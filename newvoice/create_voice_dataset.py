"""
Create Voice Dataset from out.wav and out2.wav following som-tts-dataset-v1 pipeline
with integrated ASR quality check and High-Performance Batch Processing.

Pipeline Steps:
  1. Text Cleaning & Filtering (Thai regex & length checks)
  2. Batch Voice Cloning / Synthesis using OmniVoice (k2-fsa/OmniVoice)
  3. Speaker Verification using SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb)
  4. Audio Duration & Integrity Filtering (wave header check, min duration > 2s)
  5. Batch ASR Quality Verification using Whisper (CER calculation vs ground truth)
  6. Dataset Manifest Export (CSV & HuggingFace datasets format)
"""

import os
import re
import wave
import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import soundfile as sf
from tqdm.auto import tqdm
import pandas as pd
import jiwer
from speechbrain.inference.speaker import SpeakerRecognition
from transformers import pipeline
from omnivoice import OmniVoice

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VoiceDatasetBuilder")


# Standard base files that defined the first 61,118 sentences (indices 0..61117)
ORIGINAL_ORDER = [
    "call_center_conversations.txt",
    "everyday_conversations.txt",
    "news.txt",
    "ja_th_code-switching.txt",
    "en_th_code-switching.txt",
    "sentence-collector.txt",
    "thsentences.txt",
    "Wannaphong.txt"
]


# ==============================================================================
# 1. Text Cleaning & Preparation (Following som-tts n0-clean-text)
# ==============================================================================
def is_thai_only(text: str) -> bool:
    """Check if text contains only Thai characters and whitespace."""
    if not text or text.isspace():
        return False
    thai_pattern = r"^[\u0e00-\u0e7f\s]+$"
    return bool(re.match(thai_pattern, text))


def check_thai(text: str) -> bool:
    """Filter Thai sentences: must be Thai only, no maiyamok ('ๆ'), and length > 1."""
    cleaned = text.strip()
    return is_thai_only(cleaned) and ("ๆ" not in cleaned) and len(cleaned) > 1


def load_sentences_from_text_dir(text_dir: str) -> List[str]:
    """Load and clean Thai sentences with strict, deterministic ordering preserving previous runs."""
    sentences = []
    text_path = Path(text_dir)
    all_txt = [p.name for p in text_path.glob("*.txt")]

    # Load original files in exact previous order
    ordered_files = [f for f in ORIGINAL_ORDER if f in all_txt]
    # Followed by any new text files (like thai_wiki.txt) in sorted order
    extra_files = sorted([f for f in all_txt if f not in ORIGINAL_ORDER])
    full_order = ordered_files + extra_files

    logger.info(f"Loading {len(full_order)} text files in deterministic order from {text_dir}...")
    for fname in full_order:
        fpath = text_path / fname
        file_count = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().strip('"').strip("'")
                if check_thai(line):
                    sentences.append(line)
                    file_count += 1
        logger.info(f"  - {fname}: {file_count} valid sentences")

    logger.info(f"Total valid Thai sentences loaded: {len(sentences)}")
    return sentences


# ==============================================================================
# 2. Voice Generation with Batching (OmniVoice)
# ==============================================================================
class VoiceSynthesizer:
    def __init__(self, model_name: str = "k2-fsa/OmniVoice", device: str = "cuda:0", dtype=torch.float16):
        logger.info(f"Loading OmniVoice model ({model_name}) on {device}...")
        self.device = device
        self.model = OmniVoice.from_pretrained(
            model_name,
            device_map="cuda" if "cuda" in device else device,
            dtype=dtype
        )
        self.sample_rate = 24000

    def create_clone_prompt(self, ref_audio: str, ref_text: str):
        """Precomputes reusable voice clone prompt representation once."""
        return self.model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)

    def synthesize_batch(
        self,
        texts: List[str],
        voice_clone_prompt,
        instruct: Optional[str] = None
    ) -> List[Optional[torch.Tensor]]:
        """Generate speech audio for a batch of target texts."""
        try:
            kwargs = {
                "text": texts,
                "voice_clone_prompt": voice_clone_prompt,
            }
            if instruct:
                kwargs["instruct"] = instruct

            audios = self.model.generate(**kwargs)
            return audios
        except Exception as e:
            logger.warning(f"Error synthesizing batch of {len(texts)} texts: {e}")
            # Fallback per-sample on failure
            fallback_results = []
            for t in texts:
                try:
                    kw = {"text": t, "voice_clone_prompt": voice_clone_prompt}
                    if instruct:
                        kw["instruct"] = instruct
                    res = self.model.generate(**kw)
                    fallback_results.append(res[0] if res and len(res) > 0 and len(res[0]) > 0 else None)
                except Exception:
                    fallback_results.append(None)
            return fallback_results


# ==============================================================================
# 3. Speaker Verification (SpeechBrain ECAPA-TDNN)
# ==============================================================================
class SpeakerVerifier:
    def __init__(self, device: str = "cuda:0", threshold: float = 0.25):
        logger.info("Loading Speaker Verification Model (ECAPA-TDNN)...")
        self.threshold = threshold
        self.verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )

    def verify(self, ref_audio_path: str, generated_audio_path: str) -> Tuple[bool, float]:
        """Verify speaker similarity between reference audio and generated audio."""
        if os.path.getsize(generated_audio_path) <= 44:
            return False, 0.0

        try:
            score, _ = self.verification.verify_files(ref_audio_path, generated_audio_path)
            sim_score = float(score.item())
            return sim_score >= self.threshold, sim_score
        except Exception as e:
            logger.debug(f"Speaker verification error for {generated_audio_path}: {e}")
            return False, 0.0


# ==============================================================================
# 4. Audio Duration & Sanity Check (Following som-tts n3-filter2-and-count)
# ==============================================================================
def get_wav_duration(file_path: str) -> float:
    """Reads WAV header to quickly calculate audio duration in seconds."""
    try:
        with wave.open(str(file_path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            if rate == 0:
                return 0.0
            return frames / float(rate)
    except Exception:
        return 0.0


# ==============================================================================
# 5. ASR Quality Verification with Batching (Whisper + Thai CER)
# ==============================================================================
class ASRChecker:
    def __init__(self, model_name: str = "openai/whisper-base", device: str = "cuda:0", max_cer: float = 0.35):
        logger.info(f"Loading ASR Model ({model_name}) on {device}...")
        self.max_cer = max_cer
        self.device = device
        self.asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32
        )

    def check_batch(self, audio_paths: List[str], reference_texts: List[str]) -> List[Tuple[bool, str, float]]:
        """Transcribes a batch of audio files and evaluates CER against reference texts."""
        if not audio_paths:
            return []

        try:
            results = self.asr_pipe(
                audio_paths,
                batch_size=len(audio_paths),
                generate_kwargs={"language": "th", "task": "transcribe"}
            )
        except Exception as e:
            logger.warning(f"Batch ASR error: {e}. Falling back to sequential check...")
            results = []
            for path in audio_paths:
                try:
                    res = self.asr_pipe(path, generate_kwargs={"language": "th", "task": "transcribe"})
                    results.append(res)
                except Exception:
                    results.append({"text": ""})

        eval_results = []
        for res, ref_text in zip(results, reference_texts):
            transcribed = res.get("text", "").strip() if isinstance(res, dict) else ""
            if not transcribed:
                eval_results.append((False, "", 1.0))
                continue

            clean_ref = ref_text.replace(" ", "")
            clean_hyp = transcribed.replace(" ", "")
            cer = jiwer.cer(clean_ref, clean_hyp)
            passed = cer <= self.max_cer
            eval_results.append((passed, transcribed, cer))

        return eval_results


# ==============================================================================
# 6. Unified Dataset Pipeline with Batch Processing & Incremental Resuming
# ==============================================================================
def create_voice_dataset(
    voice_config: Dict,
    sentences: List[str],
    output_base_dir: str = "voice_dataset",
    batch_size: int = 8,
    min_duration: float = 2.0,
    speaker_threshold: float = 0.25,
    max_cer: float = 0.35,
    max_samples: Optional[int] = None,
    resume: bool = True,
    start_index: Optional[int] = None,
    target_hours: Optional[float] = None,
    device: str = "cuda:0"
):
    """Build a verified TTS voice dataset in batches with automatic resume capability."""
    voice_id = voice_config["id"]
    ref_audio = voice_config["ref_audio"]
    ref_text = voice_config["ref_text"]
    instruct = voice_config.get("instruct", None)

    logger.info("=" * 60)
    logger.info(f"Starting Voice Dataset Generation: '{voice_id}'")
    logger.info(f"Reference Audio : {ref_audio}")
    logger.info(f"Reference Text  : {ref_text}")
    logger.info(f"Instruct Style  : {instruct}")
    logger.info(f"Batch Size      : {batch_size}")
    logger.info(f"Target Hours    : {target_hours if target_hours else 'None (process all)'}")
    logger.info("=" * 60)

    raw_dir = os.path.join(output_base_dir, voice_id, "raw_sound")
    filtered_dir = os.path.join(output_base_dir, voice_id, "filtered_sound")
    csv_path = os.path.join(output_base_dir, voice_id, "data_ok.csv")
    checkpoint_path = os.path.join(output_base_dir, voice_id, "checkpoint.json")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(filtered_dir, exist_ok=True)

    passed_count = 0
    total_passed_duration = 0.0
    last_processed_idx = -1
    existing_df = None

    # 1. Load existing CSV state if present
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            passed_count = len(existing_df)
            if "duration_sec" in existing_df.columns:
                total_passed_duration = float(existing_df["duration_sec"].sum())
            logger.info(f"Loaded existing dataset from {csv_path}: {passed_count} samples ({total_passed_duration / 3600.0:.2f} hours)")
        except Exception as e:
            logger.warning(f"Failed to read existing CSV {csv_path}: {e}")

    # 2. Check for checkpoint file
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                cp = json.load(f)
                last_processed_idx = cp.get("last_processed_idx", -1)
                logger.info(f"Found checkpoint file: last_processed_idx = {last_processed_idx}")
        except Exception as e:
            logger.warning(f"Failed reading checkpoint {checkpoint_path}: {e}")

    # 3. If no checkpoint exists but data exists on disk, infer starting index
    if last_processed_idx < 0 and resume:
        max_raw = -1
        for p in Path(raw_dir).glob("*.wav"):
            try:
                max_raw = max(max_raw, int(p.stem))
            except ValueError:
                pass
        max_csv = -1
        if existing_df is not None and passed_count > 0 and "index" in existing_df.columns:
            for item in existing_df["index"]:
                try:
                    max_csv = max(max_csv, int(Path(item).stem))
                except (ValueError, TypeError):
                    pass
        last_processed_idx = max(max_raw, max_csv)
        if last_processed_idx >= 0:
            logger.info(f"Inferred last processed index from existing files: {last_processed_idx}")

    # 4. Compute effective starting index
    if start_index is not None:
        effective_start_idx = start_index
    elif resume and last_processed_idx >= 0:
        effective_start_idx = last_processed_idx + 1
    else:
        effective_start_idx = 0

    logger.info(f"Starting/Resuming dataset generation at sentence index: {effective_start_idx}")

    # Initialize models
    synthesizer = VoiceSynthesizer(device=device)
    prompt = synthesizer.create_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    verifier = SpeakerVerifier(device=device, threshold=speaker_threshold)
    asr = ASRChecker(device=device, max_cer=max_cer)

    total_sentences = sentences[:max_samples] if max_samples else sentences

    spk_fail_count = 0
    dur_fail_count = 0
    asr_fail_count = 0

    num_batches = (len(total_sentences) + batch_size - 1) // batch_size
    pbar = tqdm(total=len(total_sentences), desc=f"Processing [{voice_id}] (batch={batch_size})")

    # Initial progress bar status
    pbar.set_postfix({
        "passed": passed_count,
        "hours": f"{total_passed_duration / 3600.0:.2f}h"
    })

    for b_idx in range(num_batches):
        start_idx = b_idx * batch_size
        end_idx = min(start_idx + batch_size, len(total_sentences))

        # Fast forward if this batch was already processed
        if end_idx <= effective_start_idx:
            pbar.update(end_idx - start_idx)
            continue

        batch_items = [
            (idx, total_sentences[idx], f"{idx:06d}.wav", os.path.join(raw_dir, f"{idx:06d}.wav"))
            for idx in range(start_idx, end_idx)
            if idx >= effective_start_idx
        ]

        if not batch_items:
            pbar.update(end_idx - start_idx)
            continue

        # 1. Check which items need audio synthesis
        texts_to_generate = []
        indices_to_generate = []
        for idx, text, file_name, raw_path in batch_items:
            if not os.path.exists(raw_path):
                texts_to_generate.append(text)
                indices_to_generate.append((idx, file_name, raw_path))

        # Batch synthesize missing audios
        if texts_to_generate:
            generated_audios = synthesizer.synthesize_batch(
                texts=texts_to_generate,
                voice_clone_prompt=prompt,
                instruct=instruct
            )
            for (_, _, r_path), audio_arr in zip(indices_to_generate, generated_audios):
                if audio_arr is not None and len(audio_arr) > 0:
                    sf.write(r_path, audio_arr, synthesizer.sample_rate)

        # 2. Filter by Duration & Speaker Verification
        candidates_for_asr = []
        for idx, text, file_name, raw_path in batch_items:
            if not os.path.exists(raw_path) or os.path.getsize(raw_path) <= 44:
                continue

            duration = get_wav_duration(raw_path)
            if duration < min_duration:
                dur_fail_count += 1
                continue

            spk_pass, spk_score = verifier.verify(ref_audio, raw_path)
            if not spk_pass:
                spk_fail_count += 1
                continue

            candidates_for_asr.append((idx, text, file_name, raw_path, duration, spk_score))

        # 3. Batch ASR Verification
        batch_new_records = []
        if candidates_for_asr:
            audio_paths = [c[3] for c in candidates_for_asr]
            ref_texts = [c[1] for c in candidates_for_asr]
            asr_evals = asr.check_batch(audio_paths, ref_texts)

            for cand, (asr_pass, transcribed_text, cer_score) in zip(candidates_for_asr, asr_evals):
                idx, text, file_name, raw_path, duration, spk_score = cand
                if not asr_pass:
                    asr_fail_count += 1
                    continue

                final_path = os.path.join(filtered_dir, file_name)
                shutil.copy2(raw_path, final_path)
                passed_count += 1
                total_passed_duration += duration

                batch_new_records.append({
                    "index": file_name,
                    "audio_filepath": final_path,
                    "text": text,
                    "asr_text": transcribed_text,
                    "cer": round(cer_score, 4),
                    "speaker_similarity": round(spk_score, 4),
                    "duration_sec": round(duration, 2),
                    "voice_id": voice_id
                })

        # Incremental CSV write so progress is never lost
        if batch_new_records:
            df_new = pd.DataFrame(batch_new_records)
            file_exists = os.path.exists(csv_path)
            df_new.to_csv(csv_path, mode="a", header=not file_exists, index=False)

        # Save checkpoint after each batch
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({
                "last_processed_idx": end_idx - 1,
                "passed_count": passed_count,
                "total_duration_sec": round(total_passed_duration, 2),
                "total_duration_hours": round(total_passed_duration / 3600.0, 2)
            }, f, indent=2)

        pbar.set_postfix({
            "passed": passed_count,
            "hours": f"{total_passed_duration / 3600.0:.2f}h"
        })
        pbar.update(end_idx - start_idx)

        # Check if target hours reached
        if target_hours and (total_passed_duration / 3600.0) >= target_hours:
            logger.info("\n" + "=" * 60)
            logger.info(f"🎉 Target duration of {target_hours:.2f} hours reached! (Current: {total_passed_duration / 3600.0:.2f}h)")
            logger.info("=" * 60)
            break

    pbar.close()

    logger.info("\n" + "=" * 50)
    logger.info(f"COMPLETED / CHECKPOINTED VOICE: {voice_id}")
    logger.info(f"Total Sentences In Pool   : {len(total_sentences)}")
    logger.info(f"✅ Total Passed All Checks : {passed_count}")
    logger.info(f"🎯 Total Duration (Hours)  : {total_passed_duration / 3600.0:.2f} hrs ({total_passed_duration:.1f}s)")
    logger.info(f"Saved Metadata CSV        : {csv_path}")
    logger.info(f"Checkpoint File           : {checkpoint_path}")
    logger.info("=" * 50 + "\n")

    return pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()


# ==============================================================================
# Main Runner for out.wav (Female) and out2.wav (Male)
# ==============================================================================
if __name__ == "__main__":
    import argparse

    SCRIPT_DIR = Path(__file__).resolve().parent
    WORKSPACE_DIR = SCRIPT_DIR.parent

    default_text_dir = str(WORKSPACE_DIR / "som-tts-dataset-v1" / "text")
    default_output_dir = str(SCRIPT_DIR / "dataset_output")
    out_wav_path = str(SCRIPT_DIR / "out.wav")
    out2_wav_path = str(SCRIPT_DIR / "out2.wav")

    parser = argparse.ArgumentParser(description="Create voice dataset with batching, incremental checkpointing, and resume support")
    parser.add_argument("--text-dir", default=default_text_dir, help="Directory containing Thai text files")
    parser.add_argument("--output-dir", default=default_output_dir, help="Output directory for generated voices")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for synthesis and ASR check (e.g. 8, 16, 32)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples for testing (e.g. 10)")
    parser.add_argument("--min-duration", type=float, default=2.0, help="Minimum audio duration in seconds")
    parser.add_argument("--speaker-threshold", type=float, default=0.25, help="Speaker similarity threshold")
    parser.add_argument("--max-cer", type=float, default=0.35, help="Maximum Character Error Rate for ASR check")
    parser.add_argument("--voice-id", default="newvoice_female", choices=["newvoice_female", "newvoice_male", "all"], help="Voice ID to process")
    parser.add_argument("--resume", action="store_true", default=True, help="Automatically resume from last processed index")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Do not resume; start from index 0")
    parser.add_argument("--start-index", type=int, default=None, help="Explicit starting sentence index (e.g. 61118)")
    parser.add_argument("--target-hours", type=float, default=50.0, help="Target total clean audio hours to reach (e.g. 50.0)")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Computation device")
    args = parser.parse_args()

    # Define standard reference text used in newvoice voice-design.ipynb
    COMMON_REF_TEXT = (
        "สวัสดีครับ เรากำลังทดสอบระบบแปลงข้อความเป็นเสียงภาษาไทย เพื่อตรวจสอบความชัดเจนของการออกเสียงวรรณยุกต์ "
        "และจังหวะการพูดที่ดูเป็นธรรมชาติที่สุด"
    )

    ALL_VOICE_CONFIGS = {
        "newvoice_female": {
            "id": "newvoice_female",
            "ref_audio": out_wav_path,
            "ref_text": COMMON_REF_TEXT,
            "instruct": "female"
        },
        "newvoice_male": {
            "id": "newvoice_male",
            "ref_audio": out2_wav_path,
            "ref_text": COMMON_REF_TEXT,
            "instruct": "male"
        }
    }

    if args.voice_id == "all":
        selected_configs = list(ALL_VOICE_CONFIGS.values())
    else:
        selected_configs = [ALL_VOICE_CONFIGS[args.voice_id]]

    # Load and clean sentences in deterministic order
    sentences = load_sentences_from_text_dir(args.text_dir)

    for v_cfg in selected_configs:
        create_voice_dataset(
            voice_config=v_cfg,
            sentences=sentences,
            output_base_dir=args.output_dir,
            batch_size=args.batch_size,
            min_duration=args.min_duration,
            speaker_threshold=args.speaker_threshold,
            max_cer=args.max_cer,
            max_samples=args.max_samples,
            resume=args.resume,
            start_index=args.start_index,
            target_hours=args.target_hours,
            device=args.device
        )
