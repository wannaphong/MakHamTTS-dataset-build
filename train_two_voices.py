#!/usr/bin/env python3
"""
FastThaiG2P Multi-Voice Training & Packaging Suite
==================================================
Trains and packages a Thai Kokoro-82M TTS model supporting two voices:
  - Speaker 0: newvoice_female
  - Speaker 1: newvoice_male

Pipeline Stages:
  1. prepare   : Process dataset CSVs, normalize text, convert G2P to 5-tone Kokoro,
                 validate against 115-token vocab, link audio, and create train/val/OOD lists.
  2. stage1    : Run StyleTTS2 first-stage training (acoustic & text alignment).
  3. stage2    : Run StyleTTS2 second-stage training (WavLM SLM adversarial & prosody fine-tuning).
  4. package   : Convert checkpoint to inference format, export ONNX, extract voicepacks
                 for both voices (newvoice_female and newvoice_male), and generate sound checks.
  5. test      : Run end-to-end synthesis using FastThaiG2P for both voices.
  6. all       : Execute the full pipeline sequentially (prepare -> stage1 -> stage2 -> package -> test).

Usage:
  python3 train_two_voices.py --stage all
  python3 train_two_voices.py --stage prepare
  python3 train_two_voices.py --stage stage1 --batch-size 8 --epochs-1st 20
  python3 train_two_voices.py --stage stage2 --stage2-batch-size 4 --epochs-2nd 10
  python3 train_two_voices.py --stage package
  python3 train_two_voices.py --stage test
"""

import argparse
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TrainFastThaiG2P")

ROOT_DIR = Path("/workspace/labtts")
KUKURU_DIR = ROOT_DIR / "kukuru-tts"
TRAINING_DIR = KUKURU_DIR / "training"
AUDIO_DIR = TRAINING_DIR / "audio"
CONFIG_FILE = KUKURU_DIR / "configs" / "thai_two_voices.yml"
VOCAB_CONFIG = TRAINING_DIR / "config.json"
DIST_DIR = ROOT_DIR / "dist" / "fastthaig2p_two_voices"

TEST_SENTENCES = [
    ("greeting", "สวัสดีค่ะ ยินดีต้อนรับสู่บริการของเรา"),
    ("numbers", "ยอดเงินคงเหลือสามพันห้าร้อยบาทถ้วน"),
    ("tones", "ใครขายไข่ไก่ ใกล้ๆ บ้านเรา"),
    ("complex", "กรุณากดหมายเลขหนึ่งเพื่อติดต่อฝ่ายบริการลูกค้า"),
    ("minimal_pairs", "มาใหม่ ไม้ใหม่ ไม่ไหม้ ไหม"),
]


# ==============================================================================
# 0. Base Model Setup
# ==============================================================================
def ensure_base_model() -> Path:
    """Ensure kokoro_base.pth exists in kukuru-tts/training/.

    If not found, downloads kokoro-v1_0.pth from hexgrad/Kokoro-82M on HuggingFace
    and converts it into StyleTTS2 net format.
    """
    base_pth = TRAINING_DIR / "kokoro_base.pth"
    if base_pth.exists() and base_pth.stat().st_size > 100 * 1024 * 1024:
        logger.info(f"Base model found: {base_pth} ({base_pth.stat().st_size / 1e6:.1f} MB)")
        return base_pth

    logger.info("Base model kokoro_base.pth not found. Downloading from hexgrad/Kokoro-82M...")
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    logger.info(f"Downloaded weights: {model_path}. Converting to StyleTTS2 format...")

    kokoro_state = torch.load(model_path, map_location="cpu", weights_only=True)
    net = {}
    for comp, state_dict in kokoro_state.items():
        cleaned = {k.removeprefix("module."): v for k, v in state_dict.items()}
        net[comp] = cleaned

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"net": net}, str(base_pth))
    logger.info(f"Saved base model to {base_pth} ({base_pth.stat().st_size / 1e6:.1f} MB)")
    return base_pth


# ==============================================================================
# 1. Dataset Preparation for Two Voices
# ==============================================================================
def setup_audio_symlinks(female_audio_dir: Path, male_audio_dir: Path) -> Tuple[Path, Path]:
    """Create symlinks under kukuru-tts/training/audio for both voices.

    This allows meldataset.py to read `newvoice_female/<filename>` and
    `newvoice_male/<filename>` without duplicating audio data on disk.
    """
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    f_link = AUDIO_DIR / "newvoice_female"
    m_link = AUDIO_DIR / "newvoice_male"

    if f_link.is_symlink() or f_link.exists():
        if f_link.is_symlink():
            f_link.unlink()
        else:
            shutil.rmtree(f_link)
    f_link.symlink_to(female_audio_dir.resolve())
    logger.info(f"Symlinked female audio: {f_link} -> {female_audio_dir}")

    if m_link.is_symlink() or m_link.exists():
        if m_link.is_symlink():
            m_link.unlink()
        else:
            shutil.rmtree(m_link)
    m_link.symlink_to(male_audio_dir.resolve())
    logger.info(f"Symlinked male audio: {m_link} -> {male_audio_dir}")

    return f_link, m_link


def process_voice_metadata(
    csv_path: Path,
    voice_prefix: str,
    speaker_id: int,
    g2p,
    vocab: set,
    min_phonemes: int = 10,
    max_tokens: int = 510,
    max_samples: Optional[int] = None,
) -> Tuple[List[str], int, int]:
    """Process a data_ok.csv file for a single voice.

    Returns:
        entries: list of "voice_prefix/filename|phonemes|speaker_id"
        errors: count of failed/invalid rows
        skipped: count of rows skipped (e.g., too short or too long)
    """
    from fastthaig2p import ipa_to_kokoro, normalize

    logger.info(f"Reading metadata from {csv_path} for voice '{voice_prefix}' (speaker {speaker_id})...")
    df = pd.read_csv(csv_path)

    if max_samples and max_samples > 0:
        df = df.iloc[:max_samples]
        logger.info(f"Limiting to first {max_samples} samples for quick run.")

    entries = []
    errors = 0
    skipped = 0

    target_audio_dir = AUDIO_DIR / voice_prefix

    for _, row in df.iterrows():
        fn = str(row["index"]).strip()
        text = str(row["text"]).strip()

        wav_path = target_audio_dir / fn
        if not wav_path.exists():
            errors += 1
            continue

        try:
            norm_text = normalize(text)
            ipa = g2p.convert(norm_text)
            phonemes = ipa_to_kokoro(ipa)
        except Exception:
            errors += 1
            continue

        if len(phonemes) < min_phonemes or len(phonemes) > max_tokens:
            skipped += 1
            continue

        bad_chars = {c for c in phonemes if c not in vocab}
        if bad_chars:
            errors += 1
            continue

        rel_path = f"{voice_prefix}/{fn}"
        entries.append(f"{rel_path}|{phonemes}|{speaker_id}")

    logger.info(
        f"Voice '{voice_prefix}': {len(entries)} valid utterances, "
        f"{skipped} skipped length, {errors} errors."
    )
    return entries, errors, skipped


def prepare_dataset(
    female_csv: Path,
    female_audio_dir: Path,
    male_csv: Path,
    male_audio_dir: Path,
    val_ratio: float = 0.05,
    max_samples: Optional[int] = None,
    seed: int = 1234,
) -> None:
    """Prepare train_list.txt, val_list.txt, and OOD_texts.txt for both voices."""
    logger.info("=== STEP 1: PREPARING TWO-VOICE DATASET ===")

    ensure_base_model()
    setup_audio_symlinks(female_audio_dir, male_audio_dir)

    if not VOCAB_CONFIG.exists():
        raise FileNotFoundError(f"Missing vocab config at {VOCAB_CONFIG}")
    vocab_dict = json.loads(VOCAB_CONFIG.read_text())["vocab"]
    vocab = set(vocab_dict)
    logger.info(f"Loaded Kokoro vocab: {len(vocab)} symbols (↑ tone token={vocab_dict.get('↑', 170)})")

    from fastthaig2p import G2P

    g2p = G2P()

    # Process female voice (speaker 0)
    female_entries, f_err, f_skip = process_voice_metadata(
        female_csv,
        voice_prefix="newvoice_female",
        speaker_id=0,
        g2p=g2p,
        vocab=vocab,
        max_samples=max_samples,
    )

    # Process male voice (speaker 1)
    male_entries, m_err, m_skip = process_voice_metadata(
        male_csv,
        voice_prefix="newvoice_male",
        speaker_id=1,
        g2p=g2p,
        vocab=vocab,
        max_samples=max_samples,
    )

    if not female_entries or not male_entries:
        raise RuntimeError("Failed to extract valid dataset entries for one or both voices.")

    # Stratified Train/Val split
    rng = random.Random(seed)
    rng.shuffle(female_entries)
    rng.shuffle(male_entries)

    n_val_f = max(1, int(len(female_entries) * val_ratio))
    val_f, train_f = female_entries[:n_val_f], female_entries[n_val_f:]

    n_val_m = max(1, int(len(male_entries) * val_ratio))
    val_m, train_m = male_entries[:n_val_m], male_entries[n_val_m:]

    train_all = train_f + train_m
    val_all = val_f + val_m
    rng.shuffle(train_all)
    rng.shuffle(val_all)

    # Write train_list.txt & val_list.txt
    (TRAINING_DIR / "train_list.txt").write_text("\n".join(train_all) + "\n", encoding="utf-8")
    (TRAINING_DIR / "val_list.txt").write_text("\n".join(val_all) + "\n", encoding="utf-8")

    # Generate OOD texts for both speakers
    ood_lines = []
    # Mix female training phonemes with speaker 0 and male training phonemes with speaker 1
    ood_lines.extend([e.split("|")[1] + "|0" for e in train_f[:1000]])
    ood_lines.extend([e.split("|")[1] + "|1" for e in train_m[:1000]])
    rng.shuffle(ood_lines)
    (TRAINING_DIR / "OOD_texts.txt").write_text("\n".join(ood_lines) + "\n", encoding="utf-8")

    logger.info(
        f"Dataset preparation complete!\n"
        f"  Total Training Utterances   : {len(train_all)} (Female: {len(train_f)}, Male: {len(train_m)})\n"
        f"  Total Validation Utterances : {len(val_all)} (Female: {len(val_f)}, Male: {len(val_m)})\n"
        f"  Total OOD Sentences         : {len(ood_lines)}\n"
        f"  Lists written to            : {TRAINING_DIR}/"
    )


# ==============================================================================
# 2. Training Execution with Watchdog
# ==============================================================================
def update_config_params(
    config_path: Path,
    batch_size: int,
    epochs_1st: Optional[int] = None,
    epochs_2nd: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> None:
    """Safely update yaml parameters in place."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["batch_size"] = int(batch_size)
    if epochs_1st is not None:
        cfg["epochs_1st"] = int(epochs_1st)
    if epochs_2nd is not None:
        cfg["epochs_2nd"] = int(epochs_2nd)
    if num_workers is not None:
        cfg["data_params"]["num_workers"] = int(num_workers)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    logger.info(f"Updated config {config_path}: batch_size={batch_size}")


def run_stage(
    stage_name: str,
    script_name: str,
    config_path: Path,
    gpu: int = 0,
    log_name: str = "train.log",
) -> None:
    """Run a StyleTTS2 training stage with an active watchdog."""
    logger.info(f"=== LAUNCHING {stage_name.upper()} on GPU {gpu} ===")
    style_dir = KUKURU_DIR / "StyleTTS2"
    log_path = style_dir / log_name

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONPATH"] = f"{KUKURU_DIR}:{KUKURU_DIR / 'kokoro'}:{style_dir}:" + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        script_name,
        "--config_path",
        str(config_path.resolve()),
    ]

    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(style_dir),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    logger.info(f"Process PID: {proc.pid}. Tailing log: {log_path}")

    last_size = 0
    stall_count = 0
    internal_log = style_dir / "logs" / "thai_two_voices" / "train.log"
    last_reported_line = ""

    try:
        while proc.poll() is None:
            time.sleep(10)

            # Check both logs for progress & NaN
            recent_chunks = []
            for candidate in (log_path, internal_log):
                if candidate.exists() and candidate.stat().st_size > 0:
                    with open(candidate, "r", encoding="utf-8", errors="ignore") as f:
                        f.seek(max(0, candidate.stat().st_size - 8192))
                        recent_chunks.append(f.read())

            combined_text = "\n".join(recent_chunks)

            # Watchdog for NaN / Inf
            if re.search(r"(mel loss:\s*nan|loss:\s*-?inf)", combined_text, re.IGNORECASE):
                logger.error("NaN or Inf detected in training loss! Terminating process.")
                proc.kill()
                sys.exit(2)

            # Check for progress
            curr_size = log_path.stat().st_size if log_path.exists() else 0
            if internal_log.exists():
                curr_size += internal_log.stat().st_size

            if curr_size == last_size:
                stall_count += 1
            else:
                stall_count = 0
                last_size = curr_size

            # Print latest epoch / step
            for line in combined_text.split("\n"):
                if ("Epoch [" in line or "Validation" in line or "val mel" in line.lower()) and line != last_reported_line:
                    logger.info(f"[{stage_name}] {line.strip()}")
                    last_reported_line = line

            # 30-minute stall guard
            if stall_count > 120:  # 120 * 15s = 30 minutes
                logger.error("Training stalled for 30 minutes without output. Terminating.")
                proc.kill()
                sys.exit(3)

        ret = proc.wait()
        if ret != 0:
            logger.error(f"{stage_name} exited with non-zero exit code {ret}! Check {log_path}")
            sys.exit(ret)

        logger.info(f"SUCCESS: {stage_name} completed cleanly!")

    except KeyboardInterrupt:
        logger.warning(f"Training interrupted by user. Killing PID {proc.pid}...")
        proc.kill()
        sys.exit(130)


# ==============================================================================
# 3. Packaging & Voicepack Extraction for Two Voices
# ==============================================================================
def convert_checkpoint_to_kokoro(stage2_ckpt: Path, output_pth: Path) -> Path:
    """Convert StyleTTS2 checkpoint to KModel format."""
    logger.info(f"Converting StyleTTS2 checkpoint {stage2_ckpt} -> Kokoro format {output_pth}...")
    ckpt = torch.load(str(stage2_ckpt), map_location="cpu", weights_only=False)
    net = ckpt["net"] if "net" in ckpt else ckpt

    def ensure_module_prefix(state_dict):
        return {
            ("module." + k if not k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }

    kokoro_weights = {}
    for key in ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]:
        if key in net:
            kokoro_weights[key] = ensure_module_prefix(net[key])
        else:
            logger.warning(f"Component '{key}' not found in checkpoint!")

    output_pth.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kokoro_weights, str(output_pth))
    logger.info(f"Saved Kokoro weights: {output_pth} ({output_pth.stat().st_size / 1e6:.1f} MB)")
    return output_pth


def export_onnx_model(model_pth: Path, config_json: Path, output_onnx: Path) -> Path:
    """Export Kokoro model to ONNX using classic TorchScript exporter."""
    logger.info(f"Exporting ONNX: {model_pth} -> {output_onnx}...")
    output_onnx.parent.mkdir(parents=True, exist_ok=True)

    export_script = ROOT_DIR / "FastThaiG2P" / "scripts" / "export_onnx.py"
    if not export_script.exists():
        export_script = KUKURU_DIR / "FastThaiG2P" / "scripts" / "export_onnx.py"

    cmd = [
        sys.executable,
        str(export_script),
        "--model",
        str(model_pth),
        "--config",
        str(config_json),
        "--output",
        str(output_onnx),
    ]

    subprocess.run(cmd, check=True)
    logger.info(f"ONNX Export succeeded: {output_onnx} ({output_onnx.stat().st_size / 1e6:.1f} MB)")
    return output_onnx


def extract_single_voicepack(
    model_ckpt: Path,
    style_encoder_ckpt: Path,
    audio_dir: Path,
    output_pt: Path,
    output_npy: Path,
    num_samples: int = 200,
) -> None:
    """Extract acoustic & prosodic style vectors for a single voice."""
    logger.info(f"Extracting voicepack from {audio_dir} -> {output_pt}...")
    output_pt.parent.mkdir(parents=True, exist_ok=True)

    extract_script = KUKURU_DIR / "scripts" / "extract_voicepack.py"
    cmd = [
        sys.executable,
        str(extract_script),
        "--model",
        str(model_ckpt),
        "--style-encoder-model",
        str(style_encoder_ckpt),
        "--audio-dir",
        str(audio_dir),
        "--output",
        str(output_pt),
        "--num-samples",
        str(num_samples),
        "--device",
        "cuda" if torch.cuda.is_available() else "cpu",
    ]
    subprocess.run(cmd, check=True)

    # Save numpy format for zero-torch ONNX serving
    tensor = torch.load(str(output_pt), map_location="cpu", weights_only=True)
    np.save(str(output_npy), tensor.numpy())
    logger.info(f"Saved voicepack: {output_pt} and {output_npy} (shape={tensor.shape})")


def package_two_voices(
    stage1_ckpt: Optional[Path] = None,
    stage2_ckpt: Optional[Path] = None,
    skip_onnx: bool = False,
) -> None:
    """Package model checkpoint, ONNX, and voicepacks for both voices."""
    logger.info("=== STEP 4: PACKAGING FASTTHAIG2P WITH TWO VOICES ===")

    logs_dir = KUKURU_DIR / "StyleTTS2" / "logs" / "thai_two_voices"
    if stage1_ckpt is None:
        stage1_ckpt = logs_dir / "first_stage.pth"
    if stage2_ckpt is None:
        # Pick latest epoch_2nd_*.pth or fall back to first_stage.pth
        stage2_ckpts = sorted(logs_dir.glob("epoch_2nd_*.pth"))
        if stage2_ckpts:
            stage2_ckpt = stage2_ckpts[-1]
        else:
            logger.warning(f"No stage 2 checkpoint found in {logs_dir}. Falling back to stage 1 checkpoint.")
            stage2_ckpt = stage1_ckpt

    if not stage1_ckpt.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_ckpt}")
    if not stage2_ckpt.exists():
        raise FileNotFoundError(f"Target checkpoint not found: {stage2_ckpt}")

    logger.info(f"Using Stage 1 checkpoint: {stage1_ckpt}")
    logger.info(f"Using Stage 2 checkpoint: {stage2_ckpt}")

    DIST_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Kokoro Inference Weights
    model_pth = DIST_DIR / "kokoro_thai.pth"
    convert_checkpoint_to_kokoro(stage2_ckpt, model_pth)

    # 2. Config JSON
    config_dest = DIST_DIR / "config.json"
    shutil.copy2(VOCAB_CONFIG, config_dest)

    # 3. ONNX Export
    output_onnx = DIST_DIR / "kokoro_thai.onnx"
    if not skip_onnx:
        export_onnx_model(model_pth, config_dest, output_onnx)

    # 4. Extract Voicepack for newvoice_female
    female_audio_dir = AUDIO_DIR / "newvoice_female"
    extract_single_voicepack(
        model_ckpt=stage2_ckpt,
        style_encoder_ckpt=stage1_ckpt,
        audio_dir=female_audio_dir,
        output_pt=DIST_DIR / "newvoice_female.pt",
        output_npy=DIST_DIR / "newvoice_female.npy",
    )

    # 5. Extract Voicepack for newvoice_male
    male_audio_dir = AUDIO_DIR / "newvoice_male"
    extract_single_voicepack(
        model_ckpt=stage2_ckpt,
        style_encoder_ckpt=stage1_ckpt,
        audio_dir=male_audio_dir,
        output_pt=DIST_DIR / "newvoice_male.pt",
        output_npy=DIST_DIR / "newvoice_male.npy",
    )

    logger.info(f"All packaged artifacts ready in {DIST_DIR}/")


# ==============================================================================
# 4. End-to-End Synthesis Test
# ==============================================================================
def test_synthesis(dist_dir: Path = DIST_DIR) -> None:
    """Run FastThaiG2P synthesis for both voices and save audio samples."""
    logger.info("=== STEP 5: RUNNING FASTTHAIG2P SYNTHESIS TEST ===")
    from fastthaig2p import TTS

    model_onnx = dist_dir / "kokoro_thai.onnx"
    model_pth = dist_dir / "kokoro_thai.pth"
    config_json = dist_dir / "config.json"

    # Use ONNX if available, else PyTorch
    if model_onnx.exists():
        chosen_model = model_onnx
        vp_female = dist_dir / "newvoice_female.npy"
        vp_male = dist_dir / "newvoice_male.npy"
    else:
        chosen_model = model_pth
        vp_female = dist_dir / "newvoice_female.pt"
        vp_male = dist_dir / "newvoice_male.pt"

    samples_dir = dist_dir / "samples"
    samples_f_dir = samples_dir / "female"
    samples_m_dir = samples_dir / "male"
    samples_f_dir.mkdir(parents=True, exist_ok=True)
    samples_m_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing TTS for Female Voice with {chosen_model.name}...")
    tts_f = TTS(
        model_path=chosen_model,
        voicepack_path=vp_female,
        config_path=config_json,
        device="cpu",
    )
    for tag, text in TEST_SENTENCES:
        out_wav = samples_f_dir / f"{tag}.wav"
        tts_f.synthesize(text, str(out_wav))
        logger.info(f"  [Female] {tag}: {text} -> {out_wav}")

    logger.info(f"Initializing TTS for Male Voice with {chosen_model.name}...")
    tts_m = TTS(
        model_path=chosen_model,
        voicepack_path=vp_male,
        config_path=config_json,
        device="cpu",
    )
    for tag, text in TEST_SENTENCES:
        out_wav = samples_m_dir / f"{tag}.wav"
        tts_m.synthesize(text, str(out_wav))
        logger.info(f"  [Male] {tag}: {text} -> {out_wav}")

    logger.info(f"Synthesis check complete! Listen to generated files in {samples_dir}/")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="FastThaiG2P Two-Voice Training & Packaging Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "stage1", "stage2", "package", "test", "all"],
        default="all",
        help="Pipeline stage to execute",
    )
    parser.add_argument(
        "--female-csv",
        type=Path,
        default=ROOT_DIR / "newvoice" / "dataset_output" / "newvoice_female" / "data_ok.csv",
        help="Path to newvoice_female data_ok.csv",
    )
    parser.add_argument(
        "--female-dir",
        type=Path,
        default=ROOT_DIR / "newvoice" / "dataset_output" / "newvoice_female" / "filtered_sound",
        help="Path to newvoice_female filtered WAV directory",
    )
    parser.add_argument(
        "--male-csv",
        type=Path,
        default=ROOT_DIR / "newvoice" / "dataset_output" / "newvoice_male" / "data_ok.csv",
        help="Path to newvoice_male data_ok.csv",
    )
    parser.add_argument(
        "--male-dir",
        type=Path,
        default=ROOT_DIR / "newvoice" / "dataset_output" / "newvoice_male" / "filtered_sound",
        help="Path to newvoice_male filtered WAV directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per speaker (useful for fast smoke tests; omit for full dataset)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Fraction of data reserved for validation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for Stage 1 training (4 is safe for 24GB RTX 4090)",
    )
    parser.add_argument(
        "--stage2-batch-size",
        type=int,
        default=4,
        help="Batch size for Stage 2 adversarial training (4 prevents OOM on 24GB GPUs)",
    )
    parser.add_argument(
        "--epochs-1st",
        type=int,
        default=20,
        help="Number of epochs for Stage 1",
    )
    parser.add_argument(
        "--epochs-2nd",
        type=int,
        default=10,
        help="Number of epochs for Stage 2",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA GPU index to use",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of data loader worker processes",
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX export during packaging",
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=DIST_DIR,
        help="Destination directory for packaged models & voicepacks",
    )

    args = parser.parse_args()

    # Step 1: Prepare
    if args.stage in ("prepare", "all"):
        prepare_dataset(
            female_csv=args.female_csv,
            female_audio_dir=args.female_dir,
            male_csv=args.male_csv,
            male_audio_dir=args.male_dir,
            val_ratio=args.val_ratio,
            max_samples=args.max_samples,
        )

    # Step 2: Stage 1 Training
    if args.stage in ("stage1", "all"):
        update_config_params(
            CONFIG_FILE,
            batch_size=args.batch_size,
            epochs_1st=args.epochs_1st,
            epochs_2nd=args.epochs_2nd,
            num_workers=args.num_workers,
        )
        run_stage(
            stage_name="Stage 1 Training",
            script_name="train_first.py",
            config_path=CONFIG_FILE,
            gpu=args.gpu,
            log_name="train_stage1_two_voices.log",
        )

    # Step 3: Stage 2 Training
    if args.stage in ("stage2", "all"):
        update_config_params(
            CONFIG_FILE,
            batch_size=args.stage2_batch_size,
            epochs_1st=args.epochs_1st,
            epochs_2nd=args.epochs_2nd,
            num_workers=args.num_workers,
        )
        run_stage(
            stage_name="Stage 2 Training",
            script_name="train_second.py",
            config_path=CONFIG_FILE,
            gpu=args.gpu,
            log_name="train_stage2_two_voices.log",
        )

    # Step 4: Package
    if args.stage in ("package", "all"):
        package_two_voices(skip_onnx=args.skip_onnx)

    # Step 5: Test
    if args.stage in ("test", "all"):
        test_synthesis(args.dist_dir)


if __name__ == "__main__":
    main()
