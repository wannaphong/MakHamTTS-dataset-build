# FastThaiG2P Multi-Voice Training Guide (`newvoice_female` & `newvoice_male`)

This guide covers training and deploying a 5-tone Thai Kokoro-82M TTS model supporting two distinct voices (`newvoice_female` and `newvoice_male`) with **FastThaiG2P**.

---

## 1. Architecture & Voice Design

- **Base Model**: Kokoro-82M (StyleTTS2 architecture with iSTFTNet decoder and PLBERT text encoder).
- **Phoneme Representation**: 5-tone Thai Kokoro phonemes with 115-token vocab, mapping tones to pitch markers (e.g. high tone `˦˥` → `↑` token 170, rising `˩˩˦` → `↗`).
- **Multi-Voice Representation**:
  - **Speaker 0**: `newvoice_female` (~45,600 verified utterances, ~50 hours)
  - **Speaker 1**: `newvoice_male` (~48,600 verified utterances, ~50 hours)
  - During training, the acoustic style encoder and prosodic predictor encoder learn speaker-specific representations via intra-speaker reference sampling.
  - At inference time, speaker timbres are served via voicepacks:
    - `newvoice_female.pt` / `newvoice_female.npy`
    - `newvoice_male.pt` / `newvoice_male.npy`

---

## 2. Quick Start

### Option A: One-Command Training (All Stages)
```bash
# Full pipeline on GPU 0: prepare -> stage 1 -> stage 2 -> package -> test
GPU=0 bash train_two_voices.sh all
```

### Option B: Quick Smoke Test (Verify Pipeline)
```bash
# Runs 100 samples per voice for 1 epoch to verify end-to-end functionality
bash train_two_voices.sh smoke
```

### Option C: Step-by-Step Execution via Python CLI
```bash
# Step 1: Prepare dataset & validate against Kokoro 115 vocab
python3 train_two_voices.py --stage prepare

# Step 2: Train Stage 1 (Acoustic & text alignment)
python3 train_two_voices.py --stage stage1 --batch-size 4 --epochs-1st 20

# Step 3: Train Stage 2 (SLM adversarial WavLM fine-tuning)
python3 train_two_voices.py --stage stage2 --stage2-batch-size 4 --epochs-2nd 10

# Step 4: Package models, ONNX, and voicepacks for both voices
python3 train_two_voices.py --stage package

# Step 5: Test synthesis with FastThaiG2P
python3 train_two_voices.py --stage test
```

---

## 3. Packaged Artifacts

All final model files and voicepacks are saved in `dist/fastthaig2p_two_voices/`:

| Artifact | Format | Purpose |
|---|---|---|
| `kokoro_thai.pth` | PyTorch | Inference weights (KModel loadable) |
| `kokoro_thai.onnx` | ONNX | CPU / mobile real-time serving |
| `newvoice_female.npy` / `.pt` | Tensor | Female voice style vector `[510, 1, 256]` |
| `newvoice_male.npy` / `.pt` | Tensor | Male voice style vector `[510, 1, 256]` |
| `config.json` | JSON | 115-token vocab mapping |
| `samples/female/*.wav` | Audio | Benchmark audio samples for female voice |
| `samples/male/*.wav` | Audio | Benchmark audio samples for male voice |

---

## 4. Inference in FastThaiG2P

Once packaged, use the trained voices in Python:

```python
from fastthaig2p import TTS

# 1. Synthesize with female voice
tts_female = TTS(
    model_path="dist/fastthaig2p_two_voices/kokoro_thai.onnx",
    voicepack_path="dist/fastthaig2p_two_voices/newvoice_female.npy",
    config_path="dist/fastthaig2p_two_voices/config.json",
    device="cpu",
)
tts_female.synthesize("สวัสดีค่ะ ยินดีต้อนรับสู่บริการของเรา", "female_out.wav")

# 2. Synthesize with male voice
tts_male = TTS(
    model_path="dist/fastthaig2p_two_voices/kokoro_thai.onnx",
    voicepack_path="dist/fastthaig2p_two_voices/newvoice_male.npy",
    config_path="dist/fastthaig2p_two_voices/config.json",
    device="cpu",
)
tts_male.synthesize("สวัสดีครับ ยินดีต้อนรับสู่บริการของเรา", "male_out.wav")
```
