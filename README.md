# Thai Kokoro & Thai Paradee TTS

A high-performance, open Thai Text-to-Speech system based on **Kokoro-82M** and **Paradee**, supporting 5-tone Thai phonology and standalone single-voice ONNX deployment.

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset Design & Curation](#dataset-design--curation)
3. [Training Pipeline](#training-pipeline)
4. [Thai Paradee ONNX Model](#thai-paradee-onnx-model)
5. [Benchmarks & Performance](#benchmarks--performance)
6. [Quick Start & Usage](#quick-start--usage)
   - [Using `MakHamTTS`](#1-using-makhamtts-recommended)
   - [Using `Paradee` Python API](#2-using-paradee-python-api)
   - [Using `paradee` CLI](#3-using-paradee-cli)
   - [Using `FastThaiG2P`](#4-using-fastthaig2p)
   - [Direct ONNX Runtime Inference](#5-direct-onnx-runtime-inference)
7. [Repository Structure](#repository-structure)
8. [Citation & Credits](#citation--credits)

---

## Overview

- **Base Architecture**: Kokoro-82M (StyleTTS2 architecture with iSTFTNet vocoder and PLBERT text encoder).
- **Phoneme Representation**: 5-tone Thai phonemes using a 115-token vocabulary with explicit pitch contour markers (`→` mid, `↓` low, `↘` falling, `↑` high, `↗` rising).
- **Voices**:
  - `newvoice_female`: Warm, natural Thai female voice (~50 hours).
  - `newvoice_male`: Clear, professional Thai male voice (~50 hours).
- **Thai Paradee ONNX**: A self-contained, single-voice ONNX model embedding the female style vector directly into the graph. No external voicepack is needed at runtime, matching the [Paradee](https://github.com/sahilmahendrakar/paradee) interface for CPU, server, and browser deployment.

---

## Dataset Design & Curation

The dataset follows the **Som TTS Dataset v1** lineage (by Wannaphong Phatthiyaphaibun) with automated quality control:

```
Raw Text Corpora (131k sentences)
               │
               ▼
   Text Cleaning & Filtering (Thai regex, length bounds)
               │
               ▼
  Voice Synthesis via OmniVoice (wannaphong/OmniVoice)
         ├── Speaker 0: newvoice_female (out.wav prompt)
         └── Speaker 1: newvoice_male   (out2.wav prompt)
               │
               ▼
   Speaker Verification (SpeechBrain ECAPA-TDNN)
               │
               ▼
 Audio Integrity & Duration Filtering (> 2.0s duration)
               │
               ▼
   Automated ASR Verification (Whisper CER filtering)
               │
               ▼
 Verified Dataset: 94,275 pairs (~100 hours of 24 kHz audio)
```

### Dataset Breakdown

| Split / Voice | Verified Utterances | Audio Duration | Reference Audio |
|---|---|---|---|
| `newvoice_female` | ~45,600 | ~50.0 hours | `newvoice/out.wav` |
| `newvoice_male` | ~48,600 | ~50.0 hours | `newvoice/out2.wav` |
| **Total** | **94,275** | **~100.0 hours** | **24 kHz mono WAV** |

Text sources include conversational call-center logs, everyday spoken Thai, news broadcasts, Japanese-Thai and English-Thai code-switching data, Wikipedia articles, and Common Voice sentence collections.

---

## Training Pipeline

The model was trained using the two-stage StyleTTS2 pipeline in `kukuru-tts`:

### 1. Stage 1: Acoustic & Text Alignment
- Trains the text encoder, duration predictor, prosody predictor, and acoustic style encoder.
- Uses dynamic attention masking with intra-speaker reference sampling to decouple speaker timbre from prosody.
- Learns tone-dependent duration and pitch variations across all 5 Thai tones.

### 2. Stage 2: Adversarial & Prosody Fine-Tuning
- Applies WavLM SLM (Speech Language Model) adversarial discriminators.
- Fine-tunes iSTFTNet vocoder kernels and AdaIN residual blocks for natural harmonic reconstruction and reduced buzz.

### Training Scripts
```bash
# Full automated pipeline
EPOCHS_1ST=5 EPOCHS_2ND=3 GPU=0 bash train_two_voices.sh all

# Or run stages individually:
python3 train_two_voices.py --stage prepare
python3 train_two_voices.py --stage stage1 --batch-size 4 --epochs-1st 5
python3 train_two_voices.py --stage stage2 --stage2-batch-size 4 --epochs-2nd 3
python3 train_two_voices.py --stage package
```

Multi-voice checkpoints and voicepacks are saved in `dist/fastthaig2p_two_voices/`:
- `kokoro_thai.pth` (PyTorch checkpoint)
- `kokoro_thai.onnx` (Multi-voice ONNX model requiring `ref_s` style input)
- `newvoice_female.npy` / `.pt` (Female voicepack `[510, 1, 256]`)
- `newvoice_male.npy` / `.pt` (Male voicepack `[510, 1, 256]`)

---

## Thai Paradee ONNX Model

Standard Kokoro ONNX requires managing external voicepack files (`ref_s`) sliced at runtime by phoneme length. **Thai Paradee** bakes the `newvoice_female` style representation directly into the model graph as a constant buffer, conforming to the [Paradee single-voice ONNX contract](https://github.com/sahilmahendrakar/paradee):

```
Text: "สวัสดีค่ะ" ──► FastThaiG2P ──► input_ids [1, T] ──┐
                                                        ├──► [Thai Paradee ONNX] ──► waveform [1, S]
                                       speed [1]  ──────┘
```

### Key Highlights
- **No external voicepack**: Just pass `input_ids` and `speed`.
- **Dual Export**:
  - `paradee_thai.onnx` (FP32, 325.5 MB)
  - `paradee_thai_int8.onnx` (INT8, **82.7 MB**, ~75% reduction)
- **Weight-Only Quantization**: Uses per-channel `DequantizeLinear` for weights $\ge 1024$ elements, ensuring full compatibility with CPU execution without requiring unsupported `ConvInteger` kernels.
- **Drop-in compatibility**: Works out-of-the-box with `paradee`, `fastthaig2p`, and web/in-browser engines.

Export command:
```bash
python3 scripts/export_thai_paradee.py \
  --model dist/fastthaig2p_two_voices/kokoro_thai.pth \
  --voice dist/fastthaig2p_two_voices/newvoice_female.pt \
  --config dist/fastthaig2p_two_voices/config.json \
  --out dist/paradee_thai
```

---

## Benchmarks & Performance

Evaluated on single CPU thread with 24 kHz audio output:

| Model | Size | RTF (CPU) | Input Requirements | Output Shape |
|---|---|---|---|---|
| Kokoro Thai (Multi-voice) | 325.5 MB | ~4.3x | `input_ids`, `ref_s`, `speed` | `[samples]` |
| **Thai Paradee FP32** | 325.5 MB | 4.3x – 4.4x | `input_ids`, `speed` | `[1, samples]` |
| **Thai Paradee INT8** | **82.7 MB** | **4.1x – 6.8x** | `input_ids`, `speed` | `[1, samples]` |

### Sample Audio Synthesis

| Text | Meaning / Category | Duration | INT8 RTF | Sample File |
|---|---|---|---|---|
| สวัสดีค่ะ ยินดีต้อนรับสู่บริการของเรา | Greeting / Service | 3.17 s | 4.1x RT | `dist/paradee_thai/samples/greeting_int8.wav` |
| ยอดเงินคงเหลือสามพันห้าร้อยบาทถ้วน | Banking / Numbers | 3.10 s | 4.2x RT | `dist/paradee_thai/samples/numbers_int8.wav` |
| ใครขายไข่ไก่ ใกล้ๆ บ้านเรา | 5-Tone Stress Test | 2.85 s | 4.2x RT | `dist/paradee_thai/samples/tones_int8.wav` |
| มาใหม่ ไม้ใหม่ ไม่ไหม้ ไหม | Minimal Pairs / Tones | 2.17 s | 4.1x RT | `dist/paradee_thai/samples/minimal_pairs_int8.wav` |

---

## Quick Start & Usage

### 1. Using `Paradee` Python API
The included `paradee` package automatically detects Thai script and routes to `FastThaiG2P`:

```python
import soundfile as sf
from paradee import Paradee, SAMPLE_RATE

# Loads the 82.7 MB INT8 Thai model
tts = Paradee(
    model_path="dist/paradee_thai/paradee_thai_int8.onnx",
    config_path="dist/paradee_thai/config.json",
)

audio = tts("สวัสดีค่ะ ยินดีต้อนรับสู่บริการของเรา")
sf.write("output.wav", audio, SAMPLE_RATE)
```

### 2. Using `Paradee` Python API
The included `paradee` package automatically detects Thai script and routes to `FastThaiG2P`:

```python
import soundfile as sf
from paradee import Paradee, SAMPLE_RATE

# Loads the 82.7 MB INT8 Thai model
tts = Paradee(
    model_path="dist/paradee_thai/paradee_thai_int8.onnx",
    config_path="dist/paradee_thai/config.json",
)

audio = tts("สวัสดีค่ะ ยินดีต้อนรับสู่บริการของเรา")
sf.write("output.wav", audio, SAMPLE_RATE)
```

### 3. Using `paradee` CLI

```bash
PYTHONPATH=paradee python3 -m paradee "สวัสดีค่ะ ยินดีต้อนรับสู่บริการของเรา" \
  --model-path dist/paradee_thai/paradee_thai_int8.onnx \
  --config-path dist/paradee_thai/config.json \
  -o hello_thai.wav
```

### 4. Using `FastThaiG2P`
`FastThaiG2P` detects standalone models automatically without needing `voicepack_path`:

```python
from fastthaig2p import TTS

tts = TTS(
    model_path="dist/paradee_thai/paradee_thai_int8.onnx",
    config_path="dist/paradee_thai/config.json",
)
tts.synthesize("ยอดเงินคงเหลือสามพันห้าร้อยบาทถ้วน", "balance.wav")
```

### 5. Direct ONNX Runtime Inference

```python
import json
import numpy as np
import onnxruntime as ort
import soundfile as sf
from fastthaig2p import G2P, ipa_to_kokoro

# 1. Initialize session and G2P
g2p = G2P()
cfg = json.load(open("dist/paradee_thai/config.json", encoding="utf-8"))
sess = ort.InferenceSession("dist/paradee_thai/paradee_thai_int8.onnx", providers=["CPUExecutionProvider"])

# 2. Convert text to tokens
text = "สวัสดีค่ะ ยินดีต้อนรับสู่บริการของเรา"
phonemes = ipa_to_kokoro(g2p.convert(text))
ids = [0] + [cfg["vocab"][c] for c in phonemes if c in cfg["vocab"]] + [0]

# 3. Synthesize
feed = {
    "input_ids": np.array([ids], dtype=np.int64),
    "speed": np.array([1.0], dtype=np.float32),
}
waveform = sess.run(None, feed)[0][0]  # shape: [samples]
sf.write("direct_onnx.wav", waveform, 24000)
```

---

## Repository Structure

```
├── dist/
│   ├── fastthaig2p_two_voices/    # Multi-voice Kokoro models & voicepacks
│   │   ├── kokoro_thai.pth
│   │   ├── kokoro_thai.onnx
│   │   ├── newvoice_female.npy / .pt
│   │   ├── newvoice_male.npy / .pt
│   │   └── config.json
│   └── paradee_thai/              # Standalone Thai Paradee ONNX package
│       ├── paradee_thai_int8.onnx # 82.7 MB INT8 model
│       ├── paradee_thai.onnx      # 325.5 MB FP32 model
│       ├── config.json            # 115-token vocab mapping
│       └── samples/               # Audio benchmark WAVs
├── FastThaiG2P/                   # G2P engine & TTS runtime
├── newvoice/                      # Dataset generation & OmniVoice scripts
├── paradee/                       # Paradee runtime & training suite
├── scripts/
│   └── export_thai_paradee.py     # Paradee ONNX export & INT8 quantization
├── som-tts-dataset-v1/            # Dataset curation lineage & scripts
├── train_two_voices.py            # End-to-end training pipeline
├── train_two_voices.sh            # One-command training wrapper
└── TRAINING_TWO_VOICES.md         # Multi-voice training guide
```

---

## Citation & Credits

- **Kokoro-82M**: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- **Paradee**: Sahil Mahendrakar, *Paradee: Distilling Kokoro-82M into an 8M-Parameter Single-Voice Text-to-Speech Model*, [arXiv:2610.06817](https://arxiv.org/abs/2610.06817) (2026).
- **Som TTS Dataset**: Wannaphong Phatthiyaphaibun, *Som TTS dataset: Open Data Thai TTS (Version 1.0.0)*, [Zenodo:21530909](https://doi.org/10.5281/zenodo.21530909) (2026).
- **FastThaiG2P**: [PyThaiNLP/FastThaiG2P](https://github.com/awslabs/FastThaiG2P)
