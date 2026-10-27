#!/usr/bin/env python3
"""Export Thai Kokoro checkpoint + newvoice_female style vector as a standalone Paradee ONNX model.

The resulting ONNX graph matches the Paradee single-voice interface:
- Inputs:
    input_ids: int64 [1, tokens]
    speed:     float32 [1]
- Outputs:
    waveform:  float32 [1, samples]

No external voicepack tensor is required at runtime.
Both FP32 (311 MB) and weight-only INT8 (82.7 MB) models are generated.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import soundfile as sf
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ExportThaiParadee")


class ThaiParadeeForONNX(nn.Module):
    """Single-voice Thai Kokoro wrapper matching Paradee ONNX input/output contract."""

    def __init__(self, kmodel, voice_vec: torch.Tensor):
        super().__init__()
        self.kmodel = kmodel
        if voice_vec.ndim == 3:
            voice_vec = voice_vec[0]
        if voice_vec.ndim == 1:
            voice_vec = voice_vec.unsqueeze(0)
        self.register_buffer("ref_s", voice_vec)

    def forward(
        self,
        input_ids: torch.LongTensor,
        speed: torch.FloatTensor,
    ) -> torch.FloatTensor:
        waveform, _ = self.kmodel.forward_with_tokens(input_ids, self.ref_s, speed)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return waveform


def quantize_weights_to_int8(src_onnx_path: Path, dst_onnx_path: Path) -> int:
    """Quantize 2D/3D weight matrices >= 1024 elements to int8 + DequantizeLinear per channel.
    
    This matches Paradee's official export pipeline, allowing CPU execution without
    needing ConvInteger support while reducing model file size by ~73%.
    """
    g = onnx.load(str(src_onnx_path))
    nq = 0
    new_nodes = []

    for init in list(g.graph.initializer):
        w = numpy_helper.to_array(init)
        name = init.name
        if w.dtype != np.float32:
            continue
        if w.ndim >= 2 and w.size >= 1024:
            sc = np.abs(w).reshape(w.shape[0], -1).max(1) / 127.0
            sc[sc == 0] = 1.0
            q = np.clip(np.round(w / sc.reshape((-1,) + (1,) * (w.ndim - 1))), -127, 127).astype(np.int8)
            g.graph.initializer.remove(init)
            g.graph.initializer.extend([
                numpy_helper.from_array(q, name + "_q"),
                numpy_helper.from_array(sc.astype(np.float16), name + "_s16"),
                numpy_helper.from_array(np.zeros(len(sc), np.int8), name + "_z"),
            ])
            new_nodes += [
                helper.make_node("Cast", [name + "_s16"], [name + "_s"], to=TensorProto.FLOAT),
                helper.make_node("DequantizeLinear", [name + "_q", name + "_s", name + "_z"], [name], axis=0),
            ]
            nq += 1

    for n in reversed(new_nodes):
        g.graph.node.insert(0, n)

    onnx.checker.check_model(g)
    onnx.save(g, str(dst_onnx_path))
    return nq


def export_paradee(
    model_pth: Path,
    voice_pth: Path,
    config_json: Path,
    output_dir: Path,
    opset: int = 17,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    from kokoro import KModel

    logger.info(f"Loading base Kokoro model: {model_pth}")
    kmodel = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=str(config_json),
        model=str(model_pth),
        disable_complex=True,
    ).eval()

    logger.info(f"Loading voice vector: {voice_pth}")
    vp = torch.load(voice_pth, map_location="cpu", weights_only=True)
    ref_s = vp[0] if vp.ndim == 3 else (vp if vp.ndim == 2 else vp.unsqueeze(0))
    logger.info(f"Using style vector shape: {ref_s.shape}")

    wrapper = ThaiParadeeForONNX(kmodel, ref_s).eval()

    # 1. Export FP32 ONNX
    fp32_onnx = output_dir / "paradee_thai.onnx"
    logger.info(f"Exporting FP32 ONNX model to: {fp32_onnx}")

    dummy_ids = torch.LongTensor([[0, *range(10, 60), 0]])
    dummy_speed = torch.tensor([1.0], dtype=torch.float32)

    export_kwargs = {
        "opset_version": opset,
        "do_constant_folding": True,
    }
    try:
        import inspect
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["dynamo"] = False
    except Exception:
        pass

    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_speed),
        str(fp32_onnx),
        input_names=["input_ids", "speed"],
        output_names=["waveform"],
        dynamic_axes={
            "input_ids": {1: "tokens"},
            "waveform": {1: "samples"},
        },
        **export_kwargs,
    )

    onnx_model = onnx.load(str(fp32_onnx))
    onnx.checker.check_model(onnx_model)
    fp32_size_mb = fp32_onnx.stat().st_size / 1e6
    logger.info(f"FP32 ONNX export succeeded: {fp32_onnx} ({fp32_size_mb:.1f} MB)")

    # 2. Quantize to INT8 (weight-only DequantizeLinear per channel)
    int8_onnx = output_dir / "paradee_thai_int8.onnx"
    logger.info(f"Quantizing weights to INT8: {int8_onnx}")
    nq = quantize_weights_to_int8(fp32_onnx, int8_onnx)
    int8_size_mb = int8_onnx.stat().st_size / 1e6
    logger.info(f"INT8 ONNX quantization succeeded: {int8_onnx} ({int8_size_mb:.1f} MB, {nq} matrices quantized)")

    # 3. Create Paradee-compatible config.json
    with open(config_json, "r", encoding="utf-8") as f:
        src_cfg = json.load(f)

    paradee_cfg = dict(src_cfg)
    paradee_cfg.update({
        "model_type": "paradee",
        "name": "Paradee-Thai-Female",
        "language": "th",
        "sample_rate": 24000,
        "teacher": "kokoro_thai",
        "teacher_voice": "newvoice_female",
    })

    dest_cfg = output_dir / "config.json"
    with open(dest_cfg, "w", encoding="utf-8") as f:
        json.dump(paradee_cfg, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved config.json to: {dest_cfg}")

    # 4. Validation using ONNX Runtime
    logger.info("Validating exported models with ONNX Runtime...")
    import onnxruntime as ort
    from fastthaig2p import G2P, ipa_to_kokoro

    g2p = G2P()
    test_sentences = [
        ("greeting", "สวัสดีค่ะ ยินดีต้อนรับสู่บริการของเรา"),
        ("numbers", "ยอดเงินคงเหลือสามพันห้าร้อยบาทถ้วน"),
        ("tones", "ใครขายไข่ไก่ ใกล้ๆ บ้านเรา"),
        ("minimal_pairs", "มาใหม่ ไม้ใหม่ ไม่ไหม้ ไหม"),
    ]

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    for model_name, path in [("fp32", fp32_onnx), ("int8", int8_onnx)]:
        logger.info(f"Testing {model_name} ONNX session ({path.name})...")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

        for tag, text in test_sentences:
            phonemes = ipa_to_kokoro(g2p.convert(text))
            ids = [0] + [paradee_cfg["vocab"][c] for c in phonemes if c in paradee_cfg["vocab"]] + [0]
            feed = {
                "input_ids": np.array([ids], dtype=np.int64),
                "speed": np.array([1.0], dtype=np.float32),
            }
            t0 = time.perf_counter()
            audio_out = sess.run(None, feed)[0][0]
            elapsed = time.perf_counter() - t0
            duration = len(audio_out) / 24000
            rtf = duration / max(elapsed, 1e-6)

            wav_path = samples_dir / f"{tag}_{model_name}.wav"
            sf.write(str(wav_path), audio_out, 24000)
            logger.info(f"[{model_name}] '{text}' -> {wav_path.name} ({duration:.2f}s audio in {elapsed:.3f}s, {rtf:.1f}x RT)")


def main():
    parser = argparse.ArgumentParser(description="Export Thai Paradee ONNX model")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/workspace/labtts/dist/fastthaig2p_two_voices/kokoro_thai.pth"),
        help="Path to kokoro_thai.pth",
    )
    parser.add_argument(
        "--voice",
        type=Path,
        default=Path("/workspace/labtts/dist/fastthaig2p_two_voices/newvoice_female.pt"),
        help="Path to newvoice_female.pt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/workspace/labtts/dist/fastthaig2p_two_voices/config.json"),
        help="Path to config.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/labtts/dist/paradee_thai"),
        help="Output directory for exported ONNX models and config",
    )
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_paradee(
        model_pth=args.model,
        voice_pth=args.voice,
        config_json=args.config,
        output_dir=args.out,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
