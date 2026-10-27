#!/usr/bin/env python3
"""Train and Export Thai Paradee 1.1 ONNX TTS Model (7.88M parameters, ~8.5MB INT8).

Distills the Thai Kokoro teacher (kokoro_thai.pth + newvoice_female style vector)
into the Paradee single-voice architecture:
- TextStudent (preset "s+mlp", 4.23M parameters)
- VDecoder (vhead "v2", 3.65M parameters)
Total Parameters: 7.88M!

Outputs:
- FP32 ONNX model:  paradee/models/paradee_thai/paradee_thai.onnx (~31 MB)
- INT8 ONNX model:  paradee/models/paradee_thai/paradee_thai_int8.onnx (~8.5 MB)
- Configuration:    paradee/models/paradee_thai/config.json (115-token vocab, lang: th)
"""

import argparse
import glob
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# Add paradee/training/scripts to path
PARADEE_DIR = Path("/workspace/labtts/paradee")
PARADEE_SCRIPTS = PARADEE_DIR / "training" / "scripts"
sys.path.insert(0, str(PARADEE_SCRIPTS))

from student import TextStudent, alignment
from vhead import VDecoder, SR, FHOP
from discriminators import Discriminators
from huggingface_hub import hf_hub_download
from kokoro import KModel
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TrainThaiParadee")


def strip(m):
    for x in m.modules():
        try:
            remove_weight_norm(x)
        except Exception:
            pass
        if is_parametrized(x, "weight"):
            try:
                remove_parametrizations(x, "weight", leave_parametrized=True)
            except Exception:
                pass


# -----------------------------------------------------------------------------
# ONNX Export STFT and Model Definitions
# -----------------------------------------------------------------------------
class ConvSTFT(nn.Module):
    """torch.stft / torch.istft as conv1d / conv_transpose1d."""

    def __init__(self, n, hop):
        super().__init__()
        self.n, self.hop, self.K = n, hop, n // 2 + 1
        w = torch.hann_window(n, dtype=torch.float64)
        k = torch.arange(self.K, dtype=torch.float64)[:, None]
        ang = 2 * torch.pi * ((k * torch.arange(n, dtype=torch.float64)) % n) / n
        c = torch.full((self.K, 1), 2.0, dtype=torch.float64)
        c[0] = c[-1] = 1
        self.register_buffer("fwd", (torch.cat([torch.cos(ang), -torch.sin(ang)]) * w).float()[:, None])
        self.register_buffer("inv", (torch.cat([torch.cos(ang) * c, -torch.sin(ang) * c]) / n * w).float()[:, None])
        self.register_buffer("w2", (w ** 2).float()[None, None])

    def transform(self, x):
        y = F.conv1d(F.pad(x[:, None], (self.n // 2, self.n // 2), mode="reflect"), self.fwd, stride=self.hop)
        return y[:, :self.K], y[:, self.K:]

    def inverse(self, re, im):
        y = F.conv_transpose1d(torch.cat([re, im], 1), self.inv, stride=self.hop)[:, 0]
        env = F.conv_transpose1d(torch.ones_like(re[:1, :1]), self.w2, stride=self.hop)[:, 0]
        return (y / env.clamp_min(1e-8))[:, self.n // 2 : -(self.n // 2)]


class HeadX(nn.Module):
    """SourceFilterHead2 forward compatible with ONNX."""

    def __init__(self, h):
        super().__init__()
        self.h = h
        self.st = ConvSTFT(h.n_fft, h.hop)
        self.sb = ConvSTFT(150, 75)
        self.register_buffer("k", torch.arange(1, h.K + 1).float()[None, :, None])
        self.register_buffer("fr", torch.linspace(0, SR / 2, h.nb)[None, :, None])

    def forward(self, x, f0, noise=None):
        h = self.h
        n = x.shape[2]
        T = n * h.up
        S = n * FHOP
        z = h.norm0(h.inp(x).transpose(1, 2)).transpose(1, 2)
        for b in h.blocks:
            z = b(z)
        hn = h.norm1(z.transpose(1, 2))
        o = h.out(hn).view(1, T, h.C * h.G + 1)
        o = torch.cat([o, o[:, -1:]], 1)
        g = (o[..., :-1].reshape(1, T + 1, h.C, h.G) @ h.M.T).permute(0, 2, 3, 1)
        gh, gn = torch.exp(g[:, 0].clamp(max=6)), torch.exp(g[:, 3].clamp(max=6))
        um = torch.sqrt(g[:, 1] ** 2 + g[:, 2] ** 2).clamp_min(1e-4)
        ur, ui = g[:, 1] / um, g[:, 2] / um

        r = f0.double() * FHOP / SR
        c = torch.cumsum(r, 1)
        c = (c - torch.floor(c)).float()
        d = torch.cat([r[:, 1:], torch.zeros_like(r[:, :1])], 1).float()
        t = torch.arange(S, device=x.device)
        p = (t.float() + 0.5) / FHOP - 0.5
        i = torch.clamp(torch.floor(p), min=0).long()
        i = torch.minimum(i, (n - 1) * torch.ones_like(i))
        frac = torch.clamp(p - i.float(), 0, 1)
        cyc = c[:, i] + frac * d[:, i]
        f0s = f0[:, t // FHOP]
        uv = (f0s > 10).float()
        amp = ((f0s[:, None] * self.k < 11000).float() * uv[:, None])
        e = (torch.sin(2 * torch.pi * self.k * cyc[:, None]) * amp).sum(1) * 0.1
        nz = torch.randn_like(e) if noise is None else noise
        Er, Ei = self.st.transform(e)
        Nr, Ni = self.st.transform(nz)
        j = 0
        for m in range(1, h.mod + 1):
            for fn in (torch.cos, torch.sin):
                a = g[:, 4 + j]
                mr, mi = self.st.transform(nz * uv * fn(2 * torch.pi * m * cyc))
                Nr = Nr + a * mr
                Ni = Ni + a * mi
                j += 1
        f0f = torch.cat([f0[:, torch.arange(T, device=x.device) // h.up], f0[:, -1:]], 1).clamp_min(60)
        ang = 2 * torch.pi * (self.fr / f0f[:, None]) * (0.5 * torch.tanh(o[..., -1])[:, None])
        Hr = gh * (ur * torch.cos(ang) - ui * torch.sin(ang))
        Hi = gh * (ur * torch.sin(ang) + ui * torch.cos(ang))
        y = self.st.inverse(Er * Hr - Ei * Hi + Nr * gn, Er * Hi + Ei * Hr + Ni * gn)
        if h.burst:
            gb = torch.exp((h.out_b(hn).view(1, n * 4, h.Gb) @ h.Mb.T).clamp(max=6)).transpose(1, 2)
            gb = torch.cat([gb, gb[..., -1:]], -1)
            br, bi = self.sb.transform(nz.flip(-1))
            y = y + self.sb.inverse(br * gb, bi * gb)
        return y


class Paradee2(nn.Module):
    """Paradee 1.1 single-voice full pipeline (7.88M parameters)."""

    def __init__(self, ts, vd):
        super().__init__()
        self.ts, self.vd, self.head = ts, vd, HeadX(vd.head)

    def text(self, ids):
        t = self.ts
        p = t.predictor
        s = t.style
        d_en = t.bert_encoder(t.bert(ids, attention_mask=torch.ones_like(ids))).transpose(-1, -2)
        x = d_en
        T = ids.shape[1]
        sx = s[:, :, None].expand(-1, -1, T)
        x = torch.cat([x, sx], 1)
        for b in p.text_encoder.lstms:
            if isinstance(b, nn.LSTM):
                x = b(x.transpose(-1, -2))[0].transpose(-1, -2)
            else:
                x = torch.cat([b(x.transpose(-1, -2), s).transpose(-1, -2), sx], 1)
        d = x.transpose(-1, -2)
        dur = torch.sigmoid(p.duration_proj(p.lstm(d)[0])).sum(-1)[0]
        e = t.text_encoder
        h = e.embedding(ids).transpose(1, 2)
        for c in e.cnn:
            h = c(h)
        return d, dur, t.asr_proj(e.lstm(h.transpose(1, 2))[0]).transpose(-1, -2)

    def trunk(self, asr, F0_curve, N):
        v = self.vd
        s = v.style
        F0 = v.F0_conv(F0_curve.unsqueeze(1))
        Nn = v.N_conv(N.unsqueeze(1))
        x = v.encode(torch.cat([asr, F0, Nn], 1), s)
        asr_res = v.asr_res(asr)
        res = True
        for block in v.decode:
            if res:
                x = torch.cat([x, asr_res, F0, Nn], 1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        return x

    def forward(self, input_ids: torch.LongTensor, speed: torch.FloatTensor) -> torch.FloatTensor:
        d, dur, asr_tok = self.text(input_ids)
        n = torch.clamp(torch.round(dur / speed), min=1).long()
        end = torch.cumsum(n, 0)
        start = end - n
        f = torch.arange(end[-1], device=input_ids.device)[None]
        aln = ((f >= start[:, None]) & (f < end[:, None])).float()[None]
        F0, N = self.ts.predictor.F0Ntrain(d.transpose(-1, -2) @ aln, self.ts.style)
        return self.head(self.trunk(asr_tok @ aln, F0, N), F0)


# -----------------------------------------------------------------------------
# Stage 1: Generate Teacher Corpus
# -----------------------------------------------------------------------------
def generate_teacher_shards(
    kmodel: KModel,
    voice_path: Path,
    train_list_path: Path,
    config_path: Path,
    output_dir: Path,
    target_sentences: int = 2500,
    shard_size: int = 500,
    device: str = "cuda",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(glob.glob(str(output_dir / "shard_*.pt")))
    if len(existing) >= (target_sentences // shard_size):
        logger.info(f"Found {len(existing)} teacher shards in {output_dir}, skipping generation.")
        return

    logger.info(f"Generating teacher shards from {train_list_path}...")
    cfg = json.load(open(config_path, encoding="utf-8"))
    vocab = cfg["vocab"]

    # Filter female voice sentences
    with open(train_list_path, encoding="utf-8") as f:
        lines = [line.strip().split("|") for line in f if line.strip()]
    female_items = [l for l in lines if len(l) == 3 and l[2] == "0"]
    random.seed(42)
    random.shuffle(female_items)
    selected_items = female_items[:target_sentences]

    vp = torch.load(voice_path, map_location=device, weights_only=True)
    ref_s = vp  # shape [510, 1, 256]

    kmodel.to(device).eval()

    t0 = time.time()
    total_audio_sec = 0.0
    shard_idx = 0

    for i in range(0, len(selected_items), shard_size):
        shard_path = output_dir / f"shard_{shard_idx:04d}.pt"
        if shard_path.exists():
            shard_idx += 1
            continue

        batch_items = selected_items[i : i + shard_size]
        rows = []
        for wav_path, ps, spk in batch_items:
            ids = [0] + [vocab[c] for c in ps if c in vocab] + [0]
            if len(ids) < 6 or len(ids) > 300:
                continue

            ids_t = torch.LongTensor([ids]).to(device)
            T = ids_t.shape[1]
            L = torch.full((1,), T, device=device)
            mask = torch.zeros(1, T, dtype=torch.bool, device=device)

            idx = min(T - 3, len(ref_s) - 1)
            r = ref_s[idx]
            if r.ndim == 1:
                r = r.unsqueeze(0)
            s = r[:, 128:]

            with torch.no_grad():
                bert_dur = kmodel.bert(ids_t, attention_mask=(~mask).int())
                d_en = kmodel.bert_encoder(bert_dur).transpose(-1, -2)
                d = kmodel.predictor.text_encoder(d_en, s, L, mask)
                x, _ = kmodel.predictor.lstm(d)
                dur = torch.sigmoid(kmodel.predictor.duration_proj(x)).sum(-1)
                pred_dur = torch.round(dur).clamp(min=1).long().squeeze(0)

                idx_aln = torch.repeat_interleave(torch.arange(T, device=device), pred_dur)
                aln = torch.zeros(T, idx_aln.shape[0], device=device)
                aln[idx_aln, torch.arange(idx_aln.shape[0])] = 1
                aln = aln[None]

                en = d.transpose(-1, -2) @ aln
                F0, Nn = kmodel.predictor.F0Ntrain(en, s)
                t_en = kmodel.text_encoder(ids_t, L, mask)
                asr = t_en @ aln
                audio = kmodel.decoder(asr, F0, Nn, r[:, :128]).squeeze()

                c = lambda t: t.squeeze(0).cpu()
                dur_cpu = dur.squeeze(0).cpu()
                audio_cpu = audio.cpu()
                total_audio_sec += len(audio_cpu) / 24000.0

                rows.append({
                    "ids": ids_t.squeeze(0).cpu(),
                    "dur": dur_cpu,
                    "pred_dur": pred_dur.cpu(),
                    "t_en": t_en.squeeze(0).half().cpu(),
                    "F0": F0.squeeze(0).cpu(),
                    "N": Nn.squeeze(0).cpu(),
                    "audio": (audio_cpu.clamp(-1, 1) * 32767).short(),
                    "ps": ps,
                })

        torch.save(rows, shard_path)
        logger.info(
            f"Saved {shard_path} ({len(rows)} sents, total {total_audio_sec/60:.1f}m audio, "
            f"elapsed {time.time()-t0:.1f}s)"
        )
        shard_idx += 1


# -----------------------------------------------------------------------------
# Stage 2: Train TextStudent (4.23M parameters)
# -----------------------------------------------------------------------------
def train_text_student(
    shards_dir: Path,
    output_checkpoint: Path,
    steps: int = 5000,
    bs: int = 32,
    lr: float = 5e-4,
    device: str = "cuda",
) -> TextStudent:
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model = TextStudent("s+mlp", n_token=178).to(device)

    if output_checkpoint.exists():
        logger.info(f"Loading existing TextStudent checkpoint from {output_checkpoint}")
        ck = torch.load(output_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        return model
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def collate(batch):
        B = len(batch)
        T = max(len(r["ids"]) for r in batch)
        Fm = max(len(r["F0"]) for r in batch) // 2
        ids = torch.zeros(B, T, dtype=torch.long)
        L = torch.tensor([len(r["ids"]) for r in batch])
        dur = torch.zeros(B, T)
        ten = torch.zeros(B, 512, T)
        aln = torch.zeros(B, T, Fm)
        F0 = torch.zeros(B, 2 * Fm)
        N = torch.zeros(B, 2 * Fm)
        fm = torch.zeros(B, 2 * Fm, dtype=torch.bool)
        for i, r in enumerate(batch):
            t = len(r["ids"])
            f = len(r["F0"]) // 2
            ids[i, :t] = r["ids"]
            dur[i, :t] = r["dur"]
            ten[i, :, :t] = r["t_en"].float()
            aln[i, :t, :f] = alignment(r["pred_dur"], f)
            F0[i, :2 * f] = r["F0"]
            N[i, :2 * f] = r["N"]
            fm[i, :2 * f] = True
        tm = torch.arange(T)[None] < L[:, None]
        return ids, L, dur, ten, aln, F0, N, tm, fm

    model.train()
    t0 = time.time()
    running_loss = 0.0

    for step in range(1, steps + 1):
        batch = [random.choice(train) for _ in range(bs)]
        ids, L, dur, ten, aln, F0, N, tm, fm = [x.to(device) for x in collate(batch)]

        opt.zero_grad()
        pd, pF0, pN, pasr = model(ids, L, aln)

        l_dur = (F.l1_loss(torch.log(pd.clamp_min(1e-3)), torch.log(dur.clamp_min(1e-3)), reduction="none") * tm).sum() / tm.sum()
        l_f0 = (F.l1_loss(pF0, F0, reduction="none") * fm).sum() / fm.sum() / 100
        l_n = (F.l1_loss(pN, N, reduction="none") * fm).sum() / fm.sum()
        l_asr = (F.l1_loss(pasr, ten, reduction="none") * tm[:, None]).sum() / tm.sum() / 512

        loss = l_dur + l_f0 + l_n + l_asr
        loss.backward()
        opt.step()

        running_loss += loss.item()

        if step % 1000 == 0 or step == steps:
            avg_loss = running_loss / (1000 if step % 1000 == 0 else (step % 1000))
            running_loss = 0.0
            dt = time.time() - t0
            logger.info(
                f"[TextStudent] Step {step}/{steps} | Loss: {avg_loss:.4f} "
                f"(dur: {l_dur.item():.4f}, f0: {l_f0.item():.4f}, n: {l_n.item():.4f}, asr: {l_asr.item():.4f}) | "
                f"{step/dt:.1f} steps/s"
            )

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": steps}, output_checkpoint)
    logger.info(f"Saved TextStudent checkpoint to {output_checkpoint}")
    return model


# -----------------------------------------------------------------------------
# Stage 3: Fine-tune VDecoder on Thai Audio (3.65M parameters)
# -----------------------------------------------------------------------------
def finetune_vdecoder(
    shards_dir: Path,
    pretrained_decoder_path: Path,
    output_checkpoint: Path,
    steps: int = 2000,
    bs: int = 16,
    lr: float = 2e-4,
    device: str = "cuda",
) -> VDecoder:
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model = VDecoder(head="v2").to(device)

    if output_checkpoint.exists():
        logger.info(f"Loading existing VDecoder checkpoint from {output_checkpoint}")
        model.load_ckpt(str(output_checkpoint), device=device)
        return model

    shards = sorted(glob.glob(str(shards_dir / "shard_*.pt")))
    train = []
    for s in shards:
        for r in torch.load(s, weights_only=False):
            train.append(r)

    model = VDecoder(head="v2").to(device)
    logger.info(f"Loading pretrained VDecoder weights from: {pretrained_decoder_path}")
    model.load_ckpt(str(pretrained_decoder_path), device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.8, 0.99), weight_decay=0.01)
    mel_transform = torchaudio.transforms.MelSpectrogram(24000, n_fft=1024, hop_length=256, n_mels=80).to(device)

    def spec_losses(y, t):
        lm = F.l1_loss(torch.log(mel_transform(y).clamp_min(1e-5)), torch.log(mel_transform(t).clamp_min(1e-5)))
        ls = 0
        SZ = (512, 1024, 2048)
        for n in SZ:
            w = torch.hann_window(n, device=device)
            Y = torch.stft(y, n, n // 4, window=w, return_complex=True).abs()
            T = torch.stft(t, n, n // 4, window=w, return_complex=True).abs()
            ls = ls + (T - Y).norm() / T.norm().clamp_min(1e-5) + F.l1_loss(torch.log(Y.clamp_min(1e-5)), torch.log(T.clamp_min(1e-5)))
        return lm, ls / len(SZ)

    def segment(r, seg=64):
        aln = alignment(r["pred_dur"])
        asr = r["t_en"].float() @ aln
        Fr = aln.shape[1]
        F0 = r["F0"][:2 * Fr]
        Nn = r["N"][:2 * Fr]
        au = r["audio"][:600 * Fr].float() / 32767.0
        if Fr <= seg:
            pad = seg - Fr
            asr = F.pad(asr, (0, pad))
            F0 = F.pad(F0, (0, 2 * pad))
            Nn = F.pad(Nn, (0, 2 * pad))
            au = F.pad(au, (0, 600 * pad))
            a = 0
        else:
            a = random.randint(0, Fr - seg)
        c0 = (F0[:2 * a].double() * 300 / 24000).sum() % 1
        return asr[:, a : a + seg], F0[2 * a : 2 * (a + seg)], Nn[2 * a : 2 * (a + seg)], au[600 * a : 600 * (a + seg)], c0.float()

    model.train()
    t0 = time.time()
    running_lm, running_ls = 0.0, 0.0

    for step in range(1, steps + 1):
        batch = [segment(random.choice(train)) for _ in range(bs)]
        asr_b = torch.stack([b[0] for b in batch]).to(device)
        f0_b = torch.stack([b[1] for b in batch]).to(device)
        n_b = torch.stack([b[2] for b in batch]).to(device)
        au_b = torch.stack([b[3] for b in batch]).to(device)
        c0_b = torch.stack([b[4] for b in batch]).to(device)

        opt.zero_grad()
        y = model(asr_b, f0_b, n_b, c0=c0_b)
        lm, ls = spec_losses(y, au_b)
        loss = 45.0 * lm + ls
        loss.backward()
        opt.step()

        running_lm += lm.item()
        running_ls += ls.item()

        if step % 500 == 0 or step == steps:
            dt = time.time() - t0
            count = 500 if step % 500 == 0 else (step % 500)
            logger.info(
                f"[VDecoder] Step {step}/{steps} | Mel Loss: {running_lm/count:.4f} | "
                f"STFT Loss: {running_ls/count:.4f} | {step/dt:.1f} steps/s"
            )
            running_lm, running_ls = 0.0, 0.0

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": steps}, output_checkpoint)
    logger.info(f"Saved VDecoder checkpoint to {output_checkpoint}")
    return model


# -----------------------------------------------------------------------------
# Stage 4: ONNX Assembly & Weight-only INT8 Quantization
# -----------------------------------------------------------------------------
def export_paradee_onnx(
    ts: TextStudent,
    vd: VDecoder,
    output_dir: Path,
    config_src: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    strip(ts)
    strip(vd)
    model = Paradee2(ts, vd).eval().cpu()

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    assert 7.8e6 < num_params < 8.0e6, f"Expected ~7.88M parameters, got {num_params}"

    fp32_path = output_dir / "paradee_thai.onnx"
    dummy_ids = torch.LongTensor([[0, *range(10, 50), 0]])
    dummy_speed = torch.tensor([1.0], dtype=torch.float32)

    logger.info(f"Exporting FP32 ONNX graph to {fp32_path}...")
    torch.onnx.export(
        model,
        (dummy_ids, dummy_speed),
        str(fp32_path),
        input_names=["input_ids", "speed"],
        output_names=["waveform"],
        dynamic_axes={"input_ids": {1: "tokens"}, "waveform": {1: "samples"}},
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    fp32_mb = fp32_path.stat().st_size / (1024 * 1024)
    logger.info(f"Exported FP32 ONNX: {fp32_mb:.2f} MB")

    # Quantize to INT8 + dynamic DFT basis nodes following export_paradee_v1_1.py
    logger.info("Applying Paradee 1.1 weight-only INT8 quantization and basis compression...")
    g = onnx.load(str(fp32_path))
    nq = nh = nb = 0
    new_nodes = []

    def basis(n, inverse):
        w = torch.hann_window(n, dtype=torch.float64)
        k = torch.arange(n // 2 + 1, dtype=torch.float64)[:, None]
        ang = 2 * torch.pi * ((k * torch.arange(n, dtype=torch.float64)) % n) / n
        c = torch.full((n // 2 + 1, 1), 2.0, dtype=torch.float64)
        c[0] = c[-1] = 1
        return (
            (torch.cat([torch.cos(ang) * c, -torch.sin(ang) * c]) / n * w)
            if inverse
            else (torch.cat([torch.cos(ang), -torch.sin(ang)]) * w)
        ).float()[:, None].numpy()

    def basis_nodes(name, n, inverse):
        p = name + "_gen_"
        K = n // 2 + 1
        row = np.ones((2 * K, 1), np.float32)
        if inverse:
            row[:] = 2 / n
            row[[0, K - 1, K, 2 * K - 1]] = 1 / n
        g.graph.initializer.extend([
            numpy_helper.from_array(np.arange(K, dtype=np.int64)[:, None], p + "k"),
            numpy_helper.from_array(np.arange(n, dtype=np.int64)[None], p + "n"),
            numpy_helper.from_array(np.array(n, np.int64), p + "N"),
            numpy_helper.from_array(np.array(2 * np.pi / n, np.float32), p + "c"),
            numpy_helper.from_array(torch.hann_window(n).numpy()[None], p + "w"),
            numpy_helper.from_array(row, p + "row"),
            numpy_helper.from_array(np.array([1], np.int64), p + "ax"),
        ])
        return [
            helper.make_node("Mul", [p + "k", p + "n"], [p + "kn"]),
            helper.make_node("Mod", [p + "kn", p + "N"], [p + "m"]),
            helper.make_node("Cast", [p + "m"], [p + "mf"], to=TensorProto.FLOAT),
            helper.make_node("Mul", [p + "mf", p + "c"], [p + "a"]),
            helper.make_node("Cos", [p + "a"], [p + "cos"]),
            helper.make_node("Sin", [p + "a"], [p + "sin"]),
            helper.make_node("Neg", [p + "sin"], [p + "nsin"]),
            helper.make_node("Concat", [p + "cos", p + "nsin"], [p + "cs"], axis=0),
            helper.make_node("Mul", [p + "cs", p + "w"], [p + "b0"]),
            helper.make_node("Mul", [p + "b0", p + "row"], [p + "b"]),
            helper.make_node("Unsqueeze", [p + "b", p + "ax"], [name]),
        ]

    for init in list(g.graph.initializer):
        w = numpy_helper.to_array(init)
        name = init.name
        if w.dtype != np.float32:
            continue
        hit = [
            (n, inv)
            for n in (600, 150)
            for inv in (False, True)
            if w.shape == (n + 2, 1, n) and np.abs(w - basis(n, inv)).max() < 1e-4 * np.abs(w).max()
        ]
        if hit:
            g.graph.initializer.remove(init)
            new_nodes += basis_nodes(name, *hit[0])
            nb += 1
        elif w.ndim >= 2 and w.size >= 1024:
            sc = np.abs(w).reshape(w.shape[0], -1).max(1) / 127
            sc[sc == 0] = 1
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
        elif w.size >= 64 and np.abs(w).max() < 6e4:
            g.graph.initializer.remove(init)
            g.graph.initializer.append(numpy_helper.from_array(w.astype(np.float16), name + "_16"))
            new_nodes.append(helper.make_node("Cast", [name + "_16"], [name], to=TensorProto.FLOAT))
            nh += 1

    for n_ in reversed(new_nodes):
        g.graph.node.insert(0, n_)

    int8_path = output_dir / "paradee_thai_int8.onnx"
    onnx.checker.check_model(g)
    onnx.save(g, str(int8_path))
    int8_mb = int8_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"Quantized {nq} weight tensors to INT8, {nh} to FP16, replaced {nb} DFT bases -> "
        f"{int8_path}: {int8_mb:.2f} MB"
    )

    # Copy config.json with language = "th"
    cfg = json.load(open(config_src, encoding="utf-8"))
    cfg["language"] = "th"
    cfg["model_type"] = "paradee-1.1"
    cfg["num_parameters"] = num_params
    cfg["architecture"] = "Paradee 1.1 (TextStudent s+mlp + VDecoder vhead v2)"
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    logger.info(f"Updated configuration saved to {output_dir / 'config.json'}")

    return fp32_path, int8_path


# -----------------------------------------------------------------------------
# Stage 5: Synthesis & Quality Validation
# -----------------------------------------------------------------------------
def test_synthesis(onnx_path: Path, config_path: Path, samples_dir: Path) -> None:
    import onnxruntime as ort
    from fastthaig2p import G2P, ipa_to_kokoro

    samples_dir.mkdir(parents=True, exist_ok=True)
    g2p = G2P()
    cfg = json.load(open(config_path, encoding="utf-8"))
    vocab = cfg["vocab"]

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    test_sentences = [
        ("greeting", "สวัสดีค่ะ ยินดีต้อนรับสู่บริการของเรา"),
        ("tones", "ปา ป่า ป้า ป๊า ป๋า ขา ข่า ข้า คา ค่า ค้า"),
        ("numbers", "หนึ่ง สอง สาม สี่ ห้า หก เจ็ด แปด เก้า สิบ ร้อย พัน หมื่น แสน ล้าน"),
        ("minimal_pairs", "เสือ เสื่อ เสื้อ สื่อ ซื้อ ไม้ ใหม่ ไม่ ไหม"),
    ]

    logger.info(f"Testing inference with {onnx_path.name} on CPU...")
    for tag, text in test_sentences:
        ps = ipa_to_kokoro(g2p.convert(text))
        ids = [0] + [vocab[c] for c in ps if c in vocab] + [0]
        feed = {
            "input_ids": np.array([ids], dtype=np.int64),
            "speed": np.array([1.0], dtype=np.float32),
        }
        t0 = time.perf_counter()
        wav = session.run(None, feed)[0][0]
        dt = time.perf_counter() - t0
        dur_sec = len(wav) / 24000.0
        rtf = dt / dur_sec
        wav_path = samples_dir / f"{tag}.wav"
        sf.write(str(wav_path), wav, 24000)
        logger.info(
            f"Synthesized '{text}' -> {wav_path.name} ({dur_sec:.2f}s audio in {dt*1000:.1f}ms, "
            f"speed: {1/rtf:.1f}x real-time)"
        )


def main():
    parser = argparse.ArgumentParser(description="Make Thai Paradee ONNX model (7.88M parameters).")
    parser.add_argument("--model-pth", default="dist/fastthaig2p_two_voices/kokoro_thai.pth")
    parser.add_argument("--voice-pth", default="dist/fastthaig2p_two_voices/newvoice_female.pt")
    parser.add_argument("--config-json", default="dist/fastthaig2p_two_voices/config.json")
    parser.add_argument("--train-list", default="kukuru-tts/training/train_list.txt")
    parser.add_argument("--output-dir", default="paradee/models/paradee_thai")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text-steps", type=int, default=5000)
    parser.add_argument("--dec-steps", type=int, default=1500)
    args = parser.parse_args()

    model_pth = Path(args.model_pth)
    voice_pth = Path(args.voice_pth)
    config_json = Path(args.config_json)
    train_list = Path(args.train_list)
    output_dir = Path(args.output_dir)
    device = args.device

    teacher_shards_dir = PARADEE_DIR / "training" / "data" / "teacher_thai"
    checkpoints_dir = PARADEE_DIR / "training" / "checkpoints"

    # 1. Download official Paradee-8M-v1.1 decoder checkpoint as initial weights
    logger.info("Downloading official Paradee-8M-v1.1 decoder checkpoint...")
    dec_pretrained = hf_hub_download("sahilmahendrakar/Paradee-8M-v1.1", "pytorch/decoder.pt")

    # 2. Load teacher
    logger.info("Loading Thai Kokoro teacher...")
    kmodel = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=str(config_json),
        model=str(model_pth),
        disable_complex=True,
    ).eval()

    # 3. Generate teacher shards
    generate_teacher_shards(
        kmodel=kmodel,
        voice_path=voice_pth,
        train_list_path=train_list,
        config_path=config_json,
        output_dir=teacher_shards_dir,
        target_sentences=2500,
        shard_size=500,
        device=device,
    )

    # 4. Train TextStudent (4.23M params)
    text_ckpt = checkpoints_dir / "text_thai" / "last.pt"
    ts = train_text_student(
        shards_dir=teacher_shards_dir,
        output_checkpoint=text_ckpt,
        steps=args.text_steps,
        bs=32,
        lr=5e-4,
        device=device,
    )

    # 5. Fine-tune VDecoder (3.65M params)
    dec_ckpt = checkpoints_dir / "vhead_thai" / "last.pt"
    vd = finetune_vdecoder(
        shards_dir=teacher_shards_dir,
        pretrained_decoder_path=Path(dec_pretrained),
        output_checkpoint=dec_ckpt,
        steps=args.dec_steps,
        bs=16,
        lr=2e-4,
        device=device,
    )

    # 6. Assemble & Export to ONNX
    fp32_onnx, int8_onnx = export_paradee_onnx(
        ts=ts,
        vd=vd,
        output_dir=output_dir,
        config_src=config_json,
    )

    # 7. Test Synthesis
    samples_dir = output_dir / "samples"
    test_synthesis(int8_onnx, output_dir / "config.json", samples_dir)

    logger.info("Thai Paradee 7.88M ONNX model generation complete!")


if __name__ == "__main__":
    main()
s