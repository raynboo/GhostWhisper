#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ghostwhisper.training_data import PairWaveformDataset, read_pair_manifest, split_rows_by_clean_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a complex-STFT U-Net waveform restoration model.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--compress-power", type=float, default=0.30)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--complex-weight", type=float, default=1.0)
    parser.add_argument("--mag-weight", type=float, default=0.35)
    parser.add_argument("--mrstft-weight", type=float, default=0.8)
    parser.add_argument("--highband-weight", type=float, default=1.2)
    parser.add_argument("--silence-weight", type=float, default=0.08)
    parser.add_argument("--sisdr-weight", type=float, default=0.12)
    parser.add_argument("--content-loss-weight", type=float, default=0.0)
    parser.add_argument("--content-model", type=str, help="User-supplied local content encoder directory")
    parser.add_argument("--content-layer", type=int, default=6)
    parser.add_argument("--content-sample-rate", type=int, default=16000)
    parser.add_argument("--content-loss-mode", choices=["cosine", "l1", "cosine_l1"], default="cosine")
    parser.add_argument("--content-loss-every", type=int, default=1)
    parser.add_argument("--content-local-files-only", action="store_true")
    parser.add_argument("--content-loss-mask", choices=["none", "speech"], default="none")
    parser.add_argument("--content-mask-quantile", type=float, default=0.35)
    parser.add_argument("--ctc-loss-weight", type=float, default=0.0)
    parser.add_argument("--ctc-model", type=str, help="User-supplied local CTC model directory")
    parser.add_argument("--ctc-sample-rate", type=int, default=16000)
    parser.add_argument("--ctc-loss-every", type=int, default=1)
    parser.add_argument("--ctc-local-files-only", action="store_true")
    parser.add_argument("--ctc-loss-clamp", type=float, default=0.0)
    parser.add_argument("--highband-start-hz", type=float, default=2000.0)
    parser.add_argument("--silence-quantile", type=float, default=0.28)
    parser.add_argument("--output-mode", choices=["direct", "residual"], default="direct")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--input-mag-anchor-weight", type=float, default=0.0)
    parser.add_argument("--input-highband-anchor-weight", type=float, default=0.0)
    parser.add_argument("--spec-grad-weight", type=float, default=0.0)
    parser.add_argument("--split-mode", choices=["row", "clean_id"], default="clean_id")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=4)
    parser.add_argument('--init-checkpoint', type=Path, help='Initialize weights only; new optimizer and epoch 1')
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stable_bucket(text: str) -> int:
    import hashlib

    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) % 10_000


def split_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if any(r.get('split') for r in rows) or args.split_mode == 'clean_id':
        return split_rows_by_clean_id(rows, args.val_ratio, args.seed)
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    threshold = int(args.val_ratio * 10_000)
    for row in rows:
        key_name = "clean_clip_id" if args.split_mode == "clean_id" else "pair_id"
        key = f"{args.seed}:{row.get(key_name, row.get('pair_id', ''))}"
        if stable_bucket(key) < threshold:
            val_rows.append(row)
        else:
            train_rows.append(row)
    if not val_rows and rows:
        val_rows = rows[: max(1, int(len(rows) * args.val_ratio))]
        train_rows = rows[len(val_rows) :]
    return train_rows, val_rows


def group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            group_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Dropout2d(dropout),
            group_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class STFTUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, depth: int = 4, dropout: float = 0.05) -> None:
        super().__init__()
        self.depth = depth
        channels = [base_channels * (2**idx) for idx in range(depth)]
        self.input = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for idx, channel in enumerate(channels):
            self.down_blocks.append(nn.Sequential(ResBlock(channel, dropout), ResBlock(channel, dropout)))
            if idx < depth - 1:
                self.downsample.append(nn.Conv2d(channel, channels[idx + 1], 4, stride=2, padding=1))
        self.mid = nn.Sequential(ResBlock(channels[-1], dropout), ResBlock(channels[-1], dropout))
        self.upsample = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for idx in range(depth - 2, -1, -1):
            self.upsample.append(nn.Conv2d(channels[idx + 1], channels[idx], 3, padding=1))
            self.up_blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels[idx] * 2, channels[idx], 3, padding=1),
                    ResBlock(channels[idx], dropout),
                    ResBlock(channels[idx], dropout),
                )
            )
        self.output = nn.Sequential(group_norm(channels[0]), nn.SiLU(), nn.Conv2d(channels[0], 2, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        multiple = 2 ** (self.depth - 1)
        pad_h = (multiple - height % multiple) % multiple
        pad_w = (multiple - width % multiple) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.input(x)
        skips = []
        for idx, block in enumerate(self.down_blocks):
            x = block(x)
            skips.append(x)
            if idx < len(self.downsample):
                x = self.downsample[idx](x)
        x = self.mid(x)
        for up, block, skip in zip(self.upsample, self.up_blocks, reversed(skips[:-1]), strict=True):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = up(x)
            x = block(torch.cat([x, skip], dim=1))
        x = self.output(x)
        return x[:, :, :height, :width]


def stft(wave: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=wave.device, dtype=wave.dtype)
    return torch.stft(wave, n_fft=n_fft, hop_length=hop_length, window=window, center=True, return_complex=True)


def istft(spec: torch.Tensor, length: int, n_fft: int, hop_length: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=spec.device, dtype=torch.float32)
    return torch.istft(spec.to(torch.complex64), n_fft=n_fft, hop_length=hop_length, window=window, center=True, length=length)


def compress_spec(spec: torch.Tensor, power: float) -> torch.Tensor:
    mag = spec.abs().clamp_min(1e-8)
    comp = (mag**power) * (spec / mag)
    return torch.stack([comp.real, comp.imag], dim=1)


def decompress_spec(comp: torch.Tensor, power: float) -> torch.Tensor:
    real = comp[:, 0]
    imag = comp[:, 1]
    mag_c = torch.sqrt(real.square() + imag.square()).clamp_min(1e-8)
    mag = mag_c ** (1.0 / power)
    phase = torch.complex(real / mag_c, imag / mag_c)
    return phase * mag


def input_features(noisy_spec: torch.Tensor, power: float) -> torch.Tensor:
    comp = compress_spec(noisy_spec, power)
    logmag = torch.log1p(noisy_spec.abs())
    scale = logmag.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    logmag = (logmag / scale).unsqueeze(1)
    return torch.cat([comp, logmag], dim=1)


def si_sdr_loss(reference: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    reference = reference - reference.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    ref_energy = reference.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    projection = (estimate * reference).sum(dim=-1, keepdim=True) * reference / ref_energy
    noise = estimate - projection
    ratio = projection.square().sum(dim=-1) / noise.square().sum(dim=-1).clamp_min(1e-8)
    return -10.0 * torch.log10(ratio.clamp_min(1e-8)).mean() / 20.0


def mrstft_loss(reference: torch.Tensor, estimate: torch.Tensor, sample_rate: int, highband_start_hz: float) -> tuple[torch.Tensor, torch.Tensor]:
    total = reference.new_tensor(0.0)
    high_total = reference.new_tensor(0.0)
    configs = [(512, 128), (1024, 256), (2048, 512)]
    for n_fft, hop in configs:
        ref = stft(reference, n_fft, hop).abs()
        est = stft(estimate, n_fft, hop).abs()
        sc = torch.linalg.vector_norm(est - ref, dim=(-2, -1)) / torch.linalg.vector_norm(ref, dim=(-2, -1)).clamp_min(1e-8)
        log_l1 = F.l1_loss(torch.log1p(est), torch.log1p(ref))
        total = total + sc.mean() + log_l1
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate).to(reference.device)
        mask = freqs >= highband_start_hz
        if torch.any(mask):
            high_total = high_total + F.l1_loss(torch.log1p(est[:, mask, :]), torch.log1p(ref[:, mask, :]))
    return total / len(configs), high_total / len(configs)


def highband_logmag_l1(reference_spec: torch.Tensor, estimate_spec: torch.Tensor, sample_rate: int, highband_start_hz: float) -> torch.Tensor:
    freqs = torch.fft.rfftfreq((reference_spec.shape[-2] - 1) * 2, d=1.0 / sample_rate).to(reference_spec.device)
    mask = freqs >= highband_start_hz
    if not torch.any(mask):
        return reference_spec.real.new_tensor(0.0)
    return F.l1_loss(torch.log1p(estimate_spec.abs()[:, mask, :]), torch.log1p(reference_spec.abs()[:, mask, :]))


def spectral_gradient_loss(reference_spec: torch.Tensor, estimate_spec: torch.Tensor) -> torch.Tensor:
    ref_log = torch.log1p(reference_spec.abs())
    est_log = torch.log1p(estimate_spec.abs())
    freq = F.l1_loss(est_log[:, 1:, :] - est_log[:, :-1, :], ref_log[:, 1:, :] - ref_log[:, :-1, :])
    time = F.l1_loss(est_log[:, :, 1:] - est_log[:, :, :-1], ref_log[:, :, 1:] - ref_log[:, :, :-1])
    return 0.5 * (freq + time)


def silence_loss(clean_spec: torch.Tensor, pred_spec: torch.Tensor, quantile: float) -> torch.Tensor:
    clean_power = clean_spec.abs().square().mean(dim=1)
    threshold = torch.quantile(clean_power.detach().flatten(1), quantile, dim=1).view(-1, 1)
    mask = clean_power <= threshold
    if not torch.any(mask):
        return clean_power.new_tensor(0.0)
    pred_power = pred_spec.abs().square().mean(dim=1)
    return pred_power[mask].mean()


def resample_waveform(wave: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return wave
    try:
        import torchaudio.functional as AF

        return AF.resample(wave, source_rate, target_rate)
    except Exception:
        target_length = max(1, int(round(wave.shape[-1] * target_rate / source_rate)))
        return F.interpolate(wave.unsqueeze(1), size=target_length, mode="linear", align_corners=False).squeeze(1)


def normalize_for_content(wave: torch.Tensor) -> torch.Tensor:
    wave = wave - wave.mean(dim=-1, keepdim=True)
    scale = wave.std(dim=-1, keepdim=True).clamp_min(1e-4)
    return (wave / scale).clamp(-5.0, 5.0)


def normalize_asr_label(text: str) -> str:
    raw = text.strip()
    if re.fullmatch(r"[A-Z]{2,}", raw):
        text = " ".join(raw.lower())
    else:
        text = raw.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class FrozenContentEncoder(nn.Module):
    def __init__(self, model_name: str, layer: int, source_rate: int, target_rate: int, local_files_only: bool) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.model.eval()
        self.model.requires_grad_(False)
        self.layer = layer
        self.source_rate = source_rate
        self.target_rate = target_rate

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        wave_16k = normalize_for_content(resample_waveform(wave, self.source_rate, self.target_rate))
        output = self.model(wave_16k, output_hidden_states=True)
        hidden_states = output.hidden_states
        layer = self.layer
        if layer < 0:
            layer = len(hidden_states) + layer
        layer = max(0, min(layer, len(hidden_states) - 1))
        return hidden_states[layer]


def content_frame_mask(clean: torch.Tensor, frame_count: int, args: argparse.Namespace) -> torch.Tensor | None:
    if args.content_loss_mask == "none":
        return None
    wave_16k = resample_waveform(clean, args.sample_rate, args.content_sample_rate)
    env = wave_16k.abs().detach()
    if env.shape[-1] != frame_count:
        env = F.interpolate(env.unsqueeze(1), size=frame_count, mode="linear", align_corners=False).squeeze(1)
    threshold = torch.quantile(env, args.content_mask_quantile, dim=1, keepdim=True)
    mask = (env > threshold).float()
    empty = mask.sum(dim=1, keepdim=True) < 1.0
    if torch.any(empty):
        mask = torch.where(empty, torch.ones_like(mask), mask)
    return mask


def content_embedding_loss(
    encoder: FrozenContentEncoder | None,
    clean: torch.Tensor,
    estimate: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if encoder is None or args.content_loss_weight <= 0:
        return estimate.new_tensor(0.0)
    pred_feat = encoder(estimate)
    with torch.no_grad():
        clean_feat = encoder(clean)
    if pred_feat.shape[1] != clean_feat.shape[1]:
        pred_feat = F.interpolate(pred_feat.transpose(1, 2), size=clean_feat.shape[1], mode="linear", align_corners=False).transpose(1, 2)
    mask = content_frame_mask(clean, clean_feat.shape[1], args)
    if args.content_loss_mode == "l1":
        frame_loss = torch.mean(torch.abs(pred_feat - clean_feat), dim=-1)
        if mask is None:
            return frame_loss.mean()
        return torch.sum(frame_loss * mask) / mask.sum().clamp_min(1.0)
    pred_norm = F.normalize(pred_feat, dim=-1)
    clean_norm = F.normalize(clean_feat, dim=-1)
    cosine_frame = 1.0 - (pred_norm * clean_norm).sum(dim=-1)
    if mask is None:
        cosine_loss = cosine_frame.mean()
    else:
        cosine_loss = torch.sum(cosine_frame * mask) / mask.sum().clamp_min(1.0)
    if args.content_loss_mode == "cosine_l1":
        l1_frame = torch.mean(torch.abs(pred_feat - clean_feat), dim=-1)
        if mask is None:
            l1_loss = l1_frame.mean()
        else:
            l1_loss = torch.sum(l1_frame * mask) / mask.sum().clamp_min(1.0)
        return cosine_loss + 0.1 * l1_loss
    return cosine_loss


class FrozenCTCLoss(nn.Module):
    def __init__(self, model_name: str, source_rate: int, target_rate: int, local_files_only: bool) -> None:
        super().__init__()
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.processor = Wav2Vec2Processor.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name, local_files_only=local_files_only)
        self.model.config.ctc_zero_infinity = True
        self.model.eval()
        self.model.requires_grad_(False)
        self.source_rate = source_rate
        self.target_rate = target_rate

    def encode_labels(self, transcripts: list[str], device: torch.device) -> torch.Tensor | None:
        label_ids: list[torch.Tensor] = []
        for transcript in transcripts:
            normalized = normalize_asr_label(transcript)
            ids = self.processor.tokenizer(normalized).input_ids if normalized else []
            if ids:
                label_ids.append(torch.tensor(ids, dtype=torch.long, device=device))
        if len(label_ids) != len(transcripts):
            return None
        max_len = max(int(ids.numel()) for ids in label_ids)
        labels = torch.full((len(label_ids), max_len), -100, dtype=torch.long, device=device)
        for index, ids in enumerate(label_ids):
            labels[index, : ids.numel()] = ids
        return labels

    def forward(self, wave: torch.Tensor, transcripts: list[str]) -> torch.Tensor:
        labels = self.encode_labels(transcripts, wave.device)
        if labels is None:
            return wave.new_tensor(0.0)
        wave_16k = normalize_for_content(resample_waveform(wave, self.source_rate, self.target_rate))
        return self.model(wave_16k, labels=labels).loss


def run_epoch(
    model: nn.Module,
    content_encoder: FrozenContentEncoder | None,
    ctc_loss_model: FrozenCTCLoss | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
    phase: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    started = time.time()
    for step, batch in enumerate(loader, start=1):
        noisy = batch["noisy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        transcripts = list(batch.get("transcript", [""] * noisy.shape[0]))
        with torch.set_grad_enabled(training):
            noisy_spec = stft(noisy, args.n_fft, args.hop_length)
            clean_spec = stft(clean, args.n_fft, args.hop_length)
            noisy_comp = compress_spec(noisy_spec, args.compress_power)
            model_out = model(input_features(noisy_spec, args.compress_power))
            if args.output_mode == "residual":
                pred_comp = noisy_comp + args.residual_scale * model_out
            else:
                pred_comp = model_out
            clean_comp = compress_spec(clean_spec, args.compress_power)
            pred_spec = decompress_spec(pred_comp, args.compress_power)
            pred_wave = istft(pred_spec, clean.shape[-1], args.n_fft, args.hop_length)

            complex_l1 = F.l1_loss(pred_comp, clean_comp)
            mag_l1 = F.l1_loss(torch.log1p(pred_spec.abs()), torch.log1p(clean_spec.abs()))
            mr, high = mrstft_loss(clean, pred_wave, args.sample_rate, args.highband_start_hz)
            spec_grad = spectral_gradient_loss(clean_spec, pred_spec)
            input_mag_anchor = F.l1_loss(torch.log1p(pred_spec.abs()), torch.log1p(noisy_spec.abs()))
            input_high_anchor = highband_logmag_l1(noisy_spec, pred_spec, args.sample_rate, args.highband_start_hz)
            quiet = silence_loss(clean_spec, pred_spec, args.silence_quantile)
            sisdr = si_sdr_loss(clean, pred_wave)
            use_content = (
                content_encoder is not None
                and args.content_loss_weight > 0
                and (not training or args.content_loss_every <= 1 or step % args.content_loss_every == 0)
            )
            content = content_embedding_loss(content_encoder, clean, pred_wave, args) if use_content else clean.new_tensor(0.0)
            use_ctc = (
                ctc_loss_model is not None
                and args.ctc_loss_weight > 0
                and (not training or args.ctc_loss_every <= 1 or step % args.ctc_loss_every == 0)
            )
            ctc = ctc_loss_model(pred_wave, transcripts) if use_ctc else clean.new_tensor(0.0)
            if not torch.isfinite(ctc):
                ctc = clean.new_tensor(0.0)
            elif args.ctc_loss_clamp > 0:
                ctc = ctc.clamp(max=args.ctc_loss_clamp)
            loss = (
                args.complex_weight * complex_l1
                + args.mag_weight * mag_l1
                + args.mrstft_weight * mr
                + args.highband_weight * high
                + args.silence_weight * quiet
                + args.sisdr_weight * sisdr
                + args.content_loss_weight * content
                + args.ctc_loss_weight * ctc
                + args.spec_grad_weight * spec_grad
                + args.input_mag_anchor_weight * input_mag_anchor
                + args.input_highband_anchor_weight * input_high_anchor
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at {phase} epoch={epoch} step={step}")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        values = {
            "loss": float(loss.detach().cpu()),
            "complex": float(complex_l1.detach().cpu()),
            "mag": float(mag_l1.detach().cpu()),
            "mrstft": float(mr.detach().cpu()),
            "highband": float(high.detach().cpu()),
            "spec_grad": float(spec_grad.detach().cpu()),
            "input_mag_anchor": float(input_mag_anchor.detach().cpu()),
            "input_high_anchor": float(input_high_anchor.detach().cpu()),
            "silence": float(quiet.detach().cpu()),
            "sisdr": float(sisdr.detach().cpu()),
            "content": float(content.detach().cpu()),
            "ctc": float(ctc.detach().cpu()),
        }
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
        if step == 1 or step % 20 == 0 or step == len(loader):
            elapsed = max(time.time() - started, 1e-6)
            print(f"[{phase}] epoch={epoch} step={step}/{len(loader)} loss={totals['loss']/count:.5f} speed={step/elapsed:.2f} batch/s", flush=True)
    return {key: value / max(1, count) for key, value in totals.items()}


def append_metrics(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    for weight, model_path, label in [(args.content_loss_weight, args.content_model, 'content'),
                                      (args.ctc_loss_weight, args.ctc_model, 'ctc')]:
        if weight > 0 and (not model_path or not Path(model_path).is_dir()):
            raise SystemExit(f'--{label}-model must name an existing local directory when its loss is enabled')
    args.content_local_files_only = True
    args.ctc_local_files_only = True
    if args.init_checkpoint and args.resume_checkpoint:
        raise ValueError('Choose initialization OR resume, not both')
    torch.manual_seed(args.seed)
    rows = read_pair_manifest(args.manifest)
    if args.max_items is not None:
        rows = rows[: args.max_items]
    train_rows, val_rows = split_rows(rows, args)
    if not train_rows or not val_rows:
        raise SystemExit(f"bad split: train={len(train_rows)} val={len(val_rows)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"device_resolved": str(device), "num_rows": len(rows), "num_train_rows": len(train_rows), "num_val_rows": len(val_rows)})
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"device={device}", flush=True)
    print(f"manifest={args.manifest}", flush=True)
    print(f"train_rows={len(train_rows)} val_rows={len(val_rows)} split_mode={args.split_mode}", flush=True)
    print(f"output_dir={args.output_dir}", flush=True)

    train_loader = DataLoader(
        PairWaveformDataset(train_rows, args.sample_rate, args.clip_seconds),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        PairWaveformDataset(val_rows, args.sample_rate, args.clip_seconds),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = STFTUNet(base_channels=args.base_channels, depth=args.depth, dropout=args.dropout).to(device)
    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device, weights_only=True)['model'])
    content_encoder = None
    if args.content_loss_weight > 0:
        print(
            f"loading_content_encoder={args.content_model} layer={args.content_layer} "
            f"local_only={args.content_local_files_only}",
            flush=True,
        )
        content_encoder = FrozenContentEncoder(
            args.content_model,
            args.content_layer,
            args.sample_rate,
            args.content_sample_rate,
            args.content_local_files_only,
        ).to(device)
    ctc_loss_model = None
    if args.ctc_loss_weight > 0:
        print(
            f"loading_ctc_loss_model={args.ctc_model} local_only={args.ctc_local_files_only}",
            flush=True,
        )
        ctc_loss_model = FrozenCTCLoss(
            args.ctc_model,
            args.sample_rate,
            args.ctc_sample_rate,
            args.ctc_local_files_only,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_epoch = 1
    best_loss = math.inf
    metrics_path = args.output_dir / "metrics.csv"
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"resume_checkpoint={args.resume_checkpoint} start_epoch={start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train = run_epoch(model, content_encoder, ctc_loss_model, train_loader, optimizer, args, device, epoch, "train")
        with torch.no_grad():
            val = run_epoch(model, content_encoder, ctc_loss_model, val_loader, None, args, device, epoch, "val")
        row: dict[str, object] = {"epoch": epoch, "lr": args.lr}
        row.update({f"train_{key}": value for key, value in train.items()})
        row.update({f"val_{key}": value for key, value in val.items()})
        append_metrics(metrics_path, row)
        print(f"[epoch] {epoch}/{args.epochs} train_loss={train['loss']:.5f} val_loss={val['loss']:.5f}", flush=True)
        checkpoint = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config, "train": train, "val": val}
        if epoch % args.save_every == 0:
            torch.save(checkpoint, args.output_dir / f"checkpoint_epoch_{epoch:03d}.pt")
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            torch.save(checkpoint, args.output_dir / "best.pt")
            print(f"[checkpoint] best.pt val_loss={best_loss:.5f}", flush=True)


if __name__ == "__main__":
    main()
