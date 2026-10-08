#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise SystemExit(
            "PyTorch is not installed in this environment. Run:\n"
            "  python -m pip install -r environment/requirements-training.txt\n"
            "Then rerun this script."
        ) from exc
    raise

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ghostwhisper.mel import LogMelTransform
from ghostwhisper.models import build_mel_denoise_model
from ghostwhisper.training_data import PairWaveformDataset, read_pair_manifest, split_rows_by_clean_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline U-Net on noisy/clean log-mel pairs.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Synthetic pair manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/mel_unet_v2_librosa256_rel80_unet64_gradloss"),
        help="Directory for checkpoints, logs, and config.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-items", type=int, default=None, help="Limit rows for a quick smoke test.")
    parser.add_argument(
        "--model-arch",
        choices=["resunet", "unet"],
        default="resunet",
        help="Model architecture. 'unet' keeps the original baseline for old experiments.",
    )
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--n-fft", type=int, default=4096)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-mels", type=int, default=256)
    parser.add_argument("--fmin", type=float, default=20.0)
    parser.add_argument("--fmax", type=float, default=8000.0)
    parser.add_argument("--db-min", type=float, default=-80.0)
    parser.add_argument("--db-max", type=float, default=0.0)
    parser.add_argument(
        "--absolute-db",
        dest="relative_ref",
        action="store_false",
        help="Use absolute dB values instead of per-sample relative dB.",
    )
    parser.set_defaults(relative_ref=True)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--gradient-weight", type=float, default=0.35)
    parser.add_argument(
        "--high-freq-weight",
        type=float,
        default=0.0,
        help="Extra L1 weight for upper mel bins, useful when consonant/harmonic detail is over-smoothed.",
    )
    parser.add_argument(
        "--high-freq-start",
        type=float,
        default=0.55,
        help="Fraction of mel bins where high-frequency emphasis starts.",
    )
    parser.add_argument(
        "--real-domain-augment",
        action="store_true",
        help="Apply on-the-fly waveform perturbations to noisy training audio only.",
    )
    parser.add_argument(
        "--augment-prob",
        type=float,
        default=0.85,
        help="Per-sample probability of applying real-domain input augmentation.",
    )
    parser.add_argument(
        "--augment-strength",
        type=float,
        default=1.0,
        help="Scales notch/EQ/line-noise/additive-noise augmentation intensity.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume model and optimizer state from a checkpoint, continuing epoch numbering.",
    )
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


def make_loader(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    dataset = PairWaveformDataset(rows, sample_rate=args.sample_rate, clip_seconds=args.clip_seconds)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=shuffle,
    )


def augment_noisy_waveforms(
    waveforms: torch.Tensor,
    sample_rate: int,
    probability: float,
    strength: float,
) -> torch.Tensor:
    if probability <= 0.0 or strength <= 0.0:
        return waveforms
    augmented = waveforms.clone()
    batch, length = augmented.shape
    device = augmented.device
    dtype = augmented.dtype
    freqs = torch.fft.rfftfreq(length, d=1.0 / sample_rate, device=device)
    time_axis = torch.arange(length, device=device, dtype=dtype) / float(sample_rate)

    for index in range(batch):
        if torch.rand((), device=device) > probability:
            continue
        x = augmented[index]
        x = x * torch.empty((), device=device).uniform_(0.55, 1.35)

        spectrum = torch.fft.rfft(x)
        mask = torch.ones_like(freqs)

        if torch.rand((), device=device) < 0.75:
            low_cut = float(torch.empty((), device=device).uniform_(35.0, 220.0))
            high_cut = float(torch.empty((), device=device).uniform_(3200.0, 7900.0))
            low_roll = torch.sigmoid((freqs - low_cut) / max(low_cut * 0.12, 8.0))
            high_roll = torch.sigmoid((high_cut - freqs) / max(high_cut * 0.04, 30.0))
            mask = mask * low_roll * high_roll

        notch_count = int(torch.randint(1, 5, (1,), device=device).item())
        for _ in range(notch_count):
            center = float(torch.empty((), device=device).uniform_(180.0, 7600.0))
            width = float(torch.empty((), device=device).uniform_(18.0, 120.0)) * strength
            depth = float(torch.empty((), device=device).uniform_(0.25, 0.85)) * min(strength, 1.4)
            notch = 1.0 - min(depth, 0.95) * torch.exp(-0.5 * ((freqs - center) / max(width, 1.0)).square())
            mask = mask * notch

        if torch.rand((), device=device) < 0.7:
            slope_db = float(torch.empty((), device=device).uniform_(-7.0, 7.0)) * strength
            octaves = torch.log2(torch.clamp(freqs, min=40.0) / 1000.0)
            eq = torch.pow(torch.tensor(10.0, device=device), (slope_db * octaves).clamp(-12.0, 12.0) / 20.0)
            mask = mask * eq

        x = torch.fft.irfft(spectrum * mask.to(spectrum.dtype), n=length)
        rms = x.square().mean().sqrt().clamp_min(1e-5)

        if torch.rand((), device=device) < 0.9:
            tones = torch.zeros_like(x)
            tone_count = int(torch.randint(1, 5, (1,), device=device).item())
            for _ in range(tone_count):
                freq = torch.empty((), device=device).uniform_(220.0, 7600.0)
                phase = torch.empty((), device=device).uniform_(0.0, 2.0 * torch.pi)
                amp = rms * torch.empty((), device=device).uniform_(0.015, 0.10) * strength
                tones = tones + amp * torch.sin(2.0 * torch.pi * freq * time_axis + phase)
            x = x + tones

        if torch.rand((), device=device) < 0.65:
            noise = torch.randn_like(x)
            noise_spec = torch.fft.rfft(noise)
            tilt = torch.clamp(freqs / 1000.0, min=0.2).sqrt()
            noise = torch.fft.irfft(noise_spec * tilt.to(noise_spec.dtype), n=length)
            noise = noise / noise.square().mean().sqrt().clamp_min(1e-5)
            snr_db = torch.empty((), device=device).uniform_(6.0, 24.0)
            x = x + noise * rms * torch.pow(torch.tensor(10.0, device=device), -snr_db / 20.0) * strength

        if torch.rand((), device=device) < 0.35:
            drive = torch.empty((), device=device).uniform_(1.05, 2.2)
            x = torch.tanh(drive * x) / torch.tanh(drive)

        peak = x.abs().amax().clamp_min(1e-5)
        if peak > 1.0:
            x = x / peak
        augmented[index] = x.clamp(-1.0, 1.0)

    return augmented


def run_epoch(
    model: nn.Module,
    mel_transform: LogMelTransform,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch: int,
    phase: str,
    l1_weight: float,
    gradient_weight: float,
    high_freq_weight: float,
    high_freq_start: float,
    real_domain_augment: bool,
    augment_prob: float,
    augment_strength: float,
    sample_rate: int,
    log_every: int = 25,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_batches = 0
    start = time.time()
    loss_fn = nn.L1Loss()

    for step, batch in enumerate(loader, start=1):
        noisy_wave = batch["noisy"].to(device, non_blocking=True)
        clean_wave = batch["clean"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            if training and real_domain_augment:
                noisy_wave = augment_noisy_waveforms(noisy_wave, sample_rate, augment_prob, augment_strength)
            noisy_mel = mel_transform(noisy_wave)
            clean_mel = mel_transform(clean_wave)
            pred_mel = model(noisy_mel)
            l1_loss = loss_fn(pred_mel, clean_mel)
            freq_grad_loss = loss_fn(pred_mel[:, :, 1:, :] - pred_mel[:, :, :-1, :], clean_mel[:, :, 1:, :] - clean_mel[:, :, :-1, :])
            time_grad_loss = loss_fn(pred_mel[:, :, :, 1:] - pred_mel[:, :, :, :-1], clean_mel[:, :, :, 1:] - clean_mel[:, :, :, :-1])
            high_freq_loss = pred_mel.new_tensor(0.0)
            if high_freq_weight > 0:
                start_bin = int(pred_mel.shape[-2] * high_freq_start)
                start_bin = min(max(start_bin, 0), pred_mel.shape[-2] - 1)
                bins = pred_mel.shape[-2] - start_bin
                weights = torch.linspace(1.0, 2.5, bins, device=pred_mel.device, dtype=pred_mel.dtype)
                weights = weights.view(1, 1, bins, 1)
                high_freq_loss = torch.mean(torch.abs(pred_mel[:, :, start_bin:, :] - clean_mel[:, :, start_bin:, :]) * weights)
            loss = l1_weight * l1_loss + gradient_weight * (freq_grad_loss + time_grad_loss) + high_freq_weight * high_freq_loss
            if not torch.isfinite(loss):
                pair_ids = batch.get("pair_id", [])
                raise RuntimeError(
                    f"Non-finite loss at phase={phase} epoch={epoch} step={step}: "
                    f"loss={float(loss.detach().cpu())} pair_ids={list(pair_ids)}"
                )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_batches += 1
        if step == 1 or step % log_every == 0 or step == len(loader):
            elapsed = max(time.time() - start, 1e-6)
            print(
                f"[{phase}] epoch={epoch} step={step}/{len(loader)} "
                f"loss={total_loss / total_batches:.5f} speed={step / elapsed:.2f} batch/s",
                flush=True,
            )

    return total_loss / max(total_batches, 1)


def append_metrics(path: Path, row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_best_val(metrics_path: Path) -> float:
    if not metrics_path.exists():
        return math.inf
    with metrics_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return math.inf
    return min(float(row["val_loss"]) for row in rows if row.get("val_loss"))


def json_safe_config(values: dict[str, object]) -> dict[str, object]:
    safe: dict[str, object] = {}
    for key, value in values.items():
        if isinstance(value, Path):
            safe[key] = str(value)
        else:
            safe[key] = value
    return safe


def main() -> None:
    args = parse_args()
    if args.init_checkpoint and args.resume_checkpoint:
        raise ValueError('Choose initialization OR resume, not both')
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_pair_manifest(args.manifest)
    if args.max_items is not None:
        rows = rows[: args.max_items]
    train_rows, val_rows = split_rows_by_clean_id(rows, args.val_ratio, args.seed)

    device = select_device(args.device)
    config = json_safe_config(vars(args).copy())
    config["device_resolved"] = str(device)
    config["num_rows"] = len(rows)
    config["num_train_rows"] = len(train_rows)
    config["num_val_rows"] = len(val_rows)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"device={device}", flush=True)
    print(f"manifest={args.manifest}", flush=True)
    print(f"train_rows={len(train_rows)} val_rows={len(val_rows)}", flush=True)
    print(f"output_dir={args.output_dir}", flush=True)

    train_loader = make_loader(train_rows, args, shuffle=True)
    val_loader = make_loader(val_rows, args, shuffle=False)
    mel_transform = LogMelTransform(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        db_min=args.db_min,
        db_max=args.db_max,
        relative_ref=args.relative_ref,
    ).to(device)
    model = build_mel_denoise_model(args.model_arch, base_channels=args.base_channels).to(device)
    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device, weights_only=True)['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    metrics_path = args.output_dir / "metrics.csv"
    start_epoch = 1
    best_val = read_best_val(metrics_path)
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        start_epoch = int(checkpoint["epoch"]) + 1
        if metrics_path.exists():
            best_val = min(best_val, float(checkpoint.get("val_loss", math.inf)))
        print(
            f"resume_checkpoint={args.resume_checkpoint} start_epoch={start_epoch} best_val={best_val:.5f}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(
            model,
            mel_transform,
            train_loader,
            optimizer,
            device,
            epoch,
            "train",
            args.l1_weight,
            args.gradient_weight,
            args.high_freq_weight,
            args.high_freq_start,
            args.real_domain_augment,
            args.augment_prob,
            args.augment_strength,
            args.sample_rate,
        )
        with torch.no_grad():
            val_loss = run_epoch(
                model,
                mel_transform,
                val_loader,
                None,
                device,
                epoch,
                "val",
                args.l1_weight,
                args.gradient_weight,
                args.high_freq_weight,
                args.high_freq_start,
                False,
                0.0,
                0.0,
                args.sample_rate,
                log_every=10,
            )

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": args.lr}
        append_metrics(metrics_path, row)
        print(
            f"[epoch] {epoch}/{args.epochs} train_loss={train_loss:.5f} val_loss={val_loss:.5f}",
            flush=True,
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        if epoch % args.save_every == 0:
            torch.save(checkpoint, args.output_dir / f"checkpoint_epoch_{epoch:03d}.pt")
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, args.output_dir / "best.pt")
            print(f"[checkpoint] best.pt updated val_loss={best_val:.5f}", flush=True)


if __name__ == "__main__":
    main()
