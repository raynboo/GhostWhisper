#!/usr/bin/env python3
"""Convert an HDF5 magnitude trace to WAV and an optional Mel spectrogram."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import h5py
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt


PROJECT_DIR = Path(__file__).resolve().parent


def display_path(path: Path) -> str:
    """Return an artifact-relative path without exposing a host filesystem path."""
    try:
        return str(path.resolve().relative_to(PROJECT_DIR))
    except ValueError:
        return path.name


@dataclass(frozen=True)
class ConversionConfig:
    input_path: Path
    output_dir: Path
    output_prefix: str
    magnitude_dataset: str
    input_rate_hz: float | None
    target_rate_hz: float
    resample_type: str
    pre_resample_lowpass_hz: float | None
    lowpass_hz: float | None
    lowpass_order: int
    remove_dc: bool
    peak_normalize: bool
    peak_headroom: float
    save_as_int16: bool
    plot_mel: bool
    n_mels: int
    n_fft: int
    hop_length: int
    fmin_hz: float
    fmax_hz: float | None

    def validate(self) -> None:
        if not self.input_path.is_file():
            raise ValueError(f"input file does not exist: {self.input_path}")
        if self.input_rate_hz is not None and self.input_rate_hz <= 0:
            raise ValueError("input sample rate must be positive")
        if self.target_rate_hz <= 0:
            raise ValueError("target sample rate must be positive")
        if self.lowpass_order < 1:
            raise ValueError("low-pass order must be at least one")
        if (
            self.pre_resample_lowpass_hz is not None
            and self.pre_resample_lowpass_hz <= 0
        ):
            raise ValueError("pre-resampling low-pass cutoff must be positive")
        if self.lowpass_hz is not None and not 0 < self.lowpass_hz < self.target_rate_hz / 2:
            raise ValueError("low-pass cutoff must be between zero and the target Nyquist rate")
        if not 0 < self.peak_headroom <= 1:
            raise ValueError("peak headroom must be in (0, 1]")
        if self.n_mels < 1 or self.n_fft < 2 or self.hop_length < 1:
            raise ValueError("Mel parameters must be positive")
        if self.fmin_hz < 0:
            raise ValueError("minimum Mel frequency must not be negative")


def load_h5_magnitude(
    path: Path,
    dataset: str,
    input_rate_hz: float | None,
) -> tuple[np.ndarray, float]:
    """Load a one-dimensional magnitude trace and determine its sample rate."""
    with h5py.File(path, "r") as handle:
        if dataset not in handle:
            raise KeyError(f"HDF5 dataset not found: {dataset}")
        values = np.asarray(handle[dataset][:], dtype=np.float32).squeeze()
        stored_rate = handle.attrs.get("sample_rate_hz")
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"dataset {dataset!r} must be a one-dimensional non-empty trace")
    rate = input_rate_hz if input_rate_hz is not None else stored_rate
    if rate is None:
        raise ValueError("input sample rate is missing; pass --input-rate")
    rate = float(rate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(f"invalid input sample rate: {rate!r}")
    if not np.all(np.isfinite(values)):
        raise ValueError("input magnitude trace contains NaN or infinite values")
    return values, rate


def remove_dc(values: np.ndarray, enabled: bool) -> np.ndarray:
    """Remove the mean so processing focuses on time-varying modulation."""
    if not enabled:
        return values.astype(np.float32, copy=False)
    return (values - np.mean(values)).astype(np.float32, copy=False)


def resample_with_antialias(
    values: np.ndarray,
    input_rate_hz: float,
    target_rate_hz: float,
    resample_type: str,
) -> np.ndarray:
    """Resample with librosa; SoXR modes include anti-alias filtering."""
    if input_rate_hz == target_rate_hz:
        return values.astype(np.float32, copy=False)
    result = librosa.resample(
        values.astype(np.float32, copy=False),
        orig_sr=input_rate_hz,
        target_sr=target_rate_hz,
        res_type=resample_type,
    )
    return result.astype(np.float32, copy=False)


def apply_optional_lowpass(
    values: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float | None,
    order: int,
) -> np.ndarray:
    """Apply an optional zero-phase low-pass filter at the specified sample rate."""
    if cutoff_hz is None:
        return values.astype(np.float32, copy=False)
    if not 0 < cutoff_hz < sample_rate_hz / 2:
        raise ValueError(
            f"low-pass cutoff must be between zero and Nyquist ({sample_rate_hz / 2:g} Hz)"
        )
    sos = butter(order, cutoff_hz / (sample_rate_hz / 2), btype="low", output="sos")
    return sosfiltfilt(sos, values).astype(np.float32, copy=False)


def to_wav_samples(
    values: np.ndarray,
    peak_normalize: bool,
    peak_headroom: float,
    save_as_int16: bool,
) -> np.ndarray:
    """Convert floating-point samples to the selected WAV representation."""
    result = values.astype(np.float32, copy=True)
    if peak_normalize:
        peak = float(np.max(np.abs(result)))
        if peak > 0:
            result *= peak_headroom / peak
    result = np.clip(result, -1.0, 1.0)
    return np.int16(result * 32767) if save_as_int16 else result


def save_mel_plot(
    values: np.ndarray,
    sample_rate_hz: float,
    output_path: Path,
    config: ConversionConfig,
) -> None:
    """Compute and save a Mel power spectrogram for visual inspection."""
    nyquist = sample_rate_hz / 2
    fmax_hz = config.fmax_hz if config.fmax_hz is not None else min(20_000.0, nyquist * 0.95)
    if not config.fmin_hz < fmax_hz < nyquist:
        raise ValueError(
            f"Mel frequencies must satisfy fmin < fmax < Nyquist ({nyquist:g} Hz)"
        )
    spectrum = librosa.feature.melspectrogram(
        y=values,
        sr=sample_rate_hz,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin_hz,
        fmax=fmax_hz,
        power=2.0,
    )
    spectrum_db = librosa.power_to_db(spectrum, ref=np.max)
    figure, axis = plt.subplots(figsize=(12, 5))
    image = librosa.display.specshow(
        spectrum_db,
        sr=sample_rate_hz,
        hop_length=config.hop_length,
        x_axis="time",
        y_axis="mel",
        fmin=config.fmin_hz,
        fmax=fmax_hz,
        cmap="magma",
        ax=axis,
    )
    figure.colorbar(image, ax=axis, format="%+2.0f dB")
    axis.set_title(f"Mel spectrogram ({config.fmin_hz:g}-{fmax_hz:g} Hz)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an HDF5 magnitude trace to WAV and an optional Mel plot."
    )
    parser.add_argument("input", type=Path, help="HDF5 input containing a magnitude dataset")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "converted_audio")
    parser.add_argument("--output-prefix", default="magnitude_audio")
    parser.add_argument("--dataset", default="magnitude")
    parser.add_argument(
        "--input-rate",
        type=float,
        metavar="SPS",
        help="override the sample_rate_hz HDF5 attribute",
    )
    parser.add_argument("--target-rate", type=float, default=48_000.0, metavar="SPS")
    parser.add_argument("--resample-type", default="soxr_hq")
    parser.add_argument(
        "--pre-resample-lowpass",
        type=float,
        default=100_000.0,
        metavar="HZ",
        help="input-rate low-pass cutoff applied before resampling (default: 100000)",
    )
    parser.add_argument(
        "--no-pre-resample-lowpass",
        action="store_const",
        const=None,
        dest="pre_resample_lowpass",
        help="disable the input-rate low-pass filter",
    )
    parser.add_argument("--lowpass", type=float, metavar="HZ")
    parser.add_argument("--lowpass-order", type=int, default=8)
    parser.add_argument("--keep-dc", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--peak-headroom", type=float, default=0.98)
    parser.add_argument("--float-wav", action="store_true")
    parser.add_argument("--no-mel", action="store_true")
    parser.add_argument("--n-mels", type=int, default=256)
    parser.add_argument("--n-fft", type=int, default=4096)
    parser.add_argument("--hop-length", type=int, default=480)
    parser.add_argument("--fmin", type=float, default=20.0, metavar="HZ")
    parser.add_argument("--fmax", type=float, metavar="HZ")
    return parser


def config_from_args(args: argparse.Namespace) -> ConversionConfig:
    config = ConversionConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        magnitude_dataset=args.dataset,
        input_rate_hz=args.input_rate,
        target_rate_hz=args.target_rate,
        resample_type=args.resample_type,
        pre_resample_lowpass_hz=args.pre_resample_lowpass,
        lowpass_hz=args.lowpass,
        lowpass_order=args.lowpass_order,
        remove_dc=not args.keep_dc,
        peak_normalize=not args.no_normalize,
        peak_headroom=args.peak_headroom,
        save_as_int16=not args.float_wav,
        plot_mel=not args.no_mel,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        fmin_hz=args.fmin,
        fmax_hz=args.fmax,
    )
    config.validate()
    return config


def run(config: ConversionConfig) -> tuple[Path, Path | None]:
    raw, input_rate_hz = load_h5_magnitude(
        config.input_path,
        config.magnitude_dataset,
        config.input_rate_hz,
    )
    processed = remove_dc(raw, config.remove_dc)
    processed = apply_optional_lowpass(
        processed,
        input_rate_hz,
        config.pre_resample_lowpass_hz,
        config.lowpass_order,
    )
    processed = resample_with_antialias(
        processed,
        input_rate_hz,
        config.target_rate_hz,
        config.resample_type,
    )
    processed = apply_optional_lowpass(
        processed,
        config.target_rate_hz,
        config.lowpass_hz,
        config.lowpass_order,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{config.output_prefix}_{timestamp}"
    wav_path = config.output_dir / f"{stem}.wav"
    wavfile.write(
        wav_path,
        int(round(config.target_rate_hz)),
        to_wav_samples(
            processed,
            config.peak_normalize,
            config.peak_headroom,
            config.save_as_int16,
        ),
    )
    mel_path = None
    if config.plot_mel:
        mel_path = config.output_dir / f"{stem}_mel.png"
        save_mel_plot(processed, config.target_rate_hz, mel_path, config)

    print(f"Input: {display_path(config.input_path)}")
    print(f"Input rate: {input_rate_hz:g} S/s; samples: {raw.size:,}")
    if config.pre_resample_lowpass_hz is not None:
        print(f"Pre-resampling low-pass: {config.pre_resample_lowpass_hz:g} Hz")
    print(f"Output rate: {config.target_rate_hz:g} S/s; samples: {processed.size:,}")
    print(f"WAV: {display_path(wav_path)}")
    if mel_path is not None:
        print(f"Mel plot: {display_path(mel_path)}")
    return wav_path, mel_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        config = config_from_args(parser.parse_args(argv))
        run(config)
        return 0
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
