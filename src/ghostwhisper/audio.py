from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import soundfile as sf
from scipy import signal


def normalize_peak(samples: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(samples)))
    if peak <= 0:
        return samples.astype(np.float32)
    return (samples / peak).astype(np.float32)


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(path)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    samples = samples.astype(np.float64)
    return normalize_peak(samples), int(sample_rate)


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return normalize_peak(samples.astype(np.float64))
    gcd = math.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd
    resampled = signal.resample_poly(samples.astype(np.float64), up=up, down=down)
    return normalize_peak(resampled)


def trim_or_pad(samples: np.ndarray, target_length: int) -> np.ndarray:
    if len(samples) >= target_length:
        return samples[:target_length].astype(np.float32)
    padded = np.zeros(target_length, dtype=np.float32)
    padded[: len(samples)] = samples.astype(np.float32)
    return padded


def load_h5_magnitude(path: Path, max_seconds: float | None = None) -> tuple[np.ndarray, int]:
    with h5py.File(path, "r") as handle:
        magnitude = handle["magnitude"][:].astype(np.float64)
        sample_rate = int(round(float(handle.attrs["sample_rate_hz"])))
    if max_seconds is not None:
        magnitude = magnitude[: int(max_seconds * sample_rate)]
    magnitude -= np.mean(magnitude)
    return normalize_peak(magnitude), sample_rate


def lowpass_and_resample_magnitude(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int,
    cutoff_ratio: float = 0.45,
    order: int = 8,
) -> np.ndarray:
    if target_rate >= source_rate:
        return normalize_peak(samples.astype(np.float64))
    cutoff_hz = 0.5 * target_rate * cutoff_ratio
    sos = signal.butter(order, cutoff_hz, btype="lowpass", fs=source_rate, output="sos")
    filtered = signal.sosfiltfilt(sos, samples.astype(np.float64))
    return resample_audio(filtered, source_rate, target_rate)
