from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class StableTone:
    frequency_hz: float
    prominence_db: float
    stability_db: float
    occupancy: float
    notch_width_hz: float


def _normalize_peak(samples: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 1e-12:
        return samples.astype(np.float32)
    return (samples / peak).astype(np.float32)


def _group_contiguous(indices: np.ndarray) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    breaks = np.where(np.diff(indices) > 1)[0] + 1
    return [group for group in np.split(indices, breaks) if group.size]


def detect_stable_tones(
    samples: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = 8192,
    hop_length: int = 1024,
    fmin: float = 20.0,
    fmax: float = 8000.0,
    prominence_db: float = 10.0,
    max_stability_db: float = 9.0,
    min_occupancy: float = 0.85,
    local_median_bins: int = 61,
    min_spacing_hz: float = 45.0,
    max_tones: int = 16,
    notch_width_hz: float = 35.0,
) -> list[StableTone]:
    """Detect narrowband components that are prominent and stable over time."""
    if samples.size < n_fft:
        return []

    frequencies, _, stft = signal.stft(
        samples.astype(np.float64),
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(stft), 1e-12))
    keep = (frequencies >= fmin) & (frequencies <= min(fmax, sample_rate * 0.5 - 1.0))
    if not np.any(keep):
        return []

    median_db = np.median(magnitude_db, axis=1)
    p10_db = np.percentile(magnitude_db, 10, axis=1)
    p90_db = np.percentile(magnitude_db, 90, axis=1)
    stability_db = p90_db - p10_db

    kernel = max(3, local_median_bins | 1)
    local_floor = signal.medfilt(median_db, kernel_size=kernel)
    line_prominence = median_db - local_floor
    occupancy = np.mean(magnitude_db > (local_floor[:, None] + prominence_db * 0.5), axis=1)

    candidate_mask = (
        keep
        & (line_prominence >= prominence_db)
        & (stability_db <= max_stability_db)
        & (occupancy >= min_occupancy)
    )

    grouped = _group_contiguous(np.flatnonzero(candidate_mask))
    candidates: list[StableTone] = []
    for group in grouped:
        weights = np.maximum(line_prominence[group], 1e-6)
        center_hz = float(np.sum(frequencies[group] * weights) / np.sum(weights))
        best = int(group[np.argmax(line_prominence[group])])
        candidates.append(
            StableTone(
                frequency_hz=center_hz,
                prominence_db=float(line_prominence[best]),
                stability_db=float(stability_db[best]),
                occupancy=float(occupancy[best]),
                notch_width_hz=float(notch_width_hz),
            )
        )

    candidates.sort(key=lambda tone: tone.prominence_db, reverse=True)
    selected: list[StableTone] = []
    for tone in candidates:
        if all(abs(tone.frequency_hz - other.frequency_hz) >= min_spacing_hz for other in selected):
            selected.append(tone)
        if len(selected) >= max_tones:
            break
    return sorted(selected, key=lambda tone: tone.frequency_hz)


def apply_notch_filters(
    samples: np.ndarray,
    sample_rate: int,
    tones: list[StableTone],
    *,
    q_min: float = 8.0,
    q_max: float = 120.0,
    normalize_output: bool = True,
) -> np.ndarray:
    filtered = samples.astype(np.float64)
    for tone in tones:
        freq = float(tone.frequency_hz)
        if not (0.0 < freq < sample_rate * 0.5):
            continue
        q = float(np.clip(freq / max(tone.notch_width_hz, 1.0), q_min, q_max))
        b, a = signal.iirnotch(w0=freq, Q=q, fs=sample_rate)
        filtered = signal.filtfilt(b, a, filtered)
    filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
    return _normalize_peak(filtered) if normalize_output else filtered.astype(np.float32)


def filter_stable_tones(
    samples: np.ndarray,
    sample_rate: int,
    **kwargs: object,
) -> tuple[np.ndarray, list[StableTone]]:
    tones = detect_stable_tones(samples, sample_rate, **kwargs)
    return apply_notch_filters(samples, sample_rate, tones), tones
