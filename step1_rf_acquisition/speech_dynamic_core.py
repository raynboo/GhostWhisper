#!/usr/bin/env python3
"""Short-window speech-dynamic scoring for one in-memory magnitude dwell."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import signal


EPS = np.finfo(np.float64).tiny


class DynamicScoringError(ValueError):
    """Raised when a dwell cannot be scored safely."""


@dataclass(frozen=True)
class DynamicScoreConfig:
    target_rate_hz: float = 48_000.0
    pre_resample_lowpass_hz: float = 100_000.0
    lowpass_order: int = 8
    window_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 2048
    audio_band_min_hz: float = 80.0
    audio_band_max_hz: float = 4_000.0
    baseline_percentile: float = 20.0
    scale_percentile: float = 80.0
    spectral_activity_percentile: float = 90.0
    minimum_scale_db: float = 1.0
    lower_percentile: float = 5.0
    upper_percentile: float = 95.0

    def validate(self) -> None:
        numeric = (
            self.target_rate_hz,
            self.pre_resample_lowpass_hz,
            self.window_ms,
            self.hop_ms,
            self.audio_band_min_hz,
            self.audio_band_max_hz,
            self.baseline_percentile,
            self.scale_percentile,
            self.spectral_activity_percentile,
            self.minimum_scale_db,
            self.lower_percentile,
            self.upper_percentile,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise DynamicScoringError("all scoring settings must be finite")
        if self.target_rate_hz <= 0 or self.pre_resample_lowpass_hz <= 0:
            raise DynamicScoringError("sample rate and low-pass cutoff must be positive")
        if self.lowpass_order < 1:
            raise DynamicScoringError("low-pass order must be at least one")
        if self.window_ms <= 0 or self.hop_ms <= 0:
            raise DynamicScoringError("window and hop durations must be positive")
        if self.n_fft < 2 or self.n_fft & (self.n_fft - 1):
            raise DynamicScoringError("FFT size must be a power of two")
        if not 0 < self.audio_band_min_hz < self.audio_band_max_hz < self.target_rate_hz / 2:
            raise DynamicScoringError("the speech band must lie inside the output Nyquist range")
        if not 0 <= self.baseline_percentile < self.scale_percentile <= 100:
            raise DynamicScoringError("expected baseline percentile below scale percentile")
        if not 0 <= self.spectral_activity_percentile <= 100:
            raise DynamicScoringError("spectral activity percentile must be in [0, 100]")
        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise DynamicScoringError("invalid dynamic-score percentiles")
        if self.minimum_scale_db <= 0:
            raise DynamicScoringError("minimum spectral scale must be positive")
        window_samples = int(round(self.window_ms * self.target_rate_hz / 1000.0))
        if window_samples < 2 or window_samples > self.n_fft:
            raise DynamicScoringError("window size must contain 2..n_fft samples")


@dataclass
class DynamicScoreResult:
    score: float
    time_s: np.ndarray
    activity: np.ndarray
    frequency_hz: np.ndarray
    spectral_baseline_db: np.ndarray
    spectral_scale_db: np.ndarray
    processed_magnitude: np.ndarray
    output_rate_hz: float


def preprocess_magnitude(
    values: np.ndarray,
    input_rate_hz: float,
    config: DynamicScoreConfig,
) -> tuple[np.ndarray, float]:
    """Remove the dwell mean, low-pass at 100 kHz, and resample to 48 kHz."""
    config.validate()
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or source.size < 2:
        raise DynamicScoringError("magnitude input must be a non-empty vector")
    if not np.all(np.isfinite(source)):
        raise DynamicScoringError("magnitude input contains NaN or infinite values")
    if not math.isfinite(input_rate_hz) or input_rate_hz <= 0:
        raise DynamicScoringError("input sample rate must be positive and finite")
    if config.pre_resample_lowpass_hz >= input_rate_hz / 2:
        raise DynamicScoringError("100 kHz low-pass requires an input rate above 200 kS/s")

    centered = source - float(np.mean(source))
    sos = signal.butter(
        config.lowpass_order,
        config.pre_resample_lowpass_hz,
        btype="lowpass",
        fs=input_rate_hz,
        output="sos",
    )
    try:
        filtered = signal.sosfiltfilt(sos, centered)
    except ValueError as exc:
        raise DynamicScoringError(f"dwell is too short for zero-phase filtering: {exc}") from exc

    if math.isclose(input_rate_hz, config.target_rate_hz, rel_tol=0.0, abs_tol=1e-9):
        return filtered.astype(np.float32), float(config.target_rate_hz)
    ratio = Fraction(config.target_rate_hz / input_rate_hz).limit_denominator(100_000)
    actual_rate_hz = input_rate_hz * ratio.numerator / ratio.denominator
    if not math.isclose(actual_rate_hz, config.target_rate_hz, rel_tol=1e-9, abs_tol=1e-6):
        raise DynamicScoringError(
            f"cannot represent resampling ratio {input_rate_hz:g} -> "
            f"{config.target_rate_hz:g} S/s accurately"
        )
    processed = signal.resample_poly(
        filtered,
        ratio.numerator,
        ratio.denominator,
        window=("kaiser", 8.6),
    )
    return processed.astype(np.float32), float(actual_rate_hz)


def score_magnitude_dwell(
    values: np.ndarray,
    input_rate_hz: float,
    config: DynamicScoreConfig,
) -> DynamicScoreResult:
    """Return one continuous score; no binary detection threshold is applied."""
    processed, output_rate_hz = preprocess_magnitude(values, input_rate_hz, config)
    window_samples = int(round(config.window_ms * output_rate_hz / 1000.0))
    hop_samples = int(round(config.hop_ms * output_rate_hz / 1000.0))
    if processed.size < window_samples:
        raise DynamicScoringError(
            f"processed dwell is too short for one {config.window_ms:g} ms window"
        )

    frames = np.lib.stride_tricks.sliding_window_view(processed, window_samples)[::hop_samples]
    taper = signal.windows.hann(window_samples, sym=False).astype(np.float64)
    transformed = np.fft.rfft(frames * taper, n=config.n_fft, axis=1)
    power = np.abs(transformed) ** 2 / max(float(np.sum(taper * taper)), EPS)
    power_db = 10.0 * np.log10(np.maximum(power, EPS))
    frequency_hz = np.fft.rfftfreq(config.n_fft, d=1.0 / output_rate_hz)
    speech_mask = (
        (frequency_hz >= config.audio_band_min_hz)
        & (frequency_hz <= config.audio_band_max_hz)
    )
    if not np.any(speech_mask):
        raise DynamicScoringError("the configured speech band contains no FFT bins")

    baseline_db = np.percentile(power_db, config.baseline_percentile, axis=0)
    scale_db = np.maximum(
        np.percentile(power_db, config.scale_percentile, axis=0) - baseline_db,
        config.minimum_scale_db,
    )
    normalized = (power_db - baseline_db[np.newaxis, :]) / scale_db[np.newaxis, :]
    activity = np.percentile(
        normalized[:, speech_mask],
        config.spectral_activity_percentile,
        axis=1,
    )
    low = float(np.percentile(activity, config.lower_percentile))
    high = float(np.percentile(activity, config.upper_percentile))
    time_s = (
        np.arange(frames.shape[0], dtype=np.float64) * hop_samples + window_samples / 2.0
    ) / output_rate_hz
    return DynamicScoreResult(
        score=high - low,
        time_s=time_s,
        activity=np.asarray(activity, dtype=np.float64),
        frequency_hz=frequency_hz,
        spectral_baseline_db=np.asarray(baseline_db, dtype=np.float64),
        spectral_scale_db=np.asarray(scale_db, dtype=np.float64),
        processed_magnitude=processed,
        output_rate_hz=float(output_rate_hz),
    )
