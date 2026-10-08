#!/usr/bin/env python3
"""Hardware-independent DSP for measured complex reflection ratio G(t)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal


EPS = np.finfo(np.float64).tiny


class ProcessingError(RuntimeError):
    pass


class CarrierNotFoundError(ProcessingError):
    pass


@dataclass(frozen=True)
class ProcessingSettings:
    output_rate_hz: float = 50e3
    lowpass_cutoff_hz: float = 10e3
    lowpass_order: int = 8
    transient_s: float = 0.005
    chunk_samples: int = 1_048_576
    forward_relative_floor: float = 0.05
    forward_absolute_floor: float = 1e-6

    def validate(self, input_rate_hz: float) -> int:
        values = (
            input_rate_hz,
            self.output_rate_hz,
            self.lowpass_cutoff_hz,
            self.transient_s,
            self.forward_relative_floor,
            self.forward_absolute_floor,
        )
        if not all(math.isfinite(value) for value in values):
            raise ProcessingError("all processing settings must be finite")
        if input_rate_hz <= 0 or self.output_rate_hz <= 0:
            raise ProcessingError("input and output rates must be positive")
        ratio = input_rate_hz / self.output_rate_hz
        decimation = int(round(ratio))
        if decimation < 1 or not math.isclose(ratio, decimation, rel_tol=0, abs_tol=1e-9):
            raise ProcessingError(
                "input rate must be an integer multiple of the requested output rate"
            )
        if not 0 < self.lowpass_cutoff_hz < self.output_rate_hz / 2:
            raise ProcessingError("low-pass cutoff must be below the output Nyquist frequency")
        if self.lowpass_order < 1 or self.chunk_samples < 1024:
            raise ProcessingError("filter order must be positive and chunk size >= 1024")
        if self.transient_s < 0:
            raise ProcessingError("transient duration must not be negative")
        if self.forward_relative_floor < 0 or self.forward_absolute_floor <= 0:
            raise ProcessingError("invalid Forward-channel ratio floor")
        return decimation


@dataclass(frozen=True)
class CarrierEstimate:
    frequency_hz: float
    snr_db: float
    peak_power_density: float
    noise_power_density: float
    expected_frequency_hz: float
    search_half_width_hz: float


@dataclass
class ReflectionResult:
    time_s: np.ndarray
    forward_complex: np.ndarray
    reverse_complex: np.ndarray
    g_complex: np.ndarray
    g_magnitude: np.ndarray
    g_phase: np.ndarray
    valid_ratio: np.ndarray
    forward_floor: float
    output_rate_hz: float
    decimation: int


@dataclass(frozen=True)
class ToneFit:
    r_squared: float
    correlation: float
    modulation_rms: float
    sine_coefficient: complex
    cosine_coefficient: complex
    offset: complex
    amplitude: float | None
    phase_rad: float | None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "r_squared": self.r_squared,
            "correlation": self.correlation,
            "modulation_rms": self.modulation_rms,
            "offset": _complex_dict(self.offset),
            "sine_coefficient": _complex_dict(self.sine_coefficient),
            "cosine_coefficient": _complex_dict(self.cosine_coefficient),
        }
        if self.amplitude is not None:
            result["amplitude"] = self.amplitude
        if self.phase_rad is not None:
            result["phase_rad"] = self.phase_rad
            result["phase_deg"] = math.degrees(self.phase_rad)
        return result


@dataclass
class ResponseAnalysis:
    metrics: dict[str, Any]
    g_filled: np.ndarray
    magnitude_detrended: np.ndarray
    phase_unwrapped: np.ndarray
    phase_detrended: np.ndarray
    magnitude_spectrum_hz: np.ndarray
    magnitude_psd: np.ndarray
    phase_spectrum_hz: np.ndarray
    phase_psd: np.ndarray


def _complex_dict(value: complex) -> dict[str, float]:
    return {"real": float(np.real(value)), "imag": float(np.imag(value))}


def average_power_spectrum(
    samples: Any,
    sample_rate_hz: float,
    nfft: int = 262_144,
    max_segments: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Average Hann-windowed periodograms from an ndarray or HDF5 dataset."""
    sample_count = int(len(samples))
    if sample_count < 1024:
        raise ProcessingError("at least 1024 samples are required for carrier estimation")
    if sample_rate_hz <= 0 or nfft < 1024 or nfft & (nfft - 1):
        raise ProcessingError("sample rate must be positive and nfft a power of two >= 1024")
    if max_segments < 1:
        raise ProcessingError("max_segments must be positive")

    segment_length = min(sample_count, nfft)
    if sample_count == segment_length:
        starts = np.array([0], dtype=np.int64)
    else:
        available = sample_count - segment_length
        starts = np.unique(
            np.linspace(0, available, num=min(max_segments, available + 1), dtype=np.int64)
        )
    window = signal.windows.hann(segment_length, sym=False).astype(np.float64)
    normalization = sample_rate_hz * float(np.sum(window * window))
    accumulated = np.zeros(nfft, dtype=np.float64)
    for start in starts:
        block = np.asarray(samples[int(start) : int(start) + segment_length], dtype=np.complex64)
        if not np.all(np.isfinite(block)):
            raise ProcessingError("raw IQ contains non-finite samples")
        block = block.astype(np.complex128, copy=False)
        block -= np.mean(block)
        transformed = np.fft.fft(block * window, n=nfft)
        accumulated += np.abs(transformed) ** 2 / normalization
    accumulated /= len(starts)
    frequencies = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate_hz))
    return frequencies, np.fft.fftshift(accumulated)


def estimate_carrier(
    frequencies_hz: np.ndarray,
    power_density: np.ndarray,
    expected_frequency_hz: float,
    search_half_width_hz: float = 20e3,
    minimum_snr_db: float = 10.0,
) -> CarrierEstimate:
    """Locate the Forward-channel CW in a bounded expected-IF search region."""
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    power = np.asarray(power_density, dtype=np.float64)
    if frequencies.shape != power.shape or frequencies.ndim != 1:
        raise ProcessingError("frequency and power arrays must be matching one-dimensional arrays")
    if not np.all(np.isfinite(power)) or search_half_width_hz <= 0:
        raise ProcessingError("invalid carrier spectrum or search width")
    search = np.abs(frequencies - expected_frequency_hz) <= search_half_width_hz
    if np.count_nonzero(search) < 5:
        raise ProcessingError("carrier-search region contains fewer than five FFT bins")
    search_indices = np.flatnonzero(search)
    peak_index = int(search_indices[np.argmax(power[search])])
    peak_frequency = float(frequencies[peak_index])
    resolution = float(abs(frequencies[1] - frequencies[0]))
    guard_hz = max(500.0, 5.0 * resolution)
    noise_mask = search & (np.abs(frequencies - peak_frequency) >= guard_hz)
    if not np.any(noise_mask):
        raise ProcessingError("carrier-search region has no bins available for noise estimation")
    peak_power = float(power[peak_index])
    noise_power = float(np.median(power[noise_mask]))
    snr_db = float(10.0 * math.log10(max(peak_power, EPS) / max(noise_power, EPS)))
    estimate = CarrierEstimate(
        frequency_hz=peak_frequency,
        snr_db=snr_db,
        peak_power_density=peak_power,
        noise_power_density=noise_power,
        expected_frequency_hz=float(expected_frequency_hz),
        search_half_width_hz=float(search_half_width_hz),
    )
    if snr_db < minimum_snr_db:
        raise CarrierNotFoundError(
            f"Forward carrier SNR {snr_db:.2f} dB is below {minimum_snr_db:.2f} dB "
            f"near {expected_frequency_hz/1e3:.3f} kHz"
        )
    return estimate


def compute_reflection_ratio(
    forward: np.ndarray,
    reverse: np.ndarray,
    relative_floor: float = 0.05,
    absolute_floor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float]:
    forward = np.asarray(forward)
    reverse = np.asarray(reverse)
    if forward.shape != reverse.shape or forward.ndim != 1 or forward.size == 0:
        raise ProcessingError("Forward and Reverse envelopes must be matching non-empty vectors")
    finite = np.isfinite(forward) & np.isfinite(reverse)
    if not np.any(finite):
        raise ProcessingError("envelopes contain no finite sample pairs")
    median_forward = float(np.median(np.abs(forward[finite])))
    floor = max(float(absolute_floor), float(relative_floor) * median_forward)
    valid = finite & (np.abs(forward) > floor)
    g = np.full(forward.shape, np.nan + 1j * np.nan, dtype=np.complex64)
    g[valid] = (reverse[valid] / forward[valid]).astype(np.complex64, copy=False)
    return g, valid, floor


def process_iq(
    rx0_iq: Any,
    rx1_iq: Any,
    sample_rate_hz: float,
    carrier_frequency_hz: float,
    settings: ProcessingSettings | None = None,
) -> ReflectionResult:
    """Common-DDC, low-pass and decimate two array-like IQ datasets in chunks."""
    settings = settings or ProcessingSettings()
    decimation = settings.validate(sample_rate_hz)
    sample_count = int(len(rx0_iq))
    if sample_count != int(len(rx1_iq)) or sample_count < 1:
        raise ProcessingError("raw RX channels must be non-empty and have equal lengths")
    if abs(carrier_frequency_hz) >= sample_rate_hz / 2:
        raise ProcessingError("carrier estimate lies outside Nyquist")

    sos = signal.butter(
        settings.lowpass_order,
        settings.lowpass_cutoff_hz,
        btype="lowpass",
        fs=sample_rate_hz,
        output="sos",
    )
    zi_template = signal.sosfilt_zi(sos).astype(np.complex128)
    zi_forward: np.ndarray | None = None
    zi_reverse: np.ndarray | None = None
    forward_parts: list[np.ndarray] = []
    reverse_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []

    for start in range(0, sample_count, settings.chunk_samples):
        stop = min(sample_count, start + settings.chunk_samples)
        forward_raw = np.asarray(rx0_iq[start:stop], dtype=np.complex64)
        reverse_raw = np.asarray(rx1_iq[start:stop], dtype=np.complex64)
        if not np.all(np.isfinite(forward_raw)) or not np.all(np.isfinite(reverse_raw)):
            raise ProcessingError("raw IQ contains non-finite values")
        global_indices = start + np.arange(stop - start, dtype=np.float64)
        oscillator = np.exp(
            -1j * (2.0 * np.pi * carrier_frequency_hz / sample_rate_hz) * global_indices
        )
        forward_mixed = forward_raw * oscillator
        reverse_mixed = reverse_raw * oscillator
        if zi_forward is None or zi_reverse is None:
            zi_forward = zi_template * forward_mixed[0]
            zi_reverse = zi_template * reverse_mixed[0]
        forward_filtered, zi_forward = signal.sosfilt(sos, forward_mixed, zi=zi_forward)
        reverse_filtered, zi_reverse = signal.sosfilt(sos, reverse_mixed, zi=zi_reverse)
        first = (-start) % decimation
        selected = np.arange(first, stop - start, decimation, dtype=np.int64)
        if selected.size:
            forward_parts.append(forward_filtered[selected].astype(np.complex64))
            reverse_parts.append(reverse_filtered[selected].astype(np.complex64))
            index_parts.append((start + selected).astype(np.int64))

    forward = np.concatenate(forward_parts)
    reverse = np.concatenate(reverse_parts)
    indices = np.concatenate(index_parts)
    keep = indices / sample_rate_hz >= settings.transient_s
    forward = forward[keep]
    reverse = reverse[keep]
    indices = indices[keep]
    if forward.size < 16:
        raise ProcessingError("capture is too short after removing the filter transient")
    g, valid, forward_floor = compute_reflection_ratio(
        forward,
        reverse,
        relative_floor=settings.forward_relative_floor,
        absolute_floor=settings.forward_absolute_floor,
    )
    return ReflectionResult(
        time_s=indices.astype(np.float64) / sample_rate_hz,
        forward_complex=forward,
        reverse_complex=reverse,
        g_complex=g,
        g_magnitude=np.abs(g).astype(np.float32),
        g_phase=np.angle(g).astype(np.float32),
        valid_ratio=valid,
        forward_floor=forward_floor,
        output_rate_hz=sample_rate_hz / decimation,
        decimation=decimation,
    )


def fit_known_tone(values: np.ndarray, time_s: np.ndarray, tone_frequency_hz: float) -> ToneFit:
    y = np.asarray(values)
    time_axis = np.asarray(time_s, dtype=np.float64)
    if y.shape != time_axis.shape or y.ndim != 1:
        raise ProcessingError("tone-fit values and time axis must be matching vectors")
    finite = np.isfinite(y) & np.isfinite(time_axis)
    if np.count_nonzero(finite) < 16:
        raise ProcessingError("too few finite samples for tone fitting")
    y = y[finite]
    time_axis = time_axis[finite]
    omega_t = 2.0 * np.pi * tone_frequency_hz * time_axis
    design = np.column_stack((np.ones_like(time_axis), np.sin(omega_t), np.cos(omega_t)))
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residual_power = float(np.sum(np.abs(y - fitted) ** 2))
    centered_power = float(np.sum(np.abs(y - np.mean(y)) ** 2))
    r_squared = 0.0 if centered_power <= EPS else 1.0 - residual_power / centered_power
    correlation = math.sqrt(max(0.0, min(1.0, r_squared)))
    modulation = fitted - coefficients[0]
    modulation_rms = float(np.sqrt(np.mean(np.abs(modulation) ** 2)))
    is_real = not np.iscomplexobj(y)
    amplitude = None
    phase_rad = None
    if is_real:
        sine = float(np.real(coefficients[1]))
        cosine = float(np.real(coefficients[2]))
        amplitude = float(math.hypot(sine, cosine))
        phase_rad = float(math.atan2(cosine, sine))
    return ToneFit(
        r_squared=float(r_squared),
        correlation=float(correlation),
        modulation_rms=modulation_rms,
        offset=complex(coefficients[0]),
        sine_coefficient=complex(coefficients[1]),
        cosine_coefficient=complex(coefficients[2]),
        amplitude=amplitude,
        phase_rad=phase_rad,
    )


def _fill_invalid_complex(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    indices = np.arange(len(values), dtype=np.float64)
    valid_indices = indices[valid]
    if valid_indices.size < 16:
        raise ProcessingError("too few valid G samples for analysis")
    real = np.interp(indices, valid_indices, np.real(values[valid]))
    imag = np.interp(indices, valid_indices, np.imag(values[valid]))
    return real + 1j * imag


def _linear_detrend(values: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(values):
        return signal.detrend(np.real(values), type="linear") + 1j * signal.detrend(
            np.imag(values), type="linear"
        )
    return signal.detrend(values, type="linear")


def response_spectrum(
    values: np.ndarray,
    sample_rate_hz: float,
    tone_frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 32 or not np.all(np.isfinite(values)):
        raise ProcessingError("response spectrum requires at least 32 finite real samples")
    nperseg = min(values.size, max(256, int(round(sample_rate_hz))))
    frequencies, psd = signal.welch(
        values,
        fs=sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="linear",
        scaling="density",
    )
    resolution = float(frequencies[1] - frequencies[0])
    target_half_width = max(2.0, resolution)
    target_mask = np.abs(frequencies - tone_frequency_hz) <= target_half_width
    if not np.any(target_mask):
        target_mask[int(np.argmin(np.abs(frequencies - tone_frequency_hz)))] = True
    target_indices = np.flatnonzero(target_mask)
    peak_index = int(target_indices[np.argmax(psd[target_mask])])
    noise_mask = (
        (np.abs(frequencies - tone_frequency_hz) <= 100.0)
        & (np.abs(frequencies - tone_frequency_hz) >= max(5.0, 3.0 * resolution))
    )
    if not np.any(noise_mask):
        noise_mask = frequencies >= max(10.0, tone_frequency_hz / 2.0)
        noise_mask &= frequencies <= min(sample_rate_hz / 2, tone_frequency_hz * 1.5)
        noise_mask &= ~target_mask
    noise = float(np.median(psd[noise_mask])) if np.any(noise_mask) else EPS
    peak = float(psd[peak_index])
    analysis_band = (frequencies >= 10.0) & (
        frequencies <= min(5000.0, sample_rate_hz / 2)
    )
    dominant_index = int(np.flatnonzero(analysis_band)[np.argmax(psd[analysis_band])])
    metrics = {
        "frequency_resolution_hz": resolution,
        "target_peak_frequency_hz": float(frequencies[peak_index]),
        "target_peak_psd": peak,
        "local_noise_psd": noise,
        "target_snr_db": float(10.0 * math.log10(max(peak, EPS) / max(noise, EPS))),
        "dominant_frequency_hz": float(frequencies[dominant_index]),
    }
    return frequencies, psd, metrics


def _magnitude_stats(values: np.ndarray) -> dict[str, float]:
    magnitude = np.abs(values)
    mean = float(np.mean(magnitude))
    std = float(np.std(magnitude))
    return {
        "mean": mean,
        "std": std,
        "coefficient_of_variation": std / max(mean, EPS),
    }


def _circular_stats(phase: np.ndarray) -> dict[str, float]:
    resultant = complex(np.mean(np.exp(1j * phase)))
    length = min(1.0, max(abs(resultant), EPS))
    return {
        "circular_mean_rad": float(np.angle(resultant)),
        "circular_mean_deg": float(np.degrees(np.angle(resultant))),
        "circular_std_rad": float(math.sqrt(max(0.0, -2.0 * math.log(length)))),
        "mean_resultant_length": float(abs(resultant)),
    }


def analyze_response(result: ReflectionResult, tone_frequency_hz: float) -> ResponseAnalysis:
    if tone_frequency_hz <= 0 or tone_frequency_hz >= result.output_rate_hz / 2:
        raise ProcessingError("tone frequency must be between DC and output Nyquist")
    g_filled = _fill_invalid_complex(result.g_complex, result.valid_ratio)
    complex_detrended = _linear_detrend(g_filled)
    magnitude = np.abs(g_filled)
    magnitude_detrended = _linear_detrend(magnitude)
    phase_unwrapped = np.unwrap(np.angle(g_filled))
    phase_detrended = _linear_detrend(phase_unwrapped)

    complex_fit = fit_known_tone(complex_detrended, result.time_s, tone_frequency_hz)
    magnitude_fit = fit_known_tone(magnitude_detrended, result.time_s, tone_frequency_hz)
    phase_fit = fit_known_tone(phase_detrended, result.time_s, tone_frequency_hz)
    magnitude_frequency, magnitude_psd, magnitude_spectrum_metrics = response_spectrum(
        magnitude_detrended,
        result.output_rate_hz,
        tone_frequency_hz,
    )
    phase_frequency, phase_psd, phase_spectrum_metrics = response_spectrum(
        phase_detrended,
        result.output_rate_hz,
        tone_frequency_hz,
    )
    valid_g = result.g_complex[result.valid_ratio]
    metrics: dict[str, Any] = {
        "ratio": {
            "valid_samples": int(np.count_nonzero(result.valid_ratio)),
            "total_samples": int(result.valid_ratio.size),
            "valid_fraction": float(np.mean(result.valid_ratio)),
            "forward_floor": result.forward_floor,
        },
        "stability": {
            "forward_magnitude": _magnitude_stats(result.forward_complex),
            "reverse_magnitude": _magnitude_stats(result.reverse_complex),
            "g_magnitude": _magnitude_stats(valid_g),
            "g_phase": _circular_stats(np.angle(valid_g)),
        },
        "tone": {
            "frequency_hz": float(tone_frequency_hz),
            "reference": "phase-independent sine/cosine quadrature fit",
            "complex_g": complex_fit.to_dict(),
            "g_magnitude": magnitude_fit.to_dict(),
            "g_phase": phase_fit.to_dict(),
            "g_phase_amplitude_deg": (
                None
                if phase_fit.amplitude is None
                else float(np.degrees(phase_fit.amplitude))
            ),
            "magnitude_spectrum": magnitude_spectrum_metrics,
            "phase_spectrum": phase_spectrum_metrics,
        },
    }
    return ResponseAnalysis(
        metrics=metrics,
        g_filled=g_filled.astype(np.complex64),
        magnitude_detrended=magnitude_detrended.astype(np.float32),
        phase_unwrapped=phase_unwrapped.astype(np.float32),
        phase_detrended=phase_detrended.astype(np.float32),
        magnitude_spectrum_hz=magnitude_frequency,
        magnitude_psd=magnitude_psd,
        phase_spectrum_hz=phase_frequency,
        phase_psd=phase_psd,
    )
