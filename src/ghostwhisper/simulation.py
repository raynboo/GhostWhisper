from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal

from ghostwhisper.audio import normalize_peak


NOISE_STRATEGY_NAME = "v4_uniform_weakcarrier"
REAL_CALIBRATED_NOISE_STRATEGY_NAME = "v5_realcalib_weakcarrier"
REAL_CALIBRATED_AGGRESSIVE_NOISE_STRATEGY_NAME = "v6_realcalib_aggressive_weakcarrier"


@dataclass
class NoiseParams:
    strategy: str
    snr_db: float
    profile: str
    carrier_gain: float
    carrier_attenuation_db: float
    carrier_nonlinearity: float
    broadband_color: str
    broadband_gain: float
    broadband_floor_ratio: float
    tone_count: int
    tone_freqs_hz: list[float]
    tone_gains: list[float]
    tone_drifts_hz: list[float]
    hum_base_hz: float
    hum_harmonics: int
    hum_gain: float
    gain_depth: float
    dropout_count: int
    dropout_depth: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples)) + 1e-12))


def _pink_noise(rng: np.random.Generator, length: int) -> np.ndarray:
    white = rng.normal(0.0, 1.0, length)
    freqs = np.fft.rfftfreq(length)
    spectrum = np.fft.rfft(white)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    colored = np.fft.irfft(spectrum * scale, n=length)
    return normalize_peak(colored)


def _brown_noise(rng: np.random.Generator, length: int) -> np.ndarray:
    white = rng.normal(0.0, 1.0, length)
    brown = np.cumsum(white)
    brown -= np.mean(brown)
    return normalize_peak(brown)


def _colored_noise(rng: np.random.Generator, length: int, color: str) -> np.ndarray:
    if color == "pink":
        return _pink_noise(rng, length)
    if color == "brown":
        return _brown_noise(rng, length)
    return normalize_peak(rng.normal(0.0, 1.0, length))


def _random_smooth_envelope(
    rng: np.random.Generator,
    length: int,
    sample_rate: int,
    depth: float,
) -> np.ndarray:
    control_rate_hz = 8.0
    control_count = max(4, int(length / sample_rate * control_rate_hz))
    control = rng.uniform(-1.0, 1.0, control_count)
    control = signal.savgol_filter(control, min(control_count // 2 * 2 - 1, 31), 3, mode="interp")
    envelope = np.interp(np.linspace(0, control_count - 1, length), np.arange(control_count), control)
    envelope = 1.0 + depth * envelope
    return np.clip(envelope, 0.05, None)


def _dropout_envelope(
    rng: np.random.Generator,
    length: int,
    sample_rate: int,
    count: int,
    depth: float,
    min_width_sec: float = 0.04,
    max_width_sec: float = 0.35,
) -> np.ndarray:
    envelope = np.ones(length, dtype=np.float64)
    if count <= 0:
        return envelope
    for _ in range(count):
        center = rng.integers(0, length)
        width = int(rng.uniform(min_width_sec, max_width_sec) * sample_rate)
        start = max(0, center - width // 2)
        end = min(length, center + width // 2)
        if end <= start:
            continue
        window = signal.windows.hann(end - start)
        notch = 1.0 - depth * window
        envelope[start:end] *= notch
    return envelope


def _tone_with_drift(
    rng: np.random.Generator,
    length: int,
    sample_rate: int,
    freq_hz: float,
    drift_hz: float,
) -> np.ndarray:
    t = np.arange(length, dtype=np.float64) / sample_rate
    drift_phase = rng.uniform(0.0, 2.0 * np.pi)
    drift = drift_hz * np.sin(2.0 * np.pi * rng.uniform(0.05, 0.6) * t + drift_phase)
    instantaneous_freq = np.clip(freq_hz + drift, 1.0, sample_rate / 2.2)
    phase = 2.0 * np.pi * np.cumsum(instantaneous_freq) / sample_rate
    return np.sin(phase + rng.uniform(0.0, 2.0 * np.pi))


def sample_noise_params(
    rng: np.random.Generator,
    sample_rate: int,
    buried_probability: float = 0.9,
) -> NoiseParams:
    del buried_probability
    is_buried = True
    tone_count = int(rng.integers(5, 14))
    tone_freqs = sorted(float(x) for x in rng.uniform(80.0, min(9000.0, sample_rate / 2.4), tone_count))
    tone_gains = [
        float(10 ** rng.uniform(-0.35, 0.95))
        for _ in range(tone_count)
    ]
    tone_drifts = [float(rng.uniform(0.0, 36.0)) for _ in range(tone_count)]
    return NoiseParams(
        strategy=NOISE_STRATEGY_NAME,
        snr_db=float(rng.uniform(-32.0, -12.0)),
        profile="weak_carrier",
        carrier_gain=float(10 ** rng.uniform(-1.3, -0.55)),
        carrier_attenuation_db=float(rng.uniform(18.0, 36.0)),
        carrier_nonlinearity=float(rng.uniform(0.35, 1.05)),
        broadband_color=str(rng.choice(["white", "pink", "brown"])),
        broadband_gain=float(rng.uniform(1.3, 4.5)),
        broadband_floor_ratio=float(rng.uniform(0.55, 1.8)),
        tone_count=tone_count,
        tone_freqs_hz=tone_freqs,
        tone_gains=tone_gains,
        tone_drifts_hz=tone_drifts,
        hum_base_hz=float(rng.choice([50.0, 60.0, 100.0, 120.0])),
        hum_harmonics=int(rng.integers(6, 18)),
        hum_gain=float(rng.uniform(0.16, 0.95)),
        gain_depth=float(rng.uniform(0.55, 0.95)),
        dropout_count=int(rng.integers(4, 11)),
        dropout_depth=float(rng.uniform(0.75, 0.995)),
    )


def sample_real_calibrated_noise_params(
    rng: np.random.Generator,
    sample_rate: int,
    buried_probability: float = 1.0,
) -> NoiseParams:
    del buried_probability
    tone_count = int(rng.integers(7, 18))
    tone_freqs = sorted(float(x) for x in rng.uniform(60.0, min(11_000.0, sample_rate / 2.4), tone_count))
    tone_gains = [float(10 ** rng.uniform(-0.1, 1.25)) for _ in range(tone_count)]
    tone_drifts = [float(rng.uniform(0.0, 54.0)) for _ in range(tone_count)]
    return NoiseParams(
        strategy=REAL_CALIBRATED_NOISE_STRATEGY_NAME,
        snr_db=float(rng.uniform(-38.0, -18.0)),
        profile="real_calibrated_weak_carrier",
        carrier_gain=float(10 ** rng.uniform(-1.8, -0.85)),
        carrier_attenuation_db=float(rng.uniform(28.0, 48.0)),
        carrier_nonlinearity=float(rng.uniform(0.25, 0.85)),
        broadband_color=str(rng.choice(["white", "pink", "brown"])),
        broadband_gain=float(rng.uniform(2.4, 7.0)),
        broadband_floor_ratio=float(rng.uniform(1.6, 5.2)),
        tone_count=tone_count,
        tone_freqs_hz=tone_freqs,
        tone_gains=tone_gains,
        tone_drifts_hz=tone_drifts,
        hum_base_hz=float(rng.choice([50.0, 60.0, 100.0, 120.0])),
        hum_harmonics=int(rng.integers(8, 24)),
        hum_gain=float(rng.uniform(0.32, 1.45)),
        gain_depth=float(rng.uniform(0.7, 0.98)),
        dropout_count=int(rng.integers(7, 18)),
        dropout_depth=float(rng.uniform(0.88, 0.999)),
    )


def sample_real_calibrated_aggressive_noise_params(
    rng: np.random.Generator,
    sample_rate: int,
    buried_probability: float = 1.0,
) -> NoiseParams:
    del buried_probability
    tone_count = int(rng.integers(9, 24))
    tone_freqs = sorted(float(x) for x in rng.uniform(45.0, min(12_000.0, sample_rate / 2.25), tone_count))
    tone_gains = [float(10 ** rng.uniform(0.05, 1.55)) for _ in range(tone_count)]
    tone_drifts = [float(rng.uniform(0.0, 72.0)) for _ in range(tone_count)]
    return NoiseParams(
        strategy=REAL_CALIBRATED_AGGRESSIVE_NOISE_STRATEGY_NAME,
        snr_db=float(rng.uniform(-44.0, -22.0)),
        profile="real_calibrated_aggressive_weak_carrier",
        carrier_gain=float(10 ** rng.uniform(-2.25, -1.05)),
        carrier_attenuation_db=float(rng.uniform(34.0, 56.0)),
        carrier_nonlinearity=float(rng.uniform(0.18, 0.70)),
        broadband_color=str(rng.choice(["white", "pink", "brown"])),
        broadband_gain=float(rng.uniform(3.4, 9.5)),
        broadband_floor_ratio=float(rng.uniform(2.8, 8.5)),
        tone_count=tone_count,
        tone_freqs_hz=tone_freqs,
        tone_gains=tone_gains,
        tone_drifts_hz=tone_drifts,
        hum_base_hz=float(rng.choice([50.0, 60.0, 100.0, 120.0])),
        hum_harmonics=int(rng.integers(10, 32)),
        hum_gain=float(rng.uniform(0.45, 2.1)),
        gain_depth=float(rng.uniform(0.78, 0.995)),
        dropout_count=int(rng.integers(9, 24)),
        dropout_depth=float(rng.uniform(0.92, 0.9995)),
    )


def _degrade_carrier_bandwidth(
    carrier: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    profile: str,
) -> np.ndarray:
    if not profile.startswith("real_calibrated") or len(carrier) < 8:
        return carrier
    if profile == "real_calibrated_aggressive_weak_carrier":
        cutoff_hz = float(rng.uniform(1_800.0, 4_200.0))
    else:
        cutoff_hz = float(rng.uniform(2_600.0, 5_200.0))
    sos = signal.butter(5, cutoff_hz, btype="lowpass", fs=sample_rate, output="sos")
    degraded = signal.sosfiltfilt(sos, carrier)
    notch_probability = 0.9 if profile == "real_calibrated_aggressive_weak_carrier" else 0.75
    if rng.random() < notch_probability:
        notch_freq = float(rng.uniform(1_500.0, min(7_500.0, sample_rate / 2.5)))
        quality = float(rng.uniform(8.0, 28.0))
        b, a = signal.iirnotch(notch_freq, quality, fs=sample_rate)
        degraded = signal.filtfilt(b, a, degraded)
    return degraded


def simulate_noisy_waveform(
    clean: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    params: NoiseParams | None = None,
    buried_probability: float = 0.9,
) -> tuple[np.ndarray, NoiseParams]:
    params = params or sample_noise_params(rng, sample_rate, buried_probability=buried_probability)
    clean = normalize_peak(clean.astype(np.float64))
    length = len(clean)
    t = np.arange(length, dtype=np.float64) / sample_rate

    carrier_like = params.carrier_gain * np.tanh(params.carrier_nonlinearity * clean)
    carrier_like = _degrade_carrier_bandwidth(carrier_like, sample_rate, rng, params.profile)
    carrier_like *= _random_smooth_envelope(rng, length, sample_rate, params.gain_depth)
    carrier_like *= _dropout_envelope(
        rng,
        length,
        sample_rate,
        params.dropout_count,
        params.dropout_depth,
        min_width_sec=0.08 if params.profile == "regular" else 0.12,
        max_width_sec=0.45 if params.profile == "regular" else 0.85,
    )
    carrier_like *= 10.0 ** (-params.carrier_attenuation_db / 20.0)

    broadband = _colored_noise(rng, length, params.broadband_color) * params.broadband_gain
    broadband *= _random_smooth_envelope(
        rng,
        length,
        sample_rate,
        depth=0.18 if params.profile == "regular" else 0.38,
    )

    tones = np.zeros(length, dtype=np.float64)
    for freq, gain, drift in zip(params.tone_freqs_hz, params.tone_gains, params.tone_drifts_hz):
        tones += gain * _tone_with_drift(rng, length, sample_rate, freq, drift)

    hum = np.zeros(length, dtype=np.float64)
    for harmonic in range(1, params.hum_harmonics + 1):
        freq = params.hum_base_hz * harmonic
        if freq >= sample_rate / 2:
            continue
        hum += (params.hum_gain / harmonic) * np.sin(
            2.0 * np.pi * freq * t + rng.uniform(0.0, 2.0 * np.pi)
        )

    desired_rms = _rms(carrier_like)
    broadband = broadband * max(
        desired_rms * params.broadband_floor_ratio / (_rms(broadband) + 1e-12),
        1e-12,
    )

    structured = tones + hum
    interference = broadband + structured
    structured_target = desired_rms / (10.0 ** (params.snr_db / 20.0) + 1e-12)
    extra_scale = max((structured_target - _rms(broadband)) / (_rms(structured) + 1e-12), 0.0)
    noisy = carrier_like + broadband + extra_scale * structured
    return normalize_peak(noisy), params
