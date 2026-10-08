#!/usr/bin/env python3
"""Real-time B210 carrier sweep ranked by short-window speech dynamics.

The program transmits a continuous wave on RF B, receives on RF A, scores each
frequency immediately from that frequency's current dwell, and then releases
the IQ samples.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import radio_hardware as base
from speech_dynamic_core import (
    DynamicScoreConfig,
    DynamicScoreResult,
    DynamicScoringError,
    score_magnitude_dwell,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "results" / "speech_dynamic_realtime"
PAPER_DWELL_S = 0.5
DEFAULT_SAMPLE_RATE_HZ = 5e6
DEFAULT_FFT_SIZE = 2**16
DEFAULT_NUM_AVG = 30
DEFAULT_TUNE_SETTLE_S = PAPER_DWELL_S - (
    (DEFAULT_NUM_AVG + 1) * DEFAULT_FFT_SIZE / DEFAULT_SAMPLE_RATE_HZ
)
TOP_POINT_COUNT = 10

ConfigurationError = base.ConfigurationError
HardwareError = base.HardwareError


@dataclass(frozen=True)
class SweepConfig:
    """Hardware acquisition and short-window scoring settings."""

    serial: str = ""
    start_freq_hz: float = 200e6
    stop_freq_hz: float = 800e6
    step_hz: float = 10e6
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ
    fft_size: int = DEFAULT_FFT_SIZE
    num_avg: int = DEFAULT_NUM_AVG
    tx_gain_db: float = 50.0
    rx_gain_db: float = 10.0
    tx_amplitude: float = 1.0
    rx_offset_hz: float = 250e3
    tune_settle_s: float = DEFAULT_TUNE_SETTLE_S
    spectrum_span_hz: float = 10e3
    capture_retries: int = 1
    clip_level: float = 0.98
    max_clip_fraction: float = 1e-4
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    rx_subdev: str = "A:A"
    tx_subdev: str = "A:B"
    rx_antenna: str = "TX/RX"
    tx_antenna: str = "TX/RX"
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

    def scoring_config(self) -> DynamicScoreConfig:
        """Build the signal-processing configuration for one dwell."""
        return DynamicScoreConfig(
            target_rate_hz=self.target_rate_hz,
            pre_resample_lowpass_hz=self.pre_resample_lowpass_hz,
            lowpass_order=self.lowpass_order,
            window_ms=self.window_ms,
            hop_ms=self.hop_ms,
            n_fft=self.n_fft,
            audio_band_min_hz=self.audio_band_min_hz,
            audio_band_max_hz=self.audio_band_max_hz,
            baseline_percentile=self.baseline_percentile,
            scale_percentile=self.scale_percentile,
            spectral_activity_percentile=self.spectral_activity_percentile,
            minimum_scale_db=self.minimum_scale_db,
            lower_percentile=self.lower_percentile,
            upper_percentile=self.upper_percentile,
        )

    def validate(self) -> None:
        """Validate sweep timing, frequency, gain, and scoring parameters."""
        numeric = (
            self.start_freq_hz,
            self.stop_freq_hz,
            self.step_hz,
            self.sample_rate_hz,
            self.tx_gain_db,
            self.rx_gain_db,
            self.tx_amplitude,
            self.rx_offset_hz,
            self.tune_settle_s,
            self.spectrum_span_hz,
            self.clip_level,
            self.max_clip_fraction,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ConfigurationError("all sweep settings must be finite")
        if not 0 < self.start_freq_hz < self.stop_freq_hz:
            raise ConfigurationError("start frequency must be below stop frequency")
        if self.step_hz <= 0 or self.sample_rate_hz <= 0:
            raise ConfigurationError("frequency step and sample rate must be positive")
        if self.fft_size < 16 or self.fft_size & (self.fft_size - 1):
            raise ConfigurationError("capture FFT size must be a power of two")
        if self.num_avg < 1 or self.capture_retries < 0:
            raise ConfigurationError("num_avg must be positive and retries must not be negative")
        if not 0 < self.tx_amplitude <= 1.0:
            raise ConfigurationError("TX amplitude must be in (0, 1]")
        if self.tune_settle_s < 0:
            raise ConfigurationError("tune settling time must not be negative")
        if not 0 < self.clip_level <= 1 or not 0 <= self.max_clip_fraction < 1:
            raise ConfigurationError("invalid clipping settings")
        if abs(self.rx_offset_hz) + self.spectrum_span_hz >= self.sample_rate_hz / 2:
            raise ConfigurationError("RX offset plus spectrum span exceeds Nyquist")
        self.scoring_config().validate()


@dataclass
class SweepResult:
    """Per-frequency scores, acquisition status, and window activity."""

    frequencies_hz: np.ndarray
    actual_tx_freqs_hz: np.ndarray
    actual_rx_freqs_hz: np.ndarray
    scores: np.ndarray
    valid: np.ndarray
    saturated: np.ndarray
    window_count: np.ndarray
    captured_duration_s: np.ndarray
    peak_component: np.ndarray
    clip_fraction: np.ndarray
    overflow_count: np.ndarray
    timeout_count: np.ndarray
    other_error_count: np.ndarray
    tx_underflow_count: np.ndarray
    received_samples: np.ndarray
    attempts: np.ndarray
    errors: list[str]
    activities: list[np.ndarray | None]
    activity_times_s: list[np.ndarray | None]
    duration_s: float


def acquire_and_score(
    radio: base.B210RealtimeRadio,
    frequencies_hz: np.ndarray,
    scoring_config: DynamicScoreConfig,
) -> SweepResult:
    """Capture and immediately score every frequency in one real-time sweep."""
    count = len(frequencies_hz)
    actual_tx_freqs_hz = np.full(count, np.nan, dtype=np.float64)
    actual_rx_freqs_hz = np.full(count, np.nan, dtype=np.float64)
    scores = np.full(count, np.nan, dtype=np.float64)
    valid = np.zeros(count, dtype=bool)
    saturated = np.zeros(count, dtype=bool)
    window_count = np.zeros(count, dtype=np.int32)
    captured_duration_s = np.zeros(count, dtype=np.float64)
    peak_component = np.full(count, np.nan, dtype=np.float64)
    clip_fraction = np.full(count, np.nan, dtype=np.float64)
    overflow_count = np.zeros(count, dtype=np.int32)
    timeout_count = np.zeros(count, dtype=np.int32)
    other_error_count = np.zeros(count, dtype=np.int32)
    tx_underflow_count = np.zeros(count, dtype=np.int32)
    received_samples = np.zeros(count, dtype=np.int64)
    attempts = np.zeros(count, dtype=np.int16)
    errors = [""] * count
    activities: list[np.ndarray | None] = [None] * count
    activity_times_s: list[np.ndarray | None] = [None] * count
    started = time.monotonic()

    print(f"\n[real-time sweep] {count} carrier frequencies")
    for index, frequency_hz in enumerate(frequencies_hz):
        capture = radio.capture_frequency(float(frequency_hz))
        actual_tx_freqs_hz[index] = capture.actual_tx_freq_hz
        actual_rx_freqs_hz[index] = capture.actual_rx_freq_hz
        saturated[index] = capture.saturated
        peak_component[index] = capture.peak_component
        clip_fraction[index] = capture.clip_fraction
        overflow_count[index] = capture.overflow_count
        timeout_count[index] = capture.timeout_count
        other_error_count[index] = capture.other_error_count
        tx_underflow_count[index] = capture.tx_underflow_count
        received_samples[index] = capture.received_samples
        attempts[index] = capture.attempts
        errors[index] = capture.error
        if capture.valid and capture.useful_iq is not None:
            try:
                magnitude = np.abs(capture.useful_iq)
                scored: DynamicScoreResult = score_magnitude_dwell(
                    magnitude,
                    radio.actual_rx_rate_hz,
                    scoring_config,
                )
                scores[index] = scored.score
                window_count[index] = scored.activity.size
                captured_duration_s[index] = magnitude.size / radio.actual_rx_rate_hz
                activities[index] = scored.activity
                activity_times_s[index] = scored.time_s
                valid[index] = True
            except DynamicScoringError as exc:
                errors[index] = "; ".join(value for value in (errors[index], str(exc)) if value)

        elapsed = time.monotonic() - started
        eta = elapsed / (index + 1) * (count - index - 1)
        status = f"score={scores[index]:.3f}" if valid[index] else "INVALID"
        print(
            f"  [{index + 1:3d}/{count}] {frequency_hz / 1e6:9.3f} MHz | "
            f"{status:16s} | ETA={eta / 60:.1f} min"
        )

    return SweepResult(
        frequencies_hz=np.asarray(frequencies_hz, dtype=np.float64),
        actual_tx_freqs_hz=actual_tx_freqs_hz,
        actual_rx_freqs_hz=actual_rx_freqs_hz,
        scores=scores,
        valid=valid,
        saturated=saturated,
        window_count=window_count,
        captured_duration_s=captured_duration_s,
        peak_component=peak_component,
        clip_fraction=clip_fraction,
        overflow_count=overflow_count,
        timeout_count=timeout_count,
        other_error_count=other_error_count,
        tx_underflow_count=tx_underflow_count,
        received_samples=received_samples,
        attempts=attempts,
        errors=errors,
        activities=activities,
        activity_times_s=activity_times_s,
        duration_s=time.monotonic() - started,
    )


def ranking_indices(result: SweepResult) -> np.ndarray:
    """Return valid point indices ordered by descending dynamic score."""
    indices = np.flatnonzero(result.valid & np.isfinite(result.scores))
    return indices[np.argsort(result.scores[indices])[::-1]]


def save_results(
    run_dir: Path,
    config: SweepConfig,
    hardware_info: dict[str, Any],
    result: SweepResult,
) -> None:
    """Save the sweep configuration, ranking, arrays, and summary figure."""
    order = ranking_indices(result)
    np.save(run_dir / "frequencies_hz.npy", result.frequencies_hz, allow_pickle=False)
    np.save(run_dir / "actual_tx_freqs_hz.npy", result.actual_tx_freqs_hz, allow_pickle=False)
    np.save(run_dir / "actual_rx_freqs_hz.npy", result.actual_rx_freqs_hz, allow_pickle=False)
    np.save(run_dir / "dynamic_score.npy", result.scores, allow_pickle=False)
    np.save(run_dir / "valid.npy", result.valid, allow_pickle=False)
    np.save(run_dir / "saturated.npy", result.saturated, allow_pickle=False)
    np.save(run_dir / "window_count.npy", result.window_count, allow_pickle=False)
    np.save(run_dir / "captured_duration_s.npy", result.captured_duration_s, allow_pickle=False)
    np.save(run_dir / "peak_component.npy", result.peak_component, allow_pickle=False)
    np.save(run_dir / "clip_fraction.npy", result.clip_fraction, allow_pickle=False)
    np.save(run_dir / "overflow_count.npy", result.overflow_count, allow_pickle=False)
    np.save(run_dir / "timeout_count.npy", result.timeout_count, allow_pickle=False)
    np.save(run_dir / "other_error_count.npy", result.other_error_count, allow_pickle=False)
    np.save(run_dir / "tx_underflow_count.npy", result.tx_underflow_count, allow_pickle=False)
    np.save(run_dir / "received_samples.npy", result.received_samples, allow_pickle=False)
    np.save(run_dir / "attempts.npy", result.attempts, allow_pickle=False)
    base.write_json(run_dir / "errors.json", result.errors)

    activity_arrays: dict[str, np.ndarray] = {}
    for index, (time_s, activity) in enumerate(zip(result.activity_times_s, result.activities)):
        if time_s is None or activity is None:
            continue
        activity_arrays[f"point_{index:03d}_time_s"] = time_s
        activity_arrays[f"point_{index:03d}_activity"] = activity
    np.savez_compressed(run_dir / "window_activity.npz", **activity_arrays)

    with (run_dir / "ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("rank", "requested_frequency_hz", "actual_tx_frequency_hz", "dynamic_score", "window_count")
        )
        for rank, index in enumerate(order, start=1):
            writer.writerow(
                (
                    rank,
                    result.frequencies_hz[index],
                    result.actual_tx_freqs_hz[index],
                    result.scores[index],
                    result.window_count[index],
                )
            )

    config_values = asdict(config)
    config_values.pop("serial", None)
    config_values["output_root"] = base.artifact_relative_path(run_dir.parent)
    top_points = [
        {
            "rank": rank,
            "frequency_hz": float(result.frequencies_hz[index]),
            "actual_tx_frequency_hz": float(result.actual_tx_freqs_hz[index]),
            "dynamic_score": float(result.scores[index]),
            "window_count": int(result.window_count[index]),
        }
        for rank, index in enumerate(order[:TOP_POINT_COUNT], start=1)
    ]
    base.write_json(
        run_dir / "summary.json",
        {
            "schema_version": 1,
            "completed_at": datetime.now().astimezone().isoformat(),
            "method": "single_dwell_p20_p80_short_window_dynamic_ranking",
            "selection_rule": "highest_continuous_score_no_binary_detection_threshold",
            "config": config_values,
            "hardware": hardware_info,
            "frequency_point_count": int(result.frequencies_hz.size),
            "valid_point_count": int(np.count_nonzero(result.valid)),
            "sweep_duration_s": float(result.duration_s),
            "best_frequency_hz": None if order.size == 0 else float(result.frequencies_hz[order[0]]),
            "top_points": top_points,
        },
    )

    figure, axes = plt.subplots(2, 1, figsize=(11, 8))
    axes[0].plot(result.frequencies_hz / 1e6, result.scores, marker="o", linewidth=1.0)
    axes[0].set(xlabel="Carrier frequency (MHz)", ylabel="Dynamic score", title="Real-time sweep ranking")
    axes[0].grid(True, alpha=0.25)
    if order.size:
        best = int(order[0])
        axes[0].scatter(
            [result.frequencies_hz[best] / 1e6],
            [result.scores[best]],
            color="tab:red",
            zorder=3,
            label="Highest score",
        )
        axes[0].legend(loc="best")
        axes[1].plot(result.activity_times_s[best], result.activities[best], linewidth=1.0)
        axes[1].set_title(f"Highest-score dwell: {result.frequencies_hz[best] / 1e6:.3f} MHz")
    else:
        axes[1].text(0.5, 0.5, "No valid dwell", ha="center", va="center")
    axes[1].set(xlabel="Time within captured dwell (s)", ylabel="Normalized speech-band activity")
    axes[1].grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(run_dir / "summary.png", dpi=160)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface from the configuration defaults."""
    defaults = SweepConfig()
    parser = argparse.ArgumentParser(
        description="Real-time B210 sweep ranked by short-window speech-energy variation"
    )
    parser.add_argument("--serial", default=defaults.serial)
    parser.add_argument("--start-freq", type=float, default=defaults.start_freq_hz, metavar="HZ")
    parser.add_argument("--stop-freq", type=float, default=defaults.stop_freq_hz, metavar="HZ")
    parser.add_argument("--step", type=float, default=defaults.step_hz, metavar="HZ")
    parser.add_argument("--sample-rate", type=float, default=defaults.sample_rate_hz, metavar="SPS")
    parser.add_argument("--fft-size", type=int, default=defaults.fft_size)
    parser.add_argument("--num-avg", type=int, default=defaults.num_avg)
    parser.add_argument("--tx-gain", type=float, default=defaults.tx_gain_db, metavar="DB")
    parser.add_argument("--rx-gain", type=float, default=defaults.rx_gain_db, metavar="DB")
    parser.add_argument("--tx-amplitude", type=float, default=defaults.tx_amplitude)
    parser.add_argument("--rx-offset", type=float, default=defaults.rx_offset_hz, metavar="HZ")
    parser.add_argument("--tune-settle", type=float, default=defaults.tune_settle_s, metavar="SECONDS")
    parser.add_argument("--capture-retries", type=int, default=defaults.capture_retries)
    parser.add_argument("--output-dir", default=defaults.output_root)
    return parser


def config_from_args(args: argparse.Namespace) -> SweepConfig:
    """Convert parsed command-line values into a validated configuration."""
    config = SweepConfig(
        serial=args.serial,
        start_freq_hz=args.start_freq,
        stop_freq_hz=args.stop_freq,
        step_hz=args.step,
        sample_rate_hz=args.sample_rate,
        fft_size=args.fft_size,
        num_avg=args.num_avg,
        tx_gain_db=args.tx_gain,
        rx_gain_db=args.rx_gain,
        tx_amplitude=args.tx_amplitude,
        rx_offset_hz=args.rx_offset,
        tune_settle_s=args.tune_settle,
        capture_retries=args.capture_retries,
        output_root=args.output_dir,
    )
    config.validate()
    return config


def main(argv: Sequence[str] | None = None) -> int:
    """Run preflight, real-time acquisition, scoring, and result export."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
    except ConfigurationError as exc:
        parser.error(str(exc))

    frequencies_hz = base.generate_frequency_points(
        config.start_freq_hz,
        config.stop_freq_hz,
        config.step_hz,
    )
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = base.artifact_output_root(config.output_root) / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    capture_s = config.num_avg * config.fft_size / config.sample_rate_hz
    nominal_point_s = (
        config.tune_settle_s + (config.num_avg + 1) * config.fft_size / config.sample_rate_hz
    )

    print("=" * 72)
    print("B210 real-time speech-dynamic carrier sweep")
    print("=" * 72)
    print(
        f"Sweep: {config.start_freq_hz / 1e6:.3f}-{config.stop_freq_hz / 1e6:.3f} MHz, "
        f"{len(frequencies_hz)} points, {config.step_hz / 1e6:g} MHz step"
    )
    print(
        f"Per point: {nominal_point_s:.3f} s nominal, {capture_s:.3f} s scored, "
        f"{config.window_ms:g} ms windows / {config.hop_ms:g} ms hop"
    )
    print("Selection: highest continuous dynamic score; no detection threshold")
    print(f"Output directory: {base.artifact_relative_path(run_dir)}")
    print("WARNING: This program transmits RF. Verify antennas and regulatory compliance.")

    radio: base.B210RealtimeRadio | None = None
    try:
        radio = base.B210RealtimeRadio(config)
        config_values = asdict(config)
        config_values.pop("serial", None)
        config_values["output_root"] = base.artifact_relative_path(run_dir.parent)
        base.write_json(
            run_dir / "config.json",
            {
                "created_at": datetime.now().astimezone().isoformat(),
                "analysis_mode": "real_time_single_dwell_continuous_dynamic_ranking",
                "config": config_values,
                "hardware": radio.hardware_info,
                "nominal_point_duration_s": nominal_point_s,
                "scored_capture_duration_s": capture_s,
            },
        )
        input(
            "\nKeep representative speech playing throughout the sweep, verify safe antenna "
            "placement, then press Enter to start..."
        )
        radio.prepare_frequency(float(frequencies_hz[0]))
        radio.start_tx()
        try:
            midpoint = (config.start_freq_hz + config.stop_freq_hz) / 2.0
            radio.saturation_preflight((config.start_freq_hz, midpoint, config.stop_freq_hz))
            result = acquire_and_score(radio, frequencies_hz, config.scoring_config())
        finally:
            radio.stop_tx()
        save_results(run_dir, config, radio.hardware_info, result)
        order = ranking_indices(result)
        if order.size:
            print("\nTop frequencies:")
            for rank, index in enumerate(order[:TOP_POINT_COUNT], start=1):
                print(
                    f"  {rank:2d}. {result.frequencies_hz[index] / 1e6:9.3f} MHz | "
                    f"score={result.scores[index]:.3f}"
                )
        else:
            print("\nNo valid frequency dwell was captured.")
        print(f"Results: {base.artifact_relative_path(run_dir)}")
        return 0
    except KeyboardInterrupt:
        base.write_json(
            run_dir / "failure.json",
            {"time": datetime.now().astimezone().isoformat(), "error": "Interrupted by user"},
        )
        print("\nSweep interrupted; transmission stopped.", file=sys.stderr)
        return 130
    except Exception as exc:
        base.write_json(
            run_dir / "failure.json",
            {"time": datetime.now().astimezone().isoformat(), "error": str(exc)},
        )
        print(f"\nSweep failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if radio is not None:
            radio.stop_tx()


if __name__ == "__main__":
    raise SystemExit(main())
