#!/usr/bin/env python3
"""Offline carrier recovery and measured complex reflection-ratio analysis."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from capture_io import (
    CaptureFileError,
    artifact_relative_path,
    inspect_capture,
    write_attributes,
)
from reflection_core import (
    CarrierEstimate,
    ProcessingError,
    ProcessingSettings,
    ResponseAnalysis,
    ReflectionResult,
    analyze_response,
    average_power_spectrum,
    estimate_carrier,
    process_iq,
)


ANALYSIS_FORMAT = "b210_measured_complex_reflection_analysis"
ANALYSIS_SCHEMA_VERSION = 1


def _finite_float(value: Any, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise CaptureFileError(f"capture metadata {name!r} is missing or invalid") from exc
    if not math.isfinite(converted):
        raise CaptureFileError(f"capture metadata {name!r} is not finite")
    return converted


def _carrier_dict(
    estimate: CarrierEstimate,
    center_frequency_hz: float,
    reverse_snr_db: float,
) -> dict[str, float]:
    return {
        "expected_if_frequency_hz": estimate.expected_frequency_hz,
        "estimated_if_frequency_hz": estimate.frequency_hz,
        "if_error_hz": estimate.frequency_hz - estimate.expected_frequency_hz,
        "estimated_rf_frequency_hz": center_frequency_hz + estimate.frequency_hz,
        "forward_snr_db": estimate.snr_db,
        "reverse_snr_db_at_carrier": reverse_snr_db,
        "peak_power_density": estimate.peak_power_density,
        "noise_power_density": estimate.noise_power_density,
        "search_half_width_hz": estimate.search_half_width_hz,
    }


def _snr_near_frequency(
    frequencies_hz: np.ndarray,
    power_density: np.ndarray,
    frequency_hz: float,
) -> float:
    resolution = float(abs(frequencies_hz[1] - frequencies_hz[0]))
    peak_mask = np.abs(frequencies_hz - frequency_hz) <= max(2.0 * resolution, 50.0)
    noise_mask = (
        (np.abs(frequencies_hz - frequency_hz) <= 20e3)
        & (np.abs(frequencies_hz - frequency_hz) >= max(500.0, 5.0 * resolution))
    )
    peak = float(np.max(power_density[peak_mask]))
    noise = float(np.median(power_density[noise_mask]))
    return float(10.0 * np.log10(max(peak, np.finfo(float).tiny) / max(noise, np.finfo(float).tiny)))


def _dataset_options(length: int) -> dict[str, Any]:
    chunk = min(max(1024, length // 64), 262_144, length)
    return {
        "chunks": (chunk,),
        "compression": "gzip",
        "compression_opts": 4,
        "shuffle": True,
    }


def write_analysis_h5(
    path: Path,
    source_path: Path,
    capture_metadata: dict[str, Any],
    processing_settings: ProcessingSettings,
    carrier: CarrierEstimate,
    result: ReflectionResult,
    response: ResponseAnalysis,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = ANALYSIS_FORMAT
        handle.attrs["schema_version"] = ANALYSIS_SCHEMA_VERSION
        handle.attrs["source_capture"] = artifact_relative_path(source_path)
        handle.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()

        metadata = handle.create_group("metadata")
        write_attributes(metadata, capture_metadata)
        write_attributes(
            metadata,
            {
                "carrier_estimate_hz": carrier.frequency_hz,
                "carrier_snr_db": carrier.snr_db,
                "output_rate_hz": result.output_rate_hz,
                "decimation": result.decimation,
                "lowpass_cutoff_hz": processing_settings.lowpass_cutoff_hz,
                "lowpass_order": processing_settings.lowpass_order,
                "discarded_filter_transient_s": processing_settings.transient_s,
                "forward_ratio_floor": result.forward_floor,
            },
        )

        processed = handle.create_group("processed")
        options = _dataset_options(len(result.time_s))
        processed.create_dataset("time_s", data=result.time_s, **options)
        processed.create_dataset("forward_complex", data=result.forward_complex, **options)
        processed.create_dataset("reverse_complex", data=result.reverse_complex, **options)
        processed.create_dataset("G_complex", data=result.g_complex, **options)
        processed.create_dataset("G_magnitude", data=result.g_magnitude, **options)
        processed.create_dataset("G_phase", data=result.g_phase, **options)
        processed.create_dataset("G_phase_unwrapped", data=response.phase_unwrapped, **options)
        processed.create_dataset(
            "G_magnitude_detrended", data=response.magnitude_detrended, **options
        )
        processed.create_dataset(
            "G_phase_detrended", data=response.phase_detrended, **options
        )
        processed.create_dataset("valid_ratio", data=result.valid_ratio, **options)
        tone_frequency_hz = float(capture_metadata["tone_frequency_hz"])
        processed.create_dataset(
            "audio_reference_sin",
            data=np.sin(2.0 * np.pi * tone_frequency_hz * result.time_s).astype(np.float32),
            **options,
        )
        processed.create_dataset(
            "audio_reference_cos",
            data=np.cos(2.0 * np.pi * tone_frequency_hz * result.time_s).astype(np.float32),
            **options,
        )

        spectra = handle.create_group("spectra")
        mag_options = _dataset_options(len(response.magnitude_spectrum_hz))
        phase_options = _dataset_options(len(response.phase_spectrum_hz))
        spectra.create_dataset(
            "magnitude_frequency_hz", data=response.magnitude_spectrum_hz, **mag_options
        )
        spectra.create_dataset("magnitude_psd", data=response.magnitude_psd, **mag_options)
        spectra.create_dataset(
            "phase_frequency_hz", data=response.phase_spectrum_hz, **phase_options
        )
        spectra.create_dataset("phase_psd", data=response.phase_psd, **phase_options)


def _plot_indices(length: int, maximum: int = 20_000) -> slice:
    return slice(None, None, max(1, int(math.ceil(length / maximum))))


def plot_summary(
    path: Path,
    frequencies_hz: np.ndarray,
    forward_psd: np.ndarray,
    reverse_psd: np.ndarray,
    expected_if_hz: float,
    carrier: CarrierEstimate,
    result: ReflectionResult,
    response: ResponseAnalysis,
    tone_frequency_hz: float,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle("B210 measured complex reflection ratio", fontsize=15)

    spectrum_mask = np.abs(frequencies_hz - expected_if_hz) <= 40e3
    axes[0, 0].plot(
        frequencies_hz[spectrum_mask] / 1e3,
        10.0 * np.log10(np.maximum(forward_psd[spectrum_mask], np.finfo(float).tiny)),
        label="RX0 Forward",
        linewidth=1.0,
    )
    axes[0, 0].plot(
        frequencies_hz[spectrum_mask] / 1e3,
        10.0 * np.log10(np.maximum(reverse_psd[spectrum_mask], np.finfo(float).tiny)),
        label="RX1 Reverse",
        linewidth=1.0,
        alpha=0.8,
    )
    axes[0, 0].axvline(carrier.frequency_hz / 1e3, color="black", linestyle="--")
    axes[0, 0].set(title="Carrier spectrum", xlabel="Baseband frequency (kHz)", ylabel="PSD (dB/Hz)")
    axes[0, 0].legend()

    view = _plot_indices(len(result.time_s))
    axes[0, 1].plot(
        result.time_s[view], np.abs(result.forward_complex[view]), label="|V_fwd|", linewidth=0.8
    )
    axes[0, 1].plot(
        result.time_s[view], np.abs(result.reverse_complex[view]), label="|V_rev|", linewidth=0.8
    )
    axes[0, 1].set(title="Carrier envelopes", xlabel="Time (s)", ylabel="Magnitude")
    axes[0, 1].legend()

    axes[1, 0].plot(result.time_s[view], result.g_magnitude[view], linewidth=0.8)
    axes[1, 0].set(title="|G(t)|", xlabel="Time (s)", ylabel="Magnitude")

    axes[1, 1].plot(
        result.time_s[view], np.degrees(result.g_phase[view]), linewidth=0.8
    )
    axes[1, 1].set(title="angle G(t)", xlabel="Time (s)", ylabel="Wrapped phase (deg)")

    magnitude_band = response.magnitude_spectrum_hz <= 5000.0
    axes[2, 0].plot(
        response.magnitude_spectrum_hz[magnitude_band],
        10.0
        * np.log10(np.maximum(response.magnitude_psd[magnitude_band], np.finfo(float).tiny)),
        linewidth=1.0,
    )
    axes[2, 0].axvline(tone_frequency_hz, color="red", linestyle="--", label="Known tone")
    axes[2, 0].set(title="|G| modulation spectrum", xlabel="Frequency (Hz)", ylabel="PSD (dB/Hz)")
    axes[2, 0].legend()

    phase_band = response.phase_spectrum_hz <= 5000.0
    axes[2, 1].plot(
        response.phase_spectrum_hz[phase_band],
        10.0 * np.log10(np.maximum(response.phase_psd[phase_band], np.finfo(float).tiny)),
        linewidth=1.0,
    )
    axes[2, 1].axvline(tone_frequency_hz, color="red", linestyle="--", label="Known tone")
    axes[2, 1].set(title="G phase modulation spectrum", xlabel="Frequency (Hz)", ylabel="PSD (rad^2/Hz)")
    axes[2, 1].legend()

    for axis in axes.flat:
        axis.grid(True, linestyle=":", alpha=0.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a valid dual-RX B210 capture without modifying the raw HDF5 file."
    )
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--search-half-width", type=float, default=20e3)
    parser.add_argument("--carrier-min-snr", type=float, default=10.0)
    parser.add_argument("--fft-size", type=int, default=262_144)
    parser.add_argument("--fft-segments", type=int, default=16)
    parser.add_argument("--output-rate", type=float, default=50e3)
    parser.add_argument("--lowpass-cutoff", type=float, default=10e3)
    parser.add_argument("--lowpass-order", type=int, default=8)
    parser.add_argument("--transient", type=float, default=0.005)
    parser.add_argument("--chunk-samples", type=int, default=1_048_576)
    return parser


def run_analysis(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    header = inspect_capture(args.capture, require_valid=True)
    capture_metadata = header["metadata"]
    capture_quality = header["quality"]
    sample_rate_hz = _finite_float(capture_metadata.get("sample_rate_hz"), "sample_rate_hz")
    expected_if_hz = _finite_float(
        capture_metadata.get("expected_if_frequency_hz"), "expected_if_frequency_hz"
    )
    tone_frequency_hz = _finite_float(
        capture_metadata.get("tone_frequency_hz"), "tone_frequency_hz"
    )
    center_frequency_hz = _finite_float(
        capture_metadata.get("actual_center_frequency_rx0_hz"),
        "actual_center_frequency_rx0_hz",
    )
    processing = ProcessingSettings(
        output_rate_hz=args.output_rate,
        lowpass_cutoff_hz=args.lowpass_cutoff,
        lowpass_order=args.lowpass_order,
        transient_s=args.transient,
        chunk_samples=args.chunk_samples,
    )
    processing.validate(sample_rate_hz)

    with h5py.File(args.capture, "r") as capture_file:
        rx0 = capture_file["raw/rx0_iq"]
        rx1 = capture_file["raw/rx1_iq"]
        frequencies_hz, forward_psd = average_power_spectrum(
            rx0,
            sample_rate_hz,
            nfft=args.fft_size,
            max_segments=args.fft_segments,
        )
        reverse_frequencies_hz, reverse_psd = average_power_spectrum(
            rx1,
            sample_rate_hz,
            nfft=args.fft_size,
            max_segments=args.fft_segments,
        )
        if not np.array_equal(frequencies_hz, reverse_frequencies_hz):
            raise ProcessingError("Forward and Reverse spectrum axes differ")
        carrier = estimate_carrier(
            frequencies_hz,
            forward_psd,
            expected_if_hz,
            search_half_width_hz=args.search_half_width,
            minimum_snr_db=args.carrier_min_snr,
        )
        result = process_iq(
            rx0,
            rx1,
            sample_rate_hz,
            carrier.frequency_hz,
            settings=processing,
        )

    reverse_snr_db = _snr_near_frequency(
        frequencies_hz, reverse_psd, carrier.frequency_hz
    )
    response = analyze_response(result, tone_frequency_hz)
    metrics: dict[str, Any] = {
        "format": ANALYSIS_FORMAT,
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_capture": artifact_relative_path(args.capture),
        "capture_metadata": capture_metadata,
        "capture_quality": capture_quality,
        "processing": {
            "input_rate_hz": sample_rate_hz,
            "output_rate_hz": result.output_rate_hz,
            "decimation": result.decimation,
            "lowpass_cutoff_hz": processing.lowpass_cutoff_hz,
            "lowpass_order": processing.lowpass_order,
            "discarded_filter_transient_s": processing.transient_s,
        },
        "carrier": _carrier_dict(carrier, center_frequency_hz, reverse_snr_db),
        **response.metrics,
    }

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.capture.parent / f"{args.capture.stem}_analysis_{timestamp}"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_h5 = output_dir / "analysis.h5"
    metrics_json = output_dir / "metrics.json"
    summary_png = output_dir / "summary.png"
    existing_outputs = [path for path in (analysis_h5, metrics_json, summary_png) if path.exists()]
    if existing_outputs:
        names = ", ".join(path.name for path in existing_outputs)
        raise FileExistsError(f"analysis output already exists and will not be overwritten: {names}")
    write_analysis_h5(
        analysis_h5,
        args.capture,
        capture_metadata,
        processing,
        carrier,
        result,
        response,
    )
    metrics_json.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_summary(
        summary_png,
        frequencies_hz,
        forward_psd,
        reverse_psd,
        expected_if_hz,
        carrier,
        result,
        response,
        tone_frequency_hz,
    )
    return output_dir, metrics


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_dir, metrics = run_analysis(args)
        carrier = metrics["carrier"]
        tone = metrics["tone"]
        ratio = metrics["ratio"]
        print("B210 measured complex reflection ratio analysis")
        print(
            f"  carrier IF={carrier['estimated_if_frequency_hz']/1e3:.3f} kHz, "
            f"Forward SNR={carrier['forward_snr_db']:.2f} dB"
        )
        print(f"  valid G samples={ratio['valid_fraction']*100:.3f}%")
        print(
            f"  tone={tone['frequency_hz']:.0f} Hz, "
            f"complex-G R^2={tone['complex_g']['r_squared']:.4f}, "
            f"|G| SNR={tone['magnitude_spectrum']['target_snr_db']:.2f} dB, "
            f"phase SNR={tone['phase_spectrum']['target_snr_db']:.2f} dB"
        )
        print(f"  output={artifact_relative_path(output_dir)}")
        return 0
    except (CaptureFileError, ProcessingError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
