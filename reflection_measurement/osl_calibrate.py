#!/usr/bin/env python3
"""Acquire, solve and apply one-port Open/Short/Load calibration."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibration_core import (
    CalibrationError,
    OSLModel,
    StandardEstimate,
    apply_osl,
    complex_from_dict,
    complex_to_dict,
    forward_model,
    robust_complex_center,
    solve_osl,
)
from capture import (
    DEFAULT_BANDWIDTH_HZ,
    DEFAULT_CHUNK_SAMPLES,
    DEFAULT_GAIN_DB,
    DEFAULT_LO_OFFSET_HZ,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_SERIAL,
    DEFAULT_SIGNAL_FREQUENCY_HZ,
    B210DualReceiver,
    CaptureConfigurationError,
    CaptureHardwareError,
    CaptureSettings,
)
from capture_io import (
    CaptureFileError,
    artifact_relative_path,
    inspect_capture,
    read_attributes,
    write_attributes,
)
from reflection_core import (
    CarrierEstimate,
    ProcessingError,
    ProcessingSettings,
    ReflectionResult,
    ResponseAnalysis,
    analyze_response,
    average_power_spectrum,
    estimate_carrier,
    process_iq,
)


CALIBRATION_FORMAT = "b210_one_port_osl_calibration"
CALIBRATION_SCHEMA_VERSION = 1
CALIBRATED_FORMAT = "b210_calibrated_one_port_reflection"
CALIBRATED_SCHEMA_VERSION = 1
STANDARD_ORDER = ("open", "short", "load")
DEFAULT_CALIBRATION_DURATION_S = 2.0


@dataclass
class ProcessedCapture:
    path: Path
    metadata: dict[str, Any]
    quality: dict[str, Any]
    carrier: CarrierEstimate
    result: ReflectionResult


@dataclass
class LoadedCalibration:
    path: Path
    model: OSLModel
    reference_metadata: dict[str, Any]
    processing: ProcessingSettings
    search_half_width_hz: float
    carrier_min_snr_db: float
    fft_size: int
    fft_segments: int
    metrics: dict[str, Any]


def _complex_argument(text: str) -> complex:
    try:
        value = complex(text.strip().lower().replace("i", "j"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be a complex value such as 1+0j or 0.98-0.03j"
        ) from exc
    if not np.isfinite(value):
        raise argparse.ArgumentTypeError("complex standard value must be finite")
    return value


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"missing or invalid metadata field {name!r}") from exc
    if not math.isfinite(result):
        raise CalibrationError(f"metadata field {name!r} is not finite")
    return result


def _processing_from_args(args: argparse.Namespace) -> ProcessingSettings:
    return ProcessingSettings(
        output_rate_hz=args.output_rate,
        lowpass_cutoff_hz=args.lowpass_cutoff,
        lowpass_order=args.lowpass_order,
        transient_s=args.transient,
        chunk_samples=args.analysis_chunk_samples,
    )


def _process_capture(
    path: Path,
    processing: ProcessingSettings,
    *,
    search_half_width_hz: float,
    carrier_min_snr_db: float,
    fft_size: int,
    fft_segments: int,
) -> ProcessedCapture:
    header = inspect_capture(path, require_valid=True)
    metadata = header["metadata"]
    sample_rate_hz = _finite_float(metadata.get("sample_rate_hz"), "sample_rate_hz")
    expected_if_hz = _finite_float(
        metadata.get("expected_if_frequency_hz"), "expected_if_frequency_hz"
    )
    processing.validate(sample_rate_hz)
    with h5py.File(path, "r") as capture_file:
        rx0 = capture_file["raw/rx0_iq"]
        rx1 = capture_file["raw/rx1_iq"]
        frequencies_hz, forward_psd = average_power_spectrum(
            rx0, sample_rate_hz, nfft=fft_size, max_segments=fft_segments
        )
        carrier = estimate_carrier(
            frequencies_hz,
            forward_psd,
            expected_if_hz,
            search_half_width_hz=search_half_width_hz,
            minimum_snr_db=carrier_min_snr_db,
        )
        result = process_iq(
            rx0,
            rx1,
            sample_rate_hz,
            carrier.frequency_hz,
            settings=processing,
        )
    return ProcessedCapture(
        path=Path(path),
        metadata=metadata,
        quality=header["quality"],
        carrier=carrier,
        result=result,
    )


_NUMERIC_COMPATIBILITY = {
    "signal_frequency_hz": 1.0,
    "actual_center_frequency_rx0_hz": 1.0,
    "actual_center_frequency_rx1_hz": 1.0,
    "sample_rate_hz": 1.0,
    "actual_bandwidth_rx0_hz": 1.0,
    "actual_bandwidth_rx1_hz": 1.0,
    "actual_rx0_gain_db": 0.05,
    "actual_rx1_gain_db": 0.05,
}
_EXACT_COMPATIBILITY = (
    "device_fingerprint_sha256",
    "rx0_role",
    "rx1_role",
    "rx0_antenna",
    "rx1_antenna",
    "rx_subdev_spec",
    "raw_channel_order",
)


def assert_compatible(
    reference: dict[str, Any], candidate: dict[str, Any], label: str
) -> None:
    mismatches: list[str] = []
    for key, tolerance in _NUMERIC_COMPATIBILITY.items():
        try:
            reference_value = float(reference[key])
            candidate_value = float(candidate[key])
        except (KeyError, TypeError, ValueError):
            mismatches.append(f"{key}=missing")
            continue
        if not math.isclose(reference_value, candidate_value, rel_tol=1e-9, abs_tol=tolerance):
            mismatches.append(f"{key}={candidate_value:g} (expected {reference_value:g})")
    for key in _EXACT_COMPATIBILITY:
        if reference.get(key) != candidate.get(key):
            mismatches.append(f"{key}={candidate.get(key)!r} (expected {reference.get(key)!r})")
    if mismatches:
        raise CalibrationError(
            f"{label} is incompatible with the calibration: " + "; ".join(mismatches)
        )


def _estimate_standard(
    capture: ProcessedCapture,
    standard_gamma: complex,
    settle_time_s: float,
) -> StandardEstimate:
    result = capture.result
    if settle_time_s < 0 or 2.0 * settle_time_s >= result.time_s[-1] - result.time_s[0]:
        raise CalibrationError("standard settle time leaves no usable calibration interval")
    time_mask = (result.time_s >= result.time_s[0] + settle_time_s) & (
        result.time_s <= result.time_s[-1] - settle_time_s
    )
    available_mask = time_mask & result.valid_ratio
    available = int(np.count_nonzero(available_mask))
    if available < 32:
        raise CalibrationError(f"{capture.path.name} has too few valid steady samples")
    values = result.g_complex[available_mask]
    center, dispersion, retained = robust_complex_center(values)
    used = int(np.count_nonzero(retained))
    return StandardEstimate(
        measured_g=center,
        standard_gamma=complex(standard_gamma),
        samples_used=used,
        samples_available=available,
        valid_fraction=float(np.mean(result.valid_ratio)),
        complex_std=dispersion,
        standard_error=dispersion / math.sqrt(used),
    )


def _reference_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = set(_NUMERIC_COMPATIBILITY).union(_EXACT_COMPATIBILITY)
    keys.update(("expected_if_frequency_hz", "mboard_name", "device_selection"))
    return {key: metadata[key] for key in keys if key in metadata}


def _ensure_new_outputs(output_dir: Path, names: tuple[str, ...]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in names if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            "calibration output will not overwrite existing files: "
            + ", ".join(path.name for path in existing)
        )


def _write_complex_attributes(group: h5py.Group, name: str, value: complex) -> None:
    group.attrs[f"{name}_real"] = float(np.real(value))
    group.attrs[f"{name}_imag"] = float(np.imag(value))


def _read_complex_attributes(group: h5py.Group, name: str) -> complex:
    try:
        value = complex(group.attrs[f"{name}_real"], group.attrs[f"{name}_imag"])
    except KeyError as exc:
        raise CalibrationError(f"calibration is missing coefficient {name}") from exc
    if not np.isfinite(value):
        raise CalibrationError(f"calibration coefficient {name} is not finite")
    return value


def _write_calibration_h5(
    path: Path,
    metrics: dict[str, Any],
    model: OSLModel,
    standards: dict[str, StandardEstimate],
    reference_metadata: dict[str, Any],
    processing: ProcessingSettings,
    args: argparse.Namespace,
) -> None:
    with h5py.File(path, "x") as handle:
        handle.attrs["format"] = CALIBRATION_FORMAT
        handle.attrs["schema_version"] = CALIBRATION_SCHEMA_VERSION
        handle.attrs["created_utc"] = metrics["created_utc"]
        metadata_group = handle.create_group("reference_metadata")
        write_attributes(metadata_group, reference_metadata)
        processing_group = handle.create_group("processing")
        write_attributes(
            processing_group,
            {
                "output_rate_hz": processing.output_rate_hz,
                "lowpass_cutoff_hz": processing.lowpass_cutoff_hz,
                "lowpass_order": processing.lowpass_order,
                "transient_s": processing.transient_s,
                "chunk_samples": processing.chunk_samples,
                "search_half_width_hz": args.search_half_width,
                "carrier_min_snr_db": args.carrier_min_snr,
                "fft_size": args.fft_size,
                "fft_segments": args.fft_segments,
            },
        )
        coefficients = handle.create_group("coefficients")
        for name, value in (("a", model.a), ("b", model.b), ("c", model.c)):
            _write_complex_attributes(coefficients, name, value)
        write_attributes(
            coefficients,
            {
                "condition_number": model.condition_number,
                "solve_residual_rms": model.solve_residual_rms,
                "minimum_measured_separation": model.minimum_measured_separation,
            },
        )
        standards_group = handle.create_group("standards")
        for name in STANDARD_ORDER:
            group = standards_group.create_group(name)
            estimate = standards[name]
            _write_complex_attributes(group, "measured_g", estimate.measured_g)
            _write_complex_attributes(group, "standard_gamma", estimate.standard_gamma)
            write_attributes(
                group,
                {
                    "source_capture": metrics["standards"][name]["source_capture"],
                    "samples_used": estimate.samples_used,
                    "samples_available": estimate.samples_available,
                    "valid_fraction": estimate.valid_fraction,
                    "complex_std": estimate.complex_std,
                    "standard_error": estimate.standard_error,
                },
            )


def _plot_calibration(
    path: Path,
    model: OSLModel,
    standards: dict[str, StandardEstimate],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    theta = np.linspace(0.0, 2.0 * np.pi, 721)
    unit_circle = np.exp(1j * theta)
    measured_circle = forward_model(unit_circle, model)
    axes[0].plot(measured_circle.real, measured_circle.imag, label="Mapped |Gamma|=1")
    colors = {"open": "tab:blue", "short": "tab:red", "load": "tab:green"}
    for name in STANDARD_ORDER:
        measured = standards[name].measured_g
        axes[0].scatter(measured.real, measured.imag, s=60, color=colors[name], label=name)
        axes[0].annotate(name, (measured.real, measured.imag), xytext=(5, 5), textcoords="offset points")
    axes[0].set(
        title="Measured G plane",
        xlabel="Real",
        ylabel="Imaginary",
        aspect="equal",
    )
    axes[0].legend()

    for name in STANDARD_ORDER:
        estimate = standards[name]
        calibrated, valid, _ = apply_osl(
            np.array([estimate.measured_g]), model, np.array([True])
        )
        point = calibrated[0]
        axes[1].scatter(point.real, point.imag, s=60, color=colors[name], label=name)
        axes[1].scatter(
            estimate.standard_gamma.real,
            estimate.standard_gamma.imag,
            marker="x",
            s=80,
            color="black",
        )
    axes[1].plot(unit_circle.real, unit_circle.imag, color="0.6", linestyle="--")
    axes[1].set(
        title="Calibrated standards (x = assigned value)",
        xlabel="Real Gamma",
        ylabel="Imaginary Gamma",
        aspect="equal",
    )
    axes[1].legend()
    for axis in axes:
        axis.grid(True, linestyle=":", alpha=0.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_calibration(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    paths = {name: Path(getattr(args, name)) for name in STANDARD_ORDER}
    known = {
        "open": complex(args.open_gamma),
        "short": complex(args.short_gamma),
        "load": complex(args.load_gamma),
    }
    processing = _processing_from_args(args)
    processed: dict[str, ProcessedCapture] = {}
    for name in STANDARD_ORDER:
        processed[name] = _process_capture(
            paths[name],
            processing,
            search_half_width_hz=args.search_half_width,
            carrier_min_snr_db=args.carrier_min_snr,
            fft_size=args.fft_size,
            fft_segments=args.fft_segments,
        )
    reference = processed["open"].metadata
    assert_compatible(reference, processed["short"].metadata, "Short capture")
    assert_compatible(reference, processed["load"].metadata, "Load capture")
    standards = {
        name: _estimate_standard(processed[name], known[name], args.standard_settle)
        for name in STANDARD_ORDER
    }
    measured = np.array([standards[name].measured_g for name in STANDARD_ORDER])
    actual = np.array([standards[name].standard_gamma for name in STANDARD_ORDER])
    model = solve_osl(measured, actual)
    relative_std = max(item.complex_std for item in standards.values()) / max(
        model.minimum_measured_separation, np.finfo(float).eps
    )
    if relative_std > args.max_relative_standard_std:
        raise CalibrationError(
            f"standard instability/separation ratio {relative_std:.3%} exceeds "
            f"{args.max_relative_standard_std:.3%}"
        )

    created_utc = datetime.now(timezone.utc).isoformat()
    reference_metadata = _reference_metadata(reference)
    metrics: dict[str, Any] = {
        "format": CALIBRATION_FORMAT,
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "created_utc": created_utc,
        "reference_metadata": reference_metadata,
        "processing": {
            "output_rate_hz": processing.output_rate_hz,
            "lowpass_cutoff_hz": processing.lowpass_cutoff_hz,
            "lowpass_order": processing.lowpass_order,
            "transient_s": processing.transient_s,
            "chunk_samples": processing.chunk_samples,
            "search_half_width_hz": args.search_half_width,
            "carrier_min_snr_db": args.carrier_min_snr,
            "fft_size": args.fft_size,
            "fft_segments": args.fft_segments,
            "standard_settle_s": args.standard_settle,
        },
        "standards": {},
        "model": model.to_dict(),
        "quality": {
            "maximum_relative_standard_std": args.max_relative_standard_std,
            "observed_relative_standard_std": relative_std,
        },
    }
    for name in STANDARD_ORDER:
        standard_metrics = standards[name].to_dict()
        standard_metrics.update(
            {
                "source_capture": artifact_relative_path(paths[name]),
                "carrier_if_hz": processed[name].carrier.frequency_hz,
                "carrier_snr_db": processed[name].carrier.snr_db,
            }
        )
        metrics["standards"][name] = standard_metrics

    output_dir = Path(args.output_dir)
    _ensure_new_outputs(output_dir, ("calibration.h5", "metrics.json", "summary.png"))
    _write_calibration_h5(
        output_dir / "calibration.h5",
        metrics,
        model,
        standards,
        reference_metadata,
        processing,
        args,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _plot_calibration(output_dir / "summary.png", model, standards)
    return output_dir, metrics


def load_calibration(path: Path) -> LoadedCalibration:
    path = Path(path)
    try:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("format", "") != CALIBRATION_FORMAT:
                raise CalibrationError(f"unsupported calibration format: {path}")
            coefficients = handle["coefficients"]
            model = OSLModel(
                a=_read_complex_attributes(coefficients, "a"),
                b=_read_complex_attributes(coefficients, "b"),
                c=_read_complex_attributes(coefficients, "c"),
                condition_number=float(coefficients.attrs["condition_number"]),
                solve_residual_rms=float(coefficients.attrs["solve_residual_rms"]),
                minimum_measured_separation=float(
                    coefficients.attrs["minimum_measured_separation"]
                ),
            )
            reference_metadata = read_attributes(handle["reference_metadata"])
            processing_values = read_attributes(handle["processing"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(f"cannot read calibration {path}: {exc}") from exc
    processing = ProcessingSettings(
        output_rate_hz=float(processing_values["output_rate_hz"]),
        lowpass_cutoff_hz=float(processing_values["lowpass_cutoff_hz"]),
        lowpass_order=int(processing_values["lowpass_order"]),
        transient_s=float(processing_values["transient_s"]),
        chunk_samples=int(processing_values["chunk_samples"]),
    )
    metrics_path = path.with_name("metrics.json")
    metrics: dict[str, Any] = {}
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metrics = {}
    return LoadedCalibration(
        path=path,
        model=model,
        reference_metadata=reference_metadata,
        processing=processing,
        search_half_width_hz=float(processing_values["search_half_width_hz"]),
        carrier_min_snr_db=float(processing_values["carrier_min_snr_db"]),
        fft_size=int(processing_values["fft_size"]),
        fft_segments=int(processing_values["fft_segments"]),
        metrics=metrics,
    )


def _dataset_options(length: int) -> dict[str, Any]:
    chunk = min(max(1024, length // 64), 262_144, length)
    return {"chunks": (chunk,), "compression": "gzip", "compression_opts": 4, "shuffle": True}


def _write_calibrated_h5(
    path: Path,
    calibration: LoadedCalibration,
    capture: ProcessedCapture,
    calibrated_result: ReflectionResult,
    response: ResponseAnalysis,
    denominator_floor: float,
) -> None:
    with h5py.File(path, "x") as handle:
        handle.attrs["format"] = CALIBRATED_FORMAT
        handle.attrs["schema_version"] = CALIBRATED_SCHEMA_VERSION
        handle.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        handle.attrs["source_capture"] = artifact_relative_path(capture.path)
        handle.attrs["source_calibration"] = artifact_relative_path(calibration.path)
        metadata = handle.create_group("metadata")
        write_attributes(metadata, capture.metadata)
        write_attributes(
            metadata,
            {
                "carrier_estimate_hz": capture.carrier.frequency_hz,
                "output_rate_hz": calibrated_result.output_rate_hz,
                "calibration_denominator_floor": denominator_floor,
            },
        )
        processed = handle.create_group("processed")
        options = _dataset_options(len(calibrated_result.time_s))
        processed.create_dataset("time_s", data=calibrated_result.time_s, **options)
        processed.create_dataset("forward_complex", data=calibrated_result.forward_complex, **options)
        processed.create_dataset("reverse_complex", data=calibrated_result.reverse_complex, **options)
        processed.create_dataset("measured_G_complex", data=capture.result.g_complex, **options)
        processed.create_dataset("calibrated_gamma", data=calibrated_result.g_complex, **options)
        processed.create_dataset("calibrated_magnitude", data=calibrated_result.g_magnitude, **options)
        processed.create_dataset("calibrated_phase", data=calibrated_result.g_phase, **options)
        processed.create_dataset("calibrated_phase_unwrapped", data=response.phase_unwrapped, **options)
        processed.create_dataset("valid_calibration", data=calibrated_result.valid_ratio, **options)
        spectra = handle.create_group("spectra")
        mag_options = _dataset_options(len(response.magnitude_spectrum_hz))
        phase_options = _dataset_options(len(response.phase_spectrum_hz))
        spectra.create_dataset("magnitude_frequency_hz", data=response.magnitude_spectrum_hz, **mag_options)
        spectra.create_dataset("magnitude_psd", data=response.magnitude_psd, **mag_options)
        spectra.create_dataset("phase_frequency_hz", data=response.phase_spectrum_hz, **phase_options)
        spectra.create_dataset("phase_psd", data=response.phase_psd, **phase_options)


def _plot_applied(
    path: Path,
    result: ReflectionResult,
    response: ResponseAnalysis,
    tone_frequency_hz: float,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle("OSL-calibrated one-port reflection", fontsize=15)
    step = max(1, int(math.ceil(len(result.time_s) / 20_000)))
    view = slice(None, None, step)
    gamma = result.g_complex[result.valid_ratio]
    gamma_view = gamma[:: max(1, int(math.ceil(len(gamma) / 20_000)))]
    theta = np.linspace(0.0, 2.0 * np.pi, 721)
    axes[0, 0].plot(np.cos(theta), np.sin(theta), linestyle="--", color="0.6")
    axes[0, 0].scatter(gamma_view.real, gamma_view.imag, s=2, alpha=0.35)
    axes[0, 0].set(title="Calibrated Gamma plane", xlabel="Real", ylabel="Imaginary", aspect="equal")
    axes[0, 1].plot(result.time_s[view], np.abs(result.forward_complex[view]), label="|V_fwd|")
    axes[0, 1].plot(result.time_s[view], np.abs(result.reverse_complex[view]), label="|V_rev|")
    axes[0, 1].set(title="Carrier envelopes", xlabel="Time (s)", ylabel="Magnitude")
    axes[0, 1].legend()
    axes[1, 0].plot(result.time_s[view], result.g_magnitude[view], linewidth=0.8)
    axes[1, 0].set(title="|Gamma(t)|", xlabel="Time (s)", ylabel="Magnitude")
    axes[1, 1].plot(result.time_s[view], np.degrees(result.g_phase[view]), linewidth=0.8)
    axes[1, 1].set(title="angle Gamma(t)", xlabel="Time (s)", ylabel="Phase (deg)")
    mag_band = response.magnitude_spectrum_hz <= 5000.0
    phase_band = response.phase_spectrum_hz <= 5000.0
    axes[2, 0].plot(
        response.magnitude_spectrum_hz[mag_band],
        10 * np.log10(np.maximum(response.magnitude_psd[mag_band], np.finfo(float).tiny)),
    )
    axes[2, 0].axvline(tone_frequency_hz, color="red", linestyle="--")
    axes[2, 0].set(title="|Gamma| modulation spectrum", xlabel="Frequency (Hz)", ylabel="PSD (dB/Hz)")
    axes[2, 1].plot(
        response.phase_spectrum_hz[phase_band],
        10 * np.log10(np.maximum(response.phase_psd[phase_band], np.finfo(float).tiny)),
    )
    axes[2, 1].axvline(tone_frequency_hz, color="red", linestyle="--")
    axes[2, 1].set(title="Gamma phase modulation spectrum", xlabel="Frequency (Hz)", ylabel="PSD (rad^2/Hz)")
    for axis in axes.flat:
        axis.grid(True, linestyle=":", alpha=0.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def apply_calibration(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    calibration = load_calibration(args.calibration)
    capture = _process_capture(
        args.capture,
        calibration.processing,
        search_half_width_hz=calibration.search_half_width_hz,
        carrier_min_snr_db=calibration.carrier_min_snr_db,
        fft_size=calibration.fft_size,
        fft_segments=calibration.fft_segments,
    )
    assert_compatible(calibration.reference_metadata, capture.metadata, "DUT capture")
    calibrated, valid, denominator_floor = apply_osl(
        capture.result.g_complex,
        calibration.model,
        capture.result.valid_ratio,
        relative_denominator_floor=args.denominator_floor,
    )
    calibrated_result = ReflectionResult(
        time_s=capture.result.time_s,
        forward_complex=capture.result.forward_complex,
        reverse_complex=capture.result.reverse_complex,
        g_complex=calibrated,
        g_magnitude=np.abs(calibrated).astype(np.float32),
        g_phase=np.angle(calibrated).astype(np.float32),
        valid_ratio=valid,
        forward_floor=capture.result.forward_floor,
        output_rate_hz=capture.result.output_rate_hz,
        decimation=capture.result.decimation,
    )
    tone_frequency_hz = _finite_float(
        capture.metadata.get("tone_frequency_hz"), "tone_frequency_hz"
    )
    if tone_frequency_hz <= 0:
        raise CalibrationError("DUT capture has no positive tone_frequency_hz metadata")
    response = analyze_response(calibrated_result, tone_frequency_hz)
    valid_gamma = calibrated[valid]
    metrics: dict[str, Any] = {
        "format": CALIBRATED_FORMAT,
        "schema_version": CALIBRATED_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_capture": artifact_relative_path(Path(args.capture)),
        "source_calibration": artifact_relative_path(Path(args.calibration)),
        "carrier": {
            "estimated_if_frequency_hz": capture.carrier.frequency_hz,
            "forward_snr_db": capture.carrier.snr_db,
        },
        "calibration_model": calibration.model.to_dict(),
        "calibrated": {
            "valid_samples": int(np.count_nonzero(valid)),
            "total_samples": int(valid.size),
            "valid_fraction": float(np.mean(valid)),
            "denominator_floor": denominator_floor,
            "mean_magnitude": float(np.mean(np.abs(valid_gamma))),
            "std_magnitude": float(np.std(np.abs(valid_gamma))),
            "circular_mean_phase_deg": float(
                np.degrees(np.angle(np.mean(np.exp(1j * np.angle(valid_gamma)))))
            ),
            "fraction_magnitude_above_1p05": float(np.mean(np.abs(valid_gamma) > 1.05)),
        },
        **response.metrics,
    }
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.capture).parent / f"{Path(args.capture).stem}_osl_{timestamp}"
    else:
        output_dir = Path(args.output_dir)
    _ensure_new_outputs(output_dir, ("calibrated.h5", "metrics.json", "summary.png"))
    _write_calibrated_h5(
        output_dir / "calibrated.h5",
        calibration,
        capture,
        calibrated_result,
        response,
        denominator_floor,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _plot_applied(output_dir / "summary.png", calibrated_result, response, tone_frequency_hz)
    return output_dir, metrics


def _add_processing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--search-half-width", type=float, default=20e3)
    parser.add_argument("--carrier-min-snr", type=float, default=10.0)
    parser.add_argument("--fft-size", type=int, default=262_144)
    parser.add_argument("--fft-segments", type=int, default=16)
    parser.add_argument("--output-rate", type=float, default=50e3)
    parser.add_argument("--lowpass-cutoff", type=float, default=10e3)
    parser.add_argument("--lowpass-order", type=int, default=8)
    parser.add_argument("--transient", type=float, default=0.005)
    parser.add_argument("--analysis-chunk-samples", type=int, default=1_048_576)
    parser.add_argument("--standard-settle", type=float, default=0.1)
    parser.add_argument("--max-relative-standard-std", type=float, default=0.05)
    parser.add_argument("--open-gamma", type=_complex_argument, default=1.0 + 0.0j)
    parser.add_argument("--short-gamma", type=_complex_argument, default=-1.0 + 0.0j)
    parser.add_argument("--load-gamma", type=_complex_argument, default=0.0 + 0.0j)


def _add_hardware_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--duration", type=float, default=DEFAULT_CALIBRATION_DURATION_S)
    parser.add_argument("--serial", default=DEFAULT_SERIAL)
    parser.add_argument("--signal-freq", type=float, default=DEFAULT_SIGNAL_FREQUENCY_HZ)
    parser.add_argument("--lo-offset", type=float, default=DEFAULT_LO_OFFSET_HZ)
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--bandwidth", type=float, default=DEFAULT_BANDWIDTH_HZ)
    parser.add_argument("--rx0-gain", type=float, default=DEFAULT_GAIN_DB)
    parser.add_argument("--rx1-gain", type=float, default=DEFAULT_GAIN_DB)
    parser.add_argument("--chunk-samples", type=int, default=DEFAULT_CHUNK_SAMPLES)
    parser.add_argument("--queue-depth", type=int, default=8)
    parser.add_argument("--tune-settle", type=float, default=0.2)
    parser.add_argument("--stream-start-delay", type=float, default=0.1)
    parser.add_argument("--clip-level", type=float, default=0.98)
    parser.add_argument("--max-clip-fraction", type=float, default=1e-4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire, solve, and apply a one-frequency B210 Open/Short/Load calibration."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser(
        "capture", help="interactively capture O/S/L without reinitializing the B210"
    )
    _add_hardware_arguments(capture_parser)
    _add_processing_arguments(capture_parser)
    capture_parser.add_argument("--output-dir", type=Path)

    build_parser_command = subparsers.add_parser(
        "build", help="build a calibration from three existing valid raw captures"
    )
    build_parser_command.add_argument("--open", type=Path, required=True)
    build_parser_command.add_argument("--short", type=Path, required=True)
    build_parser_command.add_argument("--load", type=Path, required=True)
    build_parser_command.add_argument("--output-dir", type=Path, required=True)
    _add_processing_arguments(build_parser_command)

    apply_parser = subparsers.add_parser(
        "apply", help="apply a calibration to a valid raw DUT capture"
    )
    apply_parser.add_argument("calibration", type=Path)
    apply_parser.add_argument("capture", type=Path)
    apply_parser.add_argument("--output-dir", type=Path)
    apply_parser.add_argument("--denominator-floor", type=float, default=1e-9)
    return parser


def _capture_settings(args: argparse.Namespace, output_dir: Path) -> CaptureSettings:
    return CaptureSettings(
        tone_frequency_hz=None,
        serial=args.serial,
        signal_frequency_hz=args.signal_freq,
        lo_offset_hz=args.lo_offset,
        sample_rate_hz=args.sample_rate,
        bandwidth_hz=args.bandwidth,
        rx0_gain_db=args.rx0_gain,
        rx1_gain_db=args.rx1_gain,
        duration_s=args.duration,
        output_dir=output_dir,
        chunk_samples=args.chunk_samples,
        queue_depth=args.queue_depth,
        tune_settle_s=args.tune_settle,
        stream_start_delay_s=args.stream_start_delay,
        clip_level=args.clip_level,
        max_clip_fraction=args.max_clip_fraction,
    )


def capture_and_build(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path(__file__).parent / "calibrations" / f"osl_{timestamp}"
    else:
        session_dir = Path(args.output_dir)
    session_dir.mkdir(parents=True, exist_ok=False)
    settings = _capture_settings(args, session_dir)
    settings.validate()
    print("B210 one-port OSL calibration capture")
    print("  B210 initialization, frequency, gain, and channel configuration remain fixed.")
    print("  The calibration plane must match the later DUT connection plane.")
    receiver = B210DualReceiver(settings)
    paths: dict[str, Path] = {}
    display_names = {"open": "OPEN", "short": "SHORT", "load": "LOAD (50 ohm)"}
    for name in STANDARD_ORDER:
        input(
            f"\nDisable the external CW and connect {display_names[name]}; restore the "
            "same safe CW power, wait for settling, then press Enter to capture..."
        )
        path = session_dir / f"{name}.h5"
        quality = receiver.capture(
            path,
            extra_metadata={
                "capture_purpose": "one_port_osl_calibration",
                "calibration_standard": name,
            },
        )
        print(
            f"  {name}: samples={quality.received_rx0}/{quality.target_samples}, "
            f"peak=({quality.peak_component_rx0:.4f}, {quality.peak_component_rx1:.4f}), "
            f"quality={'VALID' if quality.valid else 'INVALID'}"
        )
        if not quality.valid:
            raise CalibrationError(
                f"{name} capture is invalid: " + "; ".join(quality.errors)
            )
        paths[name] = path
    print("\nAll three standards are captured. Disable the external CW before disconnection.")
    build_args = argparse.Namespace(**vars(args))
    for name, path in paths.items():
        setattr(build_args, name, path)
    build_args.output_dir = session_dir / "calibration"
    return build_calibration(build_args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "capture":
            output_dir, metrics = capture_and_build(args)
            print("\nOSL calibration created successfully")
            print(f"  condition={metrics['model']['condition_number']:.3e}")
            print(f"  calibration={artifact_relative_path(output_dir / 'calibration.h5')}")
            return 0
        if args.command == "build":
            output_dir, metrics = build_calibration(args)
            print("OSL calibration created successfully")
            for name in STANDARD_ORDER:
                item = metrics["standards"][name]
                measured = complex_from_dict(item["measured_g"])
                print(
                    f"  {name}: measured G={measured.real:+.6f}{measured.imag:+.6f}j, "
                    f"std={item['complex_std']:.3e}"
                )
            print(f"  condition={metrics['model']['condition_number']:.3e}")
            print(f"  calibration={artifact_relative_path(output_dir / 'calibration.h5')}")
            return 0
        if args.command == "apply":
            output_dir, metrics = apply_calibration(args)
            calibrated = metrics["calibrated"]
            tone = metrics["tone"]
            print("OSL-calibrated one-port reflection analysis")
            print(
                f"  valid Gamma={calibrated['valid_fraction']*100:.3f}%, "
                f"mean |Gamma|={calibrated['mean_magnitude']:.6f}, "
                f"mean phase={calibrated['circular_mean_phase_deg']:.3f} deg"
            )
            print(
                f"  tone={tone['frequency_hz']:.0f} Hz, "
                f"|Gamma| SNR={tone['magnitude_spectrum']['target_snr_db']:.2f} dB, "
                f"phase SNR={tone['phase_spectrum']['target_snr_db']:.2f} dB"
            )
            print(f"  output={artifact_relative_path(output_dir)}")
            return 0
        raise AssertionError(args.command)
    except KeyboardInterrupt:
        print("\nCalibration cancelled. Verify that the external CW is disabled.", file=sys.stderr)
        return 130
    except (
        CalibrationError,
        CaptureConfigurationError,
        CaptureHardwareError,
        CaptureFileError,
        ProcessingError,
        FileExistsError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
