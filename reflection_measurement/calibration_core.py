#!/usr/bin/env python3
"""Pure OSL calibration algorithms for the measured complex ratio G."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


class CalibrationError(RuntimeError):
    """Raised when an OSL model cannot be solved or safely applied."""


@dataclass(frozen=True)
class StandardEstimate:
    measured_g: complex
    standard_gamma: complex
    samples_used: int
    samples_available: int
    valid_fraction: float
    complex_std: float
    standard_error: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "measured_g": complex_to_dict(self.measured_g),
            "standard_gamma": complex_to_dict(self.standard_gamma),
            "samples_used": self.samples_used,
            "samples_available": self.samples_available,
            "valid_fraction": self.valid_fraction,
            "complex_std": self.complex_std,
            "standard_error": self.standard_error,
        }


@dataclass(frozen=True)
class OSLModel:
    """Möbius map m=(a*Gamma+b)/(c*Gamma+1)."""

    a: complex
    b: complex
    c: complex
    condition_number: float
    solve_residual_rms: float
    minimum_measured_separation: float

    @property
    def directivity(self) -> complex:
        return self.b

    @property
    def source_match(self) -> complex:
        return -self.c

    @property
    def reflection_tracking(self) -> complex:
        # m = directivity + tracking*Gamma/(1-source_match*Gamma)
        return self.a - self.b * self.c

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "measured_g=(a*gamma+b)/(c*gamma+1)",
            "inverse": "gamma=(measured_g-b)/(a-c*measured_g)",
            "a": complex_to_dict(self.a),
            "b": complex_to_dict(self.b),
            "c": complex_to_dict(self.c),
            "directivity": complex_to_dict(self.directivity),
            "source_match": complex_to_dict(self.source_match),
            "reflection_tracking": complex_to_dict(self.reflection_tracking),
            "condition_number": self.condition_number,
            "solve_residual_rms": self.solve_residual_rms,
            "minimum_measured_separation": self.minimum_measured_separation,
        }


def complex_to_dict(value: complex) -> dict[str, float]:
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def complex_from_dict(value: Mapping[str, Any]) -> complex:
    try:
        result = complex(float(value["real"]), float(value["imag"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError("invalid serialized complex number") from exc
    if not np.isfinite(result):
        raise CalibrationError("serialized complex number is not finite")
    return result


def _finite_complex_vector(values: Any, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.complex128)
    if result.ndim != 1 or result.size < 3 or not np.all(np.isfinite(result)):
        raise CalibrationError(f"{name} must contain at least three finite complex values")
    return result


def forward_model(gamma: Any, model: OSLModel) -> np.ndarray:
    gamma_values = np.asarray(gamma, dtype=np.complex128)
    denominator = model.c * gamma_values + 1.0
    if np.any(np.abs(denominator) <= np.finfo(float).eps):
        raise CalibrationError("forward OSL model is singular for one or more values")
    return (model.a * gamma_values + model.b) / denominator


def solve_osl(
    measured_g: Any,
    standard_gamma: Any,
    *,
    maximum_condition_number: float = 1e12,
) -> OSLModel:
    """Solve a three-term one-port calibration from three or more standards."""
    measured = _finite_complex_vector(measured_g, "measured standards")
    actual = _finite_complex_vector(standard_gamma, "known standard gamma values")
    if measured.shape != actual.shape:
        raise CalibrationError("measured and known standard vectors must have equal length")
    if not math.isfinite(maximum_condition_number) or maximum_condition_number <= 1:
        raise CalibrationError("maximum condition number must be finite and greater than one")

    actual_separations = np.abs(actual[:, None] - actual[None, :])
    measured_separations = np.abs(measured[:, None] - measured[None, :])
    upper = np.triu_indices(actual.size, 1)
    if float(np.min(actual_separations[upper])) <= 1e-9:
        raise CalibrationError("known standard gamma values are not distinct")
    minimum_measured_separation = float(np.min(measured_separations[upper]))
    scale = max(float(np.max(np.abs(measured))), 1.0)
    if minimum_measured_separation <= 1e-9 * scale:
        raise CalibrationError("measured Open/Short/Load values are not distinct")

    # m_i(c*Gamma_i+1)=a*Gamma_i+b
    design = np.column_stack((actual, np.ones_like(actual), -measured * actual))
    condition_number = float(np.linalg.cond(design))
    if not math.isfinite(condition_number) or condition_number > maximum_condition_number:
        raise CalibrationError(
            f"OSL solve is ill-conditioned ({condition_number:.3e}); check standards and wiring"
        )
    coefficients, _, rank, _ = np.linalg.lstsq(design, measured, rcond=None)
    if rank != 3:
        raise CalibrationError("OSL solve does not have full rank")
    model = OSLModel(
        a=complex(coefficients[0]),
        b=complex(coefficients[1]),
        c=complex(coefficients[2]),
        condition_number=condition_number,
        solve_residual_rms=0.0,
        minimum_measured_separation=minimum_measured_separation,
    )
    predicted = forward_model(actual, model)
    residual_rms = float(np.sqrt(np.mean(np.abs(predicted - measured) ** 2)))
    return OSLModel(
        a=model.a,
        b=model.b,
        c=model.c,
        condition_number=model.condition_number,
        solve_residual_rms=residual_rms,
        minimum_measured_separation=model.minimum_measured_separation,
    )


def apply_osl(
    measured_g: Any,
    model: OSLModel,
    valid: Any | None = None,
    *,
    relative_denominator_floor: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Invert the OSL map, masking non-finite and near-singular samples."""
    measured = np.asarray(measured_g, dtype=np.complex128)
    if measured.ndim != 1 or measured.size == 0:
        raise CalibrationError("measured G must be a non-empty vector")
    if not math.isfinite(relative_denominator_floor) or relative_denominator_floor <= 0:
        raise CalibrationError("relative denominator floor must be positive and finite")
    if valid is None:
        valid_mask = np.ones(measured.shape, dtype=bool)
    else:
        valid_mask = np.asarray(valid, dtype=bool)
        if valid_mask.shape != measured.shape:
            raise CalibrationError("G validity mask has the wrong shape")
    valid_mask &= np.isfinite(measured)
    denominator = model.a - model.c * measured
    finite_denominator = np.abs(denominator[valid_mask])
    reference = (
        float(np.median(finite_denominator)) if finite_denominator.size else 0.0
    )
    floor = max(np.finfo(float).eps, relative_denominator_floor * reference)
    valid_mask &= np.isfinite(denominator) & (np.abs(denominator) > floor)
    calibrated = np.full(measured.shape, np.nan + 1j * np.nan, dtype=np.complex64)
    calibrated[valid_mask] = (
        (measured[valid_mask] - model.b) / denominator[valid_mask]
    ).astype(np.complex64)
    return calibrated, valid_mask, floor


def robust_complex_center(
    values: Any,
    *,
    outlier_sigma: float = 8.0,
) -> tuple[complex, float, np.ndarray]:
    """Return an outlier-resistant complex center and complex RMS dispersion."""
    samples = np.asarray(values, dtype=np.complex128)
    if samples.ndim != 1:
        raise CalibrationError("standard samples must be a vector")
    finite = np.isfinite(samples)
    if np.count_nonzero(finite) < 32:
        raise CalibrationError("fewer than 32 finite standard samples are available")
    samples = samples[finite]
    initial = complex(np.median(samples.real), np.median(samples.imag))
    radial = np.abs(samples - initial)
    radial_median = float(np.median(radial))
    radial_mad = 1.4826 * float(np.median(np.abs(radial - radial_median)))
    scale = max(radial_median + radial_mad, np.finfo(float).eps)
    keep = radial <= outlier_sigma * scale
    if np.count_nonzero(keep) < max(32, int(0.5 * samples.size)):
        raise CalibrationError("standard data are too unstable after outlier rejection")
    retained = samples[keep]
    center = complex(np.mean(retained))
    dispersion = float(np.sqrt(np.mean(np.abs(retained - center) ** 2)))
    full_keep = np.zeros(finite.shape, dtype=bool)
    full_keep[np.flatnonzero(finite)[keep]] = True
    return center, dispersion, full_keep
