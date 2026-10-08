#!/usr/bin/env python3
"""HDF5 I/O and capture-quality rules for dual-channel B210 recordings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np


FORMAT_NAME = "b210_dual_rx_reflection_capture"
SCHEMA_VERSION = 1
ARTIFACT_ROOT = Path(__file__).resolve().parent


def artifact_relative_path(path: Path) -> str:
    """Return a repository-relative path without exposing a host filesystem path."""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ARTIFACT_ROOT))
    except ValueError:
        return candidate.name


class CaptureFileError(RuntimeError):
    """Raised when a raw capture file is missing data or is marked invalid."""


def _attribute_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, bytes, bool, int, float, np.number)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_attributes(target: h5py.Group, values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        target.attrs[key] = _attribute_value(value)


def read_attributes(target: h5py.Group) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in target.attrs.items():
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        elif isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, str) and value[:1] in {"[", "{"}:
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        result[str(key)] = value
    return result


@dataclass(frozen=True)
class CaptureQuality:
    """All conditions used to decide whether a trace may be analyzed."""

    target_samples: int
    received_rx0: int
    received_rx1: int
    lo_locked_rx0: bool
    lo_locked_rx1: bool
    overflow_count: int = 0
    timeout_count: int = 0
    late_command_count: int = 0
    other_error_count: int = 0
    peak_component_rx0: float = 0.0
    peak_component_rx1: float = 0.0
    clip_fraction_rx0: float = 0.0
    clip_fraction_rx1: float = 0.0
    clip_level: float = 0.98
    max_clip_fraction: float = 1e-4
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def clipped(self) -> bool:
        return bool(
            self.clip_fraction_rx0 > self.max_clip_fraction
            or self.clip_fraction_rx1 > self.max_clip_fraction
        )

    @property
    def samples_complete(self) -> bool:
        return bool(
            self.received_rx0 == self.target_samples
            and self.received_rx1 == self.target_samples
        )

    @property
    def valid(self) -> bool:
        return bool(
            self.samples_complete
            and self.received_rx0 == self.received_rx1
            and self.lo_locked_rx0
            and self.lo_locked_rx1
            and self.overflow_count == 0
            and self.timeout_count == 0
            and self.late_command_count == 0
            and self.other_error_count == 0
            and not self.clipped
        )

    def to_attributes(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "samples_complete": self.samples_complete,
            "target_samples": self.target_samples,
            "received_samples_rx0": self.received_rx0,
            "received_samples_rx1": self.received_rx1,
            "lo_locked_rx0": self.lo_locked_rx0,
            "lo_locked_rx1": self.lo_locked_rx1,
            "overflow_count": self.overflow_count,
            "timeout_count": self.timeout_count,
            "late_command_count": self.late_command_count,
            "other_error_count": self.other_error_count,
            "peak_component_rx0": self.peak_component_rx0,
            "peak_component_rx1": self.peak_component_rx1,
            "clip_fraction_rx0": self.clip_fraction_rx0,
            "clip_fraction_rx1": self.clip_fraction_rx1,
            "clip_level": self.clip_level,
            "max_clip_fraction": self.max_clip_fraction,
            "clipped": self.clipped,
            "errors_json": list(self.errors),
        }


class RawCaptureWriter:
    """Sequential, uncompressed writer for one finite two-channel capture."""

    def __init__(self, path: Path, target_samples: int, chunk_samples: int):
        if target_samples < 1:
            raise ValueError("target_samples must be positive")
        if chunk_samples < 1:
            raise ValueError("chunk_samples must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.target_samples = int(target_samples)
        self.chunk_samples = min(int(chunk_samples), self.target_samples)
        self._written = 0
        self._closed = False
        self._file = h5py.File(self.path, "x")
        self._file.attrs["format"] = FORMAT_NAME
        self._file.attrs["schema_version"] = SCHEMA_VERSION
        self._file.attrs["status"] = "in_progress"
        raw = self._file.create_group("raw")
        dataset_options = {
            "shape": (self.target_samples,),
            "maxshape": (self.target_samples,),
            "dtype": np.complex64,
            "chunks": (self.chunk_samples,),
        }
        self._rx0 = raw.create_dataset("rx0_iq", **dataset_options)
        self._rx1 = raw.create_dataset("rx1_iq", **dataset_options)
        self._metadata = self._file.create_group("metadata")
        self._quality = self._file.create_group("quality")
        self._quality.attrs["valid"] = False

    @property
    def written_samples(self) -> int:
        return self._written

    def append(self, samples: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("capture writer is closed")
        block = np.asarray(samples, dtype=np.complex64)
        if block.ndim != 2 or block.shape[0] != 2:
            raise ValueError("samples must have shape (2, count)")
        count = int(block.shape[1])
        if self._written + count > self.target_samples:
            raise ValueError("capture data exceeds the declared target length")
        stop = self._written + count
        self._rx0[self._written : stop] = block[0]
        self._rx1[self._written : stop] = block[1]
        self._written = stop

    def finalize(self, metadata: Mapping[str, Any], quality: CaptureQuality) -> None:
        if self._closed:
            raise RuntimeError("capture writer is closed")
        if quality.received_rx0 != self._written or quality.received_rx1 != self._written:
            raise ValueError(
                "quality sample counts do not match HDF5 data: "
                f"quality=({quality.received_rx0}, {quality.received_rx1}), "
                f"written={self._written}"
            )
        if self._written < self.target_samples:
            self._rx0.resize((self._written,))
            self._rx1.resize((self._written,))
        write_attributes(self._metadata, metadata)
        write_attributes(self._quality, quality.to_attributes())
        self._file.attrs["status"] = "complete" if quality.valid else "invalid"
        self._file.flush()
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._file.close()
            self._closed = True

    def __enter__(self) -> "RawCaptureWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def inspect_capture(path: Path, require_valid: bool = False) -> dict[str, Any]:
    """Validate the schema and return metadata without loading IQ into memory."""
    path = Path(path)
    try:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("format", "") != FORMAT_NAME:
                raise CaptureFileError(f"unsupported capture format: {path}")
            if "raw/rx0_iq" not in handle or "raw/rx1_iq" not in handle:
                raise CaptureFileError("capture is missing one or both raw IQ datasets")
            if "metadata" not in handle or "quality" not in handle:
                raise CaptureFileError("capture is missing metadata or quality information")
            length_rx0 = int(handle["raw/rx0_iq"].shape[0])
            length_rx1 = int(handle["raw/rx1_iq"].shape[0])
            if length_rx0 != length_rx1:
                raise CaptureFileError(
                    f"raw channel lengths differ: RX0={length_rx0}, RX1={length_rx1}"
                )
            metadata = read_attributes(handle["metadata"])
            quality = read_attributes(handle["quality"])
            valid = bool(quality.get("valid", False))
            if require_valid and not valid:
                errors = quality.get("errors_json", "[]")
                raise CaptureFileError(f"capture is marked invalid: {errors}")
            return {
                "path": artifact_relative_path(path),
                "samples": length_rx0,
                "metadata": metadata,
                "quality": quality,
                "valid": valid,
            }
    except OSError as exc:
        raise CaptureFileError(f"cannot open capture {path}: {exc}") from exc
