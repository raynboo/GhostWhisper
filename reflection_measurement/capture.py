#!/usr/bin/env python3
"""Synchronously acquire Forward/Reverse IQ with one B210 RX streamer."""

from __future__ import annotations

import argparse
import hashlib
import math
import queue
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from capture_io import (
    CaptureQuality,
    RawCaptureWriter,
    artifact_relative_path,
    inspect_capture,
)


DEFAULT_SERIAL = ""
DEFAULT_SIGNAL_FREQUENCY_HZ = 601e6
DEFAULT_LO_OFFSET_HZ = 500e3
DEFAULT_SAMPLE_RATE_HZ = 5e6
DEFAULT_BANDWIDTH_HZ = 4e6
DEFAULT_GAIN_DB = 10.0
DEFAULT_DURATION_S = 10.0
DEFAULT_CHUNK_SAMPLES = 262_144
TONE_FREQUENCIES_HZ = (1000, 2000, 3000)


class CaptureConfigurationError(ValueError):
    pass


class CaptureHardwareError(RuntimeError):
    pass


@dataclass(frozen=True)
class CaptureSettings:
    tone_frequency_hz: int | None
    serial: str = DEFAULT_SERIAL
    signal_frequency_hz: float = DEFAULT_SIGNAL_FREQUENCY_HZ
    lo_offset_hz: float = DEFAULT_LO_OFFSET_HZ
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ
    bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ
    rx0_gain_db: float = DEFAULT_GAIN_DB
    rx1_gain_db: float = DEFAULT_GAIN_DB
    duration_s: float = DEFAULT_DURATION_S
    output_dir: Path = Path("captures")
    chunk_samples: int = DEFAULT_CHUNK_SAMPLES
    queue_depth: int = 8
    tune_settle_s: float = 0.2
    stream_start_delay_s: float = 0.1
    clip_level: float = 0.98
    max_clip_fraction: float = 1e-4

    @property
    def center_frequency_hz(self) -> float:
        return self.signal_frequency_hz - self.lo_offset_hz

    def validate(self) -> None:
        numeric = (
            self.signal_frequency_hz,
            self.lo_offset_hz,
            self.sample_rate_hz,
            self.bandwidth_hz,
            self.rx0_gain_db,
            self.rx1_gain_db,
            self.duration_s,
            self.tune_settle_s,
            self.stream_start_delay_s,
            self.clip_level,
            self.max_clip_fraction,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise CaptureConfigurationError("all numeric settings must be finite")
        if (
            self.tone_frequency_hz is not None
            and self.tone_frequency_hz not in TONE_FREQUENCIES_HZ
        ):
            raise CaptureConfigurationError(
                f"tone frequency must be one of {TONE_FREQUENCIES_HZ} Hz"
            )
        if self.signal_frequency_hz <= 0 or self.center_frequency_hz <= 0:
            raise CaptureConfigurationError("signal and center frequencies must be positive")
        if self.sample_rate_hz <= 0 or self.bandwidth_hz <= 0 or self.duration_s <= 0:
            raise CaptureConfigurationError("sample rate, bandwidth and duration must be positive")
        if abs(self.lo_offset_hz) + 20e3 >= self.sample_rate_hz / 2:
            raise CaptureConfigurationError(
                "LO offset plus the 20 kHz carrier-search margin must be below Nyquist"
            )
        if self.chunk_samples < 1024 or self.queue_depth < 1:
            raise CaptureConfigurationError("chunk size must be >=1024 and queue depth >=1")
        if self.stream_start_delay_s <= 0:
            raise CaptureConfigurationError("stream start delay must be positive")
        if not 0 < self.clip_level <= 1 or not 0 <= self.max_clip_fraction < 1:
            raise CaptureConfigurationError("invalid clipping thresholds")


def _load_uhd() -> Any:
    try:
        import uhd
    except ImportError as exc:
        raise CaptureHardwareError(
            "UHD Python bindings are not installed; install the system UHD package"
        ) from exc
    return uhd


def _range_bounds(range_object: Any) -> tuple[float, float]:
    return float(range_object.start()), float(range_object.stop())


def _find_usb_speed(serial: str) -> float | None:
    root = Path("/sys/bus/usb/devices")
    if not root.exists():
        return None
    for serial_path in root.glob("*/serial"):
        try:
            if serial_path.read_text(encoding="utf-8").strip() != serial:
                continue
            return float((serial_path.parent / "speed").read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
    return None


class _WriterWorker:
    """Keep HDF5 writes off the time-critical UHD receive loop."""

    def __init__(self, path: Path, target_samples: int, chunk_samples: int, depth: int):
        self.path = path
        self.target_samples = target_samples
        self.chunk_samples = chunk_samples
        self.items: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=depth)
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="hdf5-writer", daemon=True)

    def start(self) -> None:
        self._thread.start()
        if not self.ready.wait(timeout=5.0):
            raise CaptureHardwareError("HDF5 writer did not start within five seconds")
        self.raise_if_failed()

    def _run(self) -> None:
        writer: RawCaptureWriter | None = None
        try:
            writer = RawCaptureWriter(self.path, self.target_samples, self.chunk_samples)
            self.ready.set()
            while True:
                kind, payload = self.items.get()
                try:
                    if kind == "data":
                        writer.append(payload)
                    elif kind == "finish":
                        metadata, quality = payload
                        writer.finalize(metadata, quality)
                        writer = None
                        return
                    else:
                        raise RuntimeError(f"unknown writer message: {kind}")
                finally:
                    self.items.task_done()
        except BaseException as exc:
            self.error = exc
            self.ready.set()
        finally:
            if writer is not None:
                writer.close()

    def _put(self, item: tuple[str, Any]) -> None:
        while True:
            self.raise_if_failed()
            try:
                self.items.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    def append(self, samples: np.ndarray) -> None:
        self._put(("data", samples))

    def finish(self, metadata: dict[str, Any], quality: CaptureQuality) -> None:
        self._put(("finish", (metadata, quality)))
        self._thread.join(timeout=60.0)
        if self._thread.is_alive():
            raise CaptureHardwareError("HDF5 writer did not finish within 60 seconds")
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise CaptureHardwareError(f"HDF5 writer failed: {self.error}") from self.error


class B210DualReceiver:
    """Own the B210 and one synchronized two-channel RX streamer."""

    def __init__(self, settings: CaptureSettings):
        settings.validate()
        self.settings = settings
        self.uhd = _load_uhd()
        device_args = "type=b200"
        if settings.serial.strip():
            device_args += f",serial={settings.serial.strip()}"
        self.usrp = self.uhd.usrp.MultiUSRP(device_args)
        if self.usrp.get_mboard_name(0) != "B210":
            raise CaptureHardwareError(
                f"expected a B210, got {self.usrp.get_mboard_name(0)}"
            )

        device_info = dict(self.usrp.get_usrp_rx_info(0))
        actual_serial = str(
            device_info.get("mboard_serial")
            or device_info.get("serial")
            or settings.serial
        ).strip()
        usb_speed = _find_usb_speed(actual_serial) if actual_serial else None
        device_fingerprint = (
            hashlib.sha256(actual_serial.encode("utf-8")).hexdigest()
            if actual_serial
            else "unavailable"
        )
        if usb_speed is not None and usb_speed < 5000:
            raise CaptureHardwareError(
                f"B210 USB speed is {usb_speed:g} Mb/s; dual 4 MS/s capture requires USB 3.0"
            )

        self.usrp.set_rx_subdev_spec(self.uhd.usrp.SubdevSpec("A:A A:B"))
        if self.usrp.get_rx_num_channels() != 2:
            raise CaptureHardwareError("expected exactly two selected B210 RX channels")

        for channel, gain in ((0, settings.rx0_gain_db), (1, settings.rx1_gain_db)):
            antennas = set(self.usrp.get_rx_antennas(channel))
            if "RX2" not in antennas:
                raise CaptureHardwareError(
                    f"RX{channel} has no RX2 port; available antennas: {sorted(antennas)}"
                )
            gain_min, gain_max = _range_bounds(self.usrp.get_rx_gain_range(channel))
            if not gain_min <= gain <= gain_max:
                raise CaptureHardwareError(
                    f"RX{channel} gain {gain} dB is outside {gain_min}..{gain_max} dB"
                )
            freq_min, freq_max = _range_bounds(self.usrp.get_rx_freq_range(channel))
            if not freq_min <= settings.center_frequency_hz <= freq_max:
                raise CaptureHardwareError(
                    f"RX center {settings.center_frequency_hz} Hz is outside "
                    f"{freq_min}..{freq_max} Hz"
                )
            bandwidth_min, bandwidth_max = _range_bounds(
                self.usrp.get_rx_bandwidth_range(channel)
            )
            if not bandwidth_min <= settings.bandwidth_hz <= bandwidth_max:
                raise CaptureHardwareError(
                    f"RX{channel} bandwidth {settings.bandwidth_hz} Hz is outside "
                    f"{bandwidth_min}..{bandwidth_max} Hz"
                )
            self.usrp.set_rx_antenna("RX2", channel)
            self.usrp.set_rx_rate(settings.sample_rate_hz, channel)
            self.usrp.set_rx_freq(
                self.uhd.types.TuneRequest(settings.center_frequency_hz), channel
            )
            self.usrp.set_rx_gain(gain, channel)
            self.usrp.set_rx_bandwidth(settings.bandwidth_hz, channel)

        time.sleep(settings.tune_settle_s)
        self.actual_rates = tuple(float(self.usrp.get_rx_rate(ch)) for ch in (0, 1))
        self.actual_frequencies = tuple(float(self.usrp.get_rx_freq(ch)) for ch in (0, 1))
        if not np.isclose(self.actual_rates[0], self.actual_rates[1], rtol=0, atol=1.0):
            raise CaptureHardwareError(f"RX sample rates differ: {self.actual_rates}")
        if not np.isclose(
            self.actual_rates[0], settings.sample_rate_hz, rtol=1e-6, atol=1.0
        ):
            raise CaptureHardwareError(
                f"requested {settings.sample_rate_hz} S/s, got {self.actual_rates[0]} S/s"
            )
        if not np.isclose(self.actual_frequencies[0], self.actual_frequencies[1], atol=1.0):
            raise CaptureHardwareError(f"RX center frequencies differ: {self.actual_frequencies}")

        stream_args = self.uhd.usrp.StreamArgs("fc32", "sc16")
        stream_args.channels = [0, 1]
        self.streamer = self.usrp.get_rx_stream(stream_args)
        self.lo_locked = tuple(self._lo_locked(channel) for channel in (0, 1))
        self.hardware_metadata = {
            "mboard_name": self.usrp.get_mboard_name(0),
            "device_fingerprint_sha256": device_fingerprint,
            "usb_speed_mbps": usb_speed,
            "rx_subdev_spec": "A:A A:B",
            "rx0_role": "Forward",
            "rx1_role": "Reverse",
            "rx0_antenna": self.usrp.get_rx_antenna(0),
            "rx1_antenna": self.usrp.get_rx_antenna(1),
            "device_selection": "serial-specific" if settings.serial.strip() else "automatic",
        }

    def _lo_locked(self, channel: int) -> bool:
        try:
            names = self.usrp.get_rx_sensor_names(channel)
            return "lo_locked" not in names or bool(
                self.usrp.get_rx_sensor("lo_locked", channel).to_bool()
            )
        except RuntimeError:
            return False

    def _stop_stream(self) -> None:
        try:
            command = self.uhd.types.StreamCMD(self.uhd.types.StreamMode.stop_cont)
            command.stream_now = True
            self.streamer.issue_stream_cmd(command)
        except RuntimeError:
            pass

    def capture(
        self,
        output_path: Path,
        extra_metadata: dict[str, Any] | None = None,
    ) -> CaptureQuality:
        extra_metadata = dict(extra_metadata or {})
        allowed_extra_metadata = {"capture_purpose", "calibration_standard"}
        unsupported = set(extra_metadata).difference(allowed_extra_metadata)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise CaptureHardwareError(f"unsupported extra capture metadata: {names}")
        sample_rate = self.actual_rates[0]
        target_samples = int(round(self.settings.duration_s * sample_rate))
        _check_disk_space(output_path.parent, target_samples)
        writer = _WriterWorker(
            output_path,
            target_samples,
            self.settings.chunk_samples,
            self.settings.queue_depth,
        )
        writer.start()

        aggregate = np.empty((2, self.settings.chunk_samples), dtype=np.complex64)
        aggregate_count = 0
        received = 0
        peak = np.zeros(2, dtype=np.float64)
        clipped_count = np.zeros(2, dtype=np.int64)
        overflow_count = 0
        timeout_count = 0
        late_count = 0
        other_count = 0
        errors: list[str] = []
        interrupted = False

        max_packet = int(self.streamer.get_max_num_samps())
        recv_buffer = np.empty((2, max_packet), dtype=np.complex64)
        rx_metadata = self.uhd.types.RXMetadata()
        none_code = self.uhd.types.RXMetadataErrorCode.none
        overflow_code = self.uhd.types.RXMetadataErrorCode.overflow
        timeout_code = self.uhd.types.RXMetadataErrorCode.timeout
        late_code = getattr(self.uhd.types.RXMetadataErrorCode, "late_command", None)

        command = self.uhd.types.StreamCMD(self.uhd.types.StreamMode.num_done)
        command.num_samps = target_samples
        # A multi-channel streamer must use one shared future hardware time. UHD
        # rejects stream_now=True because it cannot align two independent RX DSPs.
        command.stream_now = False
        hardware_now_s = float(self.usrp.get_time_now().get_real_secs())
        scheduled_start_s = hardware_now_s + self.settings.stream_start_delay_s
        command.time_spec = self.uhd.types.TimeSpec(scheduled_start_s)
        started_utc = datetime.now(timezone.utc).isoformat()
        deadline = (
            time.monotonic()
            + self.settings.stream_start_delay_s
            + self.settings.duration_s
            + 5.0
        )
        stream_started = False

        try:
            self.streamer.issue_stream_cmd(command)
            stream_started = True
            while received < target_samples and time.monotonic() < deadline:
                wanted = min(max_packet, target_samples - received)
                count = int(self.streamer.recv(recv_buffer[:, :wanted], rx_metadata, 1.0))

                if count:
                    block = recv_buffer[:, :count]
                    components = np.maximum(np.abs(block.real), np.abs(block.imag))
                    peak = np.maximum(peak, np.max(components, axis=1))
                    clipped_count += np.count_nonzero(
                        components >= self.settings.clip_level, axis=1
                    )
                    source_start = 0
                    while source_start < count:
                        copy_count = min(
                            count - source_start,
                            self.settings.chunk_samples - aggregate_count,
                        )
                        aggregate[
                            :, aggregate_count : aggregate_count + copy_count
                        ] = block[:, source_start : source_start + copy_count]
                        aggregate_count += copy_count
                        source_start += copy_count
                        if aggregate_count == self.settings.chunk_samples:
                            writer.append(aggregate.copy())
                            aggregate_count = 0
                    received += count

                error_code = rx_metadata.error_code
                if error_code == none_code:
                    pass
                elif error_code == overflow_code:
                    overflow_count += 1
                    errors.append(rx_metadata.strerror())
                elif error_code == timeout_code:
                    timeout_count += 1
                    errors.append(rx_metadata.strerror())
                    if timeout_count >= 3:
                        break
                elif late_code is not None and error_code == late_code:
                    late_count += 1
                    errors.append(rx_metadata.strerror())
                else:
                    other_count += 1
                    errors.append(rx_metadata.strerror())
                    if other_count >= 2:
                        break
        except KeyboardInterrupt:
            interrupted = True
            errors.append("capture interrupted by user")
            other_count += 1
        except CaptureHardwareError:
            raise
        except RuntimeError as exc:
            errors.append(f"UHD stream exception: {exc}")
            other_count += 1
        finally:
            if stream_started:
                self._stop_stream()

        if aggregate_count:
            writer.append(aggregate[:, :aggregate_count].copy())
        if received < target_samples:
            errors.append(f"short capture: {received}/{target_samples} samples per channel")
        if not all(self.lo_locked):
            errors.append(f"RX LO lock failed: RX0={self.lo_locked[0]}, RX1={self.lo_locked[1]}")

        fractions = clipped_count / max(received, 1)
        if np.any(fractions > self.settings.max_clip_fraction):
            errors.append(
                "RX clipping: "
                f"RX0 peak={peak[0]:.4f}, fraction={fractions[0]:.3e}; "
                f"RX1 peak={peak[1]:.4f}, fraction={fractions[1]:.3e}"
            )
        quality = CaptureQuality(
            target_samples=target_samples,
            received_rx0=received,
            received_rx1=received,
            lo_locked_rx0=self.lo_locked[0],
            lo_locked_rx1=self.lo_locked[1],
            overflow_count=overflow_count,
            timeout_count=timeout_count,
            late_command_count=late_count,
            other_error_count=other_count,
            peak_component_rx0=float(peak[0]),
            peak_component_rx1=float(peak[1]),
            clip_fraction_rx0=float(fractions[0]),
            clip_fraction_rx1=float(fractions[1]),
            clip_level=self.settings.clip_level,
            max_clip_fraction=self.settings.max_clip_fraction,
            errors=tuple(dict.fromkeys(error for error in errors if error)),
        )
        capture_metadata = {
            **self.hardware_metadata,
            "capture_started_utc": started_utc,
            "scheduled_hardware_start_time_s": scheduled_start_s,
            "stream_start_delay_s": self.settings.stream_start_delay_s,
            "capture_interrupted": interrupted,
            "signal_frequency_hz": self.settings.signal_frequency_hz,
            "requested_center_frequency_hz": self.settings.center_frequency_hz,
            "actual_center_frequency_rx0_hz": self.actual_frequencies[0],
            "actual_center_frequency_rx1_hz": self.actual_frequencies[1],
            "expected_if_frequency_hz": (
                self.settings.signal_frequency_hz - self.actual_frequencies[0]
            ),
            "requested_sample_rate_hz": self.settings.sample_rate_hz,
            "sample_rate_hz": sample_rate,
            "requested_bandwidth_hz": self.settings.bandwidth_hz,
            "actual_bandwidth_rx0_hz": float(self.usrp.get_rx_bandwidth(0)),
            "actual_bandwidth_rx1_hz": float(self.usrp.get_rx_bandwidth(1)),
            "requested_rx0_gain_db": self.settings.rx0_gain_db,
            "requested_rx1_gain_db": self.settings.rx1_gain_db,
            "actual_rx0_gain_db": float(self.usrp.get_rx_gain(0)),
            "actual_rx1_gain_db": float(self.usrp.get_rx_gain(1)),
            "tone_frequency_hz": self.settings.tone_frequency_hz,
            "requested_duration_s": self.settings.duration_s,
            "actual_duration_s": received / sample_rate,
            "raw_dtype": "complex64",
            "raw_channel_order": ["Forward", "Reverse"],
        }
        capture_metadata.update(extra_metadata)
        writer.finish(capture_metadata, quality)
        return quality


def _check_disk_space(output_dir: Path, target_samples: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_bytes = 2 * target_samples * np.dtype(np.complex64).itemsize
    required = int(raw_bytes * 1.20) + 16 * 1024 * 1024
    available = shutil.disk_usage(output_dir).free
    if available < required:
        raise CaptureHardwareError(
            f"insufficient disk space: need about {required / 2**30:.2f} GiB, "
            f"available {available / 2**30:.2f} GiB"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture synchronized Forward/Reverse IQ from a B210. The external "
            "CW and selected audio tone must already be running."
        )
    )
    parser.add_argument("--tone-freq", type=int, choices=TONE_FREQUENCIES_HZ, required=True)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S, metavar="SECONDS")
    parser.add_argument(
        "--serial",
        default=DEFAULT_SERIAL,
        help="optional B210 serial; omit to use the first matching B200-series device",
    )
    parser.add_argument("--signal-freq", type=float, default=DEFAULT_SIGNAL_FREQUENCY_HZ)
    parser.add_argument("--lo-offset", type=float, default=DEFAULT_LO_OFFSET_HZ)
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--bandwidth", type=float, default=DEFAULT_BANDWIDTH_HZ)
    parser.add_argument("--rx0-gain", type=float, default=DEFAULT_GAIN_DB)
    parser.add_argument("--rx1-gain", type=float, default=DEFAULT_GAIN_DB)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "captures")
    parser.add_argument("--chunk-samples", type=int, default=DEFAULT_CHUNK_SAMPLES)
    parser.add_argument("--queue-depth", type=int, default=8)
    parser.add_argument("--tune-settle", type=float, default=0.2)
    parser.add_argument("--stream-start-delay", type=float, default=0.1)
    parser.add_argument("--clip-level", type=float, default=0.98)
    parser.add_argument("--max-clip-fraction", type=float, default=1e-4)
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="save valid raw IQ without running uncalibrated or calibrated analysis",
    )
    calibration_group = parser.add_mutually_exclusive_group()
    calibration_group.add_argument(
        "--calibration",
        type=Path,
        help="explicit calibration.h5 to apply after every valid capture",
    )
    calibration_group.add_argument(
        "--no-calibration",
        action="store_true",
        help="run the ordinary analysis but skip automatic OSL calibration",
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> CaptureSettings:
    return CaptureSettings(
        tone_frequency_hz=args.tone_freq,
        serial=args.serial,
        signal_frequency_hz=args.signal_freq,
        lo_offset_hz=args.lo_offset,
        sample_rate_hz=args.sample_rate,
        bandwidth_hz=args.bandwidth,
        rx0_gain_db=args.rx0_gain,
        rx1_gain_db=args.rx1_gain,
        duration_s=args.duration,
        output_dir=args.output_dir,
        chunk_samples=args.chunk_samples,
        queue_depth=args.queue_depth,
        tune_settle_s=args.tune_settle,
        stream_start_delay_s=args.stream_start_delay,
        clip_level=args.clip_level,
        max_clip_fraction=args.max_clip_fraction,
    )


def _frequency_filename_label(frequency_hz: float) -> str:
    value = f"{frequency_hz / 1e6:.6f}".rstrip("0").rstrip(".")
    return value.replace(".", "p")


def run_automatic_analysis(capture_path: Path) -> int:
    """Run the standard offline pipeline after a valid capture is closed."""
    from analyze import main as analyze_main

    print("\nCapture is valid; starting automatic analysis...")
    return int(analyze_main([str(capture_path)]))


def find_compatible_calibration(
    capture_path: Path,
    search_dir: Path | None = None,
) -> Path | None:
    """Return the newest compatible OSL file, or None when none is available."""
    from calibration_core import CalibrationError
    from osl_calibrate import assert_compatible, load_calibration

    root = search_dir or (Path(__file__).parent / "calibrations")
    if not root.is_dir():
        return None
    capture_metadata = inspect_capture(capture_path, require_valid=True)["metadata"]
    candidates = list(root.rglob("calibration.h5"))
    candidates.sort(
        key=lambda path: path.stat().st_mtime if path.exists() else float("-inf"),
        reverse=True,
    )
    for candidate in candidates:
        try:
            calibration = load_calibration(candidate)
            assert_compatible(
                calibration.reference_metadata,
                capture_metadata,
                f"capture for {candidate}",
            )
            return candidate
        except (CalibrationError, OSError, KeyError, TypeError, ValueError):
            continue
    return None


def run_automatic_calibration(capture_path: Path, calibration_path: Path) -> int:
    """Apply one compatible OSL model after the raw capture is safely closed."""
    from osl_calibrate import main as calibration_main

    print("\nApplying the OSL calibration automatically...")
    print(f"  calibration={artifact_relative_path(calibration_path)}")
    return int(calibration_main(["apply", str(calibration_path), str(capture_path)]))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = settings_from_args(args)
    try:
        settings.validate()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = settings.output_dir / (
            f"capture_{_frequency_filename_label(settings.signal_frequency_hz)}MHz_"
            f"{settings.tone_frequency_hz}Hz_{timestamp}.h5"
        )
        print("B210 dual-channel complex-reflection capture")
        print("  RX0 A:A/RX2: Forward")
        print("  RX1 A:B/RX2: Reverse")
        print(
            f"  External CW={settings.signal_frequency_hz/1e6:.6f} MHz, "
            f"B210 LO={settings.center_frequency_hz/1e6:.6f} MHz"
        )
        print(
            f"  Audio tone={settings.tone_frequency_hz} Hz, "
            f"sample rate={settings.sample_rate_hz/1e6:g} MS/s, "
            f"duration={settings.duration_s:g} s"
        )
        print("  Verify safe coupled-port power and stable external CW/audio sources.")
        receiver = B210DualReceiver(settings)
        quality = receiver.capture(output_path)
        print(f"\nRaw data: {artifact_relative_path(output_path)}")
        print(
            f"samples={quality.received_rx0}/{quality.target_samples}, "
            f"peak=({quality.peak_component_rx0:.4f}, {quality.peak_component_rx1:.4f}), "
            f"clip=({quality.clip_fraction_rx0:.3e}, {quality.clip_fraction_rx1:.3e})"
        )
        if quality.valid:
            print("Capture quality: VALID")
            if args.no_analyze:
                print("Skipped uncalibrated and OSL analysis because --no-analyze was set.")
                return 0
            analysis_status = run_automatic_analysis(output_path)
            if analysis_status != 0:
                return analysis_status
            if args.no_calibration:
                print("Skipped automatic OSL calibration because --no-calibration was set.")
                return 0
            if args.calibration is not None:
                calibration_path = args.calibration
            else:
                calibration_path = find_compatible_calibration(output_path)
                if calibration_path is None:
                    print(
                        "No OSL calibration matches the current frequency, gain, bandwidth, "
                        "and port configuration; retaining only the uncalibrated analysis."
                    )
                    return 0
                print(
                    "Selected newest compatible calibration: "
                    f"{artifact_relative_path(calibration_path)}"
                )
            return run_automatic_calibration(output_path, calibration_path)
        print("Capture quality: INVALID")
        print("The capture is invalid; automatic analysis was skipped.")
        for error in quality.errors:
            print(f"  - {error}")
        return 2
    except (CaptureConfigurationError, CaptureHardwareError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
