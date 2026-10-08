#!/usr/bin/env python3
"""B210 full-duplex hardware control for the real-time carrier sweep."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import uhd
except ImportError:  # Allow command-line help and signal processing without UHD.
    uhd = None


PROJECT_DIR = Path(__file__).resolve().parent


class ConfigurationError(ValueError):
    """Raised when a real-time sweep configuration is inconsistent."""


class HardwareError(RuntimeError):
    """Raised when the B210 cannot complete the requested operation."""


@dataclass
class PointCapture:
    """Acquisition status and useful IQ samples for one carrier frequency."""

    requested_freq_hz: float
    actual_tx_freq_hz: float
    actual_rx_freq_hz: float
    valid: bool
    saturated: bool
    peak_component: float
    clip_fraction: float
    overflow_count: int
    timeout_count: int
    other_error_count: int
    tx_underflow_count: int
    received_samples: int
    attempts: int
    error: str = ""
    useful_iq: np.ndarray | None = None


def artifact_output_root(value: str | Path) -> Path:
    """Resolve an output root relative to this code directory."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def artifact_relative_path(path: Path) -> str:
    """Return a code-directory-relative path for stored metadata and display."""
    try:
        return str(path.resolve().relative_to(PROJECT_DIR))
    except ValueError:
        return path.name


def generate_frequency_points(start_hz: float, stop_hz: float, step_hz: float) -> np.ndarray:
    """Generate ascending sweep points and include the exact stop frequency."""
    if not all(math.isfinite(value) for value in (start_hz, stop_hz, step_hz)):
        raise ConfigurationError("frequencies and step must be finite")
    if step_hz <= 0 or start_hz > stop_hz:
        raise ConfigurationError("invalid frequency interval")
    count = int(math.floor((stop_hz - start_hz) / step_hz + 1e-12))
    points = start_hz + np.arange(count + 1, dtype=np.float64) * step_hz
    tolerance = max(1e-6, abs(stop_hz) * 1e-12)
    points = points[points <= stop_hz + tolerance]
    if points.size == 0 or abs(points[-1] - stop_hz) > tolerance:
        points = np.append(points, float(stop_hz))
    else:
        points[-1] = float(stop_hz)
    return points


def _range_bounds(range_object: Any) -> tuple[float, float]:
    """Convert a UHD range object to numeric minimum and maximum values."""
    return float(range_object.start()), float(range_object.stop())


def find_usb_speed(serial: str) -> float | None:
    """Read the Linux sysfs USB link speed for the selected device."""
    sysfs_root = Path("/sys/bus/usb/devices")
    if not sysfs_root.exists():
        return None
    for serial_path in sysfs_root.glob("*/serial"):
        try:
            if serial_path.read_text(encoding="utf-8").strip() != serial:
                continue
            speed_path = serial_path.parent / "speed"
            return float(speed_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
    return None


class B210RealtimeRadio:
    """Manage B210 tuning, continuous-wave TX, and finite RX captures."""

    def __init__(self, config: Any):
        """Configure one RX chain and one TX chain from the sweep settings."""
        if uhd is None:
            raise HardwareError("Python UHD module is not installed")
        self.config = config
        device_args = "type=b200"
        if config.serial.strip():
            device_args += f",serial={config.serial.strip()}"
        self.usrp = uhd.usrp.MultiUSRP(device_args)
        if self.usrp.get_mboard_name(0) != "B210":
            raise HardwareError(f"Expected a B210, got {self.usrp.get_mboard_name(0)}")

        # Select subdevices before applying channel-specific settings.
        self.usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(config.rx_subdev))
        self.usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec(config.tx_subdev))
        if self.usrp.get_rx_num_channels() != 1 or self.usrp.get_tx_num_channels() != 1:
            raise HardwareError("Expected exactly one selected RX and TX channel")

        self._validate_requested_ranges()
        self.usrp.set_rx_antenna(config.rx_antenna, 0)
        self.usrp.set_tx_antenna(config.tx_antenna, 0)
        self.usrp.set_rx_rate(config.sample_rate_hz, 0)
        self.usrp.set_tx_rate(config.sample_rate_hz, 0)
        self.usrp.set_rx_gain(config.rx_gain_db, 0)
        self.usrp.set_tx_gain(config.tx_gain_db, 0)
        self.usrp.set_rx_bandwidth(config.sample_rate_hz, 0)
        self.usrp.set_tx_bandwidth(config.sample_rate_hz, 0)

        self.actual_rx_rate_hz = float(self.usrp.get_rx_rate(0))
        self.actual_tx_rate_hz = float(self.usrp.get_tx_rate(0))
        if not np.isclose(self.actual_rx_rate_hz, config.sample_rate_hz, rtol=1e-6, atol=1.0):
            raise HardwareError(
                f"RX rate mismatch: requested {config.sample_rate_hz}, got {self.actual_rx_rate_hz}"
            )
        if not np.isclose(self.actual_tx_rate_hz, config.sample_rate_hz, rtol=1e-6, atol=1.0):
            raise HardwareError(
                f"TX rate mismatch: requested {config.sample_rate_hz}, got {self.actual_tx_rate_hz}"
            )

        rx_args = uhd.usrp.StreamArgs("fc32", "sc16")
        rx_args.channels = [0]
        tx_args = uhd.usrp.StreamArgs("fc32", "sc16")
        tx_args.channels = [0]
        self.rx_streamer = self.usrp.get_rx_stream(rx_args)
        self.tx_streamer = self.usrp.get_tx_stream(tx_args)

        tx_packet = self.tx_streamer.get_max_num_samps()
        self._tx_buffer = np.full(
            (1, tx_packet),
            complex(config.tx_amplitude, 0.0),
            dtype=np.complex64,
        )
        self._tx_stop = threading.Event()
        self._tx_thread: threading.Thread | None = None
        self._tx_async_thread: threading.Thread | None = None
        self._tx_exception: BaseException | None = None
        self._stats_lock = threading.Lock()
        self._tx_stats = {"samples": 0, "timeouts": 0, "underflows": 0, "sequence_errors": 0}

        device_info = dict(self.usrp.get_usrp_rx_info(0))
        actual_serial = str(
            device_info.get("mboard_serial") or device_info.get("serial") or config.serial
        ).strip()
        usb_speed = find_usb_speed(actual_serial) if actual_serial else None
        if usb_speed is not None and usb_speed < 5000:
            raise HardwareError(
                f"B210 is connected at {usb_speed:g} Mb/s; USB 3.0 (5000 Mb/s) is required"
            )
        if usb_speed is None:
            print("[warning] USB speed is unavailable from sysfs; verify a USB 3.0 link.")

        self.hardware_info = {
            "mboard_name": self.usrp.get_mboard_name(0),
            "device_selection": "serial-specific" if config.serial.strip() else "automatic",
            "rx_subdev": config.rx_subdev,
            "tx_subdev": config.tx_subdev,
            "rx_antenna": self.usrp.get_rx_antenna(0),
            "tx_antenna": self.usrp.get_tx_antenna(0),
            "actual_rx_rate_hz": self.actual_rx_rate_hz,
            "actual_tx_rate_hz": self.actual_tx_rate_hz,
            "actual_rx_gain_db": float(self.usrp.get_rx_gain(0)),
            "actual_tx_gain_db": float(self.usrp.get_tx_gain(0)),
            "actual_rx_bandwidth_hz": float(self.usrp.get_rx_bandwidth(0)),
            "actual_tx_bandwidth_hz": float(self.usrp.get_tx_bandwidth(0)),
            "usb_speed_mbps": usb_speed,
        }

    def _validate_requested_ranges(self) -> None:
        """Check antenna names, RF ranges, and gain ranges before streaming."""
        config = self.config
        rx_antennas = set(self.usrp.get_rx_antennas(0))
        tx_antennas = set(self.usrp.get_tx_antennas(0))
        if config.rx_antenna not in rx_antennas:
            raise HardwareError(f"RX antenna {config.rx_antenna} unavailable: {sorted(rx_antennas)}")
        if config.tx_antenna not in tx_antennas:
            raise HardwareError(f"TX antenna {config.tx_antenna} unavailable: {sorted(tx_antennas)}")

        rx_min, rx_max = _range_bounds(self.usrp.get_rx_freq_range(0))
        tx_min, tx_max = _range_bounds(self.usrp.get_tx_freq_range(0))
        rx_requested = (
            config.start_freq_hz + config.rx_offset_hz,
            config.stop_freq_hz + config.rx_offset_hz,
        )
        if min(rx_requested) < rx_min or max(rx_requested) > rx_max:
            raise HardwareError(f"Requested RX centers {rx_requested} outside {rx_min}..{rx_max} Hz")
        if config.start_freq_hz < tx_min or config.stop_freq_hz > tx_max:
            raise HardwareError(
                f"Requested TX range {config.start_freq_hz}..{config.stop_freq_hz} outside "
                f"{tx_min}..{tx_max} Hz"
            )

        rx_gain_min, rx_gain_max = _range_bounds(self.usrp.get_rx_gain_range(0))
        tx_gain_min, tx_gain_max = _range_bounds(self.usrp.get_tx_gain_range(0))
        if not rx_gain_min <= config.rx_gain_db <= rx_gain_max:
            raise HardwareError(f"RX gain {config.rx_gain_db} outside {rx_gain_min}..{rx_gain_max} dB")
        if not tx_gain_min <= config.tx_gain_db <= tx_gain_max:
            raise HardwareError(f"TX gain {config.tx_gain_db} outside {tx_gain_min}..{tx_gain_max} dB")

    def start_tx(self) -> None:
        """Start continuous-wave TX and its asynchronous status monitor."""
        if self._tx_thread is not None and self._tx_thread.is_alive():
            raise HardwareError("TX stream is already running")
        self._tx_stop.clear()
        self._tx_exception = None
        with self._stats_lock:
            self._tx_stats = {"samples": 0, "timeouts": 0, "underflows": 0, "sequence_errors": 0}
        self._tx_thread = threading.Thread(target=self._tx_worker, name="b210-cw-tx", daemon=True)
        self._tx_async_thread = threading.Thread(
            target=self._tx_async_worker,
            name="b210-tx-async",
            daemon=True,
        )
        self._tx_async_thread.start()
        self._tx_thread.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            self.check_tx_health()
            with self._stats_lock:
                if self._tx_stats["samples"] > 0:
                    return
            time.sleep(0.01)
        self.stop_tx()
        raise HardwareError("TX streamer did not accept samples within two seconds")

    def stop_tx(self) -> None:
        """Stop the TX workers and wait for their termination."""
        self._tx_stop.set()
        if self._tx_thread is not None:
            self._tx_thread.join(timeout=3.0)
        if self._tx_async_thread is not None:
            self._tx_async_thread.join(timeout=1.0)
        self._tx_thread = None
        self._tx_async_thread = None

    def _tx_worker(self) -> None:
        """Continuously submit the constant complex TX buffer."""
        metadata = uhd.types.TXMetadata()
        metadata.start_of_burst = True
        metadata.end_of_burst = False
        metadata.has_time_spec = False
        try:
            while not self._tx_stop.is_set():
                sent = int(self.tx_streamer.send(self._tx_buffer, metadata, 0.5))
                metadata.start_of_burst = False
                with self._stats_lock:
                    self._tx_stats["samples"] += sent
                    if sent == 0:
                        self._tx_stats["timeouts"] += 1
        except BaseException as exc:  # Forward TX thread failures to the acquisition thread.
            self._tx_exception = exc
            self._tx_stop.set()
        finally:
            try:
                metadata.start_of_burst = False
                metadata.end_of_burst = True
                metadata.has_time_spec = False
                empty = np.empty((1, 0), dtype=np.complex64)
                self.tx_streamer.send(empty, metadata, 0.5)
            except Exception:
                pass

    def _tx_async_worker(self) -> None:
        """Collect TX underflow and sequence-error messages."""
        metadata = uhd.types.TXAsyncMetadata()
        while not self._tx_stop.is_set():
            try:
                if not self.tx_streamer.recv_async_msg(metadata, 0.1):
                    continue
                event = metadata.event_code
                with self._stats_lock:
                    if event in (
                        uhd.types.TXMetadataEventCode.underflow,
                        uhd.types.TXMetadataEventCode.underflow_in_packet,
                    ):
                        self._tx_stats["underflows"] += 1
                    elif event in (
                        uhd.types.TXMetadataEventCode.seq_error,
                        uhd.types.TXMetadataEventCode.seq_error_in_packet,
                    ):
                        self._tx_stats["sequence_errors"] += 1
            except RuntimeError:
                if not self._tx_stop.is_set():
                    with self._stats_lock:
                        self._tx_stats["sequence_errors"] += 1

    def tx_stats(self) -> dict[str, int]:
        """Return a thread-safe copy of the current TX counters."""
        with self._stats_lock:
            return dict(self._tx_stats)

    def check_tx_health(self) -> None:
        """Raise an error when the TX worker has failed or stopped."""
        if self._tx_exception is not None:
            raise HardwareError(f"TX worker failed: {self._tx_exception}") from self._tx_exception
        if (
            self._tx_thread is not None
            and not self._tx_thread.is_alive()
            and not self._tx_stop.is_set()
        ):
            raise HardwareError("TX worker stopped unexpectedly")

    def _sensor_locked(self, direction: str) -> bool:
        """Return the available RX or TX local-oscillator lock state."""
        try:
            if direction == "rx":
                names = self.usrp.get_rx_sensor_names(0)
                return "lo_locked" not in names or bool(
                    self.usrp.get_rx_sensor("lo_locked", 0).to_bool()
                )
            names = self.usrp.get_tx_sensor_names(0)
            return "lo_locked" not in names or bool(
                self.usrp.get_tx_sensor("lo_locked", 0).to_bool()
            )
        except RuntimeError:
            return False

    def _tune(self, carrier_hz: float) -> tuple[float, float, bool]:
        """Tune TX to the carrier and RX to the configured offset center."""
        self.check_tx_health()
        self.usrp.set_tx_freq(uhd.types.TuneRequest(float(carrier_hz)), 0)
        self.usrp.set_rx_freq(
            uhd.types.TuneRequest(float(carrier_hz + self.config.rx_offset_hz)),
            0,
        )
        time.sleep(self.config.tune_settle_s)
        deadline = time.monotonic() + 1.0
        locked = self._sensor_locked("rx") and self._sensor_locked("tx")
        while not locked and time.monotonic() < deadline:
            time.sleep(0.01)
            locked = self._sensor_locked("rx") and self._sensor_locked("tx")
        return float(self.usrp.get_tx_freq(0)), float(self.usrp.get_rx_freq(0)), locked

    def prepare_frequency(self, carrier_hz: float) -> None:
        """Tune both chains before starting the continuous TX stream."""
        _, _, locked = self._tune(carrier_hz)
        if not locked:
            raise HardwareError(f"RX or TX LO did not lock at {carrier_hz/1e6:.3f} MHz")

    def _stop_rx(self) -> None:
        """Stop an incomplete finite RX command."""
        try:
            command = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
            command.stream_now = True
            self.rx_streamer.issue_stream_cmd(command)
        except RuntimeError:
            pass

    def _capture_once(self, carrier_hz: float, num_avg: int) -> PointCapture:
        """Tune and acquire one finite IQ dwell without retrying."""
        actual_tx, actual_rx, lo_locked = self._tune(carrier_hz)
        discard_samples = self.config.fft_size
        useful_samples = self.config.fft_size * num_avg
        total_samples = discard_samples + useful_samples
        samples = np.empty(total_samples, dtype=np.complex64)
        received = 0
        overflows = 0
        timeouts = 0
        other_errors = 0
        errors: list[str] = []
        tx_before = self.tx_stats()

        command = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        command.num_samps = total_samples
        command.stream_now = True
        self.rx_streamer.issue_stream_cmd(command)
        max_packet = int(self.rx_streamer.get_max_num_samps())
        recv_buffer = np.empty((1, max_packet), dtype=np.complex64)
        metadata = uhd.types.RXMetadata()
        deadline = time.monotonic() + total_samples / self.actual_rx_rate_hz + 3.0

        while received < total_samples and time.monotonic() < deadline:
            self.check_tx_health()
            wanted = min(max_packet, total_samples - received)
            count = int(self.rx_streamer.recv(recv_buffer[:, :wanted], metadata, 1.0))
            error_code = metadata.error_code
            if error_code == uhd.types.RXMetadataErrorCode.none:
                if count:
                    samples[received : received + count] = recv_buffer[0, :count]
                    received += count
            elif error_code == uhd.types.RXMetadataErrorCode.overflow:
                overflows += 1
                errors.append(metadata.strerror())
                if count:
                    samples[received : received + count] = recv_buffer[0, :count]
                    received += count
            elif error_code == uhd.types.RXMetadataErrorCode.timeout:
                timeouts += 1
                errors.append(metadata.strerror())
                if timeouts >= 3:
                    break
            else:
                other_errors += 1
                errors.append(metadata.strerror())
                if other_errors >= 2:
                    break

        if received < total_samples:
            self._stop_rx()
            errors.append(f"short capture: {received}/{total_samples} samples")
        if not lo_locked:
            errors.append("RX or TX LO did not lock")

        tx_after = self.tx_stats()
        tx_underflows = max(0, tx_after["underflows"] - tx_before["underflows"])
        tx_sequence_errors = max(0, tx_after['sequence_errors'] - tx_before['sequence_errors'])
        tx_timeouts = max(0, tx_after['timeouts'] - tx_before['timeouts'])
        if tx_sequence_errors or tx_timeouts:
            other_errors += tx_sequence_errors + tx_timeouts
            errors.append(f'TX sequence errors={tx_sequence_errors}, timeouts={tx_timeouts}')
        if tx_underflows:
            errors.append(f"{tx_underflows} TX underflow event(s)")

        peak_component = math.nan
        clip_fraction = math.nan
        saturated = False
        useful_iq: np.ndarray | None = None
        if received >= total_samples:
            useful_iq = samples[discard_samples:total_samples].copy()
            components = np.maximum(np.abs(useful_iq.real), np.abs(useful_iq.imag))
            peak_component = float(np.max(components))
            clip_fraction = float(np.mean(components >= self.config.clip_level))
            saturated = clip_fraction > self.config.max_clip_fraction
            if saturated:
                errors.append(
                    f"RX clipping: peak component={peak_component:.4f}, "
                    f"fraction={clip_fraction:.3e}"
                )

        valid = bool(
            received >= total_samples
            and lo_locked
            and not saturated
            and overflows == 0
            and timeouts == 0
            and other_errors == 0
            and tx_underflows == 0
        )
        return PointCapture(
            requested_freq_hz=float(carrier_hz),
            actual_tx_freq_hz=actual_tx,
            actual_rx_freq_hz=actual_rx,
            valid=valid,
            saturated=saturated,
            peak_component=peak_component,
            clip_fraction=clip_fraction,
            overflow_count=overflows,
            timeout_count=timeouts,
            other_error_count=other_errors,
            tx_underflow_count=tx_underflows,
            received_samples=received,
            attempts=1,
            error="; ".join(dict.fromkeys(errors)),
            useful_iq=useful_iq,
        )

    def capture_frequency(self, carrier_hz: float, num_avg: int | None = None) -> PointCapture:
        """Capture one frequency and retry invalid hardware acquisitions."""
        averages = self.config.num_avg if num_avg is None else int(num_avg)
        attempts: list[PointCapture] = []
        for _ in range(self.config.capture_retries + 1):
            capture = self._capture_once(carrier_hz, averages)
            attempts.append(capture)
            if capture.valid:
                break
            print(f"    [retry] {carrier_hz/1e6:.3f} MHz: {capture.error}")
        final = attempts[-1]
        final.attempts = len(attempts)
        if len(attempts) > 1:
            final.overflow_count = sum(item.overflow_count for item in attempts)
            final.timeout_count = sum(item.timeout_count for item in attempts)
            final.other_error_count = sum(item.other_error_count for item in attempts)
            final.tx_underflow_count = sum(item.tx_underflow_count for item in attempts)
            combined_errors = [item.error for item in attempts if item.error]
            final.error = " | ".join(combined_errors)
        return final

    def saturation_preflight(self, frequencies_hz: Sequence[float]) -> None:
        """Check stream status and clipping at representative frequencies."""
        print("\n[preflight] Checking stream status and RX saturation at three frequencies...")
        for frequency_hz in frequencies_hz:
            capture = self.capture_frequency(float(frequency_hz), num_avg=2)
            print(
                f"  {frequency_hz/1e6:8.3f} MHz | peak={capture.peak_component:.4f} | "
                f"clip={capture.clip_fraction:.3e} | {'OK' if capture.valid else 'FAIL'}"
            )
            if not capture.valid:
                raise HardwareError(
                    f"Preflight failed at {frequency_hz/1e6:.3f} MHz: {capture.error}. "
                    "Check antenna isolation or adjust the configured gain."
                )


def _json_compatible(value: Any) -> Any:
    """Convert NumPy scalars and paths into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    """Write indented UTF-8 JSON after converting supported NumPy values."""
    path.write_text(
        json.dumps(_json_compatible(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
