#!/usr/bin/env python3
"""Transmit an unmodulated 601 MHz CW carrier from B210 RF B (A:B)."""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from radio_hardware import find_usb_speed

try:
    import uhd
except ImportError:
    uhd = None


FIXED_FREQUENCY_HZ = 601e6
DEFAULT_SAMPLE_RATE_HZ = 5e6
DEFAULT_TX_GAIN_DB = 50.0
DEFAULT_TX_AMPLITUDE = 1.0
DEFAULT_TUNE_SETTLE_S = 0.0936768
DEFAULT_TX_SUBDEV = "A:B"
DEFAULT_TX_ANTENNA = "TX/RX"


class TxConfigurationError(ValueError):
    """Raised when TX-only parameters are invalid."""


class TxHardwareError(RuntimeError):
    """Raised when the B210 cannot provide the requested TX chain."""


@dataclass(frozen=True)
class TxConfig:
    """TX-only parameters; defaults mirror the sweep program."""

    serial: str = ""
    frequency_hz: float = FIXED_FREQUENCY_HZ
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ
    tx_gain_db: float = DEFAULT_TX_GAIN_DB
    tx_amplitude: float = DEFAULT_TX_AMPLITUDE
    tune_settle_s: float = DEFAULT_TUNE_SETTLE_S
    duration_s: float = 0.0
    tx_subdev: str = DEFAULT_TX_SUBDEV
    tx_antenna: str = DEFAULT_TX_ANTENNA

    def validate(self) -> None:
        if self.frequency_hz != FIXED_FREQUENCY_HZ:
            raise TxConfigurationError("this program is fixed at exactly 601 MHz")
        if self.sample_rate_hz <= 0:
            raise TxConfigurationError("sample rate must be positive")
        if not 0 < self.tx_amplitude <= 1.0:
            raise TxConfigurationError("TX amplitude must be in (0, 1]")
        if self.tune_settle_s < 0 or self.duration_s < 0:
            raise TxConfigurationError("settling time and duration must not be negative")


def _range_bounds(range_object: object) -> tuple[float, float]:
    return float(range_object.start()), float(range_object.stop())


class B210CwTransmitter:
    """Configure RF B and stream a constant complex waveform until stopped."""

    def __init__(self, config: TxConfig):
        if uhd is None:
            raise TxHardwareError("Python UHD module is not installed")
        self.config = config
        device_args = "type=b200"
        if config.serial.strip():
            device_args += f",serial={config.serial.strip()}"
        self.usrp = uhd.usrp.MultiUSRP(device_args)
        board_name = self.usrp.get_mboard_name(0)
        if board_name != "B210":
            raise TxHardwareError(f"Expected a B210, got {board_name}")

        self.usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec(config.tx_subdev))
        if self.usrp.get_tx_num_channels() != 1:
            raise TxHardwareError("Expected exactly one selected TX channel")

        antennas = set(self.usrp.get_tx_antennas(0))
        if config.tx_antenna not in antennas:
            raise TxHardwareError(
                f"TX antenna {config.tx_antenna} unavailable: {sorted(antennas)}"
            )
        freq_min, freq_max = _range_bounds(self.usrp.get_tx_freq_range(0))
        if not freq_min <= config.frequency_hz <= freq_max:
            raise TxHardwareError(
                f"601 MHz outside the TX range {freq_min/1e6:.1f}–{freq_max/1e6:.1f} MHz"
            )
        gain_min, gain_max = _range_bounds(self.usrp.get_tx_gain_range(0))
        if not gain_min <= config.tx_gain_db <= gain_max:
            raise TxHardwareError(
                f"TX gain {config.tx_gain_db} outside {gain_min}–{gain_max} dB"
            )

        device_info = dict(self.usrp.get_usrp_tx_info(0))
        actual_serial = str(
            device_info.get("mboard_serial")
            or device_info.get("serial")
            or config.serial
        ).strip()
        usb_speed = find_usb_speed(actual_serial) if actual_serial else None
        if usb_speed is not None and usb_speed < 5000:
            raise TxHardwareError(
                f"B210 is connected at {usb_speed:g} Mb/s; USB 3.0 is required"
            )
        if usb_speed is None:
            print("[warning] USB speed is unavailable from sysfs; verify a USB 3.0 link.")

        self.usrp.set_tx_antenna(config.tx_antenna, 0)
        self.usrp.set_tx_rate(config.sample_rate_hz, 0)
        self.usrp.set_tx_gain(config.tx_gain_db, 0)
        self.usrp.set_tx_bandwidth(config.sample_rate_hz, 0)
        self.usrp.set_tx_freq(uhd.types.TuneRequest(config.frequency_hz), 0)

        self.actual_rate_hz = float(self.usrp.get_tx_rate(0))
        self.actual_gain_db = float(self.usrp.get_tx_gain(0))
        self.actual_bandwidth_hz = float(self.usrp.get_tx_bandwidth(0))
        self.actual_frequency_hz = float(self.usrp.get_tx_freq(0))
        if not np.isclose(self.actual_rate_hz, config.sample_rate_hz, rtol=1e-6, atol=1.0):
            raise TxHardwareError(
                f"TX rate mismatch: requested {config.sample_rate_hz}, got {self.actual_rate_hz}"
            )
        if not np.isclose(self.actual_frequency_hz, FIXED_FREQUENCY_HZ, rtol=0, atol=1.0):
            raise TxHardwareError(
                f"TX frequency mismatch: requested {FIXED_FREQUENCY_HZ}, "
                f"got {self.actual_frequency_hz}"
            )

        time.sleep(config.tune_settle_s)
        if not self._wait_for_lo_lock(timeout_s=1.0):
            raise TxHardwareError("TX LO did not lock at 601 MHz")

        stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
        stream_args.channels = [0]
        self.streamer = self.usrp.get_tx_stream(stream_args)
        packet_size = int(self.streamer.get_max_num_samps())
        self.waveform = np.full(
            (1, packet_size),
            complex(config.tx_amplitude, 0.0),
            dtype=np.complex64,
        )
        self._stop_async = threading.Event()
        self._async_thread: threading.Thread | None = None
        self._stats_lock = threading.Lock()
        self._stats = {
            "samples": 0,
            "send_timeouts": 0,
            "underflows": 0,
            "sequence_errors": 0,
        }

    def _wait_for_lo_lock(self, timeout_s: float) -> bool:
        names = self.usrp.get_tx_sensor_names(0)
        if "lo_locked" not in names:
            return True
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.usrp.get_tx_sensor("lo_locked", 0).to_bool():
                return True
            time.sleep(0.01)
        return bool(self.usrp.get_tx_sensor("lo_locked", 0).to_bool())

    def _async_worker(self) -> None:
        metadata = uhd.types.TXAsyncMetadata()
        while not self._stop_async.is_set():
            try:
                if not self.streamer.recv_async_msg(metadata, 0.1):
                    continue
                event = metadata.event_code
                with self._stats_lock:
                    if event in (
                        uhd.types.TXMetadataEventCode.underflow,
                        uhd.types.TXMetadataEventCode.underflow_in_packet,
                    ):
                        self._stats["underflows"] += 1
                    elif event in (
                        uhd.types.TXMetadataEventCode.seq_error,
                        uhd.types.TXMetadataEventCode.seq_error_in_packet,
                    ):
                        self._stats["sequence_errors"] += 1
            except RuntimeError:
                if not self._stop_async.is_set():
                    with self._stats_lock:
                        self._stats["sequence_errors"] += 1

    def transmit(self) -> dict[str, int]:
        """Transmit until Ctrl+C or until the optional duration expires."""
        metadata = uhd.types.TXMetadata()
        metadata.start_of_burst = True
        metadata.end_of_burst = False
        metadata.has_time_spec = False
        deadline = (
            time.monotonic() + self.config.duration_s
            if self.config.duration_s > 0
            else math.inf
        )
        self._stop_async.clear()
        self._async_thread = threading.Thread(
            target=self._async_worker,
            name="b210-601mhz-tx-async",
            daemon=True,
        )
        self._async_thread.start()

        try:
            while time.monotonic() < deadline:
                sent = int(self.streamer.send(self.waveform, metadata, 0.5))
                metadata.start_of_burst = False
                with self._stats_lock:
                    self._stats["samples"] += sent
                    if sent == 0:
                        self._stats["send_timeouts"] += 1
        except KeyboardInterrupt:
            print("\nCtrl+C received; stopping transmission...")
        finally:
            metadata.start_of_burst = False
            metadata.end_of_burst = True
            metadata.has_time_spec = False
            try:
                self.streamer.send(np.empty((1, 0), dtype=np.complex64), metadata, 0.5)
            finally:
                self._stop_async.set()
                if self._async_thread is not None:
                    self._async_thread.join(timeout=1.0)

        with self._stats_lock:
            return dict(self._stats)


def build_parser(defaults: TxConfig | None = None) -> argparse.ArgumentParser:
    defaults = TxConfig() if defaults is None else defaults
    parser = argparse.ArgumentParser(
        description="Transmit a fixed unmodulated 601 MHz CW from B210 RF B / A:B / TX-RX"
    )
    parser.add_argument(
        "--serial",
        default=defaults.serial,
        help="optional B210 serial; omit to use the first matching B200-series device",
    )
    parser.add_argument("--sample-rate", type=float, default=defaults.sample_rate_hz, metavar="SPS")
    parser.add_argument("--tx-gain", type=float, default=defaults.tx_gain_db, metavar="DB")
    parser.add_argument("--tx-amplitude", type=float, default=defaults.tx_amplitude)
    parser.add_argument("--tune-settle", type=float, default=defaults.tune_settle_s, metavar="SECONDS")
    parser.add_argument(
        "--duration",
        type=float,
        default=defaults.duration_s,
        metavar="SECONDS",
        help="0 means transmit until Ctrl+C (default: 0)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> TxConfig:
    config = TxConfig(
        serial=args.serial,
        sample_rate_hz=args.sample_rate,
        tx_gain_db=args.tx_gain,
        tx_amplitude=args.tx_amplitude,
        tune_settle_s=args.tune_settle,
        duration_s=args.duration,
    )
    config.validate()
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
    except TxConfigurationError as exc:
        parser.error(str(exc))

    print("=" * 68)
    print("B210 fixed 601 MHz CW transmitter")
    print("=" * 68)
    print("Device selection: " + ("specified serial" if config.serial else "first matching B210"))
    print(f"Port: RF B / {config.tx_subdev} / {config.tx_antenna}")
    print(f"Frequency: {config.frequency_hz/1e6:.3f} MHz (fixed)")
    print(f"Sample rate: {config.sample_rate_hz/1e6:.3f} MS/s")
    print(f"TX gain: {config.tx_gain_db:g} dB")
    print(f"Digital amplitude: {config.tx_amplitude:g}")
    print("Duration: " + (f"{config.duration_s:g} s" if config.duration_s else "until Ctrl+C"))
    print("WARNING: This program transmits RF. Verify antenna and laboratory safety.")

    try:
        input("Press Enter to confirm and transmit; press Ctrl+C to stop...")
        transmitter = B210CwTransmitter(config)
        print(
            f"Transmitting {transmitter.actual_frequency_hz/1e6:.3f} MHz CW from RF B | "
            f"rate={transmitter.actual_rate_hz/1e6:.3f} MS/s | "
            f"gain={transmitter.actual_gain_db:g} dB"
        )
        stats = transmitter.transmit()
        print(
            f"Transmission stopped. samples={stats['samples']:,}, "
            f"underflows={stats['underflows']}, "
            f"sequence_errors={stats['sequence_errors']}, "
            f"send_timeouts={stats['send_timeouts']}"
        )
        return 0
    except EOFError:
        print("No safety confirmation received; transmission was not started.", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Transmission failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
