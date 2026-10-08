# Hardware acquisition and scope

Acquisition targets the USRP B210. Compatibility with other USRPs is not
established. Offline tests never open a radio. Physical streaming and RF power
were not exercised during the source-release check.

## Environment and equipment

- Native UHD driver, firmware/images and Python bindings from the same installation.
- Python 3.10/3.11 for the original pinned acquisition dependencies:
  `python -m pip install -r environment/requirements-hardware.txt`.
- USB 3 connection for B210, adequate host throughput/storage, appropriate antennas.
- Paper laboratory chain: Rigol DSG3136B-IQ, external power amplifier and HT8 TX;
  B210 RX with HT8 or the specified VULB receiving antenna. The software does
  not automate the Rigol or configure the amplifier.

Use a driver-compatible interpreter for `import uhd`. Do not install an unrelated
PyPI package called UHD as a replacement. Keep acquisition and model environments
separate if their Python/native-library requirements differ.

**RF safety and authorization:** run only in a suitably authorized controlled
setup. Select permitted frequencies, calibrated power and safe receiver input
levels before enabling equipment. TX gain in dB is not conducted power in dBm;
the B210 scripts do not reproduce or certify the paper's generator/amplifier EIRP.
No regulatory exclusion list is implemented by the sweep. Do not run its broad
default range over the air merely because the CLI accepts it.

## Does TX/RX require two PCs?

| Program | TX/RX behavior | Host requirement |
|---|---|---|
| `carrier_search.py` | One process owns one B210. Background TX streams CW on RF B; RX records on RF A while TX continues. | One PC suffices. |
| `transmit_cw.py` | TX-only, fixed 601 MHz. | No capture. Do not open the same B210 independently with another script. |
| `capture_magnitude.py` | RX-only on channel 0 / RX2. | External carrier must already be on; one PC plus signal generator suffices. |

The sweep is **concurrent**, not sample-synchronous TX/RX: TX has no hardware
time specification and RX uses an immediate finite capture. Audio playback is not synchronized
by these scripts. Two separate USRPs can be controlled from one capable PC, but
cross-device synchronization is not implemented here. Two PCs are not inherently
required and would not by themselves establish synchronization.

## 1. Carrier search

After confirming an authorized frequency interval, consult:

```bash
python step1_rf_acquisition/carrier_search.py --help
```

Acquisition defaults: 200–800 MHz inclusive, 10 MHz step,
5 MS/s, RX gain 10 dB, TX gain 50 dB, RX center offset 250 kHz. RX/TX subdevices
are A:A/A:B with their TX/RX antenna ports. The default interval contains **61**
frequencies, not 60. Each nominal 0.5-second point contains 0.393216 seconds of
scored samples, one discarded 65536-sample block, and tune settling; retries,
preflight and computation add time.

The score is a short-window speech-dynamic percentile score after magnitude
conversion. It is **not** the combined sideband-energy/symmetry score described
in Sec. 5.1.1. Retained outputs identify the actual algorithm. The reported paper
success rate/time must not be assumed from this implementation. Fine carrier
search and receiver-position adjustment are not automatically performed.

Outputs include configuration, ranking, actual tuned frequencies, stream error
counters, clipping checks and score curves. IQ dwells are released after scoring;
this command is not an end-to-end speech-recording command.

## 2. Single-carrier capture and EM conversion

Set the external generator and power chain manually, connect RX to B210 RX2,
and run (paths are examples to replace with your actual capture):

```bash
python step1_rf_acquisition/capture_magnitude.py --center-freq 601e6 \
  --sample-rate 5e6 --gain 10 --duration 8 --no-plot --output-dir outputs/captures
python step2_em_raw/magnitude_to_audio.py outputs/captures/ACTUAL_CAPTURE.h5 \
  --output-dir outputs/em_raw --float-wav --no-mel
```

Capture now rejects overflow, timeout, short capture, rate coercion, non-finite
IQ and clipping rather than exporting a plausible-looking incomplete file.
These checks have only been verified in code/offline tests, not against a radio.
The converter reads `magnitude` and `sample_rate_hz`; use `--input-rate` only
when the acquisition metadata is missing and the true rate is known.

Conversion subtracts the global mean, applies an eighth-order 100 kHz low-pass,
then anti-aliased resampling to 48 kHz. The paper describes a smoothed baseline;
the supplied converter implements global-mean removal, not time-varying drift
tracking. Do not relabel one as the other. Full 5 MS/s recordings are memory
intensive; restoration can also start from your existing EM WAVs.

## Porting to another USRP

Review device selection (`type=b200` and B210 board checks), subdevice/channel
mapping, antenna names, supported clock/rate/bandwidth/frequency ranges, gain
ranges, USB/Ethernet transport checks, stream formats and RX/TX throughput.
Check actual coerced rates/frequencies, drops, clipping, and RF levels before
using the output. Renaming files alone does not perform this port.
