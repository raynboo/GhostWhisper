# RF reflection, calibration and impedance

This independent measurement workflow is separate from `step1`–`step4` of the
audio-recovery system. Speech restoration does not depend on it.

## Files and requirements

- `capture.py`, `capture_io.py`: synchronized forward/reverse complex I/Q capture.
- `analyze.py`, `reflection_core.py`: carrier extraction and uncalibrated complex
  reverse/forward ratio analysis.
- `osl_calibrate.py`, `calibration_core.py`: open/short/load vector calibration
  and calibrated reflection-coefficient export.
- `impedance.py`: equivalent reference-plane impedance and tone-modulation
  semiamplitude export, added during artifact preparation.

Offline analysis uses NumPy, SciPy, h5py and Matplotlib. Radio acquisition also
requires native UHD Python bindings; see `../docs/HARDWARE.md` and
`../environment/requirements-hardware.txt`.

## Measurement connection

Use an external CW signal generator, a bidirectional coupler, and a shared
transmit/receive antenna. Connect the forward coupled port to RX0 and the
reverse coupled port to RX1. The capture implementation targets a USRP B210:
both channels share one timed RX streamer on one PC. The external transmitter
is configured separately, not started or synchronized by this software.
Observe receiver input-power limits and use suitable attenuation/protection.

For another USRP, adapt the device arguments, `A:A A:B` subdevice selection,
RX2 antenna-port names, rate/bandwidth/gain capabilities, and clock/time setup
in `capture.py`. Preserve coherent, simultaneous two-channel reception.

Place the OSL reference plane at the antenna-side SMA connection of the coupler.
Capture standards at the same carrier, cabling, gains and receiver configuration
as the device under test. The default ideal standards are +1, -1 and 0; use the
standard-specific complex coefficients when available. Changing the RF setup
requires a new calibration. The reference impedance for conversion defaults to
50 ohms.

## Commands

Run from the repository root. Hardware commands require a connected,
authorized RF setup; offline commands require compatible dual-channel complex
IQ captures, real or synthetic. Magnitude-only speech H5 files are not compatible.

```bash
# Guided acquisition of open, short and load standards.
python reflection_measurement/osl_calibrate.py capture --output-dir outputs/osl

# Alternatively build from existing dual-channel captures.
python reflection_measurement/osl_calibrate.py build \
  --open measurements/open.h5 --short measurements/short.h5 \
  --load measurements/load.h5 --output-dir outputs/calibration

# Set audio playback independently; --tone-freq labels the condition.
python reflection_measurement/capture.py --tone-freq 2000 --duration 10 \
  --output-dir outputs/reflection_captures

python reflection_measurement/analyze.py measurements/dut.h5 \
  --output-dir outputs/reflection_analysis
python reflection_measurement/osl_calibrate.py apply \
  outputs/calibration/calibration.h5 measurements/dut.h5 \
  --output-dir outputs/calibrated_dut
python reflection_measurement/impedance.py outputs/calibrated_dut/calibrated.h5 \
  --tone-hz 1000 2000 3000 --output-dir outputs/impedance
```

Replace example paths with actual capture filenames. For a silent control,
disable audio playback externally and retain a positive `--tone-freq` analysis
frequency; the calibration analyzer expects this metadata. Record the silent
condition explicitly in the experiment log and compare the same fitted
frequencies under identical settings.

Impedance is computed as `Z = Z0 * (1 + Gamma) / (1 - Gamma)`. Invalid calibration
samples and near-singular denominators are masked. A complex cosine/sine fit
at each requested audio frequency gives the major/minor ellipse semiamplitudes
in ohms. These describe modulation at the calibrated reference plane, not an
isolated earphone component. The new export has synthetic numerical tests;
it has not been validated against the paper's original measurement results.

No OSL/DUT measurement dataset is included. Numerical tests synthesize arrays;
they do not constitute hardware measurements. Hardware acquisition has not been
exercised during the source-release check.
