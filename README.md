# GhostWhisper

Source code for **GhostWhisper: Remote Audio Eavesdropping on Earphones via RF
Backscatter** (IEEE Symposium on Security and Privacy 2027).

This repository contains RF acquisition, waveform conversion, speech restoration,
evaluation, simulation and training code. Network architecture definitions are
included as Python source. Model weights, pretrained initialization files, ASR
models, datasets, recordings and experiment outputs are **not included**.
Supply your own data and compatible checkpoints to run restoration. Numerical
tests run without downloaded models, recordings or RF hardware.

## Code layout

| Directory | Purpose |
| --- | --- |
| `step1_rf_acquisition/` | B210 carrier search, CW transmission and magnitude capture |
| `step2_em_raw/` | Magnitude HDF5 to 48 kHz audio conversion |
| `step3_audio_restoration/` | Stable-tone filtering, log-Mel denoising and STFT refinement |
| `step4_evaluation/` | Alignment, SNR, LSD, STOI, PESQ and optional ASR WER |
| `reflection_measurement/` | Dual-channel capture, OSL calibration, reflection and impedance analysis |
| `src/ghostwhisper/` | Signal processing, network definitions, simulation and data utilities |
| `training/` | Speech preparation, noisy-pair generation and training scripts |
| `configs/` | Network architecture specification |
| `environment/` | Software, training and native-UHD dependency lists |
| `tests/` | Synthetic numerical and network-forward tests |

Run commands from the repository root. Paths shown below are examples of files
you provide; they are not bundled samples.

## Installation

Linux is the primary target. Use Python 3.12 for software processing:

```bash
conda create -n ghostwhisper python=3.12 -y
conda activate ghostwhisper
python -m pip install -r environment/requirements.txt
python -m unittest discover -s tests -v
```

PESQ may need a C compiler on Linux (for example, the `build-essential` package
on Ubuntu). Install a PyTorch build compatible with your GPU and driver when
using CUDA. CPU execution is supported; a GPU is useful for training. MPS applies
to PyTorch restoration/training, not faster-whisper ASR.

Hardware acquisition needs a separate environment compatible with the native
UHD Python bindings, typically Python 3.10/3.11 for the pinned hardware list.
See [hardware setup](docs/HARDWARE.md). Installing a PyPI package named `uhd`
does not replace the native driver and bindings.

## 1. Convert a magnitude recording

For your own HDF5 capture with a `magnitude` dataset and `sample_rate_hz` metadata:

```bash
python step2_em_raw/magnitude_to_audio.py data/capture.h5 \
  --output-dir outputs/em_raw --float-wav --no-mel
```

Use the printed WAV filename in a manifest. Conversion removes the global mean,
applies the configured low-pass filter and resamples to 48 kHz. Acquisition
commands and their hardware requirements are in [docs/HARDWARE.md](docs/HARDWARE.md).

## 2. Restore speech with your own checkpoints

Create `data/manifest.csv`, for example:

```csv
sample_id,raw_audio,reference_audio,transcript
sample_01,em_raw.wav,reference.wav,THE REFERENCE TRANSCRIPT
```

Audio paths resolve relative to the CSV. Restoration reads only `raw_audio`;
`reference_audio` is needed for reference-based evaluation and `transcript` for
WER. Neither enters the restoration network.

```bash
python step3_audio_restoration/restore.py \
  --manifest data/manifest.csv \
  --denoiser checkpoints/denoiser.pt --refiner checkpoints/refiner.pt \
  --output-dir outputs/run_01 --device cpu
```

Use `--device cuda` for CUDA. Checkpoints must contain a `model` state dictionary
and compatible `config` values: 48 kHz, ResUNet base width 64 (26,298,817
parameters) and STFT U-Net base width 48 / depth 4 (17,411,522 parameters).
The restoration command checks these sizes. See [training](docs/TRAINING.md).

Outputs are `raw.wav`, `preprocessed.wav`, `denoised.wav`, `restored.wav` per
sample, plus `run.json` with hashes and runtime. Choose a fresh output directory;
overwriting requires `--allow-existing`.

## 3. Evaluate

```bash
python step4_evaluation/evaluate.py \
  --manifest data/manifest.csv --run-dir outputs/run_01 --pesq-mode both
```

PESQ defaults to narrowband (`nb`); select `wb` or `both` explicitly as needed.
The paper specifies wideband PESQ; these scores are not interchangeable.
Automatic alignment estimates one offset from RAW/reference audio and applies
it to every stage. Inspect it, or use `--alignment manual --offsets data/offsets.csv`
with columns `sample_id,lag_seconds` (positive means candidate delayed).

Optional WER requires a local faster-whisper model supplied separately:

```bash
python step4_evaluation/evaluate.py \
  --manifest data/manifest.csv --run-dir outputs/run_01 \
  --asr --asr-model /path/to/faster-whisper-large-v3 \
  --asr-stages raw denoised restored --local-files-only
```

ASR loading is local-only; this repository does not download or distribute a
model. The default ASR execution is CPU/int8 with beam size 5. For CUDA use
`--asr-device cuda --asr-compute-type float16` with compatible native libraries.

## Reflection measurements

See [reflection_measurement/README.md](reflection_measurement/README.md) for
coherent forward/reverse IQ capture, OSL calibration and equivalent-impedance
analysis. It requires dual-channel complex IQ; magnitude-only speech captures
cannot substitute for it. Offline algorithms accept real or synthetic captures
with the same schema. No OSL/DUT dataset is distributed here.

## Tests and scope

```bash
python -m unittest discover -s tests -v
```

Tests generate numerical inputs in memory or temporary directories and use
randomly initialized networks. They do not download weights, contact a radio or
transmit RF. The PESQ test is explicitly skipped if the native PESQ package is
unavailable; it is not replaced with a fabricated score.

This source release supports implementation inspection and experiments with
user-supplied inputs. It does not itself reproduce the paper's full training
history, device averages, range or cross-environment results. Carrier search
uses the documented speech-dynamic score; see the hardware guide for differences
from the paper's search description. Hardware acquisition was not run during
this source-release check.

## License

A project-wide open-source license has not yet been selected. Public visibility
alone does not grant an open-source license. Dependencies and any externally
obtained data or models remain subject to their respective licenses.
