# Training with user-supplied data

Install `environment/requirements-training.txt` in the software environment.
No training data, initialization weights or pretrained speech encoders are
distributed in this repository.

## Prepare clean and noisy pairs

Use speech data that you are authorized to use. Supported manifest scanners
include VCTK, LibriSpeech and Common Voice; select the layout matching your data.

```bash
python training/build_clean_manifest.py --dataset-type vctk \
  --dataset-root /path/to/VCTK-Corpus --output data/clean.csv
python training/prepare_clean_clips.py --manifest data/clean.csv \
  --output-dir data/clips --output-manifest data/clips.csv
python training/simulate_noisy_pairs.py --clean-manifest data/clips.csv \
  --output-dir data/pairs --output-manifest data/pairs.csv --preview-count 0
```

Training pair CSVs contain `pair_id`, `clean_clip_id`, `clean_audio_path` and
`noisy_audio_path`. Audio is mono 48 kHz; paths resolve relative to the pair CSV.
An explicit `split` column can select `train`/`val`. Keep clean clips and speakers
separate across these splits and keep evaluation recordings out of training.
The splitter rejects empty partitions and overlap in explicit splits.

## Train from random initialization

```bash
python training/train_mel_unet.py --manifest data/pairs.csv \
  --output-dir runs/denoiser --model-arch resunet --base-channels 64 \
  --device cpu --num-workers 0
python training/train_stft_unet.py --manifest data/pairs.csv \
  --output-dir runs/refiner --base-channels 48 --depth 4 \
  --output-mode residual --device cpu --num-workers 0
```

For a second-stage refiner, prepare pairs whose noisy inputs are denoiser outputs
and whose targets are clean speech. `prepare_refiner_pairs.py` supports this for
a VCTK layout when passed your own `--denoiser`, `--vctk-root`, `--output-dir`
and `--device`. Train/validation data volume, hyperparameters and duration are
experiment choices; these examples do not claim the original paper protocol.

`--init-checkpoint` and `--resume-checkpoint` are optional and must reference
your own compatible files. They are mutually exclusive in the main trainers.
Without either, those trainers start with random weights. Generated weights,
logs and data are ignored by Git.

## Optional speech-encoder losses

The STFT trainer disables content and CTC losses by default. Enabling them
requires a local encoder supplied through `--content-model` or `--ctc-model`.
Remote model downloads are disabled. The bounded-waveform trainer
`train_bounded_refiner.py` requires `--ctc-model /path/to/local/wav2vec2` and a
manifest of denoiser-output/clean pairs. Its optional initialization checkpoint
is no longer assumed to exist. A compatible processor/tokenizer must accompany
the local CTC model.

Inference checkpoints contain `model` and `config`. The restoration command
requires the architecture sizes in `configs/paper_models.json`; small training
smoke-test networks are not interchangeable with those sizes.
