#!/usr/bin/env python3
"""Evaluate every stage on the same RAW-derived time interval; ASR is optional."""
from pathlib import Path
import argparse
import csv
import json
import sys
import math
import numpy as np
import librosa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ghostwhisper.io_utils import read_manifest, read_audio
from ghostwhisper.metrics import (align_for_metrics, apply_alignment_shift, mel_db,
    log_mag_db, common_frames, snr_db, lsd_db, word_error_rate)


def metrics(ref, cand, sr, pesq_mode='nb'):
    if pesq_mode not in {'nb', 'wb', 'both'}:
        raise ValueError('PESQ mode must be nb, wb or both')
    rm, cm = common_frames(mel_db(ref, sr), mel_db(cand, sr))
    rl, cl = common_frames(log_mag_db(ref, sr), log_mag_db(cand, sr))
    result = {"mel_snr_db": snr_db(rm + 80, cm + 80), "waveform_snr_db": snr_db(ref, cand),
              "lsd_db": lsd_db(rl, cl)}
    # Missing dependencies and PESQ failures must be visible, never silently dropped.
    from pystoi import stoi
    from pesq import pesq
    r10 = librosa.resample(ref, orig_sr=sr, target_sr=10000)
    c10 = librosa.resample(cand, orig_sr=sr, target_sr=10000)
    result['stoi'] = float(stoi(r10, c10, 10000, extended=False))
    for mode in (['nb', 'wb'] if pesq_mode == 'both' else [pesq_mode]):
        target_sr = 8000 if mode == 'nb' else 16000
        rp = librosa.resample(ref, orig_sr=sr, target_sr=target_sr)
        cp = librosa.resample(cand, orig_sr=sr, target_sr=target_sr)
        try:
            result[f'pesq_{mode}'] = float(pesq(target_sr, rp, cp, mode))
            result[f'pesq_{mode}_status'] = 'ok'
        except Exception as e:
            result[f'pesq_{mode}'] = None
            result[f'pesq_{mode}_status'] = f'{type(e).__name__}: {e}'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True, help='CSV containing reference audio and input paths')
    p.add_argument('--run-dir', type=Path, default=ROOT / 'outputs/demo')
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--alignment', choices=['auto', 'manual', 'none'], default='auto')
    p.add_argument('--offsets', type=Path, help='CSV: sample_id,lag_seconds; positive = candidate delayed')
    p.add_argument('--max-shift', type=float, default=3.0)
    p.add_argument('--pesq-mode', choices=['nb', 'wb', 'both'], default='nb',
                   help='Select PESQ mode; evaluation settings are recorded with the report.')
    p.add_argument('--asr', action='store_true')
    p.add_argument('--asr-stages', nargs='+', choices=['raw', 'preprocessed', 'denoised', 'restored'], default=['restored'])
    p.add_argument('--asr-model', type=Path,
                   help='User-supplied local faster-whisper model directory; required with --asr')
    p.add_argument('--asr-device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--asr-compute-type', default='int8')
    p.add_argument('--asr-beam-size', type=int, default=5)
    p.add_argument('--local-files-only', action='store_true')
    a = p.parse_args()
    if a.asr and (a.asr_model is None or not a.asr_model.is_dir()):
        p.error('--asr requires --asr-model pointing to an existing local model directory')
    if a.max_shift < 0 or not math.isfinite(a.max_shift):
        p.error('--max-shift must be finite and nonnegative')
    offsets = {}
    if a.offsets:
        with a.offsets.open() as f:
            offsets = {r['sample_id']: float(r['lag_seconds']) for r in csv.DictReader(f)}
    if a.alignment == 'manual' and not offsets:
        p.error('Manual alignment requires --offsets')
    model = None
    if a.asr:
        from faster_whisper import WhisperModel
        asr_model_path = Path(a.asr_model)
        if not asr_model_path.is_absolute() and (ROOT / asr_model_path).is_dir():
            asr_model_path = ROOT / asr_model_path
        model = WhisperModel(str(asr_model_path), device=a.asr_device, compute_type=a.asr_compute_type,
                             local_files_only=True)
    records = []
    sr = 48000
    for row in read_manifest(a.manifest):
        sid = row['sample_id']
        ref = read_audio(row['reference_audio'])
        raw = read_audio(a.run_dir / sid / 'raw.wav')
        if a.alignment == 'manual':
            if sid not in offsets or not math.isfinite(offsets[sid]):
                raise ValueError(f'Missing/invalid manual offset for {sid}')
            lag = offsets[sid]
        elif a.alignment == 'auto':
            _, _, lag = align_for_metrics(ref, raw, sr, a.max_shift)
        else:
            lag = 0.0
        for stage in ['raw', 'preprocessed', 'denoised', 'restored']:
            wav = read_audio(a.run_dir / sid / f'{stage}.wav')
            r, c = apply_alignment_shift(ref, wav, round(lag * sr))
            if len(r) < sr // 2:
                raise ValueError(f'Insufficient overlap for {sid}, lag={lag}')
            rec = dict(sample_id=sid, stage=stage, alignment=a.alignment,
                       lag_seconds=lag, comparison_seconds=len(r) / sr,
                       alignment_review_required=a.alignment == 'auto')
            rec.update(metrics(r, c, sr, a.pesq_mode))
            rec.update(wer=None, wer_status='not_run', hypothesis='')
            if model and stage in a.asr_stages:
                if not row.get('transcript', '').strip():
                    raise ValueError(f'Missing reference transcript for {sid}')
                # Decode the FULL candidate; never use reference text as an ASR prompt.
                audio16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                segments, _ = model.transcribe(audio16, language='en', beam_size=a.asr_beam_size,
                                              vad_filter=False, condition_on_previous_text=False)
                hypothesis = ' '.join(s.text.strip() for s in segments)
                rec.update(wer=word_error_rate(row['transcript'], hypothesis),
                           wer_status='ok', hypothesis=hypothesis)
            records.append(rec)
            print(json.dumps(rec), flush=True)
    output = a.output or a.run_dir / f'metrics_{a.pesq_mode}.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    output.with_suffix('.settings.json').write_text(json.dumps({
        'alignment': a.alignment, 'max_shift_seconds': a.max_shift,
        'pesq_mode': a.pesq_mode,
        'pesq_sample_rates': {m: 8000 if m == 'nb' else 16000 for m in
                             (['nb', 'wb'] if a.pesq_mode == 'both' else [a.pesq_mode])},
        'pesq_notice': 'NB is the author-requested artifact default; the manuscript specifies WB. Scores are not interchangeable.',
        'asr_model': str(a.asr_model) if model else None, 'asr_compute_type': a.asr_compute_type,
        'asr_stages': a.asr_stages,
        'asr_beam_size': a.asr_beam_size, 'wer_normalization': 'lowercase; punctuation removed; no digit/letter rewriting',
        'alignment_notice': 'Automatic estimates require inspection; manual offsets change the reported metrics.'
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
