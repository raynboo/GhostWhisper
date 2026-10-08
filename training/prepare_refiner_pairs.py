#!/usr/bin/env python3
"""Create a NEW small, speaker-separated VCTK experiment; not the paper's original split."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ghostwhisper.io_utils import read_audio, write_audio, sha256
from ghostwhisper.restoration import ResUNetRestorer
from ghostwhisper.simulation import simulate_noisy_waveform
from ghostwhisper.stable_tone_filter import filter_stable_tones


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vctk-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--denoiser',type=Path,required=True,help='User-supplied denoiser checkpoint')
    p.add_argument('--train-count',type=int,default=96)
    p.add_argument('--val-count',type=int,default=24)
    p.add_argument('--seed',type=int,default=20260918)
    p.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu')
    a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error('Choose an empty output directory')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    key=lambda text:hashlib.sha256(f'{a.seed}:{text}'.encode()).hexdigest()
    wavs=list((a.vctk_root/'wav48').glob('*/*.wav'))
    speakers=sorted({x.parent.name for x in wavs},key=key)
    if len(speakers)<4:raise ValueError('Insufficient VCTK speakers')
    # Reserve speakers before selecting clips. This is explicitly a new split.
    val_speakers=set(speakers[:max(2,len(speakers)//5)])
    pools={'train':[],'val':[]};text_sets={'train':set(),'val':set()}
    for wav in sorted(wavs,key=lambda x:key(x.stem)):
        info=sf.info(wav)
        if not 1.5<=info.duration<=4.0:continue
        txt=a.vctk_root/'txt'/wav.parent.name/f'{wav.stem}.txt'
        if not txt.exists():continue
        transcript=txt.read_text().strip()
        if not 4<=len(transcript.split())<=14:continue
        split='val' if wav.parent.name in val_speakers else 'train'
        if transcript.lower() in text_sets['train']|text_sets['val']:continue
        target=a.val_count if split=='val' else a.train_count
        if len(pools[split])>=target:continue
        pools[split].append((wav,transcript));text_sets[split].add(transcript.lower())
        if len(pools['train'])==a.train_count and len(pools['val'])==a.val_count:break
    if len(pools['train'])<a.train_count or len(pools['val'])<a.val_count:
        raise ValueError('Not enough complete short utterances')
    torch.set_num_threads(4)
    model=ResUNetRestorer(a.denoiser,SimpleNamespace(sample_rate=48000,resunet_chunk_seconds=4,resunet_base_channels=64),torch.device(a.device))
    rows=[];sources=[]
    for split,pairs in pools.items():
        for wav,transcript in pairs:
            sid=wav.stem
            clean=read_audio(wav)
            duration=len(clean)/48000
            clean=np.pad(clean,(0,192000-len(clean)))
            rng=np.random.default_rng(int(key(sid)[:8],16))
            noisy,params=simulate_noisy_waveform(clean,48000,rng,buried_probability=.5)
            pre,_=filter_stable_tones(noisy,48000)
            denoised=model.restore(pre)
            for name,x in [('clean',clean),('denoised',denoised)]:
                write_audio(a.output_dir/name/f'{sid}.wav',x)
            rows.append(dict(pair_id=sid,clean_clip_id=sid,speaker_id=wav.parent.name,split=split,
                             clean_audio_path=f'clean/{sid}.wav',noisy_audio_path=f'denoised/{sid}.wav',
                             transcript=transcript,duration_sec=duration,sample_rate=48000))
            sources.append(dict(sample_id=sid,source=str(wav.relative_to(a.vctk_root)),source_sha256=sha256(wav),
                                split=split,noise=params.to_dict()))
            if len(rows)%16==0:print(f'prepared {len(rows)}/{a.train_count+a.val_count}',flush=True)
    with (a.output_dir/'manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a.output_dir/'provenance.json').write_text(json.dumps(dict(
        purpose='Speaker-separated refiner experiment, not original paper training reproduction',seed=a.seed,
        denoiser_sha256=sha256(a.denoiser),training_speakers=sorted(set(speakers)-val_speakers),
        validation_speakers=sorted(val_speakers),train_count=a.train_count,val_count=a.val_count,
        full_utterances_only=True,reference_transcript_not_cropped=True,samples=sources),indent=2)+'\n')


if __name__=='__main__':main()
