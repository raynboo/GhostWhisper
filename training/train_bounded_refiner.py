#!/usr/bin/env python3
"""Fast, explicitly new refiner training with frozen denoiser and frozen wav2vec2 CTC."""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio.functional as AF

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ghostwhisper.stft_model import STFTUNet, stft
from ghostwhisper.bounded_refiner import refine_waveform
from ghostwhisper.training_data import read_pair_manifest, split_rows_by_clean_id, PairWaveformDataset
from ghostwhisper.io_utils import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--init-checkpoint',type=Path,help='Optional user-supplied initialization; omitted means random initialization')
    p.add_argument('--ctc-model',type=Path,required=True,help='User-supplied local wav2vec2 model and processor directory')
    p.add_argument('--ctc-revision',default=None,help='Optional model revision identifier')
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--residual-scale',type=float,default=.1)
    p.add_argument('--ctc-weight',type=float,default=.01)
    p.add_argument('--seed',type=int,default=20260918)
    p.add_argument('--local-files-only',action='store_true')
    a=p.parse_args()
    if not a.ctc_model.is_dir():p.error('--ctc-model must be an existing local directory')
    a.ctc_model=str(a.ctc_model)
    a.local_files_only=True
    if a.output_dir.exists() and any(a.output_dir.iterdir()):p.error('Choose an empty output directory')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    rows=read_pair_manifest(a.manifest)
    train,val=split_rows_by_clean_id(rows,.2,a.seed)
    loaders={name:DataLoader(PairWaveformDataset(data),batch_size=1,shuffle=name=='train',num_workers=0)
             for name,data in [('train',train),('val',val)]}
    model=STFTUNet(base_channels=48,depth=4,dropout=0.).to(a.device)
    if a.init_checkpoint:
        model.load_state_dict(torch.load(a.init_checkpoint,map_location='cpu',weights_only=True)['model'])
    # Identity start is crucial: legacy output head is not a bounded residual head.
    torch.nn.init.zeros_(model.output[-1].weight);torch.nn.init.zeros_(model.output[-1].bias)
    processor=Wav2Vec2Processor.from_pretrained(a.ctc_model,revision=a.ctc_revision,local_files_only=a.local_files_only)
    asr=Wav2Vec2ForCTC.from_pretrained(a.ctc_model,revision=a.ctc_revision,local_files_only=a.local_files_only).to(a.device).eval()
    asr.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    config=dict(base_channels=48,depth=4,sample_rate=48000,clip_seconds=4.,n_fft=1024,
                hop_length=256,compress_power=.3,output_mode='bounded_waveform',residual_scale=a.residual_scale,
                seed=a.seed,ctc_weight=a.ctc_weight,ctc_model=a.ctc_model,ctc_revision=getattr(asr.config,'_commit_hash',None),
                experiment='bounded-waveform refiner training; not original paper training reproduction',
                learning_rate=a.lr,epochs=a.epochs,batch_size=1,
                loss_weights=dict(spectral=1.,detail=.1,preservation=.5,temporal=.1,ctc=a.ctc_weight),
                script_sha256=sha256(Path(__file__)),
                denoiser_frozen=True,ctc_frozen=True,training_manifest_sha256=sha256(a.manifest),
                init_checkpoint_sha256=sha256(a.init_checkpoint) if a.init_checkpoint else None,
                train_examples=len(train),validation_examples=len(val),parameters=sum(x.numel() for x in model.parameters()))
    (a.output_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    def epoch_run(phase):
        model.train(phase=='train');total={};start=time.perf_counter()
        for i,b in enumerate(loaders[phase],1):
            source=b['noisy'];clean=b['clean']
            with torch.set_grad_enabled(phase=='train'):
                pred,delta=refine_waveform(model,source,model_device=a.device,residual_scale=a.residual_scale)
                clean=clean.to(pred.device);source=source.to(pred.device)
                ps=torch.log1p(stft(pred,1024,256).abs());cs=torch.log1p(stft(clean,1024,256).abs())
                spectral=F.l1_loss(ps,cs)
                detail=F.l1_loss(ps[...,1:]-ps[...,:-1],cs[...,1:]-cs[...,:-1])
                preservation=delta.square().mean()/(source.square().mean().detach()+1e-5)
                energy=lambda x:F.avg_pool1d(x.square().unsqueeze(1),960,480)
                temporal=F.l1_loss(energy(pred),energy(source))/(energy(source).mean().detach()+1e-5)
                wave16=AF.resample(pred,48000,16000).to(a.device)
                wave16=(wave16-wave16.mean(-1,keepdim=True))/wave16.std(-1,keepdim=True).clamp_min(1e-4)
                logits=asr(wave16).logits.float().log_softmax(-1).transpose(0,1).cpu()
                labels=[re.sub(r"[^A-Z' ]",'',t.upper()) for t in b['transcript']]
                tokens=processor.tokenizer(labels,padding=True,return_tensors='pt')
                ids=tokens.input_ids;mask=ids!=processor.tokenizer.pad_token_id
                ctc=F.ctc_loss(logits,ids.masked_select(mask),torch.full((len(labels),),logits.shape[0],dtype=torch.long),
                               mask.sum(-1),blank=processor.tokenizer.pad_token_id,zero_infinity=False)
                loss=spectral+.1*detail+.5*preservation+.1*temporal+a.ctc_weight*ctc.to(pred.device)
                if not torch.isfinite(loss):raise RuntimeError(f'Non-finite {phase} loss at {b["pair_id"]}')
                if phase=='train':
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    optimizer.step()
            values=dict(loss=float(loss.detach()),spectral=float(spectral.detach()),ctc=float(ctc.detach()),
                        preservation=float(preservation.detach()),temporal=float(temporal.detach()),
                        max_correction=float(delta.detach().abs().max()))
            for k,v in values.items():total[k]=total.get(k,0.)+v
            if i%16==0:print(json.dumps(dict(phase=phase,step=i,total=len(loaders[phase]),**values)),flush=True)
        return {**{k:v/len(loaders[phase]) for k,v in total.items()},'seconds':time.perf_counter()-start}
    def save(name,epoch,result):
        torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':config,
                    'epoch':epoch,'validation':result},a.output_dir/name)
    with torch.no_grad():baseline=epoch_run('val')
    best=baseline['loss'];best_epoch=0
    save('identity_baseline.pt',0,baseline)
    records=[dict(epoch=0,phase='val',**baseline)]
    print(json.dumps(dict(epoch=0,phase='identity_baseline',**baseline)),flush=True)
    for epoch in range(1,a.epochs+1):
        tr=epoch_run('train')
        with torch.no_grad():va=epoch_run('val')
        records.extend([dict(epoch=epoch,phase='train',**tr),dict(epoch=epoch,phase='val',**va)])
        print(json.dumps(dict(epoch=epoch,train=tr,val=va)),flush=True)
        save('last.pt',epoch,va)
        if va['loss']<best-1e-5:
            best=va['loss'];best_epoch=epoch;save('best.pt',epoch,va)
        with (a.output_dir/'metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        (a.output_dir/'status.json').write_text(json.dumps(dict(completed_epochs=epoch,best_epoch=best_epoch,
                  best_val_loss=best,identity_val_loss=baseline['loss'],finished=epoch==a.epochs),indent=2)+'\n')


if __name__=='__main__':main()
