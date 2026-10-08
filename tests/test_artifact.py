"""Offline unit tests. These never open a radio or transmit RF."""
import sys
import unittest
from pathlib import Path
import tempfile
import importlib.util
from unittest.mock import patch
import numpy as np
import torch
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'step1_rf_acquisition'), str(ROOT / 'step2_em_raw')]
from ghostwhisper.models import MelDenoiseResUNet
from ghostwhisper.stft_model import STFTUNet, stft, istft, compress_spec, decompress_spec
from ghostwhisper.bounded_refiner import refine_waveform
from ghostwhisper.metrics import apply_alignment_shift, align_for_metrics, word_error_rate
from ghostwhisper.simulation import simulate_noisy_waveform
from ghostwhisper.training_data import split_rows_by_clean_id, PairWaveformDataset
from ghostwhisper.stable_tone_filter import detect_stable_tones, apply_notch_filters
from capture_magnitude import save_h5
from radio_hardware import generate_frequency_points
from magnitude_to_audio import remove_dc, resample_with_antialias, apply_optional_lowpass, load_h5_magnitude


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_parameter_counts_and_forward(self):
        for cls, count, name in [(MelDenoiseResUNet, 26298817, 'denoiser'), (STFTUNet, 17411522, 'refiner')]:
            m = cls(base_channels=64 if name == 'denoiser' else 48)
            self.assertEqual(sum(p.numel() for p in m.parameters()), count)
            m.eval()
            shape = (1, 1, 64, 65) if name == 'denoiser' else (1, 3, 65, 33)
            x = torch.zeros(shape)
            with torch.inference_mode():
                self.assertTrue(torch.isfinite(m(x)).all())

    def test_stft_roundtrip(self):
        x = torch.randn(1, 4096)
        s = stft(x, 1024, 256)
        torch.testing.assert_close(decompress_spec(compress_spec(s, .3), .3), s, atol=2e-4, rtol=2e-5)
        torch.testing.assert_close(istft(s, 4096, 1024, 256), x, atol=2e-5, rtol=2e-5)

    def test_bounded_refiner_identity_and_bound(self):
        model=STFTUNet(base_channels=8,depth=2,dropout=0.)
        torch.nn.init.zeros_(model.output[-1].weight)
        torch.nn.init.zeros_(model.output[-1].bias)
        wave=torch.randn(1,4096)*.1
        out,delta=refine_waveform(model,wave,model_device='cpu',residual_scale=.1)
        torch.testing.assert_close(out,wave)
        self.assertEqual(float(delta.abs().max()),0.)
        with torch.no_grad():model.output[-1].bias.fill_(100.)
        _,delta=refine_waveform(model,wave,model_device='cpu',residual_scale=.1)
        self.assertLessEqual(float(delta.detach().abs().max()),.100001)

    def test_offset_sign_and_alignment(self):
        x = np.arange(50,dtype=np.float32)
        y = np.pad(x,(7,0))
        r,c = apply_alignment_shift(x,y,7)
        np.testing.assert_array_equal(r,c)
        r,c = apply_alignment_shift(y,x,-7)
        np.testing.assert_array_equal(r,c)
        sr = 48000
        rng = np.random.default_rng(3)
        audio = rng.normal(size=sr*3).astype(np.float32)
        audio *= np.repeat(rng.uniform(.01,1,30),sr//10)
        delayed = np.pad(audio,(sr//5,0))
        _,_,lag = align_for_metrics(audio,delayed,sr,.5)
        self.assertAlmostEqual(lag,.2,delta=.02)

    def test_wer(self):
        self.assertEqual(word_error_rate('Hello, world!', 'hello world'),0)
        self.assertEqual(word_error_rate('one two','one three'),.5)
        self.assertEqual(word_error_rate('one','one two three'),2)

    @unittest.skipUnless(importlib.util.find_spec('pesq'), 'PESQ requires the optional native build')
    def test_pesq_modes_and_reference_identity(self):
        spec = importlib.util.spec_from_file_location('artifact_evaluate', ROOT/'step4_evaluation/evaluate.py')
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        sr = 48000
        t = np.arange(sr * 3) / sr
        x = ((.3 + .2 * np.sin(2 * np.pi * 3 * t)) *
             (np.sin(2 * np.pi * 220 * t) + .3 * np.sin(2 * np.pi * 660 * t))).astype(np.float32)
        with patch('pesq.pesq', return_value=3.25) as mocked:
            out = evaluator.metrics(x, x, sr)
            self.assertIn('pesq_nb', out)
            self.assertNotIn('pesq_wb', out)
            self.assertEqual(mocked.call_args.args[0], 8000)
            self.assertEqual(mocked.call_args.args[3], 'nb')
        out = evaluator.metrics(x, x, sr, 'both')
        self.assertGreater(out['pesq_nb'], 4.)
        self.assertGreater(out['pesq_wb'], 4.)
        with self.assertRaises(ValueError): evaluator.metrics(x, x, sr, 'unknown')

    def test_simulation_reproducible(self):
        sr=48000
        x=np.sin(2*np.pi*400*np.arange(sr)/sr).astype(np.float32)
        a,p=simulate_noisy_waveform(x,sr,np.random.default_rng(12))
        b,q=simulate_noisy_waveform(x,sr,np.random.default_rng(12))
        np.testing.assert_array_equal(a,b)
        self.assertEqual(p.to_dict(),q.to_dict())
        self.assertTrue(np.isfinite(a).all())

    def test_notch(self):
        sr=48000
        x=np.sin(2*np.pi*3000*np.arange(sr*2)/sr).astype(np.float32)
        tones=detect_stable_tones(x,sr)
        self.assertTrue(any(abs(t.frequency_hz-3000)<15 for t in tones))
        out=apply_notch_filters(x,sr,tones,normalize_output=False)
        self.assertLess(np.std(out[sr//2:]),np.std(x)/5)

    def test_capture_conversion_hdf5_contract(self):
        values=np.linspace(.1,.3,1000,dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'capture.h5'
            save_h5(values,1e6,559e6,.001,p)
            actual,rate=load_h5_magnitude(p,'magnitude',None)
            np.testing.assert_array_equal(actual,values)
            self.assertEqual(rate,1e6)

    def test_conversion(self):
        sr=500000
        t=np.arange(sr//5)/sr
        mag=(2+.1*np.sin(2*np.pi*1000*t)+.1*np.sin(2*np.pi*60000*t)).astype(np.float32)
        x=apply_optional_lowpass(remove_dc(mag,True),sr,100000,8)
        y=resample_with_antialias(x,sr,48000,'soxr_hq')
        self.assertEqual(len(y),9600)
        f=np.fft.rfftfreq(len(y),1/48000)
        self.assertAlmostEqual(f[np.argmax(abs(np.fft.rfft(y)))],1000,delta=10)

    def test_sweep_endpoints(self):
        self.assertEqual(len(generate_frequency_points(200e6,800e6,10e6)),61)

    def test_split_and_rate_guards(self):
        rows=[dict(clean_clip_id='same',split='train'),dict(clean_clip_id='same',split='val')]
        with self.assertRaises(ValueError): split_rows_by_clean_id(rows,.1,1)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'wrong_rate.wav'; sf.write(p,np.zeros(16000),16000)
            ds=PairWaveformDataset([dict(clean_audio_path=str(p),noisy_audio_path=str(p),pair_id='x')])
            with self.assertRaises(ValueError): ds[0]


if __name__ == '__main__':
    unittest.main()
