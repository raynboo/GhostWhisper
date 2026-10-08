"""Numerical measurement checks; no hardware access."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reflection_measurement'))
from calibration_core import solve_osl, forward_model, apply_osl
from impedance import gamma_to_impedance, tone_semiaxes


class ReflectionTests(unittest.TestCase):
    def test_osl_inverse(self):
        standards = np.array([1, -1, 0], dtype=complex)
        a, b, c = .8 + .1j, .02 - .01j, .1 + .03j
        measured = (a * standards + b) / (c * standards + 1)
        model = solve_osl(measured, standards)
        target = np.array([.2 + .3j, -.4 + .1j])
        recovered, valid, _ = apply_osl(forward_model(target, model), model)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(recovered, target, atol=1e-6)

    def test_impedance_conversion_and_singularity(self):
        expected = np.array([50, 37 + 108j, 0], dtype=complex)
        gamma = (expected - 50) / (expected + 50)
        z, valid = gamma_to_impedance(np.r_[gamma, 1, np.nan], np.ones(5, bool))
        np.testing.assert_allclose(z[:3], expected)
        np.testing.assert_array_equal(valid, [True, True, True, False, False])
        self.assertTrue(np.isnan(z[3:]).all())

    def test_complex_tone_semiaxes(self):
        t = np.arange(50000) / 50000
        phase = 2 * np.pi * 1000 * t
        z = 37 + 108j + np.exp(.3j) * (.04474 * np.cos(phase) + .01j * np.sin(phase))
        metrics = tone_semiaxes(t, z, np.ones(len(t), bool), 1000)
        self.assertAlmostEqual(metrics['major_axis_semiamplitude_ohm'], .04474, places=9)
        self.assertAlmostEqual(metrics['minor_axis_semiamplitude_ohm'], .01, places=9)
        self.assertLess(metrics['fit_residual_rms_ohm'], 1e-10)
