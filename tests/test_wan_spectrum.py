"""Tests for Spectrum (Chebyshev Feature Forecasting) acceleration."""

import math

import mlx.core as mx
import numpy as np
import pytest

from wan_test_helpers import _make_tiny_config

from mlx_video.models.wan.model import WanModel
from mlx_video.models.wan.spectrum import (
    ChebyshevForecaster,
    SpectrumForecaster,
    SpectrumState,
    _build_chebyshev_design,
)


class TestChebyshevDesignMatrix:
    """Unit tests for the Chebyshev design matrix builder."""

    def test_shape(self):
        taus = mx.array([0.0, 0.5, 1.0])
        phi = _build_chebyshev_design(taus, M=4)
        assert phi.shape == (3, 5)  # K=3, M+1=5

    def test_first_column_is_ones(self):
        """T_0(x) = 1 for all x."""
        taus = mx.linspace(-1.0, 1.0, 10)
        phi = _build_chebyshev_design(taus, M=3)
        np.testing.assert_allclose(np.array(phi[:, 0]), 1.0, atol=1e-6)

    def test_second_column_is_identity(self):
        """T_1(x) = x."""
        taus = mx.linspace(-1.0, 1.0, 10)
        phi = _build_chebyshev_design(taus, M=3)
        np.testing.assert_allclose(np.array(phi[:, 1]), np.array(taus), atol=1e-5)

    def test_chebyshev_recurrence(self):
        """T_2(x) = 2x*T_1(x) - T_0(x) = 2x^2 - 1."""
        taus = mx.linspace(-1.0, 1.0, 20)
        phi = _build_chebyshev_design(taus, M=2)
        t2_expected = 2.0 * np.array(taus) ** 2 - 1.0
        np.testing.assert_allclose(np.array(phi[:, 2]), t2_expected, atol=1e-5)


class TestChebyshevForecaster:
    """Unit tests for the ChebyshevForecaster ridge regression."""

    def test_fit_and_predict(self):
        """After fitting linear data, predictions should be close."""
        forecaster = ChebyshevForecaster(M=2, K=20, lam=0.01, num_steps=20)

        # Feed linear data: f(step) = step * ones
        F = 100
        for step in range(10):
            h = mx.ones((F,)) * float(step)
            forecaster.update(step, h)

        # Predict at step 5 (interpolation)
        pred = forecaster.predict(5)
        mx.eval(pred)
        assert pred.shape == (F,)
        # Should be close to 5.0
        assert abs(pred.mean().item() - 5.0) < 1.0

    def test_reset_clears_state(self):
        forecaster = ChebyshevForecaster(M=4, K=10, lam=0.1, num_steps=50)
        forecaster.update(0, mx.ones((50,)))
        assert forecaster.num_cached == 1
        forecaster.reset()
        assert forecaster.num_cached == 0
        assert forecaster._coef is None

    def test_minimum_data_for_predict(self):
        """Need at least 2 data points to fit."""
        forecaster = ChebyshevForecaster(M=4, K=10, lam=0.1, num_steps=50)
        forecaster.update(0, mx.ones((50,)))
        # With only 1 point, predict should still work (uses ridge regression)
        pred = forecaster.predict(1)
        mx.eval(pred)
        assert pred.shape == (50,)


class TestSpectrumForecaster:
    """Unit tests for the blended Chebyshev+Taylor forecaster."""

    def test_blend_weights(self):
        """w=1.0 should match pure Chebyshev prediction."""
        sf_cheb = SpectrumForecaster(M=2, K=20, lam=0.01, w=1.0, num_steps=20)
        sf_blend = SpectrumForecaster(M=2, K=20, lam=0.01, w=0.5, num_steps=20)

        F = 50
        for step in range(10):
            h = mx.ones((F,)) * float(step)
            sf_cheb.update(step, h)
            sf_blend.update(step, h)

        pred_cheb = sf_cheb.predict(12)
        pred_blend = sf_blend.predict(12)
        mx.eval(pred_cheb, pred_blend)

        # Blended should differ from pure Chebyshev (since Taylor component differs)
        assert not np.allclose(np.array(pred_cheb), np.array(pred_blend), atol=1e-3)

    def test_predict_shape(self):
        sf = SpectrumForecaster(M=4, K=10, lam=0.1, w=0.5, num_steps=50)
        F = 200
        for step in range(5):
            sf.update(step, mx.random.normal((F,)))
        pred = sf.predict(7)
        mx.eval(pred)
        assert pred.shape == (F,)


class TestSpectrumState:
    """Unit tests for SpectrumState scheduling."""

    def test_default_disabled(self):
        ss = SpectrumState()
        assert ss.enabled is False

    def test_warmup_always_computes(self):
        """All steps during warmup should return True for should_compute."""
        ss = SpectrumState(enabled=True, warmup_steps=5, num_steps=50)
        for i in range(5):
            assert ss.should_compute() is True
            ss.step(computed=True)

    def test_adaptive_schedule_produces_skips(self):
        """After warmup, should_compute should return False for some steps."""
        ss = SpectrumState(
            enabled=True, warmup_steps=3, window_size=2,
            flex_window=0.75, num_steps=20,
        )
        computed_steps = []
        for i in range(20):
            do_it = ss.should_compute()
            if do_it:
                computed_steps.append(i)
            ss.step(computed=do_it)

        # Should have fewer compute steps than total steps
        assert len(computed_steps) < 20
        # First 3 should always be computed (warmup)
        assert computed_steps[:3] == [0, 1, 2]

    def test_reset(self):
        ss = SpectrumState(enabled=True, num_steps=50)
        ss.cnt = 10
        ss.steps_computed = 5
        ss.steps_predicted = 5
        ss.curr_ws = 4.5
        ss.reset()
        assert ss.cnt == 0
        assert ss.steps_computed == 0
        assert ss.steps_predicted == 0
        assert ss.curr_ws == float(ss.window_size)

    def test_nfe_count_50_steps(self):
        """With default params and 50 steps, should get ~14 NFEs (from paper)."""
        ss = SpectrumState(
            enabled=True, warmup_steps=5, window_size=2,
            flex_window=0.75, num_steps=50,
        )
        for i in range(50):
            do_it = ss.should_compute()
            ss.step(computed=do_it)

        total_nfe = ss.steps_computed
        assert 12 <= total_nfe <= 16  # Paper claims ~14

    def test_aggressive_flex_fewer_nfe(self):
        """Higher flex_window should produce fewer NFEs."""
        ss_conservative = SpectrumState(
            enabled=True, warmup_steps=5, window_size=2,
            flex_window=0.75, num_steps=50,
        )
        ss_aggressive = SpectrumState(
            enabled=True, warmup_steps=5, window_size=2,
            flex_window=3.0, num_steps=50,
        )
        for i in range(50):
            ss_conservative.step(computed=ss_conservative.should_compute())
            ss_aggressive.step(computed=ss_aggressive.should_compute())

        assert ss_aggressive.steps_computed < ss_conservative.steps_computed


class TestSpectrumModelIntegration:
    """Integration tests for Spectrum with WanModel."""

    def setup_method(self):
        mx.random.seed(42)

    def test_spectrum_disabled_matches_original(self):
        """Output should be identical with Spectrum disabled."""
        config = _make_tiny_config()
        model = WanModel(config)
        mx.eval(model.parameters())

        C, F, H, W = config.in_dim, 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        latent = mx.random.normal((C, F, H, W))
        t = mx.array([500.0, 500.0])
        context = [mx.random.normal((6, config.text_dim)), mx.random.normal((6, config.text_dim))]

        # Run without Spectrum
        out1 = model([latent, latent], t=t, context=context, seq_len=seq_len)
        mx.eval(out1)

        # Run with Spectrum disabled (default)
        assert model.spectrum.enabled is False
        out2 = model([latent, latent], t=t, context=context, seq_len=seq_len)
        mx.eval(out2)

        np.testing.assert_allclose(
            np.array(out1[0]), np.array(out2[0]), atol=1e-5
        )

    def test_spectrum_computes_and_predicts(self):
        """Spectrum should run full forward on compute steps and predict on skip steps."""
        config = _make_tiny_config()
        model = WanModel(config)
        mx.eval(model.parameters())

        model.spectrum.enabled = True
        model.spectrum.num_steps = 10
        model.spectrum.warmup_steps = 3
        model.spectrum.window_size = 2
        model.spectrum.flex_window = 0.75
        model.spectrum.reset()

        C, F, H, W = config.in_dim, 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        latent = mx.random.normal((C, F, H, W))
        context = [mx.random.normal((6, config.text_dim))]

        # Run 10 steps with decreasing timesteps
        for ts in np.linspace(900, 100, 10):
            t = mx.array([float(ts)])
            out = model([latent], t=t, context=context, seq_len=seq_len)
            mx.eval(out)

        # Should have both compute and predicted steps
        assert model.spectrum.steps_computed > 0
        assert model.spectrum.steps_predicted > 0
        assert model.spectrum.steps_computed + model.spectrum.steps_predicted == 10

    def test_spectrum_output_shape(self):
        """Output shape should be correct with Spectrum enabled."""
        config = _make_tiny_config()
        model = WanModel(config)
        mx.eval(model.parameters())

        model.spectrum.enabled = True
        model.spectrum.num_steps = 5
        model.spectrum.warmup_steps = 5  # All compute, so shape is always correct
        model.spectrum.reset()

        C, F, H, W = config.in_dim, 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        latent = mx.random.normal((C, F, H, W))
        context = [mx.random.normal((6, config.text_dim)), mx.random.normal((6, config.text_dim))]
        t = mx.array([500.0, 500.0])

        out = model([latent, latent], t=t, context=context, seq_len=seq_len)
        mx.eval(out)

        assert len(out) == 2
        assert out[0].shape == (C, F, H, W)
        assert out[1].shape == (C, F, H, W)

    def test_spectrum_reset_between_runs(self):
        """Spectrum state should be cleanly resettable between generations."""
        config = _make_tiny_config()
        model = WanModel(config)
        mx.eval(model.parameters())

        model.spectrum.enabled = True
        model.spectrum.num_steps = 5
        model.spectrum.warmup_steps = 3
        model.spectrum.reset()

        C, F, H, W = config.in_dim, 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        latent = mx.random.normal((C, F, H, W))
        context = [mx.random.normal((6, config.text_dim))]

        # First run
        for ts in [900.0, 800.0]:
            t = mx.array([float(ts)])
            out = model([latent], t=t, context=context, seq_len=seq_len)
            mx.eval(out)

        assert model.spectrum.cnt == 2

        # Reset and verify
        model.spectrum.reset()
        assert model.spectrum.cnt == 0
        assert model.spectrum.forecaster is None

    def test_spectrum_and_teacache_mutual_exclusion(self):
        """When Spectrum is enabled, TeaCache branch should not execute."""
        config = _make_tiny_config()
        model = WanModel(config)
        mx.eval(model.parameters())

        # Enable both (Spectrum takes precedence in model.py)
        model.spectrum.enabled = True
        model.spectrum.num_steps = 5
        model.spectrum.warmup_steps = 5
        model.spectrum.reset()

        model.teacache.enabled = True
        model.teacache.threshold = 0.1
        model.teacache.coefficients = config.teacache_coefficients
        model.teacache.num_steps = 5
        model.teacache.ret_steps = 1
        model.teacache.cutoff_steps = 4

        C, F, H, W = config.in_dim, 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        latent = mx.random.normal((C, F, H, W))
        context = [mx.random.normal((6, config.text_dim))]

        for ts in [900.0, 800.0, 700.0]:
            t = mx.array([float(ts)])
            out = model([latent], t=t, context=context, seq_len=seq_len)
            mx.eval(out)

        # Spectrum should have counted steps, TeaCache should not
        assert model.spectrum.steps_computed == 3
        assert model.teacache.steps_computed == 0
