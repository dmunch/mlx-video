"""Tests for TeaCache (Timestep Embedding Aware Cache) optimization."""

import mlx.core as mx
import numpy as np
import pytest
from wan_test_helpers import _make_tiny_config

from mlx_video.models.wan.model import TeaCacheState, WanModel


class TestTeaCacheState:
    """Unit tests for TeaCacheState dataclass."""

    def test_default_disabled(self):
        tc = TeaCacheState()
        assert tc.enabled is False
        assert tc.threshold == 0.0

    def test_reset(self):
        tc = TeaCacheState(enabled=True, threshold=0.2)
        tc.cnt = 5
        tc.steps_skipped = 3
        tc.steps_computed = 2
        tc.previous_e0 = mx.ones((1,))
        tc.accumulated_distance = 1.5
        tc.reset()
        assert tc.cnt == 0
        assert tc.steps_skipped == 0
        assert tc.steps_computed == 0
        assert tc.previous_e0 is None
        assert tc.accumulated_distance == 0.0

    def test_polynomial_rescaling(self):
        """Verify polynomial rescaling matches np.poly1d."""
        # Ret-mode coefficients for T2V 14B
        coefficients = (
            -3.03318725e05,
            4.90537029e04,
            -2.65530556e03,
            5.87365115e01,
            -3.15583525e-01,
        )
        x = 0.02  # typical relative L1 distance for e0

        # np.poly1d evaluation
        expected = np.poly1d(coefficients)(x)

        # Horner's method (as used in model.py)
        result = coefficients[0]
        for c in coefficients[1:]:
            result = result * x + c

        assert abs(result - expected) < 1e-6


class TestTeaCacheIntegration:
    """Integration tests for TeaCache with WanModel."""

    def setup_method(self):
        mx.random.seed(42)

    def test_teacache_disabled_matches_original(self):
        """Output should be identical with TeaCache disabled."""
        config = _make_tiny_config()
        model = WanModel(config)
        mx.eval(model.parameters())

        C, F, H, W = config.in_dim, 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        latent = mx.random.normal((C, F, H, W))
        t = mx.array([500.0, 500.0])
        context = [
            mx.random.normal((6, config.text_dim)),
            mx.random.normal((6, config.text_dim)),
        ]

        # Run without TeaCache
        out1 = model([latent, latent], t=t, context=context, seq_len=seq_len)
        mx.eval(out1)

        # Run with TeaCache disabled (threshold=0)
        model.teacache.enabled = False
        out2 = model([latent, latent], t=t, context=context, seq_len=seq_len)
        mx.eval(out2)

        np.testing.assert_allclose(np.array(out1[0]), np.array(out2[0]), atol=1e-5)

    def test_teacache_skips_steps(self):
        """With a very high threshold, steps should be skipped after the first."""
        config = _make_tiny_config()
        model = WanModel(config)
        mx.eval(model.parameters())

        # Configure TeaCache
        model.teacache.enabled = True
        model.teacache.threshold = 100.0  # Very high = always skip
        model.teacache.coefficients = config.teacache_coefficients
        model.teacache.num_steps = 10
        model.teacache.ret_steps = 1
        model.teacache.cutoff_steps = 9

        C, F, H, W = config.in_dim, 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        latent = mx.random.normal((C, F, H, W))
        context = [mx.random.normal((6, config.text_dim))]

        # Run several steps with slightly different timesteps
        for ts in [900.0, 850.0, 800.0, 750.0, 700.0]:
            t = mx.array([ts])
            out = model([latent], t=t, context=context, seq_len=seq_len)
            mx.eval(out)

        # First step + steps after ret_steps should have skips
        assert model.teacache.steps_skipped > 0
        assert model.teacache.steps_computed > 0

    def test_teacache_output_shape(self):
        """Output shape should be correct regardless of TeaCache."""
        config = _make_tiny_config()
        model = WanModel(config)
        mx.eval(model.parameters())

        model.teacache.enabled = True
        model.teacache.threshold = 0.2
        model.teacache.coefficients = config.teacache_coefficients
        model.teacache.num_steps = 5
        model.teacache.ret_steps = 1
        model.teacache.cutoff_steps = 4

        C, F, H, W = config.in_dim, 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        latent = mx.random.normal((C, F, H, W))
        context = [
            mx.random.normal((6, config.text_dim)),
            mx.random.normal((6, config.text_dim)),
        ]
        t = mx.array([500.0, 500.0])

        out = model([latent, latent], t=t, context=context, seq_len=seq_len)
        mx.eval(out)

        assert len(out) == 2
        assert out[0].shape == (C, F, H, W)
        assert out[1].shape == (C, F, H, W)

    def test_teacache_reset_between_runs(self):
        """TeaCache state should be cleanly resettable between generations."""
        config = _make_tiny_config()
        model = WanModel(config)
        mx.eval(model.parameters())

        model.teacache.enabled = True
        model.teacache.threshold = 0.2
        model.teacache.coefficients = config.teacache_coefficients
        model.teacache.num_steps = 5
        model.teacache.ret_steps = 1
        model.teacache.cutoff_steps = 4

        C, F, H, W = config.in_dim, 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        latent = mx.random.normal((C, F, H, W))
        context = [mx.random.normal((6, config.text_dim))]

        # First run
        for ts in [900.0, 800.0]:
            t = mx.array([ts])
            out = model([latent], t=t, context=context, seq_len=seq_len)
            mx.eval(out)

        assert model.teacache.cnt == 2

        # Reset and verify
        model.teacache.reset()
        assert model.teacache.cnt == 0
        assert model.teacache.previous_e0 is None
