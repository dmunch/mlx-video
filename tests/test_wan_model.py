"""Tests for Wan model components."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from wan_test_helpers import _make_tiny_config


# ---------------------------------------------------------------------------
# Sinusoidal Embedding Tests
# ---------------------------------------------------------------------------

class TestSinusoidalEmbedding:
    def test_output_shape(self):
        from mlx_video.models.wan.model import sinusoidal_embedding_1d
        pos = mx.arange(10).astype(mx.float32)
        emb = sinusoidal_embedding_1d(256, pos)
        mx.eval(emb)
        assert emb.shape == (10, 256)

    def test_position_zero(self):
        """Position 0 should have cos=1 for all dims and sin=0."""
        from mlx_video.models.wan.model import sinusoidal_embedding_1d
        pos = mx.array([0.0])
        emb = sinusoidal_embedding_1d(64, pos)
        mx.eval(emb)
        emb_np = np.array(emb[0])
        # First half is cos, should be 1 at position 0
        np.testing.assert_allclose(emb_np[:32], 1.0, atol=1e-5)
        # Second half is sin, should be 0 at position 0
        np.testing.assert_allclose(emb_np[32:], 0.0, atol=1e-5)

    def test_different_positions_differ(self):
        from mlx_video.models.wan.model import sinusoidal_embedding_1d
        pos = mx.array([0.0, 100.0, 999.0])
        emb = sinusoidal_embedding_1d(128, pos)
        mx.eval(emb)
        emb_np = np.array(emb)
        assert not np.allclose(emb_np[0], emb_np[1])
        assert not np.allclose(emb_np[1], emb_np[2])


# ---------------------------------------------------------------------------
# Head Tests
# ---------------------------------------------------------------------------

class TestHead:
    def test_output_shape(self):
        from mlx_video.models.wan.model import Head
        head = Head(dim=64, out_dim=16, patch_size=(1, 2, 2))
        B, L = 1, 24
        x = mx.random.normal((B, L, 64))
        e = mx.random.normal((B, 64))  # time embedding: [B, dim]
        out = head(x, e)
        mx.eval(out)
        expected_proj_dim = 16 * 1 * 2 * 2  # 64
        assert out.shape == (B, L, expected_proj_dim)

    def test_modulation_shape(self):
        from mlx_video.models.wan.model import Head
        head = Head(dim=64, out_dim=16, patch_size=(1, 2, 2))
        assert head.modulation.shape == (1, 2, 64)


# ---------------------------------------------------------------------------
# WanModel (Tiny) Tests
# ---------------------------------------------------------------------------

class TestWanModel:
    def setup_method(self):
        mx.random.seed(42)

    def test_instantiation(self):
        from mlx_video.models.wan.model import WanModel
        config = _make_tiny_config()
        model = WanModel(config)
        num_params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
        assert num_params > 0

    def test_patchify_shape(self):
        from mlx_video.models.wan.model import WanModel
        config = _make_tiny_config()
        model = WanModel(config)
        # Input: [C=4, F=1, H=4, W=4]
        x = mx.random.normal((4, 1, 4, 4))
        patches, grid_size = model._patchify(x)
        mx.eval(patches)
        # Patch size (1,2,2): F'=1, H'=2, W'=2
        assert grid_size == (1, 2, 2)
        assert patches.shape == (1, 1 * 2 * 2, config.dim)

    def test_patchify_various_sizes(self):
        from mlx_video.models.wan.model import WanModel
        config = _make_tiny_config()
        model = WanModel(config)
        for f, h, w in [(1, 4, 4), (2, 6, 8), (3, 4, 6)]:
            x = mx.random.normal((config.in_dim, f, h, w))
            patches, (gf, gh, gw) = model._patchify(x)
            mx.eval(patches)
            pt, ph, pw = config.patch_size
            assert gf == f // pt
            assert gh == h // ph
            assert gw == w // pw
            assert patches.shape[1] == gf * gh * gw

    def test_unpatchify_inverse(self):
        """Patchify then unpatchify should reconstruct original spatial dims."""
        from mlx_video.models.wan.model import WanModel
        config = _make_tiny_config()
        model = WanModel(config)
        C, F, H, W = config.in_dim, 2, 4, 6
        pt, ph, pw = config.patch_size
        F_out, H_out, W_out = F // pt, H // ph, W // pw
        L = F_out * H_out * W_out
        proj_dim = config.out_dim * pt * ph * pw
        # Simulated head output
        x = mx.random.normal((1, L, proj_dim))
        out = model.unpatchify(x, [(F_out, H_out, W_out)])
        mx.eval(out[0])
        assert out[0].shape == (config.out_dim, F, H, W)

    def test_forward_pass(self):
        from mlx_video.models.wan.model import WanModel
        config = _make_tiny_config()
        model = WanModel(config)
        C, F, H, W = config.in_dim, 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        x_list = [mx.random.normal((C, F, H, W))]
        t = mx.array([500.0])
        context = [mx.random.normal((6, config.text_dim))]

        out = model(x_list, t, context, seq_len)
        mx.eval(out[0])
        assert len(out) == 1
        assert out[0].shape == (C, F, H, W)

    def test_forward_batch(self):
        from mlx_video.models.wan.model import WanModel
        config = _make_tiny_config()
        model = WanModel(config)
        C, F, H, W = config.in_dim, 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        x_list = [mx.random.normal((C, F, H, W)), mx.random.normal((C, F, H, W))]
        t = mx.array([500.0, 200.0])
        context = [mx.random.normal((6, config.text_dim)), mx.random.normal((4, config.text_dim))]

        out = model(x_list, t, context, seq_len)
        mx.eval(out[0], out[1])
        assert len(out) == 2
        for o in out:
            assert o.shape == (C, F, H, W)

    def test_output_is_float32(self):
        from mlx_video.models.wan.model import WanModel
        config = _make_tiny_config()
        model = WanModel(config)
        C, F, H, W = config.in_dim, 1, 4, 4
        seq_len = (F // 1) * (H // 2) * (W // 2)
        out = model([mx.random.normal((C, F, H, W))], mx.array([100.0]),
                     [mx.random.normal((4, config.text_dim))], seq_len)
        mx.eval(out[0])
        assert out[0].dtype == mx.float32


# ---------------------------------------------------------------------------
# Wan2.1 Model Tests
# ---------------------------------------------------------------------------

class TestWan21Model:
    """Test tiny Wan2.1-style model (single model mode)."""

    def setup_method(self):
        mx.random.seed(42)

    def _make_tiny_wan21_config(self):
        """Create a tiny config mimicking Wan2.1 (single model)."""
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig.wan21_t2v_14b()
        # Override to tiny values
        config.dim = 64
        config.ffn_dim = 128
        config.num_heads = 4
        config.num_layers = 2
        config.in_dim = 4
        config.out_dim = 4
        config.freq_dim = 32
        config.text_dim = 32
        config.text_len = 8
        return config

    def _make_tiny_wan21_1_3b_config(self):
        """Create a tiny config mimicking Wan2.1 1.3B."""
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig.wan21_t2v_1_3b()
        # Override to tiny values (preserve 1.3B head structure: 12 heads)
        config.dim = 48
        config.ffn_dim = 96
        config.num_heads = 4
        config.num_layers = 2
        config.in_dim = 4
        config.out_dim = 4
        config.freq_dim = 24
        config.text_dim = 24
        config.text_len = 8
        return config

    def test_wan21_tiny_model_forward(self):
        """Forward pass with Wan2.1 tiny config."""
        from mlx_video.models.wan.model import WanModel

        config = self._make_tiny_wan21_config()
        model = WanModel(config)

        C, F, H, W = config.in_dim, 1, 4, 4
        seq_len = (F // 1) * (H // 2) * (W // 2)

        latents = mx.random.normal((C, F, H, W))
        context = mx.random.normal((4, config.text_dim))
        t = mx.array([500.0])

        out = model([latents], t, [context], seq_len)
        mx.eval(out)
        assert out[0].shape == (C, F, H, W)

    def test_wan21_1_3b_tiny_model_forward(self):
        """Forward pass with Wan2.1 1.3B tiny config."""
        from mlx_video.models.wan.model import WanModel

        config = self._make_tiny_wan21_1_3b_config()
        model = WanModel(config)

        C, F, H, W = config.in_dim, 1, 4, 4
        seq_len = (F // 1) * (H // 2) * (W // 2)

        latents = mx.random.normal((C, F, H, W))
        context = mx.random.normal((4, config.text_dim))
        t = mx.array([500.0])

        out = model([latents], t, [context], seq_len)
        mx.eval(out)
        assert out[0].shape == (C, F, H, W)

    def test_wan21_single_model_loop(self):
        """Full diffusion loop with single model (Wan2.1 style)."""
        from mlx_video.models.wan.model import WanModel
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler

        config = self._make_tiny_wan21_config()
        model = WanModel(config)

        C, F, H, W = config.in_dim, 1, 4, 4
        seq_len = (F // 1) * (H // 2) * (W // 2)

        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(config.sample_steps, shift=config.sample_shift)

        # Use only 3 steps for speed
        latents = mx.random.normal((C, F, H, W))
        context = mx.random.normal((4, config.text_dim))
        context_null = mx.zeros((4, config.text_dim))
        gs = config.sample_guide_scale  # Should be float for Wan2.1

        assert isinstance(gs, float), "Wan2.1 guide_scale should be float"

        for i in range(3):
            t = sched.timesteps[i]
            pred_cond = model([latents], mx.array([t.item()]), [context], seq_len)[0]
            pred_uncond = model([latents], mx.array([t.item()]), [context_null], seq_len)[0]
            pred = pred_uncond + gs * (pred_cond - pred_uncond)
            latents = sched.step(pred[None], t, latents[None]).squeeze(0)
            mx.eval(latents)

        assert latents.shape == (C, F, H, W)
        assert not mx.any(mx.isnan(latents)).item()

    def test_wan21_vs_wan22_config_differences(self):
        """Verify key differences between Wan2.1 and Wan2.2 configs."""
        from mlx_video.models.wan.config import WanModelConfig

        c21 = WanModelConfig.wan21_t2v_14b()
        c22 = WanModelConfig.wan22_t2v_14b()

        # Same architecture
        assert c21.dim == c22.dim
        assert c21.num_heads == c22.num_heads
        assert c21.num_layers == c22.num_layers

        # Different pipeline settings
        assert c21.dual_model is False
        assert c22.dual_model is True
        assert isinstance(c21.sample_guide_scale, float)
        assert isinstance(c22.sample_guide_scale, tuple)
        assert c21.sample_shift != c22.sample_shift
        assert c21.sample_steps != c22.sample_steps


# ---------------------------------------------------------------------------
# Per-Token Timestep Tests
# ---------------------------------------------------------------------------

class TestPerTokenTimestep:
    """Tests for per-token sinusoidal embedding."""

    def test_1d_unchanged(self):
        from mlx_video.models.wan.model import sinusoidal_embedding_1d

        pos = mx.array([0.0, 100.0, 500.0])
        emb = sinusoidal_embedding_1d(256, pos)
        assert emb.shape == (3, 256)

    def test_2d_per_token(self):
        from mlx_video.models.wan.model import sinusoidal_embedding_1d

        pos = mx.array([[0.0, 100.0, 100.0], [50.0, 50.0, 50.0]])
        emb = sinusoidal_embedding_1d(256, pos)
        assert emb.shape == (2, 3, 256)

    def test_consistency(self):
        from mlx_video.models.wan.model import sinusoidal_embedding_1d

        pos_1d = mx.array([0.0, 100.0])
        emb_1d = sinusoidal_embedding_1d(256, pos_1d)
        pos_2d = mx.array([[0.0, 100.0]])
        emb_2d = sinusoidal_embedding_1d(256, pos_2d)
        assert mx.array_equal(emb_1d[0], emb_2d[0, 0])
        assert mx.array_equal(emb_1d[1], emb_2d[0, 1])
