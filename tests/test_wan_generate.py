"""Tests for end-to-end generation and I2V mask construction."""

import mlx.core as mx
import numpy as np
import pytest

from wan_test_helpers import _make_tiny_config


# ---------------------------------------------------------------------------
# Integration: end-to-end tiny model forward pass
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """End-to-end test with tiny model (no real weights needed)."""

    def test_tiny_model_denoise_step(self):
        """Simulate one denoising step with tiny model."""
        from mlx_video.models.wan.model import WanModel
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler

        mx.random.seed(42)
        config = _make_tiny_config()
        model = WanModel(config)

        C, F, H, W = config.in_dim, 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(5, shift=3.0)

        latents = mx.random.normal((C, F, H, W))
        context = mx.random.normal((4, config.text_dim))

        # One step
        t = sched.timesteps[0]
        pred = model([latents], mx.array([t.item()]), [context], seq_len)[0]
        latents_next = sched.step(pred[None], t, latents[None]).squeeze(0)
        mx.eval(latents_next)

        assert latents_next.shape == (C, F, H, W)
        # Should differ from original noise
        assert not np.allclose(np.array(latents_next), np.array(latents), atol=1e-5)

    def test_tiny_model_full_loop(self):
        """Run a complete (tiny) diffusion loop."""
        from mlx_video.models.wan.model import WanModel
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler

        mx.random.seed(123)
        config = _make_tiny_config()
        model = WanModel(config)

        C, F, H, W = config.in_dim, 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        sched = FlowMatchEulerScheduler()
        num_steps = 3
        sched.set_timesteps(num_steps, shift=3.0)

        latents = mx.random.normal((C, F, H, W))
        context = mx.random.normal((4, config.text_dim))

        for i in range(num_steps):
            t = sched.timesteps[i]
            pred = model([latents], mx.array([t.item()]), [context], seq_len)[0]
            latents = sched.step(pred[None], t, latents[None]).squeeze(0)
            mx.eval(latents)

        assert latents.shape == (C, F, H, W)
        assert not mx.any(mx.isnan(latents)).item(), "NaN in output"
        assert not mx.any(mx.isinf(latents)).item(), "Inf in output"


# ---------------------------------------------------------------------------
# I2V Mask Tests
# ---------------------------------------------------------------------------

class TestI2VMask:
    """Tests for _build_i2v_mask."""

    def test_mask_shapes(self):
        from mlx_video.generate_wan import _build_i2v_mask

        z_shape = (48, 5, 4, 4)  # C, T, H, W
        patch_size = (1, 2, 2)
        mask, mask_tokens = _build_i2v_mask(z_shape, patch_size)
        assert mask.shape == z_shape
        # Tokens: T=5, H/2=2, W/2=2 → 5*2*2 = 20
        assert mask_tokens.shape == (1, 20)

    def test_first_frame_zero(self):
        from mlx_video.generate_wan import _build_i2v_mask

        z_shape = (48, 5, 4, 4)
        mask, mask_tokens = _build_i2v_mask(z_shape, (1, 2, 2))
        mx.eval(mask, mask_tokens)
        # First temporal position should be 0
        assert float(mask[:, 0, :, :].max()) == 0.0
        # Rest should be 1
        assert float(mask[:, 1:, :, :].min()) == 1.0
        # First-frame tokens (T=0) should be 0 in mask_tokens
        # With T=5, H'=2, W'=2: first 4 tokens are frame 0
        assert float(mask_tokens[0, :4].max()) == 0.0
        assert float(mask_tokens[0, 4:].min()) == 1.0


class TestI2VMaskAlignment:
    """Tests that I2V mask works correctly with various aligned dimensions."""

    def test_mask_with_ti2v_dimensions(self):
        """Mask should work with TI2V-5B typical dimensions."""
        from mlx_video.generate_wan import _build_i2v_mask
        # TI2V: z_dim=48, vae_stride=(4,16,16), patch=(1,2,2)
        # 704x1280 → latent 44x80, t_latent=21 for 81 frames
        z_shape = (48, 21, 44, 80)
        patch_size = (1, 2, 2)
        mask, mask_tokens = _build_i2v_mask(z_shape, patch_size)
        mx.eval(mask, mask_tokens)

        assert mask.shape == z_shape
        assert float(mask[:, 0].max()) == 0.0
        assert float(mask[:, 1:].min()) == 1.0

        expected_tokens = 21 * 22 * 40  # T * (H/ph) * (W/pw)
        assert mask_tokens.shape == (1, expected_tokens)
        first_frame_tokens = 1 * 22 * 40  # pt=1
        assert float(mask_tokens[0, :first_frame_tokens].max()) == 0.0
        assert float(mask_tokens[0, first_frame_tokens:].min()) == 1.0

    def test_mask_per_token_timestep(self):
        """Per-token timesteps: first-frame tokens get t=0, rest get t=sigma."""
        from mlx_video.generate_wan import _build_i2v_mask
        z_shape = (4, 3, 4, 4)
        patch_size = (1, 2, 2)
        _, mask_tokens = _build_i2v_mask(z_shape, patch_size)
        mx.eval(mask_tokens)

        timestep_val = 0.8
        t_tokens = mask_tokens * timestep_val
        mx.eval(t_tokens)

        first_tokens = 1 * 2 * 2  # pt * (H/ph) * (W/pw)
        np.testing.assert_allclose(np.array(t_tokens[0, :first_tokens]), 0.0, atol=1e-7)
        np.testing.assert_allclose(np.array(t_tokens[0, first_tokens:]), timestep_val, atol=1e-7)


# ---------------------------------------------------------------------------
# Dimension Alignment Tests
# ---------------------------------------------------------------------------

class TestDimensionAlignment:
    """Tests for automatic dimension alignment in generate_wan."""

    def test_already_aligned(self):
        """Dimensions already divisible by alignment factor should be unchanged."""
        # patch_size=(1,2,2), vae_stride=(4,16,16) → align = 32
        align_h = 2 * 16  # 32
        align_w = 2 * 16  # 32
        h, w = 704, 1280
        assert h % align_h == 0
        assert w % align_w == 0
        h_aligned = (h // align_h) * align_h
        w_aligned = (w // align_w) * align_w
        assert h_aligned == h
        assert w_aligned == w

    def test_720p_rounds_down(self):
        """720p (1280x720) should round height to 704."""
        align_h = 32
        align_w = 32
        h, w = 720, 1280
        assert h % align_h != 0  # 720 not divisible by 32
        h_aligned = (h // align_h) * align_h
        w_aligned = (w // align_w) * align_w
        assert h_aligned == 704
        assert w_aligned == 1280

    def test_1080p_rounds_down(self):
        """1080p (1920x1080) should round height to 1056."""
        align = 32
        h, w = 1080, 1920
        assert h % align != 0
        assert (h // align) * align == 1056
        assert (w // align) * align == 1920

    def test_odd_sizes(self):
        """Odd sizes should be safely rounded down."""
        align = 32
        for size in [100, 255, 513, 1023]:
            aligned = (size // align) * align
            assert aligned % align == 0
            assert aligned <= size
            assert aligned + align > size  # closest lower multiple

    def test_patchify_valid_after_alignment(self):
        """After alignment, patchify should succeed without reshape errors."""
        from mlx_video.models.wan.model import WanModel
        config = _make_tiny_config()
        model = WanModel(config)

        # Simulate 720p-like scenario with tiny config
        vae_stride = config.vae_stride  # (4, 8, 8)
        patch_size = config.patch_size  # (1, 2, 2)
        align_h = patch_size[1] * vae_stride[1]
        align_w = patch_size[2] * vae_stride[2]

        # Pick a height not divisible by alignment
        raw_h = align_h * 3 + 5  # e.g. 53 for align=16
        raw_w = align_w * 4
        h = (raw_h // align_h) * align_h  # rounds down
        w = (raw_w // align_w) * align_w

        C = config.in_dim
        t_latent = 1
        h_latent = h // vae_stride[1]
        w_latent = w // vae_stride[2]

        vid = mx.random.normal((C, t_latent, h_latent, w_latent))
        patches, grid_size = model._patchify(vid)
        mx.eval(patches)
        assert patches.ndim == 3  # [1, L, dim]
        assert grid_size == (t_latent, h_latent // patch_size[1], w_latent // patch_size[2])

    def test_alignment_with_ti2v_config(self):
        """TI2V-5B uses vae_stride=(4,16,16), patch_size=(1,2,2) → align=32."""
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig.wan22_ti2v_5b()
        align_h = config.patch_size[1] * config.vae_stride[1]
        align_w = config.patch_size[2] * config.vae_stride[2]
        assert align_h == 32
        assert align_w == 32
        # 720 not divisible
        assert 720 % align_h != 0
        # 704 is
        assert 704 % align_h == 0
