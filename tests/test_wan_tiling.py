"""Tests for Wan VAE tiled decoding."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_video.models.ltx.video_vae.tiling import (
    TilingConfig,
    decode_with_tiling,
    split_in_spatial,
    split_in_temporal,
)


class TestNonCausalTemporal:
    """Tests for the causal_temporal=False path in decode_with_tiling."""

    def test_split_spatial_for_temporal(self):
        """Non-causal temporal should use split_in_spatial (no causal shift)."""
        intervals = split_in_spatial(8, 2, 20)
        # No causal adjustment: starts should be evenly spaced
        assert intervals.starts[0] == 0
        for i in range(1, len(intervals.starts)):
            assert intervals.starts[i] == intervals.starts[i - 1] + (8 - 2)

    def test_causal_vs_noncausal_output_size(self):
        """Causal temporal gives 1+(T-1)*S frames, non-causal gives T*S."""
        mx.random.seed(42)
        b, c, t, h, w = 1, 4, 4, 4, 4
        latents = mx.random.normal((b, c, t, h, w))
        scale = 4

        # Simple passthrough decoder: just repeat along dimensions
        def dummy_decoder_causal(x, **kwargs):
            b, c, t, h, w = x.shape
            out_t = 1 + (t - 1) * scale
            out_h = h * scale
            out_w = w * scale
            return mx.ones((b, 3, out_t, out_h, out_w))

        def dummy_decoder_noncausal(x, **kwargs):
            b, c, t, h, w = x.shape
            out_t = t * scale
            out_h = h * scale
            out_w = w * scale
            return mx.ones((b, 3, out_t, out_h, out_w))

        config = TilingConfig.spatial_only(tile_size=128, overlap=64)

        # Causal: 1 + (4-1)*4 = 13
        out_causal = decode_with_tiling(
            dummy_decoder_causal, latents, config,
            spatial_scale=scale, temporal_scale=scale, causal_temporal=True,
        )
        mx.eval(out_causal)
        assert out_causal.shape[2] == 1 + (t - 1) * scale  # 13

        # Non-causal: 4*4 = 16
        out_noncausal = decode_with_tiling(
            dummy_decoder_noncausal, latents, config,
            spatial_scale=scale, temporal_scale=scale, causal_temporal=False,
        )
        mx.eval(out_noncausal)
        assert out_noncausal.shape[2] == t * scale  # 16


class TestWan22TiledDecoding:
    """Tests for Wan2.2 VAE tiled decoding."""

    def _make_small_wan22_decoder(self):
        """Create a small Wan2.2 decoder for testing."""
        from mlx_video.models.wan.vae22 import Wan22VAEDecoder

        # Use very small dimensions for fast testing
        vae = Wan22VAEDecoder(z_dim=48, dim=16, dec_dim=16)
        mx.eval(vae.parameters())
        return vae

    def test_decode_tiled_output_shape(self):
        """Tiled decode should produce same shape as non-tiled."""
        mx.random.seed(42)
        vae = self._make_small_wan22_decoder()

        # Small input: [B=1, T=3, H=2, W=2, C=48]
        z = mx.random.normal((1, 3, 2, 2, 48))
        mx.eval(z)

        # Non-tiled
        out_regular = vae(z)
        mx.eval(out_regular)

        # Tiled (force tiling with very small tile sizes)
        # Use spatial tile=32px (2 latent @ scale 16) and temporal=8 frames (2 latent @ scale 4)
        config = TilingConfig(
            spatial_config=None,  # Don't tile spatially (input is tiny)
            temporal_config=None,  # Don't tile temporally (input is tiny)
        )
        # With no tiling config, decode_tiled should fall through to regular decode
        out_tiled = vae.decode_tiled(z, tiling_config=TilingConfig.default())
        mx.eval(out_tiled)

        # Both should produce the same shape
        assert out_regular.shape == out_tiled.shape, (
            f"Shape mismatch: regular={out_regular.shape} vs tiled={out_tiled.shape}"
        )

    def test_decode_tiled_falls_through_when_small(self):
        """When input is smaller than tile size, decode_tiled should produce same output as __call__."""
        mx.random.seed(42)
        vae = self._make_small_wan22_decoder()

        # Input smaller than any tile size
        z = mx.random.normal((1, 2, 2, 2, 48))
        mx.eval(z)

        out_regular = vae(z)
        mx.eval(out_regular)

        out_tiled = vae.decode_tiled(z, tiling_config=TilingConfig.default())
        mx.eval(out_tiled)

        np.testing.assert_allclose(
            np.array(out_regular), np.array(out_tiled),
            rtol=1e-4, atol=1e-4,
            err_msg="Tiled decode should match regular decode for small inputs",
        )


class TestWan21TiledDecoding:
    """Tests for Wan2.1 VAE tiled decoding."""

    def _make_small_wan21_vae(self):
        """Create a small Wan2.1 VAE for testing."""
        from mlx_video.models.wan.vae import WanVAE

        vae = WanVAE(z_dim=16)
        mx.eval(vae.parameters())
        return vae

    def test_decode_tiled_output_shape(self):
        """Tiled decode should produce correct output shape."""
        mx.random.seed(42)
        vae = self._make_small_wan21_vae()

        # [B=1, C=16, T=3, H=4, W=4]
        z = mx.random.normal((1, 16, 3, 4, 4))
        mx.eval(z)

        out_regular = vae.decode(z)
        mx.eval(out_regular)

        out_tiled = vae.decode_tiled(z, tiling_config=TilingConfig.default())
        mx.eval(out_tiled)

        assert out_regular.shape == out_tiled.shape, (
            f"Shape mismatch: regular={out_regular.shape} vs tiled={out_tiled.shape}"
        )

    def test_decode_tiled_falls_through_when_small(self):
        """When input is smaller than tile size, decode_tiled should produce same output as decode."""
        mx.random.seed(42)
        vae = self._make_small_wan21_vae()

        z = mx.random.normal((1, 16, 2, 4, 4))
        mx.eval(z)

        out_regular = vae.decode(z)
        mx.eval(out_regular)

        out_tiled = vae.decode_tiled(z, tiling_config=TilingConfig.default())
        mx.eval(out_tiled)

        np.testing.assert_allclose(
            np.array(out_regular), np.array(out_tiled),
            rtol=1e-4, atol=1e-4,
            err_msg="Tiled decode should match regular decode for small inputs",
        )


class TestWan21TemporalScale:
    """Verify Wan2.1 decoder temporal output is T*4 (non-causal)."""

    def test_wan21_decoder_temporal_output(self):
        """Wan2.1 Decoder3d should produce T*4 temporal output (non-causal doubling)."""
        from mlx_video.models.wan.vae import Decoder3d

        # Small decoder for fast test
        dec = Decoder3d(dim=16, z_dim=4, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                        temporal_upsample=[True, True, False])
        mx.eval(dec.parameters())

        x = mx.random.normal((1, 4, 3, 4, 4))  # T=3
        mx.eval(x)
        out = dec(x)
        mx.eval(out)

        # With two temporal 2× upsamples: T=3 → 6 → 12
        assert out.shape[2] == 3 * 4, f"Expected T=12, got T={out.shape[2]}"
