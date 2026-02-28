"""Tests for Wan RoPE frequency construction (Bug 6 regression tests).

These tests verify that the RoPE frequency table is built correctly by
concatenating three separate rope_params calls with different dimension
normalizations, matching the reference implementation.

Background: The reference Wan model constructs RoPE frequencies as:
    d = dim // num_heads  (128 for all Wan models)
    freqs = cat([
        rope_params(1024, d - 4*(d//6)),   # temporal (dim=44, 22 freqs)
        rope_params(1024, 2*(d//6)),         # height   (dim=42, 21 freqs)
        rope_params(1024, 2*(d//6)),         # width    (dim=42, 21 freqs)
    ])

A previous incorrect fix used a single rope_params(1024, 128) call, which
gave height/width axes only medium/high frequencies instead of full-range.
This destroyed spatial position encoding and caused grey/artifact output.
"""

import mlx.core as mx
import numpy as np
import pytest


class TestRoPEFrequencyConstruction:
    """Verify WanModel builds RoPE frequencies matching the reference."""

    def _get_model_freqs(self, dim=64, num_heads=4):
        """Instantiate a tiny WanModel and return its .freqs tensor."""
        from mlx_video.models.wan.config import WanModelConfig
        from mlx_video.models.wan.model import WanModel

        config = WanModelConfig()
        config.dim = dim
        config.ffn_dim = dim * 2
        config.num_heads = num_heads
        config.num_layers = 1
        config.in_dim = 4
        config.out_dim = 4
        config.freq_dim = 32
        config.text_dim = 32
        config.text_len = 8
        model = WanModel(config)
        mx.eval(model.freqs)
        return model.freqs, dim // num_heads

    def test_freqs_shape(self):
        """Freqs should be [1024, head_dim//2, 2] regardless of construction."""
        freqs, head_dim = self._get_model_freqs(dim=64, num_heads=4)
        assert freqs.shape == (1024, head_dim // 2, 2)

    def test_three_call_vs_single_call_differ(self):
        """Three separate rope_params calls must differ from single call."""
        from mlx_video.models.wan.rope import rope_params

        d = 128  # head_dim for all Wan models
        # Reference: three separate calls
        correct = mx.concatenate([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ], axis=1)
        # Wrong: single call
        wrong = rope_params(1024, d)
        mx.eval(correct, wrong)

        assert correct.shape == wrong.shape
        diff = np.abs(np.array(correct) - np.array(wrong)).max()
        assert diff > 0.1, f"Three-call and single-call should differ significantly, got max diff {diff}"

    def test_each_axis_starts_at_frequency_one(self):
        """Each axis (temporal/height/width) should have cos=1, sin=0 at position 0.

        This verifies each axis gets its own independent frequency range
        starting from theta^0 = 1.0 (i.e., exponent 0/dim).
        """
        from mlx_video.models.wan.rope import rope_params

        d = 128
        freqs = mx.concatenate([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ], axis=1)
        mx.eval(freqs)
        f = np.array(freqs)

        half_d = d // 2  # 64
        d_t = half_d - 2 * (half_d // 3)  # 22
        d_h = half_d // 3  # 21

        # At position 0, cos=1 and sin=0 for ALL frequency components
        np.testing.assert_allclose(f[0, :, 0], 1.0, atol=1e-6, err_msg="cos at pos 0")
        np.testing.assert_allclose(f[0, :, 1], 0.0, atol=1e-6, err_msg="sin at pos 0")

        # At position 1, each axis should have its FIRST frequency near cos(1/theta^0)=cos(1)
        # Temporal axis first freq
        np.testing.assert_allclose(f[1, 0, 0], np.cos(1.0), atol=1e-5,
                                   err_msg="temporal[0] cos at pos 1")
        # Height axis first freq (starts at index d_t)
        np.testing.assert_allclose(f[1, d_t, 0], np.cos(1.0), atol=1e-5,
                                   err_msg="height[0] cos at pos 1")
        # Width axis first freq (starts at index d_t + d_h)
        np.testing.assert_allclose(f[1, d_t + d_h, 0], np.cos(1.0), atol=1e-5,
                                   err_msg="width[0] cos at pos 1")

    def test_height_width_frequencies_identical(self):
        """Height and width axes should have identical frequency tables.

        Both use rope_params(1024, 2*(d//6)) = rope_params(1024, 42).
        """
        from mlx_video.models.wan.rope import rope_params

        d = 128
        d_h_dim = 2 * (d // 6)  # 42
        freqs = mx.concatenate([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, d_h_dim),
            rope_params(1024, d_h_dim),
        ], axis=1)
        mx.eval(freqs)
        f = np.array(freqs)

        half_d = d // 2
        d_t = half_d - 2 * (half_d // 3)
        d_h = half_d // 3

        height_freqs = f[:, d_t:d_t + d_h]
        width_freqs = f[:, d_t + d_h:]
        np.testing.assert_array_equal(height_freqs, width_freqs)

    def test_frequency_range_per_axis(self):
        """Each axis should span a full frequency range, not a slice of one range.

        With three-call construction, the inverse frequency at index 0 of each
        axis should be 1.0 (theta^0). A single-call approach would give height
        starting at ~0.04 and width at ~0.002 instead of 1.0.
        """
        from mlx_video.models.wan.rope import rope_params

        d = 128
        freqs = mx.concatenate([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ], axis=1)
        mx.eval(freqs)
        f = np.array(freqs)

        half_d = d // 2
        d_t = half_d - 2 * (half_d // 3)
        d_h = half_d // 3

        # At position 1, the first frequency component of each axis should
        # have significant magnitude (cos ≈ 0.54), not near-zero
        pos1_t = f[1, 0, 0]  # temporal first freq
        pos1_h = f[1, d_t, 0]  # height first freq
        pos1_w = f[1, d_t + d_h, 0]  # width first freq

        assert pos1_t > 0.5, f"Temporal first freq at pos 1 should be >0.5, got {pos1_t}"
        assert pos1_h > 0.5, f"Height first freq at pos 1 should be >0.5, got {pos1_h}"
        assert pos1_w > 0.5, f"Width first freq at pos 1 should be >0.5, got {pos1_w}"

    def test_model_freqs_match_manual_construction(self):
        """WanModel.freqs should match manually constructed three-call freqs."""
        from mlx_video.models.wan.rope import rope_params

        freqs_model, head_dim = self._get_model_freqs(dim=64, num_heads=4)
        d = head_dim  # 16
        freqs_manual = mx.concatenate([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ], axis=1)
        mx.eval(freqs_model, freqs_manual)
        np.testing.assert_array_equal(
            np.array(freqs_model), np.array(freqs_manual),
            err_msg="WanModel.freqs should use three-call construction"
        )

    def test_model_freqs_14b_dimensions(self):
        """Verify freq dimensions for 14B-scale head_dim=128."""
        from mlx_video.models.wan.rope import rope_params

        d = 128
        freqs = mx.concatenate([
            rope_params(1024, d - 4 * (d // 6)),  # dim=44 → 22 freq pairs
            rope_params(1024, 2 * (d // 6)),        # dim=42 → 21 freq pairs
            rope_params(1024, 2 * (d // 6)),        # dim=42 → 21 freq pairs
        ], axis=1)
        mx.eval(freqs)

        assert freqs.shape == (1024, 64, 2)
        # Verify the split dimensions used by rope_apply
        half_d = 64
        d_t = half_d - 2 * (half_d // 3)
        d_h = half_d // 3
        d_w = half_d // 3
        assert (d_t, d_h, d_w) == (22, 21, 21)
        assert d_t + d_h + d_w == half_d


class TestRoPEFrequencyMatchesReference:
    """Cross-validate MLX RoPE against PyTorch reference implementation."""

    @pytest.fixture
    def has_torch(self):
        try:
            import torch
            return True
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_freqs_match_pytorch_reference(self, has_torch):
        """Numerically compare MLX and PyTorch frequency tables."""
        import torch
        from mlx_video.models.wan.rope import rope_params

        d = 128

        # PyTorch reference (from wan/modules/model.py)
        def pt_rope_params(max_seq_len, dim, theta=10000):
            freqs = torch.outer(
                torch.arange(max_seq_len),
                1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
            freqs = torch.polar(torch.ones_like(freqs), freqs)
            return freqs

        ref = torch.cat([
            pt_rope_params(1024, d - 4 * (d // 6)),
            pt_rope_params(1024, 2 * (d // 6)),
            pt_rope_params(1024, 2 * (d // 6)),
        ], dim=1)

        # MLX
        ours = mx.concatenate([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ], axis=1)
        mx.eval(ours)

        our_cos = np.array(ours[:, :, 0])
        our_sin = np.array(ours[:, :, 1])
        ref_cos = ref.real.float().numpy()
        ref_sin = ref.imag.float().numpy()

        np.testing.assert_allclose(our_cos, ref_cos, atol=1e-6,
                                   err_msg="cos mismatch vs PyTorch reference")
        np.testing.assert_allclose(our_sin, ref_sin, atol=1e-6,
                                   err_msg="sin mismatch vs PyTorch reference")


class TestRoPEApplyWithCorrectFreqs:
    """Test that rope_apply produces correct rotations with three-call freqs."""

    def test_different_spatial_positions_get_different_rotations(self):
        """Adjacent height/width positions must produce different RoPE rotations.

        This is the key property that was broken by the single-call bug:
        height/width frequencies were too low to distinguish nearby positions.
        """
        from mlx_video.models.wan.rope import rope_params, rope_apply

        d = 128
        freqs = mx.concatenate([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ], axis=1)

        B, N = 1, 4
        F, H, W = 1, 4, 4
        L = F * H * W
        # Use a constant input so differences come purely from RoPE
        x = mx.ones((B, L, N, d))
        out = rope_apply(x, [(F, H, W)], freqs)
        mx.eval(out)
        out_np = np.array(out[0])

        # Position (0,0,0) vs (0,1,0) — different height
        pos_00 = out_np[0 * H * W + 0 * W + 0]  # (f=0, h=0, w=0)
        pos_10 = out_np[0 * H * W + 1 * W + 0]  # (f=0, h=1, w=0)
        height_diff = np.abs(pos_00 - pos_10).max()

        # Position (0,0,0) vs (0,0,1) — different width
        pos_01 = out_np[0 * H * W + 0 * W + 1]  # (f=0, h=0, w=1)
        width_diff = np.abs(pos_00 - pos_01).max()

        # Max diff should be >0.5 for both axes. With the bug, height was ~0.04
        # and width was ~0.002. With correct freqs, both are ~1.3.
        assert height_diff > 0.5, (
            f"Adjacent height positions should differ significantly, got {height_diff:.4f}"
        )
        assert width_diff > 0.5, (
            f"Adjacent width positions should differ significantly, got {width_diff:.4f}"
        )
        # Height and width should have identical frequency tables → same diffs
        np.testing.assert_allclose(height_diff, width_diff, rtol=1e-5,
            err_msg="Height and width should use identical frequency tables")

    def test_precomputed_matches_online(self):
        """rope_precompute_cos_sin + rope_apply should match non-precomputed path."""
        from mlx_video.models.wan.rope import (
            rope_apply,
            rope_params,
            rope_precompute_cos_sin,
        )

        d = 128
        freqs = mx.concatenate([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ], axis=1)

        B, N = 2, 4
        F, H, W = 2, 3, 4
        L = F * H * W
        grids = [(F, H, W), (F, H, W)]

        x = mx.random.normal((B, L, N, d))

        # Online (no precomputed)
        out_online = rope_apply(x, grids, freqs)
        # Precomputed
        cos_sin = rope_precompute_cos_sin(grids, freqs)
        out_precomp = rope_apply(x, grids, freqs, precomputed_cos_sin=cos_sin)
        mx.eval(out_online, out_precomp)

        np.testing.assert_allclose(
            np.array(out_online), np.array(out_precomp), atol=1e-5,
            err_msg="Precomputed and online RoPE should match"
        )
