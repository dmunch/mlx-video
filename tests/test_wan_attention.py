"""Tests for Wan attention components and RoPE."""

import mlx.core as mx
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# RoPE Tests
# ---------------------------------------------------------------------------

class TestRoPE:
    """Tests for 3-way factorized RoPE."""

    def test_rope_params_shape(self):
        from mlx_video.models.wan.rope import rope_params
        freqs = rope_params(1024, 64)
        mx.eval(freqs)
        assert freqs.shape == (1024, 32, 2)  # [max_seq_len, dim//2, 2]

    def test_rope_params_different_dims(self):
        from mlx_video.models.wan.rope import rope_params
        for dim in [32, 64, 128]:
            freqs = rope_params(512, dim)
            mx.eval(freqs)
            assert freqs.shape == (512, dim // 2, 2)

    def test_rope_params_cos_sin_range(self):
        from mlx_video.models.wan.rope import rope_params
        freqs = rope_params(256, 64)
        mx.eval(freqs)
        cos_vals = np.array(freqs[:, :, 0])
        sin_vals = np.array(freqs[:, :, 1])
        assert np.all(cos_vals >= -1.0) and np.all(cos_vals <= 1.0)
        assert np.all(sin_vals >= -1.0) and np.all(sin_vals <= 1.0)

    def test_rope_params_position_zero(self):
        """At position 0, cos should be 1 and sin should be 0."""
        from mlx_video.models.wan.rope import rope_params
        freqs = rope_params(10, 64)
        mx.eval(freqs)
        np.testing.assert_allclose(np.array(freqs[0, :, 0]), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.array(freqs[0, :, 1]), 0.0, atol=1e-6)

    def test_rope_apply_output_shape(self):
        from mlx_video.models.wan.rope import rope_params, rope_apply
        B, L, N, D = 1, 24, 4, 32  # batch, seq, heads, head_dim
        x = mx.random.normal((B, L, N, D))
        freqs = rope_params(1024, D)
        grid_sizes = [(2, 3, 4)]  # F*H*W = 24 = L
        out = rope_apply(x, grid_sizes, freqs)
        mx.eval(out)
        assert out.shape == (B, L, N, D)

    def test_rope_apply_preserves_norm(self):
        """RoPE rotation should preserve vector norms."""
        from mlx_video.models.wan.rope import rope_params, rope_apply
        B, N, D = 1, 2, 16
        F, H, W = 2, 3, 4
        L = F * H * W
        x = mx.random.normal((B, L, N, D))
        freqs = rope_params(1024, D)

        out = rope_apply(x, [(F, H, W)], freqs)
        mx.eval(x, out)

        x_np = np.array(x[0])
        out_np = np.array(out[0])
        for i in range(L):
            for h in range(N):
                norm_in = np.linalg.norm(x_np[i, h])
                norm_out = np.linalg.norm(out_np[i, h])
                np.testing.assert_allclose(norm_in, norm_out, rtol=1e-4)

    def test_rope_apply_with_padding(self):
        """When seq_len < L, extra tokens should be preserved unchanged."""
        from mlx_video.models.wan.rope import rope_params, rope_apply
        B, N, D = 1, 2, 16
        F, H, W = 2, 2, 2
        seq_len = F * H * W  # 8
        pad = 4
        L = seq_len + pad
        x = mx.random.normal((B, L, N, D))
        freqs = rope_params(1024, D)

        out = rope_apply(x, [(F, H, W)], freqs)
        mx.eval(x, out)
        # Padded tokens should be unchanged
        np.testing.assert_allclose(
            np.array(out[0, seq_len:]),
            np.array(x[0, seq_len:]),
            atol=1e-6,
        )

    def test_rope_apply_batch(self):
        """Test with batch_size > 1 and different grid sizes."""
        from mlx_video.models.wan.rope import rope_params, rope_apply
        B, N, D = 2, 2, 16
        grids = [(2, 3, 4), (2, 3, 4)]
        L = 2 * 3 * 4
        x = mx.random.normal((B, L, N, D))
        freqs = rope_params(1024, D)

        out = rope_apply(x, grids, freqs)
        mx.eval(out)
        assert out.shape == (B, L, N, D)

    def test_rope_frequency_split(self):
        """Verify the 3-way frequency dimension split matches Wan2.2 convention."""
        D = 128  # head_dim for 14B model
        half_d = D // 2
        d_t = half_d - 2 * (half_d // 3)
        d_h = half_d // 3
        d_w = half_d // 3
        assert d_t + d_h + d_w == half_d
        # Temporal gets more capacity
        assert d_t >= d_h
        assert d_t >= d_w


# ---------------------------------------------------------------------------
# Attention Tests
# ---------------------------------------------------------------------------

class TestWanRMSNorm:
    def test_output_shape(self):
        from mlx_video.models.wan.attention import WanRMSNorm
        norm = WanRMSNorm(64)
        x = mx.random.normal((2, 10, 64))
        out = norm(x)
        mx.eval(out)
        assert out.shape == (2, 10, 64)

    def test_zero_mean_variance(self):
        """RMS norm should make RMS ≈ 1 before scaling."""
        from mlx_video.models.wan.attention import WanRMSNorm
        norm = WanRMSNorm(64)
        x = mx.random.normal((1, 5, 64)) * 10.0
        out = norm(x)
        mx.eval(out)
        out_np = np.array(out[0])
        for i in range(5):
            rms = np.sqrt(np.mean(out_np[i] ** 2))
            # After RMS norm with weight=1, RMS should be ~1
            np.testing.assert_allclose(rms, 1.0, rtol=0.1)

    def test_dtype_preservation(self):
        """RMSNorm weight is float32, so output is promoted to float32."""
        from mlx_video.models.wan.attention import WanRMSNorm
        norm = WanRMSNorm(32)
        x = mx.random.normal((1, 4, 32)).astype(mx.bfloat16)
        out = norm(x)
        mx.eval(out)
        # Weight is float32, so multiplication promotes result to float32
        assert out.dtype == mx.float32


class TestWanLayerNorm:
    def test_output_shape(self):
        from mlx_video.models.wan.attention import WanLayerNorm
        norm = WanLayerNorm(64)
        x = mx.random.normal((2, 10, 64))
        out = norm(x)
        mx.eval(out)
        assert out.shape == (2, 10, 64)

    def test_without_affine(self):
        from mlx_video.models.wan.attention import WanLayerNorm
        norm = WanLayerNorm(64, elementwise_affine=False)
        x = mx.random.normal((1, 4, 64))
        out = norm(x)
        mx.eval(out)
        # Mean should be ~0, variance should be ~1
        out_np = np.array(out[0])
        for i in range(4):
            np.testing.assert_allclose(np.mean(out_np[i]), 0.0, atol=0.05)
            np.testing.assert_allclose(np.std(out_np[i]), 1.0, rtol=0.1)

    def test_with_affine(self):
        from mlx_video.models.wan.attention import WanLayerNorm
        norm = WanLayerNorm(32, elementwise_affine=True)
        assert hasattr(norm, "weight")
        assert hasattr(norm, "bias")
        x = mx.random.normal((1, 4, 32))
        out = norm(x)
        mx.eval(out)
        assert out.shape == (1, 4, 32)


class TestWanSelfAttention:
    def setup_method(self):
        mx.random.seed(42)
        self.dim = 64
        self.num_heads = 4

    def test_output_shape(self):
        from mlx_video.models.wan.attention import WanSelfAttention
        from mlx_video.models.wan.rope import rope_params
        attn = WanSelfAttention(self.dim, self.num_heads)
        B, L = 1, 24
        F, H, W = 2, 3, 4
        x = mx.random.normal((B, L, self.dim))
        freqs = rope_params(1024, self.dim // self.num_heads)
        out = attn(x, seq_lens=[L], grid_sizes=[(F, H, W)], freqs=freqs)
        mx.eval(out)
        assert out.shape == (B, L, self.dim)

    def test_with_qk_norm(self):
        from mlx_video.models.wan.attention import WanSelfAttention
        attn = WanSelfAttention(self.dim, self.num_heads, qk_norm=True)
        assert attn.norm_q is not None
        assert attn.norm_k is not None

    def test_without_qk_norm(self):
        from mlx_video.models.wan.attention import WanSelfAttention
        attn = WanSelfAttention(self.dim, self.num_heads, qk_norm=False)
        assert attn.norm_q is None
        assert attn.norm_k is None

    def test_masking(self):
        """Test that masking works: shorter seq_lens should mask later tokens."""
        from mlx_video.models.wan.attention import WanSelfAttention
        from mlx_video.models.wan.rope import rope_params
        attn = WanSelfAttention(self.dim, self.num_heads, qk_norm=False)
        B, L = 1, 24
        F, H, W = 2, 3, 4
        x = mx.random.normal((B, L, self.dim))
        freqs = rope_params(1024, self.dim // self.num_heads)

        # Full sequence
        out_full = attn(x, seq_lens=[L], grid_sizes=[(F, H, W)], freqs=freqs)
        # Shorter sequence (mask last 4 tokens)
        out_masked = attn(x, seq_lens=[L - 4], grid_sizes=[(F, H, W)], freqs=freqs)
        mx.eval(out_full, out_masked)

        # Outputs should differ when masking is applied
        assert not np.allclose(np.array(out_full), np.array(out_masked), atol=1e-5)


class TestWanCrossAttention:
    def setup_method(self):
        mx.random.seed(42)
        self.dim = 64
        self.num_heads = 4

    def test_output_shape(self):
        from mlx_video.models.wan.attention import WanCrossAttention
        attn = WanCrossAttention(self.dim, self.num_heads)
        B, L_q, L_kv = 1, 24, 16
        x = mx.random.normal((B, L_q, self.dim))
        context = mx.random.normal((B, L_kv, self.dim))
        out = attn(x, context)
        mx.eval(out)
        assert out.shape == (B, L_q, self.dim)

    def test_with_context_mask(self):
        from mlx_video.models.wan.attention import WanCrossAttention
        attn = WanCrossAttention(self.dim, self.num_heads)
        B, L_q, L_kv = 1, 12, 16
        x = mx.random.normal((B, L_q, self.dim))
        context = mx.random.normal((B, L_kv, self.dim))
        out = attn(x, context, context_lens=[10])
        mx.eval(out)
        assert out.shape == (B, L_q, self.dim)


# ---------------------------------------------------------------------------
# bfloat16 Autocast Tests
# ---------------------------------------------------------------------------

class TestBFloat16Autocast:
    """Tests that attention and FFN cast inputs to weight dtype (bfloat16)
    for efficient matmul, matching official PyTorch autocast behavior."""

    def setup_method(self):
        mx.random.seed(42)
        self.dim = 64
        self.num_heads = 4

    @staticmethod
    def _to_bf16(params):
        """Recursively cast all arrays in params to bfloat16."""
        if isinstance(params, dict):
            return {k: TestBFloat16Autocast._to_bf16(v) for k, v in params.items()}
        elif isinstance(params, list):
            return [TestBFloat16Autocast._to_bf16(v) for v in params]
        elif isinstance(params, mx.array):
            return params.astype(mx.bfloat16)
        return params

    def test_self_attn_casts_to_weight_dtype(self):
        """Self-attention should cast input to weight dtype for QKV projections."""
        from mlx_video.models.wan.attention import WanSelfAttention
        from mlx_video.models.wan.rope import rope_params
        attn = WanSelfAttention(self.dim, self.num_heads)
        attn.update(self._to_bf16(attn.parameters()))

        x = mx.random.normal((1, 8, self.dim))
        freqs = rope_params(1024, self.dim // self.num_heads)
        out = attn(x, seq_lens=[8], grid_sizes=[(2, 2, 2)], freqs=freqs)
        mx.eval(out)
        assert out.shape == (1, 8, self.dim)
        assert np.isfinite(np.array(out.astype(mx.float32))).all()

    def test_cross_attn_casts_to_weight_dtype(self):
        """Cross-attention should cast input to weight dtype."""
        from mlx_video.models.wan.attention import WanCrossAttention
        attn = WanCrossAttention(self.dim, self.num_heads)
        attn.update(self._to_bf16(attn.parameters()))

        x = mx.random.normal((1, 8, self.dim))
        ctx = mx.random.normal((1, 4, self.dim))
        out = attn(x, ctx)
        mx.eval(out)
        assert out.shape == (1, 8, self.dim)
        assert np.isfinite(np.array(out.astype(mx.float32))).all()

    def test_cross_attn_kv_cache_uses_weight_dtype(self):
        """prepare_kv should cast context to weight dtype."""
        from mlx_video.models.wan.attention import WanCrossAttention
        attn = WanCrossAttention(self.dim, self.num_heads)
        attn.update(self._to_bf16(attn.parameters()))

        ctx = mx.random.normal((1, 4, self.dim))
        k, v = attn.prepare_kv(ctx)
        mx.eval(k, v)
        assert k.dtype == mx.bfloat16
        assert v.dtype == mx.bfloat16

    def test_ffn_casts_to_weight_dtype(self):
        """FFN should cast input to weight dtype for linear layers."""
        from mlx_video.models.wan.transformer import WanFFN
        ffn = WanFFN(self.dim, 128)
        ffn.update(self._to_bf16(ffn.parameters()))

        x = mx.random.normal((1, 8, self.dim))
        out = ffn(x)
        mx.eval(out)
        assert out.shape == (1, 8, self.dim)
        assert np.isfinite(np.array(out.astype(mx.float32))).all()

    def test_self_attn_rope_in_float32(self):
        """RoPE should be applied in float32 for precision, even with bf16 weights."""
        from mlx_video.models.wan.attention import WanSelfAttention
        from mlx_video.models.wan.rope import rope_params
        attn = WanSelfAttention(self.dim, self.num_heads)
        attn.update(self._to_bf16(attn.parameters()))

        x = mx.random.normal((1, 8, self.dim))
        freqs = rope_params(1024, self.dim // self.num_heads)
        assert freqs.dtype == mx.float32
        out = attn(x, seq_lens=[8], grid_sizes=[(2, 2, 2)], freqs=freqs)
        mx.eval(out)
        assert np.isfinite(np.array(out.astype(mx.float32))).all()

    def test_block_float32_residual_with_bf16_weights(self):
        """Full block: residual stream stays float32, matmuls use bf16 weights."""
        from mlx_video.models.wan.transformer import WanAttentionBlock
        from mlx_video.models.wan.rope import rope_params
        block = WanAttentionBlock(self.dim, 128, self.num_heads, cross_attn_norm=True)
        block.update(self._to_bf16(block.parameters()))

        B, L = 1, 8
        x = mx.random.normal((B, L, self.dim))
        e = mx.random.normal((B, L, 6, self.dim))
        ctx = mx.random.normal((B, 4, self.dim))
        freqs = rope_params(1024, self.dim // self.num_heads)

        out = block(x, e, [L], [(2, 2, 2)], freqs, ctx)
        mx.eval(out)
        assert out.dtype == mx.float32
        assert np.isfinite(np.array(out)).all()
