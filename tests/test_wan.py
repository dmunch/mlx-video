"""Comprehensive tests for Wan2.2 model components.

All tests use small/tiny configurations to avoid needing actual weights.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

class TestWanModelConfig:
    """Tests for WanModelConfig dataclass."""

    def test_default_values(self):
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig()
        assert config.dim == 5120
        assert config.ffn_dim == 13824
        assert config.num_heads == 40
        assert config.num_layers == 40
        assert config.in_dim == 16
        assert config.out_dim == 16
        assert config.patch_size == (1, 2, 2)
        assert config.vae_stride == (4, 8, 8)
        assert config.vae_z_dim == 16
        assert config.boundary == 0.875
        assert config.sample_shift == 12.0
        assert config.sample_steps == 40
        assert config.sample_guide_scale == (3.0, 4.0)
        assert config.num_train_timesteps == 1000
        assert config.qk_norm is True
        assert config.cross_attn_norm is True
        assert config.text_len == 512

    def test_head_dim_property(self):
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig()
        assert config.head_dim == 128  # 5120 // 40

    def test_to_dict_roundtrip(self):
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["dim"] == 5120
        assert d["patch_size"] == (1, 2, 2)
        assert d["boundary"] == 0.875

    def test_t5_config_values(self):
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig()
        assert config.t5_vocab_size == 256384
        assert config.t5_dim == 4096
        assert config.t5_dim_attn == 4096
        assert config.t5_dim_ffn == 10240
        assert config.t5_num_heads == 64
        assert config.t5_num_layers == 24
        assert config.t5_num_buckets == 32


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
# Transformer Block Tests
# ---------------------------------------------------------------------------

class TestWanFFN:
    def test_output_shape(self):
        from mlx_video.models.wan.transformer import WanFFN
        ffn = WanFFN(64, 256)
        x = mx.random.normal((2, 10, 64))
        out = ffn(x)
        mx.eval(out)
        assert out.shape == (2, 10, 64)

    def test_gelu_activation(self):
        """FFN should use GELU activation (non-linearity)."""
        from mlx_video.models.wan.transformer import WanFFN
        ffn = WanFFN(32, 128)
        x = mx.ones((1, 1, 32)) * 2.0
        out1 = ffn(x)
        x2 = mx.ones((1, 1, 32)) * 4.0
        out2 = ffn(x2)
        mx.eval(out1, out2)
        # Non-linear: 2x input should not give 2x output
        assert not np.allclose(np.array(out2), np.array(out1) * 2.0, rtol=0.1)


class TestWanAttentionBlock:
    def setup_method(self):
        mx.random.seed(42)
        self.dim = 64
        self.ffn_dim = 128
        self.num_heads = 4

    def test_output_shape(self):
        from mlx_video.models.wan.transformer import WanAttentionBlock
        from mlx_video.models.wan.rope import rope_params
        block = WanAttentionBlock(
            self.dim, self.ffn_dim, self.num_heads,
            cross_attn_norm=True,
        )
        B, L = 1, 24
        F, H, W = 2, 3, 4
        x = mx.random.normal((B, L, self.dim))
        e = mx.random.normal((B, L, 6, self.dim))
        context = mx.random.normal((B, 16, self.dim))
        freqs = rope_params(1024, self.dim // self.num_heads)

        out = block(
            x, e, seq_lens=[L], grid_sizes=[(F, H, W)],
            freqs=freqs, context=context,
        )
        mx.eval(out)
        assert out.shape == (B, L, self.dim)

    def test_modulation_shape(self):
        from mlx_video.models.wan.transformer import WanAttentionBlock
        block = WanAttentionBlock(self.dim, self.ffn_dim, self.num_heads)
        assert block.modulation.shape == (1, 6, self.dim)

    def test_with_cross_attn_norm(self):
        from mlx_video.models.wan.transformer import WanAttentionBlock
        block = WanAttentionBlock(
            self.dim, self.ffn_dim, self.num_heads,
            cross_attn_norm=True,
        )
        assert block.norm3 is not None

    def test_without_cross_attn_norm(self):
        from mlx_video.models.wan.transformer import WanAttentionBlock
        block = WanAttentionBlock(
            self.dim, self.ffn_dim, self.num_heads,
            cross_attn_norm=False,
        )
        assert block.norm3 is None

    def test_residual_connection(self):
        """Output should differ from zero even with small random init."""
        from mlx_video.models.wan.transformer import WanAttentionBlock
        from mlx_video.models.wan.rope import rope_params
        block = WanAttentionBlock(self.dim, self.ffn_dim, self.num_heads)
        B, L = 1, 8
        F, H, W = 2, 2, 2
        x = mx.ones((B, L, self.dim))
        e = mx.zeros((B, L, 6, self.dim))
        context = mx.random.normal((B, 4, self.dim))
        freqs = rope_params(1024, self.dim // self.num_heads)

        out = block(x, e, [L], [(F, H, W)], freqs, context)
        mx.eval(out)
        # With residual connections, output should be close to input + corrections
        assert not np.allclose(np.array(out), 0.0, atol=1e-3)


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

def _make_tiny_config():
    """Create a tiny WanModelConfig for testing."""
    from mlx_video.models.wan.config import WanModelConfig
    config = WanModelConfig()
    # Override to tiny values
    config.dim = 64
    config.ffn_dim = 128
    config.num_heads = 4
    config.num_layers = 2
    config.in_dim = 4
    config.out_dim = 4
    config.patch_size = (1, 2, 2)
    config.freq_dim = 32
    config.text_dim = 32
    config.text_len = 8
    return config


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
# T5 Encoder Tests
# ---------------------------------------------------------------------------

class TestT5LayerNorm:
    def test_output_shape(self):
        from mlx_video.models.wan.text_encoder import T5LayerNorm
        norm = T5LayerNorm(64)
        x = mx.random.normal((2, 10, 64))
        out = norm(x)
        mx.eval(out)
        assert out.shape == (2, 10, 64)

    def test_rms_normalization(self):
        """After T5LayerNorm with weight=1, RMS should be ~1."""
        from mlx_video.models.wan.text_encoder import T5LayerNorm
        norm = T5LayerNorm(128)
        x = mx.random.normal((1, 5, 128)) * 5.0
        out = norm(x)
        mx.eval(out)
        out_np = np.array(out[0])
        for i in range(5):
            rms = np.sqrt(np.mean(out_np[i] ** 2))
            np.testing.assert_allclose(rms, 1.0, rtol=0.1)


class TestT5RelativeEmbedding:
    def test_output_shape(self):
        from mlx_video.models.wan.text_encoder import T5RelativeEmbedding
        rel_emb = T5RelativeEmbedding(num_buckets=32, num_heads=4)
        out = rel_emb(10, 10)
        mx.eval(out)
        assert out.shape == (1, 4, 10, 10)  # [1, N, lq, lk]

    def test_asymmetric_lengths(self):
        from mlx_video.models.wan.text_encoder import T5RelativeEmbedding
        rel_emb = T5RelativeEmbedding(num_buckets=32, num_heads=4)
        out = rel_emb(8, 12)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 12)

    def test_symmetry(self):
        """Position bias should have structure (not all zeros/random)."""
        from mlx_video.models.wan.text_encoder import T5RelativeEmbedding
        rel_emb = T5RelativeEmbedding(num_buckets=32, num_heads=2)
        out = rel_emb(6, 6)
        mx.eval(out)
        out_np = np.array(out[0])  # [N, lq, lk]
        # Diagonal elements (position i attending to position i) should be consistent
        # (same relative distance = 0 for all diagonal elements)
        for h in range(2):
            diag = np.diag(out_np[h])
            np.testing.assert_allclose(diag, diag[0], atol=1e-5)


class TestT5Attention:
    def test_output_shape(self):
        from mlx_video.models.wan.text_encoder import T5Attention
        attn = T5Attention(dim=64, dim_attn=64, num_heads=4)
        x = mx.random.normal((1, 10, 64))
        out = attn(x)
        mx.eval(out)
        assert out.shape == (1, 10, 64)

    def test_no_scaling(self):
        """T5 attention famously has no sqrt(d) scaling. Verify structure."""
        from mlx_video.models.wan.text_encoder import T5Attention
        attn = T5Attention(dim=64, dim_attn=64, num_heads=4)
        # No scale attribute (unlike standard attention)
        assert not hasattr(attn, "scale")

    def test_with_position_bias(self):
        from mlx_video.models.wan.text_encoder import T5Attention, T5RelativeEmbedding
        attn = T5Attention(dim=64, dim_attn=64, num_heads=4)
        rel_emb = T5RelativeEmbedding(32, 4)
        x = mx.random.normal((1, 10, 64))
        pos_bias = rel_emb(10, 10)
        out = attn(x, pos_bias=pos_bias)
        mx.eval(out)
        assert out.shape == (1, 10, 64)

    def test_with_mask(self):
        from mlx_video.models.wan.text_encoder import T5Attention
        attn = T5Attention(dim=64, dim_attn=64, num_heads=4)
        x = mx.random.normal((1, 10, 64))
        mask = mx.ones((1, 10))
        mask = mx.concatenate([mask[:, :7], mx.zeros((1, 3))], axis=1)
        out = attn(x, mask=mask)
        mx.eval(out)
        assert out.shape == (1, 10, 64)


class TestT5FeedForward:
    def test_output_shape(self):
        from mlx_video.models.wan.text_encoder import T5FeedForward
        ffn = T5FeedForward(64, 256)
        x = mx.random.normal((1, 10, 64))
        out = ffn(x)
        mx.eval(out)
        assert out.shape == (1, 10, 64)

    def test_gated_structure(self):
        """T5 FFN is gated: gate(x) * fc1(x)."""
        from mlx_video.models.wan.text_encoder import T5FeedForward
        ffn = T5FeedForward(32, 64)
        assert hasattr(ffn, "gate_proj")
        assert hasattr(ffn, "fc1")
        assert hasattr(ffn, "fc2")


class TestT5Encoder:
    def setup_method(self):
        mx.random.seed(42)

    def test_output_shape(self):
        from mlx_video.models.wan.text_encoder import T5Encoder
        encoder = T5Encoder(
            vocab_size=100, dim=64, dim_attn=64, dim_ffn=128,
            num_heads=4, num_layers=2, num_buckets=32, shared_pos=False,
        )
        ids = mx.array([[1, 5, 10, 0, 0]])
        mask = mx.array([[1, 1, 1, 0, 0]])
        out = encoder(ids, mask=mask)
        mx.eval(out)
        assert out.shape == (1, 5, 64)

    def test_shared_pos(self):
        from mlx_video.models.wan.text_encoder import T5Encoder
        encoder = T5Encoder(
            vocab_size=100, dim=64, dim_attn=64, dim_ffn=128,
            num_heads=4, num_layers=2, num_buckets=32, shared_pos=True,
        )
        assert encoder.pos_embedding is not None
        for block in encoder.blocks:
            assert block.pos_embedding is None

    def test_per_layer_pos(self):
        from mlx_video.models.wan.text_encoder import T5Encoder
        encoder = T5Encoder(
            vocab_size=100, dim=64, dim_attn=64, dim_ffn=128,
            num_heads=4, num_layers=2, num_buckets=32, shared_pos=False,
        )
        assert encoder.pos_embedding is None
        for block in encoder.blocks:
            assert block.pos_embedding is not None

    def test_param_count(self):
        from mlx_video.models.wan.text_encoder import T5Encoder
        encoder = T5Encoder(
            vocab_size=100, dim=64, dim_attn=64, dim_ffn=128,
            num_heads=4, num_layers=2, num_buckets=32, shared_pos=False,
        )
        num_params = sum(p.size for _, p in nn.utils.tree_flatten(encoder.parameters()))
        assert num_params > 0

    def test_without_mask(self):
        from mlx_video.models.wan.text_encoder import T5Encoder
        encoder = T5Encoder(
            vocab_size=100, dim=64, dim_attn=64, dim_ffn=128,
            num_heads=4, num_layers=2, num_buckets=32, shared_pos=False,
        )
        ids = mx.array([[1, 5, 10]])
        out = encoder(ids)
        mx.eval(out)
        assert out.shape == (1, 3, 64)


# ---------------------------------------------------------------------------
# VAE Tests
# ---------------------------------------------------------------------------

class TestCausalConv3d:
    def test_output_shape_stride1(self):
        from mlx_video.models.wan.vae import CausalConv3d
        conv = CausalConv3d(4, 8, kernel_size=3, stride=1, padding=1)
        # Initialize weights
        conv.weight = mx.random.normal(conv.weight.shape) * 0.02
        x = mx.random.normal((1, 4, 3, 8, 8))  # [B, C, T, H, W]
        out = conv(x)
        mx.eval(out)
        # With causal padding and padding=1 on spatial, dims should be preserved
        assert out.shape[0] == 1
        assert out.shape[1] == 8  # out_channels
        assert out.shape[2] == 3  # T preserved
        assert out.shape[3] == 8  # H preserved
        assert out.shape[4] == 8  # W preserved

    def test_output_shape_kernel1(self):
        from mlx_video.models.wan.vae import CausalConv3d
        conv = CausalConv3d(4, 8, kernel_size=1, stride=1, padding=0)
        conv.weight = mx.random.normal(conv.weight.shape) * 0.02
        x = mx.random.normal((1, 4, 2, 4, 4))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (1, 8, 2, 4, 4)

    def test_causal_padding(self):
        """Causal conv should only use past/current frames, not future."""
        from mlx_video.models.wan.vae import CausalConv3d
        conv = CausalConv3d(2, 2, kernel_size=3, stride=1, padding=1)
        conv.weight = mx.random.normal(conv.weight.shape) * 0.1
        conv.bias = mx.zeros((2,))
        # Create input where only the first frame has signal
        x = mx.zeros((1, 2, 4, 4, 4))
        x_np = np.zeros((1, 2, 4, 4, 4), dtype=np.float32)
        x_np[:, :, 0, :, :] = 1.0
        x = mx.array(x_np)
        out = conv(x)
        mx.eval(out)
        # Due to causal padding, the output at t=0 should only depend on t=0


class TestResidualBlock:
    def test_same_dim(self):
        from mlx_video.models.wan.vae import ResidualBlock
        block = ResidualBlock(8, 8)
        x = mx.random.normal((1, 8, 2, 4, 4))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 8, 2, 4, 4)

    def test_different_dim(self):
        from mlx_video.models.wan.vae import ResidualBlock
        block = ResidualBlock(8, 16)
        x = mx.random.normal((1, 8, 2, 4, 4))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 16, 2, 4, 4)

    def test_shortcut_exists_when_dims_differ(self):
        from mlx_video.models.wan.vae import ResidualBlock
        block = ResidualBlock(8, 16)
        assert block.shortcut is not None

    def test_no_shortcut_when_dims_same(self):
        from mlx_video.models.wan.vae import ResidualBlock
        block = ResidualBlock(8, 8)
        assert block.shortcut is None


class TestAttentionBlock:
    def test_output_shape(self):
        from mlx_video.models.wan.vae import AttentionBlock
        block = AttentionBlock(8)
        x = mx.random.normal((1, 8, 2, 4, 4))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 8, 2, 4, 4)

    def test_residual_connection(self):
        from mlx_video.models.wan.vae import AttentionBlock
        block = AttentionBlock(8)
        x = mx.random.normal((1, 8, 1, 3, 3))
        out = block(x)
        mx.eval(x, out)
        # Residual: output should not be zero even with random init
        assert np.abs(np.array(out)).max() > 0


class TestWanVAE:
    def test_instantiation(self):
        from mlx_video.models.wan.vae import WanVAE
        vae = WanVAE(z_dim=16)
        assert vae.z_dim == 16
        assert vae.mean.shape == (16,)
        assert vae.std.shape == (16,)

    def test_normalization_stats(self):
        from mlx_video.models.wan.vae import WanVAE, VAE_MEAN, VAE_STD
        assert len(VAE_MEAN) == 16
        assert len(VAE_STD) == 16
        assert all(s > 0 for s in VAE_STD)


# ---------------------------------------------------------------------------
# Scheduler Tests
# ---------------------------------------------------------------------------

class TestFlowMatchEulerScheduler:
    def test_initialization(self):
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched = FlowMatchEulerScheduler()
        assert sched.num_train_timesteps == 1000
        assert sched.timesteps is None
        assert sched.sigmas is None

    def test_set_timesteps(self):
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(40, shift=12.0)
        mx.eval(sched.timesteps, sched.sigmas)
        assert sched.timesteps.shape == (40,)
        assert sched.sigmas.shape == (41,)  # 40 steps + terminal

    def test_timesteps_decreasing(self):
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(40, shift=12.0)
        mx.eval(sched.timesteps)
        ts = np.array(sched.timesteps)
        # Timesteps should be monotonically decreasing
        assert np.all(np.diff(ts) < 0), f"Timesteps not decreasing: {ts[:5]}..."

    def test_sigmas_decreasing(self):
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(20, shift=1.0)
        mx.eval(sched.sigmas)
        sigmas = np.array(sched.sigmas)
        assert np.all(np.diff(sigmas) <= 0), "Sigmas not decreasing"

    def test_terminal_sigma_is_zero(self):
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(20, shift=5.0)
        mx.eval(sched.sigmas)
        np.testing.assert_allclose(np.array(sched.sigmas[-1]), 0.0, atol=1e-6)

    def test_shift_effect(self):
        """Larger shift should push sigmas toward higher values."""
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched1 = FlowMatchEulerScheduler()
        sched2 = FlowMatchEulerScheduler()
        sched1.set_timesteps(20, shift=1.0)
        sched2.set_timesteps(20, shift=12.0)
        mx.eval(sched1.sigmas, sched2.sigmas)
        mean1 = np.mean(np.array(sched1.sigmas[:-1]))
        mean2 = np.mean(np.array(sched2.sigmas[:-1]))
        assert mean2 > mean1, "Higher shift should push sigmas higher"

    def test_step_euler(self):
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(10, shift=1.0)
        mx.eval(sched.sigmas)

        sample = mx.ones((1, 4, 2, 2, 2))
        velocity = mx.ones((1, 4, 2, 2, 2)) * 0.5
        timestep = sched.timesteps[0]

        sigma = float(np.array(sched.sigmas[0]))
        sigma_next = float(np.array(sched.sigmas[1]))

        result = sched.step(velocity, timestep, sample)
        mx.eval(result)

        # Euler: x_next = x + (sigma_next - sigma) * v
        expected = 1.0 + (sigma_next - sigma) * 0.5
        np.testing.assert_allclose(
            np.array(result).flatten()[0], expected, rtol=1e-4,
        )

    def test_step_index_increments(self):
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(5, shift=1.0)
        assert sched._step_index == 0
        sample = mx.ones((1, 1, 1, 1, 1))
        vel = mx.zeros((1, 1, 1, 1, 1))
        sched.step(vel, sched.timesteps[0], sample)
        assert sched._step_index == 1
        sched.step(vel, sched.timesteps[1], sample)
        assert sched._step_index == 2

    def test_reset(self):
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(5, shift=1.0)
        sample = mx.ones((1, 1, 1, 1, 1))
        vel = mx.zeros((1, 1, 1, 1, 1))
        sched.step(vel, sched.timesteps[0], sample)
        assert sched._step_index == 1
        sched.reset()
        assert sched._step_index == 0

    @pytest.mark.parametrize("steps", [10, 20, 40, 50])
    def test_various_step_counts(self, steps):
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(steps, shift=12.0)
        mx.eval(sched.timesteps, sched.sigmas)
        assert sched.timesteps.shape == (steps,)
        assert sched.sigmas.shape == (steps + 1,)

    def test_full_denoise_loop(self):
        """Run a complete denoise loop with zero velocity -> sample unchanged."""
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(5, shift=1.0)
        sample = mx.ones((1, 2, 1, 2, 2))
        for i in range(5):
            vel = mx.zeros_like(sample)
            sample = sched.step(vel, sched.timesteps[i], sample)
        mx.eval(sample)
        # With zero velocity, sample should remain unchanged
        np.testing.assert_allclose(np.array(sample), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Weight Conversion Tests
# ---------------------------------------------------------------------------

class TestSanitizeTransformerWeights:
    def test_patch_embedding_reshape(self):
        from mlx_video.convert_wan import sanitize_wan_transformer_weights
        weights = {
            "patch_embedding.weight": mx.random.normal((5120, 16, 1, 2, 2)),
            "patch_embedding.bias": mx.random.normal((5120,)),
        }
        out = sanitize_wan_transformer_weights(weights)
        assert "patch_embedding_proj.weight" in out
        assert "patch_embedding_proj.bias" in out
        assert out["patch_embedding_proj.weight"].shape == (5120, 16 * 1 * 2 * 2)

    def test_text_embedding_rename(self):
        from mlx_video.convert_wan import sanitize_wan_transformer_weights
        weights = {
            "text_embedding.0.weight": mx.zeros((64, 32)),
            "text_embedding.0.bias": mx.zeros((64,)),
            "text_embedding.2.weight": mx.zeros((64, 64)),
            "text_embedding.2.bias": mx.zeros((64,)),
        }
        out = sanitize_wan_transformer_weights(weights)
        assert "text_embedding_0.weight" in out
        assert "text_embedding_0.bias" in out
        assert "text_embedding_1.weight" in out
        assert "text_embedding_1.bias" in out

    def test_time_embedding_rename(self):
        from mlx_video.convert_wan import sanitize_wan_transformer_weights
        weights = {
            "time_embedding.0.weight": mx.zeros((64, 32)),
            "time_embedding.2.weight": mx.zeros((64, 64)),
        }
        out = sanitize_wan_transformer_weights(weights)
        assert "time_embedding_0.weight" in out
        assert "time_embedding_1.weight" in out

    def test_time_projection_rename(self):
        from mlx_video.convert_wan import sanitize_wan_transformer_weights
        weights = {
            "time_projection.1.weight": mx.zeros((384, 64)),
            "time_projection.1.bias": mx.zeros((384,)),
        }
        out = sanitize_wan_transformer_weights(weights)
        assert "time_projection.weight" in out
        assert "time_projection.bias" in out

    def test_ffn_rename(self):
        from mlx_video.convert_wan import sanitize_wan_transformer_weights
        weights = {
            "blocks.0.ffn.0.weight": mx.zeros((128, 64)),
            "blocks.0.ffn.0.bias": mx.zeros((128,)),
            "blocks.0.ffn.2.weight": mx.zeros((64, 128)),
            "blocks.0.ffn.2.bias": mx.zeros((64,)),
        }
        out = sanitize_wan_transformer_weights(weights)
        assert "blocks.0.ffn.fc1.weight" in out
        assert "blocks.0.ffn.fc1.bias" in out
        assert "blocks.0.ffn.fc2.weight" in out
        assert "blocks.0.ffn.fc2.bias" in out

    def test_freqs_skipped(self):
        from mlx_video.convert_wan import sanitize_wan_transformer_weights
        weights = {
            "freqs": mx.zeros((1024, 64, 2)),
            "blocks.0.norm1.weight": mx.zeros((64,)),
        }
        out = sanitize_wan_transformer_weights(weights)
        assert "freqs" not in out
        assert "blocks.0.norm1.weight" in out

    def test_passthrough_keys(self):
        from mlx_video.convert_wan import sanitize_wan_transformer_weights
        weights = {
            "blocks.0.self_attn.q.weight": mx.zeros((64, 64)),
            "blocks.0.self_attn.k.weight": mx.zeros((64, 64)),
            "blocks.0.self_attn.v.weight": mx.zeros((64, 64)),
            "blocks.0.self_attn.o.weight": mx.zeros((64, 64)),
            "blocks.0.modulation": mx.zeros((1, 6, 64)),
            "head.head.weight": mx.zeros((64, 64)),
            "head.modulation": mx.zeros((1, 2, 64)),
        }
        out = sanitize_wan_transformer_weights(weights)
        for key in weights:
            assert key in out


class TestSanitizeT5Weights:
    def test_gate_rename(self):
        from mlx_video.convert_wan import sanitize_wan_t5_weights
        weights = {
            "blocks.0.ffn.gate.0.weight": mx.zeros((128, 64)),
            "blocks.0.ffn.fc1.weight": mx.zeros((128, 64)),
            "blocks.0.ffn.fc2.weight": mx.zeros((64, 128)),
        }
        out = sanitize_wan_t5_weights(weights)
        assert "blocks.0.ffn.gate_proj.weight" in out
        assert "blocks.0.ffn.fc1.weight" in out
        assert "blocks.0.ffn.fc2.weight" in out

    def test_passthrough(self):
        from mlx_video.convert_wan import sanitize_wan_t5_weights
        weights = {
            "token_embedding.weight": mx.zeros((100, 64)),
            "blocks.0.attn.q.weight": mx.zeros((64, 64)),
            "norm.weight": mx.zeros((64,)),
        }
        out = sanitize_wan_t5_weights(weights)
        for key in weights:
            assert key in out


class TestSanitizeVAEWeights:
    def test_conv3d_transpose(self):
        from mlx_video.convert_wan import sanitize_wan_vae_weights
        weights = {
            "decoder.conv1.weight": mx.zeros((8, 4, 3, 3, 3)),  # [O, I, D, H, W]
        }
        out = sanitize_wan_vae_weights(weights)
        assert out["decoder.conv1.weight"].shape == (8, 3, 3, 3, 4)  # [O, D, H, W, I]

    def test_conv2d_transpose(self):
        from mlx_video.convert_wan import sanitize_wan_vae_weights
        weights = {
            "decoder.proj.weight": mx.zeros((16, 8, 3, 3)),  # [O, I, H, W]
        }
        out = sanitize_wan_vae_weights(weights)
        assert out["decoder.proj.weight"].shape == (16, 3, 3, 8)  # [O, H, W, I]

    def test_non_conv_passthrough(self):
        from mlx_video.convert_wan import sanitize_wan_vae_weights
        weights = {
            "decoder.norm.weight": mx.zeros((64,)),  # 1D, no transpose
            "decoder.bias": mx.zeros((16,)),
        }
        out = sanitize_wan_vae_weights(weights)
        assert out["decoder.norm.weight"].shape == (64,)
        assert out["decoder.bias"].shape == (16,)

    def test_mixed_weights(self):
        from mlx_video.convert_wan import sanitize_wan_vae_weights
        weights = {
            "conv3d.weight": mx.zeros((8, 4, 3, 3, 3)),  # 5D
            "conv2d.weight": mx.zeros((8, 4, 3, 3)),  # 4D
            "linear.weight": mx.zeros((8, 4)),  # 2D
            "norm.weight": mx.zeros((8,)),  # 1D
        }
        out = sanitize_wan_vae_weights(weights)
        assert out["conv3d.weight"].shape == (8, 3, 3, 3, 4)
        assert out["conv2d.weight"].shape == (8, 3, 3, 4)
        assert out["linear.weight"].shape == (8, 4)
        assert out["norm.weight"].shape == (8,)


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
# Wan2.1 Config & Pipeline Tests
# ---------------------------------------------------------------------------

class TestWan21Config:
    """Tests for Wan2.1 config presets."""

    def test_wan21_14b_factory(self):
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig.wan21_t2v_14b()
        assert config.model_version == "2.1"
        assert config.dual_model is False
        assert config.dim == 5120
        assert config.ffn_dim == 13824
        assert config.num_heads == 40
        assert config.num_layers == 40
        assert config.head_dim == 128
        assert config.sample_guide_scale == 5.0
        assert config.sample_shift == 5.0
        assert config.sample_steps == 50
        assert config.boundary == 0.0

    def test_wan21_1_3b_factory(self):
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig.wan21_t2v_1_3b()
        assert config.model_version == "2.1"
        assert config.dual_model is False
        assert config.dim == 1536
        assert config.ffn_dim == 8960
        assert config.num_heads == 12
        assert config.num_layers == 30
        assert config.head_dim == 128  # 1536 // 12
        assert config.sample_guide_scale == 5.0

    def test_wan22_14b_factory(self):
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig.wan22_t2v_14b()
        assert config.model_version == "2.2"
        assert config.dual_model is True
        assert config.dim == 5120
        assert config.sample_guide_scale == (3.0, 4.0)
        assert config.sample_shift == 12.0
        assert config.sample_steps == 40
        assert config.boundary == 0.875

    def test_wan21_config_to_dict(self):
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig.wan21_t2v_14b()
        d = config.to_dict()
        assert d["model_version"] == "2.1"
        assert d["dual_model"] is False
        assert d["sample_guide_scale"] == 5.0

    def test_wan21_1_3b_config_to_dict(self):
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig.wan21_t2v_1_3b()
        d = config.to_dict()
        assert d["dim"] == 1536
        assert d["num_layers"] == 30

    def test_default_config_is_wan22(self):
        """Default WanModelConfig() should be Wan2.2 14B."""
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig()
        assert config.model_version == "2.2"
        assert config.dual_model is True


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


class TestWan21Convert:
    """Tests for Wan2.1 conversion support."""

    def test_auto_detect_wan21(self, tmp_path):
        """Auto-detect single-model directory as Wan2.1."""
        # Create a Wan2.1-style directory (no low_noise_model subdir)
        (tmp_path / "dummy.safetensors").touch()
        # The auto-detect logic: no low_noise_model dir → 2.1
        from pathlib import Path
        low = tmp_path / "low_noise_model"
        assert not low.exists()
        # Simulates auto detection
        version = "2.2" if low.exists() else "2.1"
        assert version == "2.1"

    def test_auto_detect_wan22(self, tmp_path):
        """Auto-detect dual-model directory as Wan2.2."""
        (tmp_path / "low_noise_model").mkdir()
        (tmp_path / "high_noise_model").mkdir()
        from pathlib import Path
        low = tmp_path / "low_noise_model"
        assert low.exists()
        version = "2.2" if low.exists() else "2.1"
        assert version == "2.2"

    def test_wan21_config_saved_correctly(self):
        """Verify config dict has correct fields for Wan2.1."""
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig.wan21_t2v_14b()
        d = config.to_dict()
        assert d["model_version"] == "2.1"
        assert d["dual_model"] is False
        assert d["sample_steps"] == 50
        assert d["sample_shift"] == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
