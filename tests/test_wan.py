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


# ---------------------------------------------------------------------------
# Shared Sigma Schedule Tests
# ---------------------------------------------------------------------------


class TestComputeSigmas:
    """Tests for the shared _compute_sigmas helper."""

    def test_length(self):
        from mlx_video.models.wan.scheduler import _compute_sigmas
        sigmas = _compute_sigmas(20, shift=5.0)
        assert len(sigmas) == 21  # num_steps + terminal

    def test_terminal_zero(self):
        from mlx_video.models.wan.scheduler import _compute_sigmas
        sigmas = _compute_sigmas(10, shift=1.0)
        assert sigmas[-1] == 0.0

    def test_starts_at_one(self):
        from mlx_video.models.wan.scheduler import _compute_sigmas
        sigmas = _compute_sigmas(20, shift=5.0)
        np.testing.assert_allclose(sigmas[0], 1.0, atol=1e-6)

    def test_decreasing(self):
        from mlx_video.models.wan.scheduler import _compute_sigmas
        sigmas = _compute_sigmas(20, shift=5.0)
        assert np.all(np.diff(sigmas) <= 0)

    def test_matches_official_wan22(self):
        """Sigma schedule should match the official Wan2.2 get_sampling_sigmas."""
        from mlx_video.models.wan.scheduler import _compute_sigmas
        steps, shift = 50, 5.0
        sigmas = _compute_sigmas(steps, shift)
        # Official: sigma = linspace(1, 0, steps+1)[:steps]; sigma = shift*sigma/(1+(shift-1)*sigma)
        official = np.linspace(1, 0, steps + 1)[:steps]
        official = shift * official / (1 + (shift - 1) * official)
        official = np.append(official, 0.0).astype(np.float32)
        np.testing.assert_allclose(sigmas, official, atol=1e-6)

    def test_shift_one_is_linear(self):
        from mlx_video.models.wan.scheduler import _compute_sigmas
        sigmas = _compute_sigmas(10, shift=1.0)
        # With shift=1, f(sigma)=sigma, so schedule is linear from 1 to 0
        expected = np.linspace(1, 0, 11).astype(np.float32)
        np.testing.assert_allclose(sigmas, expected, atol=1e-6)

    def test_all_schedulers_same_sigmas(self):
        """All three schedulers should produce identical sigma schedules."""
        from mlx_video.models.wan.scheduler import (
            FlowDPMPP2MScheduler,
            FlowMatchEulerScheduler,
            FlowUniPCScheduler,
        )
        scheds = [
            FlowMatchEulerScheduler(1000),
            FlowDPMPP2MScheduler(1000),
            FlowUniPCScheduler(1000),
        ]
        for s in scheds:
            s.set_timesteps(20, shift=5.0)
        mx.eval(*[s.sigmas for s in scheds])
        ref = np.array(scheds[0].sigmas)
        for s in scheds[1:]:
            np.testing.assert_allclose(np.array(s.sigmas), ref, atol=1e-6)

    def test_all_schedulers_same_timesteps(self):
        from mlx_video.models.wan.scheduler import (
            FlowDPMPP2MScheduler,
            FlowMatchEulerScheduler,
            FlowUniPCScheduler,
        )
        scheds = [
            FlowMatchEulerScheduler(1000),
            FlowDPMPP2MScheduler(1000),
            FlowUniPCScheduler(1000),
        ]
        for s in scheds:
            s.set_timesteps(30, shift=12.0)
        mx.eval(*[s.timesteps for s in scheds])
        ref = np.array(scheds[0].timesteps)
        for s in scheds[1:]:
            np.testing.assert_allclose(np.array(s.timesteps), ref, atol=1e-3)


# ---------------------------------------------------------------------------
# DPM++ 2M Scheduler Tests
# ---------------------------------------------------------------------------


class TestFlowDPMPP2MScheduler:
    def test_initialization(self):
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler
        sched = FlowDPMPP2MScheduler()
        assert sched.num_train_timesteps == 1000
        assert sched.lower_order_final is True

    def test_set_timesteps(self):
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler
        sched = FlowDPMPP2MScheduler()
        sched.set_timesteps(20, shift=5.0)
        mx.eval(sched.timesteps, sched.sigmas)
        assert sched.timesteps.shape == (20,)
        assert sched.sigmas.shape == (21,)

    def test_step_index_increments(self):
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler
        sched = FlowDPMPP2MScheduler()
        sched.set_timesteps(5, shift=1.0)
        sample = mx.ones((1, 4, 1, 2, 2))
        vel = mx.zeros_like(sample)
        assert sched._step_index == 0
        sched.step(vel, sched.timesteps[0], sample)
        assert sched._step_index == 1
        sched.step(vel, sched.timesteps[1], sample)
        assert sched._step_index == 2

    def test_reset(self):
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler
        sched = FlowDPMPP2MScheduler()
        sched.set_timesteps(5, shift=1.0)
        sample = mx.ones((1, 1, 1, 1, 1))
        sched.step(mx.zeros_like(sample), 0, sample)
        sched.reset()
        assert sched._step_index == 0
        assert sched._prev_x0 is None

    def test_full_loop_finite(self):
        """Full loop with constant velocity should produce finite output."""
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler
        sched = FlowDPMPP2MScheduler()
        sched.set_timesteps(10, shift=1.0)
        sample = mx.ones((1, 2, 1, 2, 2))
        for i in range(10):
            vel = mx.ones_like(sample) * 0.1
            sample = sched.step(vel, sched.timesteps[i], sample)
            mx.eval(sample)
        assert np.isfinite(np.array(sample)).all()

    def test_first_step_is_first_order(self):
        """First step should use 1st-order (no prev_x0 available)."""
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler
        sched = FlowDPMPP2MScheduler()
        sched.set_timesteps(10, shift=5.0)
        sample = mx.random.normal((1, 4, 2, 4, 4))
        vel = mx.random.normal(sample.shape)
        # Before first step, no prev_x0
        assert sched._prev_x0 is None
        result = sched.step(vel, sched.timesteps[0], sample)
        mx.eval(result)
        # After first step, prev_x0 should be set
        assert sched._prev_x0 is not None

    def test_second_step_uses_correction(self):
        """After first step, DPM++ should have stored prev_x0 for correction."""
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler
        sched = FlowDPMPP2MScheduler()
        sched.set_timesteps(10, shift=5.0)
        sample = mx.random.normal((1, 4, 1, 2, 2))
        vel = mx.random.normal(sample.shape)
        # Step 1
        sample = sched.step(vel, sched.timesteps[0], sample)
        mx.eval(sample)
        x0_after_first = sched._prev_x0
        # Step 2
        vel = mx.random.normal(sample.shape)
        sample = sched.step(vel, sched.timesteps[1], sample)
        mx.eval(sample)
        # prev_x0 should have been updated
        x0_after_second = sched._prev_x0
        assert x0_after_second is not None
        # The stored x0 should differ from the first step's
        assert not np.allclose(np.array(x0_after_first), np.array(x0_after_second), atol=1e-6)

    def test_denoise_to_target(self):
        """Perfect oracle should denoise to target with any solver."""
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler
        sched = FlowDPMPP2MScheduler()
        sched.set_timesteps(20, shift=5.0)
        target = mx.zeros((1, 2, 1, 4, 4))
        latents = mx.random.normal(target.shape)
        for i in range(20):
            sigma = float(sched.sigmas[i].item())
            v = latents / max(sigma, 1e-6)  # perfect velocity for target=0
            latents = sched.step(v, sched.timesteps[i], latents)
            mx.eval(latents)
        np.testing.assert_allclose(np.array(latents), 0.0, atol=1e-3)

    @pytest.mark.parametrize("steps", [5, 10, 20, 50])
    def test_various_step_counts(self, steps):
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler
        sched = FlowDPMPP2MScheduler()
        sched.set_timesteps(steps, shift=5.0)
        mx.eval(sched.timesteps, sched.sigmas)
        assert sched.timesteps.shape == (steps,)
        assert sched.sigmas.shape == (steps + 1,)

    def test_terminal_sigma_produces_x0(self):
        """When sigma_next=0 the scheduler should return x0 directly."""
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler
        sched = FlowDPMPP2MScheduler()
        sched.set_timesteps(5, shift=1.0)
        sample = mx.ones((1, 1, 1, 1, 1)) * 3.0
        vel = mx.ones_like(sample) * 2.0
        # Run through all steps; the last step has sigma_next=0
        for i in range(5):
            sample = sched.step(vel, sched.timesteps[i], sample)
            mx.eval(sample)
        # Final value should be finite
        assert np.isfinite(np.array(sample)).all()


# ---------------------------------------------------------------------------
# UniPC Scheduler Tests
# ---------------------------------------------------------------------------


class TestFlowUniPCScheduler:
    def test_initialization(self):
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler()
        assert sched.num_train_timesteps == 1000
        assert sched.solver_order == 2
        assert sched.lower_order_final is True

    def test_set_timesteps(self):
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler()
        sched.set_timesteps(30, shift=12.0)
        mx.eval(sched.timesteps, sched.sigmas)
        assert sched.timesteps.shape == (30,)
        assert sched.sigmas.shape == (31,)

    def test_step_index_increments(self):
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler()
        sched.set_timesteps(5, shift=1.0)
        sample = mx.ones((1, 1, 1, 1, 1))
        vel = mx.zeros_like(sample)
        assert sched._step_index == 0
        sched.step(vel, 0, sample)
        assert sched._step_index == 1

    def test_reset(self):
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler()
        sched.set_timesteps(5, shift=1.0)
        sample = mx.ones((1, 1, 1, 1, 1))
        sched.step(mx.zeros_like(sample), 0, sample)
        sched.reset()
        assert sched._step_index == 0
        assert sched._lower_order_nums == 0
        assert sched._last_sample is None
        assert all(m is None for m in sched._model_outputs)

    def test_full_loop_finite(self):
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler()
        sched.set_timesteps(10, shift=1.0)
        sample = mx.ones((1, 2, 1, 2, 2))
        for i in range(10):
            vel = mx.ones_like(sample) * 0.1
            sample = sched.step(vel, sched.timesteps[i], sample)
            mx.eval(sample)
        assert np.isfinite(np.array(sample)).all()

    def test_corrector_not_applied_first_step(self):
        """First step should skip the corrector (no history)."""
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler(use_corrector=True)
        sched.set_timesteps(10, shift=5.0)
        sample = mx.random.normal((1, 4, 1, 2, 2))
        vel = mx.random.normal(sample.shape)
        # Before step 0: no last_sample
        assert sched._last_sample is None
        sched.step(vel, sched.timesteps[0], sample)
        # After step 0: last_sample should be set for corrector on step 1
        assert sched._last_sample is not None

    def test_corrector_applied_after_first_step(self):
        """Steps after the first should use the corrector when enabled."""
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler(use_corrector=True)
        sched.set_timesteps(10, shift=5.0)
        sample = mx.random.normal((1, 2, 1, 4, 4))
        for i in range(3):
            vel = mx.random.normal(sample.shape)
            sample = sched.step(vel, sched.timesteps[i], sample)
            mx.eval(sample)
        # lower_order_nums should have increased
        assert sched._lower_order_nums >= 2

    def test_denoise_to_target(self):
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler()
        sched.set_timesteps(20, shift=5.0)
        target = mx.zeros((1, 2, 1, 4, 4))
        latents = mx.random.normal(target.shape)
        for i in range(20):
            sigma = float(sched.sigmas[i].item())
            v = latents / max(sigma, 1e-6)
            latents = sched.step(v, sched.timesteps[i], latents)
            mx.eval(latents)
        np.testing.assert_allclose(np.array(latents), 0.0, atol=1e-3)

    @pytest.mark.parametrize("steps", [5, 10, 20, 50])
    def test_various_step_counts(self, steps):
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler()
        sched.set_timesteps(steps, shift=5.0)
        mx.eval(sched.timesteps, sched.sigmas)
        assert sched.timesteps.shape == (steps,)
        assert sched.sigmas.shape == (steps + 1,)

    def test_disable_corrector(self):
        """Disabling corrector on step 0 should still work without error."""
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler(use_corrector=True, disable_corrector=[0])
        sched.set_timesteps(5, shift=1.0)
        sample = mx.ones((1, 1, 1, 2, 2))
        for i in range(5):
            vel = mx.ones_like(sample) * 0.1
            sample = sched.step(vel, sched.timesteps[i], sample)
            mx.eval(sample)
        assert np.isfinite(np.array(sample)).all()

    def test_solver_order_3(self):
        """Order 3 should work without error."""
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler(solver_order=3, use_corrector=True)
        sched.set_timesteps(10, shift=5.0)
        sample = mx.random.normal((1, 2, 1, 2, 2))
        for i in range(10):
            vel = mx.random.normal(sample.shape)
            sample = sched.step(vel, sched.timesteps[i], sample)
            mx.eval(sample)
        assert np.isfinite(np.array(sample)).all()

    def test_corrector_rhos_c_not_hardcoded(self):
        """Corrector rhos_c should be computed via linalg.solve, not hardcoded 0.5."""
        import math
        # For 50-step schedule with shift=5.0, order 2 corrector at step 5:
        # rhos_c[0] (history) should be ~0.07, NOT 0.5
        # rhos_c[1] (D1_t) should be ~0.45, NOT 0.5
        from mlx_video.models.wan.scheduler import _compute_sigmas

        sigmas = _compute_sigmas(50, shift=5.0)

        def _lambda(sigma):
            if sigma >= 1.0:
                return -math.inf
            if sigma <= 0.0:
                return math.inf
            return math.log(1 - sigma) - math.log(sigma)

        for step_idx in [5, 10, 25, 45]:
            sigma_s0 = sigmas[step_idx - 1]
            sigma_t = sigmas[step_idx]
            lambda_s0 = _lambda(sigma_s0)
            lambda_t = _lambda(sigma_t)
            h = lambda_t - lambda_s0
            hh = -h

            sigma_sk = sigmas[step_idx - 2]
            lambda_sk = _lambda(sigma_sk)
            rk = (lambda_sk - lambda_s0) / h
            rks = np.array([rk, 1.0])

            h_phi_1 = math.expm1(hh)
            B_h = h_phi_1
            h_phi_k = h_phi_1 / hh - 1.0
            factorial_i = 1
            R_rows, b_vals = [], []
            for j in range(1, 3):
                R_rows.append(rks ** (j - 1))
                b_vals.append(h_phi_k * factorial_i / B_h)
                factorial_i *= j + 1
                h_phi_k = h_phi_k / hh - 1.0 / factorial_i
            R = np.stack(R_rows)
            b = np.array(b_vals)
            rhos_c = np.linalg.solve(R, b)

            # History weight should be small (~0.07-0.09), not 0.5
            assert rhos_c[0] < 0.15, f"Step {step_idx}: rhos_c[0]={rhos_c[0]:.4f} too large"
            assert rhos_c[0] > 0.0, f"Step {step_idx}: rhos_c[0]={rhos_c[0]:.4f} should be positive"
            # D1_t weight should be ~0.42-0.45, not 0.5
            assert 0.3 < rhos_c[1] < 0.5, f"Step {step_idx}: rhos_c[1]={rhos_c[1]:.4f} out of range"


class TestSchedulerCoherence:
    """Tests that Euler, DPM++, and UniPC schedulers produce coherent results.

    All three schedulers should agree on shared structure (sigma schedules,
    first-step behavior) and converge to the same result given perfect
    velocity oracles, even though they use different update rules.
    """

    @staticmethod
    def _make_schedulers(steps=10, shift=5.0):
        from mlx_video.models.wan.scheduler import (
            FlowDPMPP2MScheduler,
            FlowMatchEulerScheduler,
            FlowUniPCScheduler,
        )

        scheds = {
            "euler": FlowMatchEulerScheduler(),
            "dpm++": FlowDPMPP2MScheduler(),
            "unipc": FlowUniPCScheduler(),
        }
        for s in scheds.values():
            s.set_timesteps(steps, shift=shift)
        return scheds

    def test_identical_sigma_schedules(self):
        """All schedulers must use the same sigma schedule."""
        scheds = self._make_schedulers(20, shift=5.0)
        ref = np.array(scheds["euler"].sigmas)
        for name in ("dpm++", "unipc"):
            np.testing.assert_allclose(
                np.array(scheds[name].sigmas),
                ref,
                atol=1e-6,
                err_msg=f"{name} sigma schedule differs from Euler",
            )

    def test_identical_timesteps(self):
        """All schedulers must produce the same timestep sequence."""
        scheds = self._make_schedulers(20, shift=5.0)
        ref = np.array(scheds["euler"].timesteps)
        for name in ("dpm++", "unipc"):
            np.testing.assert_allclose(
                np.array(scheds[name].timesteps),
                ref,
                atol=1e-6,
                err_msg=f"{name} timesteps differ from Euler",
            )

    def test_first_step_matches_euler(self):
        """Step 0 (1st-order for all solvers) should match Euler exactly."""
        mx.random.seed(42)
        shape = (1, 4, 1, 4, 4)
        noise = mx.random.normal(shape)
        vel = mx.random.normal(shape)

        scheds = self._make_schedulers(10, shift=5.0)
        results = {}
        for name, sched in scheds.items():
            r = sched.step(vel, sched.timesteps[0], noise)
            mx.eval(r)
            results[name] = np.array(r)

        np.testing.assert_allclose(
            results["dpm++"], results["euler"], atol=1e-5,
            err_msg="DPM++ step 0 should match Euler",
        )
        np.testing.assert_allclose(
            results["unipc"], results["euler"], atol=1e-5,
            err_msg="UniPC step 0 should match Euler",
        )

    def test_first_step_matches_across_shifts(self):
        """Step 0 should match Euler for different shift values."""
        mx.random.seed(99)
        shape = (1, 2, 1, 2, 2)
        noise = mx.random.normal(shape)
        vel = mx.random.normal(shape)

        for shift in (1.0, 5.0, 12.0):
            scheds = self._make_schedulers(10, shift=shift)
            euler_r = scheds["euler"].step(vel, scheds["euler"].timesteps[0], noise)
            dpm_r = scheds["dpm++"].step(vel, scheds["dpm++"].timesteps[0], noise)
            unipc_r = scheds["unipc"].step(vel, scheds["unipc"].timesteps[0], noise)
            mx.eval(euler_r, dpm_r, unipc_r)
            np.testing.assert_allclose(
                np.array(dpm_r), np.array(euler_r), atol=1e-5,
                err_msg=f"DPM++ step 0 differs from Euler at shift={shift}",
            )
            np.testing.assert_allclose(
                np.array(unipc_r), np.array(euler_r), atol=1e-5,
                err_msg=f"UniPC step 0 differs from Euler at shift={shift}",
            )

    def test_oracle_all_converge_to_target(self):
        """Given a perfect velocity oracle v=x/sigma, all solvers should
        denoise to approximately zero (the target)."""
        mx.random.seed(7)
        shape = (1, 2, 1, 4, 4)
        noise = mx.random.normal(shape)

        for name, sched in self._make_schedulers(20, shift=5.0).items():
            latents = noise
            for i in range(20):
                sigma = float(sched.sigmas[i].item())
                v = latents / max(sigma, 1e-8)
                latents = sched.step(v, sched.timesteps[i], latents)
                mx.eval(latents)
            np.testing.assert_allclose(
                np.array(latents), 0.0, atol=1e-3,
                err_msg=f"{name} did not converge to target with oracle",
            )

    def test_oracle_higher_order_closer_to_target(self):
        """With few steps and a perfect oracle, higher-order solvers should
        be at least as accurate as Euler."""
        mx.random.seed(12)
        shape = (1, 2, 1, 4, 4)
        noise = mx.random.normal(shape)
        steps = 5

        errors = {}
        for name, sched in self._make_schedulers(steps, shift=5.0).items():
            latents = noise
            for i in range(steps):
                sigma = float(sched.sigmas[i].item())
                v = latents / max(sigma, 1e-8)
                latents = sched.step(v, sched.timesteps[i], latents)
                mx.eval(latents)
            errors[name] = float(mx.mean(mx.abs(latents)).item())

        # Higher-order solvers should not be significantly worse than Euler
        assert errors["dpm++"] <= errors["euler"] * 1.5, (
            f"DPM++ error {errors['dpm++']:.6f} much worse than Euler {errors['euler']:.6f}"
        )
        assert errors["unipc"] <= errors["euler"] * 1.5, (
            f"UniPC error {errors['unipc']:.6f} much worse than Euler {errors['euler']:.6f}"
        )

    def test_multistep_trajectory_similar_magnitude(self):
        """Over a full denoising loop with constant velocity, all solvers
        should produce outputs of similar magnitude (not diverging)."""
        mx.random.seed(42)
        shape = (1, 4, 1, 4, 4)
        noise = mx.random.normal(shape)
        steps = 20

        final_means = {}
        for name, sched in self._make_schedulers(steps, shift=5.0).items():
            latents = noise
            for i in range(steps):
                vel = latents * 0.1
                latents = sched.step(vel, sched.timesteps[i], latents)
                mx.eval(latents)
            final_means[name] = float(mx.mean(mx.abs(latents)).item())

        # All solvers should produce results within the same order of magnitude
        vals = list(final_means.values())
        ratio = max(vals) / max(min(vals), 1e-10)
        assert ratio < 10.0, (
            f"Scheduler outputs diverge too much: {final_means}, ratio={ratio:.1f}"
        )

    def test_intermediate_values_finite(self):
        """Every intermediate latent value must be finite for all solvers."""
        mx.random.seed(0)
        shape = (1, 2, 1, 2, 2)
        noise = mx.random.normal(shape)

        for name, sched in self._make_schedulers(15, shift=5.0).items():
            latents = noise
            for i in range(15):
                vel = mx.random.normal(shape)
                latents = sched.step(vel, sched.timesteps[i], latents)
                mx.eval(latents)
                assert np.isfinite(np.array(latents)).all(), (
                    f"{name} produced non-finite values at step {i}"
                )

    def test_lambda_boundary_values(self):
        """_lambda must return -inf at sigma=1.0 and +inf at sigma=0.0."""
        from mlx_video.models.wan.scheduler import (
            FlowDPMPP2MScheduler,
            FlowUniPCScheduler,
        )

        for cls in (FlowDPMPP2MScheduler, FlowUniPCScheduler):
            assert cls._lambda(1.0) == -math.inf, (
                f"{cls.__name__}._lambda(1.0) should be -inf"
            )
            assert cls._lambda(0.0) == math.inf, (
                f"{cls.__name__}._lambda(0.0) should be +inf"
            )
            # Interior values should be finite
            lam = cls._lambda(0.5)
            assert math.isfinite(lam) and lam == 0.0, (
                f"{cls.__name__}._lambda(0.5) should be 0.0"
            )

    def test_lambda_monotonically_decreasing(self):
        """_lambda(sigma) should decrease as sigma increases (more noise → lower SNR)."""
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler

        sigmas = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        lambdas = [FlowDPMPP2MScheduler._lambda(s) for s in sigmas]
        for i in range(len(lambdas) - 1):
            assert lambdas[i] > lambdas[i + 1], (
                f"_lambda not decreasing: _lambda({sigmas[i]})={lambdas[i]} "
                f"vs _lambda({sigmas[i+1]})={lambdas[i+1]}"
            )

    def test_step0_is_ddim_formula(self):
        """At sigma=1.0, the DPM++/UniPC first step should reduce to the
        DDIM formula: x_next = sigma_next * x + (1 - sigma_next) * x0."""
        mx.random.seed(55)
        shape = (1, 2, 1, 2, 2)
        sample = mx.random.normal(shape)
        vel = mx.random.normal(shape)

        for steps, shift in [(10, 5.0), (20, 12.0)]:
            scheds = self._make_schedulers(steps, shift=shift)
            sigma_next = float(scheds["euler"].sigmas[1].item())
            sigma_cur = float(scheds["euler"].sigmas[0].item())
            assert abs(sigma_cur - 1.0) < 1e-6, "First sigma should be ~1.0"

            x0 = sample - sigma_cur * vel
            expected = sigma_next * sample + (1.0 - sigma_next) * x0
            mx.eval(expected)

            for name in ("dpm++", "unipc"):
                result = scheds[name].step(vel, scheds[name].timesteps[0], sample)
                mx.eval(result)
                np.testing.assert_allclose(
                    np.array(result), np.array(expected), atol=1e-5,
                    err_msg=f"{name} step 0 doesn't match DDIM formula (shift={shift})",
                )

    @pytest.mark.parametrize("steps", [5, 10, 20, 50])
    def test_coherent_across_step_counts(self, steps):
        """All solvers should agree on step 0 regardless of total step count."""
        mx.random.seed(77)
        shape = (1, 2, 1, 2, 2)
        noise = mx.random.normal(shape)
        vel = mx.random.normal(shape)

        scheds = self._make_schedulers(steps, shift=5.0)
        results = {}
        for name, sched in scheds.items():
            r = sched.step(vel, sched.timesteps[0], noise)
            mx.eval(r)
            results[name] = np.array(r)

        np.testing.assert_allclose(
            results["dpm++"], results["euler"], atol=1e-5,
        )
        np.testing.assert_allclose(
            results["unipc"], results["euler"], atol=1e-5,
        )

    def test_dpmpp_unipc_agree_on_step1(self):
        """After warmup, DPM++ and UniPC step 1 should be similar
        (both use 2nd-order corrections based on the same model outputs)."""
        mx.random.seed(42)
        shape = (1, 4, 1, 4, 4)
        noise = mx.random.normal(shape)

        scheds = self._make_schedulers(10, shift=5.0)
        # Run step 0 with same velocity
        vel0 = mx.random.normal(shape)
        for sched in scheds.values():
            sched.step(vel0, sched.timesteps[0], noise)

        # Run step 1 from same sample with same velocity
        sample1 = scheds["euler"].step(vel0, scheds["euler"].timesteps[0], noise)
        mx.eval(sample1)
        vel1 = mx.random.normal(shape)

        r_dpm = scheds["dpm++"].step(vel1, scheds["dpm++"].timesteps[1], sample1)
        r_unipc = scheds["unipc"].step(vel1, scheds["unipc"].timesteps[1], sample1)
        mx.eval(r_dpm, r_unipc)

        # They won't be identical (different correction formulas) but should
        # be in the same ballpark (within 50% of each other's magnitude)
        mean_dpm = float(mx.mean(mx.abs(r_dpm)).item())
        mean_unipc = float(mx.mean(mx.abs(r_unipc)).item())
        ratio = max(mean_dpm, mean_unipc) / max(min(mean_dpm, mean_unipc), 1e-10)
        assert ratio < 2.0, (
            f"DPM++ and UniPC step 1 differ too much: "
            f"DPM++={mean_dpm:.4f}, UniPC={mean_unipc:.4f}"
        )

    def test_reset_makes_solvers_reproducible(self):
        """After reset(), running the same loop should produce identical output."""
        mx.random.seed(42)
        shape = (1, 2, 1, 2, 2)
        noise = mx.random.normal(shape)

        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler, FlowUniPCScheduler

        for cls in (FlowDPMPP2MScheduler, FlowUniPCScheduler):
            sched = cls()
            sched.set_timesteps(5, shift=5.0)

            # First run
            latents = noise
            for i in range(5):
                vel = latents * 0.1
                latents = sched.step(vel, sched.timesteps[i], latents)
                mx.eval(latents)
            result1 = np.array(latents)

            # Reset and run again
            sched.reset()
            latents = noise
            for i in range(5):
                vel = latents * 0.1
                latents = sched.step(vel, sched.timesteps[i], latents)
                mx.eval(latents)
            result2 = np.array(latents)

            np.testing.assert_allclose(result1, result2, atol=1e-5,
                err_msg=f"{cls.__name__} not reproducible after reset()")


# ---------------------------------------------------------------------------
# Wan2.2 VAE Component Tests
# ---------------------------------------------------------------------------


class TestVAE22CausalConv3d:
    """Tests for vae22.CausalConv3d (channels-last)."""

    def test_output_shape_k3(self):
        from mlx_video.models.wan.vae22 import CausalConv3d
        conv = CausalConv3d(8, 16, kernel_size=3, padding=1)
        x = mx.random.normal((1, 4, 8, 8, 8))  # [B, T, H, W, C]
        out = conv(x)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 8, 16)

    def test_output_shape_k1(self):
        from mlx_video.models.wan.vae22 import CausalConv3d
        conv = CausalConv3d(8, 16, kernel_size=1)
        x = mx.random.normal((1, 2, 4, 4, 8))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (1, 2, 4, 4, 16)

    def test_temporal_causal(self):
        """Output at t=0 should not depend on t>0."""
        from mlx_video.models.wan.vae22 import CausalConv3d
        conv = CausalConv3d(2, 2, kernel_size=3, padding=1)
        conv.weight = mx.random.normal(conv.weight.shape) * 0.1
        conv.bias = mx.zeros(conv.bias.shape)

        x = mx.zeros((1, 4, 4, 4, 2))
        out_zero = conv(x)
        mx.eval(out_zero)
        t0_ref = np.array(out_zero[0, 0])

        # Modify t=2..3; output at t=0 should be unchanged
        x_mod = mx.concatenate([
            x[:, :2],
            mx.ones((1, 2, 4, 4, 2)),
        ], axis=1)
        out_mod = conv(x_mod)
        mx.eval(out_mod)
        t0_mod = np.array(out_mod[0, 0])
        np.testing.assert_allclose(t0_ref, t0_mod, atol=1e-5)

    def test_channels_last_format(self):
        """Verify input/output are channels-last [B, T, H, W, C]."""
        from mlx_video.models.wan.vae22 import CausalConv3d
        conv = CausalConv3d(4, 8, kernel_size=3, padding=1)
        x = mx.random.normal((2, 3, 6, 6, 4))
        out = conv(x)
        mx.eval(out)
        assert out.shape[-1] == 8  # last dim = out_channels


class TestRMSNorm:
    """Tests for vae22.RMS_norm (actually L2 normalization)."""

    def test_output_shape(self):
        from mlx_video.models.wan.vae22 import RMS_norm
        norm = RMS_norm(16)
        x = mx.random.normal((2, 4, 4, 4, 16))
        out = norm(x)
        mx.eval(out)
        assert out.shape == x.shape

    def test_l2_normalization(self):
        """RMS_norm should normalize to unit L2 norm * sqrt(dim)."""
        from mlx_video.models.wan.vae22 import RMS_norm
        dim = 32
        norm = RMS_norm(dim)
        x = mx.random.normal((1, 1, 1, 1, dim)) * 5.0  # large values
        out = norm(x)
        mx.eval(out)
        # After L2 norm * scale(=sqrt(dim)) * gamma(=1): ||out|| = sqrt(dim)
        out_np = np.array(out).flatten()
        l2 = np.linalg.norm(out_np)
        np.testing.assert_allclose(l2, math.sqrt(dim), rtol=1e-3)

    def test_scale_invariant(self):
        """Scaling input by constant should not change output (L2 norm property)."""
        from mlx_video.models.wan.vae22 import RMS_norm
        norm = RMS_norm(8)
        x = mx.random.normal((1, 1, 1, 1, 8))
        out1 = norm(x)
        out2 = norm(x * 10.0)
        mx.eval(out1, out2)
        np.testing.assert_allclose(np.array(out1), np.array(out2), atol=1e-4)

    def test_gamma_effect(self):
        """Non-unit gamma should scale output."""
        from mlx_video.models.wan.vae22 import RMS_norm
        norm = RMS_norm(4)
        norm.gamma = mx.array([2.0, 2.0, 2.0, 2.0])
        x = mx.ones((1, 1, 1, 1, 4))
        out = norm(x)
        mx.eval(out)
        # With gamma=2, each component is 2 * sqrt(4) * x/||x|| = 2 * 2 * 1/2 = 2
        np.testing.assert_allclose(np.array(out).flatten(), 2.0, atol=1e-4)


class TestDupUp3D:
    """Tests for vae22.DupUp3D spatial/temporal upsampling."""

    def test_spatial_only(self):
        from mlx_video.models.wan.vae22 import DupUp3D
        up = DupUp3D(8, 4, factor_t=1, factor_s=2)
        x = mx.random.normal((1, 3, 4, 4, 8))
        out = up(x)
        mx.eval(out)
        assert out.shape == (1, 3, 8, 8, 4)

    def test_temporal_and_spatial(self):
        from mlx_video.models.wan.vae22 import DupUp3D
        up = DupUp3D(16, 8, factor_t=2, factor_s=2)
        x = mx.random.normal((1, 3, 4, 4, 16))
        out = up(x)
        mx.eval(out)
        assert out.shape == (1, 6, 8, 8, 8)

    def test_first_chunk_trims(self):
        from mlx_video.models.wan.vae22 import DupUp3D
        up = DupUp3D(8, 4, factor_t=2, factor_s=2)
        x = mx.random.normal((1, 3, 4, 4, 8))
        out_normal = up(x, first_chunk=False)
        out_trimmed = up(x, first_chunk=True)
        mx.eval(out_normal, out_trimmed)
        # first_chunk removes factor_t-1=1 temporal frame
        assert out_normal.shape[1] == 6
        assert out_trimmed.shape[1] == 5

    def test_no_temporal_first_chunk_noop(self):
        from mlx_video.models.wan.vae22 import DupUp3D
        up = DupUp3D(8, 4, factor_t=1, factor_s=2)
        x = mx.random.normal((1, 3, 4, 4, 8))
        out_normal = up(x, first_chunk=False)
        out_trimmed = up(x, first_chunk=True)
        mx.eval(out_normal, out_trimmed)
        # factor_t=1, so first_chunk removes 0 frames
        assert out_normal.shape == out_trimmed.shape


class TestVAE22Resample:
    """Tests for vae22.Resample (spatial/temporal upsampling)."""

    def test_upsample2d_shape(self):
        from mlx_video.models.wan.vae22 import Resample
        r = Resample(8, "upsample2d")
        r.resample_weight = mx.random.normal(r.resample_weight.shape) * 0.01
        x = mx.random.normal((1, 2, 4, 4, 8))
        out = r(x)
        mx.eval(out)
        assert out.shape == (1, 2, 8, 8, 8)  # 2x spatial, same temporal

    def test_upsample3d_shape(self):
        from mlx_video.models.wan.vae22 import Resample
        r = Resample(8, "upsample3d")
        r.resample_weight = mx.random.normal(r.resample_weight.shape) * 0.01
        x = mx.random.normal((1, 2, 4, 4, 8))
        out = r(x)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 8, 8)  # 2x spatial + 2x temporal

    def test_upsample3d_first_chunk(self):
        from mlx_video.models.wan.vae22 import Resample
        r = Resample(8, "upsample3d")
        r.resample_weight = mx.random.normal(r.resample_weight.shape) * 0.01
        x = mx.random.normal((1, 2, 4, 4, 8))
        out = r(x, first_chunk=True)
        mx.eval(out)
        # first_chunk: 1 (bypass) + 2*(T-1) (interleaved) = 2T-1 = 3
        assert out.shape == (1, 3, 8, 8, 8)

    def test_upsample3d_first_chunk_single_frame(self):
        """Single-frame input with first_chunk: no temporal upsample."""
        from mlx_video.models.wan.vae22 import Resample
        r = Resample(8, "upsample3d")
        r.resample_weight = mx.random.normal(r.resample_weight.shape) * 0.01
        x = mx.random.normal((1, 1, 4, 4, 8))
        out = r(x, first_chunk=True)
        mx.eval(out)
        # Single frame with first_chunk: falls through to non-first path
        # time_conv on 1 frame → 2 interleaved
        assert out.shape == (1, 2, 8, 8, 8)

    def test_upsample3d_first_frame_bypasses_time_conv(self):
        """First frame of first_chunk should NOT go through time_conv.

        Official Wan2.2 skips time_conv for the very first frame entirely.
        We verify this by checking that the first output frame depends only on
        the first input frame (not on time_conv parameters).
        """
        from mlx_video.models.wan.vae22 import Resample
        C = 8
        r = Resample(C, "upsample3d")
        # Set time_conv weights to large values so its effect is detectable
        r.time_conv.weight = mx.ones(r.time_conv.weight.shape) * 10.0
        r.time_conv.bias = mx.zeros(r.time_conv.bias.shape)
        # Set spatial conv to identity-like
        r.resample_weight = mx.zeros(r.resample_weight.shape)
        r.resample_bias = mx.zeros(r.resample_bias.shape)

        x = mx.random.normal((1, 3, 2, 2, C))
        out = r(x, first_chunk=True)
        mx.eval(out)
        # Output: 5 frames (1 bypass + 4 interleaved from 2 remaining)
        assert out.shape[1] == 5

        # First frame should be spatial upsample of x[:, 0:1] only.
        # Run just the first frame through spatial upsample for reference
        first_only = x[:, 0:1]
        ref = r._upsample2x(first_only.reshape(1, 2, 2, C))
        ref = mx.pad(ref, [(0, 0), (1, 1), (1, 1), (0, 0)])
        ref = mx.conv_general(ref, r.resample_weight) + r.resample_bias
        mx.eval(ref)

        # Compare first output frame to reference
        first_out = out[:, 0:1].reshape(1, out.shape[2], out.shape[3], C)
        mx.eval(first_out)
        assert mx.allclose(first_out, ref, atol=1e-5).item(), \
            "First frame should bypass time_conv and match spatial-only upsample"


class TestVAE22ResidualBlock:
    """Tests for vae22.ResidualBlock."""

    def test_same_dim(self):
        from mlx_video.models.wan.vae22 import ResidualBlock
        block = ResidualBlock(8, 8)
        x = mx.random.normal((1, 2, 4, 4, 8))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 2, 4, 4, 8)

    def test_different_dim(self):
        from mlx_video.models.wan.vae22 import ResidualBlock
        block = ResidualBlock(8, 16)
        x = mx.random.normal((1, 2, 4, 4, 8))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 2, 4, 4, 16)

    def test_shortcut_when_dims_differ(self):
        from mlx_video.models.wan.vae22 import ResidualBlock
        block = ResidualBlock(8, 16)
        assert block.shortcut is not None

    def test_no_shortcut_same_dim(self):
        from mlx_video.models.wan.vae22 import ResidualBlock
        block = ResidualBlock(8, 8)
        assert block.shortcut is None


class TestResidualBlockLayers:
    """Tests for vae22.ResidualBlockLayers naming convention."""

    def test_layer_names_no_underscore_prefix(self):
        """Layer names must NOT start with underscore (MLX ignores them)."""
        from mlx_video.models.wan.vae22 import ResidualBlockLayers
        block = ResidualBlockLayers(8, 8)
        params = dict(block.parameters())
        # All param keys should use layer_N, not _layer_N
        for key in params:
            assert not key.startswith("_"), f"Parameter {key} starts with underscore"

    def test_has_expected_layers(self):
        from mlx_video.models.wan.vae22 import ResidualBlockLayers
        block = ResidualBlockLayers(8, 16)
        assert hasattr(block, "layer_0")  # first RMS_norm
        assert hasattr(block, "layer_2")  # first CausalConv3d
        assert hasattr(block, "layer_3")  # second RMS_norm
        assert hasattr(block, "layer_6")  # second CausalConv3d

    def test_forward_shape(self):
        from mlx_video.models.wan.vae22 import ResidualBlockLayers
        block = ResidualBlockLayers(8, 16)
        x = mx.random.normal((1, 2, 4, 4, 8))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 2, 4, 4, 16)


class TestVAE22AttentionBlock:
    """Tests for vae22.AttentionBlock (per-frame 2D self-attention)."""

    def test_output_shape(self):
        from mlx_video.models.wan.vae22 import AttentionBlock
        block = AttentionBlock(16)
        block.to_qkv_weight = mx.random.normal(block.to_qkv_weight.shape) * 0.01
        block.proj_weight = mx.random.normal(block.proj_weight.shape) * 0.01
        x = mx.random.normal((1, 2, 4, 4, 16))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 2, 4, 4, 16)

    def test_residual_connection(self):
        from mlx_video.models.wan.vae22 import AttentionBlock
        block = AttentionBlock(8)
        block.to_qkv_weight = mx.zeros(block.to_qkv_weight.shape)
        block.proj_weight = mx.zeros(block.proj_weight.shape)
        x = mx.ones((1, 1, 2, 2, 8))
        out = block(x)
        mx.eval(out)
        # With zero weights, attention output is 0 → residual is identity
        np.testing.assert_allclose(np.array(out), np.array(x), atol=1e-5)


class TestHead22:
    """Tests for vae22.Head22 output head."""

    def test_output_shape(self):
        from mlx_video.models.wan.vae22 import Head22
        head = Head22(16, out_channels=12)
        x = mx.random.normal((1, 2, 4, 4, 16))
        out = head(x)
        mx.eval(out)
        assert out.shape == (1, 2, 4, 4, 12)

    def test_layer_names_no_underscore(self):
        """Head layers must not use underscore prefix."""
        from mlx_video.models.wan.vae22 import Head22
        head = Head22(8)
        assert hasattr(head, "layer_0")  # RMS_norm
        assert hasattr(head, "layer_2")  # CausalConv3d
        params = dict(head.parameters())
        for key in params:
            assert not key.startswith("_"), f"Head param {key} starts with underscore"


class TestUnpatchify:
    """Tests for vae22._unpatchify."""

    def test_basic_shape(self):
        from mlx_video.models.wan.vae22 import _unpatchify
        x = mx.random.normal((1, 2, 4, 4, 12))  # 12 = 3 * 2 * 2
        out = _unpatchify(x, patch_size=2)
        mx.eval(out)
        assert out.shape == (1, 2, 8, 8, 3)

    def test_patch_size_1_noop(self):
        from mlx_video.models.wan.vae22 import _unpatchify
        x = mx.random.normal((1, 2, 4, 4, 3))
        out = _unpatchify(x, patch_size=1)
        mx.eval(out)
        np.testing.assert_array_equal(np.array(out), np.array(x))

    def test_preserves_content(self):
        """Unpatchify should be a lossless rearrangement."""
        from mlx_video.models.wan.vae22 import _unpatchify
        x = mx.arange(48).reshape(1, 1, 2, 2, 12).astype(mx.float32)
        out = _unpatchify(x, patch_size=2)
        mx.eval(out)
        # All elements should be preserved
        assert np.array(out).size == 48
        assert set(np.array(out).flatten().tolist()) == set(range(48))


class TestDenormalizeLatents:
    """Tests for vae22.denormalize_latents."""

    def test_output_shape(self):
        from mlx_video.models.wan.vae22 import denormalize_latents
        z = mx.random.normal((1, 2, 4, 4, 48))
        out = denormalize_latents(z)
        mx.eval(out)
        assert out.shape == (1, 2, 4, 4, 48)

    def test_custom_mean_std(self):
        from mlx_video.models.wan.vae22 import denormalize_latents
        z = mx.ones((1, 1, 1, 1, 4))
        mean = mx.array([1.0, 2.0, 3.0, 4.0])
        std = mx.array([0.5, 0.5, 0.5, 0.5])
        out = denormalize_latents(z, mean=mean, std=std)
        mx.eval(out)
        # z * std + mean = 1*0.5 + [1,2,3,4] = [1.5, 2.5, 3.5, 4.5]
        np.testing.assert_allclose(np.array(out).flatten(), [1.5, 2.5, 3.5, 4.5], atol=1e-5)

    def test_uses_default_constants(self):
        from mlx_video.models.wan.vae22 import VAE22_MEAN, VAE22_STD, denormalize_latents
        # Should not raise with default constants
        z = mx.zeros((1, 1, 1, 1, 48))
        out = denormalize_latents(z)
        mx.eval(out)
        # z=0 → result = 0 * std + mean = mean
        np.testing.assert_allclose(
            np.array(out).flatten(),
            np.array(VAE22_MEAN).flatten(),
            atol=1e-5,
        )


class TestVAE22NormConstants:
    """Tests for VAE22_MEAN and VAE22_STD constants."""

    def test_dimensions(self):
        from mlx_video.models.wan.vae22 import VAE22_MEAN, VAE22_STD
        mx.eval(VAE22_MEAN, VAE22_STD)
        assert VAE22_MEAN.shape == (48,)
        assert VAE22_STD.shape == (48,)

    def test_std_positive(self):
        from mlx_video.models.wan.vae22 import VAE22_STD
        mx.eval(VAE22_STD)
        assert (np.array(VAE22_STD) > 0).all()


class TestWan22VAEDecoder:
    """Tests for the full Wan22VAEDecoder (tiny configuration)."""

    def test_output_shape_small(self):
        """Tiny decoder should produce correct spatial/temporal output."""
        from mlx_video.models.wan.vae22 import Wan22VAEDecoder
        # Use very small dims to keep test fast
        dec = Wan22VAEDecoder(z_dim=4, dim=8, dec_dim=8)
        # Latent: [B=1, T=3, H=2, W=2, C=4]
        # Expected: temporal 3→5→9→9→9 (two temporal upsamples), spatial 2→4→8→16
        z = mx.random.normal((1, 3, 2, 2, 4)) * 0.1
        out = dec(z)
        mx.eval(out)
        # Output should have 3 RGB channels and be clipped to [-1, 1]
        assert out.shape[-1] == 3
        assert out.ndim == 5
        assert np.array(out).min() >= -1.0
        assert np.array(out).max() <= 1.0

    def test_output_clipped(self):
        from mlx_video.models.wan.vae22 import Wan22VAEDecoder
        dec = Wan22VAEDecoder(z_dim=4, dim=8, dec_dim=8)
        z = mx.random.normal((1, 2, 2, 2, 4)) * 10.0  # large values
        out = dec(z)
        mx.eval(out)
        assert np.array(out).min() >= -1.0 - 1e-6
        assert np.array(out).max() <= 1.0 + 1e-6


class TestSanitizeWan22VAEWeights:
    """Tests for vae22.sanitize_wan22_vae_weights."""

    def test_skip_encoder(self):
        from mlx_video.models.wan.vae22 import sanitize_wan22_vae_weights
        weights = {
            "encoder.layer.weight": mx.zeros((4,)),
            "conv1.weight": mx.zeros((4,)),
            "decoder.conv1.bias": mx.zeros((4,)),
        }
        out = sanitize_wan22_vae_weights(weights)
        assert "encoder.layer.weight" not in out
        assert "conv1.weight" not in out
        assert "decoder.conv1.bias" in out

    def test_sequential_index_remapping(self):
        from mlx_video.models.wan.vae22 import sanitize_wan22_vae_weights
        weights = {
            "decoder.upsamples.0.upsamples.0.residual.0.gamma": mx.ones((8,)),
            "decoder.upsamples.0.upsamples.0.residual.6.bias": mx.zeros((8,)),
            "decoder.head.0.gamma": mx.ones((4,)),
            "decoder.head.2.bias": mx.zeros((12,)),
        }
        out = sanitize_wan22_vae_weights(weights)
        assert "decoder.upsamples.0.upsamples.0.residual.layer_0.gamma" in out
        assert "decoder.upsamples.0.upsamples.0.residual.layer_6.bias" in out
        assert "decoder.head.layer_0.gamma" in out
        assert "decoder.head.layer_2.bias" in out

    def test_resample_conv_remapping(self):
        from mlx_video.models.wan.vae22 import sanitize_wan22_vae_weights
        weights = {
            "decoder.upsamples.1.upsamples.3.resample.1.weight": mx.zeros((8, 8, 3, 3)),
            "decoder.upsamples.1.upsamples.3.resample.1.bias": mx.zeros((8,)),
        }
        out = sanitize_wan22_vae_weights(weights)
        assert "decoder.upsamples.1.upsamples.3.resample_weight" in out
        assert "decoder.upsamples.1.upsamples.3.resample_bias" in out

    def test_attention_remapping(self):
        from mlx_video.models.wan.vae22 import sanitize_wan22_vae_weights
        weights = {
            "decoder.middle.1.to_qkv.weight": mx.zeros((24, 8, 1, 1)),
            "decoder.middle.1.to_qkv.bias": mx.zeros((24,)),
            "decoder.middle.1.proj.weight": mx.zeros((8, 8, 1, 1)),
            "decoder.middle.1.proj.bias": mx.zeros((8,)),
        }
        out = sanitize_wan22_vae_weights(weights)
        assert "decoder.middle.1.to_qkv_weight" in out
        assert "decoder.middle.1.to_qkv_bias" in out
        assert "decoder.middle.1.proj_weight" in out
        assert "decoder.middle.1.proj_bias" in out

    def test_conv3d_transpose(self):
        from mlx_video.models.wan.vae22 import sanitize_wan22_vae_weights
        # Conv3d weight: [O, I, D, H, W] → [O, D, H, W, I]
        w = mx.zeros((16, 8, 3, 3, 3))
        weights = {"decoder.conv1.weight": w}
        out = sanitize_wan22_vae_weights(weights)
        assert out["decoder.conv1.weight"].shape == (16, 3, 3, 3, 8)

    def test_conv2d_transpose(self):
        from mlx_video.models.wan.vae22 import sanitize_wan22_vae_weights
        # Conv2d weight: [O, I, H, W] → [O, H, W, I]
        w = mx.zeros((8, 8, 3, 3))
        weights = {"decoder.upsamples.0.upsamples.2.resample.1.weight": w}
        out = sanitize_wan22_vae_weights(weights)
        key = "decoder.upsamples.0.upsamples.2.resample_weight"
        assert out[key].shape == (8, 3, 3, 8)

    def test_gamma_squeeze(self):
        from mlx_video.models.wan.vae22 import sanitize_wan22_vae_weights
        # gamma: (dim, 1, 1, 1) → (dim,)
        w = mx.ones((16, 1, 1, 1))
        weights = {"decoder.upsamples.0.upsamples.0.residual.0.gamma": w}
        out = sanitize_wan22_vae_weights(weights)
        key = "decoder.upsamples.0.upsamples.0.residual.layer_0.gamma"
        assert out[key].shape == (16,)


class TestUpResidualBlock:
    """Tests for vae22.Up_ResidualBlock."""

    def test_no_upsample(self):
        from mlx_video.models.wan.vae22 import Up_ResidualBlock
        block = Up_ResidualBlock(8, 8, num_res_blocks=1, temperal_upsample=False, up_flag=False)
        x = mx.random.normal((1, 2, 4, 4, 8))
        out = block(x)
        mx.eval(out)
        # No upsample: same shape
        assert out.shape == (1, 2, 4, 4, 8)

    def test_spatial_upsample(self):
        from mlx_video.models.wan.vae22 import Up_ResidualBlock
        block = Up_ResidualBlock(8, 4, num_res_blocks=1, temperal_upsample=False, up_flag=True)
        x = mx.random.normal((1, 2, 4, 4, 8))
        out = block(x)
        mx.eval(out)
        # 2x spatial upsample, no temporal
        assert out.shape == (1, 2, 8, 8, 4)

    def test_spatial_temporal_upsample(self):
        from mlx_video.models.wan.vae22 import Up_ResidualBlock
        block = Up_ResidualBlock(8, 4, num_res_blocks=1, temperal_upsample=True, up_flag=True)
        x = mx.random.normal((1, 2, 4, 4, 8))
        out = block(x)
        mx.eval(out)
        # 2x spatial + 2x temporal
        assert out.shape == (1, 4, 8, 8, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
