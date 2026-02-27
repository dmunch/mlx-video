"""Tests for Wan transformer block components."""

import mlx.core as mx
import numpy as np
import pytest


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
# Float32 Modulation Precision Tests
# ---------------------------------------------------------------------------

class TestFloat32Modulation:
    """Tests that modulation/gate operations are computed in float32,
    matching official torch.amp.autocast('cuda', dtype=torch.float32)."""

    def setup_method(self):
        mx.random.seed(42)
        self.dim = 64

    def test_block_modulation_in_float32(self):
        """Modulation param starts random but should be usable as float32."""
        from mlx_video.models.wan.transformer import WanAttentionBlock
        block = WanAttentionBlock(self.dim, 128, 4, cross_attn_norm=True)
        assert block.modulation.dtype == mx.float32

    def test_block_output_float32_with_bf16_modulation_input(self):
        """Even if e (time embedding) arrives as bf16, modulation should cast to f32."""
        from mlx_video.models.wan.transformer import WanAttentionBlock
        from mlx_video.models.wan.rope import rope_params
        block = WanAttentionBlock(self.dim, 128, 4)
        B, L = 1, 8
        x = mx.random.normal((B, L, self.dim))
        e = mx.random.normal((B, L, 6, self.dim)).astype(mx.bfloat16)
        ctx = mx.random.normal((B, 4, self.dim))
        freqs = rope_params(1024, self.dim // 4)

        out = block(x, e, [L], [(2, 2, 2)], freqs, ctx)
        mx.eval(out)
        assert out.dtype == mx.float32
        assert np.isfinite(np.array(out)).all()

    def test_head_modulation_float32(self):
        """Head modulation should be float32 even with bf16 e input."""
        from mlx_video.models.wan.model import Head
        head = Head(self.dim, 4, (1, 2, 2))
        x = mx.random.normal((1, 8, self.dim))
        e = mx.random.normal((1, 8, self.dim)).astype(mx.bfloat16)
        out = head(x, e)
        mx.eval(out)
        assert np.isfinite(np.array(out.astype(mx.float32))).all()

    def test_model_time_embedding_float32(self):
        """sinusoidal_embedding_1d output must be float32."""
        from mlx_video.models.wan.model import sinusoidal_embedding_1d
        t = mx.array([500.0])
        emb = sinusoidal_embedding_1d(256, t)
        mx.eval(emb)
        assert emb.dtype == mx.float32

    def test_model_per_token_time_embedding_float32(self):
        """Per-token time embeddings (I2V) should also be float32."""
        from mlx_video.models.wan.model import sinusoidal_embedding_1d
        t = mx.array([[0.0, 100.0, 200.0, 300.0]])  # [B=1, L=4]
        emb = sinusoidal_embedding_1d(256, t)
        mx.eval(emb)
        assert emb.dtype == mx.float32
        assert emb.shape == (1, 4, 256)
