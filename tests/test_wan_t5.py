"""Tests for T5 encoder components."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest


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
