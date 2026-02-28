import mlx.core as mx
import mlx.nn as nn

from .attention import WanCrossAttention, WanLayerNorm, WanSelfAttention


class WanAttentionBlock(nn.Module):
    """Wan transformer block with learned modulation, self-attn, cross-attn, and FFN."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()

        # Self-attention
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)

        # Cross-attention (with optional norm on context)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else None
        )
        self.cross_attn = WanCrossAttention(dim, num_heads, qk_norm, eps)

        # Feed-forward
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = WanFFN(dim, ffn_dim)

        # Learned modulation: 6 vectors for scale/shift/gate (kept in float32 for precision)
        self.modulation = (mx.random.normal((1, 6, dim)) * (dim**-0.5)).astype(mx.float32)

    def __call__(
        self,
        x: mx.array,
        e: mx.array,
        seq_lens: list,
        grid_sizes: list,
        freqs: mx.array,
        context: mx.array,
        context_lens: list | None = None,
        cross_kv_cache: tuple | None = None,
        rope_cos_sin: tuple | None = None,
        attn_mask: mx.array | None = None,
    ) -> mx.array:
        # Modulation in float32 (e is already float32 from model forward)
        mod = self.modulation + e
        e0 = mod[:, :, 0, :]  # shift for self-attn
        e1 = mod[:, :, 1, :]  # scale for self-attn
        e2 = mod[:, :, 2, :]  # gate for self-attn
        e3 = mod[:, :, 3, :]  # shift for ffn
        e4 = mod[:, :, 4, :]  # scale for ffn
        e5 = mod[:, :, 5, :]  # gate for ffn

        # Self-attention with modulation
        # Type promotion handles bf16→f32 automatically when multiplied with f32 modulation
        x_mod = self.norm1(x) * (1 + e1) + e0
        y = self.self_attn(x_mod, seq_lens, grid_sizes, freqs, rope_cos_sin=rope_cos_sin, attn_mask=attn_mask)
        x = x + y * e2

        # Cross-attention (no modulation, just norm)
        x_cross = self.norm3(x) if self.norm3 is not None else x
        x = x + self.cross_attn(x_cross, context, context_lens, kv_cache=cross_kv_cache)

        # FFN with modulation
        x_mod = self.norm2(x) * (1 + e4) + e3
        y = self.ffn(x_mod)
        x = x + y * e5

        return x


class WanFFN(nn.Module):
    """Gated feed-forward network with GELU(tanh) activation."""

    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.act = nn.GELU(approx="precise")
        self.fc2 = nn.Linear(ffn_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        # Cast to weight dtype for efficient matmul (bfloat16 matching official autocast)
        x_w = x.astype(self.fc1.weight.dtype)
        return self.fc2(self.act(self.fc1(x_w)))
