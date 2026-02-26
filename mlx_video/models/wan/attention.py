import mlx.core as mx
import mlx.nn as nn

from .rope import rope_apply


class WanRMSNorm(nn.Module):
    """RMS normalization with learnable scale."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        output = mx.fast.rms_norm(x.astype(mx.float32), mx.ones((x.shape[-1],)), self.eps)
        return output.astype(x.dtype) * self.weight


class WanLayerNorm(nn.Module):
    """LayerNorm computed in float32, with optional affine."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones((dim,))
            self.bias = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        x_f32 = x.astype(mx.float32)
        mean = mx.mean(x_f32, axis=-1, keepdims=True)
        var = mx.var(x_f32, axis=-1, keepdims=True)
        out = (x_f32 - mean) * mx.rsqrt(var + self.eps)
        out = out.astype(x.dtype)
        if self.elementwise_affine:
            out = out * self.weight + self.bias
        return out


class WanSelfAttention(nn.Module):
    """Self-attention with QK normalization and 3-way factorized RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else None
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else None

    def __call__(
        self,
        x: mx.array,
        seq_lens: list,
        grid_sizes: list,
        freqs: mx.array,
    ) -> mx.array:
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim

        q = self.q(x)
        k = self.k(x)
        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)

        q = q.reshape(b, s, n, d)
        k = k.reshape(b, s, n, d)
        v = self.v(x).reshape(b, s, n, d)

        # Apply RoPE
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        # Scaled dot-product attention: [B, L, N, D] -> [B, N, L, D]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Build attention mask from seq_lens
        max_len = s
        mask = None
        if any(sl < max_len for sl in seq_lens):
            # Create mask: [B, 1, 1, L] where True = attend
            mask = mx.zeros((b, 1, 1, max_len))
            for i, sl in enumerate(seq_lens):
                mask[i, :, :, sl:] = -1e9

        attn_weights = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            attn_weights = attn_weights + mask
        attn_weights = mx.softmax(attn_weights.astype(mx.float32), axis=-1).astype(x.dtype)

        out = (attn_weights @ v).transpose(0, 2, 1, 3).reshape(b, s, -1)
        return self.o(out)


class WanCrossAttention(nn.Module):
    """Cross-attention: Q from hidden states, K/V from text context."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else None
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else None

    def __call__(
        self,
        x: mx.array,
        context: mx.array,
        context_lens: list | None = None,
    ) -> mx.array:
        b = x.shape[0]
        n, d = self.num_heads, self.head_dim

        q = self.q(x)
        k = self.k(context)
        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)

        q = q.reshape(b, -1, n, d).transpose(0, 2, 1, 3)
        k = k.reshape(b, -1, n, d).transpose(0, 2, 1, 3)
        v = self.v(context).reshape(b, -1, n, d).transpose(0, 2, 1, 3)

        # Optional context masking
        mask = None
        if context_lens is not None:
            ctx_len = context.shape[1]
            mask = mx.zeros((b, 1, 1, ctx_len))
            for i, cl in enumerate(context_lens):
                mask[i, :, :, cl:] = -1e9

        attn_weights = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            attn_weights = attn_weights + mask
        attn_weights = mx.softmax(attn_weights.astype(mx.float32), axis=-1).astype(x.dtype)

        out = (attn_weights @ v).transpose(0, 2, 1, 3).reshape(b, -1, n * d)
        return self.o(out)
