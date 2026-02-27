import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .attention import WanLayerNorm
from .config import WanModelConfig
from .rope import rope_params
from .transformer import WanAttentionBlock


def sinusoidal_embedding_1d(dim: int, position: mx.array) -> mx.array:
    """Compute sinusoidal positional embeddings.

    Args:
        dim: Embedding dimension (must be even).
        position: Tensor of positions — 1D [L] or 2D [B, L].

    Returns:
        Embeddings of shape [L, dim] or [B, L, dim].
    """
    assert dim % 2 == 0
    half = dim // 2
    pos = position.astype(mx.float32)
    inv_freq = mx.power(10000.0, -mx.arange(half).astype(mx.float32) / half)
    sinusoid = pos[..., None] * inv_freq  # [..., half]
    return mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=-1)


class Head(nn.Module):
    """Output projection head with learned modulation."""

    def __init__(self, dim: int, out_dim: int, patch_size: tuple, eps: float = 1e-6):
        super().__init__()
        self.out_dim = out_dim
        self.patch_size = patch_size
        proj_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, proj_dim)
        self.modulation = mx.random.normal((1, 2, dim)) * (dim**-0.5)

    def __call__(self, x: mx.array, e: mx.array) -> mx.array:
        """
        Args:
            x: [B, L, dim]
            e: [B, dim] or [B, 1, dim] (broadcast) or [B, L, dim] (per-token)
        """
        if e.ndim == 2:
            e = e[:, None, :]  # [B, 1, dim]
        e_f32 = e.astype(mx.float32)
        # modulation [1, 2, dim] broadcasts with e [B, 1/L, dim] via unsqueeze
        mod = self.modulation.astype(mx.float32)[:, None, :, :] + e_f32[:, :, None, :]  # [B, L_e, 2, dim]
        e0 = mod[:, :, 0, :]  # [B, L_e, dim] shift
        e1 = mod[:, :, 1, :]  # [B, L_e, dim] scale
        x_norm = self.norm(x).astype(mx.float32)
        x_mod = x_norm * (1 + e1) + e0  # broadcasts over L if L_e==1
        return self.head(x_mod.astype(x.dtype))


class WanModel(nn.Module):
    """Wan2.2 diffusion backbone for text-to-video generation."""

    def __init__(self, config: WanModelConfig):
        super().__init__()
        self.config = config
        dim = config.dim
        self.dim = dim
        self.num_heads = config.num_heads
        self.out_dim = config.out_dim
        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.freq_dim = config.freq_dim

        # Patch embedding: Conv3d implemented as a reshaped linear
        # For kernel (1,2,2) and stride (1,2,2): reshape input then linear
        patch_dim = config.in_dim * math.prod(config.patch_size)
        self.patch_embedding_proj = nn.Linear(patch_dim, dim)
        self._patch_size = config.patch_size

        # Text embedding MLP
        self.text_embedding_0 = nn.Linear(config.text_dim, dim)
        self.text_embedding_act = nn.GELU(approx="precise")
        self.text_embedding_1 = nn.Linear(dim, dim)

        # Time embedding MLP
        self.time_embedding_0 = nn.Linear(config.freq_dim, dim)
        self.time_embedding_act = nn.SiLU()
        self.time_embedding_1 = nn.Linear(dim, dim)

        # Time projection for modulation (6x dim)
        self.time_projection_act = nn.SiLU()
        self.time_projection = nn.Linear(dim, dim * 6)

        # Transformer blocks
        self.blocks = [
            WanAttentionBlock(
                dim=dim,
                ffn_dim=config.ffn_dim,
                num_heads=config.num_heads,
                window_size=config.window_size,
                qk_norm=config.qk_norm,
                cross_attn_norm=config.cross_attn_norm,
                eps=config.eps,
            )
            for _ in range(config.num_layers)
        ]

        # Output head
        self.head = Head(dim, config.out_dim, config.patch_size, config.eps)

        # Precompute RoPE frequencies
        d = dim // config.num_heads
        d_t = d - 4 * (d // 6)
        d_h = 2 * (d // 6)
        d_w = 2 * (d // 6)
        # Each rope_params returns [1024, d_x//2, 2]
        freqs_t = rope_params(1024, d_t)
        freqs_h = rope_params(1024, d_h)
        freqs_w = rope_params(1024, d_w)
        # Concatenate along the frequency dimension: [1024, d//2, 2]
        self.freqs = mx.concatenate([freqs_t, freqs_h, freqs_w], axis=1)

    def _patchify(self, x: mx.array) -> tuple:
        """Convert video tensor to patch embeddings.

        Args:
            x: Video latent [C, F, H, W]

        Returns:
            (patches, grid_size): patches [1, L, dim], grid_size (F', H', W')
        """
        c, f, h, w = x.shape
        pt, ph, pw = self._patch_size

        f_out = f // pt
        h_out = h // ph
        w_out = w // pw

        # Reshape: [C, F, H, W] -> [F', H', W', C, pt, ph, pw] -> [F'*H'*W', C*pt*ph*pw]
        # Order must be [C, pt, ph, pw] (C slowest) to match Conv3d weight layout
        x = x.reshape(c, f_out, pt, h_out, ph, w_out, pw)
        x = x.transpose(1, 3, 5, 0, 2, 4, 6)  # [F', H', W', C, pt, ph, pw]
        x = x.reshape(f_out * h_out * w_out, -1)  # [L, C*pt*ph*pw]

        # Project and cast to model dtype to prevent float32 cascade from input latents
        patches = self.patch_embedding_proj(x)  # [L, dim]
        patches = patches.astype(self.patch_embedding_proj.weight.dtype)
        patches = patches[None, :, :]  # [1, L, dim]

        return patches, (f_out, h_out, w_out)

    def unpatchify(self, x: mx.array, grid_sizes: list) -> list:
        """Reconstruct video from patch embeddings.

        Args:
            x: [B, L, out_dim * prod(patch_size)]
            grid_sizes: List of (F', H', W') per batch element

        Returns:
            List of tensors [C, F, H, W]
        """
        c = self.out_dim
        pt, ph, pw = self.patch_size
        out = []
        for i, (f, h, w) in enumerate(grid_sizes):
            seq_len = f * h * w
            u = x[i, :seq_len]  # [L, out_dim * pt * ph * pw]
            u = u.reshape(f, h, w, pt, ph, pw, c)
            # Rearrange: [F', H', W', pt, ph, pw, C] -> [C, F'*pt, H'*ph, W'*pw]
            u = u.transpose(6, 0, 3, 1, 4, 2, 5)  # [C, F', pt, H', ph, W', pw]
            u = u.reshape(c, f * pt, h * ph, w * pw)
            out.append(u)
        return out

    def embed_text(self, context: list) -> mx.array:
        """Precompute text embeddings (call once, reuse across steps).

        Args:
            context: List of text embeddings [L_text, text_dim]

        Returns:
            Embedded context [B, text_len, dim] in model dtype
        """
        model_dtype = self.patch_embedding_proj.weight.dtype
        context_padded = []
        for ctx in context:
            pad_len = self.text_len - ctx.shape[0]
            if pad_len > 0:
                ctx = mx.concatenate(
                    [ctx, mx.zeros((pad_len, ctx.shape[1]), dtype=ctx.dtype)],
                    axis=0,
                )
            context_padded.append(ctx)
        context_batch = mx.stack(context_padded)  # [B, text_len, text_dim]
        context_batch = self.text_embedding_1(
            self.text_embedding_act(self.text_embedding_0(context_batch))
        )
        return context_batch.astype(model_dtype)

    def prepare_cross_kv(self, context: mx.array) -> list:
        """Pre-compute cross-attention K/V for all blocks.

        Call once before the diffusion loop to cache K/V projections,
        eliminating redundant computation at each denoising step.

        Args:
            context: Pre-embedded text [B, text_len, dim]

        Returns:
            List of (k, v) tuples, one per block
        """
        kv_caches = []
        for block in self.blocks:
            kv_caches.append(block.cross_attn.prepare_kv(context))
        return kv_caches

    def __call__(
        self,
        x_list: list,
        t: mx.array,
        context: list | mx.array,
        seq_len: int,
        cross_kv_caches: list | None = None,
    ) -> list:
        """Forward pass.

        Args:
            x_list: List of video latent tensors [C, F, H, W]
            t: Timestep tensor [B]
            context: List of raw text embeddings, OR pre-embedded tensor
                     from embed_text() [B, text_len, dim]
            seq_len: Maximum sequence length for padding
            cross_kv_caches: Optional list of (k, v) tuples from
                             prepare_cross_kv(), one per block.

        Returns:
            List of denoised tensors [C, F, H, W]
        """
        # Patchify each video
        patches = []
        grid_sizes = []
        seq_lens_list = []
        for vid in x_list:
            p, gs = self._patchify(vid)  # [1, L, dim]
            patches.append(p)
            grid_sizes.append(gs)
            seq_lens_list.append(p.shape[1])

        # Pad and batch
        batch_size = len(patches)
        x = mx.concatenate(
            [
                mx.concatenate(
                    [p, mx.zeros((1, seq_len - p.shape[1], self.dim), dtype=p.dtype)],
                    axis=1,
                )
                if p.shape[1] < seq_len
                else p
                for p in patches
            ],
            axis=0,
        )  # [B, seq_len, dim]

        # Time embedding
        if t.ndim == 0:
            t = t[None]

        if t.ndim == 1:
            # Standard T2V: scalar timestep per batch element [B]
            sin_emb = sinusoidal_embedding_1d(self.freq_dim, t)  # [B, freq_dim]
            e = self.time_embedding_1(
                self.time_embedding_act(self.time_embedding_0(sin_emb))
            )  # [B, dim]
            e0 = self.time_projection(self.time_projection_act(e))  # [B, dim*6]
            # Keep e and e0 in float32 — official asserts float32 for modulation
            e0 = e0.reshape(batch_size, 1, 6, self.dim).astype(mx.float32)
            e = e.astype(mx.float32)
        else:
            # I2V: per-token timesteps [B, L]
            sin_emb = sinusoidal_embedding_1d(self.freq_dim, t)  # [B, L, freq_dim]
            e = self.time_embedding_1(
                self.time_embedding_act(self.time_embedding_0(sin_emb))
            )  # [B, L, dim]
            e0 = self.time_projection(self.time_projection_act(e))  # [B, L, dim*6]
            # Keep e and e0 in float32 — official asserts float32 for modulation
            e0 = e0.reshape(batch_size, -1, 6, self.dim).astype(mx.float32)
            e = e.astype(mx.float32)

        # Text embedding: skip MLP if context is already embedded (mx.array)
        if isinstance(context, mx.array):
            # Pre-embedded: expand to batch size if needed
            context_batch = context
            if context_batch.shape[0] == 1 and batch_size > 1:
                context_batch = mx.broadcast_to(
                    context_batch, (batch_size,) + context_batch.shape[1:]
                )
        else:
            context_batch = self.embed_text(context)

        # Run transformer blocks
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens_list,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context_batch,
            context_lens=None,
        )

        for i, block in enumerate(self.blocks):
            kv = cross_kv_caches[i] if cross_kv_caches is not None else None
            x = block(x, cross_kv_cache=kv, **kwargs)

        # Output head
        x = self.head(x, e)

        # Unpatchify
        outputs = self.unpatchify(x, grid_sizes)
        return [u.astype(mx.float32) for u in outputs]
