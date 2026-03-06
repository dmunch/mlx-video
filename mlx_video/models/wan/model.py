import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .attention import WanLayerNorm, _linear_dtype
from .config import WanModelConfig
from .rope import rope_params, rope_precompute_cos_sin
from .spectrum import SpectrumState
from .transformer import WanAttentionBlock


@dataclass
class TeaCacheState:
    """Tracks TeaCache state for skipping redundant transformer computations.

    TeaCache (Timestep Embedding Aware Cache) monitors the relative L1 distance
    between consecutive projected time embeddings (e0). When the accumulated
    rescaled distance is below a threshold, the transformer blocks are skipped
    and the cached residual from the previous step is reused.

    Uses the ret-mode approach: similarity is computed on `e0` (the projected
    time embedding passed to transformer blocks), with polynomial coefficients
    profiled against `e0`. This gives better skip calibration across different
    step counts and schedule shifts than the non-ret approach using raw `e`.


    Since batched CFG shares the same scalar timestep across cond/uncond, we
    track a single e0 / residual for the whole batch.
    """

    enabled: bool = False
    threshold: float = 0.0
    coefficients: tuple = ()
    verbose: bool = False

    # Single tracking for the whole batch (same timestep → same e0)
    previous_e0: object = None  # mx.array | None
    accumulated_distance: float = 0.0
    previous_residual: object = None  # mx.array | None

    # Step counter and bounds
    cnt: int = 0
    num_steps: int = 0
    ret_steps: int = 2  # always compute first N steps
    cutoff_steps: int = 0  # always compute last N steps (set to num_steps - 2)

    # Stats
    steps_skipped: int = 0
    steps_computed: int = 0

    def reset(self):
        self.previous_e0 = None
        self.accumulated_distance = 0.0
        self.previous_residual = None
        self.cnt = 0
        self.steps_skipped = 0
        self.steps_computed = 0


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
        self.modulation = (mx.random.normal((1, 2, dim)) * (dim**-0.5)).astype(
            mx.float32
        )

    def __call__(self, x: mx.array, e: mx.array) -> mx.array:
        """
        Args:
            x: [B, L, dim]
            e: [B, dim] or [B, 1, dim] (broadcast) or [B, L, dim] (per-token)
        """
        if e.ndim == 2:
            e = e[:, None, :]  # [B, 1, dim]
        # Compute modulation in float32 (matching reference's autocast(float32))
        mod = self.modulation[:, None, :, :] + e[:, :, None, :]  # float32
        e0 = mod[:, :, 0, :]  # [B, L_e, dim] shift
        e1 = mod[:, :, 1, :]  # [B, L_e, dim] scale
        x_norm = self.norm(x)
        x_mod = x_norm * (1 + e1) + e0
        return self.head(x_mod)


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
        self.text_embedding_act = nn.GELU(approx="tanh")
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

        # Precompute RoPE frequencies — three separate tables concatenated.
        # Reference computes three rope_params with different dim normalizations
        # so each axis (temporal/height/width) gets its own full frequency range.
        d = dim // config.num_heads
        self.freqs = mx.concatenate(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            axis=1,
        )

        # Precompute sinusoidal inv_freq for time embedding
        # Use numpy float64 for precision (matches reference torch.float64),
        # then store as float32 since MLX GPU doesn't support float64.
        half = config.freq_dim // 2
        inv_freq_np = np.power(
            10000.0, -np.arange(half, dtype=np.float64) / half
        )
        self._inv_freq = mx.array(inv_freq_np.astype(np.float32))

        # TeaCache state (disabled by default)
        self.teacache = TeaCacheState()

        # Spectrum state (disabled by default)
        self.spectrum = SpectrumState()

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
        patches = patches.astype(_linear_dtype(self.patch_embedding_proj))
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
        model_dtype = _linear_dtype(self.patch_embedding_proj)
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

    def prepare_rope(self, grid_sizes: list) -> tuple:
        """Pre-compute RoPE cos/sin for constant grid sizes.

        Call once before the diffusion loop when grid sizes don't change
        across steps. Eliminates per-step broadcast/concat overhead.

        Args:
            grid_sizes: List of (F, H, W) tuples per batch element

        Returns:
            (cos_f, sin_f) precomputed frequency tensors
        """
        w_dtype = _linear_dtype(self.patch_embedding_proj)
        return rope_precompute_cos_sin(grid_sizes, self.freqs, dtype=w_dtype)

    def __call__(
        self,
        x_list: list,
        t: mx.array,
        context: list | mx.array,
        seq_len: int,
        cross_kv_caches: list | None = None,
        y: list | None = None,
        rope_cos_sin: tuple | None = None,
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
            y: Optional list of conditioning tensors for I2V [C_y, F, H, W].
               Channel-concatenated with x before patchify.
            rope_cos_sin: Optional precomputed (cos, sin) from prepare_rope().

        Returns:
            List of denoised tensors [C, F, H, W]
        """
        # Detect identical inputs (CFG B=2) to avoid duplicate patchify work.
        # Check BEFORE I2V concat since concat creates new array objects.
        batch_size = len(x_list)
        all_same = batch_size > 1 and all(
            x_list[i] is x_list[0] for i in range(1, batch_size)
        )
        if all_same and y is not None:
            all_same = all(y[i] is y[0] for i in range(1, len(y)))

        # I2V: channel-concatenate conditioning y with noise x
        if y is not None:
            x_list = [mx.concatenate([u, v], axis=0) for u, v in zip(x_list, y)]

        if all_same:
            # Patchify once and broadcast — saves a Linear projection per step
            p, gs = self._patchify(x_list[0])  # [1, L, dim]
            grid_sizes = [gs] * batch_size
            seq_lens_list = [p.shape[1]] * batch_size
            # Pad and broadcast
            if p.shape[1] < seq_len:
                p = mx.concatenate(
                    [p, mx.zeros((1, seq_len - p.shape[1], self.dim), dtype=p.dtype)],
                    axis=1,
                )
            x = mx.broadcast_to(p, (batch_size,) + p.shape[1:])
        else:
            patches = []
            grid_sizes = []
            seq_lens_list = []
            for vid in x_list:
                p, gs = self._patchify(vid)  # [1, L, dim]
                patches.append(p)
                grid_sizes.append(gs)
                seq_lens_list.append(p.shape[1])
            x = mx.concatenate(
                [
                    (
                        mx.concatenate(
                            [
                                p,
                                mx.zeros(
                                    (1, seq_len - p.shape[1], self.dim), dtype=p.dtype
                                ),
                            ],
                            axis=1,
                        )
                        if p.shape[1] < seq_len
                        else p
                    )
                    for p in patches
                ],
                axis=0,
            )  # [B, seq_len, dim]

        # Time embedding (use cached inv_freq to avoid recomputing each step)
        if t.ndim == 0:
            t = t[None]

        pos = t.astype(mx.float32)
        sinusoid = pos[..., None] * self._inv_freq
        sin_emb = mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=-1)

        if t.ndim == 1:
            # Standard T2V: scalar timestep per batch element [B]
            e = self.time_embedding_1(
                self.time_embedding_act(self.time_embedding_0(sin_emb))
            )  # [B, dim]
            e0 = self.time_projection(self.time_projection_act(e))  # [B, dim*6]
            e0 = e0.reshape(batch_size, 1, 6, self.dim)
        else:
            # I2V: per-token timesteps [B, L]
            e = self.time_embedding_1(
                self.time_embedding_act(self.time_embedding_0(sin_emb))
            )  # [B, L, dim]
            e0 = self.time_projection(self.time_projection_act(e))  # [B, L, dim*6]
            e0 = e0.reshape(batch_size, -1, 6, self.dim)

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

        # Pre-compute attention mask from seq_lens (constant across all blocks)
        attn_mask = None
        w_dtype = _linear_dtype(self.patch_embedding_proj)
        if any(sl < seq_len for sl in seq_lens_list):
            attn_mask = mx.zeros((batch_size, 1, 1, seq_len), dtype=w_dtype)
            for i, sl in enumerate(seq_lens_list):
                attn_mask[i, :, :, sl:] = -1e9

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens_list,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context_batch,
            context_lens=None,
            rope_cos_sin=rope_cos_sin,
            attn_mask=attn_mask,
        )

        # Run transformer blocks (with optional TeaCache or Spectrum skip)
        if self.spectrum.enabled:
            sp = self.spectrum
            forecaster = sp.get_or_create_forecaster()
            do_compute = sp.should_compute()

            if do_compute:
                for i, block in enumerate(self.blocks):
                    kv = cross_kv_caches[i] if cross_kv_caches is not None else None
                    x = block(x, cross_kv_cache=kv, **kwargs)
                # Cache flattened features and update Chebyshev fit
                h_flat = x.reshape(-1)
                forecaster.update(sp.cnt, h_flat)
                mx.eval(forecaster.cheb.H_buf, forecaster.cheb.t_buf)
            else:
                # Predict features using fitted Chebyshev polynomials
                h_pred = forecaster.predict(sp.cnt)
                mx.eval(h_pred)
                x = h_pred.reshape(x.shape)

            sp.step(do_compute)
        elif self.teacache.enabled:
            tc = self.teacache
            should_skip = False

            if tc.cnt < tc.ret_steps or tc.cnt >= tc.cutoff_steps:
                # Always compute first/last steps (they change the most)
                tc.accumulated_distance = 0.0
                if tc.verbose:
                    tag = "first" if tc.cnt < tc.ret_steps else "last"
                    print(f"    [TeaCache] step {tc.cnt}: forced compute ({tag})")
            elif tc.previous_e0 is not None:
                # Compute relative L1 distance between current and previous e0
                # (projected time embedding — ret-mode coefficients are profiled
                # against e0 for proper calibration across step counts)
                rel_l1 = (
                    mx.abs(e0 - tc.previous_e0).mean() / mx.abs(tc.previous_e0).mean()
                )

                # Polynomial rescaling in MLX (Horner's method)
                rescaled = tc.coefficients[0]
                for c in tc.coefficients[1:]:
                    rescaled = rescaled * rel_l1 + c

                rescaled_val = rescaled.item()
                rel_l1_val = rel_l1.item()
                tc.accumulated_distance += rescaled_val

                if tc.accumulated_distance < tc.threshold:
                    should_skip = True
                else:
                    tc.accumulated_distance = 0.0

                if tc.verbose:
                    decision = "SKIP" if should_skip else "COMPUTE"
                    print(
                        f"    [TeaCache] step {tc.cnt}: rel_l1={rel_l1_val:.6f}  rescaled={rescaled_val:.4f}  accum={tc.accumulated_distance:.4f}  → {decision}"
                    )
            else:
                if tc.verbose:
                    print(f"    [TeaCache] step {tc.cnt}: compute (no previous)")


            tc.previous_e0 = e0

            if should_skip and tc.previous_residual is not None:
                # Reuse cached residual — skip all transformer blocks
                x = x + tc.previous_residual
                tc.steps_skipped += 1
            else:
                # Full forward pass through transformer blocks
                ori_x = x
                for i, block in enumerate(self.blocks):
                    kv = cross_kv_caches[i] if cross_kv_caches is not None else None
                    x = block(x, cross_kv_cache=kv, **kwargs)
                # Cache the residual for potential reuse
                tc.previous_residual = x - ori_x
                tc.steps_computed += 1

            tc.cnt += 1
        else:
            for i, block in enumerate(self.blocks):
                kv = cross_kv_caches[i] if cross_kv_caches is not None else None
                x = block(x, cross_kv_cache=kv, **kwargs)

        # Output head
        x = self.head(x, e)

        # Unpatchify
        outputs = self.unpatchify(x, grid_sizes)
        return [u.astype(mx.float32) for u in outputs]
