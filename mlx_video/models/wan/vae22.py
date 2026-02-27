"""Wan2.2 VAE Decoder (compression 4×16×16, z_dim=48).

Architecture differs from Wan2.1 VAE: uses RMS_norm, DupUp3D shortcuts,
spatial patchify (2×2), and different temporal upsampling pattern.

Weight keys mirror the PyTorch checkpoint hierarchy so only tensor format
conversion (channels-first → channels-last) is needed.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

CACHE_T = 2

# Per-channel normalization for z_dim=48 latent space
VAE22_MEAN = mx.array([
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,  0.1838,  0.1557,
    -0.1382,  0.0542,  0.2813,  0.0891,  0.1570, -0.0098,  0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502,
    -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649,  0.0117,  0.0723, -0.2839, -0.2083, -0.0520,  0.3748,
     0.0152,  0.1957,  0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667,
])

VAE22_STD = mx.array([
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
])


class CausalConv3d(nn.Module):
    """3D causal convolution. Input/output: [B, T, H, W, C] (channels-last).

    Decomposes the 3D conv into per-frame 2D convolutions to avoid
    excessive memory usage from MLX's conv3d implementation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self._causal_pad_t = 2 * padding[0]
        self._pad_h = padding[1]
        self._pad_w = padding[2]

        # Weight: [O, D, H, W, I] for MLX
        self.weight = mx.zeros((
            out_channels, kernel_size[0], kernel_size[1], kernel_size[2], in_channels
        ))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x):
        # x: [B, T, H, W, C]
        B, T, H, W, C = x.shape
        kd, kh, kw = self.kernel_size

        # For 1x1x1 kernel or kernel_d==1, use direct conv
        if kd == 1 and kh == 1 and kw == 1:
            # Simple pointwise: reshape to [B*T, 1, 1, C] → conv2d
            x_flat = x.reshape(B * T, H, W, C)
            w2d = self.weight[:, 0, :, :, :]  # [O, kH, kW, I]
            y = mx.conv_general(x_flat, w2d) + self.bias
            return y.reshape(B, T, y.shape[1], y.shape[2], -1)

        # Causal temporal padding (left only)
        if self._causal_pad_t > 0:
            pad_t = mx.zeros((B, self._causal_pad_t, H, W, C))
            x = mx.concatenate([pad_t, x], axis=1)

        # Spatial padding
        if self._pad_h > 0 or self._pad_w > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (self._pad_h, self._pad_h),
                           (self._pad_w, self._pad_w), (0, 0)])

        T_padded = x.shape[1]
        H_padded, W_padded = x.shape[2], x.shape[3]
        T_out = (T_padded - kd) // self.stride[0] + 1

        # Decompose 3D conv into sum of 2D convolutions over temporal kernel
        # weight shape: [O, kd, kh, kw, I] → split into kd 2D kernels [O, kh, kw, I]
        outputs = []
        for t in range(T_out):
            t_start = t * self.stride[0]
            # Sum 2D convs for each temporal kernel position
            accum = None
            for d in range(kd):
                frame = x[:, t_start + d]  # [B, H_padded, W_padded, C]
                w2d = self.weight[:, d, :, :, :]  # [O, kh, kw, I]
                conv_out = mx.conv_general(frame, w2d,
                                           stride=(self.stride[1], self.stride[2]))
                accum = conv_out if accum is None else accum + conv_out
            outputs.append(accum + self.bias)

        return mx.stack(outputs, axis=1)  # [B, T_out, H_out, W_out, O]


class RMS_norm(nn.Module):
    """RMS normalization along channel dimension."""

    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        # Weight stored as (dim,) — PyTorch stores (dim, 1, 1, 1) but we squeeze
        self.gamma = mx.ones((dim,))

    def __call__(self, x):
        # x: [..., C] (channels-last)
        # PyTorch uses F.normalize (L2 norm), not RMS: x / max(||x||_2, eps)
        l2_sq = mx.sum(x * x, axis=-1, keepdims=True)
        return x * mx.rsqrt(mx.maximum(l2_sq, mx.array(1e-24))) * self.scale * self.gamma


class ResidualBlock(nn.Module):
    """Residual block: RMS_norm → SiLU → CausalConv3d × 2 + shortcut."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Sequential residual path: [norm, silu, conv3d, norm, silu, dropout, conv3d]
        # We store as named layers matching PyTorch's indices
        self.residual = ResidualBlockLayers(in_dim, out_dim)
        self.shortcut = (
            CausalConv3d(in_dim, out_dim, 1)
            if in_dim != out_dim
            else None
        )

    def __call__(self, x):
        h = self.shortcut(x) if self.shortcut is not None else x
        return self.residual(x) + h


class ResidualBlockLayers(nn.Module):
    """The sequential layers inside a ResidualBlock.

    PyTorch stores these as nn.Sequential with indices 0-6:
    [0] RMS_norm, [1] SiLU, [2] CausalConv3d, [3] RMS_norm, [4] SiLU, [5] Dropout, [6] CausalConv3d
    We use matching attribute names for weight compatibility.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Indices match PyTorch nn.Sequential indices for weight key compat
        # Index 0: RMS_norm
        self.layer_0 = RMS_norm(in_dim)
        # Index 2: CausalConv3d
        self.layer_2 = CausalConv3d(in_dim, out_dim, 3, padding=1)
        # Index 3: RMS_norm
        self.layer_3 = RMS_norm(out_dim)
        # Index 6: CausalConv3d
        self.layer_6 = CausalConv3d(out_dim, out_dim, 3, padding=1)

    def __call__(self, x):
        x = self.layer_0(x)
        x = nn.silu(x)
        x = self.layer_2(x)
        mx.eval(x)  # Eval between convolutions to limit graph size
        x = self.layer_3(x)
        x = nn.silu(x)
        x = self.layer_6(x)
        return x


class AttentionBlock(nn.Module):
    """2D self-attention applied per frame. Input: [B, T, H, W, C]."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        # Conv2d as linear per spatial position — weight [O, H, W, I] for MLX
        # to_qkv: dim -> 3*dim, proj: dim -> dim (1x1 conv2d)
        self.to_qkv_weight = mx.zeros((3 * dim, 1, 1, dim))
        self.to_qkv_bias = mx.zeros((3 * dim,))
        self.proj_weight = mx.zeros((dim, 1, 1, dim))
        self.proj_bias = mx.zeros((dim,))

    def __call__(self, x):
        # x: [B, T, H, W, C]
        identity = x
        B, T, H, W, C = x.shape

        # Apply per frame: merge B and T
        x = x.reshape(B * T, H, W, C)
        x = self.norm(x)

        # QKV via 1x1 conv2d (equivalent to linear on last dim)
        qkv = mx.conv_general(x, self.to_qkv_weight) + self.to_qkv_bias  # [BT, H, W, 3C]
        qkv = qkv.reshape(B * T, H * W, 3 * C)
        q, k, v = mx.split(qkv, 3, axis=-1)  # each [BT, HW, C]

        # Single-head attention
        q = q[:, None, :, :]  # [BT, 1, HW, C]
        k = k[:, None, :, :]
        v = v[:, None, :, :]

        scale = C ** -0.5
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)  # [BT, 1, HW, C]
        out = out.squeeze(1).reshape(B * T, H, W, C)

        # Project output
        out = mx.conv_general(out, self.proj_weight) + self.proj_bias  # [BT, H, W, C]
        out = out.reshape(B, T, H, W, C)
        return out + identity


class DupUp3D(nn.Module):
    """Upsample by duplicating channels and reshaping. No learnable parameters."""

    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        self.repeats = out_channels * self.factor // in_channels

    def __call__(self, x, first_chunk=False):
        # x: [B, T, H, W, C]
        B, T, H, W, C = x.shape

        # Repeat channels
        x = mx.repeat(x, self.repeats, axis=-1)  # [B, T, H, W, C*repeats]

        # Reshape to [B, T, H, W, out_C, factor_t, factor_s, factor_s]
        x = x.reshape(B, T, H, W, self.out_channels, self.factor_t, self.factor_s, self.factor_s)

        # Permute to interleave: [B, T, factor_t, H, factor_s, W, factor_s, out_C]
        x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)

        # Reshape to final: [B, T*factor_t, H*factor_s, W*factor_s, out_C]
        x = x.reshape(B, T * self.factor_t, H * self.factor_s, W * self.factor_s, self.out_channels)

        if first_chunk:
            x = x[:, self.factor_t - 1:, :, :, :]
        return x


class Resample(nn.Module):
    """Spatial up/downsampling with optional temporal up/downsampling."""

    def __init__(self, dim, mode):
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            # resample.0 = Upsample (no params), resample.1 = Conv2d
            self.resample_weight = mx.zeros((dim, 3, 3, dim))  # Conv2d [O, H, W, I]
            self.resample_bias = mx.zeros((dim,))
        elif mode == "upsample3d":
            self.resample_weight = mx.zeros((dim, 3, 3, dim))
            self.resample_bias = mx.zeros((dim,))
            # time_conv: CausalConv3d(dim, dim*2, (3,1,1), padding=(1,0,0))
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _upsample2x(self, x):
        """Nearest-neighbor 2x spatial upsample. x: [N, H, W, C]."""
        N, H, W, C = x.shape
        # Repeat along H and W axes separately
        x = mx.repeat(x, repeats=2, axis=1)  # [N, 2H, W, C]
        x = mx.repeat(x, repeats=2, axis=2)  # [N, 2H, 2W, C]
        return x

    def _conv2d(self, x):
        """Apply the Conv2d with padding=1. x: [N, H, W, C]."""
        x = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
        return mx.conv_general(x, self.resample_weight) + self.resample_bias

    def __call__(self, x, first_chunk=False):
        # x: [B, T, H, W, C]
        B, T, H, W, C = x.shape

        if self.mode == "upsample3d":
            if first_chunk and T > 1:
                # Match official chunked behavior: the first frame bypasses
                # time_conv entirely (only spatial upsample). Remaining frames
                # go through time_conv with causal zero-padding, which
                # naturally gives each frame the same limited temporal context
                # as the official frame-by-frame decode with caching.
                first_frame = x[:, 0:1]  # [B, 1, H, W, C]
                rest = x[:, 1:]          # [B, T-1, H, W, C]

                # time_conv on remaining frames (causal pad gives zero context
                # before rest[0], matching the official "Rep" cache path)
                tc_out = self.time_conv(rest)  # [B, T-1, H, W, 2C]
                tc_out = tc_out.reshape(B, T - 1, H, W, 2, C)
                stream0 = tc_out[:, :, :, :, 0, :]
                stream1 = tc_out[:, :, :, :, 1, :]
                interleaved = mx.stack([stream0, stream1], axis=2)
                interleaved = interleaved.reshape(B, (T - 1) * 2, H, W, C)

                # first_frame (1) + interleaved (2*(T-1)) = 2T-1 frames
                x = mx.concatenate([first_frame, interleaved], axis=1)
            elif self.mode == "upsample3d":
                # Non-first-chunk or single frame: time_conv all frames
                tc_out = self.time_conv(x)  # [B, T, H, W, 2C]
                tc_out = tc_out.reshape(B, T, H, W, 2, C)
                stream0 = tc_out[:, :, :, :, 0, :]
                stream1 = tc_out[:, :, :, :, 1, :]
                x = mx.stack([stream0, stream1], axis=2)
                x = x.reshape(B, T * 2, H, W, C)

            mx.eval(x)
            T = x.shape[1]

        # Spatial upsample in temporal chunks to limit peak memory
        chunk_size = 8
        chunks = []
        for t_start in range(0, T, chunk_size):
            t_end = min(t_start + chunk_size, T)
            x_chunk = x[:, t_start:t_end].reshape(-1, H, W, C)
            x_chunk = self._upsample2x(x_chunk)
            x_chunk = self._conv2d(x_chunk)
            mx.eval(x_chunk)
            chunks.append(x_chunk)

        x = mx.concatenate(chunks, axis=0)
        H2, W2 = x.shape[1], x.shape[2]
        x = x.reshape(B, T, H2, W2, C)
        return x


class Up_ResidualBlock(nn.Module):
    """Upsampling residual block with optional DupUp3D shortcut."""

    def __init__(self, in_dim, out_dim, num_res_blocks, temperal_upsample=False, up_flag=False):
        super().__init__()
        self.up_flag = up_flag

        # DupUp3D shortcut (no learnable params)
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim, out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None

        # Main path: ResidualBlocks + optional Resample
        blocks = []
        dim_in = in_dim
        for _ in range(num_res_blocks):
            blocks.append(ResidualBlock(dim_in, out_dim))
            dim_in = out_dim

        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            blocks.append(Resample(out_dim, mode=mode))

        self.upsamples = blocks

    def __call__(self, x, first_chunk=False):
        x_main = x
        for module in self.upsamples:
            if isinstance(module, Resample):
                x_main = module(x_main, first_chunk)
            else:
                x_main = module(x_main)
            mx.eval(x_main)  # Limit graph size per sub-block

        if self.avg_shortcut is not None:
            x_shortcut = self.avg_shortcut(x, first_chunk)
            mx.eval(x_shortcut)
            return x_main + x_shortcut
        return x_main


class Decoder3d(nn.Module):
    """Wan2.2 3D VAE Decoder."""

    def __init__(
        self,
        dim=256,
        z_dim=48,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        temperal_upsample=(True, True, False),
    ):
        super().__init__()
        # Compute layer dimensions
        dims = [dim * dim_mult[-1]] + [dim * m for m in reversed(dim_mult)]

        # Initial conv
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # Middle blocks
        self.middle = [
            ResidualBlock(dims[0], dims[0]),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0]),
        ]

        # Upsample blocks
        self.upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up = temperal_upsample[i] if i < len(temperal_upsample) else False
            self.upsamples.append(Up_ResidualBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks + 1,
                temperal_upsample=t_up,
                up_flag=(i != len(dim_mult) - 1),
            ))

        # Output head: [RMS_norm, SiLU, CausalConv3d]
        self.head = Head22(dims[-1])

    def __call__(self, x, first_chunk=False):
        # x: [B, T, H, W, C=z_dim]
        x = self.conv1(x)

        for layer in self.middle:
            x = layer(x)
        mx.eval(x)  # Evaluate to limit graph size

        for i, layer in enumerate(self.upsamples):
            x = layer(x, first_chunk)
            mx.eval(x)  # Evaluate after each upsample block

        x = self.head(x)
        return x


class Head22(nn.Module):
    """Decoder output head: RMS_norm → SiLU → CausalConv3d(dim, 12, 3).

    PyTorch key mapping: head.0 = RMS_norm, head.2 = CausalConv3d
    (index 1 = SiLU has no params)
    """

    def __init__(self, dim, out_channels=12):
        super().__init__()
        # Index 0: RMS_norm
        self.layer_0 = RMS_norm(dim)
        # Index 2: CausalConv3d
        self.layer_2 = CausalConv3d(dim, out_channels, 3, padding=1)

    def __call__(self, x):
        x = self.layer_0(x)
        x = nn.silu(x)
        x = self.layer_2(x)
        return x


class Wan22VAEDecoder(nn.Module):
    """Full Wan2.2 VAE decoder with normalization and unpatchify."""

    def __init__(self, z_dim=48, dim=160, dec_dim=256):
        super().__init__()
        self.z_dim = z_dim
        # conv2: 1x1x1 conv before decoder
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dim=dec_dim,
            z_dim=z_dim,
            dim_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            temperal_upsample=(True, True, False),
        )

    def __call__(self, z):
        """Decode latents to video.

        Args:
            z: [B, T, H, W, C=48] latent tensor (already denormalized)

        Returns:
            video: [B, T', H', W', 3] decoded RGB in [-1, 1]
        """
        x = self.conv2(z)

        # All-at-once decode with first_chunk=True to trim extra temporal
        # frames from causal padding (matches PyTorch's chunked behavior)
        out = self.decoder(x, first_chunk=True)

        # Unpatchify: 12 channels → 3 RGB (spatial 2×2)
        out = _unpatchify(out, patch_size=2)

        return mx.clip(out, -1.0, 1.0)


def denormalize_latents(z, mean=None, std=None):
    """Denormalize latents: z = z / (1/std) + mean."""
    if mean is None:
        mean = VAE22_MEAN
    if std is None:
        std = VAE22_STD
    inv_scale = std  # scale was 1/std, so divide by scale = multiply by std
    return z * inv_scale.reshape(1, 1, 1, 1, -1) + mean.reshape(1, 1, 1, 1, -1)


def _unpatchify(x, patch_size=2):
    """Convert from packed channels to spatial: [B, T, H, W, C*p*p] → [B, T, H*p, W*p, C//(p*p)]
    Actually: [B, T, H, W, 12] → [B, T, H*2, W*2, 3]
    PyTorch: b (c r q) f h w -> b c f (h q) (w r) with q=p, r=p
    In channels-last: [B, T, H, W, C*r*q] -> [B, T, H*q, W*r, C]
    """
    if patch_size == 1:
        return x
    B, T, H, W, Cpacked = x.shape
    C = Cpacked // (patch_size * patch_size)
    # Reshape: [B, T, H, W, r, q, C] then rearrange to [B, T, H*q, W*r, C]
    # PyTorch patchify: "b c f (h q) (w r) -> b (c r q) f h w" — so c is packed as (c, r, q)
    # Unpatchify reverses: [B, T, H, W, (C, r, q)] -> [B, T, H, q, W, r, C]
    x = x.reshape(B, T, H, W, C, patch_size, patch_size)
    # Rearrange: put q next to H, r next to W
    x = x.transpose(0, 1, 2, 6, 3, 5, 4)  # [B, T, H, q, W, r, C]
    x = x.reshape(B, T, H * patch_size, W * patch_size, C)
    return x


def sanitize_wan22_vae_weights(weights: dict) -> dict:
    """Convert PyTorch Wan2.2 VAE weights to MLX format.

    Only keeps decoder + conv2 weights (encoder/conv1 not needed for generation).
    Transposes conv weights from channels-first to channels-last.
    Squeezes RMS_norm gamma from (dim, 1, 1, 1) or (dim, 1, 1) to (dim,).
    Maps PyTorch nn.Sequential indices to our named layers.
    """
    sanitized = {}

    for key, value in weights.items():
        # Skip encoder and conv1 (encoder-only)
        if key.startswith("encoder.") or key.startswith("conv1."):
            continue

        new_key = key

        # Map nn.Sequential indexed layers to our named attributes
        # ResidualBlockLayers: indices 0, 2, 3, 6 → _layer_0, _layer_2, _layer_3, _layer_6
        # Head22: indices 0, 2 → _layer_0, _layer_2
        for idx in ["0", "2", "3", "6"]:
            # Match patterns like "residual.0.gamma" → "residual.layer_0.gamma"
            # or "head.0.gamma" → "head.layer_0.gamma"
            old_pattern = f".residual.{idx}."
            new_pattern = f".residual.layer_{idx}."
            new_key = new_key.replace(old_pattern, new_pattern)

        # Head layer mapping: head.0.gamma → head.layer_0.gamma, head.2.weight → head.layer_2.weight
        for idx in ["0", "2"]:
            old_pattern = f".head.{idx}."
            new_pattern = f".head.layer_{idx}."
            new_key = new_key.replace(old_pattern, new_pattern)

        # Map Resample Conv2d: resample.1.weight → resample_weight, resample.1.bias → resample_bias
        if ".resample.1.weight" in new_key:
            new_key = new_key.replace(".resample.1.weight", ".resample_weight")
        elif ".resample.1.bias" in new_key:
            new_key = new_key.replace(".resample.1.bias", ".resample_bias")

        # Map AttentionBlock Conv2d weights
        if ".to_qkv.weight" in new_key:
            new_key = new_key.replace(".to_qkv.weight", ".to_qkv_weight")
        elif ".to_qkv.bias" in new_key:
            new_key = new_key.replace(".to_qkv.bias", ".to_qkv_bias")
        elif ".proj.weight" in new_key and "time_projection" not in new_key:
            new_key = new_key.replace(".proj.weight", ".proj_weight")
        elif ".proj.bias" in new_key and "time_projection" not in new_key:
            new_key = new_key.replace(".proj.bias", ".proj_bias")

        # Transpose conv weights to channels-last
        is_weight = new_key.endswith(".weight") or new_key.endswith("_weight")
        if is_weight:
            if value.ndim == 5:
                # Conv3d: [O, I, D, H, W] → [O, D, H, W, I]
                value = np.transpose(np.array(value), (0, 2, 3, 4, 1))
                value = mx.array(value)
            elif value.ndim == 4:
                # Conv2d: [O, I, H, W] → [O, H, W, I]
                value = np.transpose(np.array(value), (0, 2, 3, 1))
                value = mx.array(value)

        # Squeeze RMS_norm gamma: (dim, 1, 1, 1) or (dim, 1, 1) → (dim,)
        if "gamma" in new_key:
            value = mx.array(np.array(value).squeeze())

        sanitized[new_key] = value

    return sanitized
