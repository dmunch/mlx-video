"""3D VAE Decoder for Wan2.2 (VAE 2.1, compression 4×8×8)."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np


CACHE_T = 2

# Per-channel normalization statistics for z_dim=16
VAE_MEAN = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]
VAE_STD = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]


class CausalConv3d(nn.Module):
    """3D convolution with causal temporal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self._causal_pad_t = 2 * padding[0]  # Causal: pad only before in time
        self._pad_h = padding[1]
        self._pad_w = padding[2]

        # MLX Conv3d: weight shape [O, D, H, W, I]
        self.weight = mx.zeros((out_channels, kernel_size[0], kernel_size[1], kernel_size[2], in_channels))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [B, C, T, H, W] (channel-first format)
        """
        b, c, t, h, w = x.shape

        # Causal temporal padding: only pad before
        if self._causal_pad_t > 0:
            pad_t = mx.zeros((b, c, self._causal_pad_t, h, w), dtype=x.dtype)
            x = mx.concatenate([pad_t, x], axis=2)

        # Spatial padding
        if self._pad_h > 0 or self._pad_w > 0:
            pad_h = self._pad_h
            pad_w = self._pad_w
            x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)])

        # Convert to MLX conv format: [B, T, H, W, C]
        x = x.transpose(0, 2, 3, 4, 1)

        # Manual 3D convolution via reshape to use conv2d per time slice
        # or use general einsum-based approach
        out = self._conv3d(x)

        # Convert back: [B, T', H', W', O] -> [B, O, T', H', W']
        return out.transpose(0, 4, 1, 2, 3)

    def _conv3d(self, x: mx.array) -> mx.array:
        """Implement 3D conv using sliding window over time dimension.

        Args:
            x: [B, T, H, W, C_in]

        Returns:
            [B, T_out, H_out, W_out, C_out]
        """
        b, t, h, w, c_in = x.shape
        kt, kh, kw = self.kernel_size
        st, sh, sw = self.stride
        c_out = self.weight.shape[0]

        t_out = (t - kt) // st + 1
        h_out = (h - kh) // sh + 1
        w_out = (w - kw) // sw + 1

        # Unfold time dimension and use 2D conv per time window
        # Pre-reshape weight once instead of per timestep
        w_2d = self.weight.transpose(0, 2, 3, 1, 4).reshape(c_out, kh, kw, kt * c_in)
        outputs = []
        for t_i in range(t_out):
            t_start = t_i * st
            # Gather temporal window: [B, kt, H, W, C_in]
            window = x[:, t_start : t_start + kt]
            # Reshape to [B, H, W, kt * C_in]
            window = window.transpose(0, 2, 3, 1, 4).reshape(b, h, w, kt * c_in)

            # 2D convolution
            out_2d = _conv2d(window, w_2d, self.bias, (sh, sw))
            outputs.append(out_2d)

        return mx.stack(outputs, axis=1)


def _conv2d(x: mx.array, weight: mx.array, bias: mx.array, stride: tuple) -> mx.array:
    """2D convolution helper.

    Args:
        x: [B, H, W, C_in]
        weight: [C_out, kH, kW, C_in]
        bias: [C_out]
        stride: (sH, sW)

    Returns:
        [B, H_out, W_out, C_out]
    """
    # Use nn.Conv2d internally via functional approach
    out = mx.conv2d(x, weight, stride=stride)
    return out + bias


class RMSNormChannel(nn.Module):
    """Channel-first RMS normalization."""

    def __init__(self, dim: int, channel_first: bool = True):
        super().__init__()
        self.channel_first = channel_first
        self.scale = dim**0.5
        if channel_first:
            self.gamma = mx.ones((dim, 1, 1))
        else:
            self.gamma = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        norm_dim = 1 if self.channel_first else -1
        x_norm = x / (mx.sqrt(mx.sum(x * x, axis=norm_dim, keepdims=True) / x.shape[norm_dim]) + 1e-8)
        gamma = self.gamma
        if self.channel_first and x.ndim == 5:
            # Reshape (C, 1, 1) -> (1, C, 1, 1, 1) for 5D input
            gamma = gamma[None, :, :, :, None]
        return x_norm * self.scale * gamma


class ResidualBlock(nn.Module):
    """Residual block with causal 3D convolutions."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm1 = RMSNormChannel(in_dim, channel_first=True)
        self.conv1 = CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = RMSNormChannel(out_dim, channel_first=True)
        self.conv2 = CausalConv3d(out_dim, out_dim, 3, padding=1)
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else None

    def __call__(self, x: mx.array) -> mx.array:
        h = x if self.shortcut is None else self.shortcut(x)
        x = nn.silu(self.norm1(x))
        x = self.conv1(x)
        x = nn.silu(self.norm2(x))
        x = self.conv2(x)
        return x + h


class AttentionBlock(nn.Module):
    """Single-head spatial self-attention."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNormChannel(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        """x: [B, C, T, H, W]"""
        identity = x
        b, c, t, h, w = x.shape

        # Process each frame independently
        x = x.transpose(0, 2, 3, 4, 1).reshape(b * t, h, w, c)  # [BT, H, W, C]
        x_norm = self._channel_rms_norm(x)

        # QKV
        qkv = self.to_qkv(x_norm)  # [BT, H, W, 3C]
        qkv = qkv.reshape(b * t, h * w, 3, c).transpose(2, 0, 1, 3)  # [3, BT, HW, C]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        q = q[:, None, :, :]  # [BT, 1, HW, C]
        k = k[:, None, :, :]
        v = v[:, None, :, :]
        scale = c**-0.5
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out.squeeze(1)  # [BT, HW, C]
        out = out.reshape(b * t, h, w, c)

        out = self.proj(out)
        out = out.reshape(b, t, h, w, c).transpose(0, 4, 1, 2, 3)  # [B, C, T, H, W]
        return out + identity

    def _channel_rms_norm(self, x: mx.array) -> mx.array:
        """x: [B, H, W, C] -> normalize over C."""
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-8)
        return x / rms * (x.shape[-1] ** 0.5)


class Upsample2d(nn.Module):
    """2x spatial upsampling with convolution."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim // 2, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        """x: [B, C, T, H, W]"""
        b, c, t, h, w = x.shape
        # Reshape to process per-frame: [BT, H, W, C]
        x = x.transpose(0, 2, 3, 4, 1).reshape(b * t, h, w, c)
        # Nearest neighbor 2x upsample
        x = mx.repeat(x, 2, axis=1)  # [BT, 2H, W, C]
        x = mx.repeat(x, 2, axis=2)  # [BT, 2H, 2W, C]
        x = self.conv(x)  # [BT, 2H, 2W, C//2]
        c_out = x.shape[-1]
        return x.reshape(b, t, h * 2, w * 2, c_out).transpose(0, 4, 1, 2, 3)


class Upsample3d(nn.Module):
    """2x spatial + 2x temporal upsampling."""

    def __init__(self, dim: int):
        super().__init__()
        self.spatial_conv = nn.Conv2d(dim, dim // 2, 3, padding=1)
        self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

    def __call__(self, x: mx.array) -> mx.array:
        """x: [B, C, T, H, W]"""
        b, c, t, h, w = x.shape

        # Temporal upsample via learned conv
        x_t = self.time_conv(x)  # [B, 2C, T, H, W]
        # Reshape to interleave: [B, 2, C, T, H, W] -> [B, C, 2T, H, W]
        x_t = x_t.reshape(b, 2, c, t, h, w)
        x_t = mx.concatenate(
            [x_t[:, 0:1], x_t[:, 1:2]], axis=3
        ).reshape(b, c, t * 2, h, w)

        # Spatial upsample per frame
        t2 = t * 2
        x_s = x_t.transpose(0, 2, 3, 4, 1).reshape(b * t2, h, w, c)
        x_s = mx.repeat(x_s, 2, axis=1)
        x_s = mx.repeat(x_s, 2, axis=2)
        x_s = self.spatial_conv(x_s)
        c_out = x_s.shape[-1]
        return x_s.reshape(b, t2, h * 2, w * 2, c_out).transpose(0, 4, 1, 2, 3)


class Decoder3d(nn.Module):
    """3D VAE Decoder matching Wan2.1 architecture."""

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 4,
        dim_mult: list = None,
        num_res_blocks: int = 2,
        temporal_upsample: list = None,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if temporal_upsample is None:
            temporal_upsample = [False, True, True]

        # Compute channel dimensions (reversed from encoder)
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        # Init conv
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # Middle blocks
        self.mid_res1 = ResidualBlock(dims[0], dims[0])
        self.mid_attn = AttentionBlock(dims[0])
        self.mid_res2 = ResidualBlock(dims[0], dims[0])

        # Upsample blocks
        self.up_blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            block_in_dim = in_dim // 2 if i in (1, 2, 3) else in_dim
            blocks = []
            for j in range(num_res_blocks + 1):
                blocks.append(ResidualBlock(block_in_dim if j == 0 else out_dim, out_dim))
            if i != len(dim_mult) - 1:
                if temporal_upsample[i]:
                    blocks.append(Upsample3d(out_dim))
                else:
                    blocks.append(Upsample2d(out_dim))
            self.up_blocks.append(blocks)

        # Output
        self.norm_out = RMSNormChannel(dims[-1], channel_first=True)
        self.conv_out = CausalConv3d(dims[-1], 3, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Latent tensor [B, z_dim, T, H, W]

        Returns:
            Video tensor [B, 3, T_out, H_out, W_out] in [-1, 1]
        """
        x = self.conv1(x)

        # Middle
        x = self.mid_res1(x)
        x = self.mid_attn(x)
        x = self.mid_res2(x)

        # Upsample
        for stage_blocks in self.up_blocks:
            for block in stage_blocks:
                x = block(x)

        # Output
        x = nn.silu(self.norm_out(x))
        x = self.conv_out(x)
        return x


class WanVAE(nn.Module):
    """Wan2.1 VAE wrapper with per-channel normalization."""

    def __init__(self, z_dim: int = 16):
        super().__init__()
        self.z_dim = z_dim
        self.mean = mx.array(VAE_MEAN)
        self.std = mx.array(VAE_STD)
        self.inv_std = 1.0 / self.std

        # conv2 is the 1x1x1 projection before decoder
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim=96, z_dim=z_dim)

    def decode(self, z: mx.array) -> mx.array:
        """Decode latent to video.

        Args:
            z: Normalized latent [B, z_dim, T, H, W]

        Returns:
            Video [B, 3, T_out, H_out, W_out] clamped to [-1, 1]
        """
        # Denormalize
        mean = self.mean.reshape(1, -1, 1, 1, 1)
        inv_std = self.inv_std.reshape(1, -1, 1, 1, 1)
        z = z / inv_std + mean

        # Project and decode
        x = self.conv2(z)
        out = self.decoder(x)
        return mx.clip(out, -1, 1)
