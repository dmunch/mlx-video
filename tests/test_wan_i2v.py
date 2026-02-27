"""Tests for Wan2.2 I2V-14B support."""

import mlx.core as mx
import numpy as np
import pytest

from wan_test_helpers import _make_tiny_config


def _make_tiny_i2v_config():
    """Create a tiny I2V-14B config for testing."""
    config = _make_tiny_config()
    config.model_type = "i2v"
    config.in_dim = 9  # 4 noise + 4 image + 1 mask (scaled down from 16+16+4=36)
    config.out_dim = 4
    config.vae_z_dim = 4
    config.vae_stride = (4, 8, 8)
    config.dual_model = True
    config.boundary = 0.900
    config.sample_shift = 5.0
    config.sample_guide_scale = (3.5, 3.5)
    config.teacache_coefficients = None
    return config


class TestI2VConfig:
    """Test I2V-14B config preset."""

    def test_wan22_i2v_14b_preset(self):
        from mlx_video.models.wan.config import WanModelConfig

        config = WanModelConfig.wan22_i2v_14b()
        assert config.model_type == "i2v"
        assert config.in_dim == 36
        assert config.out_dim == 16
        assert config.dim == 5120
        assert config.num_layers == 40
        assert config.dual_model is True
        assert config.boundary == 0.900
        assert config.sample_shift == 5.0
        assert config.sample_guide_scale == (3.5, 3.5)
        assert config.vae_stride == (4, 8, 8)
        assert config.vae_z_dim == 16
        assert config.teacache_coefficients is None

    def test_i2v_vs_t2v_differences(self):
        from mlx_video.models.wan.config import WanModelConfig

        i2v = WanModelConfig.wan22_i2v_14b()
        t2v = WanModelConfig.wan22_t2v_14b()

        assert i2v.model_type == "i2v"
        assert t2v.model_type == "t2v"
        assert i2v.in_dim == 36 and t2v.in_dim == 16
        assert i2v.boundary == 0.900 and t2v.boundary == 0.875
        assert i2v.sample_shift == 5.0 and t2v.sample_shift == 12.0

    def test_i2v_serialization_roundtrip(self):
        from mlx_video.models.wan.config import WanModelConfig

        config = WanModelConfig.wan22_i2v_14b()
        d = config.to_dict()
        restored = WanModelConfig.from_dict(d)
        assert restored.model_type == "i2v"
        assert restored.in_dim == 36
        assert restored.boundary == 0.900


class TestModelYParameter:
    """Test y parameter channel concatenation in WanModel."""

    def test_forward_without_y(self):
        """Standard T2V forward pass (no y) still works."""
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
        assert out[0].shape == (C, F, H, W)

    def test_forward_with_y(self):
        """I2V forward pass with y channel concatenation."""
        from mlx_video.models.wan.model import WanModel

        config = _make_tiny_i2v_config()
        model = WanModel(config)

        C_noise = 4  # noise channels
        C_y = 5  # mask (1) + image (4)
        F, H, W = 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        x_list = [mx.random.normal((C_noise, F, H, W))]
        y_list = [mx.random.normal((C_y, F, H, W))]
        t = mx.array([500.0])
        context = [mx.random.normal((6, config.text_dim))]

        out = model(x_list, t, context, seq_len, y=y_list)
        mx.eval(out[0])
        # Output should match noise channels (out_dim), not concatenated in_dim
        assert out[0].shape == (config.out_dim, F, H, W)

    def test_y_none_is_noop(self):
        """Passing y=None should be identical to not passing y."""
        from mlx_video.models.wan.model import WanModel

        config = _make_tiny_config()
        model = WanModel(config)

        C, F, H, W = config.in_dim, 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        mx.random.seed(42)
        x = mx.random.normal((C, F, H, W))
        t = mx.array([500.0])
        ctx = [mx.random.normal((6, config.text_dim))]

        out1 = model([x], t, ctx, seq_len)[0]
        out2 = model([x], t, ctx, seq_len, y=None)[0]
        mx.eval(out1, out2)
        assert mx.allclose(out1, out2, atol=1e-5).item()

    def test_batched_cfg_with_y(self):
        """Batched CFG (B=2) with y should work."""
        from mlx_video.models.wan.model import WanModel

        config = _make_tiny_i2v_config()
        model = WanModel(config)

        C_noise, C_y = 4, 5
        F, H, W = 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        latents = mx.random.normal((C_noise, F, H, W))
        y = mx.random.normal((C_y, F, H, W))
        t = mx.array([500.0, 500.0])
        ctx = [mx.random.normal((6, config.text_dim)), mx.random.normal((6, config.text_dim))]

        out = model([latents, latents], t, ctx, seq_len, y=[y, y])
        mx.eval(out[0], out[1])
        assert len(out) == 2
        assert out[0].shape == (config.out_dim, F, H, W)
        assert out[1].shape == (config.out_dim, F, H, W)


class TestVAEEncoder:
    """Test Wan2.1 VAE encoder."""

    def test_encoder3d_instantiation(self):
        from mlx_video.models.wan.vae import Encoder3d

        enc = Encoder3d(dim=32, z_dim=8)  # z_dim=8 (will output 8ch, but WanVAE wraps with z*2)
        assert enc.conv1 is not None
        assert len(enc.downsamples) > 0
        assert len(enc.middle) == 3

    def test_encoder3d_output_shape(self):
        """Encoder should downsample spatially by 8x and temporally by 4x."""
        from mlx_video.models.wan.vae import Encoder3d

        enc = Encoder3d(dim=32, z_dim=8)
        # Random input: [B=1, 3, T=5, H=32, W=32]
        x = mx.random.normal((1, 3, 5, 32, 32))
        out = enc(x)
        mx.eval(out)
        # With default dim_mult=[1,2,4,4] and temporal_downsample=[True,True,False]:
        # Spatial: 32 -> 16 -> 8 -> 4 (3 spatial downsamples)
        # Temporal: 5 -> 3 -> 2 (2 temporal downsamples: downsample3d stride 2)
        assert out.shape[0] == 1
        assert out.shape[1] == 8  # z_dim
        assert out.shape[3] == 32 // 8  # spatial /8
        assert out.shape[4] == 32 // 8

    def test_wan_vae_encode(self):
        """WanVAE with encoder=True should produce normalized latents."""
        from mlx_video.models.wan.vae import WanVAE

        vae = WanVAE(z_dim=16, encoder=True)
        # Input: [B=1, 3, T=5, H=32, W=32]
        x = mx.random.normal((1, 3, 5, 32, 32))
        z = vae.encode(x)
        mx.eval(z)
        assert z.shape[0] == 1
        assert z.shape[1] == 16  # z_dim

    def test_wan_vae_encoder_flag(self):
        """WanVAE without encoder flag should not have encoder attribute."""
        from mlx_video.models.wan.vae import WanVAE

        vae_no_enc = WanVAE(z_dim=4, encoder=False)
        assert not hasattr(vae_no_enc, 'encoder')

        vae_enc = WanVAE(z_dim=4, encoder=True)
        assert hasattr(vae_enc, 'encoder')


class TestResampleDownsample:
    """Test downsample modes in Resample."""

    def test_downsample2d(self):
        from mlx_video.models.wan.vae import Resample

        r = Resample(dim=16, mode="downsample2d")
        x = mx.random.normal((1, 16, 2, 8, 8))
        out = r(x)
        mx.eval(out)
        # Spatial /2, temporal unchanged, channels same
        assert out.shape == (1, 16, 2, 4, 4)

    def test_downsample3d(self):
        from mlx_video.models.wan.vae import Resample

        r = Resample(dim=16, mode="downsample3d")
        x = mx.random.normal((1, 16, 4, 8, 8))
        out = r(x)
        mx.eval(out)
        # Spatial /2, temporal /2, channels same
        assert out.shape == (1, 16, 2, 4, 4)

    def test_upsample2d_still_works(self):
        from mlx_video.models.wan.vae import Resample

        r = Resample(dim=16, mode="upsample2d")
        x = mx.random.normal((1, 16, 2, 4, 4))
        out = r(x)
        mx.eval(out)
        assert out.shape == (1, 8, 2, 8, 8)

    def test_upsample3d_still_works(self):
        from mlx_video.models.wan.vae import Resample

        r = Resample(dim=16, mode="upsample3d")
        x = mx.random.normal((1, 16, 2, 4, 4))
        out = r(x)
        mx.eval(out)
        assert out.shape == (1, 8, 4, 8, 8)


class TestI2VMaskConstruction:
    """Test mask construction for I2V-14B."""

    def test_mask_shape(self):
        """I2V-14B mask should have 4 channels with correct temporal structure."""
        num_frames = 81
        h_latent, w_latent = 10, 18  # example latent dims
        t_latent = (num_frames - 1) // 4 + 1  # = 21

        # Build mask following reference logic
        msk = mx.ones((1, num_frames, h_latent, w_latent))
        msk = mx.concatenate([msk[:, :1], mx.zeros((1, num_frames - 1, h_latent, w_latent))], axis=1)
        msk = mx.concatenate([mx.repeat(msk[:, :1], 4, axis=1), msk[:, 1:]], axis=1)
        msk = msk.reshape(1, msk.shape[1] // 4, 4, h_latent, w_latent)
        msk = msk.transpose(0, 2, 1, 3, 4)[0]  # [4, T_lat, H_lat, W_lat]

        assert msk.shape == (4, t_latent, h_latent, w_latent)

    def test_mask_values(self):
        """First temporal position should be 1, rest 0."""
        num_frames = 9
        h_latent, w_latent = 4, 4
        t_latent = (num_frames - 1) // 4 + 1  # = 3

        msk = mx.ones((1, num_frames, h_latent, w_latent))
        msk = mx.concatenate([msk[:, :1], mx.zeros((1, num_frames - 1, h_latent, w_latent))], axis=1)
        msk = mx.concatenate([mx.repeat(msk[:, :1], 4, axis=1), msk[:, 1:]], axis=1)
        msk = msk.reshape(1, msk.shape[1] // 4, 4, h_latent, w_latent)
        msk = msk.transpose(0, 2, 1, 3, 4)[0]

        mx.eval(msk)
        # First temporal position: all 4 channels should be 1
        assert mx.all(msk[:, 0] == 1.0).item()
        # Rest: all should be 0
        assert mx.all(msk[:, 1:] == 0.0).item()

    def test_y_tensor_shape(self):
        """y = concat([mask_4ch, encoded_video_16ch]) should be 20 channels."""
        mask = mx.zeros((4, 5, 10, 18))
        encoded = mx.zeros((16, 5, 10, 18))
        y = mx.concatenate([mask, encoded], axis=0)
        assert y.shape == (20, 5, 10, 18)
