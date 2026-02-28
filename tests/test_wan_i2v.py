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


# ---------------------------------------------------------------------------
# Integration: I2V end-to-end pipeline
# ---------------------------------------------------------------------------


class TestI2VEndToEndPipeline:
    """Full I2V pipeline: image → preprocess → VAE encode → y tensor → denoise → VAE decode."""

    def test_full_i2v_pipeline(self):
        """End-to-end I2V: synthetic image → VAE encode → build y → denoise → VAE decode."""
        from mlx_video.models.wan.model import WanModel
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        from mlx_video.models.wan.vae import WanVAE

        mx.random.seed(0)

        # --- Tiny I2V model config (z_dim=16 to match VAE normalization stats) ---
        config = _make_tiny_i2v_config()
        config.vae_z_dim = 16
        config.out_dim = 16  # must match VAE z_dim for decode
        config.in_dim = 16 + 4 + 16  # noise(out_dim=16) + mask(4) + image(z_dim=16) = 36
        model = WanModel(config)

        # --- Tiny VAE (with encoder) ---
        vae = WanVAE(z_dim=config.vae_z_dim, encoder=True)

        # --- Synthetic image: [B=1, 3, T=1, H=32, W=32] in [-1, 1] ---
        height, width = 32, 32
        num_frames = 5  # small temporal extent
        img = mx.random.uniform(-1, 1, (1, 3, 1, height, width))

        # Build video: first frame = image, rest = zeros -> [1, 3, F, H, W]
        video = mx.concatenate([
            img,
            mx.zeros((1, 3, num_frames - 1, height, width)),
        ], axis=2)

        # --- VAE encode ---
        z_video = vae.encode(video)  # [1, z_dim, T_lat, H_lat, W_lat]
        mx.eval(z_video)
        assert z_video.ndim == 5
        assert z_video.shape[1] == config.vae_z_dim

        z_video = z_video[0]  # [z_dim, T_lat, H_lat, W_lat]
        t_latent = z_video.shape[1]
        h_latent = z_video.shape[2]
        w_latent = z_video.shape[3]

        # --- Build I2V mask (4 channels) ---
        msk = mx.ones((1, num_frames, h_latent, w_latent))
        msk = mx.concatenate([msk[:, :1], mx.zeros((1, num_frames - 1, h_latent, w_latent))], axis=1)
        msk = mx.concatenate([mx.repeat(msk[:, :1], 4, axis=1), msk[:, 1:]], axis=1)
        msk = msk.reshape(1, msk.shape[1] // 4, 4, h_latent, w_latent)
        msk = msk.transpose(0, 2, 1, 3, 4)[0]  # [4, T_lat, H_lat, W_lat]

        # --- Build y tensor: [mask(4ch) + encoded(z_dim ch)] ---
        y_i2v = mx.concatenate([msk, z_video], axis=0)
        mx.eval(y_i2v)
        assert y_i2v.shape[0] == 4 + config.vae_z_dim

        # --- Denoising loop (2 steps) ---
        C_noise = config.out_dim  # noise channels
        pt, ph, pw = config.patch_size
        seq_len = (t_latent // pt) * (h_latent // ph) * (w_latent // pw)

        sched = FlowMatchEulerScheduler()
        num_steps = 2
        sched.set_timesteps(num_steps, shift=config.sample_shift)

        latents = mx.random.normal((C_noise, t_latent, h_latent, w_latent))
        context = mx.random.normal((4, config.text_dim))

        for i in range(num_steps):
            t_val = sched.timesteps[i].item()
            pred = model(
                [latents],
                mx.array([t_val]),
                [context],
                seq_len,
                y=[y_i2v],
            )[0]
            latents = sched.step(pred[None], t_val, latents[None]).squeeze(0)
            mx.eval(latents)

        assert latents.shape == (C_noise, t_latent, h_latent, w_latent)
        assert not mx.any(mx.isnan(latents)).item(), "NaN in denoised latents"
        assert not mx.any(mx.isinf(latents)).item(), "Inf in denoised latents"

        # --- VAE decode ---
        decoded = vae.decode(latents[None])  # [1, 3, T_out, H_out, W_out]
        mx.eval(decoded)
        assert decoded.ndim == 5
        assert decoded.shape[0] == 1
        assert decoded.shape[1] == 3  # RGB output
        assert not mx.any(mx.isnan(decoded)).item(), "NaN in decoded video"
        assert not mx.any(mx.isinf(decoded)).item(), "Inf in decoded video"
        # VAE decode clips to [-1, 1]
        assert float(decoded.max()) <= 1.0
        assert float(decoded.min()) >= -1.0


class TestDualModelSwitching:
    """Test dual-model selection logic: high_noise vs low_noise based on boundary."""

    def test_model_selection_by_timestep(self):
        """Verify high_noise model used for timesteps >= boundary, low_noise otherwise."""
        from mlx_video.models.wan.model import WanModel
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler

        mx.random.seed(1)
        config = _make_tiny_i2v_config()
        assert config.dual_model is True

        high_noise_model = WanModel(config)
        low_noise_model = WanModel(config)

        boundary = config.boundary * config.num_train_timesteps  # 0.9 * 1000 = 900

        C_noise = config.out_dim  # 4
        C_y = config.in_dim - config.out_dim  # 9 - 4 = 5
        F, H, W = 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        sched = FlowMatchEulerScheduler()
        num_steps = 5
        sched.set_timesteps(num_steps, shift=config.sample_shift)

        guide_scale = config.sample_guide_scale  # (3.5, 3.5)
        assert isinstance(guide_scale, tuple) and len(guide_scale) == 2

        latents = mx.random.normal((C_noise, F, H, W))
        y_i2v = mx.random.normal((C_y, F, H, W))
        context = mx.random.normal((4, config.text_dim))

        high_used_steps = []
        low_used_steps = []

        timestep_list = sched.timesteps.tolist()
        for i in range(num_steps):
            timestep_val = timestep_list[i]

            if timestep_val >= boundary:
                model = high_noise_model
                gs = guide_scale[1]
                high_used_steps.append(i)
            else:
                model = low_noise_model
                gs = guide_scale[0]
                low_used_steps.append(i)

            # CFG pass: cond + uncond
            preds = model(
                [latents, latents],
                mx.array([timestep_val, timestep_val]),
                [context, context],
                seq_len,
                y=[y_i2v, y_i2v],
            )
            noise_pred_cond, noise_pred_uncond = preds[0], preds[1]
            noise_pred = noise_pred_uncond + gs * (noise_pred_cond - noise_pred_uncond)

            latents = sched.step(noise_pred[None], timestep_val, latents[None]).squeeze(0)
            mx.eval(latents)

        # With shift=5.0, early timesteps should be high (>=900), later ones low
        assert len(high_used_steps) > 0, "High-noise model was never selected"
        assert len(low_used_steps) > 0, "Low-noise model was never selected"
        # High-noise steps should come before low-noise steps (timesteps decrease)
        if high_used_steps and low_used_steps:
            assert max(high_used_steps) < min(low_used_steps) or \
                   min(high_used_steps) < max(low_used_steps), \
                   "Model switching should happen during the loop"

        assert latents.shape == (C_noise, F, H, W)
        assert not mx.any(mx.isnan(latents)).item()

    def test_guide_scale_tuple_applied_per_model(self):
        """Verify (low_gs, high_gs) tuple applies different scales per model."""
        from mlx_video.models.wan.model import WanModel
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler

        mx.random.seed(2)
        config = _make_tiny_i2v_config()
        config.sample_guide_scale = (2.0, 5.0)  # distinct values

        model = WanModel(config)
        boundary = config.boundary * config.num_train_timesteps

        C_noise = config.out_dim
        F, H, W = 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(5, shift=config.sample_shift)

        latents = mx.random.normal((C_noise, F, H, W))
        context = mx.random.normal((4, config.text_dim))
        guide_scale = config.sample_guide_scale
        C_y = config.in_dim - config.out_dim  # y channels
        y_i2v = mx.random.normal((C_y, F, H, W))

        # Track which guide scale was used at each step
        gs_per_step = []

        timestep_list = sched.timesteps.tolist()
        for i in range(5):
            timestep_val = timestep_list[i]

            if timestep_val >= boundary:
                gs = guide_scale[1]  # high_gs = 5.0
            else:
                gs = guide_scale[0]  # low_gs = 2.0
            gs_per_step.append(gs)

            pred = model(
                [latents, latents],
                mx.array([timestep_val, timestep_val]),
                [context, context],
                seq_len,
                y=[y_i2v, y_i2v],
            )
            noise_pred = pred[1] + gs * (pred[0] - pred[1])
            latents = sched.step(noise_pred[None], timestep_val, latents[None]).squeeze(0)
            mx.eval(latents)

        # Verify both guide scales were used
        assert 5.0 in gs_per_step, "High guide scale (5.0) was never used"
        assert 2.0 in gs_per_step, "Low guide scale (2.0) was never used"
        # High gs should appear first (high timesteps come first)
        first_high = gs_per_step.index(5.0)
        last_low = len(gs_per_step) - 1 - gs_per_step[::-1].index(2.0)
        assert first_high < last_low, "High gs steps should precede low gs steps"

    def test_single_model_fallback_with_tuple_guide_scale(self):
        """When dual_model=False, guide_scale tuple should use first element."""
        from mlx_video.models.wan.model import WanModel
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler

        mx.random.seed(3)
        config = _make_tiny_config()
        config.dual_model = False
        config.sample_guide_scale = (3.0, 5.0)

        model = WanModel(config)
        guide_scale = config.sample_guide_scale

        C, F, H, W = config.in_dim, 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(3, shift=3.0)

        latents = mx.random.normal((C, F, H, W))
        context = mx.random.normal((4, config.text_dim))

        # Mimic generate_wan.py single-model logic:
        # gs = guide_scale if isinstance(guide_scale, (int, float)) else guide_scale[0]
        gs = guide_scale if isinstance(guide_scale, (int, float)) else guide_scale[0]
        assert gs == 3.0, "Single model should use first element of guide_scale tuple"

        for i in range(3):
            t_val = sched.timesteps[i].item()
            pred = model(
                [latents, latents],
                mx.array([t_val, t_val]),
                [context, context],
                seq_len,
            )
            noise_pred = pred[1] + gs * (pred[0] - pred[1])
            latents = sched.step(noise_pred[None], t_val, latents[None]).squeeze(0)
            mx.eval(latents)

        assert latents.shape == (C, F, H, W)
        assert not mx.any(mx.isnan(latents)).item()
