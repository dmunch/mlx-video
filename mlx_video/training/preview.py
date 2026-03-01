"""Preview image generation during training.

Runs a short denoising loop using the current LoRA-injected model,
decodes through the VAE, and saves a single-frame preview PNG.
"""

import gc
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from mlx_video.training.config import TrainingConfig
from mlx_video.training.dataset import EncodedItem
from mlx_video.utils import Colors


def generate_preview(
    model: nn.Module,
    config: TrainingConfig,
    encoded_data: list[EncodedItem],
    epoch: int,
    output_dir: str,
    steps: int = 20,
    guide_scale: float = 1.0,
    seed: int | None = None,
) -> str | None:
    """Generate a single-frame preview image using the current model state.

    Runs a denoising loop with CFG guidance, loads the VAE decoder temporarily
    to decode the result, saves as PNG, then frees the VAE.

    Uses a fixed seed (config.seed + 9999 by default) so previews are
    comparable across epochs — same starting noise shows LoRA evolution.

    Args:
        model: WanModel with trained LoRA layers.
        config: Training configuration.
        encoded_data: Pre-encoded training data (uses first item's text embedding).
        epoch: Current epoch number (for filename).
        output_dir: Base output directory.
        steps: Number of denoising steps.
        guide_scale: CFG guidance scale (1.0 to disable CFG).
        seed: Fixed seed for preview noise. Defaults to config.seed + 9999.

    Returns:
        Path to the saved preview image, or None on failure.
    """
    try:
        return _generate_preview_impl(
            model, config, encoded_data, epoch, output_dir, steps, guide_scale,
            seed=seed if seed is not None else config.seed + 9999,
        )
    except Exception as e:
        print(f"  {Colors.DIM}⚠ Preview generation failed: {e}{Colors.RESET}")
        return None


def _generate_preview_impl(
    model: nn.Module,
    config: TrainingConfig,
    encoded_data: list[EncodedItem],
    epoch: int,
    output_dir: str,
    steps: int,
    guide_scale: float,
    seed: int = 42,
) -> str:
    from mlx_video.models.wan.config import WanModelConfig
    from mlx_video.models.wan.loading import load_vae_decoder
    from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler

    model_dir = Path(config.model_dir)
    model_config = model.config

    preview_w = config.monitoring.preview_width
    preview_h = config.monitoring.preview_height
    # Use training shift if set, otherwise fall back to model config
    shift = config.training.shift or getattr(model_config, "sample_shift", 12.0)
    z_dim = model_config.vae_z_dim
    vae_stride = model_config.vae_stride
    patch_size = model_config.patch_size

    # Compute latent dimensions for preview size
    t_latent = 1  # Single frame: (1-1)//stride + 1 = 1
    h_latent = preview_h // vae_stride[1]
    w_latent = preview_w // vae_stride[2]

    # Grid sizes for RoPE
    f_grid = max(1, t_latent // patch_size[0])
    h_grid = h_latent // patch_size[1]
    w_grid = w_latent // patch_size[2]
    seq_len = f_grid * h_grid * w_grid

    # Use first training sample's text embedding as context
    text_emb = encoded_data[0].text_embedding

    # CFG: embed both cond (real prompt) and uncond (zeros) through model's text MLP
    use_cfg = guide_scale > 1.0
    if use_cfg:
        null_emb = mx.zeros_like(text_emb)
        context_embedded = model.embed_text([text_emb, null_emb])  # [2, text_len, dim]
        cross_kv = model.prepare_cross_kv(context_embedded)
    else:
        context_embedded = model.embed_text([text_emb])  # [1, text_len, dim]
        cross_kv = model.prepare_cross_kv(context_embedded)

    if use_cfg:
        rope_cos_sin = model.prepare_rope([(f_grid, h_grid, w_grid)] * 2)
    else:
        rope_cos_sin = model.prepare_rope([(f_grid, h_grid, w_grid)])

    # Setup scheduler
    sched = FlowMatchEulerScheduler(
        num_train_timesteps=model_config.num_train_timesteps
    )
    sched.set_timesteps(steps, shift=shift)

    # Fixed seed for deterministic preview noise (same noise every epoch → shows LoRA evolution)
    # Use mx.random.key for a separate RNG stream that doesn't affect the training state
    preview_key = mx.random.key(seed)

    # Initial noise
    latents = mx.random.normal(shape=(z_dim, t_latent, h_latent, w_latent), key=preview_key)

    # Denoising loop
    for i, timestep_val in enumerate(sched.timesteps):
        if use_cfg:
            t_batch = mx.array([timestep_val, timestep_val])
            preds = model(
                [latents, latents],
                t=t_batch,
                context=context_embedded,
                seq_len=seq_len,
                cross_kv_caches=cross_kv,
                rope_cos_sin=rope_cos_sin,
            )
            noise_pred_cond, noise_pred_uncond = preds[0], preds[1]
            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            t_batch = mx.array([timestep_val])
            preds = model(
                [latents],
                t=t_batch,
                context=context_embedded,
                seq_len=seq_len,
                cross_kv_caches=cross_kv,
                rope_cos_sin=rope_cos_sin,
            )
            noise_pred = preds[0]

        latents = sched.step(noise_pred[None], timestep_val, latents[None]).squeeze(0)
        mx.eval(latents)

    # Decode through VAE
    vae_path = model_dir / "vae.safetensors"
    vae = load_vae_decoder(vae_path, model_config)

    is_wan22_vae = z_dim == 48

    # Warm-up: prepend first frame for causal conv context
    latents_for_decode = mx.concatenate([latents[:, 0:1], latents], axis=1)
    warmup_trim = vae_stride[0]

    if is_wan22_vae:
        from mlx_video.models.wan.vae22 import denormalize_latents

        z = latents_for_decode.transpose(1, 2, 3, 0)[None]  # [1, T+1, H, W, C]
        z = denormalize_latents(z)
        video = vae(z)  # [1, T', H', W', 3]
        mx.eval(video)
        video = np.array(video[0])  # [T', H', W', 3]
        video = video[warmup_trim:]
        video = (video + 1.0) / 2.0
        frame = np.clip(video[0] * 255.0, 0, 255).astype(np.uint8)  # [H, W, 3]
    else:
        video = vae.decode(latents_for_decode[None])  # [1, 3, T+1*4, H, W]
        mx.eval(video)
        video = np.array(video[0])  # [3, T', H, W]
        video = video[:, warmup_trim:]
        video = (video + 1.0) / 2.0
        frame = np.clip(video[:, 0] * 255.0, 0, 255).astype(np.uint8)  # [3, H, W]
        frame = frame.transpose(1, 2, 0)  # [H, W, 3]

    # Free VAE
    del vae
    gc.collect()
    mx.clear_cache()

    # Save preview
    preview_dir = Path(output_dir) / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"preview_epoch_{epoch:04d}.png"
    Image.fromarray(frame).save(str(preview_path))

    return str(preview_path)
