"""Wan2.2 Text-to-Video generation pipeline for MLX."""

import argparse
import gc
import math
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tqdm import tqdm

from mlx_video.models.wan.i2v_utils import build_i2v_mask, preprocess_image
from mlx_video.models.wan.loading import (
    _clean_text,
    encode_text,
    load_t5_encoder,
    load_vae_decoder,
    load_vae_encoder,
    load_wan_model,
)
from mlx_video.postprocess import save_video
from mlx_video.utils import Colors

# Backward-compat alias (tests and external code may use the old name)
_build_i2v_mask = build_i2v_mask


def generate_video(
    model_dir: str,
    prompt: str,
    negative_prompt: str | None = None,
    image: str | None = None,
    width: int = 1280,
    height: int = 720,
    num_frames: int = 81,
    steps: int = None,
    guide_scale: str | float | tuple = None,
    shift: float = None,
    seed: int = -1,
    output_path: str = "output.mp4",
    scheduler: str = "unipc",
):
    """Generate video using Wan pipeline (supports T2V and I2V).

    Args:
        model_dir: Path to converted MLX model directory
        prompt: Text prompt
        negative_prompt: Negative prompt (None = use config default, "" = no negative prompt)
        image: Path to input image for I2V (None = T2V mode)
        width: Video width
        height: Video height
        num_frames: Number of frames (must be 4n+1)
        steps: Number of diffusion steps (None = use config default)
        guide_scale: Guidance scale: float for single, (low,high) for dual (None = config default)
        shift: Noise schedule shift (None = use config default)
        seed: Random seed (-1 for random)
        output_path: Output video path
        scheduler: Solver type: 'euler', 'dpm++', or 'unipc' (default)
    """
    import json

    from mlx_video.models.wan.config import WanModelConfig
    from mlx_video.models.wan.scheduler import (
        FlowDPMPP2MScheduler,
        FlowMatchEulerScheduler,
        FlowUniPCScheduler,
    )

    model_dir = Path(model_dir)

    # Load config from model dir if available, otherwise auto-detect
    config_path = model_dir / "config.json"
    quantization = None
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        # Extract quantization config (not a model config field)
        quantization = config_dict.pop("quantization", None)
        # Handle tuple fields stored as lists in JSON
        for key in ("patch_size", "vae_stride", "window_size", "sample_guide_scale"):
            if key in config_dict and isinstance(config_dict[key], list):
                config_dict[key] = tuple(config_dict[key])
        config = WanModelConfig(**{
            k: v for k, v in config_dict.items()
            if k in WanModelConfig.__dataclass_fields__
        })
    else:
        # Auto-detect: dual model files → 2.2, single model → 2.1
        if (model_dir / "low_noise_model.safetensors").exists():
            config = WanModelConfig.wan22_t2v_14b()
        else:
            # Detect 1.3B vs 14B from weight shapes
            model_path = model_dir / "model.safetensors"
            if model_path.exists():
                probe = mx.load(str(model_path), return_metadata=False)
                for k, v in probe.items():
                    if "patch_embedding_proj.weight" in k:
                        dim = v.shape[0]
                        if dim <= 2048:
                            config = WanModelConfig.wan21_t2v_1_3b()
                        else:
                            config = WanModelConfig.wan21_t2v_14b()
                        break
                else:
                    config = WanModelConfig.wan21_t2v_14b()
                del probe
            else:
                config = WanModelConfig.wan21_t2v_14b()

    is_dual = config.dual_model
    is_i2v = image is not None

    # Validate config against actual weights (handles mismatched config.json)
    if not is_dual:
        model_path = model_dir / "model.safetensors"
        if model_path.exists():
            probe = mx.load(str(model_path), return_metadata=False)
            for k, v in probe.items():
                if "patch_embedding_proj.weight" in k:
                    actual_dim = v.shape[0]
                    if actual_dim != config.dim:
                        print(f"{Colors.YELLOW}  Config dim={config.dim} doesn't match weights dim={actual_dim}, auto-correcting...{Colors.RESET}")
                        if actual_dim <= 2048:
                            config = WanModelConfig.wan21_t2v_1_3b()
                        else:
                            config = WanModelConfig.wan21_t2v_14b()
                    break
            del probe

    # Auto-correct Wan2.2 VAE params from stale configs
    if config.in_dim == 48 and config.vae_z_dim != 48:
        print(f"{Colors.YELLOW}  Auto-correcting Wan2.2 VAE params (in_dim=48 but vae_z_dim={config.vae_z_dim}){Colors.RESET}")
        config = WanModelConfig(**{
            **{f.name: getattr(config, f.name) for f in config.__dataclass_fields__.values()},
            "vae_z_dim": 48,
            "vae_stride": (4, 16, 16),
            "sample_fps": 24,
        })

    # Apply defaults from config if not overridden
    if steps is None:
        steps = config.sample_steps
    if shift is None:
        shift = config.sample_shift
    if guide_scale is None:
        guide_scale = config.sample_guide_scale

    # Normalize guide_scale
    if isinstance(guide_scale, (int, float)):
        guide_scale = float(guide_scale)
    elif isinstance(guide_scale, str):
        parts = [float(x) for x in guide_scale.split(",")]
        guide_scale = tuple(parts) if len(parts) > 1 else parts[0]

    # Validate frame count
    assert (num_frames - 1) % 4 == 0, f"num_frames must be 4n+1, got {num_frames}"

    version_str = f"Wan{config.model_version}"
    mode_str = "dual-model" if is_dual else "single-model"
    pipeline_str = "Image-to-Video" if is_i2v else "Text-to-Video"
    # Resolve negative prompt: explicit user value > config default
    # The official Wan2.2 uses a Chinese negative prompt (config.sample_neg_prompt)
    # that prevents oversaturation, artifacts, and comic look. We use it by default.
    # Text cleaning (_clean_text) normalizes fullwidth chars to match official tokenization.
    if negative_prompt is None:
        neg_prompt_resolved = config.sample_neg_prompt
    else:
        neg_prompt_resolved = negative_prompt
    print(f"{Colors.CYAN}{'='*60}")
    print(f"  {version_str} {pipeline_str} Generation (MLX, {mode_str})")
    print(f"{'='*60}{Colors.RESET}")
    print(f"{Colors.DIM}  Prompt: {prompt}")
    if is_i2v:
        print(f"  Image: {image}")
    if neg_prompt_resolved and neg_prompt_resolved.strip():
        neg_display = neg_prompt_resolved[:60] + "..." if len(neg_prompt_resolved) > 60 else neg_prompt_resolved
        print(f"  Neg prompt: {neg_display}")
    print(f"  Size: {width}x{height}, Frames: {num_frames}")
    print(f"  Steps: {steps}, Guide: {guide_scale}, Shift: {shift}, Solver: {scheduler}")
    print(f"{Colors.RESET}")

    # Seed
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    mx.random.seed(seed)
    np.random.seed(seed)
    print(f"{Colors.DIM}  Seed: {seed}{Colors.RESET}")

    # Align dimensions to patch_size * vae_stride (required for patchify)
    vae_stride = config.vae_stride
    patch_size = config.patch_size
    align_h = patch_size[1] * vae_stride[1]  # e.g. 2*16=32
    align_w = patch_size[2] * vae_stride[2]
    if height % align_h != 0 or width % align_w != 0:
        old_h, old_w = height, width
        height = (height // align_h) * align_h
        width = (width // align_w) * align_w
        if height == 0:
            height = align_h
        if width == 0:
            width = align_w
        print(f"{Colors.DIM}  Aligned {old_w}x{old_h} → {width}x{height} (must be divisible by {align_w}x{align_h}){Colors.RESET}")

    # Compute target latent shape
    z_dim = config.vae_z_dim
    t_latent = (num_frames - 1) // vae_stride[0] + 1
    h_latent = height // vae_stride[1]
    w_latent = width // vae_stride[2]
    target_shape = (z_dim, t_latent, h_latent, w_latent)

    # Sequence length for transformer
    seq_len = math.ceil(
        (h_latent * w_latent) / (patch_size[1] * patch_size[2]) * t_latent
    )

    print(f"{Colors.DIM}  Latent shape: {target_shape}")
    print(f"  Sequence length: {seq_len}{Colors.RESET}")

    # Load T5 encoder
    t1 = time.time()
    print(f"\n{Colors.BLUE}Loading T5 encoder...{Colors.RESET}")
    t5_path = model_dir / "t5_encoder.safetensors"
    t5_encoder = load_t5_encoder(t5_path, config)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")

    # Encode prompts
    print(f"{Colors.BLUE}Encoding text...{Colors.RESET}")
    context = encode_text(t5_encoder, tokenizer, prompt, config.text_len)
    context_null = encode_text(t5_encoder, tokenizer, neg_prompt_resolved, config.text_len)
    mx.eval(context, context_null)

    # Free T5 from memory
    del t5_encoder
    gc.collect(); mx.clear_cache()
    print(f"{Colors.DIM}  T5 encoding: {time.time() - t1:.1f}s{Colors.RESET}")

    # I2V: encode image to latent space
    z_img = None
    i2v_mask = None
    i2v_mask_tokens = None
    if is_i2v:
        print(f"\n{Colors.BLUE}Encoding input image...{Colors.RESET}")
        t_img = time.time()
        img_tensor = preprocess_image(image, width, height)
        mx.eval(img_tensor)

        vae_path = model_dir / "vae.safetensors"
        vae_enc = load_vae_encoder(vae_path, config)
        z_img = vae_enc(img_tensor)  # [1, 1, H_lat, W_lat, z_dim]
        mx.eval(z_img)

        # Convert to channels-first: [z_dim, 1, H_lat, W_lat]
        z_img = z_img[0].transpose(3, 0, 1, 2)

        # Build I2V mask
        i2v_mask, i2v_mask_tokens = build_i2v_mask(target_shape, config.patch_size)

        del vae_enc, img_tensor
        gc.collect(); mx.clear_cache()
        print(f"{Colors.DIM}  Image encoding: {time.time() - t_img:.1f}s{Colors.RESET}")

    # Load transformer models
    print(f"\n{Colors.BLUE}Loading transformer model(s)...{Colors.RESET}")
    if quantization:
        print(f"{Colors.DIM}  Using {quantization['bits']}-bit quantized weights (group_size={quantization['group_size']}){Colors.RESET}")
    t2 = time.time()

    if is_dual:
        low_noise_path = model_dir / "low_noise_model.safetensors"
        high_noise_path = model_dir / "high_noise_model.safetensors"
        low_noise_model = load_wan_model(low_noise_path, config, quantization)
        high_noise_model = load_wan_model(high_noise_path, config, quantization)
    else:
        single_model = load_wan_model(model_dir / "model.safetensors", config, quantization)
    print(f"{Colors.DIM}  Models loaded: {time.time() - t2:.1f}s{Colors.RESET}")

    # Precompute text embeddings once (avoids redundant MLP in every step)
    ref_model = single_model if not is_dual else low_noise_model
    context_emb = ref_model.embed_text([context, context_null])
    mx.eval(context_emb)
    context_cond = context_emb[0:1]   # [1, text_len, dim]
    context_uncond = context_emb[1:2]  # [1, text_len, dim]
    # Stack for batched CFG: [2, text_len, dim]
    context_cfg = mx.concatenate([context_cond, context_uncond], axis=0)

    # Precompute cross-attention K/V caches (constant across all steps)
    if is_dual:
        cross_kv_low = low_noise_model.prepare_cross_kv(context_cfg)
        cross_kv_high = high_noise_model.prepare_cross_kv(context_cfg)
        mx.eval(cross_kv_low, cross_kv_high)
    else:
        cross_kv = single_model.prepare_cross_kv(context_cfg)
        mx.eval(cross_kv)

    # Setup scheduler
    _schedulers = {
        "euler": FlowMatchEulerScheduler,
        "dpm++": FlowDPMPP2MScheduler,
        "unipc": FlowUniPCScheduler,
    }
    sched_cls = _schedulers.get(scheduler, FlowUniPCScheduler)
    sched = sched_cls(num_train_timesteps=config.num_train_timesteps)
    sched.set_timesteps(steps, shift=shift)

    # Generate initial noise
    noise = mx.random.normal(target_shape)

    # I2V: blend first-frame latent into noise
    if is_i2v:
        # Broadcast z_img [z_dim, 1, H, W] across T for first-frame conditioning
        latents = (1.0 - i2v_mask) * z_img + i2v_mask * noise
    else:
        latents = noise

    # Boundary for model switching (dual model only)
    boundary = (config.boundary * config.num_train_timesteps) if is_dual else None

    # Diffusion loop
    print(f"\n{Colors.GREEN}Denoising ({steps} steps)...{Colors.RESET}")
    t3 = time.time()

    for i, t in enumerate(tqdm(range(steps), desc="Diffusion")):
        timestep_val = sched.timesteps[i].item()

        # Select model, guide scale, and cached K/V
        if is_dual:
            if timestep_val >= boundary:
                model = high_noise_model
                gs = guide_scale[1]
                kv = cross_kv_high
            else:
                model = low_noise_model
                gs = guide_scale[0]
                kv = cross_kv_low
        else:
            model = single_model
            gs = guide_scale if isinstance(guide_scale, (int, float)) else guide_scale[0]
            kv = cross_kv

        # Build per-token timesteps for I2V (first-frame patches get t=0)
        if is_i2v:
            t_tokens = i2v_mask_tokens * timestep_val  # [1, L]
            # Pad to seq_len if needed
            pad_len = seq_len - t_tokens.shape[1]
            if pad_len > 0:
                t_tokens = mx.concatenate(
                    [t_tokens, mx.full((1, pad_len), timestep_val)], axis=1
                )
            # Batch for CFG: both cond and uncond get same timesteps
            t_batch = mx.concatenate([t_tokens, t_tokens], axis=0)  # [2, L]
        else:
            t_batch = mx.array([timestep_val, timestep_val])

        # CFG: batch cond + uncond into single B=2 forward pass
        preds = model(
            [latents, latents],
            t=t_batch,
            context=context_cfg,
            seq_len=seq_len,
            cross_kv_caches=kv,
        )
        noise_pred_cond, noise_pred_uncond = preds[0], preds[1]

        # Classifier-free guidance + scheduler step
        noise_pred = noise_pred_uncond + gs * (noise_pred_cond - noise_pred_uncond)
        latents = sched.step(noise_pred[None], timestep_val, latents[None]).squeeze(0)

        # I2V: re-apply mask to keep first frame frozen
        if is_i2v:
            latents = (1.0 - i2v_mask) * z_img + i2v_mask * latents

        # Release temporaries before eval to free memory for graph execution
        del noise_pred_cond, noise_pred_uncond, noise_pred, preds
        mx.eval(latents)

    print(f"{Colors.DIM}  Denoising: {time.time() - t3:.1f}s{Colors.RESET}")

    # Free transformer models and text embeddings
    if is_dual:
        del low_noise_model, high_noise_model, cross_kv_low, cross_kv_high
    else:
        del single_model, cross_kv
    del model, kv, context, context_null, context_cfg
    gc.collect(); mx.clear_cache()

    # Load VAE and decode
    print(f"\n{Colors.BLUE}Decoding with VAE...{Colors.RESET}")
    t4 = time.time()
    vae_path = model_dir / "vae.safetensors"
    vae = load_vae_decoder(vae_path, config)

    is_wan22_vae = config.vae_z_dim == 48

    if is_wan22_vae:
        from mlx_video.models.wan.vae22 import denormalize_latents

        # latents: [C, T, H, W] → [1, T, H, W, C] (channels-last for Wan2.2 VAE)
        z = latents.transpose(1, 2, 3, 0)[None]  # [1, T, H, W, C]
        z = denormalize_latents(z)
        video = vae(z)  # [1, T', H', W', 3]
        mx.eval(video)
        print(f"{Colors.DIM}  VAE decode: {time.time() - t4:.1f}s{Colors.RESET}")

        video = np.array(video[0])  # [T', H', W', 3]
        video = (video + 1.0) / 2.0
        video = np.clip(video * 255.0, 0, 255).astype(np.uint8)
    else:
        video = vae.decode(latents[None])  # [1, 3, T, H, W]
        mx.eval(video)
        print(f"{Colors.DIM}  VAE decode: {time.time() - t4:.1f}s{Colors.RESET}")

        video = np.array(video[0])  # [3, T, H, W]
        video = (video + 1.0) / 2.0
        video = np.clip(video * 255.0, 0, 255).astype(np.uint8)
        video = video.transpose(1, 2, 3, 0)  # [T, H, W, 3]

    save_video(video, output_path, fps=config.sample_fps)
    print(f"\n{Colors.GREEN}✓ Video saved to {output_path}{Colors.RESET}")
    print(f"{Colors.DIM}  Total time: {time.time() - t1:.1f}s{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(description="Wan Text-to-Video Generation (MLX)")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to converted MLX model directory")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image for I2V (omit for T2V mode)")
    parser.add_argument("--negative-prompt", type=str, default=None,
                        help="Negative prompt for CFG (default: official Chinese prompt from config)")
    parser.add_argument("--no-negative-prompt", action="store_true",
                        help="Disable negative prompt (use empty string instead of config default)")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames (must be 4n+1)")
    parser.add_argument("--steps", type=int, default=None, help="Number of diffusion steps (default: from config)")
    parser.add_argument("--guide-scale", type=str, default=None, help="Guidance scale: single float or low,high pair")
    parser.add_argument("--shift", type=float, default=None, help="Noise schedule shift (default: from config)")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--output-path", type=str, default="output.mp4", help="Output video path")
    parser.add_argument(
        "--scheduler", type=str, default="unipc",
        choices=["euler", "dpm++", "unipc"],
        help="Diffusion solver: euler (1st order), dpm++ (2nd order), unipc (2nd order PC, default/official)",
    )
    args = parser.parse_args()

    # Parse guide scale
    guide_scale = None
    if args.guide_scale is not None:
        parts = [float(x) for x in args.guide_scale.split(",")]
        guide_scale = tuple(parts) if len(parts) > 1 else parts[0]

    # Handle negative prompt: --no-negative-prompt forces empty, otherwise pass through
    neg_prompt = args.negative_prompt
    if args.no_negative_prompt:
        neg_prompt = ""

    generate_video(
        model_dir=args.model_dir,
        prompt=args.prompt,
        negative_prompt=neg_prompt,
        image=args.image,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        steps=args.steps,
        guide_scale=guide_scale,
        shift=args.shift,
        seed=args.seed,
        output_path=args.output_path,
        scheduler=args.scheduler,
    )


if __name__ == "__main__":
    main()
