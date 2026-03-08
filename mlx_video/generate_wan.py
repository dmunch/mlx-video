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


def _best_output_size(w, h, dw, dh, max_area):
    """Compute the best output resolution that fits within max_area while
    preserving the input aspect ratio and satisfying alignment constraints.
    Matches the reference implementation's best_output_size().
    """
    ratio = w / h
    ow = (max_area * ratio) ** 0.5
    oh = max_area / ow

    # Option 1: process width first
    ow1 = int(ow // dw * dw)
    oh1 = int(max_area / ow1 // dh * dh)
    ratio1 = ow1 / oh1

    # Option 2: process height first
    oh2 = int(oh // dh * dh)
    ow2 = int(max_area / oh2 // dw * dw)
    ratio2 = ow2 / oh2

    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio):
        return ow1, oh1
    return ow2, oh2


def generate_video(
    model_dir: str,
    prompt: str,
    negative_prompt: str | None = None,
    image: str | None = None,
    width: int = 1280,
    height: int = 704,
    num_frames: int = 81,
    steps: int = None,
    guide_scale: str | float | tuple = None,
    shift: float = None,
    seed: int = -1,
    output_path: str = "output.mp4",
    scheduler: str = "unipc",
    teacache_thresh: float = 0.0,
    teacache_verbose: bool = False,
    spectrum: bool = False,
    spectrum_w: float = 0.5,
    spectrum_flex_window: float = 0.75,
    spectrum_warmup: int = 5,
    magcache: bool = False,
    magcache_thresh: float = 0.06,
    magcache_K: int = 2,
    magcache_retention_ratio: float = 0.2,
    magcache_verbose: bool = False,
    magcache_calibrate: bool = False,
    magcache_ratios: str | None = None,
    bwcache: bool = False,
    bwcache_mode: str = "block",
    bwcache_thresh: float = 0.15,
    bwcache_reuse_interval: int = 3,
    bwcache_last_step: float = 0.5,
    bwcache_boundary_blocks: int = 6,
    bwcache_verbose: bool = False,
    loras: list | None = None,
    loras_high: list | None = None,
    loras_low: list | None = None,
    tiling: str = "auto",
    no_compile: bool = False,
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
        teacache_thresh: TeaCache threshold (0=disabled, 0.1=~2x speedup, 0.2=~3x speedup)
        teacache_verbose: Print per-step TeaCache diagnostics (rel_l1, rescaled, skip/compute)
        spectrum: Enable Spectrum acceleration (Chebyshev polynomial feature forecasting)
        spectrum_w: Spectrum blend weight: 0=Taylor only, 1=Chebyshev only (default: 0.5)
        spectrum_flex_window: Window growth rate controlling speedup (0.75=~3.5x, 3.0=~5x)
        spectrum_warmup: Always compute first N steps to build cache (default: 5)
        magcache: Enable MagCache acceleration (magnitude-aware residual caching)
        magcache_thresh: MagCache accumulated error threshold (0.06=~1.5-2x speedup)
        magcache_K: Max consecutive skip steps (default: 2)
        magcache_retention_ratio: Fraction of early steps that always compute (default: 0.2)
        magcache_verbose: Print per-step MagCache diagnostics
        magcache_calibrate: Run calibration mode (full compute, record ratios to JSON)
        magcache_ratios: Path to custom calibration JSON (overrides built-in ratios)
        bwcache: Enable BWCache acceleration (block-wise caching)
        bwcache_mode: BWCache mode: 'step' (paper-faithful) or 'block' (per-block skip, default)
        bwcache_thresh: BWCache similarity threshold δ (default: 0.15)
        bwcache_reuse_interval: Periodic recomputation interval R for step mode (default: 3)
        bwcache_last_step: Fraction of tail steps to always compute in step mode (default: 0.5)
        bwcache_boundary_blocks: First/last N blocks always computed in block mode (default: 6)
        bwcache_verbose: Print per-step/block BWCache diagnostics
        loras: Optional list of (path, strength) tuples applied to all models
        loras_high: Optional list of (path, strength) tuples for high-noise model only
        loras_low: Optional list of (path, strength) tuples for low-noise model only
        tiling: Tiling mode for VAE decoding. Options:
            - "auto": Automatically determine tiling based on video size (default)
            - "none": Disable tiling
            - "default", "aggressive", "conservative": Preset tiling configs
            - "spatial": Spatial tiling only
            - "temporal": Temporal tiling only
        no_compile: If True, skip mx.compile on models (useful for debugging)
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
        config = WanModelConfig(
            **{
                k: v
                for k, v in config_dict.items()
                if k in WanModelConfig.__dataclass_fields__
            }
        )
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
                        print(
                            f"{Colors.YELLOW}  Config dim={config.dim} doesn't match weights dim={actual_dim}, auto-correcting...{Colors.RESET}"
                        )
                        if actual_dim <= 2048:
                            config = WanModelConfig.wan21_t2v_1_3b()
                        else:
                            config = WanModelConfig.wan21_t2v_14b()
                    break
            del probe

    # Auto-correct Wan2.2 VAE params from stale configs
    if config.in_dim == 48 and config.vae_z_dim != 48:
        print(
            f"{Colors.YELLOW}  Auto-correcting Wan2.2 VAE params (in_dim=48 but vae_z_dim={config.vae_z_dim}){Colors.RESET}"
        )
        config = WanModelConfig(
            **{
                **{
                    f.name: getattr(config, f.name)
                    for f in config.__dataclass_fields__.values()
                },
                "vae_z_dim": 48,
                "vae_stride": (4, 16, 16),
                "sample_fps": 24,
            }
        )

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

    # Detect CFG-disabled mode (guide_scale=1.0 for all models → skip uncond pass for 2x speedup)
    if isinstance(guide_scale, tuple):
        cfg_disabled = all(gs <= 1.0 for gs in guide_scale)
    else:
        cfg_disabled = guide_scale <= 1.0

    # Validate frame count
    assert (num_frames - 1) % 4 == 0, f"num_frames must be 4n+1, got {num_frames}"

    gen_frames = num_frames

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
        neg_display = (
            neg_prompt_resolved[:60] + "..."
            if len(neg_prompt_resolved) > 60
            else neg_prompt_resolved
        )
        print(f"  Neg prompt: {neg_display}")
    print(f"  Size: {width}x{height}, Frames: {num_frames}")
    print(
        f"  Steps: {steps}, Guide: {guide_scale}, Shift: {shift}, Solver: {scheduler}"
    )
    if cfg_disabled:
        print(f"  CFG: disabled (guide_scale≤1 → B=1 fast path, 2x denoising speedup)")
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
        print(
            f"{Colors.DIM}  Aligned {old_w}x{old_h} → {width}x{height} (must be divisible by {align_w}x{align_h}){Colors.RESET}"
        )

    # Enforce max_area constraint (model-specific resolution limit)
    if config.max_area > 0 and height * width > config.max_area:
        old_h, old_w = height, width
        width, height = _best_output_size(width, height, align_w, align_h, config.max_area)
        print(
            f"{Colors.YELLOW}  ⚠ Resolution {old_w}x{old_h} exceeds model's max area "
            f"({config.max_area:,}px). Adjusted → {width}x{height}{Colors.RESET}"
        )

    # Compute target latent shape
    z_dim = config.vae_z_dim
    t_latent = (gen_frames - 1) // vae_stride[0] + 1
    h_latent = height // vae_stride[1]
    w_latent = width // vae_stride[2]
    target_shape = (z_dim, t_latent, h_latent, w_latent)

    # Sequence length for transformer
    seq_len = math.ceil(
        (h_latent * w_latent) / (patch_size[1] * patch_size[2]) * t_latent
    )

    print(f"{Colors.DIM}  Latent shape: {target_shape}")
    if extra_frames > 0:
        print(
            f"  Generating {extra_frames} extra pixel frames to absorb VAE boundary artifacts"
        )

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
    if cfg_disabled:
        context_null = None
        mx.eval(context)
    else:
        context_null = encode_text(
            t5_encoder, tokenizer, neg_prompt_resolved, config.text_len
        )
        mx.eval(context, context_null)

    # Free T5 from memory
    del t5_encoder
    gc.collect()
    mx.clear_cache()
    print(f"{Colors.DIM}  T5 encoding: {time.time() - t1:.1f}s{Colors.RESET}")

    # I2V: encode image to latent space
    z_img = None
    i2v_mask = None
    i2v_mask_tokens = None
    y_i2v = None
    is_i2v_channel_concat = is_i2v and config.model_type == "i2v"
    is_i2v_mask_blend = is_i2v and config.model_type != "i2v"
    if is_i2v:
        print(f"\n{Colors.BLUE}Encoding input image...{Colors.RESET}")
        t_img = time.time()

        vae_path = model_dir / "vae.safetensors"

        if is_i2v_channel_concat:
            # I2V-14B: encode full video (first frame = image, rest = zeros)
            # and construct y tensor with mask + encoded latents
            from PIL import Image

            img = Image.open(image).convert("RGB")
            scale = max(width / img.width, height / img.height)
            img = img.resize(
                (round(img.width * scale), round(img.height * scale)), Image.LANCZOS
            )
            x1, y1 = (img.width - width) // 2, (img.height - height) // 2
            img = img.crop((x1, y1, x1 + width, y1 + height))
            img_arr = mx.array(
                np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0
            )  # [H, W, 3]
            img_chw = img_arr.transpose(2, 0, 1)  # [3, H, W]

            # Build video: first frame = image, rest = zeros -> [3, F, H, W]
            # Chunked encoding processes 1-frame + 4-frame chunks with temporal caching
            video = mx.concatenate(
                [
                    img_chw[:, None, :, :],
                    mx.zeros((3, num_frames - 1, height, width)),
                ],
                axis=1,
            )

            # Encode through Wan2.1 VAE -> [1, z_dim, T_lat, H_lat, W_lat]
            vae_enc = load_vae_encoder(vae_path, config)
            z_video = vae_enc.encode(video[None])  # [1, 16, T_lat, H_lat, W_lat]
            mx.eval(z_video)
            z_video = z_video[0]  # [16, T_lat, H_lat, W_lat]

            # Build mask: 1 for first frame, 0 for rest -> rearrange to [4, T_lat, H, W]
            msk = mx.ones((1, num_frames, h_latent, w_latent))
            msk = mx.concatenate(
                [msk[:, :1], mx.zeros((1, num_frames - 1, h_latent, w_latent))], axis=1
            )
            # Repeat first frame 4x, concat rest: [1, 4 + (F-1), H_lat, W_lat]
            msk = mx.concatenate(
                [
                    mx.repeat(msk[:, :1], 4, axis=1),
                    msk[:, 1:],
                ],
                axis=1,
            )
            # Reshape to [1, T_lat, 4, H_lat, W_lat] then transpose -> [4, T_lat, H_lat, W_lat]
            msk = msk.reshape(1, msk.shape[1] // 4, 4, h_latent, w_latent)
            msk = msk.transpose(0, 2, 1, 3, 4)[0]  # [4, T_lat, H_lat, W_lat]

            # y = concat([mask, encoded_video]) -> [20, T_lat, H_lat, W_lat]
            y_i2v = mx.concatenate([msk, z_video], axis=0)
            mx.eval(y_i2v)

            del vae_enc, img_arr, img_chw, video, z_video, msk
        else:
            # TI2V-5B: encode single image, blend with noise via mask
            img_tensor = preprocess_image(image, width, height)
            mx.eval(img_tensor)

            vae_enc = load_vae_encoder(vae_path, config)
            z_img = vae_enc.encode(img_tensor)  # [1, 1, H_lat, W_lat, z_dim]
            mx.eval(z_img)
            z_img = z_img[0].transpose(3, 0, 1, 2)  # [z_dim, 1, H_lat, W_lat]
            i2v_mask, i2v_mask_tokens = build_i2v_mask(target_shape, config.patch_size)

            del vae_enc, img_tensor

        gc.collect()
        mx.clear_cache()
        print(f"{Colors.DIM}  Image encoding: {time.time() - t_img:.1f}s{Colors.RESET}")

    # Load transformer models
    print(f"\n{Colors.BLUE}Loading transformer model(s)...{Colors.RESET}")
    if quantization:
        print(
            f"{Colors.DIM}  Using {quantization['bits']}-bit quantized weights (group_size={quantization['group_size']}){Colors.RESET}"
        )
    t2 = time.time()

    # Merge per-model LoRAs with shared LoRAs
    _loras_low = (loras or []) + (loras_low or []) or None
    _loras_high = (loras or []) + (loras_high or []) or None
    _loras_single = loras

    if is_dual:
        low_noise_path = model_dir / "low_noise_model.safetensors"
        high_noise_path = model_dir / "high_noise_model.safetensors"
        low_noise_model = load_wan_model(
            low_noise_path, config, quantization, loras=_loras_low
        )
        high_noise_model = load_wan_model(
            high_noise_path, config, quantization, loras=_loras_high
        )
    else:
        single_model = load_wan_model(
            model_dir / "model.safetensors", config, quantization, loras=_loras_single
        )
    print(f"{Colors.DIM}  Models loaded: {time.time() - t2:.1f}s{Colors.RESET}")

    # Precompute text embeddings once (avoids redundant MLP in every step)
    # Each model has its own text_embedding weights, so dual models need separate embeddings
    if cfg_disabled:
        # No CFG: only compute cond embeddings (B=1 forward pass, 2x faster)
        if is_dual:
            context_emb_low = low_noise_model.embed_text([context])
            context_emb_high = high_noise_model.embed_text([context])
            mx.eval(context_emb_low, context_emb_high)
            context_cond_low = context_emb_low[0:1]
            context_cond_high = context_emb_high[0:1]
        else:
            context_emb = single_model.embed_text([context])
            mx.eval(context_emb)
            context_cond = context_emb[0:1]
    else:
        if is_dual:
            context_emb_low = low_noise_model.embed_text([context, context_null])
            context_emb_high = high_noise_model.embed_text([context, context_null])
            mx.eval(context_emb_low, context_emb_high)
            context_cfg_low = mx.concatenate(
                [context_emb_low[0:1], context_emb_low[1:2]], axis=0
            )
            context_cfg_high = mx.concatenate(
                [context_emb_high[0:1], context_emb_high[1:2]], axis=0
            )
        else:
            context_emb = single_model.embed_text([context, context_null])
            mx.eval(context_emb)
            context_cfg = mx.concatenate([context_emb[0:1], context_emb[1:2]], axis=0)

    # Precompute cross-attention K/V caches (constant across all steps)
    if cfg_disabled:
        if is_dual:
            cross_kv_low = low_noise_model.prepare_cross_kv(context_cond_low)
            cross_kv_high = high_noise_model.prepare_cross_kv(context_cond_high)
            mx.eval(cross_kv_low, cross_kv_high)
        else:
            cross_kv = single_model.prepare_cross_kv(context_cond)
            mx.eval(cross_kv)
    else:
        if is_dual:
            cross_kv_low = low_noise_model.prepare_cross_kv(context_cfg_low)
            cross_kv_high = high_noise_model.prepare_cross_kv(context_cfg_high)
            mx.eval(cross_kv_low, cross_kv_high)
        else:
            cross_kv = single_model.prepare_cross_kv(context_cfg)
            mx.eval(cross_kv)

    # Precompute RoPE frequencies (grid sizes are constant across all steps)
    f_grid = t_latent // patch_size[0]
    h_grid = h_latent // patch_size[1]
    w_grid = w_latent // patch_size[2]
    if cfg_disabled:
        rope_grid_sizes = [(f_grid, h_grid, w_grid)]
    else:
        rope_grid_sizes = [(f_grid, h_grid, w_grid), (f_grid, h_grid, w_grid)]
    if is_dual:
        rope_cos_sin_low = low_noise_model.prepare_rope(rope_grid_sizes)
        rope_cos_sin_high = high_noise_model.prepare_rope(rope_grid_sizes)
        mx.eval(rope_cos_sin_low, rope_cos_sin_high)
    else:
        rope_cos_sin = single_model.prepare_rope(rope_grid_sizes)
        mx.eval(rope_cos_sin)

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

    # I2V initialization: TI2V-5B blends image with noise, I2V-14B uses pure noise
    if is_i2v_mask_blend:
        latents = (1.0 - i2v_mask) * z_img + i2v_mask * noise
    else:
        latents = noise

    # Boundary for model switching (dual model only)
    boundary = (config.boundary * config.num_train_timesteps) if is_dual else None

    # Diffusion loop
    print(f"\n{Colors.GREEN}Denoising ({steps} steps)...{Colors.RESET}")
    t3 = time.time()

    # Configure TeaCache
    if teacache_thresh > 0 and not spectrum and not magcache:
        if config.teacache_coefficients is None:
            print(
                f"{Colors.YELLOW}  Warning: TeaCache not available for this model (no profiled coefficients). Ignoring --teacache-thresh.{Colors.RESET}"
            )
        else:

            def _configure_teacache(m, steps):
                m.teacache.enabled = True
                m.teacache.threshold = teacache_thresh
                m.teacache.coefficients = config.teacache_coefficients
                m.teacache.num_steps = steps
                m.teacache.ret_steps = 2
                m.teacache.cutoff_steps = steps - 2
                m.teacache.verbose = teacache_verbose
                m.teacache.reset()

            if is_dual:
                _configure_teacache(low_noise_model, steps)
                _configure_teacache(high_noise_model, steps)
            else:
                _configure_teacache(single_model, steps)
            print(f"{Colors.DIM}  TeaCache: threshold={teacache_thresh}{Colors.RESET}")

    # Configure MagCache calibration mode (mutually exclusive with --magcache)
    if magcache_calibrate:
        if magcache:
            print(
                f"{Colors.YELLOW}  Warning: --magcache-calibrate and --magcache are mutually exclusive. Using calibration mode.{Colors.RESET}"
            )
            magcache = False

        from mlx_video.models.wan.magcache import save_calibration

        def _configure_calibration(m, num_steps):
            m.magcache.verbose = magcache_verbose
            m.magcache.configure_calibration(num_steps)

        if is_dual:
            high_steps = sum(1 for tv in sched.timesteps.tolist() if tv >= boundary)
            low_steps = steps - high_steps
            _configure_calibration(high_noise_model, high_steps)
            _configure_calibration(low_noise_model, low_steps)
            print(
                f"{Colors.DIM}  MagCache calibration: recording ratios for high-noise ({high_steps} steps) + low-noise ({low_steps} steps){Colors.RESET}"
            )
        else:
            _configure_calibration(single_model, steps)
            print(
                f"{Colors.DIM}  MagCache calibration: recording ratios for {steps} steps{Colors.RESET}"
            )

    # Configure MagCache (mutually exclusive with TeaCache, compatible with Spectrum)
    if magcache:
        if teacache_thresh > 0:
            print(
                f"{Colors.YELLOW}  Warning: MagCache and TeaCache are mutually exclusive. Using MagCache.{Colors.RESET}"
            )

        has_custom_ratios = magcache_ratios is not None
        has_builtin_ratios = config.magcache_ratios_key is not None

        if not has_custom_ratios and not has_builtin_ratios:
            print(
                f"{Colors.YELLOW}  Warning: MagCache not available for this model (no pre-calibrated ratios). "
                f"Use --magcache-calibrate to generate ratios. Ignoring --magcache.{Colors.RESET}"
            )
            magcache = False
        else:

            def _configure_magcache_common(m):
                m.magcache.enabled = True
                m.magcache.threshold = magcache_thresh
                m.magcache.K = magcache_K
                m.magcache.retention_ratio = magcache_retention_ratio
                m.magcache.verbose = magcache_verbose

            if has_custom_ratios:
                # Load ratios from calibration JSON file
                if is_dual:
                    high_steps = sum(
                        1 for tv in sched.timesteps.tolist() if tv >= boundary
                    )
                    low_steps = steps - high_steps
                    _configure_magcache_common(high_noise_model)
                    high_noise_model.magcache.configure_from_file(
                        high_steps, magcache_ratios, "high_noise"
                    )
                    # Try loading low-noise ratios if present in the file
                    try:
                        _configure_magcache_common(low_noise_model)
                        low_noise_model.magcache.configure_from_file(
                            low_steps, magcache_ratios, "low_noise"
                        )
                        print(
                            f"{Colors.DIM}  MagCache: loaded custom ratios for both models from {magcache_ratios}{Colors.RESET}"
                        )
                    except (ValueError, KeyError):
                        print(
                            f"{Colors.DIM}  MagCache: loaded custom ratios for high-noise model from {magcache_ratios}{Colors.RESET}"
                        )
                else:
                    _configure_magcache_common(single_model)
                    single_model.magcache.configure_from_file(
                        steps, magcache_ratios, "model"
                    )
                    print(
                        f"{Colors.DIM}  MagCache: loaded custom ratios from {magcache_ratios}{Colors.RESET}"
                    )
            else:
                # Use built-in pre-calibrated ratios
                ratios_key = config.magcache_ratios_key
                if config.model_type == "ti2v" and is_i2v:
                    ratios_key = "ti2v_5b_i2v"

                if is_dual:
                    high_steps = sum(
                        1 for tv in sched.timesteps.tolist() if tv >= boundary
                    )
                    low_steps = steps - high_steps
                    _configure_magcache_common(high_noise_model)
                    high_noise_model.magcache.configure(high_steps, ratios_key)
                    # Try low-noise model (MLX-calibrated ratios may be available)
                    low_key = ratios_key.replace("_high", "_low")
                    try:
                        _configure_magcache_common(low_noise_model)
                        low_noise_model.magcache.configure(low_steps, low_key)
                    except ValueError:
                        # No low-noise ratios available, disable on low-noise model
                        low_noise_model.magcache.enabled = False
                else:
                    _configure_magcache_common(single_model)
                    single_model.magcache.configure(steps, ratios_key)

            if spectrum:
                print(
                    f"{Colors.DIM}  MagCache+Spectrum hybrid: thresh={magcache_thresh}, K={magcache_K}, retention={magcache_retention_ratio}{Colors.RESET}"
                )
            elif not has_custom_ratios:
                print(
                    f"{Colors.DIM}  MagCache: thresh={magcache_thresh}, K={magcache_K}, retention={magcache_retention_ratio}{Colors.RESET}"
                )

    # Configure Spectrum (mutually exclusive with TeaCache)
    use_caching = False
    if spectrum:
        if teacache_thresh > 0 and not magcache:
            print(
                f"{Colors.YELLOW}  Warning: Spectrum and TeaCache are mutually exclusive. Using Spectrum.{Colors.RESET}"
            )

        def _configure_spectrum(m, num_steps):
            m.spectrum.enabled = True
            m.spectrum.num_steps = num_steps
            m.spectrum.m = 4
            m.spectrum.lam = 0.1
            m.spectrum.w = spectrum_w
            m.spectrum.warmup_steps = spectrum_warmup
            m.spectrum.window_size = 2
            m.spectrum.flex_window = spectrum_flex_window
            m.spectrum.reset()

        if is_dual:
            # Count steps per model using the boundary
            high_steps = sum(1 for tv in sched.timesteps.tolist() if tv >= boundary)
            low_steps = steps - high_steps
            _configure_spectrum(high_noise_model, high_steps)
            if low_steps >= 8:
                _configure_spectrum(low_noise_model, low_steps)
            else:
                print(
                    f"{Colors.DIM}  Spectrum: disabled for low-noise model ({low_steps} steps, need ≥8){Colors.RESET}"
                )
        else:
            _configure_spectrum(single_model, steps)
        use_caching = True
        if not magcache:
            print(
                f"{Colors.DIM}  Spectrum: w={spectrum_w}, flex_window={spectrum_flex_window}, warmup={spectrum_warmup}{Colors.RESET}"
            )
    elif teacache_thresh > 0 and not magcache:
        use_caching = True

    if magcache:
        use_caching = True

    if magcache_calibrate:
        use_caching = True

    # Configure BWCache (block mode compatible with MagCache/Spectrum; step mode standalone)
    if bwcache:
        if bwcache_mode == "step" and (magcache or spectrum or teacache_thresh > 0):
            print(
                f"{Colors.YELLOW}  Warning: BWCache step mode is mutually exclusive with other step-level caches. "
                f"Switching to block mode.{Colors.RESET}"
            )
            bwcache_mode = "block"

        # Block mode uses a tighter default threshold for quality
        if bwcache_mode == "block" and bwcache_thresh == 0.15:
            bwcache_thresh = 0.08

        def _configure_bwcache(m, num_steps):
            m.bwcache.enabled = True
            m.bwcache.mode = bwcache_mode
            m.bwcache.thresh = bwcache_thresh
            m.bwcache.reuse_interval = bwcache_reuse_interval
            m.bwcache.last_step_ratio = bwcache_last_step
            m.bwcache.boundary_blocks = bwcache_boundary_blocks
            m.bwcache.verbose = bwcache_verbose
            m.bwcache.num_steps = num_steps
            m.bwcache.reset()

        if is_dual:
            high_steps = sum(1 for tv in sched.timesteps.tolist() if tv >= boundary)
            low_steps = steps - high_steps
            _configure_bwcache(high_noise_model, high_steps)
            _configure_bwcache(low_noise_model, low_steps)
        else:
            _configure_bwcache(single_model, steps)

        use_caching = True
        print(
            f"{Colors.DIM}  BWCache: mode={bwcache_mode}, thresh={bwcache_thresh}"
            + (
                f", boundary_blocks={bwcache_boundary_blocks}"
                if bwcache_mode == "block"
                else f", reuse_interval={bwcache_reuse_interval}"
            )
            + f"{Colors.RESET}"
        )

    # Compile model forward for faster denoising (incompatible with caching strategies)
    if not use_caching and not no_compile:
        models_to_compile = (
            [high_noise_model, low_noise_model] if is_dual else [single_model]
        )
        for m in models_to_compile:
            m._compiled = mx.compile(m)

    # Pre-convert timesteps to Python list to avoid .item() sync each step
    timestep_list = sched.timesteps.tolist()

    for i, t in enumerate(tqdm(range(steps), desc="Diffusion")):
        timestep_val = timestep_list[i]

        # Select model, cached K/V, and precomputed RoPE
        if is_dual:
            if timestep_val >= boundary:
                model = high_noise_model
                kv = cross_kv_high
                rcs = rope_cos_sin_high
            else:
                model = low_noise_model
                kv = cross_kv_low
                rcs = rope_cos_sin_low
        else:
            model = single_model
            kv = cross_kv
            rcs = rope_cos_sin

        # Use compiled forward when available (faster after first trace)
        _call = getattr(model, "_compiled", model)

        if cfg_disabled:
            # No CFG: B=1 forward pass (2x faster than B=2 CFG batch)
            if is_i2v_mask_blend:
                t_tokens = i2v_mask_tokens * timestep_val
                pad_len = seq_len - t_tokens.shape[1]
                if pad_len > 0:
                    t_tokens = mx.concatenate(
                        [t_tokens, mx.full((1, pad_len), timestep_val)], axis=1
                    )
                t_batch = t_tokens  # [1, L]
            else:
                t_batch = mx.array([timestep_val])

            y_arg = [y_i2v] if is_i2v_channel_concat else None

            if is_dual:
                ctx = (
                    context_cond_high if timestep_val >= boundary else context_cond_low
                )
            else:
                ctx = context_cond
            preds = _call(
                [latents],
                t=t_batch,
                context=ctx,
                seq_len=seq_len,
                cross_kv_caches=kv,
                y=y_arg,
                rope_cos_sin=rcs,
            )
            noise_pred = preds[0]
            del preds
        else:
            # CFG: batch cond + uncond into single B=2 forward pass
            if is_dual:
                gs = guide_scale[1] if timestep_val >= boundary else guide_scale[0]
            else:
                gs = (
                    guide_scale
                    if isinstance(guide_scale, (int, float))
                    else guide_scale[0]
                )

            if is_i2v_mask_blend:
                t_tokens = i2v_mask_tokens * timestep_val
                pad_len = seq_len - t_tokens.shape[1]
                if pad_len > 0:
                    t_tokens = mx.concatenate(
                        [t_tokens, mx.full((1, pad_len), timestep_val)], axis=1
                    )
                t_batch = mx.concatenate([t_tokens, t_tokens], axis=0)
            else:
                t_batch = mx.array([timestep_val, timestep_val])

            y_arg = [y_i2v, y_i2v] if is_i2v_channel_concat else None

            ctx = (
                context_cfg
                if not is_dual
                else (context_cfg_high if timestep_val >= boundary else context_cfg_low)
            )
            preds = _call(
                [latents, latents],
                t=t_batch,
                context=ctx,
                seq_len=seq_len,
                cross_kv_caches=kv,
                y=y_arg,
                rope_cos_sin=rcs,
            )
            noise_pred_cond, noise_pred_uncond = preds[0], preds[1]
            noise_pred = noise_pred_uncond + gs * (noise_pred_cond - noise_pred_uncond)
            del noise_pred_cond, noise_pred_uncond, preds

        latents = sched.step(noise_pred[None], timestep_val, latents[None]).squeeze(0)

        # TI2V-5B: re-apply mask to keep first frame frozen
        if is_i2v_mask_blend:
            latents = (1.0 - i2v_mask) * z_img + i2v_mask * latents

        # Release temporaries before eval to free memory for graph execution
        del noise_pred
        mx.eval(latents)

    print(f"{Colors.DIM}  Denoising: {time.time() - t3:.1f}s{Colors.RESET}")

    if teacache_thresh > 0 and not spectrum and not magcache:
        models_to_report = (
            [("low-noise", low_noise_model), ("high-noise", high_noise_model)]
            if is_dual
            else [("model", single_model)]
        )
        total_skipped = sum(m.teacache.steps_skipped for _, m in models_to_report)
        total_computed = sum(m.teacache.steps_computed for _, m in models_to_report)
        total = total_skipped + total_computed
        if total > 0:
            print(
                f"{Colors.DIM}  TeaCache: {total_skipped}/{total} steps skipped ({total_skipped/total*100:.0f}%){Colors.RESET}"
            )

    if magcache:
        models_to_report = (
            [("high-noise", high_noise_model)] if is_dual else [("model", single_model)]
        )
        # Include low-noise model if it has magcache enabled (custom ratios)
        if is_dual and low_noise_model.magcache.enabled:
            models_to_report.append(("low-noise", low_noise_model))
        total_skipped = sum(m.magcache.steps_skipped for _, m in models_to_report)
        total_computed = sum(m.magcache.steps_computed for _, m in models_to_report)
        total_vetoed = sum(m.magcache.steps_vetoed for _, m in models_to_report)
        total = total_skipped + total_computed
        if total > 0:
            msg = f"  MagCache: {total_skipped}/{total} steps skipped ({total_skipped/total*100:.0f}%)"
            if total_vetoed > 0:
                msg += f", {total_vetoed} Spectrum vetoes"
            print(f"{Colors.DIM}{msg}{Colors.RESET}")

    if magcache_calibrate:
        # Save calibration results to JSON
        output_dir = str(Path(output_path).parent)
        cal_filename = f"magcache_ratios_{config.model_type}_{steps}steps.json"
        cal_path = str(Path(output_dir) / cal_filename)

        results = {}
        if is_dual:
            results["high_noise"] = high_noise_model.magcache.get_calibration_result()
            results["low_noise"] = low_noise_model.magcache.get_calibration_result()
        else:
            results["model"] = single_model.magcache.get_calibration_result()

        save_calibration(
            cal_path,
            model_type=config.model_type,
            model_version=config.model_version,
            steps=steps,
            results=results,
        )

    if spectrum:
        models_to_report = (
            [("low-noise", low_noise_model), ("high-noise", high_noise_model)]
            if is_dual
            else [("model", single_model)]
        )
        total_predicted = sum(m.spectrum.steps_predicted for _, m in models_to_report)
        total_computed = sum(m.spectrum.steps_computed for _, m in models_to_report)
        total = total_predicted + total_computed
        if total > 0:
            print(
                f"{Colors.DIM}  Spectrum: {total_predicted}/{total} steps predicted ({total_predicted/total*100:.0f}% skipped){Colors.RESET}"
            )

    if bwcache:
        models_to_report = (
            [("high-noise", high_noise_model), ("low-noise", low_noise_model)]
            if is_dual
            else [("model", single_model)]
        )
        if bwcache_mode == "block":
            total_block_skipped = sum(
                m.bwcache.blocks_skipped for _, m in models_to_report
            )
            total_block_computed = sum(
                m.bwcache.blocks_computed for _, m in models_to_report
            )
            total_blocks = total_block_skipped + total_block_computed
            if total_blocks > 0:
                print(
                    f"{Colors.DIM}  BWCache (block): {total_block_skipped}/{total_blocks} blocks skipped "
                    f"({total_block_skipped/total_blocks*100:.0f}%){Colors.RESET}"
                )
        else:
            total_skipped = sum(m.bwcache.steps_skipped for _, m in models_to_report)
            total_computed = sum(m.bwcache.steps_computed for _, m in models_to_report)
            total = total_skipped + total_computed
            if total > 0:
                print(
                    f"{Colors.DIM}  BWCache (step): {total_skipped}/{total} steps skipped "
                    f"({total_skipped/total*100:.0f}%){Colors.RESET}"
                )

    # Free transformer models and text embeddings
    if is_dual:
        del low_noise_model, high_noise_model, cross_kv_low, cross_kv_high
        if cfg_disabled:
            del context_cond_low, context_cond_high
        else:
            del context_cfg_low, context_cfg_high
    else:
        del single_model, cross_kv
        if cfg_disabled:
            del context_cond
        else:
            del context_cfg
    del model, kv, context
    if context_null is not None:
        del context_null
    gc.collect()
    mx.clear_cache()

    # Load VAE and decode
    print(f"\n{Colors.BLUE}Decoding with VAE...{Colors.RESET}")
    t4 = time.time()
    vae_path = model_dir / "vae.safetensors"
    vae = load_vae_decoder(vae_path, config)

    is_wan22_vae = config.vae_z_dim == 48

    # Select tiling configuration
    from mlx_video.models.ltx.video_vae.tiling import TilingConfig

    if tiling == "none":
        tiling_config = None
    elif tiling == "auto":
        tiling_config = TilingConfig.auto(height, width, num_frames)
    elif tiling == "default":
        tiling_config = TilingConfig.default()
    elif tiling == "aggressive":
        tiling_config = TilingConfig.aggressive()
    elif tiling == "conservative":
        tiling_config = TilingConfig.conservative()
    elif tiling == "spatial":
        tiling_config = TilingConfig.spatial_only()
    elif tiling == "temporal":
        tiling_config = TilingConfig.temporal_only()
    else:
        print(
            f"{Colors.YELLOW}  Unknown tiling mode '{tiling}', using auto{Colors.RESET}"
        )
        tiling_config = TilingConfig.auto(height, width, num_frames)

    if tiling_config is not None:
        spatial_info = (
            f"{tiling_config.spatial_config.tile_size_in_pixels}px"
            if tiling_config.spatial_config
            else "none"
        )
        temporal_info = (
            f"{tiling_config.temporal_config.tile_size_in_frames}f"
            if tiling_config.temporal_config
            else "none"
        )
        print(
            f"{Colors.DIM}  Tiling ({tiling}): spatial={spatial_info}, temporal={temporal_info}{Colors.RESET}"
        )

    if is_wan22_vae:
        from mlx_video.models.wan.vae22 import denormalize_latents

        # latents: [C, T, H, W] → [1, T, H, W, C] (channels-last for Wan2.2 VAE)
        z = latents.transpose(1, 2, 3, 0)[None]
        z = denormalize_latents(z)
        if tiling_config is not None:
            video = vae.decode_tiled(z, tiling_config)
        else:
            video = vae(z)
        mx.eval(video)
        print(f"{Colors.DIM}  VAE decode: {time.time() - t4:.1f}s{Colors.RESET}")

        video = np.array(video[0])  # [T', H', W', 3]
        video = (video + 1.0) / 2.0
        video = np.clip(video * 255.0, 0, 255).astype(np.uint8)
    else:
        if tiling_config is not None:
            video = vae.decode_tiled(latents[None], tiling_config)
        else:
            video = vae.decode(latents[None])
        mx.eval(video)
        print(f"{Colors.DIM}  VAE decode: {time.time() - t4:.1f}s{Colors.RESET}")

        video = np.array(video[0])  # [3, T', H, W]
        video = (video + 1.0) / 2.0
        video = np.clip(video * 255.0, 0, 255).astype(np.uint8)
        video = video.transpose(1, 2, 3, 0)  # [T, H, W, 3]

    save_video(video, output_path, fps=config.sample_fps)
    print(f"\n{Colors.GREEN}✓ Video saved to {output_path}{Colors.RESET}")
    print(f"{Colors.DIM}  Total time: {time.time() - t1:.1f}s{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(description="Wan Text-to-Video Generation (MLX)")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to converted MLX model directory",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image for I2V (omit for T2V mode)")
    parser.add_argument("--negative-prompt", type=str, default=None,
                        help="Negative prompt for CFG (default: official Chinese prompt from config)")
    parser.add_argument("--no-negative-prompt", action="store_true",
                        help="Disable negative prompt (use empty string instead of config default)")
    parser.add_argument("--width", type=int, default=1280, help="Video width (default: 1280)")
    parser.add_argument("--height", type=int, default=704, help="Video height (default: 704; 720p models use 704)")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames (must be 4n+1)")
    parser.add_argument("--steps", type=int, default=None, help="Number of diffusion steps (default: from config)")
    parser.add_argument("--guide-scale", type=str, default=None, help="Guidance scale: single float or low,high pair")
    parser.add_argument("--shift", type=float, default=None, help="Noise schedule shift (default: from config)")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--output-path", type=str, default="output.mp4", help="Output video path")
    parser.add_argument(
        "--num-frames", type=int, default=81, help="Number of frames (must be 4n+1)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of diffusion steps (default: from config)",
    )
    parser.add_argument(
        "--guide-scale",
        type=str,
        default=None,
        help="Guidance scale: single float or low,high pair",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=None,
        help="Noise schedule shift (default: from config)",
    )
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument(
        "--output-path", type=str, default="output.mp4", help="Output video path"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="unipc",
        choices=["euler", "dpm++", "unipc"],
        help="Diffusion solver: euler (1st order), dpm++ (2nd order), unipc (2nd order PC, default/official)",
    )
    parser.add_argument(
        "--teacache-thresh",
        type=float,
        default=0.0,
        help="TeaCache threshold (0=disabled, 0.1=~2x speedup, 0.2=~3x speedup)",
    )
    parser.add_argument(
        "--teacache-verbose",
        action="store_true",
        default=False,
        help="Print per-step TeaCache diagnostics (rel_l1, rescaled, skip/compute)",
    )
    parser.add_argument(
        "--spectrum",
        action="store_true",
        default=False,
        help="Enable Spectrum acceleration (Chebyshev feature forecasting, ~3.5x speedup)",
    )
    parser.add_argument(
        "--spectrum-w",
        type=float,
        default=0.5,
        help="Spectrum blend weight: 0=Taylor only, 1=Chebyshev only (default: 0.5)",
    )
    parser.add_argument(
        "--spectrum-flex-window",
        type=float,
        default=0.75,
        help="Spectrum window growth rate (0.75=~3.5x speedup, 3.0=~5x, more aggressive)",
    )
    parser.add_argument(
        "--spectrum-warmup",
        type=int,
        default=5,
        help="Spectrum warmup steps (always compute first N steps, default: 5)",
    )
    parser.add_argument(
        "--magcache",
        action="store_true",
        default=False,
        help="Enable MagCache acceleration (magnitude-aware residual caching, ~1.5-2x speedup)",
    )
    parser.add_argument(
        "--magcache-thresh",
        type=float,
        default=0.06,
        help="MagCache accumulated error threshold (default: 0.06, lower=better quality)",
    )
    parser.add_argument(
        "--magcache-K",
        type=int,
        default=2,
        help="MagCache max consecutive skip steps (default: 2)",
    )
    parser.add_argument(
        "--magcache-retention-ratio",
        type=float,
        default=0.2,
        help="MagCache fraction of early steps that always compute (default: 0.2)",
    )
    parser.add_argument(
        "--magcache-verbose",
        action="store_true",
        default=False,
        help="Print per-step MagCache diagnostics",
    )
    parser.add_argument(
        "--magcache-calibrate",
        action="store_true",
        default=False,
        help="Run MagCache calibration (full compute, records magnitude ratios to JSON for future --magcache runs)",
    )
    parser.add_argument(
        "--magcache-ratios",
        type=str,
        default=None,
        metavar="PATH",
        help="Load custom MagCache ratios from a calibration JSON file (overrides built-in ratios)",
    )
    parser.add_argument(
        "--bwcache",
        action="store_true",
        default=False,
        help="Enable BWCache acceleration (block-wise caching for DiT blocks)",
    )
    parser.add_argument(
        "--bwcache-mode",
        type=str,
        default="block",
        choices=["step", "block"],
        help="BWCache mode: 'step' (paper-faithful step-level) or 'block' (per-block skip, default)",
    )
    parser.add_argument(
        "--bwcache-thresh",
        type=float,
        default=0.15,
        help="BWCache similarity threshold δ (default: 0.15, lower=better quality)",
    )
    parser.add_argument(
        "--bwcache-reuse-interval",
        type=int,
        default=3,
        help="BWCache periodic recomputation interval for step mode (default: 3)",
    )
    parser.add_argument(
        "--bwcache-last-step",
        type=float,
        default=0.5,
        help="BWCache fraction of tail steps to always compute in step mode (default: 0.5)",
    )
    parser.add_argument(
        "--bwcache-boundary-blocks",
        type=int,
        default=6,
        help="BWCache: first/last N blocks always computed in block mode (default: 6)",
    )
    parser.add_argument(
        "--bwcache-verbose",
        action="store_true",
        default=False,
        help="Print per-step/block BWCache diagnostics",
    )
    parser.add_argument(
        "--lora",
        nargs=2,
        action="append",
        metavar=("PATH", "STRENGTH"),
        help="Apply a LoRA to all models (repeatable). Format: --lora path.safetensors 0.8",
    )
    parser.add_argument(
        "--lora-high",
        nargs=2,
        action="append",
        metavar=("PATH", "STRENGTH"),
        help="Apply a LoRA to high-noise model only (dual-model, repeatable)",
    )
    parser.add_argument(
        "--lora-low",
        nargs=2,
        action="append",
        metavar=("PATH", "STRENGTH"),
        help="Apply a LoRA to low-noise model only (dual-model, repeatable)",
    )
    parser.add_argument(
        "--tiling",
        type=str,
        default="auto",
        choices=[
            "auto",
            "none",
            "default",
            "aggressive",
            "conservative",
            "spatial",
            "temporal",
        ],
        help="VAE tiling mode to reduce memory during decoding (default: auto)",
    )
    parser.add_argument(
        "--no-compile", action="store_true",
        help="Disable mx.compile on models (for debugging)",
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

    # Parse LoRA configs: convert [path, strength_str] → (path, float)
    def _parse_lora_args(lora_list):
        if not lora_list:
            return None
        return [(path, float(strength)) for path, strength in lora_list]

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
        teacache_thresh=args.teacache_thresh,
        teacache_verbose=args.teacache_verbose,
        spectrum=args.spectrum,
        spectrum_w=args.spectrum_w,
        spectrum_flex_window=args.spectrum_flex_window,
        spectrum_warmup=args.spectrum_warmup,
        magcache=args.magcache,
        magcache_thresh=args.magcache_thresh,
        magcache_K=args.magcache_K,
        magcache_retention_ratio=args.magcache_retention_ratio,
        magcache_verbose=args.magcache_verbose,
        magcache_calibrate=args.magcache_calibrate,
        magcache_ratios=args.magcache_ratios,
        bwcache=args.bwcache,
        bwcache_mode=args.bwcache_mode,
        bwcache_thresh=args.bwcache_thresh,
        bwcache_reuse_interval=args.bwcache_reuse_interval,
        bwcache_last_step=args.bwcache_last_step,
        bwcache_boundary_blocks=args.bwcache_boundary_blocks,
        bwcache_verbose=args.bwcache_verbose,
        loras=_parse_lora_args(args.lora),
        loras_high=_parse_lora_args(args.lora_high),
        loras_low=_parse_lora_args(args.lora_low),
        tiling=args.tiling,
        no_compile=args.no_compile,
    )


if __name__ == "__main__":
    main()
