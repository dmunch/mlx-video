"""Wan2.2 Text-to-Video generation pipeline for MLX."""

import argparse
import math
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tqdm import tqdm


class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def load_wan_model(model_path: Path, config):
    """Load and initialize WanModel."""
    from mlx_video.models.wan.model import WanModel

    model = WanModel(config)
    weights = mx.load(str(model_path))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model


def load_t5_encoder(model_path: Path, config):
    """Load T5 text encoder."""
    from mlx_video.models.wan.text_encoder import T5Encoder

    encoder = T5Encoder(
        vocab_size=config.t5_vocab_size,
        dim=config.t5_dim,
        dim_attn=config.t5_dim_attn,
        dim_ffn=config.t5_dim_ffn,
        num_heads=config.t5_num_heads,
        num_layers=config.t5_num_layers,
        num_buckets=config.t5_num_buckets,
        shared_pos=False,
    )
    weights = mx.load(str(model_path))
    encoder.load_weights(list(weights.items()))
    mx.eval(encoder.parameters())
    return encoder


def load_vae_decoder(model_path: Path):
    """Load VAE decoder."""
    from mlx_video.models.wan.vae import WanVAE

    vae = WanVAE(z_dim=16)
    weights = mx.load(str(model_path))
    vae.load_weights(list(weights.items()))
    mx.eval(vae.parameters())
    return vae


def encode_text(
    encoder,
    tokenizer,
    prompt: str,
    text_len: int = 512,
) -> mx.array:
    """Encode text prompt using T5 encoder.

    Args:
        encoder: T5Encoder model
        tokenizer: HuggingFace tokenizer
        prompt: Text prompt
        text_len: Maximum text length

    Returns:
        Text embeddings [L, dim]
    """
    tokens = tokenizer(
        prompt,
        max_length=text_len,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    ids = mx.array(tokens["input_ids"])
    mask = mx.array(tokens["attention_mask"])

    embeddings = encoder(ids, mask=mask)

    # Return only non-padding tokens
    seq_len = int(mask.sum().item())
    return embeddings[0, :seq_len]


def generate_video(
    model_dir: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1280,
    height: int = 720,
    num_frames: int = 81,
    steps: int = 40,
    guide_scale: tuple = (3.0, 4.0),
    shift: float = 12.0,
    seed: int = -1,
    output_path: str = "output.mp4",
):
    """Generate video using Wan2.2 T2V pipeline.

    Args:
        model_dir: Path to converted MLX model directory
        prompt: Text prompt
        negative_prompt: Negative prompt
        width: Video width
        height: Video height
        num_frames: Number of frames (must be 4n+1)
        steps: Number of diffusion steps
        guide_scale: (low_noise_scale, high_noise_scale)
        shift: Noise schedule shift
        seed: Random seed (-1 for random)
        output_path: Output video path
    """
    from mlx_video.models.wan.config import WanModelConfig
    from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler

    model_dir = Path(model_dir)
    config = WanModelConfig()

    # Validate frame count
    assert (num_frames - 1) % 4 == 0, f"num_frames must be 4n+1, got {num_frames}"

    print(f"{Colors.CYAN}{'='*60}")
    print(f"  Wan2.2 Text-to-Video Generation (MLX)")
    print(f"{'='*60}{Colors.RESET}")
    print(f"{Colors.DIM}  Prompt: {prompt}")
    print(f"  Size: {width}x{height}, Frames: {num_frames}")
    print(f"  Steps: {steps}, Guide: {guide_scale}, Shift: {shift}")
    print(f"{Colors.RESET}")

    # Seed
    if seed < 0:
        seed = random.randint(0, sys.maxsize)
    mx.random.seed(seed)
    np.random.seed(seed)
    print(f"{Colors.DIM}  Seed: {seed}{Colors.RESET}")

    # Compute target latent shape
    vae_stride = config.vae_stride
    z_dim = config.vae_z_dim
    t_latent = (num_frames - 1) // vae_stride[0] + 1
    h_latent = height // vae_stride[1]
    w_latent = width // vae_stride[2]
    target_shape = (z_dim, t_latent, h_latent, w_latent)

    # Sequence length for transformer
    patch_size = config.patch_size
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
    if negative_prompt:
        context_null = encode_text(t5_encoder, tokenizer, negative_prompt, config.text_len)
    else:
        context_null = encode_text(t5_encoder, tokenizer, "", config.text_len)
    mx.eval(context, context_null)

    # Free T5 from memory
    del t5_encoder
    mx.metal.clear_cache() if hasattr(mx, "metal") else None
    print(f"{Colors.DIM}  T5 encoding: {time.time() - t1:.1f}s{Colors.RESET}")

    # Load transformer models
    print(f"\n{Colors.BLUE}Loading transformer models...{Colors.RESET}")
    t2 = time.time()

    low_noise_path = model_dir / "low_noise_model.safetensors"
    high_noise_path = model_dir / "high_noise_model.safetensors"
    low_noise_model = load_wan_model(low_noise_path, config)
    high_noise_model = load_wan_model(high_noise_path, config)
    print(f"{Colors.DIM}  Models loaded: {time.time() - t2:.1f}s{Colors.RESET}")

    # Setup scheduler
    scheduler = FlowMatchEulerScheduler(num_train_timesteps=config.num_train_timesteps)
    scheduler.set_timesteps(steps, shift=shift)

    # Generate initial noise
    noise = mx.random.normal(target_shape)

    # Boundary for model switching
    boundary = config.boundary * config.num_train_timesteps

    # Diffusion loop
    print(f"\n{Colors.GREEN}Denoising ({steps} steps)...{Colors.RESET}")
    latents = noise
    t3 = time.time()

    for i, t in enumerate(tqdm(range(steps), desc="Diffusion")):
        timestep_val = scheduler.timesteps[i].item()
        timestep = mx.array([timestep_val])

        # Select model and guide scale based on timestep
        if timestep_val >= boundary:
            model = high_noise_model
            gs = guide_scale[1]
        else:
            model = low_noise_model
            gs = guide_scale[0]

        # Conditional prediction
        noise_pred_cond = model(
            [latents],
            t=timestep,
            context=[context],
            seq_len=seq_len,
        )[0]

        # Unconditional prediction
        noise_pred_uncond = model(
            [latents],
            t=timestep,
            context=[context_null],
            seq_len=seq_len,
        )[0]

        # Classifier-free guidance
        noise_pred = noise_pred_uncond + gs * (noise_pred_cond - noise_pred_uncond)

        # Scheduler step
        latents = scheduler.step(noise_pred[None], timestep, latents[None]).squeeze(0)
        mx.eval(latents)

    print(f"{Colors.DIM}  Denoising: {time.time() - t3:.1f}s{Colors.RESET}")

    # Free transformer models
    del low_noise_model, high_noise_model
    mx.metal.clear_cache() if hasattr(mx, "metal") else None

    # Load VAE and decode
    print(f"\n{Colors.BLUE}Decoding with VAE...{Colors.RESET}")
    t4 = time.time()
    vae_path = model_dir / "vae.safetensors"
    vae = load_vae_decoder(vae_path)

    video = vae.decode(latents[None])  # [1, 3, T, H, W]
    mx.eval(video)
    print(f"{Colors.DIM}  VAE decode: {time.time() - t4:.1f}s{Colors.RESET}")

    # Post-process and save
    video = np.array(video[0])  # [3, T, H, W]
    video = (video + 1.0) / 2.0  # [-1,1] -> [0,1]
    video = np.clip(video * 255.0, 0, 255).astype(np.uint8)
    video = video.transpose(1, 2, 3, 0)  # [T, H, W, 3]

    save_video(video, output_path, fps=config.sample_fps)
    print(f"\n{Colors.GREEN}✓ Video saved to {output_path}{Colors.RESET}")
    print(f"{Colors.DIM}  Total time: {time.time() - t1:.1f}s{Colors.RESET}")


def save_video(frames: np.ndarray, output_path: str, fps: int = 16):
    """Save video frames to MP4.

    Args:
        frames: Video frames [T, H, W, 3] uint8
        output_path: Output file path
        fps: Frames per second
    """
    try:
        import cv2
        h, w = frames.shape[1], frames.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
    except ImportError:
        # Fallback: save as individual PNGs
        from PIL import Image
        out_dir = Path(output_path).parent / Path(output_path).stem
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(out_dir / f"frame_{i:04d}.png")
        print(f"  (cv2 not available, saved {len(frames)} frames to {out_dir}/)")


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 Text-to-Video Generation (MLX)")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to converted MLX model directory")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames (must be 4n+1)")
    parser.add_argument("--steps", type=int, default=40, help="Number of diffusion steps")
    parser.add_argument("--guide-scale", type=str, default="3.0,4.0", help="Guidance scale (low,high)")
    parser.add_argument("--shift", type=float, default=12.0, help="Noise schedule shift")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--output-path", type=str, default="output.mp4", help="Output video path")
    args = parser.parse_args()

    # Parse guide scale
    guide_scale = tuple(float(x) for x in args.guide_scale.split(","))
    if len(guide_scale) == 1:
        guide_scale = (guide_scale[0], guide_scale[0])

    generate_video(
        model_dir=args.model_dir,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        steps=args.steps,
        guide_scale=guide_scale,
        shift=args.shift,
        seed=args.seed,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
