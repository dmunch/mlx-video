"""Dataset encoding for Wan2.2 LoRA training.

Loads images and prompts, encodes them through VAE and T5,
and caches the results in memory for efficient training.
"""

import gc
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_video.training.config import TrainingConfig
from mlx_video.utils import Colors


@dataclass
class EncodedItem:
    """Pre-encoded training sample."""

    prompt: str
    image_path: Path
    clean_latents: mx.array  # [z_dim, 1, H_lat, W_lat]
    text_embedding: mx.array  # [text_len, text_dim]
    width: int
    height: int


def _load_and_resize_image(
    image_path: Path, target_size: int
) -> tuple[np.ndarray, int, int]:
    """Load image and resize to target resolution (preserving aspect ratio to nearest 32).

    Returns (image_array [H, W, 3] float32 in [-1, 1], width, height).
    """
    img = Image.open(image_path).convert("RGB")

    # Resize to fit within target_size x target_size, preserving aspect ratio
    w, h = img.size
    scale = target_size / max(w, h)
    new_w = round(w * scale)
    new_h = round(h * scale)

    # Round to nearest multiple of 32 (VAE requirement)
    new_w = max(32, (new_w // 32) * 32)
    new_h = max(32, (new_h // 32) * 32)

    img = img.resize((new_w, new_h), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0  # [H, W, 3]

    return arr, new_w, new_h


def encode_dataset(config: TrainingConfig) -> list[EncodedItem]:
    """Encode all training data through VAE and T5.

    Loads VAE encoder and T5 sequentially to minimize peak memory,
    freeing each after use.

    Args:
        config: Training configuration with data_items populated.

    Returns:
        List of EncodedItem with pre-encoded latents and text embeddings.
    """
    from mlx_video.models.wan.config import WanModelConfig
    from mlx_video.models.wan.loading import (
        encode_text,
        load_t5_encoder,
        load_vae_encoder,
    )

    model_dir = Path(config.model_dir)

    # Load model config
    import json

    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        config_dict.pop("quantization", None)
        for key in ("patch_size", "vae_stride", "window_size", "sample_guide_scale"):
            if key in config_dict and isinstance(config_dict[key], list):
                config_dict[key] = tuple(config_dict[key])
        model_config = WanModelConfig(
            **{
                k: v
                for k, v in config_dict.items()
                if k in WanModelConfig.__dataclass_fields__
            }
        )
    else:
        model_config = WanModelConfig.wan22_t2v_14b()

    # Phase 1: Encode images through VAE
    print(f"\n{Colors.BLUE}Encoding training images through VAE...{Colors.RESET}")
    t0 = time.time()
    vae_path = model_dir / "vae.safetensors"
    vae_enc = load_vae_encoder(vae_path, model_config)

    encoded_latents = []
    for item in config.data_items:
        arr, w, h = _load_and_resize_image(item.image, config.resolution)
        img_tensor = mx.array(arr)  # [H, W, 3]

        is_wan22 = model_config.vae_z_dim == 48
        if is_wan22:
            # Wan2.2 VAE encoder: [B, T, H, W, C] channels-last
            from mlx_video.models.wan.vae22 import normalize_latents

            x = img_tensor[None, None, :, :, :]  # [1, 1, H, W, 3]
            z = vae_enc(x)  # [1, 1, H_lat, W_lat, z_dim]
            z = normalize_latents(z)
            mx.eval(z)
            z = z[0].transpose(3, 0, 1, 2)  # [z_dim, 1, H_lat, W_lat]
        else:
            # Wan2.1 VAE encoder: channels-first
            img_chw = img_tensor.transpose(2, 0, 1)  # [3, H, W]
            x = img_chw[:, None, :, :]  # [3, 1, H, W]
            z = vae_enc.encode(x[None])  # [1, z_dim, 1, H_lat, W_lat]
            mx.eval(z)
            z = z[0]  # [z_dim, 1, H_lat, W_lat]

        encoded_latents.append((z, w, h))
        print(
            f"  {Colors.DIM}{item.image.name}: {w}x{h} → latent {list(z.shape)}{Colors.RESET}"
        )

    del vae_enc
    gc.collect()
    mx.clear_cache()
    print(f"{Colors.DIM}  VAE encoding: {time.time() - t0:.1f}s{Colors.RESET}")

    # Phase 2: Encode prompts through T5
    print(f"\n{Colors.BLUE}Encoding prompts through T5...{Colors.RESET}")
    t1 = time.time()
    t5_path = model_dir / "t5_encoder.safetensors"
    t5_encoder = load_t5_encoder(t5_path, model_config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")

    encoded_texts = []
    for item in config.data_items:
        text_emb = encode_text(
            t5_encoder, tokenizer, item.prompt, model_config.text_len
        )
        mx.eval(text_emb)
        encoded_texts.append(text_emb)
        print(
            f"  {Colors.DIM}{item.prompt[:60]}...{Colors.RESET}"
            if len(item.prompt) > 60
            else f"  {Colors.DIM}{item.prompt}{Colors.RESET}"
        )

    del t5_encoder, tokenizer
    gc.collect()
    mx.clear_cache()
    print(f"{Colors.DIM}  T5 encoding: {time.time() - t1:.1f}s{Colors.RESET}")

    # Combine into EncodedItems
    items = []
    for i, data_item in enumerate(config.data_items):
        z, w, h = encoded_latents[i]
        items.append(
            EncodedItem(
                prompt=data_item.prompt,
                image_path=data_item.image,
                clean_latents=z,
                text_embedding=encoded_texts[i],
                width=w,
                height=h,
            )
        )

    print(f"\n{Colors.GREEN}✓ Encoded {len(items)} training samples{Colors.RESET}")
    return items
