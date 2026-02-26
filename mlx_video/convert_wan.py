"""Weight conversion for Wan2.2 models (PyTorch -> MLX)."""

import logging
from pathlib import Path
from typing import Dict

import mlx.core as mx
import numpy as np


def load_torch_weights(path: str) -> Dict[str, mx.array]:
    """Load PyTorch .pth weights and convert to MLX arrays.

    Args:
        path: Path to .pth file

    Returns:
        Dictionary of MLX arrays
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required to load .pth weights: pip install torch")

    logging.info(f"Loading weights from {path}")
    state_dict = torch.load(path, map_location="cpu", weights_only=True)

    weights = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            np_val = value.detach().float().numpy()
            weights[key] = mx.array(np_val)

    return weights


def load_safetensors_weights(path: str) -> Dict[str, mx.array]:
    """Load safetensors weights as MLX arrays.

    Args:
        path: Path to directory with safetensors files or single file

    Returns:
        Dictionary of MLX arrays
    """
    path = Path(path)
    weights = {}
    if path.is_file():
        weights = mx.load(str(path))
    elif path.is_dir():
        for sf in sorted(path.glob("*.safetensors")):
            weights.update(mx.load(str(sf)))
    return weights


def sanitize_wan_transformer_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Convert Wan2.2 transformer weight keys to MLX model structure.

    Wan2.2 keys follow the pattern:
        patch_embedding.weight/bias
        text_embedding.{0,2}.weight/bias
        time_embedding.{0,2}.weight/bias
        time_projection.1.weight/bias
        blocks.{i}.norm1.weight
        blocks.{i}.self_attn.{q,k,v,o}.weight/bias
        blocks.{i}.self_attn.norm_q.weight
        blocks.{i}.self_attn.norm_k.weight
        blocks.{i}.norm3.weight/bias (if cross_attn_norm)
        blocks.{i}.cross_attn.{q,k,v,o}.weight/bias
        blocks.{i}.cross_attn.norm_q.weight
        blocks.{i}.cross_attn.norm_k.weight
        blocks.{i}.norm2.weight
        blocks.{i}.ffn.{0,2}.weight/bias
        blocks.{i}.modulation
        head.norm.weight
        head.head.weight/bias
        head.modulation
        freqs (buffer)

    MLX model uses:
        patch_embedding_proj.weight/bias (after patchify reshape)
        text_embedding_0.weight/bias, text_embedding_1.weight/bias
        time_embedding_0.weight/bias, time_embedding_1.weight/bias
        time_projection.weight/bias
        blocks.{i}.norm1.weight
        blocks.{i}.self_attn.{q,k,v,o}.weight/bias
        etc.
    """
    sanitized = {}

    for key, value in weights.items():
        new_key = key

        # Patch embedding: Conv3d(16, 5120, (1,2,2)) weight is [O, I, D, H, W]
        # MLX Linear expects [O, I*D*H*W] after we flatten in patchify
        if key == "patch_embedding.weight":
            # Original: [dim, in_dim, 1, 2, 2] -> reshape to [dim, in_dim*1*2*2]
            value = value.reshape(value.shape[0], -1)
            new_key = "patch_embedding_proj.weight"
            sanitized[new_key] = value
            continue
        if key == "patch_embedding.bias":
            new_key = "patch_embedding_proj.bias"
            sanitized[new_key] = value
            continue

        # Text embedding Sequential: 0=Linear, 1=GELU(no params), 2=Linear
        if key.startswith("text_embedding.0."):
            new_key = key.replace("text_embedding.0.", "text_embedding_0.")
            sanitized[new_key] = value
            continue
        if key.startswith("text_embedding.2."):
            new_key = key.replace("text_embedding.2.", "text_embedding_1.")
            sanitized[new_key] = value
            continue

        # Time embedding Sequential: 0=Linear, 1=SiLU(no params), 2=Linear
        if key.startswith("time_embedding.0."):
            new_key = key.replace("time_embedding.0.", "time_embedding_0.")
            sanitized[new_key] = value
            continue
        if key.startswith("time_embedding.2."):
            new_key = key.replace("time_embedding.2.", "time_embedding_1.")
            sanitized[new_key] = value
            continue

        # Time projection Sequential: 0=SiLU(no params), 1=Linear
        if key.startswith("time_projection.1."):
            new_key = key.replace("time_projection.1.", "time_projection.")
            sanitized[new_key] = value
            continue

        # FFN: Sequential(Linear, GELU, Linear) -> ffn.{0,2} -> ffn.fc1, ffn.fc2
        new_key = new_key.replace(".ffn.0.", ".ffn.fc1.")
        new_key = new_key.replace(".ffn.2.", ".ffn.fc2.")

        # Skip the freqs buffer (we compute it in the model)
        if key == "freqs":
            continue

        sanitized[new_key] = value

    return sanitized


def sanitize_wan_t5_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Convert Wan2.2 T5 encoder weight keys to MLX T5Encoder structure.

    Wan2.2 T5 keys:
        token_embedding.weight
        pos_embedding.embedding.weight (if shared_pos)
        blocks.{i}.norm1.weight
        blocks.{i}.attn.{q,k,v,o}.weight
        blocks.{i}.norm2.weight
        blocks.{i}.ffn.gate.0.weight  (gate linear)
        blocks.{i}.ffn.fc1.weight
        blocks.{i}.ffn.fc2.weight
        blocks.{i}.pos_embedding.embedding.weight (if not shared_pos)
        norm.weight

    MLX T5Encoder structure:
        token_embedding.weight
        blocks.{i}.norm1.weight
        blocks.{i}.attn.{q,k,v,o}.weight
        blocks.{i}.norm2.weight
        blocks.{i}.ffn.gate_proj.weight  (mapped from gate.0)
        blocks.{i}.ffn.fc1.weight
        blocks.{i}.ffn.fc2.weight
        blocks.{i}.pos_embedding.embedding.weight
        norm.weight
    """
    sanitized = {}

    for key, value in weights.items():
        new_key = key

        # Map gate.0 -> gate_proj (the GELU is a separate module, not a parameter)
        new_key = new_key.replace(".ffn.gate.0.", ".ffn.gate_proj.")

        sanitized[new_key] = value

    return sanitized


def sanitize_wan_vae_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Convert Wan2.2 VAE weight keys to MLX WanVAE structure.

    Handles Conv3d and Conv2d weight transpositions for MLX format.
    """
    sanitized = {}

    for key, value in weights.items():
        new_key = key

        # Handle Conv3d: PyTorch [O, I, D, H, W] -> MLX CausalConv3d weight [O, D, H, W, I]
        if "weight" in key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))

        # Handle Conv2d: PyTorch [O, I, H, W] -> MLX [O, H, W, I]
        if "weight" in key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        # Map decoder keys to MLX decoder structure
        # Wan2.2 uses encoder/decoder with downsamples/upsamples
        # Need to adapt naming for our simplified structure

        sanitized[new_key] = value

    return sanitized


def convert_wan_checkpoint(
    checkpoint_dir: str,
    output_dir: str,
    dtype: str = "bfloat16",
):
    """Convert a full Wan2.2 checkpoint directory to MLX format.

    Expected directory structure:
        checkpoint_dir/
            models_t5_umt5-xxl-enc-bf16.pth  (T5 encoder)
            Wan2.1_VAE.pth                     (VAE)
            low_noise_model/                   (safetensors)
            high_noise_model/                  (safetensors)

    Args:
        checkpoint_dir: Path to Wan2.2 checkpoint directory
        output_dir: Path to output MLX model directory
        dtype: Target dtype
    """
    import json

    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }
    target_dtype = dtype_map.get(dtype, mx.bfloat16)

    # Convert low-noise transformer
    low_noise_path = checkpoint_dir / "low_noise_model"
    if low_noise_path.exists():
        print("Converting low-noise transformer...")
        weights = load_safetensors_weights(str(low_noise_path))
        weights = sanitize_wan_transformer_weights(weights)
        weights = {k: v.astype(target_dtype) for k, v in weights.items()}
        out_path = output_dir / "low_noise_model.safetensors"
        mx.save_safetensors(str(out_path), weights)
        print(f"  Saved {len(weights)} weight tensors to {out_path}")

    # Convert high-noise transformer
    high_noise_path = checkpoint_dir / "high_noise_model"
    if high_noise_path.exists():
        print("Converting high-noise transformer...")
        weights = load_safetensors_weights(str(high_noise_path))
        weights = sanitize_wan_transformer_weights(weights)
        weights = {k: v.astype(target_dtype) for k, v in weights.items()}
        out_path = output_dir / "high_noise_model.safetensors"
        mx.save_safetensors(str(out_path), weights)
        print(f"  Saved {len(weights)} weight tensors to {out_path}")

    # Convert T5 encoder
    t5_path = checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    if t5_path.exists():
        print("Converting T5 encoder...")
        weights = load_torch_weights(str(t5_path))
        weights = sanitize_wan_t5_weights(weights)
        weights = {k: v.astype(target_dtype) for k, v in weights.items()}
        out_path = output_dir / "t5_encoder.safetensors"
        mx.save_safetensors(str(out_path), weights)
        print(f"  Saved {len(weights)} weight tensors to {out_path}")

    # Convert VAE
    vae_path = checkpoint_dir / "Wan2.1_VAE.pth"
    if vae_path.exists():
        print("Converting VAE...")
        weights = load_torch_weights(str(vae_path))
        weights = sanitize_wan_vae_weights(weights)
        weights = {k: v.astype(target_dtype) for k, v in weights.items()}
        out_path = output_dir / "vae.safetensors"
        mx.save_safetensors(str(out_path), weights)
        print(f"  Saved {len(weights)} weight tensors to {out_path}")

    # Save config
    from mlx_video.models.wan.config import WanModelConfig
    config = WanModelConfig()
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"  Saved config to {config_path}")

    print(f"\nConversion complete! Output: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Wan2.2 model to MLX format")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to Wan2.2 checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="wan_mlx_model",
        help="Output path for MLX model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32", "bfloat16"],
        default="bfloat16",
        help="Target dtype",
    )
    args = parser.parse_args()
    convert_wan_checkpoint(args.checkpoint_dir, args.output_dir, args.dtype)
