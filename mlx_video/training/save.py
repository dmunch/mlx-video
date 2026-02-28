"""Save LoRA weights in diffusers-compatible safetensors format."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_video.training.config import TrainingConfig
from mlx_video.training.lora_layers import TrainableLoRALinear


def _collect_lora_weights(model: nn.Module) -> dict[str, mx.array]:
    """Extract LoRA weights from model, keyed by their model path.

    Returns dict mapping e.g.:
        "blocks.0.self_attn.q.lora_A.weight" -> mx.array
        "blocks.0.self_attn.q.lora_B.weight" -> mx.array
    """
    lora_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, TrainableLoRALinear):
            lora_weights[f"{name}.lora_A.weight"] = module.lora_A
            lora_weights[f"{name}.lora_B.weight"] = module.lora_B
    return lora_weights


def _to_diffusers_keys(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Convert MLX model keys to diffusers/ai-toolkit format.

    MLX:       blocks.0.self_attn.q.lora_A.weight
    Diffusers: diffusion_model.blocks.0.self_attn.q.lora_A.weight
    """
    converted = {}
    for key, value in weights.items():
        new_key = f"diffusion_model.{key}"
        converted[new_key] = value
    return converted


def save_lora_weights(
    model: nn.Module,
    output_path: str,
    config: TrainingConfig,
) -> None:
    """Save LoRA weights to safetensors file with metadata.

    Args:
        model: WanModel with trained LoRA layers.
        output_path: Path to save the .safetensors file.
        config: Training config for metadata.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect LoRA weights
    lora_weights = _collect_lora_weights(model)

    if not lora_weights:
        raise ValueError(
            "No LoRA weights found in model. Was inject_lora_layers called?"
        )

    # Convert to diffusers key format
    lora_weights = _to_diffusers_keys(lora_weights)

    # Build metadata
    metadata = {
        "format": "mlx_video_lora",
        "lora_rank": str(config.lora.rank),
        "lora_alpha": str(config.lora.alpha),
        "model_version": "wan2.2",
        "targets": json.dumps(config.lora.targets),
    }

    mx.save_safetensors(str(output_path), lora_weights, metadata=metadata)

    n_params = sum(v.size for v in lora_weights.values())
    print(f"  Saved {len(lora_weights)} tensors ({n_params:,} params) to {output_path}")
