"""Wan2.2 LoRA training CLI entry point.

Usage:
    python -m mlx_video.train_wan --config train.json
"""

import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx

from mlx_video.utils import Colors


def main():
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter for Wan2.2 video generation (MLX)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config JSON file",
    )
    args = parser.parse_args()

    # Load config
    from mlx_video.training.config import TrainingConfig

    print(f"\n{Colors.CYAN}Loading training config: {args.config}{Colors.RESET}")
    config = TrainingConfig.from_json(args.config)

    print(f"{Colors.DIM}  Model: {config.model_dir}")
    print(f"  Data: {len(config.data_items)} training samples")
    print(f"  Resolution: {config.resolution}")
    print(f"  LoRA rank: {config.lora.rank}, alpha: {config.lora.alpha}")
    if config.trigger_word:
        print(f"  Trigger word: {config.trigger_word}")
    print(f"{Colors.RESET}")

    # Create output directory
    output_dir = Path(config.checkpoint.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the config to output dir
    config_copy = output_dir / "train_config.json"
    with open(args.config) as f:
        raw_config = json.load(f)
    with open(config_copy, "w") as f:
        json.dump(raw_config, f, indent=2)

    # Phase 1: Encode dataset
    from mlx_video.training.dataset import encode_dataset

    encoded_data = encode_dataset(config)

    # Phase 2: Load transformer model
    print(f"\n{Colors.BLUE}Loading transformer model...{Colors.RESET}")
    t0 = time.time()

    model_dir = Path(config.model_dir)
    model_config_path = model_dir / "config.json"

    from mlx_video.models.wan.config import WanModelConfig

    if model_config_path.exists():
        with open(model_config_path) as f:
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

    from mlx_video.models.wan.model import WanModel

    model = WanModel(model_config)

    # Load weights — for dual model, use low_noise_model by default (better for character training)
    if model_config.dual_model:
        weight_path = model_dir / "low_noise_model.safetensors"
        if not weight_path.exists():
            weight_path = model_dir / "high_noise_model.safetensors"
        print(
            f"{Colors.DIM}  Using dual-model weight: {weight_path.name}{Colors.RESET}"
        )
    else:
        weight_path = model_dir / "model.safetensors"

    weights = mx.load(str(weight_path))
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    del weights
    gc.collect()
    mx.clear_cache()

    print(f"{Colors.DIM}  Model loaded: {time.time() - t0:.1f}s{Colors.RESET}")

    # Phase 3: Inject LoRA layers and freeze base
    from mlx_video.training.lora_layers import (
        count_trainable_params,
        freeze_base_weights,
        inject_lora_layers,
    )

    n_injected = inject_lora_layers(model, config.lora)
    freeze_base_weights(model)

    trainable, total = count_trainable_params(model)
    print(f"\n{Colors.GREEN}LoRA injected: {n_injected} layers{Colors.RESET}")
    print(f"{Colors.DIM}  Trainable params: {trainable:,}")
    print(f"  Total params: {total:,}")
    print(f"  Trainable %: {100 * trainable / max(1, total):.2f}%{Colors.RESET}")

    # Phase 4: Train
    from mlx_video.training.trainer import train

    train(model, encoded_data, config)

    print(f"\n{Colors.GREEN}✓ Training complete!{Colors.RESET}")


if __name__ == "__main__":
    main()
