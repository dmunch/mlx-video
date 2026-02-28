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

# Wan2.2 boundary between high-noise and low-noise experts
EXPERT_BOUNDARY = 0.875


def _load_model(model_dir: Path, weight_file: str):
    """Load a WanModel from model_dir with the specified weight file."""
    from mlx_video.models.wan.config import WanModelConfig
    from mlx_video.models.wan.model import WanModel

    model_config_path = model_dir / "config.json"

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

    model = WanModel(model_config)

    weight_path = model_dir / weight_file
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    weights = mx.load(str(weight_path))
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    del weights
    gc.collect()
    mx.clear_cache()

    return model


def _setup_lora(model, lora_config):
    """Inject LoRA layers and freeze base weights. Returns (trainable, total) param counts."""
    from mlx_video.training.lora_layers import (
        count_trainable_params,
        freeze_base_weights,
        inject_lora_layers,
    )

    n_injected = inject_lora_layers(model, lora_config)
    freeze_base_weights(model)
    trainable, total = count_trainable_params(model)

    print(f"\n{Colors.GREEN}LoRA injected: {n_injected} layers{Colors.RESET}")
    print(f"{Colors.DIM}  Trainable params: {trainable:,}")
    print(f"  Total params: {total:,}")
    print(f"  Trainable %: {100 * trainable / max(1, total):.2f}%{Colors.RESET}")

    return trainable, total


def _train_single_expert(
    config, encoded_data, model_dir, weight_file, sigma_min, sigma_max, expert_label, output_suffix,
    resume_path=None,
):
    """Load model, inject LoRA, train on sigma range, unload."""
    from mlx_video.training.save import load_checkpoint
    from mlx_video.training.trainer import train

    print(f"\n{Colors.BLUE}Loading {expert_label} model ({weight_file})...{Colors.RESET}")
    t0 = time.time()
    model = _load_model(model_dir, weight_file)
    print(f"{Colors.DIM}  Model loaded: {time.time() - t0:.1f}s{Colors.RESET}")

    _setup_lora(model, config.lora)

    # Handle resume
    resume_state = None
    if resume_path:
        import mlx.optimizers as optim

        optimizer_cls = {"adam": optim.Adam, "adamw": optim.AdamW}.get(
            config.training.optimizer.lower(), optim.AdamW
        )
        optimizer = optimizer_cls(learning_rate=config.training.learning_rate)
        resume_state = load_checkpoint(resume_path, model, optimizer)
        print(f"{Colors.DIM}  Resumed from: {resume_path}{Colors.RESET}")

    train(
        model,
        encoded_data,
        config,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        expert_label=expert_label,
        output_suffix=output_suffix,
        resume_state=resume_state,
    )

    del model
    gc.collect()
    mx.clear_cache()


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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .zip file to resume training from",
    )
    args = parser.parse_args()

    # Load config
    from mlx_video.training.config import TrainingConfig

    print(f"\n{Colors.CYAN}Loading training config: {args.config}{Colors.RESET}")
    config = TrainingConfig.from_json(args.config)

    experts = config.training.experts
    expert_mode = config.training.expert_mode

    print(f"{Colors.DIM}  Model: {config.model_dir}")
    print(f"  Data: {len(config.data_items)} training samples")
    print(f"  Resolution: {config.resolution}")
    print(f"  LoRA rank: {config.lora.rank}, alpha: {config.lora.alpha}")
    print(f"  Experts: {experts}, mode: {expert_mode}")
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

    # Phase 2: Load model and train based on expert configuration
    model_dir = Path(config.model_dir)

    # Detect if this is a dual-model setup
    has_dual_model = (
        (model_dir / "low_noise_model.safetensors").exists()
        and (model_dir / "high_noise_model.safetensors").exists()
    )

    if experts == "both" and has_dual_model:
        if expert_mode == "sequential":
            # Sequential: train one expert at a time to save memory
            print(f"\n{Colors.CYAN}Sequential dual-expert training{Colors.RESET}")

            # Phase 2a: High noise expert (σ ∈ [boundary, 1.0])
            _train_single_expert(
                config,
                encoded_data,
                model_dir,
                "high_noise_model.safetensors",
                sigma_min=EXPERT_BOUNDARY,
                sigma_max=1.0,
                expert_label="high noise",
                output_suffix="_high_noise",
                resume_path=args.resume,
            )

            # Phase 2b: Low noise expert (σ ∈ [0.0, boundary))
            _train_single_expert(
                config,
                encoded_data,
                model_dir,
                "low_noise_model.safetensors",
                sigma_min=0.0,
                sigma_max=EXPERT_BOUNDARY,
                expert_label="low noise",
                output_suffix="_low_noise",
                resume_path=args.resume,
            )

        else:
            # Simultaneous: both models loaded, route by sigma each step
            print(f"\n{Colors.CYAN}Simultaneous dual-expert training{Colors.RESET}")
            from mlx_video.training.trainer import train_simultaneous

            print(f"\n{Colors.BLUE}Loading both expert models...{Colors.RESET}")
            t0 = time.time()
            high_model = _load_model(model_dir, "high_noise_model.safetensors")
            low_model = _load_model(model_dir, "low_noise_model.safetensors")
            print(f"{Colors.DIM}  Both models loaded: {time.time() - t0:.1f}s{Colors.RESET}")

            _setup_lora(high_model, config.lora)
            print(f"{Colors.DIM}  ^ High noise expert{Colors.RESET}")
            _setup_lora(low_model, config.lora)
            print(f"{Colors.DIM}  ^ Low noise expert{Colors.RESET}")

            train_simultaneous(
                high_model=high_model,
                low_model=low_model,
                encoded_data=encoded_data,
                config=config,
                boundary=EXPERT_BOUNDARY,
            )

            del high_model, low_model
            gc.collect()
            mx.clear_cache()

    elif experts == "high" and has_dual_model:
        _train_single_expert(
            config,
            encoded_data,
            model_dir,
            "high_noise_model.safetensors",
            sigma_min=EXPERT_BOUNDARY,
            sigma_max=1.0,
            expert_label="high noise",
            output_suffix="_high_noise",
            resume_path=args.resume,
        )

    elif experts == "low" and has_dual_model:
        _train_single_expert(
            config,
            encoded_data,
            model_dir,
            "low_noise_model.safetensors",
            sigma_min=0.0,
            sigma_max=EXPERT_BOUNDARY,
            expert_label="low noise",
            output_suffix="_low_noise",
            resume_path=args.resume,
        )

    else:
        # Single model (Wan2.1 or non-dual Wan2.2)
        from mlx_video.training.save import load_checkpoint
        from mlx_video.training.trainer import train

        weight_file = "model.safetensors"
        if has_dual_model:
            weight_file = "low_noise_model.safetensors"

        print(f"\n{Colors.BLUE}Loading model ({weight_file})...{Colors.RESET}")
        t0 = time.time()
        model = _load_model(model_dir, weight_file)
        print(f"{Colors.DIM}  Model loaded: {time.time() - t0:.1f}s{Colors.RESET}")

        _setup_lora(model, config.lora)

        resume_state = None
        if args.resume:
            import mlx.optimizers as optim

            optimizer_cls = {"adam": optim.Adam, "adamw": optim.AdamW}.get(
                config.training.optimizer.lower(), optim.AdamW
            )
            optimizer = optimizer_cls(learning_rate=config.training.learning_rate)
            resume_state = load_checkpoint(args.resume, model, optimizer)
            print(f"{Colors.DIM}  Resumed from: {args.resume}{Colors.RESET}")

        train(model, encoded_data, config, resume_state=resume_state)

        del model
        gc.collect()
        mx.clear_cache()

    print(f"\n{Colors.GREEN}✓ Training complete!{Colors.RESET}")


if __name__ == "__main__":
    main()
