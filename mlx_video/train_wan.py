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
import mlx.utils

from mlx_video.utils import Colors

# Wan2.2 boundary between high-noise and low-noise experts
EXPERT_BOUNDARY = 0.875


def _resolve_base_loras(config, expert: str) -> list[tuple[str, float]] | None:
    """Resolve base_loras config entries for a specific expert.

    Args:
        config: TrainingConfig with base_loras list.
        expert: "high", "low", or "single" (for non-dual models).

    Returns:
        List of (path, strength) tuples, or None if empty.
    """
    if not config.base_loras:
        return None
    result = []
    for entry in config.base_loras:
        if entry.expert == "both" or entry.expert == expert:
            result.append((entry.path, entry.strength))
        elif expert == "single" and entry.expert == "both":
            result.append((entry.path, entry.strength))
    return result or None


def _load_model(model_dir: Path, weight_file: str, base_loras=None):
    """Load a WanModel from model_dir with the specified weight file.

    Automatically detects pre-quantized models from config.json metadata
    and creates QuantizedLinear stubs before loading weights.

    Args:
        model_dir: Path to model directory.
        weight_file: Name of the weights safetensors file.
        base_loras: Optional list of (lora_path, strength) tuples to merge
                    into base weights before returning.
    """
    import mlx.nn as nn

    from mlx_video.models.wan.config import WanModelConfig
    from mlx_video.models.wan.model import WanModel

    model_config_path = model_dir / "config.json"
    quantization = None

    if model_config_path.exists():
        with open(model_config_path) as f:
            config_dict = json.load(f)
        quantization = config_dict.pop("quantization", None)
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

    # For pre-quantized models, create QuantizedLinear stubs before loading
    if quantization:
        from mlx_video.convert_wan import _quantize_predicate

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            class_predicate=lambda path, m: _quantize_predicate(path, m),
        )

    weight_path = model_dir / weight_file
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    weights = mx.load(str(weight_path))

    # Merge base LoRAs into weights before loading into model
    if base_loras:
        from mlx_video.convert_wan import load_and_apply_loras

        weights = load_and_apply_loras(dict(weights), base_loras)

    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    # Validate model weights: check for NaN/Inf in bias tensors
    bad_keys = []
    for name, param in mlx.utils.tree_flatten(model.parameters()):
        if name.endswith(".bias") and not name.endswith(".biases"):
            if mx.any(mx.isnan(param)).item() or mx.any(mx.isinf(param)).item():
                bad_keys.append(name)
    if bad_keys:
        raise RuntimeError(
            f"Corrupted model weights in {weight_file}: "
            f"{len(bad_keys)} bias tensors contain NaN/Inf "
            f"(e.g. {bad_keys[0]}). Please reconvert the model with: "
            f"python -m mlx_video.convert_wan --checkpoint-dir <src> "
            f"--output-dir {model_dir} --quantize"
        )

    del weights
    gc.collect()
    mx.clear_cache()

    return model, quantization


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


def _load_dual_resume(checkpoint_path, high_model, low_model):
    """Load LoRA weights + training state from a dual-expert checkpoint zip.

    Restores LoRA weights into both models and returns epoch/step/loss_history.
    Optimizer state is loaded later by train_simultaneous after creating optimizers.
    """
    import tempfile
    import zipfile
    from pathlib import Path

    from mlx_video.training.save import _load_lora_from_file

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with zipfile.ZipFile(checkpoint_path, "r") as zf:
            zf.extractall(tmpdir)

        # Restore LoRA weights for both experts
        _load_lora_from_file(high_model, "lora_high_noise.safetensors", tmpdir)
        _load_lora_from_file(low_model, "lora_low_noise.safetensors", tmpdir)

        # Load training state
        state_path = tmpdir / "state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
        else:
            state = {"epoch": 0, "global_step": 0, "loss_history": []}

    mx.eval(high_model.parameters(), low_model.parameters())
    # Include checkpoint path so trainer can restore optimizer state
    state["checkpoint_path"] = str(checkpoint_path)
    return state


def _train_single_expert(
    config, encoded_data, model_dir, weight_file, sigma_min, sigma_max, expert_label, output_suffix,
    resume_path=None,
):
    """Load model, inject LoRA, train on sigma range, unload."""
    from mlx_video.training.trainer import train

    print(f"\n{Colors.BLUE}Loading {expert_label} model ({weight_file})...{Colors.RESET}")
    t0 = time.time()
    # Determine expert type from label for base LoRA filtering
    expert_type = "high" if "high" in expert_label else "low"
    base_loras = _resolve_base_loras(config, expert_type)
    model, quantization = _load_model(model_dir, weight_file, base_loras=base_loras)
    if quantization:
        print(f"{Colors.DIM}  Quantized: {quantization['bits']}-bit (group_size={quantization['group_size']}){Colors.RESET}")
    print(f"{Colors.DIM}  Model loaded: {time.time() - t0:.1f}s{Colors.RESET}")

    _setup_lora(model, config.lora)

    # Handle resume: restore LoRA weights + training state
    resume_state = None
    if resume_path:
        from mlx_video.training.save import _load_lora_from_file

        import tempfile
        import zipfile
        from pathlib import Path as _Path

        ckpt = _Path(resume_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = _Path(tmpdir)
            with zipfile.ZipFile(ckpt, "r") as zf:
                zf.extractall(tmpdir)

            # Restore LoRA weights
            _load_lora_from_file(model, "lora_weights.safetensors", tmpdir)

            # Load training state
            state_path = tmpdir / "state.json"
            if state_path.exists():
                with open(state_path) as f:
                    resume_state = json.load(f)
            else:
                resume_state = {"epoch": 0, "global_step": 0, "loss_history": []}

        mx.eval(model.parameters())
        resume_state["checkpoint_path"] = str(ckpt)
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
    if config.base_loras:
        for bl in config.base_loras:
            print(f"  Base LoRA: {Path(bl.path).name} (expert={bl.expert}, strength={bl.strength})")
    if config.training.shift is not None:
        print(f"  Shift override: {config.training.shift}")
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
            high_base_loras = _resolve_base_loras(config, "high")
            low_base_loras = _resolve_base_loras(config, "low")
            high_model, q1 = _load_model(model_dir, "high_noise_model.safetensors", base_loras=high_base_loras)
            low_model, q2 = _load_model(model_dir, "low_noise_model.safetensors", base_loras=low_base_loras)
            quantization = q1 or q2
            if quantization:
                print(f"{Colors.DIM}  Quantized: {quantization['bits']}-bit{Colors.RESET}")
            print(f"{Colors.DIM}  Both models loaded: {time.time() - t0:.1f}s{Colors.RESET}")

            _setup_lora(high_model, config.lora)
            print(f"{Colors.DIM}  ^ High noise expert{Colors.RESET}")
            _setup_lora(low_model, config.lora)
            print(f"{Colors.DIM}  ^ Low noise expert{Colors.RESET}")

            # Handle resume for simultaneous training
            resume_state = None
            if args.resume:
                resume_state = _load_dual_resume(
                    args.resume, high_model, low_model,
                )
                print(f"{Colors.DIM}  Resumed from: {args.resume}{Colors.RESET}")

            train_simultaneous(
                high_model=high_model,
                low_model=low_model,
                encoded_data=encoded_data,
                config=config,
                boundary=EXPERT_BOUNDARY,
                resume_state=resume_state,
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
        single_base_loras = _resolve_base_loras(config, "single")
        model, quantization = _load_model(model_dir, weight_file, base_loras=single_base_loras)
        if quantization:
            print(f"{Colors.DIM}  Quantized: {quantization['bits']}-bit{Colors.RESET}")
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
