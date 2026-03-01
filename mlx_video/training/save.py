"""Save LoRA weights in diffusers-compatible safetensors format."""

from __future__ import annotations

import json
import zipfile
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


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    epoch: int,
    global_step: int,
    loss_history_data: list,
    output_path: str,
    expert_label: str = "",
) -> None:
    """Save a training checkpoint as a zip file.

    Contains:
      - lora_weights.safetensors: Current LoRA weights
      - optimizer_state.safetensors: Adam optimizer state for resume
      - state.json: Training state (epoch, step, loss history)
      - config.json: Training config snapshot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save LoRA weights
        _save_lora_to_file(model, "lora_weights.safetensors", tmpdir)

        # Save optimizer state
        _save_optimizer_state(optimizer, "", tmpdir)

        # Save training state
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "loss_history": loss_history_data,
            "expert_label": expert_label,
        }
        state_path = tmpdir / "state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        # Save config
        config_path = tmpdir / "config.json"
        with open(config_path, "w") as f:
            # Re-read from original if available, else serialize what we have
            json.dump(
                {
                    "model_dir": config.model_dir,
                    "resolution": config.resolution,
                    "seed": config.seed,
                    "trigger_word": config.trigger_word,
                    "lora": {
                        "rank": config.lora.rank,
                        "alpha": config.lora.alpha,
                    },
                },
                f,
                indent=2,
            )

        # Create zip
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in tmpdir.iterdir():
                zf.write(file, file.name)

    print(f"  Checkpoint saved: {output_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> dict:
    """Load a training checkpoint from a zip file.

    Restores LoRA weights and optimizer state into the given model/optimizer.

    Args:
        checkpoint_path: Path to checkpoint .zip file.
        model: Model with LoRA layers already injected.
        optimizer: Optimizer to restore state into.

    Returns:
        Dict with 'epoch', 'global_step', 'loss_history', 'expert_label'.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        with zipfile.ZipFile(checkpoint_path, "r") as zf:
            zf.extractall(tmpdir)

        # Restore LoRA weights
        lora_path = tmpdir / "lora_weights.safetensors"
        if lora_path.exists():
            saved_weights = mx.load(str(lora_path))
            # Convert from diffusers keys back to model keys
            model_weights = {}
            for key, value in saved_weights.items():
                model_key = key.replace("diffusion_model.", "", 1)
                model_weights[model_key] = value

            # Apply to model's LoRA layers
            for name, module in model.named_modules():
                if isinstance(module, TrainableLoRALinear):
                    a_key = f"{name}.lora_A.weight"
                    b_key = f"{name}.lora_B.weight"
                    if a_key in model_weights:
                        module.lora_A = model_weights[a_key]
                    if b_key in model_weights:
                        module.lora_B = model_weights[b_key]

        # Restore optimizer state
        opt_path = tmpdir / "optimizer_state.safetensors"
        if opt_path.exists():
            saved_opt = mx.load(str(opt_path))
            # Rebuild the optimizer state tree from flat dict
            if saved_opt:
                import mlx.utils

                opt_flat = list(mlx.utils.tree_flatten(optimizer.state))
                for i, (key, _) in enumerate(opt_flat):
                    lookup = f"{i}.{key}"
                    if lookup in saved_opt:
                        opt_flat[i] = (key, saved_opt[lookup])
                optimizer.state = mlx.utils.tree_unflatten(opt_flat)

        # Load training state
        state_path = tmpdir / "state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
        else:
            state = {"epoch": 0, "global_step": 0, "loss_history": []}

    mx.eval(model.parameters(), optimizer.state)
    return state


def _save_optimizer_state(optimizer, prefix: str, tmpdir: Path) -> None:
    """Save optimizer state to safetensors file.

    prefix="" -> optimizer_state.safetensors
    prefix="high" -> high_optimizer_state.safetensors
    """
    import mlx.utils

    opt_state = {}
    for i, (key, value) in enumerate(mlx.utils.tree_flatten(optimizer.state)):
        opt_state[f"{i}.{key}"] = value
    if opt_state:
        fname = f"{prefix}_optimizer_state.safetensors" if prefix else "optimizer_state.safetensors"
        opt_path = tmpdir / fname
        mx.save_safetensors(str(opt_path), opt_state)


def _load_optimizer_state(optimizer, prefix: str, tmpdir: Path) -> None:
    """Load optimizer state from safetensors file.

    prefix="" -> optimizer_state.safetensors
    prefix="high" -> high_optimizer_state.safetensors
    """
    import mlx.utils

    fname = f"{prefix}_optimizer_state.safetensors" if prefix else "optimizer_state.safetensors"
    opt_path = tmpdir / fname
    if opt_path.exists():
        saved_opt = mx.load(str(opt_path))
        if saved_opt:
            opt_flat = list(mlx.utils.tree_flatten(optimizer.state))
            for i, (key, _) in enumerate(opt_flat):
                lookup = f"{i}.{key}"
                if lookup in saved_opt:
                    opt_flat[i] = (key, saved_opt[lookup])
            optimizer.state = mlx.utils.tree_unflatten(opt_flat)


def _save_lora_to_file(model: nn.Module, filename: str, tmpdir: Path) -> None:
    """Save LoRA weights to a safetensors file in tmpdir."""
    lora_weights = _collect_lora_weights(model)
    lora_weights = _to_diffusers_keys(lora_weights)
    mx.save_safetensors(str(tmpdir / filename), lora_weights)


def _load_lora_from_file(model: nn.Module, filename: str, tmpdir: Path) -> None:
    """Load LoRA weights from a safetensors file in tmpdir."""
    lora_path = tmpdir / filename
    if lora_path.exists():
        saved_weights = mx.load(str(lora_path))
        model_weights = {}
        for key, value in saved_weights.items():
            model_key = key.replace("diffusion_model.", "", 1)
            model_weights[model_key] = value

        for name, module in model.named_modules():
            if isinstance(module, TrainableLoRALinear):
                a_key = f"{name}.lora_A.weight"
                b_key = f"{name}.lora_B.weight"
                if a_key in model_weights:
                    module.lora_A = model_weights[a_key]
                if b_key in model_weights:
                    module.lora_B = model_weights[b_key]


def save_dual_checkpoint(
    high_model: nn.Module,
    low_model: nn.Module,
    high_optimizer: optim.Optimizer,
    low_optimizer: optim.Optimizer,
    config: TrainingConfig,
    epoch: int,
    global_step: int,
    loss_history_data: list,
    output_path: str,
) -> None:
    """Save a dual-expert training checkpoint as a zip file.

    Contains:
      - lora_high_noise.safetensors: High-noise expert LoRA weights
      - lora_low_noise.safetensors: Low-noise expert LoRA weights
      - high_optimizer_state.safetensors: High-noise optimizer state
      - low_optimizer_state.safetensors: Low-noise optimizer state
      - state.json: Training state (epoch, step, loss history)
      - config.json: Training config snapshot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save LoRA weights for both experts
        _save_lora_to_file(high_model, "lora_high_noise.safetensors", tmpdir)
        _save_lora_to_file(low_model, "lora_low_noise.safetensors", tmpdir)

        # Save optimizer states
        _save_optimizer_state(high_optimizer, "high", tmpdir)
        _save_optimizer_state(low_optimizer, "low", tmpdir)

        # Save training state
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "loss_history": loss_history_data,
            "expert_mode": "simultaneous",
        }
        state_path = tmpdir / "state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        # Save config
        config_path = tmpdir / "config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "model_dir": config.model_dir,
                    "resolution": config.resolution,
                    "seed": config.seed,
                    "trigger_word": config.trigger_word,
                    "lora": {
                        "rank": config.lora.rank,
                        "alpha": config.lora.alpha,
                    },
                },
                f,
                indent=2,
            )

        # Create zip
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in tmpdir.iterdir():
                zf.write(file, file.name)

    print(f"  Checkpoint saved: {output_path}")


def load_dual_checkpoint(
    checkpoint_path: str,
    high_model: nn.Module,
    low_model: nn.Module,
    high_optimizer: optim.Optimizer,
    low_optimizer: optim.Optimizer,
) -> dict:
    """Load a dual-expert training checkpoint from a zip file.

    Restores LoRA weights and optimizer state for both experts.

    Args:
        checkpoint_path: Path to checkpoint .zip file.
        high_model: High-noise model with LoRA layers already injected.
        low_model: Low-noise model with LoRA layers already injected.
        high_optimizer: High-noise optimizer to restore state into.
        low_optimizer: Low-noise optimizer to restore state into.

    Returns:
        Dict with 'epoch', 'global_step', 'loss_history', 'expert_mode'.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        with zipfile.ZipFile(checkpoint_path, "r") as zf:
            zf.extractall(tmpdir)

        # Restore LoRA weights for both experts
        _load_lora_from_file(high_model, "lora_high_noise.safetensors", tmpdir)
        _load_lora_from_file(low_model, "lora_low_noise.safetensors", tmpdir)

        # Restore optimizer states
        _load_optimizer_state(high_optimizer, "high", tmpdir)
        _load_optimizer_state(low_optimizer, "low", tmpdir)

        # Load training state
        state_path = tmpdir / "state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
        else:
            state = {"epoch": 0, "global_step": 0, "loss_history": []}

    mx.eval(
        high_model.parameters(),
        low_model.parameters(),
        high_optimizer.state,
        low_optimizer.state,
    )
    return state
