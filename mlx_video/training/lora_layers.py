"""Trainable LoRA layers for Wan2.2 training.

Provides LoRA linear wrappers that can be injected into the model,
with freeze/unfreeze support for training only LoRA parameters.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from mlx_video.training.config import LoRAConfig


class TrainableLoRALinear(nn.Module):
    """Linear layer wrapped with trainable LoRA matrices.

    Forward: output = linear(x) + scale * (x @ lora_A.T @ lora_B.T)

    lora_A is initialized with Kaiming uniform, lora_B is initialized to zero,
    so the LoRA contribution starts at zero (standard practice).
    """

    def __init__(self, linear: nn.Linear | nn.QuantizedLinear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # QuantizedLinear packs weights into uint32 — unpack to get true dimensions
        if isinstance(linear, nn.QuantizedLinear):
            in_features = linear.weight.shape[1] * 32 // linear.bits
            out_features = linear.weight.shape[0]
        else:
            in_features = linear.weight.shape[1]
            out_features = linear.weight.shape[0]

        # Kaiming uniform init for A, zero init for B
        bound = 1.0 / math.sqrt(in_features)
        self.lora_A = mx.random.uniform(
            low=-bound, high=bound, shape=(rank, in_features)
        )
        self.lora_B = mx.zeros((out_features, rank))

    @property
    def weight(self):
        """Expose the underlying linear weight for dtype/shape introspection."""
        return self.linear.weight

    @property
    def bias(self):
        """Expose the underlying linear bias if present."""
        return getattr(self.linear, "bias", None)

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.linear(x)
        lora_out = x @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scale * lora_out


def inject_lora_layers(model: nn.Module, lora_config: LoRAConfig) -> int:
    """Inject trainable LoRA layers into targeted modules.

    Args:
        model: The WanModel to modify in-place.
        lora_config: LoRA configuration with targets and block range.

    Returns:
        Number of layers injected.
    """
    blocks = lora_config.blocks
    if blocks is None:
        from mlx_video.training.config import BlockRange

        blocks = BlockRange(start=0, end=40)

    block_indices = blocks.get_blocks()
    count = 0

    for block_idx in block_indices:
        if block_idx >= len(model.blocks):
            break

        block = model.blocks[block_idx]

        for target in lora_config.targets:
            # Navigate dot-separated path (e.g., "self_attn.q")
            parts = target.split(".")
            parent = block
            for part in parts[:-1]:
                parent = getattr(parent, part, None)
                if parent is None:
                    break

            if parent is None:
                continue

            attr_name = parts[-1]
            layer = getattr(parent, attr_name, None)
            if layer is None or not isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
                continue

            wrapped = TrainableLoRALinear(layer, lora_config.rank, lora_config.alpha)
            setattr(parent, attr_name, wrapped)
            count += 1

    return count


def freeze_base_weights(model: nn.Module) -> None:
    """Freeze all model parameters, then unfreeze only LoRA parameters."""
    model.freeze()
    _unfreeze_lora_params(model)


def _unfreeze_lora_params(module: nn.Module) -> None:
    """Recursively unfreeze lora_A and lora_B in all TrainableLoRALinear layers."""
    for name, child in module.named_modules():
        if isinstance(child, TrainableLoRALinear):
            child.unfreeze(keys=["lora_A", "lora_B"], strict=False)


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Count trainable and total parameters.

    Returns:
        (trainable_params, total_params)
    """
    trainable = 0
    total = 0

    def _count(prefix, module):
        nonlocal trainable, total
        if hasattr(module, "parameters"):
            for key, param in module.parameters().items():
                if isinstance(param, mx.array):
                    n = param.size
                    total += n
                    # Check if this param is frozen
                    frozen = getattr(module, "_frozen_keys", set())
                    if key not in frozen:
                        trainable += n

    # Simpler approach: count LoRA params directly
    trainable = 0
    total = 0
    for k, v in model.parameters().items():
        if isinstance(v, mx.array):
            total += v.size
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, mx.array):
                    total += vv.size
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, mx.array):
                    total += item.size
                elif isinstance(item, dict):
                    for kk, vv in item.items():
                        if isinstance(vv, mx.array):
                            total += vv.size

    # Count LoRA params by walking TrainableLoRALinear instances
    trainable = 0
    for _, child in model.named_modules():
        if isinstance(child, TrainableLoRALinear):
            trainable += child.lora_A.size + child.lora_B.size

    return trainable, total
