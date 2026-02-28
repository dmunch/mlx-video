"""Data structures for LoRA support."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx


@dataclass
class LoRAWeights:
    """Container for LoRA weight matrices.

    Attributes:
        lora_A: Low-rank matrix A of shape [rank, in_features]
        lora_B: Low-rank matrix B of shape [out_features, rank]
        rank: Rank of the LoRA decomposition
        alpha: LoRA scaling parameter (default: rank)
        module_name: Target module name in the model
    """

    lora_A: mx.array
    lora_B: mx.array
    rank: int
    alpha: float
    module_name: str

    @property
    def scale(self) -> float:
        """Compute the scale factor: alpha / rank."""
        return self.alpha / self.rank


@dataclass
class LoRAConfig:
    """Configuration for a single LoRA.

    Attributes:
        path: Path to the LoRA safetensors file
        strength: Strength/weight to apply this LoRA (typically 0.0-2.0)
        target_modules: Optional list of module names to apply LoRA to.
                       If None, applies to all available modules in the LoRA.
    """

    path: Path
    strength: float = 1.0
    target_modules: Optional[list[str]] = None

    def __post_init__(self):
        """Validate and normalize the configuration."""
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"LoRA file not found: {self.path}")
        if self.strength < 0:
            raise ValueError(f"LoRA strength must be non-negative, got {self.strength}")


@dataclass
class AppliedLoRA:
    """Represents a LoRA applied to a specific module.

    Attributes:
        weights: The LoRA weight matrices
        strength: Application strength for this LoRA
    """

    weights: LoRAWeights
    strength: float

    def compute_delta(self) -> mx.array:
        """Compute the weight delta: strength * scale * (lora_B @ lora_A)."""
        scale = self.weights.scale
        delta = self.weights.lora_B @ self.weights.lora_A
        return scale * self.strength * delta
