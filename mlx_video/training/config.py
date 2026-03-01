"""Training config parser for Wan2.2 LoRA training."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DataSpec:
    """A single training image with its prompt."""

    image: Path
    prompt: str


@dataclass
class BaseLoRAEntry:
    """A base LoRA to merge into model weights before training."""

    path: str
    expert: str = "both"  # "both", "high", "low"
    strength: float = 1.0


@dataclass
class TrainingLoopConfig:
    num_epochs: int = 50
    batch_size: int = 1
    learning_rate: float = 1e-4
    optimizer: str = "AdamW"
    timestep_sampling: str = "balanced"  # balanced, low_bias, high_bias
    shift: Optional[float] = None  # override model default (e.g. 12.0)
    experts: str = "both"  # "both", "low", "high"
    expert_mode: str = "simultaneous"  # "simultaneous", "sequential"
    expert_routing: str = "alternating"  # "alternating", "proportional"
    switch_every: int = 1  # steps per expert before switching (alternating only)
    high_ratio: Optional[float] = None  # H expert probability (proportional only; None = 1-boundary)
    lr_schedule: str = "cosine"  # "cosine", "constant"
    lr_warmup_ratio: float = 0.1  # fraction of total steps for warmup (cosine only)
    loss_weighting: str = "min_snr"  # "uniform", "min_snr"
    min_snr_gamma: float = 5.0  # SNR clamping value for min-SNR weighting


@dataclass
class BlockRange:
    start: int = 0
    end: int = 40

    def get_blocks(self) -> list[int]:
        return list(range(self.start, self.end))


@dataclass
class LoRAConfig:
    rank: int = 32
    alpha: float = 32.0
    targets: list[str] = field(
        default_factory=lambda: [
            "self_attn.q",
            "self_attn.k",
            "self_attn.v",
            "self_attn.o",
            "cross_attn.q",
            "cross_attn.k",
            "cross_attn.v",
            "cross_attn.o",
            "ffn.fc1",
            "ffn.fc2",
        ]
    )
    blocks: Optional[BlockRange] = None


@dataclass
class CheckpointConfig:
    save_frequency: int = 25
    output_dir: str = "./training_output/"


@dataclass
class MonitoringConfig:
    log_frequency: int = 1
    plot_frequency: int = 10
    generate_image_frequency: int = 0  # 0 = disabled
    preview_width: int = 512
    preview_height: int = 512
    preview_steps: int = 20
    preview_guide_scale: float = 1.0


@dataclass
class TrainingConfig:
    """Top-level training configuration."""

    model_dir: str
    data: str
    seed: int = 42
    resolution: int = 512
    trigger_word: Optional[str] = None
    base_loras: list[BaseLoRAEntry] = field(default_factory=list)
    training: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Resolved after loading
    data_items: list[DataSpec] = field(default_factory=list, repr=False)
    preview_prompt: Optional[str] = field(default=None, repr=False)

    @staticmethod
    def from_json(path: str) -> "TrainingConfig":
        """Load config from JSON file, discover data, and validate."""
        config_path = Path(path).resolve()
        with open(config_path) as f:
            raw = json.load(f)

        # Parse nested configs
        training = TrainingLoopConfig(**raw.get("training", {}))

        lora_raw = raw.get("lora", {})
        blocks = None
        if "blocks" in lora_raw:
            blocks = BlockRange(**lora_raw.pop("blocks"))
        lora = LoRAConfig(**lora_raw, blocks=blocks)

        checkpoint = CheckpointConfig(**raw.get("checkpoint", {}))
        monitoring = MonitoringConfig(**raw.get("monitoring", {}))

        base_loras = [
            BaseLoRAEntry(**entry) for entry in raw.get("base_loras", [])
        ]

        config = TrainingConfig(
            model_dir=raw["model_dir"],
            data=raw["data"],
            seed=raw.get("seed", 42),
            resolution=raw.get("resolution", 512),
            trigger_word=raw.get("trigger_word"),
            base_loras=base_loras,
            training=training,
            lora=lora,
            checkpoint=checkpoint,
            monitoring=monitoring,
        )

        # Resolve data path relative to config file
        data_dir = Path(config.data)
        if not data_dir.is_absolute():
            data_dir = config_path.parent / data_dir
        data_dir = data_dir.resolve()

        config.data_items = _discover_data(data_dir, config.trigger_word)
        config.preview_prompt = _resolve_preview_prompt(data_dir, config)
        _validate(config, data_dir)

        return config


def _discover_data(
    data_dir: Path, trigger_word: Optional[str] = None
) -> list[DataSpec]:
    """Auto-discover image/prompt pairs in a directory.

    Each image must have a matching .txt file with the same stem.
    """
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"Data directory not found: {data_dir}")

    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = sorted(
        p
        for p in data_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in image_exts
        and not p.stem.startswith("preview")
    )

    if not image_files:
        raise ValueError(f"No image files found in data directory: {data_dir}")

    data_items = []
    for img_path in image_files:
        prompt_file = img_path.with_suffix(".txt")
        if not prompt_file.exists():
            raise ValueError(
                f"Missing prompt file for image '{img_path.name}'. "
                f"Expected '{prompt_file.name}' in {data_dir}"
            )
        prompt = prompt_file.read_text(encoding="utf-8").strip()
        if trigger_word and trigger_word not in prompt:
            prompt = f"{trigger_word} {prompt}"
        data_items.append(DataSpec(image=img_path.resolve(), prompt=prompt))

    return data_items


def _resolve_preview_prompt(data_dir: Path, config: "TrainingConfig") -> str:
    """Determine the prompt to use for preview image generation.

    Priority: preview.txt in data dir → first training sample prompt.
    """
    preview_file = data_dir / "preview.txt"
    if preview_file.exists():
        prompt = preview_file.read_text(encoding="utf-8").strip()
        if config.trigger_word and config.trigger_word not in prompt:
            prompt = f"{config.trigger_word} {prompt}"
        return prompt
    if config.data_items:
        return config.data_items[0].prompt
    return ""


def _validate(config: TrainingConfig, data_dir: Path) -> None:
    """Validate training config."""
    model_dir = Path(config.model_dir)
    if not model_dir.exists():
        raise ValueError(f"Model directory not found: {model_dir}")

    if config.resolution < 64:
        raise ValueError(f"Resolution must be >= 64, got {config.resolution}")
    if config.resolution % 32 != 0:
        raise ValueError(f"Resolution must be divisible by 32, got {config.resolution}")

    if not config.data_items:
        raise ValueError(f"No training data found in {data_dir}")

    if config.training.learning_rate <= 0:
        raise ValueError(
            f"Learning rate must be > 0, got {config.training.learning_rate}"
        )
    if config.training.num_epochs <= 0:
        raise ValueError(f"num_epochs must be > 0, got {config.training.num_epochs}")
    if config.training.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {config.training.batch_size}")

    valid_samplings = {"balanced", "low_bias", "high_bias"}
    if config.training.timestep_sampling not in valid_samplings:
        raise ValueError(
            f"timestep_sampling must be one of {valid_samplings}, "
            f"got '{config.training.timestep_sampling}'"
        )

    valid_experts = {"both", "low", "high"}
    if config.training.experts not in valid_experts:
        raise ValueError(
            f"experts must be one of {valid_experts}, "
            f"got '{config.training.experts}'"
        )

    valid_expert_modes = {"simultaneous", "sequential"}
    if config.training.expert_mode not in valid_expert_modes:
        raise ValueError(
            f"expert_mode must be one of {valid_expert_modes}, "
            f"got '{config.training.expert_mode}'"
        )

    valid_routings = {"alternating", "proportional"}
    if config.training.expert_routing not in valid_routings:
        raise ValueError(
            f"expert_routing must be one of {valid_routings}, "
            f"got '{config.training.expert_routing}'"
        )

    if config.training.switch_every < 1:
        raise ValueError(
            f"switch_every must be >= 1, got {config.training.switch_every}"
        )

    valid_lr_schedules = {"cosine", "constant"}
    if config.training.lr_schedule not in valid_lr_schedules:
        raise ValueError(
            f"lr_schedule must be one of {valid_lr_schedules}, "
            f"got '{config.training.lr_schedule}'"
        )
    if not (0.0 <= config.training.lr_warmup_ratio < 1.0):
        raise ValueError(
            f"lr_warmup_ratio must be in [0.0, 1.0), got {config.training.lr_warmup_ratio}"
        )

    valid_loss_weightings = {"uniform", "min_snr"}
    if config.training.loss_weighting not in valid_loss_weightings:
        raise ValueError(
            f"loss_weighting must be one of {valid_loss_weightings}, "
            f"got '{config.training.loss_weighting}'"
        )
    if config.training.min_snr_gamma <= 0:
        raise ValueError(
            f"min_snr_gamma must be > 0, got {config.training.min_snr_gamma}"
        )

    if config.training.high_ratio is not None:
        if not (0.0 < config.training.high_ratio < 1.0):
            raise ValueError(
                f"high_ratio must be between 0.0 and 1.0 (exclusive), "
                f"got {config.training.high_ratio}"
            )

    valid_base_lora_experts = {"both", "high", "low"}
    for i, bl in enumerate(config.base_loras):
        if bl.expert not in valid_base_lora_experts:
            raise ValueError(
                f"base_loras[{i}].expert must be one of {valid_base_lora_experts}, "
                f"got '{bl.expert}'"
            )
        bl_path = Path(bl.path)
        if not bl_path.is_absolute():
            bl_path = Path(config.model_dir).parent / bl_path
        if not bl_path.exists():
            raise ValueError(f"base_loras[{i}].path not found: {bl.path}")

    if config.lora.rank <= 0:
        raise ValueError(f"LoRA rank must be > 0, got {config.lora.rank}")
