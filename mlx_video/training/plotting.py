"""Training loss visualization using matplotlib.

Renders a PNG plot showing individual loss values and a smoothed curve,
saved to the training output directory. Updated at each checkpoint and
at the end of training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LossHistory:
    """Accumulates per-step loss values during training."""

    steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    baseline: float | None = None

    def append(self, step: int, loss: float) -> None:
        self.steps.append(step)
        self.losses.append(loss)

    def __len__(self) -> int:
        return len(self.steps)


def _smooth(values: list[float], weight: float = 0.9) -> list[float]:
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0] if values else 0.0
    for v in values:
        last = weight * last + (1.0 - weight) * v
        smoothed.append(last)
    return smoothed


def plot_loss(
    history: LossHistory, output_path: str | Path, title: str = "Training Loss"
) -> None:
    """Render a training loss plot and save as PNG.

    Shows:
      - Individual loss values as semi-transparent dots
      - Smoothed loss curve (EMA, weight=0.9)
      - Legend with min, final, and mean loss values
      - Light grid for readability

    Args:
        history: Accumulated loss history.
        output_path: Path to save the PNG file.
        title: Plot title.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend (no display needed)
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "  ⚠ matplotlib not installed — skipping loss plot. "
            "Install with: pip install matplotlib"
        )
        return

    if len(history) < 2:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    steps = history.steps
    losses = history.losses
    smoothed = _smooth(losses, weight=0.9)

    min_loss = min(losses)
    min_step = steps[losses.index(min_loss)]
    final_loss = losses[-1]
    mean_loss = sum(losses) / len(losses)
    final_smooth = smoothed[-1]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Individual loss points
    ax.scatter(
        steps,
        losses,
        s=6,
        alpha=0.25,
        color="#5B9BD5",
        zorder=2,
        label=f"Loss (min={min_loss:.4f} @ step {min_step})",
    )

    # Smoothed curve
    ax.plot(
        steps,
        smoothed,
        color="#E04040",
        linewidth=1.8,
        zorder=3,
        label=f"Smoothed EMA (final={final_smooth:.4f})",
    )

    # Reference lines
    if history.baseline is not None:
        ax.axhline(
            y=history.baseline,
            color="#FFA500",
            linewidth=1.2,
            linestyle="--",
            alpha=0.7,
            zorder=1,
            label=f"Baseline={history.baseline:.4f}",
        )

    ax.axhline(
        y=mean_loss,
        color="#888888",
        linewidth=0.8,
        linestyle="--",
        alpha=0.5,
        zorder=1,
        label=f"Mean={mean_loss:.4f}",
    )

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Clean up axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
