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
    # Per-expert loss series (for dual-expert training)
    high_steps: list[int] = field(default_factory=list)
    high_losses: list[float] = field(default_factory=list)
    low_steps: list[int] = field(default_factory=list)
    low_losses: list[float] = field(default_factory=list)
    baseline: float | None = None

    def append(self, step: int, loss: float, expert: str | None = None) -> None:
        self.steps.append(step)
        self.losses.append(loss)
        if expert == "H":
            self.high_steps.append(step)
            self.high_losses.append(loss)
        elif expert == "L":
            self.low_steps.append(step)
            self.low_losses.append(loss)

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

    has_dual = len(history.high_losses) > 0 and len(history.low_losses) > 0

    if has_dual:
        # Separate H/L series
        if history.high_losses:
            h_smooth = _smooth(history.high_losses, weight=0.85)
            ax.scatter(
                history.high_steps,
                history.high_losses,
                s=14,
                alpha=0.5,
                color="#E04040",
                zorder=2,
                label=f"H expert (n={len(history.high_losses)}, last={history.high_losses[-1]:.4f})",
            )
            ax.plot(
                history.high_steps,
                h_smooth,
                color="#E04040",
                linewidth=1.5,
                alpha=0.8,
                zorder=3,
            )

        if history.low_losses:
            l_smooth = _smooth(history.low_losses, weight=0.85)
            ax.scatter(
                history.low_steps,
                history.low_losses,
                s=14,
                alpha=0.5,
                color="#5B9BD5",
                zorder=2,
                label=f"L expert (n={len(history.low_losses)}, last={history.low_losses[-1]:.4f})",
            )
            ax.plot(
                history.low_steps,
                l_smooth,
                color="#5B9BD5",
                linewidth=1.5,
                alpha=0.8,
                zorder=3,
            )

        # Combined EMA as reference
        ax.plot(
            steps,
            smoothed,
            color="#888888",
            linewidth=1.2,
            linestyle="--",
            zorder=3,
            label=f"Combined EMA (final={final_smooth:.4f})",
        )
    else:
        # Single series (non-dual training)
        ax.scatter(
            steps,
            losses,
            s=20,
            alpha=0.7,
            color="#5B9BD5",
            zorder=2,
            label=f"Epoch avg (min={min_loss:.4f} @ epoch {min_step})",
        )

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

    ax.set_xlabel("Epoch", fontsize=11)
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
