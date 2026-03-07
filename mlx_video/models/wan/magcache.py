"""MagCache: Magnitude-Aware Cache for accelerating Wan2.2 diffusion inference.

Uses pre-calibrated magnitude ratios of transformer residuals to decide which
denoising steps can be safely skipped, reusing cached residuals instead. More
stable than TeaCache's L2-distance approach for Wan2.2's wider feature
distribution (MoE expert fluctuations).

Reference: https://github.com/ali-vilab/MagCache
Paper: https://arxiv.org/abs/2506.09045
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


def _nearest_interp(src: np.ndarray, target_length: int) -> np.ndarray:
    """Nearest-neighbor interpolation for magnitude ratio arrays."""
    src_length = len(src)
    if target_length == 1:
        return np.array([src[-1]])
    scale = (src_length - 1) / (target_length - 1)
    indices = np.round(np.arange(target_length) * scale).astype(int)
    return src[indices]


def _convert_interleaved_ratios(interleaved: list[float]) -> np.ndarray:
    """Convert interleaved cond/uncond ratios to per-step averaged ratios.

    The reference implementation interleaves conditional and unconditional forward
    passes (cnt%2 tracking). mlx-video batches both into B=2, so we average each
    cond/uncond pair into a single per-step ratio.
    """
    arr = np.array(interleaved)
    cond = arr[0::2]
    uncond = arr[1::2]
    return (cond + uncond) / 2.0


# Pre-calibrated magnitude ratios from the reference implementation.
# Each list contains interleaved [cond, uncond, cond, uncond, ...] values.
# These are converted to per-step averages during initialization.

# T2V-A14B high-noise model (40 steps, 78 interleaved values → 39 per-step)
_RAW_RATIOS_T2V_14B_HIGH = [
    1.00124, 1.00155, 0.99822, 0.99851, 0.99696, 0.99687, 0.99703, 0.99732,
    0.9966, 0.99679, 0.99602, 0.99658, 0.99578, 0.99664, 0.99484, 0.9949,
    0.99633, 0.996, 0.99659, 0.99683, 0.99534, 0.99549, 0.99584, 0.99577,
    0.99681, 0.99694, 0.99563, 0.99554, 0.9944, 0.99473, 0.99594, 0.9964,
    0.99466, 0.99461, 0.99453, 0.99481, 0.99389, 0.99365, 0.99391, 0.99406,
    0.99354, 0.99361, 0.99283, 0.99278, 0.99268, 0.99263, 0.99057, 0.99091,
    0.99125, 0.99126, 0.65523, 0.65252, 0.98808, 0.98852, 0.98765, 0.98736,
    0.9851, 0.98535, 0.98311, 0.98339, 0.9805, 0.9806, 0.97776, 0.97771,
    0.97278, 0.97286, 0.96731, 0.96728, 0.95857, 0.95855, 0.94385, 0.94385,
    0.92118, 0.921, 0.88108, 0.88076, 0.80263, 0.80181,
]

# I2V-A14B high-noise model (40 steps, 78 interleaved values → 39 per-step)
_RAW_RATIOS_I2V_14B_HIGH = [
    0.99191, 0.99144, 0.99356, 0.99337, 0.99326, 0.99285, 0.99251, 0.99264,
    0.99393, 0.99366, 0.9943, 0.9943, 0.99276, 0.99288, 0.99389, 0.99393,
    0.99274, 0.99289, 0.99316, 0.9931, 0.99379, 0.99377, 0.99268, 0.99271,
    0.99222, 0.99227, 0.99175, 0.9916, 0.91076, 0.91046, 0.98931, 0.98933,
    0.99087, 0.99088, 0.98852, 0.98855, 0.98895, 0.98896, 0.98806, 0.98808,
    0.9871, 0.98711, 0.98613, 0.98618, 0.98434, 0.98435, 0.983, 0.98307,
    0.98185, 0.98187, 0.98131, 0.98131, 0.9783, 0.97835, 0.97619, 0.9762,
    0.97264, 0.9727, 0.97088, 0.97098, 0.96568, 0.9658, 0.96045, 0.96055,
    0.95322, 0.95335, 0.94579, 0.94594, 0.93297, 0.93311, 0.91699, 0.9172,
    0.89174, 0.89202, 0.8541, 0.85446, 0.79823, 0.79902,
]

# TI2V-5B single model, text-to-video mode (50 steps, 98 interleaved → 49 per-step)
_RAW_RATIOS_TI2V_5B_T2V = [
    0.99505, 0.99389, 0.99441, 0.9957, 0.99558, 0.99551, 0.99499, 0.9945,
    0.99534, 0.99548, 0.99468, 0.9946, 0.99463, 0.99458, 0.9946, 0.99453,
    0.99408, 0.99404, 0.9945, 0.99441, 0.99409, 0.99398, 0.99403, 0.99397,
    0.99382, 0.99377, 0.99349, 0.99343, 0.99377, 0.99378, 0.9933, 0.99328,
    0.99303, 0.99301, 0.99217, 0.99216, 0.992, 0.99201, 0.99201, 0.99202,
    0.99133, 0.99132, 0.99112, 0.9911, 0.99155, 0.99155, 0.98958, 0.98957,
    0.98959, 0.98958, 0.98838, 0.98835, 0.98826, 0.98825, 0.9883, 0.98828,
    0.98711, 0.98709, 0.98562, 0.98561, 0.98511, 0.9851, 0.98414, 0.98412,
    0.98284, 0.98282, 0.98104, 0.98101, 0.97981, 0.97979, 0.97849, 0.97849,
    0.97557, 0.97554, 0.97398, 0.97395, 0.97171, 0.97166, 0.96917, 0.96913,
    0.96511, 0.96507, 0.96263, 0.96257, 0.95839, 0.95835, 0.95483, 0.95475,
    0.94942, 0.94936, 0.9468, 0.94678, 0.94583, 0.94594, 0.94843, 0.94872,
    0.96949, 0.97015,
]

# TI2V-5B single model, image-to-video mode (50 steps, 98 interleaved → 49 per-step)
_RAW_RATIOS_TI2V_5B_I2V = [
    0.99512, 0.99559, 0.99559, 0.99561, 0.99595, 0.99577, 0.99512, 0.99512,
    0.99546, 0.99534, 0.99543, 0.99531, 0.99496, 0.99491, 0.99504, 0.99499,
    0.99444, 0.99449, 0.99481, 0.99481, 0.99435, 0.99435, 0.9943, 0.99431,
    0.99411, 0.99406, 0.99373, 0.99376, 0.99413, 0.99405, 0.99363, 0.99359,
    0.99335, 0.99331, 0.99244, 0.99243, 0.99229, 0.99229, 0.99239, 0.99236,
    0.99163, 0.9916, 0.99149, 0.99151, 0.99191, 0.99192, 0.9898, 0.98981,
    0.9899, 0.98987, 0.98849, 0.98849, 0.98846, 0.98846, 0.98861, 0.98861,
    0.9874, 0.98738, 0.98588, 0.98589, 0.98539, 0.98534, 0.98444, 0.98439,
    0.9831, 0.98309, 0.98119, 0.98118, 0.98001, 0.98, 0.97862, 0.97859,
    0.97555, 0.97558, 0.97392, 0.97388, 0.97152, 0.97145, 0.96871, 0.9687,
    0.96435, 0.96434, 0.96129, 0.96127, 0.95639, 0.95638, 0.95176, 0.95175,
    0.94446, 0.94452, 0.93972, 0.93974, 0.93575, 0.9359, 0.93537, 0.93552,
    0.96655, 0.96616,
]

# Registry: (model_key → raw interleaved ratios, calibration_steps)
_RATIO_REGISTRY: dict[str, tuple[list[float], int]] = {
    "t2v_14b_high": (_RAW_RATIOS_T2V_14B_HIGH, 40),
    "i2v_14b_high": (_RAW_RATIOS_I2V_14B_HIGH, 40),
    "ti2v_5b_t2v": (_RAW_RATIOS_TI2V_5B_T2V, 50),
    "ti2v_5b_i2v": (_RAW_RATIOS_TI2V_5B_I2V, 50),
}


def get_magcache_ratios(key: str, num_steps: int) -> np.ndarray:
    """Get per-step magnitude ratios for the given model variant and step count.

    Converts interleaved cond/uncond ratios to per-step averages, prepends
    a 1.0 padding entry (first step has no previous residual), and interpolates
    to match the requested step count.

    Args:
        key: Model variant key (e.g., "t2v_14b_high")
        num_steps: Number of denoising steps

    Returns:
        Array of magnitude ratios with shape (num_steps,), where index 0 = 1.0 (padding)
    """
    if key not in _RATIO_REGISTRY:
        raise ValueError(
            f"No pre-calibrated MagCache ratios for key '{key}'. "
            f"Available: {list(_RATIO_REGISTRY.keys())}"
        )

    raw_ratios, calibration_steps = _RATIO_REGISTRY[key]
    per_step = _convert_interleaved_ratios(raw_ratios)

    # Prepend padding (first step always computes, no previous ratio)
    ratios = np.concatenate([[1.0], per_step])  # length = calibration_steps

    if len(ratios) != calibration_steps:
        ratios = _nearest_interp(ratios, calibration_steps)

    # Interpolate to target step count if different from calibration
    if num_steps != calibration_steps:
        ratios = _nearest_interp(ratios, num_steps)

    return ratios


def load_ratios_from_file(path: str, num_steps: int, model_key: str) -> np.ndarray:
    """Load magnitude ratios from a calibration JSON file.

    Args:
        path: Path to the calibration JSON file
        num_steps: Number of denoising steps for this model
        model_key: Key to look up in the JSON ("high_noise", "low_noise", or "model")

    Returns:
        Array of magnitude ratios with shape (num_steps,)
    """
    with open(path) as f:
        data = json.load(f)

    if model_key not in data:
        available = [k for k in data if isinstance(data[k], dict) and "ratios" in data[k]]
        raise ValueError(
            f"Key '{model_key}' not found in {path}. "
            f"Available model keys: {available}"
        )

    entry = data[model_key]
    ratios = np.array([1.0] + entry["ratios"])  # prepend padding
    calibration_steps = entry["steps"]

    if len(ratios) != calibration_steps:
        ratios = _nearest_interp(ratios, calibration_steps)

    if num_steps != calibration_steps:
        ratios = _nearest_interp(ratios, num_steps)

    return ratios


def save_calibration(
    path: str,
    model_type: str,
    model_version: str,
    steps: int,
    results: dict[str, dict],
) -> None:
    """Save calibration results to a JSON file.

    Args:
        path: Output file path
        model_type: Model type (e.g., "t2v", "i2v", "ti2v")
        model_version: Model version (e.g., "2.2")
        steps: Total denoising steps
        results: Dict mapping model keys ("high_noise", "low_noise", "model")
                 to calibration result dicts from MagCacheState.get_calibration_result()
    """
    data = {
        "model_type": model_type,
        "model_version": model_version,
        "steps": steps,
        **results,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  MagCache calibration saved to {path}")


@dataclass
class MagCacheState:
    """Configuration and runtime state for MagCache acceleration.

    Tracks the accumulated magnitude error and cached residuals to decide
    whether to skip transformer block computation at each denoising step.

    Each model instance (e.g., high-noise and low-noise in Wan2.2 dual mode)
    should have its own MagCacheState.
    """

    enabled: bool = False

    # Hyperparameters
    threshold: float = 0.06       # Upper bound for accumulated error
    K: int = 2                    # Max consecutive skip steps
    retention_ratio: float = 0.2  # Fraction of early steps that always compute
    gatekeeper_low: float = 0.95  # Hybrid mode: veto Spectrum if ratio below this
    gatekeeper_high: float = 1.05  # Hybrid mode: veto Spectrum if ratio above this

    # Calibration mode
    calibrating: bool = False
    calibration_ratios: list = field(default_factory=list)
    prev_residual: object = None  # mx.array | None

    # Runtime state
    num_steps: int = 0
    cnt: int = 0
    accumulated_err: float = 0.0
    accumulated_ratio: float = 1.0
    accumulated_steps: int = 0
    residual_cache: object = None  # mx.array | None
    mag_ratios: object = field(default=None, repr=False)  # np.ndarray | None
    retention_steps: int = 0  # Computed from retention_ratio * num_steps

    verbose: bool = False

    # Stats
    steps_skipped: int = 0
    steps_computed: int = 0
    steps_vetoed: int = 0  # Hybrid mode: Spectrum skips vetoed by MagCache

    def configure(self, num_steps: int, ratios_key: str) -> None:
        """Set up MagCache for a generation run.

        Args:
            num_steps: Total denoising steps for this model
            ratios_key: Key for pre-calibrated ratio lookup
        """
        self.num_steps = num_steps
        self.retention_steps = max(2, int(num_steps * self.retention_ratio))
        self.mag_ratios = get_magcache_ratios(ratios_key, num_steps)
        self.reset()

    def configure_from_file(self, num_steps: int, path: str, model_key: str) -> None:
        """Set up MagCache using ratios loaded from a calibration JSON file."""
        self.num_steps = num_steps
        self.retention_steps = max(2, int(num_steps * self.retention_ratio))
        self.mag_ratios = load_ratios_from_file(path, num_steps, model_key)
        self.reset()

    def configure_calibration(self, num_steps: int) -> None:
        """Set up MagCache in calibration mode (full compute, record ratios)."""
        self.calibrating = True
        self.num_steps = num_steps
        self.calibration_ratios = []
        self.prev_residual = None
        self.cnt = 0

    def record_ratio(self, residual) -> None:
        """Record the magnitude ratio for the current step during calibration.

        Computes the mean norm ratio between the current residual and the
        previous step's residual. The first step just stores the residual.

        Args:
            residual: Transformer residual (x_output - x_input), mx.array
        """
        import mlx.core as mx

        if self.prev_residual is not None:
            prev_norm = mx.linalg.norm(self.prev_residual, axis=-1)
            cur_norm = mx.linalg.norm(residual, axis=-1)
            # Avoid division by zero
            safe_prev = mx.maximum(prev_norm, mx.array(1e-8))
            ratio = mx.mean(cur_norm / safe_prev).item()
            self.calibration_ratios.append(round(ratio, 5))
            if self.verbose:
                print(f"    [MagCache calibrate] step {self.cnt}: ratio={ratio:.5f}")
        else:
            if self.verbose:
                print(f"    [MagCache calibrate] step {self.cnt}: first step (no ratio)")
        self.prev_residual = residual

    def get_calibration_result(self) -> dict:
        """Return calibration results as a dict suitable for JSON serialization."""
        return {
            "steps": self.num_steps,
            "ratios": self.calibration_ratios,
        }

    def get_ratio(self, step: int) -> float:
        """Look up the pre-calibrated magnitude ratio for the given step."""
        if self.mag_ratios is None:
            return 1.0
        idx = min(step, len(self.mag_ratios) - 1)
        return float(self.mag_ratios[idx])

    def should_skip(self) -> bool:
        """Decide whether to skip computation at the current step.

        Returns True if accumulated error is below threshold and we haven't
        exceeded the maximum consecutive skip count. Returns False during
        the retention period (early steps always compute).
        """
        if self.cnt < self.retention_steps:
            return False
        if self.residual_cache is None:
            return False

        ratio = self.get_ratio(self.cnt)
        test_accumulated_ratio = self.accumulated_ratio * ratio
        test_skip_err = abs(1.0 - test_accumulated_ratio)
        test_accumulated_err = self.accumulated_err + test_skip_err
        test_accumulated_steps = self.accumulated_steps + 1

        if test_accumulated_err < self.threshold and test_accumulated_steps <= self.K:
            # Accept: update accumulators
            self.accumulated_ratio = test_accumulated_ratio
            self.accumulated_err = test_accumulated_err
            self.accumulated_steps = test_accumulated_steps
            return True
        else:
            # Reject: reset accumulators
            self.accumulated_err = 0.0
            self.accumulated_ratio = 1.0
            self.accumulated_steps = 0
            return False

    def should_veto_spectrum(self) -> bool:
        """Check if MagCache should veto a Spectrum skip (hybrid mode).

        Returns True if the magnitude ratio at the current step deviates
        significantly from 1.0, indicating the model is making substantial
        changes that Spectrum's prediction might miss.
        """
        ratio = self.get_ratio(self.cnt)
        return ratio < self.gatekeeper_low or ratio > self.gatekeeper_high

    def advance(self) -> None:
        """Advance the step counter. Call after each forward pass."""
        self.cnt += 1
        if self.cnt >= self.num_steps:
            self.cnt = 0
            self.accumulated_err = 0.0
            self.accumulated_ratio = 1.0
            self.accumulated_steps = 0

    def reset(self) -> None:
        """Reset all runtime state for a new generation."""
        self.cnt = 0
        self.accumulated_err = 0.0
        self.accumulated_ratio = 1.0
        self.accumulated_steps = 0
        self.residual_cache = None
        self.steps_skipped = 0
        self.steps_computed = 0
        self.steps_vetoed = 0
        self.calibration_ratios = []
        self.prev_residual = None
