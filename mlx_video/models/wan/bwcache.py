"""Block-Wise Caching (BWCache) for Wan2.2 DiT acceleration.

Implements the BWCache algorithm from "BWCache: Accelerating Video Diffusion
Transformers through Block-Wise Caching" (arXiv:2509.13789v3).

Two operating modes:
  - "step": Paper-faithful step-level caching. Aggregated block L1 similarity
    triggers a cal_list schedule that skips entire denoising steps.
  - "block": Enhanced per-block caching. Always computes boundary blocks
    (first/last N), skips middle blocks whose compact L1 fingerprint is
    below threshold (identity skip — zero additional memory).

Block mode is designed to layer on top of step-level methods (MagCache,
Spectrum) — it only activates within steps those methods decide to compute.

Memory design: Block mode uses spatially-pooled fingerprints ([B, dim]) for
L1 comparison instead of full hidden states, and identity-skip instead of
cached per-block residuals. This keeps memory overhead under ~2 MB regardless
of resolution/frame count.
"""

import math
from dataclasses import dataclass, field

import mlx.core as mx


@dataclass
class BWCacheState:
    """Tracks BWCache state for block-wise caching acceleration.

    The core idea: across diffusion timesteps, DiT block features exhibit a
    U-shaped variation pattern — high change at the start/end, low change in
    the middle. BWCache exploits this by skipping blocks when their input
    features barely changed from the previous step.

    Block mode uses *identity skip*: skipped blocks contribute nothing (the
    hidden state passes through unchanged). This is valid because low L1
    means the block's input barely changed, so its output delta is also small.
    This avoids storing per-block residuals (~1.5 GB each at high resolution).
    """

    enabled: bool = False
    mode: str = "step"  # "step" (paper-faithful) or "block" (enhanced)

    # --- Core parameters ---
    thresh: float = 0.15  # Similarity threshold δ (step mode default)
    reuse_interval: int = 3  # Periodic recomputation interval R
    last_step_ratio: float = 0.5  # Fraction of tail steps to always compute
    boundary_blocks: int = 6  # First/last N blocks always computed (block mode)
    verbose: bool = False

    # --- Runtime state ---
    cnt: int = 0
    num_steps: int = 0
    num_blocks: int = 0  # Total transformer blocks (set during init)

    # Step-level state
    cal_list: list = field(default_factory=list)
    cal_list_triggered: bool = False
    previous_residual: object = None  # mx.array — step-level residual cache

    # Block-level state: compact fingerprints for L1 comparison
    # Each entry is (fingerprint [B, dim], abs_mean scalar) or None
    block_fingerprints: list = field(default_factory=list)

    # Auto-calibration state
    auto_thresh: bool = True  # Enable auto-threshold calibration (block mode)
    calibration_warmup: int = 3  # Computed steps before calibration fires
    target_skip_ratio: float = 0.5  # Target fraction of middle blocks to skip
    calibration_steps_seen: int = 0  # Computed steps during warmup

    # --- Stats & diagnostics ---
    steps_skipped: int = 0
    steps_computed: int = 0
    blocks_skipped: int = 0
    blocks_computed: int = 0
    l1_history: list = field(default_factory=list)  # All non-sentinel L1 values

    def reset(self):
        """Reset all runtime state for a new generation run."""
        self.cnt = 0
        self.cal_list = []
        self.cal_list_triggered = False
        self.previous_residual = None
        self.block_fingerprints = [None] * self.num_blocks
        self.calibration_steps_seen = 0
        self.steps_skipped = 0
        self.steps_computed = 0
        self.blocks_skipped = 0
        self.blocks_computed = 0
        self.l1_history = []

    # ------------------------------------------------------------------ #
    #  Step-level mode (paper-faithful)
    # ------------------------------------------------------------------ #

    def should_skip_step(self, step_idx: int) -> bool:
        """Check if this step should be skipped (step mode).

        Returns True if cal_list has been triggered and marks this step as skip (0).
        """
        if not self.cal_list_triggered:
            return False
        if step_idx < 0 or step_idx >= len(self.cal_list):
            return False
        return self.cal_list[step_idx] == 0

    def update_schedule(self, acu_l1: float, depth: int, cur_idx: int):
        """Build the cal_list schedule when the similarity indicator fires.

        Called after each full-compute step in step mode. If the mean block L1
        is below threshold and the schedule hasn't been triggered yet, constructs
        a compute/reuse pattern for all remaining steps.

        Args:
            acu_l1: Accumulated relative L1 across all blocks for this step.
            depth: Number of blocks (used to compute mean L1).
            cur_idx: Current step index (0-based).
        """
        if self.cal_list_triggered:
            return
        if depth == 0:
            return

        mean_l1 = acu_l1 / depth
        if mean_l1 >= self.thresh:
            return

        # Trigger: build the schedule
        self.cal_list_triggered = True
        num_steps = self.num_steps

        # Start with all-compute
        self.cal_list = [1] * num_steps

        # Pattern: reuse_interval zeros followed by one 1 (compute)
        # Applied starting from the step after cur_idx
        pattern_len = self.reuse_interval + 1
        pattern = [0] * self.reuse_interval + [1]

        for i in range(cur_idx + 1, num_steps):
            self.cal_list[i] = pattern[(i - (cur_idx + 1)) % pattern_len]

        # Tail protection: always compute the last fraction of steps
        tail_start = int(cur_idx * self.last_step_ratio)
        if tail_start > 0:
            for i in range(num_steps - tail_start, num_steps):
                self.cal_list[i] = 1

        if self.verbose:
            skip_count = self.cal_list.count(0)
            print(
                f"    [BWCache] Schedule triggered at step {cur_idx}: "
                f"mean_l1={mean_l1:.4f} < thresh={self.thresh}, "
                f"{skip_count}/{num_steps} steps will be skipped"
            )

    # ------------------------------------------------------------------ #
    #  Block-level mode (enhanced, memory-efficient)
    # ------------------------------------------------------------------ #

    def compute_block_l1(self, block_idx: int, x_mod: mx.array) -> float:
        """Compute relative L1 distance using compact spatial-mean fingerprints.

        Instead of storing the full [B, seq_len, dim] tensor (which is ~1.5 GB
        per block at high resolution), we pool to [B, dim] via spatial mean.
        The L1 comparison on pooled features is a good proxy for full-tensor L1.

        Args:
            block_idx: Index of the current block (0-based).
            x_mod: Current modulated input [B, seq_len, dim] or pre-pooled [B, dim].

        Returns:
            Relative L1 distance (float). Returns 1000.0 sentinel on first call.
        """
        # Pool to compact fingerprint if not already pooled
        fingerprint = x_mod.mean(axis=-2) if x_mod.ndim == 3 else x_mod
        prev_tuple = self.block_fingerprints[block_idx]

        if prev_tuple is None:
            # First call: store fingerprint and pre-computed denominator
            abs_mean = mx.abs(fingerprint).mean()
            self.block_fingerprints[block_idx] = (fingerprint, abs_mean)
            return 1000.0

        prev, prev_abs_mean = prev_tuple

        # Relative L1 with cached denominator (avoids recomputing |prev|)
        l1 = (mx.abs(fingerprint - prev).mean() / prev_abs_mean).item()

        abs_mean = mx.abs(fingerprint).mean()
        self.block_fingerprints[block_idx] = (fingerprint, abs_mean)
        self.l1_history.append(l1)
        return l1

    def compute_block_l1_lazy(self, block_idx: int, x_mod: mx.array) -> mx.array:
        """Like compute_block_l1 but returns an mx.array (no .item() sync).

        Used in step mode to accumulate L1 values lazily across all blocks,
        deferring GPU sync to a single .item() call after all blocks complete.

        Returns:
            mx.array scalar. Returns mx.array(1000.0) sentinel on first call.
        """
        fingerprint = x_mod.mean(axis=-2) if x_mod.ndim == 3 else x_mod
        prev_tuple = self.block_fingerprints[block_idx]

        if prev_tuple is None:
            abs_mean = mx.abs(fingerprint).mean()
            self.block_fingerprints[block_idx] = (fingerprint, abs_mean)
            return mx.array(1000.0)

        prev, prev_abs_mean = prev_tuple
        l1 = mx.abs(fingerprint - prev).mean() / prev_abs_mean

        abs_mean = mx.abs(fingerprint).mean()
        self.block_fingerprints[block_idx] = (fingerprint, abs_mean)
        return l1

    def is_boundary_block(self, block_idx: int) -> bool:
        """Check if a block is in the always-compute boundary region."""
        return (
            block_idx < self.boundary_blocks
            or block_idx >= self.num_blocks - self.boundary_blocks
        )

    def should_skip_block(self, block_idx: int, l1: float) -> bool:
        """Decide whether to skip a block in block mode (identity skip).

        A block can be skipped if:
        1. It's not a boundary block
        2. It has been seen before (fingerprint exists from previous step)
        3. Its L1 similarity is below threshold

        When skipped, the block is not executed and x passes through unchanged.
        """
        if self.is_boundary_block(block_idx):
            return False
        if self.block_fingerprints[block_idx] is None:
            return False
        return l1 < self.thresh

    # ------------------------------------------------------------------ #
    #  Auto-calibration
    # ------------------------------------------------------------------ #

    def is_calibrating(self) -> bool:
        """True during warmup period when collecting L1 data (no skipping)."""
        if not self.auto_thresh or self.mode != "block":
            return False
        return self.calibration_steps_seen < self.calibration_warmup

    def finish_calibration(self):
        """Set threshold from observed L1 distribution at target skip percentile.

        Called after warmup completes and then every subsequent step for
        continuous recalibration. Uses l1_history to find the threshold
        that would skip `target_skip_ratio` fraction of middle blocks.
        """
        if not self.l1_history:
            return

        sorted_l1 = sorted(self.l1_history)
        # target_skip_ratio = 0.5 → set threshold at the 50th percentile
        # so that 50% of L1 values fall below it → 50% would be skipped
        idx = int(len(sorted_l1) * self.target_skip_ratio)
        idx = min(idx, len(sorted_l1) - 1)
        old_thresh = self.thresh
        self.thresh = sorted_l1[idx]

        if self.verbose and abs(self.thresh - old_thresh) > 1e-4:
            pcts = self.l1_percentiles()
            print(
                f"    [BWCache] Auto-calibrated: thresh {old_thresh:.4f} → {self.thresh:.4f} "
                f"(target {self.target_skip_ratio:.0%} skip, "
                f"L1 p50={pcts.get('p50', 0):.4f}, p75={pcts.get('p75', 0):.4f}, "
                f"from {len(self.l1_history)} samples)"
            )

    # ------------------------------------------------------------------ #
    #  Step-level residual helpers
    # ------------------------------------------------------------------ #

    def cache_step_residual(self, residual: mx.array):
        """Store the step-level residual (output - input across all blocks)."""
        self.previous_residual = residual

    def get_step_residual(self) -> mx.array | None:
        """Retrieve the cached step-level residual."""
        return self.previous_residual

    # ------------------------------------------------------------------ #
    #  Diagnostics
    # ------------------------------------------------------------------ #

    def l1_percentiles(self) -> dict:
        """Return L1 distribution percentiles from collected history."""
        if not self.l1_history:
            return {}
        s = sorted(self.l1_history)
        n = len(s)

        def pct(p):
            idx = min(int(n * p / 100), n - 1)
            return s[idx]

        return {
            "p10": pct(10),
            "p25": pct(25),
            "p50": pct(50),
            "p75": pct(75),
            "p90": pct(90),
            "mean": sum(s) / n,
            "count": n,
        }

    def summary(self) -> str:
        """Return a summary string of caching statistics."""
        total_steps = self.steps_computed + self.steps_skipped
        total_blocks = self.blocks_computed + self.blocks_skipped
        parts = [f"[BWCache {self.mode}]"]
        if total_steps > 0:
            parts.append(
                f"steps: {self.steps_computed} computed, "
                f"{self.steps_skipped} skipped ({self.steps_skipped/total_steps*100:.0f}%)"
            )
        if total_blocks > 0:
            parts.append(
                f"blocks: {self.blocks_computed} computed, "
                f"{self.blocks_skipped} skipped ({self.blocks_skipped/total_blocks*100:.0f}%)"
            )
        return " | ".join(parts)
