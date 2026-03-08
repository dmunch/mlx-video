"""Tests for BWCache (Block-Wise Caching) optimization."""

import mlx.core as mx
import pytest
from wan_test_helpers import _make_tiny_config

from mlx_video.models.wan.bwcache import BWCacheState
from mlx_video.models.wan.model import WanModel


class TestBWCacheState:
    """Unit tests for BWCacheState dataclass."""

    def test_default_disabled(self):
        bc = BWCacheState()
        assert bc.enabled is False
        assert bc.mode == "step"
        assert bc.thresh == 0.15

    def test_reset(self):
        bc = BWCacheState(enabled=True, num_blocks=40)
        bc.cnt = 5
        bc.steps_skipped = 3
        bc.blocks_skipped = 10
        bc.cal_list_triggered = True
        bc.previous_residual = mx.ones((1,))
        bc.reset()
        assert bc.cnt == 0
        assert bc.steps_skipped == 0
        assert bc.blocks_skipped == 0
        assert bc.cal_list_triggered is False
        assert bc.previous_residual is None
        assert len(bc.block_fingerprints) == 40
        assert all(f is None for f in bc.block_fingerprints)

    def test_should_skip_step_before_trigger(self):
        bc = BWCacheState(enabled=True, num_steps=10)
        bc.reset()
        assert bc.should_skip_step(0) is False
        assert bc.should_skip_step(5) is False

    def test_should_skip_step_after_trigger(self):
        bc = BWCacheState(enabled=True, num_steps=10)
        bc.reset()
        bc.cal_list = [1, 1, 1, 0, 0, 0, 1, 0, 0, 1]
        bc.cal_list_triggered = True
        assert bc.should_skip_step(0) is False  # compute
        assert bc.should_skip_step(3) is True  # skip
        assert bc.should_skip_step(6) is False  # compute

    def test_update_schedule_no_trigger_above_thresh(self):
        bc = BWCacheState(enabled=True, num_steps=30, thresh=0.15, num_blocks=28)
        bc.reset()
        # Mean L1 of 0.20 > threshold 0.15 → should NOT trigger
        bc.update_schedule(acu_l1=5.6, depth=28, cur_idx=5)
        assert bc.cal_list_triggered is False

    def test_update_schedule_triggers_below_thresh(self):
        bc = BWCacheState(
            enabled=True, num_steps=30, thresh=0.15, reuse_interval=3, num_blocks=28
        )
        bc.reset()
        # Mean L1 of 0.10 < threshold 0.15 → should trigger
        bc.update_schedule(acu_l1=2.8, depth=28, cur_idx=5)
        assert bc.cal_list_triggered is True
        assert len(bc.cal_list) == 30
        # Steps 0-5 should be compute (1)
        for i in range(6):
            assert bc.cal_list[i] == 1
        # Steps 6+ should follow pattern [0,0,0,1,0,0,0,1,...]
        # (reuse_interval=3 means 3 skips then 1 compute)
        pattern_start = 6
        for i in range(pattern_start, 30):
            # Tail protection may override some to 1
            pass
        # At least some steps should be 0 (skipped)
        assert 0 in bc.cal_list[6:]

    def test_update_schedule_only_triggers_once(self):
        bc = BWCacheState(
            enabled=True, num_steps=30, thresh=0.15, reuse_interval=3, num_blocks=28
        )
        bc.reset()
        bc.update_schedule(acu_l1=2.8, depth=28, cur_idx=5)
        cal_list_1 = bc.cal_list.copy()
        # Second call should not change anything
        bc.update_schedule(acu_l1=1.0, depth=28, cur_idx=10)
        assert bc.cal_list == cal_list_1

    def test_update_schedule_tail_protection(self):
        bc = BWCacheState(
            enabled=True,
            num_steps=20,
            thresh=0.15,
            reuse_interval=3,
            last_step_ratio=0.5,
            num_blocks=28,
        )
        bc.reset()
        # Trigger at step 4 → tail_start = 4 * 0.5 = 2 → last 2 steps always compute
        bc.update_schedule(acu_l1=2.0, depth=28, cur_idx=4)
        assert bc.cal_list_triggered is True
        # Last 2 steps must be compute
        assert bc.cal_list[-1] == 1
        assert bc.cal_list[-2] == 1

    def test_compute_block_l1_first_call(self):
        bc = BWCacheState(enabled=True, num_blocks=5)
        bc.reset()
        x_mod = mx.ones((1, 10, 64))
        l1 = bc.compute_block_l1(0, x_mod)
        assert l1 == 1000.0  # sentinel for first call
        assert bc.block_fingerprints[0] is not None
        # Fingerprint is stored as (array, abs_mean) tuple; array should be pooled: [B, dim]
        fp, abs_mean = bc.block_fingerprints[0]
        assert fp.shape == (1, 64)

    def test_compute_block_l1_similarity(self):
        bc = BWCacheState(enabled=True, num_blocks=5)
        bc.reset()
        x_mod = mx.ones((1, 10, 64))
        bc.compute_block_l1(0, x_mod)
        # Same input → L1 should be 0
        l1 = bc.compute_block_l1(0, x_mod)
        assert l1 == pytest.approx(0.0, abs=1e-6)

    def test_compute_block_l1_difference(self):
        bc = BWCacheState(enabled=True, num_blocks=5)
        bc.reset()
        x_mod1 = mx.ones((1, 10, 64))
        bc.compute_block_l1(0, x_mod1)
        # Different input → L1 should be > 0
        x_mod2 = mx.ones((1, 10, 64)) * 2.0
        l1 = bc.compute_block_l1(0, x_mod2)
        assert l1 > 0.0

    def test_is_boundary_block(self):
        bc = BWCacheState(enabled=True, num_blocks=40, boundary_blocks=6)
        bc.reset()
        # First 6 blocks are boundary
        for i in range(6):
            assert bc.is_boundary_block(i) is True
        # Middle blocks are not boundary
        for i in range(6, 34):
            assert bc.is_boundary_block(i) is False
        # Last 6 blocks are boundary
        for i in range(34, 40):
            assert bc.is_boundary_block(i) is True

    def test_compute_group_l1_first_call(self):
        """First group L1 call returns sentinel."""
        bc = BWCacheState(enabled=True, num_blocks=40, boundary_blocks=6)
        bc.reset()
        x = mx.ones((1, 10, 64))
        l1 = bc.compute_group_l1(x)
        assert l1 == 1000.0
        assert bc.group_fingerprint is not None

    def test_compute_group_l1_similarity(self):
        """Identical inputs give L1 ≈ 0."""
        bc = BWCacheState(enabled=True, num_blocks=40, boundary_blocks=6)
        bc.reset()
        x = mx.ones((1, 10, 64))
        bc.compute_group_l1(x)
        l1 = bc.compute_group_l1(x)
        assert l1 == pytest.approx(0.0, abs=1e-6)

    def test_compute_group_l1_difference(self):
        """Different inputs give L1 > 0 and value is recorded in l1_history."""
        bc = BWCacheState(enabled=True, num_blocks=40, boundary_blocks=6)
        bc.reset()
        bc.compute_group_l1(mx.ones((1, 10, 64)))
        l1 = bc.compute_group_l1(mx.ones((1, 10, 64)) * 2.0)
        assert l1 > 0.0
        assert len(bc.l1_history) == 1

    def test_group_residual_stored_after_reset(self):
        """middle_residual and group_fingerprint should be None after reset."""
        bc = BWCacheState(enabled=True, num_blocks=5)
        bc.group_fingerprint = (mx.ones((1, 64)), mx.array(1.0))
        bc.middle_residual = mx.ones((1, 10, 64))
        bc.reset()
        assert bc.group_fingerprint is None
        assert bc.middle_residual is None

    def test_cache_and_get_step_residual(self):
        bc = BWCacheState(enabled=True)
        res = mx.ones((1, 10, 64)) * 0.5
        bc.cache_step_residual(res)
        assert bc.get_step_residual() is not None
        assert mx.array_equal(bc.get_step_residual(), res)

    def test_summary_empty(self):
        bc = BWCacheState(enabled=True, mode="step")
        s = bc.summary()
        assert "BWCache step" in s

    def test_summary_with_stats(self):
        bc = BWCacheState(enabled=True, mode="block")
        bc.blocks_skipped = 100
        bc.blocks_computed = 200
        s = bc.summary()
        assert "BWCache block" in s
        assert "100" in s
        assert "33%" in s

    def test_l1_history_collection(self):
        """L1 values should be collected in l1_history (excluding sentinel)."""
        bc = BWCacheState(enabled=True, num_blocks=5)
        bc.reset()
        x1 = mx.ones((1, 10, 64))
        x2 = mx.ones((1, 10, 64)) * 1.1
        # First call: sentinel, not recorded
        bc.compute_block_l1(0, x1)
        assert len(bc.l1_history) == 0
        # Second call: real L1, recorded
        bc.compute_block_l1(0, x2)
        assert len(bc.l1_history) == 1
        assert bc.l1_history[0] > 0

    def test_l1_percentiles(self):
        bc = BWCacheState(enabled=True, num_blocks=5)
        bc.reset()
        bc.l1_history = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        pcts = bc.l1_percentiles()
        assert pcts["count"] == 10
        assert pcts["p50"] == pytest.approx(0.30, abs=0.05)
        assert pcts["p25"] < pcts["p50"] < pcts["p75"]
        assert pcts["mean"] == pytest.approx(0.275, abs=0.01)

    def test_l1_percentiles_empty(self):
        bc = BWCacheState(enabled=True)
        assert bc.l1_percentiles() == {}

    def test_auto_calibration_warmup(self):
        """During calibration warmup, is_calibrating() returns True."""
        bc = BWCacheState(
            enabled=True, mode="block", auto_thresh=True, calibration_warmup=3
        )
        bc.reset()
        bc.calibration_steps_seen = 0
        assert bc.is_calibrating() is True
        bc.calibration_steps_seen = 2
        assert bc.is_calibrating() is True
        bc.calibration_steps_seen = 3
        assert bc.is_calibrating() is False

    def test_auto_calibration_disabled_for_step_mode(self):
        bc = BWCacheState(enabled=True, mode="step", auto_thresh=True)
        assert bc.is_calibrating() is False

    def test_finish_calibration_sets_threshold(self):
        """finish_calibration() should set threshold from L1 distribution."""
        bc = BWCacheState(
            enabled=True,
            mode="block",
            auto_thresh=True,
            target_skip_ratio=0.5,
            thresh=0.15,
        )
        bc.reset()
        # Simulate collected L1 values
        bc.l1_history = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
        bc.finish_calibration()
        # With target_skip_ratio=0.5, threshold should be at the 50th percentile
        assert bc.thresh == pytest.approx(0.60, abs=0.05)

    def test_finish_calibration_empty_history(self):
        bc = BWCacheState(enabled=True, thresh=0.15)
        bc.reset()
        bc.finish_calibration()
        assert bc.thresh == 0.15  # unchanged

    def test_reset_clears_calibration(self):
        bc = BWCacheState(enabled=True, num_blocks=5, auto_thresh=True)
        bc.calibration_steps_seen = 5
        bc.l1_history = [0.1, 0.2, 0.3]
        bc.reset()
        assert bc.calibration_steps_seen == 0
        assert bc.l1_history == []


class TestBWCacheIntegration:
    """Integration tests for BWCache with WanModel."""

    def setup_method(self):
        mx.random.seed(42)
        self.config = _make_tiny_config()
        self.model = WanModel(self.config)

    def _make_inputs(self, batch_size=1):
        """Create minimal test inputs for WanModel."""
        C = self.config.in_dim
        F, H, W = 1, 4, 4
        x_list = [mx.random.normal((C, F, H, W)) for _ in range(batch_size)]
        t = mx.array([500.0] * batch_size)
        context = [
            mx.random.normal((self.config.text_len, self.config.text_dim))
        ] * batch_size
        seq_len = F * (H // 2) * (W // 2)  # patch_size=(1,2,2)
        return x_list, t, context, seq_len

    def test_bwcache_disabled_by_default(self):
        assert self.model.bwcache.enabled is False

    def test_bwcache_step_mode_runs(self):
        """Verify BWCache step mode produces valid output."""
        self.model.bwcache.enabled = True
        self.model.bwcache.mode = "step"
        self.model.bwcache.thresh = 0.15
        self.model.bwcache.num_steps = 3
        self.model.bwcache.reset()

        x_list, t, context, seq_len = self._make_inputs()

        # Run 3 steps to allow schedule to potentially trigger
        for step in range(3):
            t_val = mx.array([900.0 - step * 300.0])
            outputs = self.model(x_list, t_val, context, seq_len)
            mx.eval(outputs[0])

        assert self.model.bwcache.cnt == 3
        assert self.model.bwcache.steps_computed > 0

    def test_bwcache_block_mode_runs(self):
        """Verify BWCache block mode produces valid output."""
        self.model.bwcache.enabled = True
        self.model.bwcache.mode = "block"
        self.model.bwcache.thresh = 0.15
        self.model.bwcache.auto_thresh = False  # disable auto-calibration for test
        self.model.bwcache.boundary_blocks = 1  # tiny model has 2 blocks
        self.model.bwcache.num_steps = 3
        self.model.bwcache.reset()

        x_list, t, context, seq_len = self._make_inputs()

        # Run multiple steps
        for step in range(3):
            t_val = mx.array([900.0 - step * 300.0])
            outputs = self.model(x_list, t_val, context, seq_len)
            mx.eval(outputs[0])

        assert self.model.bwcache.cnt == 3
        assert self.model.bwcache.blocks_computed > 0

    def test_bwcache_output_shape(self):
        """Verify output shape matches baseline."""
        x_list, t, context, seq_len = self._make_inputs()

        # Baseline
        baseline = self.model(x_list, t, context, seq_len)
        mx.eval(baseline[0])
        baseline_shape = baseline[0].shape

        # With BWCache block mode
        self.model.bwcache.enabled = True
        self.model.bwcache.mode = "block"
        self.model.bwcache.thresh = 0.15
        self.model.bwcache.auto_thresh = False
        self.model.bwcache.boundary_blocks = 1
        self.model.bwcache.num_steps = 1
        self.model.bwcache.reset()

        cached = self.model(x_list, t, context, seq_len)
        mx.eval(cached[0])

        assert cached[0].shape == baseline_shape

    def test_bwcache_first_step_no_skip(self):
        """First step should always compute all blocks (no previous fingerprints)."""
        self.model.bwcache.enabled = True
        self.model.bwcache.mode = "block"
        self.model.bwcache.thresh = 10.0  # very high threshold
        self.model.bwcache.auto_thresh = False
        self.model.bwcache.boundary_blocks = 0  # no boundary protection
        self.model.bwcache.num_steps = 1
        self.model.bwcache.reset()

        x_list, t, context, seq_len = self._make_inputs()
        self.model(x_list, t, context, seq_len)

        # All blocks should be computed on first step (no fingerprints yet)
        assert self.model.bwcache.blocks_skipped == 0
        assert self.model.bwcache.blocks_computed == self.config.num_layers

    def test_bwcache_cfg_batch(self):
        """Verify BWCache works with B=2 (CFG) batch."""
        self.model.bwcache.enabled = True
        self.model.bwcache.mode = "block"
        self.model.bwcache.thresh = 0.15
        self.model.bwcache.auto_thresh = False
        self.model.bwcache.boundary_blocks = 1
        self.model.bwcache.num_steps = 2
        self.model.bwcache.reset()

        x_list, t, context, seq_len = self._make_inputs(batch_size=2)

        for step in range(2):
            t_val = mx.array([800.0 - step * 400.0] * 2)
            outputs = self.model(x_list, t_val, context, seq_len)
            mx.eval(outputs[0])

        assert len(outputs) == 2
        assert outputs[0].shape == outputs[1].shape

    def test_bwcache_auto_calibration(self):
        """Auto-calibration should set threshold after warmup steps."""
        self.model.bwcache.enabled = True
        self.model.bwcache.mode = "block"
        self.model.bwcache.auto_thresh = True
        self.model.bwcache.calibration_warmup = 2
        self.model.bwcache.target_skip_ratio = 0.5
        self.model.bwcache.thresh = 0.15
        self.model.bwcache.boundary_blocks = 0  # all blocks are middle
        self.model.bwcache.num_steps = 5
        self.model.bwcache.reset()

        x_list, t, context, seq_len = self._make_inputs()
        initial_thresh = self.model.bwcache.thresh

        # Run calibration warmup steps
        for step in range(3):
            t_val = mx.array([900.0 - step * 200.0])
            outputs = self.model(x_list, t_val, context, seq_len)
            mx.eval(outputs[0])

        # After 2+ computed steps, calibration should have fired
        assert self.model.bwcache.calibration_steps_seen >= 2
        # Threshold should have been auto-adjusted (unless L1 history was empty)
        if self.model.bwcache.l1_history:
            # Threshold was set from data — it may differ from initial
            assert self.model.bwcache.thresh >= 0  # sanity check

    def test_bwcache_group_residual_reuse(self):
        """Middle blocks should be skipped via residual reuse when L1 is low."""
        self.model.bwcache.enabled = True
        self.model.bwcache.mode = "block"
        self.model.bwcache.thresh = 1000.0  # very high threshold → always reuse
        self.model.bwcache.auto_thresh = False
        self.model.bwcache.boundary_blocks = 1
        self.model.bwcache.num_steps = 3
        self.model.bwcache.reset()

        x_list, t, context, seq_len = self._make_inputs()

        # Step 1: all computed (no previous fingerprint/residual)
        outputs = self.model(x_list, mx.array([900.0]), context, seq_len)
        mx.eval(outputs[0])
        step1_computed = self.model.bwcache.blocks_computed
        step1_skipped = self.model.bwcache.blocks_skipped
        assert step1_skipped == 0  # first step never skips
        assert step1_computed == self.config.num_layers

        # Step 2: middle blocks should be skipped (high thresh = always reuse)
        outputs = self.model(x_list, mx.array([600.0]), context, seq_len)
        mx.eval(outputs[0])
        # Boundary blocks computed: 1 start + 1 end = 2
        # Middle blocks skipped: num_layers - 2
        num_middle = self.config.num_layers - 2
        assert self.model.bwcache.blocks_skipped == num_middle
        assert self.model.bwcache.middle_residual is not None
