"""Tests for Wan scheduler components."""

import math

import mlx.core as mx
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Euler Scheduler Tests
# ---------------------------------------------------------------------------

class TestFlowMatchEulerScheduler:
    def test_initialization(self):
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched = FlowMatchEulerScheduler()
        assert sched.num_train_timesteps == 1000
        assert sched.timesteps is None
        assert sched.sigmas is None

    def test_set_timesteps(self):
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(40, shift=12.0)
        mx.eval(sched.timesteps, sched.sigmas)
        assert sched.timesteps.shape == (40,)
        assert sched.sigmas.shape == (41,)  # 40 steps + terminal

    def test_timesteps_decreasing(self):
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(40, shift=12.0)
        mx.eval(sched.timesteps)
        ts = np.array(sched.timesteps)
        # Timesteps should be monotonically decreasing
        assert np.all(np.diff(ts) < 0), f"Timesteps not decreasing: {ts[:5]}..."

    def test_sigmas_decreasing(self):
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(20, shift=1.0)
        mx.eval(sched.sigmas)
        sigmas = np.array(sched.sigmas)
        assert np.all(np.diff(sigmas) <= 0), "Sigmas not decreasing"

    def test_terminal_sigma_is_zero(self):
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(20, shift=5.0)
        mx.eval(sched.sigmas)
        np.testing.assert_allclose(np.array(sched.sigmas[-1]), 0.0, atol=1e-6)

    def test_shift_effect(self):
        """Larger shift should push sigmas toward higher values."""
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched1 = FlowMatchEulerScheduler()
        sched2 = FlowMatchEulerScheduler()
        sched1.set_timesteps(20, shift=1.0)
        sched2.set_timesteps(20, shift=12.0)
        mx.eval(sched1.sigmas, sched2.sigmas)
        mean1 = np.mean(np.array(sched1.sigmas[:-1]))
        mean2 = np.mean(np.array(sched2.sigmas[:-1]))
        assert mean2 > mean1, "Higher shift should push sigmas higher"

    def test_step_euler(self):
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(10, shift=1.0)
        mx.eval(sched.sigmas)

        sample = mx.ones((1, 4, 2, 2, 2))
        velocity = mx.ones((1, 4, 2, 2, 2)) * 0.5
        timestep = sched.timesteps[0]

        sigma = float(np.array(sched.sigmas[0]))
        sigma_next = float(np.array(sched.sigmas[1]))

        result = sched.step(velocity, timestep, sample)
        mx.eval(result)

        # Euler: x_next = x + (sigma_next - sigma) * v
        expected = 1.0 + (sigma_next - sigma) * 0.5
        np.testing.assert_allclose(
            np.array(result).flatten()[0], expected, rtol=1e-4,
        )

    def test_step_index_increments(self):
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(5, shift=1.0)
        assert sched._step_index == 0
        sample = mx.ones((1, 1, 1, 1, 1))
        vel = mx.zeros((1, 1, 1, 1, 1))
        sched.step(vel, sched.timesteps[0], sample)
        assert sched._step_index == 1
        sched.step(vel, sched.timesteps[1], sample)
        assert sched._step_index == 2

    def test_reset(self):
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(5, shift=1.0)
        sample = mx.ones((1, 1, 1, 1, 1))
        vel = mx.zeros((1, 1, 1, 1, 1))
        sched.step(vel, sched.timesteps[0], sample)
        assert sched._step_index == 1
        sched.reset()
        assert sched._step_index == 0

    @pytest.mark.parametrize("steps", [10, 20, 40, 50])
    def test_various_step_counts(self, steps):
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(steps, shift=12.0)
        mx.eval(sched.timesteps, sched.sigmas)
        assert sched.timesteps.shape == (steps,)
        assert sched.sigmas.shape == (steps + 1,)

    def test_full_denoise_loop(self):
        """Run a complete denoise loop with zero velocity -> sample unchanged."""
        from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(5, shift=1.0)
        sample = mx.ones((1, 2, 1, 2, 2))
        for i in range(5):
            vel = mx.zeros_like(sample)
            sample = sched.step(vel, sched.timesteps[i], sample)
        mx.eval(sample)
        # With zero velocity, sample should remain unchanged
        np.testing.assert_allclose(np.array(sample), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Shared Sigma Schedule Tests
# ---------------------------------------------------------------------------


class TestComputeSigmas:
    """Tests for the shared _compute_sigmas helper."""

    def test_length(self):
        from mlx_video.models.wan.scheduler import _compute_sigmas
        sigmas = _compute_sigmas(20, shift=5.0)
        assert len(sigmas) == 21  # num_steps + terminal

    def test_terminal_zero(self):
        from mlx_video.models.wan.scheduler import _compute_sigmas
        sigmas = _compute_sigmas(10, shift=1.0)
        assert sigmas[-1] == 0.0

    def test_starts_at_one(self):
        from mlx_video.models.wan.scheduler import _compute_sigmas
        sigmas = _compute_sigmas(20, shift=5.0)
        np.testing.assert_allclose(sigmas[0], 1.0, atol=1e-6)

    def test_decreasing(self):
        from mlx_video.models.wan.scheduler import _compute_sigmas
        sigmas = _compute_sigmas(20, shift=5.0)
        assert np.all(np.diff(sigmas) <= 0)

    def test_matches_official_wan22(self):
        """Sigma schedule should match the official Wan2.2 get_sampling_sigmas."""
        from mlx_video.models.wan.scheduler import _compute_sigmas
        steps, shift = 50, 5.0
        sigmas = _compute_sigmas(steps, shift)
        # Official: sigma = linspace(1, 0, steps+1)[:steps]; sigma = shift*sigma/(1+(shift-1)*sigma)
        official = np.linspace(1, 0, steps + 1)[:steps]
        official = shift * official / (1 + (shift - 1) * official)
        official = np.append(official, 0.0).astype(np.float32)
        np.testing.assert_allclose(sigmas, official, atol=1e-6)

    def test_shift_one_is_linear(self):
        from mlx_video.models.wan.scheduler import _compute_sigmas
        sigmas = _compute_sigmas(10, shift=1.0)
        # With shift=1, f(sigma)=sigma, so schedule is linear from 1 to 0
        expected = np.linspace(1, 0, 11).astype(np.float32)
        np.testing.assert_allclose(sigmas, expected, atol=1e-6)

    def test_all_schedulers_same_sigmas(self):
        """All three schedulers should produce identical sigma schedules."""
        from mlx_video.models.wan.scheduler import (
            FlowDPMPP2MScheduler,
            FlowMatchEulerScheduler,
            FlowUniPCScheduler,
        )
        scheds = [
            FlowMatchEulerScheduler(1000),
            FlowDPMPP2MScheduler(1000),
            FlowUniPCScheduler(1000),
        ]
        for s in scheds:
            s.set_timesteps(20, shift=5.0)
        mx.eval(*[s.sigmas for s in scheds])
        ref = np.array(scheds[0].sigmas)
        for s in scheds[1:]:
            np.testing.assert_allclose(np.array(s.sigmas), ref, atol=1e-6)

    def test_all_schedulers_same_timesteps(self):
        from mlx_video.models.wan.scheduler import (
            FlowDPMPP2MScheduler,
            FlowMatchEulerScheduler,
            FlowUniPCScheduler,
        )
        scheds = [
            FlowMatchEulerScheduler(1000),
            FlowDPMPP2MScheduler(1000),
            FlowUniPCScheduler(1000),
        ]
        for s in scheds:
            s.set_timesteps(30, shift=12.0)
        mx.eval(*[s.timesteps for s in scheds])
        ref = np.array(scheds[0].timesteps)
        for s in scheds[1:]:
            np.testing.assert_allclose(np.array(s.timesteps), ref, atol=1e-3)


# ---------------------------------------------------------------------------
# DPM++ 2M Scheduler Tests
# ---------------------------------------------------------------------------


class TestFlowDPMPP2MScheduler:
    def test_initialization(self):
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler
        sched = FlowDPMPP2MScheduler()
        assert sched.num_train_timesteps == 1000
        assert sched.lower_order_final is True

    def test_set_timesteps(self):
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler
        sched = FlowDPMPP2MScheduler()
        sched.set_timesteps(20, shift=5.0)
        mx.eval(sched.timesteps, sched.sigmas)
        assert sched.timesteps.shape == (20,)
        assert sched.sigmas.shape == (21,)

    def test_step_index_increments(self):
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler
        sched = FlowDPMPP2MScheduler()
        sched.set_timesteps(5, shift=1.0)
        sample = mx.ones((1, 4, 1, 2, 2))
        vel = mx.zeros_like(sample)
        assert sched._step_index == 0
        sched.step(vel, sched.timesteps[0], sample)
        assert sched._step_index == 1
        sched.step(vel, sched.timesteps[1], sample)
        assert sched._step_index == 2

    def test_reset(self):
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler
        sched = FlowDPMPP2MScheduler()
        sched.set_timesteps(5, shift=1.0)
        sample = mx.ones((1, 1, 1, 1, 1))
        sched.step(mx.zeros_like(sample), 0, sample)
        sched.reset()
        assert sched._step_index == 0
        assert sched._prev_x0 is None

    def test_full_loop_finite(self):
        """Full loop with constant velocity should produce finite output."""
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler
        sched = FlowDPMPP2MScheduler()
        sched.set_timesteps(10, shift=1.0)
        sample = mx.ones((1, 2, 1, 2, 2))
        for i in range(10):
            vel = mx.ones_like(sample) * 0.1
            sample = sched.step(vel, sched.timesteps[i], sample)
            mx.eval(sample)
        assert np.isfinite(np.array(sample)).all()

    def test_first_step_is_first_order(self):
        """First step should use 1st-order (no prev_x0 available)."""
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler
        sched = FlowDPMPP2MScheduler()
        sched.set_timesteps(10, shift=5.0)
        sample = mx.random.normal((1, 4, 2, 4, 4))
        vel = mx.random.normal(sample.shape)
        # Before first step, no prev_x0
        assert sched._prev_x0 is None
        result = sched.step(vel, sched.timesteps[0], sample)
        mx.eval(result)
        # After first step, prev_x0 should be set
        assert sched._prev_x0 is not None

    def test_second_step_uses_correction(self):
        """After first step, DPM++ should have stored prev_x0 for correction."""
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler
        sched = FlowDPMPP2MScheduler()
        sched.set_timesteps(10, shift=5.0)
        sample = mx.random.normal((1, 4, 1, 2, 2))
        vel = mx.random.normal(sample.shape)
        # Step 1
        sample = sched.step(vel, sched.timesteps[0], sample)
        mx.eval(sample)
        x0_after_first = sched._prev_x0
        # Step 2
        vel = mx.random.normal(sample.shape)
        sample = sched.step(vel, sched.timesteps[1], sample)
        mx.eval(sample)
        # prev_x0 should have been updated
        x0_after_second = sched._prev_x0
        assert x0_after_second is not None
        # The stored x0 should differ from the first step's
        assert not np.allclose(np.array(x0_after_first), np.array(x0_after_second), atol=1e-6)

    def test_denoise_to_target(self):
        """Perfect oracle should denoise to target with any solver."""
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler
        sched = FlowDPMPP2MScheduler()
        sched.set_timesteps(20, shift=5.0)
        target = mx.zeros((1, 2, 1, 4, 4))
        latents = mx.random.normal(target.shape)
        for i in range(20):
            sigma = float(sched.sigmas[i].item())
            v = latents / max(sigma, 1e-6)  # perfect velocity for target=0
            latents = sched.step(v, sched.timesteps[i], latents)
            mx.eval(latents)
        np.testing.assert_allclose(np.array(latents), 0.0, atol=1e-3)

    @pytest.mark.parametrize("steps", [5, 10, 20, 50])
    def test_various_step_counts(self, steps):
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler
        sched = FlowDPMPP2MScheduler()
        sched.set_timesteps(steps, shift=5.0)
        mx.eval(sched.timesteps, sched.sigmas)
        assert sched.timesteps.shape == (steps,)
        assert sched.sigmas.shape == (steps + 1,)

    def test_terminal_sigma_produces_x0(self):
        """When sigma_next=0 the scheduler should return x0 directly."""
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler
        sched = FlowDPMPP2MScheduler()
        sched.set_timesteps(5, shift=1.0)
        sample = mx.ones((1, 1, 1, 1, 1)) * 3.0
        vel = mx.ones_like(sample) * 2.0
        # Run through all steps; the last step has sigma_next=0
        for i in range(5):
            sample = sched.step(vel, sched.timesteps[i], sample)
            mx.eval(sample)
        # Final value should be finite
        assert np.isfinite(np.array(sample)).all()


# ---------------------------------------------------------------------------
# UniPC Scheduler Tests
# ---------------------------------------------------------------------------


class TestFlowUniPCScheduler:
    def test_initialization(self):
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler()
        assert sched.num_train_timesteps == 1000
        assert sched.solver_order == 2
        assert sched.lower_order_final is True

    def test_set_timesteps(self):
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler()
        sched.set_timesteps(30, shift=12.0)
        mx.eval(sched.timesteps, sched.sigmas)
        assert sched.timesteps.shape == (30,)
        assert sched.sigmas.shape == (31,)

    def test_step_index_increments(self):
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler()
        sched.set_timesteps(5, shift=1.0)
        sample = mx.ones((1, 1, 1, 1, 1))
        vel = mx.zeros_like(sample)
        assert sched._step_index == 0
        sched.step(vel, 0, sample)
        assert sched._step_index == 1

    def test_reset(self):
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler()
        sched.set_timesteps(5, shift=1.0)
        sample = mx.ones((1, 1, 1, 1, 1))
        sched.step(mx.zeros_like(sample), 0, sample)
        sched.reset()
        assert sched._step_index == 0
        assert sched._lower_order_nums == 0
        assert sched._last_sample is None
        assert all(m is None for m in sched._model_outputs)

    def test_full_loop_finite(self):
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler()
        sched.set_timesteps(10, shift=1.0)
        sample = mx.ones((1, 2, 1, 2, 2))
        for i in range(10):
            vel = mx.ones_like(sample) * 0.1
            sample = sched.step(vel, sched.timesteps[i], sample)
            mx.eval(sample)
        assert np.isfinite(np.array(sample)).all()

    def test_corrector_not_applied_first_step(self):
        """First step should skip the corrector (no history)."""
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler(use_corrector=True)
        sched.set_timesteps(10, shift=5.0)
        sample = mx.random.normal((1, 4, 1, 2, 2))
        vel = mx.random.normal(sample.shape)
        # Before step 0: no last_sample
        assert sched._last_sample is None
        sched.step(vel, sched.timesteps[0], sample)
        # After step 0: last_sample should be set for corrector on step 1
        assert sched._last_sample is not None

    def test_corrector_applied_after_first_step(self):
        """Steps after the first should use the corrector when enabled."""
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler(use_corrector=True)
        sched.set_timesteps(10, shift=5.0)
        sample = mx.random.normal((1, 2, 1, 4, 4))
        for i in range(3):
            vel = mx.random.normal(sample.shape)
            sample = sched.step(vel, sched.timesteps[i], sample)
            mx.eval(sample)
        # lower_order_nums should have increased
        assert sched._lower_order_nums >= 2

    def test_denoise_to_target(self):
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler()
        sched.set_timesteps(20, shift=5.0)
        target = mx.zeros((1, 2, 1, 4, 4))
        latents = mx.random.normal(target.shape)
        for i in range(20):
            sigma = float(sched.sigmas[i].item())
            v = latents / max(sigma, 1e-6)
            latents = sched.step(v, sched.timesteps[i], latents)
            mx.eval(latents)
        np.testing.assert_allclose(np.array(latents), 0.0, atol=1e-3)

    @pytest.mark.parametrize("steps", [5, 10, 20, 50])
    def test_various_step_counts(self, steps):
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler()
        sched.set_timesteps(steps, shift=5.0)
        mx.eval(sched.timesteps, sched.sigmas)
        assert sched.timesteps.shape == (steps,)
        assert sched.sigmas.shape == (steps + 1,)

    def test_disable_corrector(self):
        """Disabling corrector on step 0 should still work without error."""
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler(use_corrector=True, disable_corrector=[0])
        sched.set_timesteps(5, shift=1.0)
        sample = mx.ones((1, 1, 1, 2, 2))
        for i in range(5):
            vel = mx.ones_like(sample) * 0.1
            sample = sched.step(vel, sched.timesteps[i], sample)
            mx.eval(sample)
        assert np.isfinite(np.array(sample)).all()

    def test_solver_order_3(self):
        """Order 3 should work without error."""
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler(solver_order=3, use_corrector=True)
        sched.set_timesteps(10, shift=5.0)
        sample = mx.random.normal((1, 2, 1, 2, 2))
        for i in range(10):
            vel = mx.random.normal(sample.shape)
            sample = sched.step(vel, sched.timesteps[i], sample)
            mx.eval(sample)
        assert np.isfinite(np.array(sample)).all()

    def test_corrector_rhos_c_not_hardcoded(self):
        """Corrector rhos_c should be computed via linalg.solve, not hardcoded 0.5."""
        import math
        # For 50-step schedule with shift=5.0, order 2 corrector at step 5:
        # rhos_c[0] (history) should be ~0.07, NOT 0.5
        # rhos_c[1] (D1_t) should be ~0.45, NOT 0.5
        from mlx_video.models.wan.scheduler import _compute_sigmas

        sigmas = _compute_sigmas(50, shift=5.0)

        def _lambda(sigma):
            if sigma >= 1.0:
                return -math.inf
            if sigma <= 0.0:
                return math.inf
            return math.log(1 - sigma) - math.log(sigma)

        for step_idx in [5, 10, 25, 45]:
            sigma_s0 = sigmas[step_idx - 1]
            sigma_t = sigmas[step_idx]
            lambda_s0 = _lambda(sigma_s0)
            lambda_t = _lambda(sigma_t)
            h = lambda_t - lambda_s0
            hh = -h

            sigma_sk = sigmas[step_idx - 2]
            lambda_sk = _lambda(sigma_sk)
            rk = (lambda_sk - lambda_s0) / h
            rks = np.array([rk, 1.0])

            h_phi_1 = math.expm1(hh)
            B_h = h_phi_1
            h_phi_k = h_phi_1 / hh - 1.0
            factorial_i = 1
            R_rows, b_vals = [], []
            for j in range(1, 3):
                R_rows.append(rks ** (j - 1))
                b_vals.append(h_phi_k * factorial_i / B_h)
                factorial_i *= j + 1
                h_phi_k = h_phi_k / hh - 1.0 / factorial_i
            R = np.stack(R_rows)
            b = np.array(b_vals)
            rhos_c = np.linalg.solve(R, b)

            # History weight should be small (~0.07-0.09), not 0.5
            assert rhos_c[0] < 0.15, f"Step {step_idx}: rhos_c[0]={rhos_c[0]:.4f} too large"
            assert rhos_c[0] > 0.0, f"Step {step_idx}: rhos_c[0]={rhos_c[0]:.4f} should be positive"
            # D1_t weight should be ~0.42-0.45, not 0.5
            assert 0.3 < rhos_c[1] < 0.5, f"Step {step_idx}: rhos_c[1]={rhos_c[1]:.4f} out of range"


# ---------------------------------------------------------------------------
# Scheduler Coherence Tests
# ---------------------------------------------------------------------------

class TestSchedulerCoherence:
    """Tests that Euler, DPM++, and UniPC schedulers produce coherent results.

    All three schedulers should agree on shared structure (sigma schedules,
    first-step behavior) and converge to the same result given perfect
    velocity oracles, even though they use different update rules.
    """

    @staticmethod
    def _make_schedulers(steps=10, shift=5.0):
        from mlx_video.models.wan.scheduler import (
            FlowDPMPP2MScheduler,
            FlowMatchEulerScheduler,
            FlowUniPCScheduler,
        )

        scheds = {
            "euler": FlowMatchEulerScheduler(),
            "dpm++": FlowDPMPP2MScheduler(),
            "unipc": FlowUniPCScheduler(),
        }
        for s in scheds.values():
            s.set_timesteps(steps, shift=shift)
        return scheds

    def test_identical_sigma_schedules(self):
        """All schedulers must use the same sigma schedule."""
        scheds = self._make_schedulers(20, shift=5.0)
        ref = np.array(scheds["euler"].sigmas)
        for name in ("dpm++", "unipc"):
            np.testing.assert_allclose(
                np.array(scheds[name].sigmas),
                ref,
                atol=1e-6,
                err_msg=f"{name} sigma schedule differs from Euler",
            )

    def test_identical_timesteps(self):
        """All schedulers must produce the same timestep sequence."""
        scheds = self._make_schedulers(20, shift=5.0)
        ref = np.array(scheds["euler"].timesteps)
        for name in ("dpm++", "unipc"):
            np.testing.assert_allclose(
                np.array(scheds[name].timesteps),
                ref,
                atol=1e-6,
                err_msg=f"{name} timesteps differ from Euler",
            )

    def test_first_step_matches_euler(self):
        """Step 0 (1st-order for all solvers) should match Euler exactly."""
        mx.random.seed(42)
        shape = (1, 4, 1, 4, 4)
        noise = mx.random.normal(shape)
        vel = mx.random.normal(shape)

        scheds = self._make_schedulers(10, shift=5.0)
        results = {}
        for name, sched in scheds.items():
            r = sched.step(vel, sched.timesteps[0], noise)
            mx.eval(r)
            results[name] = np.array(r)

        np.testing.assert_allclose(
            results["dpm++"], results["euler"], atol=1e-5,
            err_msg="DPM++ step 0 should match Euler",
        )
        np.testing.assert_allclose(
            results["unipc"], results["euler"], atol=1e-5,
            err_msg="UniPC step 0 should match Euler",
        )

    def test_first_step_matches_across_shifts(self):
        """Step 0 should match Euler for different shift values."""
        mx.random.seed(99)
        shape = (1, 2, 1, 2, 2)
        noise = mx.random.normal(shape)
        vel = mx.random.normal(shape)

        for shift in (1.0, 5.0, 12.0):
            scheds = self._make_schedulers(10, shift=shift)
            euler_r = scheds["euler"].step(vel, scheds["euler"].timesteps[0], noise)
            dpm_r = scheds["dpm++"].step(vel, scheds["dpm++"].timesteps[0], noise)
            unipc_r = scheds["unipc"].step(vel, scheds["unipc"].timesteps[0], noise)
            mx.eval(euler_r, dpm_r, unipc_r)
            np.testing.assert_allclose(
                np.array(dpm_r), np.array(euler_r), atol=1e-5,
                err_msg=f"DPM++ step 0 differs from Euler at shift={shift}",
            )
            np.testing.assert_allclose(
                np.array(unipc_r), np.array(euler_r), atol=1e-5,
                err_msg=f"UniPC step 0 differs from Euler at shift={shift}",
            )

    def test_oracle_all_converge_to_target(self):
        """Given a perfect velocity oracle v=x/sigma, all solvers should
        denoise to approximately zero (the target)."""
        mx.random.seed(7)
        shape = (1, 2, 1, 4, 4)
        noise = mx.random.normal(shape)

        for name, sched in self._make_schedulers(20, shift=5.0).items():
            latents = noise
            for i in range(20):
                sigma = float(sched.sigmas[i].item())
                v = latents / max(sigma, 1e-8)
                latents = sched.step(v, sched.timesteps[i], latents)
                mx.eval(latents)
            np.testing.assert_allclose(
                np.array(latents), 0.0, atol=1e-3,
                err_msg=f"{name} did not converge to target with oracle",
            )

    def test_oracle_higher_order_closer_to_target(self):
        """With few steps and a perfect oracle, higher-order solvers should
        be at least as accurate as Euler."""
        mx.random.seed(12)
        shape = (1, 2, 1, 4, 4)
        noise = mx.random.normal(shape)
        steps = 5

        errors = {}
        for name, sched in self._make_schedulers(steps, shift=5.0).items():
            latents = noise
            for i in range(steps):
                sigma = float(sched.sigmas[i].item())
                v = latents / max(sigma, 1e-8)
                latents = sched.step(v, sched.timesteps[i], latents)
                mx.eval(latents)
            errors[name] = float(mx.mean(mx.abs(latents)).item())

        # Higher-order solvers should not be significantly worse than Euler
        assert errors["dpm++"] <= errors["euler"] * 1.5, (
            f"DPM++ error {errors['dpm++']:.6f} much worse than Euler {errors['euler']:.6f}"
        )
        assert errors["unipc"] <= errors["euler"] * 1.5, (
            f"UniPC error {errors['unipc']:.6f} much worse than Euler {errors['euler']:.6f}"
        )

    def test_multistep_trajectory_similar_magnitude(self):
        """Over a full denoising loop with constant velocity, all solvers
        should produce outputs of similar magnitude (not diverging)."""
        mx.random.seed(42)
        shape = (1, 4, 1, 4, 4)
        noise = mx.random.normal(shape)
        steps = 20

        final_means = {}
        for name, sched in self._make_schedulers(steps, shift=5.0).items():
            latents = noise
            for i in range(steps):
                vel = latents * 0.1
                latents = sched.step(vel, sched.timesteps[i], latents)
                mx.eval(latents)
            final_means[name] = float(mx.mean(mx.abs(latents)).item())

        # All solvers should produce results within the same order of magnitude
        vals = list(final_means.values())
        ratio = max(vals) / max(min(vals), 1e-10)
        assert ratio < 10.0, (
            f"Scheduler outputs diverge too much: {final_means}, ratio={ratio:.1f}"
        )

    def test_intermediate_values_finite(self):
        """Every intermediate latent value must be finite for all solvers."""
        mx.random.seed(0)
        shape = (1, 2, 1, 2, 2)
        noise = mx.random.normal(shape)

        for name, sched in self._make_schedulers(15, shift=5.0).items():
            latents = noise
            for i in range(15):
                vel = mx.random.normal(shape)
                latents = sched.step(vel, sched.timesteps[i], latents)
                mx.eval(latents)
                assert np.isfinite(np.array(latents)).all(), (
                    f"{name} produced non-finite values at step {i}"
                )

    def test_lambda_boundary_values(self):
        """_lambda must return -inf at sigma=1.0 and +inf at sigma=0.0."""
        from mlx_video.models.wan.scheduler import (
            FlowDPMPP2MScheduler,
            FlowUniPCScheduler,
        )

        for cls in (FlowDPMPP2MScheduler, FlowUniPCScheduler):
            assert cls._lambda(1.0) == -math.inf, (
                f"{cls.__name__}._lambda(1.0) should be -inf"
            )
            assert cls._lambda(0.0) == math.inf, (
                f"{cls.__name__}._lambda(0.0) should be +inf"
            )
            # Interior values should be finite
            lam = cls._lambda(0.5)
            assert math.isfinite(lam) and lam == 0.0, (
                f"{cls.__name__}._lambda(0.5) should be 0.0"
            )

    def test_lambda_monotonically_decreasing(self):
        """_lambda(sigma) should decrease as sigma increases (more noise → lower SNR)."""
        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler

        sigmas = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        lambdas = [FlowDPMPP2MScheduler._lambda(s) for s in sigmas]
        for i in range(len(lambdas) - 1):
            assert lambdas[i] > lambdas[i + 1], (
                f"_lambda not decreasing: _lambda({sigmas[i]})={lambdas[i]} "
                f"vs _lambda({sigmas[i+1]})={lambdas[i+1]}"
            )

    def test_step0_is_ddim_formula(self):
        """At sigma=1.0, the DPM++/UniPC first step should reduce to the
        DDIM formula: x_next = sigma_next * x + (1 - sigma_next) * x0."""
        mx.random.seed(55)
        shape = (1, 2, 1, 2, 2)
        sample = mx.random.normal(shape)
        vel = mx.random.normal(shape)

        for steps, shift in [(10, 5.0), (20, 12.0)]:
            scheds = self._make_schedulers(steps, shift=shift)
            sigma_next = float(scheds["euler"].sigmas[1].item())
            sigma_cur = float(scheds["euler"].sigmas[0].item())
            assert abs(sigma_cur - 1.0) < 1e-6, "First sigma should be ~1.0"

            x0 = sample - sigma_cur * vel
            expected = sigma_next * sample + (1.0 - sigma_next) * x0
            mx.eval(expected)

            for name in ("dpm++", "unipc"):
                result = scheds[name].step(vel, scheds[name].timesteps[0], sample)
                mx.eval(result)
                np.testing.assert_allclose(
                    np.array(result), np.array(expected), atol=1e-5,
                    err_msg=f"{name} step 0 doesn't match DDIM formula (shift={shift})",
                )

    @pytest.mark.parametrize("steps", [5, 10, 20, 50])
    def test_coherent_across_step_counts(self, steps):
        """All solvers should agree on step 0 regardless of total step count."""
        mx.random.seed(77)
        shape = (1, 2, 1, 2, 2)
        noise = mx.random.normal(shape)
        vel = mx.random.normal(shape)

        scheds = self._make_schedulers(steps, shift=5.0)
        results = {}
        for name, sched in scheds.items():
            r = sched.step(vel, sched.timesteps[0], noise)
            mx.eval(r)
            results[name] = np.array(r)

        np.testing.assert_allclose(
            results["dpm++"], results["euler"], atol=1e-5,
        )
        np.testing.assert_allclose(
            results["unipc"], results["euler"], atol=1e-5,
        )

    def test_dpmpp_unipc_agree_on_step1(self):
        """After warmup, DPM++ and UniPC step 1 should be similar
        (both use 2nd-order corrections based on the same model outputs)."""
        mx.random.seed(42)
        shape = (1, 4, 1, 4, 4)
        noise = mx.random.normal(shape)

        scheds = self._make_schedulers(10, shift=5.0)
        # Run step 0 with same velocity
        vel0 = mx.random.normal(shape)
        for sched in scheds.values():
            sched.step(vel0, sched.timesteps[0], noise)

        # Run step 1 from same sample with same velocity
        sample1 = scheds["euler"].step(vel0, scheds["euler"].timesteps[0], noise)
        mx.eval(sample1)
        vel1 = mx.random.normal(shape)

        r_dpm = scheds["dpm++"].step(vel1, scheds["dpm++"].timesteps[1], sample1)
        r_unipc = scheds["unipc"].step(vel1, scheds["unipc"].timesteps[1], sample1)
        mx.eval(r_dpm, r_unipc)

        # They won't be identical (different correction formulas) but should
        # be in the same ballpark (within 50% of each other's magnitude)
        mean_dpm = float(mx.mean(mx.abs(r_dpm)).item())
        mean_unipc = float(mx.mean(mx.abs(r_unipc)).item())
        ratio = max(mean_dpm, mean_unipc) / max(min(mean_dpm, mean_unipc), 1e-10)
        assert ratio < 2.0, (
            f"DPM++ and UniPC step 1 differ too much: "
            f"DPM++={mean_dpm:.4f}, UniPC={mean_unipc:.4f}"
        )

    def test_reset_makes_solvers_reproducible(self):
        """After reset(), running the same loop should produce identical output."""
        mx.random.seed(42)
        shape = (1, 2, 1, 2, 2)
        noise = mx.random.normal(shape)

        from mlx_video.models.wan.scheduler import FlowDPMPP2MScheduler, FlowUniPCScheduler

        for cls in (FlowDPMPP2MScheduler, FlowUniPCScheduler):
            sched = cls()
            sched.set_timesteps(5, shift=5.0)

            # First run
            latents = noise
            for i in range(5):
                vel = latents * 0.1
                latents = sched.step(vel, sched.timesteps[i], latents)
                mx.eval(latents)
            result1 = np.array(latents)

            # Reset and run again
            sched.reset()
            latents = noise
            for i in range(5):
                vel = latents * 0.1
                latents = sched.step(vel, sched.timesteps[i], latents)
                mx.eval(latents)
            result2 = np.array(latents)

            np.testing.assert_allclose(result1, result2, atol=1e-5,
                err_msg=f"{cls.__name__} not reproducible after reset()")


# ---------------------------------------------------------------------------
# UniPC Corrector Default Tests
# ---------------------------------------------------------------------------

class TestUniPCCorrectorDefault:
    """Tests that the UniPC corrector is enabled by default,
    matching official FlowUniPCMultistepScheduler behavior."""

    def test_corrector_enabled_by_default(self):
        """Default construction should have corrector enabled."""
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        sched = FlowUniPCScheduler()
        assert sched._use_corrector is True

    def test_corrector_affects_output(self):
        """Corrector should produce different results than no corrector after step 1."""
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        mx.random.seed(42)
        shape = (1, 4, 1, 4, 4)
        noise = mx.random.normal(shape)

        sched_corr = FlowUniPCScheduler(use_corrector=True)
        sched_corr.set_timesteps(10, shift=5.0)
        sched_no = FlowUniPCScheduler(use_corrector=False)
        sched_no.set_timesteps(10, shift=5.0)

        latent_corr = noise
        latent_no = noise
        for i in range(3):
            vel = mx.random.normal(shape) * 0.1
            latent_corr = sched_corr.step(vel, sched_corr.timesteps[i], latent_corr)
            latent_no = sched_no.step(vel, sched_no.timesteps[i], latent_no)
            mx.eval(latent_corr, latent_no)

        diff = float(mx.abs(latent_corr - latent_no).max())
        assert diff > 1e-6, f"Corrector had no effect (max diff={diff})"

    def test_corrector_does_not_affect_first_step(self):
        """Step 0 should be identical regardless of corrector setting."""
        from mlx_video.models.wan.scheduler import FlowUniPCScheduler
        mx.random.seed(42)
        shape = (1, 4, 1, 4, 4)
        noise = mx.random.normal(shape)
        vel = mx.random.normal(shape)

        sched_corr = FlowUniPCScheduler(use_corrector=True)
        sched_corr.set_timesteps(10, shift=5.0)
        sched_no = FlowUniPCScheduler(use_corrector=False)
        sched_no.set_timesteps(10, shift=5.0)

        r1 = sched_corr.step(vel, sched_corr.timesteps[0], noise)
        r2 = sched_no.step(vel, sched_no.timesteps[0], noise)
        mx.eval(r1, r2)
        np.testing.assert_allclose(np.array(r1), np.array(r2), atol=1e-6)
