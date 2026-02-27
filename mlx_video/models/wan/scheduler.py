"""Flow matching schedulers for Wan2.2 inference.

Provides Euler, DPM++2M, and UniPC solvers for flow matching diffusion.
Higher-order solvers (DPM++, UniPC) converge faster, needing fewer steps
for the same quality as Euler.
"""

import math

import numpy as np

import mlx.core as mx


def _compute_sigmas(num_steps: int, shift: float = 1.0) -> np.ndarray:
    """Compute shifted sigma schedule matching official Wan2.2 code.

    Returns num_steps+1 values (the last being 0.0 for the terminal state).
    """
    sigmas = np.linspace(1.0, 0.0, num_steps + 1)[:num_steps]
    sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    return np.append(sigmas, 0.0).astype(np.float32)


class FlowMatchEulerScheduler:
    """1st-order Euler scheduler for flow matching diffusion."""

    def __init__(self, num_train_timesteps: int = 1000):
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = None
        self.sigmas = None

    def set_timesteps(self, num_steps: int, shift: float = 1.0):
        sigmas = _compute_sigmas(num_steps, shift)
        self.sigmas = mx.array(sigmas)
        self.timesteps = mx.array(sigmas[:-1] * self.num_train_timesteps)
        self._step_index = 0

    def step(
        self,
        model_output: mx.array,
        timestep,
        sample: mx.array,
    ) -> mx.array:
        """Euler step: x_next = x + (sigma_next - sigma_cur) * v."""
        dt = float(self.sigmas[self._step_index + 1].item()) - float(
            self.sigmas[self._step_index].item()
        )
        x_next = sample + dt * model_output
        self._step_index += 1
        return x_next

    def reset(self):
        self._step_index = 0


class FlowDPMPP2MScheduler:
    """DPM-Solver++(2M) for flow matching diffusion.

    2nd-order multistep solver that reuses the previous step's model output
    for a correction term. Falls back to 1st order on the first and
    (optionally) last step. Reference: Wan2.2 fm_solvers.py.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        lower_order_final: bool = True,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.lower_order_final = lower_order_final
        self.timesteps = None
        self.sigmas = None

    def set_timesteps(self, num_steps: int, shift: float = 1.0):
        sigmas = _compute_sigmas(num_steps, shift)
        self.sigmas = mx.array(sigmas)
        self.timesteps = mx.array(sigmas[:-1] * self.num_train_timesteps)
        # Store sigmas as Python floats for scalar math
        self._sigmas_float = sigmas.tolist()
        self._step_index = 0
        self._num_steps = num_steps
        self._prev_x0 = None  # previous x0 prediction for 2nd-order correction

    @staticmethod
    def _lambda(sigma: float) -> float:
        """log-SNR: lambda(sigma) = log((1-sigma)/sigma).

        Returns -inf at sigma=1.0 (pure noise) and +inf at sigma=0.0 (clean),
        matching torch.log behavior in the official code.
        """
        if sigma >= 1.0:
            return -math.inf
        if sigma <= 0.0:
            return math.inf
        return math.log((1.0 - sigma) / sigma)

    def step(
        self,
        model_output: mx.array,
        timestep,
        sample: mx.array,
    ) -> mx.array:
        """DPM++(2M) step for flow matching.

        Converts velocity prediction to x0, then applies 1st or 2nd order
        update depending on available history.
        """
        i = self._step_index
        s = self._sigmas_float

        sigma_cur = s[i]
        sigma_next = s[i + 1]

        # Convert velocity -> x0 prediction: x0 = sample - sigma * v
        x0 = sample - sigma_cur * model_output

        # Decide order: 1st for first step, last step (if lower_order_final
        # and few steps), otherwise 2nd
        use_first_order = (
            self._prev_x0 is None
            or (
                self.lower_order_final
                and i == self._num_steps - 1
                and self._num_steps < 15
            )
        )

        if use_first_order or sigma_next == 0.0:
            # 1st order DPM++ (equivalent to DDIM):
            # x_next = (σ_next/σ_cur)*x - (α_next*(exp(-h)-1))*x0
            if sigma_next == 0.0:
                x_next = x0
            else:
                lambda_cur = self._lambda(sigma_cur)
                lambda_next = self._lambda(sigma_next)
                h = lambda_next - lambda_cur
                alpha_next = 1.0 - sigma_next
                coeff_x = sigma_next / sigma_cur
                coeff_x0 = alpha_next * math.expm1(-h)
                x_next = coeff_x * sample - coeff_x0 * x0
        else:
            # 2nd order DPM++(2M) with midpoint correction
            sigma_prev = s[i - 1]
            lambda_prev = self._lambda(sigma_prev)
            lambda_cur = self._lambda(sigma_cur)
            lambda_next = self._lambda(sigma_next)

            h = lambda_next - lambda_cur
            h_0 = lambda_cur - lambda_prev
            r0 = h_0 / h

            # D0 = current x0, D1 = correction from previous x0
            D0 = x0
            D1 = (1.0 / r0) * (x0 - self._prev_x0)

            alpha_next = 1.0 - sigma_next
            exp_neg_h_m1 = math.expm1(-h)  # exp(-h) - 1

            x_next = (
                (sigma_next / sigma_cur) * sample
                - (alpha_next * exp_neg_h_m1) * D0
                - 0.5 * (alpha_next * exp_neg_h_m1) * D1
            )

        self._prev_x0 = x0
        self._step_index += 1
        return x_next

    def reset(self):
        self._step_index = 0
        self._prev_x0 = None


class FlowUniPCScheduler:
    """UniPC (Unified Predictor-Corrector) for flow matching diffusion.

    Multi-step predictor-corrector solver with configurable order.
    The corrector refines each step using the model output that was already
    computed, costing no extra model evaluations. Official Wan2.2 default.
    Reference: Wan2.2 fm_solvers_unipc.py.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        lower_order_final: bool = True,
        disable_corrector: list | None = None,
        use_corrector: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.solver_order = solver_order
        self.lower_order_final = lower_order_final
        self._use_corrector = use_corrector
        self.disable_corrector = set(disable_corrector or [])
        self.timesteps = None
        self.sigmas = None

    def set_timesteps(self, num_steps: int, shift: float = 1.0):
        sigmas = _compute_sigmas(num_steps, shift)
        self.sigmas = mx.array(sigmas)
        self.timesteps = mx.array(sigmas[:-1] * self.num_train_timesteps)
        self._sigmas_float = sigmas.tolist()
        self._step_index = 0
        self._num_steps = num_steps
        self._lower_order_nums = 0
        # Model output (x0) history for multi-step, stored newest-last
        self._model_outputs = [None] * self.solver_order
        self._last_sample = None  # sample before prediction (for corrector)
        self._this_order = 1

    @staticmethod
    def _lambda(sigma: float) -> float:
        """log-SNR: lambda(sigma) = log((1-sigma)/sigma).

        Returns -inf at sigma=1.0 (pure noise) and +inf at sigma=0.0 (clean),
        matching torch.log behavior in the official code.
        """
        if sigma >= 1.0:
            return -math.inf
        if sigma <= 0.0:
            return math.inf
        return math.log((1.0 - sigma) / sigma)

    def _convert_output(self, velocity: mx.array, sample: mx.array) -> mx.array:
        """Convert velocity prediction to x0: x0 = sample - sigma * v."""
        sigma = self._sigmas_float[self._step_index]
        return sample - sigma * velocity

    def _uni_p_bh2(self, x0: mx.array, sample: mx.array, order: int) -> mx.array:
        """UniP predictor with B(h)=expm1(-h) basis (bh2 variant).

        Matches official multistep_uni_p_bh_update: computes rhos_p via
        linalg.solve for order >= 3; order <= 2 uses analytic rhos_p=[0.5].
        """
        i = self._step_index
        s = self._sigmas_float

        sigma_s0 = s[i]
        sigma_t = s[i + 1]

        if sigma_t == 0.0:
            return x0

        lambda_s0 = self._lambda(sigma_s0)
        lambda_t = self._lambda(sigma_t)
        h = lambda_t - lambda_s0
        hh = -h  # negated for predict_x0

        alpha_t = 1.0 - sigma_t
        h_phi_1 = math.expm1(hh)
        B_h = h_phi_1

        m0 = self._model_outputs[-1]
        # Base prediction
        x_t = (sigma_t / sigma_s0) * sample - (alpha_t * h_phi_1) * m0

        if order >= 2 and m0 is not None:
            rks = []
            D1s = []
            for k in range(1, order):
                si_idx = i - k
                if si_idx < 0 or self._model_outputs[-(k + 1)] is None:
                    break
                mk = self._model_outputs[-(k + 1)]
                sigma_sk = s[si_idx]
                lambda_sk = self._lambda(sigma_sk)
                rk = (lambda_sk - lambda_s0) / h
                if math.isinf(rk):
                    break
                rks.append(rk)
                D1s.append((mk - m0) / rk)

            if D1s:
                effective_order = len(D1s) + 1
                if effective_order <= 2:
                    # Analytic solution for order 2
                    rhos_p = [0.5]
                else:
                    rks_arr = np.array(rks, dtype=np.float64)
                    h_phi_k = h_phi_1 / hh - 1.0
                    factorial_i = 1
                    R_rows = []
                    b_vals = []
                    for j in range(1, effective_order):
                        R_rows.append(rks_arr ** (j - 1))
                        b_vals.append(float(h_phi_k * factorial_i / B_h))
                        factorial_i *= j + 1
                        h_phi_k = h_phi_k / hh - 1.0 / factorial_i
                    R = np.stack(R_rows)
                    b = np.array(b_vals)
                    rhos_p = np.linalg.solve(R, b).tolist()

                pred_res = sum(r * d for r, d in zip(rhos_p, D1s))
                x_t = x_t - (alpha_t * B_h) * pred_res

        return x_t

    def _uni_c_bh2(
        self,
        model_x0: mx.array,
        last_sample: mx.array,
        this_sample: mx.array,
        order: int,
    ) -> mx.array:
        """UniC corrector with B(h)=expm1(-h) basis (bh2 variant).

        Matches official multistep_uni_c_bh_update: computes rhos_c via
        linalg.solve for order >= 2 (not hardcoded 0.5).
        """
        i = self._step_index
        s = self._sigmas_float

        sigma_s0 = s[i - 1]
        sigma_t = s[i]

        if sigma_t == 0.0:
            return this_sample

        lambda_s0 = self._lambda(sigma_s0)
        lambda_t = self._lambda(sigma_t)
        h = lambda_t - lambda_s0
        hh = -h  # negated for predict_x0

        alpha_t = 1.0 - sigma_t
        h_phi_1 = math.expm1(hh)
        B_h = h_phi_1

        m0 = self._model_outputs[-1]
        # Re-derive base from last_sample
        x_t_ = (sigma_t / sigma_s0) * last_sample - (alpha_t * h_phi_1) * m0

        D1_t = model_x0 - m0

        # Gather rks and D1s from history
        rks = []
        D1s = []
        for k in range(1, order):
            si_idx = i - (k + 1)
            if si_idx < 0 or self._model_outputs[-(k + 1)] is None:
                break
            mk = self._model_outputs[-(k + 1)]
            sigma_sk = s[si_idx]
            lambda_sk = self._lambda(sigma_sk)
            rk = (lambda_sk - lambda_s0) / h
            if math.isinf(rk):
                break  # History references sigma=1.0 boundary; reduce order
            rks.append(rk)
            D1s.append((mk - m0) / rk)
        rks.append(1.0)
        effective_order = len(rks)  # = len(D1s) + 1

        # Compute rhos_c coefficients
        if effective_order == 1:
            rhos_c = [0.5]
        else:
            rks_arr = np.array(rks, dtype=np.float64)
            h_phi_k = h_phi_1 / hh - 1.0
            factorial_i = 1
            R_rows = []
            b_vals = []
            for j in range(1, effective_order + 1):
                R_rows.append(rks_arr ** (j - 1))
                b_vals.append(float(h_phi_k * factorial_i / B_h))
                factorial_i *= j + 1
                h_phi_k = h_phi_k / hh - 1.0 / factorial_i
            R = np.stack(R_rows)
            b = np.array(b_vals)
            rhos_c = np.linalg.solve(R, b).tolist()

        # Apply correction
        corr_res = mx.zeros_like(D1_t)
        for k_idx, d1 in enumerate(D1s):
            corr_res = corr_res + rhos_c[k_idx] * d1
        x_t = x_t_ - (alpha_t * B_h) * (corr_res + rhos_c[-1] * D1_t)
        return x_t

    def step(
        self,
        model_output: mx.array,
        timestep,
        sample: mx.array,
    ) -> mx.array:
        """UniPC step: correct current, then predict next."""
        i = self._step_index

        # Convert velocity -> x0
        x0 = self._convert_output(model_output, sample)

        # 1. Corrector: refine current sample if we have history
        use_corrector = (
            self._use_corrector
            and i > 0
            and (i - 1) not in self.disable_corrector
            and self._last_sample is not None
        )
        if use_corrector:
            sample = self._uni_c_bh2(x0, self._last_sample, sample, self._this_order)

        # 2. Shift model output history
        for k in range(self.solver_order - 1):
            self._model_outputs[k] = self._model_outputs[k + 1]
        self._model_outputs[-1] = x0

        # 3. Determine prediction order
        if self.lower_order_final:
            this_order = min(self.solver_order, self._num_steps - i)
        else:
            this_order = self.solver_order
        self._this_order = min(this_order, self._lower_order_nums + 1)

        # 4. Predict next sample
        self._last_sample = sample
        x_next = self._uni_p_bh2(x0, sample, self._this_order)

        if self._lower_order_nums < self.solver_order:
            self._lower_order_nums += 1

        self._step_index += 1
        return x_next

    def reset(self):
        self._step_index = 0
        self._lower_order_nums = 0
        self._model_outputs = [None] * self.solver_order
        self._last_sample = None
        self._this_order = 1
