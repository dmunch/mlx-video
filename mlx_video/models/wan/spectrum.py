"""Spectrum: Adaptive Spectral Feature Forecasting for diffusion acceleration.

Ports the Spectrum algorithm (CVPR 2026, Han et al.) to MLX for Apple Silicon.
Predicts transformer features using Chebyshev polynomials fitted via ridge
regression, enabling large skips of expensive forward passes during denoising.

Reference: https://github.com/hanjq17/Spectrum
Paper: https://arxiv.org/abs/2603.01623
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx


def _build_chebyshev_design(taus: mx.array, M: int) -> mx.array:
    """Build Chebyshev design matrix Φ from normalized timesteps.

    Evaluates T_0(τ)..T_M(τ) via the recurrence T_m = 2τ T_{m-1} - T_{m-2}.

    Args:
        taus: Normalized timesteps in [-1, 1], shape (K,)
        M: Maximum polynomial degree

    Returns:
        Design matrix of shape (K, M+1)
    """
    taus = taus.reshape(-1, 1)  # (K, 1)
    K = taus.shape[0]
    T0 = mx.ones((K, 1))
    if M == 0:
        return T0
    T1 = taus
    cols = [T0, T1]
    for _ in range(2, M + 1):
        Tm = 2 * taus * cols[-1] - cols[-2]
        cols.append(Tm)
    return mx.concatenate(cols[:M + 1], axis=1)  # (K, M+1)


class ChebyshevForecaster:
    """Fits Chebyshev polynomials to cached features via ridge regression.

    Maintains a sliding window of (timestep, feature) pairs and fits
    coefficients C such that Φ @ C ≈ H, where Φ is the Chebyshev design
    matrix and H is the feature matrix. Uses Cholesky-based ridge regression
    for efficient and stable solving.
    """

    def __init__(self, M: int = 4, K: int = 100, lam: float = 0.1, num_steps: int = 50):
        self.M = M
        self.K = K
        self.lam = lam
        self.num_steps = num_steps
        self.P = M + 1

        # Buffers
        self.t_buf: Optional[mx.array] = None   # (<=K,) step indices
        self.H_buf: Optional[mx.array] = None   # (<=K, F) flattened features
        self.feature_shape: Optional[tuple] = None

        # Cached fit results (invalidated on update)
        self._coef: Optional[mx.array] = None   # (P, F) fitted coefficients

    def _taus(self, t: mx.array) -> mx.array:
        """Map step indices to τ ∈ [-1, 1] using fixed [0, num_steps] range."""
        t_min = mx.array(0.0)
        t_max = mx.array(float(self.num_steps))
        mid = 0.5 * (t_min + t_max)
        rng = t_max - t_min
        return (t.astype(mx.float32) - mid) * 2.0 / rng

    def update(self, step: int, h_flat: mx.array) -> None:
        """Append (step, feature) to the cache and invalidate coefficients.

        Args:
            step: Current diffusion step index
            h_flat: Flattened feature vector, shape (F,)
        """
        t_val = mx.array(float(step))
        h_row = h_flat.reshape(1, -1)

        if self.t_buf is None:
            self.t_buf = t_val.reshape(1)
            self.H_buf = h_row
        else:
            self.t_buf = mx.concatenate([self.t_buf, t_val.reshape(1)])
            self.H_buf = mx.concatenate([self.H_buf, h_row])
            # Trim to max window size
            if self.t_buf.shape[0] > self.K:
                self.t_buf = self.t_buf[-self.K:]
                self.H_buf = self.H_buf[-self.K:]

        self._coef = None

    def _fit(self) -> None:
        """Fit Chebyshev coefficients via ridge regression with Cholesky solve."""
        if self._coef is not None:
            return

        taus = self._taus(self.t_buf)
        X = _build_chebyshev_design(taus, self.M).astype(mx.float32)  # (K, P)
        H = self.H_buf.astype(mx.float32)  # (K, F)
        P = X.shape[1]

        Xt = X.T  # (P, K)
        XtX = Xt @ X + self.lam * mx.eye(P)  # (P, P)
        XtH = Xt @ H  # (P, F)

        # Cholesky solve: (P, P) system — tiny, runs on CPU
        try:
            L = mx.linalg.cholesky(XtX, stream=mx.cpu)
            y = mx.linalg.solve_triangular(L, XtH, upper=False, stream=mx.cpu)
            C = mx.linalg.solve_triangular(L.T, y, upper=True, stream=mx.cpu)
        except Exception:
            # Add jitter for ill-conditioned systems
            jitter = 1e-6 * mx.mean(mx.diag(XtX))
            XtX_reg = XtX + jitter * mx.eye(P)
            L = mx.linalg.cholesky(XtX_reg, stream=mx.cpu)
            y = mx.linalg.solve_triangular(L, XtH, upper=False, stream=mx.cpu)
            C = mx.linalg.solve_triangular(L.T, y, upper=True, stream=mx.cpu)

        self._coef = C.astype(mx.bfloat16)  # (P, F)

    def predict(self, step: int) -> mx.array:
        """Predict features at the given step using fitted Chebyshev coefficients.

        Args:
            step: Diffusion step index to predict at

        Returns:
            Predicted features, shape (F,)
        """
        self._fit()
        t_star = mx.array(float(step))
        tau_star = self._taus(t_star)
        x_star = _build_chebyshev_design(tau_star.reshape(1), self.M)  # (1, P)
        h_flat = x_star.astype(mx.float32) @ self._coef.astype(mx.float32)  # (1, F)
        return h_flat.reshape(-1).astype(mx.bfloat16)

    @property
    def num_cached(self) -> int:
        return 0 if self.t_buf is None else self.t_buf.shape[0]

    def reset(self):
        self.t_buf = None
        self.H_buf = None
        self._coef = None


class SpectrumForecaster:
    """Blends Chebyshev polynomial prediction with discrete Taylor (Newton) extrapolation.

    The final prediction is: h = (1 - w) * h_taylor + w * h_chebyshev
    where w controls the blend. The Taylor component uses first-order Newton
    forward differences from the most recent cached points.
    """

    def __init__(self, M: int = 4, K: int = 100, lam: float = 0.1,
                 w: float = 0.5, num_steps: int = 50):
        self.cheb = ChebyshevForecaster(M=M, K=K, lam=lam, num_steps=num_steps)
        self.w = w

    def _taylor_predict(self, step: int) -> mx.array:
        """First-order Newton forward difference extrapolation."""
        H = self.cheb.H_buf
        t = self.cheb.t_buf

        h_i = H[-1]
        t_i = t[-1]

        if t.shape[0] < 2:
            return h_i

        h_im1 = H[-2]
        t_im1 = t[-2]

        # Forward difference
        dh = h_i - h_im1
        dt_last = mx.maximum(t_i - t_im1, mx.array(1e-8))
        k = (mx.array(float(step)) - t_i) / dt_last
        k = k.astype(h_i.dtype)

        return h_i + k * dh

    def update(self, step: int, h_flat: mx.array) -> None:
        self.cheb.update(step, h_flat)

    def predict(self, step: int) -> mx.array:
        """Predict features as a blend of Chebyshev and Taylor predictions."""
        h_cheb = self.cheb.predict(step)
        h_taylor = self._taylor_predict(step)
        return (1.0 - self.w) * h_taylor + self.w * h_cheb

    @property
    def num_cached(self) -> int:
        return self.cheb.num_cached

    def reset(self):
        self.cheb.reset()


@dataclass
class SpectrumState:
    """Configuration and runtime state for Spectrum acceleration.

    Tracks the adaptive scheduling counters and holds the forecaster instance.
    Each model instance (e.g., high-noise and low-noise in Wan2.2 dual mode)
    should have its own SpectrumState.
    """

    enabled: bool = False

    # Hyperparameters
    m: int = 4                  # Chebyshev polynomial degree
    lam: float = 0.1            # Ridge regression regularization
    w: float = 0.5              # Chebyshev/Taylor blend weight
    k_max: int = 100            # Maximum cache size
    warmup_steps: int = 5       # Always compute first N steps
    cutoff_steps: int = 0       # Always compute last N steps (detail protection)
    window_size: int = 2        # Initial compute interval
    flex_window: float = 0.75   # Window growth rate (α in paper)

    # Runtime state
    num_steps: int = 0          # Total diffusion steps (set before loop)
    cnt: int = 0                # Current step counter
    curr_ws: float = 2.0        # Current window size (grows during run)
    num_consecutive_cached: int = 0  # Consecutive predicted steps
    forecaster: Optional[SpectrumForecaster] = field(default=None, repr=False)

    # Stats
    steps_computed: int = 0
    steps_predicted: int = 0

    def should_compute(self) -> bool:
        """Determine whether to run the full transformer at the current step.

        Uses the adaptive scheduling from the Spectrum paper:
        - Always compute during warmup (first N steps)
        - Always compute during cutoff (last N steps, protects fine details)
        - After warmup, compute when consecutive cached steps hits the window threshold
        - The window grows by flex_window after each compute step
        """
        if self.cnt < self.warmup_steps:
            return True

        if self.cutoff_steps > 0 and self.cnt >= self.num_steps - self.cutoff_steps:
            return True

        should = (self.num_consecutive_cached + 1) % math.floor(self.curr_ws) == 0
        return should

    def step(self, computed: bool) -> None:
        """Advance counters after a step. Call after should_compute + forward/predict."""
        if computed:
            self.num_consecutive_cached = 0
            self.steps_computed += 1
            if self.cnt >= self.warmup_steps:
                self.curr_ws += self.flex_window
                self.curr_ws = round(self.curr_ws, 3)
        else:
            self.num_consecutive_cached += 1
            self.steps_predicted += 1

        self.cnt += 1

        # Reset at end of generation
        if self.cnt == self.num_steps:
            self.cnt = 0
            self.num_consecutive_cached = 0
            self.curr_ws = float(self.window_size)

    def reset(self):
        """Reset all runtime state for a new generation."""
        self.cnt = 0
        self.curr_ws = float(self.window_size)
        self.num_consecutive_cached = 0
        self.steps_computed = 0
        self.steps_predicted = 0
        if self.forecaster is not None:
            self.forecaster.reset()
            self.forecaster = None

    def get_or_create_forecaster(self) -> SpectrumForecaster:
        """Lazily create the forecaster on first use."""
        if self.forecaster is None:
            self.forecaster = SpectrumForecaster(
                M=self.m, K=self.k_max, lam=self.lam,
                w=self.w, num_steps=self.num_steps,
            )
        return self.forecaster
