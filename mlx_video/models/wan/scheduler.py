"""Flow matching scheduler for Wan2.2 inference."""

import numpy as np

import mlx.core as mx


class FlowMatchEulerScheduler:
    """Simple Euler scheduler for flow matching diffusion.

    Implements the flow matching formulation where the model predicts
    velocity (flow) and we use Euler steps to denoise.
    """

    def __init__(self, num_train_timesteps: int = 1000):
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = None
        self.sigmas = None

    def set_timesteps(self, num_steps: int, shift: float = 1.0):
        """Compute sigma schedule with shift.

        Args:
            num_steps: Number of inference steps.
            shift: Noise schedule shift factor.
        """
        # Linear spacing from sigma_max to sigma_min
        sigmas = np.linspace(1.0, 1.0 / self.num_train_timesteps, self.num_train_timesteps)[::-1]
        sigmas = 1.0 - sigmas

        # Select evenly spaced subset
        indices = np.linspace(0, len(sigmas) - 1, num_steps + 1).astype(int)
        sigmas = sigmas[indices[:-1]]

        # Apply shift: sigma' = shift * sigma / (1 + (shift - 1) * sigma)
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)

        # Convert to timesteps
        timesteps = sigmas * self.num_train_timesteps
        self.timesteps = mx.array(timesteps.astype(np.float32))

        # Append terminal sigma=0
        sigmas = np.append(sigmas, 0.0)
        self.sigmas = mx.array(sigmas.astype(np.float32))
        self._step_index = 0

    def step(
        self,
        model_output: mx.array,
        timestep,
        sample: mx.array,
    ) -> mx.array:
        """Euler step for flow matching.

        In flow matching, model predicts velocity v, and:
            x_{t-1} = sample + (sigma_{t-1} - sigma_t) * v

        Args:
            model_output: Predicted velocity [B, C, T, H, W]
            timestep: Current timestep (unused, step index is tracked internally)
            sample: Current noisy sample [B, C, T, H, W]

        Returns:
            Updated sample
        """
        # Use Python floats to avoid creating mx.array scalars that
        # could trigger type promotion (per fast-mlx guide)
        dt = float(self.sigmas[self._step_index + 1].item()) - float(self.sigmas[self._step_index].item())
        x_next = sample + dt * model_output

        self._step_index += 1
        return x_next

    def reset(self):
        """Reset step counter for new generation."""
        self._step_index = 0
