"""Core training loop for Wan2.2 LoRA training.

Implements flow matching loss with timestep sampling bias control,
MLX value_and_grad for LoRA parameters, and the training epoch loop.
"""

import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tqdm import tqdm

from mlx_video.training.config import TrainingConfig
from mlx_video.training.dataset import EncodedItem
from mlx_video.utils import Colors


def _sample_timestep(
    num_train_timesteps: int,
    sampling: str,
    rng: random.Random,
) -> float:
    """Sample a timestep (as sigma in [0, 1]) with optional bias.

    For flow matching, sigma represents the noise level:
      - sigma=1.0: pure noise (high noise)
      - sigma=0.0: clean data (low noise)

    Args:
        num_train_timesteps: Number of training timesteps (e.g., 1000).
        sampling: One of 'balanced', 'low_bias', 'high_bias'.
        rng: Random number generator.

    Returns:
        Sigma value in (0, 1).
    """
    if sampling == "balanced":
        # Uniform sampling across [0, 1]
        t = rng.random()
    elif sampling == "low_bias":
        # Beta distribution biased toward low sigma (fine details/identity)
        # alpha=1, beta=2 → mean ~0.33, more samples near 0
        t = rng.betavariate(1.0, 2.0)
    elif sampling == "high_bias":
        # Beta distribution biased toward high sigma (composition/motion)
        # alpha=2, beta=1 → mean ~0.67, more samples near 1
        t = rng.betavariate(2.0, 1.0)
    else:
        t = rng.random()

    # Clamp to avoid exact 0 or 1
    t = max(1e-5, min(1.0 - 1e-5, t))
    return t


def _apply_shift(sigma: float, shift: float) -> float:
    """Apply noise schedule shift (matching Wan2.2 scheduler)."""
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def compute_loss(
    model: nn.Module,
    item: EncodedItem,
    sigma: float,
    noise: mx.array,
    text_len: int,
    shift: float,
) -> mx.array:
    """Compute flow matching MSE loss for a single training sample.

    Flow matching loss: MSE(predicted_velocity, target_velocity)
    where target_velocity = noise - clean_latent (the direction from clean to noise).

    Args:
        model: WanModel with LoRA layers injected.
        item: Pre-encoded training sample.
        sigma: Noise level in [0, 1] (before shift).
        noise: Random noise matching latent shape.
        text_len: Model text sequence length.
        shift: Noise schedule shift parameter.

    Returns:
        Scalar loss value.
    """
    clean = item.clean_latents  # [z_dim, 1, H_lat, W_lat]

    # Apply shift to sigma (matching inference schedule)
    sigma_shifted = _apply_shift(sigma, shift)

    # Interpolate: noisy = (1 - sigma) * clean + sigma * noise
    noisy = (1.0 - sigma_shifted) * clean + sigma_shifted * noise

    # Timestep value for the model (sigma * num_train_timesteps)
    timestep = mx.array([sigma_shifted * 1000.0])

    # Prepare text embedding for the model (pass as list so model applies embed_text MLP)
    context = [item.text_embedding]  # list of [text_len, text_dim]

    # Compute latent spatial dimensions for sequence length
    _, _, h_lat, w_lat = clean.shape
    patch_size = model.config.patch_size
    t_lat = 1  # Single frame
    f_grid = t_lat // patch_size[0]
    h_grid = h_lat // patch_size[1]
    w_grid = w_lat // patch_size[2]
    seq_len = f_grid * h_grid * w_grid

    # Forward pass
    predicted = model(
        [noisy],
        t=timestep,
        context=context,
        seq_len=seq_len,
    )
    predicted_velocity = predicted[0]  # [z_dim, 1, H_lat, W_lat]

    # Target velocity: v = noise - clean (flow matching convention)
    target_velocity = noise - clean

    # MSE loss
    error = (predicted_velocity - target_velocity).square()
    return error.mean()


def train(
    model: nn.Module,
    encoded_data: list[EncodedItem],
    config: TrainingConfig,
) -> None:
    """Run the full training loop.

    Args:
        model: WanModel with LoRA layers injected and base weights frozen.
        encoded_data: Pre-encoded training samples.
        config: Training configuration.
    """
    from mlx_video.training.plotting import LossHistory, plot_loss
    from mlx_video.training.save import save_lora_weights

    num_epochs = config.training.num_epochs
    batch_size = config.training.batch_size
    lr = config.training.learning_rate
    sampling = config.training.timestep_sampling
    log_freq = config.monitoring.log_frequency
    plot_freq = config.monitoring.plot_frequency
    preview_freq = config.monitoring.generate_image_frequency
    save_freq = config.checkpoint.save_frequency
    output_dir = config.checkpoint.output_dir
    shift = getattr(model.config, "sample_shift", 12.0)
    text_len = model.config.text_len

    # Setup optimizer
    optimizer_cls = {"adam": optim.Adam, "adamw": optim.AdamW}.get(
        config.training.optimizer.lower(), optim.AdamW
    )
    optimizer = optimizer_cls(learning_rate=lr)

    rng = random.Random(config.seed)
    mx.random.seed(config.seed)

    # Define loss function for value_and_grad
    def loss_fn(model, items_batch, sigmas, noises):
        losses = []
        for item, sigma, noise in zip(items_batch, sigmas, noises):
            loss = compute_loss(model, item, sigma, noise, text_len, shift)
            losses.append(loss)
        return mx.mean(mx.stack(losses))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training loop
    steps_per_epoch = max(1, len(encoded_data) // batch_size)
    total_steps = num_epochs * steps_per_epoch
    global_step = 0
    running_loss = 0.0
    loss_count = 0
    loss_history = LossHistory()
    plot_path = f"{output_dir}/loss_plot.png"

    print(f"\n{Colors.CYAN}{'='*60}")
    print(f"  Wan2.2 LoRA Training")
    print(f"{'='*60}{Colors.RESET}")
    print(f"{Colors.DIM}  Training samples: {len(encoded_data)}")
    print(f"  Epochs: {num_epochs}, Steps/epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Batch size: {batch_size}, LR: {lr}")
    print(f"  Timestep sampling: {sampling}")
    print(f"  Shift: {shift}")
    print(f"  Optimizer: {config.training.optimizer}")
    print(f"{Colors.RESET}")

    t_start = time.time()

    # --- Baseline loss at step 0 (forward-only, no gradient) ---
    print(f"  {Colors.DIM}Computing baseline loss...{Colors.RESET}", end="", flush=True)
    baseline_losses = []
    for item in encoded_data:
        sigma = _sample_timestep(1000, sampling, rng)
        noise = mx.random.normal(shape=item.clean_latents.shape)
        bl = compute_loss(model, item, sigma, noise, text_len, shift)
        mx.eval(bl)
        baseline_losses.append(bl.item())
    baseline_loss = sum(baseline_losses) / len(baseline_losses)
    loss_history.append(0, baseline_loss)
    loss_history.baseline = baseline_loss
    running_loss += baseline_loss
    loss_count += 1
    print(f"\r  {Colors.DIM}Baseline loss (step 0): {baseline_loss:.4f}{Colors.RESET}")

    # Baseline preview image
    if preview_freq > 0:
        from mlx_video.training.preview import generate_preview

        preview_path = generate_preview(model, config, encoded_data, 0, output_dir)
        if preview_path:
            print(f"  {Colors.GREEN}✓ Baseline preview: {preview_path}{Colors.RESET}")

    # Baseline plot
    if plot_freq > 0:
        plot_loss(loss_history, plot_path)

    for epoch in range(num_epochs):
        # Shuffle data each epoch
        indices = list(range(len(encoded_data)))
        rng.shuffle(indices)

        epoch_loss = 0.0
        epoch_steps = 0

        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=True,
        )

        for step in pbar:
            # Gather batch
            batch_indices = []
            for b in range(batch_size):
                idx = (step * batch_size + b) % len(encoded_data)
                batch_indices.append(indices[idx])

            items_batch = [encoded_data[i] for i in batch_indices]

            # Sample timesteps and noise
            sigmas = [_sample_timestep(1000, sampling, rng) for _ in range(batch_size)]
            noises = [
                mx.random.normal(shape=items_batch[i].clean_latents.shape)
                for i in range(batch_size)
            ]

            # Forward + backward
            loss, grads = loss_and_grad(model, items_batch, sigmas, noises)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_steps += 1
            running_loss += loss_val
            loss_count += 1
            global_step += 1
            loss_history.append(global_step, loss_val)

            # Update progress bar
            avg_loss = running_loss / loss_count
            pbar.set_postfix(loss=f"{loss_val:.4f}", avg=f"{avg_loss:.4f}")

        # Epoch summary
        avg_epoch_loss = epoch_loss / max(1, epoch_steps)
        if (epoch + 1) % log_freq == 0:
            elapsed = time.time() - t_start
            print(
                f"  {Colors.DIM}Epoch {epoch + 1}: "
                f"loss={avg_epoch_loss:.4f}, "
                f"elapsed={elapsed:.1f}s{Colors.RESET}"
            )

        # Checkpoint
        if save_freq > 0 and (epoch + 1) % save_freq == 0:
            ckpt_path = f"{output_dir}/lora_epoch_{epoch + 1}.safetensors"
            save_lora_weights(model, ckpt_path, config)
            print(f"  {Colors.GREEN}✓ Checkpoint saved: {ckpt_path}{Colors.RESET}")

        # Loss plot
        if plot_freq > 0 and (epoch + 1) % plot_freq == 0:
            plot_loss(loss_history, plot_path)

        # Preview image
        if preview_freq > 0 and (epoch + 1) % preview_freq == 0:
            from mlx_video.training.preview import generate_preview

            preview_path = generate_preview(
                model, config, encoded_data, epoch + 1, output_dir
            )
            if preview_path:
                print(f"  {Colors.GREEN}✓ Preview saved: {preview_path}{Colors.RESET}")

    # Final save
    final_path = f"{output_dir}/lora_final.safetensors"
    save_lora_weights(model, final_path, config)
    plot_loss(loss_history, plot_path)

    total_time = time.time() - t_start
    avg_loss = running_loss / max(1, loss_count)
    print(f"\n{Colors.GREEN}{'='*60}")
    print(f"  Training complete!")
    print(f"{'='*60}{Colors.RESET}")
    print(f"{Colors.DIM}  Final avg loss: {avg_loss:.4f}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  LoRA saved to: {final_path}")
    print(f"  Loss plot: {plot_path}{Colors.RESET}")
