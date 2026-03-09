"""Core training loop for Wan2.2 LoRA training.

Implements flow matching loss with timestep sampling bias control,
MLX value_and_grad for LoRA parameters, and the training epoch loop.
"""

import random
import time
from functools import partial

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
    sigma_min: float = 0.0,
    sigma_max: float = 1.0,
) -> float:
    """Sample a timestep (as sigma in [sigma_min, sigma_max]) with optional bias.

    For flow matching, sigma represents the noise level:
      - sigma=1.0: pure noise (high noise)
      - sigma=0.0: clean data (low noise)

    Args:
        num_train_timesteps: Number of training timesteps (e.g., 1000).
        sampling: One of 'balanced', 'low_bias', 'high_bias'.
        rng: Random number generator.
        sigma_min: Minimum sigma value (inclusive).
        sigma_max: Maximum sigma value (inclusive).

    Returns:
        Sigma value in (sigma_min, sigma_max).
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

    # Scale to [sigma_min, sigma_max] range
    t = sigma_min + t * (sigma_max - sigma_min)

    # Clamp to avoid exact boundaries
    t = max(sigma_min + 1e-5, min(sigma_max - 1e-5, t))
    return t


def _apply_shift(sigma: float, shift: float) -> float:
    """Apply noise schedule shift (matching Wan2.2 scheduler)."""
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def _min_snr_weight_scalar(sigma_shifted: float, gamma: float) -> float:
    """Compute min-SNR weight for a single sigma value (for non-compiled path).

    SNR(σ) = (1-σ)² / σ² for flow matching.
    Weight = min(SNR, γ) / SNR — clamps contribution from low-noise (easy) samples.
    """
    if sigma_shifted <= 1e-6:
        return 1.0  # avoid division by zero at σ≈0
    snr = ((1.0 - sigma_shifted) ** 2) / (sigma_shifted ** 2)
    return min(snr, gamma) / max(snr, 1e-8)


def _min_snr_weight_mx(sigmas_shifted: mx.array, gamma: float) -> mx.array:
    """Compute min-SNR weights for a batch of sigmas (for compiled path).

    Returns per-sample weights as mx.array of shape [batch_size].
    """
    snr = ((1.0 - sigmas_shifted) ** 2) / (sigmas_shifted ** 2 + 1e-8)
    return mx.minimum(snr, gamma) / (snr + 1e-8)


def compute_loss(
    model: nn.Module,
    item: EncodedItem,
    sigma: float,
    noise: mx.array,
    text_len: int,
    shift: float,
    loss_weighting: str = "uniform",
    min_snr_gamma: float = 5.0,
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
        loss_weighting: "uniform" or "min_snr".
        min_snr_gamma: SNR clamping value for min-SNR weighting.

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

    # MSE loss with optional min-SNR weighting
    error = (predicted_velocity - target_velocity).square()
    mse = error.mean()
    if loss_weighting == "min_snr":
        weight = _min_snr_weight_scalar(sigma_shifted, min_snr_gamma)
        return weight * mse
    return mse


def compute_batch_loss(
    model: nn.Module,
    items: list[EncodedItem],
    sigmas: list[float],
    noises: list[mx.array],
    text_len: int,
    shift: float,
) -> mx.array:
    """Compute flow matching loss for a batch with a single forward pass.

    Batches all items into one model call for better GPU utilization.
    """
    batch_size = len(items)

    # Build batched inputs
    noisy_list = []
    timesteps = []
    contexts = []
    targets = []

    for item, sigma, noise in zip(items, sigmas, noises):
        clean = item.clean_latents
        sigma_shifted = _apply_shift(sigma, shift)
        noisy = (1.0 - sigma_shifted) * clean + sigma_shifted * noise
        noisy_list.append(noisy)
        timesteps.append(sigma_shifted * 1000.0)
        contexts.append(item.text_embedding)
        targets.append(noise - clean)

    t_batch = mx.array(timesteps)

    # Compute seq_len from first item (all same resolution)
    _, _, h_lat, w_lat = items[0].clean_latents.shape
    patch_size = model.config.patch_size
    f_grid = 1 // patch_size[0]
    h_grid = h_lat // patch_size[1]
    w_grid = w_lat // patch_size[2]
    seq_len = f_grid * h_grid * w_grid

    # Single batched forward pass
    predicted_list = model(
        noisy_list,
        t=t_batch,
        context=contexts,
        seq_len=seq_len,
    )

    # Compute per-sample MSE, then average
    losses = []
    for pred, target in zip(predicted_list, targets):
        error = (pred - target).square()
        losses.append(error.mean())
    return mx.mean(mx.stack(losses))


def train(
    model: nn.Module,
    encoded_data: list[EncodedItem],
    config: TrainingConfig,
    sigma_min: float = 0.0,
    sigma_max: float = 1.0,
    expert_label: str = "",
    output_suffix: str = "",
    resume_state: dict | None = None,
) -> None:
    """Run the full training loop.

    Args:
        model: WanModel with LoRA layers injected and base weights frozen.
        encoded_data: Pre-encoded training samples.
        config: Training configuration.
        sigma_min: Minimum sigma for timestep sampling (expert boundary).
        sigma_max: Maximum sigma for timestep sampling (expert boundary).
        expert_label: Label for display (e.g., "high noise", "low noise").
        output_suffix: Suffix for output files (e.g., "_high_noise", "_low_noise").
        resume_state: If resuming, dict with 'epoch', 'global_step', 'loss_history'.
    """
    from mlx_video.training.plotting import LossHistory, plot_loss
    from mlx_video.training.save import save_checkpoint, save_lora_weights

    num_epochs = config.training.num_epochs
    batch_size = config.training.batch_size
    lr = config.training.learning_rate
    sampling = config.training.timestep_sampling
    log_freq = config.monitoring.log_frequency
    plot_freq = config.monitoring.plot_frequency
    preview_freq = config.monitoring.generate_image_frequency
    preview_steps = config.monitoring.preview_steps
    preview_guide_scale = config.monitoring.preview_guide_scale
    save_freq = config.checkpoint.save_frequency
    output_dir = config.checkpoint.output_dir
    shift = config.training.shift or getattr(model.config, "sample_shift", 12.0)
    text_len = model.config.text_len
    loss_weighting = config.training.loss_weighting
    min_snr_gamma = config.training.min_snr_gamma

    # Pre-compute total steps for LR schedule
    steps_per_epoch = max(1, len(encoded_data) // batch_size)
    total_steps = num_epochs * steps_per_epoch

    # Setup optimizer with LR schedule
    optimizer_cls = {"adam": optim.Adam, "adamw": optim.AdamW}.get(
        config.training.optimizer.lower(), optim.AdamW
    )
    if config.training.lr_schedule == "cosine" and total_steps > 1:
        warmup_steps = max(1, int(total_steps * config.training.lr_warmup_ratio))
        warmup = optim.linear_schedule(init=1e-7, end=lr, steps=warmup_steps)
        decay = optim.cosine_decay(init=lr, decay_steps=total_steps - warmup_steps)
        lr_schedule = optim.join_schedules([warmup, decay], [warmup_steps])
        optimizer = optimizer_cls(learning_rate=lr_schedule)
        schedule_desc = f"cosine (warmup={warmup_steps})"
    else:
        optimizer = optimizer_cls(learning_rate=lr)
        schedule_desc = "constant"

    rng = random.Random(config.seed)
    mx.random.seed(config.seed)

    # Pre-compute seq_len (constant for all items — same resolution)
    _, _, h_lat, w_lat = encoded_data[0].clean_latents.shape
    patch_size = model.config.patch_size
    seq_len = (1 // patch_size[0]) * (h_lat // patch_size[1]) * (w_lat // patch_size[2])

    # Compile-friendly loss function with raw array inputs (no Python objects)
    def loss_fn(model, cleans, texts, sigmas, noises):
        sigmas_shifted = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
        noisy_list = []
        targets = []
        for i in range(batch_size):
            s = sigmas_shifted[i]
            noisy_list.append((1.0 - s) * cleans[i] + s * noises[i])
            targets.append(noises[i] - cleans[i])
        t_batch = sigmas_shifted * 1000.0
        predicted_list = model(noisy_list, t=t_batch, context=texts, seq_len=seq_len)
        losses = []
        for pred, target in zip(predicted_list, targets):
            losses.append((pred - target).square().mean())
        per_sample = mx.stack(losses)
        if loss_weighting == "min_snr":
            weights = _min_snr_weight_mx(sigmas_shifted, min_snr_gamma)
            per_sample = per_sample * weights
        return mx.mean(per_sample)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Compiled training step: fuses ops across transformer blocks,
    # grads released inside (reduces peak memory)
    state = [model, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(cleans, texts, sigmas, noises):
        loss, grads = loss_and_grad(model, cleans, texts, sigmas, noises)
        optimizer.update(model, grads)
        return loss

    # Training loop
    global_step = 0
    start_epoch = 0
    running_loss = 0.0
    loss_count = 0
    loss_history = LossHistory()
    title_suffix = f" ({expert_label})" if expert_label else ""
    plot_path = f"{output_dir}/loss_plot{output_suffix}.png"

    # Restore state from checkpoint if resuming
    if resume_state:
        start_epoch = resume_state.get("epoch", 0)
        global_step = resume_state.get("global_step", 0)
        for step, loss_val in resume_state.get("loss_history", []):
            loss_history.append(step, loss_val)
        if loss_history.losses:
            loss_history.baseline = loss_history.losses[0]
        print(f"{Colors.DIM}  Resuming from epoch {start_epoch}, step {global_step}{Colors.RESET}")

    print(f"\n{Colors.CYAN}{'='*60}")
    print(f"  Wan2.2 LoRA Training{title_suffix}")
    print(f"{'='*60}{Colors.RESET}")
    print(f"{Colors.DIM}  Training samples: {len(encoded_data)}")
    print(f"  Epochs: {num_epochs}, Steps/epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Batch size: {batch_size}, LR: {lr}")
    print(f"  LR schedule: {schedule_desc}")
    print(f"  Loss weighting: {loss_weighting}" + (f" (γ={min_snr_gamma})" if loss_weighting == "min_snr" else ""))
    print(f"  Timestep sampling: {sampling}")
    if sigma_min > 0.0 or sigma_max < 1.0:
        print(f"  Sigma range: [{sigma_min:.3f}, {sigma_max:.3f}]")
    print(f"  Shift: {shift}")
    print(f"  Optimizer: {config.training.optimizer}")
    if start_epoch > 0:
        print(f"  Resumed from epoch: {start_epoch}")
    print(f"{Colors.RESET}")

    t_start = time.time()

    # --- Baseline loss at step 0 (skip if resuming) ---
    if not resume_state:
        print(f"  {Colors.DIM}Computing baseline loss...{Colors.RESET}", end="", flush=True)
        baseline_losses = []
        for item in encoded_data:
            sigma = _sample_timestep(1000, sampling, rng, sigma_min, sigma_max)
            noise = mx.random.normal(shape=item.clean_latents.shape)
            bl = compute_loss(model, item, sigma, noise, text_len, shift,
                              loss_weighting=loss_weighting, min_snr_gamma=min_snr_gamma)
            baseline_losses.append(bl)
        # Single eval for all baseline losses (avoids N sync barriers)
        mx.eval(*baseline_losses)
        baseline_loss = sum(bl.item() for bl in baseline_losses) / len(baseline_losses)
        loss_history.append(0, baseline_loss)
        loss_history.baseline = baseline_loss
        running_loss += baseline_loss
        loss_count += 1
        print(f"\r  {Colors.DIM}Baseline loss (step 0): {baseline_loss:.4f}{Colors.RESET}")

        # Baseline preview image
        if preview_freq > 0:
            from mlx_video.training.preview import generate_preview

            preview_path = generate_preview(
                model, config, encoded_data, 0, output_dir,
                steps=preview_steps, guide_scale=preview_guide_scale,
                seed=config.seed + 9999,
            )
            if preview_path:
                print(f"  {Colors.GREEN}✓ Baseline preview: {preview_path}{Colors.RESET}")

        # Baseline plot
        if plot_freq > 0:
            plot_loss(loss_history, plot_path)

    for epoch in range(start_epoch, num_epochs):
        # Shuffle data each epoch
        indices = list(range(len(encoded_data)))
        rng.shuffle(indices)

        epoch_loss = 0.0
        epoch_steps = 0
        _prev_loss = None  # For deferred async loss reading

        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=True,
        )

        for step in pbar:
            # Record previous step's deferred loss (ready by now — GPU moved on)
            if _prev_loss is not None:
                loss_val = _prev_loss.item()
                epoch_loss += loss_val
                running_loss += loss_val
                loss_count += 1
                avg_loss = running_loss / loss_count
                pbar.set_postfix(loss=f"{loss_val:.4f}", avg=f"{avg_loss:.4f}")

            # Gather batch (overlaps with GPU async eval from previous step)
            batch_indices = []
            for b in range(batch_size):
                idx = (step * batch_size + b) % len(encoded_data)
                batch_indices.append(indices[idx])

            items_batch = [encoded_data[i] for i in batch_indices]

            # Extract raw arrays for compiled step
            cleans = [item.clean_latents for item in items_batch]
            texts = [item.text_embedding for item in items_batch]

            # Sample timesteps and noise
            sigmas = mx.array([
                _sample_timestep(1000, sampling, rng, sigma_min, sigma_max)
                for _ in range(batch_size)
            ])
            noises = [
                mx.random.normal(shape=items_batch[i].clean_latents.shape)
                for i in range(batch_size)
            ]

            # Forward + backward + optimizer (compiled + fused)
            loss = train_step(cleans, texts, sigmas, noises)
            # Async eval: returns immediately, GPU works while Python prepares next batch
            mx.async_eval(loss, state)
            _prev_loss = loss
            epoch_steps += 1
            global_step += 1

        # Flush last step's deferred loss
        if _prev_loss is not None:
            loss_val = _prev_loss.item()
            epoch_loss += loss_val
            running_loss += loss_val
            loss_count += 1
            _prev_loss = None

        # Record epoch average loss (cleaner than per-step for plotting)
        avg_epoch_loss = epoch_loss / max(1, epoch_steps)
        loss_history.append(epoch + 1, avg_epoch_loss)
        if (epoch + 1) % log_freq == 0:
            elapsed = time.time() - t_start
            print(
                f"  {Colors.DIM}Epoch {epoch + 1}: "
                f"loss={avg_epoch_loss:.4f}, "
                f"elapsed={elapsed:.1f}s{Colors.RESET}"
            )

        # Checkpoint (LoRA weights + resume zip)
        if save_freq > 0 and (epoch + 1) % save_freq == 0:
            ckpt_path = f"{output_dir}/lora{output_suffix}_epoch_{epoch + 1}.safetensors"
            save_lora_weights(model, ckpt_path, config)
            # Save resume checkpoint
            ckpt_zip = f"{output_dir}/checkpoint{output_suffix}_epoch_{epoch + 1}.zip"
            save_checkpoint(
                model,
                optimizer,
                config,
                epoch=epoch + 1,
                global_step=global_step,
                loss_history_data=list(zip(loss_history.steps, loss_history.losses)),
                output_path=ckpt_zip,
                expert_label=expert_label,
            )
            print(f"  {Colors.GREEN}✓ Checkpoint saved: {ckpt_path}{Colors.RESET}")

        # Loss plot
        if plot_freq > 0 and (epoch + 1) % plot_freq == 0:
            plot_loss(loss_history, plot_path)

        # Preview image
        if preview_freq > 0 and (epoch + 1) % preview_freq == 0:
            from mlx_video.training.preview import generate_preview

            preview_path = generate_preview(
                model, config, encoded_data, epoch + 1, output_dir,
                steps=preview_steps, guide_scale=preview_guide_scale,
                seed=config.seed + 9999,
            )
            if preview_path:
                print(f"  {Colors.GREEN}✓ Preview saved: {preview_path}{Colors.RESET}")

    # Final save
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


def train_simultaneous(
    high_model: nn.Module,
    low_model: nn.Module,
    encoded_data: list[EncodedItem],
    config: TrainingConfig,
    boundary: float = 0.875,
) -> None:
    """Train both experts simultaneously, routing each step to an expert.

    Supports two routing strategies:
      - alternating: Switch expert every N steps, sample σ from that expert's range.
        Each expert gets ~50% of steps. Matches AI Toolkit's switch_boundary_every.
      - proportional: Sample σ from [0,1], route by boundary. High gets ~12.5%,
        Low gets ~87.5%. Matches the natural sigma distribution.

    Args:
        high_model: High-noise expert with LoRA injected.
        low_model: Low-noise expert with LoRA injected.
        encoded_data: Pre-encoded training samples.
        config: Training configuration.
        boundary: Sigma boundary between experts (default 0.875).
    """
    from mlx_video.training.plotting import LossHistory, plot_loss
    from mlx_video.training.save import save_dual_checkpoint, save_lora_weights

    num_epochs = config.training.num_epochs
    batch_size = config.training.batch_size
    lr = config.training.learning_rate
    sampling = config.training.timestep_sampling
    routing = config.training.expert_routing
    switch_every = config.training.switch_every
    high_ratio = config.training.high_ratio
    log_freq = config.monitoring.log_frequency
    plot_freq = config.monitoring.plot_frequency
    preview_freq = config.monitoring.generate_image_frequency
    preview_steps = config.monitoring.preview_steps
    preview_guide_scale = config.monitoring.preview_guide_scale
    save_freq = config.checkpoint.save_frequency
    output_dir = config.checkpoint.output_dir
    shift = config.training.shift or getattr(high_model.config, "sample_shift", 12.0)
    text_len = high_model.config.text_len
    loss_weighting = config.training.loss_weighting
    min_snr_gamma = config.training.min_snr_gamma

    # Pre-compute total steps for LR schedule
    steps_per_epoch = max(1, len(encoded_data) // batch_size)
    total_steps = num_epochs * steps_per_epoch

    # Separate optimizers for each expert (with LR schedule)
    optimizer_cls = {"adam": optim.Adam, "adamw": optim.AdamW}.get(
        config.training.optimizer.lower(), optim.AdamW
    )
    if config.training.lr_schedule == "cosine" and total_steps > 1:
        warmup_steps = max(1, int(total_steps * config.training.lr_warmup_ratio))
        warmup = optim.linear_schedule(init=1e-7, end=lr, steps=warmup_steps)
        decay = optim.cosine_decay(init=lr, decay_steps=total_steps - warmup_steps)
        lr_schedule = optim.join_schedules([warmup, decay], [warmup_steps])
        high_optimizer = optimizer_cls(learning_rate=lr_schedule)
        low_optimizer = optimizer_cls(learning_rate=lr_schedule)
        schedule_desc = f"cosine (warmup={warmup_steps})"
    else:
        high_optimizer = optimizer_cls(learning_rate=lr)
        low_optimizer = optimizer_cls(learning_rate=lr)
        schedule_desc = "constant"

    rng = random.Random(config.seed)
    mx.random.seed(config.seed)

    # Pre-compute seq_len (constant for all items — same resolution)
    _, _, h_lat, w_lat = encoded_data[0].clean_latents.shape
    patch_size = high_model.config.patch_size
    seq_len = (1 // patch_size[0]) * (h_lat // patch_size[1]) * (w_lat // patch_size[2])

    # Compile-friendly loss functions with raw array inputs
    def high_loss_fn(model, cleans, texts, sigmas, noises):
        sigmas_shifted = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
        noisy_list = []
        targets = []
        for i in range(batch_size):
            s = sigmas_shifted[i]
            noisy_list.append((1.0 - s) * cleans[i] + s * noises[i])
            targets.append(noises[i] - cleans[i])
        t_batch = sigmas_shifted * 1000.0
        predicted_list = model(noisy_list, t=t_batch, context=texts, seq_len=seq_len)
        losses = []
        for pred, target in zip(predicted_list, targets):
            losses.append((pred - target).square().mean())
        per_sample = mx.stack(losses)
        if loss_weighting == "min_snr":
            weights = _min_snr_weight_mx(sigmas_shifted, min_snr_gamma)
            per_sample = per_sample * weights
        return mx.mean(per_sample)

    def low_loss_fn(model, cleans, texts, sigmas, noises):
        sigmas_shifted = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
        noisy_list = []
        targets = []
        for i in range(batch_size):
            s = sigmas_shifted[i]
            noisy_list.append((1.0 - s) * cleans[i] + s * noises[i])
            targets.append(noises[i] - cleans[i])
        t_batch = sigmas_shifted * 1000.0
        predicted_list = model(noisy_list, t=t_batch, context=texts, seq_len=seq_len)
        losses = []
        for pred, target in zip(predicted_list, targets):
            losses.append((pred - target).square().mean())
        per_sample = mx.stack(losses)
        if loss_weighting == "min_snr":
            weights = _min_snr_weight_mx(sigmas_shifted, min_snr_gamma)
            per_sample = per_sample * weights
        return mx.mean(per_sample)

    high_loss_and_grad = nn.value_and_grad(high_model, high_loss_fn)
    low_loss_and_grad = nn.value_and_grad(low_model, low_loss_fn)

    # Compiled training steps: separate state per expert
    high_state = [high_model, high_optimizer.state]
    low_state = [low_model, low_optimizer.state]

    @partial(mx.compile, inputs=high_state, outputs=high_state)
    def high_step(cleans, texts, sigmas, noises):
        loss, grads = high_loss_and_grad(high_model, cleans, texts, sigmas, noises)
        high_optimizer.update(high_model, grads)
        return loss

    @partial(mx.compile, inputs=low_state, outputs=low_state)
    def low_step(cleans, texts, sigmas, noises):
        loss, grads = low_loss_and_grad(low_model, cleans, texts, sigmas, noises)
        low_optimizer.update(low_model, grads)
        return loss

    # Training loop
    global_step = 0
    running_loss = 0.0
    loss_count = 0
    high_steps = 0
    low_steps = 0
    loss_history = LossHistory()
    plot_path = f"{output_dir}/loss_plot.png"

    print(f"\n{Colors.CYAN}{'='*60}")
    print(f"  Wan2.2 LoRA Training (simultaneous dual-expert)")
    print(f"{'='*60}{Colors.RESET}")
    print(f"{Colors.DIM}  Training samples: {len(encoded_data)}")
    print(f"  Epochs: {num_epochs}, Steps/epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Batch size: {batch_size}, LR: {lr}")
    print(f"  LR schedule: {schedule_desc}")
    print(f"  Loss weighting: {loss_weighting}" + (f" (γ={min_snr_gamma})" if loss_weighting == "min_snr" else ""))
    print(f"  Timestep sampling: {sampling}")
    print(f"  Expert boundary: σ={boundary:.3f} (alternating each step)")
    print(f"  Shift: {shift}")
    print(f"  Optimizer: {config.training.optimizer}")
    if routing == "alternating":
        routing_desc = f"alternating (switch every {switch_every})"
    else:
        effective = high_ratio if high_ratio is not None else (1.0 - boundary)
        routing_desc = f"proportional (H ratio={effective:.1%})"
    print(f"  Expert routing: {routing_desc}")
    print(f"{Colors.RESET}")

    t_start = time.time()

    # --- Baseline preview (before any training, uses low model for character detail) ---
    if preview_freq > 0:
        from mlx_video.training.preview import generate_preview

        preview_path = generate_preview(
            low_model, config, encoded_data, 0, output_dir,
            steps=preview_steps, guide_scale=preview_guide_scale,
            seed=config.seed + 9999,
        )
        if preview_path:
            print(f"  {Colors.GREEN}✓ Baseline preview: {preview_path}{Colors.RESET}")

    # Track which expert is active (for alternating mode)
    current_expert = "high"  # start with high
    steps_on_current = 0

    for epoch in range(num_epochs):
        indices = list(range(len(encoded_data)))
        rng.shuffle(indices)

        epoch_loss = 0.0
        epoch_steps = 0
        epoch_h_loss = 0.0
        epoch_h_steps = 0
        epoch_l_loss = 0.0
        epoch_l_steps = 0
        _prev_loss = None  # For deferred async loss reading
        _prev_expert = None

        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=True,
        )

        for step in pbar:
            # Record previous step's deferred loss (ready by now — GPU moved on)
            if _prev_loss is not None:
                loss_val = _prev_loss.item()
                if not (loss_val != loss_val):  # skip NaN (NaN != NaN is True)
                    epoch_loss += loss_val
                    if _prev_expert == "H":
                        epoch_h_loss += loss_val
                        epoch_h_steps += 1
                    else:
                        epoch_l_loss += loss_val
                        epoch_l_steps += 1
                    running_loss += loss_val
                    loss_count += 1
                avg_loss = running_loss / max(1, loss_count)
                pbar.set_postfix(
                    loss=f"{loss_val:.4f}", avg=f"{avg_loss:.4f}", expert=_prev_expert,
                    H=high_steps, L=low_steps,
                )

            # Gather batch (overlaps with GPU async eval from previous step)
            batch_indices = []
            for b in range(batch_size):
                idx = (step * batch_size + b) % len(encoded_data)
                batch_indices.append(indices[idx])

            items_batch = [encoded_data[i] for i in batch_indices]

            # Extract raw arrays for compiled step
            cleans = [item.clean_latents for item in items_batch]
            texts = [item.text_embedding for item in items_batch]

            noises = [
                mx.random.normal(shape=items_batch[i].clean_latents.shape)
                for i in range(batch_size)
            ]

            # Determine which expert trains this step
            if routing == "alternating":
                use_high = current_expert == "high"
                steps_on_current += 1
                if steps_on_current >= switch_every:
                    current_expert = "low" if current_expert == "high" else "high"
                    steps_on_current = 0
            else:
                # Proportional: use high_ratio if set, else derive from boundary
                effective_ratio = high_ratio if high_ratio is not None else (1.0 - boundary)
                use_high = rng.random() < effective_ratio

            if use_high:
                sigmas = mx.array([
                    _sample_timestep(1000, sampling, rng, boundary, 1.0)
                    for _ in range(batch_size)
                ])
                loss = high_step(cleans, texts, sigmas, noises)
                mx.async_eval(loss, high_state)
                high_steps += 1
                expert_tag = "H"
            else:
                sigmas = mx.array([
                    _sample_timestep(1000, sampling, rng, 0.0, boundary)
                    for _ in range(batch_size)
                ])
                loss = low_step(cleans, texts, sigmas, noises)
                mx.async_eval(loss, low_state)
                low_steps += 1
                expert_tag = "L"

            _prev_loss = loss
            _prev_expert = expert_tag
            epoch_steps += 1
            global_step += 1

        # Flush last step's deferred loss
        if _prev_loss is not None:
            loss_val = _prev_loss.item()
            if not (loss_val != loss_val):  # skip NaN
                epoch_loss += loss_val
                if _prev_expert == "H":
                    epoch_h_loss += loss_val
                    epoch_h_steps += 1
                else:
                    epoch_l_loss += loss_val
                    epoch_l_steps += 1
                running_loss += loss_val
                loss_count += 1
            _prev_loss = None

        # Record epoch averages (combined + per-expert)
        avg_epoch_loss = epoch_loss / max(1, epoch_steps)
        avg_h = epoch_h_loss / max(1, epoch_h_steps) if epoch_h_steps > 0 else None
        avg_l = epoch_l_loss / max(1, epoch_l_steps) if epoch_l_steps > 0 else None

        # Record combined with expert breakdown for plotting
        loss_history.append(epoch + 1, avg_epoch_loss)
        if avg_h is not None:
            loss_history.append(epoch + 1, avg_h, expert="H")
        if avg_l is not None:
            loss_history.append(epoch + 1, avg_l, expert="L")

        if (epoch + 1) % log_freq == 0:
            elapsed = time.time() - t_start
            parts = [f"Epoch {epoch + 1}: loss={avg_epoch_loss:.4f}"]
            if avg_h is not None:
                parts.append(f"H={avg_h:.4f}")
            if avg_l is not None:
                parts.append(f"L={avg_l:.4f}")
            parts.append(f"H_steps={high_steps}, L_steps={low_steps}")
            parts.append(f"elapsed={elapsed:.1f}s")
            print(f"  {Colors.DIM}{', '.join(parts)}{Colors.RESET}")

        # Checkpoint
        if save_freq > 0 and (epoch + 1) % save_freq == 0:
            high_path = f"{output_dir}/lora_high_noise_epoch_{epoch + 1}.safetensors"
            low_path = f"{output_dir}/lora_low_noise_epoch_{epoch + 1}.safetensors"
            save_lora_weights(high_model, high_path, config)
            save_lora_weights(low_model, low_path, config)
            # Save resume checkpoint with optimizer state
            ckpt_zip = f"{output_dir}/checkpoint_epoch_{epoch + 1}.zip"
            save_dual_checkpoint(
                high_model,
                low_model,
                high_optimizer,
                low_optimizer,
                config,
                epoch=epoch + 1,
                global_step=global_step,
                loss_history_data=list(zip(loss_history.steps, loss_history.losses)),
                output_path=ckpt_zip,
            )
            print(f"  {Colors.GREEN}✓ Checkpoint: {high_path}, {low_path}{Colors.RESET}")

        # Loss plot
        if plot_freq > 0 and (epoch + 1) % plot_freq == 0:
            plot_loss(loss_history, plot_path)

        # Preview (uses low noise model by default for character detail)
        if preview_freq > 0 and (epoch + 1) % preview_freq == 0:
            from mlx_video.training.preview import generate_preview

            preview_path = generate_preview(
                low_model, config, encoded_data, epoch + 1, output_dir,
                steps=preview_steps, guide_scale=preview_guide_scale,
                seed=config.seed + 9999,
            )
            if preview_path:
                print(f"  {Colors.GREEN}✓ Preview saved: {preview_path}{Colors.RESET}")

    # Final save
    high_final = f"{output_dir}/lora_high_noise_final.safetensors"
    low_final = f"{output_dir}/lora_low_noise_final.safetensors"
    save_lora_weights(high_model, high_final, config)
    save_lora_weights(low_model, low_final, config)
    plot_loss(loss_history, plot_path)

    total_time = time.time() - t_start
    avg_loss = running_loss / max(1, loss_count)
    print(f"\n{Colors.GREEN}{'='*60}")
    print(f"  Training complete! (simultaneous dual-expert)")
    print(f"{'='*60}{Colors.RESET}")
    print(f"{Colors.DIM}  Final avg loss: {avg_loss:.4f}")
    print(f"  High noise steps: {high_steps}, Low noise steps: {low_steps}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  LoRA (high): {high_final}")
    print(f"  LoRA (low): {low_final}")
    print(f"  Loss plot: {plot_path}{Colors.RESET}")
