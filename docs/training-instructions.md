# Wan2.2 LoRA Training — Practical Guide

This guide covers practical tips for training character and style LoRAs with mlx-video.

---

## Block Range Selection

Each Wan2.2 14B model has **40 transformer blocks** (numbered 0–39). The `blocks` config controls which blocks get LoRA adapters. Not all blocks contribute equally:

| Blocks | Role |
|--------|------|
| 0–10 (early) | Low-level features — edges, textures, colors |
| 10–30 (middle) | Semantic content — identity, objects, composition |
| 30–40 (late) | Fine output detail — sharpness, small refinements |

### Recommendations by dataset size

| Dataset Size | Suggested Range | Trainable Params (rank 16) | Notes |
|-------------|----------------|---------------------------|-------|
| 5–10 images | `"blocks": {"start": 12, "end": 28}` | ~30M | Conservative — less overfitting risk |
| 10–20 images | `"blocks": {"start": 8, "end": 32}` | ~45M | Good balance for character LoRAs |
| 50+ images | `"blocks": {"start": 0, "end": 40}` | ~75M | Full range, enough data to support it |

### Why not always use all blocks?

The concern is **overfitting**, not memory. With rank 16 across all 40 blocks you get ~75M trainable parameters learning from perhaps 10 images. The middle layers carry the most identity and semantic information, so restricting to those layers gives the best signal-to-noise ratio for small datasets.

Signs of overfitting during training:
- Loss drops to near zero very quickly
- Preview images look like exact copies of training images rather than generalized character likeness
- Generated videos show artifacts or frozen poses

If you see these, try narrowing the block range or reducing epochs.

### Memory impact

Block range has minimal impact on memory. The base model (~28GB) dominates regardless. Even full-range rank 16 only adds ~1.2GB (LoRA params + optimizer state + gradients):

```
Base model (bf16):     ~28 GB
LoRA + optimizer:       ~1.2 GB  (full range, rank 16)
Activations:            ~4 GB   (estimate, single frame)
Total:                 ~33 GB
```

128GB RAM is comfortable for any block range. Even 64GB should work for rank 16. Memory pressure only becomes a concern at higher ranks (64+) or with quantized base models on 32GB machines.

---

## Rank Selection

LoRA rank controls the capacity of the adaptation. Higher rank = more expressive but more parameters.

| Rank | Params (40 blocks) | Use Case |
|------|-------------------|----------|
| 4 | ~19M | Minimal adaptation, subtle style shifts |
| 8 | ~38M | Light character LoRAs with very few images |
| 16 | ~75M | Standard character LoRAs (recommended starting point) |
| 32 | ~150M | Complex characters, detailed style transfer |
| 64 | ~301M | Maximum capacity, needs larger datasets |

**Start with rank 16** and adjust based on results. If the LoRA isn't capturing enough detail, increase rank. If it's overfitting, decrease rank or narrow the block range.

---

## Timestep Sampling for Characters

Wan2.2 uses a dual-model architecture that splits denoising at σ ≈ 0.875:

- **High noise** (σ > 0.875): Controls global layout, composition, heavy motion
- **Low noise** (σ < 0.875): Controls fine details, textures, character identity

For LoRA training, the timestep sampling strategy determines which noise levels get more training signal:

| Strategy | Distribution | Best For |
|----------|-------------|----------|
| `balanced` | Uniform across all σ | General-purpose (recommended default) |
| `low_bias` | Beta(1,2), mean ≈ 0.33 | Character identity and fine detail |
| `high_bias` | Beta(2,1), mean ≈ 0.67 | Style, composition, motion patterns |

**For characters, use `balanced`** — this ensures the LoRA learns both identity (low noise) and how to integrate with motion/layout (high noise). Using `low_bias` can produce better likeness but may cause the character to look stiff or poorly integrated into scenes.

---

## Learning Rate Schedule

The trainer supports cosine annealing with linear warmup — a standard technique that prevents early destabilization and allows fine-tuning in later steps.

| Schedule | Config | Behavior |
|----------|--------|----------|
| **Cosine** (default) | `"lr_schedule": "cosine"` | Linear warmup → cosine decay to ~0 |
| **Constant** | `"lr_schedule": "constant"` | Fixed LR throughout training |

```json
"training": {
  "learning_rate": 1e-4,
  "lr_schedule": "cosine",
  "lr_warmup_ratio": 0.1
}
```

**Why it helps for character LoRAs:**
- **Warmup** (first 10% of steps): LoRA starts at zero, so early gradients are noisy. A low LR prevents destructive updates before the LoRA has learned useful features.
- **Cosine decay** (remaining 90%): Gradually reduces LR, allowing the model to fine-tune character details in later steps instead of oscillating.

`lr_warmup_ratio` controls what fraction of total steps are spent in warmup. Default is `0.1` (10%). For very short training runs (<100 steps), consider `0.05`.

---

## Loss Weighting (Min-SNR)

By default, the trainer uses **Min-SNR-γ weighting** from *"Efficient Diffusion Training via Min-SNR Weighting Strategy"* (Hang et al., 2023). This focuses the training signal on noise levels where character identity is most learnable.

| Weighting | Config | Behavior |
|-----------|--------|----------|
| **Min-SNR** (default) | `"loss_weighting": "min_snr"` | Downweights easy (low-noise) samples |
| **Uniform** | `"loss_weighting": "uniform"` | All noise levels weighted equally |

```json
"training": {
  "loss_weighting": "min_snr",
  "min_snr_gamma": 5.0
}
```

**How it works:** For flow matching, the signal-to-noise ratio at noise level σ is `SNR = (1-σ)²/σ²`. The weight for each sample is `min(SNR, γ) / SNR`:

| σ range | SNR | Weight (γ=5) | Interpretation |
|---------|-----|------|----------------|
| 0.0–0.1 | Very high (>80) | ~0.06 | Near-clean image, trivial to denoise → downweighted |
| 0.2–0.6 | Moderate (0.4–16) | 0.3–1.0 | Character features visible → full weight |
| 0.7–1.0 | Low (<0.2) | ~1.0 | Heavy noise, structure learning → full weight |

**For characters:** Min-SNR prevents the model from spending its gradient budget on trivially easy samples (near-clean images at low σ). Instead, it focuses on the intermediate noise levels where facial features, clothing details, and other identity markers are most distinguishable.

`min_snr_gamma` (default 5.0) controls the clamping threshold. Lower values (e.g., 1.0) apply stronger downweighting. Higher values (e.g., 10.0) make the weighting more uniform. 5.0 is the paper's recommended default.

---

## Expert Routing Strategies

When training both experts simultaneously, the `expert_routing` config controls how training steps are distributed between the high-noise and low-noise models.

### Why this matters

The 0.875 boundary is baked into Wan2.2's architecture — it's where the model switches experts during inference. But during training, we choose how to split the training budget. This matters because:

- The **high-noise range** [0.875, 1.0] is narrow (12.5% of σ space) but controls structure, composition, and motion
- The **low-noise range** [0.0, 0.875) is wide (87.5% of σ space) and controls fine details, textures, and character identity

### Available strategies

| Strategy | Config | H/L Split | Best for |
|----------|--------|-----------|----------|
| **Alternating** | `"alternating"` | 50/50 | Balanced motion + identity |
| **Proportional** | `"proportional"` | configurable | Identity focus with tunable H ratio |

**Alternating** (default): Switches between experts every `switch_every` steps. Each expert samples σ only from its own range. This matches AI Toolkit's `switch_boundary_every` approach and ensures the high-noise expert gets sufficient training for good motion and composition.

```json
"training": {
  "expert_routing": "alternating",
  "switch_every": 7
}
```

**Proportional**: Each step probabilistically selects H or L based on `high_ratio`. By default (`high_ratio: null`), the ratio is derived from the model's architectural boundary: `1 - 0.875 = 12.5%` H. You can override this to increase H training without going to a full 50/50 split.

```json
"training": {
  "expert_routing": "proportional",
  "high_ratio": 0.25
}
```

| `high_ratio` | H share | Use case |
|---|---|---|
| `null` (default) | ~12.5% | Natural boundary split, maximum L/identity focus |
| `0.25` | ~25% | Recommended for image-based character training |
| `0.4` | ~40% | More structural learning, closer to alternating |

### Guidance

- For **image-based character training**: Use `proportional` with `high_ratio: 0.25`. Images can't teach motion, so focus most budget on L (identity/details) with enough H (25%) for structural learning
- For **video-based training** (future): Use `alternating` to ensure both motion (H) and identity (L) get equal training
- `alternating` + `timestep_sampling: balanced` is the safest all-purpose combination

### Understanding `switch_every`

`switch_every` controls how many consecutive steps each expert trains before switching to the other. **It is purely a performance knob — it does not affect training quality.**

Why no quality impact:
- Each training step is independent (random σ, random noise, random data sample)
- Over a full epoch, each expert gets the same ~50% of total steps regardless of `switch_every`
- AdamW's exponential moving averages (β₁=0.9, β₂=0.999) are robust to the ordering of gradient updates

Why it affects speed: On Apple Silicon, both expert models (~28GB each) reside in unified memory, but the GPU has a working set / cache hierarchy. Switching experts every step (`switch_every: 1`) forces the system to page different model weights into the GPU's active cache each time, causing significant overhead. Increasing `switch_every` reduces these cache swaps.

```json
"training": {
  "expert_routing": "alternating",
  "switch_every": 7
}
```

**Recommendation**: Set `switch_every` to match your `steps_per_epoch` (= number of training samples ÷ batch size). This means each expert trains for one full epoch before switching, minimizing cache thrashing while maintaining balanced training.

### Why alternating shows higher average loss

With alternating routing, you'll notice ~2x higher average loss compared to proportional. This is expected and not a quality issue:

- The H expert predicts from nearly-pure noise (σ ∈ [0.875, 1.0]) → inherently higher MSE
- With alternating, H contributes ~50% of epoch averages (vs ~12.5% with proportional)
- The loss plot shows separate H (red) and L (blue) series so you can track each expert's learning curve independently

---

## Example Configs

### Character LoRA (10 images, 128GB Mac)

```json
{
  "model_dir": "/path/to/wan22_mlx",
  "data": "./my_character/",
  "trigger_word": "ohwx",
  "resolution": 512,
  "training": {
    "num_epochs": 40,
    "learning_rate": 1e-4,
    "timestep_sampling": "balanced",
    "experts": "both",
    "expert_mode": "simultaneous"
  },
  "lora": {
    "rank": 16,
    "alpha": 16,
    "blocks": { "start": 8, "end": 32 }
  },
  "checkpoint": {
    "save_frequency": 10,
    "output_dir": "./output_character/"
  },
  "monitoring": {
    "plot_frequency": 5,
    "generate_image_frequency": 10,
    "preview_width": 512,
    "preview_height": 512
  }
}
```

### Style LoRA (50+ images, full range)

```json
{
  "model_dir": "/path/to/wan22_mlx",
  "data": "./style_images/",
  "resolution": 512,
  "training": {
    "num_epochs": 30,
    "learning_rate": 5e-5,
    "timestep_sampling": "balanced",
    "experts": "both",
    "expert_mode": "sequential"
  },
  "lora": {
    "rank": 32,
    "alpha": 32,
    "blocks": { "start": 0, "end": 40 }
  },
  "checkpoint": {
    "save_frequency": 10,
    "output_dir": "./output_style/"
  }
}
```

---

## Wan Model Variants

| Model | Layers | Default Block End |
|-------|--------|------------------|
| Wan2.2 T2V-14B | 40 | 40 |
| Wan2.2 I2V-14B | 40 | 40 |
| Wan2.1 T2V-14B | 40 | 40 |
| Wan2.2 TI2V-5B | 30 | 30 |
| Wan2.1 T2V-1.3B | 30 | 30 |

If you're training on a 5B or 1.3B model, set `"end": 30`. The trainer handles out-of-range blocks gracefully (skips them), but setting the correct end avoids confusion.

---

## Dual Expert Training

Wan2.2 uses two transformer models (experts) that handle different noise levels:

- **High noise expert** (σ ≥ 0.875): Structure, composition, and motion
- **Low noise expert** (σ < 0.875): Fine details, textures, and character identity

For the best results, **train both experts**. The trainer supports two modes:

### Simultaneous Mode (default, recommended for 128GB+)

Both models are loaded into memory at once. Each training step samples a random sigma and routes to the correct expert based on the boundary (0.875). This matches the AI Toolkit approach (`switch_boundary_every: 1`).

```json
"training": {
  "experts": "both",
  "expert_mode": "simultaneous"
}
```

**Memory**: ~65GB (both models + LoRA + optimizer + activations)

### Sequential Mode (for 64GB or constrained systems)

Trains one expert at a time. First the high-noise expert, then the low-noise expert. Each is loaded, trained, saved, and unloaded before the next.

```json
"training": {
  "experts": "both",
  "expert_mode": "sequential"
}
```

**Memory**: ~33GB peak (only one model loaded at a time)

### Single Expert Training

If you only want to train one expert (e.g., for experimentation):

```json
"training": {
  "experts": "low"
}
```

Options: `"both"` (default), `"low"`, `"high"`

### Output Files

Dual expert training produces two LoRA files:
- `lora_high_noise_final.safetensors` — for the high noise model
- `lora_low_noise_final.safetensors` — for the low noise model

Use them at inference time:
```bash
python -m mlx_video.generate_wan \
  --lora-high ./output/lora_high_noise_final.safetensors \
  --lora-low ./output/lora_low_noise_final.safetensors \
  --prompt "A video of ohwx walking through a park"
```

---

## Resuming Training

Training saves checkpoint zip files at `save_frequency` intervals. To resume:

```bash
python -m mlx_video.train_wan --config train.json --resume ./output/checkpoint_low_noise_epoch_25.zip
```

The checkpoint contains:
- LoRA weights (current training state)
- Optimizer state (Adam momentum/variance)
- Training state (epoch, step, loss history)
- Config snapshot

Training resumes from the saved epoch with the optimizer state intact, so the learning rate warmup and momentum are preserved.

---

## Base LoRA (Lightning-Compatible Training)

To train a character LoRA that works well with [Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning) or other existing LoRAs, use `base_loras` to merge them into the base weights before training.

### Why train with a base LoRA?

If your final inference pipeline applies Lightning for speed, training *without* it means your LoRA learned velocity targets relative to the base model. But at inference time, Lightning changes the velocity field — your LoRA corrections may not align, causing artifacts or reduced quality.

By merging Lightning into the weights during training, your LoRA learns the residual on top of Lightning's velocity field. The result is a character LoRA that cooperates naturally with Lightning at inference time.

### Configuration

```json
{
  "model_dir": "/path/to/wan22_mlx",
  "data": "./my_character/",
  "trigger_word": "ohwx",
  "base_loras": [
    {
      "path": "/path/to/Wan2.2-Lightning.safetensors",
      "expert": "both",
      "strength": 1.0
    }
  ],
  "training": {
    "shift": 3.0,
    "num_epochs": 40,
    "learning_rate": 1e-4
  },
  "monitoring": {
    "preview_steps": 8,
    "preview_guide_scale": 1.0,
    "generate_image_frequency": 10
  }
}
```

### `base_loras` fields

| Field | Default | Description |
|-------|---------|-------------|
| `path` | (required) | Path to the `.safetensors` LoRA file |
| `expert` | `"both"` | Which expert(s) to apply it to: `"both"`, `"high"`, `"low"` |
| `strength` | `1.0` | Merge strength (1.0 = full, 0.5 = half) |

Multiple base LoRAs can be stacked — they're merged additively in order.

### `training.shift` override

The flow matching shift parameter controls the sigma schedule. Base Wan2.2 uses `shift=12.0`, but distilled models like Lightning use a lower shift (typically 3.0–5.0). When training on top of Lightning, override the shift to match:

```json
"training": {
  "shift": 3.0
}
```

If omitted, the model's default shift is used.

### Preview settings for Lightning

Lightning uses few-step inference without CFG, so previews should match:

```json
"monitoring": {
  "preview_steps": 8,
  "preview_guide_scale": 1.0
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `preview_steps` | `20` | Denoising steps for preview images |
| `preview_guide_scale` | `1.0` | CFG scale (1.0 disables CFG) |

### Inference with stacked LoRAs

At inference time, apply both the base LoRA and your trained LoRA:

```bash
python -m mlx_video.generate_wan \
  --lora /path/to/Wan2.2-Lightning.safetensors \
  --lora /path/to/my_character_lora.safetensors \
  --prompt "A video of ohwx walking" \
  --steps 4 --guide-scale 1.0
```

---

## QLoRA (4-Bit Quantized Training)

QLoRA dramatically reduces memory bandwidth by storing base model weights in 4-bit quantized format while keeping LoRA adapters in full bf16 precision. This is especially effective on Apple Silicon where training is memory-bandwidth-bound.

### How it works

1. **Pre-convert** the model to 4-bit quantized format (one-time step)
2. The trainer auto-detects quantization from the model's `config.json`
3. Base weights load as `QuantizedLinear` (~8GB instead of ~28GB)
4. LoRA matrices (`lora_A`, `lora_B`) remain in full bf16 precision
5. Forward: `output = quantized_matmul(x, W_q) + scale * (x @ A^T @ B^T)`
6. Saved LoRA adapters are pure bf16 — **fully portable** to full-precision inference

### Pre-converting the model

Convert from PyTorch and quantize in one step:
```bash
python -m mlx_video.convert_wan \
  --checkpoint-dir /path/to/Wan2.2-T2V-A14B \
  --output-dir /path/to/wan22_4bit \
  --quantize --bits 4 --group-size 64
```

Or quantize an already-converted MLX model (much faster — skips PyTorch conversion):
```bash
python -m mlx_video.convert_wan \
  --checkpoint-dir /path/to/wan22_mlx_bf16 \
  --output-dir /path/to/wan22_4bit \
  --quantize-only --bits 4 --group-size 64
```

This creates a quantized copy with `config.json` containing `{"quantization": {"bits": 4, "group_size": 64}}`. The original model is unchanged.

### Training with QLoRA

Point `model_dir` at the quantized model — no other config changes needed:

```json
{
  "model_dir": "/path/to/wan22_4bit",
  "data": "./my_character/",
  "trigger_word": "ohwx",
  "training": {
    "num_epochs": 40,
    "batch_size": 4,
    "learning_rate": 1e-4
  }
}
```

The trainer logs `Quantized: 4-bit` in the header when it detects a quantized model.

### Performance impact

| Metric | bf16 (28GB) | 4-bit QLoRA (8GB) |
|--------|-------------|-------------------|
| Weight memory | ~28 GB per model | ~8 GB per model |
| Memory bandwidth per step | ~56 GB (fwd+bwd) | ~16 GB (fwd+bwd) |
| Estimated speedup | 1× | 2–4× |
| LoRA quality | Baseline | Equivalent (bf16 adapters) |

### Batch size by memory tier

With QLoRA, you can increase batch size for better GPU utilization:

| RAM | Base Model | Recommended Batch Size |
|-----|-----------|----------------------|
| 32 GB | 4-bit single expert | 1–2 |
| 64 GB | 4-bit single expert | 4–8 |
| 64 GB | 4-bit dual expert | 2–4 |
| 128 GB | bf16 single expert | 2–4 |
| 128 GB | bf16 dual expert | 1–2 |
| 128 GB | 4-bit single expert | 8–16 |
| 128 GB | 4-bit dual expert | 4–8 |

Larger batch sizes improve arithmetic intensity (more compute per byte loaded), which is the key to better GPU utilization on Apple Silicon.

### Using QLoRA adapters at inference

LoRA adapters are saved in bf16 regardless of base quantization. Use them with the full-precision model:

```bash
python -m mlx_video.generate_wan \
  --lora ./output/lora_final.safetensors \
  --prompt "A video of ohwx walking"
```
