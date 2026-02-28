# Wan2.2 LoRA Training — Architecture & Design

This document describes the design decisions, sources of inspiration, and overall architecture of the Wan2.2 LoRA training pipeline in mlx-video.

---

## Sources of Inspiration

The training pipeline draws from three primary references:

### 1. mflux trainer (`/mflux/models/common/training/`)

The mflux project provides an MLX-native LoRA trainer for Flux (a diffusion image generation model). We adopted several patterns from it:

- **Config-driven workflow**: A single JSON config file defines the entire training run — model path, data directory, hyperparameters, LoRA settings, and output configuration. No code changes needed between runs.
- **Pre-encoding strategy**: Encode all training images through the VAE and all prompts through the text encoder *before* training begins. This frees the encoder models from memory during the training loop, which is critical on unified-memory Apple Silicon systems.
- **Data format convention**: A flat directory of images with matching `.txt` prompt files. Simple, human-readable, no dataset manifests or database files.
- **LoRA layer injection**: Replace targeted `nn.Linear` modules in-place with LoRA wrappers, then freeze all base parameters. Only the small LoRA matrices are updated.

### 2. ai-toolkit (`/ai-toolkit/toolkit/models/wan21/`)

ai-toolkit is the most popular open-source Wan2.1/2.2 trainer, written in PyTorch. It served as a reference for Wan-specific training details:

- **Flow matching loss formulation**: The velocity prediction target `v = noise - clean` and the noisy interpolation `z_t = (1 - σ) * clean + σ * noise`, which is Wan2.2's specific variant of flow matching.
- **Noise schedule shift**: Wan2.2 applies a schedule shift (`shift * σ / (1 + (shift-1) * σ)` with shift=12.0 for T2V-14B) that must be replicated during training to match inference behavior.
- **Diffusers key format**: ai-toolkit saves LoRA weights with a `diffusion_model.` prefix on all keys. We adopt the same convention for cross-tool compatibility (ComfyUI, diffusers, etc.).
- **Timestep bias**: ai-toolkit documents the importance of training across the full noise range for character LoRAs, not just low noise — this inspired our configurable `timestep_sampling` strategies.

### 3. Wan2.2 reference implementation (`/Wan2.2/`)

The official Wan2.2 source code provided the ground-truth model architecture details:

- **Dual-model pipeline**: Wan2.2 T2V-14B uses separate `high_noise_model` and `low_noise_model` with a boundary around σ=0.875. For character LoRA training, we default to the `low_noise_model` since it controls fine details, textures, and identity.
- **Patch embedding structure**: The Conv3d patch embedding is implemented as a reshaped Linear layer (`patch_embedding_proj`), which affects how we navigate the model tree for LoRA injection.
- **Model constants**: `num_train_timesteps=1000`, `sample_shift=12.0`, `text_len=512`, and the transformer block structure (40 layers, self_attn + cross_attn + ffn per block).

---

## Architecture Overview

The training pipeline has five phases, orchestrated by `train_wan.py`:

```
┌─────────────────────────────────────────────────────┐
│                    train_wan.py                      │
│                  (CLI entry point)                   │
└─────────┬───────────────────────────────────────────┘
          │
          ▼
┌─────────────────┐     Parse JSON config, discover
│   config.py     │     image/prompt pairs, validate
└────────┬────────┘
         │
         ▼
┌─────────────────┐     Load VAE → encode images → free VAE
│   dataset.py    │     Load T5  → encode prompts → free T5
└────────┬────────┘     Returns list[EncodedItem]
         │
         ▼
┌─────────────────┐     Load WanModel, load safetensors weights
│  (model loading) │     For dual-model: default to low_noise_model
└────────┬────────┘
         │
         ▼
┌─────────────────┐     Replace nn.Linear → TrainableLoRALinear
│ lora_layers.py  │     Freeze all base weights
└────────┬────────┘     Only lora_A, lora_B are trainable
         │
         ▼
┌─────────────────┐     Flow matching loss + AdamW optimization
│   trainer.py    │     Epoch loop with timestep sampling
└────────┬────────┘
         │
         ▼
┌─────────────────┐     Extract LoRA weights, add diffusion_model. prefix
│    save.py      │     Save as safetensors with metadata
└─────────────────┘
```

### Phase 1: Configuration (`config.py`)

The config system uses nested dataclasses parsed from JSON:

```
TrainingConfig
├── model_dir: str              (path to converted MLX weights)
├── data: str                   (path to training data directory)
├── resolution: int             (target image size, default 512)
├── trigger_word: str?          (prepended to prompts if missing)
├── training: TrainingLoopConfig
│   ├── num_epochs: int
│   ├── learning_rate: float
│   ├── batch_size: int
│   ├── optimizer: str
│   └── timestep_sampling: str  (balanced / low_bias / high_bias)
├── lora: LoRAConfig
│   ├── rank: int
│   ├── alpha: float
│   ├── targets: list[str]      (e.g., "self_attn.q", "ffn.fc1")
│   └── blocks: BlockRange?     (start/end block indices)
├── checkpoint: CheckpointConfig
└── monitoring: MonitoringConfig
    ├── log_frequency: int         (console log every N epochs, default 1)
    ├── plot_frequency: int        (update loss plot PNG every N epochs, default 10)
    ├── generate_image_frequency: int (preview image every N epochs, 0=disabled)
    ├── preview_width: int         (preview resolution, default 512)
    └── preview_height: int        (preview resolution, default 512)
```

Data discovery is automatic: scan the data directory for images (`.jpg`, `.jpeg`, `.png`, `.webp`) and match each to a `.txt` file with the same stem. The trigger word, if set, is prepended to any prompt that doesn't already contain it.

A `preview.txt` file in the data directory, if present, is used as the prompt for preview image generation. Otherwise, the first training sample's prompt is used.

### Phase 2: Dataset Encoding (`dataset.py`)

The encoding phase is deliberately sequential and memory-conscious:

1. **Load VAE encoder** → encode each image to latent space → **free VAE**
2. **Load T5 encoder** → encode each prompt to text embeddings → **free T5**

This two-pass approach means peak memory is `max(VAE, T5)` rather than `VAE + T5`, which can save several GB on machines with limited unified memory.

Image preprocessing:
- Resize to fit within `resolution × resolution` preserving aspect ratio
- Round dimensions to nearest multiple of 32 (VAE spatial compression requirement)
- Normalize pixels to `[-1, 1]`

The encoded items are stored as `EncodedItem` dataclasses with `clean_latents` and `text_embedding` arrays, ready for the training loop.

**VAE format handling**: The encoder automatically detects Wan2.1 (z_dim=16, channels-first) vs Wan2.2 (z_dim=48, channels-last) format and applies the correct encoding path, including `normalize_latents()` for Wan2.2.

### Phase 3: LoRA Injection (`lora_layers.py`)

#### TrainableLoRALinear

The core LoRA module wraps an existing `nn.Linear`:

```
forward(x):
    base_output = linear(x)                        # frozen base weights
    lora_output = x @ lora_A.T @ lora_B.T          # trainable low-rank path
    return base_output + (alpha / rank) * lora_output
```

**Initialization**: `lora_A` uses Kaiming uniform initialization, `lora_B` is zero-initialized. This means the LoRA contribution starts at exactly zero — the model behaves identically to the base model at step 0, which is standard practice to avoid initial quality degradation.

#### Injection Strategy

`inject_lora_layers()` walks the model's `blocks` list and, for each block in the specified range, replaces targeted `nn.Linear` modules with `TrainableLoRALinear` wrappers. The targets are specified as dot-separated paths relative to a block, e.g.:

| Target | Module in Block | What it Controls |
|--------|----------------|-----------------|
| `self_attn.q` | Self-attention query projection | How the model attends to video patches |
| `self_attn.k` | Self-attention key projection | Same |
| `self_attn.v` | Self-attention value projection | What information flows between patches |
| `self_attn.o` | Self-attention output projection | How attention output is projected back |
| `cross_attn.q` | Cross-attention query projection | How video attends to text |
| `cross_attn.k` | Cross-attention key projection | Same |
| `cross_attn.v` | Cross-attention value projection | What text information flows to video |
| `cross_attn.o` | Cross-attention output projection | Same |
| `ffn.fc1` | Feed-forward first layer | Non-linear feature transformation |
| `ffn.fc2` | Feed-forward second layer | Same |

By default, all 10 targets across all 40 blocks are injected (400 LoRA pairs total).

#### Freeze Strategy

After injection, `freeze_base_weights()` does:
1. `model.freeze()` — freezes everything (including LoRA parameters)
2. Walk the model tree and `unfreeze(keys=["lora_A", "lora_B"])` on every `TrainableLoRALinear`

This ensures that only the LoRA matrices participate in gradient computation, keeping the base model fixed.

### Phase 4: Training Loop (`trainer.py`)

#### Flow Matching Loss

Wan2.2 uses a flow matching (rectified flow) formulation:

1. **Sample a timestep** σ ∈ (0, 1)
2. **Apply shift**: σ_shifted = `shift * σ / (1 + (shift-1) * σ)` (shift=12.0 for T2V-14B)
3. **Create noisy latent**: z_t = `(1 - σ_shifted) * clean + σ_shifted * noise`
4. **Predict velocity**: v_pred = `model(z_t, t=σ_shifted * 1000, context=text_emb)`
5. **Compute loss**: MSE(v_pred, `noise - clean`)

The shift parameter is critical — it compresses the noise schedule to spend more steps on low-to-medium noise levels during inference. Training must use the same shift to maintain consistency.

#### Timestep Sampling Strategies

The choice of how to sample σ during training directly affects what the LoRA learns:

- **`balanced`** (uniform): Equal probability across all noise levels. Safe default.
- **`low_bias`** (Beta(1, 2), mean ≈ 0.33): Favors low σ values. Since low noise controls fine details, textures, and character identity, this is ideal when the primary goal is learning a specific appearance.
- **`high_bias`** (Beta(2, 1), mean ≈ 0.67): Favors high σ values. High noise controls global composition, layout, and heavy motion. Useful for style or motion LoRAs.

**Why this matters for Wan2.2 specifically**: The dual-model pipeline splits inference at σ ≈ 0.875, with `high_noise_model` handling early denoising (layout/structure) and `low_noise_model` handling late denoising (details/identity). Since we train on the low-noise model by default, `low_bias` or `balanced` sampling is the natural choice for character LoRAs. If you only train at low noise, the character might look right but the video won't move well, so `balanced` is the recommended default.

#### Optimization

- **Optimizer**: AdamW (default) or Adam, configurable
- **`nn.value_and_grad`**: MLX's functional gradient computation. Computes the loss and all LoRA gradients in a single forward+backward pass.
- **`mx.eval`**: Called after each optimizer step to force evaluation of the lazy computation graph, preventing unbounded memory growth.

### Phase 5: Saving (`save.py`)

#### Weight Collection

Uses MLX's `model.named_modules()` to walk the entire model tree and find all `TrainableLoRALinear` instances. For each, it extracts `lora_A` and `lora_B` with their full path as the key.

#### Key Format

Keys are converted from internal MLX format to diffusers/ai-toolkit format by prepending `diffusion_model.`:

```
MLX internal:  blocks.0.self_attn.q.lora_A.weight
Saved format:  diffusion_model.blocks.0.self_attn.q.lora_A.weight
```

This is the "original" key format (as opposed to diffusers' to_q/attn1 naming). The existing `mlx_video/lora/apply.py` loader handles this format via `_normalize_wan_lora_key()`, which strips the `diffusion_model.` prefix on load.

#### Metadata

Each safetensors file includes metadata: LoRA rank, alpha, model version, and target layers. This allows downstream tools to configure themselves automatically when loading the adapter.

### Monitoring & Preview Generation

Two monitoring tools run during training, controlled by `monitoring` config:

#### Loss Plotting (`plotting.py`)

A matplotlib-based loss plot is rendered as a PNG every `plot_frequency` epochs (default: 10) and at the end of training. The plot shows:
- Individual per-step loss values as semi-transparent blue scatter points
- Smoothed loss curve (exponential moving average, weight=0.9) in red
- Mean loss as a dashed horizontal reference line
- Legend with min loss (and step), final smoothed value, and mean

Saved to `{output_dir}/loss_plot.png`, overwritten each time. matplotlib is an optional dependency — if not installed, plotting is silently skipped with a warning.

#### Preview Image Generation (`preview.py`)

When `generate_image_frequency > 0`, a single-frame preview image is generated every N epochs to visually track training progress. The process:

1. Run a 20-step Euler denoising loop using the current LoRA-injected model at `preview_width × preview_height`
2. Load the VAE decoder temporarily from `model_dir/vae.safetensors`
3. Decode the denoised latent to a single RGB frame
4. Save as PNG to `{output_dir}/previews/preview_epoch_NNNN.png`
5. Free the VAE decoder to reclaim memory

The preview prompt comes from `preview.txt` in the data directory (if present), otherwise the first training sample's prompt is used. This is memory-intensive (loads the VAE each time) but infrequent, and the files form a visual timeline of how the LoRA evolves during training.

---

## Key Design Decisions

### 1. Single-frame training (not video clips)

We train on individual images, not video clips. Each image is encoded as a single-frame latent (temporal dimension = 1). This is standard practice for character LoRAs and has several advantages:

- **Low memory**: A single 512×512 image latent is ~16×16×48 floats. Video clips would require significantly more memory.
- **Data efficiency**: Character identity can be learned from 5–20 images. Video clips would require much more data.
- **Training speed**: Single-frame forward passes are fast, enabling more iterations per minute.

The tradeoff is that the LoRA doesn't learn motion patterns — that's handled by the base model.

### 2. Pre-encode everything, then train

Rather than encoding on-the-fly during training, we encode the entire dataset upfront and cache it in memory. With typical training sets of 5–20 images, the encoded latents + text embeddings fit easily in memory (a few hundred MB). This means:

- The VAE and T5 models can be fully unloaded before training starts
- Peak memory during training is just the transformer + LoRA parameters + optimizer state
- No I/O or encoding overhead during the training loop

### 3. Train on low_noise_model by default

Wan2.2's dual-model architecture splits denoising at σ ≈ 0.875:
- `high_noise_model`: σ > 0.875 → controls composition, layout, coarse structure
- `low_noise_model`: σ < 0.875 → controls details, textures, identity

For character LoRAs, identity lives in the low-noise regime, so we default to training the low-noise model. The balanced timestep sampling ensures we still train across the full sub-range, not just the very lowest noise levels.

### 4. Kaiming init for lora_A, zero init for lora_B

This is the standard LoRA initialization from the original paper. Since `output = base + scale * (x @ A.T @ B.T)` and B starts at zero, the LoRA contribution is exactly zero at initialization. The model starts as the unmodified base model and gradually learns the adaptation.

Kaiming uniform for A (with bound `1/√in_features`) provides appropriate scale for the input dimension, avoiding vanishing or exploding initial gradients.

### 5. Diffusers-compatible output format

The saved LoRA files use:
- **safetensors format**: Standard for ML weight files, supports metadata
- **`diffusion_model.` key prefix**: Matches ai-toolkit / diffusers conventions
- **Original key names** (not diffusers' to_q/attn1 renames): Our model uses `self_attn.q`, not `attn1.to_q`, and the save format preserves this. The existing mlx-video LoRA loader already handles this mapping.

This means trained LoRAs can be loaded in mlx-video, ComfyUI (with the right key mapping), and other tools that support the ai-toolkit format.

### 6. JSON config (not YAML)

JSON was chosen over YAML for consistency with the rest of the mlx-video codebase (which uses JSON for model configs) and because Python's standard library includes `json` — no additional dependency needed. The config is validated at load time with clear error messages for missing fields, invalid values, and missing training data.

---

## Memory Profile (approximate, M4 Max 128GB)

| Phase | Peak Memory | Notes |
|-------|------------|-------|
| VAE encoding | ~4 GB | VAE encoder loaded, then freed |
| T5 encoding | ~12 GB | T5-XXL loaded, then freed |
| Model loading | ~28 GB | 14B parameter transformer |
| Training | ~32 GB | Model + LoRA + optimizer state + gradients |

On machines with less memory (e.g., 32GB M2 Pro), reduce resolution to 256 or use a smaller LoRA rank. The 1.3B Wan2.1 model also works with this trainer and requires significantly less memory.

---

## File Reference

| File | Role |
|------|------|
| `mlx_video/train_wan.py` | CLI entry point, orchestrates all phases |
| `mlx_video/training/config.py` | JSON config parsing, data discovery, validation |
| `mlx_video/training/dataset.py` | VAE/T5 encoding, image preprocessing |
| `mlx_video/training/lora_layers.py` | TrainableLoRALinear, injection, freeze/unfreeze |
| `mlx_video/training/trainer.py` | Flow matching loss, timestep sampling, training loop |
| `mlx_video/training/save.py` | LoRA weight extraction, diffusers key mapping, safetensors save |
| `mlx_video/training/plotting.py` | Loss history tracking, matplotlib PNG rendering |
| `mlx_video/training/preview.py` | Single-frame preview generation during training |
| `tests/test_training.py` | Unit tests for config, LoRA, save, sampling, monitoring |
