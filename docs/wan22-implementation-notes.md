# Wan2.2 MLX Implementation Notes

> Learnings and key decisions from porting Wan2.2 (TI2V-5B / T2V-14B / T2V-1.3B) to Apple MLX.

## Architecture Overview

Wan2.2 is a Diffusion Transformer (DiT) for video generation. Despite early reports, the T2V/TI2V models do **not** use Mixture-of-Experts — they are dense DiT models with a dual-model architecture for the 14B variant (separate high-noise and low-noise denoisers with a boundary timestep).

### Key Parameters

| Model | dim | heads | layers | FFN mult | VAE z_dim | VAE stride |
|-------|-----|-------|--------|----------|-----------|------------|
| T2V-14B | 5120 | 40 | 40 | 4×(5120×4/3) | 16 | (4, 8, 8) |
| TI2V-5B | 3072 | 24 | 32 | 4×(3072×4/3) | 48 | (4, 16, 16) |
| T2V-1.3B | 1536 | 12 | 30 | 4×(1536×4/3) | 16 | (4, 8, 8) |

### Codebase Structure (~3900 lines of Wan2.2 code)

```
mlx_video/
├── generate_wan.py           # 483L - Generation pipeline (T2V + I2V)
├── convert_wan.py            # 564L - Weight conversion from HuggingFace
└── models/wan/
    ├── config.py             # 113L - Model configs (dataclass presets)
    ├── model.py              # 320L - DiT model (time embed, patchify, unpatchify)
    ├── transformer.py        #  91L - Attention block + FFN
    ├── attention.py          # 211L - Self-attention + cross-attention
    ├── rope.py               # 100L - 3D Rotary Position Embeddings
    ├── text_encoder.py       # 240L - T5 encoder (UMT5-XXL)
    ├── scheduler.py          # 428L - Euler, DPM++ 2M, UniPC schedulers
    ├── vae.py                # 315L - Wan2.1 VAE decoder (4×8×8)
    ├── vae22.py              # 836L - Wan2.2 VAE encoder + decoder (4×16×16)
    ├── loading.py            # 154L - Model loading utilities
    └── i2v_utils.py          #  58L - I2V mask/preprocessing
```

---

## Critical Bugs & Fixes

### 1. MLX Underscore Attribute Gotcha

**Problem**: MLX's `nn.Module` silently ignores underscore-prefixed attributes (`_layer_0`, `_layer_1`, etc.) in `parameters()` and `load_weights()`. The Wan2.2 VAE had layers named `_layer_N`, causing **87 out of 110 weights to be silently dropped** during loading.

**Fix**: Rename all `_layer_N` attributes to `layer_N`. MLX treats underscore-prefixed attributes as "private" and excludes them from the parameter tree.

**Lesson**: Never use underscore-prefixed names for `nn.Module` sub-modules in MLX.

### 2. Patchify Channel Ordering

**Problem**: The patchify/unpatchify operations transposed channels incorrectly — producing `[C fastest]` layout instead of `[C slowest]`, causing completely garbled video output.

**Fix**: Changed reshape to produce correct `[B, T', H', W', pt*ph*pw*C]` ordering matching PyTorch's contiguous memory layout.

**Lesson**: When porting PyTorch reshape/view operations to MLX, pay close attention to memory layout — PyTorch is row-major by default, and reshape semantics differ when dimensions are reordered.

### 3. VAE AttentionBlock Reshape

**Problem**: Attention block merged batch (B) with channels (C) instead of batch with temporal (T), producing a green checker pattern in output.

**Fix**: Correct reshape from `[B*C, T, H, W]` to `[B*T, C, H, W]` for spatial attention.

### 4. RMS Norm vs L2 Norm

**Problem**: The Wan2.2 VAE uses a class named `RMS_norm` in PyTorch, but it actually computes **L2 normalization** (divide by L2 norm), not RMS normalization (divide by RMS). Using actual RMS norm caused exponential value explosion.

**Fix**: Implement as `x / ||x||₂` instead of `x / sqrt(mean(x²))`.

**Lesson**: Don't trust class names in reference code — read the actual computation.

### 5. Video Codec Green Output

**Problem**: OpenCV's `mp4v` codec on macOS produces green-tinted video.

**Fix**: Switch to `imageio` with `libx264` codec. Fallback chain: imageio → cv2 (avc1) → PNG frames.

---

## Precision & Dtype Flow

### The bfloat16 Autocast Pattern

The official PyTorch implementation uses `torch.autocast("cuda", dtype=torch.bfloat16)` which automatically casts matmul inputs. In MLX, we replicate this manually:

| Operation | Official (PyTorch) | MLX Implementation |
|---|---|---|
| Modulation/gates | float32 (explicit `autocast(enabled=False)`) | `x.astype(mx.float32)` before modulation |
| QKV projections | bfloat16 (outer autocast) | Cast input to `self.q.weight.dtype` |
| RoPE computation | float64 → float32 | float32 (MLX lacks float64 on GPU) |
| Q/K after RoPE | bfloat16 (`q.to(v.dtype)`) | Cast back to weight dtype after RoPE |
| FFN matmuls | bfloat16 (outer autocast) | Cast input to `self.fc1.weight.dtype` |
| Residual stream | float32 | float32 (no cast) |

**Result**: ~16% speedup (47s vs 56s for 20 steps at 480p) with no quality regression.

**Key insight**: Modulation parameters (scale, shift, gate) must stay in float32 — they are small values (~0.01–0.1) that lose significant precision in bfloat16. The official code explicitly disables autocast for these computations.

### T5 Encoder Precision

The T5 text encoder must run in float32. Bfloat16 weights cause the attention softmax to produce degenerate distributions, which corrupts text conditioning and manifests as blurry patches in generated video. Since T5 only runs once per generation, the performance cost is negligible.

### VAE Decoder Precision

VAE weights must be float32. Bfloat16 VAE decode introduces visible quality loss in the decoded video frames.

---

## Scheduler Implementation Details

### Three Schedulers: Euler, DPM++ 2M, UniPC

All operate in the flow-matching formulation where `sigma` represents the noise level (1.0 = pure noise, 0.0 = clean).

**Euler**: Simple first-order ODE solver. Most stable, recommended for debugging.

**DPM++ 2M**: Second-order multistep solver. Uses previous step's model output for higher-order correction. Requires special handling at boundaries (return `±inf` from `_lambda()` when sigma is 0 or 1).

**UniPC** (default, matches official): Second-order predictor-corrector. The "C" (corrector) part is critical — it refines each step using the already-computed model output at **zero additional model evaluation cost**.

### UniPC Corrector: Must Be Enabled

**Discovery**: Our implementation had `use_corrector=False` by default, but the official Wan2.2 code **always** enables it (there's no flag — the corrector runs whenever `step_index > 0`).

**Impact**: Without the corrector, UniPC degrades to a simple predictor, losing its second-order accuracy advantage.

### UniPC Corrector Coefficients

The corrector coefficients (`rhos_c`) must be computed by solving a linear system, not hardcoded. For order ≥ 2, hardcoding `rhos_c[-1] = 0.5` introduces ~6–13% error in the correction term across 47+ steps. The fix uses `np.linalg.solve()` to compute exact coefficients.

### Sigma Schedule

```python
# Flow-matching sigma schedule with shift
sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
```

Default shifts: T2V-14B uses 5.0, TI2V-5B uses 3.0, T2V-1.3B uses 3.0.

---

## Image-to-Video (I2V) Pipeline

### Per-Token Timesteps

I2V conditions on a reference first frame by giving first-frame latent patches a timestep of 0 (clean) while other patches get the current diffusion timestep:

```python
# mask_tokens: [1, L] — 0 for first-frame patches, 1 for rest
t_tokens = mask_tokens * current_timestep  # first-frame → t=0
```

The model receives 2D timestep input `[B, L]` instead of scalar, enabling per-token noise levels.

### Mask Re-application

After each scheduler step, the first-frame latent is re-injected to prevent drift:

```python
latents = (1.0 - mask) * z_img + mask * latents
```

### VAE Encoder Temporal Downsample Order

The Wan2.2 VAE encoder has `temporal_downsample = (False, True, True)`:
- Stage 0: Spatial-only downsampling
- Stages 1–2: Spatial + temporal downsampling

This was incorrectly set to `(True, True, False)` initially, causing wrong spatial processing paths.

---

## Dimension Constraints

### Patchify Alignment

Video dimensions must be divisible by `patch_size × vae_stride`:
- **TI2V-5B**: patch=(1,2,2), stride=(4,16,16) → alignment = **32** pixels
- **T2V-14B**: patch=(1,2,2), stride=(4,8,8) → alignment = **16** pixels

Example: 720p (1280×720) → 720 % 32 ≠ 0, auto-aligns to **704**.

### Frame Count

Frames must satisfy `num_frames = 4n + 1` (e.g., 5, 9, 13, ..., 81) due to temporal VAE stride of 4.

---

## Performance Optimizations

### Batched CFG

Instead of two separate forward passes for conditional and unconditional predictions, batch them into a single B=2 forward pass:

```python
preds = model([latents, latents], t=t_batch, context=context_cfg, ...)
noise_pred_cond, noise_pred_uncond = preds[0], preds[1]
```

**Result**: ~40% speedup by amortizing attention overhead.

### Precomputed Text Embeddings & Cross-Attention KV Cache

Text embeddings and cross-attention K/V projections are constant across all diffusion steps. Computing them once and passing as caches eliminates redundant computation.

### Memory Management in Diffusion Loop

```python
# Release temporaries before eval to free memory for graph execution
del noise_pred_cond, noise_pred_uncond, noise_pred, preds
mx.eval(latents)
```

MLX's lazy evaluation means `mx.eval()` triggers the full computation graph. Deleting intermediate arrays before eval allows MLX to reuse their memory during execution.

---

## Weight Conversion

### Key Mapping Patterns

The PyTorch → MLX conversion (`convert_wan.py`) handles several systematic transforms:

1. **Conv3d weight transposition**: PyTorch `(out, in, D, H, W)` → MLX `(out, D, H, W, in)`
2. **Linear weight transposition**: PyTorch `(out, in)` → MLX `(out, in)` (same convention for `nn.Linear`)
3. **Nested module paths**: `blocks.0.self_attn.q.weight` → same paths, MLX loads by dotted key

### Dual-Model Splitting

The T2V-14B uses dual models (high-noise and low-noise). The conversion script splits a single checkpoint into separate files or handles pre-split checkpoints from HuggingFace.

---

## Testing Strategy

260 tests across 9 files, all running in ~4 seconds:

| File | Focus |
|------|-------|
| test_wan_config.py | Config presets, field validation |
| test_wan_attention.py | Self/cross attention, RMSNorm, bf16 autocast |
| test_wan_transformer.py | FFN, attention block, float32 modulation |
| test_wan_model.py | Full DiT forward pass, per-token timesteps |
| test_wan_t5.py | T5 encoder layers and full encoding |
| test_wan_vae.py | VAE 2.1 decoder, VAE 2.2 encoder + decoder |
| test_wan_scheduler.py | All 3 schedulers, cross-scheduler coherence |
| test_wan_convert.py | Weight sanitization and conversion |
| test_wan_generate.py | End-to-end pipeline, I2V masks, dimension alignment |

Tests use a tiny config (`dim=64, heads=2, layers=2`) for fast execution. Cross-scheduler coherence tests verify that all three schedulers produce similar outputs from the same noise.

---

## Known Issues

### I2V Quality Degradation

Frames 2–13 gradually degrade, and frame 14 often has a "flash" artifact. All implementation details have been verified against the official PyTorch code with no discrepancies found. Possible causes:
- Subtle numerical differences from float32 vs float64 RoPE (MLX lacks float64 on GPU)
- MLX-specific attention precision behavior
- Better prompts and 720p resolution (the model's native resolution) help reduce artifacts

### Chinese Negative Prompt

The official Wan2.2 uses a Chinese negative prompt that prevents oversaturation and comic-style artifacts. Correct tokenization requires `ftfy.fix_text()` to normalize fullwidth characters and double HTML unescaping. Without proper text cleaning, the negative prompt tokens don't match the training distribution, causing blurry patches.
