# Wan2.2 I2V-14B Diagnostic Report

This document records the systematic diagnostic methodology used to debug the Wan2.2 I2V-14B (Image-to-Video, 14 billion parameter) pipeline in mlx-video, along with every bug found, its root cause, and fix.

## Table of Contents

- [Overview](#overview)
- [Architecture Summary](#architecture-summary)
- [Diagnostic Methodology](#diagnostic-methodology)
- [Bug 1: Text Embedding Cross-Contamination](#bug-1-text-embedding-cross-contamination)
- [Bug 2: VAE Encoder Weights Excluded from Conversion](#bug-2-vae-encoder-weights-excluded-from-conversion)
- [Bug 3: RoPE Frequency Computation](#bug-3-rope-frequency-computation)
- [Bug 4: VAE Encoder Temporal Downsample Order](#bug-4-vae-encoder-temporal-downsample-order)
- [Bug 5: Non-Chunked VAE Encoding](#bug-5-non-chunked-vae-encoding)
- [Verified Correct Components](#verified-correct-components)
- [Performance Optimizations](#performance-optimizations)
- [Open Investigation: CFG Effectiveness](#open-investigation-cfg-effectiveness)
- [Reference Implementation](#reference-implementation)
- [Useful Diagnostic Commands](#useful-diagnostic-commands)

---

## Overview

The I2V-14B pipeline takes an input image and generates a video using a dual-model diffusion transformer. The initial implementation produced severely broken output — first frame showed the image, subsequent frames degraded to noise, checkerboard artifacts, or flat grey.

Through a systematic component-by-component comparison against the reference PyTorch implementation, **five bugs** were found and fixed. The approach was to verify each component in isolation numerically, then narrow down failures to the subsystem level.

### Timeline of Symptoms

| Stage | Symptom | Root Cause |
|-------|---------|------------|
| Initial | Grey/blurry frames after frame 1 | Non-chunked VAE encoding (Bug 5) |
| After chunked encoding fix | First frame OK, rest degrades to noise | Text embedding cross-contamination (Bug 1) + RoPE frequencies (Bug 3) |
| After text + RoPE fix | Severe 8px checkerboard on frames 4+ | VAE encoder temporal downsample order (Bug 4) |
| After VAE fix | Image in frames 0-3, grey frames 4+ | CFG effectiveness issue (open investigation) |

---

## Architecture Summary

```
I2V-14B Pipeline:
  Input Image → VAE Encoder → [16, T_lat, H_lat, W_lat]
                                      ↓
  Mask Construction → [4, T_lat, H_lat, W_lat]
                                      ↓
  y = concat(mask, encoded_video) → [20, T_lat, H_lat, W_lat]
                                      ↓
  Noise [16, T_lat, H_lat, W_lat] + y → [36, T_lat, H_lat, W_lat]
                                      ↓
  Dual DiT (40 layers, 5120 dim) × 40 denoising steps
                                      ↓
  Denoised Latent [16, T_lat, H_lat, W_lat]
                                      ↓
  VAE Decoder → Video [3, F, H, W]
```

**Key parameters:**
- `in_dim=36` (16 noise + 4 mask + 16 image latents), `out_dim=16`
- Dual model: HIGH noise (t ≥ 900) and LOW noise (t < 900)
- 40 steps, shift=5.0, guide_scale=(3.5, 3.5)
- Uses Wan2.1 VAE (z_dim=16, stride 4×8×8)

---

## Diagnostic Methodology

### 1. Component-Level Numerical Verification

Each component was tested in isolation against the reference PyTorch implementation:

1. **Load identical inputs** (same random seed, same image, same prompt)
2. **Run through reference** (on CPU where possible) and save intermediate tensors as `.npy`
3. **Run through MLX** with the same inputs
4. **Compare outputs** with `np.abs(ours - ref).max()` and relative difference metrics

Components tested this way:
- RoPE frequency parameters and rotation output
- Time embedding (sinusoidal → MLP → projection)
- Patchify (reshape+Linear vs Conv3d)
- Unpatchify (transpose-based vs einsum)
- Scheduler (UniPC) timesteps and step formulas
- VAE encoder output (frame-by-frame comparison)
- Text embeddings (per-model MLP output)
- Cross-attention K/V cache shapes
- Mask construction values

### 2. Artifact Analysis

When visual artifacts appeared, quantitative metrics were used to characterize them:

- **Checkerboard metric**: Difference between even-indexed and odd-indexed pixels at patch boundaries. Values > 20 indicate visible checkerboard.
- **FFT frequency analysis**: Power at the 8px spatial frequency (matches VAE stride). 3× normal power confirmed VAE-stride-aligned artifacts.
- **Per-frame statistics**: Mean, std, min, max for each decoded video frame to track temporal degradation.
- **Frame difference**: `mean(|frame[i] - frame[i-1]|)` to measure motion vs static content.

### 3. Isolation Testing

- **VAE round-trip test**: Encode image+zeros → decode. If clean, VAE decoder is not the source.
- **Single-step model output**: Run one diffusion step and compare cond vs uncond predictions to check CFG effectiveness.
- **Patchify/unpatchify synthetic test**: Pass structured gradient through unpatchify to verify spatial ordering.
- **Resolution sweeps**: Test at 480×272, 640×384, 1280×720 to check resolution dependence.
- **Step count sweeps**: Test at 5, 20, 40 steps to distinguish convergence issues from model bugs.

### 4. Weight Comparison

Direct comparison of converted MLX weights against original PyTorch weights:
```python
# Load both weight sets
pt_weights = torch.load("model.safetensors")
mlx_weights = mx.load("model.safetensors")
# Compare each key
for key in pt_weights:
    diff = np.abs(np.array(pt_weights[key]) - np.array(mlx_weights[key])).max()
```
Expected: max diff ≈ 0.001 (bfloat16 rounding). Actual: confirmed for all keys.

---

## Bug 1: Text Embedding Cross-Contamination

**Symptom:** Model ignores text prompt, generated frames lack semantic content.

**Root Cause:** For the dual-model architecture (high-noise and low-noise experts), text embeddings were computed using only `low_noise_model.embed_text()` and reused for both models' cross-attention K/V caches. The two models have **different** text embedding MLP weights — 42% relative mean difference in output.

**How Found:** Compared `text_embedding_0.weight` and `text_embedding_1.weight` between `high_noise_model.safetensors` and `low_noise_model.safetensors`. Found 17.9% and 26.3% relative differences in the weight matrices.

**Fix:** Compute separate text embeddings per model:
```python
# Before (broken):
context_emb = low_noise_model.embed_text([context, context_null])
cross_kv = low_noise_model.prepare_cross_kv(context_emb)  # used for BOTH models

# After (correct):
context_emb_low = low_noise_model.embed_text([context, context_null])
context_emb_high = high_noise_model.embed_text([context, context_null])
cross_kv_low = low_noise_model.prepare_cross_kv(context_emb_low)
cross_kv_high = high_noise_model.prepare_cross_kv(context_emb_high)
```

**File:** `mlx_video/generate_wan.py` (lines 333–349)
**Commit:** `a85b1c21`

---

## Bug 2: VAE Encoder Weights Excluded from Conversion

**Symptom:** VAE encoder produces constant output regardless of input image (all-zero weights after conversion).

**Root Cause:** The conversion script only included encoder weights for `model_type == "ti2v"` (TI2V-5B), not for `"i2v"` (I2V-14B). Since `load_vae_encoder()` uses `strict=False`, missing encoder weights were silently ignored, resulting in random initialization.

**How Found:** Traced through `convert_wan.py` and found `include_encoder = config.model_type == "ti2v"`. Cross-referenced with the fact that I2V-14B also requires a VAE encoder (for image conditioning).

**Fix:**
```python
# Before:
include_encoder = config.model_type == "ti2v"
# After:
include_encoder = config.model_type in ("ti2v", "i2v")
```

**Note:** The user's specific model happened to be manually converted with encoder weights already present, so this fix was preventive for future conversions.

**File:** `mlx_video/convert_wan.py` (line 424)

---

## Bug 3: RoPE Frequency Computation

**Symptom:** Progressive 2px checkerboard artifacts on generated frames, increasing with temporal distance from the conditioned frame.

**Root Cause:** The reference creates **one** frequency table via `rope_params(1024, head_dim=128)` producing 64 frequency exponents, which `rope_apply` then splits into temporal (22), height (21), and width (21) portions. This gives temporal axes LOW frequencies and spatial axes progressively HIGHER frequencies.

Our code called `rope_params` **three times** with different normalizations:
```python
# WRONG: each axis gets full frequency range [0, 1)
freqs_t = rope_params(1024, d_t=44)   # 22 exponents normalized by 44
freqs_h = rope_params(1024, d_h=42)   # 21 exponents normalized by 42
freqs_w = rope_params(1024, d_w=42)   # 21 exponents normalized by 42
```

The max frequency difference was ~1.0 (not a precision issue — a fundamental design bug). This affected **all** Wan models (T2V, I2V, TI2V).

**How Found:** Line-by-line comparison of `rope_params` usage between reference `model.py` (single call) and our `model.py` (three calls). Printed the actual frequency exponents to confirm the numerical divergence.

**Fix:**
```python
# Single unified frequency table, split by rope_apply
self.freqs = rope_params(1024, dim // config.num_heads)
```

**Impact:** ~35% reduction in checkerboard metric, 55% reduction in FFT 8px-frequency power.

**File:** `mlx_video/models/wan/model.py` (lines 154–156)
**Commit:** `3da4a637`

---

## Bug 4: VAE Encoder Temporal Downsample Order

**Symptom:** Massive checkerboard artifacts aligned to VAE spatial stride (8px period). VAE encoder output for frames 1–4 showed decreasing std (0.37→1.19) while reference showed stable std (0.95→1.34).

**Root Cause:** The VAE encoder has 3 downsampling stages. Two perform spatial+temporal downsampling (`downsample3d`) and one performs spatial-only (`downsample2d`). The order matters:

```
Reference: [False, True, True]  → stage 0: 2d, stage 1: 3d, stage 2: 3d
Ours:      [True, True, False]  → stage 0: 3d, stage 1: 3d, stage 2: 2d  ← WRONG
```

This caused temporal downsampling to happen at the wrong resolution stages (96-dim instead of 384-dim), corrupting temporal feature propagation.

**How Found:** Installed `einops` in the reference environment and ran the reference PyTorch VAE encoder on CPU. Compared frame-by-frame latent output:
- Frame 0 matched exactly (diff=0.0000) — spatial-only processing was correct
- Frames 1–4 had massive differences — proved temporal processing was broken

Then traced through the reference `_video_vae()` function and found it sets `temperal_downsample=[False, True, True]`, while our `Encoder3d` class used the wrong default `[True, True, False]`.

**Fix:**
```python
# In Encoder3d.__init__, change default:
temporal_downsample = [False, True, True]  # was [True, True, False]
```

**Impact:** Encoder output now matches reference within float32 precision (max_diff=2.2e-5). Checkerboard metric dropped from 60–80 to 0.1–7.7.

**File:** `mlx_video/models/wan/vae.py` (line 370)
**Commit:** `3da4a637`

---

## Bug 5: Non-Chunked VAE Encoding

**Symptom:** First 4–5 frames grey, then blurred version of image appears.

**Root Cause:** The reference VAE encoder uses **chunked encoding** with temporal caching (`feat_cache`):
1. Encode first frame alone (1 frame)
2. Encode remaining frames in chunks of 4, with cached temporal features propagating across chunks
3. Each `CausalConv3d` caches last 2 temporal frames from its output, prepending them to the next chunk's input

Our original implementation encoded all frames at once with zero-padded causal convolutions. The temporal feature propagation is fundamentally different because:
- Chunked: real features from previous chunks serve as causal context
- Non-chunked: zeros serve as causal context for the start

**How Found:** Studied the reference `CausalConv3d` caching mechanism (`feat_cache`, `feat_idx`) and traced the temporal dimension through all encoding stages. Confirmed that non-chunked encoding produces different output by comparing tensor shapes and values.

**Fix:** Implemented full chunked encoding with temporal caching:
- Added `cache_x` parameter to `CausalConv3d.__call__`
- Added `feat_cache`/`feat_idx` propagation to `ResidualBlock`, `Resample`, `Encoder3d`
- Rewrote `WanVAE.encode()` with chunked loop (1-frame first chunk, then 4-frame chunks)
- 24 cache slots across the encoder (1 conv1 + 18 downsamples + 4 middle + 1 head)

**File:** `mlx_video/models/wan/vae.py` (multiple methods)
**Commit:** `b6a94c4c`

---

## Verified Correct Components

These components were numerically verified against the reference and are **not** sources of bugs:

| Component | Method | Max Diff | Notes |
|-----------|--------|----------|-------|
| Weight conversion | Direct tensor comparison | ~0.001 | bfloat16 rounding only |
| RoPE rotation | Standalone comparison (float32 vs float64) | 1.3e-5 | Complex vs real multiplication equivalent |
| Time embedding | Full MLP comparison (sinusoidal→embed→project) | 7e-4 | 0.03% relative |
| Patchify | Conv3d vs reshape+Linear | 3.5e-3 | 0.16% relative |
| Unpatchify | einsum vs transpose(6,0,3,1,4,2,5) | exact | Identical operation |
| Scheduler (UniPC) | Formula-level audit + timestep comparison | exact | Predictor, corrector, lambda, rhos all match |
| Mask construction | Value comparison | exact | [4, T_lat, H_lat, W_lat], first temporal=1 |
| CFG formula | Code audit | — | `uncond + gs * (cond - uncond)` correct order |
| VAE decoder | Round-trip test (encode→decode) | clean | No checkerboard in round-trip output |
| Cross-attention K/V | Shape and value audit | — | Batch dimension preserved correctly |

---

## Performance Optimizations

Applied alongside bug fixes to improve inference speed:

### Pre-Computation (Before Diffusion Loop)
- **Cross-attention K/V caching**: Precompute K/V projections for all 40 blocks once
- **RoPE cos/sin precomputation**: Build frequency tensors once instead of per-step broadcast/concat
- **Attention mask precomputation**: Build padding mask once, pass via kwargs
- **Inverse frequency caching**: Store sinusoidal `inv_freq` in `__init__` instead of recomputing
- **Timestep list conversion**: `sched.timesteps.tolist()` before loop to avoid `.item()` sync

### Per-Step Optimizations
- **Single patchify + broadcast for CFG B=2**: Detect identical batch inputs, patchify once and broadcast instead of duplicating the Linear projection
- **Vectorized RoPE**: When all batch elements share the same grid size, apply rotation to the full batch tensor instead of looping per element
- **Redundant type cast removal**: MLX type promotion handles `bfloat16 * float32 → float32` automatically — removed 240 unnecessary graph nodes per step (6 casts × 40 blocks)
- **Euler scheduler sync fix**: Pre-store sigmas as Python floats to avoid `.item()` evaluation sync

### TeaCache Integration
- Polynomial rescaling stays in MLX lazy graph (Horner's method)
- Single `.item()` call on the accumulated distance for the skip/compute decision
- Configurable threshold, retention steps, and cutoff steps

---

## Open Investigation: CFG Effectiveness

**Current symptom:** After all bug fixes, generated video shows the input image in frames 0–3 (latent frame 0), then grey/flat frames for the rest.

**Finding:** A single forward pass at t=1000 shows cond and uncond predictions are nearly identical (|diff| mean = 0.01–0.035). With `guide_scale=3.5`, the CFG guidance term barely changes anything.

**Possible causes under investigation:**
1. Cross-attention context flow — both cond and uncond may be receiving equivalent context
2. The model may genuinely produce small cond/uncond differences for I2V (since both share the same y conditioning)
3. The `embed_text` method or `prepare_cross_kv` may not properly separate B=2 batch elements
4. There may be an issue with how cross-attention K/V caches index into batch elements

**Diagnostic approach:** Compare cross-attention K/V cache values between cond (index 0) and uncond (index 1) to confirm they contain different embeddings.

---

## Reference Implementation

The reference PyTorch implementation is at `/Users/daniel/Projects/Wan2.2/`:

| File | Contents |
|------|----------|
| `wan/image2video.py` | I2V pipeline (y construction, mask, diffusion loop) |
| `wan/modules/model.py` | DiT model (forward pass, RoPE, patchify) |
| `wan/modules/vae2_1.py` | VAE encoder/decoder with chunked encoding |
| `wan/utils/fm_solvers_unipc.py` | UniPC scheduler |
| `wan/configs/wan_i2v_A14B.py` | Model configuration |

Key structural differences between reference and our implementation:
- Reference runs **separate B=1 forward passes** for cond/uncond; we batch as B=2
- Reference uses `torch.amp.autocast('cuda', dtype=bfloat16)` with explicit float32 blocks; we cast via weight dtype
- Reference uses `Conv3d` for patchify; we use equivalent `reshape + Linear`
- Reference casts timesteps to `int64`; we keep as float (diff < 1.0)

---

## Useful Diagnostic Commands

### Run I2V-14B generation
```bash
python -m mlx_video.generate_wan \
  --prompt "A woman smiles at camera" \
  --image start.png \
  --model-dir /Volumes/SSD/Wan-AI/Wan2.2-I2V-A14B-MLX \
  --num-frames 17 --steps 40 \
  --height 384 --width 640 \
  --output output_i2v.mp4
```

### Check VAE encoder output
```python
import mlx.core as mx, numpy as np
from mlx_video.models.wan.vae import WanVAE
# Load VAE and encode an image
latents = vae.encode(video_tensor)  # [1, 16, T_lat, H_lat, W_lat]
for t in range(latents.shape[2]):
    frame = np.array(latents[0, :, t])
    print(f"Frame {t}: mean={frame.mean():.4f} std={frame.std():.4f}")
```

### Analyze video frame quality
```python
import cv2, numpy as np
cap = cv2.VideoCapture("output.mp4")
while True:
    ret, frame = cap.read()
    if not ret: break
    # Checkerboard metric: high values indicate patch-boundary artifacts
    checker = np.abs(frame[::2, ::2].astype(float) - frame[1::2, 1::2].astype(float)).mean()
    print(f"std={frame.std():.1f} checker={checker:.1f}")
```

### Compare weights between PyTorch and MLX
```python
import torch, mlx.core as mx, numpy as np
pt = torch.load("model.pt", map_location="cpu")
mlx_w = mx.load("model.safetensors")
for key in sorted(pt.keys()):
    if key in mlx_w:
        diff = np.abs(pt[key].float().numpy() - np.array(mlx_w[key])).max()
        if diff > 0.01:
            print(f"LARGE DIFF {key}: {diff:.6f}")
```
