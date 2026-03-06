# Porting Diffusion Video Models to MLX: Lessons Learned

A practical guide distilled from porting the Helios 14B DiT video generation model
from PyTorch to MLX on Apple Silicon. These lessons apply broadly to any diffusion-based
video (or image) model port.

---

## Table of Contents

1. [Debugging Methodology](#1-debugging-methodology)
2. [Precision & Dtype Pitfalls](#2-precision--dtype-pitfalls)
3. [Autoregressive Chunk Boundaries](#3-autoregressive-chunk-boundaries)
4. [VAE Decoder Artifacts](#4-vae-decoder-artifacts)
5. [Scheduler & Timestep Issues](#5-scheduler--timestep-issues)
6. [Weight Conversion](#6-weight-conversion)
7. [Text Conditioning Failures](#7-text-conditioning-failures)
8. [Position Encodings (RoPE)](#8-position-encodings-rope)
9. [Multi-Stage / Pyramid Pipelines](#9-multi-stage--pyramid-pipelines)
10. [Common Symptoms → Root Causes](#10-common-symptoms--root-causes)
11. [Verification Checklist](#11-verification-checklist)
12. [Diagnostic Tools](#12-diagnostic-tools)

---

## 1. Debugging Methodology

### Component isolation first

Never debug the full pipeline. Test each component in isolation:

1. **Text encoder** — Does it produce embeddings with reasonable statistics? (std > 0.01)
2. **Scheduler** — Do sigma/timestep values match the reference exactly?
3. **Transformer** — Does a single forward pass match the reference? (cosine similarity > 0.999)
4. **VAE decoder** — Feed reference latents into your VAE. Does the output look correct?

If every component matches individually but the pipeline fails, the bug is in
**orchestration** — how components are wired together.

### Statistical fingerprinting

Track per-step statistics through the diffusion loop:

```python
# After each denoising step
print(f"step {i}: mean={latent.mean():.6f} std={latent.std():.6f} "
      f"min={latent.min():.4f} max={latent.max():.4f}")
```

**What to look for:**
- **Progressive mean drift** (e.g., -0.002 → -0.040 → -0.123) signals accumulating errors
- **Collapsing std** (std dropping toward 0) signals broken conditioning or wrong noise schedule
- **Exploding values** signal wrong sigma scaling or scheduler formula

### Cross-framework numerical comparison

The most powerful debugging tool: save intermediate tensors from your MLX pipeline,
feed them to the PyTorch reference, compare outputs.

```python
# In MLX pipeline, save inputs before transformer call
mx.save("debug_inputs.npz", {"latent": latent, "timestep": t, "text_emb": text_emb})

# In PyTorch script, load and compare
inputs = np.load("debug_inputs.npz")
mlx_out = np.load("debug_output.npz")["flow"]
pt_out = reference_model(torch.from_numpy(inputs["latent"]), ...)
cos_sim = F.cosine_similarity(pt_out.flatten(), torch.from_numpy(mlx_out).flatten(), dim=0)
# cos_sim > 0.999 = model is correct; bug is elsewhere
# cos_sim < 0.99  = model has a bug; compare per-layer
```

### Ablation testing

When a pipeline has multiple "fixes" or features, disable them one at a time:

- **Frozen history**: Fix history to the same value for all chunks → proves whether
  history propagation is the source of drift/zoom
- **Single chunk**: Generate only 1 chunk → isolates per-chunk quality from
  multi-chunk interaction bugs
- **Disable post-processing**: Remove cross-fade, blending, corrections → reveals
  what the raw model output looks like

### Use reference on same hardware

Run the PyTorch reference on the same device (MPS for Apple Silicon). CUDA and MPS
produce different numerical results due to different float handling. Comparing your
MLX output against a CUDA reference adds noise to the comparison.

```python
# MPS may not support float64 — patch the reference:
original_linspace = torch.linspace
def patched_linspace(*args, **kwargs):
    kwargs.pop("dtype", None)
    return original_linspace(*args, dtype=torch.float32, **kwargs)
torch.linspace = patched_linspace
```

---

## 2. Precision & Dtype Pitfalls

### The #1 source of subtle bugs

Precision issues caused the most insidious bugs in our port. They don't cause
crashes — they cause progressive quality degradation that's hard to attribute.

### Residual connections MUST be float32

**Bug**: Progressive zoom/shrinking across autoregressive chunks.

**Root cause**: Residual additions (`x = x + attn_out`) in bfloat16. With 7-bit
mantissa, high-frequency spatial detail is systematically truncated. Over 144
residual ops × 6+ model calls per chunk, detail is progressively smoothed away.

**Fix**: Promote to float32 for the addition:
```python
# BAD — bfloat16 accumulation
x = x + attn_out

# GOOD — match reference's .float() pattern
x = (x.astype(mx.float32) + attn_out).astype(weight_dtype)
```

**Rule**: If the reference uses `.float()` anywhere, copy that pattern exactly. It's
there for a reason, even if a quick test seems to work without it.

### Scheduler computations need high precision

Diffusion schedulers involve:
- `x0 = xt - sigma * flow` — catastrophic cancellation near sigma ≈ 1
- `log(sigma)` and `exp()` — sensitive to small precision differences

Some references use float64 for these computations. MLX GPU doesn't support float64,
so use float32 and accept small numerical differences, but **never** use bfloat16
for scheduler math.

### Dtype propagation is invisible

Track dtype through your pipeline. A single bfloat16 intermediate can silently
downcast everything downstream:

```python
# This looks harmless but if model output is bfloat16:
result = noise - sigma * model_output  # result is bfloat16!

# Fix: explicit cast
result = (noise.astype(mx.float32) - sigma * model_output.astype(mx.float32))
```

### Type promotion rules differ across frameworks

- PyTorch: bfloat16 + float32 → float32
- MLX: bfloat16 + float32 → float32 (same, but verify)
- NumPy: no bfloat16 support

Always check what your framework does and match the reference's implicit promotions.

---

## 3. Autoregressive Chunk Boundaries

For models that generate long videos by autoregressively extending chunks (Helios,
CogVideoX, etc.), chunk boundaries are the primary source of visual artifacts.

### Don't add post-processing the reference doesn't have

**Bug**: Added pixel cross-fade to smooth boundaries → caused 40% sharpness drop.

The reference pipeline used **no cross-fade at all**. The first frame of each new
chunk is intentionally a sharp reconstruction conditioned on history. Blending it with
the previous chunk's tail (which has different content) creates blur.

**Rule**: Before adding smoothing/blending, verify the reference doesn't do it.
Reference simplicity is usually correct.

### First-frame artifacts are common

The first pixel frame of each non-first chunk is typically a distorted reconstruction
of the conditioning frame. In many models, this is expected behavior:

- **Fix**: Drop the first frame from each chunk
- **Verify frame math**: If 33 raw frames at 16fps → drop 1 → 32 frames = exactly 2 seconds

### History conditioning errors compound

Small errors in how history is prepared, sliced, patchified, or position-encoded
will compound across chunks. The error is invisible in chunk 1, small in chunk 2,
and catastrophic by chunk 5.

**Debug strategy**: Generate with frozen history (same history for every chunk).
If the artifact disappears, the bug is in history handling.

---

## 4. VAE Decoder Artifacts

### Causal temporal convolutions cause boundary warmup

Video VAEs (WanVAE, CogVideoX-VAE) use causal temporal convolutions. When decoding
each chunk independently, the first few frames lack temporal context (only zero
padding), causing:

- **~7% contrast drop** in first frames of each chunk
- **Spatial brightness redistribution** (face darkens, background brightens)

This is inherent to the architecture. The reference has the same effect but at
lower magnitude.

### Post-processing to fix VAE warmup

Two-stage correction applied to first N frames of each non-first chunk:

```python
# Stage 1: Spatially-varying brightness correction
# Downsample reference (previous chunk's last frame) and current frame
ref_small = cv2.resize(ref_frame, (w//16, h//16), interpolation=cv2.INTER_AREA)
cur_small = cv2.resize(cur_frame, (w//16, h//16), interpolation=cv2.INTER_AREA)
diff_small = ref_small - cur_small
diff_full = cv2.resize(diff_small, (w, h), interpolation=cv2.INTER_LINEAR)
corrected = cur_frame + ramp * diff_full  # ramp: 1.0 → 0.0 over N frames

# Stage 2: Per-channel contrast matching
for c in range(3):
    ref_std = np.std(ref_frame[:,:,c])
    cur_std = np.std(corrected[:,:,c])
    scale = 1.0 + ramp * (ref_std / (cur_std + 1e-6) - 1.0)
    corrected[:,:,c] = (corrected[:,:,c] - mean) * scale + mean
```

### VAE overlap decode does NOT work

**Attempted**: Prepend previous chunk's last latent frames to give the decoder
temporal context.

**Result**: Made things **worse** (22% contrast drop vs 7%). The causal convolutions
see conflicting content from different chunks and create larger artifacts than
zero-padding.

**Lesson**: Overlap only works when tiles contain the same content from the same
denoising process (e.g., spatial tiling). It fails for temporal chunks with
different content.

### Per-chunk VAE decoding is correct

Decode each chunk's latents independently, not concatenated. Concatenating all chunks
and decoding together lets boundary discontinuities propagate through temporal
convolutions, creating worse artifacts.

---

## 5. Scheduler & Timestep Issues

### Copy formulas exactly

Even small differences in scheduler formulas compound over many steps:

```python
# Dynamic time shifting — reference uses specific formula
mu = 0.5 + shift * 0.5  # NOT shift * 0.6 or any other constant

# Euler step
x_next = x + (sigma_next - sigma) * flow  # order matters: next - current
```

### Verify sigma schedules numerically

Print and compare sigma values at each step:

```python
# Reference
sigmas_ref = [1.0, 0.99375, 0.9875, ...]

# Your implementation
sigmas = scheduler.get_sigmas(steps)
for i, (r, m) in enumerate(zip(sigmas_ref, sigmas)):
    assert abs(r - m) < 1e-6, f"Step {i}: ref={r}, mlx={m}"
```

### Timestep embedding precision

Integer vs float timesteps matter. Some models expect `timestep=999` (int), others
expect `timestep=0.999` (float). Wrong type can silently produce wrong embeddings
with reasonable-looking but incorrect statistics.

---

## 6. Weight Conversion

### Always verify statistically

After converting weights from PyTorch to MLX format:

```python
for name in mlx_weights:
    pt = pytorch_weights[map_name(name)]
    mx_val = np.array(mlx_weights[name])
    pt_val = pt.numpy()
    cos_sim = np.dot(mx_val.flat, pt_val.flat) / (
        np.linalg.norm(mx_val) * np.linalg.norm(pt_val) + 1e-10
    )
    if cos_sim < 0.9999:
        print(f"MISMATCH: {name} cos_sim={cos_sim:.6f}")
```

### Conv3d → Linear reshaping

When converting 3D convolutions to linear layers (common for MLX which prefers
linear ops), the flattening order must match:

```python
# PyTorch Conv3d weight: (out_ch, in_ch, kT, kH, kW)
# Flatten to Linear: (out_ch, in_ch * kT * kH * kW)
# The reshape order MUST match how the input is patchified
```

### Sanitization functions

Write explicit weight sanitization that maps reference key names to your key names.
Don't rely on automatic matching — key naming conventions differ between frameworks.

---

## 7. Text Conditioning Failures

### Symptom: model predicts noise back to itself

If the model output correlates > 0.8 with its input noise, text conditioning is
likely broken. The model has learned nothing from the prompt and is just returning
its input.

### Check embedding statistics

```python
text_emb = text_encoder(prompt)
print(f"text_emb: mean={text_emb.mean():.4f} std={text_emb.std():.4f}")
# std < 0.01 → embeddings are collapsed → broken encoder or wrong weights
# std > 10.0 → embeddings are exploding → wrong normalization
```

### Verify with ablation

```python
# Generate with real text
output_text = denoise(latent, text_emb=real_embeddings)
# Generate with zeros
output_zero = denoise(latent, text_emb=mx.zeros_like(real_embeddings))
# Compare
text_influence = np.mean(np.abs(output_text - output_zero))
print(f"Text influence: {text_influence:.4f}")  # Should be > 0 (typically 30-60% of output)
```

---

## 8. Position Encodings (RoPE)

### Multi-scale consistency

In pyramid/multi-resolution models, RoPE must be computed consistently across scales.
If the model operates at 1/4 resolution in an early stage, the position grid must
reflect the actual spatial dimensions, not the final target dimensions.

### History vs current chunk

When conditioning on history from a previous chunk, the position encoding for
history frames must match what the model saw during training. Mismatches between
history and current-chunk position encodings can cause subtle spatial distortions
that compound across chunks.

### Factorized RoPE

3D video models often use factorized RoPE (separate temporal, height, width
frequencies). Verify each axis independently:

```python
# Compare temporal frequencies
assert np.allclose(mlx_rope_t, ref_rope_t, atol=1e-5)
# Compare spatial frequencies
assert np.allclose(mlx_rope_h, ref_rope_h, atol=1e-5)
assert np.allclose(mlx_rope_w, ref_rope_w, atol=1e-5)
```

---

## 9. Multi-Stage / Pyramid Pipelines

### Each stage is a potential failure point

Pyramid pipelines (generate at low res, upsample, refine at high res) multiply the
number of things that can go wrong:

- Downsampling method (bilinear vs area) must match reference
- Energy compensation factors (e.g., ×2 after bilinear downsample) must be present
- Alpha/beta noise mixing coefficients are stage-dependent
- Frame indices and history resolution change per stage

### Test single-stage first

If the model works at full resolution for a single stage but fails in the pyramid,
the bug is in stage orchestration — typically in how latents are passed between
stages or how position encodings adapt to different resolutions.

### Integration bugs are the hardest

We verified every Helios component matched the reference individually, but the
pyramid still produced uniform color. The bug was in dtype handling during stage
transitions. Integration bugs only appear when components interact.

---

## 10. Common Symptoms → Root Causes

| Symptom | Likely Root Causes |
|---------|-------------------|
| **Pure noise output** | Wrong sigma schedule, broken text conditioning, incorrect weight mapping |
| **Uniform color** | Model predicting noise back; text embeddings collapsed; wrong timestep format |
| **Progressive zoom/shrink** | bfloat16 residuals truncating high-freq detail; RoPE mismatch across chunks |
| **Brightness jumps at boundaries** | VAE causal warmup; cross-fade blending misaligned content |
| **Color drift across chunks** | Dtype in scheduler step; history normalization missing |
| **Blur at boundaries** | Cross-fade enabled; latent blending; wrong VAE decode order |
| **Grid/checker patterns** | Patchify channel ordering bug; latent blend artifacts |
| **Green/magenta tint** | VAE weight key mismatch; wrong denormalization constants |
| **Mean drift across steps** | bfloat16 accumulation; wrong scheduler formula; missing energy compensation |

---

## 11. Verification Checklist

Use this checklist when porting a new diffusion video model:

### Model
- [ ] Weight conversion: all keys mapped, cosine similarity > 0.9999
- [ ] Single forward pass matches reference (cos_sim > 0.999)
- [ ] Residual connections use float32 accumulation
- [ ] Attention computation matches reference precision

### Scheduler
- [ ] Sigma values match reference at every step (diff < 1e-6)
- [ ] Timestep format correct (int vs float, scale factor)
- [ ] Dynamic shifting formula copied exactly
- [ ] Step function returns correct dtype (float32)

### Text Encoder
- [ ] Embedding statistics reasonable (0.01 < std < 10)
- [ ] Text influence > 0 (ablation test)
- [ ] Tokenization matches (special tokens, padding, max length)

### VAE
- [ ] Denormalization constants match training pipeline
- [ ] Per-chunk decoding (not concatenated)
- [ ] Temporal frame count correct (account for causal padding)
- [ ] Weight keys mapped correctly (encoder vs decoder)

### Pipeline Orchestration
- [ ] Position encodings consistent across stages/chunks
- [ ] History slicing and conditioning correct
- [ ] Noise generation matches (distribution, correlation structure)
- [ ] Multi-chunk output visually consistent (no progressive degradation)

### Output
- [ ] Frame count matches expected (account for warmup/trim)
- [ ] FPS correct
- [ ] Color range [0, 255] uint8 for video
- [ ] No first-frame duplication artifacts

---

## 12. Diagnostic Tools

### General video diagnostics (`scripts/video/`)

| Script | Purpose |
|--------|---------|
| `compare_videos.py` | PSNR, SSIM, temporal coherence, color fidelity between two videos |
| `video_quality.py` | Sharpness, stability, defect detection, chunk boundary analysis |

```bash
# Quick quality check
python scripts/video/video_quality.py output.mp4 --chunk-size 32

# Compare against reference
python scripts/video/compare_videos.py reference.mp4 output.mp4 --diff-video diff.mp4
```

### Model-specific diagnostics (`scripts/helios/`)

| Script | Purpose |
|--------|---------|
| `analyze_boundaries.py` | Detailed boundary quality metrics for Helios |
| `run_reference.py` | Run PyTorch reference on MPS |
| `compare_pipelines.py` | Compare scheduler/pipeline mechanics |
| `compare_models.py` | Cross-framework model output comparison |

### Inline debugging pattern

Add temporary debug output to the diffusion loop:

```python
for i, sigma in enumerate(sigmas):
    flow = model(latent, sigma, text_emb)
    latent = scheduler.step(latent, flow, sigma, sigma_next)

    # Debug: track statistics
    print(f"[step {i}] sigma={sigma:.4f} "
          f"latent: mean={latent.mean():.6f} std={latent.std():.6f} "
          f"flow: mean={flow.mean():.6f} std={flow.std():.6f}")

    # Debug: save for cross-framework comparison
    if os.environ.get("DEBUG"):
        mx.save(f"/tmp/debug_step_{i}.npz", {
            "latent": latent, "flow": flow, "sigma": mx.array(sigma)
        })
```

---

## Key Takeaways

1. **Precision is the #1 bug source** — bfloat16 residuals, scheduler math, type
   promotion. Copy the reference's `.float()` patterns exactly.

2. **Don't add what the reference doesn't have** — cross-fade, overlap decode,
   temporal blending. If the reference works without it, you probably have a bug
   elsewhere.

3. **Component isolation → integration testing** — verify each part matches, then
   debug their interaction.

4. **Statistical comparison beats visual inspection** — mean drift, contrast ratios,
   and cosine similarity catch bugs before they're visible.

5. **Autoregressive errors compound** — a 1% error per chunk becomes 10% by chunk 10.
   Fix precision first, add corrections second.
