# Spectrum for Wan2.2 — Research & Feasibility

[Spectrum](https://github.com/hanjq17/Spectrum) (Adaptive Spectral Feature Forecasting) is a training-free diffusion acceleration technique from Stanford/ByteDance (CVPR 2026). It speeds up inference by **predicting transformer features using Chebyshev polynomials** instead of running full forward passes at every denoising step — achieving up to **4.67× speedup on Wan2.1-14B** with minimal quality loss.

**Paper**: [Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](https://arxiv.org/abs/2603.01623)
**Source**: [hanjq17/Spectrum](https://github.com/hanjq17/Spectrum) (reference implementation)

## How Spectrum Works

### The Core Idea

During diffusion sampling, the denoiser's hidden features evolve smoothly across timesteps. Instead of running the expensive 40-layer transformer at every step, Spectrum:

1. Runs the **full forward pass** at selected timesteps, caching the transformer output
2. **Fits Chebyshev polynomials** to the cached features via ridge regression
3. At remaining timesteps, **predicts** features using the fitted polynomials — skipping the entire transformer

This is similar in spirit to [TeaCache](teacache.md) (which also skips transformer blocks), but fundamentally different in approach: TeaCache reuses the *previous step's residual* (constant extrapolation), while Spectrum fits a *global polynomial model* across all cached timesteps and extrapolates along it.

### Why Chebyshev Polynomials?

Previous approaches like TaylorSeer use local Taylor expansion (finite differences from the few most recent steps). This works for small skips but the error **compounds rapidly** with the skip size — a known limitation of local approximation.

Chebyshev polynomials have a critical advantage: they are **global orthonormal bases** on [-1, 1]. When you fit M Chebyshev bases to the feature trajectory, the approximation error is bounded by the degree M **regardless of the prediction distance**:

```
Error ≤ (2B)/(ρ-1) × ρ^(-M)     (Chebyshev)
  vs.
Error ∝ (step_size)^(P+1)         (Taylor)
```

This means Spectrum can skip more steps with less quality degradation, enabling higher speedup ratios (3.5–5×) where Taylor-based methods break down.

### Algorithm Overview

```
For each denoising step i = 1..N:
    if should_compute(i):              ← determined by adaptive schedule
        h_i = transformer_blocks(x_i)  ← full 40-layer forward pass
        update_cache(h_i, t_i)          ← store features + timestep
        fit_chebyshev(cache)            ← ridge regression: C = (Φᵀ Φ + λI)⁻¹ Φᵀ H
    else:
        h_i = predict(C, t_i)           ← single matmul: φ(τ_i) @ C

    output = head(h_i)                  ← lightweight output projection (always runs)
    x_{i+1} = solver_step(output)
```

### Ridge Regression Fitting

Given K cached features at timesteps t₁..tₖ, construct:
- **Design matrix** Φ ∈ ℝ^(K × (M+1)): each row evaluates Chebyshev T₀..Tₘ at the normalized timestep τₖ
- **Feature matrix** H ∈ ℝ^(K × F): cached features (flattened transformer output)

Solve for coefficient matrix C ∈ ℝ^((M+1) × F):
```
C = (Φᵀ Φ + λI)⁻¹ Φᵀ H
```

The (M+1 × M+1) system (typically 5×5 for M=4) is solved via Cholesky decomposition. The dominant cost is the `Φᵀ @ H` matmul, which is (5 × K) @ (K × F) — fast on GPU.

### Spectrum Blend

The final implementation blends Chebyshev prediction with discrete Taylor (Newton forward differences) for robustness:

```
h_predicted = (1 - w) × h_taylor + w × h_chebyshev
```

where `w ∈ [0.5, 1.0]` controls the blend. The authors found `w=0.5` slightly enhances robustness across acceleration ratios.

### Adaptive Scheduling

Not all timesteps are equally important. Spectrum uses an adaptive schedule that:
- **Computes more at early steps** (features change rapidly at high noise)
- **Predicts more at later steps** (features stabilize as noise decreases)

This is controlled by two parameters:
- `window_size` (𝒩): initial gap between compute steps
- `flex_window` (α): how quickly the gap grows

At each compute step, the window grows by α:
```python
if (num_consecutive_cached + 1) % floor(current_window) == 0:
    do_actual_forward = True
    current_window += flex_window
```

Typical settings:
- `window_size=2, flex_window=0.75` → 14 forward passes out of 50 steps (**3.5× speedup**)
- `window_size=2, flex_window=3.0` → 10 forward passes out of 50 steps (**5× speedup**)

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | 4 | Number of Chebyshev basis polynomials |
| `λ` (lam) | 0.1 | Ridge regression regularization strength |
| `w` | 0.5 | Blend weight (0=Taylor only, 1=Chebyshev only) |
| `warmup_steps` | 5 | Always compute the first N steps (build up cache) |
| `window_size` | 2 | Initial compute interval |
| `flex_window` | 0.75 | Window growth rate per compute step |
| `K` | 100 | Maximum cache size (history buffer) |

## Mapping to Wan2.2 in mlx-video

### Architecture Alignment

The Spectrum reference implementation supports Wan2.1-14B via the diffusers `WanPipeline`. The mlx-video Wan2.2 model has an identical transformer architecture:

| Component | Reference (Wan2.1, PyTorch) | mlx-video (Wan2.2, MLX) |
|-----------|---------------------------|------------------------|
| Transformer blocks | `self.blocks` (40 layers) | `self.blocks` (40 layers) |
| Block type | `WanTransformerBlock3DModel` | `WanAttentionBlock` |
| Hidden dim | 5120 | 5120 |
| Heads | 40 | 40 |
| Output head | Final norm + projection | `self.head` (Head module) |
| Caching point | After final block, before output | After `self.blocks` loop, before `self.head` |

The caching point maps directly:

**Reference** (`wan_forward.py`):
```python
for index_block, block in enumerate(self.blocks):
    hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

feat = hidden_states.reshape(-1, hidden_states.shape[-1]).unsqueeze(0)
step_derivative_approximation(cache_dic, current, feature=feat)
```

**mlx-video** (`model.py`):
```python
for i, block in enumerate(self.blocks):
    kv = cross_kv_caches[i] if cross_kv_caches is not None else None
    x = block(x, cross_kv_cache=kv, **kwargs)

# ← Spectrum cache/predict point
x = self.head(x, e)  # always runs (lightweight)
```

### Wan2.2 vs Wan2.1 Differences

| Feature | Wan2.1 (Spectrum tested) | Wan2.2 (mlx-video) |
|---------|-------------------------|-------------------|
| Model architecture | Single model, 40 layers | **Dual model** (high-noise + low-noise), each 40 layers |
| VAE | z_dim=16, stride=(4,8,8) | z_dim=48, stride=(4,16,16) |
| Default steps | 50 | 40 |
| Guidance | Single scale (5.0) | Dual scale (3.0 low, 4.0 high) |
| Scheduler shift | 5.0 | 12.0 |
| Latent tokens | ~32,760 (480p) | ~8,190 (480p, due to larger VAE stride) |

Key implications:
- **Smaller latent space** (8,190 vs 32,760 tokens) → Spectrum's feature vectors are ~4× smaller → less memory, faster fit/predict
- **Dual model** → need separate Spectrum state per model
- **Fewer default steps** (40 vs 50) → slightly less room for aggressive skipping

### Feature Vector Dimensions

For 480×832 at 81 frames with Wan2.2:
```
Latent: 48 channels, 21 temporal, 30 height, 52 width
After patchify (1,2,2): 21 × 15 × 26 = 8,190 tokens
Hidden state: [B, 8190, 5120]
Flattened per batch element: 8,190 × 5,120 = 41,932,800 features
```

For B=2 (CFG batch): ~84M features total.

## Feasibility Analysis

### ✅ MLX Primitive Availability

All required operations exist in MLX:

| Operation | MLX API | Notes |
|-----------|---------|-------|
| Chebyshev recurrence | Basic arithmetic | `T_m = 2τ × T_{m-1} - T_{m-2}` |
| Design matrix build | `mx.concatenate` | Stack Chebyshev evaluations |
| Matrix multiply | `@` operator | `Φᵀ @ H` — dominant cost |
| Cholesky decomposition | `mx.linalg.cholesky` | **CPU-only** in MLX 0.30.1 (`stream=mx.cpu`) |
| Triangular solve | `mx.linalg.solve_triangular` | For Cholesky back-substitution |
| Direct solve (alternative) | `mx.linalg.solve` | Can replace Cholesky + back-sub |

The 5×5 Cholesky is trivial (microseconds on CPU). The bottleneck is the large matmul `Φᵀ @ H` which runs on GPU/ANE.

### ✅ Performance Benchmarks (Apple Silicon)

Benchmarked at realistic Wan2.2 scale (F ≈ 42M features, 480×832×81f):

| Operation | Time | Context |
|-----------|------|---------|
| Ridge regression fit | ~841ms | Runs once per actual forward pass |
| Chebyshev prediction | ~154ms | Replaces a ~3-5s transformer forward pass |
| Full transformer forward | ~3-5s | 40 attention blocks (what Spectrum skips) |

**Net benefit per predicted step**: Save ~3-5 seconds, pay ~154ms → **~20-30× faster per step**.
The fit cost (~841ms) is amortized over multiple predicted steps.

### ⚠️ Memory Overhead

| Component | Size (bfloat16) | Notes |
|-----------|----------------|-------|
| Coefficient matrix C | ~400 MB | (M+1) × F = 5 × 42M |
| History buffer (K=10) | ~800 MB | K × F = 10 × 42M |
| **Total per model** | **~1.2 GB** | Acceptable on 64GB+ Apple Silicon |

Mitigations:
- Reduce K from 10 to 6 (minimum K ≥ M+2 = 6 for M=4)
- Wan2.2's smaller latent space (vs Wan2.1) already helps: ~42M vs ~168M features
- On 64GB Mac: ~2% of total memory
- On 128GB Mac: ~1% of total memory

### ✅ Integration Pattern (TeaCache Precedent)

mlx-video already has TeaCache following the exact pattern Spectrum needs:

```python
# TeaCache in model.py — same pattern for Spectrum
if self.teacache.enabled:
    if should_skip:
        x = x + tc.previous_residual  # skip transformer blocks
    else:
        for block in self.blocks:     # full forward pass
            x = block(x, ...)
        tc.previous_residual = x - ori_x
```

Spectrum replaces this with:
```python
if self.spectrum.enabled:
    if should_predict:
        x = spectrum.predict(timestep)   # polynomial prediction
    else:
        for block in self.blocks:        # full forward pass
            x = block(x, ...)
        spectrum.update(timestep, x)      # update cache + refit
```

## Challenges & Mitigations

### 1. Dual Model (Wan2.2)

**Problem**: Wan2.2 switches between high-noise and low-noise models at boundary=0.875. Features from one model can't predict the other — there's a discontinuity at the switch point.

**Solution**: Maintain separate `SpectrumState` per model. With 40 steps:
- **High-noise model**: ~35 steps → excellent for Spectrum (many data points, big cache)
- **Low-noise model**: ~5 steps → marginal benefit (consider disabling Spectrum here)

The low-noise model sees very few steps, so Spectrum may not have enough history to fit accurately. A safe default would be to only enable Spectrum for the high-noise model.

### 2. CFG Batch Handling

**Problem**: mlx-video batches conditional + unconditional into B=2 for a single forward pass. The reference Spectrum implementation handles them as separate calls with separate forecasters.

**Options**:
- **Option A (recommended)**: Cache the full B=2 output as one feature vector. Both batch elements see the same timestep, so the combined vector evolves smoothly. Simplest implementation, lowest overhead.
- **Option B**: Split into separate cond/uncond forward passes with separate forecasters. More faithful to reference but loses batching efficiency and doubles Spectrum's memory usage.

### 3. mx.compile Incompatibility

**Problem**: Spectrum's dynamic control flow (compute vs predict decision) prevents `mx.compile`, which traces static computation graphs.

**Solution**: Same approach as TeaCache — disable `mx.compile` when Spectrum is enabled. The speedup from skipping transformer blocks (3-5× fewer forward passes) far exceeds the marginal benefit of compilation.

### 4. I2V (Image-to-Video)

**Problem**: I2V uses per-token timesteps (`[B, L]` instead of `[B]`) and channel-concatenated conditioning. The feature trajectory is more complex.

**Solution**: Start with T2V only. I2V support can be added as a follow-up with careful handling of the per-token timestep structure.

## Implementation Roadmap

### Phase 1: Core Algorithm (`mlx_video/models/wan/spectrum.py`)

Port the core Spectrum classes to MLX:

```python
@dataclass
class SpectrumState:
    enabled: bool = False
    # Hyperparameters
    m: int = 4              # Chebyshev bases
    lam: float = 0.1        # Ridge regularization
    w: float = 0.5          # Chebyshev/Taylor blend
    warmup_steps: int = 5   # Always compute first N steps
    window_size: int = 2    # Initial compute interval
    flex_window: float = 0.75  # Window growth rate
    # State
    forecaster: SpectrumForecaster | None = None
    cnt: int = 0
    curr_ws: float = 2.0
    num_consecutive_cached: int = 0
```

Key classes:
- `ChebyshevForecaster`: Design matrix construction, ridge regression via Cholesky, prediction
- `SpectrumForecaster`: Wraps Chebyshev + discrete Taylor blend
- `SpectrumState`: Configuration and runtime state

### Phase 2: Model Integration (`mlx_video/models/wan/model.py`)

Add Spectrum alongside TeaCache in `WanModel.__call__`:
- Add `SpectrumState` as model attribute
- Implement compute-vs-predict decision using adaptive scheduling
- Cache features after transformer blocks on compute steps
- Predict features on skip steps
- Always run `self.head()` on both real and predicted features

### Phase 3: Pipeline Integration (`mlx_video/generate_wan.py`)

- Add `--spectrum` flag with sub-parameters
- Configure per-model SpectrumState
- Disable `mx.compile` when Spectrum active
- Log skip statistics

### Phase 4: Testing & Quality Validation

- Unit tests for Chebyshev construction and ridge regression
- End-to-end generation with Spectrum enabled
- Quality comparison: same prompt/seed with and without Spectrum
- Wall-clock speedup measurement

## Expected Results

Based on the paper's Wan2.1 benchmarks and accounting for Wan2.2 differences:

| Configuration | NFE (Network Forward Evaluations) | Expected Speedup | Quality Impact |
|---------------|-----------------------------------|-------------------|---------------|
| No caching (baseline) | 40 | 1.0× | Reference |
| Spectrum (conservative) | ~14 | ~2.8× | Minimal |
| Spectrum (moderate) | ~10 | ~4.0× | Low |
| TeaCache (for comparison) | Variable | ~2-3× | Low-moderate |

Wan2.2's smaller latent space (8,190 vs 32,760 tokens) means Spectrum's overhead is proportionally smaller, which could yield even better wall-clock speedups than the paper reports for Wan2.1.

## References

- [Spectrum Paper](https://arxiv.org/abs/2603.01623) — Han et al., CVPR 2026
- [Spectrum Code](https://github.com/hanjq17/Spectrum) — Reference PyTorch implementation
- [TeaCache](teacache.md) — Existing feature caching in mlx-video (different approach)
- [Wan2.2 Implementation Notes](wan22-implementation-notes.md) — Architecture details
