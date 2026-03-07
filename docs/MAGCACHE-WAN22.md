# MagCache for Wan2.2

[MagCache](https://github.com/ali-vilab/MagCache) (Magnitude-Aware Cache) is a training-free acceleration technique for diffusion models. It speeds up inference by skipping transformer blocks when the **magnitude ratio** of residuals between consecutive steps indicates minimal change — providing ~1.5–2× speedup with minimal quality loss.

**Paper**: [MagCache: Magnitude-Aware Cache for Diffusion Sampling Acceleration](https://arxiv.org/abs/2506.09045)
**Source**: [ali-vilab/MagCache](https://github.com/ali-vilab/MagCache) (reference implementation)

## Why MagCache over TeaCache?

Both MagCache and [TeaCache](teacache.md) skip transformer blocks by caching residuals, but they use different signals to decide *when* to skip:

| | TeaCache | MagCache |
|---|---|---|
| **Signal** | L2 distance of timestep embeddings | Magnitude ratio of residual norms |
| **Polynomial rescaling** | Required (model-specific coefficients) | Not needed (uses pre-calibrated ratios) |
| **Wan2.2 stability** | Unstable — MoE expert switching causes erratic L2 distances | Stable — magnitude ratios remain smooth even during expert fluctuations |
| **Profiling** | Must profile polynomial coefficients per model | Pre-calibrated ratios included; can also self-calibrate |

**Bottom line**: MagCache is the recommended acceleration method for Wan2.2. TeaCache remains available for Wan2.1 models.

## How It Works

### The Core Idea

At each denoising step, the transformer computes a residual: `residual = transformer_output - transformer_input`. MagCache tracks the **ratio of residual norms** between consecutive steps. When this ratio stays close to 1.0 (meaning the residual hasn't changed much), the current step can reuse the cached residual instead of running the transformer.

Unlike TeaCache (which computes a live distance metric each step), MagCache uses **pre-calibrated ratios** — magnitude ratios recorded during a prior calibration run. This eliminates the need for polynomial rescaling and makes skip decisions a simple Python float comparison with zero GPU overhead.

### Skip Logic

```
for each denoising step:
    if step is in retention period (first 20% of steps):
        always compute (early steps are most impactful)
    else:
        ratio = pre_calibrated_ratios[step]
        accumulated_ratio *= ratio
        skip_error = |1 - accumulated_ratio|
        accumulated_error += skip_error

        if accumulated_error < threshold AND consecutive_skips < K:
            skip: reuse cached residual (x += cached_residual)
        else:
            compute: run all transformer blocks, cache new residual
            reset accumulators
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--magcache-thresh` | 0.06 | Accumulated error threshold. Lower = better quality, fewer skips |
| `--magcache-K` | 2 | Max consecutive skip steps. Higher = more aggressive |
| `--magcache-retention-ratio` | 0.2 | Fraction of early steps that always compute |

## Usage

### Standalone MagCache (~1.5–2× speedup)

```bash
python -m mlx_video.generate_wan \
    --model-dir wan22_mlx \
    --prompt "A cat playing piano" \
    --magcache
```

### With Spectrum (hybrid, ~2.5–3.5× speedup)

MagCache is compatible with [Spectrum](SPECTRUM-WAN22.md) for additional acceleration. In hybrid mode:

- **Phase 1 (warmup)**: MagCache accelerates Spectrum's warmup period
- **Phase 2 (post-warmup)**: Spectrum predicts features; MagCache acts as a gatekeeper that vetoes unsafe Spectrum predictions when the magnitude ratio deviates significantly

```bash
python -m mlx_video.generate_wan \
    --model-dir wan22_mlx \
    --prompt "A cat playing piano" \
    --magcache --spectrum
```

### Tuning Quality vs Speed

```bash
# Conservative (high quality, ~1.3× speedup)
--magcache --magcache-thresh 0.03

# Balanced (good quality, ~1.5–2× speedup) — default
--magcache --magcache-thresh 0.06

# Aggressive (faster, some quality loss)
--magcache --magcache-thresh 0.10 --magcache-K 3
```

### Verbose Diagnostics

```bash
# Print per-step skip/compute decisions
--magcache --magcache-verbose
```

## Calibration

MagCache comes with pre-calibrated magnitude ratios for all Wan2.2 14B variants (T2V, I2V) and TI2V-5B. These built-in ratios work well out of the box.

However, you may want to **calibrate custom ratios** when:

- Using **LoRAs** that shift the model's feature distribution
- Running at a specific step count you want optimal ratios for
- Extending MagCache to the **low-noise model** in dual-model mode (built-in ratios only cover the high-noise model)

### Step 1: Run Calibration

Calibration runs a full-quality generation (no skipping) while recording magnitude ratios at each step. The video output is identical to a normal run.

```bash
python -m mlx_video.generate_wan \
    --model-dir wan22_mlx \
    --prompt "A cat playing piano" \
    --steps 25 \
    --magcache-calibrate \
    -o calibration_output.mp4
```

This produces a JSON file (e.g., `magcache_ratios_t2v_25steps.json`) in the same directory as the output video.

For dual-model Wan2.2 (T2V-14B, I2V-14B), calibration records ratios for **both** the high-noise and low-noise models.

### Step 2: Use Custom Ratios

```bash
python -m mlx_video.generate_wan \
    --model-dir wan22_mlx \
    --prompt "A different prompt" \
    --magcache --magcache-ratios magcache_ratios_t2v_25steps.json
```

When the calibration file contains both `high_noise` and `low_noise` entries, MagCache is automatically enabled on **both** models — unlocking additional acceleration on the low-noise model that built-in ratios don't cover.

### Calibration File Format

```json
{
  "model_type": "t2v",
  "model_version": "2.2",
  "steps": 25,
  "high_noise": {
    "steps": 9,
    "ratios": [0.99822, 0.99696, ...]
  },
  "low_noise": {
    "steps": 16,
    "ratios": [0.99456, 0.99321, ...]
  }
}
```

For single-model variants (TI2V-5B), the key is `"model"` instead of `"high_noise"` / `"low_noise"`.

### Step Count Interpolation

Calibrated ratios are automatically interpolated (nearest-neighbor) when used with a different step count than what they were calibrated at. For example, ratios calibrated at 40 steps work at 25 steps. That said, calibrating at the exact step count you use gives the most accurate ratios.

## Supported Models

| Model | Built-in Ratios | Calibration |
|-------|:-:|:-:|
| Wan2.2 T2V-14B (high-noise) | ✅ | ✅ |
| Wan2.2 T2V-14B (low-noise) | — | ✅ |
| Wan2.2 I2V-14B (high-noise) | ✅ | ✅ |
| Wan2.2 I2V-14B (low-noise) | — | ✅ |
| Wan2.2 TI2V-5B (t2v mode) | ✅ | ✅ |
| Wan2.2 TI2V-5B (i2v mode) | ✅ | ✅ |
| Wan2.1 models | — | ✅ |

Models without built-in ratios require `--magcache-calibrate` first, then `--magcache-ratios`.

## Hybrid MagCache + Spectrum: How It Works

When both `--magcache` and `--spectrum` are enabled, the system operates in two phases:

### Phase 1: Warmup Acceleration

During Spectrum's warmup period (first N steps where it builds its Chebyshev polynomial model), MagCache decides whether to skip or compute. Skipped steps feed approximate features into Spectrum's cache; computed steps feed real features. This makes Spectrum's "cold start" period ~40% faster.

### Phase 2: Adaptive Gatekeeper

After warmup, Spectrum controls the prediction schedule. Before each Spectrum skip, MagCache checks the pre-calibrated magnitude ratio for the current step:

- If the ratio is within [0.95, 1.05]: Spectrum skip is allowed
- If the ratio deviates beyond that band: MagCache **vetoes** the skip, forcing full computation

This prevents Spectrum from predicting features during steps where the model is making substantial changes (e.g., MoE expert switching, sudden motion changes), avoiding "smearing" artifacts.

### Mutual Exclusivity

| Combination | Supported | Notes |
|---|:-:|---|
| MagCache + Spectrum | ✅ | Hybrid mode (recommended) |
| MagCache + TeaCache | ❌ | Both cache residuals — mutually exclusive |
| Spectrum + TeaCache | ❌ | Mutually exclusive |
| MagCache alone | ✅ | Standalone mode |
| Spectrum alone | ✅ | Standalone mode |

## Implementation Details

### Adaptation for MLX

The reference implementation uses interleaved conditional/unconditional forward passes (`cnt%2` tracking). mlx-video batches both into a single B=2 forward call, so:

- A single residual cache serves the full `[B, seq_len, dim]` tensor
- Pre-calibrated ratios are averaged from cond/uncond pairs into one ratio per step
- All skip decisions are pure Python (zero GPU sync overhead)

### Dual-Model Handling

Wan2.2 14B uses two transformer models split at a noise boundary:
- **High-noise model**: Handles early denoising steps (larger sigma values)
- **Low-noise model**: Handles later refinement steps (smaller sigma values)

Each model has its own independent `MagCacheState`. Built-in ratios only cover the high-noise model. Use calibration to extend to both.

### Files

| File | Role |
|------|------|
| `mlx_video/models/wan/magcache.py` | Core module: `MagCacheState`, pre-calibrated ratios, calibration logic |
| `mlx_video/models/wan/model.py` | Integration in `WanModel.__call__()`: calibration, standalone, and hybrid forward paths |
| `mlx_video/generate_wan.py` | CLI flags, configuration, stats reporting, calibration JSON save/load |
| `mlx_video/models/wan/config.py` | `magcache_ratios_key` field on `WanModelConfig` |
