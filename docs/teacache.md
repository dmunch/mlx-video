# TeaCache for Wan2.1/2.2

[TeaCache](https://github.com/ali-vilab/TeaCache) (Timestep Embedding Aware Cache) is a training-free acceleration technique for diffusion models. It speeds up inference by skipping transformer blocks when consecutive diffusion steps produce similar outputs — providing up to ~2–3× speedup with minimal quality loss.

**Paper**: [Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model](https://arxiv.org/abs/2411.19108)
**Source**: [ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache) (official implementation)

## How It Works

During diffusion, most consecutive steps produce very similar intermediate results — the transformer blocks do nearly identical work. TeaCache detects this by:

1. Computing the **relative L1 distance** between consecutive timestep embeddings (`e0`)
2. Applying a **polynomial rescaling** to map that distance to expected output change
3. Accumulating the rescaled distance; when it stays below a threshold, **reuse the cached residual** instead of running all transformer blocks

The key insight is that timestep embedding distance is a cheap proxy for output distance — no need to actually compute the output to know it hasn't changed much.

### Polynomial Rescaling

The relationship between embedding distance and output distance is nonlinear and model-specific. A 4th-degree polynomial maps one to the other:

```
rescaled_distance = c₀x⁴ + c₁x³ + c₂x² + c₃x + c₄
```

where `x` is the relative L1 distance between consecutive `e0` embeddings. The coefficients are profiled once per model variant (see [Profiling](#profiling-coefficients-for-new-models) below).

### Skip Logic

```
for each diffusion step:
    if step is in first 2 or last 2:
        always compute (these change the most)
    else:
        compute rel_l1 = |e0 - prev_e0|.mean() / |prev_e0|.mean()
        rescaled = polynomial(rel_l1)
        accumulated_distance += rescaled
        if accumulated_distance < threshold:
            skip: reuse cached residual (x += prev_residual)
        else:
            compute: run all transformer blocks, cache new residual
            reset accumulated_distance to 0
```

## Usage

```bash
# ~2x speedup (conservative, near-lossless)
python -m mlx_video.generate_wan \
    --model-dir wan22_mlx \
    --prompt "A cat playing piano" \
    --teacache-thresh 0.1

# ~3x speedup (good quality/speed tradeoff)
python -m mlx_video.generate_wan \
    --model-dir wan22_mlx \
    --prompt "A cat playing piano" \
    --teacache-thresh 0.2
```

Set `--teacache-thresh 0` (the default) to disable.

## Supported Models & Recommended Thresholds

| Model | Coefficients | Conservative | Fast | Source |
|-------|:---:|:---:|:---:|--------|
| Wan T2V-14B (2.1 & 2.2) | ✅ | 0.1 (~2×) | 0.2 (~3×) | Official TeaCache4Wan2.1 |
| Wan T2V-1.3B (2.1) | ✅ | 0.05 (~1.5×) | 0.08 (~2×) | Official TeaCache4Wan2.1 |
| Wan TI2V-5B (2.2) | ❌ | — | — | Needs profiling |

When TeaCache is used with an unsupported model, it prints a warning and falls back to full computation.

## Pre-Profiled Coefficients

From the [official TeaCache4Wan2.1 implementation](https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4Wan2.1/teacache_generate.py):

**T2V-14B** (Wan2.1 & Wan2.2, 40 layers, dim=5120):
```python
(-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404)
```

**T2V-1.3B** (Wan2.1, 30 layers, dim=1536):
```python
(2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01)
```

**I2V-14B 480P** (Wan2.1):
```python
(-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01)
```

**I2V-14B 720P** (Wan2.1):
```python
(-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683)
```

Format: `np.poly1d` convention (highest degree first). Stored in `mlx_video/models/wan/config.py` as the `teacache_coefficients` field.

## Profiling Coefficients for New Models

To enable TeaCache on an unsupported model variant (e.g. Wan2.2 TI2V-5B), you need to profile model-specific polynomial coefficients. This is a one-time process that runs the model without caching to measure the relationship between input and output distances.

### What You Need

- **20+ diverse text prompts** — the official TeaCache paper uses 70 prompts from [T2V-CompBench](https://github.com/KaiyueSun98/T2V-CompBench) (7 categories × 10 prompts). You can download prompts from the [T2V-CompBench prompts directory](https://github.com/KaiyueSun98/T2V-CompBench/tree/main/prompts). More diverse prompts = more robust coefficients.
- **Full inference runs** — each prompt runs all diffusion steps without caching.
- **A few hours** — for a 14B model with 50 steps and 20 prompts, expect ~2–4 hours on Apple Silicon. The polynomial fitting itself takes milliseconds.

### Step-by-Step Process

1. **Instrument the forward pass** to collect `(input_distance, output_distance)` pairs at every diffusion step:

```python
import numpy as np

all_input_dists = []
all_output_dists = []

# For each prompt, run full inference and record at each step:
prev_e0 = None
prev_output = None

for step in diffusion_steps:
    e0 = compute_timestep_embedding(timestep)
    output = run_all_transformer_blocks(x, e0, ...)

    if prev_e0 is not None:
        # Input distance: relative L1 of timestep embeddings
        input_dist = abs(e0 - prev_e0).mean() / abs(prev_e0).mean()
        # Output distance: relative L1 of block outputs
        output_dist = abs(output - prev_output).mean() / abs(prev_output).mean()

        all_input_dists.append(input_dist)
        all_output_dists.append(output_dist)

    prev_e0 = e0
    prev_output = output
```

2. **Fit a 4th-degree polynomial**:

```python
coefficients = np.polyfit(all_input_dists, all_output_dists, 4)
print(tuple(coefficients))
# e.g. (-5784.54, 5449.51, -1811.17, 256.27, -13.02)
```

3. **Add the coefficients** to the model's config constructor in `mlx_video/models/wan/config.py`:

```python
@classmethod
def wan22_ti2v_5b(cls) -> "WanModelConfig":
    return cls(
        ...
        teacache_coefficients=(
            # Your profiled coefficients here
            -302.33, 223.95, -52.55, 5.87, -0.20,
        ),
    )
```

4. **Validate** by generating a few videos with and without TeaCache and comparing quality.

### Reference

The official profiling process and coefficient discussion can be found at:
- [TeaCache Issue #2: How to get coefficients?](https://github.com/ali-vilab/TeaCache/issues/2)
- [Official source script (TeaCache4Wan2.1)](https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4Wan2.1/teacache_generate.py)
- [TeaCache paper](https://arxiv.org/abs/2411.19108)

## Implementation Details

### Our Adaptations for MLX

The official TeaCache implementation (PyTorch/CUDA) tracks conditional and unconditional predictions separately with even/odd counters (`cnt % 2`). Our MLX implementation simplifies this because we use **batched CFG** — both conditional and unconditional predictions are computed in a single B=2 forward pass with the same timestep, so they share the same `e0`. We track a single cache state per model.

### Dual-Model Support (Wan2.2)

Wan2.2 uses two separate transformer models (high-noise and low-noise) that switch at a boundary timestep. Each model has its own independent `TeaCacheState` — the cache is never shared across models. Stats are aggregated from both models when reporting.

### Code Locations

- `mlx_video/models/wan/model.py` — `TeaCacheState` dataclass and skip logic in `WanModel.__call__`
- `mlx_video/models/wan/config.py` — Per-model polynomial coefficients
- `mlx_video/generate_wan.py` — CLI flag and TeaCache configuration
- `tests/test_wan_teacache.py` — Unit and integration tests
