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
    "timestep_sampling": "balanced"
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
    "timestep_sampling": "balanced"
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
