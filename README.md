# mlx-video

MLX-Video is the best package for inference and finetuning of Image-Video-Audio generation models on your Mac using MLX.

## Installation

Install from source:

### Option 1: Install with pip (requires git):
```bash
pip install git+https://github.com/Blaizzy/mlx-video.git
```

### Option 2: Install with uv (ultra-fast package manager, optional):
```bash
uv pip install git+https://github.com/Blaizzy/mlx-video.git
```

Supported models:

- [**LTX-2**](https://huggingface.co/Lightricks/LTX-Video) — 19B parameter video generation model from Lightricks
- [**Wan2.1**](https://github.com/Wan-Video/Wan2.1) — 1.3B / 14B parameter T2V models (single-model pipeline)
- [**Wan2.2**](https://github.com/Wan-Video/Wan2.2) — 14B parameter T2V model (dual-model pipeline)

## Features

- Text-to-video generation with multiple model families
- LTX-2: Two-stage pipeline with 2x spatial upscaling
- Wan2.1/2.2: Flow-matching diffusion with classifier-free guidance
- Optimized for Apple Silicon using MLX

---

## LTX-2

> **ℹ️ Info:** Currently, only the distilled variant is supported. Full LTX-2 feature support is coming soon.

### Text-to-Video Generation

```bash
uv run mlx_video.generate --prompt "Two dogs of the poodle breed wearing sunglasses, close up, cinematic, sunset" -n 100 --width 768
```

<img src="https://github.com/Blaizzy/mlx-video/raw/main/examples/poodles.gif" width="512" alt="Poodles demo">

With custom settings:

```bash
python -m mlx_video.generate \
    --prompt "Ocean waves crashing on a beach at sunset" \
    --height 768 \
    --width 768 \
    --num-frames 65 \
    --seed 123 \
    --output my_video.mp4
```

### LTX-2 CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt`, `-p` | (required) | Text description of the video |
| `--height`, `-H` | 512 | Output height (must be divisible by 64) |
| `--width`, `-W` | 512 | Output width (must be divisible by 64) |
| `--num-frames`, `-n` | 100 | Number of frames |
| `--seed`, `-s` | 42 | Random seed for reproducibility |
| `--fps` | 24 | Frames per second |
| `--output`, `-o` | output.mp4 | Output video path |
| `--save-frames` | false | Save individual frames as images |
| `--model-repo` | Lightricks/LTX-2 | HuggingFace model repository |

### How It Works (LTX-2)

1. **Stage 1**: Generate at half resolution (e.g., 384×384) with 8 denoising steps
2. **Upsample**: 2× spatial upsampling via LatentUpsampler
3. **Stage 2**: Refine at full resolution (e.g., 768×768) with 3 denoising steps
4. **Decode**: VAE decoder converts latents to RGB video

---

## Wan2.1 / Wan2.2

Both [Wan2.1](https://github.com/Wan-Video/Wan2.1) and [Wan2.2](https://github.com/Wan-Video/Wan2.2) are text-to-video diffusion models built on a DiT (Diffusion Transformer) backbone with a T5 text encoder and 3D VAE. They share the same model architecture — the difference is in the inference pipeline:

| | Wan2.1 | Wan2.2 |
|---|--------|--------|
| **Pipeline** | Single model | Dual model (high-noise + low-noise) |
| **Sizes** | 1.3B, 14B | 14B |
| **Steps** | 50 | 40 |
| **Guidance** | 5.0 (fixed) | 3.0 / 4.0 (low/high noise) |
| **Shift** | 5.0 | 12.0 |

### Step 1: Download Weights

Download the original PyTorch checkpoints:

**Wan2.1 (14B)**
```bash
# From https://github.com/Wan-Video/Wan2.1 or HuggingFace
# Expected directory structure:
# wan21_checkpoint/
#   ├── models_t5_umt5-xxl-enc-bf16.pth
#   ├── Wan2.1_VAE.pth
#   └── diffusion_pytorch_model*.safetensors   # single model
```

**Wan2.1 (1.3B)** — same structure, smaller transformer weights.

**Wan2.2 (14B)**
```bash
# From https://github.com/Wan-Video/Wan2.2 or HuggingFace
# Expected directory structure:
# wan22_checkpoint/
#   ├── models_t5_umt5-xxl-enc-bf16.pth
#   ├── Wan2.1_VAE.pth
#   ├── low_noise_model/   # safetensors
#   └── high_noise_model/  # safetensors
```

### Step 2: Convert to MLX Format

The conversion script auto-detects whether the checkpoint is Wan2.1 or Wan2.2 based on the directory structure (presence of `low_noise_model/` subdirectory).

```bash
# Auto-detect version
python -m mlx_video.convert_wan \
    --checkpoint-dir /path/to/wan_checkpoint \
    --output-dir wan_mlx

# Explicit version
python -m mlx_video.convert_wan \
    --checkpoint-dir /path/to/wan21_checkpoint \
    --output-dir wan21_mlx \
    --model-version 2.1

python -m mlx_video.convert_wan \
    --checkpoint-dir /path/to/wan22_checkpoint \
    --output-dir wan22_mlx \
    --model-version 2.2
```

#### Conversion Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint-dir` | (required) | Path to original PyTorch checkpoint directory |
| `--output-dir` | `wan_mlx_model` | Output path for MLX model |
| `--dtype` | `bfloat16` | Target dtype (`float16`, `float32`, `bfloat16`) |
| `--model-version` | `auto` | Model version: `2.1`, `2.2`, or `auto` |
| `--quantize` | off | Quantize transformer weights for reduced memory |
| `--bits` | `4` | Quantization bits: `4` or `8` |
| `--group-size` | `64` | Quantization group size: `32`, `64`, or `128` |

The converter produces:
```
wan_mlx/
├── config.json                    # Model configuration
├── t5_encoder.safetensors         # T5 UMT5-XXL text encoder
├── vae.safetensors                # 3D VAE decoder
├── model.safetensors              # (Wan2.1) Single transformer
├── low_noise_model.safetensors    # (Wan2.2) Low-noise transformer
└── high_noise_model.safetensors   # (Wan2.2) High-noise transformer
```

### Step 3: Generate Video

```bash
# Wan2.1 — uses defaults from config (50 steps, shift=5.0, guide=5.0)
python -m mlx_video.generate_wan \
    --model-dir wan21_mlx \
    --prompt "A cat playing piano in a cozy room"

# Wan2.2 — uses defaults from config (40 steps, shift=12.0, guide=3.0,4.0)
python -m mlx_video.generate_wan \
    --model-dir wan22_mlx \
    --prompt "A cat playing piano in a cozy room"
```

With custom settings:

```bash
python -m mlx_video.generate_wan \
    --model-dir wan21_mlx \
    --prompt "Ocean waves at sunset, cinematic, 4K" \
    --negative-prompt "blurry, low quality" \
    --width 1280 \
    --height 720 \
    --num-frames 81 \
    --steps 50 \
    --guide-scale 5.0 \
    --shift 5.0 \
    --seed 42 \
    --output-path my_video.mp4
```

The pipeline auto-detects the model version from `config.json` and selects the right pipeline mode (single or dual model). You can also override any parameter via CLI flags.

#### Generation CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-dir` | (required) | Path to converted MLX model directory |
| `--prompt` | (required) | Text description of the video |
| `--negative-prompt` | `""` | Negative prompt for guidance |
| `--width` | 1280 | Video width |
| `--height` | 720 | Video height |
| `--num-frames` | 81 | Number of frames (must be 4n+1) |
| `--steps` | from config | Number of diffusion steps |
| `--guide-scale` | from config | Guidance scale: float or `low,high` pair |
| `--shift` | from config | Noise schedule shift |
| `--seed` | -1 (random) | Random seed for reproducibility |
| `--output-path` | `output.mp4` | Output video path |

### Quantization (Reduced Memory)

Quantize the transformer weights to reduce memory usage by ~3.4x. This is especially useful for the 14B model or memory-constrained devices:

```bash
# Convert with 4-bit quantization
python -m mlx_video.convert_wan \
    --checkpoint-dir /path/to/Wan2.1-T2V-1.3B \
    --output-dir wan21_mlx_q4 \
    --quantize --bits 4 --group-size 64

# Generate with quantized model (auto-detected from config.json)
python -m mlx_video.generate_wan \
    --model-dir wan21_mlx_q4 \
    --prompt "A cat playing piano"
```

**What gets quantized**: Self-attention (Q/K/V/O), cross-attention (Q/K/V/O), and FFN (fc1/fc2) — 10 layers × N blocks = ~95% of model weights. Embeddings, norms, and the output head remain in bfloat16 for precision.

| Model | BF16 Size | 4-bit Size | Notes |
|-------|-----------|------------|-------|
| 1.3B | 2.7 GB | 799 MB | ~3.4x smaller |
| 14B | ~28 GB | ~8 GB | Enables running on 16GB devices |

> **Note**: On Apple Silicon, the 1.3B model fits comfortably in unified memory at bf16. Quantization reduces memory but may not speed up inference for small models. For the 14B model, quantization is essential to fit in memory and will also improve speed.

### Wan Model Specifications

**Transformer (14B)**
- 40 layers, 40 attention heads, dim 5120, head dim 128
- 3-way factorized RoPE (temporal + spatial)
- 14.29B parameters

**Transformer (1.3B, Wan2.1 only)**
- 30 layers, 12 attention heads, dim 1536, head dim 128
- Same architecture, smaller scale

**Text Encoder** — UMT5-XXL (5.68B parameters)
- 24 layers, 64 heads, dim 4096, vocab 256K

**VAE** — 3D causal convolution decoder (72.6M parameters)
- Latent channels: 16
- Compression: 4× temporal, 8× spatial

---

## Requirements

- macOS with Apple Silicon
- Python >= 3.11
- MLX >= 0.22.0
- For weight conversion: PyTorch (`pip install torch`)

## Project Structure

```
mlx_video/
├── generate.py              # LTX-2 generation pipeline
├── generate_wan.py          # Wan2.1/2.2 generation pipeline
├── convert.py               # LTX-2 weight conversion
├── convert_wan.py           # Wan weight conversion (PyTorch → MLX)
├── postprocess.py           # Video post-processing utilities
├── utils.py                 # Helper functions
└── models/
    ├── ltx/                 # LTX-2 model
    │   ├── ltx.py           # DiT transformer
    │   ├── config.py        # Configuration
    │   ├── transformer.py   # Transformer blocks
    │   ├── attention.py     # Multi-head attention with RoPE
    │   ├── text_encoder.py  # Gemma 3 text encoder
    │   ├── upsampler.py     # 2x spatial upsampler
    │   └── video_vae/       # VAE encoder/decoder
    └── wan/                 # Wan2.1/2.2 model
        ├── config.py        # Configuration (2.1 & 2.2 presets)
        ├── model.py         # WanModel (DiT transformer)
        ├── transformer.py   # Attention blocks with 6-element modulation
        ├── attention.py     # Self/cross attention with QK-norm
        ├── rope.py          # 3-way factorized RoPE
        ├── text_encoder.py  # T5 UMT5-XXL encoder
        ├── vae.py           # 3D causal VAE decoder
        └── scheduler.py     # Flow-matching Euler scheduler
```

## License

MIT
