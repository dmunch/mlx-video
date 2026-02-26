# MLX-Video Copilot Instructions

## Overview

MLX-Video is a video/audio generation package using Apple MLX framework. It implements the LTX-2 model (19B parameter DiT) for text-to-video, image-to-video, and audio-video generation, optimized for Apple Silicon.

## Build, Test, and Lint

### Testing
```bash
# Install test dependencies first (pytest not in main deps)
pip install pytest

# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_generate_dev.py

# Run specific test
python -m pytest tests/test_generate_dev.py::TestLTX2Scheduler::test_scheduler_output_shape
```

### Linting
Pre-commit hooks configured with:
- **black**: Code formatting
- **isort**: Import sorting (profile: black)
- **autoflake**: Remove unused imports

```bash
# Run pre-commit manually
pre-commit run --all-files
```

### Running Generation
```bash
# Quick test - distilled model (two-stage pipeline)
python -m mlx_video.generate --prompt "test video" --num-frames 33

# Dev model with CFG (single-stage, higher quality)
python -m mlx_video.generate_dev --prompt "test video" --steps 40 --cfg-scale 4.0

# Audio-video generation
python -m mlx_video.generate_av --prompt "test video" --output-path out.mp4 --output-audio out.wav
```

## Architecture

### Two-Stage Pipeline (Distilled Model)
The distilled model (`generate.py`) uses a two-stage approach for efficiency:
1. **Stage 1**: Generate at half resolution with 8 denoising steps using STAGE_1_SIGMAS
2. **Upsampler**: 2x spatial upsampling via LatentUpsampler  
3. **Stage 2**: Refine at full resolution with 3 steps using STAGE_2_SIGMAS
4. **VAE Decoder**: Convert latents to RGB video (tiled decoding for memory efficiency)

### Single-Stage Pipeline (Dev Model)
The dev model (`generate_dev.py`) uses classifier-free guidance (CFG):
- Full resolution generation with configurable steps (typically 40)
- CFG guidance scale controls prompt adherence vs. diversity
- More flexible but slower than distilled model

### Core Components

**DiT Transformer** (`models/ltx/ltx.py`):
- 48 layers, 32 attention heads, 128 dim per head
- Dual modality support: video (3840-dim) and audio (2048-dim) embeddings
- Uses RoPE (Rotary Position Embeddings) in SPLIT mode with double precision
- AdaLN-Zero conditioning blocks inject timestep/text embeddings

**VAE Architecture**:
- **Video VAE**: 128 latent channels, 8x temporal + 32x spatial compression
  - Encoder: `models/ltx/video_vae/encoder.py`
  - Decoder: `models/ltx/video_vae/decoder.py` (supports tiled decoding)
- **Audio VAE**: 8 latent channels, mel-spectrogram intermediate
  - Decoder: `models/ltx/audio_vae/decoder.py`
  - HiFi-GAN vocoder: `models/ltx/audio_vae/vocoder.py`

**Text Encoder** (`models/ltx/text_encoder.py`):
- Based on Gemma 3 model
- Returns separate embeddings for video (3840-dim) and audio (2048-dim)
- Supports prompt enhancement via `enhance_t2v()` method

**Tiling System** (`models/ltx/video_vae/tiling.py`):
- Memory-efficient decoding for large videos
- Modes: auto, default (512px/64f), aggressive (256px/32f), conservative (768px/96f)
- Supports streaming via `on_frames_ready` callback

### Key Patterns

**Position Grids**: 
- Created in pixel space, then converted to latent space internally
- Video: (B, 3, num_patches, 2) with [start, end) bounds for temporal/spatial dims
- Audio: (B, 1, num_patches, 2) for temporal dimension only
- See `create_position_grid()` in generate modules

**Latent Conditioning** (`conditioning/latent.py`):
- `LatentState` tracks clean latents, noise, and sigma values
- `VideoConditionByLatentIndex` enables I2V by conditioning specific frames
- `apply_denoise_mask()` protects conditioned regions during denoising

**Weight Loading**:
- `convert.py`: Downloads from HuggingFace, converts PyTorch → MLX format
- Sanitization functions (`sanitize_transformer_weights`, `sanitize_vae_encoder_weights`) adapt keys
- Uses safetensors for efficient loading

## Key Conventions

### Model Configuration
- Always use `LTXModelConfig` to instantiate models
- `model_type` determines modality: `VideoOnly`, `AudioOnly`, or `AudioVideo`
- `rope_type=LTXRopeType.SPLIT` and `double_precision_rope=True` are standard

### Frame Count Requirements
- **Distilled model**: `num_frames = 1 + 8*k` format (e.g., 33, 65, 97)
- **Dev model**: No strict requirement, but odd numbers work better
- Audio frames auto-computed from video duration via `AUDIO_LATENTS_PER_SECOND`

### Dimension Constraints
- Video height/width must be divisible by 64 (VAE spatial compression)
- Latent dimensions are pixel dimensions divided by 32

### Audio Constants
```python
AUDIO_SAMPLE_RATE = 24000          # Output sample rate
AUDIO_LATENT_SAMPLE_RATE = 16000   # VAE internal rate
AUDIO_HOP_LENGTH = 160             # Mel hop length
AUDIO_LATENT_CHANNELS = 8          # Audio latent channels
AUDIO_MEL_BINS = 16                # Mel frequency bins
```

### Sigma Schedules
Distilled model uses predefined schedules (no scheduler class):
```python
STAGE_1_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE_2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]
```

Dev model computes schedules via `ltx2_scheduler(steps)` function.

### Code Style
- Follow black formatting (configured in pre-commit)
- Import sorting: isort with black profile
- Remove unused imports (autoflake)
- Type hints encouraged but not enforced

### Modality Enum
Use `Modality.VIDEO` and `Modality.AUDIO` from `models/ltx/transformer.py` for multi-modal operations.

### Video Post-Processing
- `postprocess.py`: Contains utilities for frame normalization and video saving
- Always denormalize latents from [-1, 1] to [0, 255] before saving
- Use opencv-python for video I/O

## Python Requirements
- Python >= 3.11
- MLX >= 0.22.0
- Primary dependencies: numpy, safetensors, transformers, opencv-python, Pillow, mlx-vlm, scipy, librosa
- Package manager: uv recommended for faster installs, pip also supported
