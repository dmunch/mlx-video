from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from mlx_video.models.ltx.config import BaseModelConfig


@dataclass
class WanModelConfig(BaseModelConfig):
    """Configuration for Wan2.2 T2V model."""

    model_type: str = "t2v"
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    text_len: int = 512
    in_dim: int = 16
    dim: int = 5120
    ffn_dim: int = 13824
    freq_dim: int = 256
    text_dim: int = 4096
    out_dim: int = 16
    num_heads: int = 40
    num_layers: int = 40
    window_size: Tuple[int, int] = (-1, -1)
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6

    # VAE
    vae_stride: Tuple[int, int, int] = (4, 8, 8)
    vae_z_dim: int = 16

    # Inference
    boundary: float = 0.875
    sample_shift: float = 12.0
    sample_steps: int = 40
    sample_guide_scale: Tuple[float, float] = (3.0, 4.0)
    num_train_timesteps: int = 1000
    sample_fps: int = 16
    frame_num: int = 81

    # T5
    t5_vocab_size: int = 256384
    t5_dim: int = 4096
    t5_dim_attn: int = 4096
    t5_dim_ffn: int = 10240
    t5_num_heads: int = 64
    t5_num_layers: int = 24
    t5_num_buckets: int = 32

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads
