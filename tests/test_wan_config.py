"""Tests for Wan model configuration."""

import pytest


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

class TestWanModelConfig:
    """Tests for WanModelConfig dataclass."""

    def test_default_values(self):
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig()
        assert config.dim == 5120
        assert config.ffn_dim == 13824
        assert config.num_heads == 40
        assert config.num_layers == 40
        assert config.in_dim == 16
        assert config.out_dim == 16
        assert config.patch_size == (1, 2, 2)
        assert config.vae_stride == (4, 8, 8)
        assert config.vae_z_dim == 16
        assert config.boundary == 0.875
        assert config.sample_shift == 12.0
        assert config.sample_steps == 40
        assert config.sample_guide_scale == (3.0, 4.0)
        assert config.num_train_timesteps == 1000
        assert config.qk_norm is True
        assert config.cross_attn_norm is True
        assert config.text_len == 512

    def test_head_dim_property(self):
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig()
        assert config.head_dim == 128  # 5120 // 40

    def test_to_dict_roundtrip(self):
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["dim"] == 5120
        assert d["patch_size"] == (1, 2, 2)
        assert d["boundary"] == 0.875

    def test_t5_config_values(self):
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig()
        assert config.t5_vocab_size == 256384
        assert config.t5_dim == 4096
        assert config.t5_dim_attn == 4096
        assert config.t5_dim_ffn == 10240
        assert config.t5_num_heads == 64
        assert config.t5_num_layers == 24
        assert config.t5_num_buckets == 32


# ---------------------------------------------------------------------------
# Wan2.1 Config Tests
# ---------------------------------------------------------------------------

class TestWan21Config:
    """Tests for Wan2.1 config presets."""

    def test_wan21_14b_factory(self):
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig.wan21_t2v_14b()
        assert config.model_version == "2.1"
        assert config.dual_model is False
        assert config.dim == 5120
        assert config.ffn_dim == 13824
        assert config.num_heads == 40
        assert config.num_layers == 40
        assert config.head_dim == 128
        assert config.sample_guide_scale == 5.0
        assert config.sample_shift == 5.0
        assert config.sample_steps == 50
        assert config.boundary == 0.0

    def test_wan21_1_3b_factory(self):
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig.wan21_t2v_1_3b()
        assert config.model_version == "2.1"
        assert config.dual_model is False
        assert config.dim == 1536
        assert config.ffn_dim == 8960
        assert config.num_heads == 12
        assert config.num_layers == 30
        assert config.head_dim == 128  # 1536 // 12
        assert config.sample_guide_scale == 5.0

    def test_wan22_14b_factory(self):
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig.wan22_t2v_14b()
        assert config.model_version == "2.2"
        assert config.dual_model is True
        assert config.dim == 5120
        assert config.sample_guide_scale == (3.0, 4.0)
        assert config.sample_shift == 12.0
        assert config.sample_steps == 40
        assert config.boundary == 0.875

    def test_wan21_config_to_dict(self):
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig.wan21_t2v_14b()
        d = config.to_dict()
        assert d["model_version"] == "2.1"
        assert d["dual_model"] is False
        assert d["sample_guide_scale"] == 5.0

    def test_wan21_1_3b_config_to_dict(self):
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig.wan21_t2v_1_3b()
        d = config.to_dict()
        assert d["dim"] == 1536
        assert d["num_layers"] == 30

    def test_default_config_is_wan22(self):
        """Default WanModelConfig() should be Wan2.2 14B."""
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig()
        assert config.model_version == "2.2"
        assert config.dual_model is True
