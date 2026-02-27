"""Shared test helpers for Wan test modules."""


def _make_tiny_config():
    """Create a tiny WanModelConfig for testing."""
    from mlx_video.models.wan.config import WanModelConfig
    config = WanModelConfig()
    # Override to tiny values
    config.dim = 64
    config.ffn_dim = 128
    config.num_heads = 4
    config.num_layers = 2
    config.in_dim = 4
    config.out_dim = 4
    config.patch_size = (1, 2, 2)
    config.freq_dim = 32
    config.text_dim = 32
    config.text_len = 8
    return config
