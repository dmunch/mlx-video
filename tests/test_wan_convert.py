"""Tests for Wan weight conversion utilities."""

import logging

import mlx.core as mx
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Transformer Weight Conversion Tests
# ---------------------------------------------------------------------------

class TestSanitizeTransformerWeights:
    def test_patch_embedding_reshape(self):
        from mlx_video.convert_wan import sanitize_wan_transformer_weights
        weights = {
            "patch_embedding.weight": mx.random.normal((5120, 16, 1, 2, 2)),
            "patch_embedding.bias": mx.random.normal((5120,)),
        }
        out = sanitize_wan_transformer_weights(weights)
        assert "patch_embedding_proj.weight" in out
        assert "patch_embedding_proj.bias" in out
        assert out["patch_embedding_proj.weight"].shape == (5120, 16 * 1 * 2 * 2)

    def test_text_embedding_rename(self):
        from mlx_video.convert_wan import sanitize_wan_transformer_weights
        weights = {
            "text_embedding.0.weight": mx.zeros((64, 32)),
            "text_embedding.0.bias": mx.zeros((64,)),
            "text_embedding.2.weight": mx.zeros((64, 64)),
            "text_embedding.2.bias": mx.zeros((64,)),
        }
        out = sanitize_wan_transformer_weights(weights)
        assert "text_embedding_0.weight" in out
        assert "text_embedding_0.bias" in out
        assert "text_embedding_1.weight" in out
        assert "text_embedding_1.bias" in out

    def test_time_embedding_rename(self):
        from mlx_video.convert_wan import sanitize_wan_transformer_weights
        weights = {
            "time_embedding.0.weight": mx.zeros((64, 32)),
            "time_embedding.2.weight": mx.zeros((64, 64)),
        }
        out = sanitize_wan_transformer_weights(weights)
        assert "time_embedding_0.weight" in out
        assert "time_embedding_1.weight" in out

    def test_time_projection_rename(self):
        from mlx_video.convert_wan import sanitize_wan_transformer_weights
        weights = {
            "time_projection.1.weight": mx.zeros((384, 64)),
            "time_projection.1.bias": mx.zeros((384,)),
        }
        out = sanitize_wan_transformer_weights(weights)
        assert "time_projection.weight" in out
        assert "time_projection.bias" in out

    def test_ffn_rename(self):
        from mlx_video.convert_wan import sanitize_wan_transformer_weights
        weights = {
            "blocks.0.ffn.0.weight": mx.zeros((128, 64)),
            "blocks.0.ffn.0.bias": mx.zeros((128,)),
            "blocks.0.ffn.2.weight": mx.zeros((64, 128)),
            "blocks.0.ffn.2.bias": mx.zeros((64,)),
        }
        out = sanitize_wan_transformer_weights(weights)
        assert "blocks.0.ffn.fc1.weight" in out
        assert "blocks.0.ffn.fc1.bias" in out
        assert "blocks.0.ffn.fc2.weight" in out
        assert "blocks.0.ffn.fc2.bias" in out

    def test_freqs_skipped(self):
        from mlx_video.convert_wan import sanitize_wan_transformer_weights
        weights = {
            "freqs": mx.zeros((1024, 64, 2)),
            "blocks.0.norm1.weight": mx.zeros((64,)),
        }
        out = sanitize_wan_transformer_weights(weights)
        assert "freqs" not in out
        assert "blocks.0.norm1.weight" in out

    def test_passthrough_keys(self):
        from mlx_video.convert_wan import sanitize_wan_transformer_weights
        weights = {
            "blocks.0.self_attn.q.weight": mx.zeros((64, 64)),
            "blocks.0.self_attn.k.weight": mx.zeros((64, 64)),
            "blocks.0.self_attn.v.weight": mx.zeros((64, 64)),
            "blocks.0.self_attn.o.weight": mx.zeros((64, 64)),
            "blocks.0.modulation": mx.zeros((1, 6, 64)),
            "head.head.weight": mx.zeros((64, 64)),
            "head.modulation": mx.zeros((1, 2, 64)),
        }
        out = sanitize_wan_transformer_weights(weights)
        for key in weights:
            assert key in out

    def test_no_unconsumed_keys(self, caplog):
        from mlx_video.convert_wan import sanitize_wan_transformer_weights
        weights = {
            "patch_embedding.weight": mx.random.normal((5120, 16, 1, 2, 2)),
            "patch_embedding.bias": mx.random.normal((5120,)),
            "text_embedding.0.weight": mx.zeros((64, 32)),
            "text_embedding.2.weight": mx.zeros((64, 64)),
            "time_embedding.0.weight": mx.zeros((64, 32)),
            "time_embedding.2.weight": mx.zeros((64, 64)),
            "time_projection.1.weight": mx.zeros((384, 64)),
            "blocks.0.ffn.0.weight": mx.zeros((128, 64)),
            "blocks.0.ffn.2.weight": mx.zeros((64, 128)),
            "blocks.0.self_attn.q.weight": mx.zeros((64, 64)),
            "blocks.0.modulation": mx.zeros((1, 6, 64)),
            "head.head.weight": mx.zeros((64, 64)),
            "freqs": mx.zeros((1024, 64, 2)),
        }
        with caplog.at_level(logging.WARNING, logger="mlx_video.convert_wan"):
            sanitize_wan_transformer_weights(weights)
        assert "Unconsumed" not in caplog.text


class TestSanitizeT5Weights:
    def test_gate_rename(self):
        from mlx_video.convert_wan import sanitize_wan_t5_weights
        weights = {
            "blocks.0.ffn.gate.0.weight": mx.zeros((128, 64)),
            "blocks.0.ffn.fc1.weight": mx.zeros((128, 64)),
            "blocks.0.ffn.fc2.weight": mx.zeros((64, 128)),
        }
        out = sanitize_wan_t5_weights(weights)
        assert "blocks.0.ffn.gate_proj.weight" in out
        assert "blocks.0.ffn.fc1.weight" in out
        assert "blocks.0.ffn.fc2.weight" in out

    def test_passthrough(self):
        from mlx_video.convert_wan import sanitize_wan_t5_weights
        weights = {
            "token_embedding.weight": mx.zeros((100, 64)),
            "blocks.0.attn.q.weight": mx.zeros((64, 64)),
            "norm.weight": mx.zeros((64,)),
        }
        out = sanitize_wan_t5_weights(weights)
        for key in weights:
            assert key in out

    def test_no_unconsumed_keys(self, caplog):
        from mlx_video.convert_wan import sanitize_wan_t5_weights
        weights = {
            "token_embedding.weight": mx.zeros((100, 64)),
            "blocks.0.ffn.gate.0.weight": mx.zeros((128, 64)),
            "blocks.0.ffn.fc1.weight": mx.zeros((128, 64)),
            "blocks.0.ffn.fc2.weight": mx.zeros((64, 128)),
            "norm.weight": mx.zeros((64,)),
        }
        with caplog.at_level(logging.WARNING, logger="mlx_video.convert_wan"):
            sanitize_wan_t5_weights(weights)
        assert "Unconsumed" not in caplog.text


class TestSanitizeVAEWeights:
    def test_conv3d_transpose(self):
        from mlx_video.convert_wan import sanitize_wan_vae_weights
        weights = {
            "decoder.conv1.weight": mx.zeros((8, 4, 3, 3, 3)),  # [O, I, D, H, W]
        }
        out = sanitize_wan_vae_weights(weights)
        assert out["decoder.conv1.weight"].shape == (8, 3, 3, 3, 4)  # [O, D, H, W, I]

    def test_conv2d_transpose(self):
        from mlx_video.convert_wan import sanitize_wan_vae_weights
        weights = {
            "decoder.proj.weight": mx.zeros((16, 8, 3, 3)),  # [O, I, H, W]
        }
        out = sanitize_wan_vae_weights(weights)
        assert out["decoder.proj.weight"].shape == (16, 3, 3, 8)  # [O, H, W, I]

    def test_non_conv_passthrough(self):
        from mlx_video.convert_wan import sanitize_wan_vae_weights
        weights = {
            "decoder.norm.weight": mx.zeros((64,)),  # 1D, no transpose
            "decoder.bias": mx.zeros((16,)),
        }
        out = sanitize_wan_vae_weights(weights)
        assert out["decoder.norm.weight"].shape == (64,)
        assert out["decoder.bias"].shape == (16,)

    def test_mixed_weights(self):
        from mlx_video.convert_wan import sanitize_wan_vae_weights
        weights = {
            "conv3d.weight": mx.zeros((8, 4, 3, 3, 3)),  # 5D
            "conv2d.weight": mx.zeros((8, 4, 3, 3)),  # 4D
            "linear.weight": mx.zeros((8, 4)),  # 2D
            "norm.weight": mx.zeros((8,)),  # 1D
        }
        out = sanitize_wan_vae_weights(weights)
        assert out["conv3d.weight"].shape == (8, 3, 3, 3, 4)
        assert out["conv2d.weight"].shape == (8, 3, 3, 4)
        assert out["linear.weight"].shape == (8, 4)
        assert out["norm.weight"].shape == (8,)

    def test_no_unconsumed_keys(self, caplog):
        from mlx_video.convert_wan import sanitize_wan_vae_weights
        weights = {
            "decoder.conv1.weight": mx.zeros((8, 4, 3, 3, 3)),
            "decoder.proj.weight": mx.zeros((16, 8, 3, 3)),
            "decoder.norm.weight": mx.zeros((64,)),
            "decoder.bias": mx.zeros((16,)),
        }
        with caplog.at_level(logging.WARNING, logger="mlx_video.convert_wan"):
            sanitize_wan_vae_weights(weights)
        assert "Unconsumed" not in caplog.text


# ---------------------------------------------------------------------------
# Wan2.1 Conversion Tests
# ---------------------------------------------------------------------------

class TestWan21Convert:
    """Tests for Wan2.1 conversion support."""

    def test_auto_detect_wan21(self, tmp_path):
        """Auto-detect single-model directory as Wan2.1."""
        # Create a Wan2.1-style directory (no low_noise_model subdir)
        (tmp_path / "dummy.safetensors").touch()
        # The auto-detect logic: no low_noise_model dir → 2.1
        from pathlib import Path
        low = tmp_path / "low_noise_model"
        assert not low.exists()
        # Simulates auto detection
        version = "2.2" if low.exists() else "2.1"
        assert version == "2.1"

    def test_auto_detect_wan22(self, tmp_path):
        """Auto-detect dual-model directory as Wan2.2."""
        (tmp_path / "low_noise_model").mkdir()
        (tmp_path / "high_noise_model").mkdir()
        from pathlib import Path
        low = tmp_path / "low_noise_model"
        assert low.exists()
        version = "2.2" if low.exists() else "2.1"
        assert version == "2.2"

    def test_wan21_config_saved_correctly(self):
        """Verify config dict has correct fields for Wan2.1."""
        from mlx_video.models.wan.config import WanModelConfig
        config = WanModelConfig.wan21_t2v_14b()
        d = config.to_dict()
        assert d["model_version"] == "2.1"
        assert d["dual_model"] is False
        assert d["sample_steps"] == 50
        assert d["sample_shift"] == 5.0


# ---------------------------------------------------------------------------
# Encoder Weight Sanitization Tests
# ---------------------------------------------------------------------------

class TestSanitizeEncoderWeights:
    """Tests for sanitize_wan22_vae_weights with include_encoder."""

    def test_exclude_encoder_by_default(self):
        from mlx_video.models.wan.vae22 import sanitize_wan22_vae_weights

        weights = {
            "encoder.conv1.weight": mx.zeros((8, 1, 3, 3, 3)),
            "conv1.weight": mx.zeros((8, 1, 1, 1, 8)),
            "conv2.weight": mx.zeros((8, 1, 1, 1, 8)),
        }
        out = sanitize_wan22_vae_weights(weights, include_encoder=False)
        assert "conv2.weight" in out
        assert not any("encoder" in k or k.startswith("conv1") for k in out)

    def test_include_encoder(self):
        from mlx_video.models.wan.vae22 import sanitize_wan22_vae_weights

        weights = {
            "encoder.conv1.weight": mx.zeros((8, 1, 3, 3, 3)),
            "conv1.weight": mx.zeros((8, 1, 1, 1, 8)),
            "conv2.weight": mx.zeros((8, 1, 1, 1, 8)),
        }
        out = sanitize_wan22_vae_weights(weights, include_encoder=True)
        assert "encoder.conv1.weight" in out
        assert "conv1.weight" in out
        assert "conv2.weight" in out

    def test_no_unconsumed_keys(self, caplog):
        from mlx_video.models.wan.vae22 import sanitize_wan22_vae_weights

        weights = {
            "encoder.conv1.weight": mx.zeros((8, 1, 3, 3, 3)),
            "conv1.weight": mx.zeros((8, 1, 1, 1, 8)),
            "conv2.weight": mx.zeros((8, 1, 1, 1, 8)),
        }
        with caplog.at_level(logging.WARNING, logger="mlx_video.models.wan.vae22"):
            sanitize_wan22_vae_weights(weights, include_encoder=True)
        assert "Unconsumed" not in caplog.text

    def test_no_unconsumed_keys_exclude_encoder(self, caplog):
        from mlx_video.models.wan.vae22 import sanitize_wan22_vae_weights

        weights = {
            "encoder.conv1.weight": mx.zeros((8, 1, 3, 3, 3)),
            "conv1.weight": mx.zeros((8, 1, 1, 1, 8)),
            "conv2.weight": mx.zeros((8, 1, 1, 1, 8)),
        }
        with caplog.at_level(logging.WARNING, logger="mlx_video.models.wan.vae22"):
            sanitize_wan22_vae_weights(weights, include_encoder=False)
        assert "Unconsumed" not in caplog.text
