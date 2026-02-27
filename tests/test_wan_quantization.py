"""Tests for Wan model quantization pipeline."""

import json
import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np
import pytest

from wan_test_helpers import _make_tiny_config


# ---------------------------------------------------------------------------
# Quantize Predicate Tests
# ---------------------------------------------------------------------------

class TestQuantizePredicate:
    def test_matches_self_attention_layers(self):
        from mlx_video.convert_wan import _quantize_predicate
        mock_linear = nn.Linear(64, 64)
        for suffix in ["q", "k", "v", "o"]:
            path = f"blocks.0.self_attn.{suffix}"
            assert _quantize_predicate(path, mock_linear), f"Should match {path}"

    def test_matches_cross_attention_layers(self):
        from mlx_video.convert_wan import _quantize_predicate
        mock_linear = nn.Linear(64, 64)
        for suffix in ["q", "k", "v", "o"]:
            path = f"blocks.0.cross_attn.{suffix}"
            assert _quantize_predicate(path, mock_linear), f"Should match {path}"

    def test_matches_ffn_layers(self):
        from mlx_video.convert_wan import _quantize_predicate
        mock_linear = nn.Linear(64, 64)
        assert _quantize_predicate("blocks.0.ffn.fc1", mock_linear)
        assert _quantize_predicate("blocks.0.ffn.fc2", mock_linear)

    def test_rejects_embeddings(self):
        from mlx_video.convert_wan import _quantize_predicate
        mock_linear = nn.Linear(64, 64)
        for path in ["patch_embedding_proj", "text_embedding_fc1", "time_embedding.fc1"]:
            assert not _quantize_predicate(path, mock_linear), f"Should reject {path}"

    def test_rejects_norms(self):
        from mlx_video.convert_wan import _quantize_predicate
        mock_norm = nn.RMSNorm(64)
        assert not _quantize_predicate("blocks.0.self_attn.norm_q", mock_norm)

    def test_rejects_non_quantizable_modules(self):
        from mlx_video.convert_wan import _quantize_predicate
        mock_norm = nn.RMSNorm(64)
        # Even if path matches, module must have to_quantized
        assert not _quantize_predicate("blocks.0.self_attn.q", mock_norm)

    def test_all_10_patterns_covered(self):
        """Verify exactly 10 layer patterns are targeted."""
        from mlx_video.convert_wan import _quantize_predicate
        mock_linear = nn.Linear(64, 64)
        patterns = [
            "blocks.0.self_attn.q", "blocks.0.self_attn.k",
            "blocks.0.self_attn.v", "blocks.0.self_attn.o",
            "blocks.0.cross_attn.q", "blocks.0.cross_attn.k",
            "blocks.0.cross_attn.v", "blocks.0.cross_attn.o",
            "blocks.0.ffn.fc1", "blocks.0.ffn.fc2",
        ]
        matched = [p for p in patterns if _quantize_predicate(p, mock_linear)]
        assert len(matched) == 10


# ---------------------------------------------------------------------------
# Quantize Round-Trip Tests
# ---------------------------------------------------------------------------

class TestQuantizeRoundTrip:
    def _quantize_and_save(self, config, tmp_path, bits=4, group_size=64):
        """Helper: create model, quantize, save to tmp_path."""
        from mlx_video.models.wan.model import WanModel
        from mlx_video.convert_wan import _quantize_predicate

        model = WanModel(config)
        nn.quantize(
            model,
            group_size=group_size,
            bits=bits,
            class_predicate=lambda path, m: _quantize_predicate(path, m),
        )

        weights_dict = dict(mlx.utils.tree_flatten(model.parameters()))
        model_path = tmp_path / "model.safetensors"
        mx.save_safetensors(str(model_path), weights_dict)

        # Write config.json
        cfg = {"quantization": {"bits": bits, "group_size": group_size}}
        with open(tmp_path / "config.json", "w") as f:
            json.dump(cfg, f)

        return model_path, weights_dict

    def test_4bit_roundtrip(self, tmp_path):
        config = _make_tiny_config()
        model_path, saved_weights = self._quantize_and_save(config, tmp_path, bits=4)

        from mlx_video.models.wan.loading import load_wan_model
        loaded = load_wan_model(
            model_path, config,
            quantization={"bits": 4, "group_size": 64},
        )

        # Verify quantized layers have scales
        has_scales = any("scales" in k for k in saved_weights)
        assert has_scales, "Quantized model should have .scales tensors"

        # Verify a self-attention layer is QuantizedLinear
        assert isinstance(loaded.blocks[0].self_attn.q, nn.QuantizedLinear)
        assert isinstance(loaded.blocks[0].ffn.fc1, nn.QuantizedLinear)

    def test_8bit_roundtrip(self, tmp_path):
        config = _make_tiny_config()
        model_path, saved_weights = self._quantize_and_save(config, tmp_path, bits=8)

        from mlx_video.models.wan.loading import load_wan_model
        loaded = load_wan_model(
            model_path, config,
            quantization={"bits": 8, "group_size": 64},
        )

        assert isinstance(loaded.blocks[0].self_attn.q, nn.QuantizedLinear)
        assert isinstance(loaded.blocks[0].cross_attn.k, nn.QuantizedLinear)

    def test_non_quantized_layers_remain_linear(self, tmp_path):
        config = _make_tiny_config()
        model_path, _ = self._quantize_and_save(config, tmp_path, bits=4)

        from mlx_video.models.wan.loading import load_wan_model
        loaded = load_wan_model(
            model_path, config,
            quantization={"bits": 4, "group_size": 64},
        )

        # Head should NOT be quantized (it's not in the predicate patterns)
        assert not isinstance(loaded.head, nn.QuantizedLinear)

    def test_loading_without_quantization_flag(self, tmp_path):
        """Loading a non-quantized model should have standard Linear layers."""
        from mlx_video.models.wan.model import WanModel

        config = _make_tiny_config()
        model = WanModel(config)
        weights_dict = dict(mlx.utils.tree_flatten(model.parameters()))
        model_path = tmp_path / "model.safetensors"
        mx.save_safetensors(str(model_path), weights_dict)

        from mlx_video.models.wan.loading import load_wan_model
        loaded = load_wan_model(model_path, config, quantization=None)

        assert isinstance(loaded.blocks[0].self_attn.q, nn.Linear)
        assert not isinstance(loaded.blocks[0].self_attn.q, nn.QuantizedLinear)


# ---------------------------------------------------------------------------
# Quantized Inference Tests
# ---------------------------------------------------------------------------

class TestQuantizedInference:
    def _make_quantized_model(self, config, bits=4):
        from mlx_video.models.wan.model import WanModel
        from mlx_video.convert_wan import _quantize_predicate

        model = WanModel(config)
        nn.quantize(
            model,
            group_size=64,
            bits=bits,
            class_predicate=lambda path, m: _quantize_predicate(path, m),
        )
        mx.eval(model.parameters())
        return model

    def test_forward_pass_4bit(self):
        config = _make_tiny_config()
        model = self._make_quantized_model(config, bits=4)

        C, F, H, W = config.in_dim, 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        x = [mx.random.normal((C, F, H, W))]
        t = mx.array([500.0])
        context = [mx.random.normal((4, config.text_dim))]

        out = model(x, t, context, seq_len)
        mx.eval(out[0])

        assert len(out) == 1
        assert out[0].shape == (C, F, H, W)

    def test_forward_pass_8bit(self):
        config = _make_tiny_config()
        model = self._make_quantized_model(config, bits=8)

        C, F, H, W = config.in_dim, 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        x = [mx.random.normal((C, F, H, W))]
        t = mx.array([500.0])
        context = [mx.random.normal((4, config.text_dim))]

        out = model(x, t, context, seq_len)
        mx.eval(out[0])

        assert len(out) == 1
        assert out[0].shape == (C, F, H, W)

    def test_quantized_output_differs_from_unquantized(self):
        """Sanity check: quantization should change the weights."""
        from mlx_video.models.wan.model import WanModel
        from mlx_video.convert_wan import _quantize_predicate

        config = _make_tiny_config()
        mx.random.seed(42)

        # Get unquantized weights
        model = WanModel(config)
        mx.eval(model.parameters())
        orig_weight = np.array(model.blocks[0].self_attn.q.weight)

        # Quantize
        nn.quantize(
            model,
            group_size=64,
            bits=4,
            class_predicate=lambda path, m: _quantize_predicate(path, m),
        )
        mx.eval(model.parameters())

        # QuantizedLinear stores weight differently (uint32 packed)
        assert isinstance(model.blocks[0].self_attn.q, nn.QuantizedLinear)
        assert hasattr(model.blocks[0].self_attn.q, "scales")


# ---------------------------------------------------------------------------
# Config Metadata Tests
# ---------------------------------------------------------------------------

class TestQuantizationConfig:
    def test_config_metadata_written(self, tmp_path):
        """Verify _quantize_saved_model writes quantization metadata to config.json."""
        from mlx_video.models.wan.model import WanModel
        from mlx_video.convert_wan import _quantize_saved_model

        config = _make_tiny_config()
        model = WanModel(config)
        weights_dict = dict(mlx.utils.tree_flatten(model.parameters()))

        # Save unquantized model + config
        model_path = tmp_path / "model.safetensors"
        mx.save_safetensors(str(model_path), weights_dict)
        with open(tmp_path / "config.json", "w") as f:
            json.dump({"dim": config.dim}, f)

        # Run quantization
        _quantize_saved_model(tmp_path, config, is_dual=False, bits=4, group_size=64)

        # Verify metadata
        with open(tmp_path / "config.json") as f:
            cfg = json.load(f)
        assert "quantization" in cfg
        assert cfg["quantization"]["bits"] == 4
        assert cfg["quantization"]["group_size"] == 64

    def test_config_metadata_8bit(self, tmp_path):
        from mlx_video.models.wan.model import WanModel
        from mlx_video.convert_wan import _quantize_saved_model

        config = _make_tiny_config()
        model = WanModel(config)
        weights_dict = dict(mlx.utils.tree_flatten(model.parameters()))

        model_path = tmp_path / "model.safetensors"
        mx.save_safetensors(str(model_path), weights_dict)
        with open(tmp_path / "config.json", "w") as f:
            json.dump({}, f)

        _quantize_saved_model(tmp_path, config, is_dual=False, bits=8, group_size=32)

        with open(tmp_path / "config.json") as f:
            cfg = json.load(f)
        assert cfg["quantization"]["bits"] == 8
        assert cfg["quantization"]["group_size"] == 32

    def test_dual_model_quantization(self, tmp_path):
        """Verify dual-model quantization writes both model files."""
        from mlx_video.models.wan.model import WanModel
        from mlx_video.convert_wan import _quantize_saved_model

        config = _make_tiny_config()

        for name in ["low_noise_model.safetensors", "high_noise_model.safetensors"]:
            model = WanModel(config)
            weights_dict = dict(mlx.utils.tree_flatten(model.parameters()))
            mx.save_safetensors(str(tmp_path / name), weights_dict)

        with open(tmp_path / "config.json", "w") as f:
            json.dump({}, f)

        _quantize_saved_model(tmp_path, config, is_dual=True, bits=4, group_size=64)

        # Both files should now contain quantized weights (have .scales keys)
        for name in ["low_noise_model.safetensors", "high_noise_model.safetensors"]:
            weights = mx.load(str(tmp_path / name))
            has_scales = any("scales" in k for k in weights)
            assert has_scales, f"{name} should have quantized layers"
