"""Tests for LoRA loading and application."""

import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


class TestLoRATypes:
    """Test LoRA data structures."""

    def test_lora_weights_scale(self):
        from mlx_video.lora.types import LoRAWeights

        w = LoRAWeights(
            lora_A=mx.zeros((16, 64)),
            lora_B=mx.zeros((128, 16)),
            rank=16,
            alpha=32.0,
            module_name="test",
        )
        assert w.scale == 2.0

    def test_lora_weights_scale_default(self):
        from mlx_video.lora.types import LoRAWeights

        w = LoRAWeights(
            lora_A=mx.zeros((16, 64)),
            lora_B=mx.zeros((128, 16)),
            rank=16,
            alpha=16.0,
            module_name="test",
        )
        assert w.scale == 1.0

    def test_applied_lora_delta(self):
        from mlx_video.lora.types import AppliedLoRA, LoRAWeights

        lora_a = mx.ones((2, 4))
        lora_b = mx.ones((8, 2))
        w = LoRAWeights(lora_A=lora_a, lora_B=lora_b, rank=2, alpha=2.0, module_name="test")
        applied = AppliedLoRA(weights=w, strength=0.5)
        delta = applied.compute_delta()
        # scale=1.0, strength=0.5, B@A = [[2,2,2,2]]*8 (each row sum of 2 ones)
        expected = 0.5 * mx.ones((8, 4)) * 2.0
        assert mx.allclose(delta, expected).item()


class TestLoRALoader:
    """Test LoRA weight loading from safetensors."""

    def _make_lora_file(self, tmp_dir, module_names, rank=4, in_dim=64, out_dim=128, key_format="AB"):
        """Helper to create a mock LoRA safetensors file."""
        weights = {}
        for name in module_names:
            if key_format == "AB":
                weights[f"{name}.lora_A.weight"] = mx.random.normal((rank, in_dim))
                weights[f"{name}.lora_B.weight"] = mx.random.normal((out_dim, rank))
            else:
                weights[f"{name}.lora_down.weight"] = mx.random.normal((rank, in_dim))
                weights[f"{name}.lora_up.weight"] = mx.random.normal((out_dim, rank))
        path = Path(tmp_dir) / "test_lora.safetensors"
        mx.save_safetensors(str(path), weights)
        return path

    def test_load_lora_a_b_format(self):
        from mlx_video.lora.loader import load_lora_weights

        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_lora_file(tmp, ["blocks.0.self_attn.q"], key_format="AB")
            lora_weights = load_lora_weights(path)
            assert "blocks.0.self_attn.q" in lora_weights
            w = lora_weights["blocks.0.self_attn.q"]
            assert w.rank == 4
            assert w.alpha == 4.0  # default: alpha == rank
            assert w.lora_A.shape == (4, 64)
            assert w.lora_B.shape == (128, 4)

    def test_load_lora_down_up_format(self):
        from mlx_video.lora.loader import load_lora_weights

        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_lora_file(
                tmp, ["blocks.0.self_attn.q"], key_format="down_up"
            )
            lora_weights = load_lora_weights(path)
            assert "blocks.0.self_attn.q" in lora_weights

    def test_load_multiple_modules(self):
        from mlx_video.lora.loader import load_lora_weights

        modules = [
            "blocks.0.self_attn.q",
            "blocks.0.self_attn.k",
            "blocks.0.ffn.fc1",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_lora_file(tmp, modules)
            lora_weights = load_lora_weights(path)
            assert len(lora_weights) == 3
            for name in modules:
                assert name in lora_weights

    def test_load_with_alpha(self):
        from mlx_video.lora.loader import load_lora_weights

        with tempfile.TemporaryDirectory() as tmp:
            weights = {
                "test.lora_A.weight": mx.random.normal((8, 64)),
                "test.lora_B.weight": mx.random.normal((128, 8)),
                "test.alpha": mx.array(16.0),
            }
            path = Path(tmp) / "lora.safetensors"
            mx.save_safetensors(str(path), weights)
            lora_weights = load_lora_weights(path)
            assert lora_weights["test"].alpha == 16.0
            assert lora_weights["test"].rank == 8
            assert lora_weights["test"].scale == 2.0

    def test_file_not_found(self):
        from mlx_video.lora.loader import load_lora_weights

        with pytest.raises(FileNotFoundError):
            load_lora_weights(Path("/nonexistent/lora.safetensors"))


class TestWanKeyNormalization:
    """Test Wan2.2 LoRA key normalization."""

    def _wan_model_keys(self):
        """Simulate typical Wan2.2 MLX model weight keys."""
        keys = set()
        for i in range(2):
            for layer in ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
                          "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o"]:
                keys.add(f"blocks.{i}.{layer}.weight")
            keys.add(f"blocks.{i}.ffn.fc1.weight")
            keys.add(f"blocks.{i}.ffn.fc2.weight")
        keys.add("text_embedding_0.weight")
        keys.add("text_embedding_1.weight")
        keys.add("time_embedding_0.weight")
        keys.add("time_embedding_1.weight")
        keys.add("time_projection.weight")
        keys.add("patch_embedding_proj.weight")
        return keys

    def test_direct_match(self):
        from mlx_video.lora.apply import _normalize_wan_lora_key

        keys = self._wan_model_keys()
        assert _normalize_wan_lora_key("blocks.0.self_attn.q", keys) == "blocks.0.self_attn.q"

    def test_strip_diffusion_model_prefix(self):
        from mlx_video.lora.apply import _normalize_wan_lora_key

        keys = self._wan_model_keys()
        result = _normalize_wan_lora_key("diffusion_model.blocks.0.self_attn.q", keys)
        assert result == "blocks.0.self_attn.q"

    def test_strip_model_prefix(self):
        from mlx_video.lora.apply import _normalize_wan_lora_key

        keys = self._wan_model_keys()
        result = _normalize_wan_lora_key("model.diffusion_model.blocks.0.self_attn.k", keys)
        assert result == "blocks.0.self_attn.k"

    def test_ffn_key_mapping(self):
        from mlx_video.lora.apply import _normalize_wan_lora_key

        keys = self._wan_model_keys()
        assert _normalize_wan_lora_key("blocks.0.ffn.0", keys) == "blocks.0.ffn.fc1"
        assert _normalize_wan_lora_key("blocks.0.ffn.2", keys) == "blocks.0.ffn.fc2"

    def test_text_embedding_mapping(self):
        from mlx_video.lora.apply import _normalize_wan_lora_key

        keys = self._wan_model_keys()
        assert _normalize_wan_lora_key("text_embedding.0", keys) == "text_embedding_0"
        assert _normalize_wan_lora_key("text_embedding.2", keys) == "text_embedding_1"

    def test_time_embedding_mapping(self):
        from mlx_video.lora.apply import _normalize_wan_lora_key

        keys = self._wan_model_keys()
        assert _normalize_wan_lora_key("time_embedding.0", keys) == "time_embedding_0"
        assert _normalize_wan_lora_key("time_embedding.2", keys) == "time_embedding_1"

    def test_time_projection_mapping(self):
        from mlx_video.lora.apply import _normalize_wan_lora_key

        keys = self._wan_model_keys()
        assert _normalize_wan_lora_key("time_projection.1", keys) == "time_projection"

    def test_patch_embedding_mapping(self):
        from mlx_video.lora.apply import _normalize_wan_lora_key

        keys = self._wan_model_keys()
        assert _normalize_wan_lora_key("patch_embedding", keys) == "patch_embedding_proj"

    def test_combined_prefix_and_ffn(self):
        from mlx_video.lora.apply import _normalize_wan_lora_key

        keys = self._wan_model_keys()
        result = _normalize_wan_lora_key("diffusion_model.blocks.1.ffn.0", keys)
        assert result == "blocks.1.ffn.fc1"


class TestApplyLoRA:
    """Test LoRA delta application to weights."""

    def test_preserves_bfloat16_dtype(self):
        """LoRA delta must not promote bfloat16 weights to float32."""
        from mlx_video.lora.apply import apply_lora_to_linear
        from mlx_video.lora.types import LoRAWeights

        original = mx.ones((8, 4), dtype=mx.bfloat16)
        # LoRA weights in float32 (typical when loaded from safetensors)
        lora_a = mx.ones((2, 4), dtype=mx.float32) * 0.1
        lora_b = mx.ones((8, 2), dtype=mx.float32) * 0.1
        w = LoRAWeights(lora_A=lora_a, lora_B=lora_b, rank=2, alpha=2.0, module_name="test")
        result = apply_lora_to_linear(original, [(w, 1.0)])
        assert result.dtype == mx.bfloat16, f"Expected bfloat16, got {result.dtype}"

    def test_preserves_float16_dtype(self):
        from mlx_video.lora.apply import apply_lora_to_linear
        from mlx_video.lora.types import LoRAWeights

        original = mx.ones((8, 4), dtype=mx.float16)
        lora_a = mx.ones((2, 4), dtype=mx.float32) * 0.1
        lora_b = mx.ones((8, 2), dtype=mx.float32) * 0.1
        w = LoRAWeights(lora_A=lora_a, lora_B=lora_b, rank=2, alpha=2.0, module_name="test")
        result = apply_lora_to_linear(original, [(w, 1.0)])
        assert result.dtype == mx.float16, f"Expected float16, got {result.dtype}"

    def test_apply_single_lora(self):
        from mlx_video.lora.apply import apply_lora_to_linear
        from mlx_video.lora.types import LoRAWeights

        original = mx.ones((8, 4))
        lora_a = mx.ones((2, 4)) * 0.1
        lora_b = mx.ones((8, 2)) * 0.1
        w = LoRAWeights(lora_A=lora_a, lora_B=lora_b, rank=2, alpha=2.0, module_name="test")
        result = apply_lora_to_linear(original, [(w, 1.0)])
        # delta = 1.0 * (B @ A) = ones(8,2)*0.1 @ ones(2,4)*0.1 = 0.02 * ones(8,4)
        expected = original + 0.02 * mx.ones((8, 4))
        assert mx.allclose(result, expected, atol=1e-6).item()

    def test_apply_multiple_loras(self):
        from mlx_video.lora.apply import apply_lora_to_linear
        from mlx_video.lora.types import LoRAWeights

        original = mx.zeros((8, 4))
        w1 = LoRAWeights(
            lora_A=mx.ones((2, 4)),
            lora_B=mx.ones((8, 2)),
            rank=2, alpha=2.0, module_name="a",
        )
        w2 = LoRAWeights(
            lora_A=mx.ones((2, 4)) * 2,
            lora_B=mx.ones((8, 2)) * 2,
            rank=2, alpha=4.0, module_name="b",
        )
        result = apply_lora_to_linear(original, [(w1, 1.0), (w2, 0.5)])
        # w1 delta: 1.0 * 1.0 * (ones(8,2) @ ones(2,4)) = 2 * ones(8,4)
        # w2 delta: 2.0 * 0.5 * (2*ones(8,2) @ 2*ones(2,4)) = 1.0 * 8*ones(8,4) = 8
        delta1 = mx.ones((8, 4)) * 2.0
        delta2 = mx.ones((8, 4)) * 8.0
        expected = delta1 + delta2
        assert mx.allclose(result, expected, atol=1e-5).item()

    def test_apply_loras_to_weights_dict(self):
        from mlx_video.lora.apply import apply_loras_to_weights
        from mlx_video.lora.types import LoRAWeights

        model_weights = {
            "blocks.0.self_attn.q.weight": mx.ones((128, 64)),
            "blocks.0.self_attn.k.weight": mx.ones((128, 64)),
            "blocks.0.ffn.fc1.weight": mx.ones((256, 64)),
        }
        w = LoRAWeights(
            lora_A=mx.ones((4, 64)) * 0.01,
            lora_B=mx.ones((128, 4)) * 0.01,
            rank=4, alpha=4.0, module_name="blocks.0.self_attn.q",
        )
        module_to_loras = {"blocks.0.self_attn.q": [(w, 1.0)]}
        result = apply_loras_to_weights(model_weights, module_to_loras)
        # Only q should be modified
        assert not mx.array_equal(
            result["blocks.0.self_attn.q.weight"],
            model_weights["blocks.0.self_attn.q.weight"],
        ).item()
        assert mx.array_equal(
            result["blocks.0.self_attn.k.weight"],
            model_weights["blocks.0.self_attn.k.weight"],
        ).item()


class TestEndToEnd:
    """End-to-end LoRA loading and application."""

    def test_load_and_apply_loras(self):
        from mlx_video.convert_wan import load_and_apply_loras

        with tempfile.TemporaryDirectory() as tmp:
            # Create mock LoRA safetensors
            rank = 4
            weights = {
                "blocks.0.self_attn.q.lora_A.weight": mx.random.normal((rank, 64)),
                "blocks.0.self_attn.q.lora_B.weight": mx.random.normal((128, rank)),
            }
            lora_path = Path(tmp) / "test.safetensors"
            mx.save_safetensors(str(lora_path), weights)

            # Create mock model weights
            model_weights = {
                "blocks.0.self_attn.q.weight": mx.ones((128, 64)),
                "blocks.0.self_attn.k.weight": mx.ones((128, 64)),
            }

            result = load_and_apply_loras(
                model_weights, [(str(lora_path), 1.0)]
            )

            # q weight should be modified, k unchanged
            assert not mx.array_equal(
                result["blocks.0.self_attn.q.weight"],
                model_weights["blocks.0.self_attn.q.weight"],
            ).item()
            assert mx.array_equal(
                result["blocks.0.self_attn.k.weight"],
                model_weights["blocks.0.self_attn.k.weight"],
            ).item()
