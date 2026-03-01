"""Tests for Wan2.2 LoRA training components."""

import json

import mlx.core as mx
import mlx.nn as nn
import pytest


class TestTrainingConfig:
    """Test config parsing and data discovery."""

    def test_config_from_json(self, tmp_path):
        """Test basic config loading with data discovery."""
        # Create data directory with images and prompts
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create fake image files (1x1 PNG)
        from PIL import Image

        for i in range(3):
            img = Image.new("RGB", (64, 64), color=(i * 50, 100, 200))
            img.save(data_dir / f"img{i}.png")
            (data_dir / f"img{i}.txt").write_text(f"A photo of a cat number {i}")

        # Create a fake model dir
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        # Save a minimal config.json
        model_config = {
            "model_version": "2.2",
            "dim": 5120,
            "num_layers": 40,
            "dual_model": True,
        }
        (model_dir / "config.json").write_text(json.dumps(model_config))

        # Create config JSON
        config = {
            "model_dir": str(model_dir),
            "data": str(data_dir),
            "seed": 123,
            "resolution": 512,
            "trigger_word": "ohwx",
            "training": {
                "num_epochs": 10,
                "batch_size": 1,
                "learning_rate": 1e-4,
            },
            "lora": {
                "rank": 16,
                "alpha": 16.0,
                "targets": ["self_attn.q", "self_attn.v"],
                "blocks": {"start": 0, "end": 5},
            },
            "checkpoint": {
                "save_frequency": 5,
                "output_dir": str(tmp_path / "output"),
            },
        }
        config_path = tmp_path / "train.json"
        config_path.write_text(json.dumps(config))

        from mlx_video.training.config import TrainingConfig

        tc = TrainingConfig.from_json(str(config_path))

        assert tc.seed == 123
        assert tc.resolution == 512
        assert tc.trigger_word == "ohwx"
        assert len(tc.data_items) == 3
        assert tc.lora.rank == 16
        assert tc.lora.blocks.start == 0
        assert tc.lora.blocks.end == 5
        assert tc.training.num_epochs == 10

        # Check trigger word was prepended
        for item in tc.data_items:
            assert item.prompt.startswith("ohwx")

    def test_config_validates_resolution(self, tmp_path):
        """Test that invalid resolution is rejected."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        from PIL import Image

        img = Image.new("RGB", (64, 64))
        img.save(data_dir / "img.png")
        (data_dir / "img.txt").write_text("test")

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        config = {
            "model_dir": str(model_dir),
            "data": str(data_dir),
            "resolution": 96 + 16,  # 112, not divisible by 32 but >= 64
        }
        config_path = tmp_path / "train.json"
        config_path.write_text(json.dumps(config))

        from mlx_video.training.config import TrainingConfig

        with pytest.raises(ValueError, match="divisible by 32"):
            TrainingConfig.from_json(str(config_path))

    def test_config_missing_prompt(self, tmp_path):
        """Test that missing prompt file raises error."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        from PIL import Image

        img = Image.new("RGB", (64, 64))
        img.save(data_dir / "img.png")
        # No img.txt

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        config = {
            "model_dir": str(model_dir),
            "data": str(data_dir),
        }
        config_path = tmp_path / "train.json"
        config_path.write_text(json.dumps(config))

        from mlx_video.training.config import TrainingConfig

        with pytest.raises(ValueError, match="Missing prompt file"):
            TrainingConfig.from_json(str(config_path))

    def test_monitoring_config_defaults(self, tmp_path):
        """Test that monitoring config fields have correct defaults."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        from PIL import Image

        img = Image.new("RGB", (64, 64))
        img.save(data_dir / "img.png")
        (data_dir / "img.txt").write_text("test prompt")

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        config = {"model_dir": str(model_dir), "data": str(data_dir)}
        config_path = tmp_path / "train.json"
        config_path.write_text(json.dumps(config))

        from mlx_video.training.config import TrainingConfig

        tc = TrainingConfig.from_json(str(config_path))
        assert tc.monitoring.log_frequency == 1
        assert tc.monitoring.plot_frequency == 10
        assert tc.monitoring.generate_image_frequency == 0
        assert tc.monitoring.preview_width == 512
        assert tc.monitoring.preview_height == 512

    def test_monitoring_config_custom(self, tmp_path):
        """Test that monitoring config fields parse from JSON."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        from PIL import Image

        img = Image.new("RGB", (64, 64))
        img.save(data_dir / "img.png")
        (data_dir / "img.txt").write_text("test prompt")

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        config = {
            "model_dir": str(model_dir),
            "data": str(data_dir),
            "monitoring": {
                "plot_frequency": 5,
                "generate_image_frequency": 50,
                "preview_width": 256,
                "preview_height": 384,
            },
        }
        config_path = tmp_path / "train.json"
        config_path.write_text(json.dumps(config))

        from mlx_video.training.config import TrainingConfig

        tc = TrainingConfig.from_json(str(config_path))
        assert tc.monitoring.plot_frequency == 5
        assert tc.monitoring.generate_image_frequency == 50
        assert tc.monitoring.preview_width == 256
        assert tc.monitoring.preview_height == 384

    def test_preview_prompt_from_data(self, tmp_path):
        """Test that preview_prompt defaults to first training sample."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        from PIL import Image

        img = Image.new("RGB", (64, 64))
        img.save(data_dir / "img.png")
        (data_dir / "img.txt").write_text("a photo of a cat")

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        config = {"model_dir": str(model_dir), "data": str(data_dir)}
        config_path = tmp_path / "train.json"
        config_path.write_text(json.dumps(config))

        from mlx_video.training.config import TrainingConfig

        tc = TrainingConfig.from_json(str(config_path))
        assert tc.preview_prompt == "a photo of a cat"

    def test_preview_prompt_from_file(self, tmp_path):
        """Test that preview.txt overrides default preview prompt."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        from PIL import Image

        img = Image.new("RGB", (64, 64))
        img.save(data_dir / "img.png")
        (data_dir / "img.txt").write_text("training prompt")
        (data_dir / "preview.txt").write_text("custom preview prompt")

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        config = {"model_dir": str(model_dir), "data": str(data_dir)}
        config_path = tmp_path / "train.json"
        config_path.write_text(json.dumps(config))

        from mlx_video.training.config import TrainingConfig

        tc = TrainingConfig.from_json(str(config_path))
        assert tc.preview_prompt == "custom preview prompt"


class TestLoRALayers:
    """Test LoRA layer injection and freezing."""

    def test_trainable_lora_linear(self):
        """Test TrainableLoRALinear forward pass."""
        from mlx_video.training.lora_layers import TrainableLoRALinear

        linear = nn.Linear(64, 128)
        lora = TrainableLoRALinear(linear, rank=8, alpha=8.0)

        x = mx.random.normal((2, 64))
        output = lora(x)
        assert output.shape == (2, 128)

        # Initially lora_B is zero, so output should equal base linear output
        base_output = linear(x)
        mx.eval(output, base_output)
        assert mx.allclose(output, base_output, atol=1e-5).item()

    def test_inject_lora_layers(self):
        """Test LoRA injection into a simple model with blocks."""
        from mlx_video.training.config import BlockRange, LoRAConfig
        from mlx_video.training.lora_layers import (
            TrainableLoRALinear,
            inject_lora_layers,
        )

        # Create a minimal model-like structure
        class FakeAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q = nn.Linear(64, 64)
                self.k = nn.Linear(64, 64)
                self.v = nn.Linear(64, 64)
                self.o = nn.Linear(64, 64)

        class FakeBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = FakeAttn()

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = [FakeBlock() for _ in range(4)]

        model = FakeModel()
        lora_config = LoRAConfig(
            rank=8,
            alpha=8.0,
            targets=["self_attn.q", "self_attn.v"],
            blocks=BlockRange(start=0, end=2),
        )

        count = inject_lora_layers(model, lora_config)
        assert count == 4  # 2 targets x 2 blocks

        # Verify injection
        assert isinstance(model.blocks[0].self_attn.q, TrainableLoRALinear)
        assert isinstance(model.blocks[0].self_attn.v, TrainableLoRALinear)
        assert isinstance(model.blocks[1].self_attn.q, TrainableLoRALinear)
        assert isinstance(model.blocks[1].self_attn.v, TrainableLoRALinear)
        # Block 2 should NOT be injected
        assert isinstance(model.blocks[2].self_attn.q, nn.Linear)
        assert isinstance(model.blocks[2].self_attn.v, nn.Linear)

    def test_freeze_base_weights(self):
        """Test that freeze_base_weights freezes everything except LoRA params."""
        from mlx_video.training.lora_layers import (
            TrainableLoRALinear,
            freeze_base_weights,
        )

        linear = nn.Linear(32, 64)
        lora = TrainableLoRALinear(linear, rank=4, alpha=4.0)

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = lora
                self.other = nn.Linear(32, 32)

        model = SimpleModel()
        freeze_base_weights(model)

        # LoRA params should be trainable (unfrozen)
        # This is verified by checking the model's trainable parameters
        trainable = model.trainable_parameters()

        # Should contain lora_A and lora_B
        has_lora_a = False
        has_lora_b = False
        for key_path, param in _flatten_params(trainable):
            if "lora_A" in key_path:
                has_lora_a = True
            if "lora_B" in key_path:
                has_lora_b = True

        assert has_lora_a, "lora_A should be trainable"
        assert has_lora_b, "lora_B should be trainable"


def _flatten_params(params, prefix=""):
    """Helper to flatten nested parameter dict."""
    results = []
    if isinstance(params, dict):
        for k, v in params.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            results.extend(_flatten_params(v, new_prefix))
    elif isinstance(params, list):
        for i, v in enumerate(params):
            new_prefix = f"{prefix}.{i}" if prefix else str(i)
            results.extend(_flatten_params(v, new_prefix))
    elif isinstance(params, mx.array):
        results.append((prefix, params))
    return results


class TestSaveLoRA:
    """Test LoRA weight saving."""

    def test_collect_and_save(self, tmp_path):
        """Test that LoRA weights can be collected and saved."""
        from mlx_video.training.lora_layers import TrainableLoRALinear
        from mlx_video.training.save import _collect_lora_weights

        class FakeAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q = TrainableLoRALinear(nn.Linear(64, 64), rank=4, alpha=4.0)

        class FakeBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = FakeAttn()

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = [FakeBlock(), FakeBlock()]

        model = FakeModel()
        weights = _collect_lora_weights(model)

        assert len(weights) == 4  # 2 blocks x (lora_A + lora_B)
        assert "blocks.0.self_attn.q.lora_A.weight" in weights
        assert "blocks.0.self_attn.q.lora_B.weight" in weights
        assert "blocks.1.self_attn.q.lora_A.weight" in weights
        assert "blocks.1.self_attn.q.lora_B.weight" in weights


class TestTimestepSampling:
    """Test timestep sampling strategies."""

    def test_balanced_sampling(self):
        """Test balanced sampling produces values in (0, 1)."""
        import random

        from mlx_video.training.trainer import _sample_timestep

        rng = random.Random(42)
        samples = [_sample_timestep(1000, "balanced", rng) for _ in range(100)]

        assert all(0 < s < 1 for s in samples)
        # Mean should be roughly 0.5
        mean = sum(samples) / len(samples)
        assert 0.3 < mean < 0.7

    def test_low_bias_sampling(self):
        """Test low_bias sampling is biased toward low sigma."""
        import random

        from mlx_video.training.trainer import _sample_timestep

        rng = random.Random(42)
        samples = [_sample_timestep(1000, "low_bias", rng) for _ in range(1000)]

        mean = sum(samples) / len(samples)
        # Low bias: mean should be below 0.5
        assert mean < 0.45

    def test_high_bias_sampling(self):
        """Test high_bias sampling is biased toward high sigma."""
        import random

        from mlx_video.training.trainer import _sample_timestep

        rng = random.Random(42)
        samples = [_sample_timestep(1000, "high_bias", rng) for _ in range(1000)]

        mean = sum(samples) / len(samples)
        # High bias: mean should be above 0.5
        assert mean > 0.55


class TestCheckpointRoundtrip:
    """Test checkpoint save/load for single and dual expert training."""

    def _make_lora_model(self):
        """Create a tiny model with LoRA layers for testing."""
        from mlx_video.training.lora_layers import TrainableLoRALinear

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 8)

            def named_modules(self):
                yield "linear", self.linear

        model = TinyModel()
        model.linear = TrainableLoRALinear(model.linear, rank=4, alpha=4.0)
        return model

    def _make_config(self, tmp_path):
        """Create a minimal TrainingConfig for testing."""
        from mlx_video.training.config import LoRAConfig, TrainingConfig

        config = TrainingConfig.__new__(TrainingConfig)
        config.model_dir = str(tmp_path / "model")
        config.resolution = 512
        config.seed = 42
        config.trigger_word = "test"
        config.lora = LoRAConfig(rank=4, alpha=4.0)
        return config

    def test_single_checkpoint_roundtrip(self, tmp_path):
        """Test save and load of a single-expert checkpoint zip."""
        import mlx.optimizers as optim

        from mlx_video.training.save import load_checkpoint, save_checkpoint

        model = self._make_lora_model()
        optimizer = optim.AdamW(learning_rate=1e-4)
        optimizer.init(model.trainable_parameters())

        # Set known LoRA values
        model.linear.lora_A = mx.ones_like(model.linear.lora_A) * 0.5
        model.linear.lora_B = mx.ones_like(model.linear.lora_B) * 0.3
        mx.eval(model.parameters(), optimizer.state)

        config = self._make_config(tmp_path)
        ckpt_path = str(tmp_path / "checkpoint.zip")

        save_checkpoint(
            model,
            optimizer,
            config,
            epoch=10,
            global_step=100,
            loss_history_data=[(1, 0.5), (2, 0.4)],
            output_path=ckpt_path,
            expert_label="low",
        )

        # Verify zip contents
        import zipfile

        with zipfile.ZipFile(ckpt_path) as zf:
            names = zf.namelist()
            assert "lora_weights.safetensors" in names
            assert "optimizer_state.safetensors" in names
            assert "state.json" in names
            assert "config.json" in names

        # Load into fresh model
        model2 = self._make_lora_model()
        opt2 = optim.AdamW(learning_rate=1e-4)
        opt2.init(model2.trainable_parameters())
        mx.eval(model2.parameters(), opt2.state)

        state = load_checkpoint(ckpt_path, model2, opt2)

        assert state["epoch"] == 10
        assert state["global_step"] == 100
        assert state["expert_label"] == "low"
        assert len(state["loss_history"]) == 2

        # Verify LoRA weights restored
        assert mx.allclose(
            model2.linear.lora_A, mx.ones_like(model2.linear.lora_A) * 0.5
        ).item()
        assert mx.allclose(
            model2.linear.lora_B, mx.ones_like(model2.linear.lora_B) * 0.3
        ).item()

    def test_dual_checkpoint_roundtrip(self, tmp_path):
        """Test save and load of a dual-expert checkpoint zip."""
        import mlx.optimizers as optim

        from mlx_video.training.save import (
            load_dual_checkpoint,
            save_dual_checkpoint,
        )

        high_model = self._make_lora_model()
        low_model = self._make_lora_model()
        high_opt = optim.AdamW(learning_rate=1e-4)
        low_opt = optim.AdamW(learning_rate=1e-4)
        high_opt.init(high_model.trainable_parameters())
        low_opt.init(low_model.trainable_parameters())

        # Set distinct known values for each expert
        high_model.linear.lora_A = mx.ones_like(high_model.linear.lora_A) * 0.7
        high_model.linear.lora_B = mx.ones_like(high_model.linear.lora_B) * 0.9
        low_model.linear.lora_A = mx.ones_like(low_model.linear.lora_A) * 0.2
        low_model.linear.lora_B = mx.ones_like(low_model.linear.lora_B) * 0.4
        mx.eval(
            high_model.parameters(),
            low_model.parameters(),
            high_opt.state,
            low_opt.state,
        )

        config = self._make_config(tmp_path)
        ckpt_path = str(tmp_path / "dual_checkpoint.zip")

        save_dual_checkpoint(
            high_model,
            low_model,
            high_opt,
            low_opt,
            config,
            epoch=5,
            global_step=50,
            loss_history_data=[(1, 0.8)],
            output_path=ckpt_path,
        )

        # Verify zip contents
        import zipfile

        with zipfile.ZipFile(ckpt_path) as zf:
            names = zf.namelist()
            assert "lora_high_noise.safetensors" in names
            assert "lora_low_noise.safetensors" in names
            assert "high_optimizer_state.safetensors" in names
            assert "low_optimizer_state.safetensors" in names
            assert "state.json" in names
            assert "config.json" in names

        # Load into fresh models
        high2 = self._make_lora_model()
        low2 = self._make_lora_model()
        high_opt2 = optim.AdamW(learning_rate=1e-4)
        low_opt2 = optim.AdamW(learning_rate=1e-4)
        high_opt2.init(high2.trainable_parameters())
        low_opt2.init(low2.trainable_parameters())
        mx.eval(high2.parameters(), low2.parameters(), high_opt2.state, low_opt2.state)

        state = load_dual_checkpoint(ckpt_path, high2, low2, high_opt2, low_opt2)

        assert state["epoch"] == 5
        assert state["expert_mode"] == "simultaneous"

        # Verify distinct weights restored to correct models
        assert mx.allclose(
            high2.linear.lora_A, mx.ones_like(high2.linear.lora_A) * 0.7
        ).item()
        assert mx.allclose(
            low2.linear.lora_A, mx.ones_like(low2.linear.lora_A) * 0.2
        ).item()
        assert mx.allclose(
            high2.linear.lora_B, mx.ones_like(high2.linear.lora_B) * 0.9
        ).item()
        assert mx.allclose(
            low2.linear.lora_B, mx.ones_like(low2.linear.lora_B) * 0.4
        ).item()

    def test_checkpoint_missing_file_raises(self):
        """Test that loading a nonexistent checkpoint raises FileNotFoundError."""
        from mlx_video.training.save import load_checkpoint

        with pytest.raises(FileNotFoundError):
            load_checkpoint("/nonexistent/path.zip", None, None)


class TestPreviewSignature:
    """Test preview function signature and CFG parameter handling."""

    def test_generate_preview_accepts_guide_scale(self):
        """Test that generate_preview has guide_scale parameter with correct default."""
        import inspect

        from mlx_video.training.preview import generate_preview

        sig = inspect.signature(generate_preview)
        assert "guide_scale" in sig.parameters
        assert sig.parameters["guide_scale"].default == 1.0
        assert sig.parameters["steps"].default == 20

    def test_generate_preview_returns_none_on_error(self):
        """Test that generate_preview catches errors and returns None."""
        from mlx_video.training.preview import generate_preview

        # Pass invalid inputs — should catch the error, not crash
        result = generate_preview(
            model=None, config=None, encoded_data=[], epoch=0, output_dir="/tmp"
        )
        assert result is None


class TestDualLossHistory:
    """Test LossHistory with separate H/L expert tracking."""

    def test_dual_loss_append(self):
        """Test that expert tags route to correct series."""
        from mlx_video.training.plotting import LossHistory

        h = LossHistory()
        h.append(1, 0.15, expert="H")
        h.append(1, 0.06, expert="L")
        h.append(2, 0.12, expert="H")
        h.append(2, 0.05)  # no expert tag

        assert len(h) == 4
        assert h.high_steps == [1, 2]
        assert h.high_losses == [0.15, 0.12]
        assert h.low_steps == [1]
        assert h.low_losses == [0.06]

    def test_dual_loss_empty_by_default(self):
        """Test that H/L series are empty when no expert tags used."""
        from mlx_video.training.plotting import LossHistory

        h = LossHistory()
        h.append(1, 0.10)
        h.append(2, 0.08)

        assert h.high_losses == []
        assert h.low_losses == []
        assert len(h) == 2

    def test_plot_loss_with_dual_series(self, tmp_path):
        """Test that plot_loss handles dual H/L series without error."""
        from mlx_video.training.plotting import LossHistory, plot_loss

        h = LossHistory()
        for i in range(1, 11):
            h.append(i, 0.15 - i * 0.005, expert="H")
            h.append(i, 0.08 - i * 0.003, expert="L")

        out = tmp_path / "dual_loss.png"
        plot_loss(h, out)
        assert out.exists()
        assert out.stat().st_size > 0


class TestBaseLoRAConfig:
    """Test base_loras config parsing and validation."""

    def _make_config_json(self, tmp_path, overrides=None):
        """Create a minimal valid config JSON for testing."""
        from PIL import Image

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        img = Image.new("RGB", (64, 64))
        img.save(data_dir / "img0.png")
        (data_dir / "img0.txt").write_text("a photo")

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"dim": 5120}))

        raw = {
            "model_dir": str(model_dir),
            "data": str(data_dir),
        }
        if overrides:
            raw.update(overrides)

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(raw))
        return config_path

    def test_base_loras_parsed(self, tmp_path):
        """Test base_loras array is parsed into BaseLoRAEntry objects."""
        from mlx_video.training.config import TrainingConfig

        # Create a fake LoRA file for validation
        lora_path = tmp_path / "lightning.safetensors"
        lora_path.write_bytes(b"fake")

        config_path = self._make_config_json(
            tmp_path,
            {
                "base_loras": [
                    {"path": str(lora_path), "expert": "both", "strength": 0.8},
                    {"path": str(lora_path), "expert": "low", "strength": 1.0},
                ]
            },
        )
        config = TrainingConfig.from_json(str(config_path))
        assert len(config.base_loras) == 2
        assert config.base_loras[0].expert == "both"
        assert config.base_loras[0].strength == 0.8
        assert config.base_loras[1].expert == "low"

    def test_base_loras_empty_default(self, tmp_path):
        """Test base_loras defaults to empty list."""
        from mlx_video.training.config import TrainingConfig

        config_path = self._make_config_json(tmp_path)
        config = TrainingConfig.from_json(str(config_path))
        assert config.base_loras == []

    def test_base_loras_invalid_expert_rejected(self, tmp_path):
        """Test invalid expert value is rejected."""
        from mlx_video.training.config import TrainingConfig

        lora_path = tmp_path / "lora.safetensors"
        lora_path.write_bytes(b"fake")

        config_path = self._make_config_json(
            tmp_path,
            {"base_loras": [{"path": str(lora_path), "expert": "invalid"}]},
        )
        with pytest.raises(ValueError, match="base_loras.*expert"):
            TrainingConfig.from_json(str(config_path))

    def test_base_loras_missing_file_rejected(self, tmp_path):
        """Test nonexistent LoRA path is rejected."""
        from mlx_video.training.config import TrainingConfig

        config_path = self._make_config_json(
            tmp_path,
            {"base_loras": [{"path": "/nonexistent/lora.safetensors"}]},
        )
        with pytest.raises(ValueError, match="base_loras.*not found"):
            TrainingConfig.from_json(str(config_path))


class TestPreviewConfigFields:
    """Test preview config fields and shift override."""

    def _make_config_json(self, tmp_path, overrides=None):
        from PIL import Image

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        img = Image.new("RGB", (64, 64))
        img.save(data_dir / "img0.png")
        (data_dir / "img0.txt").write_text("a photo")

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"dim": 5120}))

        raw = {"model_dir": str(model_dir), "data": str(data_dir)}
        if overrides:
            raw.update(overrides)

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(raw))
        return config_path

    def test_preview_steps_and_guide_scale(self, tmp_path):
        """Test custom preview_steps and preview_guide_scale from config."""
        from mlx_video.training.config import TrainingConfig

        config_path = self._make_config_json(
            tmp_path,
            {"monitoring": {"preview_steps": 8, "preview_guide_scale": 1.0}},
        )
        config = TrainingConfig.from_json(str(config_path))
        assert config.monitoring.preview_steps == 8
        assert config.monitoring.preview_guide_scale == 1.0

    def test_preview_defaults(self, tmp_path):
        """Test preview config has sensible defaults."""
        from mlx_video.training.config import TrainingConfig

        config_path = self._make_config_json(tmp_path)
        config = TrainingConfig.from_json(str(config_path))
        assert config.monitoring.preview_steps == 20
        assert config.monitoring.preview_guide_scale == 1.0

    def test_shift_override(self, tmp_path):
        """Test training.shift override from config."""
        from mlx_video.training.config import TrainingConfig

        config_path = self._make_config_json(
            tmp_path,
            {"training": {"shift": 3.0}},
        )
        config = TrainingConfig.from_json(str(config_path))
        assert config.training.shift == 3.0

    def test_shift_default_none(self, tmp_path):
        """Test training.shift defaults to None (use model default)."""
        from mlx_video.training.config import TrainingConfig

        config_path = self._make_config_json(tmp_path)
        config = TrainingConfig.from_json(str(config_path))
        assert config.training.shift is None

    def test_high_ratio_parsed(self, tmp_path):
        """Test high_ratio is parsed from config."""
        from mlx_video.training.config import TrainingConfig

        config_path = self._make_config_json(
            tmp_path,
            {"training": {"high_ratio": 0.25}},
        )
        config = TrainingConfig.from_json(str(config_path))
        assert config.training.high_ratio == 0.25

    def test_high_ratio_default_none(self, tmp_path):
        """Test high_ratio defaults to None (derive from boundary)."""
        from mlx_video.training.config import TrainingConfig

        config_path = self._make_config_json(tmp_path)
        config = TrainingConfig.from_json(str(config_path))
        assert config.training.high_ratio is None

    def test_high_ratio_invalid_rejected(self, tmp_path):
        """Test high_ratio outside (0, 1) is rejected."""
        from mlx_video.training.config import TrainingConfig

        for i, bad_value in enumerate([0.0, 1.0, -0.1, 1.5]):
            sub = tmp_path / f"run_{i}"
            sub.mkdir()
            config_path = self._make_config_json(
                sub,
                {"training": {"high_ratio": bad_value}},
            )
            with pytest.raises(ValueError, match="high_ratio"):
                TrainingConfig.from_json(str(config_path))


class TestQLoRA:
    """Test LoRA on QuantizedLinear layers (QLoRA support)."""

    def test_trainable_lora_on_quantized_linear(self):
        """Test TrainableLoRALinear wraps QuantizedLinear correctly."""
        from mlx_video.training.lora_layers import TrainableLoRALinear

        # Wrap linear in a module so nn.quantize can replace it
        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(64, 128)

        wrapper = Wrapper()
        mx.eval(wrapper.parameters())
        nn.quantize(wrapper, bits=4, group_size=64)
        mx.eval(wrapper.parameters())

        linear = wrapper.layer
        assert isinstance(linear, nn.QuantizedLinear)

        lora = TrainableLoRALinear(linear, rank=8, alpha=8.0)

        # Forward pass should work
        x = mx.random.normal((2, 64))
        output = lora(x)
        mx.eval(output)
        assert output.shape == (2, 128)

    def test_lora_on_quantized_initial_zero(self):
        """Test that LoRA output starts at zero (lora_B is zero-initialized)."""
        from mlx_video.training.lora_layers import TrainableLoRALinear

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(64, 128)

        wrapper = Wrapper()
        mx.eval(wrapper.parameters())
        nn.quantize(wrapper, bits=4, group_size=64)
        mx.eval(wrapper.parameters())

        linear = wrapper.layer
        lora = TrainableLoRALinear(linear, rank=8, alpha=8.0)
        x = mx.random.normal((2, 64))
        base_output = linear(x)
        lora_output = lora(x)
        mx.eval(base_output, lora_output)
        assert mx.allclose(base_output, lora_output, atol=1e-4).item()

    def test_inject_lora_on_quantized_model(self):
        """Test LoRA injection into a model with QuantizedLinear layers."""
        from mlx_video.training.config import BlockRange, LoRAConfig
        from mlx_video.training.lora_layers import (
            TrainableLoRALinear,
            inject_lora_layers,
        )

        class FakeAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q = nn.Linear(64, 64)
                self.k = nn.Linear(64, 64)

        class FakeBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = FakeAttn()

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = [FakeBlock() for _ in range(2)]

        model = FakeModel()
        mx.eval(model.parameters())

        # Quantize the model (targets q and k layers)
        nn.quantize(
            model,
            bits=4,
            group_size=64,
            class_predicate=lambda p, m: isinstance(m, nn.Linear),
        )
        mx.eval(model.parameters())

        # Verify layers are quantized
        assert isinstance(model.blocks[0].self_attn.q, nn.QuantizedLinear)

        # Inject LoRA on quantized layers
        lora_config = LoRAConfig(
            rank=4,
            alpha=4.0,
            targets=["self_attn.q", "self_attn.k"],
            blocks=BlockRange(start=0, end=2),
        )
        count = inject_lora_layers(model, lora_config)
        assert count == 4  # 2 blocks × 2 targets

        # Verify LoRA wraps quantized linear
        wrapped = model.blocks[0].self_attn.q
        assert isinstance(wrapped, TrainableLoRALinear)
        assert isinstance(wrapped.linear, nn.QuantizedLinear)

    def test_quantized_lora_dimensions(self):
        """Test that LoRA A/B matrices have correct dimensions for quantized base."""
        from mlx_video.training.lora_layers import TrainableLoRALinear

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(256, 512)

        wrapper = Wrapper()
        mx.eval(wrapper.parameters())
        nn.quantize(wrapper, bits=4, group_size=64)
        mx.eval(wrapper.parameters())

        lora = TrainableLoRALinear(wrapper.layer, rank=16, alpha=16.0)

        # lora_A: (rank, in_features) = (16, 256)
        assert lora.lora_A.shape == (16, 256)
        # lora_B: (out_features, rank) = (512, 16)
        assert lora.lora_B.shape == (512, 16)
