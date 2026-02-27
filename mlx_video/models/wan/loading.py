"""Wan model loading utilities."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def load_wan_model(model_path: Path, config, quantization: dict | None = None):
    """Load and initialize WanModel, with optional quantization support.

    Args:
        model_path: Path to model safetensors file
        config: WanModelConfig
        quantization: Optional dict with 'bits' and 'group_size' keys.
                      If provided, creates QuantizedLinear stubs before loading.
    """
    from mlx_video.models.wan.model import WanModel

    model = WanModel(config)

    if quantization:
        from mlx_video.convert_wan import _quantize_predicate

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            class_predicate=lambda path, m: _quantize_predicate(path, m),
        )

    weights = mx.load(str(model_path))
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return model


def load_t5_encoder(model_path: Path, config):
    """Load T5 text encoder.

    Weights are upcast to float32 for maximum precision — the T5 encoder
    only runs once per generation, so performance impact is negligible.
    This matches the official which computes softmax in float32 explicitly.
    """
    from mlx_video.models.wan.text_encoder import T5Encoder

    encoder = T5Encoder(
        vocab_size=config.t5_vocab_size,
        dim=config.t5_dim,
        dim_attn=config.t5_dim_attn,
        dim_ffn=config.t5_dim_ffn,
        num_heads=config.t5_num_heads,
        num_layers=config.t5_num_layers,
        num_buckets=config.t5_num_buckets,
        shared_pos=False,
    )
    weights = mx.load(str(model_path))
    weights = {k: v.astype(mx.float32) for k, v in weights.items()}
    encoder.load_weights(list(weights.items()))
    mx.eval(encoder.parameters())
    return encoder


def load_vae_decoder(model_path: Path, config=None):
    """Load VAE decoder (skips encoder weights with strict=False).

    For Wan2.2 (vae_z_dim=48), uses Wan22VAEDecoder.
    For Wan2.1 (vae_z_dim=16), uses WanVAE.
    """
    is_wan22 = config is not None and config.vae_z_dim == 48

    if is_wan22:
        from mlx_video.models.wan.vae22 import Wan22VAEDecoder
        vae = Wan22VAEDecoder(z_dim=48)
    else:
        from mlx_video.models.wan.vae import WanVAE
        vae = WanVAE(z_dim=16)

    weights = mx.load(str(model_path))
    # Upcast VAE weights to float32 for quality — official Wan2.2 runs VAE in float32
    weights = {k: v.astype(mx.float32) for k, v in weights.items()}
    vae.load_weights(list(weights.items()), strict=False)
    mx.eval(vae.parameters())
    return vae


def load_vae_encoder(model_path: Path, config=None):
    """Load VAE encoder for I2V image encoding.

    Only supports Wan2.2 (vae_z_dim=48).
    """
    from mlx_video.models.wan.vae22 import Wan22VAEEncoder

    encoder = Wan22VAEEncoder(z_dim=config.vae_z_dim)
    weights = mx.load(str(model_path))
    weights = {k: v.astype(mx.float32) for k, v in weights.items()}
    encoder.load_weights(list(weights.items()), strict=False)
    mx.eval(encoder.parameters())
    return encoder


def _clean_text(text: str) -> str:
    """Clean text matching official Wan2.2 tokenizer preprocessing.

    Applies ftfy.fix_text (fixes mojibake, normalizes fullwidth chars),
    double HTML unescape, and whitespace normalization. Critical for
    correct tokenization of the Chinese negative prompt.
    """
    import html
    import re

    try:
        import ftfy
        text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = html.unescape(html.unescape(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_text(
    encoder,
    tokenizer,
    prompt: str,
    text_len: int = 512,
) -> mx.array:
    """Encode text prompt using T5 encoder.

    Args:
        encoder: T5Encoder model
        tokenizer: HuggingFace tokenizer
        prompt: Text prompt
        text_len: Maximum text length

    Returns:
        Text embeddings [L, dim]
    """
    prompt = _clean_text(prompt)
    tokens = tokenizer(
        prompt,
        max_length=text_len,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    ids = mx.array(tokens["input_ids"])
    mask = mx.array(tokens["attention_mask"])

    embeddings = encoder(ids, mask=mask)

    # Return only non-padding tokens
    seq_len = int(mask.sum().item())
    return embeddings[0, :seq_len]
