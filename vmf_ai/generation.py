"""Utility helpers for sampling VMF text from a trained model."""
from __future__ import annotations

from pathlib import Path

import torch

from .model import ModelConfig, VMFTransformerLM
from .tokenizer import VMFTokenizer


def load_model(checkpoint_path: Path, tokenizer: VMFTokenizer) -> VMFTransformerLM:
    """Load a VMFTransformerLM from ``checkpoint_path`` using ``tokenizer``."""
    data = torch.load(checkpoint_path, map_location="cpu")
    config_dict = dict(data["model_config"])
    config_dict["vocab_size"] = len(tokenizer)
    model_config = ModelConfig(**config_dict)
    model = VMFTransformerLM(model_config)
    model.load_state_dict(data["model_state"])
    return model


def generate_vmf_text(
    model: VMFTransformerLM,
    tokenizer: VMFTokenizer,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "auto",
) -> str:
    """Generate VMF-like text conditioned on ``prompt``."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    model = model.to(device_obj)
    model.eval()

    encoded = torch.tensor([tokenizer.encode(prompt)], device=device_obj)
    attention_mask = torch.ones_like(encoded)
    generated = model.generate(
        encoded,
        max_new_tokens=max_tokens,
        attention_mask=attention_mask,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(generated[0].tolist())
