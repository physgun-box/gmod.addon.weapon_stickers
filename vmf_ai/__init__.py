"""Utilities for training and sampling VMF language models."""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .tokenizer import VMFTokenizer
from .dataset import VMFDataset, load_vmf_paths
from .model import ModelConfig, VMFTransformerLM

if TYPE_CHECKING:  # pragma: no cover - only evaluated by type checkers
    from .generation import generate_vmf_text
    from .training import TrainerConfig, train_language_model

__all__ = [
    "VMFTokenizer",
    "VMFDataset",
    "load_vmf_paths",
    "ModelConfig",
    "VMFTransformerLM",
    "TrainerConfig",
    "train_language_model",
    "generate_vmf_text",
]


def __getattr__(name: str) -> Any:
    if name in {"TrainerConfig", "train_language_model"}:
        module = import_module(".training", __name__)
        return getattr(module, name)
    if name == "generate_vmf_text":
        module = import_module(".generation", __name__)
        return getattr(module, name)
    raise AttributeError(name)
