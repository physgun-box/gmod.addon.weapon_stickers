"""AI utilities for procedural VMF brush generation."""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .dataset import VMFLayoutDataset, build_layout_dataset, load_vmf_paths
from .model import GeneratorConfig, VMFBrushGenerator
from .tokenizer import MaterialVocabulary

if TYPE_CHECKING:  # pragma: no cover - only evaluated by type checkers
    from .generation import GeneratedBrush, GeneratorAssets, generate_map, generate_vmf_text, load_generator
    from .training import TrainerConfig, train_generator

__all__ = [
    "MaterialVocabulary",
    "VMFLayoutDataset",
    "build_layout_dataset",
    "load_vmf_paths",
    "GeneratorConfig",
    "VMFBrushGenerator",
    "TrainerConfig",
    "train_generator",
    "GeneratedBrush",
    "GeneratorAssets",
    "load_generator",
    "generate_map",
    "generate_vmf_text",
]


def __getattr__(name: str) -> Any:
    if name in {"TrainerConfig", "train_generator"}:
        module = import_module(".training", __name__)
        return getattr(module, name)
    if name in {"GeneratedBrush", "GeneratorAssets", "load_generator", "generate_map", "generate_vmf_text"}:
        module = import_module(".generation", __name__)
        return getattr(module, name)
    raise AttributeError(name)
