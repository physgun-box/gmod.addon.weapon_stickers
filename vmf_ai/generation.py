"""Utility helpers for sampling VMF brush layouts from a trained model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch

from vmf_tools import VMFBuilder, Vector3

from .model import GeneratorConfig, VMFBrushGenerator
from .tokenizer import MaterialVocabulary


@dataclass
class GeneratedBrush:
    material: str
    min_corner: Vector3
    max_corner: Vector3


@dataclass
class GeneratorAssets:
    model: VMFBrushGenerator
    vocabulary: MaterialVocabulary
    feature_mean: torch.Tensor
    feature_std: torch.Tensor


def load_generator(checkpoint_path: Path) -> GeneratorAssets:
    """Load a :class:`VMFBrushGenerator` and its metadata from disk."""

    data = torch.load(checkpoint_path, map_location="cpu")
    config_dict = dict(data["model_config"])
    vocabulary = MaterialVocabulary.from_json(data["vocabulary"])
    config_dict["material_vocab_size"] = len(vocabulary)
    model = VMFBrushGenerator(GeneratorConfig(**config_dict))
    model.load_state_dict(data["model_state"])
    mean = torch.tensor(data["feature_mean"], dtype=torch.float32)
    std = torch.tensor(data["feature_std"], dtype=torch.float32)
    return GeneratorAssets(model=model, vocabulary=vocabulary, feature_mean=mean, feature_std=std)


def _denormalise(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return features * std + mean


def _build_brushes(
    features: torch.Tensor,
    materials: torch.Tensor,
    presence: torch.Tensor,
    vocabulary: MaterialVocabulary,
    *,
    presence_threshold: float,
) -> List[GeneratedBrush]:
    brushes: List[GeneratedBrush] = []
    probs = torch.sigmoid(presence)
    for idx in range(features.size(0)):
        if probs[idx].item() < presence_threshold:
            continue
        center = features[idx, :3]
        size = features[idx, 3:]
        size = torch.clamp(size, min=1.0)
        half = size / 2.0
        min_corner = center - half
        max_corner = center + half
        material_id = torch.argmax(materials[idx]).item()
        material = vocabulary.decode(material_id)
        brushes.append(
            GeneratedBrush(
                material=material,
                min_corner=Vector3(*min_corner.tolist()),
                max_corner=Vector3(*max_corner.tolist()),
            )
        )
    return brushes


def generate_map(
    assets: GeneratorAssets,
    *,
    device: str = "auto",
    presence_threshold: float = 0.5,
) -> List[GeneratedBrush]:
    """Sample a new layout from the trained generator."""

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    model = assets.model.to(device_obj)
    model.eval()

    sample = model.sample(num_samples=1)
    feature_pred = sample["feature_pred"].squeeze(0).cpu()
    material_logits = sample["material_logits"].squeeze(0).cpu()
    presence_logits = sample["presence_logits"].squeeze(0).cpu()

    denorm = _denormalise(feature_pred, assets.feature_mean, assets.feature_std)
    return _build_brushes(denorm, material_logits, presence_logits, assets.vocabulary, presence_threshold=presence_threshold)


def brushes_to_vmf(brushes: Iterable[GeneratedBrush]) -> str:
    builder = VMFBuilder()
    for brush in brushes:
        builder.add_axis_aligned_block(brush.min_corner, brush.max_corner, material=brush.material)
    return builder.build()


def generate_vmf_text(
    checkpoint: Path,
    *,
    device: str = "auto",
    presence_threshold: float = 0.5,
) -> str:
    """Convenience wrapper combining :func:`generate_map` and :func:`brushes_to_vmf`."""

    assets = load_generator(checkpoint)
    brushes = generate_map(assets, device=device, presence_threshold=presence_threshold)
    return brushes_to_vmf(brushes)
