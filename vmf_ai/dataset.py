"""Dataset utilities for VMF geometry generation models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from vmf_tools.geometry import Vector3
from vmf_tools.parser import VMFMap, load_vmf

from .tokenizer import MaterialVocabulary


@dataclass(frozen=True)
class BrushSample:
    """Axis-aligned bounding box extracted from a VMF brush."""

    center: Vector3
    size: Vector3
    material: str


@dataclass(frozen=True)
class MapSample:
    """Collection of brushes extracted from a VMF map."""

    path: Path
    brushes: List[BrushSample]


class VMFLayoutDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset returning normalised brush parameters for each map."""

    feature_dim = 6  # center xyz + size xyz

    def __init__(
        self,
        samples: Sequence[MapSample],
        vocabulary: MaterialVocabulary,
        max_brushes: int,
    ) -> None:
        if not samples:
            raise ValueError("VMFLayoutDataset requires at least one sample")
        self.samples = list(samples)
        self.vocabulary = vocabulary
        self.max_brushes = max_brushes

        feature_vectors: List[np.ndarray] = []
        for sample in self.samples:
            for brush in sample.brushes:
                feature_vectors.append(
                    np.array(
                        [
                            brush.center.x,
                            brush.center.y,
                            brush.center.z,
                            brush.size.x,
                            brush.size.y,
                            brush.size.z,
                        ],
                        dtype=np.float32,
                    )
                )
        if not feature_vectors:
            raise ValueError("No brush geometry found in provided VMF samples")
        stacked = np.stack(feature_vectors, axis=0)
        self.feature_mean = torch.from_numpy(stacked.mean(axis=0))
        self.feature_std = torch.from_numpy(np.maximum(stacked.std(axis=0), 1e-3))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        brush_count = min(len(sample.brushes), self.max_brushes)

        features = torch.zeros((self.max_brushes, self.feature_dim), dtype=torch.float32)
        material_ids = torch.full((self.max_brushes,), self.vocabulary.pad_id, dtype=torch.long)
        mask = torch.zeros((self.max_brushes,), dtype=torch.bool)

        for i in range(brush_count):
            brush = sample.brushes[i]
            raw = torch.tensor(
                [
                    brush.center.x,
                    brush.center.y,
                    brush.center.z,
                    brush.size.x,
                    brush.size.y,
                    brush.size.z,
                ],
                dtype=torch.float32,
            )
            normalised = (raw - self.feature_mean) / self.feature_std
            features[i] = normalised
            material_ids[i] = self.vocabulary.encode(brush.material)
            mask[i] = True

        return {
            "features": features,
            "materials": material_ids,
            "mask": mask,
        }

    # ------------------------------------------------------------------
    def denormalise(self, features: torch.Tensor) -> torch.Tensor:
        """Inverse of the normalisation applied in :meth:`__getitem__`."""

        return features * self.feature_std + self.feature_mean


# ----------------------------------------------------------------------
def load_vmf_paths(directory: Path) -> List[Path]:
    """Return all VMF files inside ``directory`` (non-recursive)."""
    paths = sorted(directory.glob("*.vmf"))
    if not paths:
        raise FileNotFoundError(f"No VMF files found in {directory}")
    return paths


def extract_brushes(map_data: VMFMap) -> List[BrushSample]:
    brushes: List[BrushSample] = []
    for solid in map_data.solids:
        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []
        material = solid.faces[0].material if solid.faces else "DEV/DEV_MEASUREWALL01C"
        for face in solid.faces:
            for vertex in face.points:
                xs.append(vertex.x)
                ys.append(vertex.y)
                zs.append(vertex.z)
        if not xs:
            continue
        min_corner = Vector3(min(xs), min(ys), min(zs))
        max_corner = Vector3(max(xs), max(ys), max(zs))
        size = Vector3(max_corner.x - min_corner.x, max_corner.y - min_corner.y, max_corner.z - min_corner.z)
        center = Vector3(
            min_corner.x + size.x / 2.0,
            min_corner.y + size.y / 2.0,
            min_corner.z + size.z / 2.0,
        )
        brushes.append(BrushSample(center=center, size=size, material=material))
    return brushes


def build_layout_dataset(
    directory: Path,
    *,
    max_brushes: int | None = None,
    material_limit: int | None = None,
) -> tuple[VMFLayoutDataset, MaterialVocabulary]:
    """Create a :class:`VMFLayoutDataset` from all VMF files in ``directory``."""

    paths = load_vmf_paths(directory)
    samples: List[MapSample] = []
    all_materials: List[str] = []
    for path in paths:
        map_data = load_vmf(str(path))
        brushes = extract_brushes(map_data)
        samples.append(MapSample(path=path, brushes=brushes))
        all_materials.extend(brush.material for brush in brushes)

    if max_brushes is None:
        max_brushes = max((len(sample.brushes) for sample in samples), default=0)
    if max_brushes <= 0:
        raise ValueError("max_brushes must be positive")

    vocab = MaterialVocabulary(all_materials, keep_top_k=material_limit)
    dataset = VMFLayoutDataset(samples, vocab, max_brushes=max_brushes)
    return dataset, vocab
