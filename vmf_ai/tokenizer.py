"""Material vocabulary management for VMF generation models."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class MaterialStatistics:
    """Summary of materials present in the training corpus."""

    counts: Dict[str, int]

    def most_common(self, n: int | None = None) -> List[tuple[str, int]]:
        counter = Counter(self.counts)
        return counter.most_common(n)


class MaterialVocabulary:
    """Bidirectional mapping between material names and integer ids."""

    pad_token = "<pad>"
    unk_token = "<unk>"

    def __init__(self, materials: Iterable[str] | None = None, *, keep_top_k: int | None = None) -> None:
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: List[str] = []
        self._counts: Counter[str] = Counter()
        self._ensure_token(self.pad_token)
        self._ensure_token(self.unk_token)
        if materials is not None:
            self.fit(materials, keep_top_k=keep_top_k)

    # ------------------------------------------------------------------
    def _ensure_token(self, token: str) -> None:
        if token not in self._token_to_id:
            self._token_to_id[token] = len(self._id_to_token)
            self._id_to_token.append(token)

    # ------------------------------------------------------------------
    @property
    def pad_id(self) -> int:
        return self._token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self._token_to_id[self.unk_token]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._id_to_token)

    # ------------------------------------------------------------------
    def fit(self, materials: Iterable[str], *, keep_top_k: int | None = None) -> None:
        """Populate the vocabulary from ``materials``.

        Parameters
        ----------
        materials:
            Collection of material names observed in the dataset.
        keep_top_k:
            When provided, limits the vocabulary to the ``keep_top_k`` most
            common materials. Rarer entries are mapped to :attr:`unk_token`.
        """

        self._counts.update(materials)
        ordered = [token for token, _ in self._counts.most_common()]
        if keep_top_k is not None:
            limit = max(keep_top_k, 2) - 2  # reserve slots for pad/unk
            ordered = ordered[:limit]
        for token in ordered:
            if token in {self.pad_token, self.unk_token}:
                continue
            self._ensure_token(token)

    # ------------------------------------------------------------------
    def encode(self, material: str) -> int:
        """Return the integer id for ``material``."""

        return self._token_to_id.get(material, self.unk_id)

    # ------------------------------------------------------------------
    def decode(self, material_id: int) -> str:
        """Inverse mapping from ``material_id`` to the original name."""

        if 0 <= material_id < len(self._id_to_token):
            return self._id_to_token[material_id]
        return self.unk_token

    # ------------------------------------------------------------------
    def statistics(self) -> MaterialStatistics:
        return MaterialStatistics(dict(self._counts))

    # ------------------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "tokens": self._id_to_token,
            "counts": dict(self._counts),
        }

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    @classmethod
    def from_json(cls, data: dict) -> "MaterialVocabulary":
        vocab = cls()
        vocab._id_to_token = list(data.get("tokens", []))
        vocab._token_to_id = {token: idx for idx, token in enumerate(vocab._id_to_token)}
        vocab._counts = Counter({str(k): int(v) for k, v in data.get("counts", {}).items()})
        # Guarantee the special tokens exist even when loading from older files
        vocab._ensure_token(cls.pad_token)
        vocab._ensure_token(cls.unk_token)
        return vocab

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "MaterialVocabulary":
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))
