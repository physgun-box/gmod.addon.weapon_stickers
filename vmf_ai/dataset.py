"""Dataset utilities for VMF language model training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch
from torch.utils.data import Dataset

from .tokenizer import VMFTokenizer


@dataclass(frozen=True)
class VMFSample:
    """A single VMF text sample."""

    path: Path
    text: str


class VMFDataset(Dataset[torch.Tensor]):
    """PyTorch dataset yielding tokenised VMF documents."""

    def __init__(
        self,
        samples: Sequence[VMFSample],
        tokenizer: VMFTokenizer,
        max_tokens: int | None = None,
    ) -> None:
        self._samples = list(samples)
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens

        if not self._samples:
            raise ValueError("VMFDataset received no samples")
        if self._max_tokens is not None and self._max_tokens < 4:
            raise ValueError("max_tokens must be at least 4 to keep BOS/EOS markers")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = self._samples[index]
        token_ids = self._tokenizer.encode(sample.text)
        if self._max_tokens is not None and len(token_ids) > self._max_tokens:
            # Keep BOS and EOS tokens intact while trimming the middle section.
            usable = self._max_tokens - 2
            half = max(1, usable // 2)
            bos = token_ids[0]
            eos = token_ids[-1]
            head = token_ids[1 : 1 + half]
            tail = token_ids[-(usable - half) - 1 : -1]
            token_ids = [bos, *head, *tail, eos]
        return torch.tensor(token_ids, dtype=torch.long)


def load_vmf_paths(directory: Path) -> List[Path]:
    """Return all VMF files inside ``directory`` (non-recursive)."""
    paths = sorted(directory.glob("*.vmf"))
    if not paths:
        raise FileNotFoundError(f"No VMF files found in {directory}")
    return paths


def build_dataset(directory: Path, tokenizer: VMFTokenizer, max_tokens: int | None = None) -> VMFDataset:
    """Create a :class:`VMFDataset` from all VMF files in ``directory``."""
    samples = [VMFSample(path=path, text=path.read_text(encoding="utf-8")) for path in load_vmf_paths(directory)]
    return VMFDataset(samples, tokenizer, max_tokens=max_tokens)


def collate_tokens(batch: Sequence[torch.Tensor], pad_id: int) -> dict[str, torch.Tensor]:
    """Pad a batch of token sequences for language-model training."""
    if not batch:
        raise ValueError("Empty batch passed to collate_tokens")

    max_len = max(item.size(0) for item in batch)
    batch_size = len(batch)

    padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    for idx, seq in enumerate(batch):
        padded[idx, : seq.size(0)] = seq

    attention_mask = (padded != pad_id).long()
    input_ids = padded[:, :-1]
    labels = padded[:, 1:].clone()
    attention_mask = attention_mask[:, :-1]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
