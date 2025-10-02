"""Text tokeniser for VMF files."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

TOKEN_PATTERN = re.compile(r'\{|\}|\(|\)|"[^"\n]*"|//[^\n]*|[^\s{}()"/]+')


class VMFTokenizer:
    """Whitespace and punctuation aware tokeniser for VMF documents."""

    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    unk_token = "<unk>"

    def __init__(self, extra_tokens: Iterable[str] | None = None) -> None:
        extra = list(extra_tokens or [])
        specials = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        self._vocab: Dict[str, int] = {token: idx for idx, token in enumerate(specials + extra)}
        self._inverse_vocab: List[str] = list(self._vocab.keys())

    # ------------------------------------------------------------------
    # Vocabulary helpers
    # ------------------------------------------------------------------
    @property
    def pad_id(self) -> int:
        return self._vocab[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self._vocab[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self._vocab[self.eos_token]

    @property
    def unk_id(self) -> int:
        return self._vocab[self.unk_token]

    def __len__(self) -> int:
        return len(self._vocab)

    # ------------------------------------------------------------------
    def fit(self, texts: Iterable[str]) -> None:
        """Populate the vocabulary with tokens found in ``texts``."""
        for text in texts:
            for token in self.tokenise(text):
                if token not in self._vocab:
                    self._vocab[token] = len(self._vocab)
                    self._inverse_vocab.append(token)

    # ------------------------------------------------------------------
    def tokenise(self, text: str) -> List[str]:
        """Split VMF text into a list of tokens."""
        tokens = TOKEN_PATTERN.findall(text)
        cleaned: List[str] = []
        for token in tokens:
            if token.startswith("//"):
                continue  # Strip single-line comments entirely
            cleaned.append(token)
        return cleaned

    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        """Return token ids with BOS/EOS markers."""
        tokens = [self.bos_token, *self.tokenise(text), self.eos_token]
        return [self._vocab.get(tok, self.unk_id) for tok in tokens]

    # ------------------------------------------------------------------
    def decode(self, token_ids: Sequence[int]) -> str:
        """Convert a sequence of token ids back to VMF text."""
        tokens: List[str] = []
        for token_id in token_ids:
            if token_id in (self.bos_id, self.eos_id, self.pad_id):
                continue
            if token_id >= len(self._inverse_vocab):
                tokens.append(self.unk_token)
            else:
                tokens.append(self._inverse_vocab[token_id])

        output: List[str] = []
        for idx, token in enumerate(tokens):
            if token in {"{", "}", "(", ")"}:
                if idx > 0 and output and not output[-1].endswith(" "):
                    output.append(" ")
                output.append(token)
                continue
            if idx > 0 and token[0] not in ')}':
                output.append(" ")
            output.append(token)
        return "".join(output)

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        state = {
            "vocab": self._vocab,
            "inverse_vocab": self._inverse_vocab,
        }
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "VMFTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        tokenizer = cls()
        tokenizer._vocab = {str(k): int(v) for k, v in data["vocab"].items()}
        tokenizer._inverse_vocab = list(data["inverse_vocab"])
        return tokenizer
