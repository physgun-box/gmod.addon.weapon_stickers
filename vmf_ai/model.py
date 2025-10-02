"""Neural network architecture for VMF language modelling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    feedforward_dim: int = 1024
    max_sequence_length: int = 2048


class VMFTransformerLM(nn.Module):
    """A causal Transformer language model specialised for VMF tokens."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embed = nn.Embedding(config.max_sequence_length, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size)

    # ------------------------------------------------------------------
    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((size, size), float("-inf"), device=device)
        mask.triu_(1)
        return mask

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len = input_ids.size()
        if seq_len > self.config.max_sequence_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds model maximum {self.config.max_sequence_length}"
            )

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embed(input_ids) + self.position_embed(positions)
        hidden = self.layer_norm(hidden)

        padding_mask = None
        if attention_mask is not None:
            padding_mask = attention_mask == 0

        logits = self.transformer(
            hidden,
            mask=self._causal_mask(seq_len, input_ids.device),
            src_key_padding_mask=padding_mask,
        )
        logits = self.output(logits)
        return logits

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        self.eval()

        generated = input_ids.clone()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        for _ in range(max_new_tokens):
            logits = self.forward(generated, attention_mask)
            next_token_logits = logits[:, -1, :]
            next_token_logits = next_token_logits / max(temperature, 1e-5)

            if top_k > 0:
                top_values, _ = torch.topk(next_token_logits, top_k)
                min_threshold = top_values[:, -1].unsqueeze(-1)
                filtered = torch.where(
                    next_token_logits < min_threshold, torch.full_like(next_token_logits, -float("inf")), next_token_logits
                )
                probs = torch.softmax(filtered, dim=-1)
            else:
                probs = torch.softmax(next_token_logits, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_tokens], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_tokens)], dim=1)

        return generated
