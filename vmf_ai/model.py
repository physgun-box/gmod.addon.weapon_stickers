"""Neural network architecture for VMF brush-layout generation."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GeneratorConfig:
    """Configuration describing the layout generator network."""

    max_brushes: int
    feature_dim: int = 6
    hidden_dim: int = 512
    latent_dim: int = 64
    decoder_hidden_dim: int = 256
    material_vocab_size: int = 32
    material_embedding_dim: int = 16


class VMFBrushGenerator(nn.Module):
    """Variational autoencoder modelling VMF brush layouts."""

    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__()
        self.config = config

        encoder_input_dim = config.max_brushes * (config.feature_dim + config.material_embedding_dim)

        self.material_embedding = nn.Embedding(config.material_vocab_size, config.material_embedding_dim)

        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.encoder_mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.encoder_logvar = nn.Linear(config.hidden_dim, config.latent_dim)

        decoder_output_dim = config.max_brushes * config.decoder_hidden_dim
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, decoder_output_dim),
            nn.GELU(),
        )
        self.feature_head = nn.Linear(config.decoder_hidden_dim, config.feature_dim)
        self.material_head = nn.Linear(config.decoder_hidden_dim, config.material_vocab_size)
        self.presence_head = nn.Linear(config.decoder_hidden_dim, 1)

    # ------------------------------------------------------------------
    def _encode(self, features: torch.Tensor, materials: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.material_embedding(materials)
        x = torch.cat([features, embedded], dim=-1)
        x = x * mask.unsqueeze(-1).float()
        x = x.view(x.size(0), -1)
        hidden = self.encoder(x)
        mu = self.encoder_mu(hidden)
        logvar = self.encoder_logvar(hidden)
        return mu, logvar

    # ------------------------------------------------------------------
    def _reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ------------------------------------------------------------------
    def _decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = z.size(0)
        hidden = self.decoder(z)
        hidden = hidden.view(batch, self.config.max_brushes, self.config.decoder_hidden_dim)
        feature_pred = self.feature_head(hidden)
        material_logits = self.material_head(hidden)
        presence_logits = self.presence_head(hidden).squeeze(-1)
        return feature_pred, material_logits, presence_logits

    # ------------------------------------------------------------------
    def forward(
        self,
        features: torch.Tensor,
        materials: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mu, logvar = self._encode(features, materials, mask)
        z = self._reparameterise(mu, logvar)
        feature_pred, material_logits, presence_logits = self._decode(z)
        return {
            "feature_pred": feature_pred,
            "material_logits": material_logits,
            "presence_logits": presence_logits,
            "mu": mu,
            "logvar": logvar,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, num_samples: int) -> dict[str, torch.Tensor]:
        z = torch.randn(num_samples, self.config.latent_dim, device=next(self.parameters()).device)
        feature_pred, material_logits, presence_logits = self._decode(z)
        return {
            "feature_pred": feature_pred,
            "material_logits": material_logits,
            "presence_logits": presence_logits,
        }
