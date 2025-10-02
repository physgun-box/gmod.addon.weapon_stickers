"""Neural network components for generative VMF map synthesis."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from vmf_tools import parser


@dataclass
class BrushToken:
    """Compact representation of a single brush primitive."""

    position: Tensor  # (3,) xyz center
    size: Tensor  # (3,) side lengths
    rotation: Tensor  # (3,) euler angles radians
    material_id: int


class VMFTokenizer:
    """Convert VMF solids/entities into normalized brush tokens."""

    def __init__(self, materials: Optional[Sequence[str]] = None) -> None:
        self._materials: List[str] = list(materials or [])

    def encode_solid(self, solid: parser.VMFSolid) -> Optional[BrushToken]:
        triangles = solid.build_geometry()
        if not triangles:
            return None
        verts = torch.tensor([[v.x, v.y, v.z] for tri in triangles for v in tri], dtype=torch.float32)
        center = verts.mean(dim=0)
        size = (verts.max(dim=0).values - verts.min(dim=0).values).clamp(min=1.0)
        material = solid.faces[0].material.lower()
        try:
            material_index = self._materials.index(material)
        except ValueError:
            self._materials.append(material)
            material_index = len(self._materials) - 1
        rotation = torch.zeros(3)
        return BrushToken(position=center, size=size, rotation=rotation, material_id=material_index)

    def decode_solid(self, token: BrushToken) -> parser.VMFSolid:
        # Reconstruction is delegated to VMFBuilder utilities.
        raise NotImplementedError("Decoding is handled by vmf_tools.builder in the generation script.")

    @property
    def material_vocab(self) -> Sequence[str]:
        return self._materials


class VMFDataset(Dataset[Tuple[Tensor, Tensor]]):
    """Dataset that serves brush token sequences and descriptor embeddings."""

    def __init__(self, vmf_paths: Iterable[Path], tokenizer: VMFTokenizer, max_brushes: int = 512) -> None:
        self._samples: List[Tuple[Tensor, Tensor]] = []
        self._tokenizer = tokenizer
        self._max_brushes = max_brushes
        for path in vmf_paths:
            vmf = parser.load_vmf(str(path))
            tokens: List[BrushToken] = []
            for solid in vmf.solids:
                token = tokenizer.encode_solid(solid)
                if token:
                    tokens.append(token)
                    if len(tokens) >= max_brushes:
                        break
            if not tokens:
                continue
            brush_tensor = torch.stack([torch.cat([t.position, t.size, t.rotation]) for t in tokens])
            material_ids = torch.tensor([t.material_id for t in tokens], dtype=torch.long)
            self._samples.append((brush_tensor, material_ids))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self._samples[index]


class DescriptorEncoder(nn.Module):
    """Text or metadata descriptor encoder returning conditioning embeddings."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=hidden_dim),
            num_layers=3,
        )

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.embedding(tokens)  # (seq, batch, embed)
        x = self.transformer(x)
        return x.mean(dim=0)


class BrushEncoder(nn.Module):
    """Encodes brush token sequences into latent vectors."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, brush_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        outputs, (hidden, _) = self.lstm(brush_tokens)
        last_hidden = hidden[-1]
        mu = self.to_mu(last_hidden)
        logvar = self.to_logvar(last_hidden)
        return mu, logvar


class BrushDecoder(nn.Module):
    """Decodes latent vectors into brush parameter sequences."""

    def __init__(self, latent_dim: int, output_dim: int) -> None:
        super().__init__()
        self.latent_to_hidden = nn.Linear(latent_dim, latent_dim)
        self.gru = nn.GRU(input_size=output_dim, hidden_size=latent_dim, num_layers=2, batch_first=True)
        self.to_brush = nn.Linear(latent_dim, output_dim)

    def forward(self, latent: Tensor, seq_len: int) -> Tensor:
        hidden = self.latent_to_hidden(latent).unsqueeze(0).repeat(2, 1, 1)
        inputs = torch.zeros(latent.size(0), seq_len, self.to_brush.out_features, device=latent.device)
        outputs, _ = self.gru(inputs, hidden)
        return self.to_brush(outputs)


class VMFGeneratorModel(nn.Module):
    """VAE-style generator that fuses descriptor conditioning with brush geometry."""

    def __init__(self, input_dim: int, latent_dim: int, descriptor_dim: int, material_vocab: int) -> None:
        super().__init__()
        self.encoder = BrushEncoder(input_dim=input_dim, hidden_dim=512, latent_dim=latent_dim)
        self.decoder = BrushDecoder(latent_dim=latent_dim + descriptor_dim, output_dim=input_dim)
        self.descriptor = DescriptorEncoder(vocab_size=material_vocab + 64, embed_dim=descriptor_dim, hidden_dim=512)
        self.material_head = nn.Linear(latent_dim + descriptor_dim, material_vocab)

    def forward(
        self,
        brush_tokens: Tensor,
        material_ids: Tensor,
        descriptor_tokens: Optional[Tensor] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encoder(brush_tokens)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        if descriptor_tokens is not None:
            desc = self.descriptor(descriptor_tokens)
            latent = torch.cat([z, desc], dim=-1)
        else:
            latent = z
        if seq_len is None:
            seq_len = brush_tokens.size(1)
        decoded_brushes = self.decoder(latent, seq_len=seq_len)
        material_logits = self.material_head(latent)
        return decoded_brushes, material_logits, mu, logvar


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """KL divergence term for the latent Gaussian."""

    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()


def reconstruction_loss(predicted: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """L2 reconstruction loss over brush parameters with masking."""

    diff = (predicted - target) ** 2
    masked = diff.sum(dim=-1) * mask
    denom = mask.sum().clamp(min=1.0)
    return masked.sum() / denom


def material_loss(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Cross-entropy over dominant material assignments per sample."""

    first_indices = mask.float().argmax(dim=1)
    gather_indices = target.gather(1, first_indices.unsqueeze(1)).squeeze(1)
    return nn.functional.cross_entropy(logits, gather_indices)


def train_step(
    model: VMFGeneratorModel,
    batch: Tuple[Tensor, Tensor, Tensor],
    optimizer: torch.optim.Optimizer,
    descriptor_tokens: Optional[Tensor] = None,
) -> Tuple[float, float, float]:
    """Single optimization step returning tuple of losses."""

    model.train()
    brush_tokens, material_ids, mask = batch
    optimizer.zero_grad()
    decoded, material_logits, mu, logvar = model(brush_tokens, material_ids, descriptor_tokens, seq_len=brush_tokens.size(1))
    rec_loss = reconstruction_loss(decoded, brush_tokens, mask)
    mat_loss = material_loss(material_logits, material_ids, mask)
    kl_loss = kl_divergence(mu, logvar)
    loss = rec_loss + mat_loss + 0.001 * kl_loss
    loss.backward()
    optimizer.step()
    return loss.item(), rec_loss.item(), mat_loss.item()
