"""Training helpers for the VMF brush layout generator."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import VMFLayoutDataset
from .model import GeneratorConfig, VMFBrushGenerator
from .tokenizer import MaterialVocabulary


@dataclass
class TrainerConfig:
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "auto"
    kl_weight: float = 1e-4
    presence_weight: float = 1.0
    feature_weight: float = 1.0
    material_weight: float = 1.0


def _prepare_device(config: TrainerConfig) -> torch.device:
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def train_generator(
    dataset: VMFLayoutDataset,
    vocabulary: MaterialVocabulary,
    trainer_config: TrainerConfig,
    model_config: GeneratorConfig,
    *,
    output_dir: Path,
) -> Path:
    """Train :class:`VMFBrushGenerator` and write checkpoints to ``output_dir``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    device = _prepare_device(trainer_config)

    dataloader = DataLoader(dataset, batch_size=trainer_config.batch_size, shuffle=True)
    model = VMFBrushGenerator(model_config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=trainer_config.learning_rate, weight_decay=trainer_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * trainer_config.epochs)

    bce_loss = nn.BCEWithLogitsLoss(reduction="none")
    mse_loss = nn.MSELoss(reduction="none")
    ce_loss = nn.CrossEntropyLoss(ignore_index=vocabulary.pad_id, reduction="none")

    for epoch in range(1, trainer_config.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{trainer_config.epochs}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            features = batch["features"].to(device)
            materials = batch["materials"].to(device)
            mask = batch["mask"].to(device)

            outputs = model(features, materials, mask)

            feature_pred = outputs["feature_pred"]
            feature_loss = mse_loss(feature_pred, features).sum(dim=-1)
            feature_loss = (feature_loss * mask.float()).sum() / torch.clamp(mask.sum(), min=1)

            material_logits = outputs["material_logits"].transpose(1, 2)
            material_loss = ce_loss(material_logits, materials)
            material_loss = (material_loss * mask.float()).sum() / torch.clamp(mask.sum(), min=1)

            presence_loss = bce_loss(outputs["presence_logits"], mask.float())
            presence_loss = presence_loss.sum() / mask.numel()

            kl_loss = _kl_divergence(outputs["mu"], outputs["logvar"]).mean()

            loss = (
                trainer_config.feature_weight * feature_loss
                + trainer_config.material_weight * material_loss
                + trainer_config.presence_weight * presence_loss
                + trainer_config.kl_weight * kl_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")

        checkpoint_path = output_dir / f"epoch_{epoch:03d}.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": asdict(model_config),
                "feature_mean": dataset.feature_mean.tolist(),
                "feature_std": dataset.feature_std.tolist(),
                "vocabulary": vocabulary.to_json(),
            },
            checkpoint_path,
        )

    config_payload = {
        "trainer": asdict(trainer_config),
        "model": asdict(model_config),
        "feature_mean": dataset.feature_mean.tolist(),
        "feature_std": dataset.feature_std.tolist(),
        "vocabulary": vocabulary.to_json(),
    }
    (output_dir / "training_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    vocab_path = output_dir / "materials.json"
    vocabulary.save(vocab_path)

    stats = {
        "feature_mean": dataset.feature_mean.tolist(),
        "feature_std": dataset.feature_std.tolist(),
    }
    (output_dir / "feature_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return checkpoint_path
