"""Training loop helpers for the VMF Transformer language model."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import VMFDataset, collate_tokens
from .model import ModelConfig, VMFTransformerLM
from .tokenizer import VMFTokenizer


@dataclass
class TrainerConfig:
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    output_dir: Path = Path("checkpoints")
    device: str = "auto"


def _prepare_device(config: TrainerConfig) -> torch.device:
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


def train_language_model(
    dataset: VMFDataset,
    tokenizer: VMFTokenizer,
    trainer_config: TrainerConfig,
    model_config: ModelConfig,
) -> Path:
    """Train a language model and persist the resulting checkpoint."""
    device = _prepare_device(trainer_config)

    dataloader = DataLoader(
        dataset,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_tokens(batch, tokenizer.pad_id),
    )

    model = VMFTransformerLM(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=trainer_config.learning_rate, weight_decay=trainer_config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * trainer_config.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    output_dir = trainer_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, trainer_config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{trainer_config.epochs}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        print(f"Epoch {epoch}: loss={avg_loss:.4f} perplexity={perplexity:.2f}")

        checkpoint_path = output_dir / f"epoch_{epoch:03d}.pt"
        torch.save({"model_state": model.state_dict(), "model_config": asdict(model_config)}, checkpoint_path)

    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    trainer_payload = asdict(trainer_config)
    trainer_payload["output_dir"] = str(trainer_config.output_dir)
    config_payload = {"trainer": trainer_payload, "model": asdict(model_config)}

    config_path = output_dir / "training_config.json"
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    return checkpoint_path
