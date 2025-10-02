"""Training entry point for the VMF generative model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from models.vmf_generative_model import (
    VMFGeneratorModel,
    VMFDataset,
    VMFTokenizer,
    kl_divergence,
    material_loss,
    reconstruction_loss,
    train_step,
)

BATCH_TYPE = Tuple[Tensor, Tensor, Tensor]


def collate_batch(batch: Iterable[Tuple[Tensor, Tensor]]) -> BATCH_TYPE:
    brushes, materials = zip(*batch)
    batch_size = len(brushes)
    max_len = max(item.size(0) for item in brushes)
    feature_dim = brushes[0].size(1)

    padded_brushes = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    padded_materials = torch.full((batch_size, max_len), fill_value=-1, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.float32)

    for idx, (brush, mat) in enumerate(zip(brushes, materials)):
        length = brush.size(0)
        padded_brushes[idx, :length] = brush
        padded_materials[idx, :length] = mat
        mask[idx, :length] = 1.0

    return padded_brushes, padded_materials, mask


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a VMF generative model")
    parser.add_argument("--data", type=Path, default=Path("maps"), help="Directory with training VMF files")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-brushes", type=int, default=512, help="Maximum brushes per map sample")
    parser.add_argument("--latent-dim", type=int, default=256, help="Latent vector size")
    parser.add_argument("--descriptor-dim", type=int, default=128, help="Descriptor embedding size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Directory for saved models")
    return parser


def run_training(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = VMFTokenizer()
    vmf_paths = list(args.data.glob("*.vmf"))
    if not vmf_paths:
        raise SystemExit(f"No VMF files found in {args.data}")

    dataset = VMFDataset(vmf_paths, tokenizer, max_brushes=args.max_brushes)
    if len(dataset) == 0:
        raise SystemExit("Dataset is empty after parsing VMF files.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    input_dim = dataset[0][0].size(1)
    material_vocab = len(tokenizer.material_vocab)
    model = VMFGeneratorModel(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        descriptor_dim=args.descriptor_dim,
        material_vocab=material_vocab,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        total_loss = total_rec = total_mat = 0.0
        for batch in dataloader:
            brush_tokens, material_ids, mask = batch
            brush_tokens = brush_tokens.to(device)
            material_ids = material_ids.to(device)
            mask = mask.to(device)

            loss, rec_loss, mat_loss = train_step(
                model,
                (brush_tokens, material_ids, mask),
                optimizer,
            )
            total_loss += loss
            total_rec += rec_loss
            total_mat += mat_loss

        batches = len(dataloader)
        print(
            f"Epoch {epoch:03d}: loss={total_loss / batches:.4f} "
            f"recon={total_rec / batches:.4f} mat={total_mat / batches:.4f}"
        )

        checkpoint_path = args.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "materials": tokenizer.material_vocab,
                "config": vars(args),
            },
            checkpoint_path,
        )


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
