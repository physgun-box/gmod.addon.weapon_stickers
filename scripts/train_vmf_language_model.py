"""Train the VMF brush layout generator."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmf_ai.dataset import build_layout_dataset
from vmf_ai.model import GeneratorConfig
from vmf_ai.training import TrainerConfig, train_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VMF brush layout generator on existing maps")
    parser.add_argument("data", type=Path, help="Directory containing VMF files")
    parser.add_argument("output", type=Path, help="Directory where checkpoints will be written")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--decoder-hidden", type=int, default=256)
    parser.add_argument("--material-embed", type=int, default=16)
    parser.add_argument("--material-limit", type=int, help="Limit the number of distinct materials tracked")
    parser.add_argument("--max-brushes", type=int, help="Override the maximum number of brushes per map")
    parser.add_argument("--device", default="auto", help="Training device (cuda/cpu/auto)")
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--presence-weight", type=float, default=1.0)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument("--material-weight", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset, vocabulary = build_layout_dataset(
        args.data,
        max_brushes=args.max_brushes,
        material_limit=args.material_limit,
    )

    trainer_config = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        kl_weight=args.kl_weight,
        presence_weight=args.presence_weight,
        feature_weight=args.feature_weight,
        material_weight=args.material_weight,
    )

    model_config = GeneratorConfig(
        max_brushes=dataset.max_brushes,
        feature_dim=dataset.feature_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        decoder_hidden_dim=args.decoder_hidden,
        material_vocab_size=len(vocabulary),
        material_embedding_dim=args.material_embed,
    )

    train_generator(dataset, vocabulary, trainer_config, model_config, output_dir=args.output)


if __name__ == "__main__":
    main()
