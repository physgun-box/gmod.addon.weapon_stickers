"""Command-line entry point for training a VMF language model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmf_ai.dataset import VMFDataset, VMFSample, load_vmf_paths
from vmf_ai.model import ModelConfig
from vmf_ai.tokenizer import VMFTokenizer
from vmf_ai.training import TrainerConfig, train_language_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer language model on VMF files")
    parser.add_argument("data", type=Path, help="Directory containing VMF files")
    parser.add_argument("output", type=Path, help="Directory where checkpoints will be written")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sequence-length", type=int, default=512, help="Maximum sequence length (without BOS/EOS)")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--max-sequence-length", type=int, default=2048, help="Maximum length the model can ingest")
    parser.add_argument("--device", default="auto", help="Training device (cuda/cpu/auto)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    vmf_paths = load_vmf_paths(args.data)
    texts = [path.read_text(encoding="utf-8") for path in vmf_paths]

    tokenizer = VMFTokenizer()
    tokenizer.fit(texts)

    samples = [VMFSample(path=path, text=text) for path, text in zip(vmf_paths, texts)]
    dataset = VMFDataset(samples, tokenizer, max_tokens=args.sequence_length + 1)

    trainer_config = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        output_dir=args.output,
        device=args.device,
    )

    model_config = ModelConfig(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.layers,
        dropout=args.dropout,
        feedforward_dim=args.ff_dim,
        max_sequence_length=args.max_sequence_length,
    )

    train_language_model(dataset, tokenizer, trainer_config, model_config)


if __name__ == "__main__":
    main()
