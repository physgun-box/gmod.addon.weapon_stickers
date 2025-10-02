"""Generate a VMF file using the trained brush layout generator."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmf_ai.generation import generate_vmf_text
from vmf_tools import preview_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a VMF brush layout from a trained generator")
    parser.add_argument("checkpoint", type=Path, help="Path to a trained model checkpoint (epoch_XXX.pt)")
    parser.add_argument("output", type=Path, help="Destination path for the generated VMF file")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--presence-threshold", type=float, default=0.5, help="Minimum brush presence probability")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    vmf_text = generate_vmf_text(args.checkpoint, device=args.device, presence_threshold=args.presence_threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(vmf_text, encoding="utf-8")
    print(f"Generated VMF saved to {args.output}")

    try:
        preview_file(str(args.output))
    except Exception as exc:  # pragma: no cover - viewer errors depend on environment
        print(f"Failed to open viewer automatically: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
