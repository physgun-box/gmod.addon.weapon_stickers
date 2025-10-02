"""Generate a VMF file using a trained language model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmf_ai.generation import generate_vmf_text, load_model
from vmf_ai.tokenizer import VMFTokenizer
from vmf_tools import preview_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VMF content with a trained model")
    parser.add_argument("checkpoint", type=Path, help="Path to a trained model checkpoint (epoch_XXX.pt)")
    parser.add_argument("tokenizer", type=Path, help="Path to the tokenizer.json saved during training")
    parser.add_argument("output", type=Path, help="Destination path for the generated VMF file")
    parser.add_argument("--prompt", type=str, default="", help="Optional priming text for generation")
    parser.add_argument("--prompt-file", type=Path, help="Load the prompt from a text file")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    else:
        prompt = args.prompt

    tokenizer = VMFTokenizer.load(args.tokenizer)
    model = load_model(args.checkpoint, tokenizer)

    generated_text = generate_vmf_text(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated_text, encoding="utf-8")
    print(f"Generated VMF saved to {args.output}")

    try:
        preview_file(str(args.output))
    except Exception as exc:  # pragma: no cover - viewer errors depend on environment
        print(f"Failed to open viewer automatically: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
