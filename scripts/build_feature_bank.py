from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a_first_memory.features.adapters import build_feature_bank, load_feature_block


def parse_block_arg(spec: str) -> tuple[str, str, float]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid --block spec '{spec}'. Expected name:path:cost")
    name, path, cost = parts
    return name, path, float(cost)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified feature bank from family blocks")
    parser.add_argument(
        "--block",
        action="append",
        required=True,
        help="Family block spec: family_name:/path/to/embeddings.npy:unit_cost",
    )
    parser.add_argument("--output", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blocks = []
    for spec in args.block:
        name, path, cost = parse_block_arg(spec)
        blocks.append(load_feature_block(path=path, family_name=name, unit_cost=cost))
    bank = build_feature_bank(blocks)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **bank)
    print(str(out.resolve()))


if __name__ == "__main__":
    main()
