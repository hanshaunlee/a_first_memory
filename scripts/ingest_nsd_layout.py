from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a_first_memory.data.nsd_layout import load_payload_from_layout
from a_first_memory.data.payload import save_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest NSD layout root into validated payload npz")
    parser.add_argument("--layout-root", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_payload_from_layout(args.layout_root)
    out = save_payload(args.output, payload)
    print(str(Path(out).resolve()))


if __name__ == "__main__":
    main()
