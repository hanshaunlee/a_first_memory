from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a_first_memory.data.payload import save_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NSD payload npz from component arrays")
    parser.add_argument("--unit-embeddings", required=True, type=str)
    parser.add_argument("--voxel-responses", required=True, type=str)
    parser.add_argument("--lags", required=True, type=str)
    parser.add_argument("--hit-rates", required=True, type=str)
    parser.add_argument("--family-index-by-unit", required=True, type=str)
    parser.add_argument("--costs-by-unit", required=True, type=str)
    parser.add_argument("--family-names", required=True, type=str)
    parser.add_argument("--novelty-index", type=str, default=None)
    parser.add_argument("--schema-congruence", type=str, default=None)
    parser.add_argument("--semantic-targets", type=str, default=None)
    parser.add_argument("--perceptual-targets", type=str, default=None)
    parser.add_argument("--subject-ids", type=str, default=None)
    parser.add_argument("--subject-names", type=str, default=None)
    parser.add_argument("--n-rois", type=int, default=None)
    parser.add_argument("--roi-voxels", type=int, default=None)
    parser.add_argument("--output", required=True, type=str)
    return parser.parse_args()


def _load(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        if "arr_0" in arr:
            return arr["arr_0"]
        raise ValueError(f"npz at {path} does not include arr_0")
    return arr


def main() -> None:
    args = parse_args()
    payload: dict[str, np.ndarray] = {
        "unit_embeddings": _load(args.unit_embeddings),
        "voxel_responses": _load(args.voxel_responses),
        "lags": _load(args.lags),
        "hit_rates": _load(args.hit_rates),
        "family_index_by_unit": _load(args.family_index_by_unit),
        "costs_by_unit": _load(args.costs_by_unit),
        "family_names": _load(args.family_names),
    }

    if args.novelty_index:
        payload["novelty_index"] = _load(args.novelty_index)
    if args.schema_congruence:
        payload["schema_congruence"] = _load(args.schema_congruence)
    if args.semantic_targets:
        payload["semantic_targets"] = _load(args.semantic_targets)
    if args.perceptual_targets:
        payload["perceptual_targets"] = _load(args.perceptual_targets)
    if args.subject_ids:
        payload["subject_ids"] = _load(args.subject_ids)
    if args.subject_names:
        payload["subject_names"] = _load(args.subject_names)
    if args.n_rois is not None:
        payload["n_rois"] = np.array(args.n_rois)
    if args.roi_voxels is not None:
        payload["roi_voxels"] = np.array(args.roi_voxels)

    out = save_payload(args.output, payload)
    print(str(Path(out).resolve()))


if __name__ == "__main__":
    main()
