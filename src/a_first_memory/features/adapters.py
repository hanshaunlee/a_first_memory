from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FamilyFeatureBlock:
    name: str
    embeddings: np.ndarray  # [n_images, n_units, emb_dim]
    unit_cost: float


def load_feature_block(path: str, family_name: str, unit_cost: float) -> FamilyFeatureBlock:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Feature block path not found: {p}")
    arr = np.load(p, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        emb = arr["arr_0"] if "arr_0" in arr else arr[list(arr.files)[0]]
    else:
        emb = arr
    if emb.ndim != 3:
        raise ValueError(f"Feature block for {family_name} must be rank-3 [n_images, n_units, emb_dim]")
    return FamilyFeatureBlock(name=family_name, embeddings=emb.astype(float), unit_cost=float(unit_cost))


def build_feature_bank(blocks: list[FamilyFeatureBlock]) -> dict[str, np.ndarray]:
    if not blocks:
        raise ValueError("build_feature_bank requires at least one block.")
    n_images = blocks[0].embeddings.shape[0]
    emb_dim = blocks[0].embeddings.shape[2]
    for block in blocks:
        if block.embeddings.shape[0] != n_images:
            raise ValueError("All blocks must share n_images")
        if block.embeddings.shape[2] != emb_dim:
            raise ValueError("All blocks must share embedding dim")

    family_names = [b.name for b in blocks]
    concat = np.concatenate([b.embeddings for b in blocks], axis=1)
    family_index_by_unit = []
    costs_by_unit = []
    for fam_idx, block in enumerate(blocks):
        n_units = block.embeddings.shape[1]
        family_index_by_unit.extend([fam_idx] * n_units)
        costs_by_unit.extend([block.unit_cost] * n_units)

    return {
        "unit_embeddings": concat,
        "family_index_by_unit": np.array(family_index_by_unit, dtype=int),
        "costs_by_unit": np.array(costs_by_unit, dtype=float),
        "family_names": np.array(family_names, dtype=object),
    }
