from __future__ import annotations

from pathlib import Path

import numpy as np


REQUIRED_KEYS = (
    "unit_embeddings",
    "voxel_responses",
    "lags",
    "hit_rates",
    "family_index_by_unit",
    "costs_by_unit",
    "family_names",
)

OPTIONAL_KEYS = (
    "novelty_index",
    "schema_congruence",
    "semantic_targets",
    "perceptual_targets",
    "n_rois",
    "roi_voxels",
    "subject_ids",
    "subject_names",
)


def validate_payload_dict(payload: dict[str, np.ndarray]) -> None:
    for key in REQUIRED_KEYS:
        if key not in payload:
            raise KeyError(f"Missing required key: {key}")

    unit_embeddings = payload["unit_embeddings"]
    voxel_responses = payload["voxel_responses"]
    lags = payload["lags"]
    hit_rates = payload["hit_rates"]
    family_index = payload["family_index_by_unit"]
    costs = payload["costs_by_unit"]

    if unit_embeddings.ndim != 3:
        raise ValueError("unit_embeddings must be rank-3 [n_images, n_units, embedding_dim]")
    if voxel_responses.ndim != 3:
        raise ValueError("voxel_responses must be rank-3 [n_images, 3, n_voxels]")
    if lags.shape != hit_rates.shape:
        raise ValueError("lags and hit_rates must share shape [n_images, 3]")
    if lags.shape[1] != 3:
        raise ValueError("lags/hit_rates second dimension must be 3 exposures")
    if family_index.shape[0] != unit_embeddings.shape[1]:
        raise ValueError("family_index_by_unit length must match n_units")
    if costs.shape[0] != unit_embeddings.shape[1]:
        raise ValueError("costs_by_unit length must match n_units")
    if unit_embeddings.shape[0] != voxel_responses.shape[0]:
        raise ValueError("unit_embeddings and voxel_responses must share n_images")
    if unit_embeddings.shape[0] != lags.shape[0]:
        raise ValueError("unit_embeddings and lags must share n_images")


def save_payload(path: str, payload: dict[str, np.ndarray]) -> Path:
    validate_payload_dict(payload)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    return out


def load_payload_npz(path: str) -> dict[str, np.ndarray]:
    npz = np.load(path, allow_pickle=True)
    data = {key: npz[key] for key in npz.files}
    validate_payload_dict(data)
    return data


def load_payload_dir(path: str) -> dict[str, np.ndarray]:
    base = Path(path)
    if not base.exists():
        raise FileNotFoundError(f"Payload directory not found: {base}")

    data: dict[str, np.ndarray] = {}
    for key in REQUIRED_KEYS + OPTIONAL_KEYS:
        npy_path = base / f"{key}.npy"
        npz_path = base / f"{key}.npz"
        if npy_path.exists():
            data[key] = np.load(npy_path, allow_pickle=True)
        elif npz_path.exists():
            bundle = np.load(npz_path, allow_pickle=True)
            data[key] = bundle["arr_0"] if "arr_0" in bundle else bundle[list(bundle.files)[0]]
    validate_payload_dict(data)
    return data
