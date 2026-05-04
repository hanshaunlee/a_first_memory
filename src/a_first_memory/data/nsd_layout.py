from __future__ import annotations

from pathlib import Path

import numpy as np


REQUIRED_FILE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "unit_embeddings": (
        "unit_embeddings.npy",
        "features/unit_embeddings.npy",
        "embeddings/unit_embeddings.npy",
    ),
    "voxel_responses": (
        "voxel_responses.npy",
        "fmri/voxel_responses.npy",
        "betas/voxel_responses.npy",
    ),
    "lags": (
        "lags.npy",
        "behavior/lags.npy",
        "metadata/lags.npy",
    ),
    "hit_rates": (
        "hit_rates.npy",
        "behavior/hit_rates.npy",
        "metadata/hit_rates.npy",
    ),
    "family_index_by_unit": (
        "family_index_by_unit.npy",
        "features/family_index_by_unit.npy",
    ),
    "costs_by_unit": (
        "costs_by_unit.npy",
        "features/costs_by_unit.npy",
    ),
    "family_names": (
        "family_names.npy",
        "features/family_names.npy",
    ),
}

OPTIONAL_FILE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "novelty_index": ("novelty_index.npy", "behavior/novelty_index.npy"),
    "schema_congruence": ("schema_congruence.npy", "behavior/schema_congruence.npy"),
    "semantic_targets": ("semantic_targets.npy", "behavior/semantic_targets.npy"),
    "perceptual_targets": ("perceptual_targets.npy", "behavior/perceptual_targets.npy"),
    "n_rois": ("n_rois.npy", "metadata/n_rois.npy"),
    "roi_voxels": ("roi_voxels.npy", "metadata/roi_voxels.npy"),
    "subject_ids": ("subject_ids.npy", "metadata/subject_ids.npy"),
    "subject_names": ("subject_names.npy", "metadata/subject_names.npy"),
}


def _resolve_candidate(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for rel in candidates:
        p = root / rel
        if p.exists():
            return p
    return None


def infer_layout_files(root: str) -> dict[str, Path]:
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(f"NSD layout root not found: {base}")

    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for key, candidates in REQUIRED_FILE_CANDIDATES.items():
        p = _resolve_candidate(base, candidates)
        if p is None:
            missing.append(key)
        else:
            resolved[key] = p
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing required NSD layout files for keys: {joined}")

    for key, candidates in OPTIONAL_FILE_CANDIDATES.items():
        p = _resolve_candidate(base, candidates)
        if p is not None:
            resolved[key] = p
    return resolved


def _load_array(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        if "arr_0" in arr:
            return arr["arr_0"]
        first = list(arr.files)[0]
        return arr[first]
    return arr


def load_payload_from_layout(root: str, overrides: dict[str, str] | None = None) -> dict[str, np.ndarray]:
    files = infer_layout_files(root)
    if overrides:
        for key, path in overrides.items():
            if path:
                files[key] = Path(path)

    return {key: _load_array(path) for key, path in files.items()}
