from __future__ import annotations

from pathlib import Path

import numpy as np

from a_first_memory.config import NSDConfig
from a_first_memory.data.nsd_layout import load_payload_from_layout
from a_first_memory.data.payload import load_payload_dir, load_payload_npz
from a_first_memory.data.synthetic import SyntheticDataset


def _load_feature_overrides(path: str) -> dict[str, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Feature override npz not found: {p}")
    npz = np.load(p, allow_pickle=True)
    required = ("unit_embeddings", "family_index_by_unit", "costs_by_unit", "family_names")
    missing = [k for k in required if k not in npz]
    if missing:
        joined = ", ".join(missing)
        raise KeyError(f"Feature override npz is missing keys: {joined}")
    return {k: npz[k] for k in npz.files}


def _payload_to_dataset(payload: dict[str, np.ndarray]) -> SyntheticDataset:
    unit_embeddings = payload["unit_embeddings"]
    voxel_responses = payload["voxel_responses"]
    lags = payload["lags"]
    hit_rates = payload["hit_rates"]
    family_index_by_unit = payload["family_index_by_unit"]
    costs_by_unit = payload["costs_by_unit"]
    family_names_raw = payload["family_names"]
    family_names = [str(x) for x in family_names_raw.tolist()]

    n_images = int(unit_embeddings.shape[0])
    n_voxels = int(voxel_responses.shape[2])
    n_rois = int(payload["n_rois"]) if "n_rois" in payload else 5
    roi_voxels = int(payload["roi_voxels"]) if "roi_voxels" in payload else max(1, n_voxels // max(n_rois, 1))

    novelty_index = payload["novelty_index"] if "novelty_index" in payload else np.ones(n_images, dtype=float) * 0.5
    schema_congruence = payload["schema_congruence"] if "schema_congruence" in payload else np.ones(n_images, dtype=float) * 0.5
    semantic_targets = payload["semantic_targets"] if "semantic_targets" in payload else np.clip(hit_rates + 0.05, 0.0, 1.0)
    perceptual_targets = payload["perceptual_targets"] if "perceptual_targets" in payload else np.clip(hit_rates - 0.05, 0.0, 1.0)
    subject_ids = payload["subject_ids"] if "subject_ids" in payload else np.zeros(n_images, dtype=int)
    subject_names = payload["subject_names"] if "subject_names" in payload else np.array(["subj01"], dtype=object)

    return SyntheticDataset(
        image_ids=np.arange(n_images),
        family_names=family_names,
        n_rois=n_rois,
        roi_voxels=roi_voxels,
        family_index_by_unit=family_index_by_unit.astype(int),
        costs_by_unit=costs_by_unit.astype(float),
        unit_embeddings=unit_embeddings.astype(float),
        voxel_responses=voxel_responses.astype(float),
        lags=lags.astype(int),
        novelty_index=novelty_index.astype(float),
        schema_congruence=schema_congruence.astype(float),
        hit_rates=hit_rates.astype(float),
        semantic_targets=semantic_targets.astype(float),
        perceptual_targets=perceptual_targets.astype(float),
        subject_ids=subject_ids.astype(int),
        subject_names=[str(x) for x in subject_names.tolist()],
    )


def _validate_strict_payload_content(payload: dict[str, np.ndarray]) -> None:
    n_images = int(payload["unit_embeddings"].shape[0])
    required_shapes = {
        "voxel_responses": (n_images, 3),
        "lags": (n_images, 3),
        "hit_rates": (n_images, 3),
        "novelty_index": (n_images,),
        "schema_congruence": (n_images,),
        "semantic_targets": (n_images, 3),
        "perceptual_targets": (n_images, 3),
        "subject_ids": (n_images,),
    }
    for key, shape_prefix in required_shapes.items():
        arr = payload[key]
        if arr.shape[: len(shape_prefix)] != shape_prefix:
            raise ValueError(f"NSD strict mode shape check failed for '{key}': got {arr.shape}, expected prefix {shape_prefix}")

    # Fail fast on obvious placeholder-like content.
    variability_checks = (
        "novelty_index",
        "schema_congruence",
        "semantic_targets",
        "perceptual_targets",
        "hit_rates",
    )
    for key in variability_checks:
        arr = np.asarray(payload[key], dtype=float)
        if float(np.nanstd(arr)) < 1e-8:
            raise ValueError(f"NSD strict mode detected near-constant '{key}', likely placeholder content.")

    subject_ids = np.asarray(payload["subject_ids"], dtype=int)
    if subject_ids.shape[0] != n_images:
        raise ValueError(f"'subject_ids' must have n_images={n_images} entries; got {subject_ids.shape[0]}.")
    subject_names = [str(x) for x in np.asarray(payload["subject_names"]).tolist()]
    if len(subject_names) == 0:
        raise ValueError("NSD strict mode requires non-empty 'subject_names'.")
    max_subject_id = int(np.max(subject_ids)) if subject_ids.size else 0
    if max_subject_id >= len(subject_names):
        raise ValueError(
            f"'subject_names' length ({len(subject_names)}) is smaller than max subject id + 1 ({max_subject_id + 1})."
        )


def load_nsd_dataset(cfg: NSDConfig) -> SyntheticDataset:
    """Load NSD-derived data from either npz bundle or raw payload directory."""
    payload: dict[str, np.ndarray]
    if cfg.layout_root:
        payload = load_payload_from_layout(cfg.layout_root)
    elif cfg.npz_path:
        npz = Path(cfg.npz_path)
        if not npz.exists():
            raise FileNotFoundError(f"NSD npz path does not exist: {npz}")
        payload = load_payload_npz(str(npz))
    elif cfg.raw_dir:
        payload = load_payload_dir(cfg.raw_dir)
    else:
        raise ValueError("NSD mode requires config.nsd.layout_root, config.nsd.npz_path, or config.nsd.raw_dir")

    if cfg.feature_npz_path:
        overrides = _load_feature_overrides(cfg.feature_npz_path)
        payload["unit_embeddings"] = overrides["unit_embeddings"]
        payload["family_index_by_unit"] = overrides["family_index_by_unit"]
        payload["costs_by_unit"] = overrides["costs_by_unit"]
        payload["family_names"] = overrides["family_names"]

    if cfg.strict:
        required_optional = (
            "novelty_index",
            "schema_congruence",
            "semantic_targets",
            "perceptual_targets",
            "subject_ids",
            "subject_names",
            "n_rois",
            "roi_voxels",
        )
        missing = [k for k in required_optional if k not in payload]
        if missing:
            joined = ", ".join(missing)
            raise KeyError(f"NSD strict mode requires keys missing from payload: {joined}")
        _validate_strict_payload_content(payload)

    return _payload_to_dataset(payload)
