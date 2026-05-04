from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from a_first_memory.config import SyntheticDataConfig
from a_first_memory.models.schema import FeatureFamily


@dataclass
class SyntheticDataset:
    image_ids: np.ndarray
    family_names: list[str]
    n_rois: int
    roi_voxels: int
    family_index_by_unit: np.ndarray
    costs_by_unit: np.ndarray
    unit_embeddings: np.ndarray
    # [n_images, 3, n_voxels]
    voxel_responses: np.ndarray
    # [n_images, 3]
    lags: np.ndarray
    # [n_images]
    novelty_index: np.ndarray
    # [n_images]
    schema_congruence: np.ndarray
    # [n_images, 3]
    hit_rates: np.ndarray
    # [n_images, 3]
    semantic_targets: np.ndarray
    # [n_images, 3]
    perceptual_targets: np.ndarray
    # [n_images]
    subject_ids: np.ndarray
    # [n_subjects]
    subject_names: list[str]


def _family_costs() -> dict[FeatureFamily, float]:
    return {
        FeatureFamily.SEMANTIC: 1.25,
        FeatureFamily.VLM_SEMANTIC: 1.30,
        FeatureFamily.OBJECT: 1.15,
        FeatureFamily.OBJECT_PART: 1.10,
        FeatureFamily.SCENE_GRAPH: 1.25,
        FeatureFamily.OCR_TEXT: 1.05,
        FeatureFamily.FACE_BODY: 1.10,
        FeatureFamily.GEOMETRY: 0.95,
        FeatureFamily.DEPTH: 1.00,
        FeatureFamily.SURFACE_NORMAL: 0.95,
        FeatureFamily.LOW_LEVEL: 0.70,
        FeatureFamily.TEXTURE: 0.75,
        FeatureFamily.COLOR: 0.70,
        FeatureFamily.SPATIAL_FREQUENCY: 0.80,
        FeatureFamily.EDGE_SHAPE: 0.80,
        FeatureFamily.PATCH: 0.85,
        FeatureFamily.SALIENCY: 0.75,
    }


def _family_mask(names: list[str], keywords: tuple[str, ...]) -> np.ndarray:
    mask = np.array([any(k in n.lower() for k in keywords) for n in names], dtype=float)
    if float(np.sum(mask)) == 0.0:
        return np.zeros(len(names), dtype=float)
    return mask


def generate_synthetic_dataset(cfg: SyntheticDataConfig) -> SyntheticDataset:
    rng = np.random.default_rng(cfg.random_seed)
    families = list(FeatureFamily)
    n_families = len(families)
    units_total = cfg.n_units_per_family * n_families
    n_voxels = cfg.n_rois * cfg.roi_voxels

    unit_embeddings = rng.normal(
        0.0,
        1.0,
        size=(cfg.n_images, units_total, cfg.embedding_dim),
    )

    family_index_by_unit = np.repeat(np.arange(n_families), cfg.n_units_per_family)
    costs_lookup = _family_costs()
    costs_by_unit = np.array(
        [costs_lookup[families[f_idx]] for f_idx in family_index_by_unit],
        dtype=float,
    )

    family_names = [f.value for f in families]
    semantic_like = _family_mask(family_names, ("semantic", "scene_graph", "ocr", "face"))
    object_like = _family_mask(family_names, ("object", "part"))
    geometry_like = _family_mask(family_names, ("geometry", "depth", "normal"))
    low_like = _family_mask(family_names, ("low", "texture", "color", "spatial", "edge", "saliency"))
    patch_like = _family_mask(family_names, ("patch",))

    if float(np.sum(semantic_like)) == 0.0:
        semantic_like[0] = 1.0
    if float(np.sum(object_like)) == 0.0:
        object_like[min(1, n_families - 1)] = 1.0
    if float(np.sum(geometry_like)) == 0.0:
        geometry_like[min(2, n_families - 1)] = 1.0
    if float(np.sum(low_like)) == 0.0:
        low_like[min(3, n_families - 1)] = 1.0
    if float(np.sum(patch_like)) == 0.0:
        patch_like[min(4, n_families - 1)] = 1.0

    semantic_like /= np.sum(semantic_like)
    object_like /= np.sum(object_like)
    geometry_like /= np.sum(geometry_like)
    low_like /= np.sum(low_like)
    patch_like /= np.sum(patch_like)

    roi_templates = np.array(
        [
            0.15 * semantic_like + 0.10 * object_like + 0.25 * geometry_like + 1.00 * low_like + 0.35 * patch_like,
            0.35 * semantic_like + 0.45 * object_like + 0.70 * geometry_like + 0.60 * low_like + 0.55 * patch_like,
            1.00 * semantic_like + 0.90 * object_like + 0.35 * geometry_like + 0.20 * low_like + 0.55 * patch_like,
            1.10 * semantic_like + 0.75 * object_like + 0.30 * geometry_like + 0.15 * low_like + 0.30 * patch_like,
            0.75 * semantic_like + 0.80 * object_like + 0.55 * geometry_like + 0.35 * low_like + 0.45 * patch_like,
        ],
        dtype=float,
    )
    if cfg.n_rois <= roi_templates.shape[0]:
        roi_family_preferences = roi_templates[: cfg.n_rois]
    else:
        extra = rng.uniform(0.2, 1.0, size=(cfg.n_rois - roi_templates.shape[0], n_families))
        roi_family_preferences = np.vstack([roi_templates, extra])

    voxel_weights = rng.normal(
        0.0,
        1.0,
        size=(cfg.n_rois, cfg.roi_voxels, cfg.embedding_dim),
    )

    sem_curve = np.array([1.00, 1.12, 1.24], dtype=float)
    obj_curve = np.array([1.00, 1.08, 1.16], dtype=float)
    geom_curve = np.array([1.00, 1.03, 1.06], dtype=float)
    low_curve = np.array([1.00, 0.93, 0.86], dtype=float)
    patch_curve = np.array([1.00, 0.99, 0.97], dtype=float)
    exposure_multipliers = np.zeros((3, n_families), dtype=float)
    for exposure_idx in range(3):
        exposure_multipliers[exposure_idx] = (
            sem_curve[exposure_idx] * semantic_like
            + obj_curve[exposure_idx] * object_like
            + geom_curve[exposure_idx] * geometry_like
            + low_curve[exposure_idx] * low_like
            + patch_curve[exposure_idx] * patch_like
        )
    exposure_multipliers /= np.maximum(np.mean(exposure_multipliers, axis=1, keepdims=True), 1e-8)

    voxel_responses = np.zeros((cfg.n_images, 3, n_voxels), dtype=float)
    lags = rng.integers(low=0, high=4, size=(cfg.n_images, 3), endpoint=False)
    novelty_index = rng.uniform(0.1, 1.0, size=cfg.n_images)
    schema_congruence = rng.uniform(0.0, 1.0, size=cfg.n_images)
    hit_rates = np.zeros((cfg.n_images, 3), dtype=float)
    semantic_targets = np.zeros((cfg.n_images, 3), dtype=float)
    perceptual_targets = np.zeros((cfg.n_images, 3), dtype=float)
    n_subjects = 8
    subject_ids = np.arange(cfg.n_images) % n_subjects

    for img_idx in range(cfg.n_images):
        units = unit_embeddings[img_idx]
        family_sums = np.zeros((n_families, cfg.embedding_dim), dtype=float)
        for fam_idx in range(n_families):
            fam_units = units[family_index_by_unit == fam_idx]
            family_sums[fam_idx] = np.mean(fam_units, axis=0)

        for exposure_idx in range(3):
            lag = lags[img_idx, exposure_idx]
            lag_decay = 1.0 / (1.0 + 0.2 * lag)
            cursor = 0
            exposure_scale = exposure_multipliers[exposure_idx]
            for roi_idx in range(cfg.n_rois):
                pref = roi_family_preferences[roi_idx] * exposure_scale * lag_decay
                roi_repr = np.sum(
                    family_sums * pref[:, None],
                    axis=0,
                )
                for vox_idx in range(cfg.roi_voxels):
                    signal = float(roi_repr @ voxel_weights[roi_idx, vox_idx])
                    noise = float(rng.normal(0.0, 0.5))
                    voxel_responses[img_idx, exposure_idx, cursor + vox_idx] = signal + noise
                cursor += cfg.roi_voxels

            # Synthetic behavioral recognition performance for process-model testing.
            family_norms = np.array([np.linalg.norm(family_sums[fam_idx]) for fam_idx in range(n_families)], dtype=float)
            semantic_strength = float(np.sum(family_norms * (0.65 * semantic_like + 0.35 * object_like)))
            perceptual_strength = float(np.sum(family_norms * (0.60 * geometry_like + 0.40 * low_like)))
            novelty_term = novelty_index[img_idx] * (1.0 + 0.08 * lag)
            schema_term = schema_congruence[img_idx] * (1.0 + 0.15 * exposure_idx)
            logits = (
                0.55 * semantic_strength
                + 0.40 * perceptual_strength
                + 1.20 * novelty_term
                + 1.05 * schema_term
                - 0.35 * lag
            )
            probs = 1.0 / (1.0 + np.exp(-0.08 * logits))
            hit_rates[img_idx, exposure_idx] = float(np.clip(probs + rng.normal(0.0, 0.04), 0.02, 0.98))
            semantic_targets[img_idx, exposure_idx] = float(
                np.clip(
                    0.55 * probs
                    + 0.35 * schema_term
                    + 0.10 * novelty_term
                    + rng.normal(0.0, 0.03),
                    0.0,
                    1.0,
                )
            )
            perceptual_targets[img_idx, exposure_idx] = float(
                np.clip(
                    0.55 * probs
                    + 0.35 * perceptual_strength / (semantic_strength + perceptual_strength + 1e-8)
                    + rng.normal(0.0, 0.03),
                    0.0,
                    1.0,
                )
            )

    return SyntheticDataset(
        image_ids=np.arange(cfg.n_images),
        family_names=family_names,
        n_rois=cfg.n_rois,
        roi_voxels=cfg.roi_voxels,
        family_index_by_unit=family_index_by_unit,
        costs_by_unit=costs_by_unit,
        unit_embeddings=unit_embeddings,
        voxel_responses=voxel_responses,
        lags=lags,
        novelty_index=novelty_index,
        schema_congruence=schema_congruence,
        hit_rates=hit_rates,
        semantic_targets=semantic_targets,
        perceptual_targets=perceptual_targets,
        subject_ids=subject_ids.astype(int),
        subject_names=[f"subj{idx + 1:02d}" for idx in range(n_subjects)],
    )
