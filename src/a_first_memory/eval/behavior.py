from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from a_first_memory.data.synthetic import SyntheticDataset


@dataclass
class BehaviorFitResult:
    name: str
    spearman: float
    mse: float


def _idx_by_keyword(names: list[str], keyword: str, fallback: int) -> int:
    for i, name in enumerate(names):
        if keyword in name.lower():
            return i
    return int(np.clip(fallback, 0, len(names) - 1))


def predict_hits_from_retention(dataset: SyntheticDataset, retention_by_image_exposure: np.ndarray) -> np.ndarray:
    n_images = len(dataset.image_ids)
    n_families = len(dataset.family_names)
    semantic_idx = _idx_by_keyword(dataset.family_names, "semantic", 0)
    object_idx = _idx_by_keyword(dataset.family_names, "object", min(1, n_families - 1))
    geometry_idx = _idx_by_keyword(dataset.family_names, "geometry", min(2, n_families - 1))
    low_idx = _idx_by_keyword(dataset.family_names, "low", min(3, n_families - 1))
    patch_idx = _idx_by_keyword(dataset.family_names, "patch", min(4, n_families - 1))
    pred = np.zeros((n_images, 3), dtype=float)
    for image_id in dataset.image_ids:
        for exposure_idx in range(3):
            selected = retention_by_image_exposure[int(image_id), exposure_idx]
            fam_energy = np.zeros(n_families, dtype=float)
            units = dataset.unit_embeddings[int(image_id)] * selected[:, None]
            for fam_idx in range(n_families):
                fam_energy[fam_idx] = float(np.linalg.norm(units[dataset.family_index_by_unit == fam_idx]))
            fam_energy = fam_energy / (np.sum(fam_energy) + 1e-8)
            semantic_like = float(
                np.sum(
                    [
                        fam_energy[i]
                        for i, name in enumerate(dataset.family_names)
                        if ("semantic" in name.lower()) or ("scene_graph" in name.lower()) or ("ocr" in name.lower())
                    ]
                )
            )
            object_like = float(
                np.sum([fam_energy[i] for i, name in enumerate(dataset.family_names) if ("object" in name.lower()) or ("part" in name.lower())])
            )
            geometry_like = float(
                np.sum(
                    [fam_energy[i] for i, name in enumerate(dataset.family_names) if ("geometry" in name.lower()) or ("depth" in name.lower()) or ("normal" in name.lower())]
                )
            )
            low_like = float(
                np.sum(
                    [
                        fam_energy[i]
                        for i, name in enumerate(dataset.family_names)
                        if ("low" in name.lower()) or ("texture" in name.lower()) or ("color" in name.lower()) or ("edge" in name.lower())
                    ]
                )
            )
            logit = (
                0.55 * fam_energy[semantic_idx]
                + 0.35 * fam_energy[object_idx]
                + 0.25 * fam_energy[geometry_idx]
                + 0.20 * fam_energy[low_idx]
                + 0.35 * fam_energy[patch_idx]
                + 0.55 * semantic_like
                + 0.40 * object_like
                + 0.35 * geometry_like
                + 0.20 * low_like
                + 0.7 * dataset.novelty_index[int(image_id)]
                + 0.55 * dataset.schema_congruence[int(image_id)]
                - 0.08 * dataset.lags[int(image_id), exposure_idx]
            )
            pred[int(image_id), exposure_idx] = float(1.0 / (1.0 + np.exp(-2.0 * (logit - 0.9))))
    return pred


def evaluate_behavior_fit(dataset: SyntheticDataset, name: str, retention_by_image_exposure: np.ndarray) -> BehaviorFitResult:
    return evaluate_behavior_fit_subset(dataset, name, retention_by_image_exposure, image_indices=None)


def evaluate_behavior_fit_subset(
    dataset: SyntheticDataset,
    name: str,
    retention_by_image_exposure: np.ndarray,
    image_indices: np.ndarray | None,
) -> BehaviorFitResult:
    predicted = predict_hits_from_retention(dataset, retention_by_image_exposure)
    if image_indices is None:
        true_vals = dataset.hit_rates.reshape(-1)
        pred_vals = predicted.reshape(-1)
    else:
        true_vals = dataset.hit_rates[image_indices].reshape(-1)
        pred_vals = predicted[image_indices].reshape(-1)
    if len(pred_vals) < 4:
        return BehaviorFitResult(name=name, spearman=0.0, mse=0.0)
    coef, _ = spearmanr(pred_vals, true_vals)
    mse = float(np.mean((pred_vals - true_vals) ** 2))
    return BehaviorFitResult(name=name, spearman=float(np.nan_to_num(coef, nan=0.0)), mse=mse)
