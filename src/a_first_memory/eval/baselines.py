from __future__ import annotations

import numpy as np

from a_first_memory.data.synthetic import SyntheticDataset


def random_budget_policy(dataset: SyntheticDataset, budget: float, seed: int = 19) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_images = len(dataset.image_ids)
    n_units = dataset.unit_embeddings.shape[1]
    retention = np.zeros((n_images, 3, n_units), dtype=float)
    for image_id in dataset.image_ids:
        for exposure_idx in range(3):
            order = rng.permutation(n_units)
            budget_left = budget
            for unit_id in order:
                cost = dataset.costs_by_unit[unit_id]
                if budget_left - cost >= 0:
                    retention[int(image_id), exposure_idx, unit_id] = 1.0
                    budget_left -= cost
    return retention


def saliency_like_policy(dataset: SyntheticDataset, budget: float) -> np.ndarray:
    n_images = len(dataset.image_ids)
    n_units = dataset.unit_embeddings.shape[1]
    retention = np.zeros((n_images, 3, n_units), dtype=float)
    fam_pref = np.ones(len(dataset.family_names), dtype=float)
    for i, name in enumerate(dataset.family_names):
        lower = name.lower()
        if "object" in lower or "part" in lower:
            fam_pref[i] = 1.05
        elif "semantic" in lower or "scene_graph" in lower or "ocr" in lower or "face" in lower:
            fam_pref[i] = 0.95
        elif "geometry" in lower or "depth" in lower or "normal" in lower:
            fam_pref[i] = 1.0
        elif "low" in lower or "texture" in lower or "color" in lower or "edge" in lower:
            fam_pref[i] = 0.8
        elif "patch" in lower:
            fam_pref[i] = 1.1
        elif "saliency" in lower:
            fam_pref[i] = 1.15
    for image_id in dataset.image_ids:
        units = dataset.unit_embeddings[int(image_id)]
        for exposure_idx in range(3):
            novelty_boost = 1.0 + 0.1 * dataset.novelty_index[int(image_id)]
            schema_boost = 1.0 + 0.08 * dataset.schema_congruence[int(image_id)] * exposure_idx
            scores = (
                np.linalg.norm(units, axis=1)
                * fam_pref[dataset.family_index_by_unit]
                * novelty_boost
                * schema_boost
            )
            order = np.argsort(scores)[::-1]
            budget_left = budget
            for unit_id in order:
                cost = dataset.costs_by_unit[unit_id]
                if budget_left - cost >= 0:
                    retention[int(image_id), exposure_idx, unit_id] = 1.0
                    budget_left -= cost
    return retention


def pca_like_compression_policy(dataset: SyntheticDataset, budget: float) -> np.ndarray:
    n_images = len(dataset.image_ids)
    n_units = dataset.unit_embeddings.shape[1]
    retention = np.zeros((n_images, 3, n_units), dtype=float)
    # Mimics generic compression by preserving units with smallest cost first.
    order = np.argsort(dataset.costs_by_unit)
    for image_id in dataset.image_ids:
        for exposure_idx in range(3):
            budget_left = budget
            for unit_id in order:
                cost = dataset.costs_by_unit[unit_id]
                if budget_left - cost >= 0:
                    retention[int(image_id), exposure_idx, unit_id] = 1.0
                    budget_left -= cost
    return retention
