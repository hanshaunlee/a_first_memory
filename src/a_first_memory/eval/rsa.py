from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from a_first_memory.config import RSAConfig
from a_first_memory.data.synthetic import SyntheticDataset
from a_first_memory.features.build import build_exposure_family_feature_tensor, build_exposure_voxel_matrix
from a_first_memory.utils.metrics import rdm, upper_triangle_values
from a_first_memory.utils.split import train_test_indices


@dataclass
class RSAResult:
    strategy: str
    exposure_correlations: list[float]
    roi_exposure_correlations: list[list[float]]
    roi_family_weights: list[list[list[float]]]


def _family_distance_matrix(family_tensor: np.ndarray, fam_idx: int) -> np.ndarray:
    fam_repr = family_tensor[:, fam_idx, :]
    return rdm(fam_repr)


def evaluate_fr_rsa(
    dataset: SyntheticDataset,
    cfg: RSAConfig,
    strategy: str,
    retention_by_image: np.ndarray | None = None,
    image_indices: np.ndarray | None = None,
) -> RSAResult:
    corrs: list[float] = []
    roi_exposure_corrs: list[list[float]] = []
    roi_family_weights: list[list[list[float]]] = []
    ids = dataset.image_ids if image_indices is None else image_indices
    n_images = len(ids)
    if n_images < 6:
        zero_roi = [[0.0 for _ in range(dataset.n_rois)] for _ in range(3)]
        zero_weights = [[[0.0 for _ in dataset.family_names] for _ in range(dataset.n_rois)] for _ in range(3)]
        return RSAResult(
            strategy=strategy,
            exposure_correlations=[0.0, 0.0, 0.0],
            roi_exposure_correlations=zero_roi,
            roi_family_weights=zero_weights,
        )
    pair_total = (n_images * (n_images - 1)) // 2
    pair_train, pair_test = train_test_indices(pair_total, cfg.train_frac, seed=23)
    for exposure_idx in range(3):
        family_tensor = build_exposure_family_feature_tensor(
            dataset=dataset,
            exposure_idx=exposure_idx,
            strategy=strategy,
            retention_by_image=retention_by_image,
            image_ids=ids,
        )
        family_rdms = [
            upper_triangle_values(_family_distance_matrix(family_tensor, fam_idx))
            for fam_idx in range(len(dataset.family_names))
        ]
        x = np.vstack(family_rdms).T

        brain = build_exposure_voxel_matrix(dataset, exposure_idx, image_ids=ids)
        roi_corrs: list[float] = []
        roi_weights: list[list[float]] = []
        for roi_idx in range(dataset.n_rois):
            s = roi_idx * dataset.roi_voxels
            e = s + dataset.roi_voxels
            brain_roi_rdm = rdm(brain[:, s:e])
            y = upper_triangle_values(brain_roi_rdm)
            model = Ridge(alpha=cfg.fr_rsa_alpha, fit_intercept=True)
            model.fit(x[pair_train], y[pair_train])
            pred = model.predict(x[pair_test])
            coef, _ = spearmanr(pred, y[pair_test])
            roi_corrs.append(float(np.nan_to_num(coef, nan=0.0)))
            coef_weights = np.nan_to_num(model.coef_, nan=0.0, posinf=0.0, neginf=0.0)
            roi_weights.append([float(w) for w in np.clip(coef_weights, 0.0, None)])
        roi_exposure_corrs.append(roi_corrs)
        roi_family_weights.append(roi_weights)
        corrs.append(float(np.mean(roi_corrs)))

    return RSAResult(
        strategy=strategy,
        exposure_correlations=corrs,
        roi_exposure_correlations=roi_exposure_corrs,
        roi_family_weights=roi_family_weights,
    )
