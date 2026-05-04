from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from a_first_memory.data.synthetic import SyntheticDataset


@dataclass
class FamilyShiftResult:
    family_names: list[str]
    exposure_retention_fraction_by_family: list[list[float]]
    policy_delta_exposure3_minus1: list[float]


@dataclass
class FamilyShiftAlignment:
    per_roi_spearman: list[float]
    mean_spearman: float


def summarize_retention_shift(
    dataset: SyntheticDataset,
    retention_by_image_exposure: np.ndarray,
) -> FamilyShiftResult:
    by_exposure: list[list[float]] = []
    for exposure_idx in range(3):
        fractions: list[float] = []
        for fam_idx, _ in enumerate(dataset.family_names):
            mask = dataset.family_index_by_unit == fam_idx
            total_possible = float(mask.sum() * len(dataset.image_ids))
            retained = float(np.sum(retention_by_image_exposure[:, exposure_idx, mask]))
            fractions.append(retained / max(total_possible, 1.0))
        by_exposure.append(fractions)

    policy_delta = [float(by_exposure[2][i] - by_exposure[0][i]) for i in range(len(dataset.family_names))]
    return FamilyShiftResult(
        family_names=dataset.family_names,
        exposure_retention_fraction_by_family=by_exposure,
        policy_delta_exposure3_minus1=policy_delta,
    )


def compare_policy_brain_shift(
    policy_delta_by_family: list[float],
    fr_rsa_roi_family_weights: list[list[list[float]]],
) -> FamilyShiftAlignment:
    # fr_rsa_roi_family_weights: [exposure][roi][family]
    roi_count = len(fr_rsa_roi_family_weights[0])
    correlations = []
    policy = np.array(policy_delta_by_family, dtype=float)
    for roi_idx in range(roi_count):
        w1 = np.array(fr_rsa_roi_family_weights[0][roi_idx], dtype=float)
        w3 = np.array(fr_rsa_roi_family_weights[2][roi_idx], dtype=float)
        brain_delta = w3 - w1
        if np.allclose(policy, policy[0]) or np.allclose(brain_delta, brain_delta[0]):
            coef = 0.0
        else:
            coef, _ = spearmanr(policy, brain_delta)
        correlations.append(float(np.nan_to_num(coef, nan=0.0)))
    return FamilyShiftAlignment(
        per_roi_spearman=correlations,
        mean_spearman=float(np.mean(correlations)),
    )
