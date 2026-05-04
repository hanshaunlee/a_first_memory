from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from a_first_memory.data.synthetic import SyntheticDataset


@dataclass
class FeatureQualityReport:
    n_images: int
    coverage_all_families_rate: float
    energy_min: float
    energy_mean: float
    energy_max: float
    variance_min: float
    variance_mean: float
    variance_max: float
    low_energy_image_ids: list[int]
    quality_pass: bool


def evaluate_feature_quality(
    dataset: SyntheticDataset,
    energy_z_floor: float = -2.0,
    min_coverage_rate: float = 0.99,
) -> FeatureQualityReport:
    x = dataset.unit_embeddings
    n_images = int(x.shape[0])
    flat = x.reshape(n_images, -1)
    energy = np.linalg.norm(flat, axis=1)
    variance = np.var(flat, axis=1)
    z = (energy - np.mean(energy)) / (np.std(energy) + 1e-8)

    unit_norms = np.linalg.norm(x, axis=2)
    coverage = []
    for image_id in dataset.image_ids:
        ok = True
        for fam_idx in range(len(dataset.family_names)):
            fam_units = np.where(dataset.family_index_by_unit == fam_idx)[0]
            if np.max(unit_norms[int(image_id), fam_units]) <= 0.1:
                ok = False
                break
        coverage.append(ok)
    coverage_arr = np.array(coverage, dtype=bool)
    coverage_rate = float(np.mean(coverage_arr))

    low_energy_ids = [int(i) for i in np.where(z < energy_z_floor)[0]]
    quality_pass = (coverage_rate >= min_coverage_rate) and (len(low_energy_ids) == 0)

    return FeatureQualityReport(
        n_images=n_images,
        coverage_all_families_rate=coverage_rate,
        energy_min=float(np.min(energy)),
        energy_mean=float(np.mean(energy)),
        energy_max=float(np.max(energy)),
        variance_min=float(np.min(variance)),
        variance_mean=float(np.mean(variance)),
        variance_max=float(np.max(variance)),
        low_energy_image_ids=low_energy_ids,
        quality_pass=quality_pass,
    )
