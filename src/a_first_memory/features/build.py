from __future__ import annotations

import numpy as np

from a_first_memory.data.synthetic import SyntheticDataset


def image_raw_representation(dataset: SyntheticDataset, image_id: int) -> np.ndarray:
    units = dataset.unit_embeddings[image_id]
    return units.reshape(-1)


def image_family_pooled_representation(dataset: SyntheticDataset, image_id: int) -> np.ndarray:
    units = dataset.unit_embeddings[image_id]
    pooled = []
    for fam_idx in range(len(dataset.family_names)):
        fam_units = units[dataset.family_index_by_unit == fam_idx]
        pooled.append(np.mean(fam_units, axis=0))
    return np.concatenate(pooled, axis=0)


def image_compressed_representation(
    dataset: SyntheticDataset,
    image_id: int,
    retention_weights: np.ndarray,
) -> np.ndarray:
    units = dataset.unit_embeddings[image_id]
    weighted = units * retention_weights[:, None]
    return weighted.reshape(-1)


def image_family_representations(
    dataset: SyntheticDataset,
    image_id: int,
    retention_weights: np.ndarray | None = None,
) -> np.ndarray:
    units = dataset.unit_embeddings[image_id]
    if retention_weights is not None:
        units = units * retention_weights[:, None]
    fam_rows = []
    for fam_idx in range(len(dataset.family_names)):
        fam_units = units[dataset.family_index_by_unit == fam_idx]
        fam_rows.append(np.mean(fam_units, axis=0))
    return np.vstack(fam_rows)


def _retention_for_exposure(
    retention_by_image: np.ndarray,
    image_id: int,
    exposure_idx: int,
) -> np.ndarray:
    if retention_by_image.ndim == 3:
        return retention_by_image[image_id, exposure_idx]
    return retention_by_image[image_id]


def build_exposure_feature_matrix(
    dataset: SyntheticDataset,
    exposure_idx: int,
    strategy: str,
    retention_by_image: np.ndarray | None = None,
    image_ids: np.ndarray | None = None,
) -> np.ndarray:
    rows = []
    ids = dataset.image_ids if image_ids is None else image_ids
    for image_id in ids:
        if strategy == "raw":
            rows.append(image_raw_representation(dataset, int(image_id)))
        elif strategy == "family_pool":
            rows.append(image_family_pooled_representation(dataset, int(image_id)))
        elif strategy == "compressed":
            if retention_by_image is None:
                raise ValueError("retention_by_image is required for compressed strategy")
            rows.append(
                image_compressed_representation(
                    dataset,
                    int(image_id),
                    _retention_for_exposure(retention_by_image, int(image_id), exposure_idx),
                )
            )
        else:
            raise ValueError(f"Unknown feature strategy: {strategy}")
    return np.vstack(rows)


def build_exposure_voxel_matrix(
    dataset: SyntheticDataset,
    exposure_idx: int,
    image_ids: np.ndarray | None = None,
) -> np.ndarray:
    ids = dataset.image_ids if image_ids is None else image_ids
    return dataset.voxel_responses[ids, exposure_idx, :]


def build_exposure_family_feature_tensor(
    dataset: SyntheticDataset,
    exposure_idx: int,
    strategy: str,
    retention_by_image: np.ndarray | None = None,
    image_ids: np.ndarray | None = None,
) -> np.ndarray:
    rows = []
    ids = dataset.image_ids if image_ids is None else image_ids
    for image_id in ids:
        if strategy == "raw":
            rows.append(image_family_representations(dataset, int(image_id)))
        elif strategy == "compressed":
            if retention_by_image is None:
                raise ValueError("retention_by_image is required for compressed strategy")
            rows.append(
                image_family_representations(
                    dataset,
                    int(image_id),
                    retention_weights=_retention_for_exposure(retention_by_image, int(image_id), exposure_idx),
                )
            )
        else:
            raise ValueError(f"Unknown strategy for family tensor: {strategy}")
    return np.stack(rows, axis=0)
