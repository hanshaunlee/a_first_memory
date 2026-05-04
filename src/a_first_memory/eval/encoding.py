from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np
from sklearn.linear_model import Ridge

from a_first_memory.config import EncodingConfig
from a_first_memory.data.synthetic import SyntheticDataset
from a_first_memory.features.build import build_exposure_feature_matrix, build_exposure_voxel_matrix
from a_first_memory.utils.metrics import explained_variance_r2
from a_first_memory.utils.split import train_test_indices


def _stabilize_matrix(x: np.ndarray, clip: float = 1e6) -> np.ndarray:
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip), -clip, clip)


def _stabilize_ridge(model: Ridge, clip: float = 1e6) -> Ridge:
    model.coef_ = _stabilize_matrix(model.coef_, clip=clip)
    model.intercept_ = np.clip(np.nan_to_num(model.intercept_, nan=0.0, posinf=clip, neginf=-clip), -clip, clip)
    return model


@dataclass
class EncodingResult:
    strategy: str
    exposure_scores: list[float]
    family_drop_scores: list[list[float]]
    best_alphas_by_exposure: list[list[float]]


def _feature_family_groups(dataset: SyntheticDataset) -> list[np.ndarray]:
    emb_dim = dataset.unit_embeddings.shape[-1]
    groups: list[np.ndarray] = []
    for fam_idx, _ in enumerate(dataset.family_names):
        unit_positions = np.where(dataset.family_index_by_unit == fam_idx)[0]
        cols = []
        for unit_pos in unit_positions:
            start = int(unit_pos) * emb_dim
            cols.extend(range(start, start + emb_dim))
        groups.append(np.array(cols, dtype=int))
    return groups


def _apply_group_scaling(x: np.ndarray, groups: list[np.ndarray], alphas: tuple[float, ...]) -> np.ndarray:
    scaled = x.copy()
    for group_idx, cols in enumerate(groups):
        scaled[:, cols] = scaled[:, cols] / np.sqrt(alphas[group_idx])
    return scaled


def _fit_banded_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    groups: list[np.ndarray],
    alpha_grid: tuple[float, ...],
    max_iter: int,
) -> tuple[Ridge, tuple[float, ...]]:
    # Coordinate-descent search scales linearly with family count.
    n_groups = len(groups)
    start_alpha = alpha_grid[min(len(alpha_grid) // 2, len(alpha_grid) - 1)]
    combo = [float(start_alpha)] * n_groups
    best_model: Ridge | None = None
    best_score = -np.inf

    for _ in range(max(max_iter, 1)):
        updated = False
        for g_idx in range(n_groups):
            local_best_alpha = combo[g_idx]
            local_best_score = -np.inf
            local_best_model: Ridge | None = None
            for alpha in alpha_grid:
                trial = combo.copy()
                trial[g_idx] = float(alpha)
                xs_train = _apply_group_scaling(x_train, groups, tuple(trial))
                xs_val = _apply_group_scaling(x_val, groups, tuple(trial))
                xs_train = _stabilize_matrix(xs_train)
                xs_val = _stabilize_matrix(xs_val)
                model = Ridge(alpha=1.0)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
                    model.fit(xs_train, y_train)
                model = _stabilize_ridge(model)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
                    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                        pred = model.predict(xs_val)
                pred = _stabilize_matrix(pred)
                score = explained_variance_r2(y_val, pred)
                if score > local_best_score:
                    local_best_score = score
                    local_best_alpha = float(alpha)
                    local_best_model = model
            if local_best_score > best_score + 1e-12:
                best_score = local_best_score
                best_model = local_best_model
                updated = True
            combo[g_idx] = local_best_alpha
        if not updated:
            break

    if best_model is None:
        xs_train = _apply_group_scaling(x_train, groups, tuple(combo))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
            best_model = Ridge(alpha=1.0).fit(xs_train, y_train)
    best_model = _stabilize_ridge(best_model)
    return best_model, tuple(combo)


def evaluate_encoding(
    dataset: SyntheticDataset,
    cfg: EncodingConfig,
    strategy: str,
    retention_by_image: np.ndarray | None = None,
    image_indices: np.ndarray | None = None,
) -> EncodingResult:
    scores: list[float] = []
    family_drop_scores: list[list[float]] = []
    best_alphas_by_exposure: list[list[float]] = []
    groups = _feature_family_groups(dataset)
    ids = dataset.image_ids if image_indices is None else image_indices
    if len(ids) < 6:
        zero_family = [[0.0 for _ in dataset.family_names] for _ in range(3)]
        default_alphas = [[cfg.ridge_alpha for _ in dataset.family_names] for _ in range(3)]
        return EncodingResult(
            strategy=strategy,
            exposure_scores=[0.0, 0.0, 0.0],
            family_drop_scores=zero_family,
            best_alphas_by_exposure=default_alphas,
        )
    for exposure_idx in range(3):
        x = build_exposure_feature_matrix(
            dataset=dataset,
            exposure_idx=exposure_idx,
            strategy=strategy,
            retention_by_image=retention_by_image,
            image_ids=ids,
        )
        y = build_exposure_voxel_matrix(dataset, exposure_idx, image_ids=ids)
        x = _stabilize_matrix(x)
        y = _stabilize_matrix(y)
        train_idx, test_idx = train_test_indices(len(ids), cfg.train_frac, seed=exposure_idx + 11)
        # Internal split for selecting per-family regularization.
        inner_train, inner_val = train_test_indices(len(train_idx), 0.8, seed=exposure_idx + 101)
        tr = train_idx[inner_train]
        va = train_idx[inner_val]

        if strategy == "raw" or strategy == "compressed":
            model, alpha_combo = _fit_banded_ridge(
                x_train=x[tr],
                y_train=y[tr],
                x_val=x[va],
                y_val=y[va],
                groups=groups,
                alpha_grid=cfg.banded_alphas,
                max_iter=cfg.banded_max_iter,
            )
            x_train_scaled = _apply_group_scaling(x[train_idx], groups, alpha_combo)
            x_test_scaled = _apply_group_scaling(x[test_idx], groups, alpha_combo)
            x_train_scaled = _stabilize_matrix(x_train_scaled)
            x_test_scaled = _stabilize_matrix(x_test_scaled)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
                model.fit(x_train_scaled, y[train_idx])
            model = _stabilize_ridge(model)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    pred = model.predict(x_test_scaled)
            pred = _stabilize_matrix(pred)
            best_alphas_by_exposure.append([float(a) for a in alpha_combo])
            full_coef = model.coef_.copy()
            family_scores = []
            full_score = explained_variance_r2(y[test_idx], pred)
            for fam_idx, cols in enumerate(groups):
                masked_coef = full_coef.copy()
                masked_coef[:, cols] = 0.0
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
                    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                        reduced_pred = x_test_scaled @ masked_coef.T + model.intercept_[None, :]
                reduced_pred = _stabilize_matrix(reduced_pred)
                reduced_score = explained_variance_r2(y[test_idx], reduced_pred)
                family_scores.append(float(full_score - reduced_score))
            family_drop_scores.append(family_scores)
        else:
            model = Ridge(alpha=cfg.ridge_alpha)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
                model.fit(x[train_idx], y[train_idx])
            model = _stabilize_ridge(model)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    pred = model.predict(x[test_idx])
            pred = _stabilize_matrix(pred)
            family_drop_scores.append([0.0 for _ in dataset.family_names])
            best_alphas_by_exposure.append([cfg.ridge_alpha for _ in dataset.family_names])
        score = explained_variance_r2(y[test_idx], pred)
        scores.append(score)
    return EncodingResult(
        strategy=strategy,
        exposure_scores=scores,
        family_drop_scores=family_drop_scores,
        best_alphas_by_exposure=best_alphas_by_exposure,
    )
