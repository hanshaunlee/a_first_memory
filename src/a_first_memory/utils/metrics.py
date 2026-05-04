from __future__ import annotations

import numpy as np


def explained_variance_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def rdm(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.maximum(norms, 1e-8)
    normalized = matrix / safe
    sims = normalized @ normalized.T
    return 1.0 - sims


def upper_triangle_values(square: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(square.shape[0], k=1)
    return square[idx]
