from __future__ import annotations

import numpy as np


def train_test_indices(n: int, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut = int(n * train_frac)
    train_idx = order[:cut]
    test_idx = order[cut:]
    return train_idx, test_idx
