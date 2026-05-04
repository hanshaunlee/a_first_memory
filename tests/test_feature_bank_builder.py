from __future__ import annotations

import numpy as np

from a_first_memory.features.adapters import FamilyFeatureBlock, build_feature_bank


def test_build_feature_bank_shapes() -> None:
    block1 = FamilyFeatureBlock("semantic", np.zeros((10, 3, 8)), 1.2)
    block2 = FamilyFeatureBlock("geometry", np.zeros((10, 2, 8)), 0.9)
    bank = build_feature_bank([block1, block2])
    assert bank["unit_embeddings"].shape == (10, 5, 8)
    assert bank["family_index_by_unit"].shape == (5,)
    assert bank["costs_by_unit"].shape == (5,)
    assert bank["family_names"].shape == (2,)
