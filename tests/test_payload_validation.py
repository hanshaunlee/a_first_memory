from __future__ import annotations

import numpy as np

from a_first_memory.data.payload import validate_payload_dict


def test_payload_validation_accepts_valid_shapes() -> None:
    n_images = 8
    n_units = 10
    payload = {
        "unit_embeddings": np.zeros((n_images, n_units, 4)),
        "voxel_responses": np.zeros((n_images, 3, 12)),
        "lags": np.zeros((n_images, 3)),
        "hit_rates": np.ones((n_images, 3)) * 0.5,
        "family_index_by_unit": np.zeros((n_units,)),
        "costs_by_unit": np.ones((n_units,)),
        "family_names": np.array(["semantic", "object"], dtype=object),
    }
    validate_payload_dict(payload)
