from __future__ import annotations

import numpy as np

from a_first_memory.config import PipelineConfig
from a_first_memory.pipeline import run_pipeline


def test_nsd_feature_override_npz(tmp_path) -> None:
    n_images = 12
    n_families = 2
    units_per_family = 3
    emb_dim = 5
    n_units = n_families * units_per_family
    n_voxels = 10
    rng = np.random.default_rng(77)

    payload_path = tmp_path / "base_payload.npz"
    np.savez(
        payload_path,
        unit_embeddings=rng.normal(size=(n_images, n_units, emb_dim)),
        voxel_responses=rng.normal(size=(n_images, 3, n_voxels)),
        lags=rng.integers(0, 4, size=(n_images, 3)),
        hit_rates=np.clip(rng.normal(0.6, 0.1, size=(n_images, 3)), 0.0, 1.0),
        family_index_by_unit=np.repeat(np.arange(n_families), units_per_family),
        costs_by_unit=np.repeat(np.array([1.2, 0.8]), units_per_family),
        family_names=np.array(["semantic", "geometry"], dtype=object),
    )

    override_path = tmp_path / "feature_override.npz"
    np.savez(
        override_path,
        unit_embeddings=rng.normal(size=(n_images, n_units, emb_dim)),
        family_index_by_unit=np.repeat(np.arange(n_families), units_per_family),
        costs_by_unit=np.repeat(np.array([1.1, 0.9]), units_per_family),
        family_names=np.array(["semantic", "geometry"], dtype=object),
    )

    cfg = PipelineConfig()
    cfg.rl.epochs = 2
    cfg.rl.reward_weight_grid = ((1.0, 0.8, 0.6),)
    cfg.encoding.banded_alphas = (1.0,)
    cfg.nsd.npz_path = str(payload_path)
    cfg.nsd.feature_npz_path = str(override_path)

    result = run_pipeline(cfg, output_dir=str(tmp_path / "out"), data_source="nsd")
    assert result["data_source"] == "nsd"
