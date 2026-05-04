from __future__ import annotations

import numpy as np

from a_first_memory.config import PipelineConfig
from a_first_memory.pipeline import run_pipeline


def test_pipeline_with_nsd_npz(tmp_path) -> None:
    n_images = 20
    n_families = 5
    units_per_family = 4
    emb_dim = 8
    n_units = n_families * units_per_family
    n_rois = 5
    roi_voxels = 6
    n_voxels = n_rois * roi_voxels
    rng = np.random.default_rng(5)

    payload_path = tmp_path / "nsd_payload.npz"
    np.savez(
        payload_path,
        unit_embeddings=rng.normal(size=(n_images, n_units, emb_dim)),
        voxel_responses=rng.normal(size=(n_images, 3, n_voxels)),
        lags=rng.integers(0, 4, size=(n_images, 3)),
        hit_rates=np.clip(rng.normal(0.6, 0.1, size=(n_images, 3)), 0.0, 1.0),
        family_index_by_unit=np.repeat(np.arange(n_families), units_per_family),
        costs_by_unit=np.repeat(np.array([1.2, 1.1, 0.9, 0.7, 0.8]), units_per_family),
        family_names=np.array(["semantic", "object", "geometry", "low_level", "patch"], dtype=object),
        novelty_index=np.clip(rng.normal(0.5, 0.1, size=n_images), 0.0, 1.0),
        schema_congruence=np.clip(rng.normal(0.5, 0.1, size=n_images), 0.0, 1.0),
        semantic_targets=np.clip(rng.normal(0.6, 0.1, size=(n_images, 3)), 0.0, 1.0),
        perceptual_targets=np.clip(rng.normal(0.6, 0.1, size=(n_images, 3)), 0.0, 1.0),
        n_rois=n_rois,
        roi_voxels=roi_voxels,
    )

    cfg = PipelineConfig()
    cfg.rl.epochs = 4
    cfg.rl.reward_weight_grid = ((1.0, 0.8, 0.6),)
    cfg.encoding.banded_alphas = (1.0, 4.0)
    cfg.nsd.enabled = True
    cfg.nsd.npz_path = str(payload_path)

    out_dir = tmp_path / "outputs"
    result = run_pipeline(cfg, output_dir=str(out_dir), data_source="nsd")
    assert result["data_source"] == "nsd"
    assert "encoding_results" in result
    assert "behavior_results" in result
