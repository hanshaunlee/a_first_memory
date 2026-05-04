from __future__ import annotations

import numpy as np

from a_first_memory.config import PipelineConfig
from a_first_memory.pipeline import run_pipeline


def test_pipeline_with_nsd_raw_dir(tmp_path) -> None:
    n_images = 18
    n_families = 5
    units_per_family = 3
    emb_dim = 6
    n_units = n_families * units_per_family
    n_rois = 3
    roi_voxels = 5
    n_voxels = n_rois * roi_voxels
    rng = np.random.default_rng(17)

    raw_dir = tmp_path / "nsd_raw_payload"
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.save(raw_dir / "unit_embeddings.npy", rng.normal(size=(n_images, n_units, emb_dim)))
    np.save(raw_dir / "voxel_responses.npy", rng.normal(size=(n_images, 3, n_voxels)))
    np.save(raw_dir / "lags.npy", rng.integers(0, 4, size=(n_images, 3)))
    np.save(raw_dir / "hit_rates.npy", np.clip(rng.normal(0.6, 0.1, size=(n_images, 3)), 0.0, 1.0))
    np.save(raw_dir / "family_index_by_unit.npy", np.repeat(np.arange(n_families), units_per_family))
    np.save(raw_dir / "costs_by_unit.npy", np.repeat(np.array([1.2, 1.1, 0.9, 0.7, 0.8]), units_per_family))
    np.save(raw_dir / "family_names.npy", np.array(["semantic", "object", "geometry", "low_level", "patch"], dtype=object))
    np.save(raw_dir / "n_rois.npy", np.array(n_rois))
    np.save(raw_dir / "roi_voxels.npy", np.array(roi_voxels))

    cfg = PipelineConfig()
    cfg.rl.epochs = 3
    cfg.rl.reward_weight_grid = ((1.0, 0.8, 0.6),)
    cfg.encoding.banded_alphas = (1.0,)
    cfg.nsd.raw_dir = str(raw_dir)

    result = run_pipeline(cfg, output_dir=str(tmp_path / "outputs"), data_source="nsd")
    assert result["data_source"] == "nsd"
    assert "rsa_results" in result
