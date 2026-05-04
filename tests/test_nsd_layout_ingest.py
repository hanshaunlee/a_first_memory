from __future__ import annotations

import numpy as np

from a_first_memory.config import PipelineConfig
from a_first_memory.pipeline import run_pipeline


def test_pipeline_with_nsd_layout_root(tmp_path) -> None:
    n_images = 16
    n_families = 5
    units_per_family = 3
    emb_dim = 6
    n_units = n_families * units_per_family
    n_rois = 4
    roi_voxels = 5
    n_voxels = n_rois * roi_voxels
    rng = np.random.default_rng(29)

    root = tmp_path / "layout_root"
    (root / "features").mkdir(parents=True, exist_ok=True)
    (root / "fmri").mkdir(parents=True, exist_ok=True)
    (root / "behavior").mkdir(parents=True, exist_ok=True)

    np.save(root / "features" / "unit_embeddings.npy", rng.normal(size=(n_images, n_units, emb_dim)))
    np.save(root / "fmri" / "voxel_responses.npy", rng.normal(size=(n_images, 3, n_voxels)))
    np.save(root / "behavior" / "lags.npy", rng.integers(0, 4, size=(n_images, 3)))
    np.save(root / "behavior" / "hit_rates.npy", np.clip(rng.normal(0.6, 0.1, size=(n_images, 3)), 0.0, 1.0))
    np.save(root / "features" / "family_index_by_unit.npy", np.repeat(np.arange(n_families), units_per_family))
    np.save(root / "features" / "costs_by_unit.npy", np.repeat(np.array([1.2, 1.1, 0.9, 0.7, 0.8]), units_per_family))
    np.save(root / "features" / "family_names.npy", np.array(["semantic", "object", "geometry", "low_level", "patch"], dtype=object))

    cfg = PipelineConfig()
    cfg.rl.epochs = 3
    cfg.rl.reward_weight_grid = ((1.0, 0.8, 0.6),)
    cfg.encoding.banded_alphas = (1.0,)
    cfg.nsd.layout_root = str(root)

    result = run_pipeline(cfg, output_dir=str(tmp_path / "outputs"), data_source="nsd")
    assert result["data_source"] == "nsd"
    assert "behavior_results" in result
