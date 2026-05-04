from __future__ import annotations

import json
import numpy as np

from a_first_memory.config import PipelineConfig
from a_first_memory.data.synthetic import generate_synthetic_dataset
from a_first_memory.pipeline import run_pipeline


def test_pipeline_smoke(tmp_path) -> None:
    cfg = PipelineConfig()
    cfg.data.n_images = 24
    cfg.rl.epochs = 5
    cfg.rl.reward_weight_grid = ((1.0, 0.8, 0.6),)
    cfg.encoding.banded_alphas = (1.0, 4.0)
    dataset = generate_synthetic_dataset(cfg.data)
    payload_path = tmp_path / "nsd_payload.npz"
    np.savez(
        payload_path,
        unit_embeddings=dataset.unit_embeddings,
        voxel_responses=dataset.voxel_responses,
        lags=dataset.lags,
        hit_rates=dataset.hit_rates,
        family_index_by_unit=dataset.family_index_by_unit,
        costs_by_unit=dataset.costs_by_unit,
        family_names=np.array(dataset.family_names, dtype=object),
        novelty_index=dataset.novelty_index,
        schema_congruence=dataset.schema_congruence,
        semantic_targets=dataset.semantic_targets,
        perceptual_targets=dataset.perceptual_targets,
        subject_ids=dataset.subject_ids,
        subject_names=np.array(dataset.subject_names, dtype=object),
        n_rois=np.array(dataset.n_rois),
        roi_voxels=np.array(dataset.roi_voxels),
    )
    cfg.nsd.npz_path = str(payload_path)
    cfg.nsd.strict = True
    output_dir = tmp_path / "outputs"
    result = run_pipeline(cfg, output_dir=str(output_dir), data_source="nsd")
    assert "encoding_results" in result
    assert "rsa_results" in result
    assert "behavior_results" in result
    assert "subject_results" in result
    assert "feature_quality" in result
    assert "retention_diagnostics" in result
    assert "setup_warnings" in result

    out_file = output_dir / "results.json"
    assert out_file.exists()
    parsed = json.loads(out_file.read_text(encoding="utf-8"))
    assert "family_shift" in parsed
