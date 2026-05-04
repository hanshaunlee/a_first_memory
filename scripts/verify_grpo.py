from __future__ import annotations

import json
import sys
import warnings
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a_first_memory.config import PipelineConfig
from a_first_memory.data.synthetic import generate_synthetic_dataset
from a_first_memory.rl.memory_policy import MemorySelectionPolicy
from a_first_memory.rl.probes import pretrain_probe_heads, score_with_frozen_probes


def _assert_binary_matrix(name: str, x: np.ndarray) -> None:
    if not np.all(np.isin(x, [0.0, 1.0])):
        raise AssertionError(f"{name} is not binary (0/1) as expected.")


def _assert_budget_feasible(name: str, retention: np.ndarray, costs: np.ndarray, budget: float) -> dict[str, float]:
    per_row_cost = np.sum(retention * costs[None, None, :], axis=2)
    max_cost = float(np.max(per_row_cost))
    min_cost = float(np.min(per_row_cost))
    if max_cost > budget + 1e-6:
        raise AssertionError(
            f"{name} violates budget. max={max_cost:.6f} budget={budget:.6f}"
        )
    return {"min_cost": min_cost, "max_cost": max_cost, "mean_cost": float(np.mean(per_row_cost))}


def _build_small_cfg() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.rl.policy_architecture = "linear"
    cfg.data.n_images = 32
    cfg.data.n_units_per_family = 4
    cfg.data.embedding_dim = 12
    cfg.data.random_seed = 11
    cfg.rl.budget = 18.0
    cfg.rl.epochs = 6
    cfg.rl.reward_weight_grid = ((1.0, 0.8, 0.6),)
    cfg.rl.grpo_group_size = 5
    cfg.rl.grpo_adv_clip = 2.5
    return cfg


def run_verifier() -> dict:
    cfg = _build_small_cfg()
    dataset = generate_synthetic_dataset(cfg.data)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*matmul.*",
            category=RuntimeWarning,
        )
        probes = pretrain_probe_heads(dataset, alpha=10.0)

    report: dict = {
        "config_snapshot": asdict(cfg.rl),
        "checks": {},
    }

    # 1) Default algorithm check
    if cfg.rl.algorithm.strip().lower() != "grpo":
        raise AssertionError("Default RL algorithm is not 'grpo'.")
    report["checks"]["default_algorithm"] = "passed"

    # 2) Probe-head sanity (trained and finite predictions on valid input)
    full_selected = np.ones(dataset.unit_embeddings.shape[1], dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*matmul.*",
            category=RuntimeWarning,
        )
        probe_scores = score_with_frozen_probes(
            probes=probes,
            dataset=dataset,
            image_id=0,
            selected=full_selected,
        )
    if not np.all(np.isfinite(np.array(probe_scores, dtype=float))):
        raise AssertionError("Probe-head sanity check failed (non-finite scores).")
    report["checks"]["probe_head_sanity"] = {
        "status": "passed",
        "scores": [float(x) for x in probe_scores],
    }

    # 3) GRPO training and structural invariants
    grpo_policy = MemorySelectionPolicy(dataset, cfg.rl, probes=None)
    grpo_initial_theta = grpo_policy.theta.copy()
    grpo_result = grpo_policy.train()
    if len(grpo_result.reward_history) != cfg.rl.epochs:
        raise AssertionError("GRPO reward history length does not match epochs.")
    if len(grpo_result.lagrangian_history) != cfg.rl.epochs:
        raise AssertionError("GRPO lagrangian history length does not match epochs.")
    if not np.all(np.isfinite(np.array(grpo_result.reward_history, dtype=float))):
        raise AssertionError("GRPO reward history contains non-finite values.")
    if not np.all(np.isfinite(np.array(grpo_result.lagrangian_history, dtype=float))):
        raise AssertionError("GRPO lagrangian history contains non-finite values.")
    if float(np.min(np.array(grpo_result.lagrangian_history, dtype=float))) < -1e-8:
        raise AssertionError("GRPO lagrangian multiplier dropped below zero.")
    if grpo_result.retention_by_image_exposure.shape != (
        cfg.data.n_images,
        3,
        dataset.unit_embeddings.shape[1],
    ):
        raise AssertionError("GRPO retention tensor shape is incorrect.")
    _assert_binary_matrix("GRPO retention", grpo_result.retention_by_image_exposure)
    budget_stats_grpo = _assert_budget_feasible(
        "GRPO retention",
        grpo_result.retention_by_image_exposure,
        dataset.costs_by_unit,
        cfg.rl.budget,
    )
    hit_preds_grpo = grpo_policy.predict_hit_rates(grpo_result.retention_by_image_exposure)
    if hit_preds_grpo.shape != (cfg.data.n_images, 3):
        raise AssertionError("GRPO hit-rate prediction shape is incorrect.")
    if float(np.min(hit_preds_grpo)) < 0.0 or float(np.max(hit_preds_grpo)) > 1.0:
        raise AssertionError("GRPO hit-rate predictions are outside [0, 1].")
    grpo_diag = grpo_result.training_diagnostics
    if grpo_diag.get("algorithm") != "grpo":
        raise AssertionError("GRPO diagnostics missing algorithm marker.")
    for key in ("clip_fraction_history", "approx_kl_history", "grad_norm_history"):
        values = np.array(grpo_diag.get(key, []), dtype=float)
        if values.shape[0] != cfg.rl.epochs:
            raise AssertionError(f"GRPO diagnostics {key} length does not match epochs.")
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"GRPO diagnostics {key} contains non-finite values.")
    clip_vals = np.array(grpo_diag["clip_fraction_history"], dtype=float)
    if float(np.min(clip_vals)) < -1e-8 or float(np.max(clip_vals)) > 1.0 + 1e-8:
        raise AssertionError("GRPO clip fraction must be inside [0, 1].")
    if float(np.min(np.array(grpo_diag["approx_kl_history"], dtype=float))) < -1e-8:
        raise AssertionError("Approximate KL should be non-negative.")

    report["checks"]["grpo_core"] = {
        "status": "passed",
        "theta_l2_shift": float(np.linalg.norm(grpo_policy.theta - grpo_initial_theta)),
        "reward_mean": float(np.mean(grpo_result.reward_history)),
        "reward_last": float(grpo_result.reward_history[-1]),
        "lagrangian_last": float(grpo_result.lagrangian_history[-1]),
        "budget_stats": budget_stats_grpo,
        "clip_fraction_mean": float(np.mean(clip_vals)),
        "approx_kl_mean": float(np.mean(np.array(grpo_diag["approx_kl_history"], dtype=float))),
        "grad_norm_mean": float(np.mean(np.array(grpo_diag["grad_norm_history"], dtype=float))),
    }

    # 4) Determinism check for GRPO given fixed dataset/config
    grpo_policy_2 = MemorySelectionPolicy(dataset, cfg.rl, probes=None)
    grpo_result_2 = grpo_policy_2.train()
    if not np.array_equal(
        grpo_result.retention_by_image_exposure,
        grpo_result_2.retention_by_image_exposure,
    ):
        raise AssertionError("GRPO determinism check failed (retention mismatch).")
    report["checks"]["grpo_determinism"] = "passed"

    # 5) REINFORCE fallback still works
    reinforce_cfg = replace(cfg.rl)
    reinforce_cfg.algorithm = "reinforce"
    reinforce_policy = MemorySelectionPolicy(dataset, reinforce_cfg, probes=None)
    reinforce_result = reinforce_policy.train()
    if len(reinforce_result.reward_history) != cfg.rl.epochs:
        raise AssertionError("REINFORCE reward history length does not match epochs.")
    if reinforce_result.training_diagnostics.get("algorithm") != "reinforce":
        raise AssertionError("REINFORCE diagnostics missing algorithm marker.")
    _assert_binary_matrix("REINFORCE retention", reinforce_result.retention_by_image_exposure)
    budget_stats_reinforce = _assert_budget_feasible(
        "REINFORCE retention",
        reinforce_result.retention_by_image_exposure,
        dataset.costs_by_unit,
        cfg.rl.budget,
    )
    report["checks"]["reinforce_fallback"] = {
        "status": "passed",
        "reward_mean": float(np.mean(reinforce_result.reward_history)),
        "reward_last": float(reinforce_result.reward_history[-1]),
        "lagrangian_last": float(reinforce_result.lagrangian_history[-1]),
        "budget_stats": budget_stats_reinforce,
    }

    # 6) Unknown algorithm should fail clearly
    bad_cfg = replace(cfg.rl)
    bad_cfg.algorithm = "not-a-real-algorithm"
    try:
        MemorySelectionPolicy(dataset, bad_cfg, probes=None).train()
        raise AssertionError("Unknown algorithm did not raise an error.")
    except ValueError:
        report["checks"]["unknown_algorithm_guard"] = "passed"

    return report


def main() -> None:
    report = run_verifier()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
