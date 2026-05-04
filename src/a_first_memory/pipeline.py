from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from a_first_memory.config import PipelineConfig
from a_first_memory.data.nsd import load_nsd_dataset
from a_first_memory.eval.behavior import evaluate_behavior_fit, evaluate_behavior_fit_subset
from a_first_memory.eval.baselines import (
    pca_like_compression_policy,
    random_budget_policy,
    saliency_like_policy,
)
from a_first_memory.eval.encoding import evaluate_encoding
from a_first_memory.eval.family_shift import compare_policy_brain_shift, summarize_retention_shift
from a_first_memory.eval.feature_quality import evaluate_feature_quality
from a_first_memory.eval.rsa import evaluate_fr_rsa
from a_first_memory.rl.memory_policy import (
    POLICY_CHECKPOINT_FILENAME,
    POLICY_TORCH_FILENAME,
    MemorySelectionPolicy,
)
from a_first_memory.rl.probes import pretrain_probe_heads


def _retention_diagnostics(dataset, retention_by_image_exposure: np.ndarray, budget: float) -> dict:
    total_cost = retention_by_image_exposure * dataset.costs_by_unit[None, None, :]
    used_budget = np.sum(total_cost, axis=2)
    selected_units = np.sum(retention_by_image_exposure, axis=2)
    return {
        "avg_budget_utilization": float(np.mean(used_budget / max(budget, 1e-8))),
        "min_budget_utilization": float(np.min(used_budget / max(budget, 1e-8))),
        "max_budget_utilization": float(np.max(used_budget / max(budget, 1e-8))),
        "avg_selected_units": float(np.mean(selected_units)),
        "min_selected_units": int(np.min(selected_units)),
        "max_selected_units": int(np.max(selected_units)),
    }


def _setup_warnings(
    data_source: str,
    feature_quality: dict,
    retention_diagnostics: dict,
    encoding_results: list[dict],
    rsa_results: list[dict],
) -> list[str]:
    warnings: list[str] = []
    if data_source == "synthetic":
        warnings.append("Pipeline is running on synthetic data; results are not directly neuroscientific claims.")
    if not bool(feature_quality.get("quality_pass", False)):
        n_low = len(feature_quality.get("low_energy_image_ids", []))
        warnings.append(f"Feature quality gate failed: {n_low} low-energy image outliers detected.")
    util = float(retention_diagnostics.get("avg_budget_utilization", 0.0))
    if util < 0.9:
        warnings.append(f"Average budget utilization is low ({util:.3f}); selector may be underfilling memory.")
    raw_enc = float(np.mean(encoding_results[0]["exposure_scores"]))
    rl_enc = float(np.mean(encoding_results[1]["exposure_scores"]))
    if rl_enc + 1e-8 < raw_enc:
        warnings.append("RL-compressed encoding underperforms raw features; tune budget/reward before interpretation.")
    raw_rsa = float(np.mean(rsa_results[0]["exposure_correlations"]))
    rl_rsa = float(np.mean(rsa_results[1]["exposure_correlations"]))
    if rl_rsa + 1e-8 < raw_rsa:
        warnings.append("RL-compressed FR-RSA underperforms raw features; feature retention policy likely too lossy.")
    return warnings


def _subjectwise_results(
    dataset,
    config: PipelineConfig,
    rl_retention: np.ndarray,
    random_retention: np.ndarray,
    saliency_retention: np.ndarray,
    pca_retention: np.ndarray,
) -> dict[str, dict]:
    subject_results: dict[str, dict] = {}
    subject_ids = np.unique(dataset.subject_ids)
    for subject_id in subject_ids:
        mask = np.where(dataset.subject_ids == subject_id)[0]
        if len(mask) < config.subject_eval.min_images_per_subject:
            continue
        label = (
            dataset.subject_names[int(subject_id)]
            if int(subject_id) < len(dataset.subject_names)
            else f"subj{int(subject_id):02d}"
        )
        encoding_rl = evaluate_encoding(
            dataset,
            config.encoding,
            strategy="compressed",
            retention_by_image=rl_retention,
            image_indices=mask,
        )
        rsa_rl = evaluate_fr_rsa(
            dataset,
            config.rsa,
            strategy="compressed",
            retention_by_image=rl_retention,
            image_indices=mask,
        )
        behavior_rl = evaluate_behavior_fit_subset(dataset, "rl_policy", rl_retention, image_indices=mask)
        behavior_random = evaluate_behavior_fit_subset(dataset, "random", random_retention, image_indices=mask)
        behavior_saliency = evaluate_behavior_fit_subset(dataset, "saliency", saliency_retention, image_indices=mask)
        behavior_generic = evaluate_behavior_fit_subset(dataset, "generic_compression", pca_retention, image_indices=mask)
        subject_results[label] = {
            "n_images": int(len(mask)),
            "encoding_rl": asdict(encoding_rl),
            "fr_rsa_rl": asdict(rsa_rl),
            "behavior": [
                asdict(behavior_rl),
                asdict(behavior_random),
                asdict(behavior_saliency),
                asdict(behavior_generic),
            ],
        }
    return subject_results


def run_pipeline(config: PipelineConfig, output_dir: str = "outputs", data_source: str = "nsd") -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if data_source == "nsd":
        dataset = load_nsd_dataset(config.nsd)
    elif data_source == "synthetic":
        raise RuntimeError(
            "Synthetic data source is disabled in research mode. "
            "Use NSD payload ingestion for train/eval runs."
        )
    else:
        raise ValueError(f"Unknown data_source: {data_source}")

    probes = pretrain_probe_heads(dataset, alpha=config.encoding.ridge_alpha)
    feature_quality = asdict(evaluate_feature_quality(dataset))

    sweep_results: list[dict] = []
    best_alignment = float("-inf")
    best_payload: dict | None = None
    best_checkpoint_data: dict | None = None
    _SWEEP_JSON_EXCLUDE = frozenset({"retention", "theta"})
    for alpha, beta, gamma in config.rl.reward_weight_grid:
        rl_cfg = replace(
            config.rl,
            alpha_recog=alpha,
            beta_sem=beta,
            gamma_perc=gamma,
        )
        rl_policy = MemorySelectionPolicy(dataset, rl_cfg, probes=probes)
        train_result = rl_policy.train()
        rl_retention = train_result.retention_by_image_exposure
        fr_rsa_rl = evaluate_fr_rsa(dataset, config.rsa, strategy="compressed", retention_by_image=rl_retention)
        family_shift = summarize_retention_shift(dataset, rl_retention)
        alignment = compare_policy_brain_shift(
            family_shift.policy_delta_exposure3_minus1,
            fr_rsa_rl.roi_family_weights,
        )
        behavior = evaluate_behavior_fit(dataset, "policy", rl_retention)
        payload = {
            "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
            "training_reward_history": train_result.reward_history,
            "lagrangian_history": train_result.lagrangian_history,
            "training_diagnostics": train_result.training_diagnostics,
            "family_shift": asdict(family_shift),
            "family_brain_alignment": asdict(alignment),
            "fr_rsa": asdict(fr_rsa_rl),
            "behavior_fit": asdict(behavior),
            "retention": rl_retention,
            "policy_architecture": rl_policy.policy_architecture,
            "lagrangian_budget": float(rl_policy.lagrangian_budget),
            "feature_dim": int(rl_policy.feature_dim),
        }
        if rl_policy.policy_architecture == "linear":
            assert rl_policy.theta is not None
            payload["theta"] = rl_policy.theta.copy()
        sweep_results.append({k: v for k, v in payload.items() if k not in _SWEEP_JSON_EXCLUDE})
        alignment_score = float(alignment.mean_spearman)
        if alignment_score > best_alignment:
            best_alignment = alignment_score
            best_payload = payload
            if rl_policy.policy_architecture == "mlp":
                assert rl_policy._policy_net is not None
                best_checkpoint_data = {
                    "kind": "mlp",
                    "state_dict": {k: v.detach().cpu().clone() for k, v in rl_policy._policy_net.state_dict().items()},
                }
            else:
                assert rl_policy.theta is not None
                best_checkpoint_data = {
                    "kind": "linear",
                    "theta": rl_policy.theta.copy(),
                    "lagrangian_budget": float(rl_policy.lagrangian_budget),
                    "feature_dim": int(rl_policy.feature_dim),
                }

    if best_payload is None or best_checkpoint_data is None:
        raise RuntimeError("Reward sweep did not produce a valid policy result.")

    rl_retention = best_payload["retention"]
    retention_diagnostics = _retention_diagnostics(dataset, rl_retention, budget=config.rl.budget)

    random_retention = random_budget_policy(dataset, budget=config.rl.budget)
    saliency_retention = saliency_like_policy(dataset, budget=config.rl.budget)
    pca_retention = pca_like_compression_policy(dataset, budget=config.rl.budget)

    encoding_results = [
        asdict(evaluate_encoding(dataset, config.encoding, strategy="raw")),
        asdict(
            evaluate_encoding(
                dataset,
                config.encoding,
                strategy="compressed",
                retention_by_image=rl_retention,
            )
        ),
        asdict(
            evaluate_encoding(
                dataset,
                config.encoding,
                strategy="compressed",
                retention_by_image=random_retention,
            )
        ),
        asdict(
            evaluate_encoding(
                dataset,
                config.encoding,
                strategy="compressed",
                retention_by_image=saliency_retention,
            )
        ),
        asdict(
            evaluate_encoding(
                dataset,
                config.encoding,
                strategy="compressed",
                retention_by_image=pca_retention,
            )
        ),
    ]

    rsa_results = [
        asdict(evaluate_fr_rsa(dataset, config.rsa, strategy="raw")),
        asdict(evaluate_fr_rsa(dataset, config.rsa, strategy="compressed", retention_by_image=rl_retention)),
        asdict(evaluate_fr_rsa(dataset, config.rsa, strategy="compressed", retention_by_image=random_retention)),
        asdict(evaluate_fr_rsa(dataset, config.rsa, strategy="compressed", retention_by_image=saliency_retention)),
        asdict(evaluate_fr_rsa(dataset, config.rsa, strategy="compressed", retention_by_image=pca_retention)),
    ]

    family_shift = asdict(summarize_retention_shift(dataset, rl_retention))
    family_brain_alignment = asdict(
        compare_policy_brain_shift(
            family_shift["policy_delta_exposure3_minus1"],
            rsa_results[1]["roi_family_weights"],
        )
    )
    behavior_results = [
        asdict(evaluate_behavior_fit(dataset, "rl_policy", rl_retention)),
        asdict(evaluate_behavior_fit(dataset, "random", random_retention)),
        asdict(evaluate_behavior_fit(dataset, "saliency", saliency_retention)),
        asdict(evaluate_behavior_fit(dataset, "generic_compression", pca_retention)),
    ]
    subject_results = (
        _subjectwise_results(
            dataset,
            config,
            rl_retention=rl_retention,
            random_retention=random_retention,
            saliency_retention=saliency_retention,
            pca_retention=pca_retention,
        )
        if config.subject_eval.enabled
        else {}
    )

    result = {
        "config": asdict(config),
        "data_source": data_source,
        "selected_reward_weights": best_payload["weights"],
        "policy_checkpoint": {
            "file": (
                POLICY_TORCH_FILENAME
                if best_payload["policy_architecture"] == "mlp"
                else POLICY_CHECKPOINT_FILENAME
            ),
            "format": "torch" if best_payload["policy_architecture"] == "mlp" else "npz",
            "policy_architecture": best_payload["policy_architecture"],
            "feature_dim": best_payload["feature_dim"],
            "lagrangian_budget": best_payload["lagrangian_budget"],
            **(
                {
                    "policy_hidden_dim": config.rl.policy_hidden_dim,
                    "policy_num_hidden_layers": config.rl.policy_num_hidden_layers,
                }
                if best_payload["policy_architecture"] == "mlp"
                else {}
            ),
        },
        "feature_quality": feature_quality,
        "retention_diagnostics": retention_diagnostics,
        "training_reward_history": best_payload["training_reward_history"],
        "lagrangian_history": best_payload["lagrangian_history"],
        "training_diagnostics": best_payload["training_diagnostics"],
        "reward_weight_sweep": sweep_results,
        "encoding_results": encoding_results,
        "rsa_results": rsa_results,
        "family_shift": family_shift,
        "family_brain_alignment": family_brain_alignment,
        "behavior_results": behavior_results,
        "subject_results": subject_results,
    }
    result["setup_warnings"] = _setup_warnings(
        data_source=data_source,
        feature_quality=feature_quality,
        retention_diagnostics=retention_diagnostics,
        encoding_results=encoding_results,
        rsa_results=rsa_results,
    )

    if best_checkpoint_data["kind"] == "mlp":
        import torch

        torch.save(
            {
                "state_dict": best_checkpoint_data["state_dict"],
                "policy_architecture": "mlp",
                "feature_dim": best_payload["feature_dim"],
                "lagrangian_budget": best_payload["lagrangian_budget"],
                "policy_hidden_dim": config.rl.policy_hidden_dim,
                "policy_num_hidden_layers": config.rl.policy_num_hidden_layers,
            },
            out / POLICY_TORCH_FILENAME,
        )
    else:
        np.savez_compressed(
            out / POLICY_CHECKPOINT_FILENAME,
            theta=best_checkpoint_data["theta"],
            lagrangian_budget=np.float64(best_checkpoint_data["lagrangian_budget"]),
            feature_dim=np.int64(best_checkpoint_data["feature_dim"]),
        )

    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
