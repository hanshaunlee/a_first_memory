from __future__ import annotations

import json
import math
from pathlib import Path

import modal

app = modal.App("a-first-memory-train")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26",
        "scikit-learn>=1.5",
        "scipy>=1.13",
        "torch>=2.0",
        "wandb>=0.20",
    )
    .add_local_dir(".", remote_path="/root/project")
)
data_volume = modal.Volume.from_name("a-first-memory-data", create_if_missing=True)
out_volume = modal.Volume.from_name("a-first-memory-outputs", create_if_missing=True)


def _as_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _log_to_wandb(
    result: dict,
    *,
    data_source: str,
    nsd_source: str,
    nsd_path: str,
    output_subdir: str,
    run_config: dict,
    enabled: bool,
    project: str,
    entity: str,
    run_name: str,
    tags_csv: str,
    api_key: str,
) -> str | None:
    if not enabled:
        return None

    import wandb

    try:
        if api_key.strip():
            wandb.login(key=api_key.strip())
        tags = [t.strip() for t in tags_csv.split(",") if t.strip()]
        run = wandb.init(
            project=project,
            entity=entity or None,
            name=run_name or None,
            config=run_config,
            tags=tags,
        )
        try:
            reward_history = result.get("training_reward_history", []) or []
            lagrangian_history = result.get("lagrangian_history", []) or []
            diag = result.get("training_diagnostics", {}) or {}
            diag_series = {
                "train/clip_fraction": diag.get("clip_fraction_history", []),
                "train/approx_kl": diag.get("approx_kl_history", []),
                "train/grad_norm": diag.get("grad_norm_history", []),
                "train/reward_std": diag.get("epoch_reward_std_history", []),
                "train/adv_std": diag.get("epoch_adv_std_history", []),
            }
            max_steps = max(
                [len(reward_history), len(lagrangian_history)]
                + [len(v) for v in diag_series.values() if isinstance(v, list)],
                default=0,
            )
            for idx in range(max_steps):
                payload: dict[str, float | int] = {"epoch": idx + 1}
                if idx < len(reward_history):
                    v = _as_float(reward_history[idx])
                    if v is not None:
                        payload["train/reward"] = v
                if idx < len(lagrangian_history):
                    v = _as_float(lagrangian_history[idx])
                    if v is not None:
                        payload["train/lagrangian_budget"] = v
                for key, history in diag_series.items():
                    if idx < len(history):
                        v = _as_float(history[idx])
                        if v is not None:
                            payload[key] = v
                wandb.log(payload, step=idx + 1)

            util = result.get("retention_diagnostics", {})
            wandb.summary["run/output_subdir"] = output_subdir
            wandb.summary["run/data_source"] = data_source
            wandb.summary["run/nsd_source"] = nsd_source
            wandb.summary["run/nsd_path"] = nsd_path
            if isinstance(util, dict):
                for key, value in util.items():
                    v = _as_float(value)
                    if v is not None:
                        wandb.summary[f"retention/{key}"] = v
        finally:
            if run is not None:
                run.finish()
    except Exception as exc:
        return f"W&B logging skipped due to error: {exc}"
    return None


@app.function(
    image=image,
    volumes={
        "/data": data_volume,
        "/outputs": out_volume,
    },
    timeout=60 * 60 * 4,
)
def train_pipeline_remote(
    data_source: str = "nsd",
    nsd_source: str = "npz",
    nsd_path: str = "",
    nsd_feature_npz_path: str = "",
    nsd_strict: bool = True,
    n_images: int = 240,
    epochs: int = 50,
    budget: float = 32.0,
    random_seed: int = 7,
    rl_algorithm: str = "grpo",
    learning_rate: float = 0.035,
    lambda_cost: float = 0.08,
    alpha_recog: float = 1.0,
    beta_sem: float = 0.9,
    gamma_perc: float = 0.7,
    delta_novelty: float = 0.45,
    eta_schema: float = 0.35,
    grpo_group_size: int = 6,
    grpo_ratio_clip_epsilon: float = 0.2,
    grpo_kl_coef: float = 0.02,
    grpo_ref_update_interval: int = 2,
    grad_clip_norm: float = 5.0,
    policy_architecture: str = "mlp",
    policy_hidden_dim: int = 128,
    policy_num_hidden_layers: int = 2,
    output_subdir: str = "run",
    wandb_enabled: bool = False,
    wandb_project: str = "a-first-memory",
    wandb_entity: str = "",
    wandb_run_name: str = "",
    wandb_tags_csv: str = "",
    wandb_api_key: str = "",
) -> dict:
    import sys

    sys.path.insert(0, "/root/project/src")
    from a_first_memory.config import PipelineConfig
    from a_first_memory.pipeline import run_pipeline

    cfg = PipelineConfig()
    cfg.data.n_images = n_images
    cfg.data.random_seed = random_seed
    cfg.rl.epochs = epochs
    cfg.rl.budget = budget
    cfg.rl.algorithm = rl_algorithm
    cfg.rl.learning_rate = learning_rate
    cfg.rl.lambda_cost = lambda_cost
    cfg.rl.alpha_recog = alpha_recog
    cfg.rl.beta_sem = beta_sem
    cfg.rl.gamma_perc = gamma_perc
    cfg.rl.delta_novelty = delta_novelty
    cfg.rl.eta_schema = eta_schema
    cfg.rl.grpo_group_size = grpo_group_size
    cfg.rl.grpo_ratio_clip_epsilon = grpo_ratio_clip_epsilon
    cfg.rl.grpo_kl_coef = grpo_kl_coef
    cfg.rl.grpo_ref_update_interval = grpo_ref_update_interval
    cfg.rl.grad_clip_norm = grad_clip_norm
    cfg.rl.policy_architecture = policy_architecture
    cfg.rl.policy_hidden_dim = policy_hidden_dim
    cfg.rl.policy_num_hidden_layers = policy_num_hidden_layers
    cfg.rl.reward_weight_grid = ((cfg.rl.alpha_recog, cfg.rl.beta_sem, cfg.rl.gamma_perc),)
    if data_source == "nsd":
        if not nsd_path:
            raise ValueError("nsd_path is required when data_source='nsd'")
        nsd_abs = Path("/data") / nsd_path
        if not nsd_abs.exists():
            raise FileNotFoundError(f"NSD path not found in Modal data volume: {nsd_abs}")
        cfg.nsd.enabled = True
        cfg.nsd.strict = bool(nsd_strict)
        if nsd_source == "npz":
            if nsd_abs.is_dir():
                raise ValueError(f"Expected npz file, got directory: {nsd_abs}")
            cfg.nsd.npz_path = str(nsd_abs)
        elif nsd_source == "dir":
            if not nsd_abs.is_dir():
                raise ValueError(f"Expected directory payload, got file: {nsd_abs}")
            cfg.nsd.raw_dir = str(nsd_abs)
        elif nsd_source == "layout":
            if not nsd_abs.is_dir():
                raise ValueError(f"Expected layout root directory, got file: {nsd_abs}")
            cfg.nsd.layout_root = str(nsd_abs)
        else:
            raise ValueError(f"Unknown nsd_source: {nsd_source}")
        if nsd_feature_npz_path:
            feature_abs = Path("/data") / nsd_feature_npz_path
            if not feature_abs.exists():
                raise FileNotFoundError(f"Feature override npz not found: {feature_abs}")
            cfg.nsd.feature_npz_path = str(feature_abs)

    output_dir = Path("/outputs") / output_subdir
    result = run_pipeline(cfg, output_dir=str(output_dir), data_source=data_source)
    run_config = {
        "data_source": data_source,
        "nsd_source": nsd_source,
        "nsd_path": nsd_path,
        "nsd_feature_npz_path": nsd_feature_npz_path,
        "nsd_strict": bool(nsd_strict),
        "n_images": n_images,
        "epochs": epochs,
        "budget": budget,
        "random_seed": random_seed,
        "rl_algorithm": rl_algorithm,
        "learning_rate": learning_rate,
        "lambda_cost": lambda_cost,
        "alpha_recog": alpha_recog,
        "beta_sem": beta_sem,
        "gamma_perc": gamma_perc,
        "delta_novelty": delta_novelty,
        "eta_schema": eta_schema,
        "grpo_group_size": grpo_group_size,
        "grpo_ratio_clip_epsilon": grpo_ratio_clip_epsilon,
        "grpo_kl_coef": grpo_kl_coef,
        "grpo_ref_update_interval": grpo_ref_update_interval,
        "grad_clip_norm": grad_clip_norm,
        "policy_architecture": policy_architecture,
        "policy_hidden_dim": policy_hidden_dim,
        "policy_num_hidden_layers": policy_num_hidden_layers,
        "output_subdir": output_subdir,
    }
    wandb_warning = _log_to_wandb(
        result,
        data_source=data_source,
        nsd_source=nsd_source,
        nsd_path=nsd_path,
        output_subdir=output_subdir,
        run_config=run_config,
        enabled=bool(wandb_enabled),
        project=wandb_project,
        entity=wandb_entity,
        run_name=wandb_run_name,
        tags_csv=wandb_tags_csv,
        api_key=wandb_api_key,
    )
    if wandb_warning:
        result.setdefault("setup_warnings", [])
        result["setup_warnings"].append(wandb_warning)
    (output_dir / "modal_result_preview.json").write_text(
        json.dumps(
            {
                "data_source": result["data_source"],
                "selected_reward_weights": result["selected_reward_weights"],
                "retention_diagnostics": result.get("retention_diagnostics", {}),
                "setup_warnings": result.get("setup_warnings", []),
                "behavior_results": result["behavior_results"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    out_volume.commit()
    return {
        "output_dir": str(output_dir),
        "selected_reward_weights": result["selected_reward_weights"],
        "retention_diagnostics": result.get("retention_diagnostics", {}),
        "setup_warnings": result.get("setup_warnings", []),
        "behavior_results": result["behavior_results"],
    }


@app.local_entrypoint()
def main(
    data_source: str = "nsd",
    nsd_source: str = "npz",
    nsd_path: str = "",
    nsd_feature_npz_path: str = "",
    nsd_strict: bool = True,
    n_images: int = 240,
    epochs: int = 50,
    budget: float = 32.0,
    random_seed: int = 7,
    rl_algorithm: str = "grpo",
    learning_rate: float = 0.035,
    lambda_cost: float = 0.08,
    alpha_recog: float = 1.0,
    beta_sem: float = 0.9,
    gamma_perc: float = 0.7,
    delta_novelty: float = 0.45,
    eta_schema: float = 0.35,
    grpo_group_size: int = 6,
    grpo_ratio_clip_epsilon: float = 0.2,
    grpo_kl_coef: float = 0.02,
    grpo_ref_update_interval: int = 2,
    grad_clip_norm: float = 5.0,
    policy_architecture: str = "mlp",
    policy_hidden_dim: int = 128,
    policy_num_hidden_layers: int = 2,
    output_subdir: str = "run",
    wandb_enabled: bool = False,
    wandb_project: str = "a-first-memory",
    wandb_entity: str = "",
    wandb_run_name: str = "",
    wandb_tags_csv: str = "",
    wandb_api_key: str = "",
    detach_job: bool = False,
) -> None:
    kwargs = dict(
        data_source=data_source,
        nsd_source=nsd_source,
        nsd_path=nsd_path,
        nsd_feature_npz_path=nsd_feature_npz_path,
        nsd_strict=nsd_strict,
        n_images=n_images,
        epochs=epochs,
        budget=budget,
        random_seed=random_seed,
        rl_algorithm=rl_algorithm,
        learning_rate=learning_rate,
        lambda_cost=lambda_cost,
        alpha_recog=alpha_recog,
        beta_sem=beta_sem,
        gamma_perc=gamma_perc,
        delta_novelty=delta_novelty,
        eta_schema=eta_schema,
        grpo_group_size=grpo_group_size,
        grpo_ratio_clip_epsilon=grpo_ratio_clip_epsilon,
        grpo_kl_coef=grpo_kl_coef,
        grpo_ref_update_interval=grpo_ref_update_interval,
        grad_clip_norm=grad_clip_norm,
        policy_architecture=policy_architecture,
        policy_hidden_dim=policy_hidden_dim,
        policy_num_hidden_layers=policy_num_hidden_layers,
        output_subdir=output_subdir,
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        wandb_tags_csv=wandb_tags_csv,
        wandb_api_key=wandb_api_key,
    )
    if detach_job:
        call = train_pipeline_remote.spawn(**kwargs)
        print(
            json.dumps(
                {
                    "status": "spawned",
                    "function_call_id": getattr(call, "object_id", ""),
                    "output_subdir": output_subdir,
                },
                indent=2,
            )
        )
        return
    result = train_pipeline_remote.remote(**kwargs)
    print(json.dumps(result, indent=2))
