from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a_first_memory.config import PipelineConfig
from a_first_memory.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run memory-selection research pipeline")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--data-source", type=str, choices=("synthetic", "nsd"), default="nsd")
    parser.add_argument("--n-images", type=int, default=None)
    parser.add_argument("--n-units-per-family", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lambda-cost", type=float, default=None)
    parser.add_argument("--alpha-recog", type=float, default=None)
    parser.add_argument("--beta-sem", type=float, default=None)
    parser.add_argument("--gamma-perc", type=float, default=None)
    parser.add_argument("--delta-novelty", type=float, default=None)
    parser.add_argument("--eta-schema", type=float, default=None)
    parser.add_argument("--rl-algorithm", type=str, choices=("grpo", "reinforce"), default=None)
    parser.add_argument("--grpo-group-size", type=int, default=None)
    parser.add_argument("--grpo-ratio-clip-epsilon", type=float, default=None)
    parser.add_argument("--grpo-kl-coef", type=float, default=None)
    parser.add_argument("--grpo-ref-update-interval", type=int, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument(
        "--policy-architecture",
        type=str,
        choices=("linear", "mlp"),
        default=None,
        help="linear = legacy θ·x scorer; mlp = full neural policy (PyTorch, default in RLConfig).",
    )
    parser.add_argument("--policy-hidden-dim", type=int, default=None)
    parser.add_argument("--policy-num-hidden-layers", type=int, default=None)
    parser.add_argument("--nsd-npz-path", type=str, default=None)
    parser.add_argument("--nsd-dir", type=str, default=None)
    parser.add_argument("--nsd-layout-root", type=str, default=None)
    parser.add_argument("--nsd-feature-npz-path", type=str, default=None)
    parser.add_argument("--nsd-strict", action="store_true")
    parser.add_argument("--allow-synthetic-dev", action="store_true")
    parser.add_argument("--allow-local-nsd-debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig()
    if args.n_images is not None:
        cfg.data.n_images = args.n_images
    if args.n_units_per_family is not None:
        cfg.data.n_units_per_family = args.n_units_per_family
    if args.random_seed is not None:
        cfg.data.random_seed = args.random_seed
    if args.budget is not None:
        cfg.rl.budget = args.budget
    if args.epochs is not None:
        cfg.rl.epochs = args.epochs
    if args.learning_rate is not None:
        cfg.rl.learning_rate = args.learning_rate
    if args.lambda_cost is not None:
        cfg.rl.lambda_cost = args.lambda_cost
    if args.alpha_recog is not None:
        cfg.rl.alpha_recog = args.alpha_recog
    if args.beta_sem is not None:
        cfg.rl.beta_sem = args.beta_sem
    if args.gamma_perc is not None:
        cfg.rl.gamma_perc = args.gamma_perc
    if args.delta_novelty is not None:
        cfg.rl.delta_novelty = args.delta_novelty
    if args.eta_schema is not None:
        cfg.rl.eta_schema = args.eta_schema
    if args.rl_algorithm is not None:
        cfg.rl.algorithm = args.rl_algorithm
    if args.grpo_group_size is not None:
        cfg.rl.grpo_group_size = args.grpo_group_size
    if args.grpo_ratio_clip_epsilon is not None:
        cfg.rl.grpo_ratio_clip_epsilon = args.grpo_ratio_clip_epsilon
    if args.grpo_kl_coef is not None:
        cfg.rl.grpo_kl_coef = args.grpo_kl_coef
    if args.grpo_ref_update_interval is not None:
        cfg.rl.grpo_ref_update_interval = args.grpo_ref_update_interval
    if args.grad_clip_norm is not None:
        cfg.rl.grad_clip_norm = args.grad_clip_norm
    if args.policy_architecture is not None:
        cfg.rl.policy_architecture = args.policy_architecture
    if args.policy_hidden_dim is not None:
        cfg.rl.policy_hidden_dim = args.policy_hidden_dim
    if args.policy_num_hidden_layers is not None:
        cfg.rl.policy_num_hidden_layers = args.policy_num_hidden_layers

    # If a custom reward tuple is passed, run a single-point sweep with that tuple.
    if args.alpha_recog is not None or args.beta_sem is not None or args.gamma_perc is not None:
        cfg.rl.reward_weight_grid = ((cfg.rl.alpha_recog, cfg.rl.beta_sem, cfg.rl.gamma_perc),)
    if args.nsd_npz_path is not None:
        cfg.nsd.enabled = True
        cfg.nsd.npz_path = args.nsd_npz_path
    if args.nsd_dir is not None:
        cfg.nsd.enabled = True
        cfg.nsd.raw_dir = args.nsd_dir
    if args.nsd_layout_root is not None:
        cfg.nsd.enabled = True
        cfg.nsd.layout_root = args.nsd_layout_root
    if args.nsd_feature_npz_path is not None:
        cfg.nsd.enabled = True
        cfg.nsd.feature_npz_path = args.nsd_feature_npz_path
    if args.nsd_strict:
        cfg.nsd.enabled = True
        cfg.nsd.strict = True

    if args.data_source == "synthetic" and not args.allow_synthetic_dev:
        raise ValueError(
            "Synthetic mode is disabled for research runs. "
            "Use --data-source nsd with a real NSD payload, or pass --allow-synthetic-dev explicitly."
        )
    if args.data_source == "nsd" and not args.allow_local_nsd_debug:
        raise ValueError(
            "Local NSD execution is blocked for heavy runs. "
            "Use Modal for production runs, or pass --allow-local-nsd-debug for explicit local debugging."
        )
    if args.data_source == "nsd":
        if not (cfg.nsd.layout_root or cfg.nsd.npz_path or cfg.nsd.raw_dir):
            raise ValueError(
                "NSD mode requires one of --nsd-layout-root, --nsd-npz-path, or --nsd-dir."
            )

    result = run_pipeline(cfg, output_dir=args.output_dir, data_source=args.data_source)
    print(json.dumps(result["encoding_results"], indent=2))
    print(json.dumps(result["rsa_results"], indent=2))


if __name__ == "__main__":
    main()
