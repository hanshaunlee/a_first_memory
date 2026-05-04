from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import wandb
from wandb.errors import CommError


def _as_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill a Modal run's results.json into W&B.")
    parser.add_argument("--results-json", required=True, help="Path to local results.json file.")
    parser.add_argument("--project", default="a-first-memory")
    parser.add_argument("--entity", default="")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--tags-csv", default="")
    parser.add_argument(
        "--wandb-api-key",
        default=os.environ.get("WANDB_API_KEY", ""),
        help="API key override; otherwise WANDB_API_KEY or ~/.netrc (see wandb.ai/authorize).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    key = (args.wandb_api_key or "").strip()
    if key:
        os.environ["WANDB_API_KEY"] = key

    results_path = Path(args.results_json)
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found: {results_path}")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    tags = [t.strip() for t in args.tags_csv.split(",") if t.strip()]

    try:
        run = wandb.init(
            project=args.project,
            entity=args.entity or None,
            name=args.run_name,
            tags=tags,
            config=payload.get("config", {}),
            resume="allow",
        )
    except CommError as exc:
        err = str(exc).lower()
        if "401" in err or "not logged in" in err or "permission" in err:
            print(
                "W&B rejected the credential (401). Typical fixes:\n"
                "  1) Paste a fresh key: wandb login --relogin  ( https://wandb.ai/authorize )\n"
                "  2) Fix ~/.netrc: machine api.wandb.ai, login user, password <single 40-char key>\n"
                "     (duplicate/wrong-length password lines cause this despite 'API key configured'.)\n"
                "  3) Retry with env override only for this shell:\n"
                "     export WANDB_API_KEY=... && python scripts/backfill_wandb_from_results.py ...\n",
                file=sys.stderr,
            )
        raise
    reward_history = payload.get("training_reward_history", []) or []
    lagrangian_history = payload.get("lagrangian_history", []) or []
    diag = payload.get("training_diagnostics", {}) or {}
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
        row: dict[str, float | int] = {"epoch": idx + 1}
        if idx < len(reward_history):
            v = _as_float(reward_history[idx])
            if v is not None:
                row["train/reward"] = v
        if idx < len(lagrangian_history):
            v = _as_float(lagrangian_history[idx])
            if v is not None:
                row["train/lagrangian_budget"] = v
        for key, history in diag_series.items():
            if idx < len(history):
                v = _as_float(history[idx])
                if v is not None:
                    row[key] = v
        wandb.log(row, step=idx + 1)

    for section_key in ("retention_diagnostics", "family_brain_alignment"):
        section = payload.get(section_key, {})
        if isinstance(section, dict):
            for k, v_raw in section.items():
                v = _as_float(v_raw)
                if v is not None:
                    wandb.summary[f"{section_key}/{k}"] = v
    wandb.summary["data_source"] = payload.get("data_source", "")
    wandb.summary["selected_reward_weights"] = payload.get("selected_reward_weights", {})
    wandb.summary["setup_warnings"] = payload.get("setup_warnings", [])
    run.finish()


if __name__ == "__main__":
    main()
