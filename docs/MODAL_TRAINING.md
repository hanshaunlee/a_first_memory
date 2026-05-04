# Modal Training Guide

This project now includes a Modal app at `scripts/modal_app.py` for remote training runs.
For research safety, heavy NSD runs should be executed on Modal, not locally.

## 1) Install and authenticate Modal locally

- `python -m pip install modal`
- `modal setup`

Follow the browser flow once; this links your machine to your Modal workspace.

## 2) Validate NSD payload before upload (recommended)

Use the readiness checker to verify schema and print exact upload/run commands:

- `python scripts/check_nsd_modal_readiness.py --nsd-source npz --nsd-path /absolute/path/nsd_payload.npz --output-subdir nsd_full_run_01 --epochs 50 --budget 32`

Supported `--nsd-source` values:

- `npz`: single payload bundle
- `dir`: directory containing split arrays by key
- `layout`: canonical-ish NSD layout root (auto-discovery)

## 3) (Optional) Upload NSD payload to Modal Volume

The Modal app expects NSD payloads under the volume mounted at `/data`.

Example upload:

- `modal volume put a-first-memory-data /absolute/path/nsd_payload.npz nsd/nsd_payload.npz`

This creates `/data/nsd/nsd_payload.npz` inside the Modal function.

## 4) Run NSD training remotely

- NPZ payload:
  - `modal run scripts/modal_app.py --data-source nsd --nsd-source npz --nsd-path nsd/nsd_payload.npz --epochs 50 --budget 32 --output-subdir nsd_run_01`
- Split-array directory payload:
  - `modal run scripts/modal_app.py --data-source nsd --nsd-source dir --nsd-path nsd/nsd_payload_dir --epochs 50 --budget 32 --output-subdir nsd_run_01`
- Canonical layout root:
  - `modal run scripts/modal_app.py --data-source nsd --nsd-source layout --nsd-path nsd/nsd_layout_root --epochs 50 --budget 32 --output-subdir nsd_run_01`

Optional feature override:

- add `--nsd-feature-npz-path nsd/feature_bank.npz`

Common research tuning flags (all remote):

- `--rl-algorithm grpo|reinforce`
- `--learning-rate <float>`
- `--lambda-cost <float>`
- `--alpha-recog <float> --beta-sem <float> --gamma-perc <float>`
- `--delta-novelty <float> --eta-schema <float>`
- `--grpo-group-size <int>`
- `--grpo-ratio-clip-epsilon <float>`
- `--grpo-kl-coef <float>`
- `--grpo-ref-update-interval <int>`
- `--grad-clip-norm <float>`
- `--random-seed <int>`

Optional W&B tracking flags:

- `--wandb-enabled true`
- `--wandb-project a-first-memory`
- `--wandb-entity <your_team_or_user>`
- `--wandb-run-name nsd_grpo_run_01`
- `--wandb-tags-csv nsd,grpo,subj01`

If using private W&B workspaces, set `WANDB_API_KEY` in the runtime environment before launching `modal run`.

## 5) Retrieve outputs

Outputs are written to Modal volume `a-first-memory-outputs` at `/outputs/<output_subdir>`.

Download results:

- `modal volume get a-first-memory-outputs nsd_run_01 ./modal_outputs/nsd_run_01`

You should see `results.json` and `modal_result_preview.json`.

## Notes

- Script: `scripts/modal_app.py`
- Data volume: `a-first-memory-data`
- Output volume: `a-first-memory-outputs`
- For strict schema validation, `nsd_strict=True` is enabled by default in the Modal app.
- Change timeout/resources directly in the `@app.function(...)` decorator if needed.
- Local `scripts/run_pipeline.py` supports NSD directly when you provide one NSD source flag (`--nsd-npz-path`, `--nsd-dir`, or `--nsd-layout-root`).
