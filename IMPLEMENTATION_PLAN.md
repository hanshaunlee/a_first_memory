# Implementation Plan: Reward-Constrained Visual Memory Selection

## Goal

Translate the research proposal in `README.md` into a runnable, incremental implementation that can be executed now on synthetic data and later swapped to NSD + foundation-model features without redesigning the codebase.

## Critical Sharpenings Integrated

- Reward now includes novelty and schema-congruence terms in addition to recognition/semantic/perceptual utility.
- Policy is sequential and budget-aware (conditioning each decision on what is already retained) with decoupled forward/backward sampling temperatures.
- Brain alignment now uses FR-RSA with feature reweighting instead of only equal-weight RSA.
- Encoding now uses banded-ridge style family-specific regularization and per-family variance-drop analysis.
- Pipeline now includes behavioral hit-rate prediction tests and reward-mixture sweep as a scientific measurement, not just tuning.
- Real-data integration path is now executable via NSD payload ingestion (`.npz`) using a shared dataset contract.

## Scope Assumptions

- This repository currently has no source code, only a conceptual proposal.
- Full NSD download and large feature extraction are intentionally deferred to a "real-data integration" phase, but all interfaces are implemented now.
- The current implementation must prove the *workflow mechanics* end-to-end:
  - construct multi-family memory units,
  - train a budgeted RL selector with delayed utility reward,
  - compare against matched-budget baselines,
  - evaluate encoding and RSA metrics exposure-wise,
  - export reproducible outputs.

## Architectural Blueprint

### Package Layout

- `src/a_first_memory/config.py`
  - Centralized experiment configuration (`SyntheticDataConfig`, `RLConfig`, `EncodingConfig`, `PipelineConfig`).
- `src/a_first_memory/data/synthetic.py`
  - Generates NSD-like repeated exposure data with semantization trends.
- `src/a_first_memory/features/build.py`
  - Converts image units into raw/compressed feature matrices for modeling.
- `src/a_first_memory/rl/memory_policy.py`
  - GRPO-based budget-constrained memory selector.
- `src/a_first_memory/eval/encoding.py`
  - Exposure-aware voxelwise encoding benchmarking (`R^2` via ridge).
- `src/a_first_memory/eval/rsa.py`
  - Exposure-aware representational similarity evaluation (Spearman on RDMs).
- `src/a_first_memory/eval/baselines.py`
  - Random/saliency/generic compression controls under matched budget.
- `src/a_first_memory/eval/family_shift.py`
  - Retention-preference analysis across feature families.
- `src/a_first_memory/pipeline.py`
  - End-to-end orchestration + output artifact writing.
- `scripts/run_pipeline.py`
  - CLI to execute the full pipeline.
- `tests/`
  - Smoke tests for successful end-to-end execution.

## Phase-by-Phase Implementation

## Phase 1: Project Scaffolding and Reproducibility

### Deliverables

- Python packaging via `pyproject.toml`.
- Installable package in `src/`.
- Minimal dependency set:
  - `numpy`, `scikit-learn`, `scipy`
  - optional dev: `pytest`

### Success Criteria

- `pip install -e .` works.
- `python scripts/run_pipeline.py` can import package modules.

## Phase 2: Synthetic NSD-Like Data Engine

### Why

Enables immediate development and testing of memory-selection logic without blocking on 7T data ingestion and large feature extraction.

### Deliverables

- Multi-family candidate units per image:
  - semantic, object, geometry, low-level, patch.
- Per-unit storage costs.
- Repeated exposure responses (3 exposures/image) with lag buckets.
- ROI/voxel response simulator with exposure-dependent family weighting
  (semantic up, low-level down) to emulate semantization trend.

### Success Criteria

- Deterministic dataset generation with fixed seed.
- Correct tensor shapes:
  - units: `[n_images, n_units, embedding_dim]`
  - fMRI: `[n_images, 3, n_voxels]`

## Phase 3: Feature Construction Layer

### Deliverables

- Raw flattening strategy (`raw`).
- Family pooled strategy (`family_pool`).
- Policy-compressed strategy (`compressed`) using image-specific retention weights.
- Exposure-wise matrix builders for features and voxel targets.

### Success Criteria

- Matrix builders produce consistent row order and dimensions for regression/RSA.

## Phase 4: RL Memory Selection Core

### Task Formulation

- Action: retain/discard each unit.
- Constraint: hard budget by cumulative unit cost.
- Reward:
  - recognition utility,
  - semantic probe utility,
  - perceptual probe utility,
  - minus storage cost penalty.

### Deliverables

- Lightweight GRPO selector (`MemorySelectionPolicy`) with optional REINFORCE fallback.
- Unit context features include:
  - family identity,
  - cost,
  - embedding norm,
  - lag bucket,
  - repeat count.
- Budget enforcement post-sampling.
- Deterministic inference mode to produce retained units per image.

### Success Criteria

- Training reward history is finite and tracked across epochs.
- Learned retention policy obeys budget for all images.

## Phase 5: Baselines and Counterfactual Policies

### Deliverables

- Matched-budget random policy.
- Saliency-like heuristic policy.
- Generic compression proxy (cost-prioritized keep policy).

### Success Criteria

- Baseline retention matrices have same shape as RL retention.
- All baselines obey budget.

## Phase 6: Brain-Comparison Metrics

### Deliverables

- Exposure-specific encoding benchmark (ridge, held-out `R^2`).
- Exposure-specific RSA benchmark (feature RDM vs brain RDM Spearman).
- Family retention summary (fraction retained per feature family).

### Success Criteria

- All metrics computed for:
  - raw,
  - RL compressed,
  - random compressed,
  - saliency compressed,
  - generic compression.

## Phase 7: End-to-End Orchestration

### Deliverables

- `run_pipeline()` function:
  - builds dataset,
  - trains policy,
  - runs baseline comparisons,
  - computes encoding + RSA + family analysis,
  - writes `outputs/results.json`.
- CLI wrapper (`scripts/run_pipeline.py`) with tunable image count and epochs.

### Success Criteria

- Single command runs full workflow and emits structured artifacts.

## Phase 8: Verification and Guardrails

### Deliverables

- Smoke tests for:
  - dataset generation,
  - successful pipeline execution,
  - existence and parseability of `results.json`.
- Lint/type hygiene checks for edited files.

### Success Criteria

- `pytest` passes.
- No lints introduced in touched files (where diagnostics are available).

## Phase 9: Real NSD Integration (Next Build Step)

### Replace Synthetic Data with Real Inputs

- Add an NSD data loader module for:
  - trial betas,
  - repeat/exposure metadata,
  - ROI definitions.
- Introduce feature extraction adapters:
  - DINOv2,
  - SigLIP/CLIP,
  - SAM slot pooling,
  - depth model outputs,
  - low-level handcrafted channels.

### Keep Existing Interfaces

- Preserve `SyntheticDataset`-like contract with a generic dataset protocol.
- Keep policy, encoding, and RSA code unchanged except for data adapter wiring.

### Additional Evaluation Targets

- ROI-specific exposure trajectory plots.
- Spacing effect stratification by lag bins.
- Probe-mixture sensitivity sweeps (semantic/perceptual reward weighting).

## Step-by-Step Execution Log (Implemented)

1. Initialized package structure and dependency manifest.
2. Created typed config layer for data, RL, and encoding phases.
3. Implemented synthetic repeated-exposure dataset generator with semantization dynamics.
4. Implemented feature builders for raw and policy-compressed representations.
5. Implemented RL memory selector with budget constraint and delayed utility reward.
6. Implemented matched-budget baselines (random, saliency-like, generic compression).
7. Implemented encoding and RSA evaluations exposure-wise.
8. Implemented retention-family analysis for interpretability.
9. Wired an orchestrated pipeline + CLI entrypoint.
10. Added testing hooks and execution docs.

## Exit Criteria for "Version 1 Complete"

- End-to-end run succeeds with default settings.
- Output artifact includes training curve + all comparison metrics.
- Codebase is ready to accept NSD and real feature banks without architectural rewrite.

## Version 2 Completion Addendum

- Added NSD payload loader at `src/a_first_memory/data/nsd.py`.
- Added raw directory NSD payload ingestion (`--nsd-dir`) in addition to `.npz`.
- Added layout-root NSD ingestion (`--nsd-layout-root`) with common path auto-discovery.
- Added feature-bank builder pipeline from per-family embedding blocks into unified unit bank.
- Added subject-wise evaluation outputs (encoding, FR-RSA, behavior) with minimum-sample safeguards.
- Added Modal cloud training integration (`scripts/modal_app.py`) and runbook (`docs/MODAL_TRAINING.md`).
- Added frozen probe pretraining step before policy training.
- Unified synthetic + NSD execution in `run_pipeline(..., data_source=...)`.
- Added CLI support for NSD mode in `scripts/run_pipeline.py`.
- Added schema documentation for NSD payloads in `docs/NSD_PAYLOAD_FORMAT.md`.
