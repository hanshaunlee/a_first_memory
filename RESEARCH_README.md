# Research Runbook (Train, Test, Evaluate)

This document is the operational protocol for running this project as a research experiment with minimal ambiguity.

If you want a single mental model: **verify environment -> lock protocol -> run policy experiments -> run baselines -> compare against brain metrics -> decide what to adjust -> rerun with controlled changes only**.

## 1) What this pipeline is testing

The core hypothesis is that a **utility-constrained memory policy** explains exposure-dependent brain representational change better than non-RL compression or heuristics.

The project currently evaluates this with:

- **Encoding performance** via voxelwise banded ridge (`encoding_results`).
- **FR-RSA alignment** via feature-family-weighted RDM prediction (`rsa_results`).
- **Family-shift alignment** between policy retention shift and ROI family shift (`family_brain_alignment`).
- **Behavior fit** between policy-implied hit rates and observed hit rates (`behavior_results`).

## 2) Hard preflight checklist (run before any experiment campaign)

1. Install environment:
   - `pip install -e .`
   - `pip install pytest`
2. Verify GRPO trainer and invariants:
   - `python scripts/verify_grpo.py`
   - Note: this verifier uses synthetic mini-data only as an internal algorithm health check, not for scientific reporting.
3. Verify tests:
   - `python -m pytest tests/test_grpo_verifier.py`
4. Optional quick smoke:
   - `python scripts/run_pipeline.py --nsd-npz-path /path/to/nsd_payload.npz --n-images 24 --epochs 2 --output-dir outputs/smoke`

Do not start long runs until all four pass.

## 3) Training and evaluation command surface

Primary production entrypoint (Modal only):

- `modal run scripts/modal_app.py --data-source nsd --nsd-source npz --nsd-path nsd/nsd_payload.npz --epochs 50 --budget 32 --output-subdir <run_name>`

All major RL/GRPO tuning knobs are available on `scripts/modal_app.py`, so you do not need local execution for parameter sweeps.

Fail-closed behavior is enabled by default:

- default data source is `nsd` (not synthetic),
- strict NSD checks are enforced,
- synthetic mode is available for rapid smoke checks and debugging,
- NSD runs are enabled when you pass a valid NSD source path (`--nsd-npz-path`, `--nsd-dir`, or `--nsd-layout-root`).

Local `scripts/run_pipeline.py` is now considered debug-only and not a production path.

Important knobs (now exposed via CLI):

- **Data scale / reproducibility**
  - `--n-images`
  - `--n-units-per-family`
  - `--random-seed`
- **RL optimization**
  - `--rl-algorithm grpo|reinforce`
  - `--epochs`
  - `--learning-rate`
  - `--budget`
  - `--lambda-cost`
- **Reward mix**
  - `--alpha-recog`
  - `--beta-sem`
  - `--gamma-perc`
  - `--delta-novelty`
  - `--eta-schema`
- **GRPO trust-region stability**
  - `--grpo-group-size`
  - `--grpo-ratio-clip-epsilon`
  - `--grpo-kl-coef`
  - `--grpo-ref-update-interval`
  - `--grad-clip-norm`

If `--alpha-recog/--beta-sem/--gamma-perc` are passed, the run uses a single reward tuple for the sweep.

## 4) Recommended experiment schedule (research-ready order)

Run these in order; do not skip.

1. **Verifier + smoke**
   - Confirms training path, invariants, determinism.
2. **Core GRPO run**
   - Default config, full intended sample size.
3. **Algorithm control**
   - Same settings with `--rl-algorithm reinforce`.
4. **Reward-mixture ablations**
   - Semantics-heavy and perceptual-heavy variants.
5. **Budget sensitivity**
   - Lower and higher memory budgets.
6. **Seed replicates**
   - At least 3 seeds for robustness.

Store each run in a separate `outputs/<name>` folder and never overwrite.

## 5) Policy comparisons you should include

For end-of-study claims, compare at minimum:

- **Primary policy**: GRPO (`rl_policy`).
- **Algorithm fallback**: REINFORCE (`--rl-algorithm reinforce`).
- **Baselines already in pipeline output**:
  - `random`
  - `saliency`
  - `generic_compression` (PCA-like cost-prioritized keep)

These baselines are automatically produced in `encoding_results`, `rsa_results`, and `behavior_results`.

## 6) Targets and brain-facing analyses to compare

Use all of these; no single metric is sufficient.

- **Encoding target**
  - Compare mean and per-exposure `exposure_scores`.
  - Compare `family_drop_scores` to identify which feature families drive prediction.
- **FR-RSA target**
  - Compare `exposure_correlations` and ROI-level correlations.
  - Inspect `roi_family_weights` for feature-family weighting patterns.
- **Family-shift target**
  - Use `family_shift` and `family_brain_alignment` to test whether policy exposure shift matches ROI shift.
- **Behavior target**
  - Compare `behavior_results` (`spearman`, `mse`) to ensure policy is behaviorally plausible, not just neurally predictive.

## 7) How to interpret GRPO diagnostics

Check `training_diagnostics` in `results.json`:

- `clip_fraction_history`
  - Near `0` always: updates may be too conservative (or nearly on-policy).
  - Near `1` always: updates may be too aggressive.
- `approx_kl_history`
  - Should be non-negative and usually moderate.
  - Rapid growth suggests trust-region drift.
- `grad_norm_history`
  - If often near/above clip threshold, learning rate may be high.
- `epoch_reward_std_history`
  - Very low variance can indicate weak exploration signal.
- `epoch_adv_std_history`
  - Should usually be non-zero under grouped reward variation.

## 8) What to adjust after a run (decision rules)

Use these rules in order; change one axis at a time.

- **If budget utilization is low**
  - Increase `--learning-rate` slightly or reduce `--lambda-cost`.
  - Check whether `--budget` is unrealistically high for your unit-cost distribution.
- **If encoding/RSA underperform raw features**
  - Increase `--budget`.
  - Reduce over-regularization: lower `--grpo-kl-coef`.
  - Improve stability if noisy: reduce `--learning-rate`.
- **If policy is unstable (high clip fraction + high KL spikes)**
  - Lower `--learning-rate`.
  - Increase `--grpo-kl-coef`.
  - Decrease `--grpo-ref-update-interval` (refresh reference more frequently).
- **If learning is too slow / stagnant**
  - Increase `--grpo-group-size`.
  - Slightly relax KL (`--grpo-kl-coef` down).
  - Slightly increase learning rate.
- **If semantic vs perceptual claims are unclear**
  - Run two reward-targeted variants:
    - semantic-leaning: increase `--beta-sem`
    - perceptual-leaning: increase `--gamma-perc`
  - Compare resulting `family_shift` and ROI weights.

## 9) Minimum report template for final brain comparison

For each run include:

- run config (`config` block from `results.json`)
- selected reward weights (`selected_reward_weights`)
- mean encoding by strategy
- mean FR-RSA by strategy
- family-brain alignment summary
- behavior fit by strategy
- GRPO diagnostics summary (`clip_fraction`, `approx_kl`, `grad_norm`)
- setup warnings (`setup_warnings`)

Then provide a cross-run table (outside this repo if needed) with:

- run name
- algorithm
- budget
- reward tuple
- seed
- key metrics above

## 10) Suggested locked protocol before expensive NSD campaign

Before launching long or costly runs:

1. Freeze this repository revision.
2. Freeze one canonical command set for each planned condition.
3. Run `scripts/verify_grpo.py` and store output JSON.
4. Run one short smoke for each planned condition.
5. Only then launch full jobs.

This prevents hidden protocol drift and keeps post-hoc interpretation credible.
