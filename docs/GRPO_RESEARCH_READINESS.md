# GRPO Research Readiness Notes

This project's GRPO implementation is a compact, task-specific adaptation for memory-selection RL. The goal is not to replicate LLM RLHF exactly, but to adopt the stability patterns that transfer well to this setting.

## Stability patterns adopted

- Group-relative advantages: normalize rewards within each sampled group to avoid a learned critic.
- Ratio clipping: clip update ratios around 1.0 with epsilon trust region behavior.
- KL anchoring: keep updates close to a reference policy and refresh that reference periodically.
- Gradient clipping: bound update magnitude to reduce occasional unstable steps.
- Deterministic verifier: enforce budget/shape/binary and diagnostics invariants on every run.

## Why these were prioritized

Across recent GRPO/RPG discussions, the recurring failure modes are: oversized policy steps, weak trust region control, and unstable off-policy-like updates. The above controls directly target those failure modes while staying lightweight for this synthetic+NSD pipeline.

## References used

- [On the Design of KL-Regularized Policy Gradient Algorithms](https://arxiv.org/pdf/2505.17508)
- [RPG project page (KL-regularized policy gradient design)](https://complex-reasoning.github.io/RPG/)
- [Generalized ratio-clipping policy optimization overview](https://www.emergentmind.com/topics/generalized-ratio-clipping-policy-optimization-grpo)
- [Illustrated GRPO explainer](https://abderrahmanskiredj.github.io/the-illustrated-grpo/)

## Practical caveat

Most GRPO literature is LLM-centered. In this repository, the policy is a sequential selector over memory units, so ratio/kl terms are implemented as lightweight surrogates using context-logit differences rather than token-level exact log-prob trajectories. This is intentional for computational simplicity and reproducibility in the current codebase.
