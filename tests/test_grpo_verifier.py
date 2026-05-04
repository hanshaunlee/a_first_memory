from __future__ import annotations

from scripts.verify_grpo import run_verifier


def test_grpo_verifier_report_structure() -> None:
    report = run_verifier()
    checks = report["checks"]
    assert checks["default_algorithm"] == "passed"
    assert checks["grpo_core"]["status"] == "passed"
    assert checks["grpo_core"]["clip_fraction_mean"] >= 0.0
    assert checks["grpo_core"]["approx_kl_mean"] >= 0.0
    assert checks["grpo_determinism"] == "passed"
    assert checks["reinforce_fallback"]["status"] == "passed"
    assert checks["unknown_algorithm_guard"] == "passed"


def test_grpo_verifier_budget_enforcement() -> None:
    report = run_verifier()
    grpo_budget = report["checks"]["grpo_core"]["budget_stats"]
    reinforce_budget = report["checks"]["reinforce_fallback"]["budget_stats"]
    budget = report["config_snapshot"]["budget"]
    assert grpo_budget["max_cost"] <= budget + 1e-6
    assert reinforce_budget["max_cost"] <= budget + 1e-6
