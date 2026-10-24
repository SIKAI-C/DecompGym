"""Scientific safeguards for joining snapshots and comparing cumulative curves."""
from copy import deepcopy
import hashlib

import pytest

from src.results import combine as figures
from src.results.campaign_report import paired_statistics


def fixture_report():
    cases = [{"suite": "bd", "problem": "ufl", "instance": f"ufl_small_seed00{i}", "size": "small", "seed": i, "sha256": {"data.json": f"data-{i}"}} for i in range(2)]
    manifest = {"instances": cases, "max_attempts": 3, "protocol_id": "fixture", "experiment_plan": {}, "controller_sha256": {name: "same" for name in ("src/agent/paired_runner.py", "src/agent/prompt_builder.py", "src/agent/output_verifier.py", "src/agent/native_solver.py")}}
    pairs = []
    for model in ("gpt-6-astra", "gemini-3.8-flash-medium"):
        for i, case in enumerate(cases):
            complete = model == "gpt-6-astra" or i == 0
            policies = {}
            for mode in figures.MODES:
                values = {str(k): {metric: k >= (1 if i == 0 else 2 if mode == "reflection" else 3) for metric in figures.METRICS} for k in range(1, 4)}
                policies[mode] = {"state": "complete" if complete else "pending", "audit_issues": [], "attempt_metrics": values}
            pairs.append({"model": model, **{k: case[k] for k in figures.CASE_FIELDS}, "state": "complete" if complete else "pending", "policies": policies})
    report = {"cohort_manifest": manifest, "configuration": {"max_attempts": 3}, "audit_errors": [], "snapshot": {"read_errors": []}, "input_root": "/fixture", "pairs": pairs, "paired_curves": paired_statistics(pairs, 3)}
    return report, manifest


def test_partial_gemini_is_separate_and_never_zero_imputed(monkeypatch):
    report, manifest = fixture_report()
    monkeypatch.setattr(figures, "validate_compatibility", lambda manifests: None)
    data = figures.combine_reports([report], [manifest])
    assert data["primary_models"] == ["gpt-6-astra"]
    partial = next(c for c in data["coverage"] if c["model"].startswith("gemini"))
    assert (partial["completed_pairs"], partial["planned_pairs"], partial["primary"]) == (1, 2, False)
    assert all(row["paired_cases"] == 1 and row["plain_rate"] == 1 for row in data["rows"] if row["model"].startswith("gemini"))
    assert next(r for r in data["rows"] if r["model"] == "gpt-6-astra" and r["dimension"] == "overall" and r["metric"] == "quality" and r["budget"] == 2)["reflection_minus_plain"] == 0.5


def test_same_count_with_different_case_identity_cannot_enter_primary(monkeypatch):
    report, manifest = fixture_report()
    monkeypatch.setattr(figures, "validate_compatibility", lambda manifests: None)
    report["pairs"][1]["instance"] = "different_case"
    with pytest.raises(ValueError, match="Model case inventory differs"):
        figures.combine_reports([report], [manifest])


@pytest.mark.parametrize("corruption", ["wrong_denominator", "wrong_count", "missing_point", "duplicate_point", "nonmonotone", "shared_first", "nesting", "unaudited"])
def test_corrupted_or_unaudited_curves_are_rejected(corruption):
    report, _ = fixture_report()
    values = report["pairs"][1]["policies"]["reflection"]["attempt_metrics"]
    if corruption == "wrong_denominator":
        report["paired_curves"][0]["paired_cases"] += 1
    elif corruption == "wrong_count":
        report["paired_curves"][0]["plain_successes"] += 1
    elif corruption == "missing_point":
        report["paired_curves"].pop()
    elif corruption == "duplicate_point":
        report["paired_curves"].append(deepcopy(report["paired_curves"][0]))
    elif corruption == "nonmonotone":
        values["3"] = {metric: False for metric in figures.METRICS}
    elif corruption == "shared_first":
        values["1"] = {metric: True for metric in figures.METRICS}
    elif corruption == "nesting":
        values["1"]["quality"] = True
    else:
        report["pairs"][0]["policies"]["plain"]["audit_issues"].append("source hash changed")
    with pytest.raises(ValueError):
        figures.validate_pairs_and_curves(report)


def test_join_rejects_different_budget_reference_or_prompt_source(monkeypatch):
    _, manifest = fixture_report()
    monkeypatch.setattr(figures, "run_identity", lambda m: {"paper_run": True})
    figures.validate_compatibility([manifest, deepcopy(manifest)])
    for field, value in (("max_attempts", 10), ("instances", []), ("controller_sha256", {}), ("experiment_plan", {"reference_policy": "changed"})):
        other = deepcopy(manifest)
        other[field] = value
        with pytest.raises(ValueError):
            figures.validate_compatibility([manifest, other])


def test_duplicate_model_cohorts_are_not_pooled(monkeypatch):
    report, manifest = fixture_report()
    monkeypatch.setattr(figures, "validate_compatibility", lambda manifests: None)
    with pytest.raises(ValueError, match="multiple cohorts"):
        figures.combine_reports([report, report], [manifest, manifest])


def test_style_registry_is_frozen_and_covers_exactly_the_paper_models():
    assert hashlib.sha256(figures.STYLE_PATH.read_bytes()).hexdigest() == "c186e848a57662e37b2df13b88aebe086354e296dd3c414137333c8ef8889fa4"
    assert list(figures.MODELS) == list(figures.PAPER_MODELS)
    assert len({s["color"] for s in figures.MODELS.values()}) == 12
    assert len({s["marker"] for s in figures.MODELS.values()}) == 12
    with pytest.raises(ValueError, match="No frozen style"):
        figures.model_style("another-model")


def test_models_outside_the_paper_are_dropped_before_combining_counts(monkeypatch):
    report, manifest = fixture_report()
    monkeypatch.setattr(figures, "validate_compatibility", lambda manifests: None)
    extra = deepcopy(report["pairs"][0])
    extra["model"] = "another-model"
    report["pairs"].append(extra)
    report["paired_curves"] = paired_statistics(report["pairs"], 3)
    data = figures.combine_reports([report], [manifest])
    assert all(r["model"] != "another-model" for key in ("rows", "pairs", "coverage") for r in data[key])
