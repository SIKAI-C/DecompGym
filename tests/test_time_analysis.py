"""Tests for resource attribution, censoring, and audited snapshot boundaries."""
from copy import deepcopy
import json

import pytest

from src.results import time_analysis as timing


@pytest.fixture(autouse=True)
def synthetic_paper_run(monkeypatch):
    monkeypatch.setattr(timing, "run_identity", lambda manifest: {"paper_run": True})
    monkeypatch.setattr(timing, "PAPER_MODELS", ("model-a",))


def fixture_report():
    case = {"suite": "bd", "problem": "ufl", "instance": "ufl_small_seed000", "size": "small", "seed": 0}
    identity = {"model": "model-a", **case}
    calls = []
    for identifier, mode, attempt, stage, seconds in (("g1", "shared", 1, "generation", 10), ("j1", "shared", 1, "algorithm_verification", 2), ("pg2", "plain", 2, "generation", 20), ("pj2", "plain", 2, "algorithm_verification", 3), ("r1", "reflection", 1, "reflection", 5), ("rg2", "reflection", 2, "generation", 15), ("rj2", "reflection", 2, "algorithm_verification", 4)):
        calls.append({**identity, "invocation_id": identifier, "mode": mode, "attempt": attempt, "stage": stage, "classification": "shared_first" if mode == "shared" else "policy", "physical_call_confirmed": True, "elapsed_seconds": seconds})
    solvers = [{**identity, "solver_id": identifier, "mode": mode, "attempt": attempt, "execution_seconds": seconds} for identifier, mode, attempt, seconds in (("s1", "shared", 1, 1), ("ps2", "plain", 2, 2), ("rs2", "reflection", 2, 3))]
    policies = {}
    for mode in timing.MODES:
        policies[mode] = {"state": "complete", "audit_issues": [], "actual_attempts": 2, "reused_slots": 1, "unexecuted_attempt_slots": 0, "call_ids": [call["invocation_id"] for call in calls if call["mode"] in ("shared", mode)], "solver_ids": [solver["solver_id"] for solver in solvers if solver["mode"] in ("shared", mode)], "attempt_metrics": {str(k): {metric: k >= 2 for metric in ("execution", "verification", "quality")} for k in range(1, 4)}}
    pair = {**identity, "state": "complete", "policies": policies}
    return {"cohort_manifest": {"protocol_id": "fixture", "instances": [case], "max_attempts": 3}, "configuration": {"protocol_id": "fixture", "max_attempts": 3}, "snapshot": {"read_errors": [], "changed_during_read": []}, "audit_errors": [], "input_root": "/fixture/original", "pairs": [pair], "calls": calls, "solver_executions": solvers}


def select(rows, **criteria):
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    assert len(matches) == 1
    return matches[0]


def test_reflection_buys_next_candidate_and_early_stop_has_no_new_work():
    data = timing.build_time_analysis([fixture_report()])
    first = select(data["budget_records"], mode="reflection", budget=1)
    second = select(data["budget_records"], mode="reflection", budget=2)
    padded = select(data["budget_records"], mode="reflection", budget=3)
    assert (first["generation_seconds"], first["reflection_seconds"], first["agent_seconds"]) == (10, 0, 10)
    assert (second["generation_seconds"], second["reflection_seconds"], second["agent_seconds"]) == (25, 5, 30)
    assert (second["judge_seconds"], second["solver_seconds"], second["active_work_seconds"]) == (6, 4, 40)
    assert all(second[measure] == padded[measure] for measure in timing.TIME_MEASURES)
    final = select(data["case_policies"], mode="reflection")
    assert (final["first_quality_attempt"], final["time_to_quality_agent_seconds"], final["time_to_quality_active_work_seconds"]) == (2, 30, 40)
    assert all(row["passed"] for row in data["accounting_checks"])


def test_shared_work_is_in_each_policy_but_once_physically_with_extras_separate():
    report = fixture_report()
    report["calls"].append(deepcopy(report["calls"][0]))
    report["solver_executions"].append(deepcopy(report["solver_executions"][0]))
    report["calls"].extend([{**report["calls"][0], "invocation_id": "rejected", "mode": "plain", "classification": "infrastructure_or_rejected", "elapsed_seconds": 99}, {**report["calls"][0], "invocation_id": "unconfirmed", "mode": "plain", "classification": "request_start_unconfirmed", "physical_call_confirmed": False, "elapsed_seconds": None}])
    data = timing.build_time_analysis([report])
    policies = data["case_policies"]
    assert sum(row["agent_seconds"] for row in policies) == 60
    physical = select(data["physical_summary"], model="__all__", category="all_recorded", stage="all")
    committed = select(data["physical_summary"], model="__all__", category="committed_policy_path", stage="all")
    extra = select(data["physical_summary"], model="__all__", category="additional", stage="all")
    assert (physical["physical_calls"], physical["unconfirmed_requests"], physical["solver_executions"]) == (8, 1, 3)
    assert (committed["seconds"], extra["seconds"], physical["seconds"]) == (65, 99, 164)
    assert (committed["physical_calls"], committed["solver_executions"]) == (7, 3)
    assert sum(row["active_work_seconds"] for row in policies) - 13 == committed["seconds"]


@pytest.mark.parametrize("bad_duration", [None, -1, float("nan"), float("inf"), "15", True])
def test_missing_or_invalid_duration_is_null_and_retains_observed_sum(bad_duration):
    report = fixture_report()
    next(call for call in report["calls"] if call["invocation_id"] == "rg2")["elapsed_seconds"] = bad_duration
    data = timing.build_time_analysis([report], agent_budgets=[10, 30])
    reflection = select(data["case_policies"], mode="reflection")
    assert reflection["generation_seconds"] is None
    assert reflection["agent_seconds"] is None
    assert reflection["active_work_seconds"] is None
    assert reflection["agent_observed_seconds"] == 15
    assert reflection["agent_missing_durations"] == 1
    assert reflection["solver_seconds"] == 4
    assert select(data["case_policies"], mode="plain")["reflection_seconds"] == 0
    curve = select(data["resource_curves"], mode="reflection", dimension="overall", agent_budget_seconds=30)
    assert curve["quality_rate"] is None
    assert (curve["unlocated_quality_successes"], curve["quality_rate_lower_bound"], curve["quality_rate_upper_bound"]) == (1, 0, 1)
    json.dumps(data, allow_nan=False)


def test_unsolved_cases_remain_in_resource_denominator_and_discordant_pair_summary():
    report = fixture_report()
    policy = report["pairs"][0]["policies"]["plain"]
    for metric in policy["attempt_metrics"].values():
        metric["quality"] = False
    data = timing.build_time_analysis([report], agent_budgets=[29, 30, 10000])
    plain = select(data["case_policies"], mode="plain")
    assert plain["quality"] is False
    assert plain["time_to_quality_agent_seconds"] is None
    assert plain["first_quality_attempt"] is None
    assert select(data["resource_curves"], mode="plain", dimension="overall", agent_budget_seconds=10000)["quality_rate"] == 0
    assert select(data["resource_curves"], mode="reflection", dimension="overall", agent_budget_seconds=29)["quality_rate"] == 0
    assert select(data["resource_curves"], mode="reflection", dimension="overall", agent_budget_seconds=30)["quality_rate"] == 1
    paired = select(data["paired_summary"], dimension="overall", measure="active_work_seconds", success_status="all")
    assert (paired["reflection_only_successes"], paired["plain_only_successes"], paired["mean"]) == (1, 0, 2)
    quality_times = select(data["paired_summary"], dimension="overall", measure="time_to_quality_agent_seconds", success_status="all")
    assert (quality_times["paired_cases"], quality_times["available_pairs"], quality_times["mean"]) == (0, 0, None)


def test_terminal_reflection_cost_is_final_expenditure_without_fictitious_candidate():
    report = fixture_report()
    policy = report["pairs"][0]["policies"]["reflection"]
    for metric in policy["attempt_metrics"].values():
        metric["quality"] = False
    report["configuration"]["max_attempts"] = 2
    report["cohort_manifest"]["max_attempts"] = 2
    for item in report["pairs"][0]["policies"].values():
        item["attempt_metrics"].pop("3")
    report["calls"].append({**report["calls"][4], "invocation_id": "terminal_reflection", "attempt": 2, "elapsed_seconds": 7})
    policy["call_ids"].append("terminal_reflection")
    data = timing.build_time_analysis([report])
    assert select(data["case_policies"], mode="reflection")["agent_seconds"] == 37
    assert select(data["budget_records"], mode="reflection", budget=2)["agent_seconds"] == 30
    assert select(data["case_policies"], mode="reflection")["time_to_quality_agent_seconds"] is None


def test_partial_model_excluded_from_primary_but_complete_pairs_and_physical_work_remain():
    report = fixture_report()
    second_case = {**report["cohort_manifest"]["instances"][0], "instance": "ufl_small_seed001", "seed": 1}
    report["cohort_manifest"]["instances"].append(second_case)
    pending = deepcopy(report["pairs"][0])
    pending.update(second_case, state="pending")
    for policy in pending["policies"].values():
        policy.update(state="pending", actual_attempts=0, call_ids=[], solver_ids=[])
    report["pairs"].append(pending)
    report["calls"].append({**report["calls"][0], **second_case, "invocation_id": "partial_reject", "classification": "infrastructure_or_rejected", "elapsed_seconds": 99})
    data = timing.build_time_analysis([report])
    assert data["metadata"]["primary_models"] == []
    assert (data["coverage"][0]["completed_pairs"], data["coverage"][0]["planned_pairs"]) == (1, 2)
    assert len(data["case_policies"]) == 2
    assert all(not row["primary_model"] for row in data["case_policies"])
    assert select(data["physical_summary"], model="model-a", category="additional", stage="all")["seconds"] == 99


def test_unscored_solver_execution_is_physical_extra_not_policy_cost():
    report = fixture_report()
    report["solver_executions"].append({**report["solver_executions"][1], "solver_id": "unscored_solver", "attempt": 3, "execution_seconds": 100})
    report["pairs"][0]["policies"]["plain"]["solver_ids"].append("unscored_solver")
    data = timing.build_time_analysis([report])
    assert select(data["case_policies"], mode="plain")["solver_seconds"] == 3
    assert select(data["physical_summary"], model="model-a", category="uncommitted_solver", stage="solver")["seconds"] == 100


def test_model_outside_the_selection_never_enters_any_cost_or_denominator():
    report = fixture_report()
    excluded = deepcopy(report["pairs"][0])
    excluded["model"] = "other-model"
    report["pairs"].append(excluded)
    report["calls"].append({**report["calls"][0], "model": "other-model", "invocation_id": "excluded_call", "elapsed_seconds": 100000})
    report["solver_executions"].append({**report["solver_executions"][0], "model": "other-model", "solver_id": "excluded_solver", "execution_seconds": 100000})
    data = timing.build_time_analysis([report])
    assert len(data["coverage"]) == 1
    assert select(data["physical_summary"], model="__all__", category="all_recorded", stage="all")["seconds"] == 65


@pytest.mark.parametrize("corruption", ["audit", "changed", "missing_call", "duplicate_call", "different_shared", "unaudited_policy", "nonmonotone", "wrong_case"])
def test_corrupt_snapshot_or_policy_attribution_is_rejected(corruption):
    report = fixture_report()
    if corruption == "audit":
        report["audit_errors"] = ["score mismatch"]
    elif corruption == "changed":
        report["snapshot"]["changed_during_read"] = ["changed.json"]
    elif corruption == "missing_call":
        report["calls"].pop(0)
    elif corruption == "duplicate_call":
        report["calls"].append({**report["calls"][0], "elapsed_seconds": 1000})
    elif corruption == "different_shared":
        report["pairs"][0]["policies"]["reflection"]["call_ids"].remove("j1")
    elif corruption == "unaudited_policy":
        report["pairs"][0]["policies"]["reflection"]["audit_issues"] = ["changed score"]
    elif corruption == "nonmonotone":
        report["pairs"][0]["policies"]["reflection"]["attempt_metrics"]["3"]["quality"] = False
    else:
        report["calls"][0]["instance"] = "different"
    with pytest.raises(ValueError):
        timing.build_time_analysis([report])


def test_export_keeps_protocol_and_root(tmp_path):
    data = timing.build_time_analysis([fixture_report()])
    timing.write_analysis(data, tmp_path / "out")
    loaded = json.loads((tmp_path / "out" / "time_analysis.json").read_text())
    assert loaded["metadata"]["sources"][0]["cohort_protocol"] == "fixture"
    assert all(row["cohort_protocol"] == "fixture" and row["input_root"] == "/fixture/original" for row in loaded["budget_records"])
    assert (tmp_path / "out" / "case_policies.csv").is_file()
