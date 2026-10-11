"""The rerun freezes a fresh cohort and cannot silently resume legacy prompts."""
from copy import deepcopy
from pathlib import Path

import pytest

from src.agent.paired_runner import ProtocolError, identity, read, save
from src.campaign_plan import load_plan, model_queue
from src.dashboard_server import Dashboard
from src.native_campaign import FAMILIES, ensure_execution_allowed, initialize, validate
from src.reporting_policy import model_exclusion


PLAN = Path(__file__).resolve().parents[1] / "docs/2026-10-12-experiment-plan.json"


def fixture_plan(tmp_path):
    plan = load_plan(PLAN)
    data = tmp_path / "input"
    cases = []
    from src.agent.paired_runner import digest
    for size in ("small", "medium", "large"):
        for seed in range(3):
            for problem in FAMILIES:
                suite = "bd" if problem in ("ufl", "fctp", "km", "lrp", "tsso") else "cg"
                name = f"{problem}_{size}_seed{seed:03d}"
                directory = data / suite / problem / name
                directory.mkdir(parents=True)
                for filename, contents in (("data.json", "{}"), ("prompt.txt", "TASK"), ("meta.json", "{}"), ("gold_monolithic.json", "{}")):
                    (directory / filename).write_text(contents)
                cases.append({"suite": suite, "problem": problem, "instance": name, "size": size, "seed": seed,
                              "sha256": {p.name: digest(p) for p in directory.iterdir()}})
    plan["dataset_inventory_sha256"] = identity(cases)
    path = tmp_path / "plan.json"
    save(path, plan)
    return data, path


def test_reviewed_queue_and_historical_reference_are_separate():
    plan = load_plan(PLAN)
    assert [s["model"] for s in plan["model_queue"]] == ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol", "gemini-3.8-flash-medium", "gemini-3.1-pro-high", "claude-opus-5-5", "claude-sonnet-5-5"]
    assert plan["historical_reference"]["model"] == "gpt-6-astra"
    assert not plan["historical_reference"]["fresh_cohort_member"]
    assert len(plan["optional_queue"]) == 1 and model_exclusion("gpt-5.6-sol")


@pytest.mark.parametrize("optional", [False, True])
def test_initialization_freezes_exact_inputs_and_stays_paused(tmp_path, optional):
    data, plan = fixture_plan(tmp_path)
    root = tmp_path / "new-campaign"
    initialize(root, data, plan, include_optional=optional)
    manifest = validate(root)
    queue = model_queue(manifest)
    assert len(queue) == 9 + int(optional)
    assert manifest["fresh_execution_pairs"] == 81 * len(queue)
    assert manifest["maximum_unique_generated_candidates"] == 81 * len(queue) * 19
    assert not (root / "runs").exists()
    assert (root / "controller/src/agent/claude_client.py").exists()
    assert read(root / "pause_request.json")["resume_requires_explicit_user_request"]
    with pytest.raises(ProtocolError, match="paused"):
        ensure_execution_allowed(root, manifest)
    dashboard = Dashboard(root).snapshot()
    assert [m["id"] for m in dashboard["models"]] == [item["model"] for item in queue if not model_exclusion(item["model"])]
    claude = next(m for m in dashboard["models"] if m["id"] == "claude-sonnet-5-5")
    assert claude["backend"] == "Claude Code" and claude["effort"] == "High"
    assert dashboard["totals"]["planned"] == 81 * (len(queue) - 1)
    assert dashboard["historical_reference"]["model"] == "gpt-6-astra"
    with pytest.raises(ProtocolError, match="existing experiments are preserved"):
        initialize(root, data, plan)


def test_mismatched_inputs_fail_before_creating_a_campaign(tmp_path):
    data, plan = fixture_plan(tmp_path)
    next(data.glob("*/*/*/data.json")).write_text('{"changed":true}')
    root = tmp_path / "new"
    with pytest.raises(ProtocolError, match="inventory"):
        initialize(root, data, plan)
    assert not root.exists()


def test_legacy_campaign_cannot_execute_with_new_prompts(tmp_path):
    with pytest.raises(ProtocolError, match="fresh cohort"):
        ensure_execution_allowed(tmp_path, {"protocol_id": "old"})


@pytest.mark.parametrize("field,value", [("workers", 3), ("shared_first_attempt", False), ("solver_timeout", 120)])
def test_plan_rejects_silent_protocol_changes(tmp_path, field, value):
    plan = deepcopy(load_plan(PLAN))
    plan["execution"][field] = value
    path = tmp_path / "bad-plan.json"
    save(path, plan)
    with pytest.raises(ProtocolError):
        load_plan(path)
