"""The rerun freezes a fresh cohort and cannot silently resume legacy prompts."""
from copy import deepcopy
from pathlib import Path

import pytest

from src.agent.paired_runner import ProtocolError, identity, read, save
from src.campaign_plan import load_plan, model_queue
from src.dashboard_server import Dashboard
from src.native_campaign import FAMILIES, ensure_execution_allowed, initialize, validate
from src.reporting_policy import model_exclusion
from src import native_campaign, runtime_environment
from src.runtime_environment import RuntimeEnvironmentError, require_runtime_environment
from src.prompt_snapshot import PROMPT_PROTOCOL, judge_prompt_hashes


PLAN = Path(__file__).resolve().parents[1] / "docs/2026-10-13-strict-decomposition-experiment-plan.json"
FRESH_PLAN = PLAN.with_name("2026-10-13-fresh-all-models-plan.json")


def test_authorized_all_fresh_queue_includes_astra_and_changes_family_order():
    plan = load_plan(FRESH_PLAN)
    assert len(plan["model_queue"]) == 10
    assert [item["backend"] for item in plan["model_queue"]] == ["codex-cli"] * 6 + ["claude-code-cli"] * 2 + ["agy-cli"] * 2
    assert plan["model_queue"][0]["model"] == "gpt-6-astra"
    assert plan["historical_reference"] is None
    assert plan["reporting"]["fresh_execution_pairs"] == 810


@pytest.mark.parametrize("change", ["historical_astra", "old_order", "wrong_effort", "unapproved_model"])
def test_fresh_queue_rejects_mixed_configuration(tmp_path, change):
    plan = load_plan(FRESH_PLAN)
    if change == "historical_astra":
        plan["historical_reference"] = load_plan(PLAN)["historical_reference"]
    elif change == "old_order":
        plan["model_queue"][6], plan["model_queue"][8] = plan["model_queue"][8], plan["model_queue"][6]
    elif change == "wrong_effort":
        plan["model_queue"][6]["effort"] = "high"
    else:
        plan["model_queue"][0]["model"] = "unapproved"
    path = tmp_path / "bad-plan.json"
    save(path, plan)
    with pytest.raises(ProtocolError):
        load_plan(path)


@pytest.fixture(autouse=True)
def synthetic_execution_environment(monkeypatch):
    # Environment validation has its own tests; campaign layout tests need no local Docker daemon.
    monkeypatch.setattr(native_campaign, "load_execution_environment", lambda *_: {"image_id": "sha256:fixture", "native_environment": {}, "evidence_root": "fixture"})
    monkeypatch.setattr(native_campaign, "validate_execution_environment", lambda *_: None)


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
    assert (root / "controller/docker/agents/worker.py").exists()
    assert manifest["agent_execution"] == "docker"
    assert manifest["solver_execution"] == "native-macos-seatbelt"
    assert manifest["runtime_environment"] == require_runtime_environment()
    assert {"pyproject.toml", ".python-version"} <= manifest["controller_sha256"].keys()
    assert native_campaign.require_campaign_environment(manifest) == manifest["runtime_environment"]
    assert manifest["prompt_protocol"] == PROMPT_PROTOCOL
    assert manifest["algorithm_judge_prompt_sha256"] == judge_prompt_hashes()
    for case in manifest["instances"]:
        relative = Path(case["suite"]) / case["problem"] / case["instance"]
        original = data / relative / "prompt.txt"
        frozen = root / "dataset" / relative
        assert original.read_text() == "TASK"
        assert (frozen / "prompt.original.txt").read_bytes() == original.read_bytes()
        assert case["source_sha256"]["prompt.txt"] == case["prompt_snapshot"]["original_sha256"]
        assert case["sha256"]["prompt.txt"] == case["prompt_snapshot"]["effective_sha256"]
        assert (frozen / "prompt.txt").read_text() != "TASK"
        if case["problem"] == "ufl":
            assert "current directory" not in (frozen / "prompt.txt").read_text()
        if case["problem"] == "lrp":
            assert "1 means open and 0 means closed" in (frozen / "prompt.txt").read_text()
    assert read(root / "pause_request.json")["resume_requires_explicit_user_request"]
    with pytest.raises(ProtocolError, match="paused"):
        ensure_execution_allowed(root, manifest)
    dashboard = Dashboard(root).snapshot()
    assert [m["id"] for m in dashboard["models"]] == [item["model"] for item in queue if not model_exclusion(item["model"])]
    claude = next(m for m in dashboard["models"] if m["id"] == "claude-sonnet-5-5")
    assert claude["backend"] == "Claude Code" and claude["effort"] == "Medium"
    assert dashboard["totals"]["planned"] == 81 * (len(queue) - 1)
    assert dashboard["historical_reference"]["model"] == "gpt-6-astra"
    with pytest.raises(ProtocolError, match="existing experiments are preserved"):
        initialize(root, data, plan)


def test_wrong_runtime_stops_before_creating_campaign(tmp_path, monkeypatch):
    data, plan = fixture_plan(tmp_path)
    root = tmp_path / "new"
    monkeypatch.setattr(runtime_environment, "version", lambda _: "12.0.0")
    with pytest.raises(RuntimeEnvironmentError, match="Gurobi"):
        initialize(root, data, plan)
    assert not root.exists()


@pytest.mark.parametrize("command", ["run", "watch"])
def test_runtime_failure_has_nonretryable_exit_code(tmp_path, monkeypatch, command):
    def wrong_runtime(_, models=None):
        raise RuntimeEnvironmentError("version changed")
    monkeypatch.setattr(native_campaign, command, wrong_runtime)
    monkeypatch.setattr(native_campaign.sys, "argv", ["native_campaign", command, "--root", str(tmp_path), "--all-models"])
    with pytest.raises(SystemExit) as stopped:
        native_campaign.main()
    assert stopped.value.code == 64


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


def test_strict_controller_rejects_predecessor_prompt_plan():
    old = PLAN.with_name("2026-10-12-experiment-plan.json")
    with pytest.raises(ProtocolError, match="prompt and output contracts|problem, judge, and output contracts"):
        load_plan(old)
