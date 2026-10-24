"""The paper config freezes a fresh campaign that stays paused and cannot silently change its protocol."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.agent.prompts.snapshot import PROMPT_PROTOCOL, judge_prompt_hashes
from src.experiments import campaign, environment
from src.experiments.campaign import FAMILIES, ensure_execution_allowed, initialize, resume, validate
from src.experiments.dashboard_server import Dashboard
from src.experiments.environment import RuntimeEnvironmentError, require_runtime_environment
from src.experiments.plan import load_plan, model_queue
from src.results.policy import PAPER_MODELS
from src.utils.errors import ProtocolError
from src.utils.records import digest, identity, read, save


CONFIG = Path(__file__).resolve().parents[1] / "configs/paper.json"


def test_paper_config_lists_the_twelve_paper_models_in_table_order():
    plan = load_plan(CONFIG)
    assert [item["model"] for item in plan["model_queue"]] == list(PAPER_MODELS)
    assert {item["backend"] for item in plan["model_queue"]} == {"codex-cli", "claude-code-cli", "agy-cli"}
    assert all(item["effort"] == "medium" for item in plan["model_queue"] if item["backend"] != "agy-cli")
    assert plan["execution"]["judge_model"] == "gpt-6-astra" and plan["execution"]["judge_effort"] == "high"


def test_paper_config_pins_the_regenerated_data_directory():
    data = Path(__file__).resolve().parents[1] / "data"
    cases = []
    for path in data.glob("*/*/*/data.json"):
        suite, problem, name = path.relative_to(data).parts[:3]
        _, size, seed = name.rsplit("_", 2)
        cases.append({"suite": suite, "problem": problem, "instance": name, "size": size, "seed": int(seed[4:]),
                      "sha256": {f: digest(path.parent / f) for f in ("data.json", "prompt.txt", "meta.json", "gold_monolithic.json")}})
    cases.sort(key=lambda c: (["small", "medium", "large"].index(c["size"]), c["seed"], FAMILIES.index(c["problem"])))
    assert identity(cases) == load_plan(CONFIG)["dataset_inventory_sha256"]


@pytest.mark.parametrize("change", ["duplicate", "unknown_backend", "missing_effort", "old_schema", "prompt_protocol"])
def test_config_rejects_malformed_queues_and_policies(tmp_path, change):
    plan = load_plan(CONFIG)
    if change == "duplicate":
        plan["model_queue"].append(plan["model_queue"][0])
    elif change == "unknown_backend":
        plan["model_queue"][0]["backend"] = "other-cli"
    elif change == "missing_effort":
        plan["model_queue"][0].pop("effort")
    elif change == "old_schema":
        plan["schema_version"] = 1
    else:
        plan["prompt_protocol"] = dict(plan["prompt_protocol"], problem_templates="older")
    path = tmp_path / "bad-config.json"
    save(path, plan)
    with pytest.raises(ProtocolError):
        load_plan(path)


@pytest.mark.parametrize("field,value", [("workers", 3), ("shared_first_attempt", False), ("solver_timeout", 120)])
def test_config_rejects_silent_protocol_changes(tmp_path, field, value):
    plan = deepcopy(load_plan(CONFIG))
    plan["execution"][field] = value
    path = tmp_path / "bad-config.json"
    save(path, plan)
    with pytest.raises(ProtocolError):
        load_plan(path)


@pytest.fixture(autouse=True)
def synthetic_execution_environment(monkeypatch):
    # Environment validation has its own tests; campaign layout tests need no local Docker daemon.
    monkeypatch.setattr(campaign, "load_execution_environment", lambda *_: {"image_id": "sha256:fixture", "native_environment": {}, "evidence_root": "fixture"})
    monkeypatch.setattr(campaign, "validate_execution_environment", lambda *_: None)


def fixture_config(tmp_path):
    plan = load_plan(CONFIG)
    data = tmp_path / "input"
    cases = []
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
    path = tmp_path / "config.json"
    save(path, plan)
    return data, path


def test_initialization_freezes_exact_inputs_and_stays_paused(tmp_path):
    data, config = fixture_config(tmp_path)
    root = tmp_path / "new-campaign"
    initialize(root, data, config, tmp_path / "runtime.json")
    manifest = validate(root)
    queue = model_queue(manifest)
    assert [item["model"] for item in queue] == list(PAPER_MODELS)
    assert manifest["fresh_execution_pairs"] == 81 * len(queue)
    assert manifest["maximum_unique_generated_candidates"] == 81 * len(queue) * 19
    assert not (root / "runs").exists()
    assert (root / "controller/src/experiments/campaign.py").exists()
    assert (root / "controller/src/agent/prompts/builder.py").exists()
    assert (root / "controller/docker/agents/worker.py").exists()
    assert manifest["agent_execution"] == "docker"
    assert manifest["solver_execution"] == "native-macos-seatbelt"
    assert manifest["runtime_environment"] == require_runtime_environment()
    assert {"pyproject.toml", ".python-version"} <= manifest["controller_sha256"].keys()
    assert campaign.require_campaign_environment(manifest) == manifest["runtime_environment"]
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
        if case["problem"] == "lrp":
            assert "1 means open and 0 means closed" in (frozen / "prompt.txt").read_text()
    assert read(root / "pause_request.json")["resume_requires_explicit_user_request"]
    with pytest.raises(ProtocolError, match="paused"):
        ensure_execution_allowed(root, manifest)
    resume(root)
    ensure_execution_allowed(root, manifest)
    dashboard = Dashboard(root).snapshot()
    assert [m["id"] for m in dashboard["models"]] == list(PAPER_MODELS)
    claude = next(m for m in dashboard["models"] if m["id"] == "claude-sonnet-5-5")
    assert claude["backend"] == "Claude Code" and claude["effort"] == "Medium"
    assert dashboard["totals"]["planned"] == 81 * len(queue)
    with pytest.raises(ProtocolError, match="existing experiments are preserved"):
        initialize(root, data, config, tmp_path / "runtime.json")


def test_wrong_runtime_stops_before_creating_campaign(tmp_path, monkeypatch):
    data, config = fixture_config(tmp_path)
    root = tmp_path / "new"
    monkeypatch.setattr(environment, "version", lambda _: "12.0.0")
    with pytest.raises(RuntimeEnvironmentError, match="Gurobi"):
        initialize(root, data, config, tmp_path / "runtime.json")
    assert not root.exists()


@pytest.mark.parametrize("command", ["run", "watch"])
def test_runtime_failure_has_nonretryable_exit_code(tmp_path, monkeypatch, command):
    def wrong_runtime(_, models=None):
        raise RuntimeEnvironmentError("version changed")
    monkeypatch.setattr(campaign, command, wrong_runtime)
    with pytest.raises(SystemExit) as stopped:
        campaign.main([command, "--root", str(tmp_path), "--all-models"])
    assert stopped.value.code == 64


def test_mismatched_inputs_fail_before_creating_a_campaign(tmp_path):
    data, config = fixture_config(tmp_path)
    next(data.glob("*/*/*/data.json")).write_text('{"changed":true}')
    root = tmp_path / "new"
    with pytest.raises(ProtocolError, match="inventory"):
        initialize(root, data, config, tmp_path / "runtime.json")
    assert not root.exists()


def test_campaign_with_other_prompts_cannot_execute(tmp_path):
    with pytest.raises(ProtocolError, match="fresh cohort"):
        ensure_execution_allowed(tmp_path, {"protocol_id": "old"})
