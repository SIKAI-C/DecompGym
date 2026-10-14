"""Operational model selection preserves the full frozen cohort and pauses at its boundary."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
import threading

import pytest

from src import hybrid_runtime, native_campaign as campaign
from src.agent import native_solver
from src.agent.docker_client import AuthenticationRequired
from src.agent.paired_runner import ProtocolError, read, save
from src.agent.prompt_policy import FEEDBACK_POLICY, REFLECTION_POLICY, WORKSPACE_POLICY
from src.prompt_snapshot import PROMPT_PROTOCOL


@pytest.fixture
def scoped_campaign(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    (root / "controller/src").mkdir(parents=True)
    monkeypatch.setattr(campaign, "__file__", str(root / "controller/src/native_campaign.py"))
    queue = [
        {"model": "gpt-a", "backend": "codex-cli", "effort": "medium"},
        {"model": "gpt-b", "backend": "codex-cli", "effort": "medium"},
        {"model": "gemini", "backend": "agy-cli", "effort": "catalog-slug"},
        {"model": "claude", "backend": "claude-code-cli", "effort": "medium"},
    ]
    cases = [{"suite": "bd", "problem": "ufl", "instance": f"case-{index}"} for index in range(2)]
    manifest = {"model_queue": queue, "instances": cases, "protocol_id": "frozen-protocol", "agent_timeout": 900,
                "judge_model": "independent-judge", "judge_effort": "high", "execution_environment": {"image_id": "sha256:fixture", "license_file": "fixture", "native_environment": {}},
                "prompt_policies": {"feedback": FEEDBACK_POLICY, "workspace": WORKSPACE_POLICY}, "reflection_policy": REFLECTION_POLICY,
                "prompt_protocol": PROMPT_PROTOCOL, "resource_guard": {}, "retry_delays_seconds": [0]}
    before = deepcopy(manifest)
    engine = SimpleNamespace(image_id=lambda name: name, bind_environment=lambda _: None, resource_samples=lambda: [], stop_campaign_workers=lambda _: None)
    monkeypatch.setattr(hybrid_runtime, "runtime", lambda: engine)
    monkeypatch.setattr(native_solver, "NativeSandboxExecutor", lambda **kwargs: object())
    monkeypatch.setattr(campaign, "validate", lambda _: manifest)
    monkeypatch.setattr(campaign, "require_campaign_environment", lambda *_: None)
    readiness = []
    monkeypatch.setattr(campaign, "require_backend_readiness", lambda env, backends: readiness.append(set(backends)) or {})
    monkeypatch.setattr(campaign, "state_root", lambda: tmp_path / "locks")
    monkeypatch.setattr(campaign, "runtime_lock", lambda *_: nullcontext())
    monkeypatch.setattr(campaign.signal, "signal", lambda *_: None)
    monkeypatch.delenv("DECOMPBENCH_AGY_KEYRING_PASSPHRASE", raising=False)
    completed, calls = set(), []
    monkeypatch.setattr(campaign, "pair_complete", lambda root, model, case, protocol: (model, case["instance"]) in completed)

    class ObservedRunner:
        def __init__(self, **kwargs):
            self.arguments = kwargs

        def run_pair(self, source, destination):
            calls.append(self.arguments)
            completed.add((self.arguments["model"], source.name))

    monkeypatch.setattr(campaign, "PairedRunner", ObservedRunner)
    return SimpleNamespace(root=root, manifest=manifest, before=before, readiness=readiness, completed=completed, calls=calls)


@pytest.mark.parametrize("models", [[], ["gpt-a", "gpt-a"], ["unknown"], ["gpt-oss-120b-medium"]])
def test_invalid_scope_stops_before_runtime_checks(scoped_campaign, models):
    fixture = scoped_campaign
    with pytest.raises(ProtocolError, match="scope"):
        campaign.run(fixture.root, models=models)
    assert not fixture.readiness and not fixture.calls
    assert fixture.manifest == fixture.before


def test_gpt_scope_requires_only_codex_and_preserves_manifest_order(scoped_campaign):
    fixture = scoped_campaign
    assert campaign.run(fixture.root, models=["gpt-b", "gpt-a"]) == 0
    assert fixture.readiness == [{"codex-cli"}]
    assert [item["model"] for item in fixture.calls] == ["gpt-a", "gpt-a", "gpt-b", "gpt-b"]
    for item in fixture.calls:
        assert item["client"].effort == item["reflection_client"].effort == "medium"
        assert item["judge_client"].model == "independent-judge" and item["judge_client"].effort == "high"
    state = read(fixture.root / "status.json")
    assert state["state"] == "paused" and state["scope_complete"] and not state["campaign_complete"]
    assert state["execution_scope"] == ["gpt-a", "gpt-b"] and state["scope_expected_pairs"] == 4
    assert state["remaining_models"] == ["gemini", "claude"] and not state["active_instances"]
    pause = read(fixture.root / "pause_request.json")
    assert pause["resume_requires_explicit_user_request"] and pause["reason"] == "launch_scope_complete"
    audit = read(next((fixture.root / "launches").glob("*.json")))
    assert audit["execution_scope"] == state["execution_scope"] and audit["protocol_id"] == fixture.manifest["protocol_id"]
    assert fixture.manifest == fixture.before


def test_scope_cannot_bypass_pause_and_completed_scope_is_reused(scoped_campaign):
    fixture = scoped_campaign
    assert campaign.run(fixture.root, models=["gpt-a", "gpt-b"]) == 0
    with pytest.raises(ProtocolError, match="paused"):
        campaign.run(fixture.root, models=["claude"])
    with pytest.raises(ProtocolError, match="paused"):
        campaign.watch(fixture.root, models=["claude"])
    save(fixture.root / "pause_request.json", {"resume_requires_explicit_user_request": False})
    before = len(fixture.calls)
    assert campaign.run(fixture.root, models=["gpt-a", "gpt-b"]) == 0
    assert len(fixture.calls) == before
    assert read(fixture.root / "status.json")["state"] == "paused"


def test_later_scopes_keep_claude_effort_and_require_agy_credentials(scoped_campaign, monkeypatch):
    fixture = scoped_campaign
    assert campaign.run(fixture.root, models=["gpt-a", "gpt-b"]) == 0
    save(fixture.root / "pause_request.json", {"resume_requires_explicit_user_request": False})
    assert campaign.run(fixture.root, models=["claude"]) == 0
    assert fixture.readiness[-1] == {"claude-code-cli", "codex-cli"}
    assert all(item["client"].effort == item["reflection_client"].effort == "medium" for item in fixture.calls if item["model"] == "claude")
    save(fixture.root / "pause_request.json", {"resume_requires_explicit_user_request": False})
    before = len(fixture.calls)
    with pytest.raises(AuthenticationRequired):
        campaign.run(fixture.root, models=["gemini"])
    assert len(fixture.calls) == before
    monkeypatch.setenv("DECOMPBENCH_AGY_KEYRING_PASSPHRASE", "synthetic-test-only")
    assert campaign.run(fixture.root, models=["gemini"]) == 0
    assert fixture.readiness[-1] == {"agy-cli", "codex-cli"}
    assert read(fixture.root / "status.json")["state"] == "complete"
    assert fixture.manifest == fixture.before


def test_watchdog_reuses_the_same_scope_on_infrastructure_retry(scoped_campaign, monkeypatch):
    fixture = scoped_campaign
    commands, results = [], iter([75, 0])

    def supervise(command, *_args, **_kwargs):
        commands.append(command)
        return next(results)

    monkeypatch.setattr(campaign, "supervise", supervise)
    monkeypatch.setattr(campaign, "sample_host", lambda: ({}, "", 50))
    monkeypatch.setattr(campaign.time, "sleep", lambda *_: None)
    assert campaign.watch(fixture.root, models=["gpt-b", "gpt-a"]) == 0
    assert len(commands) == 2 and commands[0] == commands[1]
    assert commands[0][commands[0].index("--model"):] == ["--model", "gpt-a", "--model", "gpt-b"]
    assert read(fixture.root / "watchdog.json")["execution_scope"] == ["gpt-a", "gpt-b"]


@pytest.mark.parametrize("marker", ["STOP", "pause_request"])
def test_live_pause_drains_both_admitted_pairs_without_dispatching_more(scoped_campaign, monkeypatch, marker):
    fixture = scoped_campaign
    fixture.manifest["instances"].extend({"suite": "bd", "problem": "ufl", "instance": f"case-{index}"} for index in range(2, 4))
    both_admitted = threading.Barrier(2)
    original = campaign.PairedRunner.run_pair

    def run_pair(self, source, destination):
        both_admitted.wait(timeout=5)
        if marker == "STOP":
            (fixture.root / "STOP").touch()
        elif source.name == "case-0":
            save(fixture.root / "pause_request.json", {"resume_requires_explicit_user_request": True, "reason": "user requested current two then pause"})
        both_admitted.wait(timeout=5)
        return original(self, source, destination)

    monkeypatch.setattr(campaign.PairedRunner, "run_pair", run_pair)
    assert campaign.run(fixture.root, models=["gpt-a", "gpt-b"]) == 0
    assert fixture.completed == {("gpt-a", "case-0"), ("gpt-a", "case-1")}
    state = read(fixture.root / "status.json")
    assert state["state"] == "paused" and not state["active_instances"]
    assert not state["scope_complete"] and not state["campaign_complete"]
    assert read(fixture.root / "pause_request.json")["resume_requires_explicit_user_request"]
    with pytest.raises(ProtocolError, match="paused"):
        campaign.run(fixture.root, models=["gpt-a", "gpt-b"])


@pytest.mark.parametrize("command", ["run", "watch"])
def test_cli_requires_explicit_scope_before_accessing_campaign(tmp_path, monkeypatch, command):
    monkeypatch.setattr(campaign.sys, "argv", ["native_campaign", command, "--root", str(tmp_path)])
    monkeypatch.setattr(campaign, "validate", lambda *_: pytest.fail("No scope must fail before loading a campaign"))
    with pytest.raises(SystemExit) as stopped:
        campaign.main()
    assert stopped.value.code == 2
