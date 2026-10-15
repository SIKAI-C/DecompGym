from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import signal

import pytest

from src.agent.paired_runner import ProtocolError, read, save
from src.campaign_plan import CLAUDE_EXTENSION_MODELS, load_plan
from src import extension_coordinator as coordinator
from src import native_campaign


PLAN = Path(__file__).resolve().parents[1] / "docs/2026-10-14-claude-extension-plan.json"


def test_extension_order_and_budget_are_explicit():
    plan = load_plan(PLAN)
    assert tuple(x["model"] for x in plan["model_queue"]) == CLAUDE_EXTENSION_MODELS
    assert {x["effort"] for x in plan["model_queue"]} == {"medium"}
    assert plan["execution"]["solver_timeout"] == 60
    assert plan["execution"]["judge_effort"] == "high"


@pytest.mark.parametrize("change", ["reorder", "effort", "parent", "extra", "reuse"])
def test_extension_rejects_unapproved_changes(tmp_path, change):
    plan = read(PLAN)
    if change == "reorder":
        plan["model_queue"][0], plan["model_queue"][1] = plan["model_queue"][1], plan["model_queue"][0]
    elif change == "effort":
        plan["model_queue"][0]["effort"] = "high"
    elif change == "parent":
        plan["extension_parent"]["protocol_id"] = "other"
    elif change == "extra":
        plan["model_queue"].append({"model": "claude-opus-5-5", "backend": "claude-code-cli", "effort": "medium"})
    else:
        plan["reuse_generated_context"] = True
    p = tmp_path / "plan.json"
    save(p, plan)
    with pytest.raises(ProtocolError):
        load_plan(p)


def fake_parent(tmp_path, monkeypatch):
    parent = tmp_path / "base"
    save(parent / "watchdog.json", {"pid": 123})
    save(parent / "status.json", {"state": "recovering", "active_instances": {}})
    proc = {"pid": 123, "started": "Wed Oct 14 00:00:00 2026", "state": "S", "command": f"python -m src.native_campaign watch --root {parent} --all-models"}
    monkeypatch.setattr(coordinator, "runtime_lock", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(coordinator, "process_identity", lambda pid: dict(proc) if pid == 123 else None)
    monkeypatch.setattr(coordinator, "cwd", lambda pid: str(parent / "controller"))
    monkeypatch.setattr(coordinator, "owns_runtime_namespace", lambda pid: True)
    monkeypatch.setattr(coordinator, "live_workers", lambda: [])
    monkeypatch.setattr(coordinator, "children", lambda pid: [])
    signals = []
    def send(pid, signum):
        signals.append(signum)
        proc["state"] = "T" if signum == signal.SIGSTOP else "S"
    monkeypatch.setattr(coordinator.os, "kill", send)
    return parent, proc, signals


def test_idle_handoff_parks_only_the_watchdog(tmp_path, monkeypatch):
    parent, proc, signals = fake_parent(tmp_path, monkeypatch)
    record = {"phase": "waiting_for_idle"}
    assert coordinator.park_parent(parent, record, lambda: None)
    assert signals == [signal.SIGSTOP]
    assert record["phase"] == "parked" and proc["state"] == "T"


def test_startup_race_rolls_back_before_launch(tmp_path, monkeypatch):
    parent, proc, signals = fake_parent(tmp_path, monkeypatch)
    monkeypatch.setattr(coordinator, "children", lambda pid: [456] if proc["state"] == "T" else [])
    record = {"phase": "waiting_for_idle"}
    with pytest.raises(RuntimeError, match="raced"):
        coordinator.park_parent(parent, record, lambda: None)
    assert signals == [signal.SIGSTOP, signal.SIGCONT]
    assert record["phase"] == "waiting_for_idle"


def test_unknown_stopped_process_is_not_adopted(tmp_path, monkeypatch):
    parent, proc, signals = fake_parent(tmp_path, monkeypatch)
    proc["state"] = "T"
    with pytest.raises(RuntimeError, match="unowned"):
        coordinator.park_parent(parent, {"phase": "waiting_for_idle"}, lambda: None)
    assert not signals


def test_zero_exit_partial_scope_cannot_release(tmp_path, monkeypatch):
    parent, proc, signals = fake_parent(tmp_path, monkeypatch)
    proc["state"] = "T"
    with pytest.raises(ProtocolError, match="completion"):
        coordinator.release_parent(parent, tmp_path / "extension", {"parent_watchdog": dict(proc)}, lambda: None, lambda: False)
    assert not signals


def test_user_pause_during_release_intent_cancels_signal(tmp_path, monkeypatch):
    parent, proc, signals = fake_parent(tmp_path, monkeypatch)
    proc["state"] = "T"
    extension = tmp_path / "extension"
    monkeypatch.setattr(coordinator, "surviving_owned_processes", lambda root: [])
    record = {"parent_watchdog": dict(proc)}
    def persist():
        if record.get("phase") == "release_intent":
            (parent / "STOP").write_text("user pause\n")
    with pytest.raises(ProtocolError, match="cancelled"):
        coordinator.release_parent(parent, extension, record, persist, lambda: True)
    assert not signals


def test_scope_complete_requires_every_pair_and_exact_order(tmp_path, monkeypatch):
    manifest = {"protocol_id": "protocol", "instances": [{"instance": "one"}]}
    state = {"state": "complete", "scope_complete": True, "active_instances": {}, "execution_scope": list(CLAUDE_EXTENSION_MODELS)}
    save(tmp_path / "status.json", state)
    monkeypatch.setattr(coordinator, "pair_complete", lambda root, model, case, protocol: model != CLAUDE_EXTENSION_MODELS[-1])
    assert not coordinator.extension_complete(tmp_path, manifest)
    monkeypatch.setattr(coordinator, "pair_complete", lambda *a: True)
    assert coordinator.extension_complete(tmp_path, manifest)
    state["execution_scope"].reverse()
    save(tmp_path / "status.json", state)
    assert not coordinator.extension_complete(tmp_path, manifest)


def test_duplicate_watcher_cannot_overwrite_receipt(tmp_path, monkeypatch):
    from src.agent import runtime_state
    from src import campaign_extension, hybrid_runtime
    lock_root = tmp_path / "locks"
    lock_root.mkdir(mode=0o700)
    monkeypatch.setattr(runtime_state, "state_root", lambda: lock_root)
    manifest = {"model_queue": [{"model": "model", "backend": "claude-code-cli"}], "execution_environment": {}, "resource_guard": {}}
    monkeypatch.setattr(native_campaign, "validate", lambda root: manifest)
    monkeypatch.setattr(native_campaign, "ensure_execution_allowed", lambda *a: None)
    monkeypatch.setattr(native_campaign, "require_campaign_environment", lambda *a: None)
    monkeypatch.setattr(native_campaign.signal, "signal", lambda *a: None)
    monkeypatch.setattr(hybrid_runtime, "runtime", lambda: type("Engine", (), {"bind_environment": lambda *a: None})())
    monkeypatch.setattr(campaign_extension, "watcher_lock", lambda *a: ("extension-test", False))
    save(tmp_path / "watchdog.json", {"pid": 456, "owned": True})
    with runtime_state.runtime_lock("extension-test", blocking=False):
        with pytest.raises(BlockingIOError):
            native_campaign.watch(tmp_path)
    assert read(tmp_path / "watchdog.json") == {"pid": 456, "owned": True}


def test_owned_hold_intent_can_recover_without_second_signal(tmp_path, monkeypatch):
    parent, proc, signals = fake_parent(tmp_path, monkeypatch)
    proc["state"] = "T"
    record = {"phase": "hold_intent", "parent_watchdog": dict(proc)}
    assert coordinator.park_parent(parent, record, lambda: None)
    assert record["phase"] == "parked" and not signals


@pytest.mark.parametrize("phase", ["extension_running", "release_intent"])
def test_coordinator_recovery_does_not_spawn_or_resignal(tmp_path, monkeypatch, phase):
    parent, extension = tmp_path / "base", tmp_path / "extension"
    group = {"members": [{"root": str(parent)}, {"root": str(extension)}], "active_root": str(extension)}
    group_path, state_path = tmp_path / "group.json", tmp_path / "coord.json"
    save(group_path, group)
    parent_identity = {"pid": 123, "started": "start", "command": "parent", "state": "T" if phase == "extension_running" else "S"}
    watcher = {"pid": 456, "started": "start", "command": "extension", "state": "S"}
    save(state_path, {"token": "token", "phase": phase, "parent_protocol_id": "base", "extension_protocol_id": "extension", "extension_root": str(extension), "parent_watchdog": parent_identity, "extension_watchdog": watcher})
    manifests = [{"protocol_id": "base"}, {"protocol_id": "extension"}]
    monkeypatch.setattr(coordinator, "validate_group", lambda _: manifests)
    monkeypatch.setattr(coordinator, "validate", lambda root: manifests[root == extension])
    monkeypatch.setattr(coordinator, "extension_parent", lambda _: {"coordinator_state": str(state_path)})
    monkeypatch.setattr(coordinator, "runtime_lock", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(coordinator, "process_identity", lambda pid: parent_identity if pid == 123 else None)
    monkeypatch.setattr(coordinator, "find_extension_watchers", lambda *a: [watcher])
    monkeypatch.setattr(coordinator, "extension_complete", lambda *a: True)
    monkeypatch.setattr(coordinator, "owned_live_workers", lambda *a: [])
    monkeypatch.setattr(coordinator, "surviving_owned_processes", lambda *a: [])
    monkeypatch.setattr(coordinator.subprocess, "Popen", lambda *a, **kw: pytest.fail("Duplicate launch"))
    monkeypatch.setattr(coordinator.os, "kill", lambda *a: pytest.fail("Duplicate signal"))
    def release(base, ext, record, persist, completed):
        record["phase"] = "base_resumed"
        persist()
    monkeypatch.setattr(coordinator, "release_parent", release)
    coordinator.coordinate(group_path)
    assert read(state_path)["phase"] == "base_resumed"
    assert read(group_path)["active_root"] == str(parent)


def test_real_idle_process_can_be_parked_and_released(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    import time
    from src.agent import runtime_state
    parent = tmp_path / "base"
    (parent / "controller").mkdir(parents=True)
    locks = tmp_path / "locks"
    locks.mkdir(mode=0o700)
    monkeypatch.setattr(runtime_state, "state_root", lambda: locks)
    monkeypatch.setattr(coordinator, "state_root", lambda: locks)
    monkeypatch.setattr(coordinator, "live_workers", lambda: [])
    script = "import fcntl,time; f=open(" + repr(str(locks / "fixed81-native-watchdog.lock")) + ",'w'); fcntl.flock(f,fcntl.LOCK_EX); print('ready',flush=True); time.sleep(60)"
    process = subprocess.Popen([sys.executable, "-c", script, "-m", "src.native_campaign", "watch", "--root", str(parent), "--all-models"], cwd=parent / "controller", stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "ready"
        save(parent / "watchdog.json", {"pid": process.pid})
        save(parent / "status.json", {"state": "recovering", "active_instances": {}})
        record = {"phase": "waiting_for_idle"}
        assert coordinator.park_parent(parent, record, lambda: None)
        assert "T" in coordinator.process_identity(process.pid)["state"]
        coordinator.release_parent(parent, tmp_path / "extension", record, lambda: None, lambda: True)
        for _ in range(20):
            if "T" not in coordinator.process_identity(process.pid)["state"]:
                break
            time.sleep(0.05)
        assert record["phase"] == "base_resumed"
        assert "T" not in coordinator.process_identity(process.pid)["state"]
    finally:
        os.kill(process.pid, signal.SIGCONT)
        process.terminate()
        process.wait(timeout=5)
