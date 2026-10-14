"""The dashboard distinguishes a verified automatic queue from a later manual run."""
from datetime import datetime, timezone
import fcntl

import pytest

from src import dashboard_server as server
from src.agent.paired_runner import save


@pytest.mark.parametrize("condition", ["ready", "scope_boundary", "helper_dead", "lock_free", "wrong_protocol", "wrong_scope", "unverified_credentials", "wrong_watchdog", "user_pause", "STOP"])
def test_gemini_is_queued_only_with_a_live_matching_handoff(tmp_path, monkeypatch, condition):
    queue = [
        {"model": "gpt-6-sol", "backend": "codex-cli", "effort": "medium"},
        {"model": "claude-sonnet-5-5", "backend": "claude-code-cli", "effort": "medium"},
        {"model": "gemini-3.8-flash-medium", "backend": "agy-cli", "effort": "catalog-slug"},
        {"model": "gemini-3.1-pro-high", "backend": "agy-cli", "effort": "catalog-slug"},
    ]
    case = {"instance": "ufl_small_seed000", "suite": "bd", "problem": "ufl", "size": "small", "seed": 0}
    manifest = {"backend": "native", "schema_version": 3, "protocol_id": "queue-fixture", "model_queue": queue, "instances": [case], "max_attempts": 10}
    scope = [item["model"] for item in queue[:2]]
    status = {"state": "running", "pid": 101, "active_model": "claude-sonnet-5-5", "execution_scope": scope, "active_instances": {}}
    record = {"state": "waiting_for_GPT_Claude_scope", "helper_pid": 303, "preceding_watchdog_pid": 202, "protocol_id": manifest["protocol_id"], "credential_verified": True, "all_models": True}
    pause = {"resume_requires_explicit_user_request": False}
    if condition == "wrong_protocol":
        record["protocol_id"] = "another-cohort"
    elif condition == "wrong_scope":
        status["execution_scope"] = scope[:1]
    elif condition == "unverified_credentials":
        record["credential_verified"] = False
    elif condition == "wrong_watchdog":
        record["preceding_watchdog_pid"] = 999
    elif condition in ("user_pause", "scope_boundary"):
        pause.update(resume_requires_explicit_user_request=True, reason="launch_scope_complete" if condition == "scope_boundary" else "user_requested_pause")
        status["state"] = "paused"
    elif condition == "STOP":
        (tmp_path / "STOP").touch()
    save(tmp_path / "campaign.json", manifest)
    save(tmp_path / "status.json", status)
    save(tmp_path / "pause_request.json", pause)
    save(tmp_path / "watchdog.json", {"pid": 202})
    save(tmp_path / "operations/private_launch.json", record)
    save(tmp_path / "monitor/status.json", {"state": "running", "sampled_at_utc": datetime.now(timezone.utc).isoformat()})
    monkeypatch.setattr(server, "process_alive", lambda pid: pid in (101, 202) or (pid == 303 and condition != "helper_dead"))
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    with (tmp_path / "operations/private_launch.lock").open("a") as lock:
        if condition != "lock_free":
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = server.Dashboard(tmp_path, reports_root=tmp_path / "reports").snapshot()
    gemini = [model for model in snapshot["models"] if model["backend"] == "AGY"]
    ready = condition in ("ready", "scope_boundary")
    assert snapshot["automatic_continuation"]["armed"] is ready
    assert all(model["state"] == ("queued" if ready else "later") for model in gemini)
    assert all(bool(model.get("queue_note")) is ready for model in gemini)
    assert all(not model["in_current_scope"] for model in gemini)
    assert snapshot["totals"]["planned"] == 4
    assert snapshot["execution_scope"]["planned"] == len(status["execution_scope"])
    assert all(path.read_bytes() == contents for path, contents in before.items())
