"""Private credential delivery never overrides a user pause or unfinished scope."""
import pytest
from pathlib import Path
from types import SimpleNamespace

from scripts.launch_campaign_private import authorize_handoff, read, save, verify_private_credential, VARIABLE


@pytest.mark.parametrize("blocker", [None, "STOP", "user_pause", "active_instance", "unfinished_scope", "wrong_scope"])
def test_private_handoff_only_follows_the_authorized_scope_boundary(tmp_path, blocker):
    manifest = {"model_queue": [{"model": "gpt", "backend": "codex-cli"}, {"model": "claude", "backend": "claude-code-cli"}, {"model": "gemini", "backend": "agy-cli"}]}
    pause = {"resume_requires_explicit_user_request": True, "reason": "launch_scope_complete"}
    state = {"state": "paused", "scope_complete": True, "execution_scope": ["gpt", "claude"], "active_instances": {}}
    if blocker == "STOP":
        (tmp_path / "STOP").touch()
    elif blocker == "user_pause":
        pause["reason"] = "user asked to pause"
    elif blocker == "active_instance":
        state["active_instances"] = {"worker": "busy"}
    elif blocker == "unfinished_scope":
        state["scope_complete"] = False
    elif blocker == "wrong_scope":
        state["execution_scope"] = ["gpt"]
    save(tmp_path / "pause_request.json", pause)
    save(tmp_path / "status.json", state)
    if blocker:
        with pytest.raises(RuntimeError, match="no automatic continuation"):
            authorize_handoff(tmp_path, manifest)
        assert read(tmp_path / "pause_request.json") == pause
    else:
        authorize_handoff(tmp_path, manifest)
        assert not read(tmp_path / "pause_request.json")["resume_requires_explicit_user_request"]


@pytest.mark.parametrize("unlocked", [True, False])
def test_private_launch_verifies_the_existing_keyring_without_inference(tmp_path, monkeypatch, unlocked):
    from src import hybrid_runtime
    calls = []
    engine = SimpleNamespace(bind_environment=lambda _: None, ensure_volume=lambda *args, **kwargs: None)

    def run(image, command, **kwargs):
        calls.append((image, command, kwargs))
        kwargs["state_path"].parent.mkdir(parents=True)
        assert engine.LABEL != "io.decompbench.runtime"
        assert kwargs["payload"]["credential"]["keyring_passphrase"] == "synthetic-private-input"
        return SimpleNamespace(returncode=0 if unlocked else 1, stdout='{"credential_loaded":true}', metadata={"cleanup_verified": True})

    engine.run = run
    monkeypatch.setattr(hybrid_runtime, "runtime", lambda: engine)
    monkeypatch.setenv(VARIABLE, "synthetic-private-input")
    manifest = {"execution_environment": {"image_id": "sha256:fixture"}}
    if unlocked:
        report = verify_private_credential(tmp_path, manifest)
        assert read(Path(report))["credential_verified"]
    else:
        with pytest.raises(RuntimeError, match="unlock failed"):
            verify_private_credential(tmp_path, manifest)
    assert len(calls) == 1 and calls[0][1] == ["auth-check", "agy-cli"]
    assert calls[0][2]["network"] == "none" and calls[0][2]["cpus"] == 0.25
    if not unlocked:
        assert not list(tmp_path.glob("**/report.json"))
    assert all("synthetic-private-input" not in path.read_text() for path in tmp_path.rglob("*.json"))
