"""Private credential delivery never overrides a user pause or unfinished scope."""
import pytest

from scripts.launch_campaign_private import authorize_handoff, read, save


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
