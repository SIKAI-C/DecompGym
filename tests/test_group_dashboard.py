from copy import deepcopy
from pathlib import Path

import pytest

from src.agent.paired_runner import save
from src import campaign_extension, dashboard_server


def test_group_aggregates_disjoint_models_and_routes_details(tmp_path, monkeypatch):
    base, extension = tmp_path / "base", tmp_path / "extension"
    base_models = [f"model-{i}" for i in range(7)] + ["gemini-flash", "gemini-pro"]
    extension_models = ["claude-haiku-5-5", "claude-opus-5", "claude-sonnet-5", "claude-fable-5"]
    manifests = [{"model_queue": [{"model": m} for m in base_models + ["gpt-5.6-sol"]]}, {"model_queue": [{"model": m} for m in extension_models]}]
    config = {"name": "group", "schema_version": 1, "members": [{"root": str(base), "protocol_id": "base-id"}, {"root": str(extension), "protocol_id": "extension-id"}], "active_root": str(extension), "display_order": base_models[:7] + extension_models + base_models[7:], "coordinator_state": str(tmp_path / "state.json")}
    save(tmp_path / "group.json", config)
    parent = {"pid": 12, "started": "start", "command": "watch", "state": "T"}
    save(tmp_path / "state.json", {"phase": "extension_running", "coordinator_pid": 13, "parent_watchdog": parent})
    def snapshot(models, count, active):
        rows = [{"id": m, "backend": "AGY" if m.startswith("gemini") else "Claude Code", "state": "recovering" if m.startswith("gemini") else "queued"} for m in models]
        return {"campaign": "child", "models": rows, "results": [], "recent_calls": [], "totals": {"planned": len(models) * 81, "completed": count, "remaining": len(models) * 81 - count, "passed": 0, "failed": 0, "first_attempt_completed": 0, "active": active}, "execution_scope": {}, "monitor": {}, "alerts": [], "state": "running"}
    snapshots = {base: snapshot(base_models, 632, 0), extension: snapshot(extension_models, 2, 2)}
    class Child:
        def __init__(self, root, historical):
            self.root = root
        def snapshot(self, force=False):
            return deepcopy(snapshots[self.root])
        def details(self, model, instance):
            return {"root": str(self.root), "model": model, "instance": instance}
    monkeypatch.setattr(dashboard_server, "Dashboard", Child)
    monkeypatch.setattr(campaign_extension, "validate_group", lambda config: manifests)
    monkeypatch.setattr(campaign_extension, "process_identity", lambda pid: parent)
    monkeypatch.setattr(dashboard_server, "process_alive", lambda pid: True)
    group = dashboard_server.GroupDashboard(tmp_path / "group.json")
    result = group.snapshot()
    assert result["totals"]["planned"] == 1053
    assert result["totals"]["completed"] == 634
    assert result["totals"]["active"] == 2
    assert [m["id"] for m in result["models"]] == config["display_order"]
    assert result["models"][-1]["state"] == "queued"
    assert result["automatic_continuation"]["armed"]
    assert group.details("claude-fable-5", "case")["root"] == str(extension)
    assert group.details("gemini-pro", "case")["root"] == str(base)
    with pytest.raises(ValueError):
        group.details("gpt-5.6-sol", "case")


def test_invalid_member_identity_is_not_combined(tmp_path):
    root = tmp_path / "root"
    save(root / "campaign.json", {"protocol_id": "fake"})
    with pytest.raises(ValueError, match="identity"):
        campaign_extension.validate_group({"schema_version": 1, "members": [{"root": str(root), "protocol_id": "fake"}] * 2})
