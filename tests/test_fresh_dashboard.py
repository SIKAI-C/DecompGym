"""Fresh monitoring preserves cohort eligibility, scoped progress and read-only reports."""
from copy import deepcopy
from datetime import datetime, timezone
import os

import pytest

from src.agent.paired_runner import identity, save
from src import reporting_policy
from src.dashboard_server import Dashboard, export_csv
from src.reporting_policy import AUTHORIZED_FRESH_PROTOCOL_IDS, AUTHORIZED_FRESH_PLAN, AUTHORIZED_INVENTORY_SHA256


@pytest.fixture
def fresh_dashboard(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    queue = [{"model": "gpt-6-sol", "backend": "codex-cli", "effort": "medium"}, {"model": "claude-sonnet-5-5", "backend": "claude-code-cli", "effort": "medium"}, {"model": "gpt-5.6-sol", "backend": "codex-cli", "effort": "medium"}]
    protocol = next(iter(AUTHORIZED_FRESH_PROTOCOL_IDS))
    case = {"instance": "ufl_small_seed000", "suite": "bd", "problem": "ufl", "size": "small", "seed": 0}
    manifest = {"backend": "native", "schema_version": 3, "protocol_id": protocol, "model_queue": queue, "max_attempts": 10, "instances": [case], "agent_execution": "docker", "solver_execution": "native-macos-seatbelt", "historical_reference_in_fresh_totals": False,
                "experiment_plan": {"name": AUTHORIZED_FRESH_PLAN, "dataset_inventory_sha256": AUTHORIZED_INVENTORY_SHA256, "reuse_generated_context": False, "reporting": {"mix_legacy_results_into_fresh_totals": False}}}
    manifest.pop("protocol_id")
    protocol = identity(manifest)
    manifest["protocol_id"] = protocol
    monkeypatch.setattr(reporting_policy, "AUTHORIZED_FRESH_PROTOCOL_IDS", {protocol})
    save(root / "campaign.json", manifest)
    save(root / "status.json", {"state": "running", "pid": os.getpid(), "active_model": "gpt-6-sol", "execution_scope": ["gpt-6-sol", "gpt-5.6-sol"], "active_instances": {}})
    save(root / "monitor/status.json", {"state": "running", "sampled_at_utc": datetime.now(timezone.utc).isoformat()})
    attempts = [{"attempt": i, "status": "ok", "metrics": {"generation": {"invocation_id": "shared-generation", "elapsed_seconds": 12, "usage": {"input_tokens": 100, "output_tokens": 20}}}} for i in range(1, 11)]
    excluded_attempts = deepcopy(attempts)
    for attempt in excluded_attempts:
        attempt["metrics"]["generation"].update(invocation_id="excluded-shared-generation", usage={"input_tokens": 900, "output_tokens": 99})
    for mode in ("plain", "reflection"):
        save(root / "runs/gpt-6-sol" / mode / "bd/ufl/ufl_small_seed000/instance_summary.json", {"model": "gpt-6-sol", "instance": case["instance"], "protocol_id": protocol, "max_attempts": 10, "attempts": deepcopy(attempts), "complete": True})
        save(root / "runs/gpt-5.6-sol" / mode / "bd/ufl/ufl_small_seed000/instance_summary.json", {"model": "gpt-5.6-sol", "instance": case["instance"], "protocol_id": protocol, "max_attempts": 10, "attempts": deepcopy(excluded_attempts), "complete": True})
    return Dashboard(root, reports_root=tmp_path / "reports"), manifest


def test_fresh_sol_is_excluded_before_scope_usage_and_exports_without_changing_raw_records(fresh_dashboard):
    dashboard, manifest = fresh_dashboard
    raw = {path: path.read_bytes() for path in (dashboard.root / "runs/gpt-5.6-sol").rglob("*.json")}
    assert reporting_policy.cohort_eligibility(manifest)["authorized_fresh_cohort"]
    snapshot = dashboard.snapshot()
    assert len(snapshot["models"]) == 2 and snapshot["totals"]["planned"] == 2
    assert snapshot["execution_scope"]["planned"] == snapshot["execution_scope"]["completed"] == 1
    assert snapshot["models"][1]["state"] == "later" and snapshot["models"][1]["effort"] == "Medium"
    assert [model["id"] for model in snapshot["models"]] == ["gpt-6-sol", "claude-sonnet-5-5"]
    assert snapshot["reporting_exclusions"][0]["model"] == "gpt-5.6-sol"
    assert b"gpt-5.6-sol" not in export_csv(snapshot)
    with pytest.raises(ValueError, match="excluded"):
        dashboard.details("gpt-5.6-sol", "ufl_small_seed000")
    assert dashboard.details("gpt-6-sol", "ufl_small_seed000")["model"] == "gpt-6-sol"
    assert len(snapshot["recent_calls"]) == 1
    assert snapshot["recent_calls"][0]["mode"] == "shared"
    assert snapshot["usage"]["generation"]["input_tokens"] == 100
    assert all(path.read_bytes() == value for path, value in raw.items())


def test_report_link_selects_only_matching_cohort_and_blocks_other_files(fresh_dashboard):
    dashboard, manifest = fresh_dashboard
    directory = dashboard.reports_root / "fresh"
    directory.mkdir(parents=True)
    (directory / "report.md").write_text(f"Cohort protocol: `{manifest['protocol_id']}`.\nSaved report")
    (directory / "evaluation.json").write_text('{"example":true}')
    other = dashboard.reports_root / "other"
    other.mkdir()
    (other / "report.md").write_text("Cohort protocol: `unrelated`.")
    assert dashboard.latest_report()["name"] == "fresh"
    assert "Saved report" in dashboard.report_artifact("report.md")
    assert dashboard.report_artifact("evaluation.json") == '{"example":true}'
    with pytest.raises(ValueError):
        dashboard.report_artifact("../../campaign/campaign.json")
    outside = dashboard.reports_root.parent / "private.json"
    outside.write_text("private")
    (directory / "evaluation.json").unlink()
    (directory / "evaluation.json").symlink_to(outside)
    with pytest.raises(ValueError):
        dashboard.report_artifact("evaluation.json")


def test_drain_remains_visible_and_does_not_change_pause_record(fresh_dashboard):
    dashboard, _ = fresh_dashboard
    pause = {"state": "draining", "resume_requires_explicit_user_request": True}
    save(dashboard.root / "pause_request.json", pause)
    before = (dashboard.root / "pause_request.json").read_bytes()
    assert dashboard.snapshot()["state"] == "draining_for_pause"
    assert (dashboard.root / "pause_request.json").read_bytes() == before
