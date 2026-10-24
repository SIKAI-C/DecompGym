"""The dashboard shows scoped progress and links only the report of its own campaign."""
from copy import deepcopy
from datetime import datetime, timezone
import os

import pytest

from src.experiments.dashboard_server import Dashboard, export_csv
from src.utils.records import identity, save


@pytest.fixture
def fresh_dashboard(tmp_path):
    root = tmp_path / "campaign"
    queue = [{"model": "gpt-6-sol", "backend": "codex-cli", "effort": "medium"}, {"model": "claude-sonnet-5-5", "backend": "claude-code-cli", "effort": "medium"}]
    case = {"instance": "ufl_small_seed000", "suite": "bd", "problem": "ufl", "size": "small", "seed": 0}
    manifest = {"backend": "native", "schema_version": 4, "model_queue": queue, "max_attempts": 10, "instances": [case], "agent_execution": "docker", "solver_execution": "native-macos-seatbelt", "experiment_plan": {"name": "test"}}
    manifest["protocol_id"] = identity(manifest)
    save(root / "campaign.json", manifest)
    save(root / "status.json", {"state": "running", "pid": os.getpid(), "active_model": "gpt-6-sol", "execution_scope": ["gpt-6-sol"], "active_instances": {}})
    save(root / "monitor/status.json", {"state": "running", "sampled_at_utc": datetime.now(timezone.utc).isoformat()})
    attempts = [{"attempt": i, "status": "ok", "metrics": {"generation": {"invocation_id": "shared-generation", "elapsed_seconds": 12, "usage": {"input_tokens": 100, "output_tokens": 20}}}} for i in range(1, 11)]
    for mode in ("plain", "reflection"):
        save(root / "runs/gpt-6-sol" / mode / "bd/ufl/ufl_small_seed000/instance_summary.json", {"model": "gpt-6-sol", "instance": case["instance"], "protocol_id": manifest["protocol_id"], "max_attempts": 10, "attempts": deepcopy(attempts), "complete": True})
    return Dashboard(root, reports_root=tmp_path / "reports"), manifest


def test_scope_progress_usage_and_csv_export_leave_raw_records_unchanged(fresh_dashboard):
    dashboard, manifest = fresh_dashboard
    raw = {path: path.read_bytes() for path in (dashboard.root / "runs").rglob("*.json")}
    snapshot = dashboard.snapshot()
    assert len(snapshot["models"]) == 2 and snapshot["totals"]["planned"] == 2
    assert snapshot["execution_scope"]["planned"] == snapshot["execution_scope"]["completed"] == 1
    assert snapshot["models"][1]["state"] == "later" and snapshot["models"][1]["effort"] == "Medium"
    assert [model["id"] for model in snapshot["models"]] == ["gpt-6-sol", "claude-sonnet-5-5"]
    assert b"gpt-6-sol" in export_csv(snapshot)
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


def test_native_first_pass_does_not_imply_retry_pair_is_complete(tmp_path):
    model, instance = 'gpt-6-sol', 'ufl_small_seed000'
    now = datetime.now(timezone.utc).isoformat()
    save(tmp_path / 'campaign.json', {'backend':'native', 'model_queue':[{'model':model,'backend':'codex-cli','effort':'medium'}], 'max_attempts':10, 'instances':[{'instance':instance,'suite':'bd','problem':'ufl','size':'small','seed':0}]})
    save(tmp_path / 'status.json', {'state':'running','pid':os.getpid(),'active_model':model,'active_instances':{'experiment_0':{'instance':instance,'mode':'reflection','attempt':2,'phase':'agent'}}})
    save(tmp_path / 'monitor/status.json', {'sampled_at_utc':now,'cpu_percent':12,'system_free_percent':40,'thermal':'No warnings'})
    attempt = {'attempt':1,'status':'ok','metrics':{'generation':{'invocation_id':'shared','usage':{'input_tokens':10}}}}
    for mode in ['plain','reflection']:
        save(tmp_path / 'runs' / model / mode / 'bd/ufl' / instance / 'instance_summary.json', {'model':model,'instance':instance,'attempts':[attempt],'complete':False})
    value = Dashboard(tmp_path).snapshot()
    assert value['totals']['pass_rate'] == 1
    assert value['totals']['completed'] == 0 and value['models'][0]['state'] == 'running'
    assert len(value['active_workers']) == 1 and 'attempt 2/10' in value['active_workers'][0]['phase']
    assert value['usage']['generation']['input_tokens'] == 10
    assert value['monitor']['docker_and_controller_cpu_percent'] == 12

