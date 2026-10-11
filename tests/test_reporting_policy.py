import importlib
import json
import os

import pytest

from src.agent.paired_runner import save
from src.dashboard_server import Dashboard, export_csv
from src.reporting_policy import artifact_exclusion, model_exclusion


reporter = importlib.import_module("src.05_report")
analysis = importlib.import_module("src.03_analysis")


def test_sol_exclusion_does_not_exclude_luna_or_other_codex_models():
    assert model_exclusion("gpt-5.6-sol")
    assert model_exclusion("GPT-5.6 Sol")
    assert model_exclusion("gpt-5.6-luna") is None
    assert model_exclusion("gpt-6-sol") is None
    assert model_exclusion(None) is None


def test_dashboard_exclusion_precedes_all_aggregation_and_csv_export(tmp_path):
    excluded, included = "gpt-5.6-sol", "gpt-6-sol"
    case = {"instance": "ufl_small_seed000", "suite": "bd", "problem": "ufl", "size": "small", "seed": 0}
    save(tmp_path / "campaign.json", {"backend": "native", "gpt_order": [excluded, included], "agy_order": [], "max_attempts": 10, "instances": [case]})
    save(tmp_path / "status.json", {"state": "complete", "pid": os.getpid(), "active_instances": {}})
    originals = {}
    for model, passed, tokens in [(excluded, True, 1000), (included, False, 10)]:
        attempts = [{"attempt": i, "status": "ok" if passed else "verify_failed", "exec_ok": True, "verify_ok": passed, "metrics": {"generation": {"invocation_id": model, "usage": {"input_tokens": tokens}}}} for i in range(1, 11)]
        for mode in ("plain", "reflection"):
            path = tmp_path / "runs" / model / mode / "bd/ufl/ufl_small_seed000/instance_summary.json"
            save(path, {"model": model, "instance": case["instance"], "attempts": attempts, "complete": True})
            originals[path] = path.read_bytes()
    dashboard = Dashboard(tmp_path)
    result = dashboard.snapshot()
    assert [model["id"] for model in result["models"]] == [included]
    assert result["totals"]["planned"] == result["totals"]["completed"] == 1
    assert result["totals"]["first_attempt_completed"] == 1
    assert result["totals"]["passed"] == 0 and result["totals"]["pass_rate"] == 0
    assert result["usage"]["generation"]["input_tokens"] == 10
    assert [row["model"] for row in result["results"]] == [included]
    assert result["reporting_exclusions"][0]["model"] == excluded
    assert excluded.encode() not in export_csv(result)
    with pytest.raises(ValueError, match="excluded"):
        dashboard.details(excluded, case["instance"])
    assert all(path.read_bytes() == contents for path, contents in originals.items())


def test_csv_also_filters_old_cached_snapshots():
    exported = export_csv({"results": [{"model": "gpt-5.6-sol", "instance": "excluded"}, {"model": "gpt-6-sol", "instance": "kept"}]})
    assert b"gpt-5.6-sol" not in exported and b"gpt-6-sol" in exported


def test_legacy_analysis_discovery_skips_excluded_model_metadata(tmp_path):
    for model, folder in [("gpt-5.6-sol", "anonymous_old_docker_run"), ("gpt-5.6-luna", "historical_luna")]:
        path = tmp_path / folder / "ufl_small_seed000"
        save(path / "instance_summary.json", {"model": model, "attempts": []})
    excluded = tmp_path / "anonymous_old_docker_run/ufl_small_seed000"
    assert artifact_exclusion(excluded)
    assert analysis.find_run_dirs(tmp_path) == [tmp_path / "historical_luna"]
    with pytest.raises(ValueError, match="excluded"):
        analysis.analyze_instance(excluded, 0.01, write_outputs=True)
    assert not (excluded / "analysis").exists()


def test_report_rejects_excluded_run_before_writing_or_auditing(tmp_path):
    save(tmp_path / "run_config.json", {"model": "gpt-5.6-sol"})
    with pytest.raises(ValueError, match="excluded"):
        reporter.build_report(tmp_path, tmp_path / "missing_data")


@pytest.mark.parametrize("entrypoint", ["export_report", "export_attempt_curves", "plot_report", "plot_attempt_curves", "plot_attempt_breakdowns"])
def test_cached_excluded_report_cannot_bypass_figure_policy(tmp_path, entrypoint):
    destination = tmp_path / "new_figures"
    with pytest.raises(ValueError, match="excluded"):
        getattr(reporter, entrypoint)({"configuration": {"model": "gpt-5.6-sol"}}, destination)
    assert not destination.exists()


def test_dashboard_reports_drain_as_active_and_pause_as_intentional(tmp_path):
    model = "gemini-3.8-flash-medium"
    case = {"instance": "ufl_small_seed000", "suite": "bd", "problem": "ufl", "size": "small", "seed": 0}
    save(tmp_path / "campaign.json", {"backend": "native", "gpt_order": [], "agy_order": [model], "max_attempts": 10, "instances": [case]})
    state = {"state": "running", "pid": os.getpid(), "active_model": model, "active_instances": {"experiment_0": {"model": model, "instance": case["instance"], "phase": "agent"}}}
    save(tmp_path / "status.json", state)
    save(tmp_path / "pause_request.json", {"state": "draining"})
    result = Dashboard(tmp_path).snapshot()
    assert result["state"] == "draining_for_pause" and result["totals"]["active"] == 1
    state.update(state="paused", active_instances={})
    save(tmp_path / "status.json", state)
    result = Dashboard(tmp_path).snapshot()
    assert result["state"] == "paused" and result["totals"]["active"] == 0
    assert not result["alerts"]
