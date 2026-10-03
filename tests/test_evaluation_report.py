"""Coverage and metric checks for the public result exporter."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


reporter = importlib.import_module("src.05_report")


@pytest.fixture
def matrix(tmp_path):
    root = tmp_path / "run"
    data_root = tmp_path / "data"
    root.mkdir()
    directory = data_root / "bd" / "ufl" / "ufl_small_seed000"
    directory.mkdir(parents=True)
    (directory / "data.json").write_text(json.dumps({"m": 2, "n": 1, "f": [2, 3], "D": [[4], [1]]}))
    (directory / "meta.json").write_text('{"ground_truth_objective":4}')
    (data_root / "manifest.json").write_text(json.dumps({"instances": [{"complete": True, "suite": "bd", "problem": "ufl", "instance": directory.name}]}))
    configuration = {"backend": "codex-cli", "model": None, "max_attempts": 10, "agent_timeout": 900, "solver_timeout": 600, "gap_threshold": 0.01, "python": "test", "platform": "test"}
    (root / "run_config.json").write_text(json.dumps(configuration))
    return root, data_root, directory


def write_case(matrix, mode="plain", complete=True):
    root, _, canonical = matrix
    destination = root / mode / "bd" / "ufl" / canonical.name
    directory = destination / "attempts" / "attempt_01"
    directory.mkdir(parents=True)
    (destination / "data.json").write_text((canonical / "data.json").read_text())
    first = {"attempt": 1, "status": "ok", "exec_ok": True, "verify_ok": True, "objective": 4, "gap_to_gold": 0, "llm_verdict": "pass", "reused_from_previous": False,
             "metrics": {"generation": {"usage": {"input_tokens": 10, "output_tokens": 1}, "elapsed_seconds": 2}, "algorithm_verification": {"usage": {"input_tokens": 5}, "elapsed_seconds": 1}}}
    attempts = [first]
    if complete:
        attempts += [{**first, "attempt": index, "status": "skipped_previous_solved", "reused_from_previous": True, "first_success_attempt": 1, "metrics": {}} for index in range(2, 11)]
    reflection = reporter.MODE_FLAGS[mode]
    summary = {"instance": canonical.name, "suite": "bd", "problem": "ufl", "reflection": reflection, "max_attempts": 10, "gap_threshold": 0.01, "gold_objective": 4, "attempts": attempts}
    (destination / "instance_summary.json").write_text(json.dumps(summary))
    (directory / "summary.json").write_text(json.dumps(first))
    (directory / "solver.py").write_text("# synthetic fixture\n")
    (directory / "agent_invocation.json").write_text(json.dumps({"ephemeral": True, "ignore_user_config": True, "project_doc_max_bytes": 0}))
    (directory / "alg_verifier.md").write_text("VERDICT: PASS\nNOTES: synthetic review\n")
    (directory / "output.json").write_text(json.dumps({"y": [0, 1], "x": [[0], [1]], "objective_value": 4}))
    return destination


def test_quality_requires_verification_and_reference_gap():
    attempts = [{"attempt": 1, "exec_ok": True, "verify_ok": False, "objective": 1},
                {"attempt": 2, "exec_ok": True, "verify_ok": True, "objective": 12},
                {"attempt": 3, "exec_ok": True, "verify_ok": True, "objective": 10}]
    assert reporter.attempt_scores(attempts, 10, 0.01, 1) == {"execution": True, "verification": False, "quality": False, "best_verified_gap": None}
    assert not reporter.attempt_scores(attempts, 10, 0.01, 2)["quality"]
    assert reporter.attempt_scores(attempts, 10, 0.01, 3)["quality"]


def test_cumulative_curves_find_exact_first_full_round_and_keep_modes_separate():
    histories = [("plain", [{"attempt": 1, "exec_ok": True}, {"attempt": 2, "exec_ok": True, "verify_ok": True, "objective": 12}, {"attempt": 3, "exec_ok": True, "verify_ok": True, "objective": 10}]),
                 ("plain", [{"attempt": 1, "exec_ok": False}, {"attempt": 2, "exec_ok": True}, {"attempt": 3, "exec_ok": True, "verify_ok": True, "objective": 12}, {"attempt": 4, "exec_ok": True, "verify_ok": True, "objective": 10}]),
                 ("reflection", [{"attempt": 1, "exec_ok": True, "verify_ok": True, "objective": 12}])]
    instances = [{"mode": mode, "attempt_metrics": {str(budget): reporter.attempt_scores(attempts, 10, 0.01, budget) for budget in range(1, 6)}} for mode, attempts in histories]
    plain, reflection = reporter.cumulative_attempt_curves(instances, 5)
    assert plain["num_instances"] == 2
    assert plain["points"][0]["counts"] == {"execution": 1, "verification": 0, "quality": 0}
    assert plain["points"][0]["execution"] == 0.5
    assert plain["first_full_round"] == {"execution": 2, "verification": 3, "quality": 4}
    assert plain["first_all_full_round"] == 4
    assert plain["points"][3]["counts"] == plain["points"][4]["counts"]
    assert reflection["num_instances"] == 1
    assert reflection["first_full_round"] == {"execution": 1, "verification": 1, "quality": None}
    assert reflection["first_all_full_round"] is None


def test_breakdowns_use_each_family_size_and_policy_denominator():
    scores = {"1": {"execution": True, "verification": True, "quality": True}}
    instances = [{"mode": "plain", "suite": "bd", "problem": "ufl", "size": "small", "attempt_metrics": scores},
                 {"mode": "plain", "suite": "bd", "problem": "ufl", "size": "small", "attempt_metrics": scores},
                 {"mode": "plain", "suite": "cg", "problem": "bp", "size": "medium", "attempt_metrics": scores},
                 {"mode": "reflection", "suite": "bd", "problem": "ufl", "size": "small", "attempt_metrics": scores}]
    groups = reporter.attempt_breakdowns(instances, 1)
    assert [group["group"] for group in groups["size"]] == ["small", "medium"]
    assert [curve["num_instances"] for curve in groups["suite"][0]["curves"]] == [2, 1]
    assert [curve["num_instances"] for curve in groups["problem"][0]["curves"]] == [2, 1]
    assert groups["size"][1]["curves"][0]["num_instances"] == 1
    assert groups["size"][0]["curves"][0]["points"][0]["counts"]["quality"] == 2


def test_partial_cases_are_excluded_from_denominator(matrix):
    write_case(matrix, "plain")
    write_case(matrix, "reflection", complete=False)
    report = reporter.build_report(matrix[0], matrix[1])
    assert report["coverage"]["expected_instance_runs"] == 2
    assert report["coverage"]["completed_instance_runs"] == 1
    assert len(report["coverage"]["partial_instance_runs"]) == 1
    assert not report["coverage"]["complete"]
    assert report["overall"][0]["num_instances"] == 1
    assert len(report["attempt_curves"]) == 1
    assert report["attempt_curves"][0]["num_instances"] == 1


def test_interrupted_backups_do_not_duplicate_instances(matrix):
    destination = write_case(matrix)
    backup = destination.with_name(destination.name + ".interrupted_1")
    backup.mkdir()
    (backup / "instance_summary.json").write_text((destination / "instance_summary.json").read_text())
    report = reporter.build_report(matrix[0], matrix[1])
    assert report["coverage"]["completed_instance_runs"] == 1


def test_input_tampering_is_an_audit_error(matrix):
    destination = write_case(matrix)
    (destination / "data.json").write_text("{}")
    report = reporter.build_report(matrix[0], matrix[1])
    assert not report["coverage"]["complete"]
    assert report["coverage"]["completed_instance_runs"] == 0
    assert any("runtime input differs" in issue for row in report["audit_errors"] for issue in row["issues"])


def test_reused_results_require_qualifying_success(matrix):
    destination = write_case(matrix)
    path = destination / "instance_summary.json"
    summary = json.loads(path.read_text())
    summary["attempts"][0]["verify_ok"] = False
    path.write_text(json.dumps(summary))
    report = reporter.build_report(matrix[0], matrix[1])
    assert any("reused without" in issue for row in report["audit_errors"] for issue in row["issues"])


def test_complete_report_counts_actual_calls_only(matrix, tmp_path):
    for mode in reporter.MODE_FLAGS:
        write_case(matrix, mode)
    report = reporter.build_report(matrix[0], matrix[1])
    assert report["coverage"]["complete"] and not report["audit_errors"]
    assert len(report["rows"]) == 2
    assert all(instance["actual_attempts"] == 1 and instance["reused_slots"] == 9 for instance in report["instances"])
    assert all(instance["token_usage"]["input_tokens"] == 15 for instance in report["instances"])
    destination = tmp_path / "published"
    reporter.export_report(report, destination, charts=False)
    assert (destination / "evaluation.csv").read_text().count("\n") == 7
    assert "Complete; 2/2" in (destination / "report.md").read_text()
    assert (destination / "attempt_curves.csv").read_text().count("\n") == 21
    assert all(curve["first_all_full_round"] == 1 for curve in json.loads((destination / "attempt_curves.json").read_text())["attempt_curves"])


def test_report_export_never_modifies_repository_readme(matrix, tmp_path, monkeypatch):
    for mode in reporter.MODE_FLAGS:
        write_case(matrix, mode)
    report = reporter.build_report(matrix[0], matrix[1])
    readme = tmp_path / "README.md"
    readme.write_text("Repository documentation")
    monkeypatch.chdir(tmp_path)
    reporter.export_report(report, tmp_path / "outputs" / "reports", charts=False)
    assert readme.read_text() == "Repository documentation"
    assert not (tmp_path / "outputs" / "reports" / "README.md").exists()


def test_report_coverage_respects_selected_setting(matrix):
    root, data_root, _ = matrix
    configuration = json.loads((root / "run_config.json").read_text())
    configuration["scope"] = {"modes": ["plain"], "suite": "bd", "instances_per_problem": 1}
    (root / "run_config.json").write_text(json.dumps(configuration))
    write_case(matrix, "plain")
    report = reporter.build_report(root, data_root)
    assert report["coverage"]["complete"]
    assert report["coverage"]["expected_instance_runs"] == 1


def test_short_experiment_reports_only_observed_budgets(matrix):
    root, data_root, _ = matrix
    configuration = json.loads((root / "run_config.json").read_text())
    configuration["max_attempts"] = 1
    configuration["scope"] = {"modes": ["plain"], "suite": "bd"}
    (root / "run_config.json").write_text(json.dumps(configuration))
    destination = write_case(matrix, "plain", complete=False)
    path = destination / "instance_summary.json"
    summary = json.loads(path.read_text())
    summary["max_attempts"] = 1
    path.write_text(json.dumps(summary))
    report = reporter.build_report(root, data_root)
    assert report["coverage"]["complete"]
    assert list(report["rows"][0]["metrics"]) == ["1"]
    assert [point["budget"] for point in report["attempt_curves"][0]["points"]] == [1]
    table = reporter.markdown_table(report["rows"])
    assert "solveQ@1" in table and "@1/5/10" not in table
