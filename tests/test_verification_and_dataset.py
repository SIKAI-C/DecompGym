"""Output verification, objective status, and dataset reuse."""
from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest

from src.experiments.verification import OutputVerifier
from src.utils.dataset import ensure_suite


def ufl_files(directory: Path, output: dict):
    directory.mkdir(parents=True, exist_ok=True)
    data = directory / "data.json"
    result = directory / "output.json"
    data.write_text(json.dumps({"m": 2, "n": 1, "f": [2, 3], "D": [[4], [1]]}))
    result.write_text(json.dumps({"status": "feasible", "method": "benders_decomposition", **output}))
    return data, result


def test_fractional_binary_decisions_fail(tmp_path):
    data, output = ufl_files(tmp_path, {"y": [0.5, 0.5], "x": [[0.5], [0.5]], "objective_value": 5})
    result = OutputVerifier.verify_output("bd", "ufl", data, output, 1e-6)
    assert not result["ok"] and any("binary" in issue for issue in result["issues"])


def test_objective_is_recomputed(tmp_path):
    data, output = ufl_files(tmp_path, {"y": [0, 1], "x": [[0], [1]], "objective_value": 0})
    result = OutputVerifier.verify_output("bd", "ufl", data, output, 1e-6)
    assert not result["ok"] and result["objective_value"] == 4


def test_valid_solution_passes(tmp_path):
    data, output = ufl_files(tmp_path, {"y": [0, 1], "x": [[0], [1]], "objective_value": 4})
    assert OutputVerifier.verify_output("bd", "ufl", data, output, 1e-6)["ok"]


def test_nonfinite_solution_fails(tmp_path):
    data, output = ufl_files(tmp_path, {"y": [0, 1], "x": [[0], [float("nan")]], "objective_value": 4})
    assert not OutputVerifier.verify_output("bd", "ufl", data, output, 1e-6)["ok"]


def test_zero_objective_and_missing_reference():
    assert OutputVerifier.extract_objective({"objective_value": 0})[0] == 0
    status, gap = OutputVerifier.assess_status(exec_ok=True, output={}, verify_ok=True, objective=0, gold_obj=None, gap_threshold=0.01)
    assert status == "reference_unavailable" and gap is None


def test_timeout_is_not_reported_as_missing_objective():
    status, _ = OutputVerifier.assess_status(exec_ok=False, output=None, verify_ok=False, objective=None, gold_obj=4, gap_threshold=0.01, execution_timed_out=True)
    assert status == 'timeout'


def complete_instance(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ["data.json", "meta.json", "gold_monolithic.json", "gold_cg.json"]:
        (directory / name).write_text(json.dumps({"best_objective": 1}))
    (directory / "prompt.txt").write_text("prompt")


def test_cached_dataset_needs_no_solver_import(tmp_path, monkeypatch):
    directory = tmp_path / "bp_small_seed000"
    complete_instance(directory)
    monkeypatch.setattr("src.utils.dataset.importlib.import_module", lambda *_: pytest.fail("cached data imported a solver"))
    records = ensure_suite("cg", "bp", tmp_path, ["small"], 1)
    assert records[0]["reused"]


def test_missing_instance_generated_once(tmp_path, monkeypatch):
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        complete_instance(tmp_path / "bp_small_seed000")
    monkeypatch.setattr("src.utils.dataset.importlib.import_module", lambda *_: SimpleNamespace(generate_suite=generate))
    assert not ensure_suite("cg", "bp", tmp_path, ["small"], 1)[0]["reused"]
    assert ensure_suite("cg", "bp", tmp_path, ["small"], 1)[0]["reused"]
    assert len(calls) == 1


def test_strict_cg_validation_compares_cg_objective(tmp_path):
    directory = tmp_path / "cg" / "bp" / "bp_small_seed000"
    complete_instance(directory)
    assert importlib.import_module("src.01_gen").validate_instance(directory)["status"] == "pass_simple_test"


def test_positive_decomposition_gap_survives_bad_instance_cleanup(tmp_path):
    directory = tmp_path / "cg" / "bp" / "bp_small_seed000"
    complete_instance(directory)
    (directory / "gold_cg.json").write_text('{"best_objective":2}')
    generator = importlib.import_module("src.01_gen")
    result = generator.validate_instance(directory)
    assert result["status"] == "reference_gap" and result["decomposition_gap"] == 1
    assert generator.remove_instances(tmp_path, [result], remove_mismatch=True) == 0
    assert directory.exists()


def test_objective_better_than_reference_is_flagged(tmp_path):
    directory = tmp_path / "cg" / "bp" / "bp_small_seed000"
    complete_instance(directory)
    (directory / "gold_cg.json").write_text('{"best_objective":0}')
    assert importlib.import_module("src.01_gen").validate_instance(directory)["status"] == "objective_mismatch"
