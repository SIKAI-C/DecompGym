"""License words trigger a trusted check, never determine responsibility alone."""
import json

import pytest

from src.agent import native_solver
from src.runtime_environment import RuntimeEnvironmentError


def fixture(tmp_path, monkeypatch):
    task = object.__new__(native_solver.NativeSandboxExecutor)
    task.environment = {"fixture": "environment"}
    monkeypatch.setattr(native_solver, "native_environment", lambda: task.environment)
    work = tmp_path / "case/attempts/attempt_01"
    work.mkdir(parents=True)
    (work / "solver.py").write_text("candidate")
    (work.parent.parent / "data.json").write_text("{}")
    return task, work


def record(ok, stderr="", *, output=None, violations=None):
    output = output if output is not None else ({"objective_value": 4} if ok else None)
    return {"exec_ok": ok, "stdout": "Gurobi banner", "stderr": stderr, "output": output, "output_text": json.dumps(output) if output else None,
            "integrity_violations": violations or [], "execution_seconds": 7.5}


@pytest.mark.parametrize("ok,error,violations,check_needed", [(True, "", [], False), (True, "Gurobi license log", [], False), (False, "ordinary pathlib error", [], False), (False, "license.json is missing", [], True), (False, "Gurobi unknown parameter", [], True), (False, "Gurobi failure", ["data.json"], False)])
def test_success_and_ordinary_errors_skip_diagnostic_and_false_clues_still_fail_candidate(tmp_path, monkeypatch, ok, error, violations, check_needed):
    task, work = fixture(tmp_path, monkeypatch)
    calls = []
    def execute(solver, *args, **kwargs):
        calls.append(solver)
        if solver == b"candidate":
            return record(ok, error, violations=violations)
        assert b"gp.Env(empty=True)" in solver
        return record(True, output={"status": "optimal", "objective_value": 1.0, "threads": 1})
    monkeypatch.setattr(task, "execute_bytes", execute)
    result = task(work, 60)
    assert result[0] == ok and len(calls) == 1 + check_needed
    receipt = json.loads((work / "native_solver_receipt.json").read_text())
    assert receipt["result"]["execution_seconds"] == 7.5
    task(work, 60)
    assert len(calls) == 1 + check_needed


@pytest.mark.parametrize("probe_result", [record(False, "license unavailable"), record(True, output={"status": "optimal", "objective_value": 2, "threads": 1})])
def test_failed_trusted_diagnostic_stops_without_committing_candidate(tmp_path, monkeypatch, probe_result):
    task, work = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(task, "execute_bytes", lambda code, *args, **kwargs: record(False, "Gurobi failure") if code == b"candidate" else probe_result)
    with pytest.raises(RuntimeEnvironmentError, match="independent Gurobi diagnostic failed"):
        task(work, 60)
    assert not (work / "native_solver_receipt.json").exists()
    evidence = json.loads(next(work.glob("native_execution_*.json")).read_text())
    assert evidence["gurobi_diagnostic"]["ok"] is False


def test_environment_drift_does_not_touch_existing_output(tmp_path, monkeypatch):
    task, work = fixture(tmp_path, monkeypatch)
    (work / "output.json").write_text("preserved")
    monkeypatch.setattr(native_solver, "native_environment", lambda: {"fixture": "changed"})
    with pytest.raises(RuntimeEnvironmentError, match="changed during execution"):
        task(work, 60)
    assert (work / "output.json").read_text() == "preserved"


def test_equivalent_interpreter_spelling_is_not_software_drift():
    original = {"python_executable": "/project/v2/../main/.venv/bin/python", "python_binary": "/installed/python", "site_packages": "/project/main/.venv/lib/site-packages"}
    assert native_solver.same_environment(original, original | {"python_executable": "/project/main/.venv/bin/python"})
    assert not native_solver.same_environment(original, original | {"python_binary": "/different/python"})
    assert not native_solver.same_environment(original, original | {"site_packages": "/different/packages"})
