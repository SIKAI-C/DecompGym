"""Version drift must stop execution before candidate output is changed or scored."""
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from src import runtime_environment as runtime
from src.agent.experiment_runner import ExperimentRunner
from src.runtime_environment import RuntimeEnvironmentError


def test_installation_and_frozen_runtime_pins_agree():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    assert project["requires-python"] == "==" + runtime.PYTHON_VERSION
    assert (root / ".python-version").read_text().strip() == runtime.PYTHON_VERSION
    assert "gurobipy==" + runtime.GUROBI_VERSION in project["dependencies"]


@pytest.mark.parametrize("component", runtime.VERSION_FIELDS)
def test_each_version_mismatch_blocks_execution(monkeypatch, component):
    if component == "python_version":
        monkeypatch.setattr(runtime.platform, "python_version", lambda: "3.12.12")
    elif component == "gurobipy_version":
        monkeypatch.setattr(runtime, "version", lambda _: "11.0.2")
    else:
        fake_gurobi = SimpleNamespace(gurobi=SimpleNamespace(version=lambda: (12, 0, 0)))
        monkeypatch.setattr(runtime, "importlib", SimpleNamespace(import_module=lambda _: fake_gurobi))
    with pytest.raises(RuntimeEnvironmentError, match="is required"):
        runtime.require_runtime_environment()


def test_missing_package_is_an_environment_error(monkeypatch):
    def missing(_):
        raise runtime.PackageNotFoundError("gurobipy")
    monkeypatch.setattr(runtime, "version", missing)
    with pytest.raises(RuntimeEnvironmentError, match="could not be loaded"):
        runtime.require_runtime_environment()


def test_interpreter_retargeted_during_run_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_PYTHON_BINARY", tmp_path / "original-python")
    with pytest.raises(RuntimeEnvironmentError, match="executable changed"):
        runtime.require_runtime_environment()


def test_versions_are_checked_against_the_campaign_record():
    observed = runtime.require_runtime_environment()
    assert runtime.require_runtime_environment(observed) == observed
    with pytest.raises(RuntimeEnvironmentError, match="frozen campaign"):
        runtime.require_runtime_environment(observed | {"python_version": "3.12.10"})


def test_solver_drift_does_not_launch_or_overwrite_output(tmp_path, monkeypatch):
    (tmp_path / "solver.py").write_text("raise AssertionError('must not execute')\n")
    output = tmp_path / "output.json"
    output.write_text('{"saved":"evidence"}\n')
    monkeypatch.setattr(runtime, "version", lambda _: "12.0.0")
    with pytest.raises(RuntimeEnvironmentError):
        ExperimentRunner.run_solver(tmp_path, 1)
    assert output.read_text() == '{"saved":"evidence"}\n'
    assert not (tmp_path / "gurobi.env").exists()


def test_runner_rejects_wrong_environment_before_generation(monkeypatch):
    monkeypatch.setattr(runtime.platform, "python_version", lambda: "3.13.5")
    with pytest.raises(RuntimeEnvironmentError, match="Python"):
        ExperimentRunner()
