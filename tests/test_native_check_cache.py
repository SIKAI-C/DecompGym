"""scripts/reference_native_check.py reuses a cached case only under an unchanged acceptance harness and environment.

No sandbox is started and no optimizer is imported: the cached-case path returns before any execution.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("reference_native_check", ROOT / "scripts/reference_native_check.py")
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

ENVIRONMENT = {"python_runtime_binary": "/x/python", "site_packages": "/x/site", "gurobi": "11.0.3"}


def test_the_harness_identity_covers_adapter_sandbox_verifier_contract_and_scoring():
    identity = check.harness_identity("ufl", ENVIRONMENT)
    for path in ("scripts/reference_native_check.py", "src/experiments/sandbox.py", "src/experiments/environment.py",
                 "src/agent/safe_artifacts.py", "src/experiments/verification.py", "src/experiments/solution_checks.py",
                 "src/utils/output_contract.py", "src/utils/families.py", "src/bd/ufl/d_verifier.py"):
        assert identity["files"][path] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), path
    assert "src/cg/rcsp/d_verifier.py" in check.harness_identity("rcsp", ENVIRONMENT)["files"]
    assert identity["native_environment_sha256"] == check.environment_digest(ENVIRONMENT)
    assert check.harness_identity("ufl", {**ENVIRONMENT, "gurobi": "12.0.0"}) != identity


def cached_case(tmp_path, monkeypatch, *, harness):
    """A dataset with one instance and a matching cached calibration record written under `harness`."""
    dataset = tmp_path / "dataset"
    source = dataset / "bd" / "ufl" / "ufl_small_seed000"
    source.mkdir(parents=True)
    for name, text in (("data.json", '{"m": 1}'), ("meta.json", "{}"), ("gold_monolithic.json", '{"best_objective": 1}')):
        (source / name).write_text(text)
    monkeypatch.setattr(check, "DATASET", dataset)
    destination = tmp_path / "out"
    case = destination / "formal-v2" / "bd" / "ufl" / "ufl_small_seed000"
    attempt = case / "attempts" / "attempt_01"
    attempt.mkdir(parents=True)
    solver, inventory = check.solver_source("ufl", "formal-v2")
    (attempt / "solver.py").write_text(solver)
    (attempt / "output.json").write_text("{}")
    (case / "source_inventory.json").write_text(json.dumps({"original_sources": inventory}))
    record = {"input_sha256": check.digest(source / "data.json"), "solver_sha256": hashlib.sha256(solver.encode()).hexdigest(),
              "scoring_inputs_sha256": check.scoring_inputs(source), "output_sha256": check.digest(attempt / "output.json"), "numeric_status": "ok"}
    if harness is not None:
        record["harness_identity"] = harness
    (case / "calibration.json").write_text(json.dumps(record))
    return destination, record


def test_an_unchanged_harness_reuses_the_cached_case(tmp_path, monkeypatch):
    destination, record = cached_case(tmp_path, monkeypatch, harness=check.harness_identity("ufl", ENVIRONMENT))
    executor = SimpleNamespace(environment=ENVIRONMENT)  # not callable: a cached case is never executed
    assert check.run_case(executor, destination, "ufl", "ufl_small_seed000", "formal-v2") == record


@pytest.mark.parametrize("change", ["environment", "harness_file", "unrecorded"])
def test_a_changed_or_unrecorded_harness_refuses_the_cache(tmp_path, monkeypatch, change):
    harness = check.harness_identity("ufl", ENVIRONMENT)
    if change == "harness_file":
        harness = {**harness, "files": {**harness["files"], "src/experiments/verification.py": "0" * 64}}
    destination, _ = cached_case(tmp_path, monkeypatch, harness=None if change == "unrecorded" else harness)
    environment = {**ENVIRONMENT, "gurobi": "12.0.0"} if change == "environment" else ENVIRONMENT
    with pytest.raises(RuntimeError, match="Acceptance harness"):
        check.run_case(SimpleNamespace(environment=environment), destination, "ufl", "ufl_small_seed000", "formal-v2")
