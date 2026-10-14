"""Runtime and readiness evidence must cover the exact prepared code and image."""
import hashlib
import json

import pytest

from src import execution_environment as environment
from src.runtime_environment import RuntimeEnvironmentError


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_prepared_manifest_is_checked_before_use(tmp_path, monkeypatch):
    manifest = tmp_path / "runtime.json"
    write(manifest, {"fixture": True})
    called = []
    monkeypatch.setattr(environment, "validate_execution_environment", called.append)
    plan = {"runtime_manifest": "runtime.json", "runtime_manifest_sha256": environment.digest(manifest)}
    result = environment.load_execution_environment(plan, tmp_path / "plan.json")
    assert called == [{"fixture": True}] and result["evidence_root"] == str(tmp_path)
    write(manifest, {"changed": True})
    with pytest.raises(RuntimeEnvironmentError, match="manifest changed"):
        environment.load_execution_environment(plan, tmp_path / "plan.json")


def readiness(tmp_path):
    backend = "codex-cli"
    root = tmp_path / "evidence"
    run = root / "readiness" / backend / "physical-call"
    attempt = run / "case/attempts/attempt_01"
    attempt.mkdir(parents=True)
    (attempt / "solver.py").write_text("synthetic supplied program")
    output = {"status": "optimal", "objective_value": 1.0, "threads": 1}
    write(attempt / "output.json", output)
    config = {"image_id": "sha256:fixture", "worker_files_sha256": {"worker.py": "fixture"}, "native_environment": {"pinned": True}, "evidence_root": str(root)}
    report = {"passed": True, "backend": backend, "image_id": config["image_id"], "worker_files_sha256": config["worker_files_sha256"], "native_environment_sha256": hashlib.sha256(json.dumps(config["native_environment"], sort_keys=True).encode()).hexdigest()}
    write(run / "report.json", report)
    write(root / "readiness/codex-cli.json", {"report": str(run / "report.json"), "sha256": environment.digest(run / "report.json")})
    write(run / "call/agent_metrics.json", {"ok": True, "artifact_sha256": environment.digest(attempt / "solver.py")})
    write(run / "call/agent_invocation.json", {"cleanup_verified": True, "image_id": config["image_id"]})
    write(attempt / "native_solver_receipt.json", {"request": {"environment": config["native_environment"]}, "result": {"exec_ok": True, "cleanup_verified": True, "integrity_violations": [], "output": output, "output_sha256": environment.digest(attempt / "output.json")}})
    return config, run


def test_ready_backend_requires_its_actual_handoff_artifacts(tmp_path):
    config, run = readiness(tmp_path)
    assert set(environment.require_backend_readiness(config, ["codex-cli"])) == {"codex-cli"}
    with pytest.raises(RuntimeEnvironmentError, match="pending for agy-cli"):
        environment.require_backend_readiness(config, ["agy-cli", "codex-cli"])


@pytest.mark.parametrize("changed", ["report.json", "case/attempts/attempt_01/solver.py", "case/attempts/attempt_01/output.json"])
def test_modified_readiness_records_block_dispatch(tmp_path, changed):
    config, run = readiness(tmp_path)
    (run / changed).write_text("changed")
    with pytest.raises(RuntimeEnvironmentError):
        environment.require_backend_readiness(config, ["codex-cli"])
