"""Load frozen hybrid runtime evidence and verify backend readiness before dispatch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.agent.docker_client import CLI_VERSIONS, DOCKER_EDIT_POLICY, worker_file_hashes
from src.agent.native_solver import native_environment, same_environment
from src.runtime_environment import RuntimeEnvironmentError


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_execution_environment(plan, plan_path):
    path = (Path(plan_path).resolve().parent / plan["runtime_manifest"]).resolve()
    if digest(path) != plan["runtime_manifest_sha256"]:
        raise RuntimeEnvironmentError("Prepared runtime manifest changed; review and freeze the new environment")
    value = json.loads(path.read_text())
    validate_execution_environment(value)
    return value | {"evidence_root": str(path.parent), "manifest_sha256": digest(path)}


def validate_execution_environment(value):
    if value.get("agent_policy") != DOCKER_EDIT_POLICY or value.get("cli_versions") != CLI_VERSIONS or value.get("worker_files_sha256") != worker_file_hashes():
        raise RuntimeEnvironmentError("Docker worker sources or CLI declarations differ from the prepared environment")
    profile = Path(__file__).resolve().parents[1] / "docker/agents/seccomp-userns.json"
    if digest(profile) != value.get("seccomp_profile_sha256") or not same_environment(native_environment(), value.get("native_environment", {})):
        raise RuntimeEnvironmentError("The sandbox or native environment differs from its frozen evidence")


def require_backend_readiness(value, backends):
    proofs = {}
    for backend in sorted(set(backends)):
        pointer = Path(value["evidence_root"]) / "readiness" / (backend + ".json")
        if not pointer.is_file():
            raise RuntimeEnvironmentError(f"Docker readiness is pending for {backend}; complete its private authentication/handoff check")
        entry = json.loads(pointer.read_text())
        path = Path(entry["report"])
        if digest(path) != entry["sha256"]:
            raise RuntimeEnvironmentError("Backend readiness evidence changed")
        report = json.loads(path.read_text())
        native_hash = hashlib.sha256(json.dumps(value["native_environment"], sort_keys=True).encode()).hexdigest()
        if not report.get("passed") or report.get("backend") != backend or report.get("image_id") != value["image_id"] or report.get("worker_files_sha256") != value["worker_files_sha256"] or report.get("native_environment_sha256") != native_hash:
            raise RuntimeEnvironmentError(f"Backend readiness does not cover the frozen configuration: {backend}")
        call = path.parent / "call"
        attempt = path.parent / "case/attempts/attempt_01"
        metrics = json.loads((call / "agent_metrics.json").read_text())
        invocation = json.loads((call / "agent_invocation.json").read_text())
        receipt = json.loads((attempt / "native_solver_receipt.json").read_text())
        if not metrics.get("ok") or metrics.get("artifact_sha256") != digest(attempt / "solver.py") or not invocation.get("cleanup_verified") or invocation.get("image_id") != value["image_id"]:
            raise RuntimeEnvironmentError("Backend handoff artifact does not match its actual invocation")
        result = receipt["result"]
        if receipt["request"]["environment"] != value["native_environment"] or not result["exec_ok"] or not result["cleanup_verified"] or result["integrity_violations"] or result["output"] != {"status": "optimal", "objective_value": 1.0, "threads": 1} or result["output_sha256"] != digest(attempt / "output.json"):
            raise RuntimeEnvironmentError("Backend handoff lacks matching native execution evidence")
        proofs[backend] = {"report": str(path), "sha256": entry["sha256"]}
    return proofs
