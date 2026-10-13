"""Prepare immutable runtime evidence and private authentication without starting a campaign."""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path

from src.agent.container_runtime import DockerRuntime
from src.agent.docker_client import AUTH_VOLUMES, CLI_VERSIONS, DOCKER_EDIT_POLICY, DockerCliClient, IMAGE_TAG, credential_from_environment, worker_file_hashes
from src.agent.native_solver import GUROBI_PROBE, NativeSandboxExecutor


BACKENDS = {"codex": "codex-cli", "claude": "claude-code-cli", "agy": "agy-cli"}
READINESS_MODELS = {"codex-cli": "gpt-6.1-sol", "claude-code-cli": "claude-sonnet-5-5", "agy-cli": "gemini-3.8-flash-medium"}


def runtime():
    return DockerRuntime(seccomp_profile=Path(__file__).resolve().parents[1] / "docker/agents/seccomp-userns.json")


def prepare(evidence_root, *, image=IMAGE_TAG):
    engine = runtime()
    image_id = engine.image_id(image)
    client = DockerCliClient("preflight", backend="codex-cli", image_id=image_id, runtime=engine, evidence_root=evidence_root)
    client.prepare()
    native = NativeSandboxExecutor(license_file=Path.home() / "gurobi.lic")
    version = engine.command("version", "--format", "{{json .}}")
    if version.returncode:
        raise RuntimeError("Docker version evidence is unavailable")
    docker_version = json.loads(version.stdout)
    engine_id = engine.command("info", "--format", "{{.ID}}")
    context = engine.command("context", "show")
    endpoint = engine.command("context", "inspect", context.stdout.strip(), "--format", "{{.Endpoints.docker.Host}}")
    if engine_id.returncode or context.returncode or endpoint.returncode or not endpoint.stdout.strip().startswith("unix://"):
        raise RuntimeError("The local Docker engine identity could not be established")
    manifest = {"schema_version": 1, "agent_policy": DOCKER_EDIT_POLICY, "image_id": image_id, "cli_versions": CLI_VERSIONS,
                "worker_files_sha256": worker_file_hashes(), "seccomp_profile_sha256": hashlib.sha256(engine.seccomp_profile.read_bytes()).hexdigest(),
                "docker_client_version": docker_version["Client"]["Version"], "docker_server_version": docker_version["Server"]["Version"],
                "docker_engine_id": engine_id.stdout.strip(), "docker_context": context.stdout.strip(), "docker_endpoint": endpoint.stdout.strip(),
                "native_environment": native.environment, "license_file": str(native.license_file), "auth_volumes": AUTH_VOLUMES,
                "authentication_validation": "separate; image preflight does not prove authenticated inference", "campaign_started": False}
    evidence_root.mkdir(parents=True, exist_ok=True)
    path = evidence_root / "runtime.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"runtime_manifest": str(path), "image_id": image_id, "image_preflight_passed": True, "campaign_started": False}, indent=2))
    return manifest


def interactive_credential(backend):
    if backend == "agy-cli":
        if os.environ.get("DECOMPBENCH_AGY_KEYRING_PASSPHRASE"):
            return credential_from_environment(backend)
        if not os.isatty(0):
            raise RuntimeError("Run AGY authentication in your own local terminal for the hidden passphrase prompt")
        return {"keyring_passphrase": getpass.getpass("Existing benchmark keyring passphrase (hidden; not your Google password): ")}
    credential = credential_from_environment(backend)
    if backend == "claude-code-cli" and not credential:
        if not os.isatty(0):
            return None  # A prior private-volume setup may already be sufficient.
        token = getpass.getpass("Claude subscription OAuth token (hidden; Enter reuses the private volume): ")
        return {"oauth_token": token} if token else None
    return credential


def authenticate(backend, evidence_root, *, image=IMAGE_TAG, verify=False):
    engine = runtime()
    frozen = evidence_root / "runtime.json"
    if image == IMAGE_TAG and frozen.exists():
        prepared = json.loads(frozen.read_text())
        image = prepared["image_id"]
        if "docker_endpoint" in prepared:
            engine.bind_environment(prepared)
    image_id = engine.image_id(image)
    client = DockerCliClient("auth-check", backend=backend, image_id=image_id, runtime=engine, evidence_root=evidence_root)
    mounts = client.prepare()
    credential = interactive_credential(backend)
    import uuid
    result = engine.run(image_id, ["auth-check", backend], payload={"credential": credential}, state_path=evidence_root / "authentication" / uuid.uuid4().hex / "container.json", timeout=60, network="none", volumes=mounts)
    if result.returncode:
        raise RuntimeError("Private authentication setup did not complete; no inference was submitted")
    print(json.dumps({"backend": backend, "credential_loaded": True, "inference_submitted": False, "note": "AGY's passphrase must also be supplied privately to the launch environment"}, indent=2))
    if verify:
        verify_backend(backend, evidence_root, engine=engine, image_id=image_id, credential=credential)


def verify_backend(backend, evidence_root, *, engine, image_id, credential):
    import uuid
    directory = evidence_root / "readiness" / backend / uuid.uuid4().hex
    case = directory / "case"
    attempt = case / "attempts/attempt_01"
    attempt.mkdir(parents=True)
    (case / "data.json").write_text('{"minimum":1}\n')
    target = attempt / "solver.py"
    target.write_text("")
    program = GUROBI_PROBE.replace("THREADS", "1").replace("x >= 1", "x >= json.loads(Path('../../data.json').read_text())['minimum']")
    prompt = "Copy the following provided Python program into solver.py. This is a file-handoff diagnostic, not a benchmark task. Do not execute it. It will run later in a separate runtime directory with ../../data.json.\n\n" + program
    client = DockerCliClient(READINESS_MODELS[backend], backend=backend, image_id=image_id, runtime=engine, evidence_root=evidence_root, role="readiness", effort="high" if backend == "claude-code-cli" else "medium", credential_provider=lambda _: credential)
    result = client.edit_file(target, prompt, context=[case / "data.json"], log_dir=directory / "call", step="agent")
    report = {"schema_version": 1, "backend": backend, "model": client.model, "image_id": image_id, "worker_files_sha256": worker_file_hashes(), "edit_ok": result.ok,
              "execution_ok": False, "passed": False, "purpose": "synthetic Docker-to-native file handoff", "benchmark_score": False, "campaign_started": False}
    if result.ok:
        frozen = json.loads((evidence_root / "runtime.json").read_text()) if (evidence_root / "runtime.json").exists() else None
        executor = NativeSandboxExecutor(license_file=Path.home() / "gurobi.lic", expected_environment=frozen["native_environment"] if frozen else None)
        ok, _, _, output = executor(attempt, 60, 1)
        report.update(execution_ok=ok, output=output, passed=ok and output == {"status": "optimal", "objective_value": 1.0, "threads": 1}, native_environment_sha256=hashlib.sha256(json.dumps(executor.environment, sort_keys=True).encode()).hexdigest())
    path = directory / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    (evidence_root / "readiness" / (backend + ".json")).write_text(json.dumps({"report": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2) + "\n")
    print(json.dumps(report | {"report": str(path)}, indent=2))
    if not report["passed"]:
        raise RuntimeError("The synthetic file-handoff check did not pass; inspect its preserved records")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "auth"])
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--image", default=IMAGE_TAG)
    parser.add_argument("--verify", action="store_true", help="After auth, make one unscored model edit and run the supplied synthetic Gurobi program")
    parser.add_argument("--evidence-root", type=Path, default=Path("outputs/diagnostics/hybrid_runtime_20261013"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.evidence_root.resolve(), image=args.image)
    elif not args.backend:
        parser.error("auth requires --backend")
    else:
        authenticate(BACKENDS[args.backend], args.evidence_root.resolve(), image=args.image, verify=args.verify)


if __name__ == "__main__":
    main()
