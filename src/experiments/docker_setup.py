"""Docker runtime evidence, private CLI authentication, and backend readiness checks.

`prepare` and `auth` record the evidence once per machine; campaigns verify it before any model call.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess

from src.agent.container import DockerRuntime
from src.agent.docker_client import AUTH_VOLUMES, CLI_VERSIONS, DOCKER_EDIT_POLICY, DockerCliClient, IMAGE_TAG, credential_from_environment, worker_file_hashes
from src.experiments.environment import RuntimeEnvironmentError
from src.experiments.sandbox import GUROBI_PROBE, NativeSandboxExecutor, native_environment, same_environment
from src.utils.records import digest


BACKENDS = {"codex": "codex-cli", "claude": "claude-code-cli", "agy": "agy-cli"}
READINESS_MODELS = {"codex-cli": "gpt-6.1-sol", "claude-code-cli": "claude-sonnet-5-5", "agy-cli": "gemini-3.8-flash-medium"}


SECCOMP_PROFILE = Path(__file__).resolve().parents[2] / "docker/agents/seccomp-userns.json"


def runtime():
    return DockerRuntime(seccomp_profile=SECCOMP_PROFILE)


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


def load_execution_environment(path):
    """Read the runtime.json written by `prepare`; the campaign freezes its hash at init."""
    path = Path(path).resolve()
    if not path.is_file():
        raise RuntimeEnvironmentError(f"Runtime manifest not found: {path}; run `python src/02_experiment.py docker prepare` first")
    value = json.loads(path.read_text())
    validate_execution_environment(value)
    return value | {"evidence_root": str(path.parent), "manifest_sha256": digest(path)}


def validate_execution_environment(value):
    if value.get("agent_policy") != DOCKER_EDIT_POLICY or value.get("cli_versions") != CLI_VERSIONS or value.get("worker_files_sha256") != worker_file_hashes():
        raise RuntimeEnvironmentError("Docker worker sources or CLI declarations differ from the prepared environment")
    profile = SECCOMP_PROFILE
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


def agy_login(command, *, image=IMAGE_TAG, volume=None):
    """Sign in to Antigravity once (`login`) or check the saved login (`models`) in the agent image.

    The keyring volume is created on first use. docker/agents/agy_login.py is mounted read-only for this
    container only, so the agent image and its worker hashes stay unchanged.
    """
    if not os.isatty(0) or not os.isatty(1):
        raise RuntimeError("Run this in your own terminal; it asks for a hidden passphrase and the Google sign-in code")
    engine = runtime()
    image_id = engine.image_id(image)
    volume = volume or AUTH_VOLUMES["agy-cli"]
    engine.ensure_volume(volume, "agy-native-auth", label="io.decompbench.purpose", create=command == "login")
    script = Path(__file__).resolve().parents[2] / "docker/agents/agy_login.py"
    arguments = engine.cli("run", "--rm", "-it", "--init", "--pull", "never", "--platform", "linux/arm64", "--user", "1000:1000", "--cap-drop", "ALL",
                           "--security-opt", "no-new-privileges", "--security-opt", f"seccomp={engine.seccomp_profile}", "--read-only", "--log-driver", "none",
                           "--pids-limit", "128", "--memory", "1536m", "--cpus", "2", "--network", "bridge",
                           "--tmpfs", "/tmp:rw,nosuid,nodev,size=128m", "--tmpfs", "/home/node:rw,nosuid,nodev,size=256m,uid=1000,gid=1000,mode=0700",
                           "--tmpfs", "/run/user/1000:rw,nosuid,nodev,size=16m,uid=1000,gid=1000,mode=0700",
                           "--mount", f"type=volume,src={volume},dst=/auth/keyrings", "--mount", f"type=bind,src={script},dst=/opt/login/agy_login.py,readonly",
                           "--env", "HOME=/home/node", "--env", "TERM=xterm-256color", "--entrypoint", "dbus-run-session", image_id, "--", "python3", "/opt/login/agy_login.py", command)
    returncode = subprocess.call(arguments)
    if returncode:
        raise RuntimeError("Antigravity sign-in did not finish with a sealed login and the required model; see the messages above")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "auth", "agy-login", "agy-models"], help="agy-login: one-time Antigravity sign-in; agy-models: check the saved login")
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--image", default=IMAGE_TAG)
    parser.add_argument("--verify", action="store_true", help="After auth, make one unscored model edit and run the supplied synthetic Gurobi program")
    parser.add_argument("--evidence-root", type=Path, default=Path("outputs/runtime"))
    parser.add_argument("--agy-volume", help="agy-login/agy-models only: use another keyring volume, e.g. to test a fresh sign-in")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.evidence_root.resolve(), image=args.image)
    elif args.command in ("agy-login", "agy-models"):
        agy_login(args.command.split("-")[1], image=args.image, volume=args.agy_volume)
    elif not args.backend:
        parser.error("auth requires --backend")
    else:
        authenticate(BACKENDS[args.backend], args.evidence_root.resolve(), image=args.image, verify=args.verify)


if __name__ == "__main__":
    main()
