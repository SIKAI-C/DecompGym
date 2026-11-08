"""Docker runtime evidence, private CLI authentication, and backend readiness checks.

`prepare` and `auth` record the evidence once per machine; campaigns verify it before any model call.

    docker auth --backend codex --slot 0 --login [--replace]   # Codex device sign-in into one call-gate slot's volume (repeat for --slot 1, 2)
    docker auth --backend claude [--slot 1] --login [--replace]  # store a `claude setup-token` token from a hidden prompt (slot 0 by default)
    docker agy-login | agy-models                              # Antigravity sign-in / saved-login check
    docker auth --backend X --check                            # no-inference credential check (Codex, Claude: every slot)
    docker auth --backend X --verify                           # one unscored edit plus the native sandbox handoff (readiness)

The auth volume is the only credential source of Codex and Claude Code (src/agent/auth.py); a credential is never
copied from the host unless `--import-host` (Codex, emergency) or `--from-env` (Claude) asks for it explicitly.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from src.agent.container import DockerRuntime
from src.agent.auth import AUTH_POLICY, CLAUDE_SLOT_VOLUMES, CODEX_SLOT_VOLUMES, HELPER_LABEL, SLOT_VOLUMES, backend_volumes, require_exclusive_volume, volume_lock
from src.agent.docker_client import AUTH_VOLUMES, CLI_VERSIONS, DOCKER_EDIT_POLICY, DockerCliClient, IMAGE_TAG, credential_from_environment, worker_file_hashes
from src.experiments import agent_auth
from src.experiments.environment import RuntimeEnvironmentError
from src.experiments.sandbox import GUROBI_PROBE, NativeSandboxExecutor, native_environment, same_environment
from src.utils.records import digest


BACKENDS = {"codex": "codex-cli", "claude": "claude-code-cli", "agy": "agy-cli"}
READINESS_MODELS = {"codex-cli": "gpt-6.1-sol", "claude-code-cli": "claude-sonnet-5-5", "agy-cli": "gemini-3.8-flash-medium"}


SECCOMP_PROFILE = Path(__file__).resolve().parents[2] / "docker/agents/seccomp-userns.json"


def license_file():
    """The Gurobi license the solver sandbox may read: GRB_LICENSE_FILE if set, otherwise ~/gurobi.lic."""
    return Path(os.environ.get("GRB_LICENSE_FILE") or Path.home() / "gurobi.lic")


def runtime():
    return DockerRuntime(seccomp_profile=SECCOMP_PROFILE)


def prepare(evidence_root, *, image=IMAGE_TAG):
    engine = runtime()
    image_id = engine.image_id(image)
    client = DockerCliClient("preflight", backend="codex-cli", image_id=image_id, runtime=engine, evidence_root=evidence_root)
    client.prepare()
    native = NativeSandboxExecutor(license_file=license_file())
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
                "auth_policy": AUTH_POLICY, "codex_slot_volumes": list(CODEX_SLOT_VOLUMES), "claude_slot_volumes": list(CLAUDE_SLOT_VOLUMES),
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


# Runtime evidence written before a backend had its current slots lists fewer: two Codex slots and no Claude Code list
# (its one volume, slot 0). Such evidence stays valid; a campaign that uses more slots than the evidence declares is
# refused (require_slot_volumes), so a three-Codex/two-Claude campaign needs a fresh `docker prepare`.
MINIMUM_SLOTS = {"codex-cli": 2, "claude-code-cli": 1}
SLOT_EVIDENCE = {"codex-cli": "codex_slot_volumes", "claude-code-cli": "claude_slot_volumes"}


def slot_volumes(value, backend):
    """The auth volumes of `backend`'s call-gate slots that the runtime evidence declares (a prefix of the current slots)."""
    declared = value.get(SLOT_EVIDENCE[backend], list(SLOT_VOLUMES[backend][:1]) if backend == "claude-code-cli" else None)
    if not isinstance(declared, list) or len(declared) < MINIMUM_SLOTS[backend] or declared != list(SLOT_VOLUMES[backend][:len(declared)]):
        return None
    return declared


def require_slot_volumes(value, slots):
    """Refuse a campaign whose call gates use more slots ({backend: count}) than its frozen runtime evidence declares."""
    for backend, count in slots.items():
        declared = slot_volumes(value, backend) or []
        if count > len(declared):
            raise RuntimeEnvironmentError(f"The prepared runtime evidence declares {len(declared)} {backend} auth slot(s); this campaign uses {count}. "
                                          "Rerun `python src/02_experiment.py docker prepare` before initializing the campaign")


def validate_execution_environment(value):
    if value.get("agent_policy") != DOCKER_EDIT_POLICY or value.get("cli_versions") != CLI_VERSIONS or value.get("worker_files_sha256") != worker_file_hashes():
        raise RuntimeEnvironmentError("Docker worker sources or CLI declarations differ from the prepared environment")
    if value.get("auth_volumes") != AUTH_VOLUMES or any(slot_volumes(value, backend) is None for backend in SLOT_EVIDENCE):
        raise RuntimeEnvironmentError("Prepared runtime evidence names other authentication volumes; rerun `docker prepare` and `docker auth`")
    if value.get("auth_policy") != AUTH_POLICY:
        raise RuntimeEnvironmentError(f"Prepared runtime evidence predates the authentication policy {AUTH_POLICY}; rerun `docker prepare`, then `docker auth --check` and `--verify`")
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
        allowed = (slot_volumes(value, backend) or []) if backend in SLOT_EVIDENCE else []
        if report.get("auth_volume") not in [value.get("auth_volumes", {}).get(backend), *allowed]:
            raise RuntimeEnvironmentError(f"Readiness for {backend} was proved with another authentication volume; rerun `docker auth --verify`")
        if report.get("auth_policy") != value.get("auth_policy"):
            raise RuntimeEnvironmentError(f"Readiness for {backend} predates the current authentication policy; rerun `docker auth --backend ... --verify`")
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
    """The job credential of `docker auth`: None for Codex and Claude Code (volume only), the AGY keyring passphrase."""
    if backend != "agy-cli":
        return credential_from_environment(backend)
    if os.environ.get("DECOMPBENCH_AGY_KEYRING_PASSPHRASE"):
        return credential_from_environment(backend)
    if not os.isatty(0):
        raise RuntimeError("Run AGY authentication in your own local terminal for the hidden passphrase prompt")
    return {"keyring_passphrase": getpass.getpass("Existing benchmark keyring passphrase (hidden; not your Google password): ")}


def prepared_engine(evidence_root, image=IMAGE_TAG, *, bind=True):
    """The Docker engine and image of the prepared runtime evidence (the image tag before `prepare`). A login does not
    bind the engine to that evidence (`bind=False`), so it also works before a fresh `docker prepare`."""
    engine = runtime()
    frozen = evidence_root / "runtime.json"
    if image == IMAGE_TAG and frozen.exists():
        prepared = json.loads(frozen.read_text())
        image = prepared["image_id"]
        if bind and "docker_endpoint" in prepared:
            engine.bind_environment(prepared)
    return engine, engine.image_id(image)


def authenticate(backend, evidence_root, *, image=IMAGE_TAG, verify=False, slots=None):
    """Load the saved credential in the worker without inference (`auth-check`), under each auth volume's lock."""
    engine, image_id = prepared_engine(evidence_root, image)
    credential = interactive_credential(backend)
    import uuid
    for slot in (slots if backend in SLOT_VOLUMES and slots is not None else [None]):
        client = DockerCliClient("auth-check", backend=backend, image_id=image_id, runtime=engine, evidence_root=evidence_root)
        client.auth_slot = slot
        mounts = client.prepare()
        with client.auth_lock(mounts):
            result = engine.run(image_id, ["auth-check", backend], payload={"credential": credential}, state_path=evidence_root / "authentication" / uuid.uuid4().hex / "container.json", timeout=60, network="none", volumes=mounts)
        if result.returncode:
            raise RuntimeError(f"The saved {backend} credential in {next(iter(mounts))} could not be loaded; no inference was submitted. Store one with `docker auth --backend ... --login`")
        print(json.dumps({"backend": backend, "auth_volume": next(iter(mounts)), "credential_loaded": True, "inference_submitted": False}, indent=2))
    if verify:
        verify_backend(backend, evidence_root, engine=engine, image_id=image_id, credential=credential, slot=(slots or [0])[0] if backend in SLOT_VOLUMES else None)


def check(backend, evidence_root, *, image=IMAGE_TAG, slots=None, models=None):
    """`docker auth --check`: classified no-inference reports, saved under <evidence_root>/auth-checks/."""
    engine, image_id = prepared_engine(evidence_root, image)
    passphrase = (interactive_credential(backend) or {}).get("keyring_passphrase") if backend == "agy-cli" else None
    if models is None:
        models = [READINESS_MODELS[backend]] if backend != "claude-code-cli" else []
    reports = agent_auth.check_backend(engine, image_id, backend, slots=slots, models=models, passphrase=passphrase)
    directory = evidence_root / "auth-checks"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{backend}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{time.time_ns() % 10**6:06d}.json"
    path.write_text(json.dumps({"auth_policy": AUTH_POLICY, "reports": reports}, indent=2) + "\n")
    shown = ("backend", "volume", "slot", "status", "ok", "origin", "detail", "warnings", "repair", "models_missing", "credential_refreshed", "stored_days", "validity")
    for report in reports:
        print(json.dumps({key: report.get(key) for key in shown if report.get(key) not in (None, [], "")}, indent=2))
    print(json.dumps({"report": str(path)}))
    return reports


def verify_backend(backend, evidence_root, *, engine, image_id, credential, slot=None):
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
    client.auth_slot = slot
    # The no-inference check first: a broken credential stops here, before the unscored model call.
    checked = agent_auth.check_backend(engine, image_id, backend, slots=[slot or 0] if backend in SLOT_VOLUMES else None, models=[client.model] if backend != "claude-code-cli" else [],
                                       passphrase=(credential or {}).get("keyring_passphrase"))
    agent_auth.raise_for(checked)
    result = client.edit_file(target, prompt, context=[case / "data.json"], log_dir=directory / "call", step="agent")
    report = {"schema_version": 1, "backend": backend, "model": client.model, "image_id": image_id, "worker_files_sha256": worker_file_hashes(), "auth_volume": client.auth_volume(), "auth_policy": AUTH_POLICY,
              "auth_check": [{key: item.get(key) for key in ("volume", "status", "origin", "checked_at")} for item in checked], "edit_ok": result.ok,
              "execution_ok": False, "passed": False, "purpose": "synthetic Docker-to-native file handoff", "benchmark_score": False, "campaign_started": False}
    if result.ok:
        frozen = json.loads((evidence_root / "runtime.json").read_text()) if (evidence_root / "runtime.json").exists() else None
        executor = NativeSandboxExecutor(license_file=license_file(), expected_environment=frozen["native_environment"] if frozen else None)
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
    import uuid
    name = f"decompgym-agy-{command}-{uuid.uuid4().hex[:12]}"
    arguments = engine.cli("run", "--rm", "-it", "--init", "--name", name, "--label", f"{engine.LABEL}={HELPER_LABEL}", "--pull", "never", "--platform", "linux/arm64", "--user", "1000:1000", "--cap-drop", "ALL",
                           "--security-opt", "no-new-privileges", "--security-opt", f"seccomp={engine.seccomp_profile}", "--read-only", "--log-driver", "none",
                           "--pids-limit", "128", "--memory", "1536m", "--cpus", "2", "--network", "bridge",
                           "--tmpfs", "/tmp:rw,nosuid,nodev,size=128m", "--tmpfs", "/home/node:rw,nosuid,nodev,size=256m,uid=1000,gid=1000,mode=0700",
                           "--tmpfs", "/run/user/1000:rw,nosuid,nodev,size=16m,uid=1000,gid=1000,mode=0700",
                           "--mount", f"type=volume,src={volume},dst=/auth/keyrings", "--mount", f"type=bind,src={script},dst=/opt/login/agy_login.py,readonly",
                           "--env", "HOME=/home/node", "--env", "TERM=xterm-256color", "--entrypoint", "dbus-run-session", image_id, "--", "python3", "/opt/login/agy_login.py", command)
    # The keyring is rewritten by the container's keyring daemon: hold the volume lock until the container is gone.
    with volume_lock(volume, waiting=agent_auth.notice):
        require_exclusive_volume(engine, volume)
        try:
            returncode = subprocess.call(arguments)
        finally:
            engine.remove_container(name)
    if returncode:
        raise RuntimeError("Antigravity sign-in did not finish with a sealed login and the required model; see the messages above")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "auth", "agy-login", "agy-models"], help="agy-login: one-time Antigravity sign-in; agy-models: check the saved login")
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--slot", type=int, choices=range(len(CODEX_SLOT_VOLUMES)), help="codex (0-2) and claude (0-1): the call-gate slot (auth volume); codex --login needs it, claude --login defaults to slot 0, --check to every slot, --verify to slot 0")
    parser.add_argument("--login", action="store_true", help="store a credential: codex device sign-in in a helper container, claude setup token at a hidden prompt (agy: use agy-login)")
    parser.add_argument("--replace", action="store_true", help="with --login: replace a saved credential; the old one is kept as <file>.replaced-<UTC> and never used automatically")
    parser.add_argument("--import-host", action="store_true", help="codex --login, emergency only: copy the host ~/.codex/auth.json (shares the host's OAuth refresh lineage)")
    parser.add_argument("--from-env", action="store_true", help="claude --login: import CLAUDE_CODE_OAUTH_TOKEN of this shell instead of the hidden prompt")
    parser.add_argument("--check", action="store_true", help="check the saved credential without inference (codex: app-server account, rate-limit and model reads; agy: agy models; claude: local only)")
    parser.add_argument("--model", action="append", dest="models", help="--check: a model ID that must be in the account catalog (repeatable; default: the readiness model)")
    parser.add_argument("--image", default=IMAGE_TAG)
    parser.add_argument("--verify", action="store_true", help="After auth, make one unscored model edit and run the supplied synthetic Gurobi program")
    parser.add_argument("--evidence-root", type=Path, default=Path("outputs/runtime"))
    parser.add_argument("--agy-volume", help="agy-login/agy-models only: use another keyring volume, e.g. to test a fresh sign-in")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.evidence_root.resolve(), image=args.image)
        return
    if args.command in ("agy-login", "agy-models"):
        agy_login(args.command.split("-")[1], image=args.image, volume=args.agy_volume)
        return
    if not args.backend:
        parser.error("auth requires --backend")
    backend = BACKENDS[args.backend]
    if args.slot is not None and (backend not in SLOT_VOLUMES or args.slot >= len(SLOT_VOLUMES[backend])):
        parser.error("--slot applies to codex (0-2) and claude (0-1) only")
    if (args.replace or args.import_host or args.from_env) and not args.login:
        parser.error("--replace, --import-host and --from-env apply to --login")
    if (args.import_host and backend != "codex-cli") or (args.from_env and backend != "claude-code-cli"):
        parser.error("--import-host is for codex and --from-env for claude")
    evidence_root = args.evidence_root.resolve()
    if args.login:
        if backend == "agy-cli":
            parser.error("AGY signs in with `python src/02_experiment.py docker agy-login` (and checks with agy-models)")
        if backend == "codex-cli" and args.slot is None:
            parser.error("codex --login needs --slot 0, 1 or 2 (one device sign-in per call-gate slot)")
        engine, image_id = prepared_engine(evidence_root, args.image, bind=False)
        if args.import_host:
            agent_auth.import_host_codex(engine, image_id, args.slot, replace=args.replace)
        elif backend == "codex-cli":
            agent_auth.login_codex(engine, image_id, args.slot, replace=args.replace)
        else:
            agent_auth.login_claude(engine, image_id, replace=args.replace, from_env=args.from_env, slot=args.slot)
    slots = [args.slot] if args.slot is not None else None
    if args.check:
        reports = check(backend, evidence_root, image=args.image, slots=slots, models=args.models)
        if not all(report["ok"] for report in reports):
            raise SystemExit(1)
    if args.verify:
        authenticate(backend, evidence_root, image=args.image, verify=True, slots=slots)
    elif not (args.login or args.check):
        # The plain `docker auth --backend X`: load each saved credential in the worker, without inference.
        authenticate(backend, evidence_root, image=args.image, slots=slots if slots is not None else (list(range(len(backend_volumes(backend)))) if backend in SLOT_VOLUMES else None))


if __name__ == "__main__":
    main()
