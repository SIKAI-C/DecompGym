"""`docker auth` login and no-inference checks, and the campaign's authentication admission.

Every operation runs docker/agents/auth_helper.py in a one-off container of the prepared agent image with exactly one
auth volume, while this host holds that volume's lock (src.agent.auth.volume_lock); the lock is released only after
the container is gone. No operation submits a model prompt. Credentials enter a volume only here:

- Codex: `--login` runs `codex login --device-auth` inside the helper container (one device sign-in per call-gate slot),
  so the container never shares the host's OAuth refresh lineage. `--import-host` copies ~/.codex/auth.json instead, as
  an explicitly requested emergency path recorded with origin host-copy.
- Claude Code: `--login` stores a `claude setup-token` token from a hidden prompt (or, with `--from-env`, from
  CLAUDE_CODE_OAUTH_TOKEN of this shell).
- AGY keeps `docker agy-login` / `docker agy-models`.

A check reports one status per auth volume:
    ok
    credential_missing | credential_invalid | legacy_origin | passphrase_invalid
        | commit_incomplete | refresh_not_persisted                 (owner repair: AuthenticationRequired)
    network | rate_limited | timeout | docker_unavailable          (retryable: InfrastructureError)
    model_missing | volume_foreign | protocol_error | helper_failed | unknown_error   (inspect: RuntimeEnvironmentError)
Codex is checked online through its app-server (account/read, account/rateLimits/read, model/list), AGY through
`agy models`; Claude Code has no inference-free online check, so its check is local (presence, format, storage age)
and `--verify` remains the only proof of a working token.
"""
from __future__ import annotations

from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from src.agent.auth import AUTH_POLICY, HELPER_LABEL, MOUNT_POINTS, auth_volume, backend_volumes, repair_command, require_exclusive_volume, volume_lock, volume_slot
from src.agent.safe_artifacts import read_regular_text
from src.experiments.environment import RuntimeEnvironmentError
from src.utils.errors import InfrastructureError


REPOSITORY = Path(__file__).resolve().parents[2]
HELPER = REPOSITORY / "docker/agents/auth_helper.py"
AGY_LOGIN = REPOSITORY / "docker/agents/agy_login.py"
REPAIR_STATUSES = {"credential_missing", "credential_invalid", "legacy_origin", "passphrase_invalid", "commit_incomplete", "refresh_not_persisted"}
RETRY_STATUSES = {"network", "rate_limited", "timeout", "docker_unavailable"}
# A Codex credential without the meta.json of `--login` predates this policy and may share the host's lineage.
TRUSTED_CODEX_ORIGINS = {"container-device-auth", "host-copy"}


def notice(volume):
    print(f"Waiting for the lock of auth volume {volume} (an agent call or another check is using it) ...", file=sys.stderr, flush=True)


def helper_arguments(engine, image_id, *, backend, volume, mode, name, network, interactive=False, extra=()):
    agy = backend == "agy-cli"
    arguments = engine.cli("run", "--rm", "-i", *(["-t"] if interactive else []), "--init", "--name", name, "--label", f"{engine.LABEL}={HELPER_LABEL}", "--pull", "never",
                           "--platform", "linux/arm64", "--user", "1000:1000", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--security-opt", f"seccomp={engine.seccomp_profile}",
                           "--read-only", "--log-driver", "none", "--pids-limit", "128", "--memory", "1536m", "--cpus", "2", "--network", network,
                           "--tmpfs", "/tmp:rw,nosuid,nodev,size=128m", "--tmpfs", "/home/node:rw,nosuid,nodev,size=256m,uid=1000,gid=1000,mode=0700",
                           "--tmpfs", "/run/user/1000:rw,nosuid,nodev,size=16m,uid=1000,gid=1000,mode=0700",
                           "--mount", f"type=volume,src={volume},dst={MOUNT_POINTS[backend]}", "--mount", f"type=bind,src={HELPER},dst=/opt/login/auth_helper.py,readonly",
                           *(["--mount", f"type=bind,src={AGY_LOGIN},dst=/opt/login/agy_login.py,readonly"] if agy else []),
                           "--env", "HOME=/home/node", "--env", "TERM=xterm-256color")
    entry = ["--entrypoint", "dbus-run-session", image_id, "--", "python3", "-I", "-B"] if agy else ["--entrypoint", "python3", image_id, "-I", "-B"]
    return arguments + entry + ["/opt/login/auth_helper.py", mode, *extra]


def ensure_auth_volume(engine, backend, volume, *, create):
    if backend == "agy-cli":
        engine.ensure_volume(volume, "agy-native-auth", label="io.decompbench.purpose", create=False)
    else:
        engine.ensure_volume(volume, backend + "-v2-auth", create=create)


def volume_problem(engine, backend, volume):
    """(status, detail) when the auth volume cannot be checked, else None: missing, foreign or Docker unavailable."""
    label, purpose = ("io.decompbench.purpose", "agy-native-auth") if backend == "agy-cli" else (engine.LABEL, backend + "-v2-auth")
    try:
        inspected = engine.command("volume", "inspect", volume, "--format", "{{json .Labels}}")
    except Exception as error:  # ContainerError: the Docker CLI or engine did not answer
        return "docker_unavailable", f"Docker did not answer ({type(error).__name__})"
    if inspected.returncode:
        if "no such volume" in (inspected.stderr or "").lower():
            return "credential_missing", "the auth volume does not exist yet"
        return "docker_unavailable", "Docker could not inspect the auth volume"
    try:
        labels = json.loads(inspected.stdout or "null") or {}
    except ValueError:
        labels = {}
    if labels.get(label) != purpose:
        return "volume_foreign", f"the volume's {label} label is not {purpose}"
    return None


def last_json(text):
    for line in reversed((text or "").splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def run_helper(engine, image_id, *, backend, volume, mode, network, request=None, timeout=240):
    """One non-interactive helper under the volume's lock; its JSON report (never a credential value)."""
    name = f"decompgym-auth-{uuid.uuid4().hex[:16]}"
    arguments = helper_arguments(engine, image_id, backend=backend, volume=volume, mode=mode, name=name, network=network)
    with volume_lock(volume, waiting=notice):
        require_exclusive_volume(engine, volume)
        try:
            completed = subprocess.run(arguments, input=json.dumps(request or {}) + "\n", capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            completed = None
        finally:
            # The lock is released only after the container (and any process writing the volume) is gone.
            engine.remove_container(name)
    if completed is None:
        return {"status": "timeout", "detail": f"the {mode} helper did not finish within {timeout} s"}
    value = last_json(completed.stdout)
    if value is None or "helper_error" in value:
        return {"status": "helper_failed", "detail": f"the {mode} helper failed (exit {completed.returncode}" + (f", {value['helper_error']}" if value else "") + ")"}
    return value | {"helper_returncode": completed.returncode}


def classify(backend, report):
    """The report with the host's final status: a Codex credential of unknown origin is not admitted."""
    report = dict(report)
    report.setdefault("warnings", [])
    if backend == "codex-cli" and report.get("status") == "ok":
        origin = report.get("origin")
        if origin not in TRUSTED_CODEX_ORIGINS:
            report.update(status="legacy_origin", detail="this Codex credential has no container sign-in record and may share the host's OAuth lineage; sign in again with --login --replace")
        elif origin == "host-copy":
            report["warnings"].append("emergency host import: this credential shares the host's OAuth lineage; a refresh on the host can invalidate it")
        if report.get("status") == "ok" and report.get("meta_slot") is not None and report.get("meta_slot") != report.get("slot"):
            report["warnings"].append(f"meta.json was written for slot {report['meta_slot']}")
    report["ok"] = report.get("status") == "ok"
    if not report["ok"]:
        report["repair"] = repair_command(backend, report.get("volume"))
    return report


def check_backend(engine, image_id, backend, *, slots=None, models=(), passphrase=None, timeout=60):
    """One classified report per auth volume of `backend` (Codex: the given slots, default all)."""
    if backend == "codex-cli":
        volumes = [auth_volume(backend, slot) for slot in (range(len(backend_volumes(backend))) if slots is None else slots)]
    else:
        volumes = list(backend_volumes(backend))
    reports = []
    for volume in volumes:
        base = {"backend": backend, "volume": volume, "slot": volume_slot(backend, volume), "checked_at": datetime.now(timezone.utc).isoformat(), "auth_policy": AUTH_POLICY}
        problem = volume_problem(engine, backend, volume)
        if problem:
            reports.append(classify(backend, base | {"status": problem[0], "detail": problem[1]}))
            continue
        if backend == "codex-cli":
            value = run_helper(engine, image_id, backend=backend, volume=volume, mode="codex-check", network="bridge", request={"models": sorted(set(models)), "timeout": timeout}, timeout=timeout * 4 + 60)
        elif backend == "claude-code-cli":
            value = run_helper(engine, image_id, backend=backend, volume=volume, mode="claude-check", network="none")
        else:
            if not passphrase:
                value = {"status": "passphrase_invalid", "detail": "AGY needs DECOMPBENCH_AGY_KEYRING_PASSPHRASE in the launch environment (the macOS Keychain entry decompbench-agy-keyring)"}
            else:
                value = run_helper(engine, image_id, backend=backend, volume=volume, mode="agy-check", network="bridge", request={"passphrase": passphrase, "models": sorted(set(models)), "timeout": timeout},
                                   timeout=timeout + 120)
        reports.append(classify(backend, base | value))
    return reports


def raise_for(reports):
    """Raise the most severe failure among classified reports: repair, then inspect, then retry."""
    failed = [report for report in reports if not report["ok"]]
    if not failed:
        return
    from src.agent.docker_client import AuthenticationRequired
    lines = "; ".join(f"{report['backend']} {report['volume']}: {report['status']}" + (f" ({report['detail']})" if report.get("detail") else "") for report in failed)
    repair = [report for report in failed if report["status"] in REPAIR_STATUSES]
    if repair:
        raise AuthenticationRequired(f"Authentication repair required: {lines}. Repair: " + "; ".join(sorted({report["repair"] for report in repair})),
                                     {"kind": "credential_invalid", "source": "admission_check", "reports": repair})
    if any(report["status"] not in RETRY_STATUSES for report in failed):
        raise RuntimeEnvironmentError(f"Authentication check failed: {lines}")
    raise InfrastructureError(f"Authentication check could not reach the provider: {lines}")


def require_backend_authentication(engine, image_id, backends, *, models, codex_slots, agy_passphrase, record_dir):
    """Campaign admission: check every queued backend and the Codex judge; refuse to start unless all pass."""
    reports = []
    for backend in sorted(set(backends)):
        slots = range(codex_slots) if backend == "codex-cli" else None
        reports += check_backend(engine, image_id, backend, slots=slots, models=models.get(backend, ()), passphrase=agy_passphrase)
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / f"{time.time_ns()}.json").write_text(json.dumps({"auth_policy": AUTH_POLICY, "codex_slots_checked": codex_slots, "reports": reports}, indent=2) + "\n")
    raise_for(reports)
    return {report["volume"]: {key: report.get(key) for key in ("backend", "slot", "status", "origin", "generation", "warnings", "checked_at")}
            | {"created_at": report.get("created_at") or report.get("stored_at")} for report in reports}


# ----------------------------------------------------------------------------------------------------------------------
# Login and import (owner commands)


def not_stored(what, value, check):
    """Refusals and invalid input leave the volume unchanged; any other failure may have stopped mid-commit."""
    if value.get("status") in ("refused", "credential_invalid"):
        return f"{what} was not stored ({value['status']}: {value.get('detail')}); the saved credential is unchanged"
    return f"{what} was not confirmed as stored ({value.get('status')}: {value.get('detail')}); run `python src/02_experiment.py {check}` (an interrupted commit shows as commit_incomplete)"


def login_codex(engine, image_id, slot, *, replace=False):
    """Interactive device sign-in for one slot, staged in the helper container and promoted only on success."""
    if not os.isatty(0) or not os.isatty(1):
        raise RuntimeError("Run the Codex device sign-in in your own terminal")
    volume = auth_volume("codex-cli", slot)
    ensure_auth_volume(engine, "codex-cli", volume, create=True)
    name = f"decompgym-auth-{uuid.uuid4().hex[:16]}"
    arguments = helper_arguments(engine, image_id, backend="codex-cli", volume=volume, mode="codex-login", name=name, network="bridge", interactive=True,
                                 extra=["--slot", str(slot), "--volume", volume, "--policy", AUTH_POLICY, *(["--replace"] if replace else [])])
    with volume_lock(volume, waiting=notice):
        require_exclusive_volume(engine, volume)
        try:
            returncode = subprocess.call(arguments)
        finally:
            engine.remove_container(name)
    if returncode == 3:
        raise RuntimeError(f"Slot {slot} already holds a Codex credential; nothing changed. Rerun with --replace to replace it (the old one is kept as a .replaced-<UTC> file).")
    if returncode:
        raise RuntimeError(f"The Codex device sign-in did not finish (see the messages above). Confirm what slot {slot} holds with `docker auth --backend codex --slot {slot} --check`; "
                           "an interrupted commit is reported as commit_incomplete")
    print(json.dumps({"backend": "codex-cli", "slot": slot, "volume": volume, "stored": True, "next": f"python src/02_experiment.py docker auth --backend codex --slot {slot} --check"}, indent=2))


def import_host_codex(engine, image_id, slot, *, replace=False):
    """Emergency only: copy the host's ~/.codex/auth.json into one slot (it keeps sharing the host's refresh lineage)."""
    text = read_regular_text(Path.home() / ".codex", "auth.json", missing_ok=True)
    if text is None:
        raise RuntimeError("No host ~/.codex/auth.json to import")
    print("WARNING: --import-host copies the host's Codex login. The container then shares the host's OAuth refresh lineage, so a refresh by the host "
          "(VS Code, the Codex app or the CLI) can invalidate the container's copy. Prefer --login (device sign-in).", file=sys.stderr, flush=True)
    volume = auth_volume("codex-cli", slot)
    ensure_auth_volume(engine, "codex-cli", volume, create=True)
    value = run_helper(engine, image_id, backend="codex-cli", volume=volume, mode="codex-import", network="none",
                       request={"auth": json.loads(text), "replace": replace, "origin": "host-copy", "slot": slot, "volume": volume, "policy": AUTH_POLICY})
    if not value.get("stored"):
        raise RuntimeError(not_stored("The host credential", value, f"docker auth --backend codex --slot {slot} --check"))
    print(json.dumps({key: value.get(key) for key in ("backend", "origin", "created_at", "replaced_file")} | {"slot": slot, "volume": volume}, indent=2))


def login_claude(engine, image_id, *, replace=False, from_env=False):
    """Store a Claude Code setup token (hidden prompt, or CLAUDE_CODE_OAUTH_TOKEN with --from-env)."""
    if from_env:
        token, origin = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", ""), "launch-environment-import"
        if not token:
            raise RuntimeError("CLAUDE_CODE_OAUTH_TOKEN is not set in this shell")
    else:
        if not os.isatty(0):
            raise RuntimeError("Run this in your own terminal for the hidden token prompt")
        print("Create a long-lived token on the host with `claude setup-token` (a separate token; your own Claude login is unchanged), then paste it here.", flush=True)
        token, origin = getpass.getpass("Claude setup token (hidden): ").strip(), "setup-token-prompt"
    volume = auth_volume("claude-code-cli")
    ensure_auth_volume(engine, "claude-code-cli", volume, create=True)
    value = run_helper(engine, image_id, backend="claude-code-cli", volume=volume, mode="claude-store", network="none",
                       request={"token": token, "replace": replace, "origin": origin, "volume": volume, "policy": AUTH_POLICY})
    if not value.get("stored"):
        raise RuntimeError(not_stored("The Claude token", value, "docker auth --backend claude --check"))
    print(json.dumps({key: value.get(key) for key in ("backend", "origin", "created_at", "replaced_file")} | {"volume": volume}, indent=2))
