"""Privately launch all models, or continue after the running GPT/Claude scope."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


VARIABLE = "DECOMPBENCH_AGY_KEYRING_PASSPHRASE"


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def authorize_handoff(root, manifest):
    """The earlier all-model authorization covers only a completed launch scope."""
    pause = read(root / "pause_request.json")
    status = read(root / "status.json")
    expected = [item["model"] for item in manifest["model_queue"] if item["backend"] != "agy-cli"]
    if ((root / "STOP").exists() or pause.get("reason") != "launch_scope_complete"
            or status.get("state") != "paused" or not status.get("scope_complete")
            or status.get("execution_scope") != expected or status.get("active_instances")):
        raise RuntimeError("The campaign did not reach its authorized scope boundary; no automatic continuation was performed")
    pause.update(resume_requires_explicit_user_request=False, state="authorized", reason="existing_all_model_authorization_private_Gemini_handoff", recorded_at=datetime.now(timezone.utc).isoformat())
    save(root / "pause_request.json", pause)


def verify_private_credential(root, manifest):
    """Verify the existing encrypted login offline, without submitting inference."""
    from src.hybrid_runtime import runtime
    from src.agent.docker_client import AUTH_VOLUMES
    engine = runtime()
    engine.bind_environment(manifest["execution_environment"])
    # Keep this small, bounded authentication check outside the experiment sampler.
    engine.LABEL = "io.decompbench.private-auth-check"
    volume = AUTH_VOLUMES["agy-cli"]
    engine.ensure_volume(volume, "agy-native-auth", label="io.decompbench.purpose", create=False)
    directory = root / "operations/private_auth_checks" / uuid.uuid4().hex
    result = engine.run(manifest["execution_environment"]["image_id"], ["auth-check", "agy-cli"], payload={"credential": {"keyring_passphrase": os.environ[VARIABLE]}}, state_path=directory / "container.json", timeout=60, network="none", cpus=0.25, memory="256m", volumes={volume: "/auth/keyrings"})
    observed = json.loads(result.stdout) if result.returncode == 0 else {}
    if not observed.get("credential_loaded") or not result.metadata.get("cleanup_verified"):
        raise RuntimeError("AGY keyring unlock failed; recheck the hidden passphrase in your local Terminal")
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "credential_verified": True, "inference_submitted": False, "cleanup_verified": True, "passphrase_persisted": False}
    save(directory / "report.json", report)
    return str(directory / "report.json")


def background_launch(root):
    sys.path.insert(0, str(root / "controller"))
    from src.native_campaign import validate, ensure_execution_allowed
    with (root / "operations/private_launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = validate(root)
        gate = read(root / "operations/launch_ready.json")
        if gate.get("protocol_id") != manifest["protocol_id"] or not os.environ.get(VARIABLE):
            raise RuntimeError("Private launch does not match the reviewed cohort")
        receipt = {"requested_at": datetime.now(timezone.utc).isoformat(), "helper_pid": os.getpid(), "protocol_id": manifest["protocol_id"], "all_models": True, "passphrase_persisted": False}
        receipt.update(state="checking_private_credential", credential_verified=False)
        save(root / "operations/private_launch.json", receipt)
        try:
            proof = verify_private_credential(root, manifest)
        except Exception as error:
            receipt.update(state="credential_verification_failed", error_type=type(error).__name__)
            save(root / "operations/private_launch.json", receipt)
            raise SystemExit("Private AGY unlock check failed; no experiment was interrupted. Recheck the hidden passphrase or the nonsecret diagnostic.") from None
        receipt.update(credential_verified=True, credential_verification_report=proof)
        prior = read(root / "watchdog.json")
        if alive(prior.get("pid")):
            expected = [item["model"] for item in manifest["model_queue"] if item["backend"] != "agy-cli"]
            if prior.get("execution_scope") != expected:
                raise RuntimeError("A different or all-model scope is already running; no duplicate launch")
            receipt.update(state="waiting_for_GPT_Claude_scope", preceding_watchdog_pid=prior["pid"])
            save(root / "operations/private_launch.json", receipt)
            while alive(prior["pid"]):
                pause = read(root / "pause_request.json")
                if (root / "STOP").exists() or (pause.get("resume_requires_explicit_user_request") and pause.get("reason") != "launch_scope_complete"):
                    receipt.update(state="cancelled_by_pause")
                    save(root / "operations/private_launch.json", receipt)
                    return
                time.sleep(5)
            authorize_handoff(root, manifest)
        elif read(root / "pause_request.json").get("reason") == "launch_scope_complete":
            authorize_handoff(root, manifest)
        ensure_execution_allowed(root, manifest)
        command = [sys.executable, "-u", "-m", "src.native_campaign", "watch", "--root", str(root), "--all-models"]
        with (root / "watchdog.log").open("a") as log:
            process = subprocess.Popen(command, cwd=root / "controller", stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        os.environ.pop(VARIABLE, None)
        subprocess.Popen(["/usr/bin/caffeinate", "-is", "-w", str(process.pid)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        receipt.update(state="launched", watchdog_pid=process.pid)
        save(root / "operations/private_launch.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--private-background", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "operations/launch_ready.json").is_file():
        raise SystemExit("Preflight is still running. Retry after launch_ready.json exists.")
    if args.private_background:
        background_launch(root)
        return
    environment = dict(os.environ)
    if not environment.get(VARIABLE):
        if not sys.stdin.isatty():
            raise SystemExit("Use your own Terminal for hidden input; do not enter a passphrase in chat.")
        environment[VARIABLE] = getpass.getpass("Benchmark AGY keyring passphrase (hidden; not your Google password): ")
    if not environment[VARIABLE]:
        raise SystemExit("No passphrase supplied; nothing started.")
    with (root / "operations/private_launch.log").open("a") as log:
        helper = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--root", str(root), "--private-background"], env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    environment.pop(VARIABLE, None)
    print("Checking the saved AGY login with your hidden passphrase; no model call is being made.", flush=True)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        receipt = read(root / "operations/private_launch.json")
        if receipt.get("helper_pid") == helper.pid:
            if receipt.get("credential_verified") and receipt.get("state") in ("waiting_for_GPT_Claude_scope", "launched"):
                print(f"Gemini keyring verified. Background continuation is armed (PID {helper.pid}); you can close this Terminal. No passphrase was saved. Monitor: http://localhost:8769")
                return
            if receipt.get("state") == "credential_verification_failed":
                raise SystemExit("AGY unlock verification failed; no running instance was interrupted. Recheck the benchmark keyring passphrase.")
        if helper.poll() is not None:
            raise SystemExit("Private launcher exited before arming the handoff. Inspect operations/private_launch.log for its nonsecret status.")
        time.sleep(0.25)
    raise SystemExit("Private verification is still pending. Check operations/private_launch.json before assuming Gemini is ready.")


if __name__ == "__main__":
    main()
