"""Run the approved Claude extension while preserving the parked Gemini process."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from src.agent.paired_runner import ProtocolError, read, save
from src.agent.runtime_state import runtime_lock, state_root
from src.campaign_extension import extension_parent, process_identity, same_process, validate_group
from src.campaign_plan import CLAUDE_EXTENSION_MODELS
from src.native_campaign import pair_complete, validate


def now():
    return datetime.now(timezone.utc).isoformat()


def user_paused(root):
    pause = read(root / "pause_request.json") or {}
    return (root / "STOP").exists() or bool(pause.get("resume_requires_explicit_user_request"))


def children(pid):
    result = subprocess.run(["/bin/ps", "-axo", "pid=,ppid="], capture_output=True, text=True, timeout=10, check=True)
    edges = [tuple(map(int, line.split())) for line in result.stdout.splitlines() if line.strip()]
    found = {int(pid)}
    while True:
        more = {child for child, parent in edges if parent in found} - found
        if not more:
            return sorted(found - {int(pid)})
        found.update(more)


def cwd(pid):
    result = subprocess.run(["/usr/sbin/lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"], capture_output=True, text=True, timeout=10)
    names = [line[1:] for line in result.stdout.splitlines() if line.startswith("n")]
    return names[0] if result.returncode == 0 and len(names) == 1 else None


def owns_runtime_namespace(pid):
    expected = (state_root() / "fixed81-native-watchdog.lock").resolve()
    result = subprocess.run(["/usr/sbin/lsof", "-a", "-p", str(pid), "-Fn"], capture_output=True, text=True, timeout=10)
    return result.returncode == 0 and any(Path(line[1:]).resolve() == expected for line in result.stdout.splitlines() if line.startswith("n/"))


def live_workers():
    result = subprocess.run(["docker", "ps", "--filter", "label=io.decompbench.runtime=v2-worker", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError("Cannot verify Docker worker quiescence")
    return result.stdout.split()


def owned_live_workers(root):
    names = {(read(path) or {}).get("name") for path in (root / "runs").glob("**/calls/*/*_container.json")}
    return sorted(names.intersection(live_workers()))


def wait_stopped(expected):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        actual = process_identity(expected["pid"])
        if not same_process(expected, actual):
            raise RuntimeError("Parent watchdog identity changed")
        if "T" in actual["state"]:
            return actual
        time.sleep(0.05)
    raise RuntimeError("Parent watchdog did not enter the stopped state")


def park_parent(parent_root, record, persist):
    """Reserve computation before parking; restore the original state on a prelaunch race."""
    with runtime_lock("fixed81-native-campaign", blocking=False):
        if user_paused(parent_root):
            raise ProtocolError("User pause prevents the extension handoff")
        receipt = read(parent_root / "watchdog.json") or {}
        expected = record.get("parent_watchdog") or process_identity(receipt.get("pid", 0))
        actual = process_identity(receipt.get("pid", 0))
        if (not expected or not same_process(expected, actual)
                or f"-m src.native_campaign watch --root {parent_root} --all-models" not in actual["command"]
                or cwd(actual["pid"]) != str(parent_root / "controller") or not owns_runtime_namespace(actual["pid"])):
            raise RuntimeError("Parent watchdog identity or frozen working directory differs")
        state = read(parent_root / "status.json") or {}
        if state.get("state") != "recovering" or state.get("active_instances") or children(actual["pid"]) or live_workers():
            return False
        already_owned = record.get("phase") == "hold_intent" and same_process(record.get("parent_watchdog", {}), actual)
        if "T" in actual["state"] and not already_owned:
            raise RuntimeError("Cannot claim an unowned stopped watchdog")
        record.update(parent_watchdog=expected, parent_cwd=str(parent_root / "controller"), phase="hold_intent")
        persist()
        try:
            if "T" not in actual["state"]:
                os.kill(actual["pid"], signal.SIGSTOP)
            wait_stopped(expected)
            if children(actual["pid"]) or live_workers():
                raise RuntimeError("Computation raced with the idle handoff")
        except Exception:
            # This rollback is permitted only before any extension process has been launched.
            observed = process_identity(expected["pid"])
            if not record.get("extension_watchdog") and same_process(expected, observed) and "T" in observed["state"]:
                os.kill(expected["pid"], signal.SIGCONT)
                record.update(phase="waiting_for_idle", rollback_at=now())
                persist()
            raise
        record.update(phase="parked", parked_at=now())
        persist()
        return True


def find_extension_watchers(extension_root, token):
    result = subprocess.run(["/bin/ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=10, check=True)
    found = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and f"-m src.native_campaign watch --root {extension_root} --all-models --coordinator-token {token}" in fields[1]:
            actual = process_identity(int(fields[0]))
            if actual:
                found.append(actual)
    return found


def await_watcher_receipt(root, process, token, timeout=30):
    """Wait past macOS Python's initial executable handoff before pinning identity."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        receipt = read(root / "watchdog.json") or {}
        if receipt.get("pid") == process.pid and receipt.get("coordinator_token") == token:
            actual = process_identity(process.pid)
            if (actual and f"--coordinator-token {token}" in actual["command"]
                    and cwd(process.pid) == str(root / "controller")):
                return actual
        if process.poll() is not None:
            raise RuntimeError("Extension watcher exited before publishing its startup receipt")
        time.sleep(0.1)
    raise RuntimeError("Extension watcher startup acknowledgement timed out")


def extension_complete(root, manifest):
    state = read(root / "status.json") or {}
    expected = list(CLAUDE_EXTENSION_MODELS)
    if (state.get("state") != "complete" or not state.get("scope_complete")
            or state.get("execution_scope") != expected or state.get("active_instances")):
        return False
    return all(pair_complete(root, model, case, manifest["protocol_id"]) for model in expected for case in manifest["instances"])


def surviving_owned_processes(root):
    recorded = {}
    path = root / "monitor/resources.jsonl"
    if path.exists():
        import json
        for line in path.open():
            for process in json.loads(line).get("processes", []):
                recorded.setdefault(process["pid"], set()).add(" ".join(process["started"].split()))
    result = subprocess.run(["/bin/ps", "-axo", "pid=,lstart="], capture_output=True, text=True, timeout=10, check=True)
    survivors = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and " ".join(parts[1].split()) in recorded.get(int(parts[0]), set()):
            survivors.append(int(parts[0]))
    return survivors


def release_parent(parent_root, extension_root, record, persist, completed):
    """A zero exit or stale status alone never authorizes this signal."""
    with runtime_lock("fixed81-native-campaign", blocking=False):
        if not completed() or user_paused(parent_root) or user_paused(extension_root):
            raise ProtocolError("Scope completion or user-pause guard prevents Gemini continuation")
        watcher = record.get("extension_watchdog")
        if watcher and same_process(watcher, process_identity(watcher["pid"])):
            raise RuntimeError("Extension watcher has not exited")
        if (read(extension_root / "monitor/status.json") or {}).get("state") == "cleanup_failed":
            raise RuntimeError("Extension cleanup failed")
        expected = record["parent_watchdog"]
        actual = process_identity(expected["pid"])
        if not same_process(expected, actual) or "T" not in actual["state"]:
            raise RuntimeError("Cannot release a different or unparked parent watchdog")
        if live_workers() or surviving_owned_processes(extension_root):
            raise RuntimeError("Extension computation cleanup is not yet verified")
        record.update(phase="release_intent", completed_pairs=324, completion_verified_at=now())
        persist()
        if user_paused(parent_root) or user_paused(extension_root):
            raise ProtocolError("User pause cancelled the final continuation")
        os.kill(expected["pid"], signal.SIGCONT)
        record.update(phase="base_resumed", resumed_at=now())
        persist()


def coordinate(group_path):
    group_path = group_path.resolve()
    group = read(group_path)
    manifests = validate_group(group)
    parent_root, extension_root = [Path(member["root"]).resolve() for member in group["members"]]
    for root in (parent_root, extension_root):
        validate(root)
    state_path = Path(extension_parent(manifests[1])["coordinator_state"])
    with runtime_lock("fixed81-claude-extension-coordinator", blocking=False):
        record = read(state_path) or {"token": uuid.uuid4().hex, "created_at": now(), "phase": "waiting_for_idle", "parent_root": str(parent_root), "parent_protocol_id": manifests[0]["protocol_id"], "extension_root": str(extension_root), "extension_protocol_id": manifests[1]["protocol_id"]}
        if (record["parent_protocol_id"], record["extension_protocol_id"], record["extension_root"]) != (manifests[0]["protocol_id"], manifests[1]["protocol_id"], str(extension_root)):
            raise ProtocolError("Existing coordinator ownership belongs to another extension")
        record.update(coordinator_pid=os.getpid())
        def persist():
            record["updated_at"] = now()
            save(state_path, record)
        def activate(root):
            current = read(group_path)
            current.update(active_root=str(root), coordinator_state=str(state_path), updated_at=now())
            save(group_path, current)
        persist()
        try:
            if record["phase"] in {"user_paused", "repair_required"}:
                raise ProtocolError("Recorded hold requires explicit operational recovery")
            if record["phase"] == "base_resumed":
                activate(parent_root)
                return
            if record["phase"] == "release_intent":
                actual = process_identity(record["parent_watchdog"]["pid"])
                if same_process(record["parent_watchdog"], actual) and "T" not in actual["state"]:
                    if not extension_complete(extension_root, validate(extension_root)) or owned_live_workers(extension_root) or surviving_owned_processes(extension_root):
                        raise RuntimeError("Cannot verify interrupted release")
                    record.update(phase="base_resumed", recovered_release_at=now())
                    persist()
                    activate(parent_root)
                    return
            while record["phase"] in {"waiting_for_idle", "hold_intent"}:
                if user_paused(parent_root) or user_paused(extension_root):
                    record.update(phase="user_paused")
                    persist()
                    return
                try:
                    if park_parent(parent_root, record, persist):
                        break
                except BlockingIOError:
                    pass
                time.sleep(5)
                persist()
            if user_paused(parent_root) or user_paused(extension_root):
                record.update(phase="user_paused")
                persist()
                return
            activate(extension_root)
            watchers = find_extension_watchers(extension_root, record["token"])
            if len(watchers) > 1:
                raise RuntimeError("Multiple extension watcher processes need inspection")
            prior_watch = read(extension_root / "watchdog.json")
            if watchers:
                record.update(extension_watchdog=watchers[0], phase="extension_running")
                persist()
            elif not prior_watch and record["phase"] in {"parked", "launch_intent"}:
                record.update(phase="launch_intent", launch_requested_at=now())
                persist()
                command = [sys.executable, "-u", "-m", "src.native_campaign", "watch", "--root", str(extension_root), "--all-models", "--coordinator-token", record["token"]]
                with (extension_root / "watchdog.log").open("a") as log:
                    process = subprocess.Popen(command, cwd=extension_root / "controller", stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                actual = await_watcher_receipt(extension_root, process, record["token"])
                record.update(extension_watchdog=actual, phase="extension_running")
                persist()
            while True:
                if user_paused(parent_root) or user_paused(extension_root):
                    (extension_root / "STOP").write_text("User pause cancels the extension-to-Gemini handoff.\n")
                    record.update(phase="user_paused", pause_observed_at=now())
                    persist()
                    return
                expected = record.get("extension_watchdog")
                if not expected or not same_process(expected, process_identity(expected["pid"])):
                    break
                parent_actual = process_identity(record["parent_watchdog"]["pid"])
                if not same_process(record["parent_watchdog"], parent_actual) or "T" not in parent_actual["state"]:
                    (extension_root / "STOP").write_text("Parent hold identity changed; drain current pairs for inspection.\n")
                    raise RuntimeError("Parked parent identity/state changed")
                persist()
                time.sleep(5)
            manifest = validate(extension_root)
            if not extension_complete(extension_root, manifest):
                raise RuntimeError("Extension watcher exited before all 324 pairs completed")
            release_parent(parent_root, extension_root, record, persist, lambda: extension_complete(extension_root, validate(extension_root)))
            activate(parent_root)
        except Exception as error:
            record.update(phase="user_paused" if user_paused(parent_root) or user_paused(extension_root) else "repair_required", error=f"{type(error).__name__}: {error}")
            persist()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=Path, required=True)
    args = parser.parse_args()
    coordinate(args.group)


if __name__ == "__main__":
    main()
