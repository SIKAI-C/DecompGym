"""Frozen fixed-81 native campaign, paired retries, and persistent resource supervision."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

from src.agent.agy_client import AgyClient
from src.agent.codex_client import CodexClient
from src.agent.paired_runner import InfrastructureError, PairedRunner, ProtocolError, digest, identity, read, save
from src.agent.resource_monitor import ResourceGuard, sample_host, supervise, thermal_warnings
from src.agent.runtime_state import runtime_lock


GPT_ORDER = ["gpt-5.6-sol", "gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol", "gpt-6-astra"]
AGY_ORDER = ["gpt-oss-120b-medium", "gemini-3.8-flash-medium", "gemini-3.1-pro-high", "claude-sonnet-4-6", "claude-opus-4-6-thinking"]
FAMILIES = ["ufl", "fctp", "km", "lrp", "tsso", "bp", "csp", "gap", "shift"]


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize(root, data):
    if root.exists():
        raise ProtocolError("Choose a new campaign root; existing experiments are preserved")
    paths = list(data.glob("*/*/*/data.json"))
    if len(paths) != 81:
        raise ProtocolError("Exactly 81 existing fixed instances are required")
    cases = []
    for path in paths:
        suite, problem, name = path.relative_to(data).parts[:3]
        _, size, seed = name.rsplit("_", 2)
        if problem not in FAMILIES or size not in ("small", "medium", "large") or seed not in ("seed000", "seed001", "seed002"):
            raise ProtocolError("Dataset identity differs from the fixed 81")
        destination = root / "dataset" / suite / problem / name
        destination.mkdir(parents=True)
        hashes = {}
        for filename in ("data.json", "prompt.txt", "meta.json", "gold_monolithic.json"):
            original = path.parent / filename
            shutil.copyfile(original, destination / filename)
            hashes[filename] = digest(original)
            if digest(destination / filename) != hashes[filename]:
                raise ProtocolError("Dataset copy differs")
            (destination / filename).chmod(0o444)
        cases.append({"suite": suite, "problem": problem, "instance": name, "size": size, "seed": int(seed[4:]), "sha256": hashes})
    expected = {(p, s, k) for p in FAMILIES for s in ("small", "medium", "large") for k in range(3)}
    if {(c["problem"], c["size"], c["seed"]) for c in cases} != expected:
        raise ProtocolError("Fixed dataset coverage mismatch")
    cases.sort(key=lambda c: (["small", "medium", "large"].index(c["size"]), c["seed"], FAMILIES.index(c["problem"])))
    source = Path(__file__).resolve().parent
    frozen = root / "controller"
    shutil.copytree(source, frozen / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    hashes = {str(p.relative_to(frozen)): digest(p) for p in frozen.rglob("*") if p.is_file()}
    manifest = {"schema_version": 2, "created_at": now(), "backend": "native", "scope": "existing_fixed_81", "data_regenerated": False, "original_data_root": str(data), "historical_reuse": {"model": "gpt-5.6-luna", "new_generation_sessions": 0}, "gpt_order": GPT_ORDER, "agy_order": AGY_ORDER, "generation_effort": "medium", "judge_model": "gpt-6-astra", "judge_effort": "high", "max_attempts": 10, "modes": ["plain", "reflection"], "shared_first_attempt": True, "stop_after_success": True, "workers": 2, "active_codex_invocation_limit": 1, "active_agy_invocation_limit": 1, "agent_timeout": 900, "solver_timeout": 60, "solver_threads": 1, "solver_cooldown_seconds": 10, "resource_guard": {"sample_interval": 5, "max_cpu_percent": 250, "max_rss_mib": 2048, "min_free_percent": 10, "sustained_samples": 3}, "license_route": "existing host local license", "python": sys.version, "controller_sha256": hashes, "instances": cases, "maximum_unique_generated_candidates": 81 * 10 * 19, "retry_delays_seconds": [60, 120, 300, 600, 900], "agy_isolation": "fresh workspace and conversation, native user settings retained, slash expansion disabled"}
    manifest["protocol_id"] = identity(manifest)
    save(root / "campaign.json", manifest)
    print(json.dumps({"root": str(root), "instances": len(cases), "models": GPT_ORDER + AGY_ORDER, "max_attempts": 10, "workers": 2, "shared_first_attempt": True}, indent=2))


def validate(root):
    manifest = read(root / "campaign.json")
    if not manifest:
        raise ProtocolError("Campaign manifest missing")
    for name, expected in manifest["controller_sha256"].items():
        if digest(root / "controller" / name) != expected:
            raise ProtocolError("Frozen controller changed")
    for case in manifest["instances"]:
        directory = root / "dataset" / case["suite"] / case["problem"] / case["instance"]
        if any(digest(directory / name) != sha for name, sha in case["sha256"].items()):
            raise ProtocolError("Frozen dataset changed")
    if manifest["workers"] != 2 or manifest["max_attempts"] != 10:
        raise ProtocolError("Campaign worker or attempt budget changed")
    check = dict(manifest)
    protocol = check.pop("protocol_id")
    if identity(check) != protocol:
        raise ProtocolError("Campaign protocol changed")
    return manifest


def pair_complete(root, model, case, protocol):
    from src.agent.policy_completion import TerminalCompletionError, approved_receipts, load_policy_summary, policy_is_complete
    try:
        approvals = approved_receipts(root)
    except TerminalCompletionError as error:
        raise ProtocolError(str(error)) from error
    for mode in ("plain", "reflection"):
        directory = root / "runs" / model / mode / case["suite"] / case["problem"] / case["instance"]
        try:
            summary = load_policy_summary(directory, campaign_root=root, approvals=approvals)
            if not policy_is_complete(summary, 10, protocol):
                if summary.get("complete"):
                    raise ProtocolError("Completed pair has an incompatible protocol")
                return False
        except TerminalCompletionError as error:
            raise ProtocolError(str(error)) from error
    return True


def run(root):
    manifest = validate(root)
    if Path(__file__).resolve().parents[1] != root / "controller":
        raise ProtocolError("Run from the frozen native controller")
    state = {"state": "running", "pid": os.getpid(), "started_at": now(), "active_model": None, "active_instances": {}, "workers": 2, "agy_state": "native_adapter_ready", "models": {}}
    mutex = threading.RLock()
    def checkpoint():
        with mutex:
            state["updated_at"] = now()
            save(root / "status.json", state)
    def job(model, case):
        worker = threading.current_thread().name
        def phase(**fields):
            with mutex:
                state["active_instances"][worker].update(fields)
                checkpoint()
        with mutex:
            state["active_instances"][worker] = {"model": model, "instance": case["instance"], "started_at": now(), "mode": "shared", "attempt": 1, "phase": "starting"}
            checkpoint()
        try:
            client = CodexClient(model, effort="medium") if model in GPT_ORDER else AgyClient(model)
            reflection_client = CodexClient(model, effort="medium", role="reflection") if model in GPT_ORDER else AgyClient(model, role="reflection")
            judge = CodexClient(manifest["judge_model"], effort=manifest["judge_effort"], role="algorithm_judge")
            runner = PairedRunner(client=client, judge_client=judge, reflection_client=reflection_client, protocol_id=manifest["protocol_id"], model=model, suite=case["suite"], problem=[case["problem"]], max_attempts=10, solver_timeout=60, agent_timeout=900, solver_threads=1, solver_cooldown_seconds=10, on_stage=phase)
            source = root / "dataset" / case["suite"] / case["problem"] / case["instance"]
            return runner.run_pair(source, root / "runs" / model)
        finally:
            with mutex:
                state["active_instances"].pop(worker, None)
                checkpoint()
    def stop(signum, frame):
        raise KeyboardInterrupt("Native campaign stop requested")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with runtime_lock("fixed81-native-campaign"):
        try:
            for model in manifest["gpt_order"] + manifest["agy_order"]:
                state["active_model"] = model
                pending = [case for case in manifest["instances"] if not pair_complete(root, model, case, manifest["protocol_id"])]
                state["models"][model] = {"completed_pairs": 81 - len(pending), "expected": 81}
                checkpoint()
                error = None
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="experiment") as pool:
                    queue = iter(pending)
                    active = {}
                    def submit():
                        case = next(queue, None)
                        if case:
                            active[pool.submit(job, model, case)] = case
                    submit()
                    submit()
                    while active:
                        finished, _ = wait(active, return_when=FIRST_COMPLETED)
                        for future in finished:
                            active.pop(future)
                            try:
                                future.result()
                                with mutex:
                                    state["models"][model]["completed_pairs"] += 1
                                    checkpoint()
                            except Exception as caught:
                                error = caught
                                state.update(error=f"{type(caught).__name__}: {caught}", state="draining_for_recovery")
                                checkpoint()
                        if error is None:
                            for _ in finished:
                                submit()
                if error:
                    raise error
            state["state"] = "complete"
            checkpoint()
            return 0
        except KeyboardInterrupt:
            state.update(state="stopped", error="Termination requested")
            checkpoint()
            return 130
        except Exception as error:
            state.update(state="infrastructure_stopped", error=f"{type(error).__name__}: {error}")
            checkpoint()
            print(state["error"], flush=True)
            return 64 if isinstance(error, ProtocolError) else 75


def watch(root):
    """Keep monitoring and resume durable stages after backoff or a cooling stop."""
    manifest = validate(root)
    monitor_root = root / "monitor"
    monitor_root.mkdir(exist_ok=True)
    attempts = 0
    def stop(signum, frame):
        raise KeyboardInterrupt("Watchdog termination requested")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    guard_options = {k: v for k, v in manifest["resource_guard"].items() if k != "sample_interval"}
    command = [sys.executable, "-u", "-m", "src.native_campaign", "run", "--root", str(root)]
    save(root / "watchdog.json", {"pid": os.getpid(), "started_at": now(), "command": command, "continuous_resource_monitor": True})
    try:
        with runtime_lock("fixed81-native-watchdog"):
            while True:
                try:
                    result = supervise(command, monitor_root, interval=5, guard=ResourceGuard(**guard_options))
                except (RuntimeError, OSError) as error:
                    if type(error).__name__ == "CleanupError":
                        raise
                    result = 75
                    print(f"Monitoring startup failed: {type(error).__name__}: {error}", flush=True)
                if result in (0, 64, 130):
                    return result
                delay = manifest["retry_delays_seconds"][min(attempts, len(manifest["retry_delays_seconds"]) - 1)]
                attempts += 1
                with (root / "restarts.jsonl").open("a") as stream:
                    stream.write(json.dumps({"at": now(), "returncode": result, "backoff_seconds": delay, "restart": attempts}) + "\n")
                state = read(root / "status.json") or {}
                state.update(state="recovering", watchdog_pid=os.getpid(), retry_after_seconds=delay, updated_at=now())
                save(root / "status.json", state)
                deadline, healthy = time.monotonic() + delay, 0
                while time.monotonic() < deadline or healthy < 3:
                    try:
                        _, thermal, free = sample_host()
                        warnings = thermal_warnings(thermal)
                        healthy = healthy + 1 if not warnings and free >= 15 else 0
                        save(monitor_root / "status.json", {"sampled_at_utc": now(), "state": "cooling", "system_free_percent": free, "thermal": thermal, "cpu_percent": 0, "rss_mib": 0, "reason": f"Recovery backoff / waiting for healthy resources; restart {attempts}"})
                    except (RuntimeError, OSError):
                        healthy = 0
                    time.sleep(5)
    except KeyboardInterrupt:
        state = read(root / "status.json") or {}
        state.update(state="stopped", updated_at=now(), error="Watchdog stopped by user")
        save(root / "status.json", state)
        return 130


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "run", "watch"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "init":
        initialize(root, args.data.resolve())
        return
    raise SystemExit(run(root) if args.command == "run" else watch(root))


if __name__ == "__main__":
    main()
