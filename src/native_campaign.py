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
import traceback

from src.agent.paired_runner import InfrastructureError, PairedRunner, ProtocolError, digest, identity, read, save
from src.agent.resource_monitor import CleanupError, ResourceGuard, sample_host, supervise, thermal_warnings
from src.agent.runtime_state import runtime_lock, state_root
from src.agent.prompt_policy import FEEDBACK_POLICY, WORKSPACE_POLICY, REFLECTION_POLICY
from src.campaign_plan import load_plan, model_queue
from src.runtime_environment import RuntimeEnvironmentError, require_runtime_environment
from src.prompt_snapshot import PROMPT_PROTOCOL, judge_prompt_hashes, snapshot_prompt
from src.execution_environment import load_execution_environment, validate_execution_environment, require_backend_readiness


FAMILIES = ["ufl", "fctp", "km", "lrp", "tsso", "bp", "csp", "gap", "shift"]


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize(root, data, plan_path=None, *, include_optional=False):
    if plan_path is None:
        raise ProtocolError("New campaigns require an explicit --plan; legacy frozen campaigns are preserved")
    plan = load_plan(plan_path)
    if root.exists():
        raise ProtocolError("Choose a new campaign root; existing experiments are preserved")
    environment = require_runtime_environment()
    execution_environment = load_execution_environment(plan, plan_path)
    paths = list(data.glob("*/*/*/data.json"))
    if len(paths) != 81:
        raise ProtocolError("Exactly 81 existing fixed instances are required")
    cases = []
    for path in paths:
        suite, problem, name = path.relative_to(data).parts[:3]
        _, size, seed = name.rsplit("_", 2)
        if problem not in FAMILIES or size not in ("small", "medium", "large") or seed not in ("seed000", "seed001", "seed002"):
            raise ProtocolError("Dataset identity differs from the fixed 81")
        hashes = {filename: digest(path.parent / filename) for filename in ("data.json", "prompt.txt", "meta.json", "gold_monolithic.json")}
        cases.append({"suite": suite, "problem": problem, "instance": name, "size": size, "seed": int(seed[4:]), "sha256": hashes})
    expected = {(p, s, k) for p in FAMILIES for s in ("small", "medium", "large") for k in range(3)}
    if {(c["problem"], c["size"], c["seed"]) for c in cases} != expected:
        raise ProtocolError("Fixed dataset coverage mismatch")
    cases.sort(key=lambda c: (["small", "medium", "large"].index(c["size"]), c["seed"], FAMILIES.index(c["problem"])))
    if identity(cases) != plan["dataset_inventory_sha256"]:
        raise ProtocolError("Dataset differs from the fixed-81 inventory pinned in the plan")
    for case in cases:
        case["source_sha256"] = dict(case["sha256"])
        relative = Path(case["suite"]) / case["problem"] / case["instance"]
        destination = root / "dataset" / relative
        destination.mkdir(parents=True)
        for filename, expected_hash in case["sha256"].items():
            shutil.copyfile(data / relative / filename, destination / filename)
            if digest(destination / filename) != expected_hash:
                raise ProtocolError("Dataset copy differs")
        case["prompt_snapshot"] = snapshot_prompt(destination, case["suite"], case["problem"])
        case["sha256"]["prompt.txt"] = digest(destination / "prompt.txt")
        case["sha256"]["prompt.original.txt"] = digest(destination / "prompt.original.txt")
        for filename in case["sha256"]:
            (destination / filename).chmod(0o444)
    source = Path(__file__).resolve().parent
    frozen = root / "controller"
    shutil.copytree(source, frozen / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(source.parent / "docker", frozen / "docker", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for filename in ("pyproject.toml", ".python-version"):
        shutil.copyfile(source.parent / filename, frozen / filename)
    hashes = {str(p.relative_to(frozen)): digest(p) for p in frozen.rglob("*") if p.is_file()}
    manifest = {"schema_version": 3, "created_at": now(), "backend": "native", "scope": "existing_fixed_81",
                "data_regenerated": False, "original_data_root": str(data), "modes": ["plain", "reflection"],
                "resource_guard": {"sample_interval": 5, "max_cpu_percent": 250, "max_rss_mib": 2048, "min_free_percent": 10, "sustained_samples": 3},
                "license_route": "existing host local license", "python": sys.version, "runtime_environment": environment, "controller_sha256": hashes, "instances": cases,
                "retry_delays_seconds": [60, 120, 300, 600, 900], "agy_isolation": "fresh Docker workspace and conversation; only dedicated authentication state retained; slash expansion disabled",
                "agent_execution": "docker", "solver_execution": "native-macos-seatbelt", "execution_environment": execution_environment}
    queue = plan["model_queue"] + (plan["optional_queue"] if include_optional else [])
    manifest.update(plan["execution"])
    manifest.update(schema_version=3, experiment_plan=plan, experiment_plan_id=identity(plan),
                    prompt_policies=plan["prompt_policies"], model_queue=queue, historical_reuse=plan["historical_reference"],
                    wrong_target_reflection_failure=plan["wrong_target_reflection_failure"],
                    reflection_policy=plan["reflection_policy"],
                    prompt_protocol=PROMPT_PROTOCOL, algorithm_judge_prompt_sha256=judge_prompt_hashes(),
                    gpt_order=[item["model"] for item in queue if item["backend"] == "codex-cli"],
                    agy_order=[item["model"] for item in queue if item["backend"] == "agy-cli"],
                    claude_order=[item["model"] for item in queue if item["backend"] == "claude-code-cli"],
                    maximum_unique_generated_candidates=81 * len(queue) * 19,
                    fresh_execution_pairs=81 * len(queue), historical_reference_in_fresh_totals=False)
    manifest["protocol_id"] = identity(manifest)
    save(root / "campaign.json", manifest)
    save(root / "pause_request.json", {"recorded_at": now(), "resume_requires_explicit_user_request": True,
                                      "reason": "Fresh campaign prepared for review; initialization does not authorize execution."})
    print(json.dumps({"root": str(root), "instances": len(cases), "models": [item["model"] for item in queue], "max_attempts": 10, "workers": 2, "shared_first_attempt": True}, indent=2))


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
    from src.campaign_extension import validate_lineage
    validate_lineage(root, manifest)
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


def execution_queue(manifest, models=None):
    """Resolve an operational launch scope without changing the frozen cohort."""
    queue = model_queue(manifest)
    if models is None:
        return queue
    if not models or len(models) != len(set(models)):
        raise ProtocolError("Execution scope must contain distinct model IDs")
    unknown = set(models) - {item["model"] for item in queue}
    if unknown:
        raise ProtocolError(f"Execution scope contains models outside the frozen cohort: {sorted(unknown)}")
    selected = set(models)
    return [item for item in queue if item["model"] in selected]


def run(root, models=None):
    manifest = validate(root)
    ensure_execution_allowed(root, manifest)
    queue = execution_queue(manifest, models)
    require_campaign_environment(manifest)
    environment = manifest["execution_environment"]
    readiness = require_backend_readiness(environment, [item["backend"] for item in queue] + ["codex-cli"])
    if Path(__file__).resolve().parents[1] != root / "controller":
        raise ProtocolError("Run from the frozen native controller")
    selected_models = [item["model"] for item in queue]
    state = {"state": "running", "pid": os.getpid(), "started_at": now(), "active_model": None, "active_instances": {}, "workers": 2, "agy_state": "docker_adapter_ready", "models": {}, "readiness": readiness,
             "execution_scope": selected_models, "scope_expected_pairs": len(queue) * len(manifest["instances"]), "scope_complete": False}
    save(root / "launches" / f"{time.time_ns()}.json", {"started_at": state["started_at"], "controller_pid": os.getpid(), "protocol_id": manifest["protocol_id"], "execution_scope": selected_models, "scope_expected_pairs": state["scope_expected_pairs"]})
    mutex = threading.RLock()
    from src.native_execution_phase import NativeCallGate, auth_failure_marker
    gate = NativeCallGate(state_root(), root)
    from src.agent.docker_client import DockerCliClient, AuthenticationRequired
    from src.agent.native_solver import NativeSandboxExecutor
    from src.hybrid_runtime import runtime
    engine = runtime()
    engine.bind_environment(environment)
    if engine.image_id(environment["image_id"]) != environment["image_id"]:
        raise RuntimeEnvironmentError("The frozen Docker image is unavailable")
    if any(item["backend"] == "agy-cli" for item in queue) and not os.environ.get("DECOMPBENCH_AGY_KEYRING_PASSPHRASE"):
        raise AuthenticationRequired("Supply the benchmark AGY keyring passphrase privately in the local launch environment")
    solver_executor = NativeSandboxExecutor(license_file=Path(environment["license_file"]), expected_environment=environment["native_environment"])
    class CampaignCodexClient(DockerCliClient):
        def edit_file(self, target, prompt, *, context=None, log_dir=None, step="agent"):
            result = super().edit_file(target, prompt, context=context, log_dir=log_dir, step=step)
            stderr_path = (log_dir or target.parent) / f"{step}_stderr.log"
            marker = auth_failure_marker(result, stderr_path.read_text() if stderr_path.exists() else "")
            if marker:
                gate.fallback(marker, role=self.role, invocation_id=Path(log_dir or target.parent).name)
            return result
    def call_lock(client):
        backend = client.protocol()["backend"]
        return gate.enter() if backend == "codex-cli" else runtime_lock("native-agy-auth" if backend == "agy-cli" else "native-claude-auth")
    specifications = {item["model"]: item for item in queue}
    def make_client(model, role):
        specification = {"backend": "codex-cli", "effort": manifest["judge_effort"]} if role == "algorithm_judge" else specifications[model]
        backend = specification["backend"]
        cls = CampaignCodexClient if backend == "codex-cli" else DockerCliClient
        return cls(model, backend=backend, image_id=environment["image_id"], runtime=engine, evidence_root=root / "diagnostics/runtime", role=role, effort=specification["effort"], timeout=manifest["agent_timeout"])
    def checkpoint():
        with mutex:
            state["updated_at"] = now()
            state["active_codex_invocation_limit"] = gate.limit
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
            client = make_client(model, "generation")
            reflection_client = make_client(model, "reflection")
            judge = make_client(manifest["judge_model"], "algorithm_judge")
            runner = PairedRunner(client=client, judge_client=judge, reflection_client=reflection_client, protocol_id=manifest["protocol_id"], model=model, suite=case["suite"], problem=[case["problem"]], max_attempts=10, solver_timeout=60, agent_timeout=900, solver_threads=1, solver_cooldown_seconds=10, on_stage=phase, call_lock=call_lock, solver_executor=solver_executor)
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
            admission_stopped = False
            def stop_admission():
                nonlocal admission_stopped
                if pause_requested(root):
                    admission_stopped = True
                    state.update(state="draining_for_pause")
                    checkpoint()
                return admission_stopped
            for model in specifications:
                if stop_admission():
                    break
                state["active_model"] = model
                pending = [case for case in manifest["instances"] if not pair_complete(root, model, case, manifest["protocol_id"])]
                state["models"][model] = {"completed_pairs": len(manifest["instances"]) - len(pending), "expected": len(manifest["instances"])}
                checkpoint()
                error = None
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="experiment") as pool:
                    queue = iter(pending)
                    active = {}
                    def submit():
                        if stop_admission():
                            return
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
                if admission_stopped:
                    break
            remaining_models = [item["model"] for item in model_queue(manifest) if any(not pair_complete(root, item["model"], case, manifest["protocol_id"]) for case in manifest["instances"])]
            scope_complete = not any(model in specifications for model in remaining_models)
            state.update(state="paused" if remaining_models or admission_stopped else "complete", active_model=None, scope_complete=scope_complete, campaign_complete=not remaining_models, remaining_models=remaining_models)
            if remaining_models or admission_stopped:
                pause = read(root / "pause_request.json") or {}
                pause.update(recorded_at=now(), state="paused", resume_requires_explicit_user_request=True, reason=pause.get("reason", "user_requested_drain") if admission_stopped else "launch_scope_complete", remaining_models=remaining_models)
                if scope_complete:
                    pause["completed_execution_scope"] = selected_models
                save(root / "pause_request.json", pause)
            checkpoint()
            return 0
        except KeyboardInterrupt:
            state.update(state="stopped", error="Termination requested")
            checkpoint()
            return 130
        except Exception as error:
            retryable = isinstance(error, InfrastructureError)
            state.update(state="infrastructure_stopped" if retryable else "repair_required", error=f"{type(error).__name__}: {error}")
            checkpoint()
            save(root / "controller_failures" / f"{time.time_ns()}.json", {"recorded_at": now(), "retryable": retryable, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
            print(state["error"], flush=True)
            return 75 if retryable else 64


def watch(root, models=None, coordinator_token=None):
    """Keep monitoring and resume durable stages after backoff or a cooling stop."""
    manifest = validate(root)
    ensure_execution_allowed(root, manifest)
    from src.campaign_extension import extension_parent
    parent = extension_parent(manifest)
    if parent is not None:
        coordination = read(Path(parent["coordinator_state"])) or {}
        if not coordinator_token or coordination.get("token") != coordinator_token:
            raise ProtocolError("Extension watcher needs its coordinator launch token")
    elif coordinator_token is not None:
        raise ProtocolError("Coordinator tokens apply only to linked extensions")
    selected_models = [item["model"] for item in execution_queue(manifest, models)]
    require_campaign_environment(manifest)
    from src.hybrid_runtime import runtime
    engine = runtime()
    engine.bind_environment(manifest["execution_environment"])
    monitor_root = root / "monitor"
    monitor_root.mkdir(exist_ok=True)
    attempts = 0
    def stop(signum, frame):
        raise KeyboardInterrupt("Watchdog termination requested")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    guard_options = {k: v for k, v in manifest["resource_guard"].items() if k != "sample_interval"}
    command = [sys.executable, "-u", "-m", "src.native_campaign", "run", "--root", str(root)]
    for model in selected_models:
        command.extend(["--model", model])
    from src.campaign_extension import watcher_lock
    lock_name, blocking = watcher_lock(root, manifest)
    try:
        lock = runtime_lock(lock_name) if blocking else runtime_lock(lock_name, blocking=False)
        with lock:
            save(root / "watchdog.json", {"pid": os.getpid(), "started_at": now(), "command": command, "execution_scope": selected_models, "continuous_resource_monitor": True, **({"coordinator_token": coordinator_token} if coordinator_token else {})})
            while True:
                ensure_execution_allowed(root, manifest)
                try:
                    try:
                        result = supervise(command, monitor_root, interval=5, guard=ResourceGuard(**guard_options), container_sampler=engine.resource_samples)
                    finally:
                        try:
                            engine.stop_campaign_workers(root)
                        except Exception as cleanup_error:
                            save(monitor_root / "status.json", {"state": "cleanup_failed", "reason": str(cleanup_error), "updated_at": now()})
                            raise CleanupError("Docker worker shutdown could not be verified; automatic restart disabled") from cleanup_error
                except (RuntimeError, OSError) as error:
                    if isinstance(error, RuntimeEnvironmentError) or type(error).__name__ == "CleanupError":
                        raise
                    result = 75
                    print(f"Monitoring startup failed: {type(error).__name__}: {error}", flush=True)
                if result != 75:
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


def pause_requested(root):
    pause = read(root / "pause_request.json") or {}
    from src.campaign_extension import parent_paused
    return bool(pause.get("resume_requires_explicit_user_request") or (root / "STOP").exists() or parent_paused(root))


def ensure_execution_allowed(root, manifest):
    from src.campaign_extension import require_parked_parent
    require_parked_parent(root, manifest)
    if manifest.get("prompt_policies") != {"feedback": FEEDBACK_POLICY, "workspace": WORKSPACE_POLICY}:
        raise ProtocolError("Use the original frozen controller for legacy data; this controller requires a fresh cohort")
    if manifest.get("reflection_policy") != REFLECTION_POLICY:
        raise ProtocolError("This controller requires the new reflection delivery policy; preserve the legacy controller")
    if manifest.get("prompt_protocol") != PROMPT_PROTOCOL:
        raise ProtocolError("This controller requires the reviewed prompt and output contracts")
    if pause_requested(root):
        raise ProtocolError("Campaign is paused; explicit user resumption is required")


def require_campaign_environment(manifest):
    environment = manifest.get("runtime_environment")
    if not isinstance(environment, dict):
        raise ProtocolError("Campaign has no frozen runtime environment; use its original frozen controller")
    if manifest.get("agent_execution") != "docker" or manifest.get("solver_execution") != "native-macos-seatbelt":
        raise ProtocolError("The new cohort requires Docker editing and restricted native execution")
    validate_execution_environment(manifest["execution_environment"])
    return require_runtime_environment(environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "run", "watch"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--coordinator-token", help="Nonsecret ownership token for the approved linked extension watchdog")
    parser.add_argument("--include-optional", action="store_true", help="Include the optional last-priority model in a new frozen plan")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--model", action="append", dest="models", help="Execute only this frozen model; repeat to select more, retaining frozen queue order")
    scope.add_argument("--all-models", action="store_true", help="Explicitly authorize dispatch of the complete frozen model queue")
    args = parser.parse_args()
    if args.command == "init" and (args.models or args.all_models):
        parser.error("Initialize the full plan first; execution scope applies only to run/watch")
    if args.command != "init" and args.models is None and not args.all_models:
        parser.error("run/watch requires an explicit --model selection or --all-models")
    root = args.root.resolve()
    try:
        if args.command == "init":
            initialize(root, args.data.resolve(), args.plan, include_optional=args.include_optional)
            return
        raise SystemExit(run(root, models=args.models) if args.command == "run" else watch(root, models=args.models, coordinator_token=args.coordinator_token))
    except (ProtocolError, RuntimeEnvironmentError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(64) from error
    except InfrastructureError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(75) from error
    except Exception as error:
        if root.is_dir():
            save(root / "controller_failures" / f"{time.time_ns()}.json", {"recorded_at": now(), "retryable": False, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(64) from error


if __name__ == "__main__":
    main()
