"""Run a frozen 81-instance campaign: init copies the code and data, run executes pairs, watch supervises resources.

Agents run in Docker; solvers run natively in the macOS sandbox. Each campaign root keeps its own frozen copy of
`src/` and `docker/` under `controller/`, and `run` only executes from that copy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
import traceback

from src.agent.prompts.method import DECOMPOSITION_POLICY
from src.agent.prompts.policy import FEEDBACK_POLICY, WORKSPACE_POLICY, REFLECTION_POLICY
from src.agent.prompts.snapshot import judge_prompt_hashes, snapshot_prompt
from src.experiments.agent_auth import require_backend_authentication
from src.experiments.data_context import accepted_prompt_protocol, context_delivery_policy, data_context_policy, decomposition_contract_policy
from src.experiments.docker_setup import load_execution_environment, validate_execution_environment, require_backend_readiness
from src.experiments.environment import RuntimeEnvironmentError, require_runtime_environment
from src.experiments.execution_policy import SMALL_SCALE_NATIVE_POLICY
from src.experiments.locks import runtime_lock, state_root
from src.experiments.paired_runner import PairedRunner
from src.experiments.plan import load_plan, model_queue
from src.experiments.resources import CleanupError, ResourceGuard, sample_host, supervise, thermal_warnings
from src.utils.errors import InfrastructureError, ProtocolError
from src.utils.families import PAPER_FAMILIES, cohort_tracks
from src.utils.records import digest, identity, now, read, save


# The paper's fixed 81 (a config without a cohort): its nine families, in the order that decides the inventory.
FAMILIES = list(PAPER_FAMILIES)


SOURCE = Path(__file__).resolve().parents[1]
REPOSITORY = SOURCE.parent
# The watchdog stops (repair required) after this many consecutive restarts whose failure carried an auth marker.
AUTH_RESTART_CAP = 3


def initialize(root, data, plan_path, runtime_path):
    plan = load_plan(plan_path)
    if root.exists():
        raise ProtocolError("Choose a new campaign root; existing experiments are preserved")
    environment = require_runtime_environment()
    execution_environment = load_execution_environment(runtime_path)
    # A plan may name a cohort of registered families (src.utils.families); without it, the paper's fixed 81. Each
    # family's small-scale track descriptor gives its sizes and seeds (3 x 3 for every family).
    families = plan.get("cohort", {}).get("families", FAMILIES)
    tracks = cohort_tracks(families)
    n_cases = sum(track.instance_count for track in tracks.values())
    paths = list(data.glob("*/*/*/data.json"))
    if len(paths) != n_cases:
        raise ProtocolError(f"Exactly {n_cases} existing fixed instances are required")
    cases = []
    for path in paths:
        suite, problem, name = path.relative_to(data).parts[:3]
        _, size, seed = name.rsplit("_", 2)
        if problem not in tracks or size not in tracks[problem].sizes or seed not in {f"seed{k:03d}" for k in tracks[problem].seeds}:
            raise ProtocolError("Dataset identity differs from the fixed 81")
        # A named cohort also freezes its decomposition reference, which may certify the gold value.
        frozen_files = ("data.json", "prompt.txt", "meta.json", "gold_monolithic.json") + ((f"gold_{suite}.json",) if "cohort" in plan else ())
        hashes = {filename: digest(path.parent / filename) for filename in frozen_files}
        cases.append({"suite": suite, "problem": problem, "instance": name, "size": size, "seed": int(seed[4:]), "sha256": hashes})
    expected = {(p, s, k) for p, track in tracks.items() for s in track.sizes for k in track.seeds}
    if {(c["problem"], c["size"], c["seed"]) for c in cases} != expected:
        raise ProtocolError("Fixed dataset coverage mismatch")
    sizes = list(dict.fromkeys(size for track in tracks.values() for size in track.sizes))
    cases.sort(key=lambda c: (sizes.index(c["size"]), c["seed"], families.index(c["problem"])))
    if identity(cases) != plan["dataset_inventory_sha256"]:
        raise ProtocolError("Dataset differs from the fixed-81 inventory pinned in the plan")
    # One contract id for the re-rendered instance prompts (and so the reflection prompts) and the method judge.
    contract = decomposition_contract_policy(plan["prompt_protocol"])
    for case in cases:
        case["source_sha256"] = dict(case["sha256"])
        relative = Path(case["suite"]) / case["problem"] / case["instance"]
        destination = root / "dataset" / relative
        destination.mkdir(parents=True)
        for filename, expected_hash in case["sha256"].items():
            shutil.copyfile(data / relative / filename, destination / filename)
            if digest(destination / filename) != expected_hash:
                raise ProtocolError("Dataset copy differs")
        case["prompt_snapshot"] = snapshot_prompt(destination, case["suite"], case["problem"], contract)
        case["sha256"]["prompt.txt"] = digest(destination / "prompt.txt")
        case["sha256"]["prompt.original.txt"] = digest(destination / "prompt.original.txt")
        for filename in case["sha256"]:
            (destination / filename).chmod(0o444)
    frozen = root / "controller"
    shutil.copytree(SOURCE, frozen / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(REPOSITORY / "docker", frozen / "docker", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for filename in ("pyproject.toml", ".python-version"):
        shutil.copyfile(REPOSITORY / filename, frozen / filename)
    hashes = {str(p.relative_to(frozen)): digest(p) for p in frozen.rglob("*") if p.is_file()}
    manifest = {"schema_version": 3, "created_at": now(), "backend": "native", "scope": plan["cohort"]["name"] if "cohort" in plan else "existing_fixed_81",
                "data_regenerated": False, "original_data_root": str(data), "modes": ["plain", "reflection"],
                "resource_guard": {"sample_interval": 5, "max_cpu_percent": 250, "max_rss_mib": 2048, "min_free_percent": 10, "sustained_samples": 3},
                "license_route": "existing host local license", "python": sys.version, "runtime_environment": environment, "controller_sha256": hashes, "instances": cases,
                "retry_delays_seconds": [60, 120, 300, 600, 900], "agy_isolation": "fresh Docker workspace and conversation; only dedicated authentication state retained; slash expansion disabled",
                "agent_execution": "docker", "solver_execution": "native-macos-seatbelt", "execution_environment": execution_environment}
    queue = plan["model_queue"]
    manifest.update(plan["execution"])
    manifest.update(schema_version=4, experiment_plan=plan, experiment_plan_id=identity(plan),
                    prompt_policies=plan["prompt_policies"], model_queue=queue,
                    reflection_policy=plan["reflection_policy"],
                    prompt_protocol=plan["prompt_protocol"], algorithm_judge_prompt_sha256=judge_prompt_hashes(contract),
                    maximum_unique_generated_candidates=n_cases * len(queue) * 19, fresh_execution_pairs=n_cases * len(queue))
    if "solver_output_policy" in plan:
        # Only a config that selects it records the output readback policy, so earlier manifests keep their layout.
        manifest["solver_output_policy"] = plan["solver_output_policy"]
        # ... and the storage layout of the solver-output evidence it implies (sandbox.EVIDENCE_SCHEMAS).
        from src.experiments.sandbox import EVIDENCE_SCHEMAS
        if EVIDENCE_SCHEMAS[plan["solver_output_policy"]] is not None:
            manifest["solver_evidence_schema"] = EVIDENCE_SCHEMAS[plan["solver_output_policy"]]
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
    return manifest


def pair_complete(root, model, case, protocol):
    from src.experiments.completion import TerminalCompletionError, approved_receipts, load_policy_summary, policy_is_complete
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
    if REPOSITORY != root / "controller":
        raise ProtocolError("Run from the frozen native controller")
    selected_models = [item["model"] for item in queue]
    state = {"state": "running", "pid": os.getpid(), "started_at": now(), "active_model": None, "active_instances": {}, "workers": 2, "agy_state": "docker_adapter_ready", "models": {}, "readiness": readiness,
             "execution_scope": selected_models, "scope_expected_pairs": len(queue) * len(manifest["instances"]), "scope_complete": False}
    save(root / "launches" / f"{time.time_ns()}.json", {"started_at": state["started_at"], "controller_pid": os.getpid(), "protocol_id": manifest["protocol_id"], "execution_scope": selected_models, "scope_expected_pairs": state["scope_expected_pairs"]})
    mutex = threading.RLock()
    from src.experiments.call_gate import NativeCallGate, auth_failure_marker
    gate = NativeCallGate(state_root(), root)
    from src.agent.docker_client import DockerCliClient, AuthenticationRequired
    from src.experiments.sandbox import NativeSandboxExecutor
    from src.experiments.docker_setup import runtime
    engine = runtime()
    engine.bind_environment(environment)
    if engine.image_id(environment["image_id"]) != environment["image_id"]:
        raise RuntimeEnvironmentError("The frozen Docker image is unavailable")
    solver_executor = NativeSandboxExecutor(license_file=Path(environment["license_file"]), expected_environment=environment["native_environment"], output_policy=manifest.get("solver_output_policy"),
                                            evidence_schema=manifest.get("solver_evidence_schema"))
    context_delivery = (manifest.get("prompt_protocol") or {}).get("context_delivery")
    auth_markers = []
    class CampaignClient(DockerCliClient):
        """Records non-terminal auth markers (terminal ones raise AuthenticationRequired in the client itself)."""
        def edit_file(self, target, prompt, *, context=None, log_dir=None, step="agent"):
            result = super().edit_file(target, prompt, context=context, log_dir=log_dir, step=step)
            stderr_path = (log_dir or target.parent) / f"{step}_stderr.log"
            marker = auth_failure_marker(result, stderr_path.read_text() if stderr_path.exists() else "")
            if marker:
                if not result.ok:
                    with mutex:
                        auth_markers.append({"backend": self.backend, "role": self.role, "marker": marker, "invocation_id": Path(log_dir or target.parent).name})
                if self.backend == "codex-cli":
                    gate.fallback(marker, role=self.role, invocation_id=Path(log_dir or target.parent).name)
            return result
    @contextmanager
    def call_lock(client):
        """Codex: a gate slot and that slot's own auth volume; Claude Code and AGY: one call at a time per backend."""
        backend = client.protocol()["backend"]
        if backend == "codex-cli":
            with gate.enter() as slot, client.bind_auth_slot(slot):
                yield
        else:
            with runtime_lock("native-agy-auth" if backend == "agy-cli" else "native-claude-auth"):
                yield
    specifications = {item["model"]: item for item in queue}
    def make_client(model, role):
        specification = {"backend": "codex-cli", "effort": manifest["judge_effort"]} if role == "algorithm_judge" else specifications[model]
        backend = specification["backend"]
        cls = CampaignClient
        # The judge keeps the pasted context (the solver.py under review); generation and reflection follow the protocol.
        delivery = {"context_delivery": context_delivery} if context_delivery is not None and role != "algorithm_judge" else {}
        return cls(model, backend=backend, image_id=environment["image_id"], runtime=engine, evidence_root=root / "diagnostics/runtime", role=role, effort=specification["effort"], timeout=manifest["agent_timeout"], **delivery)
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
            runner = PairedRunner(client=client, judge_client=judge, reflection_client=reflection_client, protocol_id=manifest["protocol_id"], model=model, suite=case["suite"], problem=[case["problem"]], max_attempts=10, solver_timeout=60, agent_timeout=900, solver_threads=1, solver_cooldown_seconds=10, on_stage=phase, call_lock=call_lock, solver_executor=solver_executor, data_context=(manifest.get("prompt_protocol") or {}).get("data_context"), decomposition_contract=(manifest.get("prompt_protocol") or {}).get("decomposition_contract", DECOMPOSITION_POLICY), context_delivery=context_delivery)
            source = root / "dataset" / case["suite"] / case["problem"] / case["instance"]
            return runner.run_pair(source, root / "runs" / model)
        finally:
            with mutex:
                state["active_instances"].pop(worker, None)
                checkpoint()
    def stop(signum, frame):
        from src.agent import auth
        auth.CANCEL.set()  # threads waiting for an auth-volume lock give up at once
        raise KeyboardInterrupt("Native campaign stop requested")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    auth_failures = []  # every AuthenticationRequired seen, also those a more severe exit error does not carry
    with runtime_lock("fixed81-native-campaign"):
        try:
            if any(item["backend"] == "agy-cli" for item in queue) and not os.environ.get("DECOMPBENCH_AGY_KEYRING_PASSPHRASE"):
                from src.agent.auth import AUTH_VOLUMES
                raise AuthenticationRequired("Supply the benchmark AGY keyring passphrase privately in the local launch environment",
                                             {"kind": "passphrase_missing", "backend": "agy-cli", "auth_volume": AUTH_VOLUMES["agy-cli"]})
            # Admission: every queued backend and the Codex judge must pass the no-inference credential check.
            backends = [item["backend"] for item in queue] + ["codex-cli"]
            catalog = {}
            for item in queue:
                catalog.setdefault(item["backend"], set()).add(item["model"])
            catalog.setdefault("codex-cli", set()).add(manifest["judge_model"])
            state["authentication"] = require_backend_authentication(engine, environment["image_id"], backends, models={backend: sorted(names) for backend, names in catalog.items()},
                                                                     codex_slots=gate.limit, agy_passphrase=os.environ.get("DECOMPBENCH_AGY_KEYRING_PASSPHRASE"),
                                                                     record_dir=root / "diagnostics" / "auth_checks")
            resolve_auth_sentinel(root, state["authentication"])
            checkpoint()
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
                                if isinstance(caught, AuthenticationRequired):
                                    auth_failures.append(caught)
                                # A repair-required failure is never replaced by a later retryable one while draining.
                                if error is None or (isinstance(error, InfrastructureError) and not isinstance(caught, InfrastructureError)):
                                    error = caught
                                state.update(error=f"{type(error).__name__}: {error}", state="draining_for_recovery")
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
            for failure_error in auth_failures + ([error] if isinstance(error, AuthenticationRequired) and error not in auth_failures else []):
                record_auth_sentinel(root, failure_error, state.get("authentication"))
            checkpoint()
            failure = {"recorded_at": now(), "retryable": retryable, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
            if retryable and auth_markers:
                failure["auth_markers"] = auth_markers[-5:]
            save(root / "controller_failures" / f"{time.time_ns()}.json", failure)
            print(state["error"], flush=True)
            return 75 if retryable else 64


def watch(root, models=None):
    """Keep monitoring and resume durable stages after backoff or a cooling stop."""
    manifest = validate(root)
    ensure_execution_allowed(root, manifest)
    selected_models = [item["model"] for item in execution_queue(manifest, models)]
    require_campaign_environment(manifest)
    from src.experiments.docker_setup import runtime
    engine = runtime()
    engine.bind_environment(manifest["execution_environment"])
    monitor_root = root / "monitor"
    monitor_root.mkdir(exist_ok=True)
    attempts = auth_restarts = 0
    def stop(signum, frame):
        raise KeyboardInterrupt("Watchdog termination requested")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    guard_options = {k: v for k, v in manifest["resource_guard"].items() if k != "sample_interval"}
    command = [sys.executable, "-u", "-m", "src.experiments.campaign", "run", "--root", str(root)]
    for model in selected_models:
        command.extend(["--model", model])
    try:
        with runtime_lock("fixed81-native-watchdog"):
            save(root / "watchdog.json", {"pid": os.getpid(), "started_at": now(), "command": command, "execution_scope": selected_models, "continuous_resource_monitor": True})
            while True:
                ensure_execution_allowed(root, manifest)
                launched = time.time_ns()
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
                auth_restarts = auth_restarts + 1 if latest_failure_has_auth_markers(root, launched) else 0
                if auth_restarts >= AUTH_RESTART_CAP:
                    state = read(root / "status.json") or {}
                    state.update(state="repair_required", updated_at=now(), error=f"{auth_restarts} consecutive restarts failed with authentication markers; check the credentials with `docker auth --check`")
                    save(root / "status.json", state)
                    print(state["error"], flush=True)
                    return 64
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


def latest_failure_has_auth_markers(root, since_ns):
    """Whether the newest controller failure written after `since_ns` was retryable and carried an auth marker."""
    failures = sorted((root / "controller_failures").glob("*.json"), key=lambda path: path.name)
    for path in reversed(failures):
        if not path.stem.isdigit() or int(path.stem) < since_ns:
            return False
        receipt = read(path) or {}
        return bool(receipt.get("retryable") and receipt.get("auth_markers"))
    return False


def record_auth_sentinel(root, error, authentication=None):
    """root/auth_invalid.json: the credential failures that stopped the campaign (shown by the dashboard): per failed
    volume its backend, the failure time and the credential generation the last admission check saw (informational)."""
    evidence = getattr(error, "evidence", {}) or {}
    volumes, observed = {}, getattr(error, "raised_at", None) or now()
    for report in [evidence] + list(evidence.get("reports", [])):
        volume = report.get("auth_volume") or report.get("volume")
        if volume:
            generation = report["generation"] if "generation" in report else (authentication or {}).get(volume, {}).get("generation")
            # The time the failure was observed (the call record or the check), not the later write after draining.
            failed_at = report.get("recorded_at") or report.get("checked_at") or observed
            volumes[volume] = {"backend": report.get("backend"), "generation": generation, "failed_at": failed_at}
    previous = read(root / "auth_invalid.json") or {}
    errors = (previous.get("errors") or ([previous["error"]] if previous.get("error") else [])) + [str(error)]
    save(root / "auth_invalid.json", {"recorded_at": now(), "error": str(error), "errors": errors[-10:], "evidence": evidence, "volumes": (previous.get("volumes") or {}) | volumes})


def stored_after(checked, entry):
    """Whether the checked credential was stored (meta.json created_at) after the recorded failure."""
    from datetime import datetime
    try:
        return datetime.fromisoformat(checked.get("created_at")) > datetime.fromisoformat(entry.get("failed_at"))
    except (TypeError, ValueError):
        return False


def resolve_auth_sentinel(root, summary):
    """Clear the sentinel's volumes that this admission checked and passed; archive it once none is left.

    Claude Code's check is local only, so its volume is cleared only by a token stored after the failure (a new
    `--login`); otherwise the start is refused. A volume outside this launch scope stays recorded."""
    path = root / "auth_invalid.json"
    sentinel = read(path) if path.exists() else None
    if sentinel is None:
        return
    remaining = {}
    for volume, entry in (sentinel.get("volumes") or {}).items():
        checked = (summary or {}).get(volume)
        if checked is None:
            remaining[volume] = entry
        elif entry.get("backend") == "claude-code-cli" and not stored_after(checked, entry):
            from src.agent.docker_client import AuthenticationRequired
            raise AuthenticationRequired(f"The Claude Code token in {volume} failed ({sentinel.get('recorded_at')}) and has not been replaced since; "
                                         "store a new one with `python src/02_experiment.py docker auth --backend claude --login --replace`",
                                         {"kind": "credential_invalid", "backend": "claude-code-cli", "auth_volume": volume, "generation": entry.get("generation")})
    if remaining:
        save(path, sentinel | {"volumes": remaining})
    else:
        path.replace(root / f"auth_invalid.resolved-{time.time_ns()}.json")


def pause_requested(root):
    pause = read(root / "pause_request.json") or {}
    return bool(pause.get("resume_requires_explicit_user_request") or (root / "STOP").exists())


def ensure_execution_allowed(root, manifest):
    if manifest.get("prompt_policies") != {"feedback": FEEDBACK_POLICY, "workspace": WORKSPACE_POLICY}:
        raise ProtocolError("Use the original frozen controller for legacy data; this controller requires a fresh cohort")
    if manifest.get("reflection_policy") != REFLECTION_POLICY:
        raise ProtocolError("This controller requires the new reflection delivery policy; preserve the legacy controller")
    if not accepted_prompt_protocol(manifest.get("prompt_protocol")):
        raise ProtocolError("This controller requires the reviewed prompt and output contracts")
    if (manifest.get("experiment_plan") or {}).get("execution_policy", SMALL_SCALE_NATIVE_POLICY) != SMALL_SCALE_NATIVE_POLICY:
        raise ProtocolError("This controller runs only the native execution policy")
    if "solver_output_policy" in manifest:
        from src.experiments.sandbox import OUTPUT_LIMITS
        if manifest["solver_output_policy"] is None or manifest["solver_output_policy"] not in OUTPUT_LIMITS or context_delivery_policy(manifest["prompt_protocol"]) is None:
            raise ProtocolError("This controller does not know the campaign's solver output policy")
    if "solver_evidence_schema" in manifest:
        # Only a manifest of the 32 MiB policy records it; an earlier 32 MiB manifest without it keeps embedded records.
        from src.experiments.sandbox import EVIDENCE_SCHEMAS
        if manifest.get("solver_output_policy") is None or manifest["solver_evidence_schema"] is None or manifest["solver_evidence_schema"] != EVIDENCE_SCHEMAS.get(manifest["solver_output_policy"]):
            raise ProtocolError("This controller does not know the campaign's solver evidence schema")
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


def resume(root):
    """Record an explicit start authorization; init always leaves a new campaign paused."""
    pause = read(root / "pause_request.json") or {}
    pause.update(recorded_at=now(), resume_requires_explicit_user_request=False, state="resumed")
    save(root / "pause_request.json", pause)
    (root / "STOP").unlink(missing_ok=True)


def status(root):
    manifest = validate(root)
    state = read(root / "status.json") or {"state": "not_started"}
    done = {item["model"]: sum(pair_complete(root, item["model"], case, manifest["protocol_id"]) for case in manifest["instances"]) for item in model_queue(manifest)}
    return {"root": str(root), "state": state.get("state"), "paused": pause_requested(root), "completed_pairs": done, "expected_pairs_per_model": len(manifest["instances"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "resume", "run", "watch", "status"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/paper.json"))
    parser.add_argument("--runtime", type=Path, default=Path("outputs/runtime/runtime.json"), help="runtime.json written by `docker prepare`")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--model", action="append", dest="models", help="Execute only this model; repeat to select more, keeping the config order")
    scope.add_argument("--all-models", action="store_true", help="Execute the complete model queue")
    args = parser.parse_args(argv)
    if args.command in ("run", "watch") and args.models is None and not args.all_models:
        parser.error("run/watch requires --model or --all-models")
    root = args.root.resolve()
    try:
        if args.command == "init":
            initialize(root, args.data.resolve(), args.config, args.runtime)
        elif args.command == "resume":
            resume(root)
        elif args.command == "status":
            print(json.dumps(status(root), indent=2))
        else:
            raise SystemExit(run(root, models=args.models) if args.command == "run" else watch(root, models=args.models))
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
