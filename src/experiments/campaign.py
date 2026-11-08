"""Run a frozen 81-instance campaign: init copies the code and data, run executes pairs, watch supervises resources.

Agents run in Docker; solvers run natively in the macOS sandbox. Each campaign root keeps its own frozen copy of
`src/` and `docker/` under `controller/`, and `run` only executes from that copy.

One native campaign controller runs on a host at a time (the campaign lock in the private state directory). The
small-scale protocol runs only in its lanes variant (configs/small_scale.json selects it), whose executor keeps the
host-wide ownership record of the running candidate (sandbox.OWNERSHIP_RECORD) with the launch handshake. A paper or
harder campaign keeps its sequential run and records unchanged; it only checks that record read-only at start and
refuses while an earlier lane launch may survive. Mixed-mode concurrent campaigns are unsupported.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from itertools import chain
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
import traceback

from src.agent import auth
from src.agent.prompts.method import DECOMPOSITION_POLICY
from src.agent.prompts.policy import FEEDBACK_POLICY, WORKSPACE_POLICY, REFLECTION_POLICY
from src.agent.prompts.snapshot import judge_prompt_hashes, snapshot_prompt
from src.experiments.agent_auth import require_backend_authentication
from src.experiments.data_context import accepted_prompt_protocol, context_delivery_policy, data_context_policy, decomposition_contract_policy
from src.experiments.docker_setup import load_execution_environment, require_slot_volumes, validate_execution_environment, require_backend_readiness
from src.experiments.environment import RuntimeEnvironmentError, require_runtime_environment
from src.experiments.execution_policy import SMALL_SCALE_LANES_POLICY, SMALL_SCALE_NATIVE_POLICY
from src.experiments.locks import runtime_lock, state_root
from src.experiments.paired_runner import PairedRunner
from src.experiments.plan import EXECUTION_SETTINGS, execution_policy_id, expected_specs, lane_partition, load_plan, model_queue
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
    if "lanes" in plan["execution"]:
        # Each call-gate slot mounts its own auth volume, which the prepared runtime evidence must declare.
        require_slot_volumes(execution_environment, {"codex-cli": plan["execution"]["active_codex_invocation_limit"], "claude-code-cli": plan["execution"]["active_claude_invocation_limit"]})
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
    # The execution settings become top-level manifest fields; the lanes variant's also replace the resource guard.
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
    lanes = {"lanes": {lane["name"]: {"workers": lane["workers"], "models": lane["models"]} for lane, _ in lane_partition(manifest["lanes"], queue)}} if "lanes" in manifest else {}
    print(json.dumps({"root": str(root), "instances": len(cases), "models": [item["model"] for item in queue], "max_attempts": 10, "workers": manifest["workers"], **lanes, "shared_first_attempt": True}, indent=2))


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
    # The paper's two workers; the lanes variant's workers are its lanes' (validated with the lanes in ensure_execution_allowed).
    workers = sum(lane.get("workers", 0) for lane in manifest["lanes"]) if execution_policy_id(manifest) == SMALL_SCALE_LANES_POLICY and isinstance(manifest.get("lanes"), list) else 2
    if manifest["workers"] != workers or manifest["max_attempts"] != 10:
        raise ProtocolError("Campaign worker or attempt budget changed")
    check = dict(manifest)
    protocol = check.pop("protocol_id")
    if identity(check) != protocol:
        raise ProtocolError("Campaign protocol changed")
    return manifest


def identity_settings(record):
    """The contract, context-delivery settings and spec ids of a case identity, or of a campaign's expected settings
    (completion_settings), compared by completion.case_settings: an absent contract is the paper's, absent specs none."""
    from src.experiments.completion import case_settings
    return case_settings(record)


def completion_settings(manifest):
    """What every closed pair of this campaign must carry: its prompt protocol and, under an execution variant, the spec
    ids its runners record (plan.expected_specs). The paper and harder campaigns expect no spec."""
    prompt_protocol = manifest.get("prompt_protocol") or {}
    specs = expected_specs(manifest)
    return prompt_protocol | {"specs": specs} if specs else prompt_protocol


def pair_complete(root, model, case, protocol, prompt_protocol=None):
    """Whether both policies of a (model, case) pair are closed under `protocol`. With the campaign's settings
    (`prompt_protocol`, completion_settings), each policy's case identity must also carry its contract, context
    delivery, judge context delivery and spec ids (the predicate PairedRunner.run_pair applies before it reuses a closed
    pair, and the dashboard)."""
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
        if prompt_protocol is not None and identity_settings(read(directory / "identity.json") or {}) != identity_settings(prompt_protocol):
            raise ProtocolError("Completed pair has an incompatible protocol")
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
    # The lanes variant: the selected scope partitioned into its lanes, in queue order (none: the sequential run).
    lanes = lane_partition(manifest.get("lanes"), model_queue(manifest), queue)
    lane_of = {item["model"]: lane["name"] for lane, members in lanes for item in members}
    require_campaign_environment(manifest)
    environment = manifest["execution_environment"]
    readiness = require_backend_readiness(environment, [item["backend"] for item in queue] + ["codex-cli"])
    if REPOSITORY != root / "controller":
        raise ProtocolError("Run from the frozen native controller")
    selected_models = [item["model"] for item in queue]
    state = {"state": "running", "pid": os.getpid(), "started_at": now(), "active_model": None, "active_instances": {}, "workers": sum(lane["workers"] for lane, _ in lanes) if lanes else 2, "agy_state": "docker_adapter_ready", "models": {}, "readiness": readiness,
             "execution_scope": selected_models, "scope_expected_pairs": len(queue) * len(manifest["instances"]), "scope_complete": False}
    if lanes:
        # Per lane: its workers, its models in this launch scope and the model it is working through (also in
        # active_models); the earlier active_model field follows the first lane. Gate limits are per backend (checkpoint).
        from src.experiments.sandbox import MEMORY_POLICIES
        state.update(lanes={lane["name"]: {"workers": lane["workers"], "models": [item["model"] for item in members], "active_model": None, "complete": False, "state": "waiting"} for lane, members in lanes},
                     active_models={}, solver_memory_limit_mib=MEMORY_POLICIES[manifest["solver_memory_policy"]]["limit_mib"], resource_guard=manifest["resource_guard"])
    mutex = threading.RLock()
    from src.experiments.call_gate import NativeCallGate, auth_failure_marker, exclusive_lock
    if lanes:
        # One gate per backend, shared by every lane and every method judge (Codex); each slot mounts its own volume.
        gates = {"codex-cli": NativeCallGate(state_root(), root, slots=manifest["active_codex_invocation_limit"]),
                 "claude-code-cli": NativeCallGate(state_root(), root, backend="claude-code-cli", slots=manifest["active_claude_invocation_limit"])}
        require_slot_volumes(environment, {backend: gate.slots for backend, gate in gates.items()})
    else:
        gates = {"codex-cli": NativeCallGate(state_root(), root)}
    gate = gates["codex-cli"]
    from src.agent.docker_client import DockerCliClient, AuthenticationRequired
    from src.experiments.sandbox import NativeSandboxExecutor
    from src.experiments.docker_setup import runtime
    engine = runtime()
    engine.bind_environment(environment)
    if engine.image_id(environment["image_id"]) != environment["image_id"]:
        raise RuntimeEnvironmentError("The frozen Docker image is unavailable")
    # The lanes variant's executor adds the memory budget, the durable ownership record with the launch handshake and the
    # solver-lock queue time. The ownership record is host-wide, in the private state directory of the campaign lock, so
    # a controller of any root refuses to start next to a candidate another one left running.
    from src.experiments.sandbox import OWNERSHIP_RECORD, ownership_problem
    variant = {"memory_policy": manifest["solver_memory_policy"], "ownership": state_root() / OWNERSHIP_RECORD, "record_queue_seconds": True} if lanes else {}
    solver_executor = NativeSandboxExecutor(license_file=Path(environment["license_file"]), expected_environment=environment["native_environment"], output_policy=manifest.get("solver_output_policy"),
                                            evidence_schema=manifest.get("solver_evidence_schema"), **variant)
    context_delivery = (manifest.get("prompt_protocol") or {}).get("context_delivery")
    judge_setting = (manifest.get("prompt_protocol") or {}).get("judge_context_delivery")
    from src.experiments.context_spec import context_spec, judge_delivery
    judge_transport, judge_layout = judge_delivery(judge_setting, context_spec((manifest.get("prompt_protocol") or {}).get("data_context"), context_delivery))
    auth_markers = []
    pair_markers = threading.local()
    # The lanes variant's admission latch, read and changed only under the mutex. A worker publishes its pair's outcome
    # here, in completion order, before its future completes, so no reservation can follow an unobserved failure and
    # the per-lane counters follow the order in which pairs ended (never the coordinator's consumption order). A
    # stopping failure closes admission in every lane; a retryable one blocks its own lane until the coordinator's
    # backoff re-admits it. "published"/"consumed" count the lane's retryable failures, so a backoff timer releases the
    # lane only when no newer failure awaits its own backoff.
    latch = {"closed": False, "verdicts": {}, "lanes": {lane["name"]: {"blocked": False, "auth_marked": 0, "failures": 0, "published": 0, "consumed": 0} for lane, _ in lanes}}

    def stops_campaign(error):
        """Whether a pair failure stops admission everywhere: every failure but a retryable infrastructure one (never an
        authentication, protocol, cleanup or environment failure)."""
        return not isinstance(error, InfrastructureError) or isinstance(error, (ProtocolError, CleanupError, RuntimeEnvironmentError))

    def publish(model, case, error=None):
        """In the worker, under the mutex, as its pair ends: a completed pair resets its lane's counters; a failure closes
        admission everywhere, or blocks only this pair's lane and counts toward its backoff. Retries whose calls carried
        authentication markers stop the campaign after AUTH_RESTART_CAP consecutive ones in a lane (the watchdog's rule
        for restarts)."""
        with mutex:
            name = lane_of[model]
            lane = latch["lanes"][name]
            if error is None:
                lane.update(auth_marked=0, failures=0)
                return
            stop = stops_campaign(error)
            if not stop:
                lane["auth_marked"] = lane["auth_marked"] + 1 if getattr(pair_markers, "markers", None) else 0
                if lane["auth_marked"] >= AUTH_RESTART_CAP:
                    error, stop = RuntimeEnvironmentError(f"{lane['auth_marked']} consecutive retries in lane {name} failed with authentication markers; "
                                                          "check the credentials with `docker auth --check`"), True
            if stop:
                latch["closed"] = True
            else:
                lane.update(blocked=True, failures=lane["failures"] + 1, published=lane["published"] + 1)
            latch["verdicts"][(model, case["instance"])] = ("stop" if stop else "retry", error, lane["failures"])

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
                    if getattr(pair_markers, "markers", None) is not None:
                        pair_markers.markers.append(marker)  # the markers of this worker's current pair (lane retries)
                if self.backend in gates:  # the fallback of the gate of this call's own backend
                    gates[self.backend].fallback(marker, role=self.role, invocation_id=Path(log_dir or target.parent).name)
            return result
    @contextmanager
    def call_lock(client):
        """Codex (and Claude Code in the lanes variant): a gate slot and that slot's own auth volume; otherwise one call at
        a time per backend (AGY always). The backend is the client's model specification's, never its lane's."""
        backend = client.protocol()["backend"]
        if backend in gates:
            with gates[backend].enter() as slot, client.bind_auth_slot(slot):
                yield
        else:
            with exclusive_lock(state_root(), "native-agy-auth" if backend == "agy-cli" else "native-claude-auth"):
                yield
    specifications = {item["model"]: item for item in queue}
    def make_client(model, role):
        specification = {"backend": "codex-cli", "effort": manifest["judge_effort"]} if role == "algorithm_judge" else specifications[model]
        backend = specification["backend"]
        cls = CampaignClient
        # Generation and reflection follow the protocol's context delivery; the judge pastes the sources under review
        # unless the protocol mounts them (judge_context_delivery: the transport and REVIEW_LAYOUT).
        if role == "algorithm_judge":
            delivery = {"context_delivery": judge_transport, "context_layout": judge_layout} if judge_transport is not None else {}
        else:
            delivery = {"context_delivery": context_delivery} if context_delivery is not None else {}
            if lanes:
                # The generation and reflection clients carry the runtime contract with the memory budget the runner states.
                from src.agent.specs import RuntimeSpec
                from src.experiments.sandbox import MEMORY_POLICIES
                delivery["runtime_spec"] = RuntimeSpec(60, 1, manifest.get("solver_output_policy"), MEMORY_POLICIES[manifest["solver_memory_policy"]]["limit_mib"])
        return cls(model, backend=backend, image_id=environment["image_id"], runtime=engine, evidence_root=root / "diagnostics/runtime", role=role, effort=specification["effort"], timeout=manifest["agent_timeout"], **delivery)
    def checkpoint():
        with mutex:
            state["updated_at"] = now()
            state["active_codex_invocation_limit"] = gate.limit
            if lanes:
                state.update(active_claude_invocation_limit=gates["claude-code-cli"].limit, active_agy_invocation_limit=1)
            save(root / "status.json", state)
    def job(model, case):
        worker = threading.current_thread().name
        def phase(**fields):
            with mutex:
                state["active_instances"][worker].update(fields)
                checkpoint()
        with mutex:
            if lanes and worker in state["active_instances"]:
                raise ProtocolError(f"Worker identity {worker} is already active")
            state["active_instances"][worker] = {"model": model, "instance": case["instance"], "started_at": now(), "mode": "shared", "attempt": 1, "phase": "starting"}
            if lanes:
                state["active_instances"][worker]["lane"] = lane_of[model]
            checkpoint()
        pair_markers.markers = []
        try:
            client = make_client(model, "generation")
            reflection_client = make_client(model, "reflection")
            judge = make_client(manifest["judge_model"], "algorithm_judge")
            options = {}
            if lanes:
                from src.experiments.execution_policy import SmallScaleLanesPolicy
                schema = getattr(solver_executor, "evidence_schema", None)
                options["execution_policy"] = SmallScaleLanesPolicy(executor=solver_executor, suite=case["suite"], cooldown_seconds=10,
                                                                    **({"evidence_schema": schema, "output_limit": solver_executor.output_limit} if schema else {}))
            runner = PairedRunner(client=client, judge_client=judge, reflection_client=reflection_client, protocol_id=manifest["protocol_id"], model=model, suite=case["suite"], problem=[case["problem"]], max_attempts=10, solver_timeout=60, agent_timeout=900, solver_threads=1, solver_cooldown_seconds=10, on_stage=phase, call_lock=call_lock, solver_executor=solver_executor, data_context=(manifest.get("prompt_protocol") or {}).get("data_context"), decomposition_contract=(manifest.get("prompt_protocol") or {}).get("decomposition_contract", DECOMPOSITION_POLICY), context_delivery=context_delivery, judge_context_delivery=judge_setting, **options)
            source = root / "dataset" / case["suite"] / case["problem"] / case["instance"]
            summaries = runner.run_pair(source, root / "runs" / model)
            if lanes:
                publish(model, case)
            return summaries
        except Exception as caught:
            if lanes:
                publish(model, case, caught)
            raise
        finally:
            pair_markers.markers = None
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
        # The launch record is written only by the controller that owns the campaign.
        save(root / "launches" / f"{time.time_ns()}.json", {"started_at": state["started_at"], "controller_pid": os.getpid(), "protocol_id": manifest["protocol_id"], "execution_scope": selected_models, "scope_expected_pairs": state["scope_expected_pairs"]})
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
            slots = {"claude_slots": gates["claude-code-cli"].limit} if "claude-code-cli" in gates else {}
            state["authentication"] = require_backend_authentication(engine, environment["image_id"], backends, models={backend: sorted(names) for backend, names in catalog.items()},
                                                                     codex_slots=gate.limit, agy_passphrase=os.environ.get("DECOMPBENCH_AGY_KEYRING_PASSPHRASE"),
                                                                     record_dir=root / "diagnostics" / "auth_checks", **slots)
            resolve_auth_sentinel(root, state["authentication"])
            if lanes and hasattr(solver_executor, "verify_ownership"):
                solver_executor.verify_ownership()  # no candidate of an earlier controller may still be running
            elif not lanes:
                _, problem = ownership_problem(state_root() / OWNERSHIP_RECORD)  # read-only: this mode writes no ownership record
                if problem:
                    raise CleanupError(problem)
            checkpoint()
            admission_stopped = False
            def stop_admission():
                nonlocal admission_stopped
                if pause_requested(root):
                    admission_stopped = True
                    state.update(state="draining_for_pause")
                    checkpoint()
                return admission_stopped
            expected = completion_settings(manifest)

            def run_lanes():
                """The lanes side by side under one campaign-wide admission owner (this thread reserves every pair, under the
                mutex, after checking the latch the workers publish to). A stopping failure in any lane or a pause request
                stops admission everywhere; admitted pairs still finish all their attempts, reflections and judging. A
                retryable failure blocks only its lane: that lane backs off (retry_delays_seconds, the last one repeated;
                a pause, STOP or termination ends the wait) and then re-admits the unfinished pair, whose committed calls
                are reused, while the other lanes keep admitting. Each lane works through its models in queue order, a
                model's admitted pairs finishing before its lane moves on. Returns the aggregated stopping failure
                (repair-required over retryable) once every admitted pair has finished; the caller finalizes once."""
                failure = None
                running, futures = set(), {}
                progress = {lane["name"]: {"workers": lane["workers"], "models": iter(members), "model": None, "cases": iter(()), "active": set(), "retry_at": None}
                            for lane, members in lanes}
                first = lanes[0][0]["name"]
                delays = manifest.get("retry_delays_seconds") or [60]

                def fail(caught):
                    nonlocal failure
                    with mutex:
                        latch["closed"] = True
                        if isinstance(caught, AuthenticationRequired):
                            auth_failures.append(caught)
                        # A repair-required failure is never replaced by a later retryable one while draining.
                        if failure is None or (isinstance(failure, InfrastructureError) and not isinstance(caught, InfrastructureError)):
                            failure = caught
                        state.update(error=f"{type(failure).__name__}: {failure}", state="draining_for_recovery")
                        checkpoint()

                def lane_state(name, **fields):
                    with mutex:
                        state["lanes"][name].update(fields)
                        checkpoint()

                def back_off(name, case, caught, failures):
                    """Block the lane until its next retry and put the unfinished pair back first in its queue; `failures` is
                    the lane's consecutive failure count when this failure was published."""
                    entry = progress[name]
                    delay = delays[min(failures - 1, len(delays) - 1)]
                    with mutex:
                        latch["lanes"][name]["consumed"] += 1
                    entry["cases"] = chain([case], entry["cases"])
                    entry["retry_at"] = max(entry["retry_at"] or 0.0, time.time() + delay)
                    retry_at = datetime.fromtimestamp(entry["retry_at"], timezone.utc).isoformat()
                    reason = f"{type(caught).__name__}: {caught}"
                    save(root / "lane_failures" / f"{time.time_ns()}.json", {"recorded_at": now(), "lane": name, "model": entry["model"], "instance": case["instance"], "retryable": True,
                                                                            "error_type": type(caught).__name__, "error": str(caught), "consecutive_failures": failures,
                                                                            "retry_after_seconds": delay, "retry_at": retry_at,
                                                                            "traceback": "".join(traceback.format_exception(type(caught), caught, caught.__traceback__))})
                    lane_state(name, state="backing_off", reason=reason, retry_at=retry_at, retry_after_seconds=delay, consecutive_failures=failures)

                def refill(name, pool):
                    entry = progress[name]
                    while True:
                        with mutex:  # the latch and the reservation together
                            if len(entry["active"]) >= entry["workers"] or failure is not None or latch["closed"] or latch["lanes"][name]["blocked"] or stop_admission():
                                return
                            case = next(entry["cases"], None)
                            if case is not None:
                                pair = (entry["model"], case["suite"], case["problem"], case["instance"])
                                if pair in running:
                                    raise ProtocolError(f"Pair {pair} is already admitted")
                                running.add(pair)
                                future = pool.submit(job, entry["model"], case)
                                entry["active"].add(future)
                                futures[future] = (name, entry["model"], case, pair)
                                continue
                        if entry["active"]:
                            return  # the model's admitted pairs finish before the lane moves on
                        item = next(entry["models"], None)
                        with mutex:
                            entry["model"] = state["lanes"][name]["active_model"] = item and item["model"]
                            state["lanes"][name].update(complete=item is None, state="complete" if item is None else "running")
                            state.update(active_models={lane: value["model"] for lane, value in progress.items() if value["model"]}, active_model=progress[first]["model"])
                            checkpoint()
                        if item is None:
                            return
                        try:
                            pending = [case for case in manifest["instances"] if not pair_complete(root, item["model"], case, manifest["protocol_id"], expected)]
                        except Exception as caught:
                            fail(caught)
                            return
                        with mutex:
                            state["models"][item["model"]] = {"completed_pairs": len(manifest["instances"]) - len(pending), "expected": len(manifest["instances"])}
                            checkpoint()
                        entry["cases"] = iter(pending)

                with ExitStack() as stack:
                    # Lane-qualified worker names (experiment-<lane>_<n>) key the shared active_instances.
                    pools = {lane["name"]: stack.enter_context(ThreadPoolExecutor(max_workers=lane["workers"], thread_name_prefix=f"experiment-{lane['name']}")) for lane, _ in lanes}
                    for name, pool in pools.items():
                        try:
                            refill(name, pool)
                        except ProtocolError as caught:
                            fail(caught)
                    while True:
                        with mutex:
                            stopping = failure is not None or latch["closed"]
                        waiting = [entry["retry_at"] for entry in progress.values() if entry["retry_at"] is not None]
                        if not futures and (stopping or admission_stopped or not waiting):
                            break
                        # A backoff wakes the coordinator at its retry time, and at least every 5 s for a pause or STOP.
                        timeout = min(5.0, max(0.0, min(waiting) - time.time())) if waiting else None
                        if futures:
                            finished, _ = wait(futures, timeout=timeout, return_when=FIRST_COMPLETED)
                        else:
                            finished = set()
                            if auth.CANCEL.wait(timeout):
                                raise KeyboardInterrupt("Native campaign stop requested")
                        for future in finished:
                            name, model, case, pair = futures.pop(future)
                            progress[name]["active"].discard(future)
                            running.discard(pair)
                            try:
                                future.result()
                                with mutex:
                                    state["models"][model]["completed_pairs"] += 1  # its lane's counters were reset when it ended (publish)
                                    checkpoint()
                            except Exception as caught:
                                with mutex:
                                    verdict, error, failures = latch["verdicts"].pop((model, case["instance"]), ("stop", caught, 0))
                                if verdict == "retry" and failure is None and not latch["closed"]:
                                    back_off(name, case, caught, failures)
                                elif verdict == "retry":
                                    with mutex:
                                        latch["lanes"][name]["consumed"] += 1
                                    lane_state(name, state="stopped", reason=f"{type(caught).__name__}: {caught}")
                                else:
                                    fail(error)
                        stop_admission()  # a pause request also ends a backoff without any completion
                        with mutex:
                            stopping = failure is not None or latch["closed"] or admission_stopped
                        for name, entry in progress.items():
                            if entry["retry_at"] is not None and (stopping or time.time() >= entry["retry_at"]):
                                with mutex:
                                    lane = latch["lanes"][name]
                                    if not stopping and lane["published"] > lane["consumed"]:
                                        continue  # a newer failure of this lane awaits its own backoff: the block stays
                                    entry["retry_at"] = None
                                    lane["blocked"] = stopping
                                lane_state(name, state="stopped" if stopping else "running", retry_at=None, retry_after_seconds=None)
                        for name, pool in pools.items():
                            try:
                                refill(name, pool)
                            except ProtocolError as caught:
                                fail(caught)
                with mutex:
                    state.update(active_models={}, active_model=None)
                    for value in state["lanes"].values():
                        value["active_model"] = None
                        if value.get("state") in ("running", "backing_off", "waiting") and not value["complete"]:
                            value.update(state="stopped", retry_at=None)  # a pause or a stopping failure ended this lane early
                return failure

            if lanes:
                failure = run_lanes()
                if failure:
                    raise failure
            for model in specifications if not lanes else ():  # the sequential run (no lanes)
                if stop_admission():
                    break
                state["active_model"] = model
                pending = [case for case in manifest["instances"] if not pair_complete(root, model, case, manifest["protocol_id"], expected)]
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
            remaining_models = [item["model"] for item in model_queue(manifest) if any(not pair_complete(root, item["model"], case, manifest["protocol_id"], expected) for case in manifest["instances"])]
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
            from src.agent.auth import repair_command
            from src.agent.docker_client import AuthenticationRequired
            raise AuthenticationRequired(f"The Claude Code token in {volume} failed ({sentinel.get('recorded_at')}) and has not been replaced since; "
                                         f"store a new one with `{repair_command('claude-code-cli', volume)}`",
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
    policy = (manifest.get("experiment_plan") or {}).get("execution_policy", SMALL_SCALE_NATIVE_POLICY)
    if policy not in (SMALL_SCALE_NATIVE_POLICY, SMALL_SCALE_LANES_POLICY):
        raise ProtocolError("This controller runs only the native execution policy")
    if policy == SMALL_SCALE_LANES_POLICY:
        # The reviewed lanes variant: its settings, with the memory budget the mounted prompts state.
        from src.experiments.sandbox import MEMORY_POLICIES
        if any(manifest.get(key) != value for key, value in EXECUTION_SETTINGS[policy].items()) or manifest.get("solver_memory_policy") not in MEMORY_POLICIES \
                or context_delivery_policy(manifest.get("prompt_protocol")) is None:
            raise ProtocolError("This controller does not know the campaign's execution variant")
        lane_partition(manifest.get("lanes"), model_queue(manifest))
        if manifest.get("workers") != sum(lane["workers"] for lane in manifest["lanes"]):
            raise ProtocolError("The lanes' workers must add up to the campaign's workers")
    elif "lanes" in manifest or "solver_memory_policy" in manifest:
        raise ProtocolError("Lanes and a memory budget belong to the reviewed lanes variant only")
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
    expected = completion_settings(manifest)
    done = {item["model"]: sum(pair_complete(root, item["model"], case, manifest["protocol_id"], expected) for case in manifest["instances"]) for item in model_queue(manifest)}
    summary = {"root": str(root), "state": state.get("state"), "paused": pause_requested(root), "completed_pairs": done, "expected_pairs_per_model": len(manifest["instances"])}
    if "lanes" in manifest:
        # The lanes variant: per lane its workers, models and current model (from the controller's status), the
        # per-backend call limits and the solver and aggregate budgets.
        from src.experiments.sandbox import MEMORY_POLICIES
        running = state.get("lanes") or {}
        summary.update(workers=manifest["workers"], lanes=[{"name": lane["name"], "workers": lane["workers"], "models": lane["models"], "active_model": (running.get(lane["name"]) or {}).get("active_model"),
                                                            **{key: (running.get(lane["name"]) or {}).get(key) for key in ("state", "reason", "retry_at")},
                                                            "completed_pairs": sum(done[model] for model in lane["models"]), "expected_pairs": len(lane["models"]) * len(manifest["instances"])} for lane in manifest["lanes"]],
                       call_limits={"codex-cli": state.get("active_codex_invocation_limit", manifest["active_codex_invocation_limit"]), "claude-code-cli": state.get("active_claude_invocation_limit", manifest["active_claude_invocation_limit"]),
                                    "agy-cli": manifest["active_agy_invocation_limit"]},
                       solver_memory_limit_mib=MEMORY_POLICIES[manifest["solver_memory_policy"]]["limit_mib"], resource_guard=manifest["resource_guard"])
    return summary


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
