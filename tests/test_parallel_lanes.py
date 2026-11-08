"""The small-scale lanes variant: model lanes side by side in one controller, per-backend call gates, serialized solvers
with a fail-closed admission latch, the sampled per-solver memory budget, queue-time accounting and Claude Code slots.

Controlled fakes only: no Docker, no model call and no candidate solver (the sandbox launcher, the process sampler and
the cleanup are replaced where a test drives the executor).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from src.agent import auth
from src.agent.auth import CLAUDE_SLOT_VOLUMES, CODEX_SLOT_VOLUMES
from src.agent.prompts.builder import budget_rule
from src.agent.specs import DEFAULT_RUNTIME, RuntimeSpec
from src.experiments import agent_auth, campaign, docker_setup, sandbox
from src.experiments.call_gate import NativeCallGate, exclusive_lock
from src.experiments.completion import case_settings
from src.experiments.execution_policy import LANES_EXECUTION, SMALL_SCALE_LANES_POLICY, AttemptCase, ExecutionResult, SmallScaleLanesPolicy, SmallScaleNativePolicy
from src.experiments.plan import expected_specs, lane_partition, load_plan
from src.experiments.resources import CleanupError, ProcessSample
from src.utils.errors import InfrastructureError, ProtocolError
from src.utils.records import read, save


ROOT = Path(__file__).resolve().parents[1]
EXECUTOR = sandbox.NativeSandboxExecutor  # the real class (the campaign fixture replaces the module attribute)
SMALL = load_plan(ROOT / "configs/small_scale.json")
QUEUE = [{"model": "gpt-a", "backend": "codex-cli", "effort": "medium"}, {"model": "gpt-b", "backend": "codex-cli", "effort": "medium"}, {"model": "gpt-c", "backend": "codex-cli", "effort": "medium"},
         {"model": "claude-a", "backend": "claude-code-cli", "effort": "medium"}, {"model": "claude-b", "backend": "claude-code-cli", "effort": "medium"},
         {"model": "gemini", "backend": "agy-cli", "effort": "catalog-slug"}]
LANES = [{"name": "codex", "models": ["gpt-a", "gpt-b", "gpt-c"], "workers": 3}, {"name": "claude", "models": ["claude-a", "claude-b"], "workers": 2}, {"name": "gemini", "models": ["gemini"], "workers": 1}]
ENVIRONMENT = {"image_id": "sha256:fixture", "license_file": "fixture", "native_environment": {}, "codex_slot_volumes": list(CODEX_SLOT_VOLUMES), "claude_slot_volumes": list(CLAUDE_SLOT_VOLUMES)}


def wait_for(condition, seconds=5):
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.01)


# ------------------------------------------------------------------------------------------------ lane partition


def test_the_small_scale_config_selects_three_lanes_in_queue_order():
    lanes = lane_partition(SMALL["execution"]["lanes"], SMALL["model_queue"])
    assert [(lane["name"], lane["workers"]) for lane, _ in lanes] == [("codex", 3), ("claude", 2), ("gemini", 1)] and SMALL["execution"]["workers"] == 6
    assert [[item["model"] for item in members] for _, members in lanes] == [
        ["gpt-6-astra", "gpt-5.6-luna", "gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol"],
        ["claude-sonnet-5", "claude-opus-5", "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5"], ["gemini-3.8-flash-medium"]]
    assert SMALL["execution_policy"] == SMALL_SCALE_LANES_POLICY and {k: v for k, v in SMALL["execution"].items() if k not in ("lanes", "workers")} == LANES_EXECUTION
    assert (SMALL["execution"]["active_codex_invocation_limit"], SMALL["execution"]["active_claude_invocation_limit"], SMALL["execution"]["active_agy_invocation_limit"]) == (3, 2, 1)
    assert SMALL["execution"]["resource_guard"]["max_rss_mib"] == 5120 and SMALL["execution"]["resource_guard"]["max_cpu_percent"] == 500
    # A launch scope keeps queue order within each lane and may leave a lane empty.
    scope = [item for item in SMALL["model_queue"] if item["model"] in ("gpt-6-sol", "gpt-5.6-luna", "claude-opus-5")]
    assert [[item["model"] for item in members] for _, members in lane_partition(SMALL["execution"]["lanes"], SMALL["model_queue"], scope)] == [["gpt-5.6-luna", "gpt-6-sol"], ["claude-opus-5"], []]
    assert lane_partition(None, SMALL["model_queue"]) == []


@pytest.mark.parametrize("change,message", [
    (lambda lanes: lanes[1]["models"].append("gpt-a"), "more than one lane"),          # duplicate membership
    (lambda lanes: lanes[2]["models"].clear(), "at least one model"),
    (lambda lanes: lanes[0]["models"].remove("gpt-c"), "without a lane"),
    (lambda lanes: lanes[0]["models"].append("unknown"), "outside the model queue"),
    (lambda lanes: lanes[0]["models"].reverse(), "out of queue order"),
    (lambda lanes: lanes[1].update(name="codex"), "distinct"),
    (lambda lanes: lanes[1].update(name="Claude_lane"), "distinct"),
    (lambda lanes: lanes[1].update(workers=0), "positive worker count"),
    (lambda lanes: lanes[1].update(workers=True), "positive worker count"),
    (lambda lanes: lanes[1].update(backends=["claude-code-cli"]), "distinct"),
])
def test_lanes_must_partition_the_model_queue(change, message):
    lanes = deepcopy(LANES)
    change(lanes)
    with pytest.raises(ProtocolError, match=message):
        lane_partition(lanes, QUEUE)


def test_the_variant_settings_are_reviewed_and_belong_only_to_the_variant(tmp_path):
    for change in ({"workers": 5}, {"active_claude_invocation_limit": 1}, {"solver_memory_policy": None}):
        save(tmp_path / "bad.json", SMALL | {"execution": SMALL["execution"] | change})
        with pytest.raises(ProtocolError):
            load_plan(tmp_path / "bad.json")
    pasted = {key: value for key, value in SMALL["prompt_protocol"].items() if key not in ("context_delivery", "judge_context_delivery")}
    save(tmp_path / "pasted.json", {key: value for key, value in SMALL.items() if key != "solver_output_policy"} | {"prompt_protocol": pasted})
    with pytest.raises(ProtocolError, match="mounted-files prompt"):
        load_plan(tmp_path / "pasted.json")
    for name in ("paper.json", "harder_small_all.json"):
        plan = load_plan(ROOT / "configs" / name)
        assert "execution_policy" not in plan and "lanes" not in plan["execution"] and "solver_memory_policy" not in plan["execution"] and expected_specs(plan) == {}
        save(tmp_path / name, plan | {"execution": plan["execution"] | {"lanes": SMALL["execution"]["lanes"]}})
        with pytest.raises(ProtocolError, match="Execution settings"):
            load_plan(tmp_path / name)
    allowed = {"prompt_policies": SMALL["prompt_policies"], "reflection_policy": SMALL["reflection_policy"], "prompt_protocol": SMALL["prompt_protocol"], "model_queue": QUEUE}
    (tmp_path / "root").mkdir()
    with pytest.raises(ProtocolError, match="lanes variant only"):
        campaign.ensure_execution_allowed(tmp_path / "root", allowed | {"lanes": LANES})
    variant = allowed | {"experiment_plan": {"execution_policy": SMALL_SCALE_LANES_POLICY}} | LANES_EXECUTION | {"lanes": LANES, "workers": 6}
    campaign.ensure_execution_allowed(tmp_path / "root", variant)
    for change in ({"workers": 5}, {"active_codex_invocation_limit": 2}, {"lanes": LANES[:2]}):
        with pytest.raises(ProtocolError):
            campaign.ensure_execution_allowed(tmp_path / "root", variant | change)


# ------------------------------------------------------------------------------------------------ campaign scheduling


@pytest.fixture
def lanes_campaign(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    (root / "controller/src").mkdir(parents=True)
    (tmp_path / "locks").mkdir()
    monkeypatch.setattr(campaign, "REPOSITORY", root / "controller")
    manifest = {"model_queue": QUEUE, "instances": [{"suite": "bd", "problem": "ufl", "instance": f"case-{index}"} for index in range(3)], "protocol_id": "lanes-fixture",
                "experiment_plan": {"execution_policy": SMALL_SCALE_LANES_POLICY}, "prompt_policies": SMALL["prompt_policies"], "reflection_policy": SMALL["reflection_policy"],
                "prompt_protocol": SMALL["prompt_protocol"], "solver_output_policy": SMALL["solver_output_policy"], "execution_environment": dict(ENVIRONMENT), "retry_delays_seconds": [0]}
    manifest |= deepcopy(LANES_EXECUTION) | {"lanes": deepcopy(LANES), "workers": 6}
    fixture = SimpleNamespace(root=root, manifest=manifest, completed=set(), runs=[], behavior=None, executor=object(), executor_options=[], admissions=[], lock_entries=[])
    engine = SimpleNamespace(image_id=lambda name: name, bind_environment=lambda _: None, resource_samples=lambda: [], stop_campaign_workers=lambda _: None)
    monkeypatch.setattr(docker_setup, "runtime", lambda: engine)
    monkeypatch.setattr(sandbox, "NativeSandboxExecutor", lambda **kwargs: fixture.executor_options.append(kwargs) or fixture.executor)
    monkeypatch.setattr(campaign, "validate", lambda _: fixture.manifest)
    monkeypatch.setattr(campaign, "require_campaign_environment", lambda *_: None)
    monkeypatch.setattr(campaign, "require_backend_readiness", lambda *_: {})
    monkeypatch.setattr(campaign, "require_backend_authentication", lambda *args, **kwargs: fixture.admissions.append(kwargs) or {})
    monkeypatch.setattr(campaign, "state_root", lambda: tmp_path / "locks")

    @contextmanager
    def campaign_lock(*_):
        fixture.lock_entries.append(sorted(path.name for path in (root / "launches").glob("*.json")) if (root / "launches").exists() else [])
        yield

    monkeypatch.setattr(campaign, "runtime_lock", campaign_lock)
    monkeypatch.setattr(campaign.signal, "signal", lambda *_: None)
    monkeypatch.setenv("DECOMPBENCH_AGY_KEYRING_PASSPHRASE", "synthetic-test-only")
    monkeypatch.setattr(campaign, "pair_complete", lambda root, model, case, protocol, prompt_protocol=None: (model, case["instance"]) in fixture.completed)

    class LaneRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_pair(self, source, destination):
            fixture.runs.append((self.kwargs["model"], source.name, threading.current_thread().name))
            if fixture.behavior:
                fixture.behavior(self.kwargs, source.name)
            fixture.completed.add((self.kwargs["model"], source.name))

    monkeypatch.setattr(campaign, "PairedRunner", LaneRunner)
    return fixture


def test_three_lanes_run_side_by_side_with_unique_workers_and_finalize_once(lanes_campaign):
    fixture = lanes_campaign
    together, observed = threading.Barrier(6, timeout=5), []

    def behavior(kwargs, instance):
        if len(fixture.runs) <= 6:  # the first six admitted pairs: every worker of every lane at once
            together.wait()
            observed.append(read(fixture.root / "status.json"))
            together.wait()
        if kwargs["model"] == "gpt-c" and instance == "case-2":  # the last Codex pair: Gemini's lane is long done
            wait_for(lambda: read(fixture.root / "status.json")["lanes"]["gemini"]["complete"])
            status = read(fixture.root / "status.json")
            assert status["state"] == "running" and not (read(fixture.root / "pause_request.json") or {}).get("completed_execution_scope")

    fixture.behavior = behavior
    assert campaign.run(fixture.root) == 0
    assert fixture.completed == {(item["model"], f"case-{index}") for item in QUEUE for index in range(3)}
    workers = {model: {thread for m, _, thread in fixture.runs if m == model} for model in ("gpt-a", "claude-a", "gemini")}
    assert all(re.fullmatch(r"experiment-codex_[0-2]", name) for name in workers["gpt-a"]) and len(workers["gpt-a"]) == 3
    assert all(re.fullmatch(r"experiment-claude_[01]", name) for name in workers["claude-a"]) and workers["gemini"] == {"experiment-gemini_0"}
    active = observed[-1]["active_instances"]
    assert len(active) == 6 and {entry["lane"] for entry in active.values()} == {"codex", "claude", "gemini"}  # no worker overwrote another
    assert observed[-1]["active_models"] == {"codex": "gpt-a", "claude": "claude-a", "gemini": "gemini"} and observed[-1]["active_model"] == "gpt-a"
    assert (observed[-1]["active_codex_invocation_limit"], observed[-1]["active_claude_invocation_limit"], observed[-1]["active_agy_invocation_limit"]) == (3, 2, 1)
    # Within each lane the models ran in queue order, one model after the other.
    order = [model for model, _, _ in fixture.runs]
    for lane in LANES:
        positions = [[index for index, model in enumerate(order) if model == name] for name in lane["models"]]
        assert all(max(earlier) < min(later) for earlier, later in zip(positions, positions[1:]))
    state = read(fixture.root / "status.json")
    assert state["state"] == "complete" and state["scope_complete"] and state["campaign_complete"] and state["workers"] == 6 and not state["active_instances"]
    assert state["lanes"]["codex"]["models"] == ["gpt-a", "gpt-b", "gpt-c"] and all(lane["active_model"] is None for lane in state["lanes"].values())
    assert fixture.lock_entries == [[]] and len(list((fixture.root / "launches").glob("*.json"))) == 1  # the launch record follows the campaign lock
    assert fixture.admissions[0]["codex_slots"] == 3 and fixture.admissions[0]["claude_slots"] == 2
    options = fixture.executor_options[0]
    # The ownership record is host-wide: in the private state directory that also holds the campaign lock.
    assert options["memory_policy"] == sandbox.MEMORY_POLICY_2GIB and options["record_queue_seconds"] and options["ownership"] == fixture.root.parent / "locks/native-solver-ownership.json"


def test_a_lane_scope_may_leave_lanes_empty(lanes_campaign):
    fixture = lanes_campaign
    assert campaign.run(fixture.root, models=["claude-b", "gpt-b"]) == 0
    assert {model for model, _, _ in fixture.runs} == {"gpt-b", "claude-b"}
    state = read(fixture.root / "status.json")
    assert state["lanes"]["gemini"]["models"] == [] and state["lanes"]["gemini"]["complete"] and state["state"] == "paused" and state["scope_complete"]
    assert read(fixture.root / "pause_request.json")["reason"] == "launch_scope_complete"


def test_a_stopping_failure_in_one_lane_stops_admission_in_every_lane_and_drains_admitted_pairs(lanes_campaign):
    fixture = lanes_campaign
    fixture.manifest["instances"].append({"suite": "bd", "problem": "ufl", "instance": "case-3"})
    claude_started = threading.Event()

    def draining():
        return (read(fixture.root / "status.json") or {}).get("state") == "draining_for_recovery"

    def behavior(kwargs, instance):
        if kwargs["model"] == "claude-a":
            claude_started.set()
        if kwargs["model"] == "gpt-a" and instance == "case-0":
            assert claude_started.wait(5)
            raise ProtocolError("changed inputs")  # repair-required: never retried
        wait_for(draining)  # every admitted pair finishes after the failure is known

    fixture.behavior = behavior
    assert campaign.run(fixture.root, models=["gpt-a", "claude-a"]) == 64
    assert fixture.completed == {("gpt-a", "case-1"), ("gpt-a", "case-2"), ("claude-a", "case-0"), ("claude-a", "case-1")}  # nothing admitted after it
    state = read(fixture.root / "status.json")
    assert state["state"] == "repair_required" and "changed inputs" in state["error"] and not state["active_instances"]
    assert {lane["state"] for name, lane in state["lanes"].items() if name != "gemini"} == {"stopped"}
    assert not (fixture.root / "pause_request.json").exists() and not (fixture.root / "lane_failures").exists()  # never finalized, never retried


@pytest.mark.parametrize("first", ["infrastructure", "auth"])
def test_repair_required_outranks_retryable_failures_across_lanes_and_every_auth_failure_is_kept(lanes_campaign, first):
    from src.agent.docker_client import AuthenticationRequired
    fixture = lanes_campaign
    fixture.manifest["retry_delays_seconds"] = [30]  # the retryable lane is still backing off when the other lane stops
    failed = threading.Event()
    errors = {"infrastructure": InfrastructureError("provider unavailable"),
              "auth": AuthenticationRequired("claude credential invalid", {"backend": "claude-code-cli", "auth_volume": CLAUDE_SLOT_VOLUMES[1]})}
    second = AuthenticationRequired("codex credential invalid", {"backend": "codex-cli", "auth_volume": CODEX_SLOT_VOLUMES[2]}) if first == "auth" else errors["auth"]

    def behavior(kwargs, instance):
        if (kwargs["model"], instance) == ("gpt-a" if first == "infrastructure" else "claude-a", "case-0"):
            failed.set()
            raise errors[first]
        if (kwargs["model"], instance) == ("claude-a" if first == "infrastructure" else "gpt-a", "case-0"):
            assert failed.wait(5)
            if first == "infrastructure":
                wait_for(lambda: (read(fixture.root / "status.json") or {})["lanes"]["codex"]["state"] == "backing_off")
            else:
                wait_for(lambda: (read(fixture.root / "status.json") or {}).get("state") == "draining_for_recovery")
            raise second

    fixture.behavior = behavior
    started = time.monotonic()
    assert campaign.run(fixture.root, models=["gpt-a", "claude-a"]) == 64
    assert time.monotonic() - started < 15  # the stopping failure ended the 30 s backoff at once
    state = read(fixture.root / "status.json")
    assert state["state"] == "repair_required" and state["lanes"]["codex"]["state"] == "stopped"
    assert sum(run[:2] == ("gpt-a", "case-0") for run in fixture.runs) == 1  # never re-admitted
    sentinel = read(fixture.root / "auth_invalid.json")
    expected = {CLAUDE_SLOT_VOLUMES[1]} | ({CODEX_SLOT_VOLUMES[2]} if first == "auth" else set())
    assert set(sentinel["volumes"]) == expected


def test_a_pause_request_drains_every_lane_and_admits_nothing(lanes_campaign):
    fixture = lanes_campaign
    together = threading.Barrier(6, timeout=5)

    def behavior(kwargs, instance):
        if len(fixture.runs) <= 6:
            together.wait()
            if (kwargs["model"], instance) == ("claude-a", "case-0"):
                save(fixture.root / "pause_request.json", {"resume_requires_explicit_user_request": True, "reason": "user requested a drain"})
            together.wait()  # the request exists before any admitted pair finishes

    fixture.behavior = behavior
    assert campaign.run(fixture.root) == 0
    assert {(model, instance) for model, instance, _ in fixture.runs} == {("gpt-a", f"case-{index}") for index in range(3)} | {("claude-a", "case-0"), ("claude-a", "case-1"), ("gemini", "case-0")}
    state = read(fixture.root / "status.json")
    assert state["state"] == "paused" and not state["scope_complete"] and not state["active_instances"]
    pause = read(fixture.root / "pause_request.json")
    assert pause["reason"] == "user requested a drain" and pause["resume_requires_explicit_user_request"] and "completed_execution_scope" not in pause


class CallProbe:
    """Concurrent calls per backend through the campaign's call_lock; a slot held twice at once fails."""
    def __init__(self):
        self.lock, self.active, self.peak, self.slots = threading.Lock(), {}, {}, {}

    def call(self, kwargs, key, hold=0.15):
        client = kwargs[key]
        with kwargs["call_lock"](client):
            with self.lock:
                backend, slot = client.backend, client.auth_slot
                assert slot not in self.slots.setdefault(backend, set()) or backend == "agy-cli", (backend, slot)
                self.slots[backend].add(slot)
                self.active[backend] = self.active.get(backend, 0) + 1
                self.peak[backend] = max(self.peak.get(backend, 0), self.active[backend])
                volume = client.auth_volume()
            time.sleep(hold)
            with self.lock:
                self.active[backend] -= 1
                self.slots[backend].discard(slot)
        return volume


def test_call_limits_hold_per_backend_across_lanes_with_the_judge_on_the_codex_gate(lanes_campaign):
    fixture = lanes_campaign
    probe, volumes = CallProbe(), set()

    def behavior(kwargs, instance):
        volumes.add(probe.call(kwargs, "client"))
        volumes.add(probe.call(kwargs, "judge_client"))  # every method judge is a Codex call

    fixture.behavior = behavior
    assert campaign.run(fixture.root, models=["gpt-a", "gpt-b", "claude-a", "gemini"]) == 0
    assert probe.peak["codex-cli"] == 3 and probe.peak["claude-code-cli"] == 2 and probe.peak["agy-cli"] == 1
    assert set(CODEX_SLOT_VOLUMES) <= volumes and set(CLAUDE_SLOT_VOLUMES) <= volumes


def test_gemini_keeps_one_call_at_a_time_in_a_lane_with_two_workers(lanes_campaign):
    fixture = lanes_campaign
    fixture.manifest["lanes"] = [{"name": "codex", "models": ["gpt-a", "gpt-b", "gpt-c"], "workers": 3}, {"name": "claude-then-gemini", "models": ["claude-a", "claude-b", "gemini"], "workers": 2}]
    fixture.manifest["workers"] = 5
    probe = CallProbe()
    fixture.behavior = lambda kwargs, instance: probe.call(kwargs, "client")
    assert campaign.run(fixture.root, models=["claude-a", "gemini"]) == 0
    assert {thread for model, _, thread in fixture.runs if model == "gemini"} <= {"experiment-claude-then-gemini_0", "experiment-claude-then-gemini_1"}
    assert probe.peak["agy-cli"] == 1 and probe.peak["claude-code-cli"] == 2


def test_solvers_never_overlap_across_six_lane_workers(lanes_campaign, tmp_path, monkeypatch):
    fixture = lanes_campaign
    executor = fixture.executor = object.__new__(EXECUTOR)
    executor.environment, executor.record_queue_seconds = {"fixture": "environment"}, True
    monkeypatch.setattr(sandbox, "native_environment", lambda: executor.environment)
    guard, active, peak, together = threading.Lock(), [0], [0], threading.Barrier(6, timeout=5)

    def execute_bytes(solver, data, timeout, threads, *, evidence):
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.03)
        with guard:
            active[0] -= 1
        return {"exec_ok": True, "stdout": "", "stderr": "", "output": {"objective_value": 1}, "output_text": '{"objective_value": 1}', "integrity_violations": [], "execution_seconds": 0.03}

    executor.execute_bytes = execute_bytes
    queued = []

    def behavior(kwargs, instance):
        if len(fixture.runs) <= 6:
            together.wait()
        for attempt in range(3):
            directory = tmp_path / "runs" / kwargs["model"] / instance / "attempts" / f"attempt_{attempt:02d}"
            directory.mkdir(parents=True)
            (directory / "solver.py").write_text(f"{kwargs['model']} {instance} {attempt}")
            (directory.parent.parent / "data.json").write_text("{}")
            assert kwargs["solver_executor"](directory, 60, 1)[0]
            queued.append(read(directory / "native_solver_receipt.json")["result"]["solver_queue_seconds"])

    fixture.behavior = behavior
    assert campaign.run(fixture.root) == 0
    assert peak[0] == 1 and len(queued) == 6 * 3 * 3 and max(queued) > 0  # contention existed, execution never overlapped


# ------------------------------------------------------------------------------------------------ call gates


def gate_fixture(tmp_path, **options):
    state, phase = tmp_path / "state", tmp_path / "phase"
    state.mkdir(exist_ok=True)
    phase.mkdir(exist_ok=True)
    return NativeCallGate(state, phase, **options)


@pytest.mark.parametrize("backend,slots", [("codex-cli", 3), ("claude-code-cli", 2)])
def test_a_gate_admits_its_slots_on_distinct_volumes_then_waits(tmp_path, backend, slots):
    gate = gate_fixture(tmp_path, backend=backend, slots=slots)
    entered, release, seen = threading.Barrier(slots, timeout=5), threading.Event(), []
    extra = threading.Event()

    def call():
        with gate.enter() as slot:
            seen.append(slot)
            entered.wait()
            assert release.wait(5)

    def late():
        with gate.enter() as slot:
            seen.append(slot)
            extra.set()

    with ThreadPoolExecutor(max_workers=slots + 1) as pool:
        futures = [pool.submit(call) for _ in range(slots)]
        wait_for(lambda: len(seen) == slots)
        waiting = pool.submit(late)
        assert not extra.wait(0.3)
        release.set()
        for future in futures + [waiting]:
            future.result()
    assert sorted(seen[:slots]) == list(range(slots)) and gate.active == 0
    assert [auth.auth_volume(backend, slot) for slot in range(slots)] == list(auth.SLOT_VOLUMES[backend][:slots])


def test_gate_slot_counts_are_bounded_by_the_auth_volumes(tmp_path):
    for backend, slots in (("codex-cli", 4), ("claude-code-cli", 3), ("codex-cli", 0), ("agy-cli", 1)):
        with pytest.raises(ValueError):
            gate_fixture(tmp_path, backend=backend, slots=slots)
    assert gate_fixture(tmp_path).limit == 2  # the paper's two-slot Codex gate is the default


def test_three_slot_fallback_drains_then_serializes_on_slot_zero_per_backend(tmp_path):
    codex, claude = gate_fixture(tmp_path, slots=3), gate_fixture(tmp_path, backend="claude-code-cli", slots=2)
    release, inside = threading.Event(), threading.Barrier(3, timeout=5)

    def call():
        with codex.enter():
            inside.wait()
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(call) for _ in range(3)]
        wait_for(lambda: codex.active == 3)
        codex.fallback("refresh token", role="algorithm_judge", invocation_id="fixture")
        assert codex.limit == 1 and claude.limit == 2  # a backend's fallback is its own
        release.set()
        for future in futures:
            future.result()
    with codex.enter() as slot:
        assert slot == 0
    record = read(tmp_path / "phase" / "auth_fallback.json")
    assert record["effective_codex_invocation_limit"] == 1 and list(record) == ["recorded_at", "effective_codex_invocation_limit", "reason", "role", "invocation_id", "policy"]
    claude.fallback("not logged in", role="generation", invocation_id="fixture")
    assert read(tmp_path / "phase" / "auth_fallback.claude-code-cli.json")["effective_claude_invocation_limit"] == 1 and claude.limit == 1


def test_every_gate_wait_observes_a_stop_request(tmp_path):
    gate = gate_fixture(tmp_path, slots=3)
    release, held = threading.Event(), threading.Event()

    def hold():
        with gate.enter():
            held.set()
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=4) as pool:
        holders = [pool.submit(hold) for _ in range(3)]
        wait_for(lambda: gate.active == 3)
        waiting = pool.submit(lambda: gate.enter().__enter__())  # waits at the in-process admission
        time.sleep(0.2)
        auth.CANCEL.set()
        with pytest.raises(InfrastructureError, match="stop requested"):
            waiting.result(timeout=5)
        release.set()
        for future in holders:
            future.result()
    auth.CANCEL.clear()
    # The legacy lock held exclusively by a serialized runner, and the AGY lock: both waits are cancellable.
    import fcntl
    blocker = os.open(gate.state_directory / "codex-native-auth.lock", os.O_CREAT | os.O_RDWR)
    agy = os.open(gate.state_directory / "native-agy-auth.lock", os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(blocker, fcntl.LOCK_EX)
        fcntl.flock(agy, fcntl.LOCK_EX)
        with ThreadPoolExecutor(max_workers=2) as pool:
            legacy = pool.submit(lambda: gate.enter().__enter__())
            agy_wait = pool.submit(lambda: exclusive_lock(gate.state_directory, "native-agy-auth").__enter__())
            time.sleep(0.2)
            assert not legacy.done() and not agy_wait.done()
            auth.CANCEL.set()
            for future in (legacy, agy_wait):
                with pytest.raises(InfrastructureError, match="stop requested"):
                    future.result(timeout=5)
    finally:
        os.close(blocker)
        os.close(agy)
    assert gate.active == 0


# ------------------------------------------------------------------------------------------------ solver admission


def executor_fixture(tmp_path, monkeypatch, **attributes):
    task = object.__new__(EXECUTOR)
    task.environment = {"fixture": "environment"}
    for name, value in attributes.items():
        setattr(task, name, value)
    monkeypatch.setattr(sandbox, "native_environment", lambda: task.environment)
    return task


def attempt(tmp_path, name):
    work = tmp_path / name / "attempts/attempt_01"
    work.mkdir(parents=True)
    (work / "solver.py").write_text(name)
    (work.parent.parent / "data.json").write_text("{}")
    return work


def completed_record(**fields):
    return {"exec_ok": True, "stdout": "", "stderr": "", "output": {"objective_value": 1}, "output_text": '{"objective_value": 1}', "integrity_violations": [], "execution_seconds": 1.0} | fields


def test_an_unverified_cleanup_closes_admission_before_a_waiting_worker_can_launch(tmp_path, monkeypatch):
    task = executor_fixture(tmp_path, monkeypatch)
    calls, second_waiting = [], threading.Event()

    def execute_bytes(solver, data, timeout, threads, *, evidence):
        calls.append(solver)
        assert second_waiting.wait(5)
        time.sleep(0.2)  # the second worker is blocked on the solver lock
        raise CleanupError("Owned processes remain alive: [4242]")

    task.execute_bytes = execute_bytes
    first, second = attempt(tmp_path, "first"), attempt(tmp_path, "second")
    with ThreadPoolExecutor(max_workers=2) as pool:
        failing = pool.submit(task, first, 60)
        wait_for(lambda: calls)
        def waiting():
            second_waiting.set()
            return task(second, 60)
        blocked = pool.submit(waiting)
        with pytest.raises(CleanupError, match="remain alive"):
            failing.result()
        with pytest.raises(CleanupError, match="admission is closed"):
            blocked.result()
    assert calls == [b"first"]  # the waiting worker never launched
    with pytest.raises(CleanupError, match="admission is closed"):
        task(attempt(tmp_path, "third"), 60)


def test_a_launcher_failure_with_verified_cleanup_keeps_admission_open(tmp_path, monkeypatch):
    task = executor_fixture(tmp_path, monkeypatch)
    outcomes = iter([sandbox.SolverLaunchError("bootstrap", {"pid": 7, "cleanup_verified": True}), completed_record()])

    def execute_bytes(*args, evidence):
        value = next(outcomes)
        if isinstance(value, Exception):
            raise value
        return value

    task.execute_bytes = execute_bytes
    with pytest.raises(sandbox.SolverLaunchError):
        task(attempt(tmp_path, "first"), 60)
    assert task(attempt(tmp_path, "second"), 60)[0]
    assert not sandbox.cleanup_established(sandbox.SolverLaunchError("x", {"pid": 7, "cleanup_verified": False})) and sandbox.cleanup_established(sandbox.SolverLaunchError("x", {}))


def test_a_later_controller_of_any_root_refuses_to_overlap_an_earlier_launch_that_survived(tmp_path, monkeypatch):
    ownership = tmp_path / "state/native-solver-ownership.json"  # host-wide: shared by every campaign root
    ownership.parent.mkdir()
    first_root, other_root = executor_fixture(tmp_path, monkeypatch, ownership=ownership), executor_fixture(tmp_path, monkeypatch, ownership=ownership)
    for task in (first_root, other_root):
        task.execute_bytes = lambda *args, evidence: completed_record()
    sample = lambda pid, group: ProcessSample(pid, 1, group, 0.0, 1024, "Sat Nov  7 10:00:00 2026", "python3", "R")
    processes, commands = {9: sample(9, 4242)}, {9: "python3 helper.py"}  # a reparented child, without the bootstrap token
    monkeypatch.setattr(sandbox, "read_processes", lambda: processes)
    monkeypatch.setattr(sandbox, "read_commands", lambda: commands)
    save(ownership, {"state": "running", "token": "DECOMPBENCH_READY_abc", "pid": 4242, "handshake": True, "evidence": "/root-a/attempt/native_execution_x.json"})
    with pytest.raises(CleanupError, match=r"still running: processes \[9\].*awk '\$2 == 4242'"):
        other_root.verify_ownership()  # a clean root B still sees the survivor of root A
    with pytest.raises(CleanupError, match="still running"):
        first_root(attempt(tmp_path, "first"), 60)
    processes.clear()
    other_root.verify_ownership()
    assert read(ownership)["state"] == "released" and "released_after_verification_at" in read(ownership)
    assert other_root(attempt(tmp_path, "second"), 60)[0]


@pytest.mark.parametrize("record,resolvable", [
    ({"state": "launching", "token": "DECOMPBENCH_READY_abc", "pid": None, "handshake": True}, True),   # the handshake: its candidate never ran
    ({"state": "launching", "token": "DECOMPBENCH_READY_abc", "pid": None}, False),                    # no handshake guarantee
    ({"state": "running", "token": "DECOMPBENCH_READY_abc", "pid": None, "handshake": True}, False),   # running without its group
    ({"state": "starting", "token": "DECOMPBENCH_READY_abc", "pid": 4242, "handshake": True}, False),  # unknown state
    ({}, False),
])
def test_an_unresolved_ownership_record_fails_closed_with_the_manual_check(tmp_path, monkeypatch, record, resolvable):
    ownership = tmp_path / "native-solver-ownership.json"
    task = executor_fixture(tmp_path, monkeypatch, ownership=ownership)
    monkeypatch.setattr(sandbox, "read_processes", lambda: {})
    commands = {10: "/Python -B -s -P -u -c import runpy,sys; print('DECOMPBENCH_READY_abc',flush=True); sys.exit(70) if sys.stdin.readline()!='GO\\n' else None"}
    monkeypatch.setattr(sandbox, "read_commands", lambda: commands)
    ownership.write_text(json.dumps(record))
    if not resolvable:
        with pytest.raises(CleanupError, match=f"cannot show that its launch stopped.*delete {re.escape(str(ownership))}"):
            task.verify_ownership()
        return
    with pytest.raises(CleanupError, match=r"processes \[10\]"):
        task.verify_ownership()  # the bootstrap still waits on its pipe (it exits at the end of file)
    commands.clear()
    task.verify_ownership()
    assert read(ownership)["state"] == "released"


def test_the_handshake_bootstrap_runs_the_candidate_only_after_the_go_line(tmp_path):
    import sys
    marker = tmp_path / "ran"
    solver = tmp_path / "solver.py"
    solver.write_text(f"open({str(marker)!r}, 'w').write('ran')\n")
    token = "DECOMPBENCH_READY_" + "0" * 32
    program = sandbox.bootstrap_source(token, solver, handshake=True)
    eof = subprocess.run([sys.executable, "-I", "-c", program], input=b"", capture_output=True, timeout=30)  # the controller died: end of file
    assert eof.returncode == sandbox.HANDSHAKE_EXIT and not marker.exists() and eof.stdout.decode().startswith(token)
    go = subprocess.run([sys.executable, "-I", "-c", program], input=sandbox.HANDSHAKE_LINE.encode(), capture_output=True, timeout=30)
    assert go.returncode == 0 and marker.read_text() == "ran"
    assert sandbox.bootstrap_source(token, solver) == "import runpy,sys; print(" + repr(token) + ",flush=True); sys.argv=[" + repr(str(solver)) + "]; runpy.run_path(sys.argv[0],run_name='__main__')"


# ------------------------------------------------------------------------------------------------ memory budget


def group_sample(pid, parent, group, rss_mib, state="R"):
    return ProcessSample(pid, parent, group, 0.0, int(rss_mib * 1024), "Sat Nov  7 10:00:00 2026", "python3", state)


@pytest.fixture
def killed(monkeypatch):
    signals = []
    monkeypatch.setattr(sandbox.os, "killpg", lambda group, signum: signals.append((group, signum)))
    return signals


def test_the_watcher_sums_the_group_including_children_after_the_parent_exited(killed):
    samples = iter([{100: group_sample(100, 1, 100, 600), 101: group_sample(101, 100, 100, 900), 200: group_sample(200, 1, 200, 4000), 102: group_sample(102, 100, 100, 3000, "Z")},
                    {101: group_sample(101, 1, 100, 1900), 103: group_sample(103, 1, 100, 300)}])  # the parent is gone; its children stay in the group
    watcher = sandbox.GroupMemoryWatcher(100, limit_mib=2048, interval=0.01, sampler=lambda: next(samples))
    assert watcher.done.wait(5)
    watcher.close()
    assert watcher.exceeded and watcher.error is None and watcher.samples == 2 and watcher.peak_mib == 2200 and killed == [(100, 9)]


def test_a_failed_sample_stops_the_group_and_is_not_a_memory_verdict(killed):
    def failing():
        raise subprocess.TimeoutExpired(["/bin/ps"], 5)

    watcher = sandbox.GroupMemoryWatcher(100, limit_mib=2048, interval=0.01, sampler=failing)
    assert watcher.done.wait(5)
    watcher.close()
    assert not watcher.exceeded and "TimeoutExpired" in watcher.error and killed == [(100, 9)]


class FakeStdin:
    """The handshake pipe: the candidate starts only on the GO line (and records the ownership record seen then)."""
    def __init__(self, process, ownership):
        self.process, self.ownership, self.closed, self.lines = process, ownership, False, []

    def write(self, data):
        self.lines.append(data)
        self.process.owner_at_go = read(self.ownership) if self.ownership else None
        if data == sandbox.HANDSHAKE_LINE.encode():
            self.process.start()

    def close(self):
        self.closed = True


class FakeProcess:
    """The sandbox launcher's process: prints the bootstrap token (unless the scenario says not), waits for the
    handshake line when it has a stdin pipe, may write output.json, runs until killed or done."""
    scenario, ownership = {}, None

    def __init__(self, command, cwd, stdout, stderr, stdin=None, **kwargs):
        self.pid, self.returncode, self.killed, self.cwd, self.owner_at_go = 4242, None, threading.Event(), Path(cwd), None
        self.leader_code, self.group_killed, self.status = None, threading.Event(), None
        FakeProcess.current = self
        token = re.search(r"DECOMPBENCH_READY_[0-9a-f]+", command[-1]).group(0)
        self.handshake = "sys.stdin.readline()" in command[-1]
        if not self.scenario.get("no_bootstrap"):
            stdout.write((token + "\n").encode())
            stdout.flush()
        self.stdin = FakeStdin(self, FakeProcess.ownership) if stdin == subprocess.PIPE else None
        if self.stdin is None:
            self.start()

    def start(self):
        if self.scenario.get("output") is not None:
            (self.cwd / "output.json").write_text(json.dumps(self.scenario["output"]))
        if self.scenario.get("exit_after") is not None:
            threading.Timer(self.scenario["exit_after"], self.finish, args=(0,)).start()
        if self.scenario.get("leader_exit_after") is not None:  # the leader exits with 0, a child stays in the group
            threading.Timer(self.scenario["leader_exit_after"], self.leader_exit).start()
        if self.scenario.get("child_writes_after") is not None:
            threading.Timer(self.scenario["child_writes_after"], self.child_write).start()

    def leader_exit(self):
        self.leader_code = 0  # an unreaped zombie: returncode stays None until the cleanup reaps it
        self.killed.set()

    def child_write(self):
        if not self.group_killed.is_set():
            (self.cwd / "output.json").write_text(json.dumps({"objective_value": 1}))

    def finish(self, code):
        """The group is stopped (or the leader exits); like a real child it stays unreaped (returncode None) until reap()."""
        if self.status is None:
            self.status = self.leader_code if self.leader_code is not None else code
        self.group_killed.set()
        self.killed.set()

    def reap(self):
        self.returncode = self.status if self.status is not None else -9

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if not self.killed.wait(timeout):
            raise subprocess.TimeoutExpired("sandbox", timeout)
        return self.returncode


class FakeExit:
    """resources.ChildExit for a FakeProcess: its exit, observed without reaping."""
    def __init__(self, process):
        self.process = process

    def wait(self, timeout):
        return self.process.killed.wait(timeout)

    def close(self):
        pass


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(sandbox, "child_exit", FakeExit)
    monkeypatch.setattr(sandbox, "stop_owned_tree", lambda process, grace=0.1: (process.finish(-9), process.reap()))
    monkeypatch.setattr(sandbox.os, "killpg", lambda group, signum: FakeProcess.current.finish(-9))
    task = object.__new__(EXECUTOR)
    task.environment = {"sandbox_binary": "/usr/bin/sandbox-exec", "python_runtime_binary": "/python", "site_packages": "/site", "python_runtime_root": "/runtime"}
    task.license_file = tmp_path / "gurobi.lic"

    def configure(memory=True, sampler=None, ownership=None, **scenario):
        FakeProcess.scenario, FakeProcess.ownership = scenario, ownership
        task.ownership = ownership
        task.memory_policy, task.memory_limit_mib = (sandbox.MEMORY_POLICY_2GIB, 2048) if memory else (None, None)
        if sampler is not None:
            task.memory_sampler = sampler
        return task

    return configure


def low(_=None):
    return {4242: group_sample(4242, 1, 4242, 300)}


def high():
    return {4242: group_sample(4242, 1, 4242, 1500), 4243: group_sample(4243, 4242, 4242, 900)}


def test_a_valid_output_above_the_memory_budget_is_an_execution_failure(launcher, tmp_path):
    task = launcher(sampler=high, output={"objective_value": 1})
    record = task.execute_bytes(b"solver", b"{}", 60, 1, evidence=tmp_path / "evidence.json")
    assert record["exec_ok"] is False and record["output"] is None and record["output_text"] == '{"objective_value": 1}'
    assert record["memory_limit_exceeded"] and record["termination"] == "memory_limit_exceeded" and record["memory_peak_mib"] == 2400.0
    assert record["stderr"].endswith(sandbox.memory_message(2048)) and "memory limit exceeded: the solver and its child processes used more than 2048 MiB and were stopped" in record["stderr"]
    assert (record["memory_policy"], record["memory_limit_mib"], record["memory_sampling_interval_seconds"]) == (sandbox.MEMORY_POLICY_2GIB, 2048, 0.5) and record["cleanup_verified"]


def test_the_memory_limit_takes_precedence_over_a_timeout_in_the_same_moment(launcher, tmp_path):
    def slow_high():
        time.sleep(0.6)  # the sample in progress when the deadline passes finds the group above the budget
        return high()

    task = launcher(sampler=slow_high)
    record = task.execute_bytes(b"solver", b"{}", 0.3, 1, evidence=tmp_path / "evidence.json")
    assert record["timed_out"] and record["memory_limit_exceeded"] and record["termination"] == "memory_limit_exceeded"
    assert sandbox.MEMORY_MARKER in record["stderr"] and "timed out" not in record["stderr"] and not record["exec_ok"]


def test_a_run_within_the_budget_keeps_its_result_and_records_the_sampled_peak(launcher, tmp_path):
    task = launcher(sampler=low, output={"objective_value": 1}, exit_after=0.05)
    record = task.execute_bytes(b"solver", b"{}", 60, 1, evidence=tmp_path / "evidence.json")
    assert record["exec_ok"] and record["output"] == {"objective_value": 1} and not record["memory_limit_exceeded"] and record["termination"] == "exited"
    assert record["memory_samples"] >= 1 and record["memory_peak_mib"] == 300.0 and sandbox.MEMORY_MARKER not in record["stderr"]


def test_a_sampling_failure_is_infrastructure_with_verified_cleanup(launcher, tmp_path):
    def failing():
        raise RuntimeError("Resource query failed: /bin/ps")

    task = launcher(sampler=failing, output={"objective_value": 1})
    with pytest.raises(sandbox.SolverLaunchError, match="memory sampling failed") as raised:
        task.execute_bytes(b"solver", b"{}", 60, 1, evidence=tmp_path / "evidence.json")
    assert raised.value.evidence["phase"] == "memory_sampling_failed" and raised.value.evidence["cleanup_verified"] and sandbox.cleanup_established(raised.value)


def test_the_candidate_is_let_run_only_after_its_group_is_durably_owned(launcher, tmp_path):
    ownership = tmp_path / "native-solver-ownership.json"
    task = launcher(sampler=low, ownership=ownership, output={"objective_value": 1}, exit_after=0.05)
    record = task.execute_bytes(b"solver", b"{}", 60, 1, evidence=tmp_path / "evidence.json")
    process = FakeProcess.current
    assert process.handshake and process.stdin.lines == [sandbox.HANDSHAKE_LINE.encode()] and process.stdin.closed
    assert process.owner_at_go["state"] == "running" and process.owner_at_go["pid"] == 4242 and process.owner_at_go["handshake"] is True
    assert record["exec_ok"] and read(ownership)["state"] == "released"
    assert "sys.stdin.readline()" in record["command"][-1]


@pytest.mark.parametrize("memory_exceeded", [False, True])
def test_a_missing_bootstrap_is_infrastructure_even_after_a_timeout_or_memory_stop(launcher, tmp_path, memory_exceeded):
    def sampler():
        if memory_exceeded:
            time.sleep(0.6)  # the deadline passes during the sample that finds the group above the budget
            return high()
        return low()

    task = launcher(sampler=sampler, no_bootstrap=True)
    with pytest.raises(sandbox.SolverLaunchError, match="before the candidate started") as raised:
        task.execute_bytes(b"solver", b"{}", 0.3, 1, evidence=tmp_path / "evidence.json")
    evidence = raised.value.evidence
    assert evidence["phase"] == "bootstrap_failed" and evidence["termination"] == "bootstrap_failed" and evidence["cleanup_verified"] and sandbox.cleanup_established(raised.value)
    assert evidence["memory_limit_exceeded"] is memory_exceeded  # recorded, but a candidate that may never have started is not charged


def test_a_bootstrap_that_stopped_before_the_handshake_is_not_charged(launcher, tmp_path, monkeypatch):
    task = launcher(sampler=low, ownership=tmp_path / "native-solver-ownership.json")
    monkeypatch.setattr(EXECUTOR, "let_run", staticmethod(lambda process: False))
    with pytest.raises(sandbox.SolverLaunchError, match="before the launch handshake") as raised:
        task.execute_bytes(b"solver", b"{}", 0.3, 1, evidence=tmp_path / "evidence.json")
    assert raised.value.evidence["phase"] == "handshake_failed" and raised.value.evidence["cleanup_verified"]


def test_the_paper_executor_record_keeps_its_layout(launcher, tmp_path):
    task = launcher(memory=False, output={"objective_value": 1}, exit_after=0.05)
    record = task.execute_bytes(b"solver", b"{}", 60, 1, evidence=tmp_path / "evidence.json")
    assert set(record) == {"policy", "solver_sha256", "input_sha256", "timeout", "threads", "exec_ok", "output", "output_text", "stdout", "stderr", "integrity_violations", "cleanup_verified",
                           "bootstrap_ready", "workspace", "seatbelt_profile", "seatbelt_profile_sha256", "network_enabled", "process_policy", "environment_keys", "command", "pid", "returncode",
                           "phase", "execution_seconds", "timed_out"}
    assert record["exec_ok"] and not (tmp_path / "native_solver_ownership.json").exists()


def test_the_memory_policy_is_part_of_the_request_and_never_triggers_the_gurobi_diagnostic(tmp_path, monkeypatch):
    plain = executor_fixture(tmp_path, monkeypatch)
    plain.execute_bytes = lambda *args, evidence: completed_record()
    work = attempt(tmp_path, "case")
    plain(work, 60)
    budgeted = executor_fixture(tmp_path, monkeypatch, memory_policy=sandbox.MEMORY_POLICY_2GIB, memory_limit_mib=2048)
    with pytest.raises(Exception, match="request changed"):
        budgeted(work, 60)  # a result committed without the budget is never reused under it
    calls = []

    def stopped(solver, *args, evidence):
        calls.append(solver)
        return completed_record(exec_ok=False, output=None, output_text=None, stderr="gurobipy license banner\n" + sandbox.memory_message(2048), memory_limit_exceeded=True)

    budgeted.execute_bytes = stopped
    other = attempt(tmp_path, "other")
    assert budgeted(other, 60)[0] is False and len(calls) == 1
    assert read(other / "native_solver_receipt.json")["request"]["memory_policy"] == sandbox.MEMORY_POLICY_2GIB


# ------------------------------------------------------------------------------------------------ prompts, timing, identity


def test_the_memory_budget_is_stated_only_by_the_variant_runtime():
    assert budget_rule(60, 1) == "The harness limits the entire solver.py execution to 60 seconds and numerical solvers to 1 thread(s). Design your algorithm and termination handling within this budget."
    assert RuntimeSpec(60, 1).harness_rules() == DEFAULT_RUNTIME.harness_rules() and RuntimeSpec(60, 1, None, None).contract == RuntimeSpec(60, 1, None, 2048).contract
    stated = RuntimeSpec(60, 1, "output-json-readback-32MiB-v1", 2048).harness_rules()
    assert "The solver.py process and its child processes share a resident-memory budget of 2 GiB; a run that exceeds it is stopped." in stated
    assert stated.endswith("Design your algorithm and termination handling within this budget.")
    with pytest.raises(ValueError):
        RuntimeSpec(60, 1, None, 0)


def lanes_executor(tmp_path):
    executor = object.__new__(EXECUTOR)
    executor.memory_policy, executor.memory_limit_mib, executor.record_queue_seconds = sandbox.MEMORY_POLICY_2GIB, 2048, True
    return executor


def test_the_variant_runner_records_its_policy_and_its_identity_passes_the_campaign_predicate(tmp_path, monkeypatch):
    from src.experiments.paired_runner import PairedRunner
    from src.agent.prompts.method import DECOMPOSITION_POLICY_V3
    executor = lanes_executor(tmp_path)
    protocol = SMALL["prompt_protocol"]
    runner = PairedRunner(client=None, judge_client=None, solver_executor=executor, model="gpt-a", suite="bd", problem=["ufl"], protocol_id="lanes-fixture",
                          decomposition_contract=DECOMPOSITION_POLICY_V3, context_delivery=protocol["context_delivery"], judge_context_delivery=protocol["judge_context_delivery"],
                          execution_policy=SmallScaleLanesPolicy(executor=executor, suite="bd", cooldown_seconds=10))
    manifest = {"experiment_plan": SMALL, "prompt_protocol": protocol}
    assert runner.spec_record() == expected_specs(manifest) == {"execution_policy": SMALL_SCALE_LANES_POLICY} and runner.runtime_spec.memory_mib == 2048
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.json").write_text("{}")
    (source / "prompt.txt").write_text("task")
    identity = read(runner.prepare(source, tmp_path / "runs", "plain") / "identity.json")
    settings = campaign.completion_settings(manifest)
    assert campaign.identity_settings(identity) == campaign.identity_settings(settings) == case_settings(protocol, expected_specs(manifest))
    assert campaign.identity_settings({key: value for key, value in identity.items() if key != "specs"}) != campaign.identity_settings(settings)  # a pair closed without the variant
    paper = {"prompt_protocol": protocol}
    assert campaign.completion_settings(paper) is protocol and campaign.identity_settings(identity) != campaign.identity_settings(campaign.completion_settings(paper))
    with pytest.raises(ValueError, match="memory budget"):
        PairedRunner(client=None, judge_client=None, solver_executor=executor, context_delivery=protocol["context_delivery"])  # the paper policy has none
    with pytest.raises(ValueError, match="mounted-files prompt"):
        PairedRunner(client=None, judge_client=None, solver_executor=executor, execution_policy=SmallScaleLanesPolicy(executor=executor, suite="bd"))  # pasted prompts cannot state it


def lanes_attempt(tmp_path, receipt):
    directory = tmp_path / "case/attempts/attempt_01"
    directory.mkdir(parents=True)
    (directory / "solver.py").write_text("solver")
    (directory.parent.parent / "data.json").write_text("{}")
    from src.utils.records import digest
    case = AttemptCase(directory, directory.parent.parent, directory.parent.parent / "data.json", digest(directory.parent.parent / "data.json"), "ufl")

    class Executor:
        """Commits its receipt as the native executor does; with a receipt already committed it only re-reads it."""
        memory_limit_mib = 2048

        def __call__(self, attempt_dir, timeout, threads):
            if not (attempt_dir / "native_solver_receipt.json").exists():
                save(attempt_dir / "native_solver_receipt.json", {"request": {}, "result": receipt})
            return receipt["exec_ok"], "", receipt["stderr"], None

    return case, Executor()


def test_the_variant_cooldown_follows_the_scored_execution_and_the_queue_is_kept_out(tmp_path, monkeypatch):
    from src.experiments import execution_policy
    slept = []
    monkeypatch.setattr(execution_policy.time, "sleep", slept.append)
    assert SmallScaleNativePolicy.cooldown_due(25, {"execution_seconds": 1}, True) and not SmallScaleLanesPolicy.cooldown_due(25, {"execution_seconds": 1}, True)
    assert not SmallScaleLanesPolicy.cooldown_due(0, {"execution_seconds": 6}, False) and SmallScaleLanesPolicy.cooldown_due(0, {"execution_seconds": 6}, True)
    receipt = {"exec_ok": False, "stderr": "Traceback\n" + sandbox.memory_message(2048), "execution_seconds": 6.0, "solver_queue_seconds": 20.0, "memory_limit_mib": 2048, "memory_peak_mib": 2300.0,
               "memory_limit_exceeded": True}
    case, executor = lanes_attempt(tmp_path, receipt)
    policy = SmallScaleLanesPolicy(executor=executor, suite="bd", cooldown_seconds=10)
    runtime = RuntimeSpec(60, 1, None, 2048)
    result = policy.execute(case, delivery=None, runtime=runtime, stage=lambda phase: None)
    assert slept == [10] and result.metrics["cooldown_seconds"] == 10 and result.queue_seconds == 20.0 and result.metrics["solver_queue_seconds"] == 20.0
    assert result.termination == "memory_limit_exceeded" and result.metrics["solver_memory_limit_exceeded"] and result.metrics["solver_memory_peak_mib"] == 2300.0 and not result.ok
    resumed = policy.execute(case, delivery=None, runtime=runtime, stage=lambda phase: None)
    assert resumed.queue_seconds == 0.0 and resumed.metrics["solver_queue_seconds"] == 20.0 and slept == [10]  # no queue in this run, no second cooldown


def test_a_resume_between_the_native_receipt_and_execution_json_adds_no_queue_time(tmp_path, monkeypatch):
    from src.experiments import execution_policy
    slept = []
    monkeypatch.setattr(execution_policy.time, "sleep", slept.append)
    receipt = {"exec_ok": True, "stderr": "", "execution_seconds": 6.0, "solver_queue_seconds": 100.0, "memory_limit_mib": 2048, "memory_peak_mib": 300.0, "memory_limit_exceeded": False}
    case, executor = lanes_attempt(tmp_path, receipt)
    save(case.directory / "native_solver_receipt.json", {"request": {}, "result": receipt})  # committed by a controller that stopped before execution.json
    policy = SmallScaleLanesPolicy(executor=executor, suite="bd", cooldown_seconds=10)
    result = policy.execute(case, delivery=None, runtime=RuntimeSpec(60, 1, None, 2048), stage=lambda phase: None)
    assert result.queue_seconds == 0.0 and result.metrics["solver_queue_seconds"] == 100.0  # historical evidence only
    assert slept == [] and result.metrics["cooldown_seconds"] == 0 and read(case.directory / "execution.json")["execution_seconds"] == 6.0


def test_the_receipt_verdict_decides_the_memory_outcome_not_a_printed_marker(tmp_path, monkeypatch):
    from src.experiments import execution_policy
    monkeypatch.setattr(execution_policy.time, "sleep", lambda _: None)
    receipt = {"exec_ok": True, "stderr": "print: " + sandbox.memory_message(2048), "execution_seconds": 1.0, "solver_queue_seconds": 0.0, "memory_limit_mib": 2048, "memory_peak_mib": 300.0,
               "memory_limit_exceeded": False}
    case, executor = lanes_attempt(tmp_path, receipt)
    result = SmallScaleLanesPolicy(executor=executor, suite="bd", cooldown_seconds=10).execute(case, delivery=None, runtime=RuntimeSpec(60, 1, None, 2048), stage=lambda phase: None)
    assert result.ok and result.termination == "completed" and result.metrics["solver_memory_limit_exceeded"] is False and sandbox.MEMORY_MARKER in result.stderr


def test_an_attempt_total_leaves_out_the_solver_queue(tmp_path, monkeypatch):
    from dataclasses import replace
    from src.experiments import paired_runner
    from test_native_paired import Client, source_case

    class Queued(SmallScaleNativePolicy):
        def result(self, *args):
            return replace(super().result(*args), queue_seconds=7.5)

    def executor(attempt_dir, timeout, threads):
        output = {"status": "feasible", "method": "benders_decomposition", "y": [0, 1], "x": [[0], [1]], "objective_value": 4}
        (attempt_dir / "output.json").write_text(json.dumps(output))
        return True, "", "", output

    clock = iter([100.0, 130.0] + [130.0] * 100)
    monkeypatch.setattr("src.experiments.paired_runner.runtime_lock", lambda name: nullcontext())
    monkeypatch.setattr(paired_runner, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    client = Client()
    runner = paired_runner.PairedRunner(client=client, judge_client=client, model="fixture", suite="bd", problem=["ufl"], solver_executor=executor, max_attempts=1,
                                        execution_policy=Queued(executor=executor, suite="bd"))
    summary = runner.run_pair(source_case(tmp_path), tmp_path / "runs")
    assert summary["plain"]["attempts"][0]["metrics"]["total_seconds"] == 22.5
    clock = iter([200.0, 205.0] + [205.0] * 100)  # a queue time beyond the measured attempt never makes it negative
    Queued.result = lambda self, *args: replace(SmallScaleNativePolicy.result(self, *args), queue_seconds=100.0)
    summary = runner.run_pair(source_case(tmp_path / "second"), tmp_path / "second-runs")
    assert summary["plain"]["attempts"][0]["metrics"]["total_seconds"] == 0.0


# ------------------------------------------------------------------------------------------------ Claude Code slots, evidence, state root


def test_claude_slots_have_their_own_volumes_repairs_and_admission(tmp_path, monkeypatch):
    assert auth.volume_slot("claude-code-cli", CLAUDE_SLOT_VOLUMES[0]) is None and auth.volume_slot("claude-code-cli", CLAUDE_SLOT_VOLUMES[1]) == 1
    assert auth.repair_command("claude-code-cli", CLAUDE_SLOT_VOLUMES[0]) == "python src/02_experiment.py docker auth --backend claude --login --replace"
    assert auth.repair_command("claude-code-cli", CLAUDE_SLOT_VOLUMES[1]) == "python src/02_experiment.py docker auth --backend claude --slot 1 --login --replace"
    assert auth.repair_command("codex-cli", CODEX_SLOT_VOLUMES[2]).endswith("--backend codex --slot 2 --login --replace")
    checked = []
    monkeypatch.setattr(agent_auth, "check_backend", lambda engine, image, backend, *, slots, models, passphrase: checked.append((backend, list(slots) if slots is not None else None)) or
                        [{"backend": backend, "volume": auth.auth_volume(backend, slot), "ok": True, "status": "ok"} for slot in (slots or [0])])
    summary = agent_auth.require_backend_authentication(None, "sha256:image", ["claude-code-cli", "codex-cli"], models={}, codex_slots=3, claude_slots=2, agy_passphrase=None, record_dir=tmp_path / "records")
    assert checked == [("claude-code-cli", [0, 1]), ("codex-cli", [0, 1, 2])] and set(summary) == set(CLAUDE_SLOT_VOLUMES) | set(CODEX_SLOT_VOLUMES)
    assert read(next((tmp_path / "records").glob("*.json")))["claude_slots_checked"] == 2
    checked.clear()
    agent_auth.require_backend_authentication(None, "sha256:image", ["claude-code-cli"], models={}, codex_slots=2, agy_passphrase=None, record_dir=tmp_path / "legacy")
    assert checked == [("claude-code-cli", [0])] and "claude_slots_checked" not in read(next((tmp_path / "legacy").glob("*.json")))  # the single-volume record
    # A failed slot-1 token is cleared only by a newer token in slot 1, and the refusal names slot 1's repair.
    save(tmp_path / "auth_invalid.json", {"recorded_at": "2026-11-08T10:00:00+00:00", "volumes": {CLAUDE_SLOT_VOLUMES[1]: {"backend": "claude-code-cli", "failed_at": "2026-11-08T10:00:00+00:00"}}})
    from src.agent.docker_client import AuthenticationRequired
    with pytest.raises(AuthenticationRequired, match="--slot 1 --login --replace"):
        campaign.resolve_auth_sentinel(tmp_path, {CLAUDE_SLOT_VOLUMES[1]: {"created_at": "2026-11-08T09:00:00+00:00"}})


def test_the_cli_selects_claude_slots_and_bounds_them(tmp_path, monkeypatch):
    stored = []
    monkeypatch.setattr(docker_setup, "prepared_engine", lambda *args, **kwargs: (None, "sha256:image"))
    monkeypatch.setattr(agent_auth, "login_claude", lambda engine, image, **kwargs: stored.append(kwargs))
    docker_setup.main(["auth", "--backend", "claude", "--slot", "1", "--login", "--evidence-root", str(tmp_path)])
    docker_setup.main(["auth", "--backend", "claude", "--login", "--evidence-root", str(tmp_path)])
    assert [entry["slot"] for entry in stored] == [1, None]
    for arguments in (["--backend", "claude", "--slot", "2", "--check"], ["--backend", "agy", "--slot", "0", "--check"], ["--backend", "codex", "--slot", "3", "--check"]):
        with pytest.raises(SystemExit) as stopped:
            docker_setup.main(["auth", *arguments, "--evidence-root", str(tmp_path)])
        assert stopped.value.code == 2


def test_legacy_runtime_evidence_stays_valid_but_cannot_run_more_slots_than_it_declares():
    from src.experiments.docker_setup import require_slot_volumes, slot_volumes
    legacy = {"codex_slot_volumes": list(CODEX_SLOT_VOLUMES[:2])}  # written before the third Codex and second Claude slot
    assert slot_volumes(legacy, "codex-cli") == list(CODEX_SLOT_VOLUMES[:2]) and slot_volumes(legacy, "claude-code-cli") == [CLAUDE_SLOT_VOLUMES[0]]
    require_slot_volumes(legacy, {"codex-cli": 2, "claude-code-cli": 1})  # the paper and harder campaigns
    with pytest.raises(Exception, match="Rerun `python src/02_experiment.py docker prepare`"):
        require_slot_volumes(legacy, {"codex-cli": 3, "claude-code-cli": 2})
    require_slot_volumes(ENVIRONMENT, {"codex-cli": 3, "claude-code-cli": 2})
    for broken in ({"codex_slot_volumes": [CODEX_SLOT_VOLUMES[0]]}, {"codex_slot_volumes": list(reversed(CODEX_SLOT_VOLUMES))}, ENVIRONMENT | {"claude_slot_volumes": ["other"]}):
        assert slot_volumes(broken, "codex-cli") is None or slot_volumes(broken, "claude-code-cli") is None


def test_the_state_directory_does_not_follow_the_launch_environment(monkeypatch, tmp_path):
    import tempfile
    from src.experiments import locks
    before = locks.state_root()
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", None)
    locks.user_temporary_directory.cache_clear()
    try:
        assert tempfile.gettempdir() == str(tmp_path) and locks.state_root() == before and not str(before).startswith(str(tmp_path))
    finally:
        locks.user_temporary_directory.cache_clear()


# ------------------------------------------------------------------------------------------------ runners, legacy run, status, dashboard


def test_the_variant_campaign_gives_every_runner_and_client_the_memory_budget(lanes_campaign, monkeypatch):
    from src.experiments.paired_runner import PairedRunner
    fixture = lanes_campaign
    executor = fixture.executor = object.__new__(EXECUTOR)
    executor.memory_policy, executor.memory_limit_mib, executor.record_queue_seconds = sandbox.MEMORY_POLICY_2GIB, 2048, True
    executor.output_policy, executor.output_limit = SMALL["solver_output_policy"], sandbox.OUTPUT_LIMITS[SMALL["solver_output_policy"]]
    built = []

    class Built(PairedRunner):
        def run_pair(self, source, destination):
            for step, client in (("agent", self.client), ("reflection", self.reflection_client), ("llm_verifier", self.judge_client)):
                self.check_client(client, step)  # the clients agree with the runner's layout and runtime contract
            built.append(self)

    monkeypatch.setattr(campaign, "PairedRunner", Built)
    assert campaign.run(fixture.root, models=["gpt-a", "claude-a"]) == 0
    assert len(built) == 6
    for runner in built:
        assert runner.policy.id == SMALL_SCALE_LANES_POLICY and runner.spec_record() == {"execution_policy": SMALL_SCALE_LANES_POLICY}
        assert runner.runtime_spec.memory_mib == runner.client.runtime_spec.memory_mib == runner.reflection_client.runtime_spec.memory_mib == 2048
        assert "resident-memory budget of 2 GiB" in runner.runtime_spec.harness_rules() and runner.judge_client.backend == "codex-cli"


def test_without_lanes_the_sequential_run_keeps_its_status_and_executor(lanes_campaign):
    fixture = lanes_campaign
    for key in ("lanes", "solver_memory_policy", "resource_guard", "experiment_plan", "active_codex_invocation_limit", "active_claude_invocation_limit"):
        fixture.manifest.pop(key)
    fixture.manifest["workers"] = 2
    fixture.behavior = lambda kwargs, instance: None
    assert campaign.run(fixture.root, models=["gpt-a", "claude-a"]) == 0
    state = read(fixture.root / "status.json")
    assert state["workers"] == 2 and state["active_codex_invocation_limit"] == 2 and not {"lanes", "active_models", "active_claude_invocation_limit", "solver_memory_limit_mib"} & set(state)
    assert {thread for _, _, thread in fixture.runs} <= {"experiment_0", "experiment_1"} and [model for model, _, _ in fixture.runs] == ["gpt-a"] * 3 + ["claude-a"] * 3
    assert "claude_slots" not in fixture.admissions[0] and fixture.admissions[0]["codex_slots"] == 2
    assert not {"memory_policy", "ownership", "record_queue_seconds"} & set(fixture.executor_options[0])


def test_status_and_dashboard_show_every_lane_with_its_limits_and_budgets(lanes_campaign, monkeypatch):
    from datetime import datetime, timezone
    from src.experiments.dashboard_server import Dashboard
    fixture = lanes_campaign
    fixture.behavior = lambda kwargs, instance: None
    assert campaign.run(fixture.root, models=["gpt-a", "gemini"]) == 0
    manifest = fixture.manifest | {"backend": "native", "max_attempts": 10, "instances": [case | {"size": "small", "seed": 0} for case in fixture.manifest["instances"]]}
    save(fixture.root / "campaign.json", manifest)
    summary = campaign.status(fixture.root)
    assert [(lane["name"], lane["workers"], lane["models"]) for lane in summary["lanes"]] == [(lane["name"], lane["workers"], lane["models"]) for lane in LANES]
    assert summary["call_limits"] == {"codex-cli": 3, "claude-code-cli": 2, "agy-cli": 1} and summary["solver_memory_limit_mib"] == 2048 and summary["resource_guard"]["max_rss_mib"] == 5120
    assert summary["workers"] == 6 and summary["lanes"][1]["completed_pairs"] == 0
    save(fixture.root / "monitor/status.json", {"state": "running", "sampled_at_utc": datetime.now(timezone.utc).isoformat()})
    snapshot = Dashboard(fixture.root, reports_root=fixture.root / "reports").snapshot()
    assert [lane["name"] for lane in snapshot["lanes"]] == ["codex", "claude", "gemini"] and snapshot["lanes"][0]["worker_ids"] == ["experiment-codex_0", "experiment-codex_1", "experiment-codex_2"]
    protocol = snapshot["protocol"]
    assert (protocol["workers"], protocol["codex_concurrency"], protocol["claude_concurrency"], protocol["agy_concurrency"]) == (6, 3, 2, 1)
    assert (protocol["max_cpu_percent"], protocol["max_rss_mib"], protocol["solver_memory_limit_mib"]) == (500, 5120, 2048)


def test_the_dashboard_counts_a_closed_pair_only_under_the_campaign_settings(tmp_path):
    from datetime import datetime, timezone
    from src.experiments.dashboard_server import Dashboard
    from src.utils.records import identity
    case = {"instance": "ufl_small_seed000", "suite": "bd", "problem": "ufl", "size": "small", "seed": 0}
    manifest = {"backend": "native", "model_queue": [{"model": "gpt-6-sol", "backend": "codex-cli", "effort": "medium"}], "max_attempts": 10, "instances": [case], "experiment_plan": {"execution_policy": SMALL_SCALE_LANES_POLICY},
                "prompt_protocol": SMALL["prompt_protocol"]}
    manifest["protocol_id"] = identity(manifest)
    save(tmp_path / "campaign.json", manifest)
    save(tmp_path / "status.json", {"state": "paused", "active_instances": {}})
    save(tmp_path / "monitor/status.json", {"state": "paused", "sampled_at_utc": datetime.now(timezone.utc).isoformat()})
    attempts = [{"attempt": index, "status": "ok", "metrics": {}} for index in range(1, 11)]
    for mode in ("plain", "reflection"):
        directory = tmp_path / "runs/gpt-6-sol" / mode / "bd/ufl/ufl_small_seed000"
        save(directory / "instance_summary.json", {"model": "gpt-6-sol", "instance": case["instance"], "protocol_id": manifest["protocol_id"], "max_attempts": 10, "attempts": attempts, "complete": True})
        save(directory / "identity.json", {key: SMALL["prompt_protocol"][key] for key in ("decomposition_contract", "context_delivery", "judge_context_delivery")})  # closed without the variant's spec
    snapshot = Dashboard(tmp_path, reports_root=tmp_path / "reports").snapshot()
    assert snapshot["totals"]["completed"] == 0 and any("incompatible protocol" in alert["text"] for alert in snapshot["alerts"])
    for mode in ("plain", "reflection"):
        directory = tmp_path / "runs/gpt-6-sol" / mode / "bd/ufl/ufl_small_seed000"
        save(directory / "identity.json", read(directory / "identity.json") | {"specs": {"execution_policy": SMALL_SCALE_LANES_POLICY}})
    assert Dashboard(tmp_path, reports_root=tmp_path / "reports").snapshot()["totals"]["completed"] == 1


# ------------------------------------------------------------------------------------------------ review round 2


@pytest.mark.parametrize("stubborn", [False, True])
def test_cleanup_finds_a_child_forked_into_the_group_after_the_first_census(monkeypatch, stubborn):
    """The parent forks C when it is told to stop and exits; C is reparented but keeps the group. Every census must see it."""
    import signal as signals
    from src.experiments import resources
    launchd = ProcessSample(1, 0, 1, 0.0, 1024, "Sat Nov  7 08:00:00 2026", "launchd", "S")
    parent = ProcessSample(100, 50, 100, 0.0, 1024, "Sat Nov  7 10:00:00 2026", "python3", "S")
    child = ProcessSample(101, 1, 100, 0.0, 1024, "Sat Nov  7 10:00:05 2026", "python3", "S")
    world, sent = {"live": {100: parent}}, []

    def killpg(group, signum):
        sent.append((group, signum))
        if signum == signals.SIGTERM:
            world["live"] = {101: child}  # the late fork, then the parent's exit
        elif not stubborn:
            world["live"] = {}

    clock = iter(range(1000))
    monkeypatch.setattr(resources, "read_processes", lambda: {1: launchd} | world["live"])
    monkeypatch.setattr(resources.os, "killpg", killpg)
    monkeypatch.setattr(resources, "time", SimpleNamespace(sleep=lambda _: None, monotonic=lambda: float(next(clock))))
    process = SimpleNamespace(pid=100, poll=lambda: 0, wait=lambda timeout=None: 0)
    if stubborn:
        with pytest.raises(CleanupError, match=r"remain alive: \[101\]"):
            resources.stop_owned_tree(process, grace=0)
    else:
        resources.stop_owned_tree(process, grace=0)
    assert sent[:2] == [(100, signals.SIGTERM), (100, signals.SIGKILL)]  # the group is signalled again for its late member


def test_a_failure_published_after_the_wait_snapshot_blocks_the_next_reservation(lanes_campaign, monkeypatch):
    fixture = lanes_campaign
    fixture.manifest["instances"] = [{"suite": "bd", "problem": "ufl", "instance": f"case-{index}"} for index in range(6)]
    released, real_wait = threading.Event(), campaign.wait

    def snapshot(futures, **kwargs):
        finished, pending = real_wait(futures, **kwargs)
        if not released.is_set() and any(futures[future][2]["instance"] == "case-0" for future in finished):
            released.set()  # case-1 fails now, after this snapshot and before the coordinator refills
            failing = next(future for future in futures if futures[future][2]["instance"] == "case-1")
            failing.exception(timeout=5)
        return finished, pending

    monkeypatch.setattr(campaign, "wait", snapshot)

    def behavior(kwargs, instance):
        if instance == "case-1":
            assert released.wait(5)
            raise ProtocolError("changed inputs")
        if instance == "case-2":
            assert released.wait(5)

    fixture.behavior = behavior
    assert campaign.run(fixture.root, models=["gpt-a"]) == 64
    assert sorted(instance for _, instance, _ in fixture.runs) == ["case-0", "case-1", "case-2"]  # nothing reserved after the published failure


def test_a_retryable_failure_backs_off_only_its_lane_and_retries_the_pair(lanes_campaign):
    fixture = lanes_campaign
    fixture.manifest["retry_delays_seconds"] = [0.5]
    failed, starts, seen = threading.Event(), {}, []

    def behavior(kwargs, instance):
        key = (kwargs["model"], instance)
        starts.setdefault(key, []).append(time.monotonic())
        if key == ("gpt-a", "case-0") and len(starts[key]) == 1:
            failed.set()
            raise InfrastructureError("gpt-a agent: usage limit reached")
        if kwargs["model"] == "claude-a":
            assert failed.wait(5)
            if instance == "case-2":  # admitted while the Codex lane backs off
                wait_for(lambda: read(fixture.root / "status.json")["lanes"]["codex"]["state"] == "backing_off")
                seen.append(read(fixture.root / "status.json")["lanes"]["codex"])
            time.sleep(0.05)

    fixture.behavior = behavior
    assert campaign.run(fixture.root, models=["gpt-a", "claude-a"]) == 0
    assert fixture.completed == {(model, f"case-{index}") for model in ("gpt-a", "claude-a") for index in range(3)}
    first, retry = starts[("gpt-a", "case-0")]
    assert retry - first >= 0.5 and min(starts[("claude-a", "case-2")]) < retry  # the other lane kept admitting during the backoff
    assert seen[0]["reason"] == "InfrastructureError: gpt-a agent: usage limit reached" and seen[0]["retry_at"] and seen[0]["consecutive_failures"] == 1
    record = read(next((fixture.root / "lane_failures").glob("*.json")))
    assert (record["lane"], record["instance"], record["retry_after_seconds"], record["retryable"]) == ("codex", "case-0", 0.5, True)
    state = read(fixture.root / "status.json")
    assert state["state"] == "paused" and state["scope_complete"] and state["lanes"]["codex"]["state"] == "complete" and "error" not in state  # the scope is finished


def test_repeated_retryable_failures_follow_the_backoff_schedule(lanes_campaign):
    fixture = lanes_campaign
    fixture.manifest["retry_delays_seconds"] = [0.05, 0.1]
    fixture.manifest["instances"] = fixture.manifest["instances"][:1]
    starts = []

    def behavior(kwargs, instance):
        starts.append(time.monotonic())
        if len(starts) <= 3:
            raise InfrastructureError("provider unavailable")

    fixture.behavior = behavior
    assert campaign.run(fixture.root, models=["gpt-a"]) == 0
    records = [read(path) for path in sorted((fixture.root / "lane_failures").glob("*.json"))]
    assert [record["retry_after_seconds"] for record in records] == [0.05, 0.1, 0.1] and [record["consecutive_failures"] for record in records] == [1, 2, 3]  # the last delay repeats
    assert len(starts) == 4 and all(later - earlier >= delay for earlier, later, delay in zip(starts, starts[1:], [0.05, 0.1, 0.1]))


def test_a_pause_during_a_backoff_ends_it_without_a_retry(lanes_campaign):
    fixture = lanes_campaign
    fixture.manifest["retry_delays_seconds"] = [30]

    def behavior(kwargs, instance):
        if (kwargs["model"], instance) == ("gpt-a", "case-0"):
            raise InfrastructureError("provider unavailable")
        if (kwargs["model"], instance) == ("claude-a", "case-0"):
            wait_for(lambda: read(fixture.root / "status.json")["lanes"]["codex"]["state"] == "backing_off")
            save(fixture.root / "pause_request.json", {"resume_requires_explicit_user_request": True, "reason": "user requested a drain"})

    fixture.behavior = behavior
    started = time.monotonic()
    assert campaign.run(fixture.root, models=["gpt-a", "claude-a"]) == 0
    assert time.monotonic() - started < 15 and sum(run[:2] == ("gpt-a", "case-0") for run in fixture.runs) == 1
    state = read(fixture.root / "status.json")
    assert state["state"] == "paused" and state["lanes"]["codex"]["state"] == "stopped" and not state["active_instances"]
    assert read(fixture.root / "pause_request.json")["reason"] == "user requested a drain"


def test_consecutive_retries_with_authentication_markers_stop_the_campaign(lanes_campaign, monkeypatch):
    from src.agent.docker_client import DockerCliClient
    from src.agent.result import AgentResult
    fixture = lanes_campaign
    fixture.manifest["retry_delays_seconds"] = [0.01]
    fixture.manifest["instances"] = fixture.manifest["instances"][:1]
    monkeypatch.setattr(DockerCliClient, "edit_file", lambda self, target, prompt, **kwargs: AgentResult(False, 1, 0, {}, "401 Unauthorized"))

    def behavior(kwargs, instance):
        kwargs["client"].edit_file(fixture.root / "solver.py", "Solve.", log_dir=fixture.root)  # the campaign client records the marker
        raise InfrastructureError("gpt-a agent: 401 Unauthorized")

    fixture.behavior = behavior
    assert campaign.run(fixture.root, models=["gpt-a"]) == 64
    state = read(fixture.root / "status.json")
    assert state["state"] == "repair_required" and "3 consecutive retries in lane codex failed with authentication markers" in state["error"]
    assert len(fixture.runs) == 3 and len(list((fixture.root / "lane_failures").glob("*.json"))) == 2


# ------------------------------------------------------------------------------------------------ review round 3


def test_an_exited_candidate_stays_unreaped_until_the_cleanup_reaps_it():
    """A real harmless child (/bin/sleep): kqueue reports its exit, it stays a zombie holding its PID and group number."""
    from src.experiments import resources
    process = subprocess.Popen(["/bin/sleep", "0.2"], start_new_session=True)
    exits = resources.ChildExit(process)
    try:
        assert not exits.wait(0) and exits.wait(10)
        assert process.returncode is None and resources.read_processes()[process.pid].state.startswith("Z")
        assert resources.ChildExit(process).wait(0)  # registered after the exit: reported at once
        resources.stop_owned_tree(process, grace=0)  # nothing live is left; reaped only after the verification
        assert process.returncode == 0
    finally:
        exits.close()


def test_cleanup_never_signals_a_reused_group_number(monkeypatch):
    """The root was reaped before the cleanup (no anchor); a new session leader got its PID: refused, not signalled."""
    from src.experiments import resources
    launchd = ProcessSample(1, 0, 1, 0.0, 1024, "Sat Nov  7 08:00:00 2026", "launchd", "S")
    stranger = ProcessSample(100, 1, 100, 0.0, 1024, "Sat Nov  7 11:00:00 2026", "Safari", "S")
    sent = []
    monkeypatch.setattr(resources, "read_processes", lambda: {1: launchd, 100: stranger})
    monkeypatch.setattr(resources.os, "killpg", lambda group, signum: sent.append((group, signum)))
    reaped = SimpleNamespace(pid=100, returncode=0, wait=lambda timeout=None: 0)
    with pytest.raises(CleanupError, match="identity cannot be established, so nothing is signalled"):
        resources.stop_owned_tree(reaped, grace=0)
    assert sent == []


def test_cleanup_reaps_the_root_only_after_its_final_verification(monkeypatch):
    from src.experiments import resources
    launchd = ProcessSample(1, 0, 1, 0.0, 1024, "Sat Nov  7 08:00:00 2026", "launchd", "S")
    root = ProcessSample(100, 50, 100, 0.0, 1024, "Sat Nov  7 10:00:00 2026", "python3", "S")
    zombie = ProcessSample(100, 50, 100, 0.0, 0, "Sat Nov  7 10:00:00 2026", "python3", "Z")
    events, world = [], {"live": {100: root}}

    def census():
        events.append("census")
        return {1: launchd} | world["live"]

    monkeypatch.setattr(resources, "read_processes", census)
    monkeypatch.setattr(resources.os, "killpg", lambda group, signum: world.update(live={100: zombie}))
    process = SimpleNamespace(pid=100, returncode=None, poll=lambda: pytest.fail("the root must not be reaped during the cleanup"),
                              wait=lambda timeout=None: events.append("reap") or 0)
    resources.stop_owned_tree(process, grace=0)
    assert events[-1] == "reap" and events.count("reap") == 1 and events[-2] == "census"


def test_the_memory_watcher_never_signals_an_unanchored_group(killed):
    watcher = sandbox.GroupMemoryWatcher(4242, limit_mib=2048, interval=0.01, sampler=high, anchored=lambda: False)
    assert watcher.done.wait(5)
    watcher.close()
    assert killed == [] and "can no longer be identified" in watcher.error and watcher.exceeded


def test_an_expiring_backoff_keeps_the_block_of_a_newer_unconsumed_failure(lanes_campaign, monkeypatch):
    fixture = lanes_campaign
    fixture.manifest["retry_delays_seconds"] = [0.3]
    fixture.manifest["instances"] = fixture.manifest["instances"][:2]
    real_wait, failed_at, released, starts = campaign.wait, [], threading.Event(), {}

    def snapshot(futures, **kwargs):
        finished, pending = real_wait(futures, **kwargs)
        if failed_at and not released.is_set() and time.time() >= failed_at[0] + 0.3:
            released.set()  # case-0's backoff has just expired: case-1 fails after this snapshot, before its consumption
            next(future for future in futures if futures[future][2]["instance"] == "case-1").exception(timeout=5)
        return finished, pending

    monkeypatch.setattr(campaign, "wait", snapshot)

    def behavior(kwargs, instance):
        starts.setdefault(instance, []).append(time.time_ns())
        if instance == "case-0" and len(starts[instance]) == 1:
            failed_at.append(time.time())
            raise InfrastructureError("provider unavailable")
        if instance == "case-1" and len(starts[instance]) == 1:
            assert released.wait(5)
            raise InfrastructureError("provider unavailable")

    fixture.behavior = behavior
    assert campaign.run(fixture.root, models=["gpt-a"]) == 0
    second_backoff = sorted(int(path.stem) for path in (fixture.root / "lane_failures").glob("*.json"))[1]
    assert len(starts["case-0"]) == 2 and min(starts["case-0"][1], starts["case-1"][1]) > second_backoff  # nothing re-admitted before case-1's backoff began


def test_pair_outcomes_count_in_completion_order_not_consumption_order(lanes_campaign, monkeypatch):
    from src.agent.docker_client import DockerCliClient
    from src.agent.result import AgentResult
    fixture = lanes_campaign
    fixture.manifest["retry_delays_seconds"] = [0.01]
    monkeypatch.setattr(DockerCliClient, "edit_file", lambda self, target, prompt, **kwargs: AgentResult(False, 1, 0, {}, "401 Unauthorized"))
    real_wait, released, held = campaign.wait, threading.Event(), set()

    def snapshot(futures, **kwargs):
        finished, pending = real_wait(futures, **kwargs)
        success = {future for future in finished if futures[future][2]["instance"] == "case-0"}
        if success and not released.is_set():
            released.set()  # the success has ended (and published); now case-1 and case-2 fail, marked
            failures = {future for future in futures if futures[future][2]["instance"] in ("case-1", "case-2")}
            for future in failures:
                future.exception(timeout=5)
            held.update(success)
            return (finished - success) | failures, pending  # the coordinator consumes the success last
        return finished, pending

    monkeypatch.setattr(campaign, "wait", snapshot)

    def behavior(kwargs, instance):
        if instance == "case-0":
            return
        if not released.is_set():
            assert released.wait(5)
        kwargs["client"].edit_file(fixture.root / "solver.py", "Solve.", log_dir=fixture.root)  # an authentication marker
        raise InfrastructureError("gpt-a agent: 401 Unauthorized")

    fixture.behavior = behavior
    assert campaign.run(fixture.root, models=["gpt-a"]) == 64
    state = read(fixture.root / "status.json")
    assert held and "3 consecutive retries in lane codex failed with authentication markers" in state["error"]
    # case-0, the two marked failures, then their retries: the first retry is the third marked failure in a row and stops
    # the campaign (the other retry may already be running). Counted in consumption order, the success would have reset it.
    assert len(fixture.runs) in (4, 5) and [run[1] for run in fixture.runs[:3]].count("case-0") == 1


def test_the_cooldown_evidence_survives_a_receipt_resume_a_judge_failure_and_a_second_resume(tmp_path, monkeypatch):
    from src.experiments import execution_policy
    slept = []
    monkeypatch.setattr(execution_policy.time, "sleep", slept.append)
    receipt = {"exec_ok": False, "stderr": "Traceback", "execution_seconds": 6.0, "solver_queue_seconds": 100.0, "memory_limit_mib": 2048, "memory_peak_mib": 300.0, "memory_limit_exceeded": False}
    case, executor = lanes_attempt(tmp_path, receipt)
    save(case.directory / "native_solver_receipt.json", {"request": {}, "result": receipt})  # the first controller stopped here
    policy, runtime = SmallScaleLanesPolicy(executor=executor, suite="bd", cooldown_seconds=10), RuntimeSpec(60, 1, None, 2048)
    first = policy.execute(case, delivery=None, runtime=runtime, stage=lambda phase: None)
    recorded = read(case.directory / "execution.json")
    assert (recorded["execution_source"], recorded["cooldown_decided_seconds"], recorded["cooldown_applied_seconds"]) == ("receipt", 0, 0) and first.metrics["cooldown_seconds"] == 0

    def judge():
        raise InfrastructureError("Algorithm judge produced a malformed verdict")

    with pytest.raises(InfrastructureError):
        policy.evaluate(first, None, case=case, audit=judge)  # the attempt is not scored; the lane retries it
    second = policy.execute(case, delivery=None, runtime=runtime, stage=lambda phase: None)
    assert second.metrics["cooldown_seconds"] == 0 and second.queue_seconds == 0.0 and slept == []  # never a reconstructed cooldown


def test_an_interrupted_cooldown_is_not_recorded_as_applied(tmp_path, monkeypatch):
    from src.experiments import execution_policy

    def interrupted(seconds):
        raise KeyboardInterrupt("stop requested during the cooldown")

    monkeypatch.setattr(execution_policy.time, "sleep", interrupted)
    receipt = {"exec_ok": True, "stderr": "", "execution_seconds": 6.0, "solver_queue_seconds": 1.0, "memory_limit_mib": 2048, "memory_peak_mib": 300.0, "memory_limit_exceeded": False}
    case, executor = lanes_attempt(tmp_path, receipt)
    policy, runtime = SmallScaleLanesPolicy(executor=executor, suite="bd", cooldown_seconds=10), RuntimeSpec(60, 1, None, 2048)
    with pytest.raises(KeyboardInterrupt):
        policy.execute(case, delivery=None, runtime=runtime, stage=lambda phase: None)
    recorded = read(case.directory / "execution.json")
    assert (recorded["execution_source"], recorded["cooldown_decided_seconds"], recorded["cooldown_applied_seconds"]) == ("ran", 10, 0)
    monkeypatch.setattr(execution_policy.time, "sleep", lambda seconds: pytest.fail("a resumed attempt runs no cooldown"))
    assert policy.execute(case, delivery=None, runtime=runtime, stage=lambda phase: None).metrics["cooldown_seconds"] == 0
    fresh_case, fresh_executor = lanes_attempt(tmp_path / "fresh", receipt)
    monkeypatch.setattr(execution_policy.time, "sleep", lambda seconds: None)
    done = SmallScaleLanesPolicy(executor=fresh_executor, suite="bd", cooldown_seconds=10)
    assert done.execute(fresh_case, delivery=None, runtime=runtime, stage=lambda phase: None).metrics["cooldown_seconds"] == 10
    assert read(fresh_case.directory / "execution.json")["cooldown_applied_seconds"] == 10
    assert done.execute(fresh_case, delivery=None, runtime=runtime, stage=lambda phase: None).metrics["cooldown_seconds"] == 10  # resumed: the recorded completion


@pytest.mark.parametrize("record,survivor,refused", [
    ({"state": "running", "token": "DECOMPBENCH_READY_abc", "pid": 4242, "handshake": True}, True, True),
    ({"state": "running", "token": "DECOMPBENCH_READY_abc", "pid": 4242, "handshake": True}, False, False),
    ({"state": "launching", "token": "DECOMPBENCH_READY_abc", "pid": None}, False, True),   # unresolvable: fail closed
    ({"state": "released", "token": "DECOMPBENCH_READY_abc", "pid": 4242, "handshake": True}, True, False),
])
def test_a_sequential_campaign_checks_the_host_wide_ownership_record_read_only(lanes_campaign, monkeypatch, record, survivor, refused):
    fixture = lanes_campaign
    for key in ("lanes", "solver_memory_policy", "resource_guard", "experiment_plan", "active_codex_invocation_limit", "active_claude_invocation_limit"):
        fixture.manifest.pop(key)
    fixture.manifest["workers"] = 2
    ownership = fixture.root.parent / "locks" / sandbox.OWNERSHIP_RECORD
    save(ownership, record)
    before = ownership.read_bytes()
    sample = ProcessSample(9, 1, 4242, 0.0, 1024, "Sat Nov  7 10:00:00 2026", "python3", "R")
    monkeypatch.setattr(sandbox, "read_processes", lambda: {9: sample} if survivor else {1: ProcessSample(1, 0, 1, 0.0, 1, "x", "launchd", "S")})
    monkeypatch.setattr(sandbox, "read_commands", lambda: {1: "launchd"})
    assert campaign.run(fixture.root, models=["gpt-a"]) == (64 if refused else 0)
    assert ownership.read_bytes() == before  # this mode never writes the record
    if refused:
        assert not fixture.runs and read(fixture.root / "status.json")["state"] == "repair_required" and "CleanupError" in read(fixture.root / "status.json")["error"]
    else:
        assert len(fixture.runs) == 3


# ------------------------------------------------------------------------------------------------ review round 4


def slow_low():
    time.sleep(2)  # a successful sample whose process query takes long
    return low()


def timed_kills(monkeypatch):
    kills = []

    def killpg(group, signum):
        kills.append(time.monotonic())
        FakeProcess.current.finish(-9)

    monkeypatch.setattr(sandbox.os, "killpg", killpg)
    return kills


def test_a_timeout_stops_the_group_at_once_even_while_a_slow_sample_runs(launcher, tmp_path, monkeypatch):
    task = launcher(sampler=slow_low)
    kills = timed_kills(monkeypatch)
    started = time.monotonic()
    record = task.execute_bytes(b"solver", b"{}", 0.3, 1, evidence=tmp_path / "evidence.json")
    assert kills and kills[0] - started < 1.0 and time.monotonic() - started >= 1.5  # signalled before the 2 s sample returned
    assert record["timed_out"] and not record["exec_ok"] and record["termination"] == "timeout" and record["output_changed_after_end"] is False


def test_output_a_surviving_child_would_write_after_the_leader_exits_is_never_accepted(launcher, tmp_path, monkeypatch):
    task = launcher(sampler=slow_low, leader_exit_after=0.05, child_writes_after=0.4)
    kills = timed_kills(monkeypatch)
    started = time.monotonic()
    record = task.execute_bytes(b"solver", b"{}", 60, 1, evidence=tmp_path / "evidence.json")
    assert kills and kills[0] - started < 0.4  # the group stopped when the leader's exit ended the wait
    assert not record["timed_out"] and record["returncode"] == 0 and not record["exec_ok"] and record["output"] is None
    assert "output.json missing" in record["stderr"]


def test_output_written_while_the_group_is_being_stopped_is_rejected(launcher, tmp_path, monkeypatch):
    task = launcher(sampler=low, leader_exit_after=0.05)

    def racing_kill(group, signum):  # a child's write lands after the end of the wait, just before the stop
        (FakeProcess.current.cwd / "output.json").write_text(json.dumps({"objective_value": 2}))
        FakeProcess.current.finish(-9)

    monkeypatch.setattr(sandbox.os, "killpg", racing_kill)
    record = task.execute_bytes(b"solver", b"{}", 60, 1, evidence=tmp_path / "evidence.json")
    assert record["output_changed_after_end"] and not record["exec_ok"] and record["output"] is None
    assert "output.json changed after the run ended" in record["stderr"]


def test_a_failed_end_of_run_snapshot_still_stops_joins_and_cleans_up_and_rejects_the_result(launcher, tmp_path, monkeypatch):
    # For example a candidate that made its working directory unsearchable: the snapshot raises PermissionError.
    task = launcher(sampler=low, output={"objective_value": 1}, exit_after=0.05)
    kills = timed_kills(monkeypatch)
    cleanups, original_cleanup = [], sandbox.stop_owned_tree
    monkeypatch.setattr(sandbox, "stop_owned_tree", lambda *args, **kwargs: cleanups.append(1) or original_cleanup(*args, **kwargs))
    calls = []

    def unsearchable(work):
        calls.append(1)
        raise PermissionError(13, "Permission denied", str(work / "output.json"))

    monkeypatch.setattr(sandbox.NativeSandboxExecutor, "output_state", staticmethod(unsearchable))
    record = task.execute_bytes(b"solver", b"{}", 60, 1, evidence=tmp_path / "evidence.json")
    assert kills and cleanups == [1] and record["cleanup_verified"] and record["memory_samples"] >= 1  # stopped, joined, cleaned up
    assert not [thread for thread in threading.enumerate() if thread.name.startswith("native-memory-watcher")]  # the watcher was joined
    assert record["output_changed_after_end"] and not record["exec_ok"] and record["output"] is None
    assert "output.json changed after the run ended" in record["stderr"]


def test_the_paper_executor_keeps_its_cleanup_order(launcher, tmp_path, monkeypatch):
    task = launcher(memory=False, output={"objective_value": 1}, exit_after=0.05)
    kills = timed_kills(monkeypatch)
    record = task.execute_bytes(b"solver", b"{}", 60, 1, evidence=tmp_path / "evidence.json")
    assert kills == [] and record["exec_ok"] and "output_changed_after_end" not in record  # only stop_owned_tree stops it, as before


class FakeKqueue:
    def __init__(self, behavior):
        self.behavior, self.closed = behavior, False

    def control(self, changes, maximum, timeout=None):
        return self.behavior()

    def close(self):
        self.closed = True


@pytest.mark.parametrize("outcome", ["ev_error", "interrupted", "oserror"])
def test_a_failed_exit_watch_closes_its_kqueue(monkeypatch, outcome):
    import errno
    import select
    from src.experiments import resources

    def behavior():
        if outcome == "ev_error":
            return [SimpleNamespace(flags=select.KQ_EV_ERROR, data=errno.EPERM, fflags=0)]
        if outcome == "interrupted":
            raise KeyboardInterrupt("stop requested")
        raise PermissionError(errno.EPERM, "not permitted")

    created = []
    monkeypatch.setattr(resources.select, "kqueue", lambda: created.append(FakeKqueue(behavior)) or created[-1])
    process = SimpleNamespace(pid=4242, returncode=None)
    with pytest.raises(KeyboardInterrupt if outcome == "interrupted" else CleanupError):
        resources.ChildExit(process)
    assert created[0].closed
    monkeypatch.setattr(resources.select, "kqueue", lambda: created.append(FakeKqueue(lambda: [SimpleNamespace(flags=select.KQ_EV_ERROR, data=errno.ESRCH, fflags=0)])) or created[-1])
    watch = resources.ChildExit(process)  # an unreaped child that cannot be watched is its zombie: exited, kqueue kept
    assert watch.exited and not created[-1].closed
    watch.close()
