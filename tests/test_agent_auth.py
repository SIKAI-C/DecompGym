"""Docker agent authentication: volume-only credentials, per-volume host locks, slot-bound Codex volumes, terminal
credential-failure classification, campaign admission and repair precedence.

No Docker engine, login, network, model call or solver is used: containers, CLIs and the app-server are fakes.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from src.agent import auth, docker_client
from src.agent.auth import AUTH_POLICY, CODEX_SLOT_VOLUMES, held_volumes, terminal_auth_marker, volume_lock
from src.agent.container import ContainerResult
from src.agent.docker_client import CLI_VERSIONS, AuthenticationRequired, DockerCliClient, credential_from_environment
from src.experiments import agent_auth, campaign
from src.experiments.call_gate import NativeCallGate
from src.experiments.environment import RuntimeEnvironmentError
from src.utils.errors import InfrastructureError

ROOT = Path(__file__).resolve().parents[1]
REUSED = "Your access token could not be refreshed because your refresh token was already used. Please log out and sign in again."


@pytest.fixture(autouse=True)
def clear_preflights():
    docker_client._PREFLIGHTS.clear()
    yield
    docker_client._PREFLIGHTS.clear()


class Runtime:
    """The container boundary: records each call's auth mounts and the host locks held while the container runs."""

    def __init__(self, root, backend, *, events="", stderr="", worker=None, contents="print(1)\n", error=None, gate=None, users=()):
        self.seccomp_profile = root / "profile.json"
        self.seccomp_profile.write_text("{}")
        self.backend, self.events, self.stderr, self.worker, self.contents, self.error, self.gate = backend, events, stderr, worker or {}, contents, error, gate
        self.mounts, self.held, self.jobs, self.recovered_with, self.users, self.probed = [], [], [], [], list(users), []

    def ensure_volume(self, *args, **kwargs):
        pass

    def volume_users(self, volume):
        self.probed.append((volume, held_volumes()))
        return list(self.users)

    def remove_container(self, name):
        self.users = [user for user in self.users if user[0] != name]

    def run(self, image, command, **kwargs):
        kwargs["state_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["state_path"].write_text("{}")
        if command == ["preflight"]:
            value = {"ok": True, "versions": {"codex": CLI_VERSIONS["codex-cli"], "claude": CLI_VERSIONS["claude-code-cli"], "agy": CLI_VERSIONS["agy-cli"]}, "worker_files_sha256": docker_client.worker_file_hashes()}
            return ContainerResult(0, "", "", 0, artifacts={"worker.json": json.dumps(value)})
        self.mounts.append(kwargs["volumes"])
        self.held.append(held_volumes())
        self.jobs.append(kwargs["payload"])
        if self.gate:
            self.gate()
        if self.error:
            raise self.error
        worker = {"cli_version": CLI_VERSIONS[self.backend], "requested_model": "fixture", "target": kwargs["payload"]["target"], "returncode": 0, "terminal_state": "completed", "error": None,
                  "usage": {}, "usage_available": True, "warnings": [], "observed_models": ["fixture"], "agent_seconds": 1.0} | self.worker
        self.result = ContainerResult(worker["returncode"], "", "", 1, artifacts={kwargs["payload"]["target"]: self.contents, "events.jsonl": self.events, "stderr.log": self.stderr,
                                                                                 "worker.json": json.dumps(worker), "delivery.json": json.dumps({"misplaced": {}})}, metadata={"cleanup_verified": True})
        return self.result

    def recover(self, state):
        self.recovered_with.append(held_volumes())
        return self.result


def client_for(tmp_path, backend="codex-cli", **options):
    runtime = Runtime(tmp_path, backend, **options)
    client = DockerCliClient("fixture", backend=backend, image_id="sha256:fixture", runtime=runtime, evidence_root=tmp_path / "evidence")
    target = tmp_path / "attempt/solver.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("")
    return client, runtime, target


def codex_failure(message, *, stderr=""):
    events = json.dumps({"type": "turn.failed", "error": {"message": message}}) + "\n"
    return {"events": events, "stderr": stderr, "worker": {"returncode": 1, "terminal_state": "failed", "error": message}}


# ------------------------------------------------------------------------------------------------ credential source


def test_codex_and_claude_jobs_carry_no_host_credential(tmp_path, monkeypatch):
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex/auth.json").write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "HOST_ACCESS_TOKEN_0123456789"}}))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-LAUNCH_ENVIRONMENT_TOKEN_0123456789")
    assert credential_from_environment("codex-cli") is None and credential_from_environment("claude-code-cli") is None
    monkeypatch.delenv("DECOMPBENCH_AGY_KEYRING_PASSPHRASE", raising=False)
    with pytest.raises(AuthenticationRequired):
        credential_from_environment("agy-cli")
    monkeypatch.setenv("DECOMPBENCH_AGY_KEYRING_PASSPHRASE", "synthetic-test-only")
    assert credential_from_environment("agy-cli") == {"keyring_passphrase": "synthetic-test-only"}
    client, runtime, target = client_for(tmp_path)
    assert client.edit_file(target, "Solve.", log_dir=tmp_path / "call").ok
    assert runtime.jobs[0]["credential"] is None


# ------------------------------------------------------------------------------------------------ volumes and locks


def test_each_codex_slot_mounts_its_own_volume_under_its_lock(tmp_path):
    client, runtime, target = client_for(tmp_path)
    assert client.edit_file(target, "Solve.", log_dir=tmp_path / "one").ok
    with client.bind_auth_slot(1):
        assert client.edit_file(target, "Solve.", log_dir=tmp_path / "two").ok
    assert client.auth_slot is None
    assert runtime.mounts == [{CODEX_SLOT_VOLUMES[0]: "/auth"}, {CODEX_SLOT_VOLUMES[1]: "/auth"}]
    assert runtime.held == [{CODEX_SLOT_VOLUMES[0]}, {CODEX_SLOT_VOLUMES[1]}]
    assert held_volumes() == set()
    with pytest.raises(ValueError):
        auth.auth_volume("codex-cli", 2)


@pytest.mark.parametrize("backend", ["claude-code-cli", "agy-cli"])
def test_claude_and_agy_calls_hold_their_volume_lock(tmp_path, backend):
    client, runtime, target = client_for(tmp_path, backend, worker={"observed_models": ["fixture"]})
    client.credential_provider = lambda _: {"keyring_passphrase": "synthetic"} if backend == "agy-cli" else None
    client.edit_file(target, "Solve.", log_dir=tmp_path / "call")
    assert runtime.held == [{auth.AUTH_VOLUMES[backend]}] and held_volumes() == set()


def test_lock_is_released_after_an_exception_and_a_cancelled_call(tmp_path):
    client, runtime, target = client_for(tmp_path, error=RuntimeError("container start failed"))
    with pytest.raises(RuntimeError, match="container start failed"):
        client.edit_file(target, "Solve.", log_dir=tmp_path / "one")
    runtime.error = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        client.edit_file(target, "Solve.", log_dir=tmp_path / "two")
    assert held_volumes() == set()
    with volume_lock(CODEX_SLOT_VOLUMES[0]):  # free again for the next holder
        pass


def test_nested_lock_raises_instead_of_deadlocking_and_other_threads_wait():
    volume = CODEX_SLOT_VOLUMES[0]
    entered, release, order = threading.Event(), threading.Event(), []

    def holder():
        with volume_lock(volume):
            order.append("first")
            entered.set()
            assert release.wait(5)

    def waiter():
        waited = []
        with volume_lock(volume, waiting=waited.append):
            order.append("second")
        return waited

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(holder)
        assert entered.wait(5)
        with volume_lock(CODEX_SLOT_VOLUMES[1]):  # another volume is independent
            pass
        second = pool.submit(waiter)
        time.sleep(0.2)
        assert order == ["first"]
        release.set()
        first.result()
        assert second.result() == [volume]
    assert order == ["first", "second"]
    with volume_lock(volume):
        with pytest.raises(RuntimeEnvironmentError, match="Nested lock"):
            with volume_lock(volume):
                pass
    assert held_volumes() == set()


def test_recovery_holds_the_lock_of_the_volume_the_container_mounted(tmp_path):
    client, runtime, target = client_for(tmp_path)
    with client.bind_auth_slot(1):
        client.edit_file(target, "Solve.", log_dir=tmp_path / "call")
    state = tmp_path / "call/agent_container.json"
    state.write_text(json.dumps({"mounts": [{"Type": "volume", "Name": CODEX_SLOT_VOLUMES[1], "Destination": "/auth", "RW": True}]}))
    assert client.recover_file(target, log_dir=tmp_path / "call", step="agent").ok
    assert runtime.recovered_with == [{CODEX_SLOT_VOLUMES[1]}]


def test_gate_slots_bind_distinct_volumes_and_fallback_uses_slot_zero(tmp_path):
    (tmp_path / "state").mkdir()
    gate = NativeCallGate(tmp_path / "state", tmp_path)
    seen, both, release = [], threading.Barrier(2, timeout=5), threading.Event()

    def call():
        with gate.enter() as slot:
            seen.append(slot)
            both.wait()
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(call), pool.submit(call)]
        time.sleep(0.3)
        release.set()
        for future in futures:
            future.result()
    assert sorted(seen) == [0, 1]
    gate.fallback("refresh token", role="generation", invocation_id="fixture")
    with gate.enter() as slot:
        assert slot == 0


# ------------------------------------------------------------------------------------------------ terminal failures


def event(**value):
    return json.dumps(value) + "\n"


@pytest.mark.parametrize("backend,events,stderr,succeeded,expected", [
    ("codex-cli", event(type="turn.failed", error={"message": REUSED}), "", False, "your access token could not be refreshed"),
    ("codex-cli", event(type="error", message="unexpected status 401 Unauthorized: invalidated oauth token"), "", False, "invalidated oauth token"),
    ("codex-cli", event(type="turn.failed", error={"message": "unexpected status 401 Unauthorized: {}"}), "", False, "unexpected status 401"),
    ("codex-cli", "", "ERROR codex_core::auth: refresh_token_reused", False, "refresh_token_reused"),
    ("codex-cli", event(type="turn.failed", error={"message": "stream disconnected"}), "WARN request failed: 401 Unauthorized", False, None),  # bare 401 in stderr
    ("codex-cli", event(type="turn.failed", error={"message": REUSED}), "", True, None),  # a completed call recovered
    ("codex-cli", event(type="item.completed", item={"type": "agent_message", "text": REUSED}) + event(type="turn.failed", error={"message": "stream disconnected"}), "", False, None),
    ("claude-code-cli", event(type="assistant", error="authentication_failed", message={"content": [{"type": "text", "text": "Failed to authenticate."}]}), "", False, "authentication_failed"),
    ("claude-code-cli", event(type="result", subtype="success", is_error=True, result="Invalid API key · Please run /login"), "", False, "please run /login"),
    ("claude-code-cli", event(type="assistant", error="rate_limit", message={"content": [{"type": "text", "text": "API Error: 429"}]}), "", False, None),
    ("claude-code-cli", event(type="assistant", message={"content": [{"type": "text", "text": "I would run /login. Please run /login."}]}) + event(type="result", subtype="error_max_turns", is_error=True, result=""), "", False, None),
    ("agy-cli", event(event="result", result={"status": "FAILED", "error": "Please sign in to continue"}), "", False, "please sign in"),
    ("agy-cli", "", "Welcome back! You are currently not signed in.", False, "currently not signed in"),
    ("agy-cli", event(event="result", result={"status": "FAILED", "error": "RESOURCE_EXHAUSTED: quota"}), "", False, None),
])
def test_terminal_auth_markers_come_only_from_error_channels(backend, events, stderr, succeeded, expected):
    assert terminal_auth_marker(backend, events, stderr, succeeded=succeeded) == expected


def test_terminal_failure_is_saved_uncommitted_and_stops_for_repair(tmp_path):
    client, runtime, target = client_for(tmp_path, **codex_failure(REUSED))
    with client.bind_auth_slot(1), pytest.raises(AuthenticationRequired) as stopped:
        client.edit_file(target, "Solve.", log_dir=tmp_path / "call")
    evidence = stopped.value.evidence
    assert evidence["auth_volume"] == CODEX_SLOT_VOLUMES[1] and "--slot 1 --login --replace" in evidence["repair"] and evidence["charged_to_agent"] is False
    metrics = json.loads((tmp_path / "call/agent_metrics.json").read_text())
    assert metrics["ok"] is False and metrics["terminal_state"] == "infrastructure_failed" and metrics["error"].startswith("credential_invalid:")
    assert json.loads((tmp_path / "call/agent_auth_invalid.json").read_text())["marker"] == "your access token could not be refreshed"
    assert target.read_text() == "" and held_volumes() == set()


def test_runner_retries_a_credential_failure_after_repair_without_charging_the_agent(tmp_path, monkeypatch):
    from src.experiments.paired_runner import PairedRunner
    monkeypatch.setattr("src.experiments.paired_runner.runtime_lock", lambda *_: nullcontext())
    client, runtime, _ = client_for(tmp_path, **codex_failure(REUSED))
    runner = PairedRunner(client=client, judge_client=client, solver_executor=lambda *_: pytest.fail("no solver runs in this test"))
    target = tmp_path / "case/attempt_01/solver.py"
    target.parent.mkdir(parents=True)
    with pytest.raises(AuthenticationRequired):
        runner.edit(client, target, "Solve.", [], "agent")
    assert not (target.parent / "agent_receipt.json").exists()
    runtime.events, runtime.worker = "", {}  # the owner repaired the credential
    result = runner.edit(client, target, "Solve.", [], "agent")
    assert result["ok"] and target.read_text() == "print(1)\n"
    assert len(runtime.jobs) == 2 and len(list((target.parent / "calls").iterdir())) == 2
    receipt = json.loads((target.parent / "agent_receipt.json").read_text())
    assert receipt["result"]["invocation_id"] == result["invocation_id"]


# ------------------------------------------------------------------------------------------------ campaign


def campaign_fixture(tmp_path, monkeypatch, *, backend="claude-code-cli", cases=1, runner=None, admission=None):
    from src.experiments import docker_setup
    from src.experiments import sandbox as native_solver
    root = tmp_path / "campaign"
    (root / "controller/src").mkdir(parents=True)
    monkeypatch.setattr(campaign, "REPOSITORY", root / "controller")
    manifest = {"model_queue": [{"model": "tested-model", "backend": backend, "effort": "medium"}], "instances": [{"suite": "bd", "problem": "ufl", "instance": f"case-{index}"} for index in range(cases)],
                "protocol_id": "fixture", "agent_timeout": 900, "judge_model": "judge-model", "judge_effort": "high",
                "execution_environment": {"image_id": "sha256:fixture", "license_file": "fixture", "native_environment": {}}}
    monkeypatch.setattr(docker_setup, "runtime", lambda: SimpleNamespace(image_id=lambda name: name, bind_environment=lambda _: None))
    monkeypatch.setattr(native_solver, "NativeSandboxExecutor", lambda **kwargs: object())
    for name, value in (("validate", lambda _: manifest), ("ensure_execution_allowed", lambda *_: None), ("require_campaign_environment", lambda *_: None),
                        ("require_backend_readiness", lambda *_: {}), ("pair_complete", lambda *_: False), ("state_root", lambda: tmp_path / "locks"), ("runtime_lock", lambda *_: nullcontext())):
        monkeypatch.setattr(campaign, name, value)
    (tmp_path / "locks").mkdir()
    monkeypatch.setattr(campaign.signal, "signal", lambda *_: None)
    calls = []
    monkeypatch.setattr(campaign, "require_backend_authentication", admission or (lambda *args, **kwargs: calls.append((args, kwargs)) or {}))

    class Runner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_pair(self, source, destination):
            return runner(source.name) if runner else {}

    monkeypatch.setattr(campaign, "PairedRunner", Runner)
    return root, calls


def test_admission_checks_every_queued_backend_and_the_codex_judge(tmp_path, monkeypatch):
    root, calls = campaign_fixture(tmp_path, monkeypatch)
    (root / "auth_invalid.json").write_text("{}")
    assert campaign.run(root) == 0
    (engine, image_id, backends), options = calls[0]
    assert image_id == "sha256:fixture" and set(backends) == {"claude-code-cli", "codex-cli"}
    assert options["models"] == {"claude-code-cli": ["tested-model"], "codex-cli": ["judge-model"]} and options["codex_slots"] == 2
    assert not (root / "auth_invalid.json").exists() and list(root.glob("auth_invalid.resolved-*.json"))


def test_failed_admission_refuses_to_start_and_records_the_repair(tmp_path, monkeypatch):
    started = []

    def admission(*args, **kwargs):
        raise AuthenticationRequired("codex-cli decompbench-v3-codex-auth-slot1: credential_invalid", {"kind": "credential_invalid"})

    root, _ = campaign_fixture(tmp_path, monkeypatch, runner=started.append, admission=admission)
    assert campaign.run(root) == 64 and not started
    assert json.loads((root / "status.json").read_text())["state"] == "repair_required"
    sentinel = json.loads((root / "auth_invalid.json").read_text())
    assert sentinel["evidence"] == {"kind": "credential_invalid"} and "slot1" in sentinel["error"]
    failure = json.loads(next((root / "controller_failures").glob("*.json")).read_text())
    assert failure["retryable"] is False and failure["error_type"] == "AuthenticationRequired"


@pytest.mark.parametrize("first", ["auth", "infrastructure"])
def test_repair_required_dominates_retryable_failures_while_draining(tmp_path, monkeypatch, first):
    failed = threading.Event()
    errors = {"auth": AuthenticationRequired("credential invalid"), "infrastructure": InfrastructureError("provider unavailable")}
    second = "infrastructure" if first == "auth" else "auth"

    def runner(name):
        if name == "case-0":
            failed.set()
            raise errors[first]
        assert failed.wait(5)
        time.sleep(0.3)
        raise errors[second]

    root, _ = campaign_fixture(tmp_path, monkeypatch, cases=2, runner=runner)
    assert campaign.run(root) == 64
    status = json.loads((root / "status.json").read_text())
    assert status["state"] == "repair_required" and status["error"].startswith("AuthenticationRequired")


def test_watchdog_stops_after_repeated_auth_marked_restarts(tmp_path, monkeypatch):
    from src.experiments import docker_setup
    manifest = {"execution_environment": {}, "resource_guard": {}, "retry_delays_seconds": [0], "model_queue": [{"model": "fixture", "backend": "codex-cli", "effort": "medium"}]}
    for name, value in (("validate", lambda _: manifest), ("ensure_execution_allowed", lambda *_: None), ("require_campaign_environment", lambda *_: None),
                        ("runtime_lock", lambda *_: nullcontext()), ("sample_host", lambda: (None, "", 100)), ("thermal_warnings", lambda _: [])):
        monkeypatch.setattr(campaign, name, value)
    monkeypatch.setattr(docker_setup, "runtime", lambda: SimpleNamespace(bind_environment=lambda _: None, resource_samples=lambda: [], stop_campaign_workers=lambda _: None))
    monkeypatch.setattr(campaign.signal, "signal", lambda *_: None)
    monkeypatch.setattr(campaign.time, "sleep", lambda _: None)
    runs = []

    def supervise(*args, **kwargs):
        runs.append(1)
        (tmp_path / "controller_failures").mkdir(exist_ok=True)
        (tmp_path / "controller_failures" / f"{time.time_ns()}.json").write_text(json.dumps({"retryable": True, "auth_markers": [{"marker": "refresh token"}]}))
        return 75

    monkeypatch.setattr(campaign, "supervise", supervise)
    assert campaign.watch(tmp_path, models=["fixture"]) == 64
    assert len(runs) == campaign.AUTH_RESTART_CAP
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "repair_required"


# ------------------------------------------------------------------------------------------------ checks and admission


def test_check_classification_requires_a_container_signin_for_codex():
    legacy = agent_auth.classify("codex-cli", {"status": "ok", "origin": "unknown-legacy", "volume": CODEX_SLOT_VOLUMES[1], "slot": 1})
    assert legacy["status"] == "legacy_origin" and not legacy["ok"] and "--slot 1 --login --replace" in legacy["repair"]
    imported = agent_auth.classify("codex-cli", {"status": "ok", "origin": "host-copy", "volume": CODEX_SLOT_VOLUMES[0], "slot": 0})
    assert imported["ok"] and "host" in imported["warnings"][0]
    assert agent_auth.classify("codex-cli", {"status": "ok", "origin": "container-device-auth", "volume": CODEX_SLOT_VOLUMES[0], "slot": 0})["ok"]
    assert agent_auth.classify("claude-code-cli", {"status": "ok", "origin": "unknown-legacy", "volume": auth.AUTH_VOLUMES["claude-code-cli"]})["ok"]


@pytest.mark.parametrize("statuses,error", [(["ok", "credential_invalid"], AuthenticationRequired), (["network", "model_missing"], RuntimeEnvironmentError),
                                            (["rate_limited", "network", "timeout"], InfrastructureError), (["network", "legacy_origin", "model_missing"], AuthenticationRequired),
                                            (["ok", "ok"], None)])
def test_check_failures_map_to_repair_inspect_or_retry(statuses, error):
    reports = [agent_auth.classify("claude-code-cli", {"status": status, "volume": f"volume-{index}", "backend": "claude-code-cli"}) | {"backend": "claude-code-cli"}
               for index, status in enumerate(statuses)]
    if error is None:
        agent_auth.raise_for(reports)
        return
    with pytest.raises(error) as raised:
        agent_auth.raise_for(reports)
    assert type(raised.value) is error


class Engine:
    LABEL = "io.decompbench.runtime"

    def __init__(self, tmp_path, missing=(), down=False, users=None):
        self.seccomp_profile = tmp_path / "profile.json"
        self.removed, self.missing, self.down, self.users = [], set(missing), down, users

    def cli(self, *args):
        return ["docker", *args]

    def command(self, *args, timeout=30):
        from src.agent.container import ContainerError
        if self.down:
            raise ContainerError("Docker command could not complete")
        assert args[:2] == ("volume", "inspect")
        if args[2] in self.missing:
            return subprocess.CompletedProcess(args, 1, "", "Error response from daemon: get x: no such volume")
        labels = {"io.decompbench.purpose": "agy-native-auth"} if "agy" in args[2] else {self.LABEL: ("codex-cli" if "codex" in args[2] else "claude-code-cli") + "-v2-auth"}
        return subprocess.CompletedProcess(args, 0, json.dumps(labels), "")

    def volume_users(self, volume):
        return list(self.users or [])

    def remove_container(self, name):
        self.removed.append((name, held_volumes()))
        if self.users:
            self.users = [user for user in self.users if user[0] != name]


def test_helper_runs_under_the_volume_lock_until_its_container_is_gone(tmp_path, monkeypatch):
    engine, seen = Engine(tmp_path), []

    def run(arguments, **kwargs):
        seen.append((arguments, held_volumes(), json.loads(kwargs["input"])))
        return subprocess.CompletedProcess(arguments, 0, "noise\n" + json.dumps({"status": "ok", "origin": "container-device-auth"}) + "\n", "")

    monkeypatch.setattr(agent_auth.subprocess, "run", run)
    reports = agent_auth.check_backend(engine, "sha256:image", "codex-cli", models=["judge-model"])
    assert [report["status"] for report in reports] == ["ok", "ok"] and [report["slot"] for report in reports] == [0, 1]
    assert [held for _, held, _ in seen] == [{CODEX_SLOT_VOLUMES[0]}, {CODEX_SLOT_VOLUMES[1]}] and seen[0][2]["models"] == ["judge-model"]
    assert [held for _, held in engine.removed] == [{CODEX_SLOT_VOLUMES[0]}, {CODEX_SLOT_VOLUMES[1]}] and held_volumes() == set()
    arguments = seen[0][0]
    mounts = [arguments[index + 1] for index, item in enumerate(arguments) if item == "--mount"]
    assert mounts == [f"type=volume,src={CODEX_SLOT_VOLUMES[0]},dst=/auth", f"type=bind,src={agent_auth.HELPER},dst=/opt/login/auth_helper.py,readonly"]
    assert "--read-only" in arguments and arguments[arguments.index("--network") + 1] == "bridge" and "--privileged" not in arguments

    def timeout(arguments, **kwargs):
        raise subprocess.TimeoutExpired(arguments, 1)

    monkeypatch.setattr(agent_auth.subprocess, "run", timeout)
    engine.removed.clear()
    report = agent_auth.check_backend(engine, "sha256:image", "claude-code-cli")[0]
    assert report["status"] == "timeout" and engine.removed[0][1] == {auth.AUTH_VOLUMES["claude-code-cli"]} and held_volumes() == set()


def test_admission_checks_only_slot_zero_in_single_volume_mode_and_records_reports(tmp_path, monkeypatch):
    engine = Engine(tmp_path, missing={CODEX_SLOT_VOLUMES[1]})
    monkeypatch.setattr(agent_auth.subprocess, "run", lambda arguments, **kwargs: subprocess.CompletedProcess(arguments, 0, json.dumps({"status": "ok", "origin": "container-device-auth"}), ""))
    summary = agent_auth.require_backend_authentication(engine, "sha256:image", ["codex-cli", "codex-cli"], models={}, codex_slots=1, agy_passphrase=None, record_dir=tmp_path / "records")
    assert list(summary) == [CODEX_SLOT_VOLUMES[0]]
    with pytest.raises(AuthenticationRequired, match="credential_missing"):
        agent_auth.require_backend_authentication(engine, "sha256:image", ["codex-cli"], models={}, codex_slots=2, agy_passphrase=None, record_dir=tmp_path / "records")
    with pytest.raises(AuthenticationRequired, match="passphrase"):
        agent_auth.require_backend_authentication(engine, "sha256:image", ["agy-cli"], models={}, codex_slots=2, agy_passphrase=None, record_dir=tmp_path / "records")
    assert len(list((tmp_path / "records").glob("*.json"))) == 3


def test_agy_helper_uses_the_keyring_session_and_its_login_module(tmp_path):
    arguments = agent_auth.helper_arguments(Engine(tmp_path), "sha256:image", backend="agy-cli", volume=auth.AUTH_VOLUMES["agy-cli"], mode="agy-check", name="fixture", network="bridge")
    assert arguments[arguments.index("--entrypoint") + 1] == "dbus-run-session" and f"type=volume,src={auth.AUTH_VOLUMES['agy-cli']},dst=/auth/keyrings" in arguments
    assert f"type=bind,src={agent_auth.AGY_LOGIN},dst=/opt/login/agy_login.py,readonly" in arguments


# ------------------------------------------------------------------------------------------------ runtime evidence


def test_runtime_and_readiness_evidence_must_cover_the_auth_policy(tmp_path):
    from src.experiments import docker_setup
    prepared = {"agent_policy": docker_client.DOCKER_EDIT_POLICY, "cli_versions": CLI_VERSIONS, "worker_files_sha256": docker_client.worker_file_hashes(), "auth_volumes": auth.AUTH_VOLUMES}
    with pytest.raises(RuntimeEnvironmentError, match="other authentication volumes"):
        docker_setup.validate_execution_environment(prepared)
    with pytest.raises(RuntimeEnvironmentError, match="predates the authentication policy"):
        docker_setup.validate_execution_environment(prepared | {"codex_slot_volumes": list(CODEX_SLOT_VOLUMES)})


# ------------------------------------------------------------------------------------------------ review round 1


def test_lock_holders_wait_for_remove_or_refuse_containers_that_outlived_their_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(auth.time, "sleep", lambda _: None)
    volume = CODEX_SLOT_VOLUMES[1]
    engine = Engine(tmp_path, users=[("decompgym-auth-orphan", auth.HELPER_LABEL)])
    with volume_lock(volume):
        auth.require_exclusive_volume(engine, volume, wait=0)
    assert [name for name, _ in engine.removed] == ["decompgym-auth-orphan"]
    engine = Engine(tmp_path, users=[("decompbench-v2-orphan-worker", "v2-worker"), ("recovered", "v2-worker")])
    with volume_lock(volume), pytest.raises(RuntimeEnvironmentError, match="decompbench-v2-orphan-worker"):
        auth.require_exclusive_volume(engine, volume, allowed=("recovered",), wait=0)
    assert engine.removed == []  # a worker's records and token checkpoint are never removed here
    engine = Engine(tmp_path, users=[("recovered", "v2-worker")])
    auth.require_exclusive_volume(engine, volume, allowed=("recovered",), wait=0)


def test_an_existing_call_state_is_locked_on_the_volume_its_container_mounted(tmp_path):
    client, runtime, target = client_for(tmp_path)
    state = tmp_path / "call/agent_container.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"name": "earlier", "mounts": [{"Type": "volume", "Name": CODEX_SLOT_VOLUMES[1], "Destination": "/auth", "RW": True}]}))
    with client.auth_lock({CODEX_SLOT_VOLUMES[0]: "/auth"}, state) as volume:
        assert volume == CODEX_SLOT_VOLUMES[1] and held_volumes() == {CODEX_SLOT_VOLUMES[1]}
    with client.auth_lock({CODEX_SLOT_VOLUMES[0]: "/auth"}, tmp_path / "absent.json") as volume:
        assert volume == CODEX_SLOT_VOLUMES[0]


def test_a_recovered_auth_error_before_an_unrelated_failure_is_not_terminal():
    events = event(type="error", message=REUSED) + event(type="turn.failed", error={"message": "stream disconnected before completion"})
    assert terminal_auth_marker("codex-cli", events, "", succeeded=False) is None
    assert terminal_auth_marker("codex-cli", event(type="turn.failed", error={"message": "stream disconnected"}), "ERROR " + REUSED, succeeded=False) is None
    events = event(type="assistant", error="authentication_failed", message={"content": [{"type": "text", "text": "x"}]})
    assert terminal_auth_marker("claude-code-cli", events, "", succeeded=False) == "authentication_failed"
    recovered = events + event(type="assistant", message={"content": [{"type": "text", "text": "continuing"}]}) + event(type="result", subtype="error_max_turns", is_error=True, result="")
    assert terminal_auth_marker("claude-code-cli", recovered, "Please run /login", succeeded=False) is None
    # A Codex turn that completed after an auth error and was then interrupted by a container limit is no auth failure.
    completed = event(type="error", message=REUSED) + event(type="turn.completed", usage={})
    assert terminal_auth_marker("codex-cli", completed, "ERROR " + REUSED, succeeded=False) is None


@pytest.mark.parametrize("backend,credential", [("codex-cli", {"auth_mode": "chatgpt", "tokens": {}}), ("claude-code-cli", {"oauth_token": "sk-ant-oat01-x"}),
                                                ("claude-code-cli", {"credentials": {"claudeAiOauth": {}}})])
def test_a_job_cannot_carry_a_codex_or_claude_credential(tmp_path, backend, credential):
    client, runtime, target = client_for(tmp_path, backend)
    client.credential_provider = lambda _: credential
    with pytest.raises(RuntimeEnvironmentError, match="may not carry a credential"):
        client.edit_file(target, "Solve.", log_dir=tmp_path / "call")
    assert runtime.jobs == []


def test_docker_outage_and_foreign_volumes_are_not_credential_repairs(tmp_path):
    report = agent_auth.check_backend(Engine(tmp_path, down=True), "sha256:image", "claude-code-cli")[0]
    assert report["status"] == "docker_unavailable"
    with pytest.raises(InfrastructureError):
        agent_auth.raise_for([report])
    engine = Engine(tmp_path)
    engine.command = lambda *args, timeout=30: subprocess.CompletedProcess(args, 0, json.dumps({"other": "label"}), "")
    report = agent_auth.check_backend(engine, "sha256:image", "claude-code-cli")[0]
    assert report["status"] == "volume_foreign"
    with pytest.raises(RuntimeEnvironmentError) as raised:
        agent_auth.raise_for([report])
    assert type(raised.value) is RuntimeEnvironmentError


@pytest.mark.parametrize("stored,code", [("2026-11-07T09:00:00+00:00", 64), (None, 64), ("2026-11-07T11:00:00+00:00", 0)])
def test_a_failed_claude_token_is_cleared_only_by_a_token_stored_after_the_failure(tmp_path, monkeypatch, stored, code):
    volume = auth.AUTH_VOLUMES["claude-code-cli"]
    root, _ = campaign_fixture(tmp_path, monkeypatch, admission=lambda *args, **kwargs: {volume: {"backend": "claude-code-cli", "status": "ok", "created_at": stored},
                                                                                           CODEX_SLOT_VOLUMES[0]: {"backend": "codex-cli", "status": "ok", "created_at": None}})
    failed = "2026-11-07T10:00:00+00:00"  # a token stored before the failure (for example a login between two calls) does not clear it
    (root / "auth_invalid.json").write_text(json.dumps({"recorded_at": failed, "error": "x", "volumes": {volume: {"backend": "claude-code-cli", "failed_at": failed},
                                                                                                       "volume-outside-scope": {"backend": "agy-cli", "failed_at": failed}}}))
    assert campaign.run(root) == code
    sentinel = json.loads((root / "auth_invalid.json").read_text())
    if code:
        assert json.loads((root / "status.json").read_text())["state"] == "repair_required" and volume in sentinel["volumes"]
    else:  # the replaced token's entry is cleared; a volume this launch did not check stays recorded
        assert set(sentinel["volumes"]) == {"volume-outside-scope"}


def test_a_per_call_failure_records_the_generation_of_its_volume(tmp_path, monkeypatch):
    volume = CODEX_SLOT_VOLUMES[1]

    def runner(name):
        raise AuthenticationRequired("invalid", {"kind": "credential_invalid", "backend": "codex-cli", "auth_volume": volume})

    root, _ = campaign_fixture(tmp_path, monkeypatch, runner=runner, admission=lambda *args, **kwargs: {volume: {"backend": "codex-cli", "status": "ok", "generation": "g1"}})
    assert campaign.run(root) == 64
    entry = json.loads((root / "auth_invalid.json").read_text())["volumes"][volume]
    assert entry["backend"] == "codex-cli" and entry["generation"] == "g1" and entry["failed_at"]


def test_the_sentinel_keeps_the_time_the_failure_was_observed(tmp_path):
    error = AuthenticationRequired("invalid", {"backend": "claude-code-cli", "auth_volume": "volume-a", "recorded_at": "2026-11-07T10:00:00+00:00"})
    campaign.record_auth_sentinel(tmp_path, error)
    later = AuthenticationRequired("invalid", {"backend": "claude-code-cli", "auth_volume": "volume-b"})
    later.raised_at = "2026-11-07T10:05:00+00:00"
    campaign.record_auth_sentinel(tmp_path, later)  # written after draining; the failure times stay those observed
    volumes = json.loads((tmp_path / "auth_invalid.json").read_text())["volumes"]
    assert volumes["volume-a"]["failed_at"] == "2026-11-07T10:00:00+00:00" and volumes["volume-b"]["failed_at"] == "2026-11-07T10:05:00+00:00"


def test_every_auth_failure_seen_while_draining_reaches_the_sentinel(tmp_path, monkeypatch):
    failed = threading.Event()
    volume = auth.AUTH_VOLUMES["claude-code-cli"]

    def runner(name):
        if name == "case-0":
            failed.set()
            raise RuntimeEnvironmentError("sandbox changed")
        assert failed.wait(5)
        time.sleep(0.3)
        raise AuthenticationRequired("claude credential invalid", {"backend": "claude-code-cli", "auth_volume": volume})

    root, _ = campaign_fixture(tmp_path, monkeypatch, cases=2, runner=runner)
    assert campaign.run(root) == 64
    assert json.loads((root / "status.json").read_text())["error"].startswith("RuntimeEnvironmentError")
    assert volume in json.loads((root / "auth_invalid.json").read_text())["volumes"]


def test_lock_waits_are_bounded_and_cancellable():
    volume = CODEX_SLOT_VOLUMES[0]
    holding, release = threading.Event(), threading.Event()

    def holder():
        with volume_lock(volume):
            holding.set()
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(holder)
        assert holding.wait(5)
        try:
            started = time.monotonic()
            with pytest.raises(InfrastructureError, match="stayed locked"):
                with volume_lock(volume, timeout=0.3):
                    pass
            auth.CANCEL.set()
            with pytest.raises(InfrastructureError, match="stop requested"):
                with volume_lock(volume, timeout=60):
                    pass
            assert time.monotonic() - started < 10
        finally:
            auth.CANCEL.clear()
            release.set()
        future.result()
    assert held_volumes() == set()


def test_the_client_guard_refuses_a_surviving_writer_and_exempts_the_recovered_container(tmp_path):
    client, runtime, target = client_for(tmp_path, users=[("decompbench-v2-orphan", "v2-worker")])
    import src.agent.auth as module
    original, module.ORPHAN_WAIT_SECONDS = module.ORPHAN_WAIT_SECONDS, 0
    try:
        with pytest.raises(RuntimeEnvironmentError, match="decompbench-v2-orphan"):
            client.edit_file(target, "Solve.", log_dir=tmp_path / "one")
        assert runtime.jobs == [] and runtime.probed and all(probe == (CODEX_SLOT_VOLUMES[0], {CODEX_SLOT_VOLUMES[0]}) for probe in runtime.probed)  # probed under the lock
        # An existing state of a slot-1 container: edit_file locks slot 1, and that container is not an orphan.
        runtime.users = [("earlier", "v2-worker")]
        state = tmp_path / "two/agent_container.json"
        state.parent.mkdir(parents=True)
        state.write_text(json.dumps({"name": "earlier", "mounts": [{"Type": "volume", "Name": CODEX_SLOT_VOLUMES[1], "Destination": "/auth", "RW": True}]}))
        assert client.edit_file(target, "Solve.", log_dir=tmp_path / "two").ok
        assert runtime.held[-1] == {CODEX_SLOT_VOLUMES[1]}
    finally:
        module.ORPHAN_WAIT_SECONDS = original
