"""Only explicitly classified infrastructure faults receive watchdog recovery."""
from contextlib import nullcontext

import pytest

from src import native_campaign as campaign
from src.agent.output_verifier import VerificationError
from src.agent.paired_runner import InfrastructureError, ProtocolError, read


@pytest.mark.parametrize("error,code", [(AttributeError("verifier bug"), 64), (VerificationError("broken validator"), 64), (ProtocolError("changed inputs"), 64), (InfrastructureError("provider unavailable"), 75)])
def test_controller_saves_unexpected_failure_and_stops_without_retry(tmp_path, monkeypatch, error, code):
    root = tmp_path / "campaign"
    (root / "controller/src").mkdir(parents=True)
    monkeypatch.setattr(campaign, "__file__", str(root / "controller/src/native_campaign.py"))
    case = {"suite": "bd", "problem": "ufl", "instance": "synthetic"}
    manifest = {"model_queue": [{"model": "fixture", "backend": "codex-cli", "effort": "medium"}], "instances": [case], "protocol_id": "fixture", "agent_timeout": 1, "judge_model": "fixture", "judge_effort": "high"}
    manifest["execution_environment"] = {"image_id": "sha256:fixture", "license_file": "fixture", "native_environment": {}}
    from src import hybrid_runtime
    from src.agent import native_solver
    from types import SimpleNamespace
    monkeypatch.setattr(hybrid_runtime, "runtime", lambda: SimpleNamespace(image_id=lambda name: name, bind_environment=lambda _: None))
    monkeypatch.setattr(native_solver, "NativeSandboxExecutor", lambda **kwargs: object())
    monkeypatch.setattr(campaign, "require_backend_readiness", lambda *_: {})
    monkeypatch.setattr(campaign, "validate", lambda _: manifest)
    monkeypatch.setattr(campaign, "ensure_execution_allowed", lambda *_: None)
    monkeypatch.setattr(campaign, "require_campaign_environment", lambda *_: None)
    monkeypatch.setattr(campaign, "pair_complete", lambda *_: False)
    monkeypatch.setattr(campaign, "state_root", lambda: tmp_path / "locks")
    monkeypatch.setattr(campaign, "runtime_lock", lambda *_: nullcontext())
    monkeypatch.setattr(campaign.signal, "signal", lambda *_: None)
    class FailedRunner:
        def __init__(self, **kwargs):
            pass
        def run_pair(self, *args):
            raise error
    monkeypatch.setattr(campaign, "PairedRunner", FailedRunner)
    assert campaign.run(root) == code
    receipt = read(next((root / "controller_failures").glob("*.json")))
    assert receipt["error_type"] == type(error).__name__
    assert receipt["retryable"] == (code == 75)
    assert str(error) in receipt["traceback"]
    assert read(root / "status.json")["state"] == ("infrastructure_stopped" if code == 75 else "repair_required")


@pytest.mark.parametrize("backend", ["codex-cli", "claude-code-cli", "agy-cli"])
def test_campaign_routes_every_role_through_docker_and_the_native_executor(tmp_path, monkeypatch, backend):
    from types import SimpleNamespace
    from src import hybrid_runtime
    from src.agent import native_solver

    root = tmp_path / "campaign"
    (root / "controller/src").mkdir(parents=True)
    monkeypatch.setattr(campaign, "__file__", str(root / "controller/src/native_campaign.py"))
    case = {"suite": "bd", "problem": "ufl", "instance": "synthetic"}
    environment = {"image_id": "sha256:fixture", "license_file": "fixture", "native_environment": {}}
    manifest = {"model_queue": [{"model": "tested-model", "backend": backend, "effort": "medium"}], "instances": [case], "protocol_id": "fixture", "agent_timeout": 900,
                "judge_model": "independent-judge", "judge_effort": "high", "execution_environment": environment}
    engine = SimpleNamespace(image_id=lambda name: name, bind_environment=lambda _: None)
    executor = object()
    monkeypatch.setattr(hybrid_runtime, "runtime", lambda: engine)
    monkeypatch.setattr(native_solver, "NativeSandboxExecutor", lambda **kwargs: executor)
    monkeypatch.setattr(campaign, "validate", lambda _: manifest)
    monkeypatch.setattr(campaign, "ensure_execution_allowed", lambda *_: None)
    monkeypatch.setattr(campaign, "require_campaign_environment", lambda *_: None)
    checked_backends = []
    monkeypatch.setattr(campaign, "require_backend_readiness", lambda env, backends: checked_backends.extend(backends) or {})
    monkeypatch.setattr(campaign, "pair_complete", lambda *_: False)
    monkeypatch.setattr(campaign, "state_root", lambda: tmp_path / "locks")
    monkeypatch.setattr(campaign, "runtime_lock", lambda *_: nullcontext())
    monkeypatch.setattr(campaign.signal, "signal", lambda *_: None)
    monkeypatch.setenv("DECOMPBENCH_AGY_KEYRING_PASSPHRASE", "synthetic-test-only")
    runners = []

    class ObservedRunner:
        def __init__(self, **kwargs):
            runners.append(kwargs)

        def run_pair(self, *args):
            return {"synthetic": True}

    monkeypatch.setattr(campaign, "PairedRunner", ObservedRunner)
    assert campaign.run(root) == 0
    assert len(runners) == 1 and runners[0]["solver_executor"] is executor
    for key, role, expected_backend, model in [("client", "generation", backend, "tested-model"), ("reflection_client", "reflection", backend, "tested-model"), ("judge_client", "algorithm_judge", "codex-cli", "independent-judge")]:
        client = runners[0][key]
        assert client.runtime is engine
        assert (client.protocol()["transport"], client.backend, client.role, client.model, client.image_id) == ("docker", expected_backend, role, model, environment["image_id"])
    assert set(checked_backends) == {backend, "codex-cli"}
