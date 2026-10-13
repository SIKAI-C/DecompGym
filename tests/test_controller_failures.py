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
