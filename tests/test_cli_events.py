"""Docker transport must preserve the existing backend completion semantics."""
import json

import pytest

from src.agent.cli_events import parse_events


def parse(backend, events, model="fixture", code=0):
    return parse_events(backend, "\n".join(json.dumps(event) for event in events), returncode=code, requested_model=model)


def test_codex_recovered_warning_preserves_success_usage_and_optional_model():
    result = parse("codex-cli", [{"type": "error", "message": "reconnecting"}, {"type": "turn.completed", "model": "actual-snapshot", "usage": {"input_tokens": 7, "output_tokens": 2, "cached_input_tokens": 4}}, {"type": "turn.completed", "usage": {"input_tokens": 3}}])
    assert result["terminal_state"] == "completed" and result["error"] is None
    assert result["observed_models"] == ["actual-snapshot"]
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 2, "cached_input_tokens": 4}
    assert result["warnings"] == ["reconnecting"]


def test_codex_final_failure_is_not_hidden_by_an_earlier_completion():
    result = parse("codex-cli", [{"type": "turn.completed"}, {"type": "turn.failed", "error": {"message": "quota"}}], code=1)
    assert result["terminal_state"] == "failed" and result["error"] == "quota"


@pytest.mark.parametrize("observed", [None, "wrong", "fixture", "fixture-snapshot"])
def test_claude_preserves_existing_served_model_check(observed):
    result = parse("claude-code-cli", [{"type": "assistant", "message": {"model": observed}}, {"type": "result", "subtype": "success", "usage": {"input_tokens": 10, "cache_read_input_tokens": 5}, "result": "done"}])
    assert result["terminal_state"] == ("completed" if observed in {"fixture", "fixture-snapshot"} else "infrastructure_failed")
    assert result["usage"] == {"input_tokens": 10, "cache_read_input_tokens": 5}


def test_agy_preserves_raw_token_names_and_terminal_evidence():
    result = parse("agy-cli", [{"event": "result", "result": {"status": "SUCCESS", "usage": {"cache_read_tokens": 4, "output_tokens": 2}, "response": "done"}}])
    assert result["terminal_state"] == "completed" and result["response"] == "done"
    assert result["usage"] == {"cache_read_tokens": 4, "output_tokens": 2}


@pytest.mark.parametrize("backend,event", [("codex-cli", {"type": "turn.completed"}), ("claude-code-cli", {"type": "result", "subtype": "success", "modelUsage": {"fixture": {}}}), ("agy-cli", {"event": "result", "result": {"status": "SUCCESS"}})])
def test_missing_usage_stays_explicitly_missing(backend, event):
    result = parse(backend, [None, [], True, event])
    assert result["terminal_state"] == "completed" and not result["usage_available"] and result["usage"] == {}
