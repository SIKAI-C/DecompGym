"""Concurrent Codex calls share two slots and fall back to one after an authentication failure."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from src.agent.result import AgentResult
from src.experiments.call_gate import NativeCallGate, auth_failure_marker


def gate_fixture(tmp_path):
    state, phase = tmp_path / "state", tmp_path / "phase"
    state.mkdir(); phase.mkdir()
    return NativeCallGate(state, phase)


def test_two_calls_overlap_third_waits_and_exception_releases_slot(tmp_path):
    gate = gate_fixture(tmp_path)
    entered = [threading.Event() for _ in range(3)]
    release = threading.Event()
    def call(index):
        with gate.enter():
            entered[index].set()
            if index < 2:
                assert release.wait(3)
            if index == 0:
                raise RuntimeError("controlled client failure")
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(call, 0); second = pool.submit(call, 1)
        try:
            assert entered[0].wait(2) and entered[1].wait(2)
            third = pool.submit(call, 2)
            assert not entered[2].wait(0.1)
        finally:
            release.set()
        with pytest.raises(RuntimeError):first.result()
        second.result(); third.result()
    assert gate.active == 0


def test_auth_fallback_drains_existing_calls_then_remains_serial(tmp_path):
    gate = gate_fixture(tmp_path)
    entered = [threading.Event() for _ in range(3)]
    release = [threading.Event(), threading.Event()]
    def call(index):
        with gate.enter():
            entered[index].set()
            if index < 2:assert release[index].wait(3)
    with ThreadPoolExecutor(max_workers=3) as pool:
        first, second = pool.submit(call, 0), pool.submit(call, 1)
        try:
            assert entered[0].wait(2) and entered[1].wait(2)
            gate.fallback("refresh token", role="algorithm_judge", invocation_id="fixture")
            third = pool.submit(call, 2)
            release[0].set();first.result()
            assert not entered[2].wait(0.1)
        finally:
            release[0].set();release[1].set()
        second.result();third.result()
    assert gate.limit == 1
    assert NativeCallGate(gate.state_directory, gate.phase_directory).limit == 1


@pytest.mark.parametrize("message", ["401 Unauthorized", "refresh_token_reused", "Your access token has expired", "authentication required", "failed to refresh token: rejected", "failed to refresh access token: request timed out"])
def test_auth_failure_detection(message):
    assert auth_failure_marker(AgentResult(False, 1, 0, {}, message), "")
    assert auth_failure_marker(AgentResult(True, 0, 0, {}), message)
    assert auth_failure_marker(AgentResult(True, 0, 0, {}), "") is None


def test_model_catalog_refresh_timeout_is_not_an_authentication_failure():
    diagnostics = "2026-10-08T16:16:36.869718Z ERROR codex_models_manager::manager: failed to refresh available models: request timed out"
    result = AgentResult(True, 0, 111.6, {}, terminal_state="completed")
    assert auth_failure_marker(result, diagnostics) is None
    assert auth_failure_marker(result, diagnostics + "\nERROR native auth: refresh_token_reused") == "refresh_token_reused"

