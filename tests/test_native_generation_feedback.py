"""Restore missing harness feedback without changing candidate or reflection commits."""
from contextlib import nullcontext
from dataclasses import asdict
import json

import pytest

from src.agent.result import AgentResult
from src.experiments.paired_runner import InfrastructureError, PairedRunner, ProtocolError, digest, read, save
from host_solver import host_solver


class Client:
    def __init__(self):
        self.calls = []
        self.fail_reflection = False

    def protocol(self):
        return {"backend": "test", "model": "fixture"}

    def edit_file(self, target, prompt, *, context, log_dir, step):
        self.calls.append({"step": step, "prompt": prompt, "context": [(path.name, path.read_text()) for path in context if path.exists()]})
        if step == "agent":
            result = AgentResult(False, 0, 0.01, {}, "Required solver artifact was not delivered", terminal_state="completed")
        elif step == "reflection" and self.fail_reflection:
            self.fail_reflection = False
            result = AgentResult(False, 1, 0.01, {}, "controlled reflection transport failure")
        elif step == "reflection":
            target.write_text("The previous generation did not deliver a solver.")
            result = AgentResult(True, 0, 0.01, {}, terminal_state="completed", artifact_sha256=digest(target))
        else:
            raise AssertionError("A generation failure must not be judged")
        save(log_dir / f"{step}_metrics.json", asdict(result))
        return result


@pytest.fixture
def failed_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr("src.experiments.paired_runner.runtime_lock", lambda name: nullcontext())
    source = tmp_path / "data/bd/ufl/ufl_small_seed000"
    source.mkdir(parents=True)
    save(source / "data.json", {"m": 1, "n": 1, "f": [1], "D": [[1]]})
    save(source / "meta.json", {"ground_truth_objective": 2})
    (source / "prompt.txt").write_text("Implement Benders decomposition.")
    client = Client()
    task = PairedRunner(client=client, judge_client=client, solver_executor=host_solver, model="fixture", suite="bd", problem=["ufl"])
    destination = task.prepare(source, tmp_path / "runs", "reflection")
    attempt = destination / "attempts/attempt_01"
    task.mode = "reflection"
    def forbid_execution(*args):
        raise AssertionError("No solver execution is allowed for this failed generation")
    monkeypatch.setattr(task, "run_solver", forbid_execution)
    result = task.run_attempt(1, attempt, "ufl", 2)
    assert result.status == "agent_failed"
    return task, client, attempt, result


def committed_hashes(attempt):
    return {name: digest(attempt / name) for name in ["solver.py", "score_receipt.json", "agent_receipt.json"]}


def test_cached_failure_gets_truthful_feedback_without_regeneration_or_rescoring(failed_attempt):
    task, client, attempt, _ = failed_attempt
    before = committed_hashes(attempt)
    result = task.run_attempt(1, attempt, "ufl", 2)
    reflected = task.ensure_reflection(attempt, result)
    text = (attempt / "exec.log").read_text()
    assert "[framework generation diagnostic]" in text
    assert "Solver execution was not performed" in text and "Algorithm judgment was not performed" in text
    assert "Required solver artifact was not delivered" in text
    assert reflected.status == "agent_failed" and not reflected.exec_ok
    assert [call["step"] for call in client.calls] == ["agent", "reflection"]
    assert "exec.log" in dict(client.calls[-1]["context"])
    assert not (attempt / "alg_verifier.md").exists()
    assert committed_hashes(attempt) == before
    provenance = read(attempt / "generation_failure_context.json")
    assert provenance["exec_log_sha256"] == digest(attempt / "exec.log")
    assert provenance["score_receipt_sha256"] == before["score_receipt.json"]


def test_feedback_is_stable_across_reflection_failure_and_resume(failed_attempt):
    task, client, attempt, result = failed_attempt
    client.fail_reflection = True
    with pytest.raises(InfrastructureError):
        task.ensure_reflection(attempt, result)
    hashes = {name: digest(attempt / name) for name in ["exec.log", "generation_failure_context.json"]}
    task.ensure_reflection(attempt, task.run_attempt(1, attempt, "ufl", 2))
    requests = [read(path) for path in (attempt / "calls").glob("*/request.json") if read(path)["step"] == "reflection"]
    assert len(requests) == 2 and len({request["request_sha256"] for request in requests}) == 1
    assert hashes == {name: digest(attempt / name) for name in hashes}
    assert sum(call["step"] == "agent" for call in client.calls) == 1


def test_previous_failed_context_is_retained_and_new_context_has_new_fingerprint(failed_attempt, monkeypatch):
    task, client, attempt, result = failed_attempt
    client.fail_reflection = True
    with monkeypatch.context() as patch:
        patch.setattr(task, "ensure_generation_failure_log", lambda *args: None)
        with pytest.raises(InfrastructureError):
            task.ensure_reflection(attempt, result)
    old_request = next(path for path in (attempt / "calls").glob("*/request.json") if read(path)["step"] == "reflection")
    old_hash, old_fingerprint = digest(old_request), read(old_request)["request_sha256"]
    task.ensure_reflection(attempt, result)
    assert digest(old_request) == old_hash
    assert read(attempt / "reflection_receipt.json")["request_sha256"] != old_fingerprint
    calls = [call for call in client.calls if call["step"] == "reflection"]
    assert calls[0]["prompt"] == calls[1]["prompt"]


@pytest.mark.parametrize("missing_receipt", [False, True])
def test_successful_original_context_reflections_are_reused_without_new_feedback(failed_attempt, monkeypatch, missing_receipt):
    task, client, attempt, result = failed_attempt
    with monkeypatch.context() as patch:
        patch.setattr(task, "ensure_generation_failure_log", lambda *args: None)
        task.ensure_reflection(attempt, result)
    receipt = read(attempt / "reflection_receipt.json")
    if missing_receipt:
        (attempt / "reflection_receipt.json").unlink()
    task.ensure_reflection(attempt, result)
    assert read(attempt / "reflection_receipt.json") == receipt
    assert [call["step"] for call in client.calls] == ["agent", "reflection"]
    assert not (attempt / "exec.log").exists() and not (attempt / "generation_failure_context.json").exists()


def test_recoverable_reflection_with_changed_artifact_remains_integrity_error(failed_attempt, monkeypatch):
    task, client, attempt, result = failed_attempt
    with monkeypatch.context() as patch:
        patch.setattr(task, "ensure_generation_failure_log", lambda *args: None)
        task.ensure_reflection(attempt, result)
    (attempt / "reflection_receipt.json").unlink()
    (attempt / "debug.md").write_text("changed")
    with pytest.raises(ProtocolError, match="committed artifact"):
        task.ensure_reflection(attempt, result)
    assert not (attempt / "exec.log").exists()


@pytest.mark.parametrize("filename", ["solver.py", "execution.json", "alg_verifier.md"])
def test_inconsistent_saved_state_is_not_turned_into_failure_feedback(failed_attempt, filename):
    task, client, attempt, result = failed_attempt
    (attempt / filename).write_text("changed")
    with pytest.raises(ProtocolError):
        task.ensure_reflection(attempt, result)
    assert not (attempt / "exec.log").exists()


def test_existing_execution_log_is_not_overwritten(failed_attempt):
    task, _, attempt, result = failed_attempt
    (attempt / "exec.log").write_text("Preserved existing feedback.")
    task.ensure_generation_failure_log(attempt, result)
    assert (attempt / "exec.log").read_text() == "Preserved existing feedback."
    assert not (attempt / "generation_failure_context.json").exists()


@pytest.mark.parametrize("filename", ["exec.log", "generation_failure_context.json"])
def test_uncommitted_diagnostic_tampering_is_rejected(failed_attempt, filename):
    task, _, attempt, result = failed_attempt
    task.ensure_generation_failure_log(attempt, result)
    (attempt / filename).write_text('{}' if filename.endswith('.json') else 'changed')
    with pytest.raises(ProtocolError):
        task.ensure_generation_failure_log(attempt, result)


def test_interrupted_diagnostic_creation_finishes_from_saved_provenance(failed_attempt):
    task, _, attempt, result = failed_attempt
    task.ensure_generation_failure_log(attempt, result)
    expected = (attempt / "exec.log").read_text()
    (attempt / "exec.log").unlink()
    task.ensure_generation_failure_log(attempt, result)
    assert (attempt / "exec.log").read_text() == expected
