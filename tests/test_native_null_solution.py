"""Candidate validation must resume cached work without hiding verifier defects."""
from contextlib import nullcontext
from dataclasses import asdict
import json

import pytest

from src.agent.codex_client import CodexResult
from src.agent.output_verifier import OutputVerifier
from src.agent.paired_runner import InfrastructureError, PairedRunner, ProtocolError, digest, read, save


CASES = {
    "csp": ("cg", {"roll_width": 2, "item_types": [{"id": 0, "width": 2, "demand": 1}]}, {"objective_value": 1, "solution": {"patterns": [{"usage": 1, "cuts": [{"item_id": 0, "count": 1}]}]}}),
    "gap": ("cg", {"machines": [{"id": 0, "capacity": 1}], "jobs": [{"id": 0, "assignments": [{"machine_id": 0, "resource": 1, "cost": 2}]}]}, {"objective_value": 2, "solution": {"assignments": [{"job_id": 0, "machine_id": 0}]}}),
    "shift": ("cg", {"demand": [1], "shift_min_length": 1, "shift_max_length": 1, "shift_cost": 1}, {"objective_value": 1, "solution": {"shifts": [{"start": 0, "end": 0, "count": 1}]}}),
    "ufl": ("bd", {"m": 2, "n": 1, "f": [2, 3], "D": [[4], [1]]}, {"y": [0, 1], "x": [[0], [1]], "objective_value": 4, "solution": None}),
}


class Client:
    def __init__(self, output):
        self.output = output
        self.calls = []
        self.fail_judge_once = False

    def protocol(self):
        return {"backend": "test", "model": "fixture"}

    def edit_file(self, target, prompt, *, context, log_dir, step):
        self.calls.append(step)
        if step == "agent":
            target.write_text("from pathlib import Path\nPath('output.json').write_text(" + repr(json.dumps(self.output)) + ")\n")
        elif step == "llm_verifier":
            if self.fail_judge_once:
                self.fail_judge_once = False
                return CodexResult(False, 1, 0, {}, "controlled judge transport failure")
            target.write_text("VERDICT: PASS\nNOTES: fixture")
        else:
            target.write_text("Reflection fixture.")
        result = CodexResult(True, 0, 0.01, {}, terminal_state="completed", artifact_sha256=digest(target))
        save(log_dir / f"{step}_metrics.json", asdict(result))
        return result


@pytest.fixture(autouse=True)
def no_native_auth_lock(monkeypatch):
    monkeypatch.setattr("src.agent.paired_runner.runtime_lock", lambda name: nullcontext())


def prepare(tmp_path, problem, output=None):
    suite, data, valid = CASES[problem]
    source = tmp_path / "data" / suite / problem / f"{problem}_small_seed000"
    source.mkdir(parents=True)
    save(source / "data.json", data)
    save(source / "meta.json", {"ground_truth_objective": valid["objective_value"]})
    (source / "prompt.txt").write_text("Implement the requested decomposition.")
    client = Client(valid if output is None else output)
    task = PairedRunner(client=client, judge_client=client, model="fixture", suite=suite, problem=[problem])
    destination = task.prepare(source, tmp_path / "runs", "plain")
    attempt = destination / "attempts/attempt_01"
    attempt.mkdir()
    return task, client, destination, attempt, valid["objective_value"]


def cache_generation_and_execution(task, destination, attempt):
    task.edit(task.client, attempt / "solver.py", task.generation_prompt(destination), [destination / "data.json"], "agent")
    ok, stdout, stderr, output = task.run_solver(attempt, 5)
    save(attempt / "execution.json", {"exec_ok": ok, "stdout": stdout, "stderr": stderr, "output": output, "execution_seconds": 0.01, "solver_sha256": digest(attempt / "solver.py"), "output_sha256": digest(attempt / "output.json")})
    return {name: digest(attempt / name) for name in ["solver.py", "output.json", "execution.json", "agent_receipt.json"]}


@pytest.mark.parametrize("problem", ["csp", "gap", "shift"])
@pytest.mark.parametrize("claimed_objective", [None, 1])
def test_explicit_null_solution_is_scored_and_still_judged(tmp_path, problem, claimed_objective):
    task, client, _, attempt, gold = prepare(tmp_path, problem, {"status": "infeasible", "objective_value": claimed_objective, "solution": None})
    result = task.run_attempt(1, attempt, problem, gold)
    assert result.status == ("objective_not_available" if claimed_objective is None else "verify_failed")
    assert result.exec_ok and not result.feasibility_ok and not result.verify_ok
    assert result.llm_verdict == "pass"
    assert "solution is null; no candidate solution supplied" in result.issues
    assert client.calls == ["agent", "llm_verifier"]
    assert (attempt / "score_receipt.json").is_file()


@pytest.mark.parametrize("problem", list(CASES))
def test_valid_outputs_and_top_level_benders_null_field_keep_existing_behavior(tmp_path, problem):
    task, client, _, attempt, gold = prepare(tmp_path, problem)
    result = task.run_attempt(1, attempt, problem, gold)
    assert result.status == "ok" and result.feasibility_ok
    assert client.calls == ["agent", "llm_verifier"]


def test_null_recovery_reuses_generation_execution_and_score_across_judge_failure(tmp_path, monkeypatch):
    task, client, destination, attempt, gold = prepare(tmp_path, "csp", {"status": "infeasible", "objective_value": None, "solution": None})
    hashes = cache_generation_and_execution(task, destination, attempt)
    generation_id = read(attempt / "agent_receipt.json")["result"]["invocation_id"]
    def forbid_execution(*args):
        raise AssertionError("Committed solver must not execute again")
    monkeypatch.setattr(task, "run_solver", forbid_execution)
    client.fail_judge_once = True
    with pytest.raises(InfrastructureError, match="controlled judge transport failure"):
        task.run_attempt(1, attempt, "csp", gold)
    assert not (attempt / "score_receipt.json").exists()
    result = task.run_attempt(1, attempt, "csp", gold)
    assert result.status == "objective_not_available"
    assert result.metrics["generation"]["invocation_id"] == generation_id
    assert client.calls == ["agent", "llm_verifier", "llm_verifier"]
    monkeypatch.setattr(OutputVerifier, "extract_objective", lambda *args: (_ for _ in ()).throw(AssertionError("Scored attempt must not be reevaluated")))
    assert task.run_attempt(1, attempt, "csp", gold).status == result.status
    assert client.calls == ["agent", "llm_verifier", "llm_verifier"]
    assert all(digest(attempt / name) == sha for name, sha in hashes.items())


@pytest.mark.parametrize("filename", ["solver.py", "output.json"])
def test_null_recovery_rejects_modified_committed_artifacts(tmp_path, filename):
    task, client, destination, attempt, gold = prepare(tmp_path, "csp", {"objective_value": None, "solution": None})
    cache_generation_and_execution(task, destination, attempt)
    (attempt / filename).write_text("changed")
    with pytest.raises(ProtocolError):
        task.run_attempt(1, attempt, "csp", gold)
    assert client.calls == ["agent"] and not (attempt / "score_receipt.json").exists()


def test_unrelated_verifier_attribute_error_is_not_scored(tmp_path, monkeypatch):
    task, client, _, attempt, gold = prepare(tmp_path, "csp")
    def broken_verifier(*args):
        raise AttributeError("unrelated verifier defect")
    monkeypatch.setattr(OutputVerifier, "verify_output", broken_verifier)
    with pytest.raises(AttributeError, match="unrelated verifier defect"):
        task.run_attempt(1, attempt, "csp", gold)
    assert client.calls == ["agent"] and not (attempt / "score_receipt.json").exists()
