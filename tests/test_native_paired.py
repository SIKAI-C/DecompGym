"""Behavioral checks for shared trials, independent retries, and durable recovery."""
from dataclasses import asdict
from contextlib import nullcontext
import json
from pathlib import Path

import pytest

from src.agent.codex_client import CodexResult
from src.agent.paired_runner import InfrastructureError, PairedRunner, ProtocolError, ReflectionArtifactError, digest, read, save, solver_infrastructure_error


@pytest.fixture(autouse=True)
def independent_test_auth_lock(monkeypatch):
    # Fake clients must not contend with a live campaign's subscription login lock.
    monkeypatch.setattr("src.agent.paired_runner.runtime_lock", lambda name: nullcontext())


def source_case(tmp_path):
    source = tmp_path / "data/bd/ufl/ufl_small_seed000"
    source.mkdir(parents=True)
    (source / "data.json").write_text(json.dumps({"m": 2, "n": 1, "f": [2, 3], "D": [[4], [1]]}))
    (source / "prompt.txt").write_text("Implement Benders decomposition.")
    (source / "meta.json").write_text('{"ground_truth_objective":4}')
    return source


class Client:
    def __init__(self, outcomes=None, fail_step=None, bad_verdict=False):
        self.calls = []
        self.outcomes = iter(outcomes or [True] * 100)
        self.fail_step, self.bad_verdict = fail_step, bad_verdict

    def protocol(self):
        return {"backend": "test", "model": "fixture"}

    def edit_file(self, target, prompt, *, context, log_dir, step):
        self.calls.append((step, target, list(context)))
        if step == self.fail_step:
            self.fail_step = None
            return CodexResult(False, 1, 0, {}, "controlled transport failure")
        if step == "agent":
            success = next(self.outcomes)
            if success:
                target.write_text("import json\nfrom pathlib import Path\nPath('output.json').write_text(json.dumps({'status':'feasible','method':'benders_decomposition','y':[0,1],'x':[[0],[1]],'objective_value':4}))\n")
            else:
                target.write_text("raise RuntimeError('controlled solver failure')\n")
        elif step == "llm_verifier":
            target.write_text("MALFORMED" if self.bad_verdict else "VERDICT: PASS\nNOTES: fixture")
            self.bad_verdict = False
        else:
            target.write_text("Repair the preceding solver.")
        result = CodexResult(True, 0, 0.01, {"input_tokens": 7}, terminal_state="completed", artifact_sha256=digest(target))
        save(log_dir / f"{step}_metrics.json", asdict(result))
        return result


def runner(client, judge=None, **kwargs):
    return PairedRunner(client=client, judge_client=judge or client, model="fixture", suite="bd", problem=["ufl"], **kwargs)


def test_shared_success_has_one_physical_evaluation_and_two_padded_policies(tmp_path):
    client = Client()
    source = source_case(tmp_path)
    value = runner(client).run_pair(source, tmp_path / "runs")
    assert [step for step, *_ in client.calls] == ["agent", "llm_verifier"]
    for summary in value.values():
        assert summary["complete"] and len(summary["attempts"]) == 10
        assert [x["status"] for x in summary["attempts"]] == ["ok"] + ["skipped_previous_solved"] * 9
        assert all(x["first_success_attempt"] == 1 and x["metrics"] == {} for x in summary["attempts"][1:])
    assert value["plain"]["attempts"][0] == value["reflection"]["attempts"][0]
    runner(client).run_pair(source, tmp_path / "runs")
    assert len(client.calls) == 2


def test_shared_failure_branches_with_history_only_for_reflection(tmp_path):
    client = Client([False, True, True])
    value = runner(client).run_pair(source_case(tmp_path), tmp_path / "runs")
    generations = [call for call in client.calls if call[0] == "agent"]
    assert len(generations) == 3
    assert [p.name for p in generations[1][2]] == ["data.json"]
    assert [p.name for p in generations[2][2]] == ["data.json", "solver.py", "debug.md"]
    assert sum(step == "reflection" for step, *_ in client.calls) == 1
    assert not list((tmp_path / "runs/plain").rglob("debug.md"))
    assert all(s["attempts"][1]["status"] == "ok" for s in value.values())


def test_exhaustion_counts_nineteen_candidates_not_twenty(tmp_path):
    client = Client([False] * 19)
    value = runner(client).run_pair(source_case(tmp_path), tmp_path / "runs")
    assert sum(step == "agent" for step, *_ in client.calls) == 19
    assert sum(step == "llm_verifier" for step, *_ in client.calls) == 19
    assert sum(step == "reflection" for step, *_ in client.calls) == 10
    assert all(len(s["attempts"]) == 10 and not any(a["reused_from_previous"] for a in s["attempts"]) for s in value.values())
    ids = [a["metrics"]["generation"]["invocation_id"] for s in value.values() for a in s["attempts"]]
    assert len(set(ids)) == 19


@pytest.mark.parametrize("malformed", [False, True])
def test_judge_recovery_does_not_regenerate_or_rerun_solver(tmp_path, malformed, monkeypatch):
    client = Client()
    judge = Client(bad_verdict=malformed, fail_step=None if malformed else "llm_verifier")
    task = runner(client, judge)
    source = source_case(tmp_path)
    executions = []
    execute = task.run_solver
    monkeypatch.setattr(task, "run_solver", lambda *a: (executions.append(1), execute(*a))[1])
    with pytest.raises(InfrastructureError):
        task.run_pair(source, tmp_path / "runs")
    task.run_pair(source, tmp_path / "runs")
    assert len(executions) == 1
    assert len(client.calls) == 1 and len(judge.calls) == 2
    assert len(list((tmp_path / "runs/plain").rglob("request.json"))) == 3


def test_reflection_failure_keeps_scored_shared_candidate(tmp_path):
    client = Client([False, True, True], fail_step="reflection")
    source = source_case(tmp_path)
    task = runner(client)
    with pytest.raises(InfrastructureError):
        task.run_pair(source, tmp_path / "runs")
    assert read(tmp_path / "runs/plain/bd/ufl/ufl_small_seed000/attempts/attempt_01/score_receipt.json")
    result = task.run_pair(source, tmp_path / "runs")
    assert len([x for x in client.calls if x[0] == "agent"]) == 3
    assert len([x for x in client.calls if x[0] == "reflection"]) == 2
    assert result["reflection"]["attempts"][1]["status"] == "ok"


def test_generation_client_commit_survives_missing_controller_receipt(tmp_path, monkeypatch):
    client = Client()
    task = runner(client, Client(fail_step="llm_verifier"))
    source = source_case(tmp_path)
    with pytest.raises(InfrastructureError):
        task.run_pair(source, tmp_path / "runs")
    path = tmp_path / "runs/plain/bd/ufl/ufl_small_seed000/attempts/attempt_01/agent_receipt.json"
    path.unlink()
    task.run_pair(source, tmp_path / "runs")
    assert len(client.calls) == 1


def test_mutated_input_cannot_resume(tmp_path):
    source = source_case(tmp_path)
    task = runner(Client())
    task.run_pair(source, tmp_path / "runs")
    (source / "prompt.txt").write_text("Different protocol")
    with pytest.raises(ProtocolError):
        task.run_pair(source, tmp_path / "runs")


def test_mutated_scored_code_cannot_resume(tmp_path):
    source = source_case(tmp_path)
    task = runner(Client())
    task.run_pair(source, tmp_path / "runs")
    path = tmp_path / "runs/plain/bd/ufl/ufl_small_seed000/attempts/attempt_01/solver.py"
    path.write_text("changed")
    with pytest.raises(ProtocolError):
        task.run_pair(source, tmp_path / "runs")


def test_generated_wrong_data_path_is_scored_and_not_reexecuted(tmp_path, monkeypatch):
    client = Client([False, True, True])
    original_edit = client.edit_file
    def bad_path_once(target, prompt, **kwargs):
        result = original_edit(target, prompt, **kwargs)
        if kwargs["step"] == "agent" and len(client.calls) == 1:
            target.write_text("from pathlib import Path\nwith open(Path(__file__).parent.parent / 'data.json') as stream:\n    stream.read()\n")
            result.artifact_sha256 = digest(target)
            save(kwargs["log_dir"] / "agent_metrics.json", asdict(result))
        return result
    monkeypatch.setattr(client, "edit_file", bad_path_once)
    task = runner(client)
    executed = []
    original_run = task.run_solver
    monkeypatch.setattr(task, "run_solver", lambda *args: (executed.append(args[0]), original_run(*args))[1])
    source = source_case(tmp_path)
    results = task.run_pair(source, tmp_path / "runs")
    first = tmp_path / "runs/plain/bd/ufl/ufl_small_seed000/attempts/attempt_01"
    assert results["plain"]["attempts"][0]["status"] == "exec_failed"
    assert "attempts/data.json" in read(first / "execution.json")["stderr"]
    assert digest(first / "solver.py") == read(first / "agent_receipt.json")["result"]["artifact_sha256"]
    assert not list(first.glob("solver_infrastructure_*.json"))
    task.run_pair(source, tmp_path / "runs")
    assert executed.count(first) == 1
    assert len([call for call in client.calls if call[0] == "agent"]) == 3


@pytest.mark.parametrize("problem", ["launch", "missing_input", "changed_input", "changed_solver", "dependency", "license", "network"])
def test_missing_file_exemption_preserves_infrastructure_failures(tmp_path, problem):
    source = source_case(tmp_path)
    task = runner(Client())
    directory = task.prepare(source, tmp_path / "runs", "plain") / "attempts/attempt_01"
    directory.mkdir()
    solver = directory / "solver.py"
    solver.write_text("open('../data.json')\n")
    save(directory / "agent_receipt.json", {"result": {"artifact_sha256": digest(solver)}})
    stderr = f'Traceback (most recent call last):\n  File "{solver}", line 1, in <module>\nFileNotFoundError: [Errno 2] No such file or directory: \'../data.json\'\n'
    assert not solver_infrastructure_error(directory, stderr)
    if problem == "launch":
        stderr = "[Errno 2] No such file or directory: '/missing/python'"
    elif problem == "missing_input":
        (directory.parent.parent / "data.json").unlink()
    elif problem == "changed_input":
        (directory.parent.parent / "data.json").write_text("{}")
    elif problem == "changed_solver":
        solver.write_text("changed")
    elif problem == "dependency":
        stderr = stderr.replace(str(solver), "/dependency/library.py")
    elif problem == "license":
        stderr += "Gurobi license unavailable"
    else:
        stderr += "network is unreachable"
    assert solver_infrastructure_error(directory, stderr)


class MissingAgyReflection(Client):
    def __init__(self, omissions, diagnostics=""):
        super().__init__([False, True, True])
        self.omissions = omissions
        self.diagnostics = diagnostics

    def protocol(self):
        return {"backend": "agy-cli", "model": "fixture"}

    def edit_file(self, target, prompt, *, context, log_dir, step):
        if step == "reflection" and self.omissions:
            self.omissions -= 1
            self.calls.append((step, target, list(context)))
            result = CodexResult(False, 0, 0.01, {}, "AGY did not complete an edit (exit 0); see reflection_stderr.log", terminal_state="completed")
            save(log_dir / "reflection_metrics.json", asdict(result))
            (log_dir / "reflection_stderr.log").write_text(self.diagnostics)
            return result
        return super().edit_file(target, prompt, context=context, log_dir=log_dir, step=step)


def test_reflection_omission_retries_identical_request_without_repeating_scoring(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr("src.agent.paired_runner.time.sleep", waits.append)
    client = MissingAgyReflection(2)
    task = runner(client)
    source = source_case(tmp_path)
    result = task.run_pair(source, tmp_path / "runs")
    directory = tmp_path / "runs/reflection/bd/ufl/ufl_small_seed000/attempts/attempt_01"
    requests = [read(p) for p in directory.glob("calls/*/request.json")]
    assert len(requests) == 3
    assert len({request["request_sha256"] for request in requests}) == 1
    assert len({request["invocation_id"] for request in requests}) == 3
    assert waits == [5, 10]
    assert result["reflection"]["attempts"][1]["status"] == "ok"
    assert len([call for call in client.calls if call[0] == "agent"]) == 3
    assert read(directory / "reflection_receipt.json")["result"]["ok"]
    count = len(client.calls)
    task.run_pair(source, tmp_path / "runs")
    assert len(client.calls) == count


def test_reflection_omission_exhaustion_returns_to_watchdog(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr("src.agent.paired_runner.time.sleep", waits.append)
    client = MissingAgyReflection(3)
    with pytest.raises(ReflectionArtifactError):
        runner(client).run_pair(source_case(tmp_path), tmp_path / "runs")
    assert waits == [5, 10]
    assert len([call for call in client.calls if call[0] == "reflection"]) == 3
    directory = tmp_path / "runs/reflection/bd/ufl/ufl_small_seed000/attempts/attempt_01"
    assert not (directory / "reflection_receipt.json").exists()
    assert (directory / "score_receipt.json").exists()


@pytest.mark.parametrize("diagnostics", ["authentication required", "HTTP 429 rate limit", "quota exceeded"])
def test_access_failures_bypass_local_reflection_retries(tmp_path, monkeypatch, diagnostics):
    waits = []
    monkeypatch.setattr("src.agent.paired_runner.time.sleep", waits.append)
    client = MissingAgyReflection(3, diagnostics)
    with pytest.raises(InfrastructureError) as caught:
        runner(client).run_pair(source_case(tmp_path), tmp_path / "runs")
    assert not isinstance(caught.value, ReflectionArtifactError)
    assert waits == []
    assert len([call for call in client.calls if call[0] == "reflection"]) == 1
