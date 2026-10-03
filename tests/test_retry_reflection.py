"""Exercise failure, reflection, retry, and budget exhaustion without relying on model weakness."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.codex_client import CodexResult
from src.agent.experiment_runner import ExperimentRunner, ModelAccessError


def source_instance(tmp_path: Path) -> Path:
    source = tmp_path / "data" / "bd" / "ufl" / "ufl_small_seed000"
    source.mkdir(parents=True)
    (source / "data.json").write_text(json.dumps({"m": 2, "n": 1, "f": [2, 3], "D": [[4], [1]]}))
    (source / "prompt.txt").write_text("Implement Benders decomposition and write an integer facility plan.")
    (source / "meta.json").write_text('{"ground_truth_objective":4}')
    return source


def solver_code(objective: int) -> str:
    output = {"y": [0, 1], "x": [[0], [1]], "objective_value": 4} if objective == 4 else {"y": [1, 0], "x": [[1], [0]], "objective_value": 6}
    return "import json\nfrom pathlib import Path\nPath('output.json').write_text(json.dumps(" + repr(output) + "))\n"


@pytest.mark.parametrize("failure", ["execution", "objective_gap"])
def test_failure_reflection_and_repair_are_connected(tmp_path, failure):
    class ControlledClient:
        def __init__(self):
            self.calls = []
            self.generations = 0
            self.next_context = None
        def edit_file(self, target, prompt, *, context=None, log_dir=None, step="codex"):
            self.calls.append(step)
            if step == "agent":
                self.generations += 1
                if self.generations == 1:
                    target.write_text("raise RuntimeError('controlled execution failure')\n" if failure == "execution" else solver_code(6))
                else:
                    self.next_context = list(context or [])
                    assert any(path.name == "debug.md" and "CONTROLLED_REPAIR_NOTE" in path.read_text() for path in self.next_context)
                    assert any(path.name == "solver.py" and path.parent.name == "attempt_01" for path in self.next_context)
                    target.write_text(solver_code(4))
            elif step == "llm_verifier":
                target.write_text("VERDICT: PASS\nNOTES: controlled reviewer fixture\n")
            elif step == "reflection":
                logs = next(path for path in context if path.name == "exec.log").read_text()
                assert "controlled execution failure" in logs if failure == "execution" else "0.5" in prompt
                target.write_text("CONTROLLED_REPAIR_NOTE: write the valid plan with objective 4.\n")
            return CodexResult(True, 0, 0.01, {})
    client = ControlledClient()
    runner = ExperimentRunner(client=client, reflection=True, max_attempts=4)
    summary = runner.run_instance(source_instance(tmp_path), tmp_path / "run", "ufl")
    attempts = summary["attempts"]
    assert not attempts[0]["verify_ok"] if failure == "execution" else attempts[0]["status"] == "feasible_not_optimal"
    assert attempts[1]["status"] == "ok" and attempts[1]["attempt"] == 2
    assert [attempt["status"] for attempt in attempts[2:]] == ["skipped_previous_solved", "skipped_previous_solved"]
    assert client.calls == ["agent", "llm_verifier", "reflection", "agent", "llm_verifier"]
    assert client.generations == 2


def test_maximum_attempts_is_a_hard_limit(tmp_path):
    class AlwaysFails:
        def __init__(self):
            self.generations = 0
            self.reflections = 0
        def edit_file(self, target, prompt, *, context=None, log_dir=None, step="codex"):
            if step == "agent":
                self.generations += 1
                target.write_text("raise RuntimeError('persistent controlled failure')\n")
            elif step == "llm_verifier":
                target.write_text("VERDICT: PASS\nNOTES: controlled reviewer fixture\n")
            else:
                self.reflections += 1
                target.write_text("Controlled diagnosis: failure persists.\n")
            return CodexResult(True, 0, 0.01, {})
    client = AlwaysFails()
    summary = ExperimentRunner(client=client, reflection=True, max_attempts=3).run_instance(source_instance(tmp_path), tmp_path / "run", "ufl")
    assert client.generations == 3
    assert len(summary["attempts"]) == 3
    assert all(not attempt["exec_ok"] and not attempt["reused_from_previous"] for attempt in summary["attempts"])
    assert not (Path(summary["output_dir"]) / "attempts" / "attempt_04").exists()


def test_no_reflection_retry_has_no_previous_attempt_context(tmp_path):
    class ControlledClient:
        def __init__(self):
            self.generations = 0
            self.calls = []
        def edit_file(self, target, prompt, *, context=None, log_dir=None, step="codex"):
            self.calls.append(step)
            if step == "agent":
                self.generations += 1
                assert [path.name for path in context or []] == ["data.json"]
                target.write_text("raise RuntimeError('controlled first failure')\n" if self.generations == 1 else solver_code(4))
            else:
                assert step == "llm_verifier"
                target.write_text("VERDICT: PASS\nNOTES: controlled reviewer fixture\n")
            return CodexResult(True, 0, 0.01, {})
    client = ControlledClient()
    summary = ExperimentRunner(client=client, reflection=False, max_attempts=3).run_instance(source_instance(tmp_path), tmp_path / "run", "ufl")
    assert summary["attempts"][1]["status"] == "ok"
    assert "reflection" not in client.calls
    assert not any(Path(summary["output_dir"]).rglob("debug.md"))


def test_execution_budget_is_visible_and_cooling_is_outside_scored_time(tmp_path, monkeypatch):
    class Client:
        def edit_file(self, target, prompt, *, context=None, log_dir=None, step="codex"):
            if step == "agent":
                assert 'execution to 60 seconds' in prompt and '1 thread(s)' in prompt
                target.write_text(solver_code(4))
            else:
                target.write_text("VERDICT: PASS\nNOTES: controlled reviewer fixture\n")
            return CodexResult(True, 0, 0, {})
    clock = iter([0, 0, 6, 6, 16, 17])
    pauses = []
    monkeypatch.setattr('src.agent.experiment_runner.time.perf_counter', lambda: next(clock))
    monkeypatch.setattr('src.agent.experiment_runner.time.sleep', pauses.append)
    summary = ExperimentRunner(client=Client(), solver_timeout=60, solver_threads=1, solver_cooldown_seconds=10, max_attempts=1).run_instance(source_instance(tmp_path), tmp_path / 'run', 'ufl')
    metrics = summary['attempts'][0]['metrics']
    assert summary['attempts'][0]['status'] == 'ok'
    assert metrics['execution_seconds'] == 6 and metrics['cooldown_seconds'] == 10
    assert pauses == [10]


def test_quota_failure_stops_without_using_retry_budget(tmp_path):
    class QuotaClient:
        def __init__(self):
            self.calls = 0
        def edit_file(self, *args, **kwargs):
            self.calls += 1
            return CodexResult(False, 1, 0, {}, 'usage_limit: account quota exhausted')
    client = QuotaClient()
    runner = ExperimentRunner(client=client, max_attempts=10)
    with pytest.raises(ModelAccessError):
        runner.run_instance(source_instance(tmp_path), tmp_path / 'run', 'ufl')
    assert client.calls == 1 and runner.abort_event.is_set()
    assert not (tmp_path / 'run' / 'ufl_small_seed000' / 'attempts' / 'attempt_02').exists()
