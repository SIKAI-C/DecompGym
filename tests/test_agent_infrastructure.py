"""Tests for agent isolation, repeat semantics, verification, and dataset reuse."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from src.agent.codex_client import CodexClient, CodexResult
from src.agent.experiment_runner import ExperimentRunner
from src.agent.output_verifier import OutputVerifier
from src.utils.dataset import ensure_suite


def fake_cli(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\nimport os, sys\nos.chdir(sys.argv[sys.argv.index('--cd') + 1])\n" + body)
    path.chmod(0o755)
    return path


def test_codex_isolates_context_and_copies_only_target(tmp_path):
    target = tmp_path / "solver.py"
    target.write_text("old")
    (tmp_path / "gold_monolithic.json").write_text("REFERENCE_SECRET")
    data = tmp_path / "data.json"
    data.write_text('{"m": 2}')
    cli = fake_cli(tmp_path / "fake-codex", """
import json, sys
from pathlib import Path
prompt = sys.stdin.read()
assert 'REFERENCE_SECRET' not in prompt
assert prompt.startswith('Only edit the existing solver.py in the current working directory.')
assert Path('solver.py').is_file()
assert '{"m": 2}' in prompt
assert '--ephemeral' in sys.argv and '--ignore-user-config' in sys.argv
assert 'project_doc_max_bytes=0' in sys.argv
assert Path.cwd() != Path(sys.argv[0]).parent
Path('solver.py').write_text('answer = 42')
Path('unrequested.py').write_text('unexpected')
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10,'output_tokens':2}}))
""")
    result = CodexClient(executable=str(cli)).edit_file(target, "Write the solver.", context=[data], step="generation")
    assert result.ok and result.usage == {"input_tokens": 10, "output_tokens": 2}
    assert target.read_text() == "answer = 42"
    assert not (tmp_path / "unrequested.py").exists()
    assert (tmp_path / "gold_monolithic.json").read_text() == "REFERENCE_SECRET"


def test_failed_codex_edit_keeps_original(tmp_path):
    target = tmp_path / "solver.py"
    target.write_text("original")
    cli = fake_cli(tmp_path / "fake-codex", "from pathlib import Path\nPath('solver.py').write_text('broken')\nraise SystemExit(1)\n")
    result = CodexClient(executable=str(cli)).edit_file(target, "Edit.")
    assert not result.ok and target.read_text() == "original"


def test_missing_cli_is_reported(tmp_path):
    result = CodexClient(executable="definitely-no-such-codex").edit_file(tmp_path / "solver.py", "Edit.")
    assert not result.ok and "not found" in result.error


@pytest.mark.parametrize("reported_model", [None, "gpt-6.1-sol", "gpt-6.1-sol-snapshot", 123, ""])
def test_codex_records_optional_model_metadata_without_changing_outcome(tmp_path, reported_model):
    event = {"type": "turn.completed", "usage": {"input_tokens": 10}}
    if reported_model is not None:
        event["model"] = reported_model
    body = "from pathlib import Path\nPath('solver.py').write_text('answer = 42')\n"
    body += "print(" + repr(json.dumps(event)) + ")\n"
    cli = fake_cli(tmp_path / "fake-codex", body)
    result = CodexClient(model="gpt-6.1-sol", executable=str(cli)).edit_file(tmp_path / "solver.py", "Edit.")
    invocation = json.loads((tmp_path / "codex_invocation.json").read_text())
    assert result.ok and result.usage == {"input_tokens": 10}
    assert invocation["requested_model"] == "gpt-6.1-sol"
    assert invocation["observed_models"] == ([reported_model] if isinstance(reported_model, str) and reported_model else [])


def test_recovered_connection_warning_does_not_discard_successful_edit(tmp_path):
    target = tmp_path / "solver.py"
    target.write_text("original")
    cli = fake_cli(tmp_path / "fake-codex", """
import json
from pathlib import Path
print(json.dumps({'type':'error','message':'Reconnecting after a temporary network interruption'}))
Path('solver.py').write_text('answer = 42')
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10}}))
""")
    result = CodexClient(executable=str(cli)).edit_file(target, "Edit.")
    assert result.ok and result.error is None
    assert result.warnings == ["Reconnecting after a temporary network interruption"]
    assert target.read_text() == "answer = 42"


def test_failed_terminal_turn_is_not_accepted_even_with_zero_exit_status(tmp_path):
    target = tmp_path / "solver.py"
    target.write_text("original")
    cli = fake_cli(tmp_path / "fake-codex", """
import json
from pathlib import Path
Path('solver.py').write_text('unfinished')
print(json.dumps({'type':'turn.failed','error':{'message':'turn failed'}}))
""")
    result = CodexClient(executable=str(cli)).edit_file(target, "Edit.")
    assert not result.ok and result.error == "turn failed"
    assert target.read_text() == "original"


def test_completed_turn_after_tool_host_failure_is_infrastructure_failure(tmp_path):
    target = tmp_path / "solver.py"
    target.write_text("original")
    cli = fake_cli(tmp_path / "fake-codex", """
import json,sys
print('ERROR timed out negotiating with the code-mode host',file=sys.stderr)
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10}}))
""")
    result = CodexClient(executable=str(cli)).edit_file(target, "Edit.")
    assert not result.ok and result.terminal_state == "infrastructure_failed"
    assert "code-mode host" in result.error
    assert target.read_text() == "original" and result.usage['input_tokens'] == 10


def test_codex_timeout_keeps_original(tmp_path):
    target = tmp_path / "solver.py"
    target.write_text("original")
    cli = fake_cli(tmp_path / "fake-codex", "import time\ntime.sleep(10)\n")
    result = CodexClient(executable=str(cli), timeout=0.1).edit_file(target, "Edit.")
    assert not result.ok and "timed out" in result.error and target.read_text() == "original"


def ufl_files(directory: Path, output: dict):
    directory.mkdir(parents=True, exist_ok=True)
    data = directory / "data.json"
    result = directory / "output.json"
    data.write_text(json.dumps({"m": 2, "n": 1, "f": [2, 3], "D": [[4], [1]]}))
    result.write_text(json.dumps(output))
    return data, result


def test_fractional_binary_decisions_fail(tmp_path):
    data, output = ufl_files(tmp_path, {"y": [0.5, 0.5], "x": [[0.5], [0.5]], "objective_value": 5})
    result = OutputVerifier.verify_output("bd", "ufl", data, output, 1e-6)
    assert not result["ok"] and any("binary" in issue for issue in result["issues"])


def test_objective_is_recomputed(tmp_path):
    data, output = ufl_files(tmp_path, {"y": [0, 1], "x": [[0], [1]], "objective_value": 0})
    result = OutputVerifier.verify_output("bd", "ufl", data, output, 1e-6)
    assert not result["ok"] and result["objective_value"] == 4


def test_valid_solution_passes(tmp_path):
    data, output = ufl_files(tmp_path, {"y": [0, 1], "x": [[0], [1]], "objective_value": 4})
    assert OutputVerifier.verify_output("bd", "ufl", data, output, 1e-6)["ok"]


def test_nonfinite_solution_fails(tmp_path):
    data, output = ufl_files(tmp_path, {"y": [0, 1], "x": [[0], [float("nan")]], "objective_value": 4})
    assert not OutputVerifier.verify_output("bd", "ufl", data, output, 1e-6)["ok"]


def test_zero_objective_and_missing_reference():
    assert OutputVerifier.extract_objective({"objective_value": 0})[0] == 0
    status, gap = OutputVerifier.assess_status(exec_ok=True, output={}, verify_ok=True, objective=0, gold_obj=None, gap_threshold=0.01)
    assert status == "reference_unavailable" and gap is None


def test_stderr_warning_does_not_fail_execution(tmp_path):
    (tmp_path / "solver.py").write_text("import json,sys\nfrom pathlib import Path\nprint('warning',file=sys.stderr)\nPath('output.json').write_text(json.dumps({'objective_value':0}))\n")
    ok, _, stderr, output = ExperimentRunner.run_solver(tmp_path, 5)
    assert ok and "warning" in stderr and output["objective_value"] == 0


def test_stale_output_is_not_reused(tmp_path):
    (tmp_path / "solver.py").write_text("pass\n")
    (tmp_path / "output.json").write_text('{"objective_value":0}')
    assert not ExperimentRunner.run_solver(tmp_path, 5)[0]


def test_solver_resource_limits_are_propagated(tmp_path):
    (tmp_path / "solver.py").write_text("import json,os\nfrom pathlib import Path\nPath('output.json').write_text(json.dumps({'threads':os.environ['OMP_NUM_THREADS'],'blas_threads':os.environ['OPENBLAS_NUM_THREADS'],'gurobi':Path('gurobi.env').read_text()}))\n")
    ok, _, _, output = ExperimentRunner.run_solver(tmp_path, 5, threads=1)
    assert ok and output['threads'] == '1' and output['blas_threads'] == '1'
    assert output['gurobi'] == 'ThreadLimit 1\nThreads 1\n'


def test_timeout_preserves_progress_logs(tmp_path):
    (tmp_path / "solver.py").write_text("import time\nprint('iteration 1: no feasible incumbent',flush=True)\ntime.sleep(5)\n")
    ok, stdout, stderr, output = ExperimentRunner.run_solver(tmp_path, 0.1)
    assert not ok and output is None
    assert 'iteration 1' in stdout
    assert '[framework] solver execution timed out' in stderr


def test_timeout_stops_solver_children(tmp_path):
    child = "import time;from pathlib import Path;time.sleep(0.5);Path('child_survived').write_text('unexpected')"
    (tmp_path / "solver.py").write_text("import subprocess,sys,time\nsubprocess.Popen([sys.executable,'-c'," + repr(child) + "])\nprint('child started',flush=True)\ntime.sleep(5)\n")
    ok, _, _, _ = ExperimentRunner.run_solver(tmp_path, 0.1)
    assert not ok
    time.sleep(0.6)
    assert not (tmp_path / 'child_survived').exists()


def test_timeout_stops_child_that_ignores_term_after_parent_exits(tmp_path):
    child = "import signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path('child_ready').touch();time.sleep(0.5);Path('child_survived').write_text('unexpected')"
    (tmp_path / "solver.py").write_text("import subprocess,sys,time\nfrom pathlib import Path\nsubprocess.Popen([sys.executable,'-c'," + repr(child) + "],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\nwhile not Path('child_ready').exists(): time.sleep(0.01)\nprint('child ready',flush=True)\ntime.sleep(5)\n")
    ok, stdout, _, _ = ExperimentRunner.run_solver(tmp_path, 0.2)
    assert not ok and 'child ready' in stdout
    time.sleep(0.6)
    assert not (tmp_path / 'child_survived').exists()


def test_timeout_is_not_reported_as_missing_objective():
    status, _ = OutputVerifier.assess_status(exec_ok=False, output=None, verify_ok=False, objective=None, gold_obj=4, gap_threshold=0.01, execution_timed_out=True)
    assert status == 'timeout'


def complete_instance(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ["data.json", "meta.json", "gold_monolithic.json", "gold_cg.json"]:
        (directory / name).write_text(json.dumps({"best_objective": 1}))
    (directory / "prompt.txt").write_text("prompt")


def test_cached_dataset_needs_no_solver_import(tmp_path, monkeypatch):
    directory = tmp_path / "bp_small_seed000"
    complete_instance(directory)
    monkeypatch.setattr("src.utils.dataset.importlib.import_module", lambda *_: pytest.fail("cached data imported a solver"))
    records = ensure_suite("cg", "bp", tmp_path, ["small"], 1)
    assert records[0]["reused"]


def test_missing_instance_generated_once(tmp_path, monkeypatch):
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        complete_instance(tmp_path / "bp_small_seed000")
    monkeypatch.setattr("src.utils.dataset.importlib.import_module", lambda *_: SimpleNamespace(generate_suite=generate))
    assert not ensure_suite("cg", "bp", tmp_path, ["small"], 1)[0]["reused"]
    assert ensure_suite("cg", "bp", tmp_path, ["small"], 1)[0]["reused"]
    assert len(calls) == 1


def test_strict_cg_validation_compares_cg_objective(tmp_path):
    directory = tmp_path / "cg" / "bp" / "bp_small_seed000"
    complete_instance(directory)
    assert importlib.import_module("src.01_gen").validate_instance(directory)["status"] == "pass_simple_test"


def test_positive_decomposition_gap_survives_bad_instance_cleanup(tmp_path):
    directory = tmp_path / "cg" / "bp" / "bp_small_seed000"
    complete_instance(directory)
    (directory / "gold_cg.json").write_text('{"best_objective":2}')
    generator = importlib.import_module("src.01_gen")
    result = generator.validate_instance(directory)
    assert result["status"] == "reference_gap" and result["decomposition_gap"] == 1
    assert generator.remove_instances(tmp_path, [result], remove_mismatch=True) == 0
    assert directory.exists()


def test_objective_better_than_reference_is_flagged(tmp_path):
    directory = tmp_path / "cg" / "bp" / "bp_small_seed000"
    complete_instance(directory)
    (directory / "gold_cg.json").write_text('{"best_objective":0}')
    assert importlib.import_module("src.01_gen").validate_instance(directory)["status"] == "objective_mismatch"


def test_missing_quality_is_not_zero_gap():
    analysis = importlib.import_module("src.03_analysis")
    metrics = analysis.cumulative_metrics([{"exec_ok": True, "verify_ok": True}], 0.01, None)
    assert metrics["solveQ"] == [0] and metrics["quality"] == [None]


def test_early_stop_records_reuse_without_extra_calls(tmp_path):
    source = tmp_path / "data" / "bd" / "ufl" / "ufl_small_seed000"
    ufl_files(source, {})
    (source / "prompt.txt").write_text("Implement Benders decomposition.")
    (source / "meta.json").write_text('{"ground_truth_objective":4}')
    (source / "gold_monolithic.json").write_text("REFERENCE_SECRET")
    class FakeClient:
        calls = []
        def edit_file(self, target, prompt, *, context=None, log_dir=None, step="codex"):
            self.calls.append(step)
            assert all(path.name not in {"meta.json", "gold_monolithic.json"} for path in context or [])
            if step == "agent":
                target.write_text("import json\nfrom pathlib import Path\nPath('output.json').write_text(json.dumps({'y':[0,1],'x':[[0],[1]],'objective_value':4}))\n")
            else:
                target.write_text("VERDICT: PASS\nNOTES: mocked algorithm review\n")
            return CodexResult(True, 0, 0.01, {})
    client = FakeClient()
    runner = ExperimentRunner(client=client, max_attempts=3)
    summary = runner.run_instance(source, tmp_path / "run", "ufl")
    assert [attempt["status"] for attempt in summary["attempts"]] == ["ok", "skipped_previous_solved", "skipped_previous_solved"]
    assert client.calls == ["agent", "llm_verifier"]
    assert not (Path(summary["output_dir"]) / "gold_monolithic.json").exists()
