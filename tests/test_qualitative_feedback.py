"""Check the agent-facing boundary independently of internal reference scoring."""
from contextlib import nullcontext
from dataclasses import asdict
import json

import pytest

from src.agent.codex_client import CodexResult
from src.agent.experiment_runner import AttemptResult, ExperimentRunner
from src.agent.paired_runner import PairedRunner, ProtocolError, digest, read, save
from src.agent.prompt_builder import PromptBuilder
from src.agent.prompt_policy import OBJECTIVE_RETRY


@pytest.mark.parametrize("changes,expected", [
    ({}, True), ({"exec_ok": False}, False), ({"feasibility_ok": False}, False),
    ({"objective": None}, False), ({"gold_objective": None}, False), ({"gap_to_gold": None}, False),
    ({"gap_to_gold": float("nan")}, False), ({"gold_objective": float("inf")}, False),
    ({"gap_to_gold": 0.01}, False), ({"gap_to_gold": 0}, False),
    ({"metrics": {"execution_timed_out": True}}, False),
    ({"status": "verify_failed", "verify_ok": False, "llm_verdict": "fail"}, True),
    ({"status": "verify_failed", "verify_ok": False, "llm_verdict": "fail", "gap_to_gold": 0}, False),
])
def test_objective_feedback_uses_confirmed_feasibility_and_quality(changes, expected):
    result = AttemptResult(1, "feasible_not_optimal", True, True, 6, 4, 0.5, [], feasibility_ok=True)
    for key, value in changes.items():
        setattr(result, key, value)
    flag = ExperimentRunner().objective_quality_failed(result)
    prompt = PromptBuilder.build_reflection_prompt("TASK", objective_quality_failed=flag)
    assert flag is expected and (OBJECTIVE_RETRY in prompt) == expected
    assert "gap" not in prompt and "0.5" not in prompt


@pytest.mark.parametrize("value", [0.5, "0.5", None, 1])
def test_prompt_builder_refuses_numeric_or_text_feedback(value):
    with pytest.raises(TypeError):
        PromptBuilder.build_reflection_prompt("TASK", objective_quality_failed=value)


class CaptureClient:
    def __init__(self):
        self.calls = []

    def protocol(self):
        return {"backend": "test", "model": "fixture"}

    def edit_file(self, target, prompt, *, context, log_dir, step):
        self.calls.append((step, prompt, [(p.parent.name + "/" + p.name, p.read_text()) for p in context if p.exists()]))
        if step == "agent":
            target.write_text("import json\nfrom pathlib import Path\nPath('output.json').write_text(json.dumps({'status':'feasible','method':'benders_decomposition','y':[1,0],'x':[[1],[0]],'objective_value':6}))\n")
        elif step == "llm_verifier":
            target.write_text("VERDICT: PASS\nNOTES: structural check passed\n")
        else:
            target.write_text("Reconsider the candidate plan using the supplied feedback.\n")
        result = CodexResult(True, 0, 0.01, {}, terminal_state="completed", artifact_sha256=digest(target))
        save(log_dir / f"{step}_metrics.json", asdict(result))
        return result


@pytest.mark.parametrize("paired", [False, True])
def test_changing_hidden_reference_changes_scores_but_not_any_agent_context(tmp_path, monkeypatch, paired):
    monkeypatch.setattr("src.agent.paired_runner.runtime_lock", lambda name: nullcontext())
    records, scores = [], []
    for label, gold in (("a", 4), ("b", 3)):
        source = tmp_path / label / "data/bd/ufl/ufl_small_seed000"
        source.mkdir(parents=True)
        save(source / "data.json", {"m": 2, "n": 1, "f": [2, 3], "D": [[4], [1]]})
        save(source / "meta.json", {"ground_truth_objective": gold})
        save(source / "gold_monolithic.json", {"objective_value": gold, "oracle_sentinel": "PRIVATE_REFERENCE"})
        (source / "prompt.txt").write_text("Implement Benders decomposition.")
        client = CaptureClient()
        if paired:
            task = PairedRunner(client=client, judge_client=client, model="fixture", problem=["ufl"], max_attempts=2)
            summary = task.run_pair(source, tmp_path / label / "runs")["reflection"]
        else:
            task = ExperimentRunner(client=client, reflection=True, max_attempts=2)
            summary = task.run_instance(source, tmp_path / label / "runs", "ufl")
        records.append(client.calls)
        scores.append(summary["attempts"][0])
        reflection = [c for c in client.calls if c[0] == "reflection"]
        assert reflection and all(OBJECTIVE_RETRY in c[1] for c in reflection)
        assert any('"objective_value": 6' in contents for c in reflection for name, contents in c[2] if name.endswith("output.json"))
        assert all(not name.endswith(("meta.json", "gold_monolithic.json", "summary.json", "score_receipt.json")) for c in client.calls for name, contents in c[2])
        assert "PRIVATE_REFERENCE" not in json.dumps(client.calls)
    assert records[0] == records[1]
    assert [s["gold_objective"] for s in scores] == [4, 3]
    assert [s["gap_to_gold"] for s in scores] == [0.5, 1.0]


def test_new_policy_rejects_legacy_case_identity_even_with_same_explicit_protocol(tmp_path):
    source = tmp_path / "case"
    source.mkdir()
    (source / "data.json").write_text("{}")
    (source / "prompt.txt").write_text("TASK")
    client = CaptureClient()
    task = PairedRunner(client=client, judge_client=client, model="fixture", problem=["ufl"], protocol_id="old")
    directory = task.prepare(source, tmp_path / "run", "plain")
    old = read(directory / "identity.json")
    old.pop("prompt_policies")
    save(directory / "identity.json", old)
    with pytest.raises(ProtocolError, match="Case protocol changed"):
        task.prepare(source, tmp_path / "run", "plain")
