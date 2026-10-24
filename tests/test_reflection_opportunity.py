"""Completed omissions advance once; transport faults keep their recovery path."""
from contextlib import nullcontext
from dataclasses import asdict

import pytest

from src.agent.result import AgentResult
from src.experiments.paired_runner import PairedRunner, ProtocolError, read, save
from host_solver import host_solver


class MissingReflection:
    def __init__(self, backend, delivery):
        self.backend, self.delivery, self.calls = backend, delivery, 0

    def protocol(self):
        return {"backend": self.backend}

    def edit_file(self, target, prompt, *, context, log_dir, step):
        self.calls += 1
        if self.delivery == "deleted":
            target.unlink()
        if self.delivery == "misplaced":
            path = log_dir / "unexpected_outputs/nested/debug.md"
            path.parent.mkdir(parents=True)
            path.write_text("Wrong-target content must never reach the next model.")
        result = AgentResult(False, 0, 0.25, {"input_tokens": 7, "output_tokens": 2}, "No reflection delivered", terminal_state="completed")
        save(log_dir / "reflection_metrics.json", asdict(result))
        return result


@pytest.mark.parametrize("backend", ["codex-cli", "agy-cli", "claude-code-cli"])
@pytest.mark.parametrize("delivery", ["empty", "deleted", "misplaced"])
@pytest.mark.parametrize("crash", ["after_metrics", "lost_receipt", "none"])
def test_completed_omission_is_recovered_once_with_original_usage(tmp_path, monkeypatch, backend, delivery, crash):
    monkeypatch.setattr("src.experiments.paired_runner.runtime_lock", lambda *_: nullcontext())
    client = MissingReflection(backend, delivery)
    task = PairedRunner(client=client, judge_client=client, solver_executor=host_solver)
    target = tmp_path / "attempt_01/debug.md"
    target.parent.mkdir()
    if crash == "after_metrics":
        publish = task.publish_call
        monkeypatch.setattr(task, "publish_call", lambda *_: (_ for _ in ()).throw(RuntimeError("simulated crash")))
        with pytest.raises(RuntimeError, match="simulated crash"):
            task.edit(client, target, "Reflect.", [], "reflection")
        monkeypatch.setattr(task, "publish_call", publish)
    else:
        task.edit(client, target, "Reflect.", [], "reflection")
        if crash == "lost_receipt":
            (target.parent / "reflection_receipt.json").unlink()
    result = task.edit(client, target, "Reflect.", [], "reflection")
    assert not result["ok"] and result["file_delivery_failure"]
    assert result["usage"] == {"input_tokens": 7, "output_tokens": 2}
    assert result["elapsed_seconds"] == 0.25
    assert client.calls == 1 and target.read_text() == ""
    assert len(list((target.parent / "calls").iterdir())) == 1
    assert read(target.parent / "reflection_receipt.json")["result"] == result
    target.write_text("late replacement")
    with pytest.raises(ProtocolError, match="outcome changed"):
        task.edit(client, target, "Reflect.", [], "reflection")


def test_modified_misplaced_evidence_is_not_accepted_on_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr("src.experiments.paired_runner.runtime_lock", lambda *_: nullcontext())
    client = MissingReflection("codex-cli", "misplaced")
    task = PairedRunner(client=client, judge_client=client, solver_executor=host_solver)
    target = tmp_path / "attempt_01/debug.md"
    target.parent.mkdir()
    task.edit(client, target, "Reflect.", [], "reflection")
    path = next((target.parent / "calls").glob("*/unexpected_outputs/nested/debug.md"))
    path.write_text("altered")
    with pytest.raises(ProtocolError, match="evidence changed"):
        task.edit(client, target, "Reflect.", [], "reflection")
