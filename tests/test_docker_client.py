"""Container results retain prompt, delivery, identity, and physical-call evidence."""
import json
from pathlib import Path

import pytest

from src.agent import docker_client
from src.agent.container_runtime import ContainerResult
from src.agent.docker_client import CLI_VERSIONS, DockerCliClient


class Runtime:
    def __init__(self, root, backend, contents="delivered", misplaced=None):
        self.seccomp_profile = root / "profile.json"
        self.seccomp_profile.write_text("{}")
        self.backend, self.contents, self.misplaced = backend, contents, misplaced or {}
        self.calls = []

    def ensure_volume(self, *args, **kwargs):
        pass

    def run(self, image, command, **kwargs):
        self.calls.append(command)
        kwargs["state_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["state_path"].write_text("{}")
        if command == ["preflight"]:
            value = {"ok": True, "versions": {"codex": CLI_VERSIONS["codex-cli"], "claude": CLI_VERSIONS["claude-code-cli"], "agy": CLI_VERSIONS["agy-cli"]}, "worker_files_sha256": docker_client.worker_file_hashes()}
            return ContainerResult(0, "", "", 0, artifacts={"worker.json": json.dumps(value)})
        job = kwargs["payload"]
        assert job["prompt"].startswith(f"Only edit the existing {job['target']} in the current working directory.")
        assert "PRIVATE_CREDENTIAL" not in job["prompt"]
        worker = {"cli_version": CLI_VERSIONS[self.backend], "requested_model": "fixture", "target": job["target"], "returncode": 0, "terminal_state": "completed", "error": None,
                  "usage": {"input_tokens": 10}, "usage_available": True, "warnings": [], "observed_models": ["fixture"], "agent_seconds": 1.25}
        self.result = ContainerResult(0, "", "", 2, artifacts={job["target"]: self.contents, "worker.json": json.dumps(worker), "delivery.json": json.dumps({"misplaced": self.misplaced})}, metadata={"cleanup_verified": True})
        return self.result

    def recover(self, state):
        return self.result


@pytest.fixture(autouse=True)
def clear_cache():
    docker_client._PREFLIGHTS.clear()
    yield
    docker_client._PREFLIGHTS.clear()


@pytest.mark.parametrize("backend", CLI_VERSIONS)
def test_fresh_edit_exports_one_target_and_preserves_backend_metrics(tmp_path, backend):
    runtime = Runtime(tmp_path, backend)
    target = tmp_path / "solver.py"
    target.write_text("")
    context = tmp_path / "data.json"
    context.write_text('{"example":1}')
    client = DockerCliClient("fixture", backend=backend, image_id="sha256:fixture", runtime=runtime, evidence_root=tmp_path / "preflight", credential_provider=lambda _: {"secret": "PRIVATE_CREDENTIAL"})
    result = client.edit_file(target, "Solve.", context=[context], log_dir=tmp_path / "call")
    assert result.ok and result.usage == {"input_tokens": 10} and result.elapsed_seconds == 1.25
    assert target.read_text() == "delivered"
    assert [name for name in runtime.calls if name[0] == "agent"] == [["agent", backend]]
    assert client.recover_file(target, log_dir=tmp_path / "call", step="agent").ok
    assert len(runtime.calls) == 2
    invocation = json.loads((tmp_path / "call/agent_invocation.json").read_text())
    assert invocation["transport"] == "docker" and invocation["cleanup_verified"]
    assert invocation["usage_available"] and invocation["observed_models"] == ["fixture"]


@pytest.mark.parametrize("requested,observed,allowed", [("claude-opus-5", "claude-opus-5-5", False), ("claude-sonnet-5", "claude-sonnet-5-5", False), ("claude-fable-5", "claude-fable-5-1", False), ("claude-opus-5", "claude-opus-5-20260601", True), ("claude-haiku-5-5", "claude-haiku-5-5", True)])
def test_claude_minor_version_is_not_a_dated_snapshot(tmp_path, requested, observed, allowed):
    runtime = Runtime(tmp_path, "claude-code-cli")
    client = DockerCliClient(requested, backend="claude-code-cli", image_id="sha256:fixture", runtime=runtime, evidence_root=tmp_path / "preflight")
    target = tmp_path / "solver.py"
    target.write_text("")
    log = tmp_path / "call"
    log.mkdir()
    worker = {"cli_version": CLI_VERSIONS["claude-code-cli"], "requested_model": requested, "target": "solver.py", "returncode": 0, "terminal_state": "completed", "error": None, "usage": {"input_tokens": 10}, "usage_available": True, "warnings": [], "observed_models": [observed]}
    result = ContainerResult(0, "", "", 1, artifacts={"solver.py": "candidate", "worker.json": json.dumps(worker)}, metadata={"cleanup_verified": True})
    recorded = client._record(target, log, "agent", result)
    assert recorded.ok is allowed
    assert target.read_text() == ("candidate" if allowed else "")
    if not allowed:
        assert recorded.terminal_state == "infrastructure_failed"
        assert "identity" in recorded.error


def test_completed_wrong_target_preserves_evidence_without_promoting_it(tmp_path):
    runtime = Runtime(tmp_path, "agy-cli", contents="", misplaced={"nested/debug.md": "misplaced reflection"})
    target = tmp_path / "debug.md"
    target.write_text("")
    client = DockerCliClient("fixture", backend="agy-cli", image_id="sha256:fixture", runtime=runtime, evidence_root=tmp_path / "preflight", credential_provider=lambda _: {})
    result = client.edit_file(target, "Reflect.", log_dir=tmp_path / "call", step="reflection")
    assert not result.ok and result.returncode == 0 and result.terminal_state == "completed"
    assert target.read_text() == ""
    assert (tmp_path / "call/unexpected_outputs/nested/debug.md").read_text() == "misplaced reflection"
    assert (tmp_path / "call/reflection_delivery_failure.json").exists()


def test_committed_generation_omission_is_recovered_without_an_extra_call(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from src.agent.paired_runner import PairedRunner
    monkeypatch.setattr("src.agent.paired_runner.runtime_lock", lambda *_: nullcontext())
    runtime = Runtime(tmp_path, "codex-cli", contents="")
    client = DockerCliClient("fixture", backend="codex-cli", image_id="sha256:fixture", runtime=runtime, evidence_root=tmp_path / "preflight", credential_provider=lambda _: {})
    task = PairedRunner(client=client, judge_client=client, solver_executor=lambda *_: pytest.fail("This edit-only test must not execute a solver"))
    target = tmp_path / "attempt_01/solver.py"
    target.parent.mkdir()
    original = task.publish_call
    monkeypatch.setattr(task, "publish_call", lambda *_: (_ for _ in ()).throw(RuntimeError("interrupted")))
    with pytest.raises(RuntimeError, match="interrupted"):
        task.edit(client, target, "Solve.", [], "agent")
    monkeypatch.setattr(task, "publish_call", original)
    result = task.edit(client, target, "Solve.", [], "agent")
    assert not result["ok"] and result["terminal_state"] == "completed"
    assert len([call for call in runtime.calls if call[0] == "agent"]) == 1
