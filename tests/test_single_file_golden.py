"""The single-file (paper) delivery path is byte-identical to the code before multi-file delivery was added.

The digests below were captured from a pristine export of commit 213b7bc (before `src/agent/multifile.py`, `docker/agents/multifile_worker.py`
and the `DockerRuntime.WORKER` / `COLLECT_OUTPUT_BYTES` class attributes existed). They cover everything the single-file path emits:
the Docker create/start/collect command lines and the durable container state, the worker job sent to the container, every record the
client writes (prompts, workspace record, metrics, invocation, delivery evidence), the committed target and the returned result.
No Docker engine, model call or solver is used. This is fixture-based regression coverage of the client/runtime records (the
worker's own outputs are fixtures, and the paper worker files are pinned by hash below); the 4970 recorded paper prompts are
checked separately by tests/test_paper_equivalence.py.

Re-capturing the baseline (only to extend coverage, never to absorb a behavior change): export the baseline commit with
``git archive 213b7bc | tar -x -C <dir>``, copy this file into ``<dir>/tests/``, temporarily empty the golden digest mapping
there and run ``python -m pytest <dir>/tests/test_single_file_golden.py -k golden -vv`` from ``<dir>``; the assertion lists them.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

import pytest

from src.agent import container as containers
from src.agent import docker_client
from src.agent.container import ContainerResult, DockerRuntime
from src.agent.docker_client import CLI_VERSIONS, DockerCliClient
from src.agent.prompts.policy import record_workspace_prompt

ROOT = Path(__file__).resolve().parents[1]

GOLDEN = {
    "runtime": "86477e4d66216fa5be8fd6f94afc2178e1b504c96abbcd54fd6a6a4c87e722f3",
    "client:codex-cli": "c8a3dc32d5849c37f1c7916dc946f217246989b50e8135a2cf3b846495359f75",
    "client:claude-code-cli": "223664d47ad56b40c45619690931ba01bf9023d6ee56d2fab77fd0db89c07388",
    "client:agy-cli": "d9826bf7fda74b6267c1d08fe92a873bb2c31c5303cc7098c39611de7fcdb6e3",
    "client:misplaced": "ff84f9eeef3b06094353e653bc3daaa3756b9c7e83f4d2f47f90e7c119e89107",
    "prompt": "41131c780c11be6dba9de70fbd7fdb6c8981260e55164d77342c5a2ba5651e3f",
}


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def tree(root: Path, scratch: Path) -> dict:
    """sha256 of every file's bytes (the scratch directory replaced by `<tmp>`), keyed by relative path."""
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes().replace(str(scratch).encode(), b"<tmp>")).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def capture_runtime(tmp_path, monkeypatch, cls=DockerRuntime) -> dict:
    """One full DockerRuntime.run (create, start, collect, dispose) with the Docker CLI replaced by a recorder."""
    runtime = cls(seccomp_profile=tmp_path / "profile.json")
    (tmp_path / "profile.json").write_text("{}")
    commands, processes = [], []
    monkeypatch.setattr(containers.uuid, "uuid4", lambda: uuid.UUID(int=7))
    monkeypatch.setattr(containers.time, "time", lambda: 1000.0)

    def command(*args, timeout=30):
        commands.append(list(args))
        if args[:2] == ("image", "inspect"):
            return subprocess.CompletedProcess(args, 0, "sha256:fixture\n", "")
        if args[:2] == ("volume", "inspect"):
            return subprocess.CompletedProcess(args, 0, json.dumps({runtime.LABEL: "v2-ephemeral-worker"}), "")
        if args[0] == "create":
            return subprocess.CompletedProcess(args, 0, "container-id\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    states = iter([{"HostConfig": {"Privileged": False, "ReadonlyRootfs": True, "Binds": None, "SecurityOpt": ["no-new-privileges"]},
                    "Mounts": [{"Type": "volume", "Name": "w", "Destination": "/work", "RW": True}]},
                   {"State": {"Running": False, "ExitCode": 0, "OOMKilled": False}}])

    def inspect(name):
        return next(states, None)

    def bounded(args, *, stdin="", timeout, max_output_bytes=8 * 1024 * 1024, heartbeat=False):
        processes.append({"args": [str(item) for item in args], "stdin": stdin, "timeout": timeout, "max_output_bytes": max_output_bytes, "heartbeat": heartbeat})
        if "collect" in args:
            return ContainerResult(0, json.dumps({"ok": True, "artifacts": {"solver.py": "print(1)\n", "worker.json": "{}"}, "rejected": []}), "", 0.5)
        return ContainerResult(0, "", "", 1.5)

    monkeypatch.setattr(runtime, "command", command)
    monkeypatch.setattr(runtime, "inspect", inspect)
    monkeypatch.setattr(containers, "bounded_process", bounded)
    state_path = tmp_path / "run" / "agent_container.json"
    result = runtime.run("fixture-image", ["agent", "codex-cli"], payload={"target": "solver.py"}, state_path=state_path, timeout=900, network="bridge",
                         volumes={"auth-volume": "/auth"}, artifacts=["solver.py", "worker.json"], public_metadata={"backend": "codex-cli"})
    records = tree(state_path.parent, tmp_path)
    return {"commands": [[str(item).replace(str(tmp_path), "<tmp>") for item in entry] for entry in commands],
            "processes": [json.loads(json.dumps(entry).replace(str(tmp_path), "<tmp>")) for entry in processes],
            "records": records, "result": json.loads(json.dumps(asdict(result)).replace(str(tmp_path), "<tmp>"))}


class FakeRuntime:
    """The container boundary of DockerCliClient; it records the job the worker would receive."""

    def __init__(self, root, backend, contents="def solve():\n    return 1\n", misplaced=None):
        self.seccomp_profile = root / "profile.json"
        self.seccomp_profile.write_text("{}")
        self.backend, self.contents, self.misplaced = backend, contents, misplaced or {}
        self.jobs, self.runs = [], []

    def ensure_volume(self, *args, **kwargs):
        pass

    def run(self, image, command, **kwargs):
        self.runs.append({"image": image, "command": command, **{key: (str(value) if isinstance(value, Path) else value) for key, value in kwargs.items() if key != "payload"}})
        kwargs["state_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["state_path"].write_text("{}")
        if command == ["preflight"]:
            value = {"ok": True, "versions": {"codex": CLI_VERSIONS["codex-cli"], "claude": CLI_VERSIONS["claude-code-cli"], "agy": CLI_VERSIONS["agy-cli"]}, "worker_files_sha256": docker_client.worker_file_hashes()}
            return ContainerResult(0, "", "", 0, artifacts={"worker.json": json.dumps(value)})
        job = dict(kwargs["payload"])
        self.jobs.append(job)
        worker = {"cli_version": CLI_VERSIONS[self.backend], "requested_model": "fixture-model", "target": job["target"], "returncode": 0, "terminal_state": "completed", "error": None,
                  "usage": {"input_tokens": 10, "output_tokens": 3}, "usage_available": True, "warnings": [], "observed_models": ["fixture-model"], "agent_seconds": 1.25}
        return ContainerResult(0, "", "", 2, artifacts={job["target"]: self.contents, "events.jsonl": "", "stderr.log": "diagnostic\n", "response.md": "done", "worker.json": json.dumps(worker),
                                                         "delivery.json": json.dumps({"rejected_target": None, "misplaced": self.misplaced})}, metadata={"cleanup_verified": True, "image_id": image})


def capture_client(tmp_path, backend, *, contents="def solve():\n    return 1\n", misplaced=None) -> dict:
    docker_client._PREFLIGHTS.clear()
    runtime = FakeRuntime(tmp_path, backend, contents=contents, misplaced=misplaced)
    case = tmp_path / "case"
    attempt = case / "attempt_01"
    attempt.mkdir(parents=True)
    (case / "data.json").write_text('{"minimum": 1}\n')
    target = attempt / "solver.py"
    target.write_text("# previous\n")
    client = DockerCliClient("fixture-model", backend=backend, image_id="sha256:fixture", runtime=runtime, evidence_root=tmp_path / "evidence", effort="high", timeout=900,
                             credential_provider=lambda _: {"secret": "PRIVATE_CREDENTIAL_VALUE_0123456789"})
    result = client.edit_file(target, "Solve the task.\n\nUse ../../data.json.", context=[case / "data.json"], log_dir=attempt / "calls/one", step="agent")
    scrub = lambda text: text.replace(str(tmp_path), "<tmp>")
    docker_client._PREFLIGHTS.clear()
    return {"jobs": json.loads(scrub(json.dumps(runtime.jobs))), "runs": json.loads(scrub(json.dumps(runtime.runs, default=str))), "protocol": client.protocol(),
            "files": tree(attempt, tmp_path), "result": asdict(result)}


def capture_prompt(tmp_path) -> dict:
    out = {}
    for absolute in (False, True):
        log = tmp_path / f"prompt_{absolute}"
        log.mkdir()
        effective = record_workspace_prompt(Path("/work"), "solver.py", "Edit only solver.py. All task context is supplied below.\n\nBody", log, "agent", old_absolute_note=absolute)
        out[str(absolute)] = {"effective": effective, "files": tree(log, tmp_path)}
    return out


def captures(tmp_path, monkeypatch) -> dict:
    values = {"runtime": capture_runtime(tmp_path / "runtime", monkeypatch)}
    for backend in CLI_VERSIONS:
        values["client:" + backend] = capture_client(tmp_path / backend, backend)
    values["client:misplaced"] = capture_client(tmp_path / "misplaced", "codex-cli", contents="", misplaced={"nested/solver.py": "print(2)\n"})
    values["prompt"] = capture_prompt(tmp_path)
    return values


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    tmp_path = tmp_path.resolve()
    for name in ("runtime", *CLI_VERSIONS, "misplaced"):
        (tmp_path / name).mkdir()
    return tmp_path, monkeypatch


def test_single_file_path_matches_the_pre_multifile_golden_records(isolated):
    tmp_path, monkeypatch = isolated
    observed = {key: canonical(value) for key, value in captures(tmp_path, monkeypatch).items()}
    assert observed == GOLDEN


def test_paper_worker_sources_are_unchanged():
    """The paper image's worker files (hashed by its preflight and the frozen runtime evidence) are untouched."""
    assert docker_client.worker_file_hashes() == PAPER_WORKER_FILES
    assert docker_client.IMAGE_TAG == "decompbench-agents-v2:20261013"


PAPER_WORKER_FILES = {
    "worker.py": "e0d7fb17196fdf4565ee5331ea2e360e8d42616887d8bf46cf2859046e3c3126",
    "agy_credentials.py": "4edd5bcc54fdaf461e48ccc83a502c3e752ca223e7dccf89b489c694678c8126",
    "safe_artifacts.py": "2cd0987da10a7155dfe918e7d70259c427b74095a457b14e2569be3a2f28a55d",
    "cli_events.py": "1ad60fb6df3cb7cc5233c8e4eb60d3e8c2ef5dae1f779768517bd4c5beb4c975",
}
