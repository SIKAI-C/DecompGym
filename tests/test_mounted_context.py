"""The mounted-files context delivery (prompt_protocol.context_delivery = "mounted-files-v1") and its structured prompt.

No Docker engine, model call or optimizer is used: the Docker CLI and the container boundary are recorders, solvers are
trivial host Python. The paper's pasted protocol stays byte-identical (tests/test_single_file_golden.py and the pinned
digest below); the real mounts are checked by the opt-in tests/test_docker_worker_integration.py and by
scripts/context_delivery_check.py.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import uuid

import pytest

from src.agent import container as containers
from src.agent import docker_client
from src.agent.container import ContainerResult, DockerRuntime, archive_manifest, context_archive
from src.agent.docker_client import CLI_VERSIONS, DockerCliClient
from src.agent.mounted_context import MAX_OUTPUT_JSON_BYTES, mounted_files, volume_contents
from src.agent.prompts.builder import LOCATION_NOTE, PromptBuilder
from src.agent.prompts.method import COMMON_REQUIREMENTS, DECOMPOSITION_POLICY, DECOMPOSITION_POLICY_V3, method_requirements
from src.agent.prompts.mounted import (CONTEXT_DELIVERY_MOUNTED, DATA_START, HISTORY_DIR, HISTORY_USE, INSTANCE_DIR, JUDGE_CONTEXT_DELIVERY, MOUNTED_WORKSPACE_POLICY, TASK_HEADINGS, TASK_LEAD,
                                       TASK_RULES, mount_location, task_sections)
from src.agent.specs import MOUNT_LAYOUT, REVIEW_LAYOUT
from src.agent.prompts.policy import record_workspace_prompt
from src.agent.result import AgentResult
from src.experiments import campaign
from src.experiments.data_context import DATA_CONTEXT_PREVIEW, accepted_prompt_protocol, context_delivery_policy
from src.experiments.paired_runner import PairedRunner
from src.experiments.plan import load_plan
from src.experiments.sandbox import OUTPUT_LIMITS, OUTPUT_POLICY_32MIB, NativeSandboxExecutor
from src.utils.errors import ProtocolError
from src.utils.families import FAMILIES, SMALL_SCALE_CAMPAIGN_DATA, dataset_roots
from src.utils.records import digest, identity, read, save
from host_solver import host_solver

ROOT = Path(__file__).resolve().parents[1]
COMBINED = ROOT / SMALL_SCALE_CAMPAIGN_DATA
MOUNTED = CONTEXT_DELIVERY_MOUNTED


def combined_root() -> Path:
    """The combined small-scale root (outputs/ is not part of the repository): missing fails unless explicitly opted out."""
    if not COMBINED.is_dir():
        message = f"{COMBINED} is not present (build it with scripts/build_small_scale_data.py build)"
        if os.environ.get("DECOMPGYM_ALLOW_MISSING_REFERENCE_DATA") == "1":
            pytest.skip(message)
        pytest.fail(message)
    return COMBINED


@pytest.fixture(autouse=True)
def independent_test_auth_lock(monkeypatch):
    monkeypatch.setattr("src.experiments.paired_runner.runtime_lock", lambda name: nullcontext())
    docker_client._PREFLIGHTS.clear()
    yield
    docker_client._PREFLIGHTS.clear()


# ---------------------------------------------------------------------------------------------------------------------
# The pasted protocol is unchanged.

# sha256 over every pasted effective prompt (3 backends; generation, regeneration with history, reflection with and
# without the objective feedback, judge) of the 135 tracked instances of data/ and data_harder/ under the paper's v2
# contract, captured with `pasted_prompt_digest` from a pristine export of commit 6b51773 (before this protocol existed).
PASTED_GOLDEN = "f6555277ebeaddfd3c56b557b7ece6711e1ab4c85f8e596e46bc171b41c5fe26"


def pasted_prompt_digest(scratch: Path) -> str:
    from src.agent.prompts.judge import algorithm_review_prompt
    from src.agent.prompts.snapshot import render_case_prompt

    history = scratch / "attempts"
    for index in (1, 2):
        directory = history / f"attempt_{index:02d}"
        directory.mkdir(parents=True)
        for name, text in (("solver.py", f"print({index})\n"), ("debug.md", f"Reflection {index}\n"), ("exec.log", "log\n"), ("alg_verifier.md", "VERDICT: PASS\nNOTES: x\n"), ("output.json", '{"status":"feasible"}')):
            (directory / name).write_text(text)
    (history / "attempt_01/reflection_feedback.md").write_text("feedback\n")
    regeneration = [history / f"attempt_{i:02d}" / name for i in (1, 2) for name in ("solver.py", "debug.md", "reflection_feedback.md")]
    reflection = [history / "attempt_02" / name for name in ("solver.py", "exec.log", "alg_verifier.md", "output.json")]
    hashes = []
    for root in ("data", "data_harder"):
        for path in sorted((ROOT / root).glob("*/*/*/data.json")):
            suite, problem = path.relative_to(ROOT / root).parts[:2]
            rendered = render_case_prompt(suite, problem, json.loads(path.read_text()), DECOMPOSITION_POLICY)
            generation = PromptBuilder.build_generation_prompt(rendered, solver_timeout=60, solver_threads=1)
            calls = [("solver.py", generation, [path]), ("solver.py", generation, [path] + regeneration),
                     ("debug.md", PromptBuilder.build_reflection_prompt(generation, objective_quality_failed=False), [path] + reflection),
                     ("debug.md", PromptBuilder.build_reflection_prompt(generation, objective_quality_failed=True), [path] + reflection),
                     ("alg_verifier.md", algorithm_review_prompt(suite, DECOMPOSITION_POLICY), reflection[:1])]
            for backend in CLI_VERSIONS:
                client = DockerCliClient("m", backend=backend, image_id="sha256:x", runtime=None, evidence_root=None)
                for index, (target, prompt, context) in enumerate(calls):
                    log = scratch / f"log_{root}_{path.parent.name}_{backend}_{index}"
                    log.mkdir()
                    hashes.append(hashlib.sha256(record_workspace_prompt(Path("/work"), target, client.compose_prompt(target, prompt, context), log, "s",
                                                                         old_absolute_note=backend == "agy-cli").encode()).hexdigest())
    return hashlib.sha256(json.dumps(hashes).encode()).hexdigest()


def test_pasted_prompts_of_the_paper_and_harder_data_are_byte_identical(tmp_path):
    assert pasted_prompt_digest(tmp_path) == PASTED_GOLDEN


def test_default_settings_keep_every_paper_record():
    client = DockerCliClient("m", backend="codex-cli", image_id="sha256:x", runtime=None, evidence_root=None)
    assert client.context_delivery is None and "context_delivery" not in client.protocol()
    runner = PairedRunner(client=None, judge_client=None, solver_executor=None)
    assert runner.context_delivery is None
    for path in ("paper.json", "harder_small_all.json"):
        plan = load_plan(ROOT / "configs" / path)
        assert context_delivery_policy(plan["prompt_protocol"]) is None and "solver_output_policy" not in plan
    executor = object.__new__(NativeSandboxExecutor)
    assert (executor.output_policy, executor.output_limit) == (None, 4 * 1024 * 1024)


# ---------------------------------------------------------------------------------------------------------------------
# Layout and Part 1.

def history_case(tmp_path, attempts=1, names=("solver.py", "debug.md")):
    case = tmp_path / "case"
    (case / "attempts").mkdir(parents=True)
    (case / "data.json").write_text('{"m": 2}\n')
    paths = [case / "data.json"]
    for index in range(1, attempts + 1):
        directory = case / "attempts" / f"attempt_{index:02d}"
        directory.mkdir()
        for name in names:
            (directory / name).write_text(f"CONTENT-{name}-{index}\n")
            paths.append(directory / name)
    return case, paths


def test_layout_maps_instance_data_and_history_files_and_refuses_anything_else(tmp_path):
    case, paths = history_case(tmp_path, attempts=2, names=("solver.py", "debug.md", "reflection_feedback.md", "exec.log", "alg_verifier.md", "output.json"))
    files = mounted_files(paths + [case / "absent.json"])
    assert [item.path for item in files][:3] == [f"{INSTANCE_DIR}/data.json", f"{HISTORY_DIR}/attempt_01/solver.py", f"{HISTORY_DIR}/attempt_01/debug.md"]
    assert all(item.sha256 == digest(path) and item.size == path.stat().st_size for item, path in zip(files, paths))
    assert volume_contents(files)[INSTANCE_DIR] == {"data.json": b'{"m": 2}\n'} and "attempt_02/output.json" in volume_contents(files)[HISTORY_DIR]
    for bad in (case / "meta.json", case / "gold_monolithic.json", case / "attempts/attempt_01/source.json", case / "attempts/attempt_01/data.json", case / "prompt.txt"):
        bad.write_text("{}")
        with pytest.raises(ProtocolError):
            mount_location(bad)
        with pytest.raises(ProtocolError):
            mounted_files([case / "data.json", bad])
    other = tmp_path / "other"
    other.mkdir()
    (other / "data.json").write_text("{}")
    with pytest.raises(ProtocolError, match="same mounted path"):
        mounted_files([case / "data.json", other / "data.json"])
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "data.json").symlink_to(case / "data.json")
    with pytest.raises(Exception, match="safely"):
        mounted_files([linked / "data.json"])  # links are never followed


@pytest.mark.parametrize("backend", list(CLI_VERSIONS))
def test_part_one_states_the_one_writable_target_and_the_runtime_paths(tmp_path, backend):
    case, paths = history_case(tmp_path)
    client = DockerCliClient("m", backend=backend, image_id="sha256:x", runtime=None, evidence_root=None, context_delivery=MOUNTED)
    for target, context in (("solver.py", paths[:1]), ("solver.py", paths), ("debug.md", paths)):
        composed, note = client.compose_mounted_prompt(target, "# Part 2: Task", mounted_files(context))
        log = tmp_path / f"log_{target}_{len(context)}"
        log.mkdir()
        text = record_workspace_prompt(Path("/work"), target, composed, log, "s", old_absolute_note=backend == "agy-cli", note=note, version=MOUNTED_WORKSPACE_POLICY)
        headings = [line for line in text.splitlines() if line.startswith("#")]
        assert headings == ["# Part 1: Workspace and files", "## File to modify", "## Input data", "## Output file", "## Files from previous attempts", "## No other task files", "# Part 2: Task"]
        assert (f"## File to modify\nOnly edit the existing {target} in the current working directory. That file is /work/{target}; do not create, rename or delete any file. "
                "Every other path in this workspace is read-only context.") in text
        assert "physical line" not in text and "soft wrapping" not in text  # the paper wrapper's prose rule is not part of the mounted prompt
        assert f"Only /work/{target} is collected after this call: anything written elsewhere is ignored." in text
        assert "solver.py must read its input from ../../data.json relative to that working directory" in text
        assert "solver.py must write its result to output.json in its working directory" in text and "a missing or misplaced output.json is an execution failure" in text
        assert f"{INSTANCE_DIR}/data.json (9 bytes)" in text
        for item in mounted_files(context)[1:]:
            assert f"  - {item.path} ({item.size} bytes): " in text
        assert ("No files from previous attempts are provided." in text) == (len(context) == 1)
        assert ("(the attempt under review)" in text) == (target == "debug.md")
        # Only a regeneration with history is told what the history is (requirements item 25).
        assert (HISTORY_USE in text) == (target == "solver.py" and len(context) > 1)
        # Every backend and call is told that nothing can be run here and what it has to write (requirements item 28);
        # only Antigravity, whose denied command tool ends its run, is told to use its file tools.
        assert ("This workspace has no gurobipy, NumPy or other solver packages, so solver.py cannot be run or tested here; the harness runs it later in its own "
                f"environment, where gurobipy and NumPy are available. Here you only need to read the files listed above and write /work/{target}.") in text
        assert ("Read and write files with your file tools; this workspace does not allow command execution." in text) == (backend == "agy-cli")
        assert ("Do not use memory, skills, subagents or prior conversations." in text) == (backend == "agy-cli")
        # The boundary sentence names task files only: the container also holds the OS and the agent CLI's installation.
        assert (f"No other task files are provided: apart from the files listed above and /work/{target}, the workspace contains no data, "
                "references or results of this or any other instance; do not inspect other directories, use external tools or services, "
                "or search for reference implementations.") in text and "nothing else is available" not in text
        assert "All task context is supplied below" not in text and "BEGIN CONTEXT" not in text
        record = read(log / "s_workspace.json")
        assert record["version"] == MOUNTED_WORKSPACE_POLICY and record["note"] == note
    with pytest.raises(ValueError):
        client.compose_mounted_prompt("alg_verifier.md", "x", mounted_files(paths[:1]))


# ---------------------------------------------------------------------------------------------------------------------
# Part 2: the prompt.txt pieces regrouped, nothing reworded or dropped.

def sentences(text: str) -> list[str]:
    body = " ".join(line for line in text.splitlines() if not line.startswith("#"))
    return sorted(part for part in re.split(r"(?<=[.!?:])\s+", " ".join(body.split())) if part)


def check_regrouping(prompt: str, part2: str, suite: str, contract: str) -> None:
    headings = [line for line in part2.splitlines() if line.startswith("#")]
    assert headings == ["# Part 2: Task"] + [f"## {heading}" for heading in TASK_HEADINGS]
    # Every line of prompt.txt is kept verbatim, in order within its section; only the Task line is cut after its two rule sentences.
    task_line = TASK_RULES + " " + TASK_LEAD
    old = [line for line in prompt.splitlines() if line.strip()]
    new = [line for line in part2.splitlines() if line.strip() and not line.startswith("#")]
    expected = sorted([TASK_RULES, TASK_LEAD] + [line for line in old if line != task_line])
    # The only added lines are the harness rules, which the pasted protocol appends after the prompt.
    harness = [line for line in new if line.startswith(("Implement a standalone solver.py", "The harness limits the entire solver.py"))]
    assert len(harness) == 2 and old.count(task_line) == 1
    assert sorted(line for line in new if line not in harness) == expected
    assert sentences(part2) == sorted(sentences(prompt) + sentences("\n".join(harness)))
    assert method_requirements(suite, contract).split("\n\n")[1] in part2 and all(paragraph in part2 for paragraph in COMMON_REQUIREMENTS)


def test_every_stored_instance_regroups_without_changing_a_sentence():
    root = combined_root()
    plan = load_plan(ROOT / "configs/small_scale.json")
    assert sorted(plan["cohort"]["families"]) == sorted(FAMILIES) == sorted(DATA_START)
    checked = 0
    for path in sorted(root.glob("*/*/*/prompt.txt")):
        suite, problem = path.relative_to(root).parts[:2]
        prompt = path.read_text()
        part2 = PromptBuilder.build_structured_generation_prompt(prompt, suite=suite, problem=problem, contract=DECOMPOSITION_POLICY_V3, solver_timeout=60, solver_threads=1)
        check_regrouping(prompt, part2, suite, DECOMPOSITION_POLICY_V3)
        sections = task_sections(prompt, suite=suite, problem=problem, contract=DECOMPOSITION_POLICY_V3)
        assert sections["Problem"][0].startswith("You are an expert operations research engineer and Python programmer.")
        assert sections["Data"][0].startswith(DATA_START[problem]) and "data.json" in sections["Data"][0]
        assert sections["Output"][0].startswith("Task:\n" + TASK_LEAD) and "output.json" in sections["Output"][0]
        reflection = PromptBuilder.build_structured_reflection_prompt(prompt, suite=suite, problem=problem, contract=DECOMPOSITION_POLICY_V3, solver_timeout=60, solver_threads=1, objective_quality_failed=True)
        assert reflection.startswith(part2.split("\n\n", 1)[0] + "\n\nYou are making an attempt to solve the following problem.\n\n## Problem")
        assert reflection.split("\n\n# Part 3: Reflection\n")[0].replace("\n\nYou are making an attempt to solve the following problem.", "", 1) == part2
        checked += 1
    assert checked == 180


def test_a_prompt_outside_the_contract_or_layout_is_refused():
    prompt = (ROOT / "data/bd/ufl/ufl_small_seed000/prompt.txt").read_text()  # the paper's v2 render
    with pytest.raises(ProtocolError, match="method contract"):
        task_sections(prompt, suite="bd", problem="ufl", contract=DECOMPOSITION_POLICY_V3)
    assert task_sections(prompt, suite="bd", problem="ufl", contract=DECOMPOSITION_POLICY)["Method requirements"]
    with pytest.raises(ProtocolError, match="Task block"):
        task_sections(prompt.replace(TASK_RULES, "Assume nothing."), suite="bd", problem="ufl", contract=DECOMPOSITION_POLICY)
    with pytest.raises(ProtocolError, match="data marker"):
        task_sections(prompt.replace("\n- There are ", "\n* There are "), suite="bd", problem="ufl", contract=DECOMPOSITION_POLICY)


# ---------------------------------------------------------------------------------------------------------------------
# The client: mounted files, evidence, only the declared target collected.

class FakeRuntime:
    def __init__(self, root, backend, contents="def solve():\n    return 1\n", misplaced=None):
        self.seccomp_profile = root / "profile.json"
        self.seccomp_profile.write_text("{}")
        self.backend, self.contents, self.misplaced = backend, contents, misplaced or {}
        self.calls = []

    def ensure_volume(self, *args, **kwargs):
        pass

    def run(self, image, command, **kwargs):
        kwargs["state_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["state_path"].write_text("{}")
        if command == ["preflight"]:
            value = {"ok": True, "versions": {"codex": CLI_VERSIONS["codex-cli"], "claude": CLI_VERSIONS["claude-code-cli"], "agy": CLI_VERSIONS["agy-cli"]}, "worker_files_sha256": docker_client.worker_file_hashes()}
            return ContainerResult(0, "", "", 0, artifacts={"worker.json": json.dumps(value)})
        self.calls.append(kwargs)
        job = kwargs["payload"]
        worker = {"cli_version": CLI_VERSIONS[self.backend], "requested_model": "m", "target": job["target"], "returncode": 0, "terminal_state": "completed", "error": None,
                  "usage": {"input_tokens": 1}, "usage_available": True, "warnings": [], "observed_models": ["m"], "agent_seconds": 1.0}
        return ContainerResult(0, "", "", 1, artifacts={job["target"]: self.contents, "worker.json": json.dumps(worker), "delivery.json": json.dumps({"rejected_target": None, "misplaced": self.misplaced})},
                               metadata={"cleanup_verified": True, "context_volumes": {INSTANCE_DIR: "v"}, "mounts": []})


@pytest.mark.parametrize("backend", list(CLI_VERSIONS))
def test_mounted_call_sends_the_files_as_read_only_context_and_records_their_sha256(tmp_path, backend):
    runtime = FakeRuntime(tmp_path, backend)
    case, paths = history_case(tmp_path)
    target = case / "attempts/attempt_02/solver.py"
    target.parent.mkdir()
    target.write_text("")
    client = DockerCliClient("m", backend=backend, image_id="sha256:x", runtime=runtime, evidence_root=tmp_path / "evidence", credential_provider=lambda _: {}, context_delivery=MOUNTED)
    result = client.edit_file(target, "# Part 2: Task\n\nTASK", context=paths, log_dir=tmp_path / "call")
    assert result.ok and target.read_text() == runtime.contents
    call = runtime.calls[0]
    assert call["artifacts"] == ["solver.py", "events.jsonl", "stderr.log", "response.md", "worker.json", "delivery.json"]  # only the declared target besides the worker records
    assert call["context"] == {INSTANCE_DIR: {"data.json": (case / "data.json").read_bytes()},
                               HISTORY_DIR: {f"attempt_01/{name}": (case / "attempts/attempt_01" / name).read_bytes() for name in ("solver.py", "debug.md")}}
    assert "CONTENT-" not in call["payload"]["prompt"] and '"m": 2' not in call["payload"]["prompt"]  # no file content is pasted
    evidence = read(tmp_path / "call/agent_context.json")
    assert evidence["writable_target"] == "/work/solver.py" and evidence["read_only_mounts"] == [HISTORY_DIR, INSTANCE_DIR]
    assert [(entry["path"], entry["sha256"]) for entry in evidence["files"]] == [(item.path, item.sha256) for item in mounted_files(paths)]
    invocation = read(tmp_path / "call/agent_invocation.json")
    assert invocation["context_delivery"] == MOUNTED and invocation["context_volumes"] == {INSTANCE_DIR: "v"}
    assert read(tmp_path / "call/agent_workspace.json")["version"] == MOUNTED_WORKSPACE_POLICY


def test_writes_outside_the_target_never_rescue_a_call_and_mounted_copies_are_not_deliveries(tmp_path):
    runtime = FakeRuntime(tmp_path, "codex-cli", contents="", misplaced={"history/attempt_01/solver.py": "CONTENT-solver.py-1\n", "scratch/solver.py": "print(2)\n"})
    case, paths = history_case(tmp_path)
    target = case / "attempts/attempt_02/solver.py"
    target.parent.mkdir()
    target.write_text("")
    client = DockerCliClient("m", backend="codex-cli", image_id="sha256:x", runtime=runtime, evidence_root=tmp_path / "evidence", credential_provider=lambda _: {}, context_delivery=MOUNTED)
    result = client.edit_file(target, "# Part 2: Task", context=paths, log_dir=tmp_path / "call")
    assert not result.ok and result.error == "Completed CLI did not deliver the requested file" and target.read_text() == ""
    failure = read(tmp_path / "call/agent_delivery_failure.json")
    assert [item["relative_path"] for item in failure["misplaced_files"]] == ["scratch/solver.py"]  # evidence only, never promoted
    assert read(tmp_path / "call/agent_context_copies_ignored.json") == {"ignored_mounted_copies": ["history/attempt_01/solver.py"]}
    assert not (tmp_path / "call/unexpected_outputs/history").exists()


def test_paired_runner_scores_a_misplaced_delivery_as_a_failed_generation(tmp_path):
    source = tmp_path / "data/bd/ufl/ufl_small_seed000"
    source.mkdir(parents=True)
    (source / "data.json").write_text(json.dumps({"m": 2, "n": 1, "f": [2, 3], "D": [[4], [1]]}))
    (source / "prompt.txt").write_text(UFL_PROMPT)
    (source / "meta.json").write_text('{"ground_truth_objective":4}')
    runtime = FakeRuntime(tmp_path, "codex-cli", contents="", misplaced={"elsewhere/solver.py": "print(2)\n"})
    client = DockerCliClient("m", backend="codex-cli", image_id="sha256:x", runtime=runtime, evidence_root=tmp_path / "evidence", credential_provider=lambda _: {}, context_delivery=MOUNTED)
    judge = MountedClient()
    judge.context_delivery = None
    runner = PairedRunner(client=client, judge_client=judge, reflection_client=client, model="m", suite="bd", problem=["ufl"], max_attempts=1, solver_executor=lambda *_: pytest.fail("no solver may run"),
                          decomposition_contract=DECOMPOSITION_POLICY_V3, context_delivery=MOUNTED)
    summaries = runner.run_pair(source, tmp_path / "runs")
    assert all(summary["attempts"][0]["status"] == "agent_failed" for summary in summaries.values())


def test_an_author_layout_client_refuses_the_judge_target(tmp_path):
    # The mounted judge has its own layout (tests/test_mounted_judge.py); the author's layout never edits alg_verifier.md.
    client = DockerCliClient("m", backend="codex-cli", image_id="sha256:x", runtime=FakeRuntime(tmp_path, "codex-cli"), evidence_root=tmp_path, context_delivery=MOUNTED)
    target = tmp_path / "alg_verifier.md"
    target.write_text("")
    with pytest.raises(ValueError, match="only with the review layout"):
        client.edit_file(target, "x", context=[], log_dir=tmp_path / "call", step="llm_verifier")
    with pytest.raises(ValueError):
        DockerCliClient("m", backend="codex-cli", image_id="sha256:x", runtime=None, evidence_root=None, context_delivery="pasted-v0")


# ---------------------------------------------------------------------------------------------------------------------
# The runtime: fresh read-only volumes, filled through a never-started container and checked before the agent starts.

def capture(tmp_path, monkeypatch, *, read_only=True, tamper=False, context=None, interrupt=None):
    runtime = DockerRuntime(seccomp_profile=tmp_path / "profile.json")
    (tmp_path / "profile.json").write_text("{}")
    commands, transfers, processes, created, removed = [], [], [], set(), set()
    monkeypatch.setattr(containers.uuid, "uuid4", lambda: uuid.UUID(int=7))
    monkeypatch.setattr(containers.time, "time", lambda: 1000.0)
    context = context or {INSTANCE_DIR: {"data.json": b'{"m": 2}'}, HISTORY_DIR: {"attempt_01/solver.py": b"print(1)\n", "attempt_01/debug.md": b"notes\n"}}
    copied = {}

    def command(*args, timeout=30):
        commands.append(list(args))
        if args[:2] == ("image", "inspect"):
            return subprocess.CompletedProcess(args, 0, "sha256:fixture\n", "")
        if args[:2] == ("volume", "inspect"):
            purpose = "v2-ephemeral-context" if "context" in args[2] else "v2-ephemeral-worker"
            return subprocess.CompletedProcess(args, 0, json.dumps({runtime.LABEL: purpose}), "")
        if args[0] == "create":
            created.add(args[args.index("--name") + 1])
            return subprocess.CompletedProcess(args, 0, "container-id\n", "")
        if args[0] == "rm":
            removed.add(args[-1])
        return subprocess.CompletedProcess(args, 0, "", "")

    def transfer(*args, data=None, timeout=120):
        transfers.append(list(args))
        if interrupt == "loading" and len(transfers) == 2:
            raise KeyboardInterrupt("controller stopped while the context was loading")
        if args[1] == "-":
            copied[args[2].split(":", 1)[1]] = data
            return subprocess.CompletedProcess(args, 0, b"", b"")
        destination = args[1].split(":", 1)[1]
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as out, tarfile.open(fileobj=io.BytesIO(copied[destination])) as source:
            out.addfile(_directory(destination.rsplit("/", 1)[1]))
            for member in source.getmembers():
                contents = source.extractfile(member).read() if member.isfile() else None
                if tamper and contents is not None:
                    contents = b"tampered"
                    member.size = len(contents)
                member.name = destination.rsplit("/", 1)[1] + "/" + member.name
                out.addfile(member, io.BytesIO(contents) if contents is not None else None)
        return subprocess.CompletedProcess(args, 0, archive.getvalue(), b"")

    mounts = [{"Type": "volume", "Name": "decompbench-v2-work-" + "0" * 31 + "7", "Destination": "/work", "RW": True}]
    mounts += [{"Type": "volume", "Name": f"decompbench-v2-context-{name}-" + "0" * 31 + "7", "Destination": f"/work/{name}", "RW": not read_only} for name in ("history", "instance")]
    states = iter([{"HostConfig": {"Privileged": False, "ReadonlyRootfs": True, "Binds": None, "SecurityOpt": ["no-new-privileges"]}, "Mounts": mounts},
                   {"State": {"Running": False, "ExitCode": 0, "OOMKilled": False}}])

    def inspect(name):
        if name not in created or name in removed:
            return None
        return {"State": {"Running": False, "Status": "created"}} if name.endswith("-context") else next(states, None)

    def bounded(args, *, stdin="", timeout, max_output_bytes=8 * 1024 * 1024, heartbeat=False):
        processes.append([str(item) for item in args])
        if interrupt == "running" and "start" in args:
            raise KeyboardInterrupt("controller stopped while the worker ran")
        if "collect" in args:
            return ContainerResult(0, json.dumps({"ok": True, "artifacts": {"solver.py": "print(1)\n"}, "rejected": []}), "", 0.5)
        return ContainerResult(0, "", "", 1.5)

    monkeypatch.setattr(runtime, "command", command)
    monkeypatch.setattr(runtime, "transfer", transfer)
    monkeypatch.setattr(runtime, "inspect", inspect)
    monkeypatch.setattr(containers, "bounded_process", bounded)
    state_path = tmp_path / "run/agent_container.json"
    run = lambda: runtime.run("fixture-image", ["agent", "codex-cli"], payload={"target": "solver.py"}, state_path=state_path, timeout=900, network="bridge",
                              volumes={"auth-volume": "/auth"}, artifacts=["solver.py"], context=context)
    return run, commands, transfers, state_path, copied, processes


def _directory(name):
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    return info


def test_runtime_fills_fresh_context_volumes_and_mounts_them_read_only(tmp_path, monkeypatch):
    run, commands, transfers, state_path, copied, processes = capture(tmp_path, monkeypatch)
    result = run()
    identifier = "0" * 31 + "7"
    loader = [command for command in commands if command[0] == "create" and f"decompbench-v2-{identifier}-context" in command]
    agent = [command for command in commands if command[0] == "create" and f"decompbench-v2-{identifier}-context" not in command]
    assert len(loader) == 1 and len(agent) == 1
    assert any("start" in process for process in processes) and not any(f"decompbench-v2-{identifier}-context" in process for process in processes)  # the loader never starts
    assert f"type=volume,src=decompbench-v2-context-instance-{identifier},dst={INSTANCE_DIR}" in loader[0] and "--network" in loader[0]
    assert f"type=volume,src=decompbench-v2-context-instance-{identifier},dst={INSTANCE_DIR},readonly" in agent[0]
    assert f"type=volume,src=decompbench-v2-context-history-{identifier},dst={HISTORY_DIR},readonly" in agent[0]
    assert transfers == [["cp", "-", f"decompbench-v2-{identifier}-context:{HISTORY_DIR}"], ["cp", f"decompbench-v2-{identifier}-context:{HISTORY_DIR}", "-"],
                         ["cp", "-", f"decompbench-v2-{identifier}-context:{INSTANCE_DIR}"], ["cp", f"decompbench-v2-{identifier}-context:{INSTANCE_DIR}", "-"]]
    with tarfile.open(fileobj=io.BytesIO(copied[HISTORY_DIR])) as archive:
        assert {member.name: (member.mode, member.type) for member in archive.getmembers()} == {"attempt_01": (0o555, tarfile.DIRTYPE), "attempt_01/debug.md": (0o444, tarfile.REGTYPE), "attempt_01/solver.py": (0o444, tarfile.REGTYPE)}
    state = json.loads(state_path.read_text())
    assert state["context_manifest"][INSTANCE_DIR] == {"data.json": hashlib.sha256(b'{"m": 2}').hexdigest()}
    assert result.metadata["context_volumes"] == state["context_volumes"]
    removed = [command[2] for command in commands if command[:2] == ["volume", "rm"]]
    assert sorted(removed) == sorted([f"decompbench-v2-work-{identifier}", f"decompbench-v2-record-{identifier}", f"decompbench-v2-context-history-{identifier}", f"decompbench-v2-context-instance-{identifier}"])
    assert ["rm", "--force", f"decompbench-v2-{identifier}-context"] in commands


def test_runtime_refuses_writable_or_altered_context_before_the_agent_starts(tmp_path, monkeypatch):
    for case, kwargs, error in (("rw", {"read_only": False}, "not read-only"), ("tamper", {"tamper": True}, "differs from its manifest")):
        directory = tmp_path / case
        directory.mkdir()
        run, commands, _, _, _, processes = capture(directory, monkeypatch, **kwargs)
        with pytest.raises(Exception, match=error):
            run()
        assert not any("start" in process for process in processes)  # the agent never starts
        assert any(command[:2] == ["volume", "rm"] and "context" in command[2] for command in commands)
    (tmp_path / "bad").mkdir()
    for bad in ({"/work": {"a": b""}}, {INSTANCE_DIR: {"../meta.json": b""}}, {INSTANCE_DIR: {"/abs": b""}}, {INSTANCE_DIR: {}}, {HISTORY_DIR: {"a//b": b""}}, {INSTANCE_DIR: {"data.json": "text"}}):
        with pytest.raises(ValueError):
            capture(tmp_path / "bad", monkeypatch, context=bad)[0]()


@pytest.mark.parametrize("moment", ["loading", "running"])
def test_interrupted_calls_remove_the_loader_and_dispose_the_context_volumes_on_recovery(tmp_path, monkeypatch, moment):
    run, commands, _, state_path, _, processes = capture(tmp_path, monkeypatch, interrupt=moment)
    with pytest.raises(KeyboardInterrupt):
        run()
    identifier = "0" * 31 + "7"
    loader = f"decompbench-v2-{identifier}-context"
    assert ["rm", "--force", loader] in commands  # the loader never outlives the call that created it
    state = json.loads(state_path.read_text())
    contexts = sorted(state["context_volumes"].values())
    removed = lambda: sorted(command[2] for command in commands if command[:2] == ["volume", "rm"] and "context" in command[2])
    if moment == "loading":
        assert state["phase"] == "preparing" and removed() == contexts  # nothing to collect: disposed at once
    else:
        assert state["phase"] == "stopped" and removed() == []  # kept until the worker's records are collected
        runtime = DockerRuntime(seccomp_profile=tmp_path / "profile.json")
        monkeypatch.setattr(runtime, "command", lambda *args, timeout=30: commands.append(list(args)) or subprocess.CompletedProcess(args, 0, "", ""))
        monkeypatch.setattr(runtime, "inspect", lambda name: None)
        result = runtime.recover(state_path)
        assert result.artifacts == {"solver.py": "print(1)\n"} and removed() == contexts
        assert not any("--mount" in process and any("context" in item for item in process) for process in processes if "collect" in process)


def test_archive_round_trip_and_refusal_of_unexpected_entries():
    files = {"data.json": b"{}"}
    archive = context_archive(files)
    assert context_archive(files) == archive  # deterministic
    assert archive_manifest(archive, "/work/instance") == {"data.json": hashlib.sha256(b"{}").hexdigest()}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as out:
        link = tarfile.TarInfo("instance/meta.json")
        link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
        out.addfile(link)
    with pytest.raises(Exception, match="unexpected entry"):
        archive_manifest(buffer.getvalue(), "/work/instance")


# ---------------------------------------------------------------------------------------------------------------------
# The paired runner and the campaign under the mounted protocol.

UFL_PROMPT = (ROOT / "data/bd/ufl/ufl_small_seed000/prompt.txt").read_text().replace("Use a master problem for the complicating decisions and recourse-cost estimates.",
                                                                                   "Use a master problem for the complicating decisions and, when applicable, recourse-cost estimates.")


class MountedClient:
    context_delivery = MOUNTED

    def __init__(self, outcomes=None):
        self.calls = []
        self.outcomes = iter(outcomes or [True] * 100)

    def protocol(self):
        return {"backend": "test", "model": "fixture"}

    def edit_file(self, target, prompt, *, context, log_dir, step):
        self.calls.append((step, target, list(context), prompt))
        if step == "agent":
            text = ("import json\nfrom pathlib import Path\nPath('output.json').write_text(json.dumps({'status':'feasible','method':'benders_decomposition','y':[0,1],'x':[[0],[1]],'objective_value':4}))\n"
                    if next(self.outcomes) else "raise RuntimeError('controlled solver failure')\n")
            target.write_text(text)
        elif step == "llm_verifier":
            target.write_text("VERDICT: PASS\nNOTES: fixture")
        else:
            target.write_text("Repair the preceding solver.")
        result = AgentResult(True, 0, 0.01, {"input_tokens": 7}, terminal_state="completed", artifact_sha256=digest(target))
        save(log_dir / f"{step}_metrics.json", asdict(result))
        return result


def test_runner_records_the_layout_and_builds_structured_prompts(tmp_path):
    source = tmp_path / "data/bd/ufl/ufl_small_seed000"
    source.mkdir(parents=True)
    (source / "data.json").write_text(json.dumps({"m": 2, "n": 1, "f": [2, 3], "D": [[4], [1]]}))
    (source / "prompt.txt").write_text(UFL_PROMPT)
    (source / "meta.json").write_text('{"ground_truth_objective":4}')
    client, judge = MountedClient([False, False, True, True]), MountedClient()
    judge.context_delivery = None
    runner = PairedRunner(client=client, judge_client=judge, model="fixture", suite="bd", problem=["ufl"], solver_executor=host_solver, decomposition_contract=DECOMPOSITION_POLICY_V3, context_delivery=MOUNTED)
    runner.run_pair(source, tmp_path / "runs")
    steps = [step for step, *_ in client.calls]
    assert "reflection" in steps and steps.count("agent") >= 3
    for step, target, context, prompt in client.calls:
        assert prompt.startswith("# Part 2: Task") and LOCATION_NOTE not in prompt
        assert ("# Part 3: Reflection" in prompt) == (step == "reflection")
    regenerations = [context for step, target, context, _ in client.calls if step == "agent" and target.parents[5].name == "reflection" and target.parent.name != "attempt_01"]
    assert regenerations and all(any(path.name == "debug.md" for path in context) for context in regenerations)
    for path in (tmp_path / "runs").rglob("request.json"):
        request = read(path)
        if request["step"] == "llm_verifier":
            assert "context_delivery" not in request
        else:
            files = request["context_delivery"]["files"]
            assert request["context_delivery"]["policy"] == MOUNTED and files[0]["path"] == f"{INSTANCE_DIR}/data.json"
            assert all(entry["path"].startswith(f"{HISTORY_DIR}/attempt_") for entry in files[1:])
            assert request["request_sha256"] == identity({k: v for k, v in request.items() if k not in ("request_sha256", "invocation_id")})
    assert all(read(path)["context_delivery"] == MOUNTED for path in (tmp_path / "runs").rglob("identity.json"))
    calls = len(client.calls)
    PairedRunner(client=client, judge_client=judge, model="fixture", suite="bd", problem=["ufl"], solver_executor=host_solver, decomposition_contract=DECOMPOSITION_POLICY_V3, context_delivery=MOUNTED).run_pair(source, tmp_path / "runs")
    assert len(client.calls) == calls  # resume reuses every committed call
    with pytest.raises(ProtocolError, match="Case protocol changed"):
        PairedRunner(client=client, judge_client=judge, model="fixture", suite="bd", problem=["ufl"], solver_executor=host_solver, decomposition_contract=DECOMPOSITION_POLICY_V3).run_pair(source, tmp_path / "runs")


def test_runner_refuses_mixed_delivery_settings(tmp_path):
    with pytest.raises(ValueError, match="no data-context preview"):
        PairedRunner(client=None, judge_client=None, solver_executor=None, data_context=DATA_CONTEXT_PREVIEW, context_delivery=MOUNTED)
    with pytest.raises(ValueError):
        PairedRunner(client=None, judge_client=None, solver_executor=None, context_delivery="pasted-v0")
    runner = PairedRunner(client=None, judge_client=None, solver_executor=None, context_delivery=MOUNTED, decomposition_contract=DECOMPOSITION_POLICY_V3)
    target = tmp_path / "attempt_01/solver.py"
    target.parent.mkdir()
    pasted = MountedClient()
    pasted.context_delivery = None
    with pytest.raises(ProtocolError, match="context delivery differs"):
        runner.edit(pasted, target, "x", [], "agent")
    with pytest.raises(ProtocolError, match="context delivery differs"):
        runner.edit(MountedClient(), tmp_path / "attempt_01/alg_verifier.md", "x", [], "llm_verifier")


def test_campaign_routes_the_mounted_protocol_and_the_output_policy(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.experiments import docker_setup
    from src.experiments import sandbox as native_solver

    root = tmp_path / "campaign"
    (root / "controller/src").mkdir(parents=True)
    monkeypatch.setattr(campaign, "REPOSITORY", root / "controller")
    plan = load_plan(ROOT / "configs/small_scale.json")
    manifest = {"model_queue": [{"model": "tested-model", "backend": "claude-code-cli", "effort": "medium"}], "instances": [{"suite": "bd", "problem": "ufl", "instance": "synthetic"}],
                "protocol_id": "fixture", "agent_timeout": 900, "judge_model": "independent-judge", "judge_effort": "high", "prompt_protocol": plan["prompt_protocol"],
                "solver_output_policy": plan["solver_output_policy"], "execution_environment": {"image_id": "sha256:fixture", "license_file": "fixture", "native_environment": {}}}
    executors = []
    monkeypatch.setattr(docker_setup, "runtime", lambda: SimpleNamespace(image_id=lambda name: name, bind_environment=lambda _: None))
    monkeypatch.setattr(native_solver, "NativeSandboxExecutor", lambda **kwargs: executors.append(kwargs) or object())
    for name, value in (("validate", lambda _: manifest), ("ensure_execution_allowed", lambda *_: None), ("require_campaign_environment", lambda *_: None),
                        ("require_backend_readiness", lambda *_: {}), ("require_backend_authentication", lambda *args, **kwargs: {}), ("pair_complete", lambda *_: False), ("state_root", lambda: tmp_path / "locks"), ("runtime_lock", lambda *_: nullcontext())):
        monkeypatch.setattr(campaign, name, value)
    monkeypatch.setattr(campaign.signal, "signal", lambda *_: None)
    runners = []

    class ObservedRunner:
        def __init__(self, **kwargs):
            runners.append(kwargs)

        def run_pair(self, *args):
            return {}

    monkeypatch.setattr(campaign, "PairedRunner", ObservedRunner)
    assert campaign.run(root) == 0
    assert executors[0]["output_policy"] == OUTPUT_POLICY_32MIB and executors[0]["evidence_schema"] is None  # a manifest without the schema: embedded records
    manifest["solver_evidence_schema"] = "native-output-by-reference-v1"
    assert campaign.run(root) == 0
    assert executors[1]["output_policy"] == OUTPUT_POLICY_32MIB and executors[1]["evidence_schema"] == "native-output-by-reference-v1"
    assert runners[0]["context_delivery"] == MOUNTED and runners[0]["data_context"] is None and runners[0]["decomposition_contract"] == DECOMPOSITION_POLICY_V3
    assert runners[0]["client"].context_delivery == MOUNTED and runners[0]["reflection_client"].context_delivery == MOUNTED
    # The small-scale protocol mounts the judge's sources under review with the review layout.
    assert runners[0]["judge_context_delivery"] == JUDGE_CONTEXT_DELIVERY
    assert runners[0]["judge_client"].context_delivery == MOUNTED and runners[0]["judge_client"].context_layout == REVIEW_LAYOUT
    assert runners[0]["client"].context_layout == MOUNT_LAYOUT
    # Without the setting (mounted author, pasted judge) the judge keeps the paper's pasted context.
    manifest["prompt_protocol"] = {key: value for key, value in plan["prompt_protocol"].items() if key != "judge_context_delivery"}
    fresh = tmp_path / "campaign-pasted-judge"
    (fresh / "controller/src").mkdir(parents=True)
    monkeypatch.setattr(campaign, "REPOSITORY", fresh / "controller")
    assert campaign.run(fresh) == 0
    assert len(runners) == 2
    assert runners[-1]["judge_context_delivery"] is None and runners[-1]["judge_client"].context_delivery is None and runners[-1]["client"].context_delivery == MOUNTED


def test_config_and_manifest_validation(tmp_path):
    plan = load_plan(ROOT / "configs/small_scale.json")
    assert context_delivery_policy(plan["prompt_protocol"]) == MOUNTED and "data_context" not in plan["prompt_protocol"]
    assert plan["solver_output_policy"] == OUTPUT_POLICY_32MIB and OUTPUT_LIMITS[OUTPUT_POLICY_32MIB] == 33_554_432 == MAX_OUTPUT_JSON_BYTES
    assert not accepted_prompt_protocol({**plan["prompt_protocol"], "data_context": DATA_CONTEXT_PREVIEW})
    assert not accepted_prompt_protocol({**plan["prompt_protocol"], "context_delivery": "mounted-files-v0"})
    for change in ({"solver_output_policy": "output-json-readback-64MiB-v1"}, {"solver_output_policy": None},
                   {"prompt_protocol": {k: v for k, v in plan["prompt_protocol"].items() if k != "context_delivery"}}):
        save(tmp_path / "bad.json", plan | change)
        with pytest.raises(ProtocolError):
            load_plan(tmp_path / "bad.json")
    allowed = {"prompt_policies": plan["prompt_policies"], "reflection_policy": plan["reflection_policy"], "prompt_protocol": plan["prompt_protocol"], "solver_output_policy": OUTPUT_POLICY_32MIB}
    (tmp_path / "root").mkdir()
    campaign.ensure_execution_allowed(tmp_path / "root", allowed)
    with pytest.raises(ProtocolError):
        campaign.ensure_execution_allowed(tmp_path / "root", allowed | {"solver_output_policy": "unknown"})
    assert (ROOT / SMALL_SCALE_CAMPAIGN_DATA).resolve() in dataset_roots()


# ---------------------------------------------------------------------------------------------------------------------
# The solver output readback limit.

def km_outputs():
    root = combined_root()
    outputs = []
    for seed in range(3):
        gold = json.loads((root / f"bd/km/km_large_seed{seed:03d}/gold_monolithic.json").read_text())
        outputs.append({"status": "optimal", "method": "benders_decomposition", **gold["output"]})
    return outputs


def executor(policy):
    value = object.__new__(NativeSandboxExecutor)
    if policy is not None:
        value.output_policy, value.output_limit = policy, OUTPUT_LIMITS[policy]
    return value


def test_dense_km_outputs_in_every_json_layout_are_read_in_full_under_the_32mib_policy(tmp_path):
    sizes = []
    for index, output in enumerate(km_outputs()):
        for indent in (None, 2, 4):
            work = tmp_path / f"{index}_{indent}"
            work.mkdir()
            text = json.dumps(output, indent=indent)
            (work / "output.json").write_text(text)
            sizes.append(len(text.encode()))
            assert json.loads(executor(OUTPUT_POLICY_32MIB).read_output(work)) == output
            if len(text.encode()) > 4 * 1024 * 1024:
                with pytest.raises(Exception, match="size limit"):
                    executor(None).read_output(work)  # the paper's 4 MiB limit
    assert min(sizes) > 1_300_000 and max(sizes) > 4_600_000  # about 1.4, 3.0 and 4.7 MB


def test_an_output_above_32mib_is_rejected_and_the_policy_is_recorded(tmp_path, monkeypatch):
    work = tmp_path / "case/attempts/attempt_01"
    work.mkdir(parents=True)
    (work / "output.json").write_text('{"x": "' + "a" * (32 * 1024 * 1024 - 9) + '"}')
    assert (work / "output.json").stat().st_size == 32 * 1024 * 1024
    assert executor(OUTPUT_POLICY_32MIB).read_output(work) is not None
    with (work / "output.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(Exception, match="size limit"):
        executor(OUTPUT_POLICY_32MIB).read_output(work)
    # The cached-output read uses the same limit, and the request fingerprint records the policy.
    import src.experiments.sandbox as sandbox
    monkeypatch.setattr(sandbox, "native_environment", lambda: {"env": 1})
    (tmp_path / "case/data.json").write_text("{}")
    (work / "solver.py").write_text("print(1)\n")
    for policy in (None, OUTPUT_POLICY_32MIB):
        value = executor(policy)
        value.environment, value.license_file = {"env": 1}, tmp_path / "license"
        text = json.dumps(km_outputs()[0], indent=4)
        value.execute_bytes = lambda solver, data, timeout, threads, evidence, text=text: {"exec_ok": True, "output": json.loads(text), "output_text": text, "stdout": "", "stderr": "", "integrity_violations": []}
        (work / "native_solver_receipt.json").unlink(missing_ok=True)
        value(work, 60, 1)
        receipt = read(work / "native_solver_receipt.json")
        assert ("output_policy" in receipt["request"]) == (policy is not None)
        if policy is None:
            with pytest.raises(Exception, match="size limit"):
                value(work, 60, 1)  # a 4.7 MB cached output cannot be re-read under the paper's limit
        else:
            assert value(work, 60, 1)[3] == json.loads(text)


# ---------------------------------------------------------------------------------------------------------------------
# The combined dataset.

def test_combined_dataset_matches_its_sources_and_the_pinned_inventory():
    import importlib.util

    root = combined_root()
    spec = importlib.util.spec_from_file_location("build_small_scale_data", ROOT / "scripts/build_small_scale_data.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    summary = builder.check(root, ROOT / "configs/small_scale.json")
    changed = {name: value["prompts_changed_v2_to_v3"] for name, value in summary["families"].items() if value["prompts_changed_v2_to_v3"]}
    assert changed == {"tsso": 9, "lrp": 9, "lbbd": 9} and all(value["suite"] == "bd" for name, value in summary["families"].items() if name in changed)
    plan = load_plan(ROOT / "configs/small_scale.json")
    families = plan["cohort"]["families"]
    cases = []
    for path in root.glob("*/*/*/data.json"):
        suite, problem, name = path.relative_to(root).parts[:3]
        _, size, seed = name.rsplit("_", 2)
        cases.append({"suite": suite, "problem": problem, "instance": name, "size": size, "seed": int(seed[4:]),
                      "sha256": {f: digest(path.parent / f) for f in ("data.json", "prompt.txt", "meta.json", "gold_monolithic.json", f"gold_{suite}.json")}})
    cases.sort(key=lambda c: (["small", "medium", "large"].index(c["size"]), c["seed"], families.index(c["problem"])))
    assert len(cases) == 180 and identity(cases) == plan["dataset_inventory_sha256"]
