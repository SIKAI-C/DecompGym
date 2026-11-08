"""The method judge's mounted delivery (prompt_protocol.judge_context_delivery = "mounted-review-sources-v1").

No Docker engine, model call or optimizer is used: the container boundary is a recorder and the candidate solvers are
trivial host Python. The judge's review text is unchanged (algorithm_review_prompt); only its transport differs. The
paper's pasted judge stays byte-identical (tests/test_single_file_golden.py, tests/test_mounted_context.py).
"""
from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path

import pytest

from src.agent import docker_client
from src.agent.docker_client import DockerCliClient
from src.agent.mounted_context import mounted_files
from src.agent.prompts.judge import algorithm_review_prompt
from src.agent.prompts.method import DECOMPOSITION_POLICY_V3
from src.agent.prompts.mounted import (CONTEXT_DELIVERY_MOUNTED, JUDGE_CONTEXT_DELIVERY, MOUNTED_REVIEW_WORKSPACE_POLICY, REVIEW_HEADING, review_workspace_part)
from src.agent.specs import MOUNT_LAYOUT, REVIEW_DIR, REVIEW_LAYOUT
from src.experiments.data_context import accepted_prompt_protocol, judge_context_delivery_policy
from src.experiments.paired_runner import PairedRunner
from src.experiments.plan import load_plan
from src.utils.errors import ProtocolError
from src.utils.records import identity, read
from host_solver import host_solver
from test_mounted_context import UFL_PROMPT, FakeRuntime, MountedClient

ROOT = Path(__file__).resolve().parents[1]
MOUNTED = CONTEXT_DELIVERY_MOUNTED
VERDICT = "VERDICT: PASS\nNOTES: fixture\n"


@pytest.fixture(autouse=True)
def independent_test_auth_lock(monkeypatch):
    monkeypatch.setattr("src.experiments.paired_runner.runtime_lock", lambda name: nullcontext())
    docker_client._PREFLIGHTS.clear()
    yield
    docker_client._PREFLIGHTS.clear()


def reviewed(tmp_path, sources=("solver.py",), attempt="attempt_01"):
    directory = tmp_path / "case/attempts" / attempt
    for name in sources:
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(f"CONTENT-{name}\n")
    return [directory / name for name in sources]


def judge_client(tmp_path, runtime=None, backend="codex-cli", layout=REVIEW_LAYOUT):
    return DockerCliClient("m", backend=backend, image_id="sha256:x", runtime=runtime or FakeRuntime(tmp_path, backend, contents=VERDICT), evidence_root=tmp_path / "evidence",
                           credential_provider=lambda _: {}, role="algorithm_judge", effort="high", context_delivery=MOUNTED, context_layout=layout)


def test_the_mounted_judge_reads_only_the_sources_under_review_with_the_unchanged_review_text(tmp_path):
    runtime = FakeRuntime(tmp_path, "codex-cli", contents=VERDICT)
    sources = reviewed(tmp_path)
    target = sources[0].parent / "alg_verifier.md"
    review = algorithm_review_prompt("bd", DECOMPOSITION_POLICY_V3)
    result = judge_client(tmp_path, runtime).edit_file(target, review, context=sources, log_dir=tmp_path / "call", step="llm_verifier")
    assert result.ok and target.read_text() == VERDICT
    call = runtime.calls[0]
    prompt = call["payload"]["prompt"]
    assert call["context"] == {REVIEW_DIR: {"attempt_01/solver.py": b"CONTENT-solver.py\n"}} and call["context_directories"] == (REVIEW_DIR,)
    assert call["artifacts"][0] == "alg_verifier.md"
    assert [line for line in prompt.splitlines() if line.startswith("#")] == ["# Part 1: Workspace and files", "## File to modify", "## Files under review", "## No other task files", REVIEW_HEADING]
    assert prompt.endswith(REVIEW_HEADING + "\n\n" + review)  # the review text is the paper's, byte for byte
    assert prompt.startswith("# Part 1: Workspace and files\n\n## File to modify\nOnly edit the existing alg_verifier.md in the current working directory. "
                             "That file is /work/alg_verifier.md; do not create, rename or delete any file. Every other path in this workspace is read-only context. "
                             "Only /work/alg_verifier.md is collected after this call: anything written elsewhere is ignored.\n\n## Files under review\n")
    assert "physical line" not in prompt
    assert "  - /work/review/attempt_01/solver.py (18 bytes): the solver.py delivered by attempt 1 (under review)." in prompt
    assert "Read every file listed above completely before writing alg_verifier.md." in prompt
    assert ("This workspace has no gurobipy, NumPy or other solver packages, so solver.py cannot be run or tested here. Here you only need to read the files "
            "listed above and write /work/alg_verifier.md.") in prompt and "runs it later" not in prompt.split(REVIEW_HEADING)[0]
    assert "CONTENT-" not in prompt and "BEGIN CONTEXT" not in prompt and "/work/instance" not in prompt and "data.json" not in prompt.split(REVIEW_HEADING)[0]
    evidence = read(tmp_path / "call/llm_verifier_context.json")
    assert evidence["writable_target"] == "/work/alg_verifier.md" and evidence["read_only_mounts"] == [REVIEW_DIR] and evidence["layout"] == REVIEW_LAYOUT.id
    assert [(entry["path"], entry["sha256"]) for entry in evidence["files"]] == [(item.path, item.sha256) for item in mounted_files(sources, REVIEW_LAYOUT)]
    assert read(tmp_path / "call/llm_verifier_workspace.json")["version"] == MOUNTED_REVIEW_WORKSPACE_POLICY
    assert read(tmp_path / "call/llm_verifier_invocation.json")["context_delivery"] == MOUNTED


def test_a_multi_file_delivery_is_listed_file_by_file(tmp_path):
    sources = reviewed(tmp_path, ("solver.py", "cpp/core.cpp", "cpp/core.hpp"))
    files = mounted_files(sources, REVIEW_LAYOUT)
    assert [item.path for item in files] == ["/work/review/attempt_01/solver.py", "/work/review/attempt_01/cpp/core.cpp", "/work/review/attempt_01/cpp/core.hpp"]
    part, _ = review_workspace_part("alg_verifier.md", files, "codex-cli")
    assert "  - /work/review/attempt_01/cpp/core.cpp (21 bytes): the C++ source cpp/core.cpp delivered by attempt 1 (under review)." in part
    assert "  - /work/review/attempt_01/cpp/core.hpp (21 bytes): the C++ header cpp/core.hpp delivered by attempt 1 (under review)." in part


@pytest.mark.parametrize("case", ["missing", "nested", "order", "two_attempts", "data"])
def test_an_incomplete_or_foreign_review_inventory_is_refused(tmp_path, case):
    if case == "missing":
        sources = reviewed(tmp_path) + [tmp_path / "case/attempts/attempt_01/cpp/absent.cpp"]
    elif case == "nested":
        sources = reviewed(tmp_path, ("solver.py", "cpp/sub/core.cpp"))
    elif case == "order":
        sources = reviewed(tmp_path, ("cpp/core.cpp", "solver.py"))
    elif case == "two_attempts":
        sources = reviewed(tmp_path) + reviewed(tmp_path, attempt="attempt_02")
    else:
        sources = reviewed(tmp_path) + [tmp_path / "case/data.json"]
        sources[-1].write_text("{}")
    target = tmp_path / "case/attempts/attempt_01/alg_verifier.md"
    with pytest.raises(ProtocolError):
        judge_client(tmp_path).edit_file(target, "review", context=sources, log_dir=tmp_path / "call", step="llm_verifier")


def test_the_review_layout_and_the_judge_target_belong_together(tmp_path):
    sources = reviewed(tmp_path)
    with pytest.raises(ValueError, match="only with the review layout"):
        judge_client(tmp_path).edit_file(sources[0].parent / "solver.py", "x", context=sources, log_dir=tmp_path / "call", step="agent")
    with pytest.raises(ValueError, match="only with the review layout"):
        judge_client(tmp_path, layout=MOUNT_LAYOUT).edit_file(sources[0].parent / "alg_verifier.md", "x", context=sources, log_dir=tmp_path / "call2", step="llm_verifier")


def ufl_source(tmp_path):
    source = tmp_path / "data/bd/ufl/ufl_small_seed000"
    source.mkdir(parents=True)
    (source / "data.json").write_text(json.dumps({"m": 2, "n": 1, "f": [2, 3], "D": [[4], [1]]}))
    (source / "prompt.txt").write_text(UFL_PROMPT)
    (source / "meta.json").write_text('{"ground_truth_objective":4}')
    return source


def mounted_judge():
    judge = MountedClient()
    judge.context_layout = REVIEW_LAYOUT
    return judge


def runner(client, judge, setting=JUDGE_CONTEXT_DELIVERY):
    return PairedRunner(client=client, judge_client=judge, model="fixture", suite="bd", problem=["ufl"], solver_executor=host_solver, decomposition_contract=DECOMPOSITION_POLICY_V3,
                        context_delivery=MOUNTED, judge_context_delivery=setting)


def test_the_runner_fingerprints_the_mounted_review_and_freezes_the_setting_in_the_case_identity(tmp_path):
    source = ufl_source(tmp_path)
    client, judge = MountedClient([False] + [True] * 20), mounted_judge()
    runner(client, judge).run_pair(source, tmp_path / "runs")
    reviews = [read(path) for path in (tmp_path / "runs").rglob("request.json") if read(path)["step"] == "llm_verifier"]
    assert reviews
    for request in reviews:
        delivery = request["context_delivery"]
        assert delivery["policy"] == MOUNTED and delivery["judge"] == JUDGE_CONTEXT_DELIVERY and delivery["layout"] == REVIEW_LAYOUT.id
        assert [entry["path"] for entry in delivery["files"]] == [f"/work/review/{request['context'][0]['label']}"]
        assert delivery["files"][0]["sha256"] == request["context"][0]["sha256"]
        assert request["request_sha256"] == identity({k: v for k, v in request.items() if k not in ("request_sha256", "invocation_id")})
    for step, target, context, prompt in judge.calls:
        assert step == "llm_verifier" and [path.name for path in context] == ["solver.py"] and prompt.startswith("Inspect the supplied solver.py")
    assert all(read(path)["judge_context_delivery"] == JUDGE_CONTEXT_DELIVERY for path in (tmp_path / "runs").rglob("identity.json"))
    calls = len(client.calls) + len(judge.calls)
    runner(client, judge).run_pair(source, tmp_path / "runs")
    assert len(client.calls) + len(judge.calls) == calls  # resume reuses every committed call
    pasted = MountedClient()
    pasted.context_delivery = None
    with pytest.raises(ProtocolError, match="Case protocol changed"):
        runner(client, pasted, setting=None).run_pair(source, tmp_path / "runs")


def test_the_runner_refuses_a_judge_client_or_setting_that_differs_from_the_protocol(tmp_path):
    target = tmp_path / "attempt_01/alg_verifier.md"
    target.parent.mkdir()
    pasted = MountedClient()
    pasted.context_delivery = None
    with pytest.raises(ProtocolError, match="context delivery differs"):
        runner(None, None).edit(pasted, target, "x", [], "llm_verifier")
    with pytest.raises(ProtocolError, match="mounted layout differs"):
        runner(None, None).edit(MountedClient(), target, "x", [], "llm_verifier")
    with pytest.raises(ProtocolError, match="context delivery differs"):
        runner(None, None, setting=None).edit(mounted_judge(), target, "x", [], "llm_verifier")
    with pytest.raises(ValueError, match="Unknown judge context delivery"):
        runner(None, None, setting="mounted-review-sources-v0")
    with pytest.raises(ValueError, match="mounted-files protocol"):
        PairedRunner(client=None, judge_client=None, solver_executor=None, judge_context_delivery=JUDGE_CONTEXT_DELIVERY)


def test_only_the_small_scale_protocol_mounts_the_judge():
    small = load_plan(ROOT / "configs/small_scale.json")["prompt_protocol"]
    assert judge_context_delivery_policy(small) == JUDGE_CONTEXT_DELIVERY
    for name in ("paper.json", "harder_small_all.json"):
        assert judge_context_delivery_policy(load_plan(ROOT / "configs" / name)["prompt_protocol"]) is None
    assert accepted_prompt_protocol({key: value for key, value in small.items() if key != "judge_context_delivery"})  # mounted author, pasted judge
    assert not accepted_prompt_protocol({key: value for key, value in small.items() if key != "context_delivery"})  # a mounted judge needs the mounted protocol
    assert not accepted_prompt_protocol({**small, "judge_context_delivery": "mounted-review-sources-v0"})
    assert not accepted_prompt_protocol({**small, "judge_context_delivery": None})


def test_a_committed_review_is_reused_or_recovered_and_a_missing_source_is_refused_before_reuse(tmp_path):
    sources = reviewed(tmp_path)
    target = sources[0].parent / "alg_verifier.md"
    judge = mounted_judge()
    paired = runner(None, None)
    first = paired.edit(judge, target, "review", sources, "llm_verifier")
    assert first["ok"] and len(judge.calls) == 1
    assert paired.edit(judge, target, "review", sources, "llm_verifier") == first and len(judge.calls) == 1  # the receipt is reused
    (target.parent / "llm_verifier_receipt.json").unlink()  # interrupted after the call's metrics, before the runner's receipt
    assert paired.edit(judge, target, "review", sources, "llm_verifier")["artifact_sha256"] == first["artifact_sha256"] and len(judge.calls) == 1
    with pytest.raises(ProtocolError, match="never skips"):
        paired.edit(judge, target, "review", sources + [sources[0].parent / "cpp/absent.cpp"], "llm_verifier")
    with pytest.raises(ProtocolError, match="exactly one attempt"):
        paired.edit(judge, target, "review", sources + reviewed(tmp_path, attempt="attempt_02"), "llm_verifier")
    assert len(judge.calls) == 1


class VerdictRuntime:
    """A container boundary for scripts/judge_delivery_check.py: PASS unless the reviewed source solves a full model."""

    def __init__(self, root):
        self.seccomp_profile = root / "profile.json"
        self.seccomp_profile.write_text("{}")
        self.calls = []

    def bind_environment(self, environment):
        pass

    def image_id(self, image):
        return image

    def ensure_volume(self, *args, **kwargs):
        pass

    def run(self, image, command, **kwargs):
        from src.agent.container import ContainerResult
        from src.agent.docker_client import CLI_VERSIONS

        kwargs["state_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["state_path"].write_text("{}")
        if command == ["preflight"]:
            value = {"ok": True, "versions": {"codex": CLI_VERSIONS["codex-cli"], "claude": CLI_VERSIONS["claude-code-cli"], "agy": CLI_VERSIONS["agy-cli"]}, "worker_files_sha256": docker_client.worker_file_hashes()}
            return ContainerResult(0, "", "", 0, artifacts={"worker.json": json.dumps(value)})
        job = kwargs["payload"]
        self.calls.append(kwargs)
        mounted = b"".join(contents for directory in (kwargs.get("context") or {}).values() for contents in directory.values()).decode()
        source = mounted or job["prompt"]
        verdict = "VERDICT: FAIL\nNOTES: full model\n" if 'gp.Model("full")' in source or 'gp.Model("compact")' in source else "VERDICT: PASS\nNOTES: decomposition\n"
        events = json.dumps({"type": "item.completed", "item": {"type": "command_execution", "command": "cat /work/review/attempt_01/solver.py"}}) if mounted else ""
        worker = {"cli_version": CLI_VERSIONS["codex-cli"], "requested_model": job["model"], "target": job["target"], "returncode": 0, "terminal_state": "completed", "error": None,
                  "usage": {"input_tokens": 1}, "usage_available": True, "warnings": [], "observed_models": [job["model"]], "agent_seconds": 1.0}
        return ContainerResult(0, "", "", 1, artifacts={job["target"]: verdict, "worker.json": json.dumps(worker), "events.jsonl": events, "delivery.json": json.dumps({"rejected_target": None, "misplaced": {}})},
                               metadata={"cleanup_verified": True, "mounts": []})


def test_the_real_call_check_script_runs_eight_controlled_judge_calls(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("judge_delivery_check", ROOT / "scripts/judge_delivery_check.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    engine = VerdictRuntime(tmp_path)
    monkeypatch.setattr(script, "REPO", tmp_path)
    monkeypatch.setattr(script.docker_setup, "load_execution_environment", lambda path: {"image_id": "sha256:fixture"})
    monkeypatch.setattr(script.docker_setup, "runtime", lambda: engine)
    monkeypatch.setattr(script.agent_auth, "check_backend", lambda *args, **kwargs: [{"ok": True}])
    monkeypatch.setattr(script, "DockerCliClient", lambda *args, **kwargs: DockerCliClient(*args, credential_provider=lambda _: {}, **kwargs))
    with pytest.raises(SystemExit) as exit_info:
        script.main([])
    assert exit_info.value.code == 0
    summary = read(next((tmp_path / "outputs/diagnostics").glob("judge_delivery_check_*/summary.json")))
    calls = summary["calls"]
    assert summary["passed"] and len(calls) == 8 == len(engine.calls)
    assert summary["judge_model"] == "gpt-6-astra" and summary["judge_effort"] == "high"
    assert {(call["fixture"], call["transport"]) for call in calls} == {(fixture, transport) for _, fixture, _ in script.CASES for transport in script.TRANSPORTS}
    for fixture in {call["fixture"] for call in calls}:
        pair = [call for call in calls if call["fixture"] == fixture]
        assert len({call["source_sha256"] for call in pair}) == 1  # identical bytes under both transports
    assert all(call["review_path_in_completed_tool_input"] is True for call in calls if call["transport"] == "mounted")
    assert [engine.calls[index]["payload"]["effort"] for index in range(8)] == ["high"] * 8


def test_the_path_mention_diagnostic_separates_missing_evidence_from_absence():
    import importlib.util

    spec = importlib.util.spec_from_file_location("judge_delivery_check", ROOT / "scripts/judge_delivery_check.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    path = "/work/review/attempt_01/solver.py"
    hit = json.dumps({"type": "item.completed", "item": {"type": "command_execution", "command": f"sed -n 1,200p {path}"}})
    miss = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": path}})
    assert script.mentioned(hit, path) is True and script.mentioned(miss, path) is False
    assert script.mentioned("", path) is None and script.mentioned("not json\n[1, 2]\n", path) is None
