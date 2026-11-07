"""The recorded data-context setting: full data.json up to 512 KiB, a marked structural preview above it."""
from contextlib import nullcontext
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.docker_client import DockerCliClient
from src.agent.prompts.snapshot import PROMPT_PROTOCOL
from src.experiments import campaign
from src.experiments.data_context import (DATA_CONTEXT_PREVIEW, PREVIEW_NAME, PREVIEW_THRESHOLD_BYTES, accepted_prompt_protocol,
                                          data_context_file, data_context_policy)
from src.experiments.paired_runner import PairedRunner
from src.experiments.plan import load_plan
from src.utils.errors import ProtocolError
from src.utils.records import digest, read, save
from host_solver import host_solver
from test_native_paired import Client


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def independent_test_auth_lock(monkeypatch):
    monkeypatch.setattr("src.experiments.paired_runner.runtime_lock", lambda name: nullcontext())


def case(tmp_path, data):
    source = tmp_path / "data/bd/ufl/ufl_small_seed000"
    source.mkdir(parents=True)
    (source / "data.json").write_text(json.dumps(data))
    (source / "prompt.txt").write_text("Implement Benders decomposition.")
    (source / "meta.json").write_text('{"ground_truth_objective":4}')
    return source


SMALL = {"m": 2, "n": 1, "f": [2, 3], "D": [[4], [1]]}


# The successful solver reads the runtime ../../data.json and fails unless it is the complete file.
READING_SOLVER = ("import hashlib, json\nfrom pathlib import Path\nraw = Path('../../data.json').read_bytes()\n"
                  "assert hashlib.sha256(raw).hexdigest() == EXPECTED, 'runtime input is not the full data.json'\n"
                  "Path('output.json').write_text(json.dumps({'status':'feasible','method':'benders_decomposition','y':[0,1],'x':[[0],[1]],'objective_value':4}))\n")


class ReadingClient(Client):
    expected = None  # sha256 of the source data.json

    def edit_file(self, target, prompt, *, context, log_dir, step):
        result = super().edit_file(target, prompt, context=context, log_dir=log_dir, step=step)
        if step == "agent" and "controlled solver failure" not in target.read_text():
            target.write_text(f"EXPECTED = {self.expected!r}\n" + READING_SOLVER)
            result = replace(result, artifact_sha256=digest(target))
            save(log_dir / f"{step}_metrics.json", asdict(result))
        return result


def run(tmp_path, data, policy):
    client = ReadingClient([False, True, True])  # a failed shared attempt, so reflection contexts are built too
    source = case(tmp_path, data)
    client.expected = digest(source / "data.json")
    runs = tmp_path / "runs"
    summaries = PairedRunner(client=client, judge_client=client, model="fixture", suite="bd", problem=["ufl"], solver_executor=host_solver, data_context=policy).run_pair(source, runs)
    assert all(s["attempts"][1]["status"] == "ok" for s in summaries.values())  # the reading solver succeeded
    calls = len(client.calls)
    PairedRunner(client=client, judge_client=client, model="fixture", suite="bd", problem=["ufl"], solver_executor=host_solver, data_context=policy).run_pair(source, runs)
    assert len(client.calls) == calls  # resume reuses every committed call, preview included
    return client, source, runs


def contexts(client, root):
    """Each call's target and context files relative to the run root (logs mention the absolute temporary path)."""
    def body(p):
        return p.read_bytes().replace(str(root.parent).encode(), b"<tmp>") if p.exists() else None
    return [(step, str(target.relative_to(root)), [(str(p.relative_to(root)), body(p)) for p in context]) for step, target, context in client.calls]


def compose(paths):
    return DockerCliClient.compose_prompt(SimpleNamespace(backend="codex-cli"), "solver.py", "TASK", paths)


def test_below_threshold_context_is_identical_to_the_paper_rule(tmp_path):
    legacy, _, legacy_runs = run(tmp_path / "legacy", SMALL, None)
    new, _, new_runs = run(tmp_path / "new", SMALL, DATA_CONTEXT_PREVIEW)
    assert contexts(legacy, legacy_runs) == contexts(new, new_runs)
    assert all(context[0].name == "data.json" for step, _, context in new.calls if step in ("agent", "reflection"))
    assert sum(step == "reflection" for step, *_ in new.calls) >= 1
    assert compose([c[0] for s, _, c in new.calls if s == "agent"][0:1]) == compose([c[0] for s, _, c in legacy.calls if s == "agent"][0:1])
    assert not list(new_runs.rglob(PREVIEW_NAME))
    requests = [read(p) for p in new_runs.rglob("request.json")]
    deliveries = [r["data_context"] for r in requests if r["step"] != "llm_verifier"]
    assert len(deliveries) == sum(s != "llm_verifier" for s, *_ in new.calls)
    assert all(d == {"policy": DATA_CONTEXT_PREVIEW, "data_json_bytes": len(json.dumps(SMALL)), "delivered": "full"} for d in deliveries)
    assert all(set(read(p)) == {"protocol", "reflection_policy", "prompt_policies", "step", "prompt", "target", "context", "request_sha256", "invocation_id"}
               for p in legacy_runs.rglob("request.json"))  # legacy requests keep their layout; their contexts were compared above
    # The legacy rule records nothing new, so earlier request fingerprints and identities are unchanged.
    assert not any("data_context" in read(p) for p in legacy_runs.rglob("request.json"))
    assert all("data_context" not in read(p) for p in legacy_runs.rglob("identity.json"))
    assert all(read(p)["data_context"] == DATA_CONTEXT_PREVIEW for p in new_runs.rglob("identity.json"))


def test_threshold_boundary(tmp_path):
    for size, delivered in ((PREVIEW_THRESHOLD_BYTES, "full"), (PREVIEW_THRESHOLD_BYTES + 1, "structural_preview")):
        directory = tmp_path / str(size)
        directory.mkdir()
        text = json.dumps({"x": "a"})
        (directory / "data.json").write_text(text[:-2] + "a" * (size - len(text)) + text[-2:])
        assert (directory / "data.json").stat().st_size == size
        path, record = data_context_file(directory, DATA_CONTEXT_PREVIEW)
        assert record["delivered"] == delivered and path.name == ("data.json" if delivered == "full" else PREVIEW_NAME)
        assert data_context_file(directory, None) == (directory / "data.json", None)


def test_above_threshold_context_holds_a_marked_preview_and_the_solver_reads_the_full_file(tmp_path):
    big = dict(SMALL, extra=list(range(150_000)))  # verifiers ignore unknown keys; only the size matters here
    client, source, runs = run(tmp_path, big, DATA_CONTEXT_PREVIEW)
    assert (source / "data.json").stat().st_size > PREVIEW_THRESHOLD_BYTES
    for step, target, context in client.calls:
        if step in ("agent", "reflection"):
            first = context[0]
            assert first.name == PREVIEW_NAME and first.parent == target.parents[2] and "data.json" in first.name
            text = first.read_text()
            assert text.startswith("STRUCTURAL PREVIEW of data.json") and "solver.py must load the complete data.json at run time" in text
            assert json.loads(text.split("\n", 1)[1]) == dict(SMALL, extra=[0, 1, 2, "... (150000 items in total)"])
    prompt = compose([c[0] for s, _, c in client.calls if s == "agent"][:1])
    assert f"--- BEGIN CONTEXT: ufl_small_seed000/{PREVIEW_NAME} ---" in prompt and str(149_999) not in prompt
    # The runtime input stays the real, full data.json.
    for mode in ("plain", "reflection"):
        assert digest(runs / mode / "bd/ufl/ufl_small_seed000/data.json") == digest(source / "data.json")
    records = [read(p)["data_context"] for p in runs.rglob("request.json") if read(p)["step"] != "llm_verifier"]
    assert len(records) == sum(s != "llm_verifier" for s, *_ in client.calls) and all(r["delivered"] == "structural_preview" and r["data_json_bytes"] == (source / "data.json").stat().st_size for r in records)
    preview = runs / "plain/bd/ufl/ufl_small_seed000" / PREVIEW_NAME
    assert not list(preview.parent.glob(f".{PREVIEW_NAME}.*"))  # no temporary file left behind
    preview.write_text("tampered")
    with pytest.raises(ProtocolError):
        data_context_file(preview.parent, DATA_CONTEXT_PREVIEW)


def test_config_key_is_validated(tmp_path):
    # No shipped config uses the preview rule any more (the small-scale campaign mounts its context files); it stays a
    # reviewed setting, checked here on a paper-based plan.
    paper = load_plan(REPO / "configs/paper.json")
    small_scale = dict(paper, prompt_protocol={**PROMPT_PROTOCOL, "data_context": DATA_CONTEXT_PREVIEW})
    save(tmp_path / "preview.json", small_scale)
    assert data_context_policy(load_plan(tmp_path / "preview.json")["prompt_protocol"]) == DATA_CONTEXT_PREVIEW
    assert data_context_policy(paper["prompt_protocol"]) is None
    assert data_context_policy(load_plan(REPO / "configs/small_scale.json")["prompt_protocol"]) is None
    for value in ("full-v0", None, ""):
        plan = dict(small_scale, prompt_protocol={**PROMPT_PROTOCOL, "data_context": value})
        save(tmp_path / "bad.json", plan)
        with pytest.raises(ProtocolError):
            load_plan(tmp_path / "bad.json")
    assert not accepted_prompt_protocol({**PROMPT_PROTOCOL, "data_context": DATA_CONTEXT_PREVIEW, "extra": 1})
    with pytest.raises(ValueError):
        PairedRunner(client=None, judge_client=None, solver_executor=None, data_context="full-v0")
    root = tmp_path / "root"
    root.mkdir()
    manifest = {"prompt_policies": small_scale["prompt_policies"], "reflection_policy": small_scale["reflection_policy"], "prompt_protocol": small_scale["prompt_protocol"]}
    campaign.ensure_execution_allowed(root, manifest)
    with pytest.raises(ProtocolError):
        campaign.ensure_execution_allowed(root, dict(manifest, prompt_protocol={**PROMPT_PROTOCOL, "data_context": "full-v0"}))


def test_km_literature_template_requires_the_stated_distance_rule():
    from src.bd.km.b_problem import render_prompt
    shared = {"m": 2, "n": 2, "k": 1, "Xc": [[0, 0], [1, 1]], "Xf": [[0, 0], [1, 1]]}
    assert "rounded down to an integer" in render_prompt({**shared, "D": [[0, 1], [1, 0]]})
    assert "rounded down to an integer" not in render_prompt({**shared, "D": [[0, 2], [2, 0]]})
