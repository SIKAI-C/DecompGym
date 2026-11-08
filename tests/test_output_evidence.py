"""The solver-output evidence schema of the 32 MiB policy (sandbox.OUTPUT_BY_REFERENCE): output.json is the only
copy of the candidate's output; the native evidence, the native receipt and execution.json keep its sha256 instead,
and the shared first attempt's output.json is a copy-on-write clone. Without the schema (the paper, and every
executor constructed without it) the records embed the output exactly as before.

Solver-free: every executor here is a stub; no solver.py, sandbox or optimizer runs."""
from __future__ import annotations

from contextlib import nullcontext
import io
import contextlib
import json
import math
import os
from pathlib import Path

import pytest

from src.experiments import campaign
from src.experiments import sandbox
from src.experiments.environment import RuntimeEnvironmentError
from src.experiments.execution_policy import AttemptCase, SmallScaleNativePolicy
from src.experiments.paired_runner import PairedRunner, clone_or_copy
from src.experiments.sandbox import EVIDENCE_SCHEMAS, OUTPUT_BY_REFERENCE, OUTPUT_LIMITS, OUTPUT_POLICY_32MIB, NativeSandboxExecutor, stored_record
from src.utils.errors import ProtocolError
from src.utils.records import digest, read, save

from test_native_paired import Client, source_case

ROOT = Path(__file__).resolve().parents[1]
EMBEDDED = ("output_text", "output")
# UTF-8 non-ASCII text, a duplicate key (the last wins), NaN, Infinity and a signed zero, as a candidate may write them.
ODD = '{"status": "feasible", "name": "caf\\u00e9 été", "k": 1, "k": 2, "nan": NaN, "inf": -Infinity, "zero": -0.0}'


def same(a, b) -> bool:
    """Equality that also identifies NaN with NaN and tells -0.0 from 0.0."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True) and repr(a) == repr(b)


# ---------------------------------------------------------------------------------------------------------------------
# Native sandbox records (A + B).

def native(tmp_path, monkeypatch, *, policy=OUTPUT_POLICY_32MIB, schema=OUTPUT_BY_REFERENCE, record=None):
    monkeypatch.setattr(sandbox, "native_environment", lambda: {"env": 1})
    tmp_path.mkdir(parents=True, exist_ok=True)
    license_file = tmp_path / "license"
    license_file.write_text("x")
    executor = NativeSandboxExecutor(license_file=license_file, output_policy=policy, evidence_schema=schema)
    calls = []
    executor.execute_bytes = lambda solver, data, timeout, threads, evidence: (calls.append(evidence.name), dict(record))[1]
    work = tmp_path / "case/attempts/attempt_01"
    work.mkdir(parents=True, exist_ok=True)
    (tmp_path / "case/data.json").write_text("{}")
    (work / "solver.py").write_text("print(1)\n")
    return executor, work, calls


def record(text, ok):
    output = json.loads(text) if ok else None
    return {"policy": sandbox.NATIVE_SOLVER_POLICY, "exec_ok": ok, "output": output, "output_text": text, "stdout": "out", "stderr": "" if ok else "[framework] solver execution timed out after 60 seconds",
            "integrity_violations": [], "cleanup_verified": True, "execution_seconds": 1.5}


CASES = {"completed": (ODD, True), "timed_out_with_valid_partial_output": ('{"status": "feasible", "objective_value": 3}', False),
         "empty_output": ("", False), "malformed_output": ("{", False), "no_output": (None, False), "empty_object": ("{}", True)}


@pytest.mark.parametrize("case", CASES)
def test_native_records_keep_only_the_hash_and_a_resume_returns_the_fresh_result(tmp_path, monkeypatch, case):
    text, ok = CASES[case]
    executor, work, calls = native(tmp_path, monkeypatch, record=record(text, ok))
    fresh = executor(work, 60, 1)
    expected_output = json.loads(text) if ok else None
    assert fresh[0] is ok and same(fresh[3], expected_output)
    receipt, evidence = read(work / "native_solver_receipt.json"), read(work / calls[0])
    assert receipt["request"]["evidence_schema"] == OUTPUT_BY_REFERENCE and read(work / "solver_invocation.json")["request"] == receipt["request"]
    for stored in (receipt["result"], evidence):
        assert not any(key in stored for key in EMBEDDED)
        assert stored["evidence_schema"] == OUTPUT_BY_REFERENCE and stored["output_parsed"] is ok
        if text is None:
            assert stored["output_sha256"] is None and stored["output_bytes"] is None and not (work / "output.json").exists()
        else:  # also the empty file: a hash of the empty bytes, size 0
            assert (work / "output.json").read_bytes() == text.encode("utf-8")
            assert stored["output_sha256"] == digest(work / "output.json") and stored["output_bytes"] == len(text.encode("utf-8"))
    assert receipt["result"] == evidence
    again = executor(work, 60, 1)
    assert len(calls) == 1 and again[:3] == fresh[:3] and same(again[3], expected_output)


def test_a_changed_or_deleted_output_and_a_schema_change_are_refused(tmp_path, monkeypatch):
    executor, work, _ = native(tmp_path, monkeypatch, record=record(ODD, True))
    executor(work, 60, 1)
    original = (work / "output.json").read_bytes()
    (work / "output.json").write_bytes(original + b" ")
    with pytest.raises(RuntimeEnvironmentError, match="output changed"):
        executor(work, 60, 1)
    (work / "output.json").unlink()
    with pytest.raises(RuntimeEnvironmentError, match="output changed"):
        executor(work, 60, 1)
    (work / "output.json").write_bytes(original)
    legacy, _, _ = native(tmp_path, monkeypatch, schema=None, record=record(ODD, True))
    with pytest.raises(RuntimeEnvironmentError, match="request changed"):
        legacy(work, 60, 1)  # a schema receipt is never read as an embedded one, and vice versa
    receipt = read(work / "native_solver_receipt.json")
    for change in ({"output": {"x": 1}}, {"output_parsed": False}, {"exec_ok": False}, {"output_bytes": 1}, {"output_parsed": "yes"}):
        save(work / "native_solver_receipt.json", receipt | {"result": receipt["result"] | change})
        with pytest.raises(RuntimeEnvironmentError, match="unknown layout"):
            executor(work, 60, 1)
    # An embedded (legacy) receipt is never read by a schema executor.
    other = tmp_path / "legacy"
    legacy, legacy_work, _ = native(other, monkeypatch, schema=None, record=record(ODD, True))
    legacy(legacy_work, 60, 1)
    schema_executor, _, _ = native(other, monkeypatch, record=record(ODD, True))
    with pytest.raises(RuntimeEnvironmentError, match="request changed"):
        schema_executor(legacy_work, 60, 1)


@pytest.mark.parametrize("policy", [None, OUTPUT_POLICY_32MIB])
def test_without_the_schema_the_records_embed_the_output_as_before(tmp_path, monkeypatch, policy):
    executor, work, calls = native(tmp_path, monkeypatch, policy=policy, schema=None, record=record('{"x": 1}', True))
    executor(work, 60, 1)
    receipt = read(work / "native_solver_receipt.json")
    assert "evidence_schema" not in receipt["request"] and ("output_policy" in receipt["request"]) == (policy is not None)
    assert receipt["result"]["output_text"] == '{"x": 1}' and receipt["result"]["output"] == {"x": 1} and "output_parsed" not in receipt["result"]
    assert read(work / calls[0]) == receipt["result"]


def test_schema_and_policy_must_match():
    assert EVIDENCE_SCHEMAS == {None: None, OUTPUT_POLICY_32MIB: OUTPUT_BY_REFERENCE}
    for policy, schema in ((None, OUTPUT_BY_REFERENCE), (OUTPUT_POLICY_32MIB, "native-output-by-reference-v0")):
        with pytest.raises(ValueError, match="Evidence schema"):
            NativeSandboxExecutor(license_file=Path("/nonexistent"), output_policy=policy, evidence_schema=schema)
    with pytest.raises(ValueError):
        stored_record({"exec_ok": True, "output": None, "output_text": "{}"}, OUTPUT_BY_REFERENCE)


@pytest.mark.parametrize("probe_ok", [True, False])
def test_the_gurobi_diagnostic_keeps_embedded_evidence(tmp_path, monkeypatch, probe_ok):
    failed = record('{"partial": true}', False) | {"stderr": "gurobipy license error"}
    executor, work, _ = native(tmp_path, monkeypatch, record=failed)
    probe = {"exec_ok": True, "output": {"status": "optimal", "objective_value": 1.0 if probe_ok else 2.0, "threads": 1}, "output_text": "x"}
    def execute_bytes(solver, data, timeout, threads, evidence):
        value = dict(probe) if evidence.name.startswith("gurobi_diagnostic_") else dict(failed)
        sandbox.save(evidence, value)  # as execute_bytes saves its complete record
        return value
    executor.execute_bytes = execute_bytes
    if probe_ok:
        executor(work, 60, 1)
        evidence = read(next(work.glob("native_execution_*.json")))
        assert evidence["gurobi_diagnostic"]["ok"] and not any(key in evidence for key in EMBEDDED)
        assert read(work / evidence["gurobi_diagnostic"]["path"])["output"] == probe["output"]  # the probe's evidence stays embedded
    else:
        with pytest.raises(RuntimeEnvironmentError, match="diagnostic failed"):
            executor(work, 60, 1)
        # The candidate's output was never published: its evidence keeps the text.
        evidence = read(next(work.glob("native_execution_*.json")))
        assert evidence["output_text"] == '{"partial": true}' and not evidence["gurobi_diagnostic"]["ok"] and not (work / "output.json").exists()
        assert read(work / evidence["gurobi_diagnostic"]["path"])["output"]["objective_value"] == 2.0


# ---------------------------------------------------------------------------------------------------------------------
# execution.json (C).

class StubNative:
    """A native-executor stand-in: publishes `text` as output.json and returns the sandbox's tuple."""
    output_policy, output_limit, evidence_schema = OUTPUT_POLICY_32MIB, OUTPUT_LIMITS[OUTPUT_POLICY_32MIB], OUTPUT_BY_REFERENCE

    def __init__(self, text=ODD, ok=True, tamper=None):
        self.text, self.ok, self.tamper, self.calls = text, ok, tamper, 0

    def __call__(self, attempt_dir, timeout, threads=1):
        self.calls += 1
        (attempt_dir / "output.json").write_bytes(self.text.encode("utf-8"))
        if self.tamper:
            self.tamper(attempt_dir)
        return self.ok, "out", "" if self.ok else "failed", json.loads(self.text) if self.ok else None


def attempt_case(tmp_path):
    instance = tmp_path / "case"
    directory = instance / "attempts/attempt_01"
    directory.mkdir(parents=True)
    (instance / "data.json").write_text('{"a": 1}')
    (tmp_path / "canonical.json").write_text('{"a": 1}')
    (directory / "solver.py").write_text("print(1)\n")
    return AttemptCase(directory, instance, tmp_path / "canonical.json", digest(tmp_path / "canonical.json"), "ufl")


def execute(policy, case):
    return policy.execute(case, delivery=sandbox_delivery(), runtime=runtime(), stage=lambda phase: None)


def sandbox_delivery():
    from src.agent.specs import SINGLE_FILE_DELIVERY
    return SINGLE_FILE_DELIVERY


def runtime():
    from src.agent.specs import RuntimeSpec
    return RuntimeSpec(60, 1, OUTPUT_POLICY_32MIB)


def by_reference(executor):
    return SmallScaleNativePolicy(executor=executor, suite="bd", evidence_schema=executor.evidence_schema, output_limit=executor.output_limit)


def test_execution_json_names_output_json_and_a_resume_returns_the_fresh_result(tmp_path):
    executor = StubNative()
    case = attempt_case(tmp_path)
    policy = by_reference(executor)
    fresh = execute(policy, case)
    stored = read(case.directory / "execution.json")
    assert stored["output"] is None and stored["output_in_output_json"] is True and stored["evidence_schema"] == OUTPUT_BY_REFERENCE
    assert stored["output_sha256"] == digest(case.directory / "output.json") and stored["exec_ok"] is True
    assert fresh.ok and same(fresh.output, json.loads(ODD))  # the fresh result keeps the parsed output in memory
    again = execute(policy, case)
    assert executor.calls == 1 and (again.ok, again.stdout, again.stderr, again.seconds, again.termination, dict(again.metrics)) == (fresh.ok, fresh.stdout, fresh.stderr, fresh.seconds, fresh.termination, dict(fresh.metrics))
    assert same(again.output, fresh.output)
    for change in (lambda path: path.write_bytes(path.read_bytes() + b" "), lambda path: path.unlink()):
        original = (case.directory / "output.json").read_bytes()
        change(case.directory / "output.json")
        with pytest.raises(ProtocolError, match="Saved solver output changed"):
            execute(policy, case)
        (case.directory / "output.json").unlink(missing_ok=True)
        (case.directory / "output.json").write_bytes(original)
    # A record of another schema is refused in both directions.
    with pytest.raises(ProtocolError, match="evidence schema"):
        execute(SmallScaleNativePolicy(executor=executor, suite="bd"), case)
    legacy_case = attempt_case(tmp_path / "legacy")
    execute(SmallScaleNativePolicy(executor=StubNative(), suite="bd"), legacy_case)
    with pytest.raises(ProtocolError, match="evidence schema"):
        execute(policy, legacy_case)


@pytest.mark.parametrize("failing", ["native_solver_receipt.json", "solver_invocation.json", "evidence"])
def test_an_interrupted_publication_reruns_and_commits_consistent_records(tmp_path, monkeypatch, failing):
    executor, work, calls = native(tmp_path, monkeypatch, record=record(ODD, True))
    original = sandbox.save
    def save_or_fail(path, value):
        if (path.name == failing or (failing == "evidence" and path.name.startswith("native_execution_"))) and not getattr(save_or_fail, "failed", False):
            save_or_fail.failed = True
            raise OSError("interrupted")
        original(path, value)
    monkeypatch.setattr(sandbox, "save", save_or_fail)
    with pytest.raises(OSError):
        executor(work, 60, 1)
    assert not (work / "native_solver_receipt.json").exists() and (work / "output.json").read_bytes() == ODD.encode()
    fresh = executor(work, 60, 1)  # no receipt: the run is repeated and its records are committed together
    receipt = read(work / "native_solver_receipt.json")
    assert len(calls) == 2 and receipt["result"]["output_sha256"] == digest(work / "output.json") and same(fresh[3], json.loads(ODD))
    assert read(work / read(work / "solver_invocation.json")["evidence"]) == receipt["result"] and same(executor(work, 60, 1)[3], fresh[3])


@pytest.mark.parametrize("partial", ["", "{", "[" * 200000, '{"a": "\udcff"}'.encode("utf-8", "surrogatepass").decode("latin-1")])
def test_a_failed_run_with_any_partial_output_resumes(tmp_path, partial):
    class Raw(StubNative):
        def __call__(self, attempt_dir, timeout, threads=1):
            self.calls += 1
            (attempt_dir / "output.json").write_bytes(partial.encode("latin-1") if partial.startswith('{"a"') else partial.encode())
            return False, "out", "failed", None
    case = attempt_case(tmp_path)
    executor = Raw()
    policy = by_reference(executor)
    fresh = execute(policy, case)
    assert read(case.directory / "execution.json")["output_in_output_json"] is False
    again = execute(policy, case)
    assert executor.calls == 1 and (again.ok, again.output, again.stderr) == (fresh.ok, fresh.output, fresh.stderr) == (False, None, "failed")


def test_a_lost_execution_record_is_rebuilt_from_the_native_receipt(tmp_path, monkeypatch):
    executor, work, calls = native(tmp_path, monkeypatch, record=record(ODD, True))
    case = AttemptCase(work, work.parent.parent, work.parent.parent / "data.json", digest(work.parent.parent / "data.json"), "ufl")
    policy = by_reference(executor)
    fresh = execute(policy, case)
    stored = read(work / "execution.json")
    (work / "execution.json").unlink()  # interrupted after the native receipt, before execution.json
    again = execute(policy, case)
    assert len(calls) == 1 and read(work / "execution.json") == stored and same(again.output, fresh.output) and again.ok


def test_an_8_mib_output_is_stored_once(tmp_path, monkeypatch):
    text = json.dumps({"status": "feasible", "x": [i * 0.5 for i in range(1_000_000)]})
    assert len(text) > 8_000_000
    executor, work, _ = native(tmp_path, monkeypatch, record=record(text, True))
    case = AttemptCase(work, work.parent.parent, work.parent.parent / "data.json", digest(work.parent.parent / "data.json"), "ufl")
    execute(by_reference(executor), case)
    others = sum(path.stat().st_size for path in work.iterdir() if path.name != "output.json")
    assert (work / "output.json").stat().st_size == len(text) and others < 20_000


def test_failed_and_restored_input_runs_keep_a_null_output(tmp_path):
    case = attempt_case(tmp_path)
    policy = by_reference(StubNative('{"status": "feasible", "objective_value": 3}', ok=False))
    result = execute(policy, case)
    stored = read(case.directory / "execution.json")
    assert result.output is None and stored["output"] is None and stored["output_in_output_json"] is False and stored["output_sha256"] == digest(case.directory / "output.json")
    assert execute(policy, case).output is None
    # The policy's own failure after a completed native run (the runtime input changed): exec_ok False, output None.
    other = attempt_case(tmp_path / "restored")
    policy = by_reference(StubNative(tamper=lambda attempt: (attempt.parent.parent / "data.json").write_text('{"a": 2}')))
    result = execute(policy, other)
    stored = read(other.directory / "execution.json")
    assert not result.ok and result.output is None and stored["output"] is None and stored["output_in_output_json"] is False
    resumed = execute(policy, other)
    assert not resumed.ok and resumed.output is None


def test_an_output_that_differs_from_output_json_stays_embedded(tmp_path):
    class Divergent(StubNative):
        def __call__(self, attempt_dir, timeout, threads=1):
            ok, out, err, output = super().__call__(attempt_dir, timeout, threads)
            return ok, out, err, {"status": "feasible", "other": True}
    case = attempt_case(tmp_path)
    policy = by_reference(Divergent())
    execute(policy, case)
    stored = read(case.directory / "execution.json")
    assert stored["output"] == {"status": "feasible", "other": True} and stored["output_in_output_json"] is False
    assert execute(policy, case).output == {"status": "feasible", "other": True}


def test_without_the_schema_execution_json_embeds_the_output(tmp_path):
    case = attempt_case(tmp_path)
    execute(SmallScaleNativePolicy(executor=StubNative('{"x": 1}'), suite="bd"), case)
    stored = read(case.directory / "execution.json")
    assert stored["output"] == {"x": 1} and "output_in_output_json" not in stored and "evidence_schema" not in stored


# ---------------------------------------------------------------------------------------------------------------------
# The paired runner (wiring, scores, resume, shared first attempt: E').

GOOD = '{"status": "feasible", "method": "benders_decomposition", "y": [0, 1], "x": [[0], [1]], "objective_value": 4, "note": "é"}'
WORSE = GOOD.replace('"objective_value": 4', '"objective_value": 6').replace('"y": [0, 1]', '"y": [1, 1]')  # feasible, 50 % above the gold


@pytest.fixture(autouse=True)
def independent_test_auth_lock(monkeypatch):
    monkeypatch.setattr("src.experiments.paired_runner.runtime_lock", lambda name: nullcontext())


def summaries(root):
    """The scored attempt records of both policies, without the metrics (call ids, wall-clock times)."""
    value = {}
    for path in sorted(root.rglob("summary.json")):
        record = read(path)
        record["metrics"] = {k: v for k, v in record.get("metrics", {}).items() if k in ("execution_timed_out", "cooldown_seconds")}
        value[str(path.relative_to(root))] = record
    return value


class Sequence(StubNative):
    def __init__(self, texts):
        super().__init__()
        self.texts = list(texts)

    def __call__(self, attempt_dir, timeout, threads=1):
        self.text = self.texts.pop(0)
        return super().__call__(attempt_dir, timeout, threads)


class Embedded(Sequence):
    evidence_schema = None


def pair(tmp_path, executor):
    source = source_case(tmp_path)
    client = Client()
    PairedRunner(client=client, judge_client=client, model="fixture", suite="bd", problem=["ufl"], solver_executor=executor, max_attempts=3).run_pair(source, tmp_path / "runs")
    return source, client


def test_the_runner_scores_identically_and_stores_the_output_once(tmp_path):
    reference = pair(tmp_path / "embedded", Embedded([WORSE, WORSE, GOOD, WORSE, GOOD]))
    executor = Sequence([WORSE, WORSE, GOOD, WORSE, GOOD])
    source, client = pair(tmp_path / "schema", executor)
    runs = tmp_path / "schema/runs"
    assert summaries(runs) == summaries(tmp_path / "embedded/runs")
    statuses = {mode: [a["status"] for a in read(runs / mode / "bd/ufl/ufl_small_seed000/instance_summary.json")["attempts"]] for mode in ("plain", "reflection")}
    assert statuses["plain"][0] == statuses["reflection"][0] == "feasible_not_optimal" and "ok" in statuses["plain"] and "ok" in statuses["reflection"]
    for execution in runs.rglob("execution.json"):
        record = read(execution)
        assert record["output"] is None and record["output_in_output_json"] is True and record["evidence_schema"] == OUTPUT_BY_REFERENCE
    # Shared first attempt: an independent file with the committed bytes (a clone where the filesystem allows it).
    first, projected = (runs / mode / "bd/ufl/ufl_small_seed000/attempts/attempt_01/output.json" for mode in ("plain", "reflection"))
    assert first.read_bytes() == projected.read_bytes() and os.stat(first).st_ino != os.stat(projected).st_ino
    assert os.stat(first).st_nlink == os.stat(projected).st_nlink == 1
    from src.agent.mounted_context import mounted_files  # a reflection reads the projection through the single-link check
    assert mounted_files([projected])[0].contents == first.read_bytes()
    # Resume after a lost score receipt: the execution record is re-read (not re-run) and scored the same.
    attempt = runs / "plain/bd/ufl/ufl_small_seed000/attempts/attempt_02"
    before = summaries(runs)
    (attempt / "score_receipt.json").unlink()
    (attempt / "summary.json").unlink()
    calls = executor.calls
    PairedRunner(client=client, judge_client=client, model="fixture", suite="bd", problem=["ufl"], solver_executor=executor, max_attempts=3).run_pair(source, runs)
    assert executor.calls == calls and summaries(runs) == before
    # A later change of a projected file is detected on resume.
    projected.write_bytes(projected.read_bytes() + b" ")
    with pytest.raises(ProtocolError):
        PairedRunner(client=client, judge_client=client, model="fixture", suite="bd", problem=["ufl"], solver_executor=executor, max_attempts=3).run_pair(source, runs)


def test_the_paper_path_copies_and_plain_callables_keep_embedded_records(tmp_path, monkeypatch):
    clones = []
    monkeypatch.setattr("src.experiments.paired_runner.clone_or_copy", lambda *args: clones.append(args))
    pair(tmp_path, Embedded([GOOD]))
    assert not clones
    record = read(tmp_path / "runs/plain/bd/ufl/ufl_small_seed000/attempts/attempt_01/execution.json")
    assert record["output"] == json.loads(GOOD) and "output_in_output_json" not in record


def test_clone_or_copy_falls_back_to_a_copy_and_never_replaces_a_file(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_bytes(b'{"x": 1}')
    first = clone_or_copy(source, tmp_path / "a.json")
    assert first in ("clone", "copy") and (tmp_path / "a.json").read_bytes() == source.read_bytes() and os.stat(tmp_path / "a.json").st_nlink == 1
    with pytest.raises(ProtocolError):
        clone_or_copy(source, tmp_path / "a.json")
    import ctypes
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("no clonefile")))
    assert clone_or_copy(source, tmp_path / "b.json") == "copy" and (tmp_path / "b.json").read_bytes() == source.read_bytes()
    from types import SimpleNamespace
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(clonefile=lambda *a: -1))
    assert clone_or_copy(source, tmp_path / "c.json") == "copy" and (tmp_path / "c.json").read_bytes() == source.read_bytes()
    (tmp_path / "a.json").write_bytes(b"changed")
    assert source.read_bytes() == b'{"x": 1}'  # independent files


# ---------------------------------------------------------------------------------------------------------------------
# Campaign: the manifest records the schema; admission checks it; the executor receives it.

def test_campaign_records_and_checks_the_evidence_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "require_runtime_environment", lambda *args, **kwargs: {"fixture": "environment"})
    from src.agent.auth import CLAUDE_SLOT_VOLUMES, CODEX_SLOT_VOLUMES
    # Runtime evidence of the current `docker prepare`: the lanes variant's three Codex and two Claude Code slots.
    monkeypatch.setattr(campaign, "load_execution_environment", lambda path: {"fixture": "execution-environment", "codex_slot_volumes": list(CODEX_SLOT_VOLUMES), "claude_slot_volumes": list(CLAUDE_SLOT_VOLUMES)})
    manifests = {}
    for config, data in (("small_scale.json", "outputs/small_scale_data"), ("harder_small_all.json", "data_harder")):
        if not (ROOT / data).is_dir():
            pytest.skip(f"{data} is not present")
        with contextlib.redirect_stdout(io.StringIO()):
            campaign.initialize(tmp_path / config, (ROOT / data).resolve(), ROOT / "configs" / config, None)
        manifests[config] = read(tmp_path / config / "campaign.json")
    small = manifests["small_scale.json"]
    assert small["solver_output_policy"] == OUTPUT_POLICY_32MIB and small["solver_evidence_schema"] == OUTPUT_BY_REFERENCE
    # The lanes variant reaches the manifest (its guard replaces the paper's); the harder manifest keeps the paper's settings.
    assert [lane["name"] for lane in small["lanes"]] == ["codex", "claude", "gemini"] and small["workers"] == 6 and small["resource_guard"]["max_rss_mib"] == 5120
    assert campaign.validate(tmp_path / "small_scale.json")["protocol_id"] == small["protocol_id"]
    harder = manifests["harder_small_all.json"]
    assert harder["workers"] == 2 and harder["resource_guard"] == {"sample_interval": 5, "max_cpu_percent": 250, "max_rss_mib": 2048, "min_free_percent": 10, "sustained_samples": 3}
    assert not {"lanes", "solver_memory_policy"} & set(harder)
    monkeypatch.setattr(campaign, "load_execution_environment", lambda path: {"fixture": "execution-environment", "codex_slot_volumes": list(CODEX_SLOT_VOLUMES[:2])})
    with pytest.raises(RuntimeEnvironmentError, match="docker prepare"):  # evidence prepared before the third Codex and second Claude slot
        campaign.initialize(tmp_path / "legacy-evidence", (ROOT / "outputs/small_scale_data").resolve(), ROOT / "configs/small_scale.json", None)
    assert not (tmp_path / "legacy-evidence").exists()
    assert "solver_evidence_schema" not in manifests["harder_small_all.json"] and "solver_output_policy" not in manifests["harder_small_all.json"]
    root = tmp_path / "admission"
    root.mkdir()
    campaign.ensure_execution_allowed(root, small)
    campaign.ensure_execution_allowed(root, {k: v for k, v in small.items() if k != "solver_evidence_schema"})  # an earlier 32 MiB manifest
    for change in ({"solver_evidence_schema": "native-output-by-reference-v0"}, {"solver_evidence_schema": None}):
        with pytest.raises(ProtocolError, match="evidence schema"):
            campaign.ensure_execution_allowed(root, small | change)
    with pytest.raises(ProtocolError, match="evidence schema"):
        campaign.ensure_execution_allowed(root, {k: v for k, v in small.items() if k != "solver_output_policy"})
