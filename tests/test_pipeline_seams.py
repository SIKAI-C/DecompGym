"""The decoupled agent pipeline: each part (execution policy, context / delivery / runtime specs, Part 2 sections,
audit extensions, family tracks, shared-first artifacts) accepts a second implementation without touching the paired
runner's core, while the defaults keep the paper and small-scale records (tests/test_single_file_golden.py,
tests/test_mounted_context.py and the paper-equivalence tests pin those byte for byte).

Synthetic only: no Docker engine, model call, compiler or optimizer. The large-scale policy below scores with the real
large-scale scorer (src.experiments.xl_scoring, pure Python) on scripted supervisor-like results.
"""
from __future__ import annotations

import ast
import base64
from contextlib import nullcontext
from dataclasses import asdict, replace
from fractions import Fraction
import importlib
import json
from pathlib import Path
import sys

import pytest

from src.agent import docker_client, multifile
from src.agent.container import ContainerResult, check_context
from src.agent.docker_client import CLI_VERSIONS, DockerCliClient
from src.agent.mounted_context import mounted_files
from src.agent.prompts.builder import PromptBuilder
from src.agent.prompts.judge import ALGORITHM_JUDGE, PAPER_AUDIT_TARGET, AuditExtension, AuditTarget, algorithm_review_prompt, judge_policy
from src.agent.prompts.method import COMMON_REQUIREMENTS, CONTRACTS, DECOMPOSITION_POLICY, DECOMPOSITION_POLICY_V3, METHOD_HEADER
from src.agent.prompts.mounted import (CONTEXT_DELIVERY_MOUNTED, MOUNTED_CPP_WORKSPACE_POLICY, PROSE_SECTIONS, TASK_HEADINGS, DirectSections, check_sections,
                                       render_task_part, workspace_part)
from src.agent.prompts.policy import OBJECTIVE_RETRY
from src.agent.result import AgentResult
from src.agent.specs import (DEFAULT_RUNTIME, HISTORY_DIR, INSTANCE_DIR, MOUNT_LAYOUT, MULTI_FILE, PREVIEW_ALONGSIDE_NAME, PREVIEW_LAYOUT, PREVIEW_NAME, SINGLE_FILE_DELIVERY,
                             SOURCES_LAYOUT, ContextRole, DeliverySpec, MountLayout, RuntimeSpec)
from src.experiments import campaign
from src.experiments.context_spec import (BOTH, LEGACY_CONTEXTS, MOUNTED, MOUNTED_CONTEXT, MOUNTED_WITH_PREVIEW, MOUNTED_WITH_SOURCES, PAPER_CONTEXT, PAPER_HISTORY, PREVIEW_CONTEXT,
                                          ContextSpec, HistorySets, context_spec)
from src.experiments.data_context import DATA_CONTEXT_BOTH, DATA_CONTEXT_PREVIEW, PREVIEW_THRESHOLD_BYTES, accepted_prompt_protocol
from src.experiments.execution_policy import GAP_TARGET, SMALL_SCALE_NATIVE_POLICY, Evaluation, ExecutionPolicy, ExecutionResult, SmallScaleNativePolicy
from src.experiments.paired_runner import AttemptResult, PairedRunner
from src.experiments.plan import EXECUTION, load_plan, planned_cases
from src.experiments.xl_scoring import Attempt, Reference, score_attempt
from src.utils.errors import InfrastructureError, ProtocolError
from src.utils.families import FAMILIES, LARGE_SCALE, SEEDS, SIZES, SMALL_SCALE, Track, cohort_tracks, track_family
from src.utils.records import digest, read, save
from host_solver import host_solver

ROOT = Path(__file__).resolve().parents[1]
MOUNTED_ID = CONTEXT_DELIVERY_MOUNTED
UFL_V3 = (ROOT / "data/bd/ufl/ufl_small_seed000/prompt.txt").read_text().replace("Use a master problem for the complicating decisions and recourse-cost estimates.",
                                                                               "Use a master problem for the complicating decisions and, when applicable, recourse-cost estimates.")


@pytest.fixture(autouse=True)
def independent_test_locks(monkeypatch):
    monkeypatch.setattr("src.experiments.paired_runner.runtime_lock", lambda name: nullcontext())
    docker_client._PREFLIGHTS.clear()
    multifile._PREFLIGHTS.clear()
    yield
    docker_client._PREFLIGHTS.clear()
    multifile._PREFLIGHTS.clear()


def source_case(tmp_path, prompt="Implement Benders decomposition.", data=None):
    source = tmp_path / "data/bd/ufl/ufl_small_seed000"
    source.mkdir(parents=True)
    (source / "data.json").write_text(json.dumps(data or {"m": 2, "n": 1, "f": [2, 3], "D": [[4], [1]]}))
    (source / "prompt.txt").write_text(prompt)
    (source / "meta.json").write_text('{"ground_truth_objective":4}')
    return source


class Client:
    """A scripted agent/judge/reflection client (the container boundary of the paired runner)."""

    def __init__(self, solvers=(), verdicts=(), context_delivery=None, **attributes):
        self.calls, self.solvers, self.verdicts = [], iter(solvers), iter(verdicts)
        self.context_delivery = context_delivery
        for name, value in attributes.items():
            setattr(self, name, value)

    def protocol(self):
        return {"backend": "test", "model": "fixture"}

    def edit_file(self, target, prompt, *, context, log_dir, step):
        self.calls.append((step, target, list(context), prompt))
        if step == "agent":
            target.write_text(next(self.solvers, "print('scenario: none')\n"))
        elif step == "llm_verifier":
            target.write_text(next(self.verdicts, "VERDICT: PASS\nNOTES: fixture"))
        else:
            target.write_text("Repair the preceding solver.")
        result = AgentResult(True, 0, 0.01, {"input_tokens": 7}, terminal_state="completed", artifact_sha256=digest(target))
        save(log_dir / f"{step}_metrics.json", asdict(result))
        return result


# ---------------------------------------------------------------------------------------------------------------------
# Execution and evaluation policy.

def original_stop(result, threshold):
    return result.status == "ok" and result.exec_ok and result.verify_ok and result.gap_to_gold is not None and result.gap_to_gold <= threshold


def original_feedback(result, threshold):
    import math
    values = (result.objective, result.gold_objective, result.gap_to_gold)
    return bool(result.exec_ok and result.feasibility_ok and not result.metrics.get("execution_timed_out")
                and all(isinstance(value, (int, float)) and math.isfinite(value) for value in values) and result.gap_to_gold > threshold)


@pytest.mark.parametrize("status", ["ok", "feasible_not_optimal", "verify_failed", "timeout", "exec_failed", "objective_not_available", "reference_unavailable", "agent_failed"])
def test_the_small_scale_policy_decides_exactly_as_the_paper_rule(status):
    policy = SmallScaleNativePolicy(executor=None, suite="bd")
    for exec_ok in (True, False):
        for verify_ok in (True, False):
            for feasible in (True, False):
                for gap in (None, 0.0, 0.005, 0.01, 0.0101, 0.5, float("nan"), float("inf")):
                    for timed_out in (True, False):
                        result = AttemptResult(1, status, exec_ok, verify_ok, 6.0, 4.0, gap, [], feasibility_ok=feasible, metrics={"execution_timed_out": timed_out})
                        evaluation = policy.evaluation(result)
                        assert evaluation.solved == original_stop(result, 0.01) and evaluation.headline == GAP_TARGET
                        assert evaluation.objective_target_missed == original_feedback(result, 0.01)
    runner = PairedRunner(client=None, judge_client=None, solver_executor=None)
    assert runner.policy.id == SMALL_SCALE_NATIVE_POLICY and runner.spec_record() == {}
    assert EXECUTION["solver_timeout"] == 60 and EXECUTION["solver_threads"] == 1 and EXECUTION["gap_threshold"] == 0.01


class SupervisedPolicy(ExecutionPolicy):
    """A large-scale-style policy: a supervisor run publishes incumbents (and certificate checkpoints) during T_f; the
    large-scale scorer decides per-target success, which counts also after a timeout and without a primal result."""
    id = "synthetic-supervised-publications-v0"
    nested = ("publication",)

    def __init__(self, reference):
        self.reference, self.executions = reference, []

    def references(self, source):
        return self.reference

    def reference_objective(self, references):
        return float(references.z_best)

    def execute(self, case, *, delivery, runtime, stage):
        stage("solver")
        scenario = json.loads(case.directory.joinpath("solver.py").read_text().split("SCENARIO=", 1)[1].splitlines()[0])
        self.executions.append(case.directory.name)
        publication = case.directory / "publication"
        publication.mkdir(exist_ok=True)
        for index, (t, value) in enumerate(scenario.get("incumbents", [])):
            (publication / f"incumbent_{index:03d}.json").write_text(json.dumps({"t": t, "objective": value}))
        (publication / "checks").mkdir(exist_ok=True)
        (publication / "checks/receipts.json").write_text(json.dumps(scenario.get("certificate", [])))
        termination = scenario["termination"]
        return ExecutionResult(termination == "completed", "", "[framework] solver execution timed out" if termination == "timeout" else "", None, float(runtime.timeout), termination,
                               {"execution_seconds": float(runtime.timeout), "publications": len(scenario.get("incumbents", []))},
                               {"incumbents": scenario.get("incumbents", []), "certificate": scenario.get("certificate")})

    def evaluate(self, result, references, *, case, audit):
        verdict, notes = audit()
        record = score_attempt(Attempt(incumbents=[tuple(row) for row in result.evidence["incumbents"]], method_ok=verdict == "pass", certificate=result.evidence["certificate"],
                                       termination=result.termination), references)
        return self._evaluation(record, references.headline, result.termination, verdict, notes, tuple(record["infrastructure_errors"]))

    def evaluation(self, result):
        """Restored from the self-contained record alone (it travels with skipped rows as well)."""
        saved = result.metrics["evaluation"]
        return self._evaluation(saved, saved["headline"]["target"], saved["termination"], saved["method_verdict"], saved["method_notes"], ())

    def _evaluation(self, record, headline, termination, verdict, notes, infrastructure):
        success = dict(record["success"])
        solved = success.get(headline) is True
        feasible = bool(record.get("feasible"))
        kept = {"success": success, "headline": {"target": headline, "success": success.get(headline)}, "feasible": feasible, "objective": record.get("objective"), "termination": termination,
                "method_verdict": verdict, "method_notes": notes}
        return Evaluation(status="ok" if solved else ("target_missed" if feasible else "no_eligible_result"), feasible=feasible, verified=verdict == "pass" and (feasible or solved),
                          objective=record.get("objective"), reference_objective=float(self.reference.z_best), gap=None, termination=termination, success=success, headline=headline,
                          issues=[] if verdict == "pass" else ["algorithm_verifier_failed: " + verdict], method_verdict=verdict, method_notes=notes,
                          objective_target_missed=feasible and not solved and verdict == "pass", infrastructure=infrastructure, record=kept)


RESTORED = ("success", "headline", "termination", "feasible", "verified", "method_verdict", "method_notes", "objective", "solved")


def assert_restored_like(policy, row, successful):
    """A skipped row restores the successful attempt's Evaluation, field by field."""
    restored, expected = policy.evaluation(AttemptResult(**row)), policy.evaluation(successful)
    assert {name: getattr(restored, name) for name in RESTORED} == {name: getattr(expected, name) for name in RESTORED} and restored.solved


def scenario(**value):
    return "SCENARIO=" + json.dumps(value) + "\nprint('supervised candidate')\n"


def test_success_after_a_timeout_counts_and_the_nested_evidence_is_shared(tmp_path):
    reference = Reference(T_f=60.0, z_best=Fraction(4), integral=True, eps_p=Fraction("0.01"), headline="near_best")
    policy = SupervisedPolicy(reference)
    source = source_case(tmp_path)
    client = Client([scenario(termination="timeout", incumbents=[[12.0, 5], [40.0, 4], [61.0, 3]])])
    summaries = PairedRunner(client=client, judge_client=client, model="fixture", suite="bd", problem=["ufl"], solver_executor=None, max_attempts=3, execution_policy=policy).run_pair(source, tmp_path / "runs")
    for mode in ("plain", "reflection"):
        attempts = summaries[mode]["attempts"]
        first = AttemptResult(**attempts[0])
        assert first.status == "ok" and first.exec_ok is False and first.metrics["evaluation"]["termination"] == "timeout"
        assert first.metrics["evaluation"]["success"]["near_best"] is True and first.metrics["evaluation"]["objective"] == 4  # the receipt at 61 s is not eligible
        assert not PairedRunner.can_early_stop(first, 0.01)  # the paper's rule would have continued
        assert [a["status"] for a in attempts[1:]] == ["skipped_previous_solved"] * 2
        assert all(a["exec_ok"] is False and a["metrics"]["evaluation"] == first.metrics["evaluation"] for a in attempts[1:])  # never restated as a completed run
        for row in attempts[1:]:
            assert_restored_like(policy, row, first)
        assert policy.evaluation(first).termination == "timeout" and policy.evaluation(first).method_notes == "VERDICT: PASS\nNOTES: fixture"
    assert policy.executions == ["attempt_01"]  # one shared first execution
    plain = tmp_path / "runs/plain/bd/ufl/ufl_small_seed000/attempts/attempt_01"
    shared = tmp_path / "runs/reflection/bd/ufl/ufl_small_seed000/attempts/attempt_01"
    nested = sorted(p.relative_to(plain).as_posix() for p in (plain / "publication").rglob("*") if p.is_file())
    assert nested == ["publication/checks/receipts.json", "publication/incumbent_000.json", "publication/incumbent_001.json", "publication/incumbent_002.json"]
    assert all(digest(shared / name) == digest(plain / name) for name in nested)
    assert not (shared / "calls").exists()  # call records stay with the canonical first attempt, as in the paper layout
    assert set(nested) <= set(read(plain / "score_receipt.json")["artifact_sha256"])  # the score receipt pins the nested evidence
    identity = read(tmp_path / "runs/plain/bd/ufl/ufl_small_seed000/identity.json")
    assert identity["specs"] == {"execution_policy": policy.id}
    assert all(read(path)["specs"] == {"execution_policy": policy.id} for path in (tmp_path / "runs").rglob("request.json"))
    # Resume restores the decision from the durable record without executing again.
    calls = len(client.calls)
    PairedRunner(client=client, judge_client=client, model="fixture", suite="bd", problem=["ufl"], solver_executor=None, max_attempts=3, execution_policy=policy).run_pair(source, tmp_path / "runs")
    assert len(client.calls) == calls and policy.executions == ["attempt_01"]
    # A changed nested artifact in the projection is refused like a changed solver.py.
    (shared / "publication/incumbent_001.json").write_text("{}")
    runner = PairedRunner(client=client, judge_client=client, model="fixture", suite="bd", problem=["ufl"], solver_executor=None, max_attempts=3, execution_policy=policy)
    with pytest.raises(ProtocolError, match="projection differs"):
        runner.copy_shared_first(plain, shared)
    # The policy id is part of the case protocol: the default policy cannot resume this pair.
    with pytest.raises(ProtocolError, match="Case protocol changed"):
        PairedRunner(client=client, judge_client=client, model="fixture", suite="bd", problem=["ufl"], solver_executor=host_solver, max_attempts=3).run_pair(source, tmp_path / "runs")


BOUND = dict(T_f=900.0, z_best=Fraction(1010), z_mip=Fraction(1012), lp_lo=Fraction("999.9999"), lp_hi=Fraction(1000), eps_b=Fraction("0.0001"), eps_p=Fraction("0.001"), headline="bound_target")


def checkpoints(*bounds):
    return [{"checkpoint": 225.0 * (k + 1), "lb": str(bound), "cert_ok": True, "receipt_t": 100.0 * (k + 1)} for k, bound in enumerate(bounds)]


def test_a_bound_only_success_stops_and_a_missed_target_gives_the_objective_feedback(tmp_path):
    policy = SupervisedPolicy(Reference(**BOUND))
    source = source_case(tmp_path)
    missed = scenario(termination="completed", incumbents=[[100.0, 1100]], certificate=checkpoints(900, 950, 990, 995))  # feasible, bound gap 0.5 %: missed
    bound_only = scenario(termination="crashed", incumbents=[], certificate=checkpoints(900, 999.95, 999.95, 999.99))  # no primal, certified bound: success
    client = Client([missed, bound_only, bound_only])  # plain attempts 1 (shared) and 2, then the reflection policy's attempt 2
    summaries = PairedRunner(client=client, judge_client=client, model="fixture", suite="bd", problem=["ufl"], solver_executor=None, max_attempts=3, execution_policy=policy).run_pair(source, tmp_path / "runs")
    reflection = summaries["reflection"]["attempts"]
    assert [a["status"] for a in reflection] == ["target_missed", "ok", "skipped_previous_solved"]
    second = AttemptResult(**reflection[1])
    assert second.feasibility_ok is False and second.metrics["evaluation"]["success"] == {"matches_or_beats_MIP": False, "near_best": False, "bound_target": True}
    assert reflection[2]["feasibility_ok"] is False and reflection[2]["exec_ok"] is False and reflection[2]["metrics"]["evaluation"] == second.metrics["evaluation"]
    for mode in ("plain", "reflection"):
        rows = summaries[mode]["attempts"]
        assert_restored_like(policy, rows[2], AttemptResult(**rows[1]))
    restored = policy.evaluation(AttemptResult(**reflection[2]))
    assert restored.headline == "bound_target" and not restored.feasible and restored.verified and restored.termination == "crashed" and restored.success["bound_target"] is True
    prompts = [prompt for step, _, _, prompt in client.calls if step == "reflection"]
    assert len(prompts) == 1 and OBJECTIVE_RETRY in prompts[0]  # feedback from Evaluation.objective_target_missed, not from a gap to the gold
    assert [a["status"] for a in summaries["plain"]["attempts"]] == ["target_missed", "ok", "skipped_previous_solved"]


class OmittingClient(Client):
    """The first generation completes without delivering solver.py (a scored agent failure, never executed)."""

    def edit_file(self, target, prompt, *, context, log_dir, step):
        if step == "agent" and not any(call[0] == "agent" for call in self.calls):
            self.calls.append((step, target, list(context), prompt))
            target.write_text("")
            result = AgentResult(False, 0, 0.01, {}, "Completed CLI did not deliver the requested file", terminal_state="completed")
            save(log_dir / f"{step}_metrics.json", asdict(result))
            return result
        return super().edit_file(target, prompt, context=context, log_dir=log_dir, step=step)


def test_a_generation_failure_is_a_lifecycle_record_that_no_policy_has_to_interpret(tmp_path):
    reference = Reference(T_f=60.0, z_best=Fraction(4), integral=True, eps_p=Fraction("0.01"), headline="near_best")
    policy = SupervisedPolicy(reference)
    client = OmittingClient([scenario(termination="completed", incumbents=[[5.0, 4]])] * 3)
    summaries = PairedRunner(client=client, judge_client=client, model="fixture", suite="bd", problem=["ufl"], solver_executor=None, max_attempts=3, execution_policy=policy).run_pair(source_case(tmp_path), tmp_path / "runs")
    for mode in ("plain", "reflection"):
        assert [a["status"] for a in summaries[mode]["attempts"]] == ["agent_failed", "ok", "skipped_previous_solved"]
    assert policy.executions == ["attempt_02", "attempt_02"]  # the failed generation was never executed
    reflection = [prompt for step, _, _, prompt in client.calls if step == "reflection"]
    assert len(reflection) == 1 and OBJECTIVE_RETRY not in reflection[0]
    assert (tmp_path / "runs/reflection/bd/ufl/ufl_small_seed000/attempts/attempt_01/exec.log").read_text().startswith("[framework generation diagnostic]")


def test_layout_specs_are_hashable_values():
    assert hash(MOUNT_LAYOUT) == hash(MountLayout(MOUNT_LAYOUT.id, MOUNT_LAYOUT.roles)) and len({MOUNT_LAYOUT, PREVIEW_LAYOUT, MOUNT_LAYOUT}) == 2
    assert len({PAPER_CONTEXT, MOUNTED_CONTEXT, MOUNTED_WITH_PREVIEW, context_spec()}) == 3


def test_an_undefined_score_is_infrastructure_and_never_scored(tmp_path):
    policy = SupervisedPolicy(Reference(**BOUND))
    source = source_case(tmp_path)
    client = Client([scenario(termination="completed", incumbents=[[100.0, 1100]], certificate=None)])  # a BD-LP score needs checked checkpoints
    with pytest.raises(InfrastructureError, match="no_checked_certificate_checkpoints"):
        PairedRunner(client=client, judge_client=client, model="fixture", suite="bd", problem=["ufl"], solver_executor=None, max_attempts=2, execution_policy=policy).run_pair(source, tmp_path / "runs")
    attempt = tmp_path / "runs/plain/bd/ufl/ufl_small_seed000/attempts/attempt_01"
    assert not (attempt / "score_receipt.json").exists() and not (attempt / "summary.json").exists()


# ---------------------------------------------------------------------------------------------------------------------
# Context, delivery and runtime specs.

def test_the_recorded_settings_select_the_reviewed_context_specs():
    assert context_spec() is PAPER_CONTEXT and context_spec(DATA_CONTEXT_PREVIEW) is PREVIEW_CONTEXT and context_spec(None, MOUNTED_ID) is MOUNTED_CONTEXT
    assert [(spec.data_context, spec.context_delivery) for spec in LEGACY_CONTEXTS.values()] == list(LEGACY_CONTEXTS)
    assert (MOUNTED_CONTEXT.id, PREVIEW_CONTEXT.id) == (MOUNTED_ID, DATA_CONTEXT_PREVIEW) and all(spec.history is PAPER_HISTORY for spec in LEGACY_CONTEXTS.values())
    with pytest.raises(ValueError, match="no data-context preview"):
        context_spec(DATA_CONTEXT_PREVIEW, MOUNTED_ID)
    with pytest.raises(ValueError, match="Unknown data-context"):
        context_spec(DATA_CONTEXT_BOTH)
    # Representation is independent of transport at the spec level; the reviewed protocols still refuse the combination.
    assert MOUNTED_WITH_PREVIEW.mounted and MOUNTED_WITH_PREVIEW.representation == BOTH and MOUNTED_WITH_PREVIEW.data_context == DATA_CONTEXT_BOTH
    plan = load_plan(ROOT / "configs/small_scale.json")
    assert not accepted_prompt_protocol({**plan["prompt_protocol"], "data_context": DATA_CONTEXT_PREVIEW})
    with pytest.raises(ValueError, match="data_preview role"):
        ContextSpec("mounted-preview-without-role", transport=MOUNTED, representation=BOTH)
    with pytest.raises(ValueError, match="either a ContextSpec"):
        PairedRunner(client=None, judge_client=None, solver_executor=None, context=MOUNTED_WITH_PREVIEW, context_delivery=MOUNTED_ID)


def test_the_paper_history_sets_keep_plain_regeneration_without_history(tmp_path):
    case = tmp_path / "case"
    for index in (1, 2):
        directory = case / "attempts" / f"attempt_{index:02d}"
        directory.mkdir(parents=True)
        for name in ("solver.py", "debug.md", "reflection_feedback.md", "exec.log", "alg_verifier.md", "output.json", "notes.txt"):
            (directory / name).write_text(name)
    (case / "data.json").write_text("{}")
    attempts = case / "attempts"
    assert PAPER_CONTEXT.generation(case, attempts, 3, "plain") == ([case / "data.json"], None)
    assert PAPER_CONTEXT.generation(case, attempts, 1, "reflection") == ([case / "data.json"], None)
    files, _ = PAPER_CONTEXT.generation(case, attempts, 3, "reflection")
    assert [(p.parent.name, p.name) for p in files[1:]] == [(f"attempt_{i:02d}", n) for i in (1, 2) for n in ("solver.py", "debug.md", "reflection_feedback.md")]
    assert [p.name for p in PAPER_CONTEXT.reflection(case, attempts / "attempt_02")[0][1:]] == ["solver.py", "exec.log", "alg_verifier.md", "output.json"]
    other = replace(PAPER_CONTEXT, id="history-with-notes", history=HistorySets("with-notes", ("solver.py", "notes.txt"), ("solver.py", "notes.txt")))
    assert [p.name for p in other.generation(case, attempts, 3, "reflection")[0][1:]] == ["solver.py", "notes.txt"] * 2
    assert other.generation(case, attempts, 3, "plain")[0] == [case / "data.json"]  # no history set can reach a plain regeneration


class MountedFakeRuntime:
    """The container boundary of DockerCliClient: records each call's job and context."""

    def __init__(self, root, backend="codex-cli", contents="def solve():\n    return 1\n"):
        self.seccomp_profile = root / "profile.json"
        self.seccomp_profile.write_text("{}")
        self.backend, self.contents, self.calls = backend, contents, []

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
        return ContainerResult(0, "", "", 1, artifacts={job["target"]: self.contents, "worker.json": json.dumps(worker), "delivery.json": json.dumps({"rejected_target": None, "misplaced": {}})},
                               metadata={"cleanup_verified": True})


def big_case(tmp_path):
    case = tmp_path / "case"
    (case / "attempts/attempt_01").mkdir(parents=True)
    (case / "data.json").write_text(json.dumps({"n": 1, "values": list(range(150_000))}))
    assert (case / "data.json").stat().st_size > PREVIEW_THRESHOLD_BYTES
    return case


def test_mounted_context_can_carry_the_full_data_and_its_preview(tmp_path):
    case = big_case(tmp_path)
    files, record = MOUNTED_WITH_PREVIEW.instance_files(case)
    assert [p.name for p in files] == ["data.json", PREVIEW_ALONGSIDE_NAME] and record["delivered"] == "full_and_structural_preview" and record["policy"] == DATA_CONTEXT_BOTH
    assert "The complete data.json is provided as well" in (case / PREVIEW_ALONGSIDE_NAME).read_text()
    assert PREVIEW_CONTEXT.instance_files(case)[0] == [case / PREVIEW_NAME]  # the replacing preview is another file: both representations coexist
    runtime = MountedFakeRuntime(tmp_path)
    client = DockerCliClient("m", backend="codex-cli", image_id="sha256:x", runtime=runtime, evidence_root=tmp_path / "evidence", credential_provider=lambda _: {},
                             context_delivery=MOUNTED_ID, context_layout=PREVIEW_LAYOUT)
    target = case / "attempts/attempt_01/solver.py"
    target.write_text("")
    assert client.edit_file(target, "# Part 2: Task", context=files, log_dir=tmp_path / "call").ok
    prompt = runtime.calls[0]["payload"]["prompt"]
    lines = prompt.split("## Input data\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert lines[0].startswith(f"- For reading during this call: {INSTANCE_DIR}/data.json (") and lines[1].startswith(f"- For reading during this call: {INSTANCE_DIR}/{PREVIEW_ALONGSIDE_NAME} (")
    assert lines[2].startswith("- At run time:")
    assert sorted(runtime.calls[0]["context"][INSTANCE_DIR]) == ["data.json", PREVIEW_ALONGSIDE_NAME] and runtime.calls[0]["context_directories"] == PREVIEW_LAYOUT.directories
    evidence = read(tmp_path / "call/agent_context.json")
    assert evidence["layout"] == PREVIEW_LAYOUT.id and [entry["path"] for entry in evidence["files"]] == [f"{INSTANCE_DIR}/data.json", f"{INSTANCE_DIR}/{PREVIEW_ALONGSIDE_NAME}"]
    with pytest.raises(ProtocolError, match="No mounted location"):
        mounted_files(files)  # the default layout declares no preview role (fail-closed)


def test_the_runner_composes_and_records_an_explicit_context_spec(tmp_path):
    source = source_case(tmp_path, UFL_V3, data={"m": 2, "n": 1, "f": [2, 3], "D": [[4], [1]], "padding": list(range(150_000))})
    good = "import json\nfrom pathlib import Path\nPath('output.json').write_text(json.dumps({'status':'feasible','method':'benders_decomposition','y':[0,1],'x':[[0],[1]],'objective_value':4}))\n"
    client = Client([good], context_delivery=MOUNTED_ID, context_layout=PREVIEW_LAYOUT)
    judge = Client()
    runner = PairedRunner(client=client, judge_client=judge, model="fixture", suite="bd", problem=["ufl"], solver_executor=host_solver, max_attempts=1, decomposition_contract=DECOMPOSITION_POLICY_V3,
                          context=MOUNTED_WITH_PREVIEW)
    summaries = runner.run_pair(source, tmp_path / "runs")
    assert summaries["plain"]["attempts"][0]["status"] == "ok"
    request = next(read(path) for path in (tmp_path / "runs/plain").rglob("calls/*/request.json") if read(path)["step"] == "agent")
    assert request["specs"] == {"context": MOUNTED_WITH_PREVIEW.id} and request["data_context"]["delivered"] == "full_and_structural_preview"
    assert [entry["path"] for entry in request["context_delivery"]["files"]] == [f"{INSTANCE_DIR}/data.json", f"{INSTANCE_DIR}/{PREVIEW_ALONGSIDE_NAME}"]
    identity = read(tmp_path / "runs/plain/bd/ufl/ufl_small_seed000/identity.json")
    assert identity["context_delivery"] == MOUNTED_ID and identity["data_context"] == DATA_CONTEXT_BOTH and identity["specs"] == {"context": MOUNTED_WITH_PREVIEW.id}
    # A client that places files with another layout is refused before any call.
    other = PairedRunner(client=Client(context_delivery=MOUNTED_ID), judge_client=judge, model="fixture", suite="bd", problem=["ufl"], solver_executor=host_solver, decomposition_contract=DECOMPOSITION_POLICY_V3,
                         context=MOUNTED_WITH_PREVIEW)
    target = tmp_path / "x/attempt_01/solver.py"
    target.parent.mkdir(parents=True)
    with pytest.raises(ProtocolError, match="layout or runtime contract"):
        other.edit(other.client, target, "x", [], "agent")


def test_layout_roles_are_declared_and_fail_closed(tmp_path):
    docs = ContextRole("docs", "/work/docs", {"api.md": "the publication API reference"})
    layout = MountLayout("instance-history-docs-test", (*MOUNT_LAYOUT.roles, docs))
    (tmp_path / "api.md").write_text("publish_incumbent(payload)\n")
    (tmp_path / "data.json").write_text("{}")
    assert layout.locate(tmp_path / "api.md")[:2] == ("/work/docs/api.md", "the publication API reference")
    assert layout.directories == (HISTORY_DIR, INSTANCE_DIR, "/work/docs")
    check_context({"/work/docs": {"api.md": b"x"}}, layout.directories)
    with pytest.raises(ValueError):
        check_context({"/work/docs": {"api.md": b"x"}})  # not a directory of the default layout
    with pytest.raises(ValueError):
        MountLayout("bad", (ContextRole("deep", "/work/a/b", {"x": "y"}),))
    for name in ("meta.json", "gold_monolithic.json", "prompt.txt"):
        (tmp_path / name).write_text("{}")
        with pytest.raises(ProtocolError):
            layout.locate(tmp_path / name)


def test_single_file_delivery_and_default_runtime_render_the_earlier_text():
    for target in ("solver.py", "debug.md"):
        writable = f"/work/{target}"
        assert SINGLE_FILE_DELIVERY.modify_text(target) == (f"That file is {writable}; do not create, rename or delete any file. Every other path in this workspace is read-only context. "
                                                            f"Only {writable} is collected after this call: anything written elsewhere is ignored.")
        assert SINGLE_FILE_DELIVERY.writable_text(target) == writable and SINGLE_FILE_DELIVERY.instruction(target) == f"Only edit the existing {target} in the current working directory."
    assert DEFAULT_RUNTIME.input_lines() == ["- At run time: the harness later runs solver.py in a separate attempt directory, which is its working directory, and solver.py must read its input from ../../data.json relative to that working directory. "
                                             "The /work paths of this workspace do not exist at run time, so solver.py must not use them."]
    assert DEFAULT_RUNTIME.output_section() == ("## Output file\nAt run time solver.py must write its result to output.json in its working directory. The harness reads only that file: a missing or misplaced output.json is an execution failure. "
                                                "Do not execute the generated solver. The harness executes it later in its documented runtime directory.")
    assert DEFAULT_RUNTIME.harness_rules() == ("Implement a standalone solver.py using gurobipy, NumPy, and the Python standard library. Write output.json using the documented contract. Do not import repository implementations or load reference solutions.\n"
                                               "The harness limits the entire solver.py execution to 60 seconds and numerical solvers to 1 thread(s). Design your algorithm and termination handling within this budget.")
    assert (DEFAULT_RUNTIME.id, DEFAULT_RUNTIME.output_limit, RuntimeSpec(60, 1, "output-json-readback-32MiB-v1").output_limit) == ("separate-edit-runtime-v1", 4 * 2**20, 32 * 2**20)
    with pytest.raises(ValueError):
        RuntimeSpec(60, 1, "output-json-readback-64MiB-v1")


class SupervisorRuntime(RuntimeSpec):
    """A large-scale-style runtime: cwd /work, input /work/data.json, and a publication API."""
    id = "synthetic-supervisor-cwd-runtime-v0"
    input_path = "/work/data.json"
    publication = "Publish every useful incumbent during the run with decompgym_publish.publish_incumbent(payload); only receipts strictly before the deadline count."

    def input_lines(self, workspace="/work"):
        return [f"- At run time: the harness runs solver.py with working directory {workspace}; solver.py must read its input from {self.input_path}."]

    def output_section(self):
        return "## Output file\nThe harness scores the published incumbents; a final output.json is kept for inspection only."

    def harness_rules(self):
        return ("Implement a standalone solver.py using gurobipy, NumPy, the Python standard library and decompgym_publish. Do not import repository implementations or load reference solutions.\n"
                f"The harness limits the entire solver.py execution to {self.timeout} seconds and numerical solvers to {self.threads} thread(s).\n" + self.publication)


def test_a_second_runtime_contract_reaches_part_one_and_part_two_and_is_recorded(tmp_path):
    runtime = SupervisorRuntime(900, 1)
    case = tmp_path / "case"
    (case / "attempts/attempt_01").mkdir(parents=True)
    (case / "data.json").write_text("{}")
    workspace, _ = workspace_part("solver.py", mounted_files([case / "data.json"]), "codex-cli", runtime=runtime)
    assert "solver.py must read its input from /work/data.json." in workspace and "kept for inspection only" in workspace and "../../data.json" not in workspace
    part2 = PromptBuilder.structured_generation(PROSE_SECTIONS(source_case(tmp_path / "s", UFL_V3), suite="bd", problem="ufl", contract=DECOMPOSITION_POLICY_V3), runtime=runtime)
    assert part2.endswith("1 thread(s).\n" + runtime.publication) and "to 900 seconds" in part2 and "Write output.json" not in part2
    with pytest.raises(ValueError, match="native policy"):
        PairedRunner(client=None, judge_client=None, solver_executor=None, solver_timeout=900, runtime_spec=runtime)  # the native sandbox cannot run it as stated
    policy = SupervisedPolicy(Reference(T_f=900.0, z_best=Fraction(4), eps_p=Fraction("0.01"), headline="near_best"))
    runner = PairedRunner(client=Client(context_delivery=MOUNTED_ID, runtime_spec=runtime), judge_client=Client(), solver_executor=None, solver_timeout=900, context_delivery=MOUNTED_ID,
                          decomposition_contract=DECOMPOSITION_POLICY_V3, runtime_spec=runtime, execution_policy=policy)
    assert runner.spec_record() == {"runtime": runtime.id, "execution_policy": policy.id}
    with pytest.raises(ValueError, match="budget"):
        PairedRunner(client=None, judge_client=None, solver_executor=None, runtime_spec=runtime)
    target = tmp_path / "attempt_01/solver.py"
    target.parent.mkdir()
    with pytest.raises(ProtocolError, match="layout or runtime contract"):
        runner.edit(Client(context_delivery=MOUNTED_ID), target, "x", [], "agent")  # the client would describe the default runtime


@pytest.fixture(scope="module")
def worker():
    """The multi-file container module, imported with the image's flat /opt/decompbench layout (tests/test_multifile_delivery.py)."""
    added = [str(ROOT / "docker/agents"), str(ROOT / "src/agent")]
    sys.path[:0] = added
    try:
        return importlib.import_module("multifile_worker")
    finally:
        for path in added:
            sys.path.remove(path)


class MultiFileFakeRuntime(multifile.MultiFileDockerRuntime):
    def __init__(self, root, bundle, solver):
        super().__init__(seccomp_profile=root / "profile.json")
        self.seccomp_profile.write_text("{}")
        self.bundle, self.solver, self.calls = bundle, solver, []

    def ensure_volume(self, *args, **kwargs):
        pass

    def volume_users(self, volume):
        return []

    def run(self, image, command, **kwargs):
        kwargs["state_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["state_path"].write_text("{}")
        if command == ["preflight"]:
            value = {"ok": True, "versions": {"codex": CLI_VERSIONS["codex-cli"], "claude": CLI_VERSIONS["claude-code-cli"], "agy": CLI_VERSIONS["agy-cli"]},
                     "worker_files_sha256": docker_client.worker_file_hashes(), "extension_files_sha256": multifile.extension_file_hashes()}
            return ContainerResult(0, "", "", 0, artifacts={"worker.json": json.dumps(value)})
        self.calls.append(kwargs)
        worker = {"cli_version": CLI_VERSIONS["codex-cli"], "requested_model": "fixture", "target": "solver.py", "returncode": 0, "terminal_state": "completed", "error": None,
                  "usage": {}, "usage_available": True, "warnings": [], "observed_models": ["fixture"], "agent_seconds": 1.0}
        misplaced = {"history/attempt_01/solver.py": "print(1)\n"}  # the worker's misplaced-target search also meets the mounted copies
        return ContainerResult(0, "", "", 1, artifacts={"solver.py": self.solver, "worker.json": json.dumps(worker), "delivery.json": json.dumps({"rejected_target": None, "misplaced": misplaced}),
                                                        multifile.BUNDLE_NAME: self.bundle}, metadata={"cleanup_verified": True})


def test_multi_file_delivery_composes_with_mounted_context(tmp_path, worker):
    files = {"solver.py": b"import os\n", "cpp/core.cpp": b"int f(){return 1;}\n"}
    rejected = [{"path": "history", "reason": "not allowed"}, {"path": "instance", "reason": "not allowed"}]  # the read-only mounts, as the worker lists them
    manifest = worker.manifest_for(files, rejected, "cpp_allowed")
    bundle = json.dumps({"version": worker.RECORD_VERSION, "mode": MULTI_FILE, "track": "cpp_allowed", "manifest": manifest, "error": None, "error_rejected": [],
                         "files": {path: base64.b64encode(data).decode() for path, data in files.items()}})
    runtime = MultiFileFakeRuntime(tmp_path, bundle, "import os\n")
    delivery = DeliverySpec(MULTI_FILE, "cpp_allowed")
    client = multifile.MultiFileDockerCliClient("fixture", backend="codex-cli", image_id="sha256:fixture", runtime=runtime, evidence_root=tmp_path / "evidence", delivery=delivery,
                                                credential_provider=lambda _: {}, context_delivery=MOUNTED_ID)
    case = tmp_path / "case"
    (case / "attempts/attempt_01").mkdir(parents=True)
    (case / "attempts/attempt_01/solver.py").write_text("print(1)\n")
    (case / "data.json").write_text('{"n": 1}')
    target = case / "attempts/attempt_02/solver.py"
    target.parent.mkdir()
    result = client.edit_file(target, "# Part 2: Task", context=[case / "data.json", case / "attempts/attempt_01/solver.py"], log_dir=tmp_path / "call")
    assert result.ok and multifile.load_delivery(target, tmp_path / "call") == (manifest, files) and (target.parent / "cpp/core.cpp").read_bytes() == files["cpp/core.cpp"]
    call = runtime.calls[0]
    assert sorted(call["context"]) == [HISTORY_DIR, INSTANCE_DIR] and call["payload"]["delivery"] == {"mode": MULTI_FILE, "track": "cpp_allowed"}
    prompt = call["payload"]["prompt"]
    assert prompt.startswith("# Part 1: Workspace and files\n\n## File to modify\nOnly edit the existing solver.py in the current working directory. " + multifile.CPP_NOTE)
    assert "Apart from /work/solver.py and those C++ sources in /work/cpp/, every path in this workspace is read-only context." in prompt
    assert "No other task files are provided: apart from the files listed above and /work/solver.py and /work/cpp/, the workspace contains no data, references or results of this or any other instance;" in prompt and "BEGIN CONTEXT" not in prompt
    assert read(tmp_path / "call/agent_workspace.json")["version"] == MOUNTED_CPP_WORKSPACE_POLICY
    assert read(tmp_path / "call/agent_context.json")["writable"] == ["/work/solver.py", "/work/cpp/*.cpp", "/work/cpp/*.hpp"]
    assert read(tmp_path / "call/agent_context_copies_ignored.json") == {"ignored_mounted_copies": ["history/attempt_01/solver.py"]}
    assert read(tmp_path / "call/agent_invocation.json")["context_delivery"] == MOUNTED_ID
    # The method judge reviews every delivered source; a runner and a client must agree on the delivery.
    assert [p.relative_to(target.parent).as_posix() for p in delivery.reviewed_sources(target.parent, tmp_path / "call")] == ["solver.py", "cpp/core.cpp"]
    (target.parent / "cpp/extra.cpp").write_text("int g;\n")
    with pytest.raises(ProtocolError, match="differs from the committed delivery"):
        delivery.reviewed_sources(target.parent, tmp_path / "call")  # the inspection copy must be exactly the delivery
    (target.parent / "cpp/extra.cpp").unlink()
    with pytest.raises(ValueError, match="native policy"):
        PairedRunner(client=Client(), judge_client=Client(), solver_executor=None, delivery=delivery)  # the native sandbox runs no C++ build
    policy = SupervisedPolicy(Reference(T_f=60.0, z_best=Fraction(4), eps_p=Fraction("0.01"), headline="near_best"))
    runner = PairedRunner(client=Client(), judge_client=Client(), solver_executor=None, delivery=delivery, execution_policy=policy)
    assert runner.spec_record() == {"delivery": "multi-file-cpp_allowed-v1", "execution_policy": policy.id} and SINGLE_FILE_DELIVERY.nested == () and delivery.nested == ("cpp",)
    with pytest.raises(ProtocolError, match="delivery differs"):
        runner.edit(Client(), target, "x", [], "agent")


# ---------------------------------------------------------------------------------------------------------------------
# Part 2 sections and the method audit.

def synthetic_sections(data, *, suite, problem, contract):
    return {"Problem": [f"Synthetic {problem} task: choose sites to minimise cost."], "Data": [f"data.json holds {sorted(data)}."],
            "Output": ["Task:\nWrite a Python program solver.py that writes output.json with the chosen sites."],
            "Method requirements": [METHOD_HEADER + "\n\n" + CONTRACTS[contract][suite], "Publish every certified bound through the publication API."],
            "Rules": ["Assume gurobipy and NumPy are available.", *COMMON_REQUIREMENTS]}


def test_a_family_track_can_supply_its_sections_directly(tmp_path):
    adapter = DirectSections("synthetic-family-sections-v0", synthetic_sections)
    case = tmp_path / "case"
    case.mkdir()
    (case / "data.json").write_text('{"sites": 3, "cost": [1, 2, 3]}')
    sections = adapter(case, suite="bd", problem="synthetic_sites", contract=DECOMPOSITION_POLICY_V3)
    text = PromptBuilder.structured_generation(sections, runtime=DEFAULT_RUNTIME)
    assert [line for line in text.splitlines() if line.startswith("#")] == ["# Part 2: Task"] + [f"## {heading}" for heading in TASK_HEADINGS]
    assert "Synthetic synthetic_sites task" in text and "data.json holds ['cost', 'sites']." in text and text.endswith(DEFAULT_RUNTIME.harness_rules())
    reflection = PromptBuilder.structured_reflection(sections, runtime=DEFAULT_RUNTIME, objective_quality_failed=True)
    assert reflection.split("\n\n# Part 3: Reflection\n")[0].replace("\n\nYou are making an attempt to solve the following problem.", "", 1) == text
    for broken in ({k: v for k, v in sections.items() if k != "Rules"}, dict(sections, Rules=["Assume nothing."]), dict(sections, Data=[" "]),
                   dict(sections, **{"Method requirements": [METHOD_HEADER + "\n\n" + CONTRACTS[DECOMPOSITION_POLICY]["bd"]]})):
        with pytest.raises(ProtocolError):
            check_sections(broken, suite="bd", problem="synthetic_sites", contract=DECOMPOSITION_POLICY_V3)
    # The prose adapter is the earlier task_sections parser on the stored prompt.txt.
    source = source_case(tmp_path / "prose", UFL_V3)
    assert render_task_part(PROSE_SECTIONS(source, suite="bd", problem="ufl", contract=DECOMPOSITION_POLICY_V3), harness_rules=DEFAULT_RUNTIME.harness_rules()) == \
        PromptBuilder.build_structured_generation_prompt(UFL_V3, suite="bd", problem="ufl", contract=DECOMPOSITION_POLICY_V3, solver_timeout=60, solver_threads=1)
    # In the runner: Part 2 comes from the adapter and the adapter id is recorded.
    good = "import json\nfrom pathlib import Path\nPath('output.json').write_text(json.dumps({'status':'feasible','method':'benders_decomposition','y':[0,1],'x':[[0],[1]],'objective_value':4}))\n"
    client = Client([good], context_delivery=MOUNTED_ID)
    runner = PairedRunner(client=client, judge_client=Client(), model="fixture", suite="bd", problem=["ufl"], solver_executor=host_solver, max_attempts=1, decomposition_contract=DECOMPOSITION_POLICY_V3,
                          context_delivery=MOUNTED_ID, task_sections=adapter)
    runner.run_pair(source_case(tmp_path / "run", UFL_V3), tmp_path / "runs")
    assert client.calls[0][3].startswith("# Part 2: Task\n\n## Problem\nSynthetic ufl task") and runner.spec_record() == {"task_sections": adapter.id}


def test_audit_extensions_and_reviewed_sources_extend_the_judge_prompt():
    for suite in ("bd", "cg"):
        for contract in (DECOMPOSITION_POLICY, DECOMPOSITION_POLICY_V3):
            assert algorithm_review_prompt(suite, contract) == algorithm_review_prompt(suite, contract, sources=("solver.py",), extensions=())
    base = algorithm_review_prompt("bd", DECOMPOSITION_POLICY_V3)
    assert base.startswith("Inspect the supplied solver.py and determine whether it implements Benders decomposition and substantively uses that method")
    assert base.endswith("Do not execute or modify solver.py. Do not claim to have observed runtime behavior when reviewing source only. Write alg_verifier.md with exactly a first line VERDICT: PASS or VERDICT: FAIL, followed by NOTES: and concise code evidence supporting the verdict.")
    certificate = AuditExtension("certificate-paths-v0", ("Also trace every code path that publishes a bound certificate and confirm that its duals come from the Benders subproblems.",))
    extended = algorithm_review_prompt("bd", DECOMPOSITION_POLICY_V3, sources=("solver.py", "cpp/core.cpp"), extensions=(certificate,))
    paragraphs = extended.split("\n\n")
    assert paragraphs[0].startswith("Inspect the supplied source files (solver.py, cpp/core.cpp; the entry point is solver.py) and determine whether they implement Benders decomposition and substantively use")
    assert paragraphs[-2] == certificate.paragraphs[0] and paragraphs[-1].startswith("Do not execute or modify the source files.")
    assert base.split("\n\n")[1:-1] == paragraphs[1:-2]  # the contract and the review rules are unchanged
    assert judge_policy() == ALGORITHM_JUDGE and judge_policy((certificate,)) == ALGORITHM_JUDGE + "+certificate-paths-v0"
    runner = PairedRunner(client=None, judge_client=None, solver_executor=None, audit_extensions=(certificate,))
    assert runner.spec_record() == {"algorithm_judge": judge_policy((certificate,))}
    for sources in ((), ("cpp/core.cpp", "solver.py"), ("solver.py", "solver.py")):
        with pytest.raises(ValueError):
            algorithm_review_prompt("bd", sources=sources)


# ---------------------------------------------------------------------------------------------------------------------
# Family tracks.

def scaleup_sources():
    """src.experiments.scaleup_calibration.SOURCES without importing it (it imports gurobipy)."""
    tree = ast.parse((ROOT / "src/experiments/scaleup_calibration.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SOURCES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("SOURCES not found")


@pytest.mark.parametrize("name", list(FAMILIES))
def test_every_family_has_a_wired_small_scale_track_and_a_large_scale_slot(name):
    entry = FAMILIES[name]
    small, large = entry.tracks[SMALL_SCALE], entry.tracks[LARGE_SCALE]
    assert set(entry.tracks) == {SMALL_SCALE, LARGE_SCALE}
    assert small.wired and (small.family, small.sizes, small.seeds, small.dataset, small.config, small.instance_count) == (name, SIZES, SEEDS, entry.dataset, entry.config, 9)
    assert small.location == f"src/{entry.suite}/{name}" and all((ROOT / small.location / f"{module}.py").is_file() for module in small.modules)
    assert small.instance_name("large", 2) == f"{name}_large_seed002"
    if large is not None:
        assert not large.wired and large.family == name and large.instance_count == 0 and large.dataset is None
        assert all((ROOT / large.location / f"{module}.py").is_file() for module in large.modules)
        assert {module for module in large.modules if module.endswith("_xl")} == {p.stem for p in (ROOT / large.location).glob("*_xl.py")}
        listed = scaleup_sources().get(large.tool_key, [])
        assert {Path(path).stem for path in listed if Path(path).parent.as_posix() == large.location} <= set(large.modules)


def test_large_scale_lbbd_is_the_planning_and_scheduling_family():
    assert track_family("lbbd") == "lbbd_ps" and FAMILIES["lbbd"].tracks[LARGE_SCALE] is None
    track = FAMILIES["lbbd_ps"].tracks[LARGE_SCALE]
    assert (track.location, track.tool_key, track.family) == ("src/bd/lbbd", "lbbd", "lbbd_ps") and "b_mip_xl" in track.modules
    assert "planning and scheduling" in (ROOT / "src/bd/lbbd/b_mip_xl.py").read_text().split("\n", 1)[0]
    assert {name for name, entry in FAMILIES.items() if entry.tracks[LARGE_SCALE] is None} == {"lrp", "lbbd", "bp", "csp", "shift", "gcol", "nrp"}
    assert set(scaleup_sources()) == {FAMILIES[name].tracks[LARGE_SCALE].tool_key for name in map(track_family, scaleup_sources())}
    with pytest.raises(KeyError):
        cohort_tracks(["lbbd_ps"], LARGE_SCALE)  # registry support is not campaign eligibility
    with pytest.raises(KeyError):
        track_family("lbbd_ps")


def test_campaign_and_check_config_take_their_counts_from_the_tracks(tmp_path, monkeypatch):
    assert [planned_cases(load_plan(ROOT / "configs" / name)) for name in ("paper.json", "harder_small_all.json", "small_scale.json")] == [81, 54, 180]
    one = {name: Track(SMALL_SCALE, name, f"src/bd/{name}", (), name, sizes=("small",), seeds=(0,), wired=True) for name in FAMILIES}
    monkeypatch.setattr(campaign, "cohort_tracks", lambda families: {name: one[name] for name in families})
    monkeypatch.setattr(campaign, "require_runtime_environment", lambda *args, **kwargs: {})
    monkeypatch.setattr(campaign, "load_execution_environment", lambda path: {})
    with pytest.raises(ProtocolError, match="Exactly 6 existing fixed instances"):
        campaign.initialize(tmp_path / "root", ROOT / "data_harder", ROOT / "configs/harder_small_all.json", None)
    plan = json.loads((ROOT / "configs/paper.json").read_text())
    save(tmp_path / "policy.json", plan | {"execution_policy": "synthetic-supervised-publications-v0"})
    with pytest.raises(ProtocolError, match="Execution settings"):
        load_plan(tmp_path / "policy.json")
    with pytest.raises(ProtocolError, match="native execution policy"):
        campaign.ensure_execution_allowed(tmp_path, {"prompt_policies": plan["prompt_policies"], "reflection_policy": plan["reflection_policy"], "prompt_protocol": plan["prompt_protocol"],
                                                     "experiment_plan": plan | {"execution_policy": "synthetic-supervised-publications-v0"}})


# ---------------------------------------------------------------------------------------------------------------------
# Combinations: a whole multi-file lifecycle with mounted sources, exact evidence inventories, refused mixtures.

class QueuedMultiFileRuntime(multifile.MultiFileDockerRuntime):
    """Each generation call delivers the next {path: bytes} bundle (validated by the host as in a real call)."""

    def __init__(self, root, worker, deliveries):
        super().__init__(seccomp_profile=root / "profile.json")
        self.seccomp_profile.write_text("{}")
        self.worker, self.deliveries, self.calls = worker, iter(deliveries), []

    def ensure_volume(self, *args, **kwargs):
        pass

    def volume_users(self, volume):
        return []

    def run(self, image, command, **kwargs):
        kwargs["state_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["state_path"].write_text("{}")
        if command == ["preflight"]:
            value = {"ok": True, "versions": {"codex": CLI_VERSIONS["codex-cli"], "claude": CLI_VERSIONS["claude-code-cli"], "agy": CLI_VERSIONS["agy-cli"]},
                     "worker_files_sha256": docker_client.worker_file_hashes(), "extension_files_sha256": multifile.extension_file_hashes()}
            return ContainerResult(0, "", "", 0, artifacts={"worker.json": json.dumps(value)})
        self.calls.append(kwargs)
        files = next(self.deliveries)
        manifest = self.worker.manifest_for(files, [], "cpp_allowed")
        bundle = json.dumps({"version": self.worker.RECORD_VERSION, "mode": MULTI_FILE, "track": "cpp_allowed", "manifest": manifest, "error": None, "error_rejected": [],
                             "files": {path: base64.b64encode(data).decode() for path, data in files.items()}})
        worker = {"cli_version": CLI_VERSIONS["codex-cli"], "requested_model": "fixture", "target": "solver.py", "returncode": 0, "terminal_state": "completed", "error": None,
                  "usage": {}, "usage_available": True, "warnings": [], "observed_models": ["fixture"], "agent_seconds": 1.0}
        return ContainerResult(0, "", "", 1, artifacts={"solver.py": files["solver.py"].decode(), "worker.json": json.dumps(worker), "delivery.json": json.dumps({"rejected_target": None, "misplaced": {}}),
                                                        multifile.BUNDLE_NAME: bundle}, metadata={"cleanup_verified": True})


def test_a_multi_file_delivery_runs_the_whole_paired_lifecycle_with_mounted_sources(tmp_path, worker):
    policy = SupervisedPolicy(Reference(T_f=60.0, z_best=Fraction(4), integral=True, eps_p=Fraction("0.01"), headline="near_best"))
    missed, solved = scenario(termination="completed", incumbents=[[5.0, 6]]), scenario(termination="completed", incumbents=[[5.0, 4]])
    deliveries = [{"solver.py": missed.encode(), "cpp/core_1.cpp": b"int one(){return 1;}\n"}, {"solver.py": solved.encode(), "cpp/core_2.cpp": b"int two(){return 2;}\n"},
                  {"solver.py": solved.encode(), "cpp/core_3.cpp": b"int three(){return 3;}\n", "cpp/core_3.hpp": b"int three();\n"}]
    generation_runtime = QueuedMultiFileRuntime(tmp_path, worker, deliveries)
    delivery = DeliverySpec(MULTI_FILE, "cpp_allowed")
    generation = multifile.MultiFileDockerCliClient("fixture", backend="codex-cli", image_id="sha256:fixture", runtime=generation_runtime, evidence_root=tmp_path / "evidence", delivery=delivery,
                                                    credential_provider=lambda _: {}, context_delivery=MOUNTED_ID, context_layout=SOURCES_LAYOUT)
    reflection_runtime = MountedFakeRuntime(tmp_path, contents="Port the master to C++ and keep the cuts.\n")
    reflection = DockerCliClient("m", backend="codex-cli", image_id="sha256:x", runtime=reflection_runtime, evidence_root=tmp_path / "evidence", credential_provider=lambda _: {},
                                 role="reflection", context_delivery=MOUNTED_ID, context_layout=SOURCES_LAYOUT)
    judge = Client()
    summaries = PairedRunner(client=generation, judge_client=judge, reflection_client=reflection, model="fixture", suite="bd", problem=["ufl"], solver_executor=None, max_attempts=3,
                             decomposition_contract=DECOMPOSITION_POLICY_V3, context=MOUNTED_WITH_SOURCES, delivery=delivery, execution_policy=policy).run_pair(source_case(tmp_path, UFL_V3), tmp_path / "runs")
    for mode in ("plain", "reflection"):
        assert [a["status"] for a in summaries[mode]["attempts"]] == ["target_missed", "ok", "skipped_previous_solved"]
    plain, reflected = generation_runtime.calls[1], generation_runtime.calls[2]
    assert HISTORY_DIR not in plain["context"]  # plain regeneration: no history
    assert sorted(reflected["context"][HISTORY_DIR]) == ["attempt_01/cpp/core_1.cpp", "attempt_01/debug.md", "attempt_01/solver.py"]
    assert reflected["context"][HISTORY_DIR]["attempt_01/cpp/core_1.cpp"] == deliveries[0]["cpp/core_1.cpp"]
    review = reflection_runtime.calls[0]
    assert sorted(review["context"][HISTORY_DIR]) == ["attempt_01/alg_verifier.md", "attempt_01/cpp/core_1.cpp", "attempt_01/exec.log", "attempt_01/solver.py"]
    assert f"  - {HISTORY_DIR}/attempt_01/cpp/core_1.cpp (21 bytes): the C++ source cpp/core_1.cpp of attempt 1." in review["payload"]["prompt"]
    judged = [(context, prompt) for step, _, context, prompt in judge.calls if step == "llm_verifier"]
    assert [[p.relative_to(p.parents[1] if p.parent.name == "cpp" else p.parent).as_posix() for p in context] for context, _ in judged] == \
        [["solver.py", "cpp/core_1.cpp"], ["solver.py", "cpp/core_2.cpp"], ["solver.py", "cpp/core_3.cpp", "cpp/core_3.hpp"]]
    assert judged[0][1].startswith("Inspect the supplied source files (solver.py, cpp/core_1.cpp; the entry point is solver.py)")
    first = tmp_path / "runs/plain/bd/ufl/ufl_small_seed000/attempts/attempt_01"
    shared = tmp_path / "runs/reflection/bd/ufl/ufl_small_seed000/attempts/attempt_01"
    assert (shared / "cpp/core_1.cpp").read_bytes() == deliveries[0]["cpp/core_1.cpp"] and {"cpp/core_1.cpp", "publication/incumbent_000.json"} <= set(read(first / "score_receipt.json")["artifact_sha256"])
    assert read(tmp_path / "runs/plain/bd/ufl/ufl_small_seed000/identity.json")["specs"] == {"context": MOUNTED_WITH_SOURCES.id, "delivery": delivery.id, "execution_policy": policy.id}


def test_the_committed_nested_inventory_is_exact(tmp_path):
    reference = Reference(T_f=60.0, z_best=Fraction(4), integral=True, eps_p=Fraction("0.01"), headline="near_best")
    source = source_case(tmp_path)
    client = Client([scenario(termination="timeout", incumbents=[[12.0, 4]])])
    run = lambda: PairedRunner(client=client, judge_client=client, model="fixture", suite="bd", problem=["ufl"], solver_executor=None, max_attempts=2,
                               execution_policy=SupervisedPolicy(reference)).run_pair(source, tmp_path / "runs")
    run()
    publication = tmp_path / "runs/plain/bd/ufl/ufl_small_seed000/attempts/attempt_01/publication"
    (publication / "late.json").write_text("{}")  # never part of the scored attempt
    with pytest.raises(ProtocolError, match="nested artifacts changed"):
        run()
    (publication / "late.json").unlink()
    (publication / "link.json").symlink_to(publication / "incumbent_000.json")
    with pytest.raises(ProtocolError, match="Unsafe entry"):
        run()
    (publication / "link.json").unlink()
    run()
    projection = tmp_path / "runs/reflection/bd/ufl/ufl_small_seed000/attempts/attempt_01/publication"
    (projection / "late.json").write_text("{}")  # an extra file in the reflection projection
    with pytest.raises(ProtocolError, match="projection differs"):
        run()
    (projection / "late.json").unlink()
    copied = projection / "incumbent_000.json"
    (projection / "kept.json").write_bytes(copied.read_bytes())
    copied.unlink()
    copied.symlink_to(projection / "kept.json")  # same bytes, but a link
    with pytest.raises(ProtocolError, match="Unsafe entry"):
        run()
    copied.unlink()
    (projection / "kept.json").rename(copied)
    run()
    with pytest.raises(ValueError, match="plain names"):
        PairedRunner(client=None, judge_client=None, solver_executor=None, execution_policy=type("Deep", (SupervisedPolicy,), {"nested": ("../escape",)})(reference))


def test_a_pasted_context_refuses_prompt_parts_it_would_ignore():
    adapter = DirectSections("synthetic-family-sections-v0", synthetic_sections)
    policy = SupervisedPolicy(Reference(T_f=900.0, z_best=Fraction(4), eps_p=Fraction("0.01"), headline="near_best"))
    rule = AuditExtension("bound-only-rule-v0", ("A bound-only attempt must produce its certificates from the Benders subproblems.",), author=True)
    for options in ({"task_sections": adapter}, {"runtime_spec": SupervisorRuntime(900, 1), "solver_timeout": 900, "execution_policy": policy}, {"audit_extensions": (rule,)}):
        with pytest.raises(ValueError, match="pasted context"):
            PairedRunner(client=None, judge_client=None, solver_executor=None, **options)
    with pytest.raises(ValueError, match="mounted-files prompt only"):
        DockerCliClient("m", backend="codex-cli", image_id="sha256:x", runtime=None, evidence_root=None, runtime_spec=SupervisorRuntime(900, 1))


def test_an_audit_target_replaces_the_review_target_and_shared_rules_reach_the_author(tmp_path):
    target = AuditTarget("published-incumbents-and-bounds-v0", "the published incumbents and certified bounds", "the code that publishes incumbents and certificates")
    text = algorithm_review_prompt("bd", DECOMPOSITION_POLICY_V3, target=target).split("\n\n")
    assert "substantively uses that method to produce the published incumbents and certified bounds." in text[0] and "reported solution" not in text[0]
    trace = next(paragraph for paragraph in text if paragraph.startswith("Trace the entry point"))
    assert "and the code that publishes incumbents and certificates." in trace and "output.json" not in trace
    assert algorithm_review_prompt("bd", DECOMPOSITION_POLICY_V3, target=PAPER_AUDIT_TARGET) == algorithm_review_prompt("bd", DECOMPOSITION_POLICY_V3)
    assert judge_policy((), target) == ALGORITHM_JUDGE + "+published-incumbents-and-bounds-v0" and judge_policy((), PAPER_AUDIT_TARGET) == ALGORITHM_JUDGE
    rule = AuditExtension("bound-only-rule-v0", ("Publish every certified bound through the publication API.",), author=True)
    case = source_case(tmp_path, UFL_V3)
    stated = PairedRunner(client=None, judge_client=None, solver_executor=None, problem=["ufl"], context_delivery=MOUNTED_ID, decomposition_contract=DECOMPOSITION_POLICY_V3, audit_target=target, audit_extensions=(rule,),
                          task_sections=DirectSections("synthetic-family-sections-v0", synthetic_sections))
    assert stated.spec_record()["algorithm_judge"] == ALGORITHM_JUDGE + "+published-incumbents-and-bounds-v0+bound-only-rule-v0"
    assert "Publish every certified bound" in stated.generation_prompt(case)
    unstated = PairedRunner(client=None, judge_client=None, solver_executor=None, problem=["ufl"], context_delivery=MOUNTED_ID, decomposition_contract=DECOMPOSITION_POLICY_V3, audit_extensions=(rule,))
    with pytest.raises(ProtocolError, match="lack the audit's method rules"):
        unstated.generation_prompt(case)


def test_specs_validate_their_inputs():
    with pytest.raises(ValueError):
        MountLayout("escape", (ContextRole("up", "/work/..", {"x": "y"}),))
    with pytest.raises(ValueError, match="same Part 1 section"):
        MountLayout("mixed", (ContextRole("a", "/work/notes", {"a.md": "a"}), ContextRole("b", "/work/notes", {"b.md": "b"}, per_attempt=True, section="Files from previous attempts")))
    with pytest.raises(ValueError):
        check_context({"/work/..": {"x": b"1"}}, ("/work/..",))
    for files, per_attempt in (({"../x": "y"}, True), ({"/abs": "y"}, True), ({"a/b": "y"}, False), ({"": "y"}, False)):
        with pytest.raises(ValueError):
            ContextRole("bad", "/work/docs", files, per_attempt=per_attempt)
    files = {"api.md": "the API"}
    role = ContextRole("docs", "/work/docs", files)
    files["api.md"], files["secret.txt"] = "changed", "a new admission"
    assert dict(role.files) == {"api.md": "the API"}  # the role keeps its frozen copy
    sections = synthetic_sections({"n": 1}, suite="bd", problem="x", contract=DECOMPOSITION_POLICY_V3)
    for broken in (dict(sections, Data="data.json holds n."), dict(sections, Data=None), dict(sections, Rules=[1])):
        with pytest.raises(ProtocolError, match="non-empty list"):
            check_sections(broken, suite="bd", problem="x", contract=DECOMPOSITION_POLICY_V3)


class LeakyContext(ContextSpec):
    def generation(self, instance, attempts_root, idx, mode):
        files, record = super().generation(instance, attempts_root, idx, "reflection")  # ignores the mode
        return files, record


def test_the_runner_keeps_plain_regeneration_free_of_history_whatever_the_spec(tmp_path):
    good = "import json\nfrom pathlib import Path\nPath('output.json').write_text(json.dumps({'status':'feasible','method':'benders_decomposition','y':[0,1],'x':[[0],[1]],'objective_value':6}))\n"
    client = Client([good, good])
    runner = PairedRunner(client=client, judge_client=client, model="fixture", suite="bd", problem=["ufl"], solver_executor=host_solver, max_attempts=3,
                          context=LeakyContext("leaky-history-v0"))
    with pytest.raises(ProtocolError, match="plain regeneration receives no file of an earlier attempt"):
        runner.run_pair(source_case(tmp_path), tmp_path / "runs")


def test_one_reviewed_attempt_may_span_several_mount_directories(tmp_path):
    certificates = ContextRole("certificates", "/work/certificates", {"publication/*.json": "the published certificate {path} of attempt {n}"}, per_attempt=True, section="Files from previous attempts")
    layout = MountLayout("history-and-certificates-test", (*MOUNT_LAYOUT.roles, certificates))
    case = tmp_path / "case"
    for index in (1, 2):
        (case / f"attempts/attempt_{index:02d}/publication").mkdir(parents=True)
        (case / f"attempts/attempt_{index:02d}/solver.py").write_text("print(1)\n")
        (case / f"attempts/attempt_{index:02d}/publication/certificate.json").write_text("{}")
    (case / "data.json").write_text("{}")
    one = mounted_files([case / "data.json", case / "attempts/attempt_01/solver.py", case / "attempts/attempt_01/publication/certificate.json"], layout)
    text, _ = workspace_part("debug.md", one, "codex-cli", layout=layout)
    assert "- Attempt 1 (the attempt under review), mounted read-only in /work/history/attempt_01/:" in text
    assert "- Attempt 1 (the attempt under review), mounted read-only in /work/certificates/attempt_01/:" in text
    assert "  - /work/certificates/attempt_01/publication/certificate.json (2 bytes): the published certificate publication/certificate.json of attempt 1." in text
    two = mounted_files([case / "data.json", case / "attempts/attempt_01/solver.py", case / "attempts/attempt_02/publication/certificate.json"], layout)
    with pytest.raises(ProtocolError, match="exactly one attempt"):
        workspace_part("debug.md", two, "codex-cli", layout=layout)
