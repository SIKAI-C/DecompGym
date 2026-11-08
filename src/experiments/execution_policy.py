"""How a delivered candidate is executed and evaluated: the execution policy of the paired runner.

The paired lifecycle (src.experiments.paired_runner: generation, execution, method audit, reflection, shared first
attempt, stopping) never interprets an execution itself. It asks the policy:

    references(source)                          -> the frozen references of a case (opaque to the runner)
    execute(case, delivery=, runtime=, stage=)  -> ExecutionResult   (durable: a resumed attempt re-reads its record)
    evaluate(result, references, case=, audit=) -> Evaluation        (calls audit() -> (verdict, notes) for the method
                                                                       judge at the point the policy decides)
    evaluation(result)                          -> Evaluation        (of a durable AttemptResult record)
    reused(successful)                          -> the fields of the skipped_previous_solved rows after a success
    check(delivery=, runtime=)                  -> refuses a delivery or runtime contract the policy cannot run
    runtime_environment()                       -> the controller-side environment check (the local Python/Gurobi one)

and reads every decision from the Evaluation: the stop rule is Evaluation.solved (success of the headline target), the
reflection's qualitative feedback flag Evaluation.objective_target_missed. An Evaluation with infrastructure errors is
never scored (the runner raises InfrastructureError). `record` carries the policy's own scores into the attempt
metrics ("evaluation") so that evaluation() can restore them; `nested` names attempt subdirectories that belong to the
attempt's evidence (pinned in the score receipt and copied with the shared first attempt; durable evidence must live
there, not in new top-level files), and `reference_files` the case files frozen in the case identity as references.

SmallScaleNativePolicy (SMALL_SCALE_NATIVE_POLICY) is the paired protocol of the paper and the small-scale track: the
native sandbox (src.experiments.sandbox) runs solver.py once within the runtime's 60 s and one thread, the final
output.json is verified by the family verifier and scored against the gold objective with the 1 % gap rule
(OutputVerifier.assess_status); a timeout, crash or invalid output is never a success. Its records are exactly those of
the earlier fixed implementation. A large-scale policy (supervisor publications, eligible incumbents, certified bounds,
per-target success; src.experiments.xl_scoring) implements the same four methods.

SmallScaleLanesPolicy (SMALL_SCALE_LANES_POLICY) is the same execution and scoring under the small-scale lanes variant
(LANES_EXECUTION, selected by configs/small_scale.json): the campaign runs its model lanes side by side in one
controller (src.experiments.campaign), candidate solvers stay serialized by the native solver lock, and each candidate
and its child processes have a sampled 2 GiB resident-memory budget (sandbox.MEMORY_POLICY_2GIB): a run above it is an
execution failure, exactly as a crash. Its differences are in the records only: the cooldown is applied by one rule
(scored execution time of at least 5 s) and recorded as applied, the solver-lock queue time is recorded separately
(solver_queue_seconds) and left out of the attempt's total_seconds, and the memory outcome is recorded. Its id is
recorded in every case identity and request (PairedRunner.spec_record).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping
import uuid

from src.agent.specs import DEFAULT_RUNTIME, SINGLE_FILE_DELIVERY
from src.experiments.environment import require_runtime_environment
from src.agent.safe_artifacts import MAX_ARTIFACT_BYTES, UnsafeArtifact, read_regular
from src.experiments.sandbox import MEMORY_POLICY_2GIB, OUTPUT_BY_REFERENCE, SolverLaunchError
from src.experiments.verification import OutputVerifier
from src.utils.errors import InfrastructureError, ProtocolError
from src.utils.records import digest, read, save


SMALL_SCALE_NATIVE_POLICY = "native-final-output-gap-to-gold-v1"
GAP_TARGET = "within_gap_of_gold"
TIMEOUT_MARKER = "[framework] solver execution timed out"
# The paired protocol's frozen execution settings (configs/*.json "execution"; src.experiments.plan validates them).
PAIRED_EXECUTION = {"workers": 2, "max_attempts": 10, "solver_timeout": 60, "solver_threads": 1,
                    "solver_cooldown_seconds": 10, "agent_timeout": 900, "active_codex_invocation_limit": 2,
                    "active_agy_invocation_limit": 1, "active_claude_invocation_limit": 1,
                    "shared_first_attempt": True, "stop_after_success": True, "gap_threshold": 0.01,
                    "judge_model": "gpt-6-astra", "judge_effort": "high"}
SMALL_SCALE_LANES_POLICY = "native-final-output-gap-to-gold-lanes-v1"
# The reviewed settings of the lanes variant: the paired protocol with three Codex and two Claude Code calls at a time
# (one auth volume per slot; the judge shares the Codex slots), one AGY call, the 2 GiB per-solver memory budget and the
# aggregate resource guard of the controller tree and its Docker workers. The config adds its LANE_FIELDS: "lanes", a
# list of {"name", "models", "workers"} that partitions the model queue (src.experiments.plan.lane_partition), and
# "workers", their total. The call limits hold per backend whatever the lanes, through one gate per backend.
LANES_EXECUTION = {key: value for key, value in PAIRED_EXECUTION.items() if key != "workers"} | {
    "active_codex_invocation_limit": 3, "active_claude_invocation_limit": 2, "solver_memory_policy": MEMORY_POLICY_2GIB,
    "resource_guard": {"sample_interval": 5, "max_cpu_percent": 500, "max_rss_mib": 5120, "min_free_percent": 10, "sustained_samples": 3}}
LANE_FIELDS = ("lanes", "workers")


@dataclass(frozen=True)
class AttemptCase:
    """What the runner hands a policy about one attempt."""
    directory: Path  # the attempt directory (the delivered solver.py and the execution records)
    instance: Path  # the case directory (its data.json is the runtime input)
    canonical_input: Path  # the frozen scoring input of the case identity
    input_sha256: str
    problem: str


@dataclass(frozen=True)
class ExecutionResult:
    ok: bool  # the run completed and its output is available
    stdout: str
    stderr: str
    output: dict | None
    seconds: float  # scored execution time
    termination: str  # completed | failed | timeout (a policy may add its own, e.g. crashed, preempted)
    metrics: Mapping[str, Any] = field(default_factory=dict)  # merged into the attempt metrics, in this order
    evidence: Mapping[str, Any] = field(default_factory=dict)  # policy-specific (e.g. a supervisor record)
    queue_seconds: float = 0.0  # time this run waited for the solver lock, left out of the attempt's total_seconds


@dataclass(frozen=True)
class Evaluation:
    status: str
    feasible: bool
    verified: bool  # feasible and the method audit passed (the paper's verify_ok)
    objective: float | None
    reference_objective: float | None
    gap: float | None
    termination: str
    success: Mapping[str, bool | None]  # per target, gated by the method audit; None: undefined
    headline: str  # the target that decides the stop rule
    issues: list = field(default_factory=list)
    message: str | None = None
    method_verdict: str | None = None
    method_notes: str | None = None
    objective_target_missed: bool = False  # feasible and verified-objective feedback: the target was not met
    infrastructure: tuple[str, ...] = ()
    record: Mapping[str, Any] = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        return self.success.get(self.headline) is True


class ExecutionPolicy:
    """The interface (see the module docstring); `id` is recorded by the runner when it is not the default policy."""
    id: str
    nested: tuple[str, ...] = ()
    reference_files: tuple[str, ...] = ("meta.json", "gold_monolithic.json")

    def check(self, *, delivery, runtime) -> None:
        """Refuse (ValueError) a delivery or runtime contract this policy cannot execute as stated to the agent."""

    def runtime_environment(self) -> dict:
        """The environment this controller needs for the policy (by default the local Python/Gurobi versions)."""
        return require_runtime_environment()

    def reused(self, successful) -> dict:
        """The facts of a skipped_previous_solved row after `successful` (an AttemptResult): its own flags and the
        policy's evaluation record, so that a bound-only or post-timeout success is never restated as anything else."""
        return {"exec_ok": successful.exec_ok, "verify_ok": successful.verify_ok, "objective": successful.objective, "gap_to_gold": successful.gap_to_gold,
                "feasibility_ok": successful.feasibility_ok, "metrics": {"evaluation": successful.metrics["evaluation"]} if "evaluation" in successful.metrics else {}}

    def references(self, source: Path):
        raise NotImplementedError

    def reference_objective(self, references) -> float | None:
        raise NotImplementedError

    def execute(self, case: AttemptCase, *, delivery, runtime, stage: Callable[[str], None]) -> ExecutionResult:
        raise NotImplementedError

    def evaluate(self, result: ExecutionResult, references, *, case: AttemptCase, audit: Callable[[], tuple[str, str]]) -> Evaluation:
        raise NotImplementedError

    def evaluation(self, result) -> Evaluation:
        raise NotImplementedError


class SmallScaleNativePolicy(ExecutionPolicy):
    """The paper's execution and scoring (module docstring). `executor(attempt_dir, timeout, threads)` returns
    (exec_ok, stdout, stderr, output): src.experiments.sandbox.NativeSandboxExecutor in a campaign."""
    id = SMALL_SCALE_NATIVE_POLICY
    settings = PAIRED_EXECUTION

    def __init__(self, *, executor, suite: str, tol: float = 1e-6, gap_threshold: float = 0.01, cooldown_seconds: float = 0,
                 evidence_schema: str | None = None, output_limit: int = MAX_ARTIFACT_BYTES):
        self.executor, self.suite, self.tol, self.gap_threshold, self.cooldown_seconds = executor, suite, tol, gap_threshold, cooldown_seconds
        # The executor's sandbox.EVIDENCE_SCHEMAS value: under OUTPUT_BY_REFERENCE execution.json names output.json
        # (stored_execution) instead of embedding the parsed output; None keeps the paper's record.
        if evidence_schema not in (None, OUTPUT_BY_REFERENCE):
            raise ValueError(f"Unknown solver evidence schema: {evidence_schema!r}")
        self.evidence_schema, self.output_limit = evidence_schema, output_limit

    def stored_execution(self, attempt_dir, execution):
        """execution.json as committed: under OUTPUT_BY_REFERENCE a parsed output that equals the parse of output.json
        (always, for the native sandbox) is replaced by a reference to that file; any other output stays embedded."""
        if self.evidence_schema is None:
            return execution
        referenced = False
        if execution["output"] is not None and execution["output_sha256"] is not None:
            try:
                referenced = self.parse_output(self.output_bytes(attempt_dir, execution["output_sha256"])) == execution["output"]
            except ProtocolError:
                referenced = False
        return execution | {"output": None if referenced else execution["output"], "evidence_schema": self.evidence_schema, "output_in_output_json": referenced}

    def output_bytes(self, attempt_dir, expected_sha256) -> bytes:
        """The committed output.json bytes (bounded single-link read) after the sha256 check."""
        try:
            contents = read_regular(attempt_dir, "output.json", maximum=self.output_limit)
        except (OSError, UnsafeArtifact) as error:
            raise ProtocolError("Saved solver output changed") from error
        if contents is None or hashlib.sha256(contents).hexdigest() != expected_sha256:
            raise ProtocolError("Saved solver output changed")
        return contents

    @staticmethod
    def parse_output(contents: bytes):
        """The same strict UTF-8 decoding and json.loads as a fresh native run."""
        try:
            return json.loads(contents.decode("utf-8"))
        except (UnicodeError, ValueError, RecursionError) as error:
            raise ProtocolError("Saved solver output is not the committed parsed output") from error

    def check(self, *, delivery, runtime):
        if delivery != SINGLE_FILE_DELIVERY or runtime.contract != DEFAULT_RUNTIME.contract:
            raise ValueError("The native policy executes one solver.py under the separate-edit runtime (../../data.json in, output.json out)")

    def reused(self, successful):
        """The paper's skipped rows: a success is a completed, verified, feasible final output (as in every summary so far)."""
        return {"exec_ok": True, "verify_ok": True, "objective": successful.objective, "gap_to_gold": successful.gap_to_gold, "feasibility_ok": True, "metrics": {}}

    def references(self, source: Path) -> float | None:
        return OutputVerifier.load_gold_objective(source, self.suite)

    def reference_objective(self, references) -> float | None:
        return references

    def execute(self, case, *, delivery, runtime, stage):
        attempt_dir = case.directory
        execution_path = attempt_dir / "execution.json"
        execution = read(execution_path)
        # Where this execution comes from: "resumed" (a committed execution.json), "receipt" (the executor re-reads its
        # committed native receipt: the controller stopped between the two records) or "ran" (the solver runs now).
        source = "resumed" if execution else "receipt" if (attempt_dir / "native_solver_receipt.json").exists() else "ran"
        code_hash = digest(attempt_dir / "solver.py")
        if execution:
            if execution["solver_sha256"] != code_hash:
                raise ProtocolError("Executed solver changed")
            if execution.get("evidence_schema") != self.evidence_schema:
                raise ProtocolError("A committed execution record has another evidence schema")
            if self.evidence_schema is not None:
                if not isinstance(execution.get("output_in_output_json"), bool) or (execution["output_in_output_json"] and (execution["output"] is not None or not execution.get("output_sha256"))):
                    raise ProtocolError("A committed execution record has an unknown layout")
                if execution.get("output_sha256"):
                    contents = self.output_bytes(attempt_dir, execution["output_sha256"])  # the integrity check of every retained output
                    if execution["output_in_output_json"]:  # parsed only when it is the committed result (a failed run's partial file may be any bytes)
                        output = self.parse_output(contents)
                        if not isinstance(output, dict) or execution.get("exec_ok") is not True:
                            raise ProtocolError("A committed execution record has an unknown layout")
                        execution = execution | {"output": output}
                elif (attempt_dir / "output.json").exists():
                    raise ProtocolError("Saved solver output changed")
            elif execution.get("output_sha256") and digest(attempt_dir / "output.json") != execution["output_sha256"]:
                raise ProtocolError("Saved solver output changed")
        else:
            stage("solver")
            execution_started = time.perf_counter()
            try:
                exec_ok, stdout, stderr, output = self.executor(attempt_dir, runtime.timeout, runtime.threads)
            except SolverLaunchError as error:
                save(attempt_dir / f"solver_infrastructure_{uuid.uuid4().hex}.json", error.evidence | {"solver_sha256": code_hash})
                raise InfrastructureError("Solver launcher unavailable; see structured execution diagnostics") from error
            elapsed = time.perf_counter() - execution_started
            native_receipt = read(attempt_dir / "native_solver_receipt.json")
            scored_seconds = native_receipt["result"]["execution_seconds"] if native_receipt else elapsed
            if digest(case.canonical_input) != case.input_sha256:
                raise ProtocolError("Canonical scoring input changed during execution")
            runtime_input = case.instance / "data.json"
            if not runtime_input.is_file() or runtime_input.is_symlink() or digest(runtime_input) != case.input_sha256:
                save(attempt_dir / "input_integrity_failure.json", {"expected_sha256": case.input_sha256, "observed_sha256": digest(runtime_input) if runtime_input.is_file() and not runtime_input.is_symlink() else None})
                if runtime_input.exists() or runtime_input.is_symlink():
                    runtime_input.rename(attempt_dir / f"altered_input_{uuid.uuid4().hex}")
                shutil.copyfile(case.canonical_input, runtime_input)
                exec_ok, output = False, None
                stderr += "\n[framework] execution input changed; the candidate failed and the canonical input was restored"
            execution = {"exec_ok": exec_ok, "stdout": stdout, "stderr": stderr, "output": output, "execution_seconds": scored_seconds, "solver_sha256": code_hash, "output_sha256": digest(attempt_dir / "output.json") if (attempt_dir / "output.json").exists() else None}
            cooldown = self.cooldown_seconds if self.cooldown_seconds and self.cooldown_due(elapsed, execution, source == "ran") else 0
            save(execution_path, self.stored_execution(attempt_dir, execution) | self.cooldown_record(source, cooldown))
            if not (attempt_dir / "solver_invocation.json").exists():
                save(attempt_dir / "solver_invocation.json", {"backend": "host-python", "timeout": runtime.timeout, "threads": runtime.threads, "solver_sha256": code_hash})
            if cooldown:
                stage("solver cooldown")
                time.sleep(cooldown)
                self.cooldown_done(execution_path, cooldown)
        exec_ok, stdout, stderr, output = (execution[k] for k in ("exec_ok", "stdout", "stderr", "output"))
        timed_out = TIMEOUT_MARKER in stderr
        metrics = {"execution_seconds": execution["execution_seconds"], "execution_timed_out": timed_out, "cooldown_seconds": self.cooldown_seconds if execution["execution_seconds"] >= 5 else 0}
        return self.result(attempt_dir, execution, metrics, "timeout" if timed_out else ("completed" if exec_ok else "failed"), source)

    @staticmethod
    def cooldown_due(elapsed, execution, ran) -> bool:
        """The paper's rule: the wall time around the executor (with any solver-lock wait) was at least 5 s. The recorded
        cooldown_seconds instead follows the scored execution time."""
        return elapsed >= 5

    def cooldown_record(self, source, cooldown) -> dict:
        """Fields execution.json adds about the execution's source and cooldown (none for the paper's record)."""
        return {}

    def cooldown_done(self, execution_path, cooldown) -> None:
        """Called once the cooldown after an execution has been completed (the paper records nothing)."""

    def result(self, attempt_dir, execution, metrics, termination, source) -> ExecutionResult:
        exec_ok, stdout, stderr, output = (execution[k] for k in ("exec_ok", "stdout", "stderr", "output"))
        return ExecutionResult(exec_ok, stdout, stderr, output, execution["execution_seconds"], termination, metrics)

    def evaluate(self, result, references, *, case, audit):
        """Verify the final output (only after a completed run), then the method audit, then the gap rule."""
        objective, issues, message = OutputVerifier.extract_objective(result.output)
        verification = {"ok": False, "issues": []}
        if result.ok:
            verification = OutputVerifier.verify_output(self.suite, case.problem, case.canonical_input, case.directory / "output.json", self.tol)
            issues.extend(verification.get("issues", []))
            if verification.get("objective_value") is not None:
                objective = float(verification["objective_value"])
        else:
            issues.append(result.stderr or "solver execution failed")
        verdict, notes = audit()
        if verdict != "pass":
            issues.append("algorithm_verifier_failed: " + verdict)
        verify_ok = bool(verification["ok"]) and verdict == "pass"
        status, gap = OutputVerifier.assess_status(exec_ok=result.ok, output=result.output, verify_ok=verify_ok, objective=objective, gold_obj=references, gap_threshold=self.gap_threshold,
                                                   execution_timed_out=result.termination == "timeout")
        return self._evaluation(status=status, exec_ok=result.ok, verify_ok=verify_ok, feasible=bool(verification["ok"]), objective=objective, gold=references, gap=gap,
                                timed_out=result.termination == "timeout", issues=issues, message=message, verdict=verdict, notes=notes)

    def evaluation(self, result):
        """The Evaluation of a durable attempt record (AttemptResult): the same decisions as when it was scored."""
        return self._evaluation(status=result.status, exec_ok=result.exec_ok, verify_ok=result.verify_ok, feasible=result.feasibility_ok, objective=result.objective,
                                gold=result.gold_objective, gap=result.gap_to_gold, timed_out=bool(result.metrics.get("execution_timed_out")), issues=result.issues,
                                message=None, verdict=result.llm_verdict, notes=result.llm_notes)

    def _evaluation(self, *, status, exec_ok, verify_ok, feasible, objective, gold, gap, timed_out, issues, message, verdict, notes):
        solved = status == "ok" and exec_ok and verify_ok and gap is not None and gap <= self.gap_threshold
        # Qualitative objective feedback: a completed, feasible final output whose gap to the gold misses the threshold
        # (independent of the method audit; never the numeric gap itself).
        missed = bool(exec_ok and feasible and not timed_out and all(isinstance(value, (int, float)) and math.isfinite(value) for value in (objective, gold, gap))
                      and gap > self.gap_threshold)
        return Evaluation(status=status, feasible=feasible, verified=verify_ok, objective=objective, reference_objective=gold, gap=gap,
                          termination="timeout" if timed_out else ("completed" if exec_ok else "failed"), success={GAP_TARGET: solved}, headline=GAP_TARGET,
                          issues=issues, message=message, method_verdict=verdict, method_notes=notes, objective_target_missed=missed)


class SmallScaleLanesPolicy(SmallScaleNativePolicy):
    """SmallScaleNativePolicy under the lanes variant (module docstring): the same execution, evaluation and stop rule;
    the executor (NativeSandboxExecutor with memory_policy, ownership and record_queue_seconds) adds the memory
    budget, and this policy's records name it and leave the solver-lock queue out of the attempt time."""
    id = SMALL_SCALE_LANES_POLICY
    settings = LANES_EXECUTION

    def check(self, *, delivery, runtime):
        super().check(delivery=delivery, runtime=runtime)
        if runtime.memory_mib != getattr(self.executor, "memory_limit_mib", None) or runtime.memory_mib is None:
            raise ValueError("The lanes policy states the executor's memory budget in the runtime contract")

    @staticmethod
    def cooldown_due(elapsed, execution, ran) -> bool:
        """One rule for applying and recording the cooldown: the solver ran in this call and its scored execution time
        (never the solver-lock wait) was at least 5 s."""
        return ran and execution["execution_seconds"] >= 5

    def cooldown_record(self, source, cooldown):
        """The durable evidence a resumed attempt reads instead of reconstructing it: where the execution came from ("ran"
        or "receipt", a re-read native receipt), the cooldown decided for it and the cooldown applied (set once the
        cooldown has been completed; 0 while it has not, also when the controller stopped during it)."""
        return {"execution_source": source, "cooldown_decided_seconds": cooldown, "cooldown_applied_seconds": 0}

    def cooldown_done(self, execution_path, cooldown):
        save(execution_path, read(execution_path) | {"cooldown_applied_seconds": cooldown})

    def result(self, attempt_dir, execution, metrics, termination, source):
        """The receipt's structured memory verdict decides the memory outcome (the stderr marker is feedback only: a
        candidate may print it). Its queue time is historical evidence; only a solver that ran in this call adds queue
        time to this attempt. The cooldown recorded is the one execution.json records as completed, in every call (a
        re-read native receipt and an interrupted cooldown record none)."""
        receipt = (read(attempt_dir / "native_solver_receipt.json") or {}).get("result", {})
        queue, exceeded = receipt.get("solver_queue_seconds"), receipt.get("memory_limit_exceeded") is True
        metrics = dict(metrics, solver_queue_seconds=queue, solver_memory_limit_mib=receipt.get("memory_limit_mib"), solver_memory_peak_mib=receipt.get("memory_peak_mib"),
                       solver_memory_limit_exceeded=exceeded, cooldown_seconds=(read(attempt_dir / "execution.json") or {}).get("cooldown_applied_seconds", 0))
        exec_ok, stdout, stderr, output = (execution[k] for k in ("exec_ok", "stdout", "stderr", "output"))
        if exceeded:
            termination = "memory_limit_exceeded"  # an execution failure of the candidate, scored as any crash
        return ExecutionResult(exec_ok, stdout, stderr, output, execution["execution_seconds"], termination, metrics, queue_seconds=(queue or 0.0) if source == "ran" else 0.0)


EXECUTION_POLICIES = {SMALL_SCALE_NATIVE_POLICY: SmallScaleNativePolicy, SMALL_SCALE_LANES_POLICY: SmallScaleLanesPolicy}
