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
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping
import uuid

from src.agent.specs import DEFAULT_RUNTIME, SINGLE_FILE_DELIVERY
from src.experiments.environment import require_runtime_environment
from src.experiments.sandbox import SolverLaunchError
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

    def __init__(self, *, executor, suite: str, tol: float = 1e-6, gap_threshold: float = 0.01, cooldown_seconds: float = 0):
        self.executor, self.suite, self.tol, self.gap_threshold, self.cooldown_seconds = executor, suite, tol, gap_threshold, cooldown_seconds

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
        code_hash = digest(attempt_dir / "solver.py")
        if execution:
            if execution["solver_sha256"] != code_hash:
                raise ProtocolError("Executed solver changed")
            if execution.get("output_sha256") and digest(attempt_dir / "output.json") != execution["output_sha256"]:
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
            save(execution_path, execution)
            if not (attempt_dir / "solver_invocation.json").exists():
                save(attempt_dir / "solver_invocation.json", {"backend": "host-python", "timeout": runtime.timeout, "threads": runtime.threads, "solver_sha256": code_hash})
            if elapsed >= 5 and self.cooldown_seconds:
                stage("solver cooldown")
                time.sleep(self.cooldown_seconds)
        exec_ok, stdout, stderr, output = (execution[k] for k in ("exec_ok", "stdout", "stderr", "output"))
        timed_out = TIMEOUT_MARKER in stderr
        metrics = {"execution_seconds": execution["execution_seconds"], "execution_timed_out": timed_out, "cooldown_seconds": self.cooldown_seconds if execution["execution_seconds"] >= 5 else 0}
        return ExecutionResult(exec_ok, stdout, stderr, output, execution["execution_seconds"], "timeout" if timed_out else ("completed" if exec_ok else "failed"), metrics)

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


EXECUTION_POLICIES = {SMALL_SCALE_NATIVE_POLICY: SmallScaleNativePolicy}
