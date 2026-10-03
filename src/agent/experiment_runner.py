"""Codex generation, execution, verification, and optional reflection."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

from src.agent.algorithm_verifier import AlgorithmVerifier
from src.agent.codex_client import CodexClient
from src.agent.exec_logger import ExecLogger
from src.agent.files_manager import ArtifactManager, InstanceManager
from src.agent.output_verifier import OutputVerifier
from src.agent.prompt_builder import PromptBuilder
from src.agent.resource_monitor import CleanupError, stop_owned_tree
from src.utils.json_utils import safe_load_json


SUITE_PROBLEMS = {"bd": ["tsso", "fctp", "km", "lrp", "ufl"], "cg": ["bp", "csp", "gap", "shift"]}
ALL_PROBLEMS = sorted({problem for problems in SUITE_PROBLEMS.values() for problem in problems})


class ModelAccessError(RuntimeError):
    """Access or quota failure must not be counted as model-solving performance."""


@dataclass
class AttemptResult:
    attempt: int
    status: str
    exec_ok: bool
    verify_ok: bool
    objective: float | None
    gold_objective: float | None
    gap_to_gold: float | None
    issues: list[str]
    llm_verdict: str | None = None
    llm_notes: str | None = None
    reused_from_previous: bool = False
    first_success_attempt: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    feasibility_ok: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentRunner:
    def __init__(self, *, model: str | None = None, note: str | None = None, suite: str = "bd",
                 problem: Sequence[str] | None = None, instance: str | None = None, benchmark_root: str = "data",
                 max_attempts: int = 10, solver_timeout: int = 60, agent_timeout: int = 900, solver_threads: int = 1,
                 solver_cooldown_seconds: float = 0,
                 tol: float = 1e-6, gap_threshold: float = 0.01, reflection: bool = False,
                 codex_executable: str = "codex", client: CodexClient | None = None) -> None:
        if suite not in SUITE_PROBLEMS:
            raise ValueError("Choose suite bd or cg.")
        if max_attempts < 1 or solver_threads < 1 or min(solver_timeout, agent_timeout, tol) <= 0 or gap_threshold < 0 or solver_cooldown_seconds < 0:
            raise ValueError("Attempt/time/tolerance budgets must be positive; gap threshold must be nonnegative.")
        self.model, self.note, self.suite = model, note, suite
        self.problems = list(problem) if problem else list(SUITE_PROBLEMS[suite])
        if any(p not in SUITE_PROBLEMS[suite] for p in self.problems):
            raise ValueError(f"Problems must belong to {suite}: {SUITE_PROBLEMS[suite]}")
        self.instance, self.benchmark_root = instance, Path(benchmark_root)
        self.max_attempts, self.solver_timeout = max_attempts, solver_timeout
        self.solver_threads = solver_threads
        self.solver_cooldown_seconds = solver_cooldown_seconds
        self.abort_event = threading.Event()
        self.tol, self.gap_threshold, self.reflection = tol, gap_threshold, reflection
        self.client = client or CodexClient(model, executable=codex_executable, timeout=agent_timeout)

    def run(self) -> list[dict[str, Any]]:
        summaries = []
        for problem in self.problems:
            if self.instance:
                source = Path(self.instance).resolve()
                inferred = InstanceManager.infer_problem_from_path(source, ALL_PROBLEMS)
                if inferred != problem:
                    continue
                if InstanceManager.infer_suite_from_path(source, SUITE_PROBLEMS) != self.suite:
                    raise ValueError("Instance suite does not match --suite.")
                instances = [source]
            else:
                instances = InstanceManager.find_instances(self.benchmark_root / self.suite / problem)
            if not instances:
                raise RuntimeError(f"No instances for {problem}. Run python -m src.01_gen first.")
            root = ArtifactManager.ensure_output_root(self.suite, problem, self.model or "codex-default", note=self.note)
            results = [self.run_instance(source, root, problem) for source in instances]
            summary = {"suite": self.suite, "problem": problem, "backend": "codex-cli", "model": self.model,
                       "output_root": str(root), "num_instances": len(results), "instances": results}
            ArtifactManager.write_text(root, "problem_summary.json", json.dumps(summary, indent=2))
            summaries.append(summary)
        if not summaries:
            raise ValueError("No selected problem matches --instance.")
        print(json.dumps(summaries, indent=2))
        return summaries

    @staticmethod
    def run_solver(attempt_dir: Path, timeout: int, threads: int = 1) -> tuple[bool, str, str, dict[str, Any] | None]:
        if not (attempt_dir / "solver.py").exists() or not (attempt_dir / "solver.py").stat().st_size:
            return False, "", "solver.py missing or empty", None
        (attempt_dir / "output.json").unlink(missing_ok=True)
        # Gurobi reads this before creating its default environment; ThreadLimit bounds parallel solves.
        (attempt_dir / "gurobi.env").write_text(f"ThreadLimit {threads}\nThreads {threads}\n")
        env = os.environ.copy()
        for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
            env[key] = str(threads)
        try:
            process = subprocess.Popen([sys.executable, "solver.py"], cwd=attempt_dir.resolve(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                stop_owned_tree(process, grace=0.1)
                try:
                    stdout, stderr = process.communicate(timeout=2)
                except subprocess.TimeoutExpired as error:
                    raise CleanupError("Solver pipes remained open after owned process cleanup") from error
                return False, stdout[-12000:], stderr[-12000:] + f"\n[framework] solver execution timed out after {timeout} seconds", None
        except OSError as error:
            return False, "", str(error), None
        stdout, stderr = stdout[-12000:], stderr[-12000:]
        if process.returncode:
            return False, stdout, stderr or f"returncode={process.returncode}", None
        output = safe_load_json(attempt_dir / "output.json")
        if not isinstance(output, dict):
            return False, stdout, stderr + "\noutput.json missing, invalid, or not an object", None
        return True, stdout, stderr, output

    def run_attempt(self, idx: int, attempt_dir: Path, problem: str, gold: float | None) -> AttemptResult:
        started = time.perf_counter()
        instance_root = attempt_dir.parent.parent
        prompt = PromptBuilder.ensure_location_prompt((instance_root / "prompt.txt").read_text())
        context = [instance_root / "data.json"]
        if self.reflection and idx > 1:
            context += [Path(path) for path in ArtifactManager.collect_previous_attempt_files(attempt_dir.parent, idx)]
        prompt += "\n\nImplement a standalone solver.py using gurobipy and standard Python libraries. Write output.json using the documented contract. Do not import repository implementations or load reference solutions."
        prompt += f"\nThe harness limits the entire solver.py execution to {self.solver_timeout} seconds and numerical solvers to {self.solver_threads} thread(s). Design your algorithm and termination handling within this budget."
        (attempt_dir / "solver.py").write_text("")
        generation = self.client.edit_file(attempt_dir / "solver.py", prompt, context=context, log_dir=attempt_dir, step="agent")
        if self.abort_event.is_set():
            raise RuntimeError("Sweep aborted before solver execution")
        metrics = {"generation": asdict(generation)}
        if not generation.ok:
            access_error = " ".join([generation.error or "", *generation.warnings]).lower()
            if any(marker in access_error for marker in ["usage limit", "usage_limit", "rate limit", "rate_limit", "quota", "not supported", "unauthorized", "authentication", "invalid api key"]):
                self.abort_event.set()
                raise ModelAccessError(f"Codex access/quota failure; preserve traces and resume after access is restored: {generation.error}")
            result = AttemptResult(idx, "agent_failed", False, False, None, gold, None, [generation.error or "Codex generation failed"], metrics=metrics)
        else:
            execution_started = time.perf_counter()
            exec_ok, stdout, stderr, output = self.run_solver(attempt_dir, self.solver_timeout, self.solver_threads)
            metrics["execution_seconds"] = time.perf_counter() - execution_started
            metrics["execution_timed_out"] = not exec_ok and "[framework] solver execution timed out" in stderr
            if self.solver_cooldown_seconds and metrics["execution_seconds"] >= 5:
                cooldown_started = time.perf_counter()
                time.sleep(self.solver_cooldown_seconds)
                metrics["cooldown_seconds"] = time.perf_counter() - cooldown_started
            objective, issues, output_message = OutputVerifier.extract_objective(output)
            verification = {"ok": False, "issues": []}
            if exec_ok:
                verification = OutputVerifier.verify_output(self.suite, problem, instance_root / "data.json", attempt_dir / "output.json", self.tol)
                issues.extend(verification.get("issues", []))
                if verification.get("objective_value") is not None:
                    objective = float(verification["objective_value"])
            else:
                issues.append(stderr or "solver execution failed")
            verdict, notes = AlgorithmVerifier.run_llm_verifier(attempt_dir, self.client, self.suite)
            if verdict != "pass":
                issues.append(f"algorithm_verifier_failed: {verdict or 'unknown'}")
            verify_ok = bool(verification["ok"]) and verdict == "pass"
            status, gap = OutputVerifier.assess_status(exec_ok=exec_ok, output=output, verify_ok=verify_ok, objective=objective, gold_obj=gold, gap_threshold=self.gap_threshold, execution_timed_out=metrics["execution_timed_out"])
            result = AttemptResult(idx, status, exec_ok, verify_ok, objective, gold, gap, issues, verdict, notes,
                                   stdout=stdout, stderr=stderr, feasibility_ok=bool(verification["ok"]), metrics=metrics)
            lines = ExecLogger.build_exec_lines(stdout, stderr)
            ExecLogger.append_output_message(lines, output_message)
            ExecLogger.append_verification_issues(lines, issues)
            ExecLogger.append_llm_verifier(lines, notes)
            ExecLogger.write_exec_log(attempt_dir, lines)
            verifier_metrics = safe_load_json(attempt_dir / "llm_verifier_metrics.json")
            if verifier_metrics:
                metrics["algorithm_verification"] = verifier_metrics
        if self.reflection and result.status != "ok":
            reflection_prompt = PromptBuilder.build_reflection_prompt(prompt, gap_info=str(result.gap_to_gold))
            reflection_prompt += "\nKeep the reflection concise and actionable. Edit only debug.md."
            paths = [instance_root / "data.json"] + [attempt_dir / name for name in ["solver.py", "exec.log", "alg_verifier.md", "output.json"]]
            reflected = self.client.edit_file(attempt_dir / "debug.md", reflection_prompt, context=paths, log_dir=attempt_dir, step="reflection")
            metrics["reflection"] = asdict(reflected)
        metrics["total_seconds"] = time.perf_counter() - started
        ArtifactManager.write_text(attempt_dir, "summary.json", json.dumps(result.to_dict(), indent=2))
        return result

    @staticmethod
    def can_early_stop(result: AttemptResult, gap_threshold: float) -> bool:
        return result.status == "ok" and result.exec_ok and result.verify_ok and result.gap_to_gold is not None and result.gap_to_gold <= gap_threshold

    def run_instance(self, source: Path, root: Path, problem: str | None = None) -> dict[str, Any]:
        if self.abort_event.is_set():
            raise RuntimeError("Sweep aborted before starting another instance")
        problem = problem or InstanceManager.infer_problem_from_path(source, ALL_PROBLEMS)
        destination, attempts_root = ArtifactManager.copy_instance(source, root, self.max_attempts, suite=self.suite)
        gold = OutputVerifier.load_gold_objective(source, self.suite)
        results = []
        successful = None
        first_success = None
        for idx in range(1, self.max_attempts + 1):
            if self.abort_event.is_set():
                raise RuntimeError("Sweep aborted before next attempt")
            if successful is not None:
                result = AttemptResult(idx, "skipped_previous_solved", True, True, successful.objective, gold,
                                       successful.gap_to_gold, [], reused_from_previous=True,
                                       first_success_attempt=first_success, feasibility_ok=True)
            else:
                directory = attempts_root / f"attempt_{idx:02d}"
                directory.mkdir(parents=True, exist_ok=True)
                result = self.run_attempt(idx, directory, problem, gold)
                if self.can_early_stop(result, self.gap_threshold):
                    successful, first_success = result, idx
            results.append(result.to_dict())
            summary = {"instance": source.name, "suite": self.suite, "problem": problem, "source": str(source),
                       "output_dir": str(destination), "backend": "codex-cli", "model": self.model,
                       "gold_objective": gold, "reflection": self.reflection,
                       "solver_threads": self.solver_threads, "solver_timeout": self.solver_timeout,
                       "solver_cooldown_seconds": self.solver_cooldown_seconds,
                       "max_attempts": self.max_attempts, "gap_threshold": self.gap_threshold, "attempts": results}
            ArtifactManager.write_text(destination, "instance_summary.json", json.dumps(summary, indent=2))
            print(f"{source.name}: attempt {idx}/{self.max_attempts} {result.status}", flush=True)
        return summary
