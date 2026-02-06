"""
Experiment 2 runner logic: attempts, instances, problems, and suites.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

from src.agent.algorithm_verifier import AlgorithmVerifier
from src.agent.exec_logger import ExecLogger
from src.agent.files_manager import ArtifactManager, InstanceManager
from src.agent.output_verifier import OutputVerifier
from src.agent.prompt_builder import PromptBuilder
from src.utils.json_utils import safe_load_json


SUITE_PROBLEMS = {
    "bd": ["tsso", "fctp", "km", "lrp", "ufl"],
    "cg": ["bp", "csp", "gap", "shift"],
}
ALL_PROBLEMS = sorted({p for probs in SUITE_PROBLEMS.values() for p in probs})


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "status": self.status,
            "exec_ok": self.exec_ok,
            "verify_ok": self.verify_ok,
            "objective": self.objective,
            "gold_objective": self.gold_objective,
            "gap_to_gold": self.gap_to_gold,
            "issues": self.issues,
            "llm_verdict": self.llm_verdict,
            "llm_notes": self.llm_notes,
            "reused_from_previous": self.reused_from_previous,
            "first_success_attempt": self.first_success_attempt,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


class ExperimentRunner:
    def __init__(
        self,
        *,
        model_name: str,
        note: str | None,
        suite: str,
        problems: Sequence[str] | None,
        instance: str | None,
        benchmark_root: str,
        max_attempts: int,
        solver_timeout: int,
        tol: float,
        gap_threshold: float,
        auto_yes: bool,
        stream: bool,
        reflection: bool,
        run_all: bool,
    ) -> None:
        self.model_name = model_name
        self.note = note
        self.suite = suite
        self.problems = list(problems) if problems is not None else None
        self.instance = instance
        self.benchmark_root = benchmark_root
        self.max_attempts = max_attempts
        self.solver_timeout = solver_timeout
        self.tol = tol
        self.gap_threshold = gap_threshold
        self.auto_yes = auto_yes
        self.stream = stream
        self.reflection = reflection
        self.run_all = run_all

    def run(self) -> None:
        benchmark_root = Path(self.benchmark_root) / self.suite
        if self.instance:
            self._run_single_instance()
            return

        problems = self._select_problems(benchmark_root)

        summaries = ExperimentRunner.run_all_problems(
            suite=self.suite,
            model_name=self.model_name,
            note=self.note,
            benchmark_root=benchmark_root,
            problems=problems,
            max_attempts=self.max_attempts,
            solver_timeout=self.solver_timeout,
            tol=self.tol,
            gap_threshold=self.gap_threshold,
            auto_yes=self.auto_yes,
            stream=self.stream,
            reflection=self.reflection,
        )
        print(json.dumps(summaries, indent=2))

    def _run_single_instance(self) -> None:
        inst_path = Path(self.instance).resolve()
        suite = InstanceManager.infer_suite_from_path(inst_path, SUITE_PROBLEMS)
        problem = (
            self.problems[0]
            if self.problems
            else InstanceManager.infer_problem_from_path(inst_path, ALL_PROBLEMS)
        )
        out_root = ArtifactManager.ensure_output_root(suite, problem, self.model_name, note=self.note)
        summary = ExperimentRunner.run_instance(
            instance_src=inst_path,
            out_root=out_root,
            model_name=self.model_name,
            max_attempts=self.max_attempts,
            solver_timeout=self.solver_timeout,
            tol=self.tol,
            gap_threshold=self.gap_threshold,
            auto_yes=self.auto_yes,
            stream=self.stream,
            suite=suite,
            reflection=self.reflection,
        )
        print(json.dumps(summary, indent=2))

    def _select_problems(self, benchmark_root: Path) -> list[str]:
        if self.run_all or self.problems is None:
            return [p for p in SUITE_PROBLEMS[self.suite] if (benchmark_root / p).exists()]
        return self.problems

    @staticmethod
    def run_solver(attempt_dir: Path, timeout: int) -> tuple[bool, str, str, dict[str, Any] | None]:
        attempt_dir = attempt_dir.resolve()
        solver_path = attempt_dir / "solver.py"
        # Guard against missing/empty solver.py
        if (not solver_path.exists()) or solver_path.stat().st_size == 0:
            return False, "", "solver.py missing or empty", None

        # Copy current env for subprocess
        env = os.environ.copy()
        try:
            proc = subprocess.run(
                [sys.executable, "solver.py"],
                cwd=attempt_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return False, "", "timeout", None
        except Exception as exc:
            return False, "", str(exc), None

        stdout = proc.stdout[-4000:]
        stderr = proc.stderr[-4000:]
        if proc.returncode != 0:
            # Return stderr so caller can see the actual error message
            return False, stdout, stderr or f"returncode={proc.returncode}", None

        output_path = attempt_dir / "output.json"
        if not output_path.exists():
            return False, stdout, "output.json missing", None
        output = safe_load_json(output_path)
        if output is None:
            return False, stdout, "failed to parse output.json", None
        return True, stdout, stderr, output

    @staticmethod
    def run_attempt(
        attempt_idx: int,
        attempt_dir: Path,
        model: Model,
        auto_yes: bool,
        stream: bool,
        solver_timeout: int,
        tol: float,
        gap_threshold: float,
        prev_debug: str,
        gold_obj: float | None,
        suite: str,
        reflection: bool,
    ) -> AttemptResult:
        # print(f"Running attempt {attempt_idx} in {attempt_dir}")
        # print(f"Previous debug: {prev_debug}")
        # print(f"Gold objective: {gold_obj}")

        instance_root = attempt_dir.parent.parent
        data_path = instance_root / "data.json"
        prompt_path = instance_root / "prompt.txt"
        base_prompt = prompt_path.read_text()
        base_prompt = PromptBuilder.ensure_location_prompt(base_prompt)
        ArtifactManager.write_text(instance_root, "prompt.txt", base_prompt)
        prompt_for_agent = base_prompt if not prev_debug else PromptBuilder.build_prev_debug_prompt(base_prompt, prev_debug)

        # First, let the agent edit solver.py using prior reflections (if any).
        ArtifactManager.ensure_dir(attempt_dir, exist_ok=True)
        solver_path = ArtifactManager.write_text(attempt_dir, "solver.py", "")
        fnames = [str(solver_path)]
        attempts_root = attempt_dir.parent
        prev_ro = (
            ArtifactManager.collect_previous_attempt_files(attempts_root, attempt_idx)
            if reflection
            else []
        )
        ro_files = [str(data_path.resolve()), str(prompt_path.resolve()), *prev_ro]
        coder = Coder.create(
            main_model=model,
            fnames=fnames,
            read_only_fnames=ro_files,
            io=InputOutput(yes=auto_yes),
            stream=stream,
            use_git=False,
            edit_format="diff",
        )
        coder.run(prompt_for_agent)

        exec_ok, stdout, stderr, output = ExperimentRunner.run_solver(attempt_dir, solver_timeout)
        exec_lines = ExecLogger.build_exec_lines(stdout, stderr)

        issues: list[str] = []
        objective, objective_issues, output_msg = OutputVerifier.extract_objective(output)
        issues.extend(objective_issues)
        verify_ok = False

        if exec_ok and output is not None:
            problem = InstanceManager.infer_problem_from_path(attempt_dir, ALL_PROBLEMS)
            verify_res = OutputVerifier.verify_output(suite, problem, data_path, attempt_dir / "output.json", tol)
            verify_ok = bool(verify_res.get("ok"))
            if not verify_ok:
                issues.extend(verify_res.get("issues", []))
        else:
            issues.append(stderr or "execution failed")

        ExecLogger.append_output_message(exec_lines, output_msg)
        ExecLogger.append_verification_issues(exec_lines, issues if exec_ok and output is not None else [])

        llm_verdict, llm_notes = AlgorithmVerifier.run_llm_verifier(
            attempt_dir=attempt_dir,
            model=model,
            auto_yes=auto_yes,
            stream=stream,
            suite=suite,
        )
        if llm_verdict is None or llm_verdict != "pass":
            verify_ok = False
            issues.append(f"llm_verifier_failed: {llm_verdict or 'unknown'}")
        ExecLogger.append_llm_verifier(exec_lines, llm_notes)
        exec_log = ExecLogger.write_exec_log(attempt_dir, exec_lines)

        status, gap = OutputVerifier.assess_status(
            exec_ok=exec_ok,
            output=output,
            verify_ok=verify_ok,
            objective=objective,
            gold_obj=gold_obj,
            gap_threshold=gap_threshold,
        )
        
        if reflection:
            debug_md = attempt_dir / "debug.md"
            if status != "ok":
                ArtifactManager.write_text(attempt_dir, "debug.md", "")  # placeholder; aider will fill reflection
                # Ask aider to update debug.md based on exec.log and current solver (read-only)
                gap_info = f"Gap to gold: {gap}" if gap is not None else None
                reflection_prompt = PromptBuilder.build_reflection_prompt(base_prompt, gap_info=gap_info)
                reflection_coder = Coder.create(
                    main_model=model,
                    fnames=[str(debug_md)],
                    read_only_fnames=[
                        str(solver_path.resolve()),
                        str(prompt_path.resolve()),
                        str(exec_log.resolve()),
                        str((attempt_dir / "alg_verifier.md").resolve()),
                        str((attempt_dir / "output.json").resolve()),
                    ],
                    io=InputOutput(yes=auto_yes),
                    stream=stream,
                    use_git=False,
                    edit_format="diff",
                )
                reflection_coder.run(reflection_prompt)
            else:
                ArtifactManager.write_text(attempt_dir, "debug.md", "Passed verification.")

        summary_path = attempt_dir / "summary.json"
        ArtifactManager.write_text(
            attempt_dir,
            "summary.json",
            json.dumps(
                AttemptResult(
                    attempt=attempt_idx,
                    status=status,
                    exec_ok=exec_ok,
                    verify_ok=verify_ok,
                    objective=objective,
                    gold_objective=gold_obj,
                    gap_to_gold=gap,
                    issues=issues,
                    llm_verdict=llm_verdict,
                    llm_notes=llm_notes,
                    stdout=stdout,
                    stderr=stderr,
                ).to_dict(),
                indent=2,
            )
        )

        return AttemptResult(
            attempt=attempt_idx,
            status=status,
            exec_ok=exec_ok,
            verify_ok=verify_ok,
            objective=objective,
            gold_objective=gold_obj,
            gap_to_gold=gap,
            issues=issues,
            llm_verdict=llm_verdict,
            llm_notes=llm_notes,
            stdout=stdout,
            stderr=stderr,
        )

    @staticmethod
    def can_early_stop(res: AttemptResult, gap_threshold: float) -> bool:
        if not (res.exec_ok and res.verify_ok):
            return False
        if res.gap_to_gold is None:
            return True
        return res.gap_to_gold <= gap_threshold

    @staticmethod
    def run_instance(
        instance_src: Path,
        out_root: Path,
        model_name: str,
        max_attempts: int,
        solver_timeout: int,
        tol: float,
        gap_threshold: float,
        auto_yes: bool,
        stream: bool,
        suite: str,
        reflection: bool,
    ) -> dict[str, Any]:
        dest_inst, attempts_root = ArtifactManager.copy_instance(instance_src, out_root, max_attempts, suite=suite)
        model = Model(model_name)
        gold_obj = OutputVerifier.load_gold_objective(dest_inst, suite=suite)

        results: list[AttemptResult] = []
        last_good: AttemptResult | None = None
        first_success_idx: int | None = None
        for idx in range(1, max_attempts + 1):
            a_dir = attempts_root / f"attempt_{idx:02d}"
            if last_good and ExperimentRunner.can_early_stop(last_good, gap_threshold):
                res = AttemptResult(
                    attempt=idx,
                    status="skipped_previous_solved",
                    exec_ok=True,
                    verify_ok=True,
                    objective=last_good.objective,
                    gold_objective=last_good.gold_objective,
                    gap_to_gold=last_good.gap_to_gold,
                    issues=["skipped: previous attempt already solved"],
                    reused_from_previous=True,
                    first_success_attempt=first_success_idx,
                    stdout=None,
                    stderr=None,
                )
            else:
                prev_debug = (
                    ArtifactManager.collect_previous_debug(dest_inst / "attempts", idx)
                    if (idx > 1 and reflection)
                    else ""
                )
                ArtifactManager.ensure_dir(a_dir, exist_ok=True)
                res = ExperimentRunner.run_attempt(
                    attempt_idx=idx,
                    attempt_dir=a_dir,
                    model=model,
                    auto_yes=auto_yes,
                    stream=stream,
                    solver_timeout=solver_timeout,
                    tol=tol,
                    gap_threshold=gap_threshold,
                    prev_debug=prev_debug,
                    gold_obj=gold_obj,
                    suite=suite,
                    reflection=reflection,
                )
                if res.exec_ok and res.verify_ok:
                    if first_success_idx is None:
                        first_success_idx = idx
                    last_good = res
            results.append(res)
            if ExperimentRunner.can_early_stop(res, gap_threshold):
                # allow remaining attempts to skip but keep loop to populate reused entries
                continue

        inst_summary = {
            "instance": instance_src.name,
            "source": str(instance_src),
            "output_dir": str(dest_inst),
            "gold_objective": gold_obj,
            "attempts": [r.to_dict() for r in results],
        }
        ArtifactManager.write_text(dest_inst, "instance_summary.json", json.dumps(inst_summary, indent=2))
        return inst_summary

    @staticmethod
    def run_problem(
        suite: str,
        problem: str,
        model_name: str,
        note: str | None,
        benchmark_root: Path,
        max_attempts: int,
        solver_timeout: int,
        tol: float,
        gap_threshold: float,
        auto_yes: bool,
        stream: bool,
        reflection: bool,
    ) -> dict[str, Any]:
        inst_root = benchmark_root / problem
        instances = InstanceManager.find_instances(inst_root)
        if not instances:
            raise RuntimeError(f"No instances found under {inst_root}")
        out_root = ArtifactManager.ensure_output_root(suite, problem, model_name, note=note)

        summaries: list[dict[str, Any]] = []
        for inst in instances:
            summaries.append(
                ExperimentRunner.run_instance(
                    inst,
                    out_root,
                    model_name=model_name,
                    max_attempts=max_attempts,
                    solver_timeout=solver_timeout,
                    tol=tol,
                    gap_threshold=gap_threshold,
                    auto_yes=auto_yes,
                    stream=stream,
                    suite=suite,
                    reflection=reflection,
                )
            )

        problem_summary = {
            "problem": problem,
            "model": model_name,
            "output_root": str(out_root),
            "num_instances": len(summaries),
            "instances": summaries,
        }
        ArtifactManager.write_text(out_root, "problem_summary.json", json.dumps(problem_summary, indent=2))
        return problem_summary

    @staticmethod
    def run_all_problems(
        suite: str,
        model_name: str,
        note: str | None,
        benchmark_root: Path,
        problems: Sequence[str],
        max_attempts: int,
        solver_timeout: int,
        tol: float,
        gap_threshold: float,
        auto_yes: bool,
        stream: bool,
        reflection: bool,
    ) -> list[dict[str, Any]]:
        all_summaries: list[dict[str, Any]] = []
        for prob in problems:
            if not (benchmark_root / prob).exists():
                continue
            all_summaries.append(
                ExperimentRunner.run_problem(
                    suite,
                    prob,
                    model_name=model_name,
                    note=note,
                    benchmark_root=benchmark_root,
                    max_attempts=max_attempts,
                    solver_timeout=solver_timeout,
                    tol=tol,
                    gap_threshold=gap_threshold,
                    auto_yes=auto_yes,
                    stream=stream,
                    reflection=reflection,
                )
            )
        return all_summaries
