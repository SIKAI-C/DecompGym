"""
Run Experiment 2: Base Benders Decomposition & Column Generation with Reflections and History.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

from src.agent.artifact_manager import ArtifactManager
from src.utils.json_utils import safe_load_json


# Supported problem folders under benchmark root and outputs.
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


def build_reflection_prompt(base_prompt: str, gap_info: str | None = None) -> str:
    """
    Construct the follow-up prompt for aider after a failed attempt.
    The agent should read solver.py and exec.log (read-only), diagnose the failure,
    and write guidance for the next attempt into debug.md. Do not edit solver.py in this step.
    """
    gap_text = gap_info if gap_info is not None else "unknown"
    debug_context = [
        "You are making an attempt to solve the following problem.",
        base_prompt,
        "",
        "You have generated the solver.py file based on the problem description and the data.json file.",
        "And the exec.log file shows the output of the solver.py file.",
        "You also have alg_verifier.md with an LLM-based algorithm check; use it to improve the reflection.",
        f"The current objective gap to the optimal solution is: {gap_text}",
        "Your tasks:",
        "1) Read exec.log and solver.py to understand why the current attempt failed to execute or failed to find the optimal solution.\
            you need to reasoning the current solver.py and exec.log.",
        "2) Write a concise reflection to debug.md: what likely went wrong and what the next attempt should change",
        "3) Provide concrete tips for the following attempts, based on the reflection and write into the debug.md file.",
    ]
    return "\n".join(debug_context)


def build_prev_debug_prompt(base_prompt: str, prior_notes: str) -> str:
    """
    Prompt used before generating a new solver attempt, surfacing lessons from previous failures.
    """
    context = [
        base_prompt,
        "",
        "You have prior failed attempts with reflections (and you can read their solver.py, debug.md, summary.json):",
        prior_notes or "(no prior reflections)",
        "",
        "Incorporate the lessons above to avoid repeating mistakes. Use the prior files as read-only context.",
    ]
    return "\n".join(context)


def find_instances(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "data.json").exists() and (p / "prompt.txt").exists():
            out.append(p)
    return out


def infer_problem_from_path(path: Path) -> str:
    for parent in path.resolve().parents:
        if parent.name in ALL_PROBLEMS:
            return parent.name
    raise ValueError(f"Could not infer problem type from path {path}")


def infer_suite_from_path(path: Path) -> str:
    for parent in path.resolve().parents:
        if parent.name in SUITE_PROBLEMS:
            return parent.name
    raise ValueError(f"Could not infer suite from path {path}")


def load_gold_objective(inst_dir: Path, suite: str) -> float | None:
    meta_path = inst_dir / "meta.json"
    if meta_path.exists():
        meta = safe_load_json(meta_path)
        if meta is not None:
            val = meta.get("ground_truth_objective") or meta.get("gold_objective")
            if val is not None:
                try:
                    return float(val)
                except Exception:
                    pass

    gold_file = "gold_bd.json" if suite == "bd" else "gold_cg.json"
    for name in ["gold_monolithic.json", gold_file]:
        p = inst_dir / name
        if not p.exists():
            continue
        payload = safe_load_json(p)
        if payload is None:
            continue
        for key in ("best_objective", "objective_value"):
            if key in payload and payload[key] is not None:
                try:
                    return float(payload[key])
                except Exception:
                    continue
    return None


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


def run_llm_verifier(
    attempt_dir: Path,
    model: Model,
    auto_yes: bool,
    stream: bool,
    suite: str,
) -> tuple[str | None, str]:
    instance_root = attempt_dir.parent.parent
    verifier_md = ArtifactManager.write_text(attempt_dir, "alg_verifier.md", "")
    solver_path = attempt_dir / "solver.py"
    data_path = instance_root / "data.json"
    prompt_path = instance_root / "prompt.txt"
    prompt = (
        "You are the LLM verifier. Inspect solver.py and judge whether it actually implements "
        f"{'Benders Decomposition' if suite == 'bd' else 'Column Generation'} rather than solving a trivial monolithic MIP.\n"
        "Return a short decision in alg_verifier.md.\n\n"
        "Format:\n"
        "VERDICT: PASS or FAIL\n"
        "NOTES: <1-3 bullet points>\n"
    )
    coder = Coder.create(
        main_model=model,
        fnames=[str(verifier_md)],
        read_only_fnames=[str(solver_path.resolve()), str(data_path.resolve()), str(prompt_path.resolve())],
        io=InputOutput(yes=auto_yes),
        stream=stream,
        use_git=False,
        edit_format="diff",
    )
    coder.run(prompt)
    text = verifier_md.read_text().strip()
    verdict = None
    for line in text.splitlines():
        if line.strip().upper().startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip().lower()
            break
    return verdict, text


def import_verifier(suite: str, problem: str):
    verifier_path = Path(__file__).resolve().parent / suite / problem / "d_verifier.py"
    if not verifier_path.exists():
        return None, f"verifier not found at {verifier_path}"

    module_name = f"{suite}_{problem}_verifier"
    spec = importlib.util.spec_from_file_location(module_name, verifier_path)
    if spec is None or spec.loader is None:
        return None, f"failed to load verifier from {verifier_path}"
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        return None, f"failed to import verifier '{verifier_path}': {exc}"
    return module, None


def verify_output(suite: str, problem: str, data_path: Path, output_path: Path, tol: float) -> dict[str, Any]:
    verifier, import_err = import_verifier(suite, problem)
    if import_err:
        return {"ok": False, "issues": [import_err]}
    if not hasattr(verifier, "verify_solution"):
        return {"ok": False, "issues": ["verify_solution not found"]}
    return verifier.verify_solution(str(data_path), str(output_path), tol=tol)


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
) -> AttemptResult:
    # print(f"Running attempt {attempt_idx} in {attempt_dir}")
    # print(f"Previous debug: {prev_debug}")
    # print(f"Gold objective: {gold_obj}")

    instance_root = attempt_dir.parent.parent
    data_path = instance_root / "data.json"
    prompt_path = instance_root / "prompt.txt"
    base_prompt = prompt_path.read_text()
    location_note = (
        "IMPORTANT: data.json and prompt.txt are located two levels up (../../). "
        "Read data from ../../data.json and ../../prompt.txt, and write output.json in the current folder. "
        "Do NOT read from ../data.json or attempts/data.json.\n\n"
    )
    if location_note not in base_prompt:
        base_prompt = base_prompt.rstrip() + "\n\n" + location_note
        ArtifactManager.write_text(instance_root, "prompt.txt", base_prompt)
    prompt_for_agent = base_prompt if not prev_debug else build_prev_debug_prompt(base_prompt, prev_debug)

    # First, let the agent edit solver.py using prior reflections (if any).
    ArtifactManager.ensure_dir(attempt_dir, exist_ok=True)
    solver_path = ArtifactManager.write_text(attempt_dir, "solver.py", "")
    fnames = [str(solver_path)]
    attempts_root = attempt_dir.parent
    prev_ro = collect_previous_attempt_files(attempts_root, attempt_idx)
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

    exec_ok, stdout, stderr, output = run_solver(attempt_dir, solver_timeout)
    exec_log = attempt_dir / "exec.log"
    exec_lines = [
        "=== EXECUTION ===",
        f"STDOUT:\n{stdout}",
        "",
        f"STDERR:\n{stderr}",
    ]

    issues: list[str] = []
    objective = None
    verify_ok = False

    if exec_ok and output is not None:
        objective = output.get("objective_value") or output.get("best_objective")
        if objective is not None:
            try:
                objective = float(objective)
            except Exception:
                issues.append("objective not numeric")
                objective = None
        output_msg = output.get("error_message") or output.get("message")
        if output_msg:
            exec_lines.append("")
            exec_lines.append("=== OUTPUT MESSAGE ===")
            exec_lines.append(str(output_msg))

        problem = infer_problem_from_path(attempt_dir)
        verify_res = verify_output(suite, problem, data_path, attempt_dir / "output.json", tol)
        verify_ok = bool(verify_res.get("ok"))
        if not verify_ok:
            issues.extend(verify_res.get("issues", []))
            exec_lines.append("")
            exec_lines.append("=== VERIFICATION ISSUES ===")
            exec_lines.extend(issues)
    else:
        issues.append(stderr or "execution failed")

    llm_verdict, llm_notes = run_llm_verifier(
        attempt_dir=attempt_dir,
        model=model,
        auto_yes=auto_yes,
        stream=stream,
        suite=suite,
    )
    if llm_verdict is None or llm_verdict != "pass":
        verify_ok = False
        issues.append(f"llm_verifier_failed: {llm_verdict or 'unknown'}")
    exec_lines.append("")
    exec_lines.append("=== LLM VERIFIER ===")
    exec_lines.append(llm_notes or "No verifier output.")
    ArtifactManager.write_text(attempt_dir, "exec.log", "\n".join(exec_lines))

    gap = None
    if objective is not None and gold_obj is not None:
        gap = abs(objective - gold_obj) / max(1.0, abs(gold_obj))

    gap_ok = (gap is None) or (gap_threshold is None) or (gap <= gap_threshold)

    if not exec_ok or output is None:
        status = "exec_failed"
    elif not verify_ok:
        status = "verify_failed"  # infeasible or failed verifier
    elif not gap_ok:
        status = "feasible_not_optimal"  # feasible but not within gap threshold
    else:
        status = "ok"
    debug_md = attempt_dir / "debug.md"
    if status != "ok":
        ArtifactManager.write_text(attempt_dir, "debug.md", "")  # placeholder; aider will fill reflection
        # Ask aider to update debug.md based on exec.log and current solver (read-only)
        gap_info = f"Gap to gold: {gap}" if gap is not None else None
        reflection_prompt = build_reflection_prompt(base_prompt, gap_info=gap_info)
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


def collect_previous_debug(attempts_root: Path, current_idx: int) -> str:
    pieces: list[str] = []
    for idx in range(1, current_idx):
        dbg = attempts_root / f"attempt_{idx:02d}" / "debug.md"
        if dbg.exists():
            pieces.append(dbg.read_text())
    return "\n\n".join(pieces)


def collect_previous_attempt_files(attempts_root: Path, current_idx: int) -> list[str]:
    """
    Collect read-only files from prior attempts to give the coder more context:
    - solver.py
    - debug.md
    - summary.json
    """
    ro: list[str] = []
    for idx in range(1, current_idx):
        a_dir = attempts_root / f"attempt_{idx:02d}"
        for name in ("solver.py", "debug.md", "summary.json"):
            p = a_dir / name
            if p.exists():
                ro.append(str(p.resolve()))
    return ro


def can_early_stop(res: AttemptResult, gap_threshold: float) -> bool:
    if not (res.exec_ok and res.verify_ok):
        return False
    if res.gap_to_gold is None:
        return True
    return res.gap_to_gold <= gap_threshold


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
) -> dict[str, Any]:
    dest_inst, attempt_dirs = ArtifactManager.copy_instance(instance_src, out_root, max_attempts, suite=suite)
    model = Model(model_name)
    gold_obj = load_gold_objective(dest_inst, suite=suite)

    results: list[AttemptResult] = []
    last_good: AttemptResult | None = None
    first_success_idx: int | None = None
    for idx, a_dir in enumerate(attempt_dirs, start=1):
        if last_good and can_early_stop(last_good, gap_threshold):
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
            prev_debug = collect_previous_debug(dest_inst / "attempts", idx) if idx > 1 else ""
            res = run_attempt(
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
            )
            if res.exec_ok and res.verify_ok:
                if first_success_idx is None:
                    first_success_idx = idx
                last_good = res
        results.append(res)
        if can_early_stop(res, gap_threshold):
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
) -> dict[str, Any]:
    inst_root = benchmark_root / problem
    instances = find_instances(inst_root)
    if not instances:
        raise RuntimeError(f"No instances found under {inst_root}")
    out_root = ArtifactManager.ensure_output_root(suite, problem, model_name, note=note)

    summaries: list[dict[str, Any]] = []
    for inst in instances:
        summaries.append(
            run_instance(
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
) -> list[dict[str, Any]]:
    all_summaries: list[dict[str, Any]] = []
    for prob in problems:
        if not (benchmark_root / prob).exists():
            continue
        all_summaries.append(
            run_problem(
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
            )
        )
    return all_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run aider-driven benchmarks with multi-level orchestration.")
    parser.add_argument(
        "--model", "-m", type=str, default="azure/adeza3-DE00023575-gpt-4.1", help="Model identifier for aider."
    )
    parser.add_argument(
        "--note", "-n", type=str, default=None, help="Optional note to tag output folder for experiment tracking."
    )
    parser.add_argument(
        "--suite",
        "-s",
        type=str,
        default="bd",
        choices=["bd", "cg"],
        help="Benchmark suite (bd or cg).",
    )
    parser.add_argument(
        "--problem", "-p", type=str, nargs="*", default=None, help="Problem types to run (e.g., tsso, gap)."
    )
    parser.add_argument("--instance", "-i", type=str, default=None, help="Path to a single instance folder.")
    parser.add_argument(
        "--benchmark-root", "-br", type=str, default="z_artifacts/benchmarks", help="Root containing problem folders."
    )
    parser.add_argument("--max-attempts", "-ma", type=int, default=10, help="Max attempts per instance.")
    parser.add_argument("--solver-timeout", "-sot", type=int, default=600, help="Timeout per solver run (seconds).")
    parser.add_argument("--tol", "-t", type=float, default=1e-6, help="Tolerance passed to verifiers.")
    parser.add_argument(
        "--gap-threshold",
        "-gt",
        type=float,
        default=0.01,
        help="Require relative gap to gold <= threshold before early stopping; 0 means only stop at optimal.",
    )
    parser.add_argument("--auto-yes", "-ay", action="store_true", default=True, help="Auto-confirm for aider.")
    parser.add_argument("--stream", "-st", action="store_true", help="Stream aider output.")
    parser.add_argument(
        "--run-all", "-ra", action="store_true", help="Run all available problems under benchmark root."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_root = Path(args.benchmark_root) / args.suite
    if args.instance:
        inst_path = Path(args.instance).resolve()
        suite = infer_suite_from_path(inst_path)
        problem = args.problem[0] if args.problem else infer_problem_from_path(inst_path)
        out_root = ArtifactManager.ensure_output_root(suite, problem, args.model, note=args.note)
        summary = run_instance(
            inst_path,
            out_root,
            model_name=args.model,
            max_attempts=args.max_attempts,
            solver_timeout=args.solver_timeout,
            tol=args.tol,
            gap_threshold=args.gap_threshold,
            auto_yes=args.auto_yes,
            stream=args.stream,
            suite=suite,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.run_all or args.problem is None:
        problems = [p for p in SUITE_PROBLEMS[args.suite] if (benchmark_root / p).exists()]
    else:
        problems = args.problem

    summaries = run_all_problems(
        suite=args.suite,
        model_name=args.model,
        note=args.note,
        benchmark_root=benchmark_root,
        problems=problems,
        max_attempts=args.max_attempts,
        solver_timeout=args.solver_timeout,
        tol=args.tol,
        gap_threshold=args.gap_threshold,
        auto_yes=args.auto_yes,
        stream=args.stream,
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
