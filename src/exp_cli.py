"""
CLI entrypoint for Experiment 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent.experiment_runner import ALL_PROBLEMS, SUITE_PROBLEMS, ExperimentRunner
from src.agent.files_manager import ArtifactManager, InstanceManager


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
        suite = InstanceManager.infer_suite_from_path(inst_path, SUITE_PROBLEMS)
        problem = args.problem[0] if args.problem else InstanceManager.infer_problem_from_path(inst_path, ALL_PROBLEMS)
        out_root = ArtifactManager.ensure_output_root(suite, problem, args.model, note=args.note)
        summary = ExperimentRunner.run_instance(
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

    summaries = ExperimentRunner.run_all_problems(
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
