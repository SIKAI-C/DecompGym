"""Run each selected instance with and without reflection; reuse completed runs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import platform
import sys
import threading

from src.agent.experiment_runner import ExperimentRunner, ModelAccessError, SUITE_PROBLEMS
from src.agent.resource_monitor import CleanupError
from src.agent.files_manager import InstanceManager
from src.utils.json_utils import safe_load_json


MODES = {"plain": False, "reflection": True}


def instance_metrics(summary: dict, budget: int) -> dict:
    attempts = summary.get("attempts", [])[:budget]
    verified = [attempt for attempt in attempts if attempt.get("verify_ok")]
    gaps = [attempt["gap_to_gold"] for attempt in verified if attempt.get("gap_to_gold") is not None]
    return {"execution": any(attempt.get("exec_ok") for attempt in attempts), "verification": bool(verified),
            "quality": any(attempt.get("status") in {"ok", "skipped_previous_solved"} for attempt in attempts),
            "best_gap": min(gaps) if gaps else None}


def summarize(root: Path, modes: list[str] | None = None) -> dict:
    groups = {}
    for path in sorted(root.glob("*/*/*/*/instance_summary.json")):
        parts = path.relative_to(root).parts
        if ".interrupted_" in parts[3] or parts[0] not in (modes or MODES):
            continue
        summary = safe_load_json(path)
        if not summary or len(summary.get("attempts", [])) != summary.get("max_attempts"):
            continue
        groups.setdefault(tuple(parts[:3]), []).append(summary)
    rows = []
    for (mode, suite, problem), instances in sorted(groups.items()):
        metrics = {}
        for budget in [1, 5, 10]:
            values = [instance_metrics(summary, budget) for summary in instances]
            metrics[str(budget)] = {key: sum(value[key] for value in values) / len(values) for key in ["execution", "verification", "quality"]}
            gaps = [value["best_gap"] for value in values if value["best_gap"] is not None]
            metrics[str(budget)]["mean_verified_gap"] = sum(gaps) / len(gaps) if gaps else None
        rows.append({"mode": mode, "suite": suite, "problem": problem, "num_instances": len(instances), "metrics": metrics})
    result = {"rows": rows, "num_instance_runs": sum(len(values) for values in groups.values())}
    (root / "evaluation.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/experiments/default_run"))
    parser.add_argument("--suite", choices=["bd", "cg", "all"], default="all")
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument("--model", default=None)
    parser.add_argument("--workers", type=int, default=2, help="Maximum concurrent instance experiments; attempts within an instance remain sequential.")
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--agent-timeout", type=int, default=900)
    parser.add_argument("--solver-timeout", type=int, default=60, help="Deadline in seconds for the entire generated solver execution.")
    parser.add_argument("--solver-threads", type=int, default=1, help="Thread limit per solver; workers controls concurrent instances.")
    parser.add_argument("--solver-cooldown-seconds", type=float, default=0, help="Pause after executions lasting at least five seconds; outside the solver deadline.")
    parser.add_argument("--gap-threshold", type=float, default=0.01)
    parser.add_argument("--limit-per-problem", type=int, default=None)
    args = parser.parse_args()
    if args.workers < 1 or (args.limit_per_problem is not None and args.limit_per_problem < 1):
        parser.error("Worker and instance limits must be positive")
    args.modes = list(dict.fromkeys(args.modes))
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    configuration = {"backend": "codex-cli", "model": args.model, "max_attempts": args.max_attempts,
                     "agent_timeout": args.agent_timeout, "solver_timeout": args.solver_timeout,
                     "gap_threshold": args.gap_threshold, "data_root": str(args.data_root.resolve()),
                     "python": sys.version, "platform": platform.platform(),
                     "scope": {"modes": args.modes, "suite": args.suite, "instances_per_problem": args.limit_per_problem},
                     "workers": args.workers}
    configuration["solver_threads"] = args.solver_threads
    configuration["solver_cooldown_seconds"] = args.solver_cooldown_seconds
    previous = safe_load_json(root / "run_config.json")
    protocol_keys = ["backend", "model", "max_attempts", "agent_timeout", "solver_timeout", "gap_threshold", "data_root", "python", "platform"]
    if previous is not None and any(previous.get(key) != configuration[key] for key in protocol_keys):
        raise ValueError("Protocol differs from the existing run; choose another output root.")
    if previous is not None and previous.get("solver_threads", 0) != args.solver_threads:
        raise ValueError("Solver thread limit differs from the saved run; use a separate output root to preserve its resource settings.")
    if previous is not None and previous.get("solver_cooldown_seconds", 0) != args.solver_cooldown_seconds:
        raise ValueError("Cooldown differs from the saved run; choose another output root.")
    (root / "run_config.json").write_text(json.dumps(configuration, indent=2))
    jobs = []
    abort_event = threading.Event()
    selected_count = 0
    reused_count = 0
    suites = ["bd", "cg"] if args.suite == "all" else [args.suite]
    for mode in args.modes:
        for suite in suites:
            for problem in SUITE_PROBLEMS[suite]:
                instances = InstanceManager.find_instances(args.data_root / suite / problem)
                instances.sort(key=lambda path: (next((i for i, size in enumerate(["small", "medium", "large"]) if f"_{size}_" in path.name), 3), path.name))
                if args.limit_per_problem is not None:
                    instances = instances[:args.limit_per_problem]
                if not instances:
                    raise RuntimeError(f"No dataset for {suite}/{problem}; generate it first.")
                destination = root / mode / suite / problem
                destination.mkdir(parents=True, exist_ok=True)
                runner = ExperimentRunner(model=args.model, suite=suite, problem=[problem], max_attempts=args.max_attempts,
                    agent_timeout=args.agent_timeout, solver_timeout=args.solver_timeout, gap_threshold=args.gap_threshold,
                    reflection=MODES[mode], solver_threads=args.solver_threads, solver_cooldown_seconds=args.solver_cooldown_seconds)
                runner.abort_event = abort_event
                for instance in instances:
                    selected_count += 1
                    path = destination / instance.name / "instance_summary.json"
                    completed = safe_load_json(path)
                    if completed and len(completed.get("attempts", [])) == args.max_attempts:
                        reused_count += 1
                        print(f"Reused completed {mode}/{suite}/{instance.name}", flush=True)
                        continue
                    if path.parent.exists():
                        suffix = 1
                        backup = path.parent.with_name(path.parent.name + f".interrupted_{suffix}")
                        while backup.exists():
                            suffix += 1
                            backup = path.parent.with_name(path.parent.name + f".interrupted_{suffix}")
                        path.parent.rename(backup)
                    jobs.append((runner, instance, destination, problem, mode))
    # Interleave policies and families, so partial monitoring covers both settings from the start.
    jobs.sort(key=lambda job: (next((i for i, size in enumerate(["small", "medium", "large"]) if f"_{size}_" in job[1].name), 3), job[3], job[1].name, args.modes.index(job[4])))
    def save_progress(completed_count: int, failures: list[dict]) -> None:
        (root / "progress.json").write_text(json.dumps({"expected_instance_runs": selected_count,
            "reused_completed_runs": reused_count, "completed_this_invocation": completed_count,
            "pending": len(jobs) - completed_count, "errors": failures}, indent=2))
    save_progress(0, [])
    print(f"Selected {selected_count} runs; reused {reused_count}; scheduled {len(jobs)} with {args.workers} workers", flush=True)
    failures = []
    def run_job(runner, instance, destination, problem):
        try:
            return runner.run_instance(instance, destination, problem)
        except (CleanupError, ModelAccessError):
            abort_event.set()
            raise
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, runner, instance, destination, problem): (mode, instance.name)
                   for runner, instance, destination, problem, mode in jobs}
        completed_count = 0
        for future in as_completed(futures):
            mode, name = futures[future]
            try:
                future.result()
                print(f"Completed {mode}/{name}", flush=True)
            except (CleanupError, ModelAccessError) as error:
                for pending in futures:
                    pending.cancel()
                failures.append({"mode": mode, "instance": name, "error": str(error), "fatal_infrastructure_error": True})
                save_progress(completed_count, failures)
                raise
            except Exception as error:
                failures.append({"mode": mode, "instance": name, "error": str(error)})
                print(f"Failed {mode}/{name}: {error}", flush=True)
            completed_count += 1
            save_progress(completed_count, failures)
            summarize(root, args.modes)
    result = summarize(root, args.modes)
    (root / "completion.json").write_text(json.dumps({"complete": not failures, "expected_instance_runs": selected_count,
                                                    "errors": failures, **result}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
