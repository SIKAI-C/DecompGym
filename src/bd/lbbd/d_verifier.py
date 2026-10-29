"""Independent checker for LBBD (assignment + single-machine scheduling) outputs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from src.utils.output_contract import finite_number, integer_value


def _load(obj: Any) -> dict[str, Any]:
    if isinstance(obj, (str, Path)):
        return json.loads(Path(obj).read_text())
    if isinstance(obj, dict):
        return obj
    raise TypeError("expected dict or path-like input")


def _result(ok: bool, message: str, objective_value: float | None = None) -> dict[str, Any]:
    return {"ok": ok, "issues": [] if ok else [message], "objective_value": objective_value}


def objective_of(data: dict[str, Any], plan: dict[int, tuple[int, float]]) -> float:
    total = 0.0
    for j, (i, start) in plan.items():
        p = data["proc_time"][i][j]
        total += data["proc_cost"][i][j] + data["weight"][j] * max(0.0, start + p - data["due"][j])
    return total


def verify_solution(data: Any, output: Any, tol: float | None = None) -> dict[str, Any]:
    eps = 1e-6 if tol is None else max(tol, 1e-9)
    try:
        data, output = _load(data), _load(output)
        n, m = int(data["n_tasks"]), int(data["n_machines"])
    except Exception as exc:
        return _result(False, f"data load error: {exc}")
    solution = output.get("solution")
    if not isinstance(solution, dict) or not isinstance(solution.get("assignments"), list):
        return _result(False, "solution.assignments missing or not a list")
    plan: dict[int, tuple[int, float]] = {}
    for idx, row in enumerate(solution["assignments"]):
        if not isinstance(row, dict):
            return _result(False, f"assignments[{idx}] is not an object")
        task, machine, start = row.get("task_id"), row.get("machine_id"), row.get("start")
        if not integer_value(task) or not integer_value(machine):
            return _result(False, f"assignments[{idx}] task_id/machine_id must be integer JSON numbers")
        if not finite_number(start):
            return _result(False, f"assignments[{idx}].start must be a finite number")
        task, machine = int(task), int(machine)
        if not 0 <= task < n or not 0 <= machine < m:
            return _result(False, f"assignments[{idx}] index out of range")
        if task in plan:
            return _result(False, f"task {task} assigned more than once")
        if start < data["release"][task] - eps:
            return _result(False, f"task {task} starts before its release time")
        plan[task] = (machine, float(start))
    if len(plan) != n:
        return _result(False, f"{n - len(plan)} task(s) not assigned")
    for i in range(m):
        jobs = sorted((start, j) for j, (mi, start) in plan.items() if mi == i)
        for (s1, j1), (s2, j2) in zip(jobs, jobs[1:]):
            if s1 + data["proc_time"][i][j1] > s2 + eps:
                return _result(False, f"tasks {j1} and {j2} overlap on machine {i}")
    try:
        objective = objective_of(data, plan)
    except OverflowError:
        return _result(False, "start times are too large to evaluate")
    if not math.isfinite(objective):
        return _result(False, "objective is not finite")
    return _result(True, "ok", objective)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an LBBD solution.")
    parser.add_argument("--dir", type=Path, default=Path("."))
    parser.add_argument("--output", default="output.json")
    args = parser.parse_args()
    print(json.dumps(verify_solution(args.dir / "data.json", args.dir / args.output), indent=2))


if __name__ == "__main__":
    main()
