"""Independent checker for RCSP outputs: duties {"depot": <crew base station>, "tasks": [<task>, ...]}.

Rebuilds every duty's connections from data.json (SOURCE depot -> first task, one CONNECTION per consecutive
pair, SINK last task -> depot; a missing connection makes the duty infeasible) and checks the labour rules:
duty length (sum of the connection durations) <= 34 200 s, and some break connection a (timeBeforeBreak >= 0)
with (durations before a) + timeBeforeBreak(a) <= 19 800 s and timeAfterBreak(a) + (durations after a)
<= 19 800 s. Every task must be in at least one duty. The objective is the sum of the connection costs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.utils.output_contract import integer_value

MAX_DUTY = 34_200
MAX_WORK = 19_800


def _load(obj: Any) -> dict[str, Any]:
    if isinstance(obj, (str, Path)):
        return json.loads(Path(obj).read_text())
    if isinstance(obj, dict):
        return obj
    raise TypeError("expected dict or path-like input")


def _result(ok: bool, message: str, objective_value: float | None = None) -> dict[str, Any]:
    return {"ok": ok, "issues": [] if ok else [message], "objective_value": objective_value}


def chain_cost(chain: list[list[Any]]) -> tuple[int | None, str]:
    """(cost, "") for a duty given as its connection rows, else (None, reason)."""
    length = sum(c[6] for c in chain)
    if length > MAX_DUTY:
        return None, f"duty length {length} s exceeds {MAX_DUTY} s"
    elapsed = 0
    for c in chain:
        if c[4] >= 0 and elapsed + c[4] <= MAX_WORK and c[5] + (length - elapsed - c[6]) <= MAX_WORK:
            return sum(c[7] for c in chain), ""
        elapsed += c[6]
    return None, "no meal break with at most 5.5 h of work before and after it"


def verify_solution(data: Any, output: Any, tol: float | None = None) -> dict[str, Any]:
    _ = tol
    try:
        d, output = _load(data), _load(output)
        n = int(d["n_tasks"])
        bases = {int(b) for b in d["crew_bases"]}
        table: dict[str, dict] = {"SOURCE": {}, "CONNECTION": {}, "SINK": {}}
        for c in d["connections"]:
            table[c[1]][(int(c[2]), int(c[3]))] = [int(c[0]), c[1], *(int(v) for v in c[2:8])]
    except Exception as exc:
        return _result(False, f"data load error: {exc}")
    solution = output.get("solution")
    if not isinstance(solution, dict) or not isinstance(solution.get("duties"), list):
        return _result(False, "solution.duties missing or not a list")
    covered: set[int] = set()
    total = 0
    for k, duty in enumerate(solution["duties"]):
        if not isinstance(duty, dict):
            return _result(False, f"duties[{k}] must be an object with depot and tasks")
        depot, tasks = duty.get("depot"), duty.get("tasks")
        if not integer_value(depot) or int(depot) not in bases:
            return _result(False, f"duties[{k}].depot must be a crew base ({sorted(bases)})")
        if not isinstance(tasks, list) or not tasks or not all(integer_value(t) for t in tasks):
            return _result(False, f"duties[{k}].tasks must be a non-empty list of task indices")
        depot, tasks = int(depot), [int(t) for t in tasks]
        if any(not 0 <= t < n for t in tasks):
            return _result(False, f"duties[{k}] has a task index outside 0..{n - 1}")
        chain = [table["SOURCE"].get((depot, tasks[0]))] + [table["CONNECTION"].get(p) for p in zip(tasks, tasks[1:])] + [table["SINK"].get((tasks[-1], depot))]
        if any(c is None for c in chain):
            i = chain.index(None)
            what = "no SOURCE connection" if i == 0 else "no SINK connection" if i == len(chain) - 1 else f"no connection {tasks[i - 1]} -> {tasks[i]}"
            return _result(False, f"duties[{k}]: {what}")
        cost, why = chain_cost(chain)
        if cost is None:
            return _result(False, f"duties[{k}]: {why}")
        total += cost
        covered |= set(tasks)
    if len(covered) != n:
        return _result(False, f"{n - len(covered)} task(s) not covered")
    return _result(True, "ok", float(total))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an RCSP solution.")
    parser.add_argument("--dir", type=Path, default=Path("."))
    parser.add_argument("--output", default="output.json")
    args = parser.parse_args()
    print(json.dumps(verify_solution(args.dir / "data.json", args.dir / args.output), indent=2))


if __name__ == "__main__":
    main()
