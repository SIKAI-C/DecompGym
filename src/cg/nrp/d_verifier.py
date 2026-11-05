"""Independent checker for nurse rostering outputs: one roster per nurse (-1 = off, 0/1/2 = shift)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.output_contract import integer_value


def _load(obj: Any) -> dict[str, Any]:
    if isinstance(obj, (str, Path)):
        return json.loads(Path(obj).read_text())
    if isinstance(obj, dict):
        return obj
    raise TypeError("expected dict or path-like input")


def _result(ok: bool, message: str, objective_value: float | None = None) -> dict[str, Any]:
    return {"ok": ok, "issues": [] if ok else [message], "objective_value": objective_value}


def roster_issue(d: dict[str, Any], n: int, roster: list[int]) -> str | None:
    worked = [s for s in roster if s >= 0]
    if not d["min_days"][n] <= len(worked) <= d["max_days"][n]:
        return "working days outside [min_days, max_days]"
    if sum(1 for s in roster if s == 2) > d["max_nights"][n]:
        return "too many nights"
    run = 0
    for day, s in enumerate(roster):
        run = run + 1 if s >= 0 else 0
        if run > d["max_consecutive"]:
            return "too many consecutive working days"
        if day > 0 and [roster[day - 1], s] in d["forbidden_successions"]:
            return f"forbidden succession on day {day}"
    return None


def evaluate(d: dict[str, Any], rosters: list[list[int]]) -> float:
    total = sum(d["cost"][n][day][s] for n, r in enumerate(rosters) for day, s in enumerate(r) if s >= 0)
    for day in range(d["n_days"]):
        for s in range(3):
            covered = sum(1 for r in rosters if r[day] == s)
            total += d["shortage_penalty"] * max(0, d["demand"][day][s] - covered)
    return float(total)


def verify_solution(data: Any, output: Any, tol: float | None = None) -> dict[str, Any]:
    _ = tol
    try:
        d, output = _load(data), _load(output)
        N, D = int(d["n_nurses"]), int(d["n_days"])
    except Exception as exc:
        return _result(False, f"data load error: {exc}")
    solution = output.get("solution")
    if not isinstance(solution, dict) or not isinstance(solution.get("rosters"), list) or len(solution["rosters"]) != N:
        return _result(False, f"solution.rosters must list one roster per nurse ({N})")
    rosters = []
    for n, r in enumerate(solution["rosters"]):
        if not isinstance(r, list) or len(r) != D or not all(integer_value(s) and -1 <= int(s) <= 2 for s in r):
            return _result(False, f"rosters[{n}] must have {D} entries in -1..2")
        r = [int(s) for s in r]
        issue = roster_issue(d, n, r)
        if issue:
            return _result(False, f"rosters[{n}]: {issue}")
        rosters.append(r)
    return _result(True, "ok", evaluate(d, rosters))
