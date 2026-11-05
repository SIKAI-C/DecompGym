"""Independent checker for PDPTW outputs: routes of node indices from and back to the depot."""
from __future__ import annotations

import json
import math
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


def route_issue(d: dict[str, Any], route: list[int]) -> str | None:
    """Pairing, precedence, load and exact (integer-tenth) timing of one route; None if feasible."""
    n = int(d["n_requests"])
    dist10 = lambda i, j: math.floor(10.0 * math.hypot(d["x"][i] - d["x"][j], d["y"][i] - d["y"][j]))
    to10 = lambda v: int(round(10 * v))
    pos = {v: k for k, v in enumerate(route)}
    for v in route:
        mate = v + n if v <= n else v - n
        if mate not in pos:
            return f"node {v} without its paired node {mate}"
        if v <= n and pos[v] > pos[mate]:
            return f"delivery {mate} before pickup {v}"
    # Departure at the depot's opening time.
    t, load, prev = to10(d["ready"][0]), 0, 0
    for v in route + [0]:
        t = max(t + (to10(d["service"][prev]) if prev else 0) + dist10(prev, v), to10(d["ready"][v]))
        if t > to10(d["due"][v]):
            return f"time window of node {v} violated"
        load += d["demand"][v]
        if load > d["capacity"] or load < 0:
            return f"load out of range at node {v}"
        prev = v
    return None


def verify_solution(data: Any, output: Any, tol: float | None = None) -> dict[str, Any]:
    _ = tol
    try:
        d, output = _load(data), _load(output)
        n = int(d["n_requests"])
    except Exception as exc:
        return _result(False, f"data load error: {exc}")
    dist10 = lambda i, j: math.floor(10.0 * math.hypot(d["x"][i] - d["x"][j], d["y"][i] - d["y"][j]))
    solution = output.get("solution")
    if not isinstance(solution, dict) or not isinstance(solution.get("routes"), list):
        return _result(False, "solution.routes missing or not a list")
    seen: set[int] = set()
    total10 = 0
    for r, route in enumerate(solution["routes"]):
        if not isinstance(route, list) or not route or not all(integer_value(v) and 1 <= int(v) <= 2 * n for v in route):
            return _result(False, f"routes[{r}] must be a non-empty list of node indices 1..{2 * n}")
        route = [int(v) for v in route]
        if seen & set(route) or len(set(route)) != len(route):
            return _result(False, f"routes[{r}] repeats a node")
        seen |= set(route)
        issue = route_issue(d, route)
        if issue:
            return _result(False, f"routes[{r}]: {issue}")
        total10 += sum(dist10(a, b) for a, b in zip([0] + route, route + [0]))
    if len(seen) != 2 * n:
        return _result(False, f"{2 * n - len(seen)} node(s) not served")
    return _result(True, "ok", total10 / 10.0)
