"""Independent checker for VRPTW outputs: a list of routes, each a customer sequence from and to the depot."""
from __future__ import annotations

import argparse
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


def route_check(d: dict[str, Any], route: list[int]) -> tuple[str | None, float]:
    """Return (issue, length) for one route given as customers only (depot implied at both ends).

    Times are compared exactly in integer tenths: distances are floor(10 * euclidean) / 10 and the
    windows and service times lie on the 0.1 grid. The vehicle leaves the depot at its opening time.
    """
    if sum(d["demand"][c] for c in route) > d["capacity"]:
        return "capacity exceeded", 0.0
    to10 = lambda v: int(round(10 * v))
    dist10 = lambda i, j: math.floor(10.0 * math.hypot(d["x"][i] - d["x"][j], d["y"][i] - d["y"][j]))
    t, prev, length10 = to10(d["ready"][0]), 0, 0
    for c in route + [0]:
        length10 += dist10(prev, c)
        t = max(t + (to10(d["service"][prev]) if prev else 0) + dist10(prev, c), to10(d["ready"][c]))
        if t > to10(d["due"][c]):
            return f"time window of node {c} violated", 0.0
        prev = c
    return None, length10 / 10.0


def verify_solution(data: Any, output: Any, tol: float | None = None) -> dict[str, Any]:
    _ = tol
    try:
        d, output = _load(data), _load(output)
        n = int(d["n_customers"])
    except Exception as exc:
        return _result(False, f"data load error: {exc}")
    solution = output.get("solution")
    if not isinstance(solution, dict) or not isinstance(solution.get("routes"), list):
        return _result(False, "solution.routes missing or not a list")
    seen: set[int] = set()
    total = 0.0
    for r, route in enumerate(solution["routes"]):
        if not isinstance(route, list) or not route or not all(integer_value(c) for c in route):
            return _result(False, f"routes[{r}] must be a non-empty list of customer indices")
        route = [int(c) for c in route]
        if any(not 1 <= c <= n for c in route):
            return _result(False, f"routes[{r}] has an index outside 1..{n}")
        if seen & set(route) or len(set(route)) != len(route):
            return _result(False, f"routes[{r}] repeats a customer")
        seen |= set(route)
        issue, length = route_check(d, route)
        if issue:
            return _result(False, f"routes[{r}]: {issue}")
        total += length
    if len(seen) != n:
        return _result(False, f"{n - len(seen)} customer(s) not served")
    return _result(True, "ok", round(total, 6))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a VRPTW solution.")
    parser.add_argument("--dir", type=Path, default=Path("."))
    parser.add_argument("--output", default="output.json")
    args = parser.parse_args()
    print(json.dumps(verify_solution(args.dir / "data.json", args.dir / args.output), indent=2))


if __name__ == "__main__":
    main()
