"""Independent checker for CLRP outputs: routes {"depot": d, "customers": [i, ...]} from and to their depot.

Checks coverage (each customer exactly once), vehicle capacity per route and depot capacity per depot,
and recomputes the objective: opening costs of the depots used + per route the vehicle cost and
ceil(100 * Euclidean distance) per arc.
"""
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


def verify_solution(data: Any, output: Any, tol: float | None = None) -> dict[str, Any]:
    _ = tol
    try:
        d, output = _load(data), _load(output)
        n, m = int(d["n_customers"]), int(d["n_depots"])
    except Exception as exc:
        return _result(False, f"data load error: {exc}")
    solution = output.get("solution")
    if not isinstance(solution, dict) or not isinstance(solution.get("routes"), list):
        return _result(False, "solution.routes missing or not a list")
    pts = [(d["depot_x"][k], d["depot_y"][k]) for k in range(m)] + [(d["customer_x"][i], d["customer_y"][i]) for i in range(n)]
    cost = lambda a, b: math.ceil(100.0 * math.dist(pts[a], pts[b]) - 1e-9)
    seen: set[int] = set()
    load = [0] * m
    used: set[int] = set()
    total = 0
    for r, route in enumerate(solution["routes"]):
        if not isinstance(route, dict):
            return _result(False, f"routes[{r}] must be an object with depot and customers")
        depot, customers = route.get("depot"), route.get("customers")
        if not integer_value(depot) or not 0 <= int(depot) < m:
            return _result(False, f"routes[{r}].depot must be a depot index in 0..{m - 1}")
        if not isinstance(customers, list) or not customers or not all(integer_value(c) for c in customers):
            return _result(False, f"routes[{r}].customers must be a non-empty list of customer indices")
        depot, customers = int(depot), [int(c) for c in customers]
        if any(not 0 <= c < n for c in customers):
            return _result(False, f"routes[{r}] has a customer index outside 0..{n - 1}")
        if seen & set(customers) or len(set(customers)) != len(customers):
            return _result(False, f"routes[{r}] repeats a customer")
        seen |= set(customers)
        demand = sum(d["demand"][c] for c in customers)
        if demand > d["vehicle_capacity"]:
            return _result(False, f"routes[{r}]: vehicle capacity exceeded")
        load[depot] += demand
        used.add(depot)
        nodes = [depot, *(m + c for c in customers), depot]
        total += d["vehicle_cost"] + sum(cost(a, b) for a, b in zip(nodes, nodes[1:]))
    if len(seen) != n:
        return _result(False, f"{n - len(seen)} customer(s) not served")
    for k in range(m):
        if load[k] > d["depot_capacity"][k]:
            return _result(False, f"depot {k} capacity exceeded")
    total += sum(d["opening_cost"][k] for k in used)
    return _result(True, "ok", float(total))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a CLRP solution.")
    parser.add_argument("--dir", type=Path, default=Path("."))
    parser.add_argument("--output", default="output.json")
    args = parser.parse_args()
    print(json.dumps(verify_solution(args.dir / "data.json", args.dir / args.output), indent=2))


if __name__ == "__main__":
    main()
