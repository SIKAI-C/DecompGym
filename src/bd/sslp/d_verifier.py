"""Independent checker for SSLP outputs: openings plus every scenario's client assignment.

Overflow is not read from the output: given the openings and assignments, the cheapest feasible
overflow is max(0, load_j - u * y_j), which the checker computes itself.
"""
from __future__ import annotations

import argparse
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


def evaluate(data: dict[str, Any], open_sites: set[int], assignment: list[dict[int, int]]) -> float:
    total = float(sum(data["open_cost"][j] for j in open_sites))
    for s, plan in enumerate(assignment):
        load = [0] * data["n_sites"]
        revenue = 0
        for i, j in plan.items():
            load[j] += data["demand"][i][j]
            revenue += data["revenue"][i][j]
        overflow = sum(max(0, load[j] - (data["capacity"] if j in open_sites else 0)) for j in range(data["n_sites"]))
        total += data["probability"][s] * (data["overflow_penalty"] * overflow - revenue)
    return total


def verify_solution(data: Any, output: Any, tol: float | None = None) -> dict[str, Any]:
    _ = tol
    try:
        data, output = _load(data), _load(output)
        J, I, S = data["n_sites"], data["n_clients"], data["n_scenarios"]
    except Exception as exc:
        return _result(False, f"data load error: {exc}")
    solution = output.get("solution")
    if not isinstance(solution, dict):
        return _result(False, "solution must be an object")
    sites = solution.get("open_sites")
    if not isinstance(sites, list) or not all(integer_value(j) and 0 <= int(j) < J for j in sites):
        return _result(False, "solution.open_sites must be a list of site indices")
    open_sites = {int(j) for j in sites}
    if len(open_sites) != len(sites):
        return _result(False, "solution.open_sites has duplicates")
    if len(open_sites) > data["max_open"]:
        return _result(False, "more sites open than max_open")
    scenarios = solution.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != S:
        return _result(False, f"solution.scenarios must list all {S} scenarios")
    assignment: list[dict[int, int]] = [{} for _ in range(S)]
    seen = set()
    for idx, sc in enumerate(scenarios):
        if not isinstance(sc, dict) or not integer_value(sc.get("scenario")) or not 0 <= int(sc["scenario"]) < S:
            return _result(False, f"scenarios[{idx}] needs a valid scenario index")
        s = int(sc["scenario"])
        if s in seen:
            return _result(False, f"scenario {s} listed twice")
        seen.add(s)
        rows = sc.get("assignments")
        if not isinstance(rows, list):
            return _result(False, f"scenario {s}: assignments must be a list")
        for row in rows:
            if not isinstance(row, dict) or not integer_value(row.get("client")) or not integer_value(row.get("site")):
                return _result(False, f"scenario {s}: each assignment needs integer client and site")
            i, j = int(row["client"]), int(row["site"])
            if not 0 <= i < I or not 0 <= j < J:
                return _result(False, f"scenario {s}: index out of range")
            if not data["presence"][s][i]:
                return _result(False, f"scenario {s}: client {i} is absent but assigned")
            if i in assignment[s]:
                return _result(False, f"scenario {s}: client {i} assigned twice")
            assignment[s][i] = j
        missing = [i for i in range(I) if data["presence"][s][i] and i not in assignment[s]]
        if missing:
            return _result(False, f"scenario {s}: present clients {missing[:5]} not assigned")
    return _result(True, "ok", evaluate(data, open_sites, assignment))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an SSLP solution.")
    parser.add_argument("--dir", type=Path, default=Path("."))
    parser.add_argument("--output", default="output.json")
    args = parser.parse_args()
    print(json.dumps(verify_solution(args.dir / "data.json", args.dir / args.output), indent=2))


if __name__ == "__main__":
    main()
