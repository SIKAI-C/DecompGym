"""
Verifier for FCTP solutions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def verify_solution(data_path: str | Path, output_path: str | Path, tol: float = 1e-6) -> Dict[str, Any]:
    data = json.loads(Path(data_path).read_text())
    out = json.loads(Path(output_path).read_text())

    m = int(data.get("m", 0))
    n = int(data.get("n", 0))
    supply = [float(v) for v in data.get("supply", [])]
    demand = [float(v) for v in data.get("demand", [])]
    capacity = data.get("capacity", [])

    y = out.get("y")
    x = out.get("x")

    issues: List[str] = []

    if y is None or x is None:
        return {"ok": False, "issues": ["x or y missing in output.json"]}
    if len(y) != m or any(len(row) != n for row in y):
        issues.append("y has wrong shape")
    if len(x) != m or any(len(row) != n for row in x):
        issues.append("x has wrong shape")
    if issues:
        return {"ok": False, "issues": issues}

    for i in range(m):
        total = sum(float(x[i][j]) for j in range(n))
        if total - supply[i] > tol:
            issues.append(f"supply violated at node {i}: shipped {total} > {supply[i]}")

    for j in range(n):
        total = sum(float(x[i][j]) for i in range(m))
        if demand[j] - total > tol:
            issues.append(f"demand not satisfied at node {j}: received {total} < {demand[j]}")

    for i in range(m):
        for j in range(n):
            lhs = float(x[i][j])
            if lhs < -tol:
                issues.append(f"x[{i},{j}] negative")
            cap = float(capacity[i][j]) if capacity and capacity[i] else float("inf")
            if lhs - cap * float(y[i][j]) > tol:
                issues.append(f"capacity/link violation at ({i},{j})")
            if float(y[i][j]) < -tol or float(y[i][j]) > 1 + tol:
                issues.append(f"y[{i},{j}] not in [0,1]")

    return {"ok": len(issues) == 0, "issues": issues}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify FCTP output.json feasibility.")
    parser.add_argument("--data", required=True, help="Path to data.json")
    parser.add_argument("--output", required=True, help="Path to output.json")
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance for equality/inequality checks")
    args = parser.parse_args()

    res = verify_solution(args.data, args.output, tol=args.tol)
    if res["ok"]:
        print("OK: solution satisfies supply/demand and linking constraints.")
    else:
        print("INVALID:")
        for msg in res["issues"]:
            print(f" - {msg}")
