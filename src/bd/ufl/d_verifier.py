"""
Simple verifier for UFL solutions produced per b_problem.py contract.

Checks:
- For every customer j: sum_i x_ij == 1 (within tolerance)
- For every i, j: x_ij <= y_i + tol

Assumes:
- data.json provides m, n
- output.json provides x (m x n list) and y (length m list)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def verify_solution(data_path: str | Path, output_path: str | Path, tol: float = 1e-6) -> Dict[str, Any]:
    """
    Verify feasibility of x, y in output.json against basic UFL constraints.
    Returns dict with ok: bool and issues: list[str].
    """
    data = json.loads(Path(data_path).read_text())
    out = json.loads(Path(output_path).read_text())

    m = int(data.get("m", 0))
    n = int(data.get("n", 0))

    y = out.get("y")
    x = out.get("x")

    issues: List[str] = []

    if y is None or x is None:
        return {"ok": False, "issues": ["x or y missing in output.json"]}

    # Shape checks
    if len(y) != m:
        issues.append(f"len(y)={len(y)} != m={m}")
    if len(x) != m:
        issues.append(f"len(x)={len(x)} != m={m}")
    else:
        for i, row in enumerate(x):
            if len(row) != n:
                issues.append(f"len(x[{i}])={len(row)} != n={n}")

    # Early exit if shapes off
    if issues:
        return {"ok": False, "issues": issues}

    # Constraint: sum_i x_ij == 1 for every j
    for j in range(n):
        col_sum = sum(float(x[i][j]) for i in range(m))
        if abs(col_sum - 1.0) > tol:
            issues.append(f"sum_i x[{i}][{j}] = {col_sum} (expected 1) for customer {j}")

    # Constraint: x_ij <= y_i
    for i in range(m):
        yi = float(y[i])
        for j in range(n):
            xij = float(x[i][j])
            if xij - yi > tol:
                issues.append(f"x[{i}][{j}]={xij} > y[{i}]={yi}")

    return {"ok": len(issues) == 0, "issues": issues}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify UFL output.json feasibility.")
    parser.add_argument("--data", required=True, help="Path to data.json")
    parser.add_argument("--output", required=True, help="Path to output.json")
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance for equality/inequality checks")
    args = parser.parse_args()

    res = verify_solution(args.data, args.output, tol=args.tol)
    if res["ok"]:
        print("OK: solution satisfies basic constraints.")
    else:
        print("INVALID:")
        for msg in res["issues"]:
            print(f" - {msg}")

