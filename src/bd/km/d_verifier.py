"""
Verifier for k-median solutions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def verify_solution(data_path: str | Path, output_path: str | Path, tol: float = 1e-6) -> dict[str, Any]:
    data = json.loads(Path(data_path).read_text())
    out = json.loads(Path(output_path).read_text())

    m = int(data.get("m", 0))
    n = int(data.get("n", 0))
    k = int(data.get("k", 0))

    y = out.get("y")
    x = out.get("x")

    issues: list[str] = []
    if y is None or x is None:
        return {"ok": False, "issues": ["y or x missing in output.json"]}
    if len(y) != m:
        issues.append(f"len(y)={len(y)} != m={m}")
    if len(x) != m or any(len(row) != n for row in x):
        issues.append("x has wrong shape")
    if issues:
        return {"ok": False, "issues": issues}

    open_total = sum(float(val) for val in y)
    if abs(open_total - k) > max(1.0, k) * tol:
        issues.append(f"sum(y)={open_total} differs from k={k}")

    for j in range(n):
        col = sum(float(x[i][j]) for i in range(m))
        if abs(col - 1.0) > tol:
            issues.append(f"customer {j} assignment sum {col} != 1")

    for i in range(m):
        yi = float(y[i])
        if yi < -tol or yi > 1 + tol:
            issues.append(f"y[{i}] not in [0,1]")
        for j in range(n):
            if float(x[i][j]) - yi > tol:
                issues.append(f"x[{i},{j}] > y[{i}]")
            if float(x[i][j]) < -tol:
                issues.append(f"x[{i},{j}] negative")

    return {"ok": len(issues) == 0, "issues": issues}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify k-median output.json feasibility.")
    parser.add_argument("--data", required=True, help="Path to data.json")
    parser.add_argument("--output", required=True, help="Path to output.json")
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance for equality/inequality checks")
    args = parser.parse_args()

    res = verify_solution(args.data, args.output, tol=args.tol)
    if res["ok"]:
        print("OK: assignments satisfy linking and open exactly k facilities.")
    else:
        print("INVALID:")
        for msg in res["issues"]:
            print(f" - {msg}")
