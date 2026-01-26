"""
Verifier for LRP solutions based on the prompt contract.
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
    route_depot = data.get("route_depot", [])
    incidence = data.get("incidence", [])
    max_vehicles = data.get("max_vehicles", [0.0] * m)

    y = out.get("y")
    lam = out.get("lambda")

    issues: List[str] = []
    if y is None or lam is None:
        return {"ok": False, "issues": ["y or lambda missing in output.json"]}
    if len(y) != m:
        issues.append(f"len(y)={len(y)} != m={m}")
    if len(lam) != len(route_depot):
        issues.append("lambda length mismatch with route_depot")
    if issues:
        return {"ok": False, "issues": issues}

    for j, val in enumerate(y):
        if float(val) < -tol or float(val) > 1 + tol:
            issues.append(f"y[{j}] not in [0,1]")

    for r, val in enumerate(lam):
        if float(val) < -tol:
            issues.append(f"lambda[{r}] negative")

    for i in range(n):
        cov = 0.0
        for r, lam_r in enumerate(lam):
            cov += float(incidence[i][r]) * float(lam_r)
        if abs(cov - 1.0) > tol:
            issues.append(f"customer {i} coverage sum {cov} != 1")

    for j in range(m):
        fleet = 0.0
        for r, depot in enumerate(route_depot):
            if int(depot) == j:
                fleet += float(lam[r])
        rhs = float(max_vehicles[j]) * float(y[j])
        if fleet - rhs > tol:
            issues.append(f"fleet constraint violated at depot {j}: {fleet} > {rhs}")

    return {"ok": len(issues) == 0, "issues": issues}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify LRP output.json feasibility.")
    parser.add_argument("--data", required=True, help="Path to data.json")
    parser.add_argument("--output", required=True, help="Path to output.json")
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance for equality/inequality checks")
    args = parser.parse_args()

    res = verify_solution(args.data, args.output, tol=args.tol)
    if res["ok"]:
        print("OK: solution satisfies coverage and fleet constraints.")
    else:
        print("INVALID:")
        for msg in res["issues"]:
            print(f" - {msg}")
