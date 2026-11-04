"""Independent verifier for k-median solutions (kmedian-xl-v1 JSON instances, scale-up study; standard library only).

Output contract: {"status": "optimal" | "feasible", "method": str, "objective_value": number,
                  "y": [0/1 per facility], optionally "assign": [open facility per customer]}.
Checks: y is a 0/1 list of length m with exactly k ones; the design value sum_j min_{i open} D_ij is recomputed
here from the coordinates with D_ij = r(s * sqrt(dx * dx + dy * dy)) (r = floor or identity, as documented in
a_data_xl; the same IEEE operations in the same order, so the values are bit-identical to the solver side), and the
claimed objective must match it within tol * max(1, |value|). If an assignment is given, every customer must be
assigned to an open facility; its cost is reported (the score is the design value: each customer at its cheapest
open facility, the paper's semantics). The instance file is parsed here, independently of the solver code.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def read_instance(path: str | Path) -> dict[str, Any]:
    obj = json.loads(Path(path).read_text())
    if obj.get("format") != "kmedian-xl-v1":
        raise ValueError("not a kmedian-xl-v1 instance")
    fac = [(float(a), float(b)) for a, b in obj["facilities"]]
    cus = [(float(a), float(b)) for a, b in obj["customers"]]
    k = int(obj["k"])
    rounding = obj.get("rounding", "floor")
    scale = float(obj.get("scale", 1.0))
    if rounding not in ("floor", "none") or not (math.isfinite(scale) and scale > 0) or not (1 <= k <= len(fac)):
        raise ValueError("invalid instance header")
    if not all(math.isfinite(x) for p in fac + cus for x in p):
        raise ValueError("non-finite coordinates")
    return {"fac": fac, "cus": cus, "k": k, "rounding": rounding, "scale": scale, "m": len(fac), "n": len(cus)}


def distance(inst: dict[str, Any], i: int, j: int) -> float:
    xf, yf = inst["fac"][i]
    xc, yc = inst["cus"][j]
    dx = xf - xc
    dy = yf - yc
    d = math.sqrt(dx * dx + dy * dy)
    if inst["scale"] != 1.0:
        d = inst["scale"] * d
    if inst["rounding"] == "floor":
        d = float(math.floor(d))
    return d


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def design_value(inst: dict[str, Any], open_fac: list[int]) -> tuple[float, list[int]]:
    """(sum_j min_{i open} D_ij, the cheapest open facility per customer; ties: lowest index)."""
    costs, best = [], []
    for j in range(inst["n"]):
        bd, bi = math.inf, -1
        for i in open_fac:
            d = distance(inst, i, j)
            if d < bd:
                bd, bi = d, i
        costs.append(bd)
        best.append(bi)
    return math.fsum(costs), best


def verify_solution(data_path: str | Path, output_path: str | Path, tol: float = 1e-6) -> dict[str, Any]:
    inst = read_instance(data_path)
    m, n, k = inst["m"], inst["n"], inst["k"]
    try:
        out = json.loads(Path(output_path).read_text())
    except (OSError, ValueError) as exc:
        return {"ok": False, "issues": [f"unreadable output: {exc}"]}
    if not isinstance(out, dict):
        return {"ok": False, "issues": ["output must be a JSON object"]}
    y = out.get("y")
    if not isinstance(y, list) or len(y) != m or not all(_is_num(v) and v in (0, 1) for v in y):
        return {"ok": False, "issues": [f"y must be a list of {m} values in {{0, 1}}"]}
    open_fac = [i for i in range(m) if y[i] == 1]
    if len(open_fac) != k:
        return {"ok": False, "issues": [f"{len(open_fac)} facilities open, exactly k={k} required"]}
    total, _ = design_value(inst, open_fac)
    issues: list[str] = []
    result: dict[str, Any] = {"objective_value": total, "open": open_fac}
    a = out.get("assign")
    if a is not None:
        opened = set(open_fac)
        if not isinstance(a, list) or len(a) != n or not all(isinstance(i, int) and not isinstance(i, bool) for i in a):
            issues.append(f"assign must be a list of {n} facility indices")
        elif any(i not in opened for i in a):
            issues.append("a customer is assigned to a closed facility")
        else:
            result["assign_cost"] = math.fsum(distance(inst, a[j], j) for j in range(n))
    claimed = out.get("objective_value")
    if not _is_num(claimed) or not math.isfinite(claimed) or abs(claimed - total) > tol * max(1.0, abs(total)):
        issues.append(f"objective mismatch or non-finite: claimed {claimed}, recomputed {total}")
    if out.get("status") not in ("optimal", "feasible"):
        issues.append("status must be 'optimal' or 'feasible' for a returned design")
    if not isinstance(out.get("method"), str) or not out.get("method"):
        issues.append("method missing")
    return {"ok": not issues, "issues": issues, **result}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Verify a k-median (XL) output.json")
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()
    res = verify_solution(a.data, a.output, a.tol)
    print(json.dumps({k: v for k, v in res.items() if k != "open"}))
