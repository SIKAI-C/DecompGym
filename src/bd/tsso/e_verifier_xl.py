"""Independent verifier for scale-up TSSO solutions (tsso-xl-v1 instances; standard library only).

Output contract: {"status": "optimal" | "feasible", "method": str, "objective_value": number,
                  "y": [[0/1] * T] * I}.
The solution is the first-stage design y. Its value is sum_it F_it y_it + sum_(i,s) p_s Q_is(y_i) with the paper's
recourse (a_data_xl), which this verifier computes itself, exactly, by a dynamic program over the net position
n_t = inv_t - back_t (no LP solver):
  * for a net position n the cheapest split is inv = n^+, back = n^- (h, b >= 0), costing e_t(n) = h_t n^+ + b_t n^-;
  * producing p >= 0 in period t costs C_t(p) = c_reg_t min(p, K_t y_t) + c_ot_t (p - K_t y_t)^+ (regular first,
    c_reg <= c_ot), a convex piecewise-linear function;
  * F_t(n) = e_t(n) + min_{p >= 0} [C_t(p) + F_t-1(n + d_t - p)],  F_-1 = 0 at n = I0 - B0 only, Q = min_n F_T-1(n).
Every F_t is convex piecewise linear on [x0, +inf); it is stored as (x0, F(x0), segments (length, slope) with
increasing slopes, the last one infinite). The infimal convolution with C_t merges the slope lists (truncated after
the first infinite segment), the shift by d_t moves x0, and adding e_t splits the segment containing 0. The minimum is
read off by walking the segments while the slope is negative. Equivalent to the LP (both are min-cost flows on the
same path network); the test suite compares the two on random designs.
Checks: y binary of shape (I, T); the claimed objective must match the recomputed value within
tol * max(1, |value|); a missing method or an unknown status is rejected. The instance file is parsed here,
independently of the solver code.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

INF = float("inf")


def read_instance(path: str | Path) -> dict[str, Any]:
    obj = json.loads(Path(path).read_text())
    if obj.get("format") not in (None, "tsso-xl-v1"):
        raise ValueError("not a tsso-xl-v1 instance")
    F = [[float(v) for v in row] for row in obj["F"]]
    I, T = len(F), len(F[0])
    arrs = {k: [[float(v) for v in row] for row in obj[k]] for k in ("K", "c_reg", "c_ot", "h", "b")}
    d = [[[float(v) for v in col] for col in prod] for prod in obj["demand"]]
    S = len(d[0][0])
    p = [float(v) for v in obj["prob"]]
    tot = math.fsum(p)
    if len(p) != S or any(v <= 0 for v in p) or tot <= 0:
        raise ValueError("invalid scenario probabilities")
    if abs(tot - 1.0) > 1e-12:
        p = [v / tot for v in p]
    I0 = [float(v) for v in obj.get("I0", [0.0] * I)]
    B0 = [float(v) for v in obj.get("B0", [0.0] * I)]
    for k, a in arrs.items():
        if len(a) != I or any(len(r) != T for r in a):
            raise ValueError(f"{k}: inconsistent dimensions")
    if len(d) != I or any(len(r) != T or any(len(c) != S for c in r) for r in d):
        raise ValueError("demand: inconsistent dimensions")
    return {"I": I, "T": T, "S": S, "F": F, "p": p, "d": d, "I0": I0, "B0": B0, **arrs}


def recourse(c_reg: list, c_ot: list, h: list, b: list, cap: list, d: list, n0: float = 0.0) -> float:
    """Exact optimal recourse cost of one (product, scenario) with regular capacities cap_t = K_t y_t."""
    x0, f0, segs = n0, 0.0, []
    T = len(d)
    for t in range(T):
        x0 -= d[t]
        if c_reg[t] < c_ot[t] and cap[t] > 0:
            add = [(cap[t], c_reg[t]), (INF, c_ot[t])]
        else:
            add = [(INF, min(c_reg[t], c_ot[t]) if cap[t] == INF else c_ot[t])]
        merged = sorted(segs + add, key=lambda s: s[1])
        segs = []
        for L, s in merged:
            segs.append((L, s))
            if L == INF:
                break
        ht, bt = h[t], b[t]
        f0 += ht * max(x0, 0.0) + bt * max(-x0, 0.0)
        out = []
        x = x0
        for L, s in segs:
            if x >= 0:
                out.append((L, s + ht))
            elif L == INF or x + L > 0:
                out.append((-x, s - bt))
                out.append((INF if L == INF else x + L, s + ht))
            else:
                out.append((L, s - bt))
            x = INF if L == INF else x + L
        segs = out
    val = f0
    for L, s in segs:
        if s >= 0:
            break
        if L == INF:
            raise RuntimeError("unbounded recourse (negative final slope)")
        val += s * L
    return val


def design_value(inst: dict[str, Any], y: list) -> float:
    I, T, S = inst["I"], inst["T"], inst["S"]
    total = math.fsum(inst["F"][i][t] * y[i][t] for i in range(I) for t in range(T))
    parts = []
    for i in range(I):
        cap = [inst["K"][i][t] * y[i][t] for t in range(T)]
        args = (inst["c_reg"][i], inst["c_ot"][i], inst["h"][i], inst["b"][i], cap)
        n0 = inst["I0"][i] - inst["B0"][i]
        for s in range(S):
            parts.append(inst["p"][s] * recourse(*args, [inst["d"][i][t][s] for t in range(T)], n0))
    return total + math.fsum(parts)


def verify_solution(data_path: str | Path, output_path: str | Path, tol: float = 1e-6) -> dict[str, Any]:
    inst = read_instance(data_path)
    issues: list[str] = []
    try:
        out = json.loads(Path(output_path).read_text())
    except (OSError, ValueError) as exc:
        return {"ok": False, "issues": [f"unreadable output: {exc}"], "objective_value": None}
    if out.get("status") not in ("optimal", "feasible"):
        issues.append(f"status {out.get('status')!r} is not optimal/feasible")
    if not out.get("method"):
        issues.append("method missing")
    y = out.get("y")
    I, T = inst["I"], inst["T"]
    if not isinstance(y, list) or len(y) != I or any(not isinstance(r, list) or len(r) != T for r in y):
        return {"ok": False, "issues": issues + ["y missing or not of shape (I, T)"], "objective_value": None}
    yy = []
    for row in y:
        r = []
        for v in row:
            try:
                f = float(v)
            except (TypeError, ValueError):
                return {"ok": False, "issues": issues + [f"y entry {v!r} not numeric"], "objective_value": None}
            if f not in (0.0, 1.0):
                return {"ok": False, "issues": issues + [f"y entry {v!r} not binary"], "objective_value": None}
            r.append(int(f))
        yy.append(r)
    value = design_value(inst, yy)
    claimed = out.get("objective_value")
    try:
        claimed = float(claimed)
    except (TypeError, ValueError):
        claimed = None
    if claimed is None or not math.isfinite(claimed):
        issues.append("objective_value missing or not finite")
    elif abs(claimed - value) > tol * max(1.0, abs(value)):
        issues.append(f"objective_value {claimed} differs from the recomputed {value}")
    return {"ok": not issues, "issues": issues, "objective_value": value, "claimed": claimed}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Verify a scale-up TSSO output.json.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()
    print(json.dumps(verify_solution(a.data, a.output, a.tol)))
