"""Independent verifier for CLRP route plans (Prodhon instances, scale-up study).

Output contract: {"status", "method", "objective_value", "routes": [{"depot": d, "customers": [i, ...]}]}
with depots 0..m-1 and customers 0..n-1 in visiting order. Checks coverage (each customer exactly once),
vehicle capacity per route, depot capacity per depot, and recomputes the objective: opening costs of the
depots used + ceil(100 x Euclidean distance) per arc + the vehicle cost per route.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _read(path: str | Path) -> dict[str, Any]:
    tok = [float(t) for t in Path(path).read_text().split()]
    n, m = int(tok[0]), int(tok[1])
    p = 2
    depots = [tok[p + 2 * k: p + 2 * k + 2] for k in range(m)]; p += 2 * m
    custs = [tok[p + 2 * k: p + 2 * k + 2] for k in range(n)]; p += 2 * n
    Q = tok[p]; p += 1
    W = tok[p:p + m]; p += m
    q = tok[p:p + n]; p += n
    O = tok[p:p + m]; p += m
    return {"n": n, "m": m, "pts": depots + custs, "Q": Q, "W": W, "q": q, "O": O, "F": tok[p]}


def verify_solution(data_path: str | Path, output_path: str | Path, tol: float = 1e-6) -> dict[str, Any]:
    d = _read(data_path)
    out = json.loads(Path(output_path).read_text())
    issues: list[str] = []
    routes = out.get("routes")
    if not isinstance(routes, list):
        return {"ok": False, "issues": ["routes missing"]}
    n, m = d["n"], d["m"]
    seen = [0] * n
    load = [0.0] * m
    used = set()
    total = 0.0
    dist = lambda a, b: math.ceil(100.0 * math.dist(d["pts"][a], d["pts"][b]) - 1e-9)
    for k, r in enumerate(routes):
        dep, cs = r.get("depot"), r.get("customers")
        if not isinstance(dep, int) or not 0 <= dep < m or not isinstance(cs, list) or not cs or not all(isinstance(c, int) and 0 <= c < n for c in cs):
            issues.append(f"route {k}: invalid depot or customers")
            continue
        for c in cs:
            seen[c] += 1
        demand = sum(d["q"][c] for c in cs)
        if demand > d["Q"] + tol:
            issues.append(f"route {k}: load {demand} exceeds vehicle capacity {d['Q']}")
        load[dep] += demand
        used.add(dep)
        seq = [dep] + [m + c for c in cs] + [dep]
        total += sum(dist(a, b) for a, b in zip(seq, seq[1:])) + d["F"]
    issues += [f"customer {c} visited {s} times" for c, s in enumerate(seen) if s != 1]
    issues += [f"depot {k} load {load[k]} exceeds capacity {d['W'][k]}" for k in range(m) if load[k] > d["W"][k] + tol]
    total += sum(d["O"][k] for k in used)
    claimed = out.get("objective_value")
    if not isinstance(claimed, (int, float)) or abs(claimed - total) > tol * max(1.0, abs(total)):
        issues.append(f"objective mismatch: claimed {claimed}, recomputed {total}")
    return {"ok": not issues, "issues": issues, "objective_value": total}
