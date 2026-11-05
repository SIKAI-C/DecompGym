"""Independent verifier for CVRP route plans (CVRPLIB X instances, scale-up study). Standard library only.

Output contract:
    {"status": "optimal" | "feasible", "method": str, "objective_value": number, "routes": [[customer, ...], ...]}
with customers numbered 1..n as in the CVRPLIB .sol files (the non-depot nodes of the .vrp file in file order; in the X
instances the depot is node 1, so customer i is node i + 1), each route in visiting order, the depot implicit at both
ends. Checks, from the .vrp text only: every route is a nonempty list of customer numbers; its load (sum of demands)
is at most CAPACITY; every customer is visited exactly once; the objective (sum over routes of the Euclidean distances
rounded to the nearest integer, TSPLIB nint = floor(d + 0.5)) matches the claim within tol * max(1, |objective|); the
claim is a finite number; status is "optimal" or "feasible" and method is a string.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def read_vrp(path: str | Path) -> dict[str, Any]:
    header: dict[str, str] = {}
    sec: dict[str, list[list[str]]] = {}
    cur = None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "EOF":
            break
        first = line.split()[0].rstrip(":").upper()
        if first.endswith("_SECTION"):
            cur = first
            sec[cur] = []
        elif cur is None and ":" in line:
            k, v = line.split(":", 1)
            header[k.strip().upper()] = v.strip()
        elif cur is not None:
            sec[cur].append(line.split())
        else:
            raise ValueError(f"unexpected line {line!r}")
    if header.get("EDGE_WEIGHT_TYPE", "").upper() != "EUC_2D":
        raise ValueError("only EUC_2D instances are supported")
    coords = {int(r[0]): (float(r[1]), float(r[2])) for r in sec["NODE_COORD_SECTION"]}
    demand = {int(r[0]): int(r[1]) for r in sec["DEMAND_SECTION"]}
    depots = [int(r[0]) for r in sec.get("DEPOT_SECTION", [["1"]]) if int(r[0]) != -1]
    if len(depots) != 1:
        raise ValueError("exactly one depot is supported")
    order = [depots[0]] + [v for v in sorted(coords) if v != depots[0]]
    return {"pts": [coords[v] for v in order], "q": [demand[v] for v in order], "Q": int(header["CAPACITY"]),
            "n": len(order) - 1}


def verify_solution(data_path: str | Path, output_path: str | Path, tol: float = 1e-6) -> dict[str, Any]:
    d = read_vrp(data_path)
    try:
        out = json.loads(Path(output_path).read_text())
    except (OSError, json.JSONDecodeError) as err:
        return {"ok": False, "issues": [f"unreadable output: {err}"]}
    if not isinstance(out, dict):
        return {"ok": False, "issues": ["output must be a JSON object"]}
    routes = out.get("routes")
    if not isinstance(routes, list):
        return {"ok": False, "issues": ["routes missing"]}
    n, pts, q, Q = d["n"], d["pts"], d["q"], d["Q"]
    dist = lambda a, b: math.floor(math.dist(pts[a], pts[b]) + 0.5)
    issues: list[str] = []
    seen = [0] * (n + 1)
    total = 0
    for k, r in enumerate(routes):
        if not isinstance(r, list) or not r or not all(isinstance(c, int) and not isinstance(c, bool) and 1 <= c <= n for c in r):
            issues.append(f"route {k}: must be a nonempty list of customer numbers in 1..{n}")
            continue
        load = sum(q[c] for c in r)
        if load > Q:
            issues.append(f"route {k}: load {load} exceeds capacity {Q}")
        for c in r:
            seen[c] += 1
        seq = [0] + r + [0]
        total += sum(dist(a, b) for a, b in zip(seq, seq[1:]))
    wrong = [c for c in range(1, n + 1) if seen[c] != 1]
    if wrong:
        issues.append(f"{len(wrong)} customers not visited exactly once (e.g. {wrong[:5]})")
    claimed = out.get("objective_value")
    if isinstance(claimed, bool) or not isinstance(claimed, (int, float)) or not math.isfinite(claimed) or abs(claimed - total) > tol * max(1.0, abs(total)):
        issues.append(f"objective mismatch or non-finite: claimed {claimed}, recomputed {total}")
    if out.get("status") not in ("optimal", "feasible"):
        issues.append("status must be 'optimal' or 'feasible' for a returned plan")
    if not isinstance(out.get("method"), str):
        issues.append("method missing")
    return {"ok": not issues, "issues": issues, "objective_value": float(total), "routes": len(routes)}
