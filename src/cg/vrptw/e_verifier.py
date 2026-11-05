"""Independent verifier for VRPTW route plans (Solomon / Gehring-Homberger files in PyVRP's VRPLIB format; scale-up
study). Standard library only; it re-reads the instance file and shares no code with the solvers.

Output contract:
    {"status": "optimal" | "feasible", "method": str, "objective_value": number, "routes": [[customer, ...], ...]}
with customers numbered 1..n as in the .sol files (the non-depot nodes of the .vrp file in file order; the depot is
node 1 in every bundled file, so customer i is node i + 1), each route in visiting order, the depot implicit at both
ends. Checks (DIMACS convention, everything in exact integer tenths):
- every route is a nonempty list of customer numbers; every customer is visited exactly once; at most VEHICLES routes;
- load (sum of demands) <= CAPACITY on every route;
- time windows: the vehicle leaves the depot at the depot's earliest time; travel time = distance =
  floor(10 x Euclidean distance) / 10 (computed exactly with integer square roots); service starts at
  max(arrival, earliest) and must be <= latest; service takes SERVICE_TIME (or the SERVICE_TIME_SECTION value); the
  vehicle must be back at the depot by the depot's latest time;
- objective: the sum of the truncated distances; the claimed value must equal it on the 0.1 grid (|claim - total| <=
  1e-6), and be a finite number; status must be "optimal" or "feasible" and method a string.
"""
from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any


def _tenths(text: str) -> int:
    v = Fraction(text) * 10
    if v.denominator != 1:
        raise ValueError(f"{text} is not on the 0.1 grid")
    return int(v)


def read_vrptw(path: str | Path) -> dict[str, Any]:
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
    if header.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper() != "EUC_2D":
        raise ValueError("only EUC_2D instances are supported")
    coords = {int(r[0]): (Fraction(r[1]), Fraction(r[2])) for r in sec["NODE_COORD_SECTION"]}
    demand = {int(r[0]): int(r[1]) for r in sec["DEMAND_SECTION"]}
    tw = {int(r[0]): (_tenths(r[1]), _tenths(r[2])) for r in sec["TIME_WINDOW_SECTION"]}
    depots = [int(r[0]) for r in sec.get("DEPOT_SECTION", [["1"]]) if int(r[0]) != -1]
    if len(depots) != 1:
        raise ValueError("exactly one depot is supported")
    order = [depots[0]] + [v for v in sorted(coords) if v != depots[0]]
    if "SERVICE_TIME_SECTION" in sec:
        svc = {int(r[0]): _tenths(r[1]) for r in sec["SERVICE_TIME_SECTION"]}
        service = [svc[v] for v in order]
    else:
        service = [_tenths(header.get("SERVICE_TIME", "0"))] * len(order)
    service[0] = 0
    n = len(order) - 1
    return {"pts": [coords[v] for v in order], "q": [demand[v] for v in order], "Q": int(header["CAPACITY"]),
            "e": [tw[v][0] for v in order], "l": [tw[v][1] for v in order], "s": service, "n": n,
            "K": int(header["VEHICLES"]) if "VEHICLES" in header else n}


def _dist10(a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]) -> int:
    """floor(10 * Euclidean distance), exactly (floor(sqrt(x)) = isqrt(floor(x)) for rational x >= 0)."""
    return math.isqrt(math.floor(100 * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)))


def verify_solution(data_path: str | Path, output_path: str | Path, tol: float = 1e-6) -> dict[str, Any]:
    d = read_vrptw(data_path)
    try:
        out = json.loads(Path(output_path).read_text())
    except (OSError, json.JSONDecodeError) as err:
        return {"ok": False, "issues": [f"unreadable output: {err}"]}
    if not isinstance(out, dict):
        return {"ok": False, "issues": ["output must be a JSON object"]}
    routes = out.get("routes")
    if not isinstance(routes, list):
        return {"ok": False, "issues": ["routes missing"]}
    n, pts, q, Q, e, l, s = d["n"], d["pts"], d["q"], d["Q"], d["e"], d["l"], d["s"]
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
        t, prev = e[0], 0
        for c in r:
            dd = _dist10(pts[prev], pts[c])
            total += dd
            t = max(t + s[prev] + dd, e[c])
            if t > l[c]:
                issues.append(f"route {k}: service at customer {c} starts at {t / 10} after its latest time {l[c] / 10}")
                break
            prev = c
        else:
            dd = _dist10(pts[prev], pts[0])
            total += dd
            if t + s[prev] + dd > l[0]:
                issues.append(f"route {k}: returns to the depot at {(t + s[prev] + dd) / 10} after {l[0] / 10}")
    wrong = [c for c in range(1, n + 1) if seen[c] != 1]
    if wrong:
        issues.append(f"{len(wrong)} customers not visited exactly once (e.g. {wrong[:5]})")
    if len(routes) > d["K"]:
        issues.append(f"{len(routes)} routes exceed the fleet of {d['K']} vehicles")
    claimed = out.get("objective_value")
    value = total / 10
    if isinstance(claimed, bool) or not isinstance(claimed, (int, float)) or not math.isfinite(claimed) or abs(claimed - value) > tol:
        issues.append(f"objective mismatch or non-finite: claimed {claimed}, recomputed {value}")
    if out.get("status") not in ("optimal", "feasible"):
        issues.append("status must be 'optimal' or 'feasible' for a returned plan")
    if not isinstance(out.get("method"), str):
        issues.append("method missing")
    return {"ok": not issues, "issues": issues, "objective_value": value, "routes": len(routes)}
