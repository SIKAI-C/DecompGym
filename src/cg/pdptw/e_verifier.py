"""Independent verifier for PDPTW route plans (Li & Lim instances in PyVRP's VRPLIB format, scale-up study).

Standard library only; it reads the raw .vrp file and imports no solver code.

Output contract:
    {"status": "optimal" | "feasible", "method": str, "objective_value": int,
     "routes": [[<node>, ...], ...]}
Nodes are numbered 0..N with 0 the depot (file id - 1, the numbering of PyVRP's .sol files); each route lists its
client nodes in visiting order, without the depot.

Checks (PyVRP's `exact` convention: distances round(1000 x Euclidean), travel time = distance, time windows and
service times x 1000):
- every client node 1..N appears in exactly one route; each route is a nonempty list of client nodes;
- pairing and precedence: a pickup and its delivery are on the same route, the pickup first;
- capacity: the load (+q at a pickup, -q at a delivery, starting empty) never exceeds the capacity;
- time windows: the vehicle leaves the depot at the depot's ready time, may wait, starts service at node j no later
  than due_j, and is back at the depot no later than the depot's due time;
- fleet: at most VEHICLES routes;
- the claimed objective equals the recomputed total distance exactly (integral).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

SCALE = 1000


def read_instance(path: str | Path) -> dict[str, Any]:
    header: dict[str, str] = {}
    sections: dict[str, list[list[str]]] = {}
    current = None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "EOF":
            break
        if line.endswith("_SECTION"):
            current = line
            sections[current] = []
        elif current is None:
            key, _, value = line.partition(":")
            header[key.strip().upper()] = value.strip()
        else:
            sections[current].append(line.split())
    dim = int(header["DIMENSION"])
    coords = sections["NODE_COORD_SECTION"]
    rows = sections["PICKUP_AND_DELIVERY_SECTION"]
    if len(coords) != dim or len(rows) != dim:
        raise ValueError(f"{path}: expected {dim} coordinate and pickup-delivery rows")
    depots = [int(r[0]) for r in sections.get("DEPOT_SECTION", []) if int(r[0]) != -1]
    if depots != [1]:
        raise ValueError(f"{path}: exactly one depot (node 1) expected")
    xy, rec = {}, {}
    for r in coords:
        xy[int(r[0]) - 1] = (float(r[1]), float(r[2]))
    for r in rows:
        rec[int(r[0]) - 1] = [int(v) for v in r[1:]]
    if sorted(xy) != list(range(dim)) or sorted(rec) != list(range(dim)):
        raise ValueError(f"{path}: node ids must be 1..{dim}")
    d: dict[str, Any] = {"name": header.get("NAME", Path(path).stem), "N": dim - 1, "Q": int(header["CAPACITY"]),
                         "K": int(header["VEHICLES"]) if "VEHICLES" in header else dim - 1,
                         "xy": [xy[i] for i in range(dim)]}
    d["demand"] = [rec[i][0] for i in range(dim)]
    d["ready"] = [SCALE * rec[i][1] for i in range(dim)]
    d["due"] = [SCALE * rec[i][2] for i in range(dim)]
    d["service"] = [SCALE * rec[i][3] for i in range(dim)]
    # Partner node in 0-based numbering; pickup_of[j] for a delivery j, delivery_of[i] for a pickup i.
    d["delivery_of"] = {i: rec[i][5] - 1 for i in range(1, dim) if rec[i][5] > 0}
    d["pickup_of"] = {j: rec[j][4] - 1 for j in range(1, dim) if rec[j][4] > 0}
    for i, j in d["delivery_of"].items():
        if d["pickup_of"].get(j) != i or d["demand"][i] <= 0 or d["demand"][j] != -d["demand"][i]:
            raise ValueError(f"{path}: inconsistent request ({i}, {j})")
    if len(d["delivery_of"]) + len(d["pickup_of"]) != dim - 1:
        raise ValueError(f"{path}: every client must be a pickup or a delivery")
    return d


def distance(d: dict[str, Any], a: int, b: int) -> int:
    (xa, ya), (xb, yb) = d["xy"][a], d["xy"][b]
    dx, dy = xa - xb, ya - yb
    return int(round(1000.0 * math.sqrt(dx * dx + dy * dy)))


def route_issue(d: dict[str, Any], route: list[int]) -> tuple[int | None, str]:
    """(distance of the route, "") if it is feasible on its own, else (None, reason)."""
    pos = {v: k for k, v in enumerate(route)}
    for v in route:
        if v in d["delivery_of"]:
            mate = d["delivery_of"][v]
            if mate not in pos:
                return None, f"pickup {v} without its delivery {mate}"
            if pos[mate] < pos[v]:
                return None, f"delivery {mate} before pickup {v}"
        elif d["pickup_of"][v] not in pos:
            return None, f"delivery {v} without its pickup {d['pickup_of'][v]}"
    t, load, prev, total = d["ready"][0], 0, 0, 0
    for v in route + [0]:
        step = distance(d, prev, v)
        total += step
        t = max(t + d["service"][prev] + step, d["ready"][v])
        if t > d["due"][v]:
            return None, f"time window of node {v} violated (arrival {t} > due {d['due'][v]})"
        load += d["demand"][v]
        if load > d["Q"] or load < 0:
            return None, f"load {load} out of range at node {v}"
        prev = v
    return total, ""


def check_routes(d: dict[str, Any], routes: Any) -> tuple[int, list[str]]:
    issues: list[str] = []
    if not isinstance(routes, list):
        return 0, ["routes missing or not a list"]
    N = d["N"]
    seen = [0] * (N + 1)
    total = 0
    for k, route in enumerate(routes):
        if not isinstance(route, list) or not route or not all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= N for v in route):
            issues.append(f"route {k}: must be a nonempty list of client nodes 1..{N}")
            continue
        for v in route:
            seen[v] += 1
        if len(set(route)) != len(route):
            issues.append(f"route {k}: repeats a node")
            continue
        cost, why = route_issue(d, route)
        if cost is None:
            issues.append(f"route {k}: {why}")
            continue
        total += cost
    issues += [f"node {v} visited {seen[v]} times" for v in range(1, N + 1) if seen[v] != 1][:20]
    if len(routes) > d["K"]:
        issues.append(f"{len(routes)} routes exceed the fleet of {d['K']} vehicles")
    return total, issues


def verify_solution(data_path: str | Path, output_path: str | Path, tol: float | None = None) -> dict[str, Any]:
    _ = tol  # the objective is integral: exact agreement is required
    d = read_instance(data_path)
    try:
        out = json.loads(Path(output_path).read_text())
    except (OSError, json.JSONDecodeError) as err:
        return {"ok": False, "issues": [f"unreadable output: {err}"]}
    if not isinstance(out, dict):
        return {"ok": False, "issues": ["output must be a JSON object"]}
    total, issues = check_routes(d, out.get("routes"))
    claimed = out.get("objective_value")
    if isinstance(claimed, bool) or not isinstance(claimed, (int, float)) or not math.isfinite(claimed) or claimed != total:
        issues.append(f"objective mismatch or non-finite: claimed {claimed}, recomputed {total}")
    if out.get("status") not in ("optimal", "feasible"):
        issues.append("status must be 'optimal' or 'feasible' for a returned plan")
    if not isinstance(out.get("method"), str) or not out.get("method"):
        issues.append("method missing")
    return {"ok": not issues, "issues": issues, "objective_value": float(total),
            "routes": len(out["routes"]) if isinstance(out.get("routes"), list) else None}


def read_sol(path: str | Path) -> tuple[list[list[int]], int | None]:
    """Routes and cost of a PyVRP .sol file ("Route #k: i j ...", then "Cost c"); nodes in 0..N numbering."""
    routes, cost = [], None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("Route"):
            routes.append([int(v) for v in line.split(":", 1)[1].split()])
        elif line.startswith("Cost"):
            cost = int(float(line.split()[1]))
    return routes, cost


def verify_sol_file(data_path: str | Path, sol_path: str | Path) -> dict[str, Any]:
    d = read_instance(data_path)
    routes, claimed = read_sol(sol_path)
    total, issues = check_routes(d, routes)
    if claimed != total:
        issues.append(f"cost mismatch: file {claimed}, recomputed {total}")
    return {"ok": not issues, "issues": issues, "objective_value": total, "claimed": claimed, "routes": len(routes), "fleet": d["K"]}
