"""VRPTW instances in PyVRP's VRPLIB format (Solomon 100-customer and Gehring-Homberger sets; scale-up study).

File layout (`outputs/scaleup/raw/cvrp/Instances-main/VRPTW/{Solomon,GH200,GH400,...}/*.vrp`, data only): header lines
"KEY : value" (NAME, TYPE = CVRPTW, DIMENSION, VEHICLES, CAPACITY, SERVICE_TIME, EDGE_WEIGHT_TYPE = EUC_2D), then
NODE_COORD_SECTION (node id, x, y), DEMAND_SECTION (node id, demand), TIME_WINDOW_SECTION (node id, earliest, latest),
DEPOT_SECTION (depot node id, closed by -1) and EOF. Every bundled file has integer coordinates, integer windows, one
depot (node 1), positive demands and one uniform SERVICE_TIME for the customers (the depot has none); a
SERVICE_TIME_SECTION, if present, would give per-node service times instead.

Conventions (the folder's README: DIMACS rounding; reproduced exactly on the bundled best-known solutions, see
tests/test_scaleup_vrptw.py):
- customers are numbered 1..n in file order of the non-depot nodes (customer i is node i + 1 of the file), as in the
  .sol files ("Route #k: i j ...", then "Cost c");
- every distance and every travel duration is the Euclidean distance truncated to one decimal, floor(10 d) / 10;
  everything is kept in integer tenths here (`dist`, `ready`, `due`, `service`), so feasibility tests and costs are
  exact; with integer (or decimal) coordinates, 10 d = sqrt(100 (dx^2 + dy^2)) is truncated exactly by an integer
  square root, never by floating point;
- a route leaves the depot at its earliest time, waits when it arrives before a window opens, must start service at
  each customer no later than the window's end (start = max(arrival, earliest) <= latest), spends the service time,
  and must be back at the depot by the depot's latest time; route load (sum of demands) <= CAPACITY;
- objective: total distance (sum of truncated arc distances), no vehicle cost;
- fleet: at most VEHICLES routes (the header states the limit: 25 for Solomon, 50 for 200-customer GH, 100 for 400;
  every bundled best-known solution uses fewer routes, so the limit is not binding there). Without a VEHICLES line the
  fleet is unlimited (n routes at most).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np


@dataclass
class VRPTWInstance:
    name: str
    coords: np.ndarray    # (n + 1, 2) as read; row 0 is the depot
    demand: np.ndarray    # (n + 1,) integers, demand[0] = 0
    ready: np.ndarray     # (n + 1,) earliest service start, integer tenths
    due: np.ndarray       # (n + 1,) latest service start (depot: latest return), integer tenths
    service: np.ndarray   # (n + 1,) service durations, integer tenths, service[0] = 0
    capacity: int
    vehicles: int         # fleet limit (number of routes)
    dist: np.ndarray = field(repr=False)  # (n + 1, n + 1) truncated distances = travel times, integer tenths

    @property
    def n(self) -> int:
        return len(self.demand) - 1

    @property
    def min_vehicles(self) -> int:
        """ceil(total demand / capacity), a lower bound on the number of routes."""
        return int(math.ceil(int(self.demand.sum()) / self.capacity))


def tenths(value: str | float | int | Fraction) -> int:
    """An exact time or distance on the 0.1 grid, in integer tenths (ValueError if it is off the grid)."""
    v = Fraction(str(value)) * 10 if not isinstance(value, Fraction) else value * 10
    if v.denominator != 1:
        raise ValueError(f"{value} is not on the 0.1 grid")
    return int(v)


def truncated_tenths(a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]) -> int:
    """floor(10 * Euclidean distance) exactly: floor(sqrt(x)) = isqrt(floor(x)) for rational x >= 0."""
    s = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    return math.isqrt(math.floor(100 * s))


def distance_matrix(points: list[tuple[Fraction, Fraction]]) -> np.ndarray:
    m = len(points)
    if all(p[0].denominator == 1 and p[1].denominator == 1 for p in points):
        xy = np.array([[int(p[0]), int(p[1])] for p in points], dtype=np.int64)
        sq = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2) * 100
        # Integer square root, vectorized: a float estimate corrected by +-1 (values here stay far below 2^52).
        r = np.floor(np.sqrt(sq.astype(np.float64))).astype(np.int64)
        r -= (r * r > sq).astype(np.int64)
        r += ((r + 1) * (r + 1) <= sq).astype(np.int64)
        return r
    return np.array([[truncated_tenths(points[i], points[j]) for j in range(m)] for i in range(m)], dtype=np.int64)


def _sections(path: str | Path) -> tuple[dict[str, str], dict[str, list[list[str]]]]:
    header: dict[str, str] = {}
    sections: dict[str, list[list[str]]] = {}
    current = None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "EOF":
            break
        first = line.split()[0].rstrip(":").upper()
        if first.endswith("_SECTION"):
            current = first
            sections[current] = []
            continue
        if current is None and ":" in line:
            k, v = line.split(":", 1)
            header[k.strip().upper()] = v.strip().strip('"')
            continue
        if current is None:
            raise ValueError(f"{path}: unexpected line {line!r}")
        sections[current].append(line.split())
    return header, sections


def load_vrptw(path: str | Path) -> VRPTWInstance:
    """Reads a PyVRP/VRPLIB VRPTW file (EUC_2D, one depot); see the module docstring for numbering and units."""
    header, sec = _sections(path)
    if header.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper() != "EUC_2D":
        raise ValueError(f"{path}: only EDGE_WEIGHT_TYPE EUC_2D is supported")
    dim = int(header["DIMENSION"])
    Q = int(header["CAPACITY"])
    coords = {int(r[0]): (Fraction(r[1]), Fraction(r[2])) for r in sec["NODE_COORD_SECTION"]}
    dem = {int(r[0]): int(r[1]) for r in sec["DEMAND_SECTION"]}
    tw = {int(r[0]): (tenths(r[1]), tenths(r[2])) for r in sec["TIME_WINDOW_SECTION"]}
    depots = [int(r[0]) for r in sec.get("DEPOT_SECTION", [["1"]]) if int(r[0]) != -1]
    if len(depots) != 1:
        raise ValueError(f"{path}: exactly one depot is supported, got {depots}")
    nodes = list(range(1, dim + 1))
    if sorted(coords) != nodes or sorted(dem) != nodes or sorted(tw) != nodes:
        raise ValueError(f"{path}: DIMENSION {dim} does not match the sections")
    dep = depots[0]
    order = [dep] + [v for v in nodes if v != dep]
    if "SERVICE_TIME_SECTION" in sec:
        svc = {int(r[0]): tenths(r[1]) for r in sec["SERVICE_TIME_SECTION"]}
        service = [svc[v] for v in order]
    else:
        s = tenths(header.get("SERVICE_TIME", "0"))
        service = [0] + [s] * (dim - 1)
    service[0] = 0
    q = np.array([dem[v] for v in order], dtype=np.int64)
    if q[0] != 0 or (q[1:] < 0).any():
        raise ValueError(f"{path}: depot demand must be 0 and demands nonnegative")
    vehicles = int(header["VEHICLES"]) if "VEHICLES" in header else dim - 1
    pts = [coords[v] for v in order]
    return VRPTWInstance(
        name=header.get("NAME", Path(path).stem),
        coords=np.array([[float(x), float(y)] for x, y in pts]),
        demand=q,
        ready=np.array([tw[v][0] for v in order], dtype=np.int64),
        due=np.array([tw[v][1] for v in order], dtype=np.int64),
        service=np.array(service, dtype=np.int64),
        capacity=Q,
        vehicles=vehicles,
        dist=distance_matrix(pts),
    )


def load_sol(path: str | Path) -> tuple[list[list[int]], int]:
    """Routes (customer numbers 1..n) and the stated cost, in integer tenths, of a .sol file."""
    routes: list[list[int]] = []
    cost = None
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s.lower().startswith("route"):
            routes.append([int(t) for t in s.split(":", 1)[1].split()])
        elif s.lower().startswith("cost"):
            cost = tenths(s.split()[1])
    if cost is None:
        raise ValueError(f"{path}: no Cost line")
    return routes, cost


# ---------------------------------------------------------------- evaluation (integer tenths)

def route_schedule(inst: VRPTWInstance, route: list[int]) -> tuple[int, int] | None:
    """(distance in tenths, return time) of one route (customers 1..n in visiting order), or None if a time window
    is missed. The vehicle leaves the depot at its earliest time and waits when it arrives early."""
    d, e, l, s = inst.dist, inst.ready, inst.due, inst.service
    t = int(e[0])
    prev = 0
    length = 0
    for c in route:
        length += int(d[prev, c])
        t = max(t + int(s[prev]) + int(d[prev, c]), int(e[c]))
        if t > l[c]:
            return None
        prev = c
    length += int(d[prev, 0])
    t += int(s[prev]) + int(d[prev, 0])
    if t > l[0]:
        return None
    return length, t


def plan_issues(inst: VRPTWInstance, routes: list) -> tuple[int | None, list[str]]:
    """(total distance in tenths, []) for a feasible plan, (None, issues) otherwise: every route a nonempty list of
    customer numbers 1..n, every customer exactly once, loads within capacity, windows met, at most `vehicles` routes."""
    n, Q, q = inst.n, inst.capacity, inst.demand
    issues: list[str] = []
    seen = np.zeros(n + 1, dtype=np.int64)
    total = 0
    if not isinstance(routes, (list, tuple)):
        return None, ["routes must be a list"]
    for k, r in enumerate(routes):
        if not isinstance(r, (list, tuple)) or not r or not all(
                isinstance(i, (int, np.integer)) and not isinstance(i, bool) and 1 <= i <= n for i in r):
            issues.append(f"route {k}: must be a nonempty list of customer numbers in 1..{n}")
            continue
        r = [int(i) for i in r]
        for i in r:
            seen[i] += 1
        load = int(q[r].sum())
        if load > Q:
            issues.append(f"route {k}: load {load} exceeds capacity {Q}")
        sched = route_schedule(inst, r)
        if sched is None:
            issues.append(f"route {k}: a time window is violated")
        else:
            total += sched[0]
    wrong = [c for c in range(1, n + 1) if seen[c] != 1]
    if wrong:
        issues.append(f"{len(wrong)} customers not visited exactly once (e.g. {wrong[:5]})")
    if len(routes) > inst.vehicles:
        issues.append(f"{len(routes)} routes exceed the fleet of {inst.vehicles}")
    return (None, issues) if issues else (total, [])
