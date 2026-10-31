"""Pickup and delivery with time windows (PDPTW): Li & Lim instances in PyVRP's VRPLIB format (scale-up study).

Source: the PyVRP Instances repository (PDPTW/, Li100/ .. Li800/; converted from SINTEF). Only the data files are read.
File layout: "KEY: value" header lines (NAME, TYPE = PDPTW, DIMENSION, VEHICLES, CAPACITY, EDGE_WEIGHT_TYPE = EUC_2D);
NODE_COORD_SECTION ("id x y", ids 1..DIMENSION, id 1 the depot); PICKUP_AND_DELIVERY_SECTION ("id demand ready due
service pickup delivery": a pickup has pickup = 0 and delivery = the file id of its delivery node, a delivery has
delivery = 0 and pickup = the file id of its pickup node; demand +q at a pickup and -q at its delivery);
DEPOT_SECTION ("1", "-1"); EOF.

Conventions (the repository README: PyVRP's `exact` rounding, distance-only objective; reproduced exactly on all 354
bundled .sol files, whose costs are PyVRP's best found under this objective, not literature best-known solutions):
- nodes are renumbered 0..N (file id - 1), node 0 the depot; the .sol files use this numbering;
- distance d_ij = round(1000 x Euclidean distance) (nearest integer); travel time = distance; time windows and service
  times are multiplied by 1000, so all arithmetic is integral;
- a route leaves the depot at the depot's ready time, may wait, starts service at node j no later than due_j, and
  returns to the depot no later than the depot's due time;
- load starts at 0, +q at a pickup, -q at its delivery, never above the capacity; a request's pickup and delivery are
  on the same route, the pickup first;
- at most VEHICLES routes (never binding for the bundled solutions: at most 19 of 25 routes on Li100);
- objective: total distance (integral, in units of 1/1000).
Rounding to the nearest integer breaks the triangle inequality by at most one unit; `insertion_slack` and
`skip_slack` quantify this for the algorithms that rely on it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCALE = 1000


@dataclass
class PDPTWInstance:
    name: str
    coords: np.ndarray        # (N + 1, 2); row 0 is the depot
    demand: np.ndarray        # (N + 1,) +q at pickups, -q at deliveries, 0 at the depot
    ready: np.ndarray         # (N + 1,) scaled by 1000
    due: np.ndarray           # (N + 1,) scaled by 1000
    service: np.ndarray       # (N + 1,) scaled by 1000
    capacity: int
    vehicles: int
    pickups: list[int] = field(default_factory=list)     # pickup node of request r (requests ordered by pickup node)
    deliveries: list[int] = field(default_factory=list)  # delivery node of request r

    @property
    def N(self) -> int:
        return len(self.coords) - 1

    @property
    def n(self) -> int:
        return len(self.pickups)

    def __post_init__(self) -> None:
        N = len(self.coords) - 1
        self.request_of = [-1] * (N + 1)
        self.is_pickup = [False] * (N + 1)
        for r, (p, d) in enumerate(zip(self.pickups, self.deliveries)):
            self.request_of[p] = self.request_of[d] = r
            self.is_pickup[p] = True
        self.q = [int(self.demand[p]) for p in self.pickups]

    def partner(self, v: int) -> int:
        r = self.request_of[v]
        return self.deliveries[r] if self.is_pickup[v] else self.pickups[r]

    def distances(self) -> np.ndarray:
        """(N + 1) x (N + 1) integer distances round(1000 x Euclidean), depot first (= travel times)."""
        p = self.coords.astype(float)
        dx = p[:, None, 0] - p[None, :, 0]
        dy = p[:, None, 1] - p[None, :, 1]
        return np.rint(1000.0 * np.sqrt(dx * dx + dy * dy)).astype(np.int64)


def load_pdptw(path: str | Path) -> PDPTWInstance:
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
    if header.get("TYPE", "PDPTW").upper() != "PDPTW":
        raise ValueError(f"{path}: not a PDPTW instance")
    coords = np.zeros((dim, 2))
    rec = np.zeros((dim, 6), dtype=np.int64)
    seen_c, seen_r = set(), set()
    for r in sections["NODE_COORD_SECTION"]:
        i = int(r[0]) - 1
        coords[i] = (float(r[1]), float(r[2]))
        seen_c.add(i)
    for r in sections["PICKUP_AND_DELIVERY_SECTION"]:
        i = int(r[0]) - 1
        rec[i] = [int(v) for v in r[1:7]]
        seen_r.add(i)
    if seen_c != set(range(dim)) or seen_r != set(range(dim)):
        raise ValueError(f"{path}: node ids must be 1..{dim}")
    depots = [int(r[0]) for r in sections.get("DEPOT_SECTION", []) if int(r[0]) != -1]
    if depots != [1]:
        raise ValueError(f"{path}: exactly one depot (node 1) expected")
    pickups, deliveries = [], []
    for i in range(1, dim):
        if rec[i, 5] > 0:  # a pickup: its delivery's file id
            j = int(rec[i, 5]) - 1
            if not (1 <= j < dim) or int(rec[j, 4]) - 1 != i or rec[i, 0] <= 0 or rec[j, 0] != -rec[i, 0]:
                raise ValueError(f"{path}: inconsistent request ({i}, {j})")
            pickups.append(i)
            deliveries.append(j)
        elif rec[i, 4] <= 0:
            raise ValueError(f"{path}: node {i} is neither a pickup nor a delivery")
    if 2 * len(pickups) != dim - 1:
        raise ValueError(f"{path}: every client must belong to exactly one request")
    return PDPTWInstance(name=header.get("NAME", Path(path).stem), coords=coords, demand=rec[:, 0].copy(),
                         ready=SCALE * rec[:, 1], due=SCALE * rec[:, 2], service=SCALE * rec[:, 3],
                         capacity=int(header["CAPACITY"]), vehicles=int(header.get("VEHICLES", dim - 1)),
                         pickups=pickups, deliveries=deliveries)


def read_sol(path: str | Path) -> tuple[list[list[int]], int | None]:
    """Routes (0..N numbering, depot omitted) and cost of a PyVRP .sol file."""
    routes, cost = [], None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("Route"):
            routes.append([int(v) for v in line.split(":", 1)[1].split()])
        elif line.startswith("Cost"):
            cost = int(float(line.split()[1]))
    return routes, cost


# ---------------------------------------------------------------- evaluation (shared by every method)

def route_check(inst: PDPTWInstance, dist, route: list[int]) -> tuple[int | None, str]:
    """(distance, "") of a feasible route (client nodes in visiting order, depot omitted), else (None, reason)."""
    pos = {v: k for k, v in enumerate(route)}
    if len(pos) != len(route):
        return None, "repeated node"
    N = inst.N
    for v in route:
        if not 1 <= v <= N:
            return None, f"node {v} is not a client"
        w = inst.partner(v)
        if w not in pos:
            return None, f"node {v} without its partner {w}"
        if inst.is_pickup[v] and pos[w] < pos[v]:
            return None, f"delivery {w} before pickup {v}"
    e, l, s, dem = inst.ready, inst.due, inst.service, inst.demand
    t, load, prev, total = int(e[0]), 0, 0, 0
    for v in route + [0]:
        step = int(dist[prev][v])
        total += step
        t = max(t + int(s[prev]) + step, int(e[v]))
        if t > l[v]:
            return None, f"time window of node {v} violated"
        load += int(dem[v])
        if load > inst.capacity:
            return None, f"capacity exceeded at node {v}"
        prev = v
    return total, ""


def plan_cost(inst: PDPTWInstance, dist, routes: list[list[int]]) -> int | None:
    """Total distance of a route plan, or None if a route is infeasible, a client is not served exactly once, or the
    plan uses more routes than the fleet."""
    if len(routes) > inst.vehicles:
        return None
    served = sorted(v for r in routes for v in r)
    if served != list(range(1, inst.N + 1)):
        return None
    total = 0
    for r in routes:
        if not r:
            return None
        c, _ = route_check(inst, dist, list(r))
        if c is None:
            return None
        total += c
    return total


def insertion_slack(inst: PDPTWInstance, dist: np.ndarray) -> np.ndarray:
    """delta[u] = max over nodes i != w of (t_iw - t_iu - s_u - t_uw)^+: by how much visiting u between i and w can make
    the arrival at w earlier than the direct arc (rounding breaks the triangle inequality by at most one unit, so it is
    nonzero only for nodes with zero service time)."""
    D = dist.astype(np.int64)
    s = inst.service.astype(np.int64)
    out = np.zeros(len(D), dtype=np.int64)
    for u in range(len(D)):
        if s[u] >= 1:  # violations are at most one unit
            continue
        viol = D - D[:, u][:, None] - s[u] - D[u, :][None, :]
        np.fill_diagonal(viol, 0)
        viol[u, :] = 0
        viol[:, u] = 0
        out[u] = max(0, int(viol.max()))
    return out


def skip_slack(dist: np.ndarray, nodes: list[int] | None = None) -> np.ndarray:
    """gamma[u] = max over nodes i != w of (c_iw - c_iu - c_uw)^+: the most that skipping u can add to a path's cost
    (computed for `nodes`, all nodes by default; zero elsewhere)."""
    D = dist.astype(np.int64)
    out = np.zeros(len(D), dtype=np.int64)
    for u in (range(len(D)) if nodes is None else nodes):
        viol = D - D[:, u][:, None] - D[u, :][None, :]
        np.fill_diagonal(viol, 0)
        viol[u, :] = 0
        viol[:, u] = 0
        out[u] = max(0, int(viol.max()))
    return out


def feasible_arcs(inst: PDPTWInstance, dist: np.ndarray) -> np.ndarray:
    """Boolean (N + 1) x (N + 1) matrix of arcs that some feasible route may use (node 0 is the depot at both ends).

    Removed: arcs into a delivery from the depot, from a pickup to the depot, from a delivery to its own pickup;
    direct time-window violations; pairs whose loads exceed the capacity; and the pairing tests of Dumas, Desrosiers
    and Soumis (1991), which check the short paths an arc forces (for (p_a, p_b): 0-p_a-p_b-d_a-d_b-0 and
    0-p_a-p_b-d_b-d_a-0; for (p_a, d_b): 0-p_b-p_a-d_b-d_a-0; for (d_a, p_b): 0-p_a-d_a-p_b-d_b-0; for (d_a, d_b):
    0-p_a-p_b-d_a-d_b-0 and 0-p_b-p_a-d_a-d_b-0). Inserting other nodes can only delay a path except by the rounding
    slack of zero-service nodes, so a path test fails only if it is late by more than the total insertion slack."""
    N = inst.N
    D = dist.astype(np.int64)
    e, l, s = inst.ready.astype(np.int64), inst.due.astype(np.int64), inst.service.astype(np.int64)
    Q = inst.capacity
    slack = int(insertion_slack(inst, dist).sum())
    ok = np.ones((N + 1, N + 1), dtype=bool)
    np.fill_diagonal(ok, False)
    for r, (p, d) in enumerate(zip(inst.pickups, inst.deliveries)):
        ok[0, d] = False
        ok[p, 0] = False
        ok[d, p] = False
    # Direct time windows (exact for a single arc).
    ok &= (e[:, None] + s[:, None] + D) <= l[None, :]
    # Return to the depot from the earliest possible start (nodes visited before can make it earlier only by the slack).
    ok[:, 0] &= (np.maximum(e, e[0] + D[0, :] - slack) + s + D[:, 0]) <= l[0]

    def late(path: list[int]) -> bool:
        t, load = int(e[0]), 0
        prev = 0
        for v in path + [0]:
            t = max(t + int(s[prev]) + int(D[prev, v]), int(e[v]))
            if t > l[v] + slack:
                return True
            load += int(inst.demand[v])
            if load > Q:
                return True
            prev = v
        return False

    P, Dl = inst.pickups, inst.deliveries
    n = inst.n
    for a in range(n):
        pa, da = P[a], Dl[a]
        for b in range(n):
            if a == b:
                continue
            pb, db = P[b], Dl[b]
            if ok[pa, pb] and late([pa, pb, da, db]) and late([pa, pb, db, da]):
                ok[pa, pb] = False
            if ok[pa, db] and late([pb, pa, db, da]):
                ok[pa, db] = False
            if ok[da, pb] and late([pa, da, pb, db]):
                ok[da, pb] = False
            if ok[da, db] and late([pa, pb, da, db]) and late([pb, pa, da, db]):
                ok[da, db] = False
    return ok
