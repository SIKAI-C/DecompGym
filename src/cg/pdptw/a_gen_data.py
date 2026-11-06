"""Pickup and delivery with time windows (PDPTW): the Li & Lim (2001) construction on random Solomon subsets.

Source: H. Li and A. Lim, "A metaheuristic for the pickup and delivery problem with time windows", Proc. 13th IEEE
ICTAI, 2001, pp. 160-167 (instances: https://www.sintef.no/projectweb/top/pdptw/li-lim-benchmark/). The benchmark is
built from Solomon's (1987) VRPTW instances by solving each VRPTW heuristically and pairing customers at random
inside each route of that solution (Li & Lim 2001, Section 5.1: random pairing within the routes obtained by the
Li-Lim-Huang VRPTW heuristic; also arXiv:2511.07681). Comparing every local Li & Lim 100-customer file with its
Solomon parent (e.g. lr101 vs r101; all 56 files) shows these observed properties:
- every Solomon customer keeps its coordinates, ready time, due date and service time; depot, capacity and the
  25-vehicle fleet are the parent's;
- in each pair the pickup keeps its own Solomon demand q and the delivery gets -q (its own demand is discarded);
- a route with an odd number of customers leaves one customer unpaired: it becomes a pickup with service time 0, and
  a new delivery node is added at the same location with the same window, the customer's original service time and
  demand -q (lr101 has 6 such nodes, 106 tasks; lr201 has 2, 102 tasks).
The files do not contain the generating routes, so the visit order behind each pair's direction is not observable.
Inferred reconstruction (our choice): the pickup is the customer visited first on the heuristic route, which keeps
that route feasible (every published pair passes a depot-pickup-delivery-depot timing check). This generator applies
the construction to a random subset of a Solomon file:
- parent file and customer subset: the frozen VRPTW sampling protocol (`src.cg.vrptw.a_gen_data`: slot
  k = 3 * size_index + seed takes class (R1, R2, C1, C2, RC1, RC2)[k mod 6], a uniform file of the class and a
  uniform subset of n_customers of its 100 customers), with the stream "lilim-subset/<size>/<seed>";
- heuristic VRPTW solution (named substitution for the Li-Lim-Huang heuristic): Solomon's sequential insertion
  heuristic I1 (Solomon 1987; mu = 1, lambda = 1, alpha1 = 0.5, routes seeded by the farthest unrouted customer; `src.cg.vrptw.heuristic.basic_initializer`),
  evaluated under this family's distance convention below (Li & Lim's own heuristic solutions are not published);
- pairing: each route's customers are shuffled by the instance stream and paired consecutively (a uniform random
  matching); with an odd count the last shuffled customer is the unpaired one;
- numbering: requests are ordered by their pickup's parent customer number; request k has pickup node k and
  delivery node n + k (n = number of requests), node 0 = depot.
The heuristic routes stay feasible after pairing (generation asserts it), so every instance is feasible.

Frozen convention, identical to the scale-up PDPTW (`src.cg.pdptw.a_data`, PyVRP's VRPLIB conversion of Li & Lim):
- distance = travel time = round(1000 x Euclidean distance) (nearest integer; Solomon coordinates are integers, so a
  tie never occurs and the value is computed exactly with an integer square root); ready, due and service times are
  the parent's integers multiplied by 1000, so data.json stores them in the same unit and all arithmetic is integral;
- a route leaves the depot at its ready time, waits when early, starts service no later than due, and returns by the
  depot's due time; load starts at 0, never exceeds the capacity; pickup before delivery on the same route;
- fleet: at most the parent's 25 vehicles; a plan never has more routes than requests and the requests here are far
  fewer than 25, so the limit cannot bind (generation asserts it) and the model states an unlimited fleet;
- objective: total distance only (SINTEF ranks vehicles first; its best-known values are not optima here).
Size adaptation: Li & Lim's smallest set has 100 customers (about 53 requests); n_customers is shrunk and depends on
the parent's horizon type (type 1: 12 / 15 / 18, type 2: 12 / 15 / 18 customers), calibrated on pilot seeds 100+
(never the evaluation seeds 0-2) so that the branch-and-price reference proves optimality within its 60 s budget and
the monolithic MIP proves optimality within 1800 s, both with one thread. The number of requests is (n_customers + number of odd
heuristic routes) / 2, so it varies slightly with the instance.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, cast

import numpy as np

from src.cg.vrptw.solomon import CLASSES, CLASS_FILES, load_solomon
from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


SIZES = ("small", "medium", "large")
# Keyed by the parent's horizon type (calibrated on pilot seeds 100+, full-process seconds). Type 2: the reference is
# the limit (c204 times out at 20 customers, 17.0 s at 18; r204/r208/rc204/rc208 <= 30.4 s at 20). Type 1: the
# reference stays <= 10.2 s up to 28 customers, but the MIP of C1 parents with loose windows is the limit: c104
# exceeds the 1800 s MIP at 20 customers (2 of 3 pilots); at 18 every pilot MIP proves in <= 42 s.
SIZE_CONFIG = {
    "small": {"n_customers": {"1": 12, "2": 12}},
    "medium": {"n_customers": {"1": 15, "2": 15}},
    "large": {"n_customers": {"1": 18, "2": 18}},
}
SCALE = 1000


def scaled_distance(dx: int, dy: int) -> int:
    """round(1000 x sqrt(dx^2 + dy^2)) for integers, exactly: floor((floor(2000 d) + 1) / 2)."""
    return (math.isqrt(4 * SCALE * SCALE * (dx * dx + dy * dy)) + 1) // 2


@dataclass
class PDPTWInstance(JsonDataMixin):
    instance_size: str
    n_requests: int
    capacity: int
    x: list[int]
    y: list[int]
    demand: list[int]  # +q at a pickup, -q at its delivery, 0 at the depot
    ready: list[int]  # thousandths of the parent's time unit
    due: list[int]
    service: list[int]

    def __post_init__(self) -> None:
        if len(self.x) != 2 * self.n_requests + 1:
            raise ValueError("expected 2 * n_requests + 1 nodes")
        for name in ("x", "y", "ready", "due", "service"):
            values = getattr(self, name)
            if any(float(v) != int(v) for v in values):
                raise ValueError(f"{name} must be integers")
            setattr(self, name, [int(v) for v in values])

    @classmethod
    def from_json_file(cls, path: str | Path) -> PDPTWInstance:
        return cast("PDPTWInstance", super().from_json_file(path))

    def dist(self, i: int, j: int) -> int:
        return scaled_distance(self.x[i] - self.x[j], self.y[i] - self.y[j])


class PDPTWGenerator(BaseInstanceGenerator[PDPTWInstance]):
    def generate_instance(self, *, seed: int | None = None, size_label: str = "medium") -> PDPTWInstance:
        if size_label not in SIZE_CONFIG:
            raise ValueError(f"Unknown size_label '{size_label}'. Expected one of {list(SIZE_CONFIG)}.")
        return build_instance(int(seed or 0), size_label)[0]


def build_instance(seed: int, size_label: str) -> tuple[PDPTWInstance, dict[str, Any]]:
    from src.cg.pdptw.d_verifier import route_issue
    from src.cg.vrptw.a_data import VRPTWInstance as GridInstance
    from src.cg.vrptw.heuristic import basic_initializer

    stream = f"lilim-subset/{size_label}/{seed}"
    rng = random.Random(stream)
    cls = CLASSES[(3 * SIZES.index(size_label) + seed) % len(CLASSES)]
    parent = rng.choice(CLASS_FILES[cls])
    sol = load_solomon(parent)
    cfg = SIZE_CONFIG[size_label]["n_customers"]
    m = cfg if isinstance(cfg, int) else cfg[parent[-3]]  # keyed by horizon type: r101 -> "1", rc208 -> "2"
    customers = sorted(rng.sample(range(1, 101), m))
    rows = [sol.rows[0]] + [sol.rows[c] for c in customers]  # (x, y, q, ready, due, service)
    pts = [(r[0], r[1]) for r in rows]
    dist = np.array([[scaled_distance(a[0] - b[0], a[1] - b[1]) for b in pts] for a in pts], dtype=np.int64)
    col = lambda k, scale=1: np.array([r[k] * scale for r in rows], dtype=np.int64)
    grid = GridInstance(parent, np.array(pts, dtype=float), col(2), col(3, SCALE), col(4, SCALE), col(5, SCALE), sol.capacity, sol.vehicles, dist)
    routes = basic_initializer(grid)  # subset positions 1..m
    pairs: list[tuple[int, int | None]] = []  # (pickup, delivery) as subset positions; None = duplicated location
    for route in routes:
        order = list(route)
        rng.shuffle(order)
        for a, b in zip(order[0::2], order[1::2]):
            pairs.append((a, b) if route.index(a) < route.index(b) else (b, a))
        if len(order) % 2:
            pairs.append((order[-1], None))
    pairs.sort(key=lambda p: p[0])
    n = len(pairs)
    if n > sol.vehicles:
        raise ValueError("the fleet limit could bind; the unlimited-fleet model would differ from the parent's")
    node = {"x": [pts[0][0]], "y": [pts[0][1]], "demand": [0], "ready": [rows[0][3]], "due": [rows[0][4]], "service": [rows[0][5]]}
    deliveries = []
    for p, d in pairs:
        r = rows[p]
        for key, value in zip(node, (r[0], r[1], r[2], r[3], r[4], r[5] if d is not None else 0)):
            node[key].append(value)
        r_d = rows[d] if d is not None else r  # the duplicate copies the pickup's location, window and service time
        deliveries.append((r_d[0], r_d[1], -r[2], r_d[3], r_d[4], r_d[5]))
    for row in deliveries:
        for key, value in zip(node, row):
            node[key].append(value)
    inst = PDPTWInstance(size_label, n, sol.capacity, node["x"], node["y"], node["demand"],
                         *([SCALE * v for v in node[key]] for key in ("ready", "due", "service")))
    # The paired heuristic routes are a feasible plan (the construction keeps their visiting order).
    new_id = {}
    for k, (p, d) in enumerate(pairs, start=1):
        new_id[p] = k
        new_id[("dup", p) if d is None else d] = n + k
    data = inst.to_dict()
    for route in routes:
        seq = []
        for c in route:
            seq.append(new_id[c])
            if ("dup", c) in new_id:
                seq.append(new_id[("dup", c)])
        if route_issue(data, seq) is not None:
            raise RuntimeError(f"{parent}: a paired heuristic route is infeasible ({route_issue(data, seq)})")
    source = {"recipe": "Li & Lim (2001) pairing on a Solomon (1987) random subset", "parent": parent, "parent_class": cls,
              "parent_customers": customers, "parent_vehicles": sol.vehicles, "heuristic_routes": len(routes),
              "duplicated_deliveries": sum(d is None for _, d in pairs),
              "requests": [[customers[p - 1], customers[d - 1] if d is not None else None] for p, d in pairs], "stream": stream}
    return inst, source


def generate_pdptw_instance(seed: int, size: str = "medium", output_dir: str | Path | None = None) -> dict[str, Any]:
    inst, source = build_instance(seed, size)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
        # Provenance stays out of data.json (the only file the agent sees).
        (out_dir / "source.json").write_text(json.dumps(source, indent=2))
    return inst.to_dict()
