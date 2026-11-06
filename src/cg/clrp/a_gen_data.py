"""Capacitated location-routing (CLRP): small instances built like the Prodhon (Prins et al.) benchmark.

Model. m candidate depots (coordinates, capacity W_d, opening cost O_d) and n customers (coordinates, demand q_i);
identical vehicles of capacity Q with a fixed cost F per route, unlimited in number. Every customer is visited once
by a route that starts and ends at the same open depot; the demand served from a depot is at most its capacity.
Travel cost between two points is ceil(100 * Euclidean distance). Objective: opening costs + F per route + travel.

Literature source. The 30 CLRP instances of C. Prins, C. Prodhon, R. Wolfler Calvo, "Solving the capacitated
location-routing problem by a GRASP complemented by a learning process and a path relinking", 4OR 4(3):221-238,
2006 (also C. Prodhon, PhD thesis, UTT 2006), distributed at http://prodhonc.free.fr/Instances/instances_us.htm
(files `coord{n}-{m}-{code}[b].dat`, format in http://prodhonc.free.fr/Instances/Format.txt). The same set is
described in C. Prins, C. Prodhon, A. Ruiz, P. Soriano, R. Wolfler Calvo, "Solving the capacitated location-routing
problem by a cooperative Lagrangean relaxation-granular tabu search heuristic", Transportation Science 41(4):470-483,
2007, Section 5.1. We could not open the 2006 papers or the thesis (no open-access copy, 2026-11-05); the 2007
Section 5.1 statements below were reported by the Codex review of 2026-11-05 and not re-read by us. Each rule is
marked [V] verified from a primary source or the published files, [S] stated in the 2007 Section 5.1 (via the
review), or [I] inferred from the statistics of the 30 published files ("Prodhon-inspired"; read-only script).
- [V] File layout, F = 1000 per route in every file, Q in {70 ("a" files), 150 ("b" files)}.
- [V][S] Travel cost ceil(100 * Euclidean distance) ([S]: "multiplied by 100 and rounded up"). Format.txt says the
  scaled distance is truncated, but the published optimum of coord20-5-1 (54 793) is reproduced by our exact column
  generation only with rounding up (54 769 with truncation); 20-5-1b (39 104) and 20-5-2 (48 908) also match.
- [V] Sizes n in {20, 50, 100, 200}, m in {5, 10}; integer coordinates in [1, 50] (none at 0 in the 2 880 customer
  records of the 30 files).
- [S][I] Customers are uniform or clustered; [S] names the cluster counts {0, 2, 3} (0 = uniform). The last index of
  a file name ("code", 1, 2 or 3) matches them: code-1 files are uniform on [1, 50]^2 (per-axis sd 14.4 = 50 /
  sqrt 12), code-2 and code-3 files put the customers in equal shares around fixed centres, (12.5, 12.5) and
  (37.5, 37.5) for two clusters, and (12.5, 12.5), (12.5, 37.5), (37.5, 25) for three, roughly uniform in 25 x 25
  squares (per-axis sd about 7.2 = 25 / sqrt 12; 5-15 % of the points lie slightly outside). [I] We draw clustered
  customers uniformly in those squares (the third square of three is [26, 50] x [13, 37]) and give cluster k the
  customers i with i mod (number of clusters) = k.
- [I] Depots uniform on [1, 50]^2 in every class (depots lie outside the clusters in several clustered files).
- [S][I] Demands: [S] says uniform on [10, 20]; the files hold 11-20 except one 10 (coord200-10-1 and its b copy),
  which fits ceil of a continuous U(10, 20] better than integers U{10..20} (which would give about 9 % tens). We
  draw integers U{11..20} (adaptation: the files' range).
- [S][I] Depot capacities: [S] says they are set so that at least two or three depots must be opened. [I] The files
  realise this with heterogeneous capacities (multiples of Q in the 20-customer files, of 70 in larger ones). We use
  k uniform in {2, 3} and W_d = ceil(D / k * U[1.0, 1.2]) independently per depot (D = total demand), so k - 1
  depots never suffice ((k - 1) * 1.2 / k < 1) and k may (Prodhon-inspired; the multiplier range is ours).
- [I] Opening costs uniform integers in [5 000, 15 000] (the 20- and 50-customer files range over 5 029-14 703).
Adaptations (named): sizes shrink to n in {12, 14, 16} with m = 5 (Prodhon's smallest files have n = 20, m = 5).
Size calibration (round 2, 2026-11-05; development seeds >= 100 only; full-process exact column generation with the
60 s budget, Threads = 1): the reference's tail grows steeply with n under Q = 150 (Q = 70 seeds stay below 5 s at
n = 20). Maximum over the Q = 150 pilot seeds: n = 12 0.5 s, n = 14 4.8 s, n = 16 12.0 s (10 seeds), n = 17 50.9 s
(10 seeds), n = 18 above 55 s with one seed unproved at 60 s, n = 20 one seed failed (its route tables also need
about m * 2^n * n * 16 bytes, so n <= 20 at m = 5). The compact MIP (1800 s limit, MIPGap 0) proved n = 16 pilots
in 1-71 s. Earlier, 60 s-MIP calibration (round 1) had used n in {8, 10, 12}. The class follows a balanced schedule by seed: cluster count
0 / 2 / 3 for seed mod 3 = 0 / 1 / 2, and Q = 70 for even seeds, 150 for odd seeds; feasibility rule: the depot
capacities are redrawn until first-fit decreasing packs the customer demands into the depots (unlimited vehicles
make every such packing routable, since q_i <= Q).
The four published 20-customer files are parser and reference fixtures only (tests, notes), never evaluation
instances.

Node numbering inside the solvers: depots 0..m-1, then customer i (0..n-1) as node m + i.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, cast

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


SIZE_CONFIG = {
    "small": {"n_customers": 12, "n_depots": 5},
    "medium": {"n_customers": 14, "n_depots": 5},
    "large": {"n_customers": 16, "n_depots": 5},
}

DEFAULT_CONFIG = {
    "grid": (1, 50),
    "demand_range": (11, 20),
    "vehicle_capacities": (70, 150),
    "vehicle_cost": 1000,
    "opening_cost_range": (5000, 15000),
    "depots_needed": (2, 3),
    "capacity_slack": (1.0, 1.2),
    # Cluster squares (x_lo, x_hi, y_lo, y_hi) per cluster count; 0 clusters = uniform on the whole grid.
    "clusters": {
        0: [(1, 50, 1, 50)],
        2: [(1, 25, 1, 25), (26, 50, 26, 50)],
        3: [(1, 25, 1, 25), (1, 25, 26, 50), (26, 50, 13, 37)],
    },
    "max_capacity_draws": 1000,
}


@dataclass
class CLRPInstance(JsonDataMixin):
    instance_size: str
    n_customers: int
    n_depots: int
    vehicle_capacity: int
    vehicle_cost: int
    depot_x: list[int]
    depot_y: list[int]
    depot_capacity: list[int]
    opening_cost: list[int]
    customer_x: list[int]
    customer_y: list[int]
    demand: list[int]

    def __post_init__(self) -> None:
        for name in ("depot_x", "depot_y", "depot_capacity", "opening_cost"):
            if len(getattr(self, name)) != self.n_depots:
                raise ValueError(f"{name} must have n_depots entries")
        for name in ("customer_x", "customer_y", "demand"):
            if len(getattr(self, name)) != self.n_customers:
                raise ValueError(f"{name} must have n_customers entries")

    @classmethod
    def from_json_file(cls, path: str | Path) -> CLRPInstance:
        return cast("CLRPInstance", super().from_json_file(path))

    def point(self, u: int) -> tuple[float, float]:
        m = self.n_depots
        return (self.depot_x[u], self.depot_y[u]) if u < m else (self.customer_x[u - m], self.customer_y[u - m])

    def cost(self, u: int, v: int) -> int:
        """Travel cost between solver nodes u and v (depots first): ceil(100 * Euclidean distance)."""
        return math.ceil(100.0 * math.dist(self.point(u), self.point(v)) - 1e-9)

    def route_cost(self, depot: int, customers: tuple[int, ...] | list[int]) -> int:
        """Vehicle cost plus travel cost of one route; customers are 0..n-1 in visiting order."""
        nodes = [depot, *(self.n_depots + c for c in customers), depot]
        return self.vehicle_cost + sum(self.cost(a, b) for a, b in zip(nodes, nodes[1:]))


def packs(demand: list[int], capacity: list[int]) -> bool:
    """First-fit decreasing: True if the demands fit into the depot capacities (a sufficient feasibility test)."""
    left = list(capacity)
    for q in sorted(demand, reverse=True):
        k = next((k for k, w in enumerate(left) if w >= q), None)
        if k is None:
            return False
        left[k] -= q
    return True


def read_prodhon(path: str | Path, size_label: str = "fixture") -> CLRPInstance:
    """Parse a published Prodhon file (Format.txt layout) into a CLRPInstance (integer-cost files only)."""
    t = Path(path).read_text().split()
    n, m = int(t[0]), int(t[1])
    vals = [float(v) for v in t[2:]]
    k = 0
    take = lambda c: [vals[k + j] for j in range(c)]
    depots = take(2 * m); k += 2 * m
    customers = take(2 * n); k += 2 * n
    Q = vals[k]; k += 1
    W = take(m); k += m
    q = take(n); k += n
    O = take(m); k += m
    F, flag = vals[k], vals[k + 1]
    if flag != 0:
        raise ValueError("only integer-cost (code 0) Prodhon files are supported")
    i = lambda v: int(round(v))
    return CLRPInstance(size_label, n, m, i(Q), i(F), [i(v) for v in depots[0::2]], [i(v) for v in depots[1::2]],
                        [i(v) for v in W], [i(v) for v in O], [i(v) for v in customers[0::2]], [i(v) for v in customers[1::2]],
                        [i(v) for v in q])


class CLRPGenerator(BaseInstanceGenerator[CLRPInstance]):
    def generate_instance(self, *, seed: int | None = None, size_label: str = "medium") -> CLRPInstance:
        if size_label not in SIZE_CONFIG:
            raise ValueError(f"Unknown size_label '{size_label}'. Expected one of {list(SIZE_CONFIG)}.")
        cfg = DEFAULT_CONFIG
        seed = 0 if seed is None else int(seed)
        rng = random.Random(f"clrp-{size_label}-{seed}")
        n, m = SIZE_CONFIG[size_label]["n_customers"], SIZE_CONFIG[size_label]["n_depots"]
        lo, hi = cfg["grid"]
        clusters = (0, 2, 3)[seed % 3]
        Q = cfg["vehicle_capacities"][seed % 2]
        dx, dy = [rng.randint(lo, hi) for _ in range(m)], [rng.randint(lo, hi) for _ in range(m)]
        squares = cfg["clusters"][clusters]
        cx, cy = [], []
        for i in range(n):
            x0, x1, y0, y1 = squares[i % len(squares)]
            cx.append(rng.randint(x0, x1))
            cy.append(rng.randint(y0, y1))
        demand = [rng.randint(*cfg["demand_range"]) for _ in range(n)]
        total = sum(demand)
        k = rng.randint(*cfg["depots_needed"])
        for _ in range(cfg["max_capacity_draws"]):
            capacity = [math.ceil(total / k * rng.uniform(*cfg["capacity_slack"])) for _ in range(m)]
            if packs(demand, capacity):
                break
        else:
            raise RuntimeError(f"no packable depot capacities for {size_label} seed {seed}")
        opening = [rng.randint(*cfg["opening_cost_range"]) for _ in range(m)]
        return CLRPInstance(size_label, n, m, Q, cfg["vehicle_cost"], dx, dy, capacity, opening, cx, cy, demand)


def generate_clrp_instance(seed: int, size: str = "medium", output_dir: str | Path | None = None) -> dict[str, Any]:
    inst = CLRPGenerator().generate_instance(seed=seed, size_label=size)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return inst.to_dict()
