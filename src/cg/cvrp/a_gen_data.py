"""Capacitated vehicle routing (CVRP): the Uchoa et al. (2017) X-instance generator at small n.

Source: E. Uchoa, D. Pecin, A. Pessoa, M. Poggi, T. Vidal, A. Subramanian, "New benchmark instances for the
Capacitated Vehicle Routing Problem", European Journal of Operational Research 257(3):845-858, 2017, Section 3
(preprint: https://optimization-online.org/wp-content/uploads/2014/10/4597.pdf). Where the paper leaves an
implementation detail open, the CVRPLIB generator script is followed (https://galgos.inf.puc-rio.br/cvrplib/en/xml100,
`generator.py`, which CVRPLIB states is the generator used for the X instances; Queiroga et al. 2022). Attributes:
- grid: depot and customers at distinct integer points of [0, 1000] x [0, 1000] ("superpositions are not allowed");
- depot positioning: central C = (500, 500), eccentric E = (0, 0), random R = a uniform grid point;
- customer positioning: random R (uniform grid points); clustered C: S ~ UD[3, 8] customers are the cluster seeds,
  placed uniformly; every other customer is drawn by accept-reject: a uniform free grid point p is accepted with
  probability sum_s exp(-d(p, s) / 40) / S, so a customer lands on p with probability proportional to the paper's
  attraction sum_s exp(-d(p, s) / 40) (S bounds the attraction everywhere, a valid envelope; the script instead divides
  by the attraction at the strongest seed, which can be exceeded between seeds and then saturates at 1);
  random-clustered RC: floor(n / 2) customers random, the others clustered;
- demand distribution (seven types): U unitary (all 1); 1-10 = UD[1, 10]; 5-10 = UD[5, 10]; 1-100 = UD[1, 100];
  50-100 = UD[50, 100]; Q (quadrant): UD[1, 50] in the even quadrants w.r.t. (500, 500) (x < 500 <= y or y < 500 <= x),
  UD[51, 100] in the odd ones; SL: most demands (a share drawn from U[0.70, 0.95]) from UD[1, 10], the remaining
  customers (at least one, chosen uniformly) from UD[50, 100];
- average route size r ~ triangular T(3, 6, 25) (min 3, mode 6, max 25); capacity Q = ceil(r * sum(q) / n) (the
  paper's formula, also for unit demands, where the script uses floor(r) instead), raised to max(q) if smaller (the
  script's safeguard, so every customer fits a vehicle);
- distances: Euclidean rounded to the nearest integer (TSPLIB nint); objective: total distance; unlimited fleet.
Balanced attribute schedule (paper Section 3.2, not a factorial grid): instances are taken in a fixed order, slot
k = 3 * size_index + seed (size_index 0/1/2 for small/medium/large; seeds 0..2 give slots 0..8 in order of n, like the
X set's order by n). Depot positioning: slot k takes element k mod 3 of a random permutation of (C, E, R) drawn for
the block k // 3 (stream "x-depot/<block>"); customer positioning likewise over (R, C, RC); demand distribution over
the seven types with blocks of seven. The paper partitions 100 draws of r into quintiles and gives the k-th instance
an r from quintile ((k - 1) mod 5) + 1 (1-based); here slot k draws r from quintile (k mod 5) of T(3, 6, 25) itself
(inverse CDF of a uniform draw inside that quintile, stream "x-r/<k>"), the population version of that rule.
Positions and demands use the stream "x-cvrp/<size>/<seed>". No instance is selected or redrawn on solver outcomes.

Adaptations, named: (1) n is shrunk from 100-1000 to the sizes below (15 / 17 / 19 customers for small / medium /
large), calibrated on pilot seeds 100+ so that the subset-table reference (exact, at most 21 customers) finishes within
the 60 s budget (largest pilot: 22 s at n = 19, 51 s at n = 20) and the monolithic MIP proves optimality; customers are the non-depot nodes (n customers, n + 1 nodes). (2) S ~ UD[3, min(8, c)] where c is the number of
clustered customers (at small n an RC instance may have fewer than 8 clustered customers; the script aborts there).
(3) the SL share's distribution U[0.70, 0.95] and the at-least-one-large rule fill a detail the paper leaves open (the
script instead makes about 1.5 n / r customers large). (4) the paper's exp(-d/40) is used, not the script's 2^(-d/40),
and the paper's S ~ UD[3, 8], not the script's UD[2, 6]; the accept-reject envelope is S (see above), not the
script's saturating normalization. (5) With small n, r can exceed n, in which case the capacity does not bind (one
vehicle may serve every customer); this follows from keeping the paper's absolute route sizes.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, cast

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


SIZES = ("small", "medium", "large")
# Calibrated on pilot seeds 100+ (see the family report): the monolithic MIP proves every pilot in < 1.5 s at n = 20;
# the binding limit is the subset-table CG reference (ten n = 19 pilots <= 22 s; one n = 20 pilot took 51 s).
SIZE_CONFIG = {
    "small": {"n_customers": 15},
    "medium": {"n_customers": 17},
    "large": {"n_customers": 19},
}

GRID = 1000
DECAY = 40.0
DEPOT_POSITIONS = ("C", "E", "R")
CUSTOMER_POSITIONS = ("R", "C", "RC")
DEMAND_TYPES = ("U", "1-10", "5-10", "1-100", "50-100", "Q", "SL")
DEMAND_RANGES = {"U": (1, 1), "1-10": (1, 10), "5-10": (5, 10), "1-100": (1, 100), "50-100": (50, 100)}
ROUTE_SIZE = (3.0, 6.0, 25.0)  # triangular (min, mode, max)


@dataclass
class CVRPInstance(JsonDataMixin):
    n_customers: int
    capacity: int
    x: list[int]  # node coordinates, index 0 = depot
    y: list[int]
    demand: list[int]  # demand[0] = 0

    def __post_init__(self) -> None:
        for name in ("x", "y", "demand"):
            if len(getattr(self, name)) != self.n_customers + 1:
                raise ValueError(f"{name} must have n_customers + 1 entries")

    @classmethod
    def from_json_file(cls, path: str | Path) -> CVRPInstance:
        return cast("CVRPInstance", super().from_json_file(path))

    def dist(self, i: int, j: int) -> int:
        return math.floor(math.hypot(self.x[i] - self.x[j], self.y[i] - self.y[j]) + 0.5)


def _balanced(values: tuple[str, ...], k: int, name: str) -> str:
    """Element k mod |values| of a random permutation drawn for block k // |values| (paper Section 3.2)."""
    block = list(values)
    random.Random(f"x-{name}/{k // len(values)}").shuffle(block)
    return block[k % len(values)]


def triangular_quantile(u: float, lo: float = ROUTE_SIZE[0], mode: float = ROUTE_SIZE[1], hi: float = ROUTE_SIZE[2]) -> float:
    cut = (mode - lo) / (hi - lo)
    if u < cut:
        return lo + math.sqrt(u * (hi - lo) * (mode - lo))
    return hi - math.sqrt((1 - u) * (hi - lo) * (hi - mode))


def x_attributes(seed: int, size_label: str) -> dict[str, Any]:
    """The slot's attributes under the balanced schedule (independent of the positions/demands stream)."""
    k = 3 * SIZES.index(size_label) + seed
    quintile = k % 5
    u = (quintile + random.Random(f"x-r/{k}").random()) / 5
    return {"slot": k, "depot": _balanced(DEPOT_POSITIONS, k, "depot"), "customers": _balanced(CUSTOMER_POSITIONS, k, "customers"),
            "demand": _balanced(DEMAND_TYPES, k, "demand"), "r_quintile": quintile + 1, "r": triangular_quantile(u)}


class CVRPGenerator(BaseInstanceGenerator[CVRPInstance]):
    def generate_instance(self, *, seed: int | None = None, size_label: str = "medium") -> CVRPInstance:
        if size_label not in SIZE_CONFIG:
            raise ValueError(f"Unknown size_label '{size_label}'. Expected one of {list(SIZE_CONFIG)}.")
        return build_instance(int(seed or 0), size_label)[0]


def build_instance(seed: int, size_label: str) -> tuple[CVRPInstance, dict[str, Any]]:
    n = SIZE_CONFIG[size_label]["n_customers"]
    att = x_attributes(seed, size_label)
    rng = random.Random(f"x-cvrp/{size_label}/{seed}")
    point = lambda: (rng.randint(0, GRID), rng.randint(0, GRID))
    depot = {"C": (GRID // 2, GRID // 2), "E": (0, 0)}.get(att["depot"]) or point()
    used = {depot}

    def fresh():
        p = point()
        while p in used:
            p = point()
        return p

    n_random = {"R": n, "C": 0, "RC": n // 2}[att["customers"]]
    pts = []
    for _ in range(n_random):
        pts.append(fresh())
        used.add(pts[-1])
    n_clustered, n_seeds = n - n_random, 0
    if n_clustered:
        n_seeds = rng.randint(3, min(8, n_clustered))
        seeds = []
        for _ in range(n_seeds):
            seeds.append(fresh())
            used.add(seeds[-1])
        pull = lambda p: sum(math.exp(-math.dist(p, s) / DECAY) for s in seeds)
        pts += seeds
        while len(pts) < n:
            p = fresh()
            if rng.random() < pull(p) / n_seeds:
                pts.append(p)
                used.add(p)
    kind = att["demand"]
    if kind in DEMAND_RANGES:
        demand = [rng.randint(*DEMAND_RANGES[kind]) for _ in pts]
    elif kind == "Q":
        demand = [rng.randint(1, 50) if (px < GRID / 2) != (py < GRID / 2) else rng.randint(51, 100) for px, py in pts]
    else:  # SL
        share = rng.uniform(0.70, 0.95)
        large = set(rng.sample(range(n), max(1, round((1 - share) * n))))
        demand = [rng.randint(50, 100) if i in large else rng.randint(1, 10) for i in range(n)]
    capacity = max(max(demand), math.ceil(att["r"] * sum(demand) / n))
    inst = CVRPInstance(n, capacity, [depot[0]] + [p[0] for p in pts], [depot[1]] + [p[1] for p in pts], [0] + demand)
    source = {"recipe": "Uchoa et al. (2017) X generator", **att, "n_seeds": n_seeds, "k_lower_bound": math.ceil(sum(demand) / capacity),
              "stream": f"x-cvrp/{size_label}/{seed}"}
    return inst, source


def generate_cvrp_instance(seed: int, size: str = "medium", output_dir: str | Path | None = None) -> dict[str, Any]:
    inst, source = build_instance(seed, size)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
        # Provenance stays out of data.json (the only file the agent sees).
        (out_dir / "source.json").write_text(json.dumps(source, indent=2))
    return inst.to_dict()
