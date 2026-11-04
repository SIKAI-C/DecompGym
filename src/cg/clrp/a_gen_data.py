"""Capacitated location-routing (CLRP), Prodhon-style random instances at small scale.

m candidate depots (coordinates, capacity, opening cost) and n customers (coordinates, demand); identical
vehicles of capacity Q with a fixed cost F per route, unlimited in number. Every customer is visited once
by a route that starts and ends at the same open depot; the demand served from a depot is at most its
capacity. Travel cost between two points is ceil(100 * Euclidean distance) (Prodhon's convention, as in
the scale-up CLRP). Objective: opening costs + per-route fixed costs + travel costs.

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
    "small": {"n_customers": 9, "n_depots": 3},
    "medium": {"n_customers": 13, "n_depots": 4},
    "large": {"n_customers": 16, "n_depots": 4},
}

DEFAULT_CONFIG = {
    "grid": 50,
    "demand_range": (11, 20),
    "vehicle_capacities": (70, 100),
    "vehicle_cost": 1000,
    "opening_cost_range": (4000, 8000),
    "depots_needed": (2, 3),
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


class CLRPGenerator(BaseInstanceGenerator[CLRPInstance]):
    def generate_instance(self, *, seed: int | None = None, size_label: str = "medium") -> CLRPInstance:
        if size_label not in SIZE_CONFIG:
            raise ValueError(f"Unknown size_label '{size_label}'. Expected one of {list(SIZE_CONFIG)}.")
        cfg = DEFAULT_CONFIG
        rng = random.Random(seed)
        n, m, g = SIZE_CONFIG[size_label]["n_customers"], SIZE_CONFIG[size_label]["n_depots"], cfg["grid"]
        dx, dy = [rng.randint(0, g) for _ in range(m)], [rng.randint(0, g) for _ in range(m)]
        cx, cy = [rng.randint(0, g) for _ in range(n)], [rng.randint(0, g) for _ in range(n)]
        demand = [rng.randint(*cfg["demand_range"]) for _ in range(n)]
        Q = rng.choice(cfg["vehicle_capacities"])
        # Each depot can serve 1/k of the total demand (k = 2 or 3, plus up to 25% slack), so at least k depots open.
        k = rng.randint(*cfg["depots_needed"])
        capacity = [max(Q, math.ceil(sum(demand) / k * rng.uniform(1.0, 1.25))) for _ in range(m)]
        opening = [rng.randint(*cfg["opening_cost_range"]) for _ in range(m)]
        return CLRPInstance(size_label, n, m, Q, cfg["vehicle_cost"], dx, dy, capacity, opening, cx, cy, demand)


def generate_clrp_instance(seed: int, size: str = "medium", output_dir: str | Path | None = None) -> dict[str, Any]:
    inst = CLRPGenerator().generate_instance(seed=seed, size_label=size)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return inst.to_dict()
