"""Capacitated vehicle routing (CVRP), Uchoa X-style random instances at small scale.

Node 0 is the depot. Coordinates are integers in [0, 1000]; travel distance is the Euclidean distance
rounded to the nearest integer (TSPLIB nint, floor(d + 0.5)); the objective is total distance. Vehicles
are identical with capacity Q and unlimited in number (no vehicle cost); every customer is visited once.
Semantics follow the scale-up CVRP (CVRPLIB X instances).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, cast

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


SIZE_CONFIG = {
    "small": {"n_customers": 11},
    "medium": {"n_customers": 15},
    "large": {"n_customers": 20},
}

DEFAULT_CONFIG = {
    "grid": 1000,
    "demand_ranges": ((1, 10), (1, 100)),
    "customers_per_route": (3, 6),
}


@dataclass
class CVRPInstance(JsonDataMixin):
    instance_size: str
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


class CVRPGenerator(BaseInstanceGenerator[CVRPInstance]):
    """Depot at the center, a corner or a random point; customers random or clustered (X-set positioning)."""

    def generate_instance(self, *, seed: int | None = None, size_label: str = "medium") -> CVRPInstance:
        if size_label not in SIZE_CONFIG:
            raise ValueError(f"Unknown size_label '{size_label}'. Expected one of {list(SIZE_CONFIG)}.")
        cfg = DEFAULT_CONFIG
        rng = random.Random(seed)
        n, g = SIZE_CONFIG[size_label]["n_customers"], cfg["grid"]
        depot = rng.choice(["center", "corner", "random"])
        dx, dy = {"center": (g // 2, g // 2), "corner": (0, 0), "random": (rng.randint(0, g), rng.randint(0, g))}[depot]
        xs, ys = [dx], [dy]
        if rng.random() < 0.5:
            pts = [(rng.randint(0, g), rng.randint(0, g)) for _ in range(n)]
        else:
            seeds = [(rng.randint(0, g), rng.randint(0, g)) for _ in range(rng.randint(2, 4))]
            pts = []
            while len(pts) < n:
                cx, cy = rng.choice(seeds)
                px, py = round(rng.gauss(cx, 60)), round(rng.gauss(cy, 60))
                if 0 <= px <= g and 0 <= py <= g:
                    pts.append((px, py))
        xs += [p[0] for p in pts]
        ys += [p[1] for p in pts]
        lo, hi = rng.choice(cfg["demand_ranges"])
        demand = [0] + [rng.randint(lo, hi) for _ in range(n)]
        r = rng.uniform(*cfg["customers_per_route"])
        capacity = max(max(demand), math.ceil(r * sum(demand) / n))
        return CVRPInstance(size_label, n, capacity, xs, ys, demand)


def generate_cvrp_instance(seed: int, size: str = "medium", output_dir: str | Path | None = None) -> dict[str, Any]:
    inst = CVRPGenerator().generate_instance(seed=seed, size_label=size)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return inst.to_dict()
