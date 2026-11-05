"""Capacitated vehicle routing with time windows (VRPTW), Solomon-style random instances.

Node 0 is the depot. Travel time equals Euclidean distance (rounded down to one decimal, as in Solomon's
benchmark convention), and the objective is total travel distance. Vehicles are identical with capacity Q
and unlimited in number; every route starts and ends at the depot within the depot's window, and a vehicle
may wait at a customer until its window opens.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, cast

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


SIZE_CONFIG = {
    "small": {"n_customers": 10},
    "medium": {"n_customers": 15},
    "large": {"n_customers": 20},
}

DEFAULT_CONFIG = {
    "grid": 100,
    "demand_range": (5, 30),
    "capacity": 100,
    "service_time": 10,
    "horizon": 400,
    "window_width_range": (40, 120),
}


@dataclass
class VRPTWInstance(JsonDataMixin):
    instance_size: str
    n_customers: int
    capacity: int
    x: list[float]  # node coordinates, index 0 = depot
    y: list[float]
    demand: list[int]  # demand[0] = 0
    ready: list[float]
    due: list[float]
    service: list[float]

    def __post_init__(self) -> None:
        n = self.n_customers + 1
        for name in ("x", "y", "demand", "ready", "due", "service"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must have n_customers + 1 entries")

    @classmethod
    def from_json_file(cls, path: str | Path) -> VRPTWInstance:
        return cast("VRPTWInstance", super().from_json_file(path))

    def dist(self, i: int, j: int) -> float:
        return math.floor(10.0 * math.hypot(self.x[i] - self.x[j], self.y[i] - self.y[j])) / 10.0


class VRPTWGenerator(BaseInstanceGenerator[VRPTWInstance]):
    def generate_instance(self, *, seed: int | None = None, size_label: str = "medium") -> VRPTWInstance:
        if size_label not in SIZE_CONFIG:
            raise ValueError(f"Unknown size_label '{size_label}'. Expected one of {list(SIZE_CONFIG)}.")
        cfg = DEFAULT_CONFIG
        rng = random.Random(seed)
        n = SIZE_CONFIG[size_label]["n_customers"]
        g, H, svc = cfg["grid"], cfg["horizon"], cfg["service_time"]
        xs, ys = [g / 2], [g / 2]
        demand, ready, due, service = [0], [0.0], [float(H)], [0.0]
        for _ in range(n):
            xi, yi = float(rng.randint(0, g)), float(rng.randint(0, g))
            reach = math.hypot(xi - g / 2, yi - g / 2)
            # Window center chosen so the customer can be served and the vehicle can still return.
            lo, hi = math.ceil(reach), math.floor(H - svc - reach)
            width = rng.randint(*cfg["window_width_range"])
            center = rng.randint(lo + width // 2, max(lo + width // 2, hi - width // 2))
            xs.append(xi)
            ys.append(yi)
            demand.append(rng.randint(*cfg["demand_range"]))
            ready.append(float(max(lo, center - width // 2)))
            due.append(float(min(hi, center + width // 2)))
            service.append(float(svc))
        return VRPTWInstance(size_label, n, cfg["capacity"], xs, ys, demand, ready, due, service)


def generate_vrptw_instance(seed: int, size: str = "medium", output_dir: str | Path | None = None) -> dict[str, Any]:
    inst = VRPTWGenerator().generate_instance(seed=seed, size_label=size)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return inst.to_dict()
