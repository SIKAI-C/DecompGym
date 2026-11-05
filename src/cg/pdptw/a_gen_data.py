"""Pickup and delivery with time windows (PDPTW), Li-Lim-style random instances.

Node 0 is the depot; nodes 1..n are pickups and n+1..2n the matching deliveries (request k = (k, n+k)).
Each request is carried by one vehicle, picked up before it is delivered, with vehicle load within
[0, Q] throughout. Travel time equals Euclidean distance rounded down to one decimal; the objective is
total distance with an unlimited fleet.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, cast

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


SIZE_CONFIG = {
    "small": {"n_requests": 4},
    "medium": {"n_requests": 6},
    "large": {"n_requests": 8},
}

DEFAULT_CONFIG = {"grid": 100, "demand_range": (10, 30), "capacity": 60, "service_time": 10, "horizon": 600, "window_width_range": (60, 150)}


@dataclass
class PDPTWInstance(JsonDataMixin):
    instance_size: str
    n_requests: int
    capacity: int
    x: list[float]
    y: list[float]
    demand: list[int]  # +q at a pickup, -q at its delivery, 0 at the depot
    ready: list[float]
    due: list[float]
    service: list[float]

    def __post_init__(self) -> None:
        if len(self.x) != 2 * self.n_requests + 1:
            raise ValueError("expected 2 * n_requests + 1 nodes")

    @classmethod
    def from_json_file(cls, path: str | Path) -> PDPTWInstance:
        return cast("PDPTWInstance", super().from_json_file(path))

    def dist(self, i: int, j: int) -> float:
        return math.floor(10.0 * math.hypot(self.x[i] - self.x[j], self.y[i] - self.y[j])) / 10.0


class PDPTWGenerator(BaseInstanceGenerator[PDPTWInstance]):
    def generate_instance(self, *, seed: int | None = None, size_label: str = "medium") -> PDPTWInstance:
        if size_label not in SIZE_CONFIG:
            raise ValueError(f"Unknown size_label '{size_label}'. Expected one of {list(SIZE_CONFIG)}.")
        cfg = DEFAULT_CONFIG
        rng = random.Random(seed)
        n = SIZE_CONFIG[size_label]["n_requests"]
        g, H, svc = cfg["grid"], cfg["horizon"], cfg["service_time"]
        pts = [(g / 2, g / 2)] + [(float(rng.randint(0, g)), float(rng.randint(0, g))) for _ in range(2 * n)]
        d = lambda a, b: math.floor(10.0 * math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1])) / 10.0
        demand, ready, due = [0] * (2 * n + 1), [0.0] * (2 * n + 1), [float(H)] * (2 * n + 1)
        for k in range(1, n + 1):
            p, q = k, n + k
            demand[p] = rng.randint(*cfg["demand_range"])
            demand[q] = -demand[p]
            # Pickup window first; the delivery window opens once the direct trip is possible.
            latest_p = H - svc - d(p, q) - svc - d(q, 0)
            wp = rng.randint(*cfg["window_width_range"])
            a_p = rng.randint(math.ceil(d(0, p)), max(math.ceil(d(0, p)), int(latest_p) - wp))
            ready[p], due[p] = float(a_p), float(min(a_p + wp, latest_p))
            wq = rng.randint(*cfg["window_width_range"])
            a_q = math.ceil(ready[p] + svc + d(p, q))
            ready[q], due[q] = float(a_q), float(min(a_q + wq, H - svc - d(q, 0)))
        return PDPTWInstance(size_label, n, cfg["capacity"], [p[0] for p in pts], [p[1] for p in pts], demand, ready, due, [0.0] + [float(svc)] * (2 * n))


def generate_pdptw_instance(seed: int, size: str = "medium", output_dir: str | Path | None = None) -> dict[str, Any]:
    inst = PDPTWGenerator().generate_instance(seed=seed, size_label=size)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return inst.to_dict()
