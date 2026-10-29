"""Nurse rostering instances (a compact INRC-style model).

Nurses n work at most one shift per day (0 = Early, 1 = Late, 2 = Night, or off) over D days.
Hard rules per nurse: min_days..max_days working days, at most max_nights nights, at most
max_consecutive consecutive working days, and forbidden successions Night -> Early/Late and Late -> Early
on consecutive days. Coverage demand[d][s] is soft: each missing nurse costs `shortage_penalty`.
Objective: sum of the chosen assignment costs (preferences) + shortage penalties.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, cast

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


SHIFTS = ("Early", "Late", "Night")
FORBIDDEN_SUCCESSIONS = [[2, 0], [2, 1], [1, 0]]

SIZE_CONFIG = {
    "small": {"n_nurses": 6, "n_days": 7},
    "medium": {"n_nurses": 8, "n_days": 14},
    "large": {"n_nurses": 10, "n_days": 14},
}

DEFAULT_CONFIG = {"cost_range": (0, 9), "shortage_penalty": 30, "max_consecutive": 5, "work_share": (0.5, 0.75)}


@dataclass
class NRPInstance(JsonDataMixin):
    instance_size: str
    n_nurses: int
    n_days: int
    demand: list[list[int]]  # demand[d][s]
    cost: list[list[list[int]]]  # cost[n][d][s]
    min_days: list[int]
    max_days: list[int]
    max_nights: list[int]
    max_consecutive: int
    forbidden_successions: list[list[int]]
    shortage_penalty: int

    @classmethod
    def from_json_file(cls, path: str | Path) -> NRPInstance:
        return cast("NRPInstance", super().from_json_file(path))


class NRPGenerator(BaseInstanceGenerator[NRPInstance]):
    def generate_instance(self, *, seed: int | None = None, size_label: str = "medium") -> NRPInstance:
        if size_label not in SIZE_CONFIG:
            raise ValueError(f"Unknown size_label '{size_label}'. Expected one of {list(SIZE_CONFIG)}.")
        cfg, size = DEFAULT_CONFIG, SIZE_CONFIG[size_label]
        rng = random.Random(seed)
        N, D = size["n_nurses"], size["n_days"]
        max_days = [round(D * rng.uniform(*cfg["work_share"])) for _ in range(N)]
        min_days = [max(0, m - 2) for m in max_days]
        capacity = sum(max_days)
        # Demand uses about 90% of the available nurse-days, spread over shifts (fewer nights).
        weights = [0.4, 0.35, 0.25]
        demand = [[0, 0, 0] for _ in range(D)]
        for _ in range(int(0.9 * capacity)):
            d = rng.randrange(D)
            s = rng.choices(range(3), weights)[0]
            demand[d][s] += 1
        cost = [[[rng.randint(*cfg["cost_range"]) for _ in range(3)] for _ in range(D)] for _ in range(N)]
        max_nights = [max(1, round(m / 3)) for m in max_days]
        return NRPInstance(size_label, N, D, demand, cost, min_days, max_days, max_nights, cfg["max_consecutive"], FORBIDDEN_SUCCESSIONS, cfg["shortage_penalty"])


def generate_nrp_instance(seed: int, size: str = "medium", output_dir: str | Path | None = None) -> dict[str, Any]:
    inst = NRPGenerator().generate_instance(seed=seed, size_label=size)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return inst.to_dict()
