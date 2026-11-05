"""Stochastic server location (SSLP-type) instances: binary first stage, mixed-binary recourse.

First stage: open servers y_j in {0,1} at opening cost c_j, at most max_open of them.
Scenario s (equally likely): client i is present iff h[s][i] = 1. Each present client is assigned to
exactly one site (z_ijs in {0,1}), earning revenue q_ij and using d_ij capacity. Site j's load beyond
u * y_j is overflow v_js >= 0 at unit penalty q0 (canonical SSLP semantics: a closed site has zero
capacity, so every unit assigned to it is overflow). This gives complete recourse.
Objective: min sum_j c_j y_j + sum_s p_s (q0 sum_j v_js - sum_ij q_ij z_ijs).
Follows the SSLP recipe of Ntaimo and Sen (2005); instances are generated, not taken from SIPLIB.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, cast

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


SIZE_CONFIG = {
    "small": {"n_sites": 5, "n_clients": 15, "n_scenarios": 10},
    "medium": {"n_sites": 8, "n_clients": 20, "n_scenarios": 20},
    "large": {"n_sites": 10, "n_clients": 25, "n_scenarios": 30},
}

DEFAULT_CONFIG = {
    "open_cost_range": (40, 80),
    "demand_range": (0, 25),  # d_ij = q_ij, as in SSLP
    "capacity_factor": 1.5,  # u = factor * sum_ij d_ij / n_sites, scaled by the presence rate
    "presence_prob": 0.5,
    "overflow_penalty": 1000,
}


@dataclass
class SSLPInstance(JsonDataMixin):
    instance_size: str
    n_sites: int
    n_clients: int
    n_scenarios: int
    open_cost: list[int]
    max_open: int
    capacity: int
    demand: list[list[int]]  # demand[i][j]
    revenue: list[list[int]]  # revenue[i][j]
    overflow_penalty: int
    presence: list[list[int]]  # presence[s][i] in {0, 1}
    probability: list[float]

    def __post_init__(self) -> None:
        if len(self.presence) != self.n_scenarios or any(len(r) != self.n_clients for r in self.presence):
            raise ValueError("presence must be n_scenarios x n_clients")
        if abs(sum(self.probability) - 1.0) > 1e-9:
            raise ValueError("scenario probabilities must sum to 1")

    @classmethod
    def from_json_file(cls, path: str | Path) -> SSLPInstance:
        return cast("SSLPInstance", super().from_json_file(path))

    def recourse_lower_bound(self, s: int) -> float:
        """Valid lower bound on scenario s's recourse cost: every present client earns its best revenue, no overflow."""
        return -float(sum(max(self.revenue[i]) for i in range(self.n_clients) if self.presence[s][i]))


class SSLPGenerator(BaseInstanceGenerator[SSLPInstance]):
    def generate_instance(self, *, seed: int | None = None, size_label: str = "medium") -> SSLPInstance:
        if size_label not in SIZE_CONFIG:
            raise ValueError(f"Unknown size_label '{size_label}'. Expected one of {list(SIZE_CONFIG)}.")
        cfg, size = DEFAULT_CONFIG, SIZE_CONFIG[size_label]
        rng = random.Random(seed)
        J, I, S = size["n_sites"], size["n_clients"], size["n_scenarios"]
        demand = [[rng.randint(*cfg["demand_range"]) for _ in range(J)] for _ in range(I)]
        capacity = round(cfg["capacity_factor"] * cfg["presence_prob"] * sum(map(sum, demand)) / J)
        presence = [[int(rng.random() < cfg["presence_prob"]) for _ in range(I)] for _ in range(S)]
        return SSLPInstance(
            instance_size=size_label, n_sites=J, n_clients=I, n_scenarios=S,
            open_cost=[rng.randint(*cfg["open_cost_range"]) for _ in range(J)], max_open=J, capacity=capacity,
            demand=demand, revenue=[row[:] for row in demand], overflow_penalty=cfg["overflow_penalty"],
            presence=presence, probability=[1.0 / S] * S,
        )


def generate_sslp_instance(seed: int, size: str = "medium", output_dir: str | Path | None = None) -> dict[str, Any]:
    inst = SSLPGenerator().generate_instance(seed=seed, size_label=size)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return inst.to_dict()
