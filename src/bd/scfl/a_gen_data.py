"""Two-stage stochastic capacitated facility location (SCFL) at small scale.

Semantics follow the scale-up SCFL (Louveaux 1986; Bodur, Dash, Guenluek and Luedtke 2017, formulation (15)):
    first stage   open facilities y_i in {0, 1} (fixed cost f_i, capacity u_i) with sum_i u_i y_i >= max_s sum_j d_sj;
    second stage  per scenario s (probability p_s): x_sij >= 0, sum_i x_sij = d_sj, sum_j x_sij <= u_i y_i,
                  cost Q_s(y) = min sum_ij c_ij x_sij;
    minimize      f.y + sum_s p_s Q_s(y).
Instances follow the scale-up `make_cst` recipe (Cornuejols, Sridharan and Thizy 1991 base, Bodur et al. 2017
stochastic demands): facilities and customers uniform in the unit square, mean demands dbar_j ~ U{5..35},
capacities U[10, 160] rescaled so that sum_i u_i = ratio * sum_j dbar_j (rounded up to integers), fixed costs
U[0, 90] + U[100, 110] sqrt(u_i), unit costs 10 * Euclidean distance; then sigma_j ~ U(0.1 dbar_j, 0.3 dbar_j) and
d_sj ~ N(dbar_j, sigma_j), rounded to the nearest integer and truncated at 0, equal probabilities; all from
numpy default_rng(seed) in that order. Small-scale choices: ratio ~ U(2, 3) (tighter than CST's 3-10, so capacity
matters), unit costs rounded to 3 decimals and fixed costs to 2 decimals (exact JSON values).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.utils.base_data import JsonDataMixin


SIZE_CONFIG = {
    "small": {"n_facilities": 5, "n_customers": 10, "n_scenarios": 5},
    "medium": {"n_facilities": 8, "n_customers": 15, "n_scenarios": 10},
    "large": {"n_facilities": 10, "n_customers": 20, "n_scenarios": 20},
}

DEFAULT_CONFIG = {"ratio": (2.0, 3.0), "sigma": (0.1, 0.3)}


@dataclass
class SCFLInstance(JsonDataMixin):
    instance_size: str
    n_facilities: int
    n_customers: int
    n_scenarios: int
    capacity: list[int]           # u_i
    fixed_cost: list[float]       # f_i
    unit_cost: list[list[float]]  # c_ij, n_facilities x n_customers
    demand: list[list[int]]       # d_sj, n_scenarios x n_customers
    probability: list[float]      # p_s

    def __post_init__(self) -> None:
        I, J, S = self.n_facilities, self.n_customers, self.n_scenarios
        if len(self.capacity) != I or len(self.fixed_cost) != I or len(self.unit_cost) != I or any(len(r) != J for r in self.unit_cost):
            raise ValueError("facility data must have n_facilities entries (unit_cost rows of n_customers)")
        if len(self.demand) != S or any(len(r) != J for r in self.demand) or len(self.probability) != S:
            raise ValueError("demand must be n_scenarios x n_customers and probability of length n_scenarios")

    @classmethod
    def from_json_file(cls, path: str | Path) -> SCFLInstance:
        return cast("SCFLInstance", super().from_json_file(path))

    @property
    def max_demand(self) -> int:
        return max(sum(row) for row in self.demand)


def generate_scfl_instance(seed: int, size: str = "medium", output_dir: str | Path | None = None) -> dict[str, Any]:
    if size not in SIZE_CONFIG:
        raise ValueError(f"Unknown size_label '{size}'. Expected one of {list(SIZE_CONFIG)}.")
    I, J, S = (SIZE_CONFIG[size][k] for k in ("n_facilities", "n_customers", "n_scenarios"))
    rng = np.random.default_rng(seed)
    ratio = rng.uniform(*DEFAULT_CONFIG["ratio"])
    fpos, cpos = rng.random((I, 2)), rng.random((J, 2))
    dbar = rng.integers(5, 36, size=J).astype(float)
    cap = rng.uniform(10, 160, size=I)
    cap = np.ceil(cap * ratio * dbar.sum() / cap.sum())
    fixed = rng.uniform(0, 90, size=I) + rng.uniform(100, 110, size=I) * np.sqrt(cap)
    cost = 10.0 * np.sqrt(((fpos[:, None, :] - cpos[None, :, :]) ** 2).sum(axis=2))
    lo, hi = DEFAULT_CONFIG["sigma"]
    sig = rng.uniform(lo * dbar, hi * dbar)
    demand = np.maximum(np.rint(rng.normal(dbar[None, :], sig[None, :], size=(S, J))), 0.0)
    inst = SCFLInstance(size, I, J, S, [int(v) for v in cap], [round(float(v), 2) for v in fixed],
                        [[round(float(v), 3) for v in row] for row in cost], [[int(v) for v in row] for row in demand], [1.0 / S] * S)
    if sum(inst.capacity) < inst.max_demand:
        raise ValueError("generated instance is infeasible (total capacity below a scenario demand)")
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return inst.to_dict()
