from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.utils.base_data import JsonDataMixin


@dataclass
class FCTPInstance(JsonDataMixin):
    """Fixed-charge transportation problem data."""

    m: int
    n: int
    supply: np.ndarray  # (m,)
    demand: np.ndarray  # (n,)
    unit_cost: np.ndarray  # (m, n)
    fixed_cost: np.ndarray  # (m, n)
    capacity: np.ndarray  # (m, n)

    def __post_init__(self) -> None:
        self.supply = np.asarray(self.supply, dtype=float).reshape(-1)
        self.demand = np.asarray(self.demand, dtype=float).reshape(-1)
        self.unit_cost = np.asarray(self.unit_cost, dtype=float)
        self.fixed_cost = np.asarray(self.fixed_cost, dtype=float)
        self.capacity = np.asarray(self.capacity, dtype=float)

        if self.supply.shape[0] != self.m:
            raise ValueError("supply length mismatch")
        if self.demand.shape[0] != self.n:
            raise ValueError("demand length mismatch")
        if self.unit_cost.shape != (self.m, self.n):
            raise ValueError("unit_cost shape mismatch")
        if self.fixed_cost.shape != (self.m, self.n):
            raise ValueError("fixed_cost shape mismatch")
        if self.capacity.shape != (self.m, self.n):
            raise ValueError("capacity shape mismatch")

    @classmethod
    def from_json_file(cls, path: str | Path) -> FCTPInstance:
        return cast("FCTPInstance", super().from_json_file(path))


# -----------------------
# Roberti-Bartolini-Mingozzi (Data Set 3/4) recipe (small-scale benchmark generator since 2026-11-05)
# -----------------------

LIT_SIZES = ("small", "medium", "large")
LIT_SIZE_CONFIG = {"small": {"n": 11}, "medium": {"n": 13}, "large": {"n": 14}}  # m = n; calibrated 2026-11-06 (round 2) on pilot seeds >= 100
LIT_FIXED_RANGE = (200, 800)  # f_ij, integers
LIT_B = 20  # a_i, b_j integers in [1, B]; B = 20 as in the B = 20 classes of Data Sets 3-4 (and Agarwal-Aneja's set)
LIT_THETAS = (0.0, 0.2, 0.5)
MAX_BALANCE_DRAWS = 100_000


def generate_fctp_instance(
    seed: int,
    size: str = "small",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Balanced square FCTP instance following the Roberti-Bartolini-Mingozzi recipe (Data Sets 3-4 of Mingozzi-Roberti).

    Source: Mingozzi and Roberti (2018), "An exact algorithm for the fixed charge transportation problem based on
    matching source and sink patterns", Transportation Science 52(2):229-238, Section 7. Data Set 3 (introduced by
    Roberti, Bartolini and Mingozzi 2015, Management Science 61(6):1275-1291) and Data Set 4 (same recipe, other
    (n, B)) share with the earlier Agarwal and Aneja (2012, Operations Research 60(3):638-654) set (Data Set 2, whose
    unit costs are instead random values scaled to a target ratio) the common part: m = n; fixed costs f_ij randomly
    generated in [200, 800]; supplies a_i and demands b_j randomly generated in [1, B]; balanced, sum_i a_i = sum_j b_j.
    The unit costs used here are those of Data Sets 3-4 only:
    c_ij = floor(theta * f_ij * (2n - 1) / sum_i a_i) with theta in {0.0, 0.2, 0.5} (theta is the target ratio
    between total variable and total fixed cost of a basic solution with 2n - 1 arcs). Formulation F0 links
    x_ij <= m_ij y_ij with m_ij = min(a_i, b_j); `capacity` holds m_ij.

    Adaptations (named):
    - size: the literature has n = 15-120; we keep every distribution and shrink n = m to LIT_SIZE_CONFIG[size]
      with the calibration target that the reference Benders (60 s budget, one thread, full process) takes at most
      30 s (small, medium) and 45 s (large) on pilot seeds >= 100 (in-process pilot times plus full-process checks); the monolithic MIP only proves optimality (MIPGap 0,
      one thread, limit 1800 s); B = 20 throughout;
    - integer draws: f_ij ~ U{200..800}, a_i, b_j ~ U{1..B} (uniform integers, bounds included; the sources say
      "randomly generated in the interval");
    - balance: the sources do not state how balance is obtained; we redraw (a, b) together until sum a = sum b
      (declared feasibility rule; this keeps a, b exchangeable and uniform conditioned on balance);
    - theta follows the declared schedule theta = (0.0, 0.2, 0.5)[seed mod 3]; draw order: numpy default_rng(seed),
      then f (row-major), then (a, b) pairs until balanced.
    The paper's MIP and Benders keep `sum_j x_ij <= a_i` and `sum_i x_ij >= b_j`, which equal the literature's
    equalities on balanced instances.
    """
    if size not in LIT_SIZE_CONFIG:
        raise ValueError(f"Unknown size '{size}'. Expected one of {list(LIT_SIZE_CONFIG)}.")
    n = int(LIT_SIZE_CONFIG[size]["n"])
    theta = LIT_THETAS[seed % len(LIT_THETAS)]
    rng = np.random.default_rng(seed)
    fixed = rng.integers(LIT_FIXED_RANGE[0], LIT_FIXED_RANGE[1] + 1, size=(n, n))
    for draws in range(1, MAX_BALANCE_DRAWS + 1):
        a = rng.integers(1, LIT_B + 1, size=n)
        b = rng.integers(1, LIT_B + 1, size=n)
        if a.sum() == b.sum():
            break
    else:
        raise RuntimeError("no balanced supply/demand draw")
    # Exact integer arithmetic for theta in tenths: floor(theta * f * (2n - 1) / A) = floor(t * f * (2n - 1) / (10 A)).
    tenths = round(theta * 10)
    unit = (tenths * fixed * (2 * n - 1)) // (10 * int(a.sum()))
    capacity = np.minimum(a[:, None], b[None, :])
    info = {"recipe": "roberti-bartolini-mingozzi-ds3", "theta": theta, "B": LIT_B, "fixed_cost_range": list(LIT_FIXED_RANGE),
            "unit_cost_rule": "floor(theta * f_ij * (2n - 1) / sum_i a_i)", "capacity_rule": "min(supply_i, demand_j)",
            "balanced": True, "balance_draws": draws}
    inst = FCTPInstance(m=n, n=n, supply=a.astype(float), demand=b.astype(float), unit_cost=unit.astype(float),
                        fixed_cost=fixed.astype(float), capacity=capacity.astype(float))
    data = inst.to_dict()
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
        # Recipe provenance stays outside data.json (agents receive only data.json and prompt.txt).
        (out_dir / "source.json").write_text(json.dumps(info, indent=2))
    return data
