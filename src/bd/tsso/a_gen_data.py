from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from utils.base_data import BaseInstanceGenerator, JsonDataMixin


DIFFICULTY_CONFIG = {
    "easy": {"products": (2, 3), "periods": (3, 5), "scenarios": (2, 3)},
    "medium": {"products": (3, 5), "periods": (5, 8), "scenarios": (3, 5)},
    "hard": {"products": (5, 7), "periods": (8, 10), "scenarios": (6, 8)},
}


@dataclass
class TSSOInstance(JsonDataMixin):
    """Two-stage stochastic production-inventory instance."""

    n_products: int
    n_periods: int
    n_scenarios: int
    prob: np.ndarray
    F: np.ndarray
    K: np.ndarray
    c_reg: np.ndarray
    c_ot: np.ndarray
    h: np.ndarray
    b: np.ndarray
    demand: np.ndarray
    I0: np.ndarray
    B0: np.ndarray

    def __post_init__(self) -> None:
        self.prob = np.asarray(self.prob, dtype=float).reshape(-1)
        self.F = np.asarray(self.F, dtype=float)
        self.K = np.asarray(self.K, dtype=float)
        self.c_reg = np.asarray(self.c_reg, dtype=float)
        self.c_ot = np.asarray(self.c_ot, dtype=float)
        self.h = np.asarray(self.h, dtype=float)
        self.b = np.asarray(self.b, dtype=float)
        self.demand = np.asarray(self.demand, dtype=float)
        self.I0 = np.asarray(self.I0, dtype=float).reshape(-1)
        self.B0 = np.asarray(self.B0, dtype=float).reshape(-1)

        total_prob = self.prob.sum()
        if total_prob <= 0:
            raise ValueError("Scenario probabilities must sum to a positive value.")
        if not np.isclose(total_prob, 1.0):
            self.prob = self.prob / total_prob

        expected = (self.n_products, self.n_periods)
        if self.F.shape != expected or self.K.shape != expected or self.c_reg.shape != expected:
            raise ValueError("First-stage arrays shape mismatch.")
        if self.c_ot.shape != expected or self.h.shape != expected or self.b.shape != expected:
            raise ValueError("Cost arrays shape mismatch.")
        expected_demand = (self.n_products, self.n_periods, self.n_scenarios)
        if self.demand.shape != expected_demand:
            raise ValueError("Demand shape mismatch.")
        if self.I0.shape[0] != self.n_products or self.B0.shape[0] != self.n_products:
            raise ValueError("Initial inventory/backlog length mismatch.")

    @classmethod
    def from_json_file(cls, path: str | Path) -> TSSOInstance:
        return cast("TSSOInstance", super().from_json_file(path))


class TSSOGenerator(BaseInstanceGenerator[TSSOInstance]):
    """Generator matching the benchmark implementation."""

    def __init__(self, *, seed: int = 7) -> None:
        self.default_seed = seed

    def generate_instance(
        self,
        n_products: int = 5,
        n_periods: int = 10,
        n_scenarios: int = 7,
        *,
        seed: int | None = None,
        demand_low: int = 12,
        demand_high: int = 55,
    ) -> TSSOInstance:
        rng = np.random.default_rng(seed if seed is not None else self.default_seed)

        prob = rng.random(n_scenarios)
        prob /= prob.sum()

        F = rng.integers(30, 90, size=(n_products, n_periods)).astype(float)
        K = rng.integers(25, 70, size=(n_products, n_periods)).astype(float)

        c_reg = rng.uniform(1.0, 3.0, size=(n_products, n_periods))
        c_ot = c_reg + rng.uniform(0.8, 2.0, size=(n_products, n_periods))
        h = rng.uniform(0.2, 0.8, size=(n_products, n_periods))
        b = rng.uniform(2.0, 4.5, size=(n_products, n_periods))

        demand = rng.integers(
            demand_low,
            demand_high + 1,
            size=(n_products, n_periods, n_scenarios),
            endpoint=False,
        ).astype(float)

        I0 = np.zeros(n_products, dtype=float)
        B0 = np.zeros(n_products, dtype=float)

        return TSSOInstance(
            n_products=n_products,
            n_periods=n_periods,
            n_scenarios=n_scenarios,
            prob=prob,
            F=F,
            K=K,
            c_reg=c_reg,
            c_ot=c_ot,
            h=h,
            b=b,
            demand=demand,
            I0=I0,
            B0=B0,
        )

    def plot_instance(self, inst: TSSOInstance, output_path: str | Path | None = None) -> None:
        _ = inst, output_path  # table-based instance, nothing to plot
        return None

    def plot_solution(self, inst: TSSOInstance, result: dict[str, Any], output_path: str | Path | None = None) -> None:
        _ = inst, result, output_path
        return None


def generate_tsso_instance(
    seed: int,
    difficulty: str = "easy",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Sample sizes from a difficulty definition and write data.json under output_dir.
    Returns the raw dictionary representation.
    """
    if difficulty not in DIFFICULTY_CONFIG:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Expected one of {list(DIFFICULTY_CONFIG)}.")

    cfg = DIFFICULTY_CONFIG[difficulty]
    rng = np.random.default_rng(seed)

    n_products = int(rng.integers(cfg["products"][0], cfg["products"][1] + 1))
    n_periods = int(rng.integers(cfg["periods"][0], cfg["periods"][1] + 1))
    n_scenarios = int(rng.integers(cfg["scenarios"][0], cfg["scenarios"][1] + 1))

    gen = TSSOGenerator(seed=seed)
    inst = gen.generate_instance(n_products=n_products, n_periods=n_periods, n_scenarios=n_scenarios, seed=seed)
    data = inst.to_dict()

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")

    return data
