from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


SIZE_CONFIG = {
    "small": {"m_range": (2, 3), "n_range": (3, 5)},
    "medium": {"m_range": (3, 4), "n_range": (4, 6)},
    "large": {"m_range": (4, 5), "n_range": (6, 8)},
}


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


class FCTPGenerator(BaseInstanceGenerator[FCTPInstance]):
    """Synthetic FCTP instance generator."""

    def __init__(
        self,
        *,
        seed: int = 1804,
        supply_range: tuple[int, int] = (15, 35),
        demand_range: tuple[int, int] = (10, 25),
    ) -> None:
        self.default_seed = seed
        self.supply_range = supply_range
        self.demand_range = demand_range

    def generate_instance(
        self,
        m: int = 5,
        n: int = 8,
        *,
        seed: int | None = None,
    ) -> FCTPInstance:
        rng = np.random.default_rng(seed if seed is not None else self.default_seed)
        supply = rng.integers(self.supply_range[0], self.supply_range[1] + 1, size=m).astype(float)
        demand = rng.integers(self.demand_range[0], self.demand_range[1] + 1, size=n).astype(float)

        total_supply = supply.sum()
        total_demand = demand.sum()
        if total_demand == 0:
            raise ValueError("Total demand must be positive.")
        scale = min(0.95 * total_supply / total_demand, 1.0)
        demand = np.maximum(1.0, np.floor(demand * scale)).astype(float)
        gap = total_supply - demand.sum()
        demand[-1] += gap

        unit_cost = rng.uniform(1.0, 10.0, size=(m, n))
        fixed_cost = rng.integers(5, 25, size=(m, n)).astype(float)
        capacity = np.maximum(supply[:, None], demand[None, :])

        return FCTPInstance(
            m=m,
            n=n,
            supply=supply,
            demand=demand,
            unit_cost=unit_cost,
            fixed_cost=fixed_cost,
            capacity=capacity,
        )

    def _as_numpy(self, arr: Iterable, *, dtype=float) -> np.ndarray:
        return np.asarray(arr, dtype=dtype)

    def plot_instance(self, inst: FCTPInstance, output_path: str | Path | None = None) -> None:
        supply_y = np.linspace(0, inst.m - 1, inst.m)
        demand_y = np.linspace(0, inst.n - 1, inst.n)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title("FCTP instance")
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.1, max(inst.m, inst.n))

        ax.scatter(np.zeros(inst.m), supply_y, marker="s", s=70, label="Supply")
        ax.scatter(np.ones(inst.n), demand_y, marker="o", s=70, color="tab:red", label="Demand")

        for i in range(inst.m):
            for j in range(inst.n):
                ax.plot([0, 1], [supply_y[i], demand_y[j]], color="lightgray", linewidth=0.5, alpha=0.4)

        for i in range(inst.m):
            ax.text(-0.1, supply_y[i], f"S{i}={inst.supply[i]:.0f}", ha="right", va="center", fontsize=8)
        for j in range(inst.n):
            ax.text(1.1, demand_y[j], f"D{j}={inst.demand[j]:.0f}", ha="left", va="center", fontsize=8)

        ax.legend(loc="upper left")
        if output_path is None:
            plt.show()
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, format="pdf", bbox_inches="tight")
            plt.close(fig)

    def plot_solution(
        self,
        inst: FCTPInstance,
        result: dict[str, Any],
        *,
        output_path: str | Path | None = None,
        open_tol: float = 0.5,
    ) -> None:
        """Plot using finalized result dict (baseline or Benders)."""
        vars_dict = result.get("final_vars", {}) or {}
        sub_meta = result.get("sub_metadata", {}) or {}
        master_meta = result.get("master_metadata", {}) or {}

        if "x_assignment" in sub_meta:
            x_arr = np.asarray(sub_meta["x_assignment"], dtype=float).reshape(inst.m, inst.n)
        elif "x_assignment" in master_meta:
            x_arr = np.asarray(master_meta["x_assignment"], dtype=float).reshape(inst.m, inst.n)
        else:
            x_arr = np.zeros((inst.m, inst.n), dtype=float)
            for i in range(inst.m):
                for j in range(inst.n):
                    x_arr[i, j] = float(vars_dict.get(f"x_{i}_{j}", 0.0) or 0.0)

        if "y_assignment" in sub_meta:
            y_arr = np.asarray(sub_meta["y_assignment"], dtype=float).reshape(inst.m, inst.n)
        elif "y_assignment" in master_meta:
            y_arr = np.asarray(master_meta["y_assignment"], dtype=float).reshape(inst.m, inst.n)
        else:
            y_arr = np.array([vars_dict.get(f"y_{i}_{j}", 0.0) or 0.0 for i in range(inst.m) for j in range(inst.n)])
            y_arr = y_arr.reshape(inst.m, inst.n)

        supply_y = np.linspace(0, inst.m - 1, inst.m)
        demand_y = np.linspace(0, inst.n - 1, inst.n)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title("FCTP solution")
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.1, max(inst.m, inst.n))

        for i in range(inst.m):
            for j in range(inst.n):
                if x_arr[i, j] > 1e-6:
                    ax.plot(
                        [0, 1],
                        [supply_y[i], demand_y[j]],
                        color="black",
                        linewidth=0.5 + 2.0 * (x_arr[i, j] / max(1.0, inst.capacity[i, j])),
                        alpha=0.7,
                    )

        ax.scatter(np.zeros(inst.m), supply_y, marker="s", s=70, label="Supply")
        ax.scatter(np.ones(inst.n), demand_y, marker="o", s=70, color="tab:red", label="Demand")

        for i in range(inst.m):
            for j in range(inst.n):
                if y_arr[i, j] >= open_tol:
                    ax.scatter([0.5], [(supply_y[i] + demand_y[j]) / 2], color="darkblue", s=12, alpha=0.8)

        ax.legend(loc="upper left")
        if output_path is None:
            plt.show()
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, format="pdf", bbox_inches="tight")
            plt.close(fig)


def generate_fctp_instance(
    seed: int,
    size: str = "small",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    if size not in SIZE_CONFIG:
        raise ValueError(f"Unknown size '{size}'. Expected one of {list(SIZE_CONFIG)}.")
    cfg = SIZE_CONFIG[size]
    rng = np.random.default_rng(seed)
    m = int(rng.integers(cfg["m_range"][0], cfg["m_range"][1] + 1))
    n = int(rng.integers(cfg["n_range"][0], cfg["n_range"][1] + 1))

    gen = FCTPGenerator(seed=seed)
    inst = gen.generate_instance(m=m, n=n, seed=seed)
    data = inst.to_dict()

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")

    return data
