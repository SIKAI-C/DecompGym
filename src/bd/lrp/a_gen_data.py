from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, pairwise, permutations
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from utils.base_data import BaseInstanceGenerator, JsonDataMixin


DIFFICULTY_CONFIG = {
    "easy": {"m_range": (3, 4), "n_range": (6, 8), "max_stops": 4},
    "medium": {"m_range": (4, 6), "n_range": (8, 10), "max_stops": 5},
    "hard": {"m_range": (5, 7), "n_range": (10, 14), "max_stops": 6},
}


@dataclass
class LRPInstance(JsonDataMixin):
    """Route-enumerated Location Routing Problem data."""

    m: int  # depots
    n: int  # customers
    max_stops: int
    max_vehicles: np.ndarray  # (m,)
    depot_cost: np.ndarray  # (m,)
    Xd: np.ndarray  # (m,2)
    Xc: np.ndarray  # (n,2)
    routes: list[list[int]]  # list of customer sequences
    route_depot: np.ndarray  # (R,)
    route_cost: np.ndarray  # (R,)
    incidence: np.ndarray  # (n, R)

    def __post_init__(self) -> None:
        self.max_vehicles = np.asarray(self.max_vehicles, dtype=float).reshape(-1)
        self.depot_cost = np.asarray(self.depot_cost, dtype=float).reshape(-1)
        self.Xd = np.asarray(self.Xd, dtype=float)
        self.Xc = np.asarray(self.Xc, dtype=float)
        self.route_depot = np.asarray(self.route_depot, dtype=int).reshape(-1)
        self.route_cost = np.asarray(self.route_cost, dtype=float).reshape(-1)
        self.incidence = np.asarray(self.incidence, dtype=float)

        if len(self.max_vehicles) != self.m or len(self.depot_cost) != self.m:
            raise ValueError("max_vehicles and depot_cost must have length m.")
        if self.Xd.shape != (self.m, 2):
            raise ValueError("Xd shape mismatch.")
        if self.Xc.shape != (self.n, 2):
            raise ValueError("Xc shape mismatch.")
        if self.incidence.shape != (self.n, len(self.route_cost)):
            raise ValueError("incidence shape mismatch vs routes.")
        if len(self.routes) != len(self.route_cost):
            raise ValueError("routes length mismatch.")
        if self.route_depot.shape[0] != len(self.route_cost):
            raise ValueError("route_depot length mismatch.")

    @property
    def R(self) -> int:
        return len(self.route_cost)

    @classmethod
    def from_json_file(cls, path: str | Path) -> LRPInstance:
        return cast("LRPInstance", super().from_json_file(path))


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _best_route_cost(depot: np.ndarray, customers: Sequence[int], Xc: np.ndarray) -> tuple[float, list[int]]:
    best_cost = float("inf")
    best_order: list[int] = []
    for perm in permutations(customers):
        cost = _euclidean(depot, Xc[perm[0]])
        for u, v in pairwise(perm):
            cost += _euclidean(Xc[u], Xc[v])
        cost += _euclidean(Xc[perm[-1]], depot)
        if cost < best_cost:
            best_cost = cost
            best_order = list(perm)
    return best_cost, best_order


class LRPGenerator(BaseInstanceGenerator[LRPInstance]):
    """Generate small enumerated-route LRP instances."""

    def __init__(self, *, seed: int = 1804, max_stops: int = 10, max_routes_per_depot: int = 200) -> None:
        self.default_seed = seed
        self.max_stops = max_stops
        self.max_routes_per_depot = max_routes_per_depot

    def generate_instance(
        self,
        m: int = 10,
        n: int = 10,
        *,
        seed: int | None = None,
    ) -> LRPInstance:
        if self.max_stops <= 0:
            raise ValueError("max_stops must be positive.")
        rng = np.random.default_rng(seed if seed is not None else self.default_seed)

        Xc = rng.random((n, 2))
        Xd = rng.uniform(-0.2, 1.2, size=(m, 2))
        depot_cost = rng.integers(12, 35, size=m).astype(float)

        vehicles = np.full(m, max(1, int(np.ceil(n / (m * self.max_stops)))), dtype=float) * 2
        vehicles = np.maximum(vehicles, 1.0)
        vehicles = np.ceil(vehicles)

        routes: list[list[int]] = []
        route_depot: list[int] = []
        route_cost: list[float] = []

        for j in range(m):
            per_depot = 0
            for rsize in range(1, self.max_stops + 1):
                customer_indices = list(range(n))
                rng.shuffle(customer_indices)
                for comb in combinations(customer_indices, rsize):
                    if per_depot >= self.max_routes_per_depot:
                        break
                    cost, order = _best_route_cost(Xd[j], comb, Xc)
                    routes.append(order)
                    route_depot.append(j)
                    route_cost.append(cost)
                    per_depot += 1
                if per_depot >= self.max_routes_per_depot:
                    break

        coverage = np.zeros(n, dtype=int)
        for cust_list in routes:
            for i in cust_list:
                coverage[i] += 1

        if np.any(coverage == 0):
            uncovered = np.where(coverage == 0)[0]
            depot_dist = np.linalg.norm(Xd[:, None, :] - Xc[None, :, :], axis=2)
            for cust in uncovered:
                depot_idx = int(np.argmin(depot_dist[:, cust]))
                cost, order = _best_route_cost(Xd[depot_idx], (cust,), Xc)
                routes.append(order)
                route_depot.append(depot_idx)
                route_cost.append(cost)
                coverage[cust] += 1

        R = len(route_cost)
        incidence = np.zeros((n, R))
        for r, cust_list in enumerate(routes):
            for i in cust_list:
                incidence[i, r] = 1.0

        return LRPInstance(
            m=m,
            n=n,
            max_stops=self.max_stops,
            max_vehicles=vehicles,
            depot_cost=depot_cost,
            Xd=Xd,
            Xc=Xc,
            routes=routes,
            route_depot=np.asarray(route_depot, dtype=int),
            route_cost=np.asarray(route_cost, dtype=float),
            incidence=incidence,
        )

    def plot_instance(self, inst: LRPInstance, output_path: str | Path | None = None) -> None:
        fig, ax = plt.subplots()
        ax.set_title("LRP instance")
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.3, 1.3)
        ax.scatter(inst.Xc[:, 0], inst.Xc[:, 1], marker="o", s=40, color="tab:red", label="Customers")
        ax.scatter(inst.Xd[:, 0], inst.Xd[:, 1], marker="s", s=70, color="tab:blue", label="Depots")
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
        inst: LRPInstance,
        result: dict[str, Any],
        *,
        output_path: str | Path | None = None,
        open_tol: float = 0.5,
        use_order: bool = True,
    ) -> None:
        vars_dict = result.get("final_vars", {}) or {}
        sub_meta = result.get("sub_metadata", {}) or {}
        master_meta = result.get("master_metadata", {}) or {}

        lam_source = sub_meta.get("lambda_assignment") or master_meta.get("lambda_assignment")
        if lam_source is None:
            lam_arr = np.array([vars_dict.get(f"lambda_{r}", 0.0) or 0.0 for r in range(inst.R)], dtype=float)
        else:
            lam_arr = np.asarray(lam_source, dtype=float).reshape(inst.R)

        y_source = sub_meta.get("y_assignment") or master_meta.get("y_assignment")
        if y_source is None:
            y_arr = np.array([vars_dict.get(f"y_{j}", 0.0) or 0.0 for j in range(inst.m)], dtype=float)
        else:
            y_arr = np.asarray(y_source, dtype=float).reshape(inst.m)

        fig, ax = plt.subplots()
        ax.set_title("LRP solution")
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.3, 1.3)

        for r, lam_val in enumerate(lam_arr):
            if lam_val <= 1e-6:
                continue
            j = int(inst.route_depot[r])
            custs = inst.routes[r]
            if not custs:
                continue
            seq = custs
            pts = [inst.Xd[j]] + [inst.Xc[i] for i in seq] + [inst.Xd[j]]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], linewidth=1.5, alpha=0.7, color="black")

        open_mask = y_arr >= open_tol
        ax.scatter(inst.Xd[~open_mask, 0], inst.Xd[~open_mask, 1], marker="s", s=70, color="lightblue")
        ax.scatter(inst.Xd[open_mask, 0], inst.Xd[open_mask, 1], marker="s", s=70, color="darkblue", label="Depots")
        ax.scatter(inst.Xc[:, 0], inst.Xc[:, 1], marker="o", s=40, color="tab:red", label="Customers")
        ax.legend(loc="upper left")

        if output_path is None:
            plt.show()
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, format="pdf", bbox_inches="tight")
            plt.close(fig)


def generate_lrp_instance(
    seed: int,
    difficulty: str = "easy",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    if difficulty not in DIFFICULTY_CONFIG:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Expected one of {list(DIFFICULTY_CONFIG)}.")
    cfg = DIFFICULTY_CONFIG[difficulty]
    rng = np.random.default_rng(seed)
    m = int(rng.integers(cfg["m_range"][0], cfg["m_range"][1] + 1))
    n = int(rng.integers(cfg["n_range"][0], cfg["n_range"][1] + 1))
    max_stops = cfg["max_stops"]

    gen = LRPGenerator(seed=seed, max_stops=max_stops)
    inst = gen.generate_instance(m=m, n=n, seed=seed)
    data = inst.to_dict()

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")

    return data
