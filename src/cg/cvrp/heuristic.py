"""Shared CVRP primal helpers (scale-up study): the basic initializer shared by the naive and the engineered column
generation, the repair of a covering plan into a partition, and the engineered reference's local search.

Routes are lists of customer numbers 1..n (a_data's numbering); the distance matrix is a_data's (depot = row 0).
The CLRP helpers are reused through a one-depot view (depot 0, customers 0..n-1 at matrix rows 1..n, no vehicle,
opening or depot-capacity costs): Clarke-Wright savings, intra-route 2-opt and the inter-route local search.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bd.clrp.c_heuristic import savings, two_opt
from src.cg.cvrp.a_data import CVRPInstance


@dataclass
class SingleDepotView:
    """The attributes of a CLRPInstance that src.bd.clrp's NGPricer and LocalSearch read, for one uncapacitated depot
    without opening or vehicle costs (customer k of the view is customer k + 1 of the CVRP instance)."""
    m: int
    n: int
    vehicle_capacity: int
    demand: np.ndarray
    vehicle_cost: float
    opening_cost: np.ndarray
    depot_capacity: np.ndarray


def single_depot_view(inst: CVRPInstance) -> SingleDepotView:
    return SingleDepotView(1, inst.n, int(inst.capacity), inst.demand[1:].astype(np.int64), 0.0, np.zeros(1),
                           np.array([float(inst.demand.sum()) + 1.0]))


def route_length(dist: np.ndarray, route: list[int]) -> int:
    seq = [0] + list(route) + [0]
    return int(sum(dist[a, b] for a, b in zip(seq, seq[1:])))


def basic_initializer(inst: CVRPInstance, dist: np.ndarray) -> list[list[int]]:
    """Clarke-Wright parallel savings, then 2-opt inside every route (deterministic)."""
    custs = list(range(1, inst.n + 1))
    routes = savings(0, custs, dist, inst.demand[1:], inst.capacity, 1)
    return [two_opt(0, list(r), dist) for r in routes if r]


def repair_cover(routes: list[list[int]], dist: np.ndarray) -> list[list[int]]:
    """A covering plan made a partition: each repeated customer stays only in the route where removing it would save
    least (rounded distances can break the triangle inequality by one unit, so a removal may cost; the result is
    always feasible and is re-evaluated by the caller)."""
    routes = [list(r) for r in routes if r]
    count: dict[int, int] = {}
    for r in routes:
        for i in r:
            count[i] = count.get(i, 0) + 1
    for i, k in count.items():
        while k > 1:
            best = None
            for t, r in enumerate(routes):
                if i in r:
                    s = [0] + r + [0]
                    p = s.index(i)
                    save = int(dist[s[p - 1], i] + dist[i, s[p + 1]] - dist[s[p - 1], s[p + 1]])
                    if best is None or save > best[0]:
                        best = (save, t)
            routes[best[1]].remove(i)
            k -= 1
    return [r for r in routes if r]


class CVRPLocalSearch:
    """The CLRP reference's inter-route local search on one depot: granular relocate and swap, 2-opt* tail exchanges
    and 2-opt inside every changed route (route capacity kept; emptied routes disappear)."""

    def __init__(self, inst: CVRPInstance, dist: np.ndarray, near: int = 15) -> None:
        from src.bd.clrp.d_ref_cg_xl import LocalSearch
        self.ls = LocalSearch(single_depot_view(inst), dist.astype(float), near=near)

    def improve(self, routes: list[list[int]], deadline: float) -> list[list[int]]:
        plan = [(0, [i - 1 for i in r]) for r in routes if r]
        out = self.ls.improve(plan, deadline)
        return [[i + 1 for i in p] for _, p in out if p]

