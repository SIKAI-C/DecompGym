"""Decomposition-free CVRP control (protocol v4 Section 6, "no grandfathering": a strong standalone heuristic).

Ruin-and-recreate simulated annealing (src.cg.clrp.f_sisr_xl, SISR-style; one uncapacitated depot, no vehicle or
opening costs) started from the shared basic initializer (Clarke-Wright savings + 2-opt, the same first plan as the
naive and the engineered column generation), with the engineered reference's inter-route local search
(heuristic.CVRPLocalSearch: granular relocate / swap, 2-opt*, intra-route 2-opt) polishing new best plans at most
every `polish_every` seconds. No LP, MIP or column generation; Python 3 + NumPy only.

Interface as the decomposition references: CVRPRuinRecreate(time_limit=..., on_incumbent=...).run(path) publishes every
improving plan (routes of customers 1..n, re-checked with b_mip_xl.route_cost before publication, rate-limited to one
publication per `publish_every` seconds with the pending best always flushed before returning) and returns a result
dict. The search stops `publish_reserve` seconds (<= 0.5 s) before the deadline so that the pending best is always
flushed before it (Codex review CG-1 #3); nothing is published at or after the deadline.
"""
from __future__ import annotations

import inspect
import math
import time
from typing import Any, Callable

import numpy as np

from src.cg.clrp.f_sisr_xl import RoutingModel, RuinRecreate
from src.cg.cvrp.a_data import CVRPInstance, load_vrp
from src.cg.cvrp.b_mip_xl import route_cost
from src.cg.cvrp.heuristic import CVRPLocalSearch, basic_initializer


def routing_model(inst: CVRPInstance, dist: np.ndarray) -> RoutingModel:
    """Depot = node 0, customer i = node i (m = 1, so the CVRP numbering is the engine's node numbering)."""
    return RoutingModel(dist.astype(float), inst.demand[1:].astype(float), inst.capacity, 1)


class Publisher:
    """Rate-limited publication of improving plans: a better plan is published at once if the last publication is at
    least `every` seconds old, otherwise kept pending and published by `tick`/`flush`; never at or after `deadline`."""

    def __init__(self, hook: Callable, deadline: float, every: float = 0.25, stage: str = "ls") -> None:
        self.hook, self.deadline, self.every, self.stage = hook, deadline, every, stage
        try:
            params = inspect.signature(hook).parameters
            self.takes_stage = "stage" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())
        except (TypeError, ValueError):
            self.takes_stage = False
        self.last = -math.inf
        self.pending: Any = None
        self.published = 0

    def offer(self, solution: Any) -> None:
        self.pending = solution
        self.tick()

    def tick(self, force: bool = False) -> None:
        now = time.perf_counter()
        if self.pending is None or now >= self.deadline or (not force and now - self.last < self.every):
            return
        sol, self.pending = self.pending, None
        self.hook(sol, stage=self.stage) if self.takes_stage else self.hook(sol)
        self.last = now
        self.published += 1

    def flush(self) -> None:
        self.tick(force=True)


class CVRPRuinRecreate:
    """The decomposition-free CVRP control of the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 T0: float = 100.0, Tf: float = 1.0, polish_every: float = 2.0, publish_every: float = 0.25,
                 max_iterations: int | None = None) -> None:
        self.time_limit, self.threads, self.seed = time_limit, threads, seed
        self.on_incumbent = on_incumbent or (lambda routes: None)
        self.T0, self.Tf, self.polish_every, self.publish_every = T0, Tf, polish_every, publish_every
        self.max_iterations = max_iterations

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        search_deadline = deadline - min(0.5, 0.05 * self.time_limit)   # reserve for the final flush
        inst = load_vrp(data_path)
        dist = inst.distances()
        model = routing_model(inst, dist)
        ls = CVRPLocalSearch(inst, dist)
        pub = Publisher(self.on_incumbent, deadline, self.publish_every, stage="ruin_recreate")
        best = {"obj": math.inf, "routes": None, "t": None, "first": None}

        def offer(routes_1n: list[list[int]]) -> None:
            obj = route_cost(inst, routes_1n, dist)
            if obj is None or obj >= best["obj"] or time.perf_counter() >= deadline:   # the publication deadline
                return
            best.update(obj=obj, routes=routes_1n, t=time.perf_counter() - t0)
            if best["first"] is None:
                best["first"] = best["t"]
            pub.offer(routes_1n)

        init = basic_initializer(inst, dist)
        offer(init)
        init = ls.improve(init, min(search_deadline, time.perf_counter() + 30.0))
        offer(init)

        def polish(routes, dl):   # engine routes [0, [nodes 1..n]] <-> lists of customers 1..n (same numbers)
            out = ls.improve([p for _, p in routes], min(dl, time.perf_counter() + 10.0))
            return [[0, list(p)] for p in out if p]

        rr = RuinRecreate(model, seed=self.seed, T0=self.T0, Tf=self.Tf)
        start = [[0, list(r)] for r in (best["routes"] or init)]
        rr.run(start, search_deadline, on_best=lambda routes, cost: (offer([list(p) for _, p in routes]), pub.tick()),
               polish=polish, polish_every=self.polish_every, max_iterations=self.max_iterations, tick=pub.tick)
        pub.flush()
        return {"best_objective": best["obj"] if best["routes"] else None, "routes": best["routes"],
                "termination_reason": "time_limit" if self.max_iterations is None else "iteration_limit",
                "first_incumbent_seconds": best["first"], "best_seconds": best["t"], "iterations": rr.iterations,
                "accepted": rr.accepted, "publications": pub.published, "seed": self.seed,
                "total_seconds": time.perf_counter() - t0}
