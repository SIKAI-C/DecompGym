"""Naive CVRP column generation control (protocol v3.1, Section 20.6 "Naive CG"; mirrors src/bd/clrp/e_naive_cg.py).

Root column generation plus an integer restricted master, as a textbook implementation would do it:
- master LP: min sum_r c_r lam_r s.t. sum_r a_ir lam_r >= 1 for every customer i, lam >= 0 (set covering, a relaxation
  of set partitioning); no cuts, no stabilization, no vehicle-count row;
- deterministic seed columns: one single-customer route per customer and the routes of the shared basic initializer
  (heuristic.basic_initializer: Clarke-Wright savings + 2-opt, unpolished; also the first incumbent);
- pricing: the fixed exact oracle (src.bd.clrp's NGPricer on one depot: ng-route labelling with the capacity resource),
  returning the single best negative column per iteration; duplicates are not added. Strict: a pricing round that is
  not completed exactly (deadline or the pricer's label guards) is discarded and stops column generation;
- a lower bound is published only when a complete exact round finds no negative column: every plan has at most n
  routes, each of reduced cost >= min(r*, -tol) (r* the exact round's minimum), so sum_i pi_i + n min(r*, -tol) is
  valid (rounded up: distances are integral);
- the last 20% of the budget is an integer restricted master (covering, binary) over the elementary columns; a
  covering solution is made a partition by heuristic.repair_cover, and every plan is checked with route_cost before
  it is published.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.clrp.d_ref_cg_xl import NGPricer
from src.cg.cvrp.a_data import load_vrp
from src.cg.cvrp.b_mip_xl import route_cost
from src.cg.cvrp.heuristic import basic_initializer, repair_cover, route_length, single_depot_view


class CVRPNaiveCG:
    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0, ng: int = 8,
                 integer_share: float = 0.2, tol: float = 1e-6) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.ng = time_limit, threads, soft_mem_gb, ng
        self.on_incumbent = on_incumbent or (lambda routes: None)
        self.on_bound = on_bound or (lambda bound: None)
        self.integer_share, self.tol = integer_share, tol

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        cg_deadline = t0 + (1.0 - self.integer_share) * self.time_limit
        inst = load_vrp(data_path)
        dist = inst.distances()
        n, q = inst.n, inst.demand
        best: dict[str, Any] = {"obj": math.inf, "routes": None, "first": None}

        def offer(routes: list[list[int]]) -> None:
            obj = route_cost(inst, routes, dist)
            if obj is not None and obj < best["obj"]:
                best.update(obj=obj, routes=routes)
                if best["first"] is None:
                    best["first"] = time.perf_counter() - t0
                self.on_incumbent(routes)

        init = basic_initializer(inst, dist)
        offer(init)

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", self.threads)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()
        lp = gp.Model(env=env)
        cover = [lp.addLConstr(gp.LinExpr(), GRB.GREATER_EQUAL, 1.0) for _ in range(n)]  # row i - 1 is customer i
        cols: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()

        def add(route: list[int]) -> bool:
            key = tuple(route)
            if key in seen or q[list(route)].sum() > inst.capacity:
                return False
            seen.add(key)
            visits: dict[int, float] = {}
            for i in route:
                visits[i] = visits.get(i, 0.0) + 1.0
            lp.addVar(lb=0.0, obj=float(route_length(dist, route)), column=gp.Column(list(visits.values()), [cover[i - 1] for i in visits]))
            cols.append(list(route))
            return True

        for i in range(1, n + 1):
            add([i])
        for r in init:
            add(r)

        ng = self.ng
        while n * (1 << max(0, ng - 1)) * (inst.capacity + 1) * 8 > 2.5e8 and ng > 3:
            ng -= 1  # memory of the pricer's dominance surface (customers x ng-masks x loads)
        pricer = NGPricer(single_depot_view(inst), dist.astype(float), ng=ng)
        iters, converged, bound, stop_reason, lp_value = 0, False, None, "deadline", None
        while time.perf_counter() < cg_deadline:
            lp.Params.TimeLimit = max(0.01, cg_deadline - time.perf_counter())
            lp.optimize()
            if lp.Status != GRB.OPTIMAL:
                break
            iters += 1
            lp_value = float(lp.ObjVal)
            pi = np.maximum(np.array(lp.getAttr("Pi", cover)), 0.0)
            # Strict exact pricing: a round cut short by the deadline or the pricer's resource guards (which would
            # fall back to a beam) is discarded entirely, and column generation stops there.
            res = pricer.price(pi, 0.0, pricer.C, pricer.c_out[0], pricer.c_in[0], exact=True, max_cols=1,
                               deadline=cg_deadline, tol=self.tol)
            if not res["complete"]:
                stop_reason = "deadline" if time.perf_counter() >= cg_deadline else "pricing_resource_limit"
                break
            added = sum(add([i + 1 for i in path]) for rc, path in res["routes"] if rc < -self.tol)
            if added == 0:
                converged, stop_reason = True, "converged"
                # Duplicates can come back slightly negative (LP tolerances): the exact minimum enters the bound.
                value = float(pi.sum()) + n * min(float(res["min_rc"]), -self.tol)
                bound = float(math.ceil(value - 1e-7 * max(1.0, abs(value))))
                self.on_bound(bound)
                break

        # Integer restricted master over the elementary columns with the remaining budget.
        rmp_status = None
        if time.perf_counter() < deadline - 0.01:
            lp.update()
            mip = lp.copy()
            lv = mip.getVars()
            for v, route in zip(lv, cols):
                v.VType = GRB.BINARY
                if len(set(route)) < len(route):
                    v.UB = 0.0
            mip.Params.TimeLimit = max(0.01, deadline - time.perf_counter())

            def decode(values) -> list[list[int]]:
                return repair_cover([r for v, r in zip(values, cols) if v > 0.5], dist)

            def callback(model, where):
                if time.perf_counter() > deadline:
                    model.terminate()
                    return
                if where == GRB.Callback.MIPSOL:
                    offer(decode(model.cbGetSolution(lv)))

            mip.optimize(callback)
            if mip.SolCount:
                offer(decode(mip.getAttr("X", lv)))
            rmp_status = int(mip.Status)
            mip.dispose()
        lp.dispose()
        env.dispose()
        return {"best_objective": best["obj"] if best["routes"] else None, "global_lb": bound,
                "termination_reason": "time_limit", "cg_iterations": iters, "cg_converged": converged,
                "cg_stop_reason": stop_reason, "columns": len(cols), "lp_value": lp_value, "rmp_status": rmp_status,
                "ng": ng, "first_incumbent_seconds": best["first"], "routes": best["routes"],
                "total_seconds": time.perf_counter() - t0}
