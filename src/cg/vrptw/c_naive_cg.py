"""Naive VRPTW column generation control (protocol v3.1, Section 20.6 "Naive CG"; mirrors src/bd/clrp/e_naive_cg.py).

Root column generation plus an integer restricted master, as a textbook implementation would do it:
- master LP (integer tenths): min sum_r c_r lam_r  s.t.  sum_r a_ir lam_r >= 1 for every customer i (set covering, a
  relaxation of set partitioning), sum_r lam_r - a <= K (the fleet; a is an artificial at a cost above any plan, so the
  LP is feasible even if the seed columns needed more than K routes), lam >= 0; no cuts, no stabilization;
- deterministic seed columns: one single-customer route per customer and the routes of the shared basic initializer
  (heuristic.basic_initializer, Solomon's I1, unpolished; also the first incumbent);
- pricing: the fixed exact oracle (pricing.TWPricer: ng-route labelling with time windows and capacity, ng = 8),
  returning the single best negative column per iteration; duplicates are not added. Strict: a pricing round that
  is not completed exactly (deadline or the pricer's label guards) is discarded and stops column generation;
- a lower bound is published only when a complete exact round finds no negative column: every plan is a partition by
  at most K elementary routes, each of reduced cost >= min(r*, -tol) (r* the exact round's minimum over ng-routes), so
  sum_i pi_i + K (mu + min(r*, -tol)) is valid (mu <= 0 the fleet dual; rounded up to the 0.1 grid);
- the last 20% of the budget is an integer restricted master (set partitioning, binary, at most K routes) over the
  elementary columns; every plan is checked with route_cost before it is published.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.cg.vrptw.a_data import load_vrptw
from src.cg.vrptw.b_mip_xl import route_cost
from src.cg.vrptw.heuristic import basic_initializer
from src.cg.vrptw.pricing import TWPricer


class VRPTWNaiveCG:
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
        inst = load_vrptw(data_path)
        n, K = inst.n, int(inst.vehicles)
        dist = inst.dist
        best: dict[str, Any] = {"obj": math.inf, "routes": None, "first": None}

        def offer(routes: list[list[int]]) -> None:
            obj = route_cost(inst, routes)
            if obj is not None and obj < best["obj"] - 1e-9:
                best.update(obj=obj, routes=routes)
                if best["first"] is None:
                    best["first"] = time.perf_counter() - t0
                self.on_incumbent(routes)

        init = basic_initializer(inst)
        offer(init)
        pricer = TWPricer(inst, ng=self.ng)

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", self.threads)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()
        lp = gp.Model(env=env)
        cover = [lp.addLConstr(gp.LinExpr(), GRB.GREATER_EQUAL, 1.0) for _ in range(n)]  # row i - 1 is customer i
        big = 1.0 + 2.0 * float(dist[0, 1:].sum() + dist[1:, 0].sum() + dist.max(axis=1).sum())
        art = lp.addVar(lb=0.0, obj=big)
        fleet = lp.addLConstr(gp.LinExpr([-1.0], [art]), GRB.LESS_EQUAL, float(K))
        cols: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()

        def length(route: list[int]) -> int:
            seq = [0] + list(route) + [0]
            return int(sum(int(dist[a, b]) for a, b in zip(seq, seq[1:])))

        def add(route: list[int]) -> bool:
            key = tuple(route)
            if key in seen:
                return False
            seen.add(key)
            visits: dict[int, float] = {}
            for i in route:
                visits[i] = visits.get(i, 0.0) + 1.0
            lp.addVar(lb=0.0, obj=float(length(route)),
                      column=gp.Column(list(visits.values()) + [1.0], [cover[i - 1] for i in visits] + [fleet]))
            cols.append(list(route))
            return True

        for i in range(1, n + 1):
            if pricer.out_ok[i - 1] and pricer.in_ok[i - 1]:
                add([i])
        for r in init:
            add(r)

        iters, converged, bound, stop_reason, lp_value = 0, False, None, "deadline", None
        while time.perf_counter() < cg_deadline:
            lp.Params.TimeLimit = max(0.01, cg_deadline - time.perf_counter())
            lp.optimize()
            if lp.Status != GRB.OPTIMAL:
                break
            iters += 1
            lp_value = float(lp.ObjVal)
            pi = np.maximum(np.array(lp.getAttr("Pi", cover)), 0.0)
            mu = min(float(fleet.Pi), 0.0)
            # Strict exact pricing: a round cut short by the deadline or the pricer's resource guards (which would
            # fall back to a beam) is discarded entirely, and column generation stops there.
            res = pricer.price(pi, -mu, pricer.C, pricer.c_out, pricer.c_in, exact=True, max_cols=1,
                               deadline=cg_deadline, tol=self.tol)
            if not res["complete"]:
                stop_reason = "deadline" if time.perf_counter() >= cg_deadline else "pricing_resource_limit"
                break
            added = sum(add([i + 1 for i in path]) for rc, path in res["routes"] if rc < -self.tol)
            if added == 0:
                converged, stop_reason = True, "converged"
                # Duplicates can come back slightly negative (LP tolerances): the exact minimum enters the bound.
                value = float(pi.sum()) + K * (mu + min(float(res["min_rc"]), -self.tol))
                bound = math.ceil(value - 1e-7 * max(1.0, abs(value))) / 10
                self.on_bound(bound)
                break

        # Integer restricted master over the elementary columns with the remaining budget.
        rmp_status = None
        if time.perf_counter() < deadline - 0.01:
            mip = gp.Model(env=env)
            elem = [r for r in cols if len(set(r)) == len(r)]
            lam = [mip.addVar(vtype=GRB.BINARY, obj=float(length(r))) for r in elem]
            rows = [gp.LinExpr() for _ in range(n)]
            for v, r in zip(lam, elem):
                for i in r:
                    rows[i - 1].addTerms(1.0, v)
            for i in range(n):
                mip.addLConstr(rows[i], GRB.EQUAL, 1.0)
            mip.addLConstr(gp.quicksum(lam), GRB.LESS_EQUAL, float(K))
            mip.Params.TimeLimit = max(0.01, deadline - time.perf_counter())

            def decode(values) -> list[list[int]]:
                return [list(r) for v, r in zip(values, elem) if v > 0.5]

            def callback(model, where):
                if time.perf_counter() > deadline:
                    model.terminate()
                    return
                if where == GRB.Callback.MIPSOL:
                    offer(decode(model.cbGetSolution(lam)))

            mip.optimize(callback)
            if mip.SolCount:
                offer(decode(mip.getAttr("X", lam)))
            rmp_status = int(mip.Status)
            mip.dispose()
        lp.dispose()
        env.dispose()
        return {"best_objective": best["obj"] if best["routes"] else None, "global_lb": bound,
                "termination_reason": "time_limit", "cg_iterations": iters, "cg_converged": converged,
                "cg_stop_reason": stop_reason, "columns": len(cols), "lp_value": None if lp_value is None else lp_value / 10,
                "rmp_status": rmp_status, "first_incumbent_seconds": best["first"], "routes": best["routes"],
                "total_seconds": time.perf_counter() - t0}
