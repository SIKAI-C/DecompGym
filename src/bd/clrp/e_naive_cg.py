"""Naive CLRP column generation control (protocol v3.1, Section 20.6 "Naive CG").

Root column generation plus an integer restricted master:
- master LP: min sum_r c_r lambda_r + sum_d O_d y_d, each customer covered (>= 1), per-depot capacity
  sum_r q_r lambda_r <= W_d y_d, 0 <= y_d <= 1; no cuts, no stabilization, no linking strengthening;
- deterministic seed columns: one single-customer route per (depot, customer) and the routes of the shared
  basic initializer (regret assignment + savings on the cheapest-capacity depot set, unpolished);
- pricing: the exact ng-route labeling oracle per depot, returning one best negative column per depot per
  iteration;
- a lower bound is published only when a full exact pricing round finds no negative column (the converged
  master LP over an ng-relaxation is a valid relaxation);
- the last 20% of the budget is an integer restricted master over the elementary columns; a covering
  solution is turned into a partition by dropping repeated visits (valid because costs satisfy the
  triangle inequality), and every plan is checked with `route_cost` before publication.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.clrp.a_data import load_prodhon
from src.bd.clrp.b_mip_xl import route_cost
from src.bd.clrp.c_heuristic import plan_for
from src.bd.clrp.d_ref_cg_xl import NGPricer, repair_cover


class CLRPNaiveCG:
    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0, ng: int = 8,
                 integer_share: float = 0.2) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.ng = time_limit, threads, soft_mem_gb, ng
        self.on_incumbent = on_incumbent or (lambda routes: None)
        self.on_bound = on_bound or (lambda bound: None)
        self.integer_share = integer_share

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        cg_deadline = t0 + (1.0 - self.integer_share) * self.time_limit
        inst = load_prodhon(data_path)
        cost = inst.travel("ceil")
        m, n, q, F = inst.m, inst.n, inst.demand, float(inst.vehicle_cost)
        best = {"obj": float("inf"), "routes": None}

        def offer(routes: list[dict[str, Any]]) -> None:
            obj = route_cost(inst, cost, routes)
            if obj is not None and obj < best["obj"] - 1e-9:
                best.update(obj=obj, routes=routes)
                self.on_incumbent(routes)

        # Shared basic initializer (unpolished).
        order = sorted(range(m), key=lambda d: inst.opening_cost[d] / inst.depot_capacity[d])
        start, cap = [], 0.0
        for d in order:
            start.append(d)
            cap += inst.depot_capacity[d]
            if cap >= q.sum():
                break
        init = plan_for(inst, cost, start)
        if init:
            offer(init)

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", self.threads)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()
        lp = gp.Model(env=env)
        y = [lp.addVar(lb=0.0, ub=1.0, obj=float(inst.opening_cost[d])) for d in range(m)]
        cover = [lp.addLConstr(gp.LinExpr(), GRB.GREATER_EQUAL, 1.0) for _ in range(n)]
        capr = [lp.addLConstr(gp.LinExpr([(-float(inst.depot_capacity[d]), y[d])]), GRB.LESS_EQUAL, 0.0) for d in range(m)]
        cols: list[tuple[int, list[int]]] = []
        lam: list[gp.Var] = []
        seen: set[tuple[int, tuple[int, ...]]] = set()

        def travel(d: int, path: list[int]) -> float:
            seq = [d] + [m + i for i in path] + [d]
            return float(sum(cost[a, b] for a, b in zip(seq, seq[1:])))

        def add(d: int, path: list[int]) -> bool:
            key = (d, tuple(path))
            if key in seen:
                return False
            seen.add(key)
            visits: dict[int, float] = {}
            for i in path:
                visits[i] = visits.get(i, 0.0) + 1.0
            col = gp.Column([visits[i] for i in visits] + [float(q[path].sum())], [cover[i] for i in visits] + [capr[d]])
            lam.append(lp.addVar(lb=0.0, obj=travel(d, path) + F, column=col))
            cols.append((d, list(path)))
            return True

        for d in range(m):
            for i in range(n):
                if q[i] <= inst.vehicle_capacity:
                    add(d, [i])
        for r in init or []:
            add(r["depot"], r["customers"])

        pricer = NGPricer(inst, cost, ng=self.ng)
        iters, converged, bound = 0, False, None
        while time.perf_counter() < cg_deadline:
            lp.Params.TimeLimit = max(0.01, cg_deadline - time.perf_counter())
            lp.optimize()
            if lp.Status != GRB.OPTIMAL:
                break
            iters += 1
            pi = np.array(lp.getAttr("Pi", cover))
            nu = np.array(lp.getAttr("Pi", capr))
            added, complete = 0, True
            for d in range(m):
                res = pricer.price(pi + nu[d] * q, F, pricer.C, pricer.c_out[d], pricer.c_in[d], exact=True,
                                   max_cols=1, deadline=cg_deadline)
                complete &= res["complete"]
                for rc, path in res["routes"]:
                    if rc < -1e-6:
                        added += add(d, path)
            if added == 0:
                if complete:
                    converged, bound = True, float(lp.ObjVal)
                    self.on_bound(bound)
                break

        # Integer restricted master over the elementary columns with the remaining budget.
        if time.perf_counter() < deadline - 0.01:
            lp.update()
            mip = lp.copy()
            yv, lv = mip.getVars()[:m], mip.getVars()[m:]
            for v in yv:
                v.VType = GRB.BINARY
            for v, (d, path) in zip(lv, cols):
                v.VType = GRB.BINARY
                if len(set(path)) < len(path):
                    v.UB = 0.0
            mip.Params.TimeLimit = max(0.01, deadline - time.perf_counter())

            def decode(values) -> list[dict[str, Any]]:
                chosen = [(d, list(p)) for v, (d, p) in zip(values, cols) if v > 0.5]
                return [{"depot": d, "customers": p} for d, p in repair_cover(chosen, cost.tolist(), m) if p]

            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    offer(decode(model.cbGetSolution(lv)))

            mip.optimize(callback)
            if mip.SolCount:
                offer(decode(mip.getAttr("X", lv)))
            mip.dispose()
        env.dispose()
        return {"best_objective": best["obj"] if best["routes"] else None, "global_lb": bound,
                "termination_reason": "time_limit", "cg_iterations": iters, "cg_converged": converged,
                "columns": len(cols), "routes": best["routes"], "total_seconds": time.perf_counter() - t0}
