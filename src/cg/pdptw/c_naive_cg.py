"""Naive PDPTW column generation control (protocol v3.1, Section 20.6 "Naive CG").

Root column generation plus an integer restricted master, as a textbook implementation would do it:
- master LP: min sum_k c_k lam_k  s.t.  sum_{k serves r} lam_k >= 1 for every request r (pi_r >= 0),
  sum_k lam_k - a <= K (fleet; mu <= 0), lam, a >= 0, where the artificial fleet slack a costs more than any plan
  (a valid phase I: it only enlarges the master); no stabilization, no column management, no cuts;
- deterministic seed columns: one single-request route 0 -> p -> d -> 0 per request and the routes of the shared basic
  initializer (pricing.insertion_plan, unpolished; also the first incumbent);
- pricing: the fixed exact oracle (pricing.PDPPricer, exact mode: elementary labelling with open-request sets, time
  windows and capacity), adding the single most negative route per iteration (one block: a homogeneous fleet);
  duplicates are not added. Strict: a pricing round that is not completed exactly (deadline or label cap) is
  discarded and stops column generation;
- a lower bound is published only when a complete exact round finds no negative route: the Lagrangian value
  sum_r pi_r + K mu + K min(0, min_rc) at those duals (the LP value at convergence), rounded up after a floating-point
  allowance (costs are integral);
- the last 20% of the budget is an integer restricted master (set partitioning over the generated columns, binary,
  fleet row, no artificial); every plan is checked with plan_cost before publication.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.cg.pdptw.a_data import load_pdptw, plan_cost, route_check
from src.cg.pdptw.pricing import PDPPricer, insertion_plan


class PDPTWNaiveCG:
    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 integer_share: float = 0.2, tol: float = 1e-6) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.on_incumbent = on_incumbent or (lambda routes: None)
        self.on_bound = on_bound or (lambda bound: None)
        self.integer_share, self.tol = integer_share, tol

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        cg_deadline = t0 + (1.0 - self.integer_share) * self.time_limit
        inst = load_pdptw(data_path)
        dist = inst.distances()
        c = dist.tolist()
        n, K = inst.n, inst.vehicles
        pricer = PDPPricer(inst, dist)
        best: dict[str, Any] = {"obj": math.inf, "routes": None, "first_t": None}

        def offer(routes: list[list[int]]) -> None:
            obj = plan_cost(inst, c, routes)
            if obj is not None and obj < best["obj"]:
                best.update(obj=obj, routes=routes)
                if best["first_t"] is None:
                    best["first_t"] = time.perf_counter() - t0
                self.on_incumbent(routes)

        init = insertion_plan(inst, c)
        if init:
            offer(init)

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", self.threads)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()
        lp = gp.Model(env=env)
        cover = [lp.addLConstr(gp.LinExpr(), GRB.GREATER_EQUAL, 1.0) for _ in range(n)]
        fleet = lp.addLConstr(gp.LinExpr(), GRB.LESS_EQUAL, float(K))
        singles = [route_check(inst, c, [p, d])[0] for p, d in zip(inst.pickups, inst.deliveries)]
        big = float(sum(x for x in singles if x is not None)) + 1.0
        art = lp.addVar(lb=0.0, obj=big, column=gp.Column([-1.0], [fleet]))  # the first variable of the master
        cols: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()

        def add(route: list[int]) -> bool:
            key = tuple(int(v) for v in route)
            if key in seen:
                return False
            cost, _ = route_check(inst, c, list(key))
            if cost is None:  # cannot happen with the exact oracle; never add an infeasible column
                return False
            seen.add(key)
            reqs = sorted({inst.request_of[v] for v in key})
            lp.addVar(lb=0.0, obj=float(cost), column=gp.Column([1.0] * len(reqs) + [1.0], [cover[r] for r in reqs] + [fleet]))
            cols.append(key)
            return True

        for p, d in zip(inst.pickups, inst.deliveries):
            add([p, d])
        for route in init or []:
            add(route)
        iters, converged, bound, stop_reason, lp_value = 0, False, None, "deadline", None
        while time.perf_counter() < cg_deadline:
            lp.Params.TimeLimit = max(0.01, cg_deadline - time.perf_counter())
            lp.optimize()
            if lp.Status != GRB.OPTIMAL:
                stop_reason = "lp_not_optimal"
                break
            iters += 1
            lp_value = float(lp.ObjVal)
            pi = np.maximum(np.array(lp.getAttr("Pi", cover)), 0.0)
            mu = min(float(fleet.Pi), 0.0)
            res = pricer.price(pi, -mu, exact=True, max_cols=1, deadline=cg_deadline, tol=self.tol)
            if not res["complete"]:
                stop_reason = "deadline" if time.perf_counter() >= cg_deadline else "pricing_label_limit"
                break
            added = sum(add(route) for _, route in res["routes"])
            if added == 0:
                converged, stop_reason = True, "converged"
                min_rc = min(0.0, float(res["min_rc"]))
                value = float(pi.sum()) + K * mu + K * min_rc
                scale = float(pi.sum()) + K * abs(mu) + K * abs(min_rc)
                bound = float(math.ceil(value - 1e-7 * max(1.0, scale)))
                self.on_bound(bound)
                break
        art_value = float(art.X) if lp.SolCount else None

        # Integer restricted master (set partitioning over the generated columns) with the remaining budget.
        rmp_status = None
        if time.perf_counter() < deadline - 0.01:
            lp.update()
            mip = lp.copy()
            mvars = mip.getVars()
            a, lam = mvars[0], mvars[1:]
            a.UB = 0.0
            for v in lam:
                v.VType = GRB.BINARY
            for row in mip.getConstrs()[:n]:
                row.Sense = GRB.EQUAL
            mip.Params.TimeLimit = max(0.01, deadline - time.perf_counter())
            mip.Params.Seed = self.seed

            def decode(values) -> list[list[int]]:
                return [list(cols[k]) for k, v in enumerate(values) if v > 0.5]

            def callback(model, where):
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
                "termination_reason": "time_limit", "routes": best["routes"], "cg_iterations": iters,
                "cg_converged": converged, "cg_stop_reason": stop_reason, "columns": len(cols), "lp_value": lp_value,
                "artificial": art_value, "rmp_status": rmp_status, "first_incumbent_seconds": best["first_t"],
                "pricing_seconds": pricer.stats["seconds"], "labels": pricer.stats["labels"],
                "total_seconds": time.perf_counter() - t0}
