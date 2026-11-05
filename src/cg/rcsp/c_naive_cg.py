"""Naive RCSP column generation control (protocol v3.1, Section 20.6 "Naive CG").

Root column generation plus an integer restricted master, as a textbook implementation would do it:
- master LP: min sum_j c_j lam_j s.t. sum_{j covers t} lam_j >= 1 for every task t, lam >= 0; no stabilization,
  no column management, no cuts;
- deterministic seed columns: the duties of the shared basic initializer (pricing.greedy_cover, also the first
  incumbent), so the master is feasible from the start;
- pricing: the fixed exact oracle (pricing.DutyPricer in exact mode: resource-constrained shortest path with the
  duty-length, time-without-break and break-taken resources), returning the single most negative duty per crew base
  per iteration; duplicates are not added. Strict: a pricing round that is not completed exactly (deadline or label
  cap) is discarded and stops column generation;
- a lower bound is published only when a complete exact round finds no negative duty (the converged LP value, minus
  the pricing tolerance times the largest possible number of duties, rounded up since costs are integral);
- the last 20% of the budget is an integer restricted master over all generated columns; every plan is checked with
  plan_cost before publication.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.cg.rcsp.a_data import load_rcsp
from src.cg.rcsp.b_mip_xl import duty_check, plan_cost
from src.cg.rcsp.pricing import DutyPricer, greedy_cover


class RCSPNaiveCG:
    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 integer_share: float = 0.2, tol: float = 1e-4) -> None:
        self.time_limit, self.threads, self.soft_mem_gb = time_limit, threads, soft_mem_gb
        self.on_incumbent = on_incumbent or (lambda duties: None)
        self.on_bound = on_bound or (lambda bound: None)
        self.integer_share, self.tol = integer_share, tol

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        cg_deadline = t0 + (1.0 - self.integer_share) * self.time_limit
        inst = load_rcsp(data_path)
        pricer = DutyPricer(inst)
        best: dict[str, Any] = {"obj": math.inf, "duties": None}

        def offer(duties: list[dict[str, Any]]) -> None:
            obj = plan_cost(inst, duties)
            if obj is not None and obj < best["obj"] - 1e-9:
                best.update(obj=obj, duties=duties)
                self.on_incumbent(duties)

        init = greedy_cover(pricer, deadline)
        if init:
            offer(init)
        out = {"cg_iterations": 0, "cg_converged": False, "cg_stop_reason": "no_initial_cover", "columns": 0, "lp_value": None}
        bound = None
        if init is None:
            return {"best_objective": None, "global_lb": None, "termination_reason": "infeasible_or_deadline", "duties": None,
                    **out, "total_seconds": time.perf_counter() - t0}
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", self.threads)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()
        lp = gp.Model(env=env)
        cover = [lp.addLConstr(gp.LinExpr(), GRB.GREATER_EQUAL, 1.0) for _ in range(inst.n)]
        cols: list[tuple[int, tuple[int, ...]]] = []
        seen: set[tuple[int, tuple[int, ...]]] = set()

        def add(d: int, tasks: list[int]) -> bool:
            key = (int(d), tuple(int(t) for t in tasks))
            if key in seen:
                return False
            cost, _ = duty_check(inst, key[0], list(key[1]))
            if cost is None:  # cannot happen with the exact oracle; never add an infeasible column
                return False
            seen.add(key)
            lp.addVar(lb=0.0, obj=cost, column=gp.Column([1.0] * len(key[1]), [cover[t] for t in key[1]]))
            cols.append(key)
            return True

        for duty in init:
            add(duty["depot"], duty["tasks"])
        kappa = best["obj"] / float(np.min(pricer.src_cost[np.isfinite(pricer.src_cost)]))
        iters, converged, stop_reason = 0, False, "deadline"
        while time.perf_counter() < cg_deadline:
            lp.Params.TimeLimit = max(0.01, cg_deadline - time.perf_counter())
            lp.optimize()
            if lp.Status != GRB.OPTIMAL:
                break
            iters += 1
            pi = np.maximum(np.array(lp.getAttr("Pi", cover)), 0.0)
            res = pricer.price(pi, exact=True, max_cols=pricer.D, per_depot=1, deadline=cg_deadline, tol=self.tol)
            if not res["complete"]:
                stop_reason = "deadline" if time.perf_counter() >= cg_deadline else "pricing_label_limit"
                break
            added = sum(add(d, tasks) for _, d, tasks in res["cols"])
            if added == 0:
                converged, stop_reason = True, "converged"
                # Lagrangian bound at these duals: every duty has reduced cost >= min_rc - tol (exact round), and an
                # optimal LP solution uses at most kappa = UB / (cheapest duty) duties.
                value = float(pi.sum()) + kappa * (min(0.0, float(res["min_rc"].min())) - self.tol)
                bound = float(math.ceil(value - 1e-7 * max(1.0, abs(value))))
                self.on_bound(bound)
                break
        out.update(cg_iterations=iters, cg_converged=converged, cg_stop_reason=stop_reason, columns=len(cols),
                   lp_value=float(lp.ObjVal) if lp.SolCount else None)

        # Integer restricted master over all generated columns with the remaining budget.
        if time.perf_counter() < deadline - 0.01:
            lp.update()
            mip = lp.copy()
            lv = mip.getVars()
            for v in lv:
                v.VType = GRB.BINARY
            mip.Params.TimeLimit = max(0.01, deadline - time.perf_counter())

            def decode(values) -> list[dict[str, Any]]:
                return [{"depot": d, "tasks": list(t)} for v, (d, t) in zip(values, cols) if v > 0.5]

            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    offer(decode(model.cbGetSolution(lv)))

            mip.optimize(callback)
            if mip.SolCount:
                offer(decode(mip.getAttr("X", lv)))
            out["rmp_status"] = int(mip.Status)
            mip.dispose()
        lp.dispose()
        env.dispose()
        return {"best_objective": best["obj"] if best["duties"] else None, "global_lb": bound, "termination_reason": "time_limit",
                "duties": best["duties"], **out, "pricing_seconds": pricer.stats["seconds"], "total_seconds": time.perf_counter() - t0}
