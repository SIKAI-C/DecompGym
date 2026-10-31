"""Naive BPPC column generation control (protocol v3.1, Section 20.6 "Naive CG"; mirrors src/cg/cvrp/c_naive_cg.py).

Root column generation plus an integer restricted master, as a textbook implementation would do it:
- master LP: min sum_S lam_S s.t. sum_{S containing i} lam_S >= 1 for every item i, lam >= 0 (set covering; bins are
  closed under taking subsets, so covering and partitioning have the same optimum); no stabilization;
- deterministic seed columns: one single-item bin per item and the bins of the shared basic initializer
  (heuristic.basic_initializer: first fit decreasing with conflicts; also the first incumbent);
- pricing: the fixed exact oracle (pricing.KPCOracle, the engineered reference's exact oracle), returning the single
  best bin of value above 1 + tol per iteration; duplicates are not added. Strict: a pricing round that is not
  completed exactly (deadline) is discarded and stops column generation;
- a lower bound is published only when a complete exact round finds no improving bin: Farley's bound
  sum_i pi_i / max(1, v) with v the oracle's upper bound on the pricing optimum (pi / v is dual feasible for the full
  master), rounded up (the bin count is integral);
- the last 20% of the budget is an integer restricted master (covering, binary) over the generated bins; a covering
  solution is trimmed into a partition (heuristic.trim_cover), and every plan is checked with bin_count before it is
  published. termination_reason "finished_early" if the integer master ends before the deadline (allowed by 20.6).
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.cg.bppc.a_data import load_bppc
from src.cg.bppc.b_mip_xl import bin_count, public
from src.cg.bppc.heuristic import basic_initializer, edge_clique_cover, trim_cover
from src.cg.bppc.pricing import KPCOracle


class BPPCNaiveCG:
    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 integer_share: float = 0.2, tol: float = 1e-6) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.on_incumbent = on_incumbent or (lambda bins: None)
        self.on_bound = on_bound or (lambda bound: None)
        self.integer_share, self.tol = integer_share, tol

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        cg_deadline = t0 + (1.0 - self.integer_share) * self.time_limit
        inst = load_bppc(data_path)
        n = inst.n
        best: dict[str, Any] = {"obj": math.inf, "bins": None, "first": None}

        def offer(bins: list[list[int]]) -> None:
            pub = public(bins)
            obj = bin_count(inst, pub)
            if obj is not None and obj < best["obj"] and time.perf_counter() < deadline:  # never after the deadline
                best.update(obj=obj, bins=pub)
                if best["first"] is None:
                    best["first"] = time.perf_counter() - t0
                self.on_incumbent(pub)

        init = basic_initializer(inst)
        offer(init)

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", self.threads)
        env.setParam("Seed", self.seed)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()
        lp = gp.Model(env=env)
        cover = [lp.addLConstr(gp.LinExpr(), GRB.GREATER_EQUAL, 1.0) for _ in range(n)]
        cols: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()

        def add(items) -> bool:
            key = tuple(sorted(int(i) for i in items))
            if not key or key in seen:
                return False
            seen.add(key)
            lp.addVar(lb=0.0, obj=1.0, column=gp.Column([1.0] * len(key), [cover[i] for i in key]))
            cols.append(key)
            return True

        for i in range(n):
            add([i])
        for b in init:
            add(b)

        oracle = KPCOracle(inst.weights, inst.capacity, inst.adj, edge_clique_cover(inst.adj), env, tol=self.tol)
        iters, converged, bound, stop_reason, lp_value = 0, False, None, "deadline", None
        while time.perf_counter() < cg_deadline:
            lp.Params.TimeLimit = max(0.01, cg_deadline - time.perf_counter())
            lp.optimize()
            if lp.Status != GRB.OPTIMAL:
                break
            iters += 1
            lp_value = float(lp.ObjVal)
            pi = np.maximum(np.array(lp.getAttr("Pi", cover)), 0.0)
            # Strict exact pricing: a round cut short by the deadline is discarded and column generation stops.
            res = oracle.solve(pi, max_cols=1, deadline=cg_deadline)
            if not res["complete"]:
                stop_reason = "deadline" if time.perf_counter() >= cg_deadline else "pricing_incomplete"
                break
            if res["cols"] and add(res["cols"][0][1]):
                continue
            converged, stop_reason = True, "converged"
            v = max(1.0, float(res["bound"]))
            bound = float(math.ceil(float(pi.sum()) / v - 1e-6))
            if best["obj"] < math.inf:
                bound = min(bound, float(best["obj"]))
            self.on_bound(bound)
            break

        # Integer restricted master over the generated bins with the remaining budget.
        rmp_status = None
        if time.perf_counter() < deadline - 0.01:
            mip = gp.Model(env=env)
            lam = list(mip.addVars(len(cols), vtype=GRB.BINARY, obj=1.0).values())
            rows = [gp.LinExpr() for _ in range(n)]
            for v_, key in zip(lam, cols):
                for i in key:
                    rows[i].addTerms(1.0, v_)
            for i in range(n):
                mip.addLConstr(rows[i], GRB.GREATER_EQUAL, 1.0)
            mip.Params.TimeLimit = max(0.01, deadline - time.perf_counter())

            def decode(values) -> list[list[int]]:
                return trim_cover([list(key) for val, key in zip(values, cols) if val > 0.5])

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
        oracle.dispose()
        lp.dispose()
        env.dispose()
        return {"best_objective": best["obj"] if best["bins"] else None, "global_lb": bound,
                "termination_reason": "time_limit" if time.perf_counter() >= deadline - 1.0 else "finished_early",
                "cg_iterations": iters, "cg_converged": converged, "cg_stop_reason": stop_reason, "columns": len(cols),
                "lp_value": lp_value, "rmp_status": rmp_status, "pricing_calls": oracle.calls,
                "pricing_seconds": oracle.seconds, "first_incumbent_seconds": best["first"], "bins": best["bins"],
                "total_seconds": time.perf_counter() - t0}
