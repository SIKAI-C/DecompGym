"""Naive vertex-coloring column generation control (protocol v3.1, Section 20.6 "Naive CG").

Root column generation plus an integer restricted master, as a textbook implementation would do it (Mehrotra and
Trick 1996 without their engineering):
- master LP: min sum_S lam_S s.t. sum_{S contains v} lam_S >= 1 for every vertex v, lam >= 0 (set covering over
  independent sets; a subset of an independent set is independent, so covering and partitioning have the same
  integer optimum); no stabilization, no column management;
- deterministic seed columns: the color classes of the shared basic initializer (heuristic.dsatur, also the first
  incumbent), so the master is feasible from the start;
- pricing: a fixed exact oracle, the edge-formulation MIP for the maximum-weight independent set
  (max sum_v pi_v y_v s.t. y_u + y_v <= 1 for every edge, y binary; Gurobi with MIPGap 0), returning the single
  optimal set per iteration (as returned, not extended); duplicates are not added. Strict: a pricing solve that does
  not finish with a proven optimum (deadline) is discarded and stops column generation;
- a lower bound is published only when an exact round finds no column of weight > 1 + tol (convergence): the Farley
  bound sum_v pi_v / max(1, w*) with the proven maximum weight w* (valid for every pi >= 0), made safe against
  floating-point error and rounded up (the objective is integral);
- the last 20% of the budget is an integer restricted master (covering, binary) over all generated columns; a
  covering solution is turned into a coloring (each vertex takes the first chosen set containing it) and checked with
  color_count before it is published.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from src.cg.coloring.a_data import load_col
from src.cg.coloring.b_mip_xl import color_count
from src.cg.coloring.heuristic import canonical, classes_of, coloring_from_cover, dsatur


def farley_bound(pi_sum: float, w_ub: float) -> int:
    """ceil(sum pi / max(1, w_ub)) made safe: w_ub is inflated by 1e-6 (1 + sum pi) (solver tolerances on the pricing
    optimum) and 1e-6 is subtracted before rounding up. Valid for every pi >= 0 when w_ub bounds the maximum weight of
    an independent set: every k-coloring splits sum pi into k class weights, each at most w*."""
    w = max(1.0, float(w_ub)) + 1e-6 * (1.0 + abs(pi_sum))
    return int(math.ceil(pi_sum / w - 1e-6))


class ColoringNaiveCG:
    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 integer_share: float = 0.2, tol: float = 1e-6) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.on_incumbent = on_incumbent or (lambda coloring: None)
        self.on_bound = on_bound or (lambda bound: None)
        self.integer_share, self.tol = integer_share, tol

    @staticmethod
    def price(pm: gp.Model, ylist: list, pi: np.ndarray, deadline: float) -> dict[str, Any]:
        """One exact pricing solve: complete only if Gurobi proves the optimum (MIPGap 0)."""
        pm.setAttr("Obj", ylist, pi.tolist())
        pm.Params.TimeLimit = max(0.01, deadline - time.perf_counter())
        pm.optimize()
        if pm.Status != GRB.OPTIMAL:
            return {"complete": False, "reason": f"pricing_status_{pm.Status}"}
        value = float(pm.ObjVal)
        return {"complete": True, "value": value, "w_ub": max(value, float(pm.ObjBound)),
                "set": np.nonzero(np.array(pm.getAttr("X", ylist)) > 0.5)[0]}

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        cg_deadline = t0 + (1.0 - self.integer_share) * self.time_limit
        inst = load_col(data_path)
        n = inst.n
        best: dict[str, Any] = {"obj": math.inf, "coloring": None, "first": None}

        def offer(coloring) -> None:
            if coloring is None:
                return
            col = canonical(coloring)
            obj = color_count(inst, col)
            if obj is not None and obj < best["obj"]:
                best.update(obj=obj, coloring=col)
                if best["first"] is None:
                    best["first"] = time.perf_counter() - t0
                self.on_incumbent(col)

        init = dsatur(inst.adjacency)
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
        cols: list[np.ndarray] = []
        seen: set[tuple[int, ...]] = set()

        def add(S: np.ndarray) -> bool:
            key = tuple(int(v) for v in S)
            if not key or key in seen:
                return False
            seen.add(key)
            lp.addVar(lb=0.0, obj=1.0, column=gp.Column([1.0] * len(key), [cover[v] for v in key]))
            cols.append(np.array(key, dtype=np.int64))
            return True

        for S in classes_of(init):
            add(S)

        # Pricing oracle: maximum-weight independent set, edge formulation.
        pm = gp.Model(env=env)
        y = pm.addMVar(n, vtype=GRB.BINARY)
        if inst.m:
            E = inst.edges
            inc = sp.csr_matrix((np.ones(2 * len(E)), (np.repeat(np.arange(len(E)), 2), E.reshape(-1))), shape=(len(E), n))
            pm.addConstr(inc @ y <= 1.0)
        pm.ModelSense = GRB.MAXIMIZE
        pm.Params.MIPGap = 0.0
        pm.update()
        ylist = y.tolist()

        iters, converged, bound, stop_reason, lp_value = 0, False, None, "deadline", None
        while time.perf_counter() < cg_deadline:
            lp.Params.TimeLimit = max(0.01, cg_deadline - time.perf_counter())
            lp.optimize()
            if lp.Status != GRB.OPTIMAL:
                break
            iters += 1
            lp_value = float(lp.ObjVal)
            pi = np.maximum(np.array(lp.getAttr("Pi", cover)), 0.0)
            res = self.price(pm, ylist, pi, cg_deadline)
            if not res["complete"]:
                # Strict exact pricing: an unfinished round is discarded and column generation stops.
                stop_reason = "deadline" if time.perf_counter() >= cg_deadline else res["reason"]
                break
            if res["value"] > 1.0 + self.tol and add(res["set"]):
                continue
            converged, stop_reason = True, "converged"
            bound = farley_bound(float(pi.sum()), res["w_ub"])
            self.on_bound(bound)
            break
        pm.dispose()

        # Integer restricted master over all columns with the remaining budget.
        rmp_status = None
        if time.perf_counter() < deadline - 0.01:
            lp.update()
            mip = lp.copy()
            lv = mip.getVars()
            for v in lv:
                v.VType = GRB.BINARY
            mip.Params.TimeLimit = max(0.01, deadline - time.perf_counter())

            def decode(values) -> np.ndarray | None:
                chosen = [S for val, S in zip(values, cols) if val > 0.5]
                return coloring_from_cover(n, chosen)

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
        return {"best_objective": best["obj"] if best["coloring"] is not None else None, "global_lb": bound,
                # "finished_early": the integer restricted master ended before the deadline (allowed by 20.6).
                "termination_reason": "time_limit" if time.perf_counter() >= deadline - 1.0 else "finished_early",
                "cg_iterations": iters, "cg_converged": converged,
                "cg_stop_reason": stop_reason, "columns": len(cols), "lp_value": lp_value, "rmp_status": rmp_status,
                "first_incumbent_seconds": best["first"], "coloring": best["coloring"],
                "total_seconds": time.perf_counter() - t0}
