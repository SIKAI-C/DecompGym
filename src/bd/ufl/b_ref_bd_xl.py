"""Engineered Benders reference for large UFL (scale-up study; the paper's references are unchanged).

Branch-and-cut on the opening variables y with one recourse variable per customer.
For any y in [0, 1]^m the customer-j subproblem LP has a closed-form optimum: sort the
facilities by cost, fill the cheapest ones up to y-mass 1, and let c_k be the cost of the
critical facility. The cut theta_j >= c_k - sum_i max(0, c_k - c_ij) y_i is a valid Benders
optimality cut, tight at y (Fischetti, Ljubic, Sinnl 2017, Management Science).

Stages, all within one wall-clock budget:
1. Greedy add/drop local search for an initial incumbent.
2. Root LP cutting-plane loop with in-out stabilization; only cuts binding at the end are kept.
3. Gurobi branch-and-cut on y with lazy cuts at integer points and user cuts at the root.
"""
from __future__ import annotations

import math
import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.ufl.a_gen_data import UFLInstance


class UFLCutSeparator:
    """Closed-form per-customer cut separation, vectorized over customers."""

    def __init__(self, D: np.ndarray) -> None:
        self.D = D
        self.order = np.argsort(D, axis=0, kind="stable")  # (m, n) facility order per customer
        self.sorted = np.take_along_axis(D, self.order, axis=0)
        self.cols = np.arange(D.shape[1])

    def critical(self, y: np.ndarray) -> np.ndarray:
        """Cost of the critical facility per customer at y (the closed-form dual value)."""
        cum = np.cumsum(y[self.order], axis=0)
        k = np.argmax(cum >= 1.0 - 1e-9, axis=0)
        # If y-mass never reaches 1 the subproblem is infeasible; the master's sum(y) >= 1 rules this out.
        return self.sorted[k, self.cols]

    def rhs(self, ck: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Right-hand side of the cuts with critical costs ck, evaluated at y."""
        return ck - (np.maximum(ck[None, :] - self.D, 0.0) * y[:, None]).sum(axis=0)

    def value(self, y: np.ndarray) -> np.ndarray:
        """Subproblem value per customer at y (the tight cut evaluated at y)."""
        return self.rhs(self.critical(y), y)

    def cut(self, j: int, ck: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (facility indices, coefficients, constant): theta_j >= const - sum coef_i y_i."""
        coef = ck - self.D[:, j]
        idx = np.nonzero(coef > 0)[0]
        return idx, coef[idx], float(ck)


def assignment_cost(D: np.ndarray, f: np.ndarray, y: np.ndarray) -> float:
    open_ = np.nonzero(y > 0.5)[0]
    return float(f[open_].sum() + D[open_].min(axis=0).sum())


def local_search(D: np.ndarray, f: np.ndarray, deadline: float) -> np.ndarray:
    """Best-improvement add/drop local search from the best single facility."""
    m, _ = D.shape
    y = np.zeros(m, dtype=bool)
    y[int(np.argmin(f + D.sum(axis=1)))] = True
    while time.perf_counter() < deadline:
        sub = D[y]
        best = sub.min(axis=0)
        add_delta = f + np.minimum(D - best[None, :], 0.0).sum(axis=1)
        add_delta[y] = np.inf
        drop_delta = np.full(m, np.inf)
        if y.sum() > 1:
            open_idx = np.nonzero(y)[0]
            arg = sub.argmin(axis=0)
            part = np.partition(sub, 1, axis=0)
            second = part[1]
            for pos, i in enumerate(open_idx):
                mask = arg == pos
                drop_delta[i] = (second[mask] - best[mask]).sum() - f[i]
        i_add, i_drop = int(np.argmin(add_delta)), int(np.argmin(drop_delta))
        if min(add_delta[i_add], drop_delta[i_drop]) >= -1e-9:
            break
        if add_delta[i_add] <= drop_delta[i_drop]:
            y[i_add] = True
        else:
            y[i_drop] = False
    return y.astype(float)


class UFLBendersXL:
    """Engineered Benders branch-and-cut reference; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, root_rounds: int = 200, inout_lambda: float = 0.3, verbose: bool = False,
                 on_incumbent=None, on_bound=None) -> None:
        self.on_incumbent = on_incumbent or (lambda open_: None)  # receives the list of open facilities
        self.on_bound = on_bound or (lambda bound: None)  # receives every valid lower bound
        self.time_limit = time_limit
        self.threads = threads
        self.root_rounds = root_rounds
        self.inout_lambda = inout_lambda
        self.verbose = verbose
        self.trace: list[dict[str, Any]] = []

    def _log(self, t0: float, incumbent: float | None, bound: float | None, stage: str) -> None:
        if bound is not None:
            self.on_bound(bound)
        last = self.trace[-1] if self.trace else None
        if last and last["incumbent"] == incumbent and last["bound"] is not None and bound is not None and bound <= last["bound"] + 1e-6 * max(1.0, abs(bound)):
            return
        self.trace.append({"t": time.perf_counter() - t0, "incumbent": incumbent, "bound": bound, "stage": stage})

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        inst = UFLInstance.from_json_file(data_path)
        D, f = inst.D, inst.f
        m, n = D.shape
        sep = UFLCutSeparator(D)

        y_inc = local_search(D, f, t0 + min(60.0, 0.05 * self.time_limit))
        best_obj = assignment_cost(D, f, y_inc)
        self.on_incumbent(np.nonzero(y_inc > 0.5)[0].tolist())
        self._log(t0, best_obj, None, "local_search")

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 1 if self.verbose else 0)
        env.setParam("Threads", self.threads)
        env.start()
        model = gp.Model(env=env)
        y = model.addMVar(m, lb=0.0, ub=1.0, name="y")
        theta = model.addMVar(n, lb=0.0, name="theta")
        model.addConstr(y.sum() >= 1)
        # Every customer pays at least its cheapest connection cost.
        model.addConstr(theta >= D.min(axis=0))
        model.setObjective(f @ y + theta.sum(), GRB.MINIMIZE)

        cuts: list[gp.Constr] = []
        core = np.full(m, 0.5)
        lam = self.inout_lambda
        root_bound, stalled, rounds = -math.inf, 0, 0
        while rounds < self.root_rounds and time.perf_counter() < deadline:
            rounds += 1
            model.Params.TimeLimit = max(1.0, deadline - time.perf_counter())
            model.optimize()
            if model.Status != GRB.OPTIMAL:
                break
            y_lp, th = y.X, theta.X
            bound = float(model.ObjVal)
            self._log(t0, best_obj, bound, "root_lp")
            stalled = stalled + 1 if bound <= root_bound + 1e-7 * max(1.0, abs(bound)) else 0
            root_bound = max(root_bound, bound)
            point = lam * y_lp + (1 - lam) * core if lam > 0 else y_lp
            # Cuts are generated at the stabilized point and kept only if they cut off the LP point.
            ck = sep.critical(point)
            viol = sep.rhs(ck, y_lp) - th
            added = 0
            for j in np.nonzero(viol > 1e-6 * np.maximum(1.0, np.abs(ck)))[0]:
                idx, coef, const = sep.cut(int(j), float(ck[j]))
                cuts.append(model.addConstr(theta[int(j)] + gp.quicksum(float(c) * y[int(i)] for i, c in zip(idx, coef)) >= const))
                added += 1
            core = 0.5 * core + 0.5 * y_lp
            if added == 0 or stalled >= 5:
                if lam > 0:
                    lam, stalled = 0.0, 0  # finish with pure Kelley cuts at the LP point
                    continue
                break
        # Drop non-binding cuts so the branch-and-cut master stays small.
        kept = len(cuts)
        if cuts and time.perf_counter() < deadline:
            model.optimize()  # the last round may have added cuts after the final solve
            root_bound = max(root_bound, float(model.ObjVal)) if model.Status == GRB.OPTIMAL else root_bound
        if cuts and model.Status == GRB.OPTIMAL:
            loose = [c for c in cuts if c.Slack > 1e-6]
            model.remove(loose)
            kept -= len(loose)
        root_seconds = time.perf_counter() - t0

        y.VType = GRB.BINARY
        y.Start = y_inc
        model.Params.LazyConstraints = 1
        model.Params.PreCrush = 1
        model.Params.TimeLimit = max(1.0, deadline - time.perf_counter())
        stats = {"lazy": 0, "user": 0}
        yv = y.tolist()
        tv = theta.tolist()

        def add_cut(j: int, ck: float, lazy: bool) -> None:
            idx, coef, const = sep.cut(j, ck)
            expr = tv[j] + gp.quicksum(float(c) * yv[int(i)] for i, c in zip(idx, coef))
            if lazy:
                model.cbLazy(expr >= const)
                stats["lazy"] += 1
            else:
                model.cbCut(expr >= const)
                stats["user"] += 1

        def callback(cb_model: gp.Model, where: int) -> None:
            if where == GRB.Callback.MIPSOL:
                yy = np.asarray(cb_model.cbGetSolution(yv))
                th = np.asarray(cb_model.cbGetSolution(tv))
                self.on_incumbent(np.nonzero(yy > 0.5)[0].tolist())
                ck = sep.critical(yy)
                viol = sep.value(yy) - th
                for j in np.nonzero(viol > 1e-6 * np.maximum(1.0, np.abs(ck)))[0]:
                    add_cut(int(j), float(ck[j]), True)
            elif where == GRB.Callback.MIPNODE and cb_model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL and cb_model.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0:
                yy = np.asarray(cb_model.cbGetNodeRel(yv))
                th = np.asarray(cb_model.cbGetNodeRel(tv))
                ck = sep.critical(yy)
                viol = sep.value(yy) - th
                for j in np.nonzero(viol > 1e-4 * np.maximum(1.0, np.abs(ck)))[0]:
                    add_cut(int(j), float(ck[j]), False)
            elif where == GRB.Callback.MIP:
                inc, bnd = cb_model.cbGet(GRB.Callback.MIP_OBJBST), cb_model.cbGet(GRB.Callback.MIP_OBJBND)
                self._log(t0, inc if inc < GRB.INFINITY else None, bnd, "branch_and_cut")

        if time.perf_counter() < deadline:
            model.optimize(callback)
        has_sol = model.SolCount > 0
        y_best = np.round(y.X) if has_sol else y_inc
        obj = assignment_cost(D, f, y_best)  # recomputed from y, never trusted from the master
        if obj > best_obj:
            obj, y_best = best_obj, y_inc
        bound = max(root_bound, float(model.ObjBound)) if model.SolCount or model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) else root_bound
        self._log(t0, obj, bound, "final")
        env.dispose()
        return {
            "best_objective": obj,
            "global_lb": bound,
            "global_ub": obj,
            "gap": (obj - bound) / abs(obj) if obj else None,
            "status": int(model.Status),
            "termination_reason": "optimal" if model.Status == GRB.OPTIMAL else ("time_limit" if model.Status == GRB.TIME_LIMIT else f"status_{model.Status}"),
            "open": np.nonzero(y_best > 0.5)[0].tolist(),
            "root_rounds": rounds,
            "root_bound": root_bound,
            "root_seconds": root_seconds,
            "root_cuts_total": len(cuts),
            "root_cuts_kept": kept,
            "lazy_cuts": stats["lazy"],
            "user_cuts": stats["user"],
            "node_count": float(model.NodeCount) if model.Status != GRB.LOADED else 0.0,
            "total_seconds": time.perf_counter() - t0,
            "trace": self.trace,
        }
