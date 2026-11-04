"""Naive textbook Benders controls for TSSO at scale (scale-up study, protocol v3.1 Section 20.6 and Codex review 43).

Both use the same recourse oracle as the engineered reference (b_mip_xl.ProductRecourse: exact LP recourse of every
product over all its scenarios, duals pi and mu = max(0, pi - c_reg)) and the same shared initializer
(a_data_xl.initial_design), and both add ONE aggregated optimality cut per iteration,
    theta >= sum_(i,s) p_s (pi_is . d_is - sum_t mu_ist K_it y_it),
logged term by term (g_certificate_xl.CutLog.add_aggregate) so the certificate checker can rebuild it.

TSSONaiveLPBenders -- the bound-side naive control: Kelley's cutting-plane method on the LP master
    min F.y + theta,  y in [0, 1]^(I x T),  0 <= theta <= sum_k p_k U_k,
unstabilized (the next separation point is the master's LP solution), one aggregated cut per iteration, stopped when
the master value meets the best LP value F.y + sum p Q(y) seen at a separation point (that is LP*). The master value
is a valid lower bound (solver-reported); the certified bound comes from the cut log. Its only design is the shared
initializer (published at the start).

TSSONaiveBenders -- the integer-master naive (diagnostic and primal-side naive): outer loop, integer master with the
same rows and y binary re-solved to MIPGap 1e-6 every iteration, every master design evaluated exactly and published,
master ObjBound published as its (solver-reported) bound. No multi-cut, stabilization, Pareto selection, partial
decomposition or callbacks in either.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.tsso.a_data_xl import TSSOXLInstance, initial_design, load_instance
from src.bd.tsso.b_mip_xl import DesignEvaluator, is_memory_error, make_env
from src.bd.tsso.g_certificate_xl import CutLog

INF = float("inf")


def evaluate_point(inst: TSSOXLInstance, ev: DesignEvaluator, y: np.ndarray, deadline: float) -> dict[str, Any] | None:
    """Recourse of every product at y in [0, 1]^(I x T): q, pi, mu (I, S[, T]); None when out of time."""
    I, S, T = inst.I, inst.S, inst.T
    q = np.zeros((I, S))
    pi = np.zeros((I, S, T))
    mu = np.zeros((I, S, T))
    for i in range(I):
        left = deadline - time.perf_counter()
        if left <= 0:
            return None
        res = ev.lp(i).solve(y[i], left, duals=True)
        if res["status"] != "optimal":
            if res["status"] == "time_limit":
                return None
            raise RuntimeError(f"recourse of product {i}: {res['status']}")
        q[i], pi[i], mu[i] = res["q"], res["pi"], res["mu"]
    return {"q": q, "pi": pi, "mu": mu}


def aggregate_cut(inst: TSSOXLInstance, pi: np.ndarray, mu: np.ndarray) -> tuple[float, np.ndarray]:
    """(const, coef) of theta + coef . y >= const for the aggregated cut of duals pi, mu (I, S, T)."""
    p = inst.prob
    const = float(np.einsum("s,ist,its->", p, pi, inst.demand))
    coef = np.einsum("s,ist->it", p, mu) * inst.K
    return const, coef


class _NaiveBase:
    def __init__(self, *, time_limit: float = 900.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0, verbose: bool = False,
                 tol: float = 1e-7, model_budget_gb: float = 8.0, **_: Any) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.on_incumbent = on_incumbent or (lambda y, stage=None: None)
        self.on_bound = on_bound or (lambda bound: None)
        self.verbose, self.tol, self.model_budget_gb = verbose, tol, model_budget_gb
        self.trace: list[dict[str, Any]] = []
        self.cutlog: CutLog | None = None

    def left(self) -> float:
        return self.deadline - time.perf_counter()

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def log(self, stage: str, **kw: Any) -> None:
        self.trace.append({"t": round(self.now(), 3), "incumbent": self.ub if self.ub < INF else None,
                           "bound": self.lb if self.lb > -INF else None, "stage": stage, **kw})
        if self.verbose:
            print(self.trace[-1], flush=True)

    def publish_bound(self, value: float, stage: str) -> None:
        if value > self.lb and self.left() > 0:
            self.lb = value
            self.on_bound(value)
            self.log(stage)

    def publish(self, y: np.ndarray, value: float, stage: str) -> None:
        if value < self.ub - 1e-9 * max(1.0, abs(value)) and self.left() > 0:
            self.ub, self.best = value, y.copy()
            self.on_incumbent(y, stage=stage)
            self.log(stage)

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        self.inst = inst = load_instance(data_path)
        self.cutlog = CutLog(inst)
        self.ub, self.lb, self.best = INF, -INF, None
        self.stats: dict[str, Any] = {"iterations": 0, "cuts": 0, "master_seconds": 0.0, "sub_seconds": 0.0}
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        ev = DesignEvaluator(inst, env, model_budget_gb=self.model_budget_gb)
        try:
            reason = self.solve(inst, env, ev)
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        finally:
            ev.close()
            env.dispose()
        ub = self.ub if self.ub < INF else None
        lb = self.lb if self.lb > -INF else None
        return {"best_objective": ub, "global_lb": lb, "termination_reason": reason,
                "budget_exhausted": self.now() >= self.time_limit - 1.0,
                "design": self.best.astype(int).tolist() if self.best is not None else None, **self.stats,
                "total_seconds": self.now(), "trace": self.trace}

    def separate(self, ev: DesignEvaluator, y: np.ndarray) -> dict[str, Any] | None:
        t = time.perf_counter()
        res = evaluate_point(self.inst, ev, y, self.deadline)
        self.stats["sub_seconds"] += time.perf_counter() - t
        if res is None:
            return None
        inst = self.inst
        res["value"] = float((inst.F * y).sum() + np.einsum("s,is->", inst.prob, res["q"]))
        res["const"], res["coef"] = aggregate_cut(inst, res["pi"], res["mu"])
        sub = (np.arange(inst.I)[:, None] * inst.S + np.arange(inst.S)[None, :]).ravel()
        self.cutlog.add_aggregate(sub, res["pi"].reshape(-1, inst.T), res["mu"].reshape(-1, inst.T), self.now())
        self.stats["cuts"] += 1
        return res

    def master(self, env: gp.Env, binary: bool) -> tuple[gp.Model, gp.MVar, gp.Var]:
        inst = self.inst
        md = gp.Model(env=env)
        md.Params.OutputFlag = 0
        md.Params.FeasibilityTol = 1e-9
        md.Params.OptimalityTol = 1e-9
        if binary:
            md.Params.MIPGap = 1e-6
        y = md.addMVar((inst.I, inst.T), lb=0.0, ub=1.0, vtype=GRB.BINARY if binary else GRB.CONTINUOUS)
        theta = md.addVar(lb=0.0, ub=float(np.einsum("s,is->", inst.prob, inst.total_demand
                                                     * (inst.c_reg + inst.c_ot + inst.h + inst.b).sum(axis=1)[:, None])))
        md.setObjective((inst.F * y).sum() + theta, GRB.MINIMIZE)
        return md, y, theta

    def solve(self, inst: TSSOXLInstance, env: gp.Env, ev: DesignEvaluator) -> str:
        raise NotImplementedError


class TSSONaiveLPBenders(_NaiveBase):
    """Unstabilized single-cut Kelley method on the LP master (see the module docstring)."""

    def solve(self, inst: TSSOXLInstance, env: gp.Env, ev: DesignEvaluator) -> str:
        self.publish_bound(0.0, "trivial")
        self.lp_ub = INF
        y_bar = initial_design(inst).astype(float)
        md, y, theta = self.master(env, binary=False)
        try:
            first = True
            while True:
                if self.left() <= 0:
                    return "time_limit"
                res = self.separate(ev, y_bar)
                if res is None:
                    return "time_limit"
                if first:  # the shared initializer is binary: its LP value is its design value
                    self.publish(y_bar > 0.5, res["value"], "initializer")
                    first = False
                self.lp_ub = min(self.lp_ub, res["value"])
                md.addConstr(theta + (res["coef"] * y).sum() >= res["const"])
                if self.lp_ub < INF and self.lp_ub - self.lb <= self.tol * max(1.0, abs(self.lp_ub)):
                    self.log("lp_converged", lp_ub=self.lp_ub)
                    return "lp_optimal"
                t = time.perf_counter()
                md.Params.TimeLimit = max(0.01, self.left())
                md.optimize()
                self.stats["master_seconds"] += time.perf_counter() - t
                self.stats["iterations"] += 1
                if md.Status != GRB.OPTIMAL:
                    return "time_limit" if md.Status == GRB.TIME_LIMIT else f"master_status_{md.Status}"
                self.publish_bound(float(md.ObjVal), "lp_master")
                self.stats["lp_ub"] = self.lp_ub
                if self.lp_ub < INF and self.lp_ub - self.lb <= self.tol * max(1.0, abs(self.lp_ub)):
                    self.log("lp_converged", lp_ub=self.lp_ub)
                    return "lp_optimal"
                y_bar = np.clip(np.asarray(y.X), 0.0, 1.0)
        finally:
            self.stats["lp_ub"] = self.lp_ub if self.lp_ub < INF else None
            md.dispose()


class TSSONaiveBenders(_NaiveBase):
    """Integer-master naive Benders (diagnostic; see the module docstring)."""

    def solve(self, inst: TSSOXLInstance, env: gp.Env, ev: DesignEvaluator) -> str:
        self.publish_bound(0.0, "trivial")
        y_bar = initial_design(inst)
        theta_bar = -INF
        md, y, theta = self.master(env, binary=True)
        try:
            while True:
                if self.left() <= 0:
                    return "time_limit"
                res = self.separate(ev, y_bar.astype(float))
                if res is None:
                    return "time_limit"
                self.publish(y_bar, res["value"], "master_design")
                q = res["value"] - float((inst.F * y_bar).sum())
                if q - theta_bar <= self.tol * max(1.0, abs(q)):
                    return "optimal"
                md.addConstr(theta + (res["coef"] * y).sum() >= res["const"])
                if self.ub < INF and self.ub - self.lb <= 1e-6 * max(1.0, abs(self.ub)):
                    return "optimal"
                t = time.perf_counter()
                md.Params.TimeLimit = max(0.01, self.left())
                md.optimize()
                self.stats["master_seconds"] += time.perf_counter() - t
                self.stats["iterations"] += 1
                if md.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED) and math.isfinite(md.ObjBound):
                    self.publish_bound(float(md.ObjBound), "int_master")
                if md.Status != GRB.OPTIMAL:
                    return "time_limit" if md.Status == GRB.TIME_LIMIT else f"master_status_{md.Status}"
                if self.ub < INF and self.ub - self.lb <= 1e-6 * max(1.0, abs(self.ub)):
                    return "optimal"
                y_bar = np.asarray(y.X) > 0.5
                theta_bar = float(theta.X)
        finally:
            md.dispose()
