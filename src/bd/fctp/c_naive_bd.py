"""Naive textbook Benders decomposition for the FCTP (scale-up control), the shared basic initializer and the cut
algebra shared with the engineered reference (scale-up study; the paper's FCTP files are unchanged).

Recourse of a design y: the transportation LP on the open arcs
    T(y) = min sum c_ij x_ij  s.t.  sum_j x_ij = s_i (u_i),  sum_i x_ij = d_j (v_j),  0 <= x_ij <= m_ij y_ij (w_ij >= 0),
m_ij = min(s_i, d_j). Every (u, v) with w_ij >= (u_i + v_j - c_ij)^+ gives
    optimality cut   theta >= sum_i s_i u_i + sum_j d_j v_j - sum_ij m_ij w_ij y_ij,
and every Farkas ray (u, v, w) with w_ij >= (u_i + v_j)^+ and s.u + d.v > sum m_ij w_ij ybar_ij gives
    feasibility cut  sum_ij m_ij w_ij y_ij >= s.u + d.v.
Both are valid for every feasible binary y (weak duality; m_ij y_ij bounds every feasible flow on an open arc).
The cheapest completion w = (u + v - c)^+ (feasibility: (u + v)^+) gives the strongest cut for given (u, v).

NAIVE BENDERS (the executable control of the scale-up protocol, Section 20.6): outer loop; the integer master
    min f.y + theta,  theta >= 0,  y binary,  plus all cuts so far
is re-solved by Gurobi to optimality (MIPGap 0) every iteration; the recourse LP (explicit linking rows
x_ij <= m_ij ybar_ij, right-hand sides changed, warm-started) is solved at the master's design; one aggregate cut per
iteration from the LP's own duals (optimality: Pi; feasibility: FarkasDual), with w raised to (u + v - c)^+ only as
a floating-point safeguard. The master bound (ObjBound) is published every iteration; the incumbent is the best
master design whose recourse is feasible, published as its LP flows. No stabilization, Pareto selection, master
inequalities, strengthening or callbacks.

Shared basic initializer (also used by the engineered reference and the heuristic control): Balinski's (1961)
linearization, the transportation LP over all arcs with unit costs c_ij + f_ij / m_ij. Its optimal value is the LP
relaxation bound of the compact model; its optimal vertex is a feasible plan (the arcs with flow form the design).
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.fctp.a_data_xl import FCTPInstance, load_fctp
from src.bd.fctp.b_mip_xl import TransportLP, clean_flows, make_env, matrix_to_flows, plan_cost

INF = float("inf")


# ---------------------------------------------------------------- cut algebra

def cut_constant(inst: FCTPInstance, u: np.ndarray, v: np.ndarray) -> float:
    return float(inst.supply @ u + inst.demand @ v)


def cut_coefficients(inst: FCTPInstance, u: np.ndarray, v: np.ndarray, feasibility: bool, w: np.ndarray | None = None) -> np.ndarray:
    """Arc coefficients m_ij * w_ij with w = max(w_given, (u_i + v_j - c_ij)^+) (c = 0 for a feasibility cut)."""
    g = u[:, None] + v[None, :] - (0.0 if feasibility else inst.tcost)
    g = np.maximum(g, 0.0)
    if w is not None:
        g = np.maximum(g, np.maximum(w, 0.0))
    return inst.cap * g


def certify(bound: float) -> float:
    """A computed lower bound made safe against floating-point error in cuts and master LPs."""
    return bound - 1e-7 * max(1.0, abs(bound)) - 1e-6


def is_memory_error(err: gp.GurobiError) -> bool:
    return err.errno == GRB.Error.OUT_OF_MEMORY or "memory" in str(err).lower()


# ---------------------------------------------------------------- shared initializer

def initial_plan(inst: FCTPInstance, env: gp.Env, time_limit: float) -> dict[str, Any] | None:
    """Balinski's linearization (the shared basic initializer). Returns the plan X (cleaned LP vertex), its true cost,
    the LP value (= the compact model's LP relaxation bound) and the LP duals (u, v); None if out of time."""
    lp = TransportLP(inst, env, cost=inst.tcost + inst.fcost / inst.cap)
    res = lp.solve(inst.cap, time_limit)
    lp.dispose()
    if res["status"] != "optimal":
        if res["status"] == "infeasible":
            raise ValueError("the instance is infeasible")
        return None
    X = clean_flows(inst, res["flows"])
    return {"X": X, "cost": plan_cost(inst, X), "lp_value": res["obj"], "u": res["u"], "v": res["v"]}


# ---------------------------------------------------------------- textbook recourse LP

class NaiveRecourse:
    """Recourse LP with explicit linking rows x_ij <= m_ij ybar_ij (right-hand sides changed per design) and the
    LP's own duals: optimality u = Pi(supply), v = Pi(demand), w = -Pi(link); Farkas u = -FD(supply), v = -FD(demand),
    w = +FD(link)."""

    def __init__(self, inst: FCTPInstance, env: gp.Env) -> None:
        self.inst = inst
        m, n = inst.m, inst.n
        self.model = gp.Model(env=env)
        self.model.Params.OutputFlag = 0
        self.model.Params.InfUnbdInfo = 1
        self.model.Params.FeasibilityTol = 1e-9
        self.model.Params.OptimalityTol = 1e-9
        self.x = self.model.addMVar((m, n), lb=0.0, name="x")
        self.srows = self.model.addConstr(self.x.sum(axis=1) == inst.supply)
        self.drows = self.model.addConstr(self.x.sum(axis=0) == inst.demand)
        self.link = self.model.addConstr(self.x <= np.zeros((m, n)))
        self.model.setObjective((inst.tcost * self.x).sum(), GRB.MINIMIZE)
        self.seconds = 0.0

    def solve(self, y: np.ndarray, time_limit: float) -> dict[str, Any]:
        t = time.perf_counter()
        self.link.RHS = self.inst.cap * np.asarray(y, dtype=float)
        self.model.Params.TimeLimit = max(0.01, time_limit)
        self.model.optimize()
        st = self.model.Status
        if st == GRB.INF_OR_UNBD:
            self.model.Params.DualReductions = 0
            self.model.optimize()
            self.model.Params.DualReductions = 1
            st = self.model.Status
        if st == GRB.OPTIMAL:
            out = {"status": "optimal", "obj": float(self.model.ObjVal), "u": np.asarray(self.srows.Pi),
                   "v": np.asarray(self.drows.Pi), "w": -np.asarray(self.link.Pi), "flows": np.asarray(self.x.X)}
        elif st == GRB.INFEASIBLE:
            out = {"status": "infeasible", "u": -np.asarray(self.srows.FarkasDual), "v": -np.asarray(self.drows.FarkasDual),
                   "w": np.asarray(self.link.FarkasDual)}
        elif st == GRB.TIME_LIMIT:
            out = {"status": "time_limit"}
        else:
            out = {"status": f"status_{st}"}
        self.seconds += time.perf_counter() - t
        return out

    def dispose(self) -> None:
        self.model.dispose()


# ---------------------------------------------------------------- naive Benders

class FCTPNaiveBenders:
    """Naive textbook Benders control; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0, verbose: bool = False) -> None:
        self.time_limit = time_limit
        self.threads = threads
        self.on_incumbent = on_incumbent or (lambda flows: None)  # receives sparse flows [[i, j, q], ...]
        self.on_bound = on_bound or (lambda bound: None)  # receives every new valid global lower bound
        self.soft_mem_gb = soft_mem_gb
        self.seed = seed
        self.verbose = verbose
        self.trace: list[dict[str, Any]] = []

    def left(self) -> float:
        return self.deadline - time.perf_counter()

    def log(self, stage: str) -> None:
        self.trace.append({"t": round(time.perf_counter() - self.t0, 3), "incumbent": self.ub if self.ub < INF else None,
                           "bound": self.lb if self.lb > -INF else None, "stage": stage})
        if self.verbose:
            print(self.trace[-1], flush=True)

    def publish_bound(self, value: float) -> None:
        value = certify(value)
        if value > self.lb and self.left() > 0:
            self.lb = value
            self.on_bound(value)
            self.log("bound")

    def publish_plan(self, X: np.ndarray) -> None:
        cost = plan_cost(self.inst, X)
        if cost < self.ub - 1e-9 and self.left() > 0:
            self.ub, self.best = cost, X.copy()
            self.on_incumbent(matrix_to_flows(X))
            self.log("incumbent")

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        self.inst = inst = load_fctp(data_path)
        self.ub, self.lb, self.best = INF, -INF, None
        self.stats = {"iterations": 0, "optimality_cuts": 0, "feasibility_cuts": 0, "master_seconds": 0.0,
                      "sub_seconds": 0.0, "init_seconds": 0.0, "last_master_status": None}
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        try:
            reason = self.solve(inst, env)
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        finally:
            env.dispose()
        self.stats["cuts"] = self.stats["optimality_cuts"] + self.stats["feasibility_cuts"]
        ub = self.ub if self.ub < INF else None
        lb = self.lb if self.lb > -INF else None
        return {"best_objective": ub, "global_lb": lb, "global_ub": ub,
                "gap": (ub - lb) / abs(ub) if ub is not None and lb is not None and ub else None,
                "termination_reason": reason, "budget_exhausted": time.perf_counter() - self.t0 >= self.time_limit - 1.0,
                "flows": matrix_to_flows(self.best) if self.best is not None else None,
                "stats": self.stats, **self.stats, "total_seconds": time.perf_counter() - self.t0, "trace": self.trace}

    def solve(self, inst: FCTPInstance, env: gp.Env) -> str:
        self.publish_bound(0.0)  # costs are nonnegative
        t = time.perf_counter()
        init = initial_plan(inst, env, self.left())
        self.stats["init_seconds"] = time.perf_counter() - t
        if init is None or self.left() <= 0:
            return "time_limit"
        self.publish_plan(init["X"])
        sub = NaiveRecourse(inst, env)
        master = gp.Model(env=env)
        master.Params.OutputFlag = 0
        master.Params.MIPGap = 0.0
        y = master.addMVar((inst.m, inst.n), vtype=GRB.BINARY, name="y")
        theta = master.addVar(lb=0.0, name="theta")
        master.setObjective((inst.fcost * y).sum() + theta, GRB.MINIMIZE)
        tol = 1e-6
        y_bar, theta_bar = init["X"] > 0, -INF
        try:
            while True:
                if self.left() <= 0:
                    return "time_limit"
                t = time.perf_counter()
                res = sub.solve(y_bar, self.left())
                self.stats["sub_seconds"] += time.perf_counter() - t
                if res["status"] == "optimal":
                    self.publish_plan(clean_flows(inst, res["flows"]))
                    const = cut_constant(inst, res["u"], res["v"])
                    h = cut_coefficients(inst, res["u"], res["v"], False, res["w"])
                    if const - float((h * y_bar).sum()) - theta_bar > tol * max(1.0, abs(res["obj"])):
                        master.addConstr(theta + (h * y).sum() >= const)
                        self.stats["optimality_cuts"] += 1
                    else:
                        return "optimal"  # theta_bar >= T(y_bar) - tol: the master optimum is attained by a feasible design
                elif res["status"] == "infeasible":
                    const = cut_constant(inst, res["u"], res["v"])
                    h = cut_coefficients(inst, res["u"], res["v"], True, res["w"])
                    if const - float((h * y_bar).sum()) <= 1e-9 * max(1.0, abs(const)):
                        raise RuntimeError("Farkas certificate does not cut off the infeasible design")
                    master.addConstr((h * y).sum() >= const)
                    self.stats["feasibility_cuts"] += 1
                else:
                    return "time_limit" if res["status"] == "time_limit" else f"recourse_{res['status']}"
                if self.closed():
                    return "optimal"
                if self.left() <= 0:
                    return "time_limit"
                t = time.perf_counter()
                master.Params.TimeLimit = max(0.01, self.left())
                master.optimize()
                self.stats["master_seconds"] += time.perf_counter() - t
                self.stats["iterations"] += 1
                self.stats["last_master_status"] = int(master.Status)
                if master.Status == GRB.INFEASIBLE:
                    raise RuntimeError("Benders master infeasible: the instance has no feasible design")
                if master.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.MEM_LIMIT) and math.isfinite(master.ObjBound):
                    self.publish_bound(float(master.ObjBound))
                if master.Status == GRB.TIME_LIMIT:
                    return "time_limit"
                if master.Status == GRB.MEM_LIMIT:
                    return "memory_limit"
                if master.Status != GRB.OPTIMAL:
                    return f"master_status_{master.Status}"
                if self.closed():
                    return "optimal"
                y_bar = np.asarray(y.X) > 0.5
                theta_bar = float(theta.X)
        finally:
            sub.dispose()
            master.dispose()

    def closed(self) -> bool:
        return self.ub < INF and self.lb >= self.ub - 1e-6 * max(1.0, abs(self.ub))
