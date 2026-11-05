"""Naive textbook Benders (L-shaped) decomposition for the SCFL (scale-up control), the shared basic initializer and
the cut algebra shared with the engineered reference.

Cuts. For scenario s, every (v, alpha) with alpha >= 0 (and v <= M for the penalized recourse) and
beta_ij >= (v_j - alpha_i - c_ij)^+ is dual feasible for the recourse LP with strong linking, so
    theta_s >= sum_j d_sj v_j - sum_i (u_i alpha_i + sum_j d_sj beta_ij) y_i
is valid for every y in [0, 1]^I (weak duality); the cheapest completion beta = (v - alpha - c)^+ gives the strongest
cut for given (v, alpha). A Farkas ray (v, alpha) of an infeasible point gives the feasibility cut
    sum_i (u_i alpha_i + sum_j d_sj (v_j - alpha_i)^+) y_i >= sum_j d_sj v_j.
Closed-facility lifting (`lift_closed`, engineered only): at a binary design ybar the cut value at ybar does not
depend on alpha_i for a closed facility i, so alpha_i can be chosen to minimize its coefficient
u_i a + sum_j d_sj (v_j - c_ij - a)^+ over a >= 0 (a convex piecewise-linear function: the minimizer is the value
w = v_j - c_ij at which the demand of the customers with larger w first exceeds u_i, or 0). The lifted cut is still
tight at ybar and dominates the unlifted one at every y >= 0.

NAIVE BENDERS (the executable control of protocol Section 20.6): outer loop; the integer master
    min f.y + theta,  theta >= 0,  sum_i u_i y_i >= max_s D_s ((15d), a first-stage row of the formulation),  y binary,
plus all cuts so far, re-solved by Gurobi to optimality every iteration; the recourse LP of the extensive form
(explicit demand, capacity and strong-linking rows, right-hand sides changed per design and scenario, warm-started)
is solved for every scenario at the master's design; ONE aggregate cut per iteration,
    theta >= sum_s p_s (cut of scenario s),
from the LP's own duals (Pi), with beta raised to (v - alpha - c)^+ only as a floating-point safeguard. Every master
design is evaluated exactly (it is the full recourse) and published; the master bound (ObjBound) is published every
iteration. No multi-cut, stabilization, Pareto selection, lifting, partial decomposition or callbacks.

Shared basic initializer (naive, engineered and heuristic control): open facilities in increasing order of an
estimated unit cost rho_i = f_i / u_i + sum_j dbar_j c_ij / sum_j dbar_j until (15d) holds.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.scfl.a_data import SCFLInstance, load_instance
from src.bd.scfl.b_mip_xl import is_memory_error, make_env, open_list

INF = float("inf")


# ---------------------------------------------------------------- cut algebra

def cut_from_duals(inst: SCFLInstance, s: int, v: np.ndarray, alpha: np.ndarray, feasibility: bool = False,
                   beta: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """(constant, coefficients) of the cut theta_s >= const - coef . y (feasibility: coef . y >= const)."""
    d = inst.demand[s]
    g = v[None, :] - alpha[:, None] - (0.0 if feasibility else inst.cost)
    g = np.maximum(g, 0.0)
    if beta is not None:
        g = np.maximum(g, np.maximum(beta, 0.0))
    return float(d @ v), inst.capacity * alpha + (g * d[None, :]).sum(axis=1)


def cuts_from_duals(inst: SCFLInstance, scen: np.ndarray, V: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized optimality cuts of several scenarios: const (k,), coef (k, I); V (k, J), A (k, I)."""
    D = inst.demand[scen]
    const = (D * V).sum(axis=1)
    coef = np.empty((len(scen), inst.I))
    for a in range(0, len(scen), 64):  # chunks keep the (k, I, J) temporary small
        b = min(len(scen), a + 64)
        G = np.maximum(V[a:b, None, :] - A[a:b, :, None] - inst.cost[None, :, :], 0.0)
        coef[a:b] = inst.capacity[None, :] * A[a:b] + (G * D[a:b, None, :]).sum(axis=2)
    return const, coef


def lift_closed(inst: SCFLInstance, s: int, v: np.ndarray, coef: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Coefficients with every closed facility's alpha_i re-chosen to minimize its coefficient (see the docstring)."""
    closed = np.nonzero(np.asarray(y) < 0.5)[0]
    if not len(closed):
        return coef
    d = inst.demand[s]
    W = v[None, :] - inst.cost[closed]  # (k, J)
    order = np.argsort(-W, axis=1, kind="stable")
    Ws = np.take_along_axis(W, order, axis=1)
    C = np.cumsum(d[order], axis=1)
    u = inst.capacity[closed]
    over = C > u[:, None] + 1e-9
    has = over.any(axis=1)
    first = np.argmax(over, axis=1)
    a = np.where(has, np.maximum(Ws[np.arange(len(closed)), first], 0.0), 0.0)
    lifted = u * a + (np.maximum(W - a[:, None], 0.0) * d[None, :]).sum(axis=1)
    out = coef.copy()
    out[closed] = np.minimum(coef[closed], lifted)
    return out


def certify(bound: float) -> float:
    """A computed lower bound made safe against floating-point error in cuts and master LPs."""
    return bound - 1e-7 * max(1.0, abs(bound)) - 1e-6


# ---------------------------------------------------------------- shared initializer

def greedy_design(inst: SCFLInstance) -> np.ndarray:
    dbar = inst.mean_demand
    rho = inst.fixed / inst.capacity + inst.cost @ dbar / max(float(dbar.sum()), 1e-12)
    y = np.zeros(inst.I, dtype=bool)
    for i in np.argsort(rho, kind="stable"):
        y[i] = True
        if inst.feasible(y):
            break
    return y


# ---------------------------------------------------------------- textbook recourse LP

class NaiveRecourse:
    """The extensive form's recourse LP with explicit rows and the LP's own duals: v = Pi(demand),
    alpha = -Pi(capacity), beta = -Pi(link); Farkas: v = -FD(demand), alpha = +FD(capacity), beta = +FD(link)."""

    def __init__(self, inst: SCFLInstance, env: gp.Env) -> None:
        self.inst = inst
        I, J = inst.I, inst.J
        md = gp.Model(env=env)
        md.Params.OutputFlag = 0
        md.Params.InfUnbdInfo = 1
        md.Params.FeasibilityTol = 1e-9
        md.Params.OptimalityTol = 1e-9
        self.model = md
        self.x = md.addMVar((I, J), lb=0.0, name="x")
        self.drows = md.addConstr(self.x.sum(axis=0) == np.zeros(J))
        self.crows = md.addConstr(self.x.sum(axis=1) <= np.zeros(I))
        self.link = md.addConstr(self.x <= np.zeros((I, J)))
        md.setObjective((inst.cost * self.x).sum(), GRB.MINIMIZE)
        self.seconds = 0.0

    def solve(self, s: int, y: np.ndarray, time_limit: float) -> dict[str, Any]:
        t = time.perf_counter()
        d = self.inst.demand[s]
        y = np.asarray(y, dtype=float)
        self.drows.RHS = d
        self.crows.RHS = self.inst.capacity * y
        self.link.RHS = y[:, None] * d[None, :]
        md = self.model
        md.Params.TimeLimit = max(0.01, time_limit)
        md.optimize()
        st = md.Status
        if st == GRB.INF_OR_UNBD:
            md.Params.DualReductions = 0
            md.optimize()
            md.Params.DualReductions = 1
            st = md.Status
        if st == GRB.OPTIMAL:
            out = {"status": "optimal", "obj": float(md.ObjVal), "v": np.asarray(self.drows.Pi),
                   "alpha": np.maximum(-np.asarray(self.crows.Pi), 0.0), "beta": -np.asarray(self.link.Pi)}
        elif st == GRB.INFEASIBLE:
            out = {"status": "infeasible", "v": -np.asarray(self.drows.FarkasDual),
                   "alpha": np.maximum(np.asarray(self.crows.FarkasDual), 0.0), "beta": np.asarray(self.link.FarkasDual)}
        elif st == GRB.TIME_LIMIT:
            out = {"status": "time_limit"}
        else:
            out = {"status": f"status_{st}"}
        self.seconds += time.perf_counter() - t
        return out

    def dispose(self) -> None:
        self.model.dispose()


# ---------------------------------------------------------------- naive Benders

class SCFLNaiveBenders:
    """Naive textbook L-shaped control; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0, verbose: bool = False) -> None:
        self.time_limit = time_limit
        self.threads = threads
        self.on_incumbent = on_incumbent or (lambda design: None)  # receives the open-facility list
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

    def publish(self, y: np.ndarray, value: float) -> None:
        if value < self.ub - 1e-9 * max(1.0, abs(value)) and self.left() > 0:
            self.ub, self.best = value, y.copy()
            self.on_incumbent(open_list(y))
            self.log("incumbent")

    def closed(self) -> bool:
        return self.ub < INF and self.lb >= self.ub - 1e-6 * max(1.0, abs(self.ub))

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        self.inst = inst = load_instance(data_path)
        self.ub, self.lb, self.best = INF, -INF, None
        self.stats = {"iterations": 0, "optimality_cuts": 0, "feasibility_cuts": 0, "master_seconds": 0.0,
                      "sub_seconds": 0.0, "evaluations": 0, "last_master_status": None}
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        try:
            reason = self.solve(inst, env)
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        finally:
            env.dispose()
        ub = self.ub if self.ub < INF else None
        lb = self.lb if self.lb > -INF else None
        return {"best_objective": ub, "global_lb": lb, "global_ub": ub,
                "gap": (ub - lb) / abs(ub) if ub is not None and lb is not None and ub else None,
                "termination_reason": reason, "budget_exhausted": time.perf_counter() - self.t0 >= self.time_limit - 1.0,
                "design": open_list(self.best) if self.best is not None else None,
                "stats": self.stats, **self.stats, "total_seconds": time.perf_counter() - self.t0, "trace": self.trace}

    def recourse(self, sub: NaiveRecourse, y: np.ndarray) -> dict[str, Any]:
        """All scenarios at the design y: the aggregate cut (or a Farkas cut) and the exact recourse value."""
        inst = self.inst
        t = time.perf_counter()
        const, coef, q = 0.0, np.zeros(inst.I), 0.0
        try:
            for s in range(inst.S):
                if self.left() <= 0:
                    return {"status": "time_limit"}
                res = sub.solve(s, y, self.left())
                if res["status"] == "infeasible":
                    c, h = cut_from_duals(inst, s, res["v"], res["alpha"], True, res["beta"])
                    return {"status": "infeasible", "const": c, "coef": h}
                if res["status"] != "optimal":
                    return {"status": "time_limit" if res["status"] == "time_limit" else f"recourse_{res['status']}"}
                c, h = cut_from_duals(inst, s, res["v"], res["alpha"], False, res["beta"])
                p = float(inst.prob[s])
                const += p * c
                coef += p * h
                q += p * res["obj"]
        finally:
            self.stats["sub_seconds"] += time.perf_counter() - t
        self.stats["evaluations"] += 1
        return {"status": "optimal", "const": const, "coef": coef, "q": q}

    def solve(self, inst: SCFLInstance, env: gp.Env) -> str:
        self.publish_bound(0.0)  # costs are nonnegative
        sub = NaiveRecourse(inst, env)
        master = gp.Model(env=env)
        master.Params.OutputFlag = 0
        master.Params.MIPGap = 0.0
        y = master.addMVar(inst.I, vtype=GRB.BINARY, name="y")
        theta = master.addVar(lb=0.0, name="theta")
        master.addConstr(inst.capacity @ y >= inst.max_demand, name="cover")
        master.setObjective(inst.fixed @ y + theta, GRB.MINIMIZE)
        y_bar, theta_bar = greedy_design(inst), -INF
        try:
            while True:
                if self.left() <= 0:
                    return "time_limit"
                res = self.recourse(sub, y_bar)
                if res["status"] == "optimal":
                    value = float(inst.fixed @ y_bar) + res["q"]
                    self.publish(y_bar, value)
                    if res["const"] - float(res["coef"] @ y_bar) - theta_bar > 1e-6 * max(1.0, abs(res["q"])):
                        master.addConstr(theta + res["coef"] @ y >= res["const"])
                        self.stats["optimality_cuts"] += 1
                    else:
                        return "optimal"  # theta_bar >= Q(y_bar) - tol: the master optimum is attained
                elif res["status"] == "infeasible":
                    if res["const"] - float(res["coef"] @ y_bar) <= 1e-9 * max(1.0, abs(res["const"])):
                        raise RuntimeError("Farkas certificate does not cut off the infeasible design")
                    master.addConstr(res["coef"] @ y >= res["const"])
                    self.stats["feasibility_cuts"] += 1
                else:
                    return res["status"]
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
