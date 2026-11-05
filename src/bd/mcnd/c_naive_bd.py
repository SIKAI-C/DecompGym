"""Naive textbook Benders decomposition for the MCND (scale-up control) and the Benders machinery it shares with
the engineered reference (scale-up study; the paper's references are unchanged).

Recourse of a design y: the multicommodity min-cost flow LP
    Q(y) = min sum_{a,k} c_a x_ak  s.t.  N x_k = b_k (every k),  sum_k x_ak <= u_a y_a,  [x_ak <= L_ak y_a],  x >= 0,
with node potentials pi_k (duals of N x_k = b_k), alpha_a >= 0 (capacity) and, in the strong recourse only,
beta_ak >= 0 (linking, L_ak = min(d_k, u_a)). Every pi with dual-feasible (alpha, beta) gives
    Q(y) >= sum_k d_k (pi_k[o_k] - pi_k[t_k]) - sum_a h_a y_a                (optimality cut, costs c)
    sum_a h_a y_a >= sum_k d_k (pi_k[o_k] - pi_k[t_k])                       (feasibility cut, Farkas ray, costs 0)
where, with g_ak = max(0, pi_k[tail a] - pi_k[head a] - c_a):
    plain recourse:  h_a = u_a alpha_a with alpha_a >= max_k g_ak;
    strong recourse: h_a = min_{t >= 0} u_a t + sum_k L_ak (g_ak - t)^+  (the cheapest dual-feasible completion).
Both are valid for every binary y: for binary y some optimal (or feasible) flow is cycle-free, so x_ak <= L_ak y_a.

NAIVE BENDERS (the executable control of the scale-up protocol, Section 20.6): outer loop; the integer master
    min f y + theta,  theta >= 0,  y binary,  plus all cuts so far
is re-solved by Gurobi to optimality (MIPGap 0) every iteration; the plain recourse LP is solved at the master's y
(persistent model, right-hand sides changed, warm-started); one aggregate cut per iteration from the LP's own duals
(optimality: Pi; feasibility: FarkasDual), with alpha raised to max_k g_ak only as a floating-point safeguard.
The master bound (ObjBound) is published every iteration; the incumbent is the best master design whose recourse
is feasible, published with its true cost f y + Q(y). No stabilization, Pareto selection, strengthening, master
inequalities or callbacks. Initializer (shared with the engineered reference): one slope-scaling step, i.e. the
min-cost multicommodity flow over all arcs with unit costs c_a + f_a / u_a; the arcs carrying flow form the design.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.mcnd.a_data import MCNDInstance, load_dow
from src.bd.mcnd.b_mip_xl import add_flow_conservation, useless_arcs

INF = float("inf")
FLOW_EPS = 1e-9


# ---------------------------------------------------------------- cuts from node potentials

def gather(rows: list, attr: str, K: int) -> np.ndarray:
    """(K, n) array of a row attribute (Pi or FarkasDual) of the per-node flow-conservation blocks (0 for absent nodes)."""
    out = np.zeros((K, len(rows)))
    for i, r in enumerate(rows):
        if r is not None:
            out[:, i] = np.asarray(r.getAttr(attr)).ravel()
    return out


def arc_excess(inst: MCNDInstance, pi: np.ndarray, cost: np.ndarray | float, skip: np.ndarray | None = None) -> np.ndarray:
    """g_ak = max(0, pi_k[tail a] - pi_k[head a] - c_a), zero on skipped commodity-arc pairs."""
    G = np.maximum(pi[:, inst.tail] - pi[:, inst.head] - cost, 0.0)
    if skip is not None:
        G[skip] = 0.0
    return G


def strong_arc_coefficients(G: np.ndarray, u: np.ndarray, L: np.ndarray) -> np.ndarray:
    """h_a = min_{t >= 0} u_a t + sum_k L_ak (g_ak - t)^+ (exact minimizer; any t >= 0 would still be valid)."""
    if G.shape[0] == 0:
        return np.zeros(G.shape[1])
    order = np.argsort(-G, axis=0, kind="stable")
    Gs = np.take_along_axis(G, order, axis=0)
    S = np.cumsum(np.take_along_axis(L, order, axis=0), axis=0)
    over = S > u[None, :]
    j = np.argmax(over, axis=0)
    t = np.where(over.any(axis=0), Gs[j, np.arange(G.shape[1])], 0.0)
    return u * t + (L * np.maximum(G - t[None, :], 0.0)).sum(axis=0)


def cut_constant(inst: MCNDInstance, pi: np.ndarray) -> np.ndarray:
    """Per-commodity d_k (pi_k[o_k] - pi_k[t_k]); the cut constant is its sum."""
    k = np.arange(inst.K)
    return inst.demand * (pi[k, inst.orig] - pi[k, inst.dest])


def plain_cut(inst: MCNDInstance, pi: np.ndarray, alpha_lp: np.ndarray | None, feasibility: bool) -> tuple[float, np.ndarray]:
    """Textbook cut (const, h) of the plain recourse: theta >= const - h y (or h y >= const for a ray)."""
    G = arc_excess(inst, pi, 0.0 if feasibility else inst.cost)
    alpha = G.max(axis=0) if inst.K else np.zeros(inst.m)
    if alpha_lp is not None:
        alpha = np.maximum(alpha, np.maximum(alpha_lp, 0.0))
    return float(cut_constant(inst, pi).sum()), inst.cap * alpha


# ---------------------------------------------------------------- recourse LP

class RecourseLP:
    """Persistent recourse LP over all arcs; `solve(point)` sets capacities u * point (and, if strong, the linking
    bounds L * point) and re-optimizes from the previous basis."""

    def __init__(self, inst: MCNDInstance, env: gp.Env, strong: bool) -> None:
        self.inst, self.strong = inst, strong
        K, m = inst.K, inst.m
        self.model = gp.Model(env=env)
        self.model.Params.OutputFlag = 0
        self.model.Params.InfUnbdInfo = 1
        self.skip = useless_arcs(inst) if strong else None
        self.L = np.where(self.skip, 0.0, inst.link_bound()) if strong else None
        self.x = self.model.addMVar((K, m), lb=0.0, ub=self.L if strong else GRB.INFINITY, name="x")
        self.rows = add_flow_conservation(self.model, self.x, inst)
        self.cap = self.model.addConstr(self.x.sum(axis=0) <= inst.cap, name="cap")
        self.model.setObjective((inst.cost[None, :] * self.x).sum(), GRB.MINIMIZE)
        self.solves = 0
        self.seconds = 0.0

    def solve(self, point: np.ndarray, time_limit: float, flows: bool = False) -> dict[str, Any]:
        t = time.perf_counter()
        point = np.clip(np.asarray(point, dtype=float), 0.0, 1.0)
        self.cap.RHS = self.inst.cap * point
        if self.strong:
            self.x.UB = self.L * point[None, :]
        self.model.Params.TimeLimit = max(0.01, time_limit)
        self.model.optimize()
        self.solves += 1
        st = self.model.Status
        if st == GRB.INF_OR_UNBD:  # bounded variables: infeasible; re-solve without dual reductions for the ray
            self.model.Params.DualReductions = 0
            self.model.optimize()
            self.model.Params.DualReductions = 1
            st = self.model.Status
        K = self.inst.K
        if st == GRB.OPTIMAL:
            out = {"status": "optimal", "obj": float(self.model.ObjVal), "pi": gather(self.rows, "Pi", K),
                   "alpha": -np.asarray(self.cap.Pi)}
            if flows:
                out["flows"] = np.asarray(self.x.X)
        elif st == GRB.INFEASIBLE:
            # Gurobi's Farkas certificate: the dual ray is pi = -FarkasDual(flow rows), alpha = +FarkasDual(capacity rows).
            out = {"status": "infeasible", "pi": -gather(self.rows, "FarkasDual", K), "alpha": np.asarray(self.cap.FarkasDual)}
        elif st == GRB.TIME_LIMIT:
            out = {"status": "time_limit"}
        else:
            out = {"status": f"status_{st}"}
        self.seconds += time.perf_counter() - t
        return out

    def dispose(self) -> None:
        self.model.dispose()


# ---------------------------------------------------------------- shared initializer

class SlopeScaling:
    """Slope scaling (Crainic, Gendron, Hernu 2004): min-cost multicommodity flow over all arcs with linearized unit
    costs rho_a, initially c_a + f_a / u_a, then c_a + f_a / xbar_a on arcs that carried flow xbar_a > 0 (unchanged
    otherwise). The arcs carrying flow form a design. The first step is the shared basic initializer."""

    def __init__(self, inst: MCNDInstance, env: gp.Env) -> None:
        self.inst = inst
        self.model = gp.Model(env=env)
        self.model.Params.OutputFlag = 0
        self.x = self.model.addMVar((inst.K, inst.m), lb=0.0, name="x")
        add_flow_conservation(self.model, self.x, inst)
        self.model.addConstr(self.x.sum(axis=0) <= inst.cap, name="cap")
        self.rho = inst.cost + inst.fixed / inst.cap
        self.seconds = 0.0
        self.steps = 0

    def step(self, time_limit: float) -> np.ndarray | None:
        """One LP; returns the aggregate arc flows, or None (infeasible or out of time)."""
        t = time.perf_counter()
        self.model.setObjective((self.rho[None, :] * self.x).sum(), GRB.MINIMIZE)
        self.model.Params.TimeLimit = max(0.01, time_limit)
        self.model.optimize()
        self.seconds += time.perf_counter() - t
        if self.model.Status != GRB.OPTIMAL:
            if self.model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
                raise ValueError("the instance is infeasible even with every arc open")
            return None
        self.steps += 1
        flow = np.asarray(self.x.X).sum(axis=0)
        inst = self.inst
        self.rho = np.where(flow > FLOW_EPS, inst.cost + inst.fixed / np.maximum(flow, FLOW_EPS), self.rho)
        return flow

    def dispose(self) -> None:
        self.model.dispose()


def initial_design(ss: SlopeScaling, time_limit: float) -> np.ndarray | None:
    """The shared basic initializer: arcs carrying flow after the first slope-scaling step (None if out of time)."""
    flow = ss.step(time_limit)
    return None if flow is None else flow > FLOW_EPS


# ---------------------------------------------------------------- naive Benders

def make_env(threads: int, soft_mem_gb: float | None) -> gp.Env:
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", threads)
    if soft_mem_gb:
        env.setParam("SoftMemLimit", soft_mem_gb)
    env.start()
    return env


def certify(bound: float) -> float:
    """A computed lower bound made safe against floating-point error in cuts and master LPs."""
    return bound - 1e-7 * max(1.0, abs(bound)) - 1e-6


def is_memory_error(err: gp.GurobiError) -> bool:
    return err.errno == GRB.Error.OUT_OF_MEMORY or "memory" in str(err).lower()


class MCNDNaiveBenders:
    """Naive textbook Benders control; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0, verbose: bool = False) -> None:
        self.time_limit = time_limit
        self.threads = threads
        self.on_incumbent = on_incumbent or (lambda design: None)  # receives the list of open arc indices
        self.on_bound = on_bound or (lambda bound: None)  # receives every new valid global lower bound
        self.soft_mem_gb = soft_mem_gb
        self.seed = seed
        self.verbose = verbose
        self.trace: list[dict[str, Any]] = []

    def left(self) -> float:
        return self.deadline - time.perf_counter()

    def publish_bound(self, value: float) -> None:
        value = certify(value)
        if value > self.lb:
            self.lb = value
            self.on_bound(value)
            self.log("bound")

    def publish_design(self, y: np.ndarray, cost: float) -> None:
        if cost < self.ub - 1e-9:
            self.ub, self.best = cost, y.copy()
            self.on_incumbent(np.nonzero(y)[0].tolist())
            self.log("incumbent")

    def log(self, stage: str) -> None:
        self.trace.append({"t": round(time.perf_counter() - self.t0, 3), "incumbent": self.ub if self.ub < INF else None,
                           "bound": self.lb if self.lb > -INF else None, "stage": stage})
        if self.verbose:
            print(self.trace[-1], flush=True)

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        inst = load_dow(data_path)
        self.ub, self.lb, self.best = INF, -INF, None
        self.stats = {"iterations": 0, "cuts": 0, "optimality_cuts": 0, "feasibility_cuts": 0, "master_seconds": 0.0,
                      "sub_seconds": 0.0, "init_seconds": 0.0}
        reason = "time_limit"
        env = make_env(self.threads, self.soft_mem_gb)
        try:
            reason = self.solve(inst, env)
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        env.dispose()
        self.stats["cuts"] = self.stats["optimality_cuts"] + self.stats["feasibility_cuts"]
        ub = self.ub if self.ub < INF else None
        lb = self.lb if self.lb > -INF else None
        return {"best_objective": ub, "global_lb": lb, "global_ub": ub,
                "gap": (ub - lb) / abs(ub) if ub is not None and lb is not None and ub else None,
                "termination_reason": reason, "open": np.nonzero(self.best)[0].tolist() if self.best is not None else None,
                "stats": self.stats, **self.stats, "total_seconds": time.perf_counter() - self.t0, "trace": self.trace}

    def solve(self, inst: MCNDInstance, env: gp.Env) -> str:
        self.publish_bound(0.0)  # costs are nonnegative
        t = time.perf_counter()
        ss = SlopeScaling(inst, env)
        y0 = initial_design(ss, self.left())
        ss.dispose()
        self.stats["init_seconds"] = time.perf_counter() - t
        if y0 is None or self.left() <= 0:
            return "time_limit"
        sub = RecourseLP(inst, env, strong=False)
        master = gp.Model(env=env)
        master.Params.OutputFlag = 0
        master.Params.MIPGap = 0.0
        master.Params.Seed = self.seed
        y = master.addMVar(inst.m, vtype=GRB.BINARY, name="y")
        theta = master.addVar(lb=0.0, name="theta")
        master.setObjective(inst.fixed @ y + theta, GRB.MINIMIZE)
        tol = 1e-6
        y_bar, theta_bar = y0, -INF
        while True:
            # Recourse at the current design: incumbent update and one aggregate cut.
            if self.left() <= 0:
                return "time_limit"
            t = time.perf_counter()
            res = sub.solve(y_bar.astype(float), self.left())
            self.stats["sub_seconds"] += time.perf_counter() - t
            if res["status"] == "optimal":
                self.publish_design(y_bar, float(inst.fixed[y_bar].sum()) + res["obj"])
                const, h = plain_cut(inst, res["pi"], res["alpha"], feasibility=False)
                if const - h @ y_bar - theta_bar > tol * max(1.0, abs(res["obj"])):
                    master.addConstr(theta + h @ y >= const)
                    self.stats["optimality_cuts"] += 1
                else:
                    # theta_bar >= Q(y_bar) - tol: the master optimum is attained by a feasible design (converged).
                    return "optimal"
            elif res["status"] == "infeasible":
                const, h = plain_cut(inst, res["pi"], res["alpha"], feasibility=True)
                if const - h @ y_bar <= 1e-9 * max(1.0, abs(const)):
                    raise RuntimeError("Farkas certificate does not cut off the infeasible design")
                master.addConstr(h @ y >= const)
                self.stats["feasibility_cuts"] += 1
            else:
                return "time_limit"
            if self.closed():
                return "optimal"
            # Master.
            if self.left() <= 0:
                return "time_limit"
            t = time.perf_counter()
            master.Params.TimeLimit = max(0.01, self.left())
            master.optimize()
            self.stats["master_seconds"] += time.perf_counter() - t
            self.stats["iterations"] += 1
            if master.Status == GRB.INFEASIBLE:
                raise RuntimeError("Benders master infeasible: the instance has no feasible design")
            if master.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.MEM_LIMIT):
                self.publish_bound(float(master.ObjBound))
            if master.Status != GRB.OPTIMAL:
                return "memory_limit" if master.Status == GRB.MEM_LIMIT else "time_limit"
            if self.closed():
                return "optimal"
            y_bar = np.asarray(y.X) > 0.5
            theta_bar = float(theta.X)

    def closed(self) -> bool:
        return self.ub < INF and self.lb >= self.ub - 1e-6 * max(1.0, abs(self.ub))
