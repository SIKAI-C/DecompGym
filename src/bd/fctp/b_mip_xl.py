"""Compact FCTP MIP, the transportation-LP evaluator and the solution contract (scale-up study).

Compact model (the standard strong formulation; Glover et al. 2005, MIPLIB 2017 "n37" group use the same rows):
    min  sum f_ij y_ij + sum c_ij x_ij
    s.t. sum_j x_ij = s_i  (suppliers),  sum_i x_ij = d_j  (customers),
         x_ij <= m_ij y_ij  with m_ij = min(s_i, d_j)  (the tightest valid big-M: no feasible flow exceeds it),
         0 <= x_ij <= m_ij,  y binary.
It is exact: for a fixed binary y the remaining LP is the transportation problem restricted to the open arcs, and a
design is feasible in the MIP iff that LP is feasible (tested against the verifier on tiny instances). Gurobi's default
root algorithm is kept (the root LP is a 150-row transportation LP plus 5000 linking rows at 50 x 100: it solves in well
under a second, unlike the large RCSP models where barrier was needed); `mip_focus` is available for a documented probe.
With integer data the objective is an integer, so MIPGap 0 with MIPGapAbs 0.99 makes OPTIMAL a proof, and bounds are
rounded up to integers.

Solutions (published by every method and stored by the runner) are sparse flows [[i, j, q], ...] with q > 0
(integers when the data are integral); the fixed cost of every listed arc is paid. `flow_cost` recomputes the
objective in closed form (no LP), returning None for an infeasible flow.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.fctp.a_data_xl import FCTPInstance, load_fctp

FEAS_TOL = 1e-6


# ---------------------------------------------------------------- solution contract

def normalize_flows(flows: list) -> list[list]:
    """Sorted [[i, j, q]] with duplicates summed and exact zeros dropped (integers kept as int)."""
    acc: dict[tuple[int, int], float] = {}
    for entry in flows:
        i, j, q = entry
        key = (int(i), int(j))
        acc[key] = acc.get(key, 0) + q
    out = []
    for (i, j), q in sorted(acc.items()):
        if q == 0:
            continue
        qf = float(q)
        out.append([i, j, int(qf) if qf.is_integer() else qf])
    return out


def flows_to_matrix(inst: FCTPInstance, flows: list) -> np.ndarray | None:
    X = np.zeros((inst.m, inst.n))
    for entry in flows:
        if len(entry) != 3:
            return None
        i, j, q = entry
        if not (0 <= int(i) < inst.m and 0 <= int(j) < inst.n) or not math.isfinite(float(q)):
            return None
        X[int(i), int(j)] += float(q)
    return X


def flow_cost(inst: FCTPInstance, flows: list, tol: float = FEAS_TOL) -> float | None:
    """Objective of a flow (fixed cost of every listed arc with positive flow + unit costs), or None if infeasible."""
    flows = normalize_flows(flows)
    X = flows_to_matrix(inst, flows)
    if X is None or np.any(X < -tol):
        return None
    if np.any(np.abs(X.sum(axis=1) - inst.supply) > tol * np.maximum(1.0, inst.supply)):
        return None
    if np.any(np.abs(X.sum(axis=0) - inst.demand) > tol * np.maximum(1.0, inst.demand)):
        return None
    used = X > 0
    return float(inst.fcost[used].sum() + (inst.tcost * X).sum())


def matrix_to_flows(X: np.ndarray) -> list[list]:
    i, j = np.nonzero(X > 0)
    return normalize_flows([[int(a), int(b), float(X[a, b])] for a, b in zip(i, j)])


def fctp_output(inst: FCTPInstance, flows: list, status: str = "feasible", method: str = "benders_decomposition") -> dict:
    """Output contract of e_verifier_xl: y (m x n, the arcs with positive flow) and sparse flows."""
    flows = normalize_flows(flows)
    y = np.zeros((inst.m, inst.n), dtype=int)
    for i, j, q in flows:
        if q > 0:
            y[i, j] = 1
    return {"status": status, "method": method, "objective_value": flow_cost(inst, flows), "y": y.tolist(), "flows": flows}


def clean_flows(inst: FCTPInstance, X: np.ndarray) -> np.ndarray:
    """Flows of an LP vertex made exact: rounded to integers when the data are integral and the rounding is still
    feasible (vertices are integral), otherwise tiny values zeroed."""
    X = np.where(np.abs(X) < 1e-9, 0.0, np.maximum(X, 0.0))
    if inst.integral:
        R = np.round(X)
        if np.all(R.sum(axis=1) == inst.supply) and np.all(R.sum(axis=0) == inst.demand):
            return R
    return X


def plan_cost(inst: FCTPInstance, X: np.ndarray) -> float:
    return float(inst.fcost[X > 0].sum() + (inst.tcost * X).sum())


# ---------------------------------------------------------------- transportation LP

def make_env(threads: int = 1, soft_mem_gb: float | None = None, seed: int = 0) -> gp.Env:
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", threads)
    env.setParam("Seed", seed)
    if soft_mem_gb:
        env.setParam("SoftMemLimit", soft_mem_gb)
    env.start()
    return env


class TransportLP:
    """Persistent transportation LP over all arcs: min cost . x, sum_j x_ij = s_i, sum_i x_ij = d_j, 0 <= x <= ub.
    `solve(ub)` changes the bounds and re-optimizes from the previous basis. Duals: u (supplier rows), v (customer
    rows); for an infeasible ub, (u, v) is the Farkas ray -FarkasDual (sum ub_ij (u_i + v_j)^+ < s.u + d.v)."""

    def __init__(self, inst: FCTPInstance, env: gp.Env, cost: np.ndarray | None = None) -> None:
        self.inst = inst
        m, n = inst.m, inst.n
        self.model = gp.Model(env=env)
        self.model.Params.OutputFlag = 0
        self.model.Params.InfUnbdInfo = 1
        self.model.Params.FeasibilityTol = 1e-9
        self.model.Params.OptimalityTol = 1e-9
        self.x = self.model.addMVar((m, n), lb=0.0, ub=inst.cap, name="x")
        self.srows = self.model.addConstr(self.x.sum(axis=1) == inst.supply, name="s")
        self.drows = self.model.addConstr(self.x.sum(axis=0) == inst.demand, name="d")
        self.set_cost(inst.tcost if cost is None else cost)
        self.solves = 0
        self.seconds = 0.0

    def set_cost(self, cost: np.ndarray) -> None:
        self.cost = np.asarray(cost, dtype=float)
        self.model.setObjective((self.cost * self.x).sum(), GRB.MINIMIZE)

    def solve(self, ub: np.ndarray, time_limit: float, flows: bool = True) -> dict[str, Any]:
        t = time.perf_counter()
        self.x.UB = np.asarray(ub, dtype=float)
        self.model.Params.TimeLimit = max(0.01, time_limit)
        self.model.optimize()
        st = self.model.Status
        if st == GRB.INF_OR_UNBD:  # bounded variables: infeasible; re-solve without dual reductions for the ray
            self.model.Params.DualReductions = 0
            self.model.optimize()
            self.model.Params.DualReductions = 1
            st = self.model.Status
        if st == GRB.OPTIMAL:
            out = {"status": "optimal", "obj": float(self.model.ObjVal), "u": np.asarray(self.srows.Pi, dtype=float),
                   "v": np.asarray(self.drows.Pi, dtype=float)}
            if flows:
                out["flows"] = np.asarray(self.x.X, dtype=float)
        elif st == GRB.INFEASIBLE:
            out = {"status": "infeasible", "u": -np.asarray(self.srows.FarkasDual, dtype=float),
                   "v": -np.asarray(self.drows.FarkasDual, dtype=float)}
        elif st == GRB.TIME_LIMIT:
            out = {"status": "time_limit"}
        else:
            out = {"status": f"status_{st}"}
        self.solves += 1
        self.seconds += time.perf_counter() - t
        return out

    def evaluate(self, design: np.ndarray, time_limit: float) -> dict[str, Any]:
        """Exact recourse of a binary design (the transportation LP on its open arcs); on success also the cleaned
        flows ("X") and the true cost of the cleaned plan (arcs without flow are closed)."""
        res = self.solve(self.inst.cap * np.asarray(design, dtype=float), time_limit)
        if res["status"] == "optimal":
            res["X"] = clean_flows(self.inst, res["flows"])
            res["cost"] = plan_cost(self.inst, res["X"])
        return res

    def dispose(self) -> None:
        self.model.dispose()


def design_flows(inst: FCTPInstance, design: np.ndarray, env: gp.Env | None = None) -> np.ndarray | None:
    """Optimal (cleaned) flows of a design, or None if it is infeasible."""
    own = env is None
    env = make_env() if own else env
    try:
        lp = TransportLP(inst, env)
        res = lp.evaluate(design, 60.0)
        lp.dispose()
    finally:
        if own:
            env.dispose()
    if res["status"] == "infeasible":
        return None
    if res["status"] != "optimal":
        raise RuntimeError(f"transportation LP ended with status {res['status']}")
    return res["X"]


def design_cost(inst: FCTPInstance, design: np.ndarray, env: gp.Env | None = None) -> float | None:
    X = design_flows(inst, design, env)
    return None if X is None else plan_cost(inst, X)


# ---------------------------------------------------------------- compact MIP

def build_fctp_mip(inst: FCTPInstance, env: gp.Env | None = None) -> tuple[gp.Model, dict[str, Any]]:
    m, n = inst.m, inst.n
    model = gp.Model(env=env) if env else gp.Model()
    cap = inst.cap
    y = model.addMVar((m, n), vtype=GRB.BINARY, name="y")
    x = model.addMVar((m, n), lb=0.0, ub=cap, name="x")
    model.addConstr(x.sum(axis=1) == inst.supply, name="supply")
    model.addConstr(x.sum(axis=0) == inst.demand, name="demand")
    model.addConstr(x <= cap * y, name="link")
    model.setObjective((inst.fcost * y).sum() + (inst.tcost * x).sum(), GRB.MINIMIZE)
    model.update()
    return model, {"y": y, "x": x, "num_vars": int(model.NumVars), "num_constrs": int(model.NumConstrs)}


def integer_bound(inst: FCTPInstance, value: float) -> float:
    """A solver bound made safe: rounded up to an integer for integral data (with a relative safety margin)."""
    value = value - 1e-7 * max(1.0, abs(value)) - 1e-6
    return float(math.ceil(value)) if inst.integral else value


class FCTPCompactMIP:
    """The compact MIP with the solver interface of the decomposition references. Every integer solution's design
    (arcs with y = 1) is re-evaluated with the transportation LP and its cleaned integral flows are published, so a
    published plan is never worse than the MIP's own; nothing is published at or after the deadline."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 mip_focus: int | None = None) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.mip_focus = mip_focus
        self.on_incumbent = on_incumbent or (lambda flows: None)
        self.on_bound = on_bound or (lambda bound: None)

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        inst = load_fctp(data_path)
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        model, info = build_fctp_mip(inst, env)
        lp = TransportLP(inst, env)
        build = time.perf_counter() - t0
        yv = info["y"].tolist()
        flat = [v for row in yv for v in row]
        best = {"obj": math.inf, "X": None, "bound": -math.inf, "first": None, "solutions": 0, "infeasible": 0,
                "deadline_stop": False}

        def offer(values) -> None:
            if time.perf_counter() >= deadline:
                return
            design = np.asarray(values, dtype=float).reshape(inst.m, inst.n) > 0.5
            res = lp.evaluate(design, max(0.01, deadline - time.perf_counter()))
            best["solutions"] += 1
            if res["status"] != "optimal":
                best["infeasible"] += 1  # never expected: the model is exact
                return
            if res["cost"] < best["obj"] - 1e-9 and time.perf_counter() < deadline:
                best.update(obj=res["cost"], X=res["X"])
                if best["first"] is None:
                    best["first"] = time.perf_counter() - t0
                self.on_incumbent(matrix_to_flows(res["X"]))

        def bound(value: float) -> None:
            if math.isfinite(value) and abs(value) < 1e90 and time.perf_counter() < deadline:
                b = integer_bound(inst, value)
                if b > best["bound"]:
                    best["bound"] = b
                    self.on_bound(b)

        errors: list[BaseException] = []

        def callback(cb, where):
            try:
                if time.perf_counter() >= deadline:
                    best["deadline_stop"] = True
                    cb.terminate()
                    return
                if where == GRB.Callback.MIPSOL:
                    offer(cb.cbGetSolution(flat))
                elif where == GRB.Callback.MIP:
                    bound(cb.cbGet(GRB.Callback.MIP_OBJBND))
            except BaseException as exc:  # noqa: BLE001 - gurobipy would swallow it
                errors.append(exc)
                cb.terminate()

        reason, status = "deadline_during_build", None
        if deadline - time.perf_counter() > 0.01:
            model.Params.TimeLimit = deadline - time.perf_counter()
            model.Params.MIPGap = 0.0
            if inst.integral:
                model.Params.MIPGapAbs = 0.99
            if self.mip_focus is not None:
                model.Params.MIPFocus = self.mip_focus
            model.optimize(callback)
            if errors:
                raise errors[0]
            status = int(model.Status)
            if model.SolCount:
                offer(model.getAttr("X", flat))
            if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.MEM_LIMIT):
                bound(float(model.ObjBound))
            # The actual Gurobi status is kept; only TIME_LIMIT or our own deadline stop is a timeout.
            if status == GRB.OPTIMAL:
                reason = "optimal"
            elif status == GRB.TIME_LIMIT or (status == GRB.INTERRUPTED and best["deadline_stop"]):
                reason = "time_limit"
            else:
                reason = {GRB.INTERRUPTED: "interrupted", GRB.MEM_LIMIT: "memory_limit",
                          GRB.INFEASIBLE: "infeasible"}.get(status, f"status_{status}")
        out = {"best_objective": best["obj"] if best["X"] is not None else None,
               "global_lb": best["bound"] if best["bound"] > -math.inf else None,
               "termination_reason": reason, "gurobi_status": status, "deadline_callback_stop": best["deadline_stop"],
               "budget_exhausted": time.perf_counter() - t0 >= self.time_limit - 1.0,
               "flows": matrix_to_flows(best["X"]) if best["X"] is not None else None, "build_seconds": build,
               "first_incumbent_seconds": best["first"], "num_vars": info["num_vars"], "num_constrs": info["num_constrs"],
               "node_count": float(model.NodeCount) if status is not None else 0.0, "mip_solutions": best["solutions"],
               "rejected_solutions": best["infeasible"], "mip_focus": self.mip_focus,
               "total_seconds": time.perf_counter() - t0}
        lp.dispose()
        model.dispose()
        env.dispose()
        return out
