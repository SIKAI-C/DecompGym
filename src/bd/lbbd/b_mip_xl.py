"""Monolithic time-indexed MIP for planning and scheduling (minimum cost) and the solution contract (scale-up study).

Formulation (the discrete-time model, the standard MIP for cumulative scheduling with integer data and the MIP
baseline of Hooker 2004/2007 and Cire, Coban and Hooker 2016): x_ijt = 1 if job j starts on facility i at time t,
for every facility i that j fits on and every integer t in [r_j, d_j - p_ij]:
    min  sum_ijt f_ij x_ijt
    s.t. sum_it x_ijt = 1                                                     (every job once)
         sum_j sum_{t' in (t - p_ij, t]} c_ij x_ijt' <= C_i     for all i, t  (cumulative resource at time t)
         x binary.
It is exact for integer starts (lossless with integer data, see a_data): a fixed assignment with fixed starts is
admitted iff the verifier accepts it, with the same cost (tested). Variables and rows are built from sparse matrices.
The decoded solution of every MIP incumbent is published as [[facility, start] per job]. All costs are integers, so
MIPGapAbs = 1 - 1e-6 proves integral optimality (otherwise MIPGap 1e-6); `mip_focus` and `warm_start` (the shared
basic initializer as MIP start, a diagnostic arm) are probe knobs.

Contract (e_verifier): {"status": "optimal" | "feasible", "method": str, "objective_value": number,
"schedule": [{"job": j, "facility": i, "start": s}, ...]} with 0-based indices and integer starts.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from src.bd.lbbd.a_data import PSInstance, load_instance, normalize_solution, schedule_cost


# ---------------------------------------------------------------- contract

def ps_output(inst: PSInstance, sol: list, status: str = "feasible", method: str = "logic_based_benders") -> dict:
    pairs = normalize_solution(sol)
    return {"status": status, "method": method, "objective_value": schedule_cost(inst, pairs),
            "schedule": [{"job": j, "facility": i, "start": s} for j, (i, s) in enumerate(pairs)]}


def make_env(threads: int = 1, soft_mem_gb: float | None = None, seed: int = 0, mem_limit_gb: float | None = None) -> gp.Env:
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", threads)
    env.setParam("Seed", seed)
    if soft_mem_gb:
        env.setParam("SoftMemLimit", soft_mem_gb)
    if mem_limit_gb:
        env.setParam("MemLimit", mem_limit_gb)
    env.start()
    return env


def is_memory_error(err: gp.GurobiError) -> bool:
    return err.errno == GRB.Error.OUT_OF_MEMORY or "memory" in str(err).lower()


def integral_costs(inst: PSInstance) -> bool:
    return inst.cost.dtype.kind in "iu"


def set_gap(model: gp.Model, inst: PSInstance) -> None:
    if integral_costs(inst):
        model.Params.MIPGap = 0.0
        model.Params.MIPGapAbs = 1.0 - 1e-6
    else:
        model.Params.MIPGap = 1e-6


# ---------------------------------------------------------------- time-indexed model

def build_time_indexed(inst: PSInstance, env: gp.Env | None = None) -> tuple[gp.Model, dict[str, Any]]:
    el = inst.eligible
    fi, fj, ft = [], [], []
    for i in range(inst.m):
        for j in range(inst.n):
            if el[i, j]:
                lo, hi = int(inst.release[j]), int(inst.deadline[j]) - int(inst.proc[i, j])
                ts = np.arange(lo, hi + 1)
                fi.append(np.full(len(ts), i))
                fj.append(np.full(len(ts), j))
                ft.append(ts)
    vi, vj, vt = np.concatenate(fi), np.concatenate(fj), np.concatenate(ft)
    N = len(vi)
    H = inst.horizon
    model = gp.Model(env=env) if env else gp.Model()
    x = model.addMVar(N, vtype=GRB.BINARY, obj=inst.cost[vi, vj].astype(float), name="x")
    model.ModelSense = GRB.MINIMIZE
    A = sp.csr_matrix((np.ones(N), (vj, np.arange(N))), shape=(inst.n, N))
    model.addConstr(A @ x == np.ones(inst.n), name="assign")
    p = inst.proc[vi, vj]
    rows = np.repeat(vi * H, p) + (np.repeat(vt, p) + (np.arange(p.sum()) - np.repeat(np.cumsum(p) - p, p)))
    cols = np.repeat(np.arange(N), p)
    vals = np.repeat(inst.rate[vi, vj].astype(float), p)
    R = sp.csr_matrix((vals, (rows, cols)), shape=(inst.m * H, N))
    keep = np.nonzero(np.asarray(R.sum(axis=1)).ravel() > np.repeat(inst.capacity, H))[0]  # rows that can bind
    if len(keep):
        model.addConstr(R[keep] @ x <= np.repeat(inst.capacity, H)[keep].astype(float), name="cap")
    model.update()
    return model, {"x": x, "i": vi, "j": vj, "t": vt, "num_vars": int(model.NumVars), "num_constrs": int(model.NumConstrs),
                   "cap_rows": int(len(keep))}


def decode(inst: PSInstance, info: dict[str, Any], values: np.ndarray) -> list[list[int]]:
    """[[facility, start] per job] from x values (the largest x of every job)."""
    v = np.asarray(values)
    sol: list[list[int] | None] = [None] * inst.n
    best = np.full(inst.n, -1.0)
    on = np.nonzero(v > 0.5)[0]
    for k in on:
        j = int(info["j"][k])
        if v[k] > best[j]:
            best[j] = v[k]
            sol[j] = [int(info["i"][k]), int(info["t"][k])]
    if any(s is None for s in sol):
        raise RuntimeError("MIP solution leaves a job unassigned")
    return sol


def fix_solution(model: gp.Model, info: dict[str, Any], sol: list[list[int]] | None) -> None:
    """Fix x to a given schedule (tests); None restores the bounds."""
    x = info["x"]
    if sol is None:
        x.LB, x.UB = 0.0, 1.0
        return
    want = {(j, int(i), int(s)) for j, (i, s) in enumerate(sol)}
    val = np.array([1.0 if (int(j), int(i), int(t)) in want else 0.0 for i, j, t in zip(info["i"], info["j"], info["t"])])
    x.LB, x.UB = val, val
    model.update()


class PSTimeIndexedMIP:
    """The time-indexed MIP with the solver interface of the decomposition methods (see the module docstring). Every
    new incumbent is decoded, published strictly before the deadline, and rechecked by the caller; bounds are
    Gurobi's MIP bound."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 mip_focus: int | None = None, warm_start: bool = False, mem_limit_gb: float | None = None) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.mip_focus, self.warm_start, self.mem_limit_gb = mip_focus, warm_start, mem_limit_gb
        self.on_incumbent = on_incumbent or (lambda sol: None)  # one argument, the runner's MIP convention
        self.on_bound = on_bound or (lambda bound: None)

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        inst = load_instance(data_path)
        env = make_env(self.threads, self.soft_mem_gb, self.seed, self.mem_limit_gb)
        best = {"obj": math.inf, "sol": None, "bound": -math.inf, "first": None, "published": 0, "deadline_stop": False,
                "stage": None}
        base = {"mip_focus": self.mip_focus, "warm_start": self.warm_start}
        try:
            model, info = build_time_indexed(inst, env)
        except gp.GurobiError as err:
            env.dispose()
            if not is_memory_error(err):
                raise
            return {"best_objective": None, "global_lb": None, "termination_reason": "memory_limit_during_build",
                    "gurobi_status": None, "budget_exhausted": False, "error": str(err),
                    "build_seconds": time.perf_counter() - t0, "total_seconds": time.perf_counter() - t0, **base}
        build = time.perf_counter() - t0
        xl = info["x"].tolist()

        def offer(values, stage: str) -> None:
            if time.perf_counter() >= deadline:
                return
            sol = decode(inst, info, values)
            obj = schedule_cost(inst, sol)
            if obj is None:
                raise RuntimeError("MIP incumbent fails the exact schedule check")
            if obj < best["obj"] - 0.5 * (1 if integral_costs(inst) else 1e-6):
                best.update(obj=obj, sol=sol, stage=stage)
                best["published"] += 1
                if best["first"] is None:
                    best["first"] = time.perf_counter() - t0
                self.on_incumbent(sol)

        def bound(value: float) -> None:
            if math.isfinite(value) and abs(value) < 1e90 and time.perf_counter() < deadline and value > best["bound"]:
                best["bound"] = value
                self.on_bound(value)

        errors: list[BaseException] = []

        def callback(cb, where):
            try:
                if time.perf_counter() >= deadline:
                    best["deadline_stop"] = True
                    cb.terminate()
                    return
                if where == GRB.Callback.MIPSOL:
                    offer(cb.cbGetSolution(xl), "mip")
                elif where == GRB.Callback.MIP:
                    bound(float(cb.cbGet(GRB.Callback.MIP_OBJBND)))
            except BaseException as exc:  # noqa: BLE001 - gurobipy would swallow it
                errors.append(exc)
                cb.terminate()

        reason, status = "deadline_during_build", None
        if self.warm_start:
            from src.bd.lbbd.a_scheduling import greedy_solution
            g = greedy_solution(inst)
            if g is not None:
                fix = {(j, i, s) for j, (i, s) in enumerate(g[0])}
                info["x"].Start = np.array([1.0 if (int(j), int(i), int(t)) in fix else 0.0
                                            for i, j, t in zip(info["i"], info["j"], info["t"])])
                if time.perf_counter() < deadline:
                    best.update(obj=float(g[1]), sol=g[0], published=1, first=time.perf_counter() - t0, stage="initializer")
                    self.on_incumbent(g[0])
        try:
            if deadline - time.perf_counter() > 0.01:
                model.Params.TimeLimit = deadline - time.perf_counter()
                set_gap(model, inst)
                if self.mip_focus is not None:
                    model.Params.MIPFocus = self.mip_focus
                model.optimize(callback)
                if errors:
                    raise errors[0]
                status = int(model.Status)
                if model.SolCount:
                    offer(model.getAttr("X", xl), "mip_final")
                if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.MEM_LIMIT):
                    bound(float(model.ObjBound))
                unit = 1.0 if integral_costs(inst) else 1e-6
                if status == GRB.OPTIMAL and best["sol"] is not None and best["obj"] <= float(model.ObjVal) + 0.5 * unit:
                    reason = "optimal"  # the retained incumbent (published before the deadline) attains the optimum
                elif status == GRB.TIME_LIMIT or (status == GRB.INTERRUPTED and best["deadline_stop"]) or \
                        (status in (GRB.OPTIMAL, GRB.INFEASIBLE, GRB.INF_OR_UNBD) and time.perf_counter() >= deadline):
                    reason = "time_limit"
                else:
                    reason = {GRB.INTERRUPTED: "interrupted", GRB.MEM_LIMIT: "memory_limit", GRB.INFEASIBLE: "infeasible",
                              GRB.INF_OR_UNBD: "infeasible"}.get(status, f"status_{status}")  # binaries: never unbounded
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        out = {"best_objective": best["obj"] if best["sol"] is not None else None,
               "global_lb": best["bound"] if best["bound"] > -math.inf else None,
               "termination_reason": reason, "gurobi_status": status, "deadline_callback_stop": best["deadline_stop"],
               "budget_exhausted": time.perf_counter() - t0 >= self.time_limit - 1.0, "solution": best["sol"],
               "build_seconds": build, "first_incumbent_seconds": best["first"], "num_vars": info["num_vars"],
               "num_constrs": info["num_constrs"], "cap_rows": info["cap_rows"],
               "node_count": float(model.NodeCount) if status is not None else 0.0,
               "published": best["published"], "best_stage": best["stage"], "total_seconds": time.perf_counter() - t0, **base}
        model.dispose()
        env.dispose()
        return out
