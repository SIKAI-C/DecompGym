"""Naive textbook logic-based Benders decomposition for planning and scheduling (minimum cost; scale-up control).

NAIVE LBBD (the executable control; protocol Section 20.6 asks for a separately specified baseline when the recourse
is not an LP): the outer-loop LBBD of Hooker and Ottosson (2003) / Hooker (2007, Section 3) with the plain nogood
cut (11) of Cire, Coban and Hooker (2016):
  master   min sum_ij f_ij x_ij  s.t.  sum_i x_ij = 1 (every job once),  x_ij in {0, 1} for the facilities job j fits
           on (p_ij <= d_j - r_j, c_ij <= C_i: part of the problem data), plus all cuts so far; re-solved by Gurobi
           to proved optimality every iteration;
  check    for every facility i, the set J_i of jobs the master assigns to i is checked by the shared exact oracle
           (a_scheduling.FacilityOracle: CP-SAT with exact caches, relaxation and SGS shortcuts);
  cuts     ONE nogood per infeasible facility per iteration on the full assigned set:
               sum_{j in J_i} x_ij <= |J_i| - 1     (not every job of J_i can stay on i);
  stop     when every facility is feasible: the master optimum is then optimal; it is published with its schedules.
The master optimum of every iteration is a valid lower bound (published). No subproblem relaxation in the master, no
cut strengthening (cores), no branch-and-check, no repair heuristic. With `relaxation=True` (a diagnostic arm, Cire,
Coban and Hooker's "relax + weak cuts") the interval relaxation rows of a_scheduling.Relaxation are added to the
master.
The shared basic initializer (a_scheduling.greedy_solution) is published first, as for the engineered method.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from src.bd.lbbd.a_data import PSInstance, load_instance, schedule_cost
from src.bd.lbbd.a_scheduling import (FEASIBLE, INFEASIBLE, FacilityOracle, Relaxation, bits, greedy_solution,
                                      solution_from_schedules, summarize)
from src.bd.lbbd.b_mip_xl import integral_costs, is_memory_error, make_env, set_gap

INF = float("inf")


def certify(bound: float) -> float:
    """A computed lower bound made safe against floating-point error."""
    return bound - 1e-7 * max(1.0, abs(bound)) - 1e-6


class AssignmentMaster:
    """x_ij binary for the eligible pairs, assignment rows, optional relaxation rows; helpers shared with the
    engineered reference."""

    def __init__(self, inst: PSInstance, env: gp.Env, relax: Relaxation | None, binary: bool = True) -> None:
        self.inst = inst
        el = inst.eligible
        self.pi, self.pj = np.nonzero(el)
        self.index = -np.ones((inst.m, inst.n), dtype=np.int64)
        self.index[self.pi, self.pj] = np.arange(len(self.pi))
        N = len(self.pi)
        md = gp.Model(env=env)
        md.Params.OutputFlag = 0
        self.model = md
        self.x = md.addMVar(N, vtype=GRB.BINARY if binary else GRB.CONTINUOUS, lb=0.0, ub=1.0,
                            obj=inst.cost[self.pi, self.pj].astype(float), name="x")
        md.ModelSense = GRB.MINIMIZE
        A = sp.csr_matrix((np.ones(N), (self.pj, np.arange(N))), shape=(inst.n, N))
        md.addConstr(A @ self.x == np.ones(inst.n), name="assign")
        self.relax_rows = 0
        if relax is not None:
            for i in range(inst.m):
                R, rhs, _, _ = relax.rows[i]
                if not len(rhs):
                    continue
                cols = self.index[i]
                nz_r, nz_j = np.nonzero(R)  # relaxation rows only involve jobs that fit on facility i
                if np.any(cols[nz_j] < 0):
                    raise RuntimeError("relaxation row on an ineligible pair")
                M = sp.csr_matrix((R[nz_r, nz_j], (nz_r, cols[nz_j])), shape=(len(rhs), N))
                md.addConstr(M @ self.x <= rhs, name=f"relax{i}")
                self.relax_rows += len(rhs)
        md.update()
        self.xl = self.x.tolist()

    def assignment(self, values) -> list[int]:
        v = np.asarray(values)
        assign = [-1] * self.inst.n
        for k in np.nonzero(v > 0.5)[0]:
            assign[int(self.pj[k])] = int(self.pi[k])
        if min(assign) < 0:
            raise RuntimeError("master solution leaves a job unassigned")
        return assign

    def start_vector(self, assign: list[int]) -> np.ndarray:
        s = np.zeros(len(self.pi))
        s[self.index[assign, np.arange(self.inst.n)]] = 1.0
        return s

    def nogood(self, i: int, mask: int) -> tuple[list[float], list[gp.Var], float]:
        """sum_{j in mask} x_ij <= |mask| - 1 as (coefficients, variables, rhs)."""
        jobs = bits(mask)
        return [1.0] * len(jobs), [self.xl[int(self.index[i, j])] for j in jobs], float(len(jobs) - 1)

    def dispose(self) -> None:
        self.model.dispose()


def masks_of(inst: PSInstance, assign: list[int]) -> list[int]:
    masks = [0] * inst.m
    for j, i in enumerate(assign):
        masks[i] |= 1 << j
    return masks


class PSNaiveLBBD:
    """Naive textbook LBBD control; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 relaxation: bool = False, verbose: bool = False) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.relaxation = relaxation
        self.on_incumbent = on_incumbent or (lambda sol: None)  # one argument (provenance kept in best_stage)
        self.on_bound = on_bound or (lambda bound: None)
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

    def publish(self, sol: list[list[int]], stage: str) -> None:
        value = schedule_cost(self.inst, sol)
        if value is None:
            raise RuntimeError("naive LBBD built an infeasible schedule")
        if value < self.ub - 0.5 * self.unit and self.left() > 0:
            self.ub, self.best, self.best_stage = value, [list(v) for v in sol], stage
            self.on_incumbent(self.best)
            self.log(stage)

    def closed(self) -> bool:
        return self.ub < INF and self.lb > self.ub - self.unit

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        self.inst = inst = load_instance(data_path)
        self.unit = 1.0 if integral_costs(inst) else 1e-6
        self.ub, self.lb, self.best, self.best_stage = INF, -INF, None, None
        self.stats: dict[str, Any] = {"iterations": 0, "cuts": 0, "master_seconds": 0.0, "last_master_status": None}
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        self.oracle = FacilityOracle(inst, Relaxation(inst), seed=self.seed)
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
        return {"best_objective": ub, "global_lb": lb, "global_ub": ub, "termination_reason": reason,
                "budget_exhausted": time.perf_counter() - self.t0 >= self.time_limit - 1.0, "solution": self.best,
                "best_stage": self.best_stage, "stats": self.stats, **self.stats, "oracle": summarize(self.oracle.stats),
                "total_seconds": time.perf_counter() - self.t0, "trace": self.trace}

    def solve(self, inst: PSInstance, env: gp.Env) -> str:
        g = greedy_solution(inst)
        if g is not None:
            self.publish(g[0], "initializer")
        master = AssignmentMaster(inst, env, self.oracle.relax if self.relaxation else None)
        self.stats["relaxation_rows"] = master.relax_rows
        md = master.model
        set_gap(md, inst)
        try:
            while True:
                if self.left() <= 0:
                    return "time_limit"
                t = time.perf_counter()
                md.Params.TimeLimit = max(0.01, self.left())
                md.optimize()
                self.stats["master_seconds"] += time.perf_counter() - t
                self.stats["iterations"] += 1
                self.stats["last_master_status"] = int(md.Status)
                if md.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
                    return "infeasible"  # no assignment is schedulable (every cut is valid; binaries: never unbounded)
                if md.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.MEM_LIMIT) and math.isfinite(md.ObjBound):
                    self.publish_bound(float(md.ObjBound))
                if md.Status == GRB.TIME_LIMIT:
                    return "time_limit"
                if md.Status == GRB.MEM_LIMIT:
                    return "memory_limit"
                if md.Status != GRB.OPTIMAL:
                    return f"master_status_{md.Status}"
                if self.closed():
                    return "optimal"
                assign = master.assignment(md.getAttr("X", master.xl))
                schedules: list[dict[int, int]] = [{} for _ in range(inst.m)]
                ok = True
                for i, mask in enumerate(masks_of(inst, assign)):
                    if self.left() <= 0:
                        return "time_limit"
                    status, starts = self.oracle.check(i, mask, self.left())
                    if status == FEASIBLE:
                        schedules[i] = starts
                    elif status == INFEASIBLE:
                        ok = False
                        coef, xs, rhs = master.nogood(i, mask)
                        md.addLConstr(gp.LinExpr(coef, xs), GRB.LESS_EQUAL, rhs)
                        self.stats["cuts"] += 1
                    else:
                        return "time_limit"  # the oracle ran out of time
                if ok:
                    self.publish(solution_from_schedules(inst, assign, schedules), "master_optimum")
                    return "optimal"
        finally:
            master.dispose()
