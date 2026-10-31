"""Compact PDPTW MIP (scale-up study): two-index vehicle flow with time, load and route-identity variables (the compact
formulation of Furtado, Munari and Morabito 2017) on the reduced arc set of `a_data.feasible_arcs`.

Variables: x_ij (arc used; node 0 is the depot at both ends), T_i (start of service), L_i (load after service),
v_i (index of the first node of i's route).
- degree: every client has one incoming and one outgoing arc; between 1 and K (fleet) arcs leave the depot;
- time: T_j >= T_i + s_i + t_ij - M_ij (1 - x_ij) on client arcs, return to the depot on arcs (i, 0); windows as bounds,
  tightened by the depot (earliest arrival from it, latest return to it) and by each request's pickup-delivery pair;
- load: L_j = L_i + q_j on used client arcs (two rows), L_j = q_j after the depot, L_i = 0 before returning;
  bounds max(0, q_i) <= L_i <= min(Q, Q + q_i);
- pairing: v_j = j on arcs from the depot, v_i = v_j along used client arcs, v_p = v_d for every request;
- precedence: T_d >= T_p + s_p + t_pd - slack;
- 2-cycle cuts x_ij + x_ji <= 1.
`slack` is the total rounding slack of `a_data.insertion_slack` (nodes with zero service can make a path up to one
unit faster than the direct arc); every tightening that relies on the triangle inequality is relaxed by it, so every
feasible route plan satisfies the model and Gurobi's bound is valid. Objective: total distance.
Integer solutions are decoded into routes and published only after `plan_cost` (exact integer arithmetic) accepts
them; IntegralityFocus = 1 keeps the big-M time rows from being satisfied only through integrality tolerances.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.cg.pdptw.a_data import PDPTWInstance, feasible_arcs, insertion_slack, load_pdptw, plan_cost

INF = float("inf")


def route_cost(inst: PDPTWInstance, dist, routes: list[list[int]]) -> int | None:
    """Total distance of a route plan (coverage, pairing and precedence, capacity, time windows, fleet), else None."""
    return plan_cost(inst, dist, routes)


def normalize_plan(routes: list) -> list[list[int]]:
    return [[int(v) for v in r] for r in routes if len(r)]


def pdptw_output(inst: PDPTWInstance, dist, routes: list, status: str = "feasible", method: str = "decomposition") -> dict[str, Any]:
    plan = normalize_plan(routes)
    return {"status": status, "method": method, "objective_value": route_cost(inst, dist, plan), "routes": plan}


def time_bounds(inst: PDPTWInstance, dist: np.ndarray, slack: int) -> tuple[np.ndarray, np.ndarray]:
    """Valid bounds on the start of service at every node (depot: its window)."""
    D = dist.astype(np.int64)
    e, l, s = inst.ready.astype(np.int64), inst.due.astype(np.int64), inst.service.astype(np.int64)
    lo = np.maximum(e, e[0] + D[0, :] - slack)
    hi = np.minimum(l, l[0] - s - D[:, 0] + slack)
    lo[0], hi[0] = e[0], l[0]
    for p, d in zip(inst.pickups, inst.deliveries):
        lo[d] = max(lo[d], lo[p] + s[p] + D[p, d] - slack)
        hi[p] = min(hi[p], hi[d] - s[p] - D[p, d] + slack)
    return lo, hi


def build_pdptw_mip(inst: PDPTWInstance, dist: np.ndarray, env: gp.Env | None = None,
                    arcs: np.ndarray | None = None) -> tuple[gp.Model, dict[str, Any]]:
    N, Q, K = inst.N, inst.capacity, inst.vehicles
    D = dist.astype(np.int64)
    s = inst.service.astype(np.int64)
    dem = inst.demand.astype(np.int64)
    ok = feasible_arcs(inst, dist) if arcs is None else arcs
    slack = int(insertion_slack(inst, dist).sum())
    lo, hi = time_bounds(inst, dist, slack)
    A = [(int(i), int(j)) for i, j in zip(*np.nonzero(ok))]
    model = gp.Model(env=env) if env else gp.Model()
    x = model.addVars(A, vtype=GRB.BINARY, name="x")
    C = range(1, N + 1)
    T = model.addVars(C, lb={i: float(lo[i]) for i in C}, ub={i: float(max(hi[i], lo[i])) for i in C}, name="T")
    llo = {i: float(max(0, dem[i])) for i in C}
    lhi = {i: float(min(Q, Q + dem[i])) for i in C}
    L = model.addVars(C, lb=llo, ub=lhi, name="L")
    v = model.addVars(C, lb=1.0, ub=float(N), name="v")
    out = {i: [] for i in range(N + 1)}
    inn = {i: [] for i in range(N + 1)}
    for (i, j) in A:
        out[i].append(j)
        inn[j].append(i)
    for i in C:
        model.addConstr(gp.quicksum(x[i, j] for j in out[i]) == 1)
        model.addConstr(gp.quicksum(x[j, i] for j in inn[i]) == 1)
    model.addConstr(gp.quicksum(x[0, j] for j in out[0]) <= K)
    model.addConstr(gp.quicksum(x[0, j] for j in out[0]) >= 1)
    for (i, j) in A:
        if i and j:
            M = float(hi[i] + s[i] + D[i, j] - lo[j])
            if M > 0:
                model.addConstr(T[j] >= T[i] + float(s[i] + D[i, j]) - M * (1 - x[i, j]))
            M1 = lhi[i] + dem[j] - llo[j]
            if M1 > 0:
                model.addConstr(L[j] >= L[i] + float(dem[j]) - M1 * (1 - x[i, j]))
            M2 = lhi[j] - llo[i] - dem[j]
            if M2 > 0:
                model.addConstr(L[j] <= L[i] + float(dem[j]) + M2 * (1 - x[i, j]))
            model.addConstr(v[j] - v[i] <= (N - 1) * (1 - x[i, j]))
            model.addConstr(v[i] - v[j] <= (N - 1) * (1 - x[i, j]))
            if (j, i) in x and i < j:
                model.addConstr(x[i, j] + x[j, i] <= 1)
        elif i == 0:
            # Leaving the depot at its ready time at the earliest: service at j starts no earlier than e_0 + s_0 + t_0j.
            # (The variable bound lo[j] is relaxed by the rounding slack, so this row is needed for exactness.)
            M = float(inst.ready[0] + s[0] + D[0, j] - lo[j])
            if M > 0:
                model.addConstr(T[j] >= float(inst.ready[0] + s[0] + D[0, j]) - M * (1 - x[0, j]))
            if lhi[j] > dem[j]:
                model.addConstr(L[j] <= float(dem[j]) + (lhi[j] - dem[j]) * (1 - x[0, j]))
            model.addConstr(v[j] >= j * x[0, j])
            model.addConstr(v[j] <= j + (N - j) * (1 - x[0, j]))
        else:
            M = float(hi[i] + s[i] + D[i, 0] - inst.due[0])
            if M > 0:
                model.addConstr(T[i] + float(s[i] + D[i, 0]) <= float(inst.due[0]) + M * (1 - x[i, 0]))
            if lhi[i] > 0:
                model.addConstr(L[i] <= lhi[i] * (1 - x[i, 0]))
    # The slack-relaxed precedence row still orders p before d on a common route: if d preceded p, the arc rows would
    # give T_p >= T_d + s_d + t_dp - slack, contradicting the row below whenever s_p + s_d + t_pd + t_dp > 2 slack.
    precedence_exact = all(int(s[p] + s[d] + D[p, d] + D[d, p]) > 2 * slack for p, d in zip(inst.pickups, inst.deliveries))
    for p, d in zip(inst.pickups, inst.deliveries):
        model.addConstr(v[p] == v[d])
        model.addConstr(T[d] >= T[p] + float(s[p] + D[p, d] - slack))
    model.setObjective(gp.quicksum(float(D[i, j]) * x[i, j] for (i, j) in A), GRB.MINIMIZE)
    model.update()
    return model, {"x": x, "T": T, "L": L, "v": v, "arcs": A, "slack": slack, "precedence_exact": precedence_exact}


def decode_routes(N: int, arc_values: dict[tuple[int, int], float]) -> list[list[int]] | None:
    """Routes from integral arc values (successor chains from the depot); None if a client is not reached once."""
    succ, starts = {}, []
    for (i, j), val in arc_values.items():
        if val > 0.5:
            if i == 0:
                starts.append(j)
            else:
                if i in succ:
                    return None
                succ[i] = j
    routes, seen = [], set()
    for j in sorted(starts):
        route = []
        while j != 0:
            if j in seen or j not in succ:
                return None
            seen.add(j)
            route.append(j)
            j = succ[j]
        routes.append(route)
    return routes if len(seen) == N else None


def mip_size(inst: PDPTWInstance, dist: np.ndarray, env: gp.Env | None = None) -> dict[str, int]:
    model, info = build_pdptw_mip(inst, dist, env=env)
    out = {"nodes": inst.N + 1, "requests": inst.n, "arcs": len(info["arcs"]), "full_arcs": (inst.N + 1) * inst.N,
           "variables": model.NumVars, "binaries": model.NumBinVars, "constraints": model.NumConstrs, "nonzeros": model.NumNZs}
    model.dispose()
    return out


class PDPTWCompactMIP:
    """The compact MIP as a timed method: every integer solution is decoded, checked with plan_cost and published if
    it improves; Gurobi's best bound is published as it rises."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 integrality_focus: int = 1, warm_start: bool = False) -> None:
        # warm_start: the shared basic initializer's plan (pricing.insertion_plan) is published first and given to
        # Gurobi as a MIP start (the "MIP with initializer" variant; off for the plain baseline).
        # The objective is integral: MIPGap 0 and MIPGapAbs < 1 make the MIP stop only on a proof of optimality.
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.on_incumbent = on_incumbent or (lambda routes: None)
        self.on_bound = on_bound or (lambda bound: None)
        self.integrality_focus = integrality_focus
        self.warm_start = warm_start

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        inst = load_pdptw(data_path)
        dist = inst.distances()
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", self.threads)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()
        best = {"obj": INF, "routes": None, "bound": -INF, "rejected": 0, "first_t": None}
        try:
            model, info = build_pdptw_mip(inst, dist, env=env)
        except gp.GurobiError as exc:
            env.dispose()
            reason = "memory_limit" if exc.errno == GRB.Error.OUT_OF_MEMORY else "error"
            return {"best_objective": None, "global_lb": None, "termination_reason": reason, "error": str(exc),
                    "total_seconds": time.perf_counter() - t0}
        build = time.perf_counter() - t0
        A = info["arcs"]
        xv = [info["x"][a] for a in A]
        model.Params.Seed = self.seed
        model.Params.IntegralityFocus = self.integrality_focus
        model.Params.MIPGap = 0.0
        model.Params.MIPGapAbs = 1.0 - 1e-6

        def offer(values) -> None:
            routes = decode_routes(inst.N, dict(zip(A, values)))
            obj = plan_cost(inst, dist, routes) if routes is not None else None
            if obj is None:
                best["rejected"] += 1
                return
            if obj < best["obj"]:
                best.update(obj=obj, routes=routes)
                if best["first_t"] is None:
                    best["first_t"] = time.perf_counter() - t0
                self.on_incumbent(routes)

        def bound(value: float) -> None:
            if math.isfinite(value) and value > best["bound"] + 1e-9 * max(1.0, abs(value)):
                best["bound"] = value
                self.on_bound(value)

        def callback(m, where):
            if where == GRB.Callback.MIPSOL:
                offer(m.cbGetSolution(xv))
            elif where == GRB.Callback.MIP:
                bound(m.cbGet(GRB.Callback.MIP_OBJBND))

        if self.warm_start:
            from src.cg.pdptw.pricing import insertion_plan
            plan = insertion_plan(inst, dist.tolist())
            if plan is not None and plan_cost(inst, dist, plan) is not None:
                used = {a for r in plan for a in zip([0] + r, r + [0])}
                if used <= set(A):
                    for a, var in zip(A, xv):
                        var.Start = 1.0 if a in used else 0.0
                    offer([1.0 if a in used else 0.0 for a in A])
        status = "deadline_during_build"
        left = self.time_limit - (time.perf_counter() - t0)
        if left > 0.01:
            model.Params.TimeLimit = left
            model.optimize(callback)
            if model.SolCount:
                offer(model.getAttr("X", xv))
            # The actual Gurobi status is kept: only TIME_LIMIT is a solver timeout.
            status = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit", GRB.MEM_LIMIT: "memory_limit",
                      GRB.INFEASIBLE: "infeasible"}.get(model.Status, f"status_{model.Status}")
            if model.Status == GRB.OPTIMAL:
                bound(model.ObjBound)
        out = {"best_objective": best["obj"] if best["routes"] else None,
               "global_lb": best["bound"] if best["bound"] > -INF else None, "termination_reason": status,
               "routes": best["routes"], "build_seconds": build, "first_incumbent_seconds": best["first_t"],
               "rejected_solutions": best["rejected"], "gurobi_status": model.Status, "precedence_exact": info["precedence_exact"],
               "budget_exhausted": time.perf_counter() - t0 >= self.time_limit - 1.0, "arcs": len(A), "variables": model.NumVars,
               "constraints": model.NumConstrs, "nodes": model.NodeCount if model.Status != GRB.LOADED else 0,
               "total_seconds": time.perf_counter() - t0}
        model.dispose()
        env.dispose()
        return out
