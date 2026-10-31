"""VRPTW route evaluator and compact MIP (scale-up study).

Evaluator. `route_cost(inst, routes)` takes routes as lists of customer numbers 1..n (a_data's numbering, in visiting
order, the depot implicit at both ends), checks that every customer is visited exactly once, every route is nonempty,
within the capacity and on time (the vehicle leaves the depot at its earliest time, waits at early arrivals, starts
every service by the window's end and is back by the depot's latest time; travel time = distance = Euclidean distance
truncated to one decimal), and that at most `inst.vehicles` routes are used; it returns the total distance in
distance units (an exact multiple of 0.1, computed in integer tenths), or None. It reproduces every bundled
best-known cost exactly (tests/test_scaleup_vrptw.py).

Compact MIP: the directed two-index vehicle-flow model with service-start times (MTZ-style time propagation, the
standard compact VRPTW formulation; e.g. Toth and Vigo 2014, ch. 5) and load variables, all in integer tenths:
    min sum_a d_a x_a
    x(delta+(i)) = x(delta-(i)) = 1 for every customer i;   K_lo <= x(delta+(0)) <= K   (K = VEHICLES, K_lo = ceil(q(N)/Q))
    t_j >= t_i + s_i + d_ij - M_ij (1 - x_ij),  M_ij = L_i + s_i + d_ij - E_j   (customer arcs)
    t_i <= R_i + (L_i - R_i)(1 - x_i0),  R_i = l_0 - s_i - d_i0                (only where L_i > R_i)
    E_i <= t_i <= L_i;   lifted MTZ loads (Desrochers and Laporte 1991; Kara, Laporte and Bektas 2004):
    u_i - u_j + Q x_ij + (Q - q_i - q_j) x_ji <= Q - q_j,  q_i + sum_j q_j x_ji <= u_i <= Q - sum_j q_j x_ij,
    u_i <= Q - (Q - q_i) x_0i;   x_ij + x_ji <= 1;   x binary.
Strengthening from the windows: E_i / L_i are the earliest / latest feasible service starts over all paths
(pricing.time_bounds), big-M values use them, and every arc that no route can use (E_i + s_i + d_ij > L_j, or
q_i + q_j > Q) is left out. Rounded capacity inequalities x(delta+(S)) >= ceil(q(S)/Q) are separated as Gurobi user
cuts at the root (greedy set growth of src.bd.clrp on the symmetric arc values). The model is exact when every arc
takes positive time (s_i + d_ij >= 0.1 on every kept arc; true in every bundled file, checked at build time), because
the time propagation then excludes subtours. Every integer solution is decoded into routes and re-checked with
route_cost before it is published.
Size: n customers give at most n (n + 1) arcs (fewer after the window-based elimination), 2 n continuous variables and
about one big-M row per arc for times and one for loads.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.clrp.d_ref_cg_xl import separate_rcc
from src.cg.vrptw.a_data import VRPTWInstance, load_vrptw, plan_issues
from src.cg.vrptw.pricing import time_bounds


# ---------------------------------------------------------------- evaluator

def normalize_routes(routes: list) -> list[list[int]]:
    return [[int(i) for i in r] for r in routes]


def route_cost(inst: VRPTWInstance, routes: list) -> float | None:
    """Total distance (distance units, on the 0.1 grid) of a feasible plan, or None (see the module docstring)."""
    total, _ = plan_issues(inst, routes)
    return None if total is None else total / 10


def route_cost_tenths(inst: VRPTWInstance, routes: list) -> int | None:
    total, _ = plan_issues(inst, routes)
    return total


def vrptw_output(inst: VRPTWInstance, routes: list, status: str = "feasible", method: str = "decomposition") -> dict:
    routes = normalize_routes(routes)
    return {"status": status, "method": method, "objective_value": route_cost(inst, routes), "routes": routes}


# ---------------------------------------------------------------- compact MIP

def mip_arcs(inst: VRPTWInstance) -> dict[str, Any]:
    """Arcs that some feasible route can use (nodes 0..n, depot 0) and the tightened windows (index 0 unused)."""
    tb = time_bounds(inst)
    n = inst.n
    arcs = [(0, j + 1) for j in np.flatnonzero(tb["out_ok"])]
    arcs += [(i + 1, 0) for i in np.flatnonzero(tb["in_ok"])]
    ii, jj = np.nonzero(tb["arc_ok"])
    arcs += [(int(i) + 1, int(j) + 1) for i, j in zip(ii, jj)]
    E = np.r_[inst.ready[0], tb["E"]]
    L = np.r_[inst.due[0], tb["L"]]
    return {"arcs": [(int(a), int(b)) for a, b in arcs], "E": E, "L": L, "n": n, "delta": tb["delta"]}


def build_vrptw_mip(inst: VRPTWInstance, env: gp.Env | None = None) -> tuple[gp.Model, dict[str, Any]]:
    info = mip_arcs(inst)
    if info["delta"] < 1:
        raise ValueError("the compact model needs s_i + d_ij >= 0.1 on every customer arc (subtour exclusion)")
    n, arcs, E, L = inst.n, info["arcs"], info["E"], info["L"]
    d, s, q, Q = inst.dist, inst.service, inst.demand, int(inst.capacity)
    l0 = int(inst.due[0])
    C = range(1, n + 1)
    model = gp.Model(env=env) if env else gp.Model()
    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    t = model.addVars(C, lb={i: float(E[i]) for i in C}, ub={i: float(L[i]) for i in C}, name="t")
    u = model.addVars(C, lb={i: float(q[i]) for i in C}, ub=float(Q), name="u")
    out = {v: [] for v in range(n + 1)}
    inn = {v: [] for v in range(n + 1)}
    for (a, b) in arcs:
        out[a].append(b)
        inn[b].append(a)
    arcset = set(arcs)
    for i in C:
        model.addLConstr(gp.quicksum(x[i, j] for j in out[i]), GRB.EQUAL, 1.0)
        model.addLConstr(gp.quicksum(x[j, i] for j in inn[i]), GRB.EQUAL, 1.0)
    dep = gp.quicksum(x[0, j] for j in out[0])
    model.addLConstr(dep, GRB.LESS_EQUAL, float(inst.vehicles))
    model.addLConstr(dep, GRB.GREATER_EQUAL, float(inst.min_vehicles))
    for (a, b) in arcs:
        if a and b:
            tau = int(s[a]) + int(d[a, b])
            M = int(L[a]) + tau - int(E[b])
            if M > 0:
                model.addLConstr(t[b] - t[a] - M * x[a, b], GRB.GREATER_EQUAL, float(tau - M))
            # Lifted MTZ load rows.
            if (b, a) in arcset:
                if a < b:
                    model.addLConstr(x[a, b] + x[b, a], GRB.LESS_EQUAL, 1.0)
                model.addLConstr(u[a] - u[b] + Q * x[a, b] + (Q - int(q[a]) - int(q[b])) * x[b, a], GRB.LESS_EQUAL, float(Q - int(q[b])))
            else:
                model.addLConstr(u[a] - u[b] + Q * x[a, b], GRB.LESS_EQUAL, float(Q - int(q[b])))
        elif a:  # return arc
            R = l0 - int(s[a]) - int(d[a, 0])
            if L[a] > R:
                model.addLConstr(t[a] + (int(L[a]) - R) * x[a, 0], GRB.LESS_EQUAL, float(L[a]))
    for i in C:
        model.addLConstr(u[i] - gp.quicksum(int(q[j]) * x[j, i] for j in inn[i] if j), GRB.GREATER_EQUAL, float(q[i]))
        model.addLConstr(u[i] + gp.quicksum(int(q[j]) * x[i, j] for j in out[i] if j), GRB.LESS_EQUAL, float(Q))
        if (0, i) in arcset:
            model.addLConstr(u[i] + (Q - int(q[i])) * x[0, i], GRB.LESS_EQUAL, float(Q))
    model.setObjective(gp.quicksum(int(d[a, b]) * x[a, b] for (a, b) in arcs), GRB.MINIMIZE)
    model.update()
    info.update({"x": x, "t": t, "u": u, "num_vars": model.NumVars, "num_constrs": model.NumConstrs, "num_arcs": len(arcs)})
    return model, info


def decode_routes(n: int, arc_values: dict[tuple[int, int], float]) -> list[list[int]] | None:
    """Routes from integral arc values (None if some customer is not on a route from the depot)."""
    succ: dict[int, int] = {}
    starts = []
    for (a, b), v in arc_values.items():
        if v > 0.5:
            if a == 0:
                starts.append(b)
            else:
                if a in succ:
                    return None
                succ[a] = b
    routes = []
    seen = set()
    for v in sorted(starts):
        r = []
        while v != 0:
            if v in seen or v not in succ:
                return None
            seen.add(v)
            r.append(v)
            v = succ[v]
        routes.append(r)
    if len(seen) != n:
        return None
    return routes


def mip_size(inst: VRPTWInstance) -> dict[str, int]:
    info = mip_arcs(inst)
    return {"arcs": len(info["arcs"]), "customers": inst.n, "all_arcs": inst.n * (inst.n + 1)}


class VRPTWCompactMIP:
    """The compact MIP with the solver interface of the decomposition references: incumbents are decoded and checked
    with route_cost before publication; bounds are Gurobi's best bounds rounded up to the 0.1 grid; MIPGap 0 and
    MIPGapAbs 0.99 on the objective in integer tenths, so an OPTIMAL status is a proof of optimality."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0, rcc: bool = True,
                 rcc_rounds: int = 30) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.rcc, self.rcc_rounds = rcc, rcc_rounds
        self.on_incumbent = on_incumbent or (lambda routes: None)
        self.on_bound = on_bound or (lambda bound: None)

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        inst = load_vrptw(data_path)
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", self.threads)
        env.setParam("Seed", self.seed)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()
        model, info = build_vrptw_mip(inst, env)
        build = time.perf_counter() - t0
        x = info["x"]
        keys = list(x.keys())
        xs = [x[a] for a in keys]
        n, q, Q = inst.n, inst.demand, int(inst.capacity)
        best = {"obj": math.inf, "routes": None, "bound": -math.inf, "first": None, "undecodable": 0, "cuts": 0, "rounds": 0}

        def offer(values) -> None:
            routes = decode_routes(n, dict(zip(keys, values)))
            if routes is None:
                best["undecodable"] += 1
                return
            obj = route_cost(inst, routes)
            if obj is not None and obj < best["obj"] - 1e-9:
                best.update(obj=obj, routes=routes)
                if best["first"] is None:
                    best["first"] = time.perf_counter() - t0
                self.on_incumbent(routes)

        def bound(value: float) -> None:
            if math.isfinite(value) and value > -1e90:
                b = math.ceil(value - 1e-6 * max(1.0, abs(value))) / 10  # objective in integer tenths
                if b > best["bound"]:
                    best["bound"] = b
                    self.on_bound(b)

        pos = {a: k for k, a in enumerate(keys)}
        cust = [(a, b) for (a, b) in keys if a and b]
        cidx = [pos[a] for a in cust]
        dep_out = {b: pos[(0, b)] for (a, b) in keys if a == 0}
        dep_in = {a: pos[(a, 0)] for (a, b) in keys if b == 0}
        out_of = {}
        for (a, b) in keys:
            if a:
                out_of.setdefault(a, []).append(b)
        qv = q[1:].astype(float)

        def separate(m) -> None:
            vals = m.cbGetNodeRel(xs)
            X = np.zeros((n, n))
            for (a, b), k in zip(cust, cidx):
                v = vals[k]
                if v > 1e-9:
                    X[a - 1, b - 1] += v
                    X[b - 1, a - 1] += v
            deg = X.sum(axis=1)
            for b_, k in dep_out.items():
                deg[b_ - 1] += vals[k]
            for a_, k in dep_in.items():
                deg[a_ - 1] += vals[k]
            found = separate_rcc(X, deg, qv, float(Q), max_size=min(n, 30), min_viol=0.1)
            for _, U in found[:20]:
                S = set((np.flatnonzero(U) + 1).tolist())
                expr = gp.quicksum(x[a, b] for a in S for b in out_of.get(a, ()) if b not in S)
                m.cbCut(expr >= math.ceil(float(q[list(S)].sum()) / Q - 1e-9))
                best["cuts"] += 1

        def callback(m, where):
            if time.perf_counter() > deadline:
                m.terminate()
                return
            if where == GRB.Callback.MIPSOL:
                offer(m.cbGetSolution(xs))
            elif where == GRB.Callback.MIP:
                bound(m.cbGet(GRB.Callback.MIP_OBJBND))
            elif where == GRB.Callback.MIPNODE and self.rcc and best["rounds"] < self.rcc_rounds:
                if m.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL and m.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0:
                    best["rounds"] += 1
                    separate(m)

        reason = "deadline"
        if deadline - time.perf_counter() > 0.01:
            model.Params.TimeLimit = deadline - time.perf_counter()
            # Integral objective (tenths): OPTIMAL then certifies optimality (no relative-gap stop at 1e-4).
            model.Params.MIPGap = 0.0
            model.Params.MIPGapAbs = 0.99
            if self.rcc:
                model.Params.PreCrush = 1
            model.optimize(callback)
            if model.SolCount:
                offer(model.getAttr("X", xs))
            if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.MEM_LIMIT, GRB.INTERRUPTED):
                bound(float(model.ObjBound))
            reason = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit", GRB.MEM_LIMIT: "memory_limit",
                      GRB.INTERRUPTED: "time_limit", GRB.INFEASIBLE: "infeasible"}.get(model.Status, f"status_{model.Status}")
        out = {"best_objective": best["obj"] if best["routes"] else None,
               "global_lb": best["bound"] if best["bound"] > -math.inf else None,
               "termination_reason": reason, "routes": best["routes"], "build_seconds": build,
               "first_incumbent_seconds": best["first"], "num_vars": info["num_vars"], "num_constrs": info["num_constrs"],
               "num_arcs": info["num_arcs"], "rcc_cuts": best["cuts"], "undecodable_solutions": best["undecodable"],
               "node_count": float(model.NodeCount) if reason != "deadline" else 0.0, "total_seconds": time.perf_counter() - t0}
        model.dispose()
        env.dispose()
        return out
