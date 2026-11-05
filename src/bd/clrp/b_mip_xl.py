"""Compact CLRP MIP (scale-up study): two-index vehicle flow with a single-commodity load flow.

Variables: y_d (open depot), z_id (customer i served from depot d), x_uv (arc used), f_uv (load on arc).
- Each customer has one incoming and one outgoing arc; depots have equal in- and out-degree.
- Load flow: customers absorb their demand; q_v x_uv <= f_uv <= (Q - q_u) x_uv on customer arcs, so a
  route never carries more than the vehicle capacity Q and no subtour avoids a depot.
- Depot consistency: z is equal along every customer-customer arc and matches the depot on depot arcs,
  so a route returns to the depot it left.
- Depot capacity: sum_i q_i z_id <= W_d y_d.
Objective: opening costs + travel costs + vehicle cost per route (one per arc leaving a depot).
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.clrp.a_data import CLRPInstance, load_prodhon


def build_clrp_mip(inst: CLRPInstance, cost: np.ndarray, env: gp.Env | None = None) -> tuple[gp.Model, dict[str, Any]]:
    m, n = inst.m, inst.n
    Q, q = inst.vehicle_capacity, inst.demand
    D = range(m)
    C = range(m, m + n)  # customers are indexed after the depots, as in `travel()`
    model = gp.Model(env=env) if env else gp.Model()
    y = model.addVars(D, vtype=GRB.BINARY, name="y")
    z = model.addVars(C, D, vtype=GRB.BINARY, name="z")
    arcs = [(u, v) for u in range(m + n) for v in range(m + n) if u != v and not (u < m and v < m)]
    # Arcs that can never be used because the pair exceeds the vehicle capacity are dropped.
    arcs = [(u, v) for (u, v) in arcs if not (u >= m and v >= m and q[u - m] + q[v - m] > Q)]
    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    f = model.addVars(arcs, lb=0.0, name="f")
    out = {u: [] for u in range(m + n)}
    inn = {u: [] for u in range(m + n)}
    for (u, v) in arcs:
        out[u].append(v)
        inn[v].append(u)
    for i in C:
        model.addConstr(gp.quicksum(x[i, v] for v in out[i]) == 1)
        model.addConstr(gp.quicksum(x[u, i] for u in inn[i]) == 1)
        model.addConstr(gp.quicksum(f[u, i] for u in inn[i]) - gp.quicksum(f[i, v] for v in out[i]) == q[i - m])
        model.addConstr(z.sum(i, "*") == 1)
    for d in D:
        model.addConstr(gp.quicksum(x[d, v] for v in out[d]) == gp.quicksum(x[u, d] for u in inn[d]))
        model.addConstr(gp.quicksum(q[i - m] * z[i, d] for i in C) <= inst.depot_capacity[d] * y[d])
        for i in C:
            model.addConstr(z[i, d] <= y[d])
            model.addConstr(x[d, i] <= z[i, d])
            if (i, d) in x:
                model.addConstr(x[i, d] <= z[i, d])
                model.addConstr(f[i, d] == 0)
    for (u, v) in arcs:
        if u >= m and v >= m:
            model.addConstr(f[u, v] <= (Q - q[u - m]) * x[u, v])
            model.addConstr(f[u, v] >= q[v - m] * x[u, v])
            for d in D:
                model.addConstr(z[u, d] - z[v, d] <= 1 - x[u, v])
                model.addConstr(z[v, d] - z[u, d] <= 1 - x[u, v])
        elif u < m:
            model.addConstr(f[u, v] <= Q * x[u, v])
            model.addConstr(f[u, v] >= q[v - m] * x[u, v])
    # Valid inequalities: open depots must cover total demand; at least ceil(total demand / Q) routes.
    model.addConstr(gp.quicksum(inst.depot_capacity[d] * y[d] for d in D) >= q.sum())
    model.addConstr(gp.quicksum(x[d, v] for d in D for v in out[d]) >= int(np.ceil(q.sum() / Q)))
    model.setObjective(gp.quicksum(inst.opening_cost[d] * y[d] for d in D)
                       + gp.quicksum(cost[u, v] * x[u, v] for (u, v) in arcs)
                       + gp.quicksum(inst.vehicle_cost * x[d, v] for d in D for v in out[d]), GRB.MINIMIZE)
    return model, {"y": y, "z": z, "x": x, "f": f, "arcs": arcs}


def decode_routes(inst: CLRPInstance, arc_values: dict[tuple[int, int], float]) -> list[dict[str, Any]]:
    """Routes (depot, ordered customers in 0..n-1) from integral arc values."""
    m = inst.m
    succ = {}
    starts = []
    for (u, v), val in arc_values.items():
        if val > 0.5:
            if u < m:
                starts.append((u, v))
            else:
                succ[u] = v
    routes = []
    for d, v in starts:
        seq = []
        while v >= m:
            seq.append(v - m)
            v = succ[v]
        routes.append({"depot": d, "customers": seq})
    return routes


def route_cost(inst: CLRPInstance, cost: np.ndarray, routes: list[dict[str, Any]]) -> float | None:
    """Objective of a route plan, or None if it violates a vehicle or depot capacity or coverage."""
    m, q = inst.m, inst.demand
    served = sorted(c for r in routes for c in r["customers"])
    if served != list(range(inst.n)):
        return None
    load = np.zeros(m)
    total = 0.0
    for r in routes:
        seq = [r["depot"]] + [c + m for c in r["customers"]] + [r["depot"]]
        demand = q[r["customers"]].sum()
        if demand > inst.vehicle_capacity:
            return None
        load[r["depot"]] += demand
        total += sum(cost[a, b] for a, b in zip(seq, seq[1:])) + inst.vehicle_cost
    if np.any(load > inst.depot_capacity):
        return None
    opened = sorted({r["depot"] for r in routes})
    return total + float(inst.opening_cost[opened].sum())


class CLRPCompactMIP:
    """The compact MIP (build_clrp_mip, unchanged) with the solver interface of the decomposition references, for the
    protocol v4 runner (scripts/phoenix/scaleup_cg_stage1.py): every integer solution is decoded (decode_routes) and
    re-checked with route_cost before it is published; bounds are Gurobi's best bounds rounded up (all costs are
    integral: ceil'd travel costs, integer opening and vehicle costs), never above the incumbent. Memory: SoftMemLimit
    on the environment before construction, as the v3.1 calibration runner did."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.on_incumbent = on_incumbent or (lambda routes: None)
        self.on_bound = on_bound or (lambda bound: None)

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        inst = load_prodhon(data_path)
        cost = inst.travel("ceil")
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", self.threads)
        env.setParam("Seed", self.seed)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()
        model, v = build_clrp_mip(inst, cost, env)
        arcs = v["arcs"]
        xs = [v["x"][a] for a in arcs]
        integral = bool(np.all(inst.opening_cost == np.round(inst.opening_cost)) and float(inst.vehicle_cost).is_integer())
        build = time.perf_counter() - t0
        best: dict[str, Any] = {"obj": math.inf, "routes": None, "bound": -math.inf, "first": None, "rejected": 0}

        def offer(values: list[float]) -> None:
            try:
                routes = decode_routes(inst, dict(zip(arcs, values)))
            except (KeyError, IndexError):   # a malformed integer point (should not happen): never published
                best["rejected"] += 1
                return
            obj = route_cost(inst, cost, routes)
            if obj is None:
                best["rejected"] += 1
                return
            if obj < best["obj"]:
                best.update(obj=obj, routes=routes)
                if best["first"] is None:
                    best["first"] = time.perf_counter() - t0
                self.on_incumbent(routes)

        def bound(value: float) -> None:
            if math.isfinite(value) and value > -1e90:
                b = float(math.ceil(value - 1e-6 * max(1.0, abs(value)))) if integral else value - 1e-6 * max(1.0, abs(value))
                if best["obj"] < math.inf:
                    b = min(b, float(best["obj"]))
                if b > best["bound"]:
                    best["bound"] = b
                    self.on_bound(b)

        def callback(mdl, where):
            if time.perf_counter() > deadline:
                mdl.terminate()
                return
            if where == GRB.Callback.MIPSOL:
                offer(mdl.cbGetSolution(xs))
            elif where == GRB.Callback.MIP:
                bound(mdl.cbGet(GRB.Callback.MIP_OBJBND))

        reason = "time_limit"
        if deadline - time.perf_counter() > 0.01:
            model.Params.TimeLimit = deadline - time.perf_counter()
            if integral:   # integer objective: stop only at a proof within less than one unit (Codex review CG-1 #7)
                model.Params.MIPGap = 0.0
                model.Params.MIPGapAbs = 0.5
            model.optimize(callback)
            if model.SolCount:
                offer(model.getAttr("X", xs))
            if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.MEM_LIMIT):
                bound(float(model.ObjBound))
            reason = {GRB.OPTIMAL: "optimal" if integral else "tolerance_optimal",
                      GRB.MEM_LIMIT: "memory_limit"}.get(model.Status, "time_limit")
        out = {"best_objective": best["obj"] if best["routes"] else None,
               "global_lb": best["bound"] if best["bound"] > -math.inf else None, "termination_reason": reason,
               "routes": best["routes"], "build_seconds": build, "num_vars": int(model.NumVars),
               "num_constrs": int(model.NumConstrs), "rejected_points": best["rejected"],
               "first_incumbent_seconds": best["first"], "node_count": float(model.NodeCount),
               "total_seconds": time.perf_counter() - t0}
        model.dispose()
        env.dispose()
        return out
