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

from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.clrp.a_data import CLRPInstance


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
