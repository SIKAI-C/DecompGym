from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.cg.pdptw.a_gen_data import PDPTWInstance


class PDPTWBaseline:
    """Two-index arc MIP; a route-identity variable (the index of the route's first node) ties each pickup to its delivery."""

    def __init__(self, *, params: dict[str, Any] | None = None) -> None:
        # Integer distances: a bound within 0.5 of the incumbent proves it. Times are in thousandths (up to ~3.4e6), so
        # the integrality tolerance is tightened: a big-M link with x = 1 - IntFeasTol must not open a whole time unit.
        self.params = {"Threads": 1, "MIPGap": 0.0, "MIPGapAbs": 0.5, "IntFeasTol": 1e-9, **(params or {})}

    def run(self, path: str) -> dict[str, Any]:
        inst = PDPTWInstance.from_json_file(path)
        n, N, Q = inst.n_requests, 2 * inst.n_requests, inst.capacity
        V, C = range(N + 1), range(1, N + 1)
        d = {(i, j): inst.dist(i, j) for i in V for j in V if i != j}
        arcs = [(i, j) for (i, j) in d if not (i > n and i - n == j) and not (j == 0 and 1 <= i <= n) and not (i == 0 and j > n)]
        m = gp.Model("pdptw_monolithic")
        m.Params.OutputFlag = 0
        for k, v in self.params.items():
            m.setParam(k, v)
        x = m.addVars(arcs, vtype=GRB.BINARY)
        t = m.addVars(V, lb={i: inst.ready[i] for i in V}, ub={i: inst.due[i] for i in V})
        u = m.addVars(C, lb={i: max(0, inst.demand[i]) for i in C}, ub={i: Q + min(0, inst.demand[i]) for i in C})
        r = m.addVars(C, lb=1, ub=N)
        m.addConstrs(x.sum(i, "*") == 1 for i in C)
        m.addConstrs(x.sum("*", j) == 1 for j in C)
        m.addConstr(t[0] == inst.ready[0])
        for (i, j) in arcs:
            if j == 0:
                over = inst.due[i] + inst.service[i] + d[i, j] - inst.due[0]
                if over > 0:
                    m.addConstr(t[i] + inst.service[i] + d[i, j] <= inst.due[0] + over * (1 - x[i, j]))
                continue
            big = inst.due[i] + inst.service[i] + d[i, j] - inst.ready[j]
            if big > 0:
                m.addConstr(t[j] >= t[i] + inst.service[i] + d[i, j] - big * (1 - x[i, j]))
            if i == 0:
                m.addConstr(u[j] >= inst.demand[j] * x[i, j])
                m.addConstr(r[j] >= j * x[i, j])
                m.addConstr(r[j] <= j + N * (1 - x[i, j]))
            else:
                m.addConstr(u[j] >= u[i] + inst.demand[j] - 2 * Q * (1 - x[i, j]))
                m.addConstr(r[j] >= r[i] - N * (1 - x[i, j]))
                m.addConstr(r[j] <= r[i] + N * (1 - x[i, j]))
        # Rounded distances can violate the triangle inequality by a unit, so the pickup-to-delivery travel bound is the
        # shortest-path distance over all nodes, not the direct arc (service times on the way only add to it).
        sp = {(i, j): (0 if i == j else d[i, j]) for i in V for j in V}
        for k in V:
            for i in V:
                for j in V:
                    if sp[i, k] + sp[k, j] < sp[i, j]:
                        sp[i, j] = sp[i, k] + sp[k, j]
        for k in range(1, n + 1):
            m.addConstr(r[k] == r[n + k])
            m.addConstr(t[n + k] >= t[k] + inst.service[k] + sp[k, n + k])
        m.setObjective(gp.quicksum(d[a] * x[a] for a in arcs), GRB.MINIMIZE)
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Baseline model not optimal (status={m.Status})")
        used = [a for a in arcs if x[a].X > 0.5]
        succ = {i: j for (i, j) in used if i != 0}
        routes = []
        for start in [j for (i, j) in used if i == 0]:
            route, node = [], start
            while node != 0:
                route.append(node)
                node = succ[node]
            routes.append(route)
        return {"best_objective": float(m.ObjVal), "global_lb": float(m.ObjBound), "global_ub": float(m.ObjVal), "routes": routes}
