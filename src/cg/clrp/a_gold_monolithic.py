from __future__ import annotations

import math
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.cg.clrp.a_gen_data import CLRPInstance
from src.utils.base_baseline_class import BaseBaseline, BaselineSolution


class CLRPBaseline(BaseBaseline[CLRPInstance, gp.Model]):
    """The scale-up compact CLRP MIP: two-index vehicle flow with a single-commodity load flow.

    y_d opens depot d, z_id serves customer i from d, x_uv uses arc u -> v, f_uv is the load on it.
    Customers have one in- and one out-arc and absorb their demand; q_v x_uv <= f_uv <= (Q - q_u) x_uv on
    customer arcs, so loads stay within Q and no subtour avoids a depot. z is equal along customer arcs and
    matches the depot on depot arcs (a route returns to the depot it left); sum_i q_i z_id <= W_d y_d.
    Objective: opening costs + travel costs + the vehicle cost of every arc leaving a depot. Rounded capacity
    inequalities x(delta+(S)) >= ceil(q(S) / Q) on customer sets are separated as user cuts at the root, with
    the depots merged into one node (the CVRP separation).
    """

    def __init__(self, *, params: dict[str, Any] | None = None, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        # Zero gap tolerances: costs are integers, so a bound within 0.5 of the incumbent proves it exactly.
        self.params = {"Threads": 1, "MIPGap": 0.0, "MIPGapAbs": 0.5, **(params or {})}

    def load_instance(self, instance_name: str) -> CLRPInstance:
        return CLRPInstance.from_json_file(instance_name)

    def build_model(self, inst: CLRPInstance) -> gp.Model:
        m, n, Q = inst.n_depots, inst.n_customers, inst.vehicle_capacity
        q = {m + i: inst.demand[i] for i in range(n)}
        D, C = range(m), range(m, m + n)
        arcs = [(u, v) for u in range(m + n) for v in range(m + n) if u != v and not (u < m and v < m)]
        arcs = [(u, v) for (u, v) in arcs if not (u >= m and v >= m and q[u] + q[v] > Q)]
        model = gp.Model("clrp_monolithic")
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.params.items():
            model.setParam(k, v)
        y = model.addVars(D, vtype=GRB.BINARY, name="y")
        z = model.addVars(C, D, vtype=GRB.BINARY, name="z")
        x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
        f = model.addVars(arcs, lb=0.0, name="f")
        out = {u: [v for (a, v) in arcs if a == u] for u in range(m + n)}
        inn = {v: [u for (u, b) in arcs if b == v] for v in range(m + n)}
        for i in C:
            model.addConstr(gp.quicksum(x[i, v] for v in out[i]) == 1)
            model.addConstr(gp.quicksum(x[u, i] for u in inn[i]) == 1)
            model.addConstr(gp.quicksum(f[u, i] for u in inn[i]) - gp.quicksum(f[i, v] for v in out[i]) == q[i])
            model.addConstr(z.sum(i, "*") == 1)
        for d in D:
            model.addConstr(gp.quicksum(x[d, v] for v in out[d]) == gp.quicksum(x[u, d] for u in inn[d]))
            model.addConstr(gp.quicksum(q[i] * z[i, d] for i in C) <= inst.depot_capacity[d] * y[d])
            for i in C:
                model.addConstr(z[i, d] <= y[d])
                model.addConstr(x[d, i] <= z[i, d])
                model.addConstr(x[i, d] <= z[i, d])
                model.addConstr(f[i, d] == 0)
        for (u, v) in arcs:
            if u >= m and v >= m:
                model.addConstr(f[u, v] <= (Q - q[u]) * x[u, v])
                model.addConstr(f[u, v] >= q[v] * x[u, v])
                for d in D:
                    model.addConstr(z[u, d] - z[v, d] <= 1 - x[u, v])
                    model.addConstr(z[v, d] - z[u, d] <= 1 - x[u, v])
            elif u < m:
                model.addConstr(f[u, v] <= Q * x[u, v])
                model.addConstr(f[u, v] >= q[v] * x[u, v])
        # Valid inequalities: open depots cover the total demand; at least ceil(total demand / Q) routes.
        model.addConstr(gp.quicksum(inst.depot_capacity[d] * y[d] for d in D) >= sum(inst.demand))
        model.addConstr(gp.quicksum(x[d, v] for d in D for v in out[d]) >= math.ceil(sum(inst.demand) / Q))
        model.setObjective(gp.quicksum(inst.opening_cost[d] * y[d] for d in D)
                           + gp.quicksum(inst.cost(u, v) * x[u, v] for (u, v) in arcs)
                           + gp.quicksum(inst.vehicle_cost * x[d, v] for d in D for v in out[d]), GRB.MINIMIZE)
        # Linking (valid for every optimal plan): the load leaving a depot is the demand assigned to it, and an
        # open depot sends out at least one route (opening an unused depot only adds its positive cost).
        for d in D:
            model.addConstr(gp.quicksum(f[d, v] for v in out[d]) == gp.quicksum(q[i] * z[i, d] for i in C))
            model.addConstr(gp.quicksum(x[d, v] for v in out[d]) >= y[d])
        model._x = x  # type: ignore[attr-defined]
        model._arcs = arcs  # type: ignore[attr-defined]
        return model

    def solve_model(self, inst: CLRPInstance, model: gp.Model) -> BaselineSolution[gp.Model]:
        from src.cg.cvrp.a_gold_monolithic import CVRPBaseline

        m, x, arcs = inst.n_depots, model._x, model._arcs  # type: ignore[attr-defined]
        xs = [x[a] for a in arcs]
        node = lambda u: 0 if u < m else u - m + 1  # depots merged into node 0, customer i -> i + 1
        q = [0] + list(inst.demand)

        def callback(cb, where):
            if where != GRB.Callback.MIPNODE or cb.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL or cb.cbGet(GRB.Callback.MIPNODE_NODCNT) > 0:
                return
            flow: dict[tuple[int, int], float] = {}
            for (u, v), val in zip(arcs, cb.cbGetNodeRel(xs)):
                flow[node(u), node(v)] = flow.get((node(u), node(v)), 0.0) + val
            for S in CVRPBaseline.violated_sets(inst.n_customers, inst.vehicle_capacity, q, flow):
                cb.cbCut(gp.quicksum(x[u, v] for (u, v) in arcs if node(u) in S and node(v) not in S) >= math.ceil(sum(q[i] for i in S) / inst.vehicle_capacity - 1e-9))

        model.Params.PreCrush = 1
        model.optimize(callback)
        if model.Status != GRB.OPTIMAL or math.ceil(model.ObjBound - 1e-6) < model.ObjVal - 1e-6:
            raise RuntimeError(f"Baseline model not proved optimal (status={model.Status}, bound={model.ObjBound}, incumbent={model.ObjVal})")
        m = inst.n_depots
        used = [(u, v) for (u, v), var in model._x.items() if var.X > 0.5]  # type: ignore[attr-defined]
        succ = {u: v for (u, v) in used if u >= m}
        routes = []
        for d, start in [(u, v) for (u, v) in used if u < m]:
            route, node = [], start
            while node >= m:
                route.append(node - m)
                node = succ[node]
            routes.append({"depot": d, "customers": route})
        return BaselineSolution(objective=float(model.ObjVal), vars={}, best_bound=float(model.ObjBound), model_state=model, metadata={"routes": routes})
