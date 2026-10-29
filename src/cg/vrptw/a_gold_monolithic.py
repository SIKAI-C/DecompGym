from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.cg.vrptw.a_gen_data import VRPTWInstance
from src.utils.base_baseline_class import BaseBaseline, BaselineSolution


class VRPTWBaseline(BaseBaseline[VRPTWInstance, gp.Model]):
    """Two-index arc MIP with service-start times and loads (MTZ-type big-M linking)."""

    def __init__(self, *, params: dict[str, Any] | None = None, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        self.params = {"Threads": 1, **(params or {})}

    def load_instance(self, instance_name: str) -> VRPTWInstance:
        return VRPTWInstance.from_json_file(instance_name)

    def build_model(self, inst: VRPTWInstance) -> gp.Model:
        n, Q = inst.n_customers, inst.capacity
        V, C = range(n + 1), range(1, n + 1)
        d = {(i, j): inst.dist(i, j) for i in V for j in V if i != j}
        # Prune arcs that can never be time- or capacity-feasible.
        arcs = [(i, j) for (i, j) in d if (i == 0 or j == 0 or (inst.ready[i] + inst.service[i] + d[i, j] <= inst.due[j] and inst.demand[i] + inst.demand[j] <= Q))]
        model = gp.Model("vrptw_monolithic")
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.params.items():
            model.setParam(k, v)
        x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
        t = model.addVars(V, lb={i: inst.ready[i] for i in V}, ub={i: inst.due[i] for i in V}, name="t")
        u = model.addVars(C, lb={i: inst.demand[i] for i in C}, ub=Q, name="u")
        model.addConstrs((x.sum(i, "*") == 1 for i in C), name="out")
        model.addConstrs((x.sum("*", j) == 1 for j in C), name="in")
        model.addConstr(t[0] == inst.ready[0])
        for (i, j) in arcs:
            if j == 0:
                if i != 0:
                    over = inst.due[i] + inst.service[i] + d[i, j] - inst.due[0]
                    if over > 0:
                        model.addConstr(t[i] + inst.service[i] + d[i, j] <= inst.due[0] + over * (1 - x[i, j]))
                continue
            big_m = inst.due[i] + inst.service[i] + d[i, j] - inst.ready[j]
            if big_m > 0:
                model.addConstr(t[j] >= t[i] + inst.service[i] + d[i, j] - big_m * (1 - x[i, j]))
            if i != 0:
                model.addConstr(u[j] >= u[i] + inst.demand[j] - Q * (1 - x[i, j]))
        model.setObjective(gp.quicksum(d[a] * x[a] for a in arcs), GRB.MINIMIZE)
        model._x = x  # type: ignore[attr-defined]
        return model

    def solve_model(self, inst: VRPTWInstance, model: gp.Model) -> BaselineSolution[gp.Model]:
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Baseline model not optimal (status={model.Status})")
        used = [(i, j) for (i, j), var in model._x.items() if var.X > 0.5]  # type: ignore[attr-defined]
        succ = {i: j for (i, j) in used if i != 0}
        routes = []
        for start in [j for (i, j) in used if i == 0]:
            route, node = [], start
            while node != 0:
                route.append(node)
                node = succ[node]
            routes.append(route)
        return BaselineSolution(objective=float(model.ObjVal), vars={}, best_bound=float(model.ObjBound), model_state=model, metadata={"routes": routes})
