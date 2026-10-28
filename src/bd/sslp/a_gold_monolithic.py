from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.bd.sslp.a_gen_data import SSLPInstance
from src.utils.base_baseline_class import BaseBaseline, BaselineSolution


def add_recourse(model: gp.Model, inst: SSLPInstance, s: int, y, *, binary: bool = True):
    """Scenario s's recourse block; y is either master variables or fixed 0/1 values."""
    present = [i for i in range(inst.n_clients) if inst.presence[s][i]]
    sites = range(inst.n_sites)
    z = model.addVars(present, sites, vtype=GRB.BINARY if binary else GRB.CONTINUOUS, lb=0.0, ub=1.0)
    v = model.addVars(sites, lb=0.0)
    model.addConstrs(z.sum(i, "*") == 1 for i in present)
    cap = model.addConstrs(gp.quicksum(inst.demand[i][j] * z[i, j] for i in present) - v[j] <= inst.capacity * y[j] for j in sites)
    cost = inst.overflow_penalty * v.sum() - gp.quicksum(inst.revenue[i][j] * z[i, j] for i in present for j in sites)
    return z, v, cap, cost


class SSLPBaseline(BaseBaseline[SSLPInstance, gp.Model]):
    """Deterministic-equivalent MIP over all scenarios."""

    def __init__(self, *, params: dict[str, Any] | None = None, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        self.params = {"Threads": 1, **(params or {})}

    def load_instance(self, instance_name: str) -> SSLPInstance:
        return SSLPInstance.from_json_file(instance_name)

    def build_model(self, inst: SSLPInstance) -> gp.Model:
        model = gp.Model("sslp_de")
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.params.items():
            model.setParam(k, v)
        y = model.addVars(inst.n_sites, vtype=GRB.BINARY, name="y")
        model.addConstr(y.sum() <= inst.max_open)
        blocks, objective = [], gp.quicksum(inst.open_cost[j] * y[j] for j in range(inst.n_sites))
        for s in range(inst.n_scenarios):
            z, _, _, cost = add_recourse(model, inst, s, y)
            blocks.append(z)
            objective += inst.probability[s] * cost
        model.setObjective(objective, GRB.MINIMIZE)
        model._y, model._z = y, blocks  # type: ignore[attr-defined]
        return model

    def solve_model(self, inst: SSLPInstance, model: gp.Model) -> BaselineSolution[gp.Model]:
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Baseline model not optimal (status={model.Status})")
        open_sites = [j for j in range(inst.n_sites) if model._y[j].X > 0.5]  # type: ignore[attr-defined]
        scenarios = [
            {"scenario": s, "assignments": [{"client": i, "site": j} for (i, j), var in z.items() if var.X > 0.5]}
            for s, z in enumerate(model._z)  # type: ignore[attr-defined]
        ]
        return BaselineSolution(
            objective=float(model.ObjVal), vars={f"y_{j}": float(j in open_sites) for j in range(inst.n_sites)},
            best_bound=float(model.ObjBound), model_state=model, metadata={"open_sites": open_sites, "scenarios": scenarios},
        )
