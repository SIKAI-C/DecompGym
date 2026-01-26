from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from benchmark.benders_decomp.base_baseline_class import BaseBaseline, BaselineSolution
from lrp.a_gen_data import LRPInstance


class LRPBaseline(BaseBaseline[LRPInstance, gp.Model]):
    """Full route-based LRP MILP with Gurobi."""

    def __init__(self, *, params: dict[str, Any] | None = None, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        self.params = params or {}

    def load_instance(self, instance_name: str) -> LRPInstance:
        return LRPInstance.from_json_file(instance_name)

    def build_model(self, inst: LRPInstance) -> gp.Model:
        m, n, R = inst.m, inst.n, inst.R
        model = gp.Model()
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.params.items():
            model.setParam(k, v)

        y = model.addMVar(m, vtype=GRB.BINARY, name="y")
        lam = model.addMVar(R, vtype=GRB.BINARY, name="lambda")

        model.addConstrs(
            (gp.quicksum(inst.incidence[i, r] * lam[r] for r in range(R)) == 1 for i in range(n)),
            name="cover",
        )
        for j in range(m):
            routes_j = [r for r in range(R) if int(inst.route_depot[r]) == j]
            if routes_j:
                model.addConstr(
                    gp.quicksum(lam[r] for r in routes_j) <= inst.max_vehicles[j] * y[j],
                    name=f"fleet_{j}",
                )

        model.addConstr(y.sum() >= 1, name="open_one")
        model.setObjective(inst.depot_cost @ y + inst.route_cost @ lam, GRB.MINIMIZE)
        model._vars = {"y": y, "lambda": lam}
        return model

    def solve_model(self, inst: LRPInstance, model: gp.Model) -> BaselineSolution[gp.Model]:
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Baseline model not optimal (status={model.Status})")

        vars_ = model._vars  # type: ignore[attr-defined]
        y_val = np.asarray(vars_["y"].X, dtype=float)
        lam_val = np.asarray(vars_["lambda"].X, dtype=float)

        var_dict: dict[str, Any] = {}
        for j in range(inst.m):
            var_dict[f"y_{j}"] = float(y_val[j])
        for r in range(inst.R):
            var_dict[f"lambda_{r}"] = float(lam_val[r])

        best_bound = float(model.ObjBound) if model.ObjBound < 1e20 else None
        return BaselineSolution(
            objective=float(model.ObjVal),
            vars=var_dict,
            best_bound=best_bound,
            model_state=model,
            metadata={"y_assignment": y_val.tolist(), "lambda_assignment": lam_val.tolist()},
        )
