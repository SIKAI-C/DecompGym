from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.ufl.a_gen_data import UFLInstance
from src.utils.base_baseline_class import BaseBaseline, BaselineSolution


class UFLBaseline(BaseBaseline[UFLInstance, gp.Model]):
    """One-shot MILP for UFL using Gurobi."""

    def __init__(self, *, params: dict[str, Any] | None = None, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        self.params = params or {}

    def load_instance(self, instance_name: str) -> UFLInstance:
        return UFLInstance.from_json_file(instance_name)

    def build_model(self, inst: UFLInstance) -> gp.Model:
        m, n = inst.m, inst.n
        model = gp.Model()
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.params.items():
            model.setParam(k, v)

        y = model.addMVar(m, vtype=GRB.BINARY, name="y")
        x = model.addMVar((m, n), vtype=GRB.BINARY, name="x")

        model.addConstrs((x[:, j].sum() == 1 for j in range(n)), name="assign")
        model.addConstrs((x[i, j] <= y[i] for i in range(m) for j in range(n)), name="open_link")

        model.setObjective(inst.f @ y + (inst.D * x).sum(), GRB.MINIMIZE)
        model._vars = {"y": y, "x": x}
        return model

    def solve_model(self, inst: UFLInstance, model: gp.Model) -> BaselineSolution[gp.Model]:
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Baseline model not optimal (status={model.Status})")

        vars_ = model._vars  # type: ignore[attr-defined]
        y_val = np.asarray(vars_["y"].X, dtype=float)
        x_val = np.asarray(vars_["x"].X, dtype=float)

        # Flatten vars into dict
        var_dict: dict[str, Any] = {}
        for i, val in enumerate(y_val):
            var_dict[f"y_{i}"] = float(val)
        for i in range(inst.m):
            for j in range(inst.n):
                var_dict[f"x_{i}_{j}"] = float(x_val[i, j])

        best_bound = float(model.ObjBound) if model.ObjBound < 1e20 else None
        return BaselineSolution(
            objective=float(model.ObjVal),
            vars=var_dict,
            best_bound=best_bound,
            model_state=model,
            metadata={"x_assignment": x_val.tolist(), "y": y_val.tolist()},
        )
