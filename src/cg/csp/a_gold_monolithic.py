from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.cg.csp.a_gen_data import CuttingStockInstance
from src.utils.base_baseline_class import BaseBaseline, BaselineSolution


def _extract(inst: CuttingStockInstance) -> tuple[list[int], list[int]]:
    items = sorted(inst.item_types, key=lambda it: it.id)
    widths = [it.size for it in items]
    demands = [it.demand for it in items]
    return widths, demands


class CuttingStockBaseline(BaseBaseline[CuttingStockInstance, gp.Model]):
    """Monolithic MIP for cutting stock."""

    def __init__(self, *, params: dict[str, Any] | None = None, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        self.params = params or {}

    def load_instance(self, instance_name: str) -> CuttingStockInstance:
        return CuttingStockInstance.from_json_file(instance_name)

    def build_model(self, inst: CuttingStockInstance) -> gp.Model:
        widths, demands = _extract(inst)
        n_items = len(widths)
        max_rolls = sum(demands)

        model = gp.Model("csp_monolithic")
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.params.items():
            model.setParam(k, v)

        rolls = range(max_rolls)
        items = range(n_items)

        y = model.addVars(rolls, vtype=GRB.BINARY, name="y")
        x = model.addVars(items, rolls, vtype=GRB.INTEGER, lb=0, name="x")

        for r in rolls:
            model.addConstr(gp.quicksum(widths[i] * x[i, r] for i in items) <= inst.stock_size * y[r], name=f"cap_{r}")
        for i in items:
            model.addConstr(gp.quicksum(x[i, r] for r in rolls) >= demands[i], name=f"demand_{i}")

        model.setObjective(gp.quicksum(y[r] for r in rolls), GRB.MINIMIZE)
        model._vars = {"x": x, "y": y, "rolls": list(rolls), "widths": widths, "demands": demands}  # type: ignore[attr-defined]
        return model

    def solve_model(self, inst: CuttingStockInstance, model: gp.Model) -> BaselineSolution[gp.Model]:
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Baseline model not optimal (status={model.Status})")

        vars_ = model._vars  # type: ignore[attr-defined]
        x = vars_["x"]
        y = vars_["y"]
        rolls = vars_["rolls"]
        widths: list[int] = vars_["widths"]

        x_val = np.asarray([[x[i, r].X for r in rolls] for i in range(len(widths))], dtype=float)
        y_val = np.asarray([y[r].X for r in rolls], dtype=float)

        var_dict: dict[str, Any] = {}
        for r in rolls:
            var_dict[f"y_{r}"] = float(y[r].X)
            for i in range(len(widths)):
                var_dict[f"x_{i}_{r}"] = float(x[i, r].X)

        patterns = []
        pid = 0
        for r in rolls:
            if y[r].X <= 0.5:
                continue
            cuts = []
            for i in range(len(widths)):
                cnt = round(x[i, r].X)
                if cnt > 0:
                    cuts.append({"item_id": i, "count": cnt})
            patterns.append({"pattern_id": pid, "usage": 1, "cuts": cuts})
            pid += 1

        best_bound = float(model.ObjBound) if model.ObjBound < 1e20 else None
        return BaselineSolution(
            objective=float(model.ObjVal),
            vars=var_dict,
            best_bound=best_bound,
            model_state=model,
            metadata={"patterns": patterns, "y_assignment": y_val.tolist(), "x_assignment": x_val.tolist()},
        )
