from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.cg.bp.a_gen_data import BinPackingInstance
from src.utils.base_baseline_class import BaseBaseline, BaselineSolution


def _aggregate_items(inst: BinPackingInstance) -> tuple[list[int], list[int]]:
    """Convert item list into (sizes, demands) by unique size."""
    size_counts: dict[int, int] = {}
    for it in inst.items:
        size_counts[it.size] = size_counts.get(it.size, 0) + 1
    sizes = sorted(size_counts.keys())
    demands = [size_counts[s] for s in sizes]
    return sizes, demands


class BinPackingBaseline(BaseBaseline[BinPackingInstance, gp.Model]):
    """Full MIP for bin packing using BaseBaseline structure."""

    def __init__(self, *, params: dict[str, Any] | None = None, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        self.params = params or {}

    def load_instance(self, instance_name: str) -> BinPackingInstance:
        return BinPackingInstance.from_json_file(instance_name)

    def build_model(self, inst: BinPackingInstance) -> gp.Model:
        sizes, demands = _aggregate_items(inst)
        n_types = len(sizes)
        max_bins = sum(demands)

        model = gp.Model("bp_monolithic")
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.params.items():
            model.setParam(k, v)

        bins = range(max_bins)
        items = range(n_types)

        y = model.addVars(bins, vtype=GRB.BINARY, name="y")
        x = model.addVars(items, bins, vtype=GRB.INTEGER, lb=0, name="x")

        for b in bins:
            model.addConstr(
                gp.quicksum(sizes[i] * x[i, b] for i in items) <= inst.bin_capacity * y[b],
                name=f"cap_{b}",
            )
        for i in items:
            model.addConstr(gp.quicksum(x[i, b] for b in bins) >= demands[i], name=f"demand_{i}")

        model.setObjective(gp.quicksum(y[b] for b in bins), GRB.MINIMIZE)
        model._vars = {"x": x, "y": y, "sizes": sizes, "demands": demands, "bins": list(bins)}  # type: ignore[attr-defined]
        return model

    def solve_model(self, inst: BinPackingInstance, model: gp.Model) -> BaselineSolution[gp.Model]:
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Baseline model not optimal (status={model.Status})")

        vars_ = model._vars  # type: ignore[attr-defined]
        x = vars_["x"]
        y = vars_["y"]
        bins = vars_["bins"]
        sizes: list[int] = vars_["sizes"]

        x_val = np.asarray([[x[i, b].X for b in bins] for i in range(len(sizes))], dtype=float)
        y_val = np.asarray([y[b].X for b in bins], dtype=float)

        var_dict: dict[str, Any] = {}
        for b, yb in y.items():
            var_dict[f"y_{b}"] = float(yb.X)
            for i in range(len(sizes)):
                var_dict[f"x_{i}_{b}"] = float(x[i, b].X)

        patterns = []
        pattern_id = 0
        for b in bins:
            if y[b].X <= 0.5:
                continue
            contents = []
            for i in range(len(sizes)):
                cnt = round(x[i, b].X)
                if cnt > 0:
                    contents.append({"item_id": i, "count": cnt})
            patterns.append({"pattern_id": pattern_id, "usage": 1, "contents": contents})
            pattern_id += 1

        best_bound = float(model.ObjBound) if model.ObjBound < 1e20 else None
        return BaselineSolution(
            objective=float(model.ObjVal),
            vars=var_dict,
            best_bound=best_bound,
            model_state=model,
            metadata={
                "patterns": patterns,
                "y_assignment": y_val.tolist(),
                "x_assignment": x_val.tolist(),
            },
        )
