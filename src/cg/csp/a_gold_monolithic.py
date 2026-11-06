from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.cg.csp.a_gen_data import CuttingStockInstance
from src.cg.bp.arcflow import build_arcflow, decompose_flow
from src.utils.base_baseline_class import BaseBaseline, BaselineSolution


def _extract(inst: CuttingStockInstance) -> tuple[list[int], list[int]]:
    items = sorted(inst.item_types, key=lambda it: it.id)
    widths = [it.size for it in items]
    demands = [it.demand for it in items]
    return widths, demands


class CuttingStockBaseline(BaseBaseline[CuttingStockInstance, gp.Model]):
    """Arc-flow MIP (Valerio de Carvalho 1999; src/cg/bp/arcflow.py); its LP bound is the Gilmore-Gomory bound."""

    def __init__(self, *, params: dict[str, Any] | None = None, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        self.params = params or {}

    def load_instance(self, instance_name: str) -> CuttingStockInstance:
        return CuttingStockInstance.from_json_file(instance_name)

    def build_model(self, inst):
        sizes, demands = _extract(inst)
        model, state = build_arcflow(sizes, demands, inst.stock_size, "csp_monolithic_arcflow")
        model.Params.OutputFlag = 1 if self.verbose else 0
        model.Params.Threads = 1  # single-thread reference default; the caller sets TimeLimit/MIPGap; user params may override
        for k, v in self.params.items():
            model.setParam(k, v)
        model._vars = state  # type: ignore[attr-defined]
        return model

    def solve_model(self, inst, model: gp.Model) -> BaselineSolution[gp.Model]:
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Baseline model not optimal (status={model.Status})")
        state = model._vars  # type: ignore[attr-defined]
        patterns = [{"pattern_id": p, "usage": usage, "cuts": [{"item_id": i, "count": c} for i, c in enumerate(counts) if c > 0]}
                    for p, (counts, usage) in enumerate(decompose_flow(state))]
        var_dict = {"z": float(state["z"].X)}
        best_bound = float(model.ObjBound) if model.ObjBound < 1e20 else None
        return BaselineSolution(
            objective=float(model.ObjVal),
            vars=var_dict,
            best_bound=best_bound,
            model_state=model,
            metadata={"patterns": patterns, "formulation": "arc-flow"},
        )
