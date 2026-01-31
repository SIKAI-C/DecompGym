from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.cg.shift.a_gen_data import ShiftInstance
from src.utils.base_baseline_class import BaseBaseline, BaselineSolution


def _enumerate_shifts(inst: ShiftInstance) -> list[tuple[int, int]]:
    T = len(inst.time_slots)
    shifts: list[tuple[int, int]] = []
    for start in range(T):
        for length in range(inst.shift_min_length, inst.shift_max_length + 1):
            end = start + length - 1
            if end >= T:
                break
            shifts.append((start, end))
    return shifts


class ShiftBaseline(BaseBaseline[ShiftInstance, gp.Model]):
    """Monolithic set-covering model for shifts."""

    def __init__(self, *, params: dict[str, Any] | None = None, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        self.params = params or {}

    def load_instance(self, instance_name: str) -> ShiftInstance:
        return ShiftInstance.from_json_file(instance_name)

    def build_model(self, inst: ShiftInstance) -> gp.Model:
        shifts = _enumerate_shifts(inst)
        T = len(inst.time_slots)
        shift_idx = range(len(shifts))
        slots = range(T)

        model = gp.Model("shift_monolithic")
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.params.items():
            model.setParam(k, v)

        x = model.addVars(shift_idx, vtype=GRB.INTEGER, lb=0, name="x")

        for t in slots:
            cover_expr = gp.quicksum(x[s] for s, (start, end) in enumerate(shifts) if start <= t <= end)
            model.addConstr(cover_expr >= inst.demand[t], name=f"cover_{t}")

        model.setObjective(gp.quicksum(x[s] for s in shift_idx), GRB.MINIMIZE)
        model._vars = {"x": x, "shifts": shifts}
        return model

    def solve_model(self, inst: ShiftInstance, model: gp.Model) -> BaselineSolution[gp.Model]:
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Shift baseline not optimal (status={model.Status})")

        vars_ = model._vars  # type: ignore[attr-defined]
        x = vars_["x"]
        shifts = vars_["shifts"]
        x_vals = np.array([x[i].X for i in range(len(shifts))], dtype=float)

        var_dict: dict[str, Any] = {}
        shift_list = []
        for idx, (start, end) in enumerate(shifts):
            val = round(x_vals[idx])
            var_dict[f"x_{idx}"] = float(x_vals[idx])
            if val > 0:
                shift_list.append({"start": start, "end": end, "usage": val})

        best_bound = float(model.ObjBound) if model.ObjBound < 1e20 else None
        return BaselineSolution(
            objective=float(model.ObjVal),
            vars=var_dict,
            best_bound=best_bound,
            model_state=model,
            metadata={"shifts": shift_list},
        )
