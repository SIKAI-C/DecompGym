from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from benchmark.benders_decomp.base_baseline_class import BaseBaseline, BaselineSolution
from tsso.a_gen_data import TSSOInstance


class TSSOBaseline(BaseBaseline[TSSOInstance, gp.Model]):
    """Deterministic equivalent MILP solve for the two-stage stochastic program."""

    def __init__(self, *, params: dict[str, Any] | None = None, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        self.params = params or {}

    def load_instance(self, instance_name: str) -> TSSOInstance:
        return TSSOInstance.from_json_file(instance_name)

    def build_model(self, inst: TSSOInstance) -> gp.Model:
        I, T, S = inst.n_products, inst.n_periods, inst.n_scenarios

        model = gp.Model()
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.params.items():
            model.setParam(k, v)

        y = model.addMVar((I, T), vtype=GRB.BINARY, name="y")
        x_reg = model.addMVar((I, T, S), lb=0.0, name="x_reg")
        x_ot = model.addMVar((I, T, S), lb=0.0, name="x_ot")
        inv = model.addMVar((I, T, S), lb=0.0, name="inv")
        back = model.addMVar((I, T, S), lb=0.0, name="back")

        for s in range(S):
            for i in range(I):
                for t in range(T):
                    inv_prev = inst.I0[i] if t == 0 else inv[i, t - 1, s]
                    back_prev = inst.B0[i] if t == 0 else back[i, t - 1, s]
                    rhs = inst.demand[i, t, s] - inv_prev + back_prev
                    model.addConstr(
                        x_reg[i, t, s] + x_ot[i, t, s] - inv[i, t, s] + back[i, t, s] == rhs,
                        name=f"balance_{i}_{t}_{s}",
                    )
                    model.addConstr(
                        x_reg[i, t, s] <= inst.K[i, t] * y[i, t],
                        name=f"capacity_{i}_{t}_{s}",
                    )

        recourse_cost = gp.LinExpr()
        for s in range(S):
            recourse_cost += float(inst.prob[s]) * (
                (inst.c_reg * x_reg[:, :, s]).sum()
                + (inst.c_ot * x_ot[:, :, s]).sum()
                + (inst.h * inv[:, :, s]).sum()
                + (inst.b * back[:, :, s]).sum()
            )

        first_stage_cost = (inst.F * y).sum()
        model.setObjective(first_stage_cost + recourse_cost, GRB.MINIMIZE)
        model._vars = {"y": y, "x_reg": x_reg, "x_ot": x_ot, "inv": inv, "back": back}
        return model

    def solve_model(self, inst: TSSOInstance, model: gp.Model) -> BaselineSolution[gp.Model]:
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Baseline model not optimal (status={model.Status})")

        vars_ = model._vars  # type: ignore[attr-defined]
        y_val = np.asarray(vars_["y"].X, dtype=float)
        x_reg_val = np.asarray(vars_["x_reg"].X, dtype=float)
        x_ot_val = np.asarray(vars_["x_ot"].X, dtype=float)
        inv_val = np.asarray(vars_["inv"].X, dtype=float)
        back_val = np.asarray(vars_["back"].X, dtype=float)

        var_dict: dict[str, Any] = {
            "y": y_val.tolist(),
            "x_reg": x_reg_val.tolist(),
            "x_ot": x_ot_val.tolist(),
            "inv": inv_val.tolist(),
            "back": back_val.tolist(),
        }

        best_bound = float(model.ObjBound) if model.ObjBound < 1e20 else None
        return BaselineSolution(
            objective=float(model.ObjVal),
            vars=var_dict,
            best_bound=best_bound,
            model_state=model,
            metadata={
                "y_assignment": y_val.tolist(),
                "recourse": {
                    "x_reg": x_reg_val.tolist(),
                    "x_ot": x_ot_val.tolist(),
                    "inv": inv_val.tolist(),
                    "back": back_val.tolist(),
                },
            },
        )
