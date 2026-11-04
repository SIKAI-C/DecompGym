from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.bd.scfl.a_gen_data import SCFLInstance
from src.utils.base_baseline_class import BaseBaseline, BaselineSolution


# Objective values are fractional, so optimality is certified by an explicit bound check at this relative tolerance.
CERT_TOL = 1e-7


class SCFLBaseline(BaseBaseline[SCFLInstance, gp.Model]):
    """Extensive form (Bodur et al. 2017, eq. (15)) with strong linking, as in the scale-up MIP:

        min f.y + sum_s p_s sum_ij c_ij x_sij
        s.t. sum_i x_sij = d_sj,  sum_j x_sij <= u_i y_i,  x_sij <= min(d_sj, u_i) y_i,  sum_i u_i y_i >= max_s D_s,
             x >= 0, y binary.
    For a binary y the strong rows are redundant and the rest separates into the scenario transportation LPs.
    """

    def __init__(self, *, params: dict[str, Any] | None = None, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        self.params = {"Threads": 1, "MIPGap": 0.0, "MIPGapAbs": 0.0, **(params or {})}

    def load_instance(self, instance_name: str) -> SCFLInstance:
        return SCFLInstance.from_json_file(instance_name)

    def build_model(self, inst: SCFLInstance) -> gp.Model:
        I, J, S = range(inst.n_facilities), range(inst.n_customers), range(inst.n_scenarios)
        u, d = inst.capacity, inst.demand
        model = gp.Model("scfl_extensive_form")
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.params.items():
            model.setParam(k, v)
        y = model.addVars(I, vtype=GRB.BINARY, name="y")
        x = model.addVars(S, I, J, lb=0.0, name="x")
        model.addConstrs((x.sum(s, "*", j) == d[s][j] for s in S for j in J), name="demand")
        model.addConstrs((x.sum(s, i, "*") <= u[i] * y[i] for s in S for i in I), name="capacity")
        model.addConstrs((x[s, i, j] <= min(d[s][j], u[i]) * y[i] for s in S for i in I for j in J), name="strong")
        model.addConstr(gp.quicksum(u[i] * y[i] for i in I) >= inst.max_demand, name="cover")
        model.setObjective(gp.quicksum(inst.fixed_cost[i] * y[i] for i in I)
                           + gp.quicksum(inst.probability[s] * inst.unit_cost[i][j] * x[s, i, j] for s in S for i in I for j in J), GRB.MINIMIZE)
        model._y = y  # type: ignore[attr-defined]
        return model

    def solve_model(self, inst: SCFLInstance, model: gp.Model) -> BaselineSolution[gp.Model]:
        model.optimize()
        if model.Status != GRB.OPTIMAL or model.ObjVal - model.ObjBound > CERT_TOL * max(1.0, abs(model.ObjVal)):
            raise RuntimeError(f"Baseline model not proved optimal (status={model.Status}, bound={model.ObjBound}, incumbent={model.ObjVal})")
        y = [int(model._y[i].X > 0.5) for i in range(inst.n_facilities)]  # type: ignore[attr-defined]
        return BaselineSolution(objective=float(model.ObjVal), vars={}, best_bound=float(model.ObjBound), model_state=model, metadata={"y": y})
