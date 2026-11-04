from __future__ import annotations

import math
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.bd.lbbd_ps.a_gen_data import PSInstance
from src.utils.base_baseline_class import BaseBaseline, BaselineSolution


class PSBaseline(BaseBaseline[PSInstance, gp.Model]):
    """Time-indexed MIP, as in the scale-up `b_mip_xl.py`: x_ijt = 1 if job j starts on facility i at integer time t,
    for every facility j fits on and t in [r_j, d_j - p_ij]:
        min  sum f_ij x_ijt
        s.t. sum_it x_ijt = 1                                                 (every job once)
             sum_j sum_{t' in (t - p_ij, t]} c_ij x_ijt' <= C_i   for all i, t  (cumulative resource at time t)
    Integer starts lose nothing with integer data, so the model is exact. Rows that cannot bind are omitted.
    """

    def __init__(self, *, params: dict[str, Any] | None = None, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        # Zero gap tolerances: costs are integers, so a bound within 0.5 of the incumbent proves it exactly.
        self.params = {"Threads": 1, "MIPGap": 0.0, "MIPGapAbs": 0.5, **(params or {})}

    def load_instance(self, instance_name: str) -> PSInstance:
        return PSInstance.from_json_file(instance_name)

    def build_model(self, inst: PSInstance) -> gp.Model:
        m, n = inst.n_facilities, inst.n_jobs
        keys = [(i, j, t) for i in range(m) for j in range(n) if inst.eligible(i, j)
                for t in range(inst.release[j], inst.deadline[j] - inst.proc[i][j] + 1)]
        model = gp.Model("lbbd_ps_monolithic")
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.params.items():
            model.setParam(k, v)
        x = model.addVars(keys, vtype=GRB.BINARY, name="x")
        model.addConstrs((x.sum("*", j, "*") == 1 for j in range(n)), name="assign")
        running: dict[tuple[int, int], list[tuple[int, gp.Var]]] = {}
        for (i, j, t) in keys:
            for u in range(t, t + inst.proc[i][j]):
                running.setdefault((i, u), []).append((inst.rate[i][j], x[i, j, t]))
        for (i, u), terms in running.items():
            if sum(c for c, _ in terms) > inst.capacity[i]:
                model.addConstr(gp.quicksum(c * v for c, v in terms) <= inst.capacity[i], name=f"cap[{i},{u}]")
        model.setObjective(gp.quicksum(inst.cost[i][j] * x[i, j, t] for (i, j, t) in keys), GRB.MINIMIZE)
        model._x = x  # type: ignore[attr-defined]
        return model

    def solve_model(self, inst: PSInstance, model: gp.Model) -> BaselineSolution[gp.Model]:
        model.optimize()
        if model.Status != GRB.OPTIMAL or math.ceil(model.ObjBound - 1e-6) < model.ObjVal - 1e-6:
            raise RuntimeError(f"Baseline model not proved optimal (status={model.Status})")
        schedule = sorted(({"job": j, "facility": i, "start": t} for (i, j, t), var in model._x.items() if var.X > 0.5),  # type: ignore[attr-defined]
                          key=lambda e: e["job"])
        return BaselineSolution(objective=float(round(model.ObjVal)), vars={}, best_bound=float(model.ObjBound), model_state=model,
                                metadata={"schedule": schedule})
