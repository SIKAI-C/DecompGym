from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.cg.gap.a_gen_data import GAPInstance
from src.utils.base_baseline_class import BaseBaseline, BaselineSolution


def _extract(inst: GAPInstance) -> tuple[list[list[int]], list[list[float]], list[int]]:
    machines = sorted(inst.machines, key=lambda m: m.id)
    jobs = sorted(inst.jobs, key=lambda j: j.id)
    b = [m.capacity for m in machines]
    n_jobs = len(jobs)
    n_machines = len(machines)
    a = [[0] * n_machines for _ in range(n_jobs)]
    c = [[0.0] * n_machines for _ in range(n_jobs)]
    for job in jobs:
        for assign in job.assignments:
            j = int(assign.machine_id)
            a[job.id][j] = assign.resource
            c[job.id][j] = assign.cost
    return a, c, b


class GAPBaseline(BaseBaseline[GAPInstance, gp.Model]):
    """Monolithic MIP for GAP."""

    def __init__(self, *, params: dict[str, Any] | None = None, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        self.params = params or {}

    def load_instance(self, instance_name: str) -> GAPInstance:
        return GAPInstance.from_json_file(instance_name)

    def build_model(self, inst: GAPInstance) -> gp.Model:
        a, c, b = _extract(inst)
        n_jobs = len(a)
        n_machines = len(b)
        jobs = range(n_jobs)
        machines = range(n_machines)

        model = gp.Model("gap_monolithic")
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.params.items():
            model.setParam(k, v)

        x = model.addVars(n_jobs, n_machines, vtype=GRB.BINARY, name="x")

        for i in jobs:
            model.addConstr(gp.quicksum(x[i, j] for j in machines) == 1, name=f"assign_{i}")
        for j in machines:
            model.addConstr(gp.quicksum(a[i][j] * x[i, j] for i in jobs) <= b[j], name=f"cap_{j}")

        model.setObjective(gp.quicksum(c[i][j] * x[i, j] for i in jobs for j in machines), GRB.MINIMIZE)
        model._vars = {"x": x, "a": a, "c": c}  # type: ignore[attr-defined]
        return model

    def solve_model(self, inst: GAPInstance, model: gp.Model) -> BaselineSolution[gp.Model]:
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"GAP monolithic not optimal (status={model.Status})")

        vars_ = model._vars  # type: ignore[attr-defined]
        x = vars_["x"]
        n_jobs = len(inst.jobs)
        n_machines = len(inst.machines)

        var_dict: dict[str, Any] = {}
        assignments = []
        x_mat = np.zeros((n_jobs, n_machines), dtype=float)
        for i in range(n_jobs):
            for j in range(n_machines):
                val = float(x[i, j].X)
                var_dict[f"x_{i}_{j}"] = val
                x_mat[i, j] = val
                if val > 0.5:
                    assignments.append({"job_id": i, "machine_id": j})

        best_bound = float(model.ObjBound) if model.ObjBound < 1e20 else None
        return BaselineSolution(
            objective=float(model.ObjVal),
            vars=var_dict,
            best_bound=best_bound,
            model_state=model,
            metadata={"assignments": assignments, "x_assignment": x_mat.tolist()},
        )
