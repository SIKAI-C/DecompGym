from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.bd.lbbd.a_gen_data import LBBDInstance
from src.utils.base_baseline_class import BaseBaseline, BaselineSolution


class LBBDBaseline(BaseBaseline[LBBDInstance, gp.Model]):
    """Time-indexed MIP over all machines: z[i, j, t] = 1 if task j starts at time t on machine i."""

    def __init__(self, *, params: dict[str, Any] | None = None, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        self.params = {"Threads": 1, **(params or {})}

    def load_instance(self, instance_name: str) -> LBBDInstance:
        return LBBDInstance.from_json_file(instance_name)

    def build_model(self, inst: LBBDInstance) -> gp.Model:
        H = inst.horizon()
        machines, tasks = range(inst.n_machines), range(inst.n_tasks)
        model = gp.Model("lbbd_monolithic")
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.params.items():
            model.setParam(k, v)
        keys = [(i, j, t) for i in machines for j in tasks for t in range(inst.release[j], H - inst.proc_time[i][j] + 1)]
        cost = {
            (i, j, t): inst.proc_cost[i][j] + inst.weight[j] * max(0, t + inst.proc_time[i][j] - inst.due[j])
            for (i, j, t) in keys
        }
        z = model.addVars(keys, vtype=GRB.BINARY, name="z")
        model.addConstrs((z.sum("*", j, "*") == 1 for j in tasks), name="assign")
        running: dict[tuple[int, int], list] = {}
        for (i, j, t) in keys:
            for u in range(t, t + inst.proc_time[i][j]):
                running.setdefault((i, u), []).append(z[i, j, t])
        model.addConstrs((gp.quicksum(running[k]) <= 1 for k in running), name="one_at_a_time")
        model.setObjective(gp.quicksum(cost[k] * z[k] for k in keys), GRB.MINIMIZE)
        model._z = z  # type: ignore[attr-defined]
        return model

    def solve_model(self, inst: LBBDInstance, model: gp.Model) -> BaselineSolution[gp.Model]:
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Baseline model not optimal (status={model.Status})")
        assignments = [
            {"task_id": j, "machine_id": i, "start": t}
            for (i, j, t), var in model._z.items()  # type: ignore[attr-defined]
            if var.X > 0.5
        ]
        assignments.sort(key=lambda a: a["task_id"])
        return BaselineSolution(
            objective=float(model.ObjVal),
            vars={f"task_{a['task_id']}": [a["machine_id"], a["start"]] for a in assignments},
            best_bound=float(model.ObjBound),
            model_state=model,
            metadata={"assignments": assignments},
        )
