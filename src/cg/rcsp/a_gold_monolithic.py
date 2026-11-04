from __future__ import annotations

import math
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.cg.rcsp.a_gen_data import RCSPInstance, enumerate_duties
from src.utils.base_baseline_class import BaseBaseline, BaselineSolution


class RCSPBaseline(BaseBaseline[RCSPInstance, gp.Model]):
    """Set covering over every feasible duty (an exact extended formulation at this size).

    All duties are enumerated by depth-first search over the connection graph from every SOURCE connection
    (`a_gen_data.enumerate_duties`, which applies the labour rules directly); the graph is acyclic and has at most
    one connection per (type, tail, head), so a duty is identified by its crew base and task sequence.
    x_k = 1 if duty k is used; every task is covered at least once; objective = sum of the duty costs.
    """

    def __init__(self, *, params: dict[str, Any] | None = None, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        # Zero gap tolerances: costs are integers, so a bound within 0.5 of the incumbent proves it exactly.
        self.params = {"Threads": 1, "MIPGap": 0.0, "MIPGapAbs": 0.5, **(params or {})}

    def load_instance(self, instance_name: str) -> RCSPInstance:
        return RCSPInstance.from_json_file(instance_name)

    def build_model(self, inst: RCSPInstance) -> gp.Model:
        duties = enumerate_duties(inst)
        model = gp.Model("rcsp_monolithic")
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.params.items():
            model.setParam(k, v)
        x = model.addVars(len(duties), vtype=GRB.BINARY, name="x")
        by_task: list[list[int]] = [[] for _ in range(inst.n_tasks)]
        for k, (_, path, _) in enumerate(duties):
            for t in path:
                by_task[t].append(k)
        for t, ks in enumerate(by_task):
            if not ks:
                raise RuntimeError(f"task {t} is in no feasible duty")
            model.addConstr(gp.quicksum(x[k] for k in ks) >= 1, name=f"cover[{t}]")
        model.setObjective(gp.quicksum(cost * x[k] for k, (_, _, cost) in enumerate(duties)), GRB.MINIMIZE)
        model._x, model._duties = x, duties  # type: ignore[attr-defined]
        return model

    def solve_model(self, inst: RCSPInstance, model: gp.Model) -> BaselineSolution[gp.Model]:
        model.optimize()
        if model.Status != GRB.OPTIMAL or math.ceil(model.ObjBound - 1e-6) < model.ObjVal - 1e-6:
            raise RuntimeError(f"Baseline model not proved optimal (status={model.Status}, bound={model.ObjBound}, incumbent={model.ObjVal})")
        x, duties = model._x, model._duties  # type: ignore[attr-defined]
        plan = [{"depot": base, "tasks": list(path)} for k, (base, path, _) in enumerate(duties) if x[k].X > 0.5]
        return BaselineSolution(objective=float(model.ObjVal), vars={}, best_bound=float(model.ObjBound), model_state=model,
                                metadata={"duties": plan, "n_duties_enumerated": len(duties)})
