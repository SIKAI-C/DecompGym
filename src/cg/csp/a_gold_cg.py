from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.cg.csp.a_gen_data import CuttingStockInstance
from src.utils.base_cg_class import (
    BaseColumnGeneration,
    IntegerMasterSolution,
    MasterLPSolution,
    PricingResult,
)


def _extract(inst: CuttingStockInstance) -> tuple[list[int], list[int]]:
    items = sorted(inst.item_types, key=lambda it: it.id)
    widths = [it.size for it in items]
    demands = [it.demand for it in items]
    return widths, demands


class CuttingStockColumnGeneration(BaseColumnGeneration[CuttingStockInstance, list[int]]):
    """Column generation for cutting stock using the shared base CG driver."""

    def __init__(
        self,
        *,
        master_params: dict[str, Any] | None = None,
        pricing_params: dict[str, Any] | None = None,
        max_iters: int = 200,
        rc_tol: float = 1e-6,
        verbose: bool = False,
    ) -> None:
        super().__init__(max_iters=max_iters, verbose=verbose)
        self.master_params = master_params or {}
        self.pricing_params = pricing_params or {}
        self.rc_tolerance = rc_tol

    def load_instance(self, instance_name: str) -> CuttingStockInstance:
        return CuttingStockInstance.from_json_file(instance_name)

    def initial_patterns(self, inst: CuttingStockInstance) -> list[list[int]]:
        widths, _ = _extract(inst)
        patterns: list[list[int]] = []
        m = len(widths)
        for i in range(m):
            max_cnt = inst.stock_size // widths[i]
            if max_cnt > 0:
                pat = [0] * m
                pat[i] = max_cnt
                patterns.append(pat)
        if not patterns:
            raise RuntimeError("No initial patterns constructed; check data.")
        return patterns

    def solve_restricted_master(
        self,
        inst: CuttingStockInstance,
        patterns: list[list[int]],
    ) -> MasterLPSolution[list[int]]:
        widths, demands = _extract(inst)
        m = len(widths)
        P = range(len(patterns))
        items = range(m)

        model = gp.Model("csp_master_lp")
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.master_params.items():
            model.setParam(k, v)

        lam = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0.0, name="lambda")
        demand_constrs = []
        for i in items:
            c = model.addConstr(gp.quicksum(patterns[p][i] * lam[p] for p in P) >= demands[i], name=f"demand_{i}")
            demand_constrs.append(c)

        model.setObjective(gp.quicksum(lam[p] for p in P), GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Master LP not optimal (status={model.Status})")

        duals = [c.Pi for c in demand_constrs]
        return MasterLPSolution(objective=float(model.ObjVal), duals=duals, metadata={"patterns": patterns})

    def solve_pricing(
        self,
        inst: CuttingStockInstance,
        master_solution: MasterLPSolution[list[int]],
        patterns: list[list[int]],
    ) -> PricingResult[list[int]]:
        widths, _ = _extract(inst)
        m = len(widths)
        items = range(m)

        model = gp.Model("csp_pricing")
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.pricing_params.items():
            model.setParam(k, v)

        a = model.addVars(items, vtype=GRB.INTEGER, lb=0, name="a")
        model.addConstr(gp.quicksum(widths[i] * a[i] for i in items) <= inst.stock_size, name="capacity")
        model.addConstr(gp.quicksum(a[i] for i in items) >= 1, name="nontrivial")

        duals = master_solution.duals
        model.setObjective(gp.quicksum(duals[i] * a[i] for i in items), GRB.MAXIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Pricing problem not optimal (status={model.Status})")

        z_price = float(model.ObjVal)
        pattern = [round(a[i].X) for i in items]
        reduced_cost = 1.0 - z_price

        return PricingResult(pattern=pattern, reduced_cost=reduced_cost, metadata={"dual_value": z_price})

    def solve_integer_master(
        self,
        inst: CuttingStockInstance,
        patterns: list[list[int]],
    ) -> IntegerMasterSolution[list[int]]:
        widths, demands = _extract(inst)
        m = len(widths)
        items = range(m)
        P = range(len(patterns))

        model = gp.Model("csp_master_int")
        model.Params.OutputFlag = 1 if self.verbose else 0

        x = model.addVars(P, vtype=GRB.INTEGER, lb=0, name="x")

        for i in items:
            model.addConstr(gp.quicksum(patterns[p][i] * x[p] for p in P) >= demands[i], name=f"demand_{i}")

        model.setObjective(gp.quicksum(x[p] for p in P), GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Integer master not optimal (status={model.Status})")

        usage_dict: dict[int, float] = {}
        used_patterns = []
        for p in P:
            val = x[p].X
            if val > 1e-6:
                usage_dict[p] = float(val)
                used_patterns.append(
                    {
                        "pattern_id": p,
                        "usage": round(val),
                        "cuts": [{"item_id": i, "count": patterns[p][i]} for i in items if patterns[p][i] > 0],
                    }
                )

        return IntegerMasterSolution(
            objective=float(model.ObjVal),
            pattern_usages=usage_dict,
            metadata={"patterns": patterns, "solution_patterns": used_patterns},
        )
