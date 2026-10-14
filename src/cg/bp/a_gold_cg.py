from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.cg.bp.a_gen_data import BinPackingInstance
from src.utils.base_cg_class import (
    BaseColumnGeneration,
    IntegerMasterSolution,
    MasterLPSolution,
    PricingResult,
)


def _aggregate_items(inst: BinPackingInstance) -> tuple[list[int], list[int]]:
    size_counts: dict[int, int] = {}
    for it in inst.items:
        size_counts[it.size] = size_counts.get(it.size, 0) + 1
    sizes = sorted(size_counts.keys())
    demands = [size_counts[s] for s in sizes]
    return sizes, demands


def _trim_surplus_bins(patterns: list[list[int]], usages: dict[int, float], demands: list[int]) -> dict[int, float]:
    """Drop redundant bins in a feasible integer incumbent, in output-conversion order."""
    remaining = list(demands)
    kept = {}
    for index, usage in usages.items():
        for _ in range(round(usage)):
            contents = [min(left, count) for left, count in zip(remaining, patterns[index])]
            if any(contents):
                kept[index] = kept.get(index, 0.0) + 1.0
                remaining = [left - count for left, count in zip(remaining, contents)]
    if any(remaining):
        raise RuntimeError("Integer restricted master incumbent does not cover every item")
    return kept


class BinPackingColumnGeneration(BaseColumnGeneration[BinPackingInstance, list[int]]):
    """Column generation for bin packing, wrapped in BaseColumnGeneration."""

    def __init__(
        self,
        *,
        master_params: dict[str, Any] | None = None,
        pricing_params: dict[str, Any] | None = None,
        max_iters: int = 200,
        rc_tol: float = 1e-6,
        verbose: bool = False,
        time_limit: float = 60.0,
    ) -> None:
        super().__init__(max_iters=max_iters, verbose=verbose, time_limit=time_limit)
        self.master_params = master_params or {}
        self.pricing_params = pricing_params or {}
        self.rc_tolerance = rc_tol

    def load_instance(self, instance_name: str) -> BinPackingInstance:
        return BinPackingInstance.from_json_file(instance_name)

    def initial_patterns(self, inst: BinPackingInstance) -> list[list[int]]:
        sizes, demands = _aggregate_items(inst)
        patterns: list[list[int]] = []
        m = len(sizes)
        for i in range(m):
            max_cnt = min(demands[i], inst.bin_capacity // sizes[i])
            if max_cnt > 0:
                pat = [0] * m
                pat[i] = max_cnt
                patterns.append(pat)
        if not patterns:
            raise RuntimeError("No initial patterns constructed; check instance.")
        return patterns

    def solve_restricted_master(
        self,
        inst: BinPackingInstance,
        patterns: list[list[int]],
    ) -> MasterLPSolution[list[int]]:
        sizes, demands = _aggregate_items(inst)
        m = len(sizes)
        P = range(len(patterns))
        items = range(m)

        model = gp.Model("bp_master_lp")
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.master_params.items():
            model.setParam(k, v)

        lam = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0.0, name="lambda")
        demand_constrs = []
        for i in items:
            c = model.addConstr(
                gp.quicksum(patterns[p][i] * lam[p] for p in P) >= demands[i],
                name=f"demand_{i}",
            )
            demand_constrs.append(c)

        model.setObjective(gp.quicksum(lam[p] for p in P), GRB.MINIMIZE)
        self.runtime.optimize(model, role="restricted_master", allow_incumbent=False)
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Master LP not optimal (status={model.Status})")

        duals = [c.Pi for c in demand_constrs]
        obj = float(model.ObjVal)

        return MasterLPSolution(objective=obj, duals=duals, metadata={"patterns": patterns})


    def solve_pricing(
        self,
        inst: BinPackingInstance,
        master_solution: MasterLPSolution[list[int]],
        patterns: list[list[int]],
    ) -> PricingResult[list[int]]:
        sizes, demands = _aggregate_items(inst)
        m = len(sizes)
        items = range(m)

        model = gp.Model("bp_pricing")
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.pricing_params.items():
            model.setParam(k, v)

        model.Params.PoolSearchMode = 2
        model.Params.PoolSolutions = 64
        a = model.addVars(items, vtype=GRB.INTEGER, lb=0, ub={i: demands[i] for i in items}, name="a")
        model.addConstr(gp.quicksum(sizes[i] * a[i] for i in items) <= inst.bin_capacity, name="capacity")
        model.addConstr(gp.quicksum(a[i] for i in items) >= 1, name="nontrivial")

        duals = master_solution.duals
        model.setObjective(gp.quicksum(duals[i] * a[i] for i in items), GRB.MAXIMIZE)
        self.runtime.optimize(model, role="pricing", allow_incumbent=False)
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Pricing problem not optimal (status={model.Status})")

        z_price = float(model.ObjVal)
        pattern = [round(a[i].X) for i in items]
        reduced_cost = 1.0 - float(model.ObjBound)
        known = {tuple(pat) for pat in patterns}
        for index in range(model.SolCount):
            model.Params.SolutionNumber = index
            candidate = [round(a[i].Xn) for i in items]
            # The base loop adds the improving best column; pool alternatives enrich the integer RMP.
            if candidate == pattern and reduced_cost < -self.rc_tolerance:
                continue
            if tuple(candidate) not in known:
                patterns.append(candidate)
                known.add(tuple(candidate))
        return PricingResult(pattern=pattern, reduced_cost=reduced_cost, metadata={"dual_value": z_price, "pricing_pool_solutions": model.SolCount, "lp_bound": master_solution.objective / max(1.0, float(model.ObjBound))})


    def solve_integer_master(
        self,
        inst: BinPackingInstance,
        patterns: list[list[int]],
    ) -> IntegerMasterSolution[list[int]]:
        sizes, demands = _aggregate_items(inst)
        m = len(sizes)
        items = range(m)
        P = range(len(patterns))

        model = gp.Model("bp_master_int")
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.master_params.items():
            model.setParam(k, v)

        x = model.addVars(P, vtype=GRB.INTEGER, lb=0, name="x")

        for i in items:
            model.addConstr(gp.quicksum(patterns[p][i] * x[p] for p in P) >= demands[i], name=f"demand_{i}")

        model.setObjective(gp.quicksum(x[p] for p in P), GRB.MINIMIZE)
        self.runtime.optimize(model, role="integer_master", allow_incumbent=True)
        if model.SolCount == 0:
            raise RuntimeError(f"Integer master not optimal (status={model.Status})")

        # A TIME_LIMIT incumbent may contain redundant bins. Normalize before committing
        # the objective, usages, and output metadata, so all three describe the same plan.
        usage_dict = _trim_surplus_bins(patterns, {p: float(x[p].X) for p in P if x[p].X > 1e-6}, demands)
        used_patterns = []
        for p in P:
            val = usage_dict.get(p, 0.0)
            if val > 1e-6:
                used_patterns.append(
                    {
                        "pattern_id": p,
                        "usage": int(round(val)),
                        "contents": [{"item_id": i, "count": patterns[p][i]} for i in items if patterns[p][i] > 0],
                    }
                )

        return IntegerMasterSolution(
            objective=sum(usage_dict.values()),
            pattern_usages=usage_dict,
            metadata={"patterns": patterns, "solution_patterns": used_patterns, "restricted_master_objective": float(model.ObjVal)},
        )
