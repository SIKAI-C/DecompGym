from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.cg.shift.a_gen_data import ShiftInstance
from src.utils.base_cg_class import (
    BaseColumnGeneration,
    IntegerMasterSolution,
    MasterLPSolution,
    PricingResult,
)


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


class ShiftColumnGeneration(BaseColumnGeneration[ShiftInstance, tuple[int, int]]):
    """Column generation for shift scheduling with contiguous shifts."""

    def __init__(
        self,
        *,
        master_params: dict[str, Any] | None = None,
        max_iters: int = 200,
        rc_tol: float = 1e-6,
        verbose: bool = False,
    ) -> None:
        super().__init__(max_iters=max_iters, verbose=verbose)
        self.master_params = master_params or {}
        self.rc_tolerance = rc_tol
        self._all_patterns: list[tuple[int, int]] | None = None

    def load_instance(self, instance_name: str) -> ShiftInstance:
        return ShiftInstance.from_json_file(instance_name)

    def initial_patterns(self, inst: ShiftInstance) -> list[tuple[int, int]]:
        if self._all_patterns is None:
            self._all_patterns = _enumerate_shifts(inst)
        # Seed with single-slot shifts for feasibility
        return [(t, t) for t in range(len(inst.time_slots))]

    def solve_restricted_master(
        self,
        inst: ShiftInstance,
        patterns: list[tuple[int, int]],
    ) -> MasterLPSolution[tuple[int, int]]:
        T = len(inst.time_slots)
        slots = range(T)
        P = range(len(patterns))

        model = gp.Model("shift_master_lp")
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.master_params.items():
            model.setParam(k, v)

        lam = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0.0, name="lambda")

        demand_constrs = []
        for t in slots:
            cstr = model.addConstr(
                gp.quicksum(lam[p] for p, (s, e) in enumerate(patterns) if s <= t <= e) >= inst.demand[t],
                name=f"cover_{t}",
            )
            demand_constrs.append(cstr)

        model.setObjective(gp.quicksum(lam[p] for p in P), GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Shift master LP not optimal (status={model.Status})")

        duals = [c.Pi for c in demand_constrs]
        return MasterLPSolution(objective=float(model.ObjVal), duals=duals)

    def solve_pricing(
        self,
        inst: ShiftInstance,
        master_solution: MasterLPSolution[tuple[int, int]],
        patterns: list[tuple[int, int]],
    ) -> PricingResult[tuple[int, int]]:
        duals = master_solution.duals  # list[float] per slot
        if self._all_patterns is None:
            self._all_patterns = _enumerate_shifts(inst)

        best_rc = 0.0
        best_pat: tuple[int, int] | None = None
        for pat in self._all_patterns:
            s, e = pat
            dual_sum = sum(duals[t] for t in range(s, e + 1))
            rc = 1.0 - dual_sum  # cost=1 per shift
            if rc < best_rc - 1e-12:
                best_rc = rc
                best_pat = pat

        return PricingResult(pattern=best_pat, reduced_cost=best_rc)

    def solve_integer_master(
        self,
        inst: ShiftInstance,
        patterns: list[tuple[int, int]],
    ) -> IntegerMasterSolution[tuple[int, int]]:
        T = len(inst.time_slots)
        slots = range(T)
        P = range(len(patterns))

        model = gp.Model("shift_master_int")
        model.Params.OutputFlag = 1 if self.verbose else 0

        x = model.addVars(P, vtype=GRB.INTEGER, lb=0, name="x")
        for t in slots:
            model.addConstr(
                gp.quicksum(x[p] for p, (s, e) in enumerate(patterns) if s <= t <= e) >= inst.demand[t],
                name=f"cover_{t}",
            )

        model.setObjective(gp.quicksum(x[p] for p in P), GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Shift integer master not optimal (status={model.Status})")

        usage: dict[int, float] = {}
        shifts_out = []
        for p in P:
            val = x[p].X
            if val > 1e-6:
                usage[p] = float(val)
                s, e = patterns[p]
                shifts_out.append({"start": s, "end": e, "usage": round(val)})

        return IntegerMasterSolution(
            objective=float(model.ObjVal),
            pattern_usages=usage,
            metadata={"shifts": shifts_out},
        )
