from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.ufl.a_gen_data import UFLInstance
from src.utils.base_bd_class import BaseBenders, MasterSolution, SubproblemResult


@dataclass
class UflOptCut:
    customer: int
    cost_at_ybar: float
    lam: list[float]
    ybar: list[float]


@dataclass
class UflMasterState:
    model: gp.Model
    y: gp.MVar
    theta: gp.MVar


class UFLBenders(BaseBenders[UFLInstance, UflMasterState, None, UflOptCut]):
    """Benders decomposition for UFL using Gurobi subproblems."""

    def __init__(
        self,
        *,
        master_params: dict[str, Any] | None = None,
        subproblem_params: dict[str, Any] | None = None,
        max_iters: int = 250,
        verbose: bool = False,
    ) -> None:
        super().__init__(max_iters=max_iters, verbose=verbose)
        self.master_params = master_params or {}
        self.subproblem_params = subproblem_params or {}

    def load_instance(self, instance_name: str) -> UFLInstance:
        return UFLInstance.from_json_file(instance_name)

    def build_initial_master(self, inst: UFLInstance) -> UflMasterState:
        m, n = inst.m, inst.n
        model = gp.Model()
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.master_params.items():
            model.setParam(k, v)

        y = model.addMVar(m, vtype=GRB.BINARY, name="y")
        theta = model.addMVar(n, lb=0.0, name="theta")

        model.addConstr(y.sum() >= 1, name="open_at_least_one")
        model.setObjective(inst.f @ y + theta.sum(), GRB.MINIMIZE)
        return UflMasterState(model=model, y=y, theta=theta)

    def solve_master(self, inst: UFLInstance, master: UflMasterState) -> MasterSolution[UflMasterState]:
        master.model.optimize()
        if master.model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(f"Master solve failed with status {master.model.Status}")

        y_val = np.asarray(master.y.X, dtype=float)
        theta_val = np.asarray(master.theta.X, dtype=float)

        vars_dict: dict[str, Any] = {}
        for i, val in enumerate(y_val):
            vars_dict[f"y_{i}"] = float(val)
        for j, val in enumerate(theta_val):
            vars_dict[f"theta_{j}"] = float(val)

        return MasterSolution(
            objective=float(master.model.ObjVal),
            vars=vars_dict,
            master_state=master,
        )

    def _solve_subproblem(self, d: np.ndarray, y_vals: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        m = len(d)
        model = gp.Model()
        model.Params.OutputFlag = 0
        model.Params.InfUnbdInfo = 1
        for k, v in self.subproblem_params.items():
            model.setParam(k, v)

        x = model.addMVar(m, lb=0.0, name="x")
        model.addConstr(x.sum() == 1, name="assign")
        link_constr = model.addConstrs((x[i] <= float(y_vals[i]) for i in range(m)), name="link")

        model.setObjective(d @ x, GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Subproblem not optimal (status={model.Status})")

        lam = np.array([link_constr[i].Pi for i in range(m)], dtype=float)
        x_val = np.asarray(x.X, dtype=float)
        return float(model.ObjVal), lam, x_val

    def solve_subproblems(
        self,
        inst: UFLInstance,
        master_solution: MasterSolution[UflMasterState],
    ) -> SubproblemResult[None, UflOptCut]:
        m, n = inst.m, inst.n
        y_vals = np.array([master_solution.vars[f"y_{i}"] for i in range(m)], dtype=float)
        theta_vals = np.array([master_solution.vars[f"theta_{j}"] for j in range(n)], dtype=float)

        cuts: list[UflOptCut] = []
        worst_violation = 0.0
        total_sub_cost = 0.0

        x_mat = np.zeros((m, n), dtype=float)
        for j in range(n):
            zj, lam_j, x_j = self._solve_subproblem(inst.D[:, j], y_vals)
            x_mat[:, j] = x_j
            total_sub_cost += zj
            violation = zj - theta_vals[j]
            worst_violation = max(worst_violation, violation)
            if violation > self.violation_tolerance:
                cuts.append(
                    UflOptCut(
                        customer=j,
                        cost_at_ybar=zj,
                        lam=lam_j.tolist(),
                        ybar=y_vals.tolist(),
                    )
                )

        metadata = {
            "total_sub_cost": total_sub_cost,
            "theta_sum": float(theta_vals.sum()),
            "x_assignment": x_mat.tolist(),
        }
        return SubproblemResult(
            feasibility_cuts=[],
            optimality_cuts=cuts,
            all_feasible=True,
            worst_violation=worst_violation,
            sub_bound=total_sub_cost,  # not used directly; compute bounds is overridden
            metadata=metadata,
        )

    def compute_iteration_bounds(
        self,
        inst: UFLInstance,
        master_solution: MasterSolution[UflMasterState],
        sub_result: SubproblemResult[None, UflOptCut],
    ) -> tuple[float, float]:
        # Master objective is a lower bound; UB estimated via subproblem costs.
        master_obj = master_solution.objective
        sub_cost = sub_result.metadata.get("total_sub_cost", None) if sub_result.metadata else None
        theta_sum = sub_result.metadata.get("theta_sum", None) if sub_result.metadata else None

        if sub_cost is None or theta_sum is None:
            return super().compute_iteration_bounds(inst, master_solution, sub_result)

        delta = sub_cost - theta_sum
        ub_candidate = master_obj + delta

        if math.isinf(self.global_lb):
            self.global_lb = master_obj
        else:
            self.global_lb = max(self.global_lb, master_obj)

        if math.isinf(self.global_ub):
            self.global_ub = ub_candidate
        else:
            self.global_ub = min(self.global_ub, ub_candidate)

        return self.global_lb, self.global_ub

    def add_feasibility_cuts(
        self,
        inst: UFLInstance,
        master: UflMasterState,
        cuts: Sequence[None],
    ) -> None:
        # UFL has no feasibility cuts.
        return

    def add_optimality_cuts(
        self,
        inst: UFLInstance,
        master: UflMasterState,
        cuts: Sequence[UflOptCut],
    ) -> None:
        for idx, cut in enumerate(cuts):
            expr = cut.cost_at_ybar + gp.quicksum(cut.lam[i] * (master.y[i] - cut.ybar[i]) for i in range(inst.m))
            master.model.addConstr(master.theta[cut.customer] >= expr, name=f"opt_cut_{cut.customer}_{idx}")
