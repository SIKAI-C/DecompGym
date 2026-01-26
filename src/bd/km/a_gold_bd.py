from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from benchmark.benders_decomp.base_bd_class import BaseBenders, MasterSolution, SubproblemResult
from km.a_gen_data import KMedianInstance


@dataclass
class KMedianOptCut:
    alpha_sum: float
    beta_sum: np.ndarray  # shape (m,)


@dataclass
class KMedianMasterState:
    model: gp.Model
    y: gp.MVar
    z: gp.Var


class KMedianBenders(BaseBenders[KMedianInstance, KMedianMasterState, None, KMedianOptCut]):
    """Benders decomposition for k-median (Gurobi)."""

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

    def load_instance(self, instance_name: str) -> KMedianInstance:
        return KMedianInstance.from_json_file(instance_name)

    def build_initial_master(self, inst: KMedianInstance) -> KMedianMasterState:
        model = gp.Model()
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.master_params.items():
            model.setParam(k, v)

        y = model.addMVar(inst.m, vtype=GRB.BINARY, name="y")
        z = model.addVar(lb=0.0, name="z")

        model.addConstr(y.sum() == inst.k, name="open_k")
        model.setObjective(z, GRB.MINIMIZE)
        return KMedianMasterState(model=model, y=y, z=z)

    def solve_master(self, inst: KMedianInstance, master: KMedianMasterState) -> MasterSolution[KMedianMasterState]:
        master.model.optimize()
        if master.model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(f"Master solve failed (status={master.model.Status})")

        y_val = np.asarray(master.y.X, dtype=float)
        z_val = float(master.z.X)

        vars_dict: dict[str, Any] = {f"y_{i}": float(y_val[i]) for i in range(inst.m)}
        vars_dict["z"] = z_val

        return MasterSolution(
            objective=float(master.model.ObjVal),
            vars=vars_dict,
            master_state=master,
        )

    def _solve_subproblem(
        self,
        inst: KMedianInstance,
        y_vals: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        m, n = inst.m, inst.n
        y_vals = y_vals.reshape(m)

        model = gp.Model()
        model.Params.OutputFlag = 0
        model.Params.InfUnbdInfo = 1
        for k, v in self.subproblem_params.items():
            model.setParam(k, v)

        x = model.addMVar((m, n), lb=0.0, name="x")
        assign_constr = model.addConstrs((x[:, j].sum() == 1 for j in range(n)), name="assign")
        link_constr = model.addConstrs((x[i, j] <= float(y_vals[i]) for i in range(m) for j in range(n)), name="link")

        model.setObjective((inst.D * x).sum(), GRB.MINIMIZE)
        model.optimize()

        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Subproblem not optimal (status={model.Status})")

        alpha = np.array([assign_constr[j].Pi for j in range(n)], dtype=float)
        beta = np.zeros((m, n), dtype=float)
        for i in range(m):
            for j in range(n):
                beta[i, j] = link_constr[i, j].Pi
        beta_sum = np.minimum(beta.sum(axis=1), 0.0)
        x_val = np.asarray(x.X, dtype=float)
        return float(model.ObjVal), alpha, beta_sum, x_val

    def solve_subproblems(
        self,
        inst: KMedianInstance,
        master_solution: MasterSolution[KMedianMasterState],
    ) -> SubproblemResult[None, KMedianOptCut]:
        m = inst.m
        y_vals = np.array([master_solution.vars[f"y_{i}"] for i in range(m)], dtype=float)
        sp_obj, alpha, beta_sum, x_val = self._solve_subproblem(inst, y_vals)

        alpha_sum = float(alpha.sum())
        rhs_val = alpha_sum + float(beta_sum @ y_vals)
        theta_val = float(master_solution.vars.get("z", 0.0) or 0.0)
        violation = rhs_val - theta_val

        metadata = {
            "assignment_cost": sp_obj,
            "theta": theta_val,
            "x_assignment": x_val.tolist(),
            "y_assignment": y_vals.tolist(),
        }

        cuts: list[KMedianOptCut] = []
        if violation > self.violation_tolerance:
            cuts.append(KMedianOptCut(alpha_sum=alpha_sum, beta_sum=beta_sum))

        return SubproblemResult(
            feasibility_cuts=[],
            optimality_cuts=cuts,
            all_feasible=True,
            worst_violation=max(0.0, violation),
            sub_bound=sp_obj,
            metadata=metadata,
        )

    def compute_iteration_bounds(
        self,
        inst: KMedianInstance,
        master_solution: MasterSolution[KMedianMasterState],
        sub_result: SubproblemResult[None, KMedianOptCut],
    ) -> tuple[float, float]:
        # Master objective is LB (z); UB from subproblem cost.
        lb_candidate = master_solution.objective
        ub_candidate = sub_result.sub_bound if sub_result.sub_bound is not None else lb_candidate

        if math.isinf(self.global_lb):
            self.global_lb = lb_candidate
        else:
            self.global_lb = max(self.global_lb, lb_candidate)

        if math.isinf(self.global_ub):
            self.global_ub = ub_candidate
        else:
            self.global_ub = min(self.global_ub, ub_candidate)

        return self.global_lb, self.global_ub

    def add_feasibility_cuts(
        self,
        inst: KMedianInstance,
        master: KMedianMasterState,
        cuts: Sequence[None],
    ) -> None:
        return

    def add_optimality_cuts(
        self,
        inst: KMedianInstance,
        master: KMedianMasterState,
        cuts: Sequence[KMedianOptCut],
    ) -> None:
        for idx, cut in enumerate(cuts):
            expr = cut.alpha_sum + gp.quicksum(cut.beta_sum[i] * master.y[i] for i in range(inst.m))
            master.model.addConstr(master.z >= expr, name=f"opt_cut_{idx}")
