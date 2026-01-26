from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from benchmark.benders_decomp.base_bd_class import BaseBenders, MasterSolution, SubproblemResult
from fctp.a_gen_data import FCTPInstance


@dataclass
class FCTPFeasCut:
    const_term: float
    coeffs: np.ndarray  # shape (m, n)


@dataclass
class FCTPOptCut:
    const_term: float
    coeffs: np.ndarray  # same shape as y


@dataclass
class FCTPMasterState:
    model: gp.Model
    y: gp.MVar
    z: gp.Var


class FCTPBenders(BaseBenders[FCTPInstance, FCTPMasterState, FCTPFeasCut, FCTPOptCut]):
    """Benders decomposition for the fixed-charge transportation problem."""

    def __init__(
        self,
        *,
        master_params: dict[str, Any] | None = None,
        subproblem_params: dict[str, Any] | None = None,
        max_iters: int = 500,
        verbose: bool = False,
    ) -> None:
        super().__init__(max_iters=max_iters, verbose=verbose)
        self.master_params = master_params or {}
        self.subproblem_params = subproblem_params or {}

    def load_instance(self, instance_name: str) -> FCTPInstance:
        return FCTPInstance.from_json_file(instance_name)

    def build_initial_master(self, inst: FCTPInstance) -> FCTPMasterState:
        m, n = inst.m, inst.n
        model = gp.Model()
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.master_params.items():
            model.setParam(k, v)

        y = model.addMVar((m, n), vtype=GRB.BINARY, name="y")
        z = model.addVar(lb=0.0, name="z")

        # Each demand node must have at least one open arc (weak feasibility guard).
        model.addConstrs((y[:, j].sum() >= 1 for j in range(n)), name="demand_covered")
        model.setObjective((inst.fixed_cost * y).sum() + z, GRB.MINIMIZE)
        return FCTPMasterState(model=model, y=y, z=z)

    def solve_master(self, inst: FCTPInstance, master: FCTPMasterState) -> MasterSolution[FCTPMasterState]:
        master.model.optimize()
        if master.model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(f"Master solve failed (status={master.model.Status})")

        y_val = np.asarray(master.y.X, dtype=float)
        z_val = float(master.z.X)

        vars_dict: dict[str, Any] = {}
        for i in range(inst.m):
            for j in range(inst.n):
                vars_dict[f"y_{i}_{j}"] = float(y_val[i, j])
        vars_dict["z"] = z_val

        return MasterSolution(
            objective=float(master.model.ObjVal),
            vars=vars_dict,
            master_state=master,
        )

    def _solve_subproblem(
        self, inst: FCTPInstance, y_vals: np.ndarray
    ) -> tuple[float | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, bool]:
        m, n = inst.m, inst.n
        y_vals = y_vals.reshape(m, n)
        model = gp.Model()
        model.Params.OutputFlag = 0
        model.Params.InfUnbdInfo = 1
        for k, v in self.subproblem_params.items():
            model.setParam(k, v)

        x = model.addMVar((m, n), lb=0.0, name="x")

        supply_constr = model.addConstrs((x[i, :].sum() <= inst.supply[i] for i in range(m)), name="supply")
        demand_constr = model.addConstrs((x[:, j].sum() >= inst.demand[j] for j in range(n)), name="demand")
        link_constr = model.addConstrs(
            (x[i, j] <= inst.capacity[i, j] * float(y_vals[i, j]) for i in range(m) for j in range(n)), name="link"
        )

        model.setObjective((inst.unit_cost * x).sum(), GRB.MINIMIZE)
        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            pi = -np.array([supply_constr[i].FarkasDual for i in range(m)], dtype=float)
            sigma = -np.array([demand_constr[j].FarkasDual for j in range(n)], dtype=float)
            mu = np.zeros((m, n), dtype=float)
            for i in range(m):
                for j in range(n):
                    mu[i, j] = -link_constr[i, j].FarkasDual
            return None, pi, sigma, mu, None, False

        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Subproblem not optimal (status={model.Status})")

        pi = np.array([supply_constr[i].Pi for i in range(m)], dtype=float)
        sigma = np.array([demand_constr[j].Pi for j in range(n)], dtype=float)
        mu = np.zeros((m, n), dtype=float)
        for i in range(m):
            for j in range(n):
                mu[i, j] = link_constr[i, j].Pi

        x_val = np.asarray(x.X, dtype=float)
        return float(model.ObjVal), pi, sigma, mu, x_val, True

    def solve_subproblems(
        self,
        inst: FCTPInstance,
        master_solution: MasterSolution[FCTPMasterState],
    ) -> SubproblemResult[FCTPFeasCut, FCTPOptCut]:
        m, n = inst.m, inst.n
        y_vals = np.array(
            [master_solution.vars[f"y_{i}_{j}"] for i in range(m) for j in range(n)], dtype=float
        ).reshape(m, n)

        sp_obj, pi, sigma, mu, x_val, feasible = self._solve_subproblem(inst, y_vals)

        feasibility_cuts: list[FCTPFeasCut] = []
        optimality_cuts: list[FCTPOptCut] = []
        all_feasible = True
        worst_violation = 0.0
        metadata: dict[str, Any] = {"y_assignment": y_vals.tolist()}

        if not feasible:
            all_feasible = False
            const_term = float(inst.supply @ pi + inst.demand @ sigma)
            coeff = inst.capacity * mu
            dual_expr = const_term + float((coeff * y_vals).sum())
            # Farkas feasibility cuts must be violated at the current y (expr > 0).
            if dual_expr <= 1e-6:
                raise RuntimeError(
                    f"Feasibility cut not violated at current y (expr={dual_expr:.3e}); check dual signs."
                )
            feasibility_cuts.append(FCTPFeasCut(const_term=const_term, coeffs=coeff))
            metadata.update({"transport_cost": None, "theta": float(master_solution.vars.get("z", 0.0) or 0.0)})
            sub_bound = None
        else:
            const_term = float(inst.supply @ pi + inst.demand @ sigma)
            coeff = inst.capacity * mu  # note mu <= 0, capacity >=0
            dual_value = const_term + float((coeff * y_vals).sum())
            if not math.isclose(dual_value, float(sp_obj), rel_tol=1e-6, abs_tol=1e-6):
                diff = dual_value - float(sp_obj)
                raise RuntimeError(f"Dual mismatch in subproblem: diff={diff:.3e}")
            cut = FCTPOptCut(const_term=const_term, coeffs=coeff)

            theta_estimate = float(master_solution.vars.get("z", 0.0) or 0.0)
            violation = (const_term + float((coeff * y_vals).sum())) - theta_estimate
            worst_violation = max(0.0, violation)

            metadata.update(
                {
                    "transport_cost": sp_obj,
                    "theta": theta_estimate,
                    "x_assignment": x_val.tolist() if x_val is not None else None,
                }
            )
            sub_bound = sp_obj

            if violation > self.violation_tolerance:
                optimality_cuts.append(cut)

        return SubproblemResult(
            feasibility_cuts=feasibility_cuts,
            optimality_cuts=optimality_cuts,
            all_feasible=all_feasible,
            worst_violation=worst_violation,
            sub_bound=sub_bound,
            metadata=metadata,
        )

    def compute_iteration_bounds(
        self,
        inst: FCTPInstance,
        master_solution: MasterSolution[FCTPMasterState],
        sub_result: SubproblemResult[FCTPFeasCut, FCTPOptCut],
    ) -> tuple[float, float]:
        fixed_term = float(
            sum(
                inst.fixed_cost[i, j] * master_solution.vars[f"y_{i}_{j}"] for i in range(inst.m) for j in range(inst.n)
            )
        )
        sp_cost = (
            sub_result.metadata.get("transport_cost", sub_result.sub_bound)
            if sub_result.metadata
            else sub_result.sub_bound
        )
        # If the subproblem is infeasible, the current master solution is not feasible
        # for the original problem, so we must not update bounds.
        if sp_cost is None:
            return self.global_lb, self.global_ub

        ub_candidate = fixed_term + sp_cost
        lb_candidate = master_solution.objective

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
        inst: FCTPInstance,
        master: FCTPMasterState,
        cuts: Sequence[FCTPFeasCut],
    ) -> None:
        for idx, cut in enumerate(cuts):
            expr = cut.const_term + gp.quicksum(
                cut.coeffs[i, j] * master.y[i, j] for i in range(inst.m) for j in range(inst.n)
            )
            master.model.addConstr(expr <= 0.0, name=f"feas_cut_{idx}")

    def add_optimality_cuts(
        self,
        inst: FCTPInstance,
        master: FCTPMasterState,
        cuts: Sequence[FCTPOptCut],
    ) -> None:
        for idx, cut in enumerate(cuts):
            expr = cut.const_term + gp.quicksum(
                cut.coeffs[i, j] * master.y[i, j] for i in range(inst.m) for j in range(inst.n)
            )
            master.model.addConstr(master.z >= expr, name=f"opt_cut_{idx}")
