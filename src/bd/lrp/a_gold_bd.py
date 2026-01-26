from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from benchmark.benders_decomp.base_bd_class import BaseBenders, MasterSolution, SubproblemResult
from lrp.a_gen_data import LRPInstance


@dataclass
class LRPOptCut:
    const_term: float
    coeffs: np.ndarray  # shape (m,)


@dataclass
class LRPFeasCut:
    const_term: float
    coeffs: np.ndarray  # shape (m,)


@dataclass
class LRPMasterState:
    model: gp.Model
    y: gp.MVar
    z: gp.Var


class LRPBenders(BaseBenders[LRPInstance, LRPMasterState, LRPFeasCut, LRPOptCut]):
    """Benders decomposition for the route-enumerated LRP."""

    def __init__(
        self,
        *,
        master_params: dict[str, Any] | None = None,
        subproblem_params: dict[str, Any] | None = None,
        integer_params: dict[str, Any] | None = None,
        max_iters: int = 250,
        verbose: bool = False,
    ) -> None:
        super().__init__(max_iters=max_iters, verbose=verbose)
        self.master_params = master_params or {}
        self.subproblem_params = subproblem_params or {}
        self.integer_params = integer_params or {}

    def load_instance(self, instance_name: str) -> LRPInstance:
        return LRPInstance.from_json_file(instance_name)

    def build_initial_master(self, inst: LRPInstance) -> LRPMasterState:
        model = gp.Model()
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.master_params.items():
            model.setParam(k, v)

        y = model.addMVar(inst.m, vtype=GRB.BINARY, name="y")
        z = model.addVar(lb=0.0, name="z")

        model.addConstr(y.sum() >= 1, name="open_one")
        model.setObjective(inst.depot_cost @ y + z, GRB.MINIMIZE)
        return LRPMasterState(model=model, y=y, z=z)

    def solve_master(self, inst: LRPInstance, master: LRPMasterState) -> MasterSolution[LRPMasterState]:
        master.model.optimize()
        if master.model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(f"Master solve failed (status={master.model.Status})")

        y_val = np.asarray(master.y.X, dtype=float)
        z_val = float(master.z.X)
        vars_dict: dict[str, Any] = {f"y_{j}": float(y_val[j]) for j in range(inst.m)}
        vars_dict["z"] = z_val

        return MasterSolution(
            objective=float(master.model.ObjVal),
            vars=vars_dict,
            master_state=master,
        )

    def _solve_subproblem(
        self,
        inst: LRPInstance,
        y_vals: np.ndarray,
    ) -> tuple[float | None, np.ndarray, np.ndarray, np.ndarray | None, bool]:
        m, n, R = inst.m, inst.n, inst.R
        y_vals = y_vals.reshape(m)
        model = gp.Model()
        model.Params.OutputFlag = 0
        model.Params.InfUnbdInfo = 1
        for k, v in self.subproblem_params.items():
            model.setParam(k, v)

        lam = model.addMVar(R, lb=0.0, name="lambda")

        cover_constr = model.addConstrs(
            (gp.quicksum(inst.incidence[i, r] * lam[r] for r in range(R)) == 1 for i in range(n)),
            name="cover",
        )

        fleet_constr = {}
        for j in range(m):
            routes_j = [r for r in range(R) if int(inst.route_depot[r]) == j]
            if routes_j:
                fleet_constr[j] = model.addConstr(
                    gp.quicksum(lam[r] for r in routes_j) <= inst.max_vehicles[j] * y_vals[j],
                    name=f"fleet_{j}",
                )

        model.setObjective(inst.route_cost @ lam, GRB.MINIMIZE)
        model.optimize()

        if model.Status == GRB.INF_OR_UNBD:
            model.setParam(GRB.Param.DualReductions, 0)
            model.optimize()

        if model.Status == GRB.INFEASIBLE:
            alpha = -np.array([cover_constr[i].FarkasDual for i in range(n)], dtype=float)
            beta = np.zeros(m)
            for j, constr in fleet_constr.items():
                beta[j] = -constr.FarkasDual
            return None, alpha, beta, None, False

        if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(f"Subproblem not solved (status={model.Status})")

        alpha = np.array([cover_constr[i].Pi for i in range(n)], dtype=float)
        beta = np.zeros(m)
        for j, constr in fleet_constr.items():
            beta[j] = constr.Pi
        lam_val = np.asarray(lam.X, dtype=float)
        return float(model.ObjVal), alpha, beta, lam_val, True

    def _solve_integer_recourse(
        self, inst: LRPInstance, y_vals: np.ndarray
    ) -> tuple[float | None, np.ndarray | None, int]:
        m, n, R = inst.m, inst.n, inst.R
        y_vals = y_vals.reshape(m)
        model = gp.Model()
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.integer_params.items():
            model.setParam(k, v)

        lam = model.addMVar(R, vtype=GRB.BINARY, name="lambda")

        model.addConstrs(
            (gp.quicksum(inst.incidence[i, r] * lam[r] for r in range(R)) == 1 for i in range(n)),
            name="cover",
        )

        for j in range(m):
            routes_j = [r for r in range(R) if int(inst.route_depot[r]) == j]
            if routes_j:
                model.addConstr(gp.quicksum(lam[r] for r in routes_j) <= inst.max_vehicles[j] * y_vals[j])

        model.setObjective(inst.route_cost @ lam, GRB.MINIMIZE)
        model.optimize()

        status = model.Status
        if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or model.SolCount == 0:
            return None, None, status

        lam_val = np.asarray(lam.X, dtype=float)
        return float(model.ObjVal), lam_val, status

    def solve_subproblems(
        self,
        inst: LRPInstance,
        master_solution: MasterSolution[LRPMasterState],
    ) -> SubproblemResult[LRPFeasCut, LRPOptCut]:
        y_vals = np.array([master_solution.vars[f"y_{j}"] for j in range(inst.m)], dtype=float)
        sp_obj, alpha, beta, lam_val, feasible = self._solve_subproblem(inst, y_vals)

        feasibility_cuts: list[LRPFeasCut] = []
        optimality_cuts: list[LRPOptCut] = []
        all_feasible = True
        worst_violation = 0.0
        metadata: dict[str, Any] = {"y_assignment": y_vals.tolist()}

        if not feasible:
            all_feasible = False
            const_term = float(alpha.sum())
            coeffs = beta * inst.max_vehicles
            dual_expr = const_term + float(coeffs @ y_vals)
            if dual_expr <= self.violation_tolerance:
                raise RuntimeError(
                    f"Feasibility cut not violated at current y (expr={dual_expr:.3e}); check dual signs."
                )
            feasibility_cuts.append(LRPFeasCut(const_term=const_term, coeffs=coeffs))
            metadata.update({"lp_route_cost": None, "theta": float(master_solution.vars.get("z", 0.0) or 0.0)})
            sub_bound = None
        else:
            const_term = float(alpha.sum())
            coeffs = beta * inst.max_vehicles
            dual_value = const_term + float(coeffs @ y_vals)
            if not math.isclose(dual_value, float(sp_obj), rel_tol=1e-6, abs_tol=1e-6):
                diff = dual_value - float(sp_obj)
                raise RuntimeError(f"Dual mismatch in routing LP: diff={diff:.3e}")
            rhs_val = const_term + float(coeffs @ y_vals)
            theta_val = float(master_solution.vars.get("z", 0.0) or 0.0)
            violation = rhs_val - theta_val
            worst_violation = max(0.0, violation)

            metadata.update(
                {
                    "lp_route_cost": sp_obj,
                    "theta": theta_val,
                    "lambda_assignment_lp": lam_val.tolist() if lam_val is not None else None,
                }
            )

            sub_bound = sp_obj
            if violation > self.violation_tolerance:
                optimality_cuts.append(LRPOptCut(const_term=const_term, coeffs=coeffs))

            int_obj, int_lam, int_status = self._solve_integer_recourse(inst, y_vals)
            metadata.update(
                {
                    "int_route_cost": int_obj,
                    "lambda_assignment_int": int_lam.tolist() if int_lam is not None else None,
                    "int_status": int_status,
                }
            )
            lam_for_result: list[float] | None = None
            if int_lam is not None:
                lam_for_result = int_lam.tolist()
            elif lam_val is not None:
                lam_for_result = lam_val.tolist()
            if lam_for_result is not None:
                metadata["lambda_assignment"] = lam_for_result

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
        inst: LRPInstance,
        master_solution: MasterSolution[LRPMasterState],
        sub_result: SubproblemResult[LRPFeasCut, LRPOptCut],
    ) -> tuple[float, float]:
        if not sub_result.all_feasible or sub_result.sub_bound is None:
            return self.global_lb, self.global_ub

        lb_candidate = master_solution.objective
        depot_term = float(sum(inst.depot_cost[j] * master_solution.vars[f"y_{j}"] for j in range(inst.m)))
        int_cost = sub_result.metadata.get("int_route_cost") if sub_result.metadata else None
        ub_candidate = depot_term + int_cost if int_cost is not None else math.inf

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
        inst: LRPInstance,
        master: LRPMasterState,
        cuts: Sequence[LRPFeasCut],
    ) -> None:
        for idx, cut in enumerate(cuts):
            expr = cut.const_term + gp.quicksum(cut.coeffs[j] * master.y[j] for j in range(inst.m))
            master.model.addConstr(expr <= 0.0, name=f"feas_cut_{idx}")

    def add_optimality_cuts(
        self,
        inst: LRPInstance,
        master: LRPMasterState,
        cuts: Sequence[LRPOptCut],
    ) -> None:
        for idx, cut in enumerate(cuts):
            expr = cut.const_term + gp.quicksum(cut.coeffs[j] * master.y[j] for j in range(inst.m))
            master.model.addConstr(master.z >= expr, name=f"opt_cut_{idx}")
