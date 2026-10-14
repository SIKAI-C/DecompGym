from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.tsso.a_gen_data import TSSOInstance
from src.utils.base_bd_class import BaseBenders, MasterSolution, SubproblemResult


@dataclass
class TSSOOptCut:
    alpha: float
    grad: np.ndarray  # shape (I, T)


@dataclass
class TSSOMasterState:
    model: gp.Model
    y: gp.MVar
    theta: gp.Var


class TSSOBenders(BaseBenders[TSSOInstance, TSSOMasterState, None, TSSOOptCut]):
    """Benders decomposition for two-stage stochastic production."""

    def __init__(
        self,
        *,
        master_params: dict[str, Any] | None = None,
        subproblem_params: dict[str, Any] | None = None,
        max_iters: int = 200,
        verbose: bool = False,
        time_limit: float = 60.0,
        target_gap: float = 0.005,
    ) -> None:
        super().__init__(max_iters=max_iters, verbose=verbose, time_limit=time_limit, target_gap=target_gap)
        self.master_params = master_params or {}
        self.subproblem_params = subproblem_params or {}

    def load_instance(self, instance_name: str) -> TSSOInstance:
        return TSSOInstance.from_json_file(instance_name)

    def build_initial_master(self, inst: TSSOInstance) -> TSSOMasterState:
        I, T = inst.n_products, inst.n_periods
        model = gp.Model()
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.master_params.items():
            model.setParam(k, v)

        y = model.addMVar((I, T), vtype=GRB.BINARY, name="y")
        theta = model.addVar(lb=0.0, name="theta")

        model.setObjective((inst.F * y).sum() + theta, GRB.MINIMIZE)
        return TSSOMasterState(model=model, y=y, theta=theta)

    def solve_master(self, inst: TSSOInstance, master: TSSOMasterState) -> MasterSolution[TSSOMasterState]:
        self.runtime.optimize(master.model, role="master", allow_incumbent=False)
        if master.model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Master solve failed (status={master.model.Status})")

        y_val = np.asarray(master.y.X, dtype=float)
        theta_val = float(master.theta.X)

        vars_dict: dict[str, Any] = {"theta": theta_val}
        for i in range(inst.n_products):
            for t in range(inst.n_periods):
                vars_dict[f"y_{i}_{t}"] = float(y_val[i, t])

        return MasterSolution(
            objective=float(master.model.ObjVal),
            vars=vars_dict,
            master_state=master,
            bound=float(master.model.ObjBound),
        )


    def _solve_subproblem(
        self,
        inst: TSSOInstance,
        y_vals: np.ndarray,
        scenario: int,
    ) -> tuple[float, float, np.ndarray, dict[str, np.ndarray]]:
        I, T = inst.n_products, inst.n_periods
        y_vals = y_vals.reshape(I, T)

        model = gp.Model()
        model.Params.OutputFlag = 0
        model.Params.InfUnbdInfo = 1
        for k, v in self.subproblem_params.items():
            model.setParam(k, v)

        x_reg = model.addMVar((I, T), lb=0.0, name="x_reg")
        x_ot = model.addMVar((I, T), lb=0.0, name="x_ot")
        inv = model.addMVar((I, T), lb=0.0, name="inv")
        back = model.addMVar((I, T), lb=0.0, name="back")

        cap_constr: dict[tuple[int, int], gp.Constr] = {}
        for i in range(I):
            for t in range(T):
                inv_prev = inst.I0[i] if t == 0 else inv[i, t - 1]
                back_prev = inst.B0[i] if t == 0 else back[i, t - 1]
                rhs = inst.demand[i, t, scenario] - inv_prev + back_prev
                model.addConstr(x_reg[i, t] + x_ot[i, t] - inv[i, t] + back[i, t] == rhs, name=f"bal_{i}_{t}")
                cap_constr[(i, t)] = model.addConstr(
                    x_reg[i, t] <= inst.K[i, t] * y_vals[i, t],
                    name=f"cap_{i}_{t}",
                )

        model.setObjective(
            (inst.c_reg * x_reg).sum() + (inst.c_ot * x_ot).sum() + (inst.h * inv).sum() + (inst.b * back).sum(),
            sense=GRB.MINIMIZE,
        )
        self.runtime.optimize(model, role="recourse_lp", allow_incumbent=False)
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Subproblem not optimal (status={model.Status})")

        mu = np.zeros((I, T))
        for (i, t), constr in cap_constr.items():
            mu[i, t] = float(constr.Pi)

        grad = mu * inst.K
        alpha = float(model.ObjVal) - float((grad * y_vals).sum())
        dual_value = alpha + float((grad * y_vals).sum())
        if not math.isclose(dual_value, float(model.ObjVal), rel_tol=1e-6, abs_tol=1e-6):
            raise RuntimeError(f"Dual mismatch in scenario {scenario}: obj={model.ObjVal:.6f}, dual={dual_value:.6f}")
        if self.verbose:
            print(f"[DEBUG] scenario {scenario}: obj={model.ObjVal:.6f}, dual_value={dual_value:.6f}")
        primals = {
            "x_reg": np.asarray(x_reg.X, dtype=float),
            "x_ot": np.asarray(x_ot.X, dtype=float),
            "inv": np.asarray(inv.X, dtype=float),
            "back": np.asarray(back.X, dtype=float),
        }
        return float(model.ObjVal), alpha, grad, primals


    def solve_subproblems(
        self,
        inst: TSSOInstance,
        master_solution: MasterSolution[TSSOMasterState],
    ) -> SubproblemResult[None, TSSOOptCut]:
        I, T, S = inst.n_products, inst.n_periods, inst.n_scenarios
        y_vals = np.array(
            [master_solution.vars[f"y_{i}_{t}"] for i in range(I) for t in range(T)], dtype=float
        ).reshape(I, T)

        expected_Q = 0.0
        alpha_sum = 0.0
        grad_sum = np.zeros((I, T))
        recourse_snapshots: dict[int, dict[str, np.ndarray]] = {}

        for s in range(S):
            obj_s, alpha_s, grad_s, primals_s = self._solve_subproblem(inst, y_vals, s)
            weight = float(inst.prob[s])
            expected_Q += weight * obj_s
            alpha_sum += weight * alpha_s
            grad_sum += weight * grad_s
            recourse_snapshots[s] = primals_s

        theta_val = float(master_solution.vars.get("theta", 0.0) or 0.0)
        violation = expected_Q - theta_val

        metadata = {
            "expected_Q": expected_Q,
            "theta": theta_val,
            "recourse": {
                int(k): {name: v.tolist() for name, v in snap.items()} for k, snap in recourse_snapshots.items()
            },
            "y_assignment": y_vals.tolist(),
        }

        cuts: list[TSSOOptCut] = []
        if violation > self.violation_tolerance:
            cuts.append(TSSOOptCut(alpha=alpha_sum, grad=grad_sum))

        return SubproblemResult(
            feasibility_cuts=[],
            optimality_cuts=cuts,
            all_feasible=True,
            worst_violation=max(0.0, violation),
            sub_bound=expected_Q,
            metadata=metadata,
        )

    def compute_iteration_bounds(
        self,
        inst: TSSOInstance,
        master_solution: MasterSolution[TSSOMasterState],
        sub_result: SubproblemResult[None, TSSOOptCut],
    ) -> tuple[float, float]:
        # Master objective is LB; UB from first-stage cost + expected recourse.
        lb_candidate = master_solution.bound if master_solution.bound is not None else master_solution.objective
        y_vals = np.array(
            [master_solution.vars[f"y_{i}_{t}"] for i in range(inst.n_products) for t in range(inst.n_periods)],
            dtype=float,
        ).reshape(inst.n_products, inst.n_periods)
        fixed_cost = float((inst.F * y_vals).sum())
        expected_Q = sub_result.sub_bound if sub_result.sub_bound is not None else 0.0
        ub_candidate = fixed_cost + expected_Q

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
        inst: TSSOInstance,
        master: TSSOMasterState,
        cuts: Sequence[None],
    ) -> None:
        return

    def add_optimality_cuts(
        self,
        inst: TSSOInstance,
        master: TSSOMasterState,
        cuts: Sequence[TSSOOptCut],
    ) -> None:
        for idx, cut in enumerate(cuts):
            expr = cut.alpha + gp.quicksum(
                cut.grad[i, t] * master.y[i, t] for i in range(inst.n_products) for t in range(inst.n_periods)
            )
            master.model.addConstr(master.theta >= expr, name=f"opt_cut_{idx}")
