"""Benders reference for k-median (p-median), multi-cut with closed-form subproblem duals.

Model: min sum_j theta_j s.t. sum_i y_i = k, y binary, theta_j >= recourse_j(y), where for a customer j
    recourse_j(y) = min { sum_i D_ij x_ij : sum_i x_ij = 1, 0 <= x_ij <= y_i }.
For any y in [0, 1]^m with sum_i y_i >= 1, sort the sites by D_ij (ties by index) and let k_j be the first site at
which the cumulative y reaches 1 (the critical site). Then u_j = D_{k_j j}, w_ij = max(0, D_{k_j j} - D_ij) is an
optimal dual solution of the recourse LP, which gives the Benders optimality cut
    theta_j >= D_{k_j j} - sum_i max(0, D_{k_j j} - D_ij) y_i,
valid for every y and tight at the current y (the p-median Benders cuts of Duran-Mateluna, Ales and Elloumi 2023;
Cornuejols, Nemhauser and Wolsey 1990 for UFL). The reference first runs the cut loop on the LP
relaxation of the master (Phase 1, as in Duran-Mateluna et al.), then on the binary master until the bounds meet.
Replaces the paper's single aggregated cut with Gurobi recourse LPs (2026-11-05), which does not converge in 60 s on
the TSPLIB-derived instances. The model, the output contract and the verifier are unchanged.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.km.a_gen_data import KMedianInstance
from src.utils.base_bd_class import BaseBenders, MasterSolution, SubproblemResult
from src.utils.reference_runtime import ReferenceLimit


@dataclass
class KMedianOptCut:
    customer: int
    constant: float  # D_{k_j j}
    coeff: np.ndarray  # (m,) max(0, D_{k_j j} - D_ij); cut: theta_j >= constant - coeff @ y


@dataclass
class KMedianMasterState:
    model: gp.Model
    y: gp.MVar
    theta: gp.MVar


def closed_form_cuts(D: np.ndarray, order: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per customer j: recourse value, critical constant D_{k_j j} and the cut coefficients (m x n); requires sum(y) >= 1."""
    m, n = D.shape
    values = np.empty(n)
    constants = np.empty(n)
    coeffs = np.empty((m, n))
    for j in range(n):
        idx = order[:, j]
        cum = np.cumsum(y[idx])
        # First position whose cumulative y reaches 1 (a 1e-12 slack absorbs round-off in sums of binaries).
        pos = int(np.searchsorted(cum, 1.0 - 1e-12))
        pos = min(pos, m - 1)
        k = idx[pos]
        c = D[k, j]
        constants[j] = c
        coeffs[:, j] = np.maximum(0.0, c - D[:, j])
        values[j] = c - float(coeffs[:, j] @ y)
    return values, constants, coeffs


class KMedianBenders(BaseBenders[KMedianInstance, KMedianMasterState, None, KMedianOptCut]):
    """Benders decomposition for k-median (Gurobi master, closed-form recourse)."""

    def __init__(
        self,
        *,
        master_params: dict[str, Any] | None = None,
        subproblem_params: dict[str, Any] | None = None,
        max_iters: int = 500,
        verbose: bool = False,
        time_limit: float = 60.0,
        lp_phase_iters: int = 200,
        lp_phase_share: float = 0.5,
    ) -> None:
        super().__init__(max_iters=max_iters, verbose=verbose, time_limit=time_limit)
        self.master_params = master_params or {}
        self.subproblem_params = subproblem_params or {}  # kept for interface compatibility; recourse is closed form
        self.lp_phase_iters = lp_phase_iters
        self.lp_phase_share = lp_phase_share  # the LP phase stops once it would leave less than this share of the budget

    def load_instance(self, instance_name: str) -> KMedianInstance:
        return KMedianInstance.from_json_file(instance_name)

    def _add_cuts(self, master: KMedianMasterState, constants: np.ndarray, coeffs: np.ndarray, customers: Sequence[int]) -> None:
        for j in customers:
            nz = np.flatnonzero(coeffs[:, j])
            master.model.addConstr(master.theta[j] + gp.quicksum(float(coeffs[i, j]) * master.y[int(i)] for i in nz) >= float(constants[j]),
                                   name=f"opt_{j}_{master.model.NumConstrs}")

    def build_initial_master(self, inst: KMedianInstance) -> KMedianMasterState:
        self._order = np.argsort(inst.D, axis=0, kind="stable")
        model = gp.Model()
        model.Params.OutputFlag = 1 if self.verbose else 0
        model.Params.MIPGap = 0.0  # integer data: prove the optimum exactly (user master_params may override)
        for k, v in self.master_params.items():
            model.setParam(k, v)
        y = model.addMVar(inst.m, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")
        # theta_j >= min_i D_ij is implied by every cut; it keeps the first master bounded.
        theta = model.addMVar(inst.n, lb=inst.D.min(axis=0), name="theta")
        model.addConstr(y.sum() == inst.k, name="open_k")
        model.setObjective(theta.sum(), GRB.MINIMIZE)
        master = KMedianMasterState(model=model, y=y, theta=theta)

        # Phase 1: cut loop on the LP relaxation of the master (cuts stay valid for the binary master).
        self.lp_phase = {"iterations": 0, "bound": None, "stopped": "converged"}
        # The phase reserve is held by the runtime, so no LP solve or cut computation can eat into the binary master's share.
        self.runtime.stage_reserve = self.lp_phase_share * (self.runtime.time_limit - self.runtime.output_reserve)
        try:
            for _ in range(self.lp_phase_iters):
                try:
                    self.runtime.optimize(model, role="master_lp", allow_incumbent=False)
                except ReferenceLimit:
                    self.lp_phase["stopped"] = "time_share"
                    break
                if model.Status != GRB.OPTIMAL:
                    raise RuntimeError(f"LP master failed (status={model.Status})")
                yv, tv = np.asarray(y.X, dtype=float), np.asarray(theta.X, dtype=float)
                values, constants, coeffs = closed_form_cuts(inst.D, self._order, yv)
                self.lp_phase.update(iterations=self.lp_phase["iterations"] + 1, bound=float(model.ObjVal))
                violated = np.flatnonzero(values - tv > 1e-6 * np.maximum(1.0, np.abs(values)))
                if violated.size == 0:
                    break
                self._add_cuts(master, constants, coeffs, violated.tolist())
        finally:
            self.runtime.stage_reserve = 0.0
            model.Params.TimeLimit = float(self.master_params.get("TimeLimit", GRB.INFINITY))  # drop the phase-capped limit
        y.VType = GRB.BINARY
        model.update()
        return master

    def solve_master(self, inst: KMedianInstance, master: KMedianMasterState) -> MasterSolution[KMedianMasterState]:
        self.runtime.optimize(master.model, role="master", allow_incumbent=False)
        if master.model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Master solve failed (status={master.model.Status})")
        y_val = np.round(np.asarray(master.y.X, dtype=float))
        theta_val = np.asarray(master.theta.X, dtype=float)
        vars_dict: dict[str, Any] = {f"y_{i}": float(y_val[i]) for i in range(inst.m)}
        vars_dict.update({f"theta_{j}": float(theta_val[j]) for j in range(inst.n)})
        return MasterSolution(objective=float(master.model.ObjVal), vars=vars_dict, master_state=master, bound=float(master.model.ObjBound))

    def solve_subproblems(
        self,
        inst: KMedianInstance,
        master_solution: MasterSolution[KMedianMasterState],
    ) -> SubproblemResult[None, KMedianOptCut]:
        m, n = inst.m, inst.n
        y_vals = np.array([master_solution.vars[f"y_{i}"] for i in range(m)], dtype=float)
        theta = np.array([master_solution.vars[f"theta_{j}"] for j in range(n)], dtype=float)
        values, constants, coeffs = closed_form_cuts(inst.D, self._order, y_vals)
        open_sites = np.flatnonzero(y_vals > 0.5)
        best = open_sites[np.argmin(inst.D[open_sites, :], axis=0)]
        x_val = np.zeros((m, n))
        x_val[best, np.arange(n)] = 1.0
        cost = float(inst.D[best, np.arange(n)].sum())
        cuts = [KMedianOptCut(customer=j, constant=float(constants[j]), coeff=coeffs[:, j].copy())
                for j in range(n) if values[j] - theta[j] > 1e-6 * max(1.0, abs(values[j]))]
        metadata = {"assignment_cost": cost, "theta_sum": float(theta.sum()), "x_assignment": x_val.tolist(),
                    "y_assignment": y_vals.tolist(), "lp_phase": dict(self.lp_phase)}
        return SubproblemResult(feasibility_cuts=[], optimality_cuts=cuts, all_feasible=True,
                                worst_violation=float(max(0.0, (values - theta).max())), sub_bound=cost, metadata=metadata)

    def compute_iteration_bounds(
        self,
        inst: KMedianInstance,
        master_solution: MasterSolution[KMedianMasterState],
        sub_result: SubproblemResult[None, KMedianOptCut],
    ) -> tuple[float, float]:
        lb = master_solution.bound if master_solution.bound is not None else master_solution.objective
        self.global_lb = lb if math.isinf(self.global_lb) else max(self.global_lb, lb)
        self.global_ub = sub_result.sub_bound if math.isinf(self.global_ub) else min(self.global_ub, sub_result.sub_bound)
        return self.global_lb, self.global_ub

    def add_feasibility_cuts(self, inst: KMedianInstance, master: KMedianMasterState, cuts: Sequence[None]) -> None:
        return

    def add_optimality_cuts(self, inst: KMedianInstance, master: KMedianMasterState, cuts: Sequence[KMedianOptCut]) -> None:
        for cut in cuts:
            nz = np.flatnonzero(cut.coeff)
            master.model.addConstr(master.theta[cut.customer] + gp.quicksum(float(cut.coeff[i]) * master.y[int(i)] for i in nz) >= cut.constant,
                                   name=f"opt_{cut.customer}_{master.model.NumConstrs}")
