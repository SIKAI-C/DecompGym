"""Benders reference for the fixed-charge transportation problem (FCTP).

Master: min sum f_ij y_ij + z over binary y; recourse: the transportation LP with arc bounds x_ij <= m_ij y_ij
(m_ij = data["capacity"]); Farkas feasibility cuts and LP-dual optimality cuts on z.
2026-11-05 strengthening (the literature-recipe instances, n = m up to about 12, defeat the paper's iterative loop,
which re-solves the master MIP after every single cut and found no feasible design within 60 s):
  - the master gets the valid capacity-cover rows sum_i m_ij y_ij >= b_j and
    sum_j m_ij y_ij >= sum_k b_k - sum_{k != i} a_k (= a_i on balanced data; any feasible flow satisfies them because
    x_ij <= m_ij y_ij);
  - Phase 1 runs the same cut loop on the LP relaxation of the master (bounded share of the time budget);
  - the binary master is then solved once as branch-and-Benders-cut: at every integer candidate a lazy-constraint
    callback solves the transportation LP and adds the violated feasibility or optimality cut.
The final design is re-evaluated by `solve_subproblems`, so the published plan, its objective and the bounds are the
usual paired Benders result. Every nested solve (LP phase, callback recourse LPs) is capped by the shared runtime
budget; if the binary master runs out of time, its incumbent design is still re-evaluated exactly. Model, output contract and verifier are unchanged.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.fctp.a_gen_data import FCTPInstance
from src.utils.base_bd_class import BaseBenders, MasterSolution, SubproblemResult
from src.utils.reference_runtime import ReferenceLimit


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
        time_limit: float = 60.0,
        lp_phase_iters: int = 200,
        lp_phase_share: float = 0.5,
        final_eval_reserve: float = 2.0,
    ) -> None:
        super().__init__(max_iters=max_iters, verbose=verbose, time_limit=time_limit)
        self.master_params = master_params or {}
        self.subproblem_params = subproblem_params or {}
        self.lp_phase_iters = lp_phase_iters
        self.lp_phase_share = lp_phase_share  # Phase 1 stops once it would leave less than this share of the budget
        self.final_eval_reserve = final_eval_reserve  # seconds the binary master leaves for the final recourse LP

    def load_instance(self, instance_name: str) -> FCTPInstance:
        return FCTPInstance.from_json_file(instance_name)

    def build_initial_master(self, inst: FCTPInstance) -> FCTPMasterState:
        m, n = inst.m, inst.n
        model = gp.Model()
        model.Params.OutputFlag = 1 if self.verbose else 0
        model.Params.MIPGap = 0.0  # integer data: prove the optimum exactly (user master_params may override)
        for k, v in self.master_params.items():
            model.setParam(k, v)

        y = model.addMVar((m, n), vtype=GRB.BINARY, name="y")
        z = model.addVar(lb=0.0, name="z")

        # Each demand node must have at least one open arc (weak feasibility guard).
        model.addConstrs((y[:, j].sum() >= 1 for j in range(n)), name="demand_covered")
        # Capacity covers, valid for any data because x_ij <= m_ij y_ij: column j must carry b_j, and row i must carry
        # what the other rows cannot (total demand minus their supply; this equals a_i on the balanced instances).
        model.addConstrs(((inst.capacity[:, j] * y[:, j]).sum() >= inst.demand[j] for j in range(n)), name="cover_demand")
        total_demand, total_supply = float(inst.demand.sum()), float(inst.supply.sum())
        for i in range(m):
            rhs = total_demand - (total_supply - float(inst.supply[i]))
            if rhs > 0:
                model.addConstr((inst.capacity[i, :] * y[i, :]).sum() >= rhs, name=f"cover_supply[{i}]")
        model.setObjective((inst.fixed_cost * y).sum() + z, GRB.MINIMIZE)
        master = FCTPMasterState(model=model, y=y, z=z)

        # Phase 1: Benders cut loop on the LP relaxation (cuts stay valid for the binary master).
        y.VType = GRB.CONTINUOUS
        self.lp_phase = {"iterations": 0, "bound": None, "stopped": "converged"}
        # The runtime holds the phase reserve, so no LP solve or recourse LP can eat into the binary master's share.
        self.runtime.stage_reserve = self.lp_phase_share * (self.runtime.time_limit - self.runtime.output_reserve)
        try:
            for _ in range(self.lp_phase_iters):
                try:
                    self.runtime.optimize(model, role="master_lp", allow_incumbent=False)
                    if model.Status != GRB.OPTIMAL:
                        raise RuntimeError(f"LP master failed (status={model.Status})")
                    y_frac = np.asarray(y.X, dtype=float)
                    cut = self._cut_at(inst, y_frac, float(z.X), self._solve_subproblem(inst, y_frac))
                except ReferenceLimit:
                    self.lp_phase["stopped"] = "time_share"
                    break
                self.lp_phase.update(iterations=self.lp_phase["iterations"] + 1, bound=float(model.ObjVal))
                if cut is None:
                    break
                model.addConstr(self._cut_expr(master, cut[1]) <= (master.z if cut[0] == "opt" else 0.0))
        finally:
            self.runtime.stage_reserve = 0.0
            model.Params.TimeLimit = float(self.master_params.get("TimeLimit", GRB.INFINITY))  # drop the phase-capped limit
        y.VType = GRB.BINARY
        model.Params.LazyConstraints = 1
        model.update()
        self._sub_env = gp.Env(params={"OutputFlag": 0, "Threads": 1})
        return master

    def _cut_at(self, inst: FCTPInstance, y_vals: np.ndarray, z_val: float, sub) -> tuple[str, FCTPFeasCut | FCTPOptCut] | None:
        """The violated cut at (y, z) from a transportation-LP result, or None."""
        sp_obj, pi, sigma, mu, _, feasible = sub
        const_term = float(inst.supply @ pi + inst.demand @ sigma)
        coeff = inst.capacity * mu
        value = const_term + float((coeff * y_vals).sum())
        if not feasible:
            if value <= 1e-6:
                raise RuntimeError(f"Feasibility cut not violated at current y (expr={value:.3e}); check dual signs.")
            return "feas", FCTPFeasCut(const_term=const_term, coeffs=coeff)
        if value - z_val > 1e-6 * max(1.0, abs(value)):
            return "opt", FCTPOptCut(const_term=const_term, coeffs=coeff)
        return None

    @staticmethod
    def _cut_expr(master: FCTPMasterState, cut: FCTPFeasCut | FCTPOptCut):
        m, n = cut.coeffs.shape
        return cut.const_term + gp.quicksum(float(cut.coeffs[i, j]) * master.y[i, j] for i in range(m) for j in range(n) if cut.coeffs[i, j] != 0.0)

    def _callback_subproblem(self, inst: FCTPInstance, y_vals: np.ndarray):
        """Transportation LP for a callback (own environment, no runtime bookkeeping); same dual conventions."""
        m, n = inst.m, inst.n
        with gp.Model(env=self._sub_env) as model:
            model.Params.InfUnbdInfo = 1
            x = model.addMVar((m, n), lb=0.0)
            supply = [model.addConstr(x[i, :].sum() <= inst.supply[i]) for i in range(m)]
            demand = [model.addConstr(x[:, j].sum() >= inst.demand[j]) for j in range(n)]
            link = [[model.addConstr(x[i, j] <= inst.capacity[i, j] * float(y_vals[i, j])) for j in range(n)] for i in range(m)]
            model.setObjective((inst.unit_cost * x).sum(), GRB.MINIMIZE)
            model.Params.TimeLimit = self.runtime.check()  # nested solve: capped by the shared budget (raises when spent)
            model.optimize()
            if model.Status == GRB.TIME_LIMIT:
                raise ReferenceLimit("callback recourse LP reached the shared time limit")
            if model.Status == GRB.INFEASIBLE:
                pi = -np.array([c.FarkasDual for c in supply]); sigma = -np.array([c.FarkasDual for c in demand])
                mu = -np.array([[c.FarkasDual for c in row] for row in link])
                return None, pi, sigma, mu, None, False
            if model.Status != GRB.OPTIMAL:
                raise RuntimeError(f"Subproblem not optimal (status={model.Status})")
            pi = np.array([c.Pi for c in supply]); sigma = np.array([c.Pi for c in demand])
            mu = np.array([[c.Pi for c in row] for row in link])
            return float(model.ObjVal), pi, sigma, mu, np.asarray(x.X, dtype=float), True

    def solve_master(self, inst: FCTPInstance, master: FCTPMasterState) -> MasterSolution[FCTPMasterState]:
        errors: list[BaseException] = []
        y_vars = [v for row in master.y.tolist() for v in row]

        def lazy(model, where):
            if where != GRB.Callback.MIPSOL or errors:
                return
            try:
                y_sol = np.round(np.asarray(model.cbGetSolution(y_vars), dtype=float)).reshape(inst.m, inst.n)
                z_sol = float(model.cbGetSolution(master.z))
                cut = self._cut_at(inst, y_sol, z_sol, self._callback_subproblem(inst, y_sol))
                if cut is not None:
                    expr = self._cut_expr(master, cut[1])
                    model.cbLazy(expr <= (master.z if cut[0] == "opt" else 0.0))
            except BaseException as exc:  # surfaced after optimize(); never let a candidate pass unchecked
                errors.append(exc)
                model.terminate()

        # Keep time for the final recourse evaluation of the returned design (BaseBenders.run re-evaluates it exactly).
        self.runtime.stage_reserve = self.final_eval_reserve
        timed_out = False
        try:
            self.runtime.optimize(_CallbackModel(master.model, lazy), role="master", allow_incumbent=True)
        except ReferenceLimit:
            timed_out = True  # no incumbent at the cutoff: propagated below unless a callback error explains it
        finally:
            self.runtime.stage_reserve = 0.0
        hard = [e for e in errors if not isinstance(e, ReferenceLimit)]
        if hard:
            raise RuntimeError(f"Benders callback failed: {hard[0]!r}")
        status = master.model.Status
        if status != GRB.OPTIMAL:
            # Time ran out (master limit or a recourse LP in the callback). A stored incumbent is still usable: its
            # design is re-evaluated by the exact transportation LP, and ObjBound stays a valid lower bound.
            if timed_out or status not in (GRB.TIME_LIMIT, GRB.INTERRUPTED) or master.model.SolCount == 0 \
                    or (status == GRB.INTERRUPTED and not errors):
                raise ReferenceLimit(f"Master stopped without a usable incumbent (status={status})")
        y_val = np.round(np.asarray(master.y.X, dtype=float))
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
            bound=float(master.model.ObjBound),
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
        self.runtime.optimize(model, role="recourse_lp", allow_incumbent=False)

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
        metadata: dict[str, Any] = {"y_assignment": y_vals.tolist(), "lp_phase": dict(getattr(self, "lp_phase", {}))}

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
        lb_candidate = master_solution.bound if master_solution.bound is not None else master_solution.objective

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


class _CallbackModel:
    """Lets ReferenceRuntime.optimize (which calls model.optimize()) run a model with a callback."""

    def __init__(self, model: gp.Model, callback) -> None:
        self._model, self._callback = model, callback

    def optimize(self) -> None:
        self._model.optimize(self._callback)

    def __getattr__(self, name):
        return getattr(self._model, name)
