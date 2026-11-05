from __future__ import annotations

import math
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.bd.scfl.a_gen_data import SCFLInstance
from src.utils.base_baseline_class import BaseBaseline, BaselineSolution


# Optimality certificates.
# Lattice (exact up to a stated coefficient perturbation). Every CAP allocation cost is lambda_bar_j times a multiple of
# 1/80, so every exact unit cost c_ij = alloc_ij / lambda_bar_j is a multiple of 1/80 (checked on all 16 vendored
# files); fixed costs are integers, demands and capacities integers, p_s = 1/S. Hence every design value of the exact
# rational instance f.y + (1/S) sum_s Q_s(y) (integral transportation vertices) is a multiple of 1/(80 S). The JSON
# holds the doubles alloc/lambda_bar (the scale-up conversion, not snapped); they differ from the exact multiples by
# at most `dev` (~1e-14), which moves any design value by at most dev * max_s D_s (`objective_step` requires this to
# be below step/100). A bound and an incumbent with no lattice value strictly between them (half-step margins) prove
# the incumbent optimal for the exact instance, provided the solver's own float error stays below about a quarter
# step (an engineering assumption, as before; step = 0.0125 / S, e.g. 2.5e-4 at S = 50).
# Relative fallback. When the solver cannot close the gap to that margin (or the data lie on no small grid), an
# incumbent within CERT_TOL = 1e-9 relative of the bound is accepted as numerically optimal (1e-4 to 1e-3 absolute at
# objectives 1e5-1e6). Results name the certificate used ("lattice" or "relative").
CERT_TOL = 1e-9
GRID_DENOMINATORS = (1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400, 500, 1000)


def objective_step(inst: SCFLInstance) -> float | None:
    """Spacing 1/(q S) of the exact design values for the smallest q in GRID_DENOMINATORS that puts every unit and
    fixed cost on a 1/q grid (within 1e-12 relative; see above), or None when no such grid exists."""
    S = inst.n_scenarios
    if (any(p != 1.0 / S for p in inst.probability)
            or not all(float(v).is_integer() for v in [*inst.capacity, *(d for row in inst.demand for d in row)])):
        return None
    costs = [*inst.fixed_cost, *(c for row in inst.unit_cost for c in row)]
    for q in GRID_DENOMINATORS:
        dev = max(abs(v - round(v * q) / q) for v in costs)
        if dev <= 1e-12 * max(1.0, max(abs(v) for v in costs)):
            step = 1.0 / (q * S)
            return step if dev * max(1.0, inst.max_demand) * (inst.n_facilities + 1) < step / 100 else None
    return None


def lattice_bound(bound: float, step: float) -> float:
    """A conservative lattice lower bound: a lattice value no greater than the optimum."""
    return math.ceil(bound / step - 0.5) * step


def certificate(incumbent: float, bound: float, step: float | None, tol: float = CERT_TOL) -> str | None:
    """'lattice', 'relative' or None (see above). Non-finite values are rejected. 'lattice' needs the bound within half a
    step above the incumbent (a larger crossing violates the error assumption) and no lattice value strictly between;
    otherwise 'relative' needs |incumbent - bound| <= tol * max(1, |incumbent|) (a crossing within tol is roundoff)."""
    if not (math.isfinite(incumbent) and math.isfinite(bound) and math.isfinite(tol) and tol >= 0):
        return None
    if step and bound <= incumbent + 0.5 * step and incumbent <= lattice_bound(bound, step) + 0.5 * step:
        return "lattice"
    if abs(incumbent - bound) <= tol * max(1.0, abs(incumbent)):
        return "relative"
    return None


def certified(incumbent: float, bound: float, step: float | None) -> bool:
    return certificate(incumbent, bound, step) is not None


def design_value(inst: SCFLInstance, y: list[int]) -> float:
    """f.y + sum_s p_s Q_s(y) recomputed by the independent verifier (min-cost flow, no solver)."""
    from src.bd.scfl.d_verifier import verify_solution

    res = verify_solution(inst.to_dict(), {"y": y})
    if not res["ok"]:
        raise RuntimeError(f"design {y} rejected by the verifier: {res['issues']}")
    return float(res["objective_value"])


def agree(a: float, b: float, step: float | None) -> bool:
    """Two computations of one design value agree (a quarter step, or 1e-10 relative without a lattice)."""
    return abs(a - b) <= (0.25 * step if step else 1e-10 * max(1.0, abs(a)))


class SCFLBaseline(BaseBaseline[SCFLInstance, gp.Model]):
    """Extensive form (Bodur et al. 2017, eq. (15)) with strong linking, as in the scale-up MIP:

        min f.y + sum_s p_s sum_ij c_ij x_sij
        s.t. sum_i x_sij = d_sj,  sum_j x_sij <= u_i y_i,  x_sij <= min(d_sj, u_i) y_i,  sum_i u_i y_i >= max_s D_s,
             x >= 0, y binary.
    For a binary y the strong rows are redundant and the rest separates into the scenario transportation LPs.
    """

    def __init__(self, *, params: dict[str, Any] | None = None, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        self.params = {"Threads": 1, "MIPGap": 0.0, "MIPGapAbs": 0.0, **(params or {})}

    def load_instance(self, instance_name: str) -> SCFLInstance:
        return SCFLInstance.from_json_file(instance_name)

    def build_model(self, inst: SCFLInstance) -> gp.Model:
        I, J, S = range(inst.n_facilities), range(inst.n_customers), range(inst.n_scenarios)
        u, d = inst.capacity, inst.demand
        model = gp.Model("scfl_extensive_form")
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.params.items():
            model.setParam(k, v)
        y = model.addVars(I, vtype=GRB.BINARY, name="y")
        x = model.addVars(S, I, J, lb=0.0, name="x")
        model.addConstrs((x.sum(s, "*", j) == d[s][j] for s in S for j in J), name="demand")
        model.addConstrs((x.sum(s, i, "*") <= u[i] * y[i] for s in S for i in I), name="capacity")
        model.addConstrs((x[s, i, j] <= min(d[s][j], u[i]) * y[i] for s in S for i in I for j in J), name="strong")
        model.addConstr(gp.quicksum(u[i] * y[i] for i in I) >= inst.max_demand, name="cover")
        model.setObjective(gp.quicksum(inst.fixed_cost[i] * y[i] for i in I)
                           + gp.quicksum(inst.probability[s] * inst.unit_cost[i][j] * x[s, i, j] for s in S for i in I for j in J), GRB.MINIMIZE)
        model._y = y  # type: ignore[attr-defined]
        return model

    def solve_model(self, inst: SCFLInstance, model: gp.Model) -> BaselineSolution[gp.Model]:
        model.optimize()
        if model.Status != GRB.OPTIMAL or model.SolCount == 0:
            raise RuntimeError(f"Baseline model not proved optimal (status={model.Status})")
        y = [int(model._y[i].X > 0.5) for i in range(inst.n_facilities)]  # type: ignore[attr-defined]
        step = objective_step(inst)
        value = design_value(inst, y)  # the published objective is the verifier's value of the design
        kind = certificate(value, float(model.ObjBound), step)
        if not agree(value, float(model.ObjVal), step) or kind is None:
            raise RuntimeError(f"Baseline model not certified (bound={model.ObjBound}, incumbent={model.ObjVal}, verified={value})")
        return BaselineSolution(objective=value, vars={}, best_bound=float(model.ObjBound), model_state=model,
                                metadata={"y": y, "certificate": kind, "objective_step": step, "solver_objective": float(model.ObjVal)})
