"""License-free HiGHS baselines for the route-enumerated LRP pilot.

These are new pilot implementations, NOT the paper's Gurobi timing baseline.
All cuts and incumbent witnesses are recorded to support independent audits.
"""

from __future__ import annotations

import math
import time
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, OptimizeWarning, linprog, milp
from scipy.sparse import csc_matrix, diags, eye, hstack, vstack

from .data import validate_data, verify


DEFAULT_METHODS = ("monolithic", "lp_benders", "integer_benders")
METHODS = (
    *DEFAULT_METHODS,
    "integer_benders_lp_first",
    "integer_benders_lp_first_repair",
    "integer_benders_exact_cover",
)


class BudgetExpired(Exception):
    pass


def relative_gap(lower, upper):
    if lower is None or upper is None or not math.isfinite(lower) or not math.isfinite(upper):
        return None
    if lower > upper + 1e-6 * max(1, abs(upper)):
        return None
    return max(0.0, upper - lower) / max(1.0, abs(upper))


def finite(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def point_cut(y, recourse_lower_bound):
    """theta >= q * (1 - Hamming(y, ybar)); assumes global recourse >= 0.

    At ybar this enforces theta>=q. At every other binary vector the RHS
    is nonpositive. q must be a valid lower bound at ybar, not just an incumbent.
    """
    y = np.asarray(y, dtype=int)
    q = float(recourse_lower_bound)
    return {
        "kind": "integer_optimality",
        "constant": q * (1 - int(y.sum())),
        "coefficients": (q * (2 * y - 1)).tolist(),
        "y": y.tolist(),
        "q": q,
    }


def exclusion_cut(y):
    """Hamming(y, ybar)>=1: exclude only a proved-infeasible binary plan."""
    y = np.asarray(y, dtype=int)
    return {
        "kind": "integer_feasibility",
        "constant": float(1 - y.sum()),
        "coefficients": (2 * y - 1).tolist(),
        "y": y.tolist(),
    }


class Solver:
    def __init__(self, data, method, seconds=30.0, max_iterations=1000, witness=None, checkpoint=None):
        if method not in METHODS or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid method or time budget")
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.start = time.monotonic()
        self.deadline = self.start + seconds
        self.data, self.method = data, method
        self.seconds, self.max_iterations = seconds, max_iterations
        self.checkpoint = checkpoint or (lambda _: None)
        validate_data(data)
        self.m, self.n, self.r = data["m"], data["n"], len(data["routes"])
        self.a = csc_matrix(np.asarray(data["incidence"], dtype=float))
        self.fleet = csc_matrix((np.ones(self.r), (data["route_depot"], np.arange(self.r))), shape=(self.m, self.r))
        self.f, self.c = np.asarray(data["depot_cost"]), np.asarray(data["route_cost"])
        self.vehicles = np.asarray(data["max_vehicles"], dtype=float)
        self.incumbent = None
        self.lower = 0.0  # all costs are nonnegative, checked by validate_data
        self.solves, self.trace, self.cuts = [], [], []
        self.reason = "running"
        if witness is not None:
            self.accept(witness, source="shared_generator_witness")

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0.001:
            raise BudgetExpired
        return remaining

    def optimize(self, role, objective, matrix, lower, upper, integer, bounds, seconds=None):
        before = time.monotonic()
        limit = self.remaining()
        if seconds is not None:
            limit = min(limit, seconds)
        with warnings.catch_warnings():
            # scipy forwards this documented HiGHS option but warns it is not
            # part of scipy's public option subset. Do not suppress other warnings.
            warnings.filterwarnings(
                "ignore", message="Unrecognized options detected:.*threads.*", category=RuntimeWarning
            )
            result = milp(
                np.asarray(objective, dtype=float),
                integrality=integer,
                bounds=bounds,
                constraints=LinearConstraint(matrix, lower, upper),
                options={"time_limit": limit, "mip_rel_gap": 0.0, "threads": 1},
            )
        self.solves.append(
            {
                "role": role,
                "seconds": time.monotonic() - before,
                "time_limit_seconds": limit,
                "status": int(result.status),
                "objective": finite(result.fun),
                "dual_bound": finite(result.get("mip_dual_bound")),
                "nodes": finite(result.get("mip_node_count")),
                "message": result.message,
            }
        )
        return result

    def recourse_lp(self, y, phase_one=False):
        before = time.monotonic()
        if phase_one:
            # Unmet-customer slacks make phase I feasible for every depot plan.
            cost = np.r_[np.zeros(self.r), np.ones(self.n)]
            aeq = hstack([self.a, eye(self.n)], format="csc")
            aub = hstack([self.fleet, csc_matrix((self.m, self.n))], format="csc")
        else:
            cost, aeq, aub = self.c, self.a, self.fleet
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Unrecognized options detected:.*threads.*", category=OptimizeWarning
            )
            result = linprog(
                cost,
                A_eq=aeq,
                b_eq=np.ones(self.n),
                A_ub=aub,
                b_ub=self.vehicles * y,
                bounds=(0, None),
                method="highs-ds",
                options={"time_limit": self.remaining(), "threads": 1},
            )
        self.solves.append(
            {
                "role": "phase_one" if phase_one else "recourse_lp",
                "seconds": time.monotonic() - before,
                "status": int(result.status),
                "objective": finite(result.fun),
                "message": result.message,
            }
        )
        return result

    def lp_cut(self, result, y, feasibility=False):
        alpha, beta = result.eqlin.marginals, result.ineqlin.marginals
        # Reject dubious duals instead of using them to claim an optimality bound.
        dual_lhs = self.a.T @ alpha + self.fleet.T @ beta
        rhs = np.zeros(self.r) if feasibility else self.c
        if not np.isfinite(alpha).all() or not np.isfinite(beta).all() or np.max(beta) > 1e-7:
            raise RuntimeError("Invalid LP dual multipliers")
        if np.max(dual_lhs - rhs) > 1e-7 or (feasibility and np.max(alpha) > 1 + 1e-7):
            raise RuntimeError("LP dual constraints violated")
        constant = float(alpha.sum())
        coefficients = beta * self.vehicles
        if not math.isclose(constant + coefficients @ y, result.fun, abs_tol=1e-6, rel_tol=1e-7):
            raise RuntimeError("LP primal/dual mismatch")
        # Record the unmodified dual certificate. Solvers use floating-point tolerances.
        return {
            "kind": "lp_feasibility" if feasibility else "lp_optimality",
            "constant": constant,
            "coefficients": coefficients.tolist(),
            "y": y.tolist(),
            "recourse_value": float(result.fun),
        }

    def recourse_integer(self, y, seconds=None):
        matrix = vstack([self.a, self.fleet], format="csc")
        return self.optimize(
            "integer_recourse",
            self.c,
            matrix,
            np.r_[np.ones(self.n), np.full(self.m, -np.inf)],
            np.r_[np.ones(self.n), self.vehicles * y],
            np.ones(self.r),
            Bounds(np.zeros(self.r), np.ones(self.r)),
            seconds=seconds,
        )

    def repair_openings(self, y, iteration):
        """Primal heuristic: allow one extra depot after unresolved integer recourse.

        Each neighbor uses at most two seconds within the original wall budget.
        No cut or global bound is inferred from these auxiliary solves. Any
        incumbent must pass the same independent whole-route verifier.
        """
        closed = sorted(np.flatnonzero(y == 0), key=lambda j: (self.f[j], j))
        for depot in closed:
            neighbor = y.copy()
            neighbor[depot] = 1
            result = self.recourse_integer(neighbor, seconds=2.0)
            self.solves[-1].update(role="repair_integer_recourse", y=neighbor.tolist())
            if result.x is not None:
                # An allowed depot need not remain open if no chosen route uses
                # it. Nonnegative opening costs make this a valid improvement.
                used = (self.fleet @ np.rint(result.x) > 0).astype(int)
                self.accept(self.output(used, result.x), "one_extra_depot_repair")
            self.record(iteration, y=neighbor.tolist(), repair_added_depot=int(depot), repair_status=int(result.status))

    def recourse_exact_cover(self, y, lp, iteration):
        from .exact_cover import solve_exact_cover

        def incumbent(routes, _cost):
            lam = np.zeros(self.r)
            lam[routes] = 1
            self.accept(self.output(y, lam), "exact_cover_recourse")
            self.record(iteration, y=y.tolist(), exact_cover_incumbent=True)

        result = solve_exact_cover(self.data, y, self.remaining(), lp.eqlin.marginals, lp.ineqlin.marginals, incumbent)
        self.solves.append({"role": "exact_cover_recourse", "y": y.tolist(), **result})
        lam = None
        if result["routes"] is not None:
            lam = np.zeros(self.r)
            lam[result["routes"]] = 1
        return OptimizeResult(
            status={"optimal": 0, "time_limit": 1, "infeasible": 2}[result["status"]],
            x=lam,
            fun=result["objective"],
            mip_dual_bound=result["lower_bound"],
        )

    def accept(self, output, source):
        checked = verify(self.data, output)
        if not checked["ok"]:
            raise RuntimeError(f"Invalid incumbent: {checked['issues']}")
        if self.incumbent is None or checked["recomputed_objective"] < self.incumbent["objective_value"] - 1e-8:
            self.incumbent = {
                **output,
                "objective_value": checked["recomputed_objective"],
                "source": source,
                "found_at_seconds": time.monotonic() - self.start,
            }

    def output(self, y, lam):
        return {
            "y": np.rint(y).astype(int).tolist(),
            "lambda": np.rint(lam).astype(int).tolist(),
            "objective_value": float(self.f @ np.rint(y) + self.c @ np.rint(lam)),
        }

    def record(self, iteration, **details):
        upper = self.incumbent["objective_value"] if self.incumbent else None
        if upper is not None and self.lower > upper + 1e-6 * max(1, abs(upper)):
            raise RuntimeError("Lower bound exceeds verified incumbent")
        self.trace.append(
            {
                "iteration": iteration,
                "seconds": time.monotonic() - self.start,
                "lower_bound": self.lower,
                "upper_bound": upper,
                "relative_gap": relative_gap(self.lower, upper),
                "cuts": len(self.cuts),
                **details,
            }
        )
        self.checkpoint(self.result())

    def closed(self):
        upper = self.incumbent["objective_value"] if self.incumbent else None
        gap = relative_gap(self.lower, upper)
        return gap is not None and gap <= 1e-7

    def run_monolithic(self):
        matrix = vstack(
            [
                hstack([csc_matrix((self.n, self.m)), self.a]),
                hstack([-diags(self.vehicles), self.fleet]),
                csc_matrix(np.r_[np.ones(self.m), np.zeros(self.r)][None, :]),
            ],
            format="csc",
        )
        result = self.optimize(
            "monolithic",
            np.r_[self.f, self.c],
            matrix,
            np.r_[np.ones(self.n), np.full(self.m, -np.inf), 1.0],
            np.r_[np.ones(self.n), np.zeros(self.m), np.inf],
            np.ones(self.m + self.r),
            Bounds(np.zeros(self.m + self.r), np.ones(self.m + self.r)),
        )
        if result.x is not None:
            self.accept(self.output(result.x[: self.m], result.x[self.m :]), "monolithic")
        bound = finite(result.get("mip_dual_bound"))
        if bound is not None:
            self.lower = max(self.lower, bound)
        self.reason = {0: "optimal", 1: "time_limit", 2: "infeasible"}.get(result.status, "solver_error")
        self.record(1)

    def master(self):
        rows, lower, upper = [np.r_[np.ones(self.m), 0.0]], [1.0], [np.inf]
        for cut in self.cuts:
            optimality = cut["kind"].endswith("optimality")
            rows.append(np.r_[-np.asarray(cut["coefficients"]), float(optimality)])
            lower.append(cut["constant"])
            upper.append(np.inf)
        return self.optimize(
            "master",
            np.r_[self.f, 1.0],
            csc_matrix(np.array(rows)),
            lower,
            upper,
            np.r_[np.ones(self.m), 0.0],
            Bounds(np.zeros(self.m + 1), np.r_[np.ones(self.m), np.inf]),
        )

    def run_benders(self):
        for iteration in range(1, self.max_iterations + 1):
            self.remaining()
            master = self.master()
            bound = finite(master.get("mip_dual_bound"))
            if bound is not None:
                self.lower = max(self.lower, bound)
            if self.closed():
                self.reason = "bound_closed"
                self.record(iteration)
                return
            if master.status == 2:
                self.reason = "infeasible"
                self.record(iteration)
                return
            if master.x is None or master.status != 0:
                self.reason = "master_unresolved"
                self.record(iteration)
                return
            y = np.rint(master.x[: self.m]).astype(int)
            theta = master.x[-1]
            lp = self.recourse_lp(y)
            if lp.status == 2:
                phase = self.recourse_lp(y, phase_one=True)
                if phase.status != 0 or phase.fun <= 1e-7:
                    self.reason = "feasibility_unresolved"
                    self.record(iteration, y=y.tolist())
                    return
                self.cuts.append(self.lp_cut(phase, y, feasibility=True))
                self.record(iteration, y=y.tolist(), phase_one_value=float(phase.fun))
                continue
            if lp.status != 0:
                self.reason = "lp_unresolved"
                self.record(iteration, y=y.tolist())
                return
            before = len(self.cuts)
            if lp.fun > theta + 1e-7 * max(1.0, abs(theta)):
                self.cuts.append(self.lp_cut(lp, y))
                if self.method in (
                    "integer_benders_lp_first",
                    "integer_benders_lp_first_repair",
                    "integer_benders_exact_cover",
                ):
                    # Reoptimize the master with this cheap LP information before
                    # committing the remaining budget to this depot plan's MIP.
                    self.record(iteration, y=y.tolist(), lp_recourse=float(lp.fun), integer_recourse_deferred=True)
                    continue
            if self.method == "integer_benders_exact_cover":
                integer = self.recourse_exact_cover(y, lp, iteration)
            elif self.method == "integer_benders_lp_first_repair":
                integer = self.recourse_integer(y, seconds=2.0)
                if integer.x is not None:
                    self.accept(self.output(y, integer.x), "integer_recourse")
                if integer.status == 1:
                    self.record(iteration, y=y.tolist(), repair_trigger="integer_recourse_time_limit")
                    self.repair_openings(y, iteration)
                    if self.closed():
                        self.reason = "bound_closed"
                        return
                    # Return to the unresolved master choice using the remaining
                    # budget. Auxiliary neighbors were never excluded or used as
                    # global lower bounds. This is a restart, not a warm start.
                    integer = self.recourse_integer(y)
            else:
                integer = self.recourse_integer(y)
            if integer.x is not None:
                self.accept(self.output(y, integer.x), "integer_recourse")
            if self.method != "lp_benders":
                if integer.status == 2:
                    self.cuts.append(exclusion_cut(y))
                elif integer.status == 0:
                    # Use the MIP dual bound, never a time-limited incumbent, in a cut.
                    q = finite(integer.get("mip_dual_bound"))
                    if q is None:
                        raise RuntimeError("Integer recourse returned no bound")
                    self.cuts.append(point_cut(y, max(0.0, q)))
                else:
                    self.reason = "integer_recourse_unresolved"
                    self.record(iteration, y=y.tolist())
                    return
            self.record(iteration, y=y.tolist(), lp_recourse=float(lp.fun), integer_recourse=finite(integer.fun))
            if self.closed():
                self.reason = "bound_closed"
                return
            if len(self.cuts) == before:
                self.reason = "lp_stalled_with_integer_gap"
                return
        self.reason = "iteration_limit"

    def run(self):
        self.record(0)
        try:
            if self.method == "monolithic":
                self.run_monolithic()
            else:
                self.run_benders()
        except BudgetExpired:
            self.reason = "time_limit"
        result = self.result()
        if self.reason == "infeasible" and self.incumbent is not None:
            raise RuntimeError("Infeasibility contradicts a verified incumbent")
        self.checkpoint(result)
        return result

    def result(self):
        upper = self.incumbent["objective_value"] if self.incumbent else None
        if self.closed():
            status = "optimal"
        elif self.incumbent is not None:
            status = "feasible"
        else:
            status = "infeasible" if self.reason == "infeasible" else "no_incumbent"
        return {
            "method": self.method,
            "backend": "scipy-highs",
            "status": status,
            "termination_reason": self.reason,
            "lower_bound": self.lower,
            "upper_bound": upper,
            "relative_gap": relative_gap(self.lower, upper),
            "solver_budget_seconds": self.seconds,
            "method_seconds": time.monotonic() - self.start,
            "threads": 1,
            "incumbent": self.incumbent,
            "trace": self.trace,
            "cuts": self.cuts,
            "solves": self.solves,
            "verification": verify(self.data, self.incumbent) if self.incumbent else None,
            "scope": "optimum over supplied routes; not the paper Gurobi baseline",
        }
