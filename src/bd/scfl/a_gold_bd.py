"""Multi-cut L-shaped (Benders) reference for SCFL: an outer loop over an integer master.

Master:  min f.y + sum_s p_s theta_s  over y in {0, 1}^I with the (15d) cover sum_i u_i y_i >= max_s D_s,
         theta_s >= sum_j d_sj min_i c_ij (every unit at its cheapest cost) and all optimality cuts so far.
Subproblem per scenario at a binary design ybar: the transportation LP over all facilities with capacity rows
sum_j x_ij <= u_i ybar_i (closed facilities get capacity 0), duals v_j (demand rows) and alpha_i >= 0 (capacity).
Every (v, alpha >= 0) with beta_ij = (v_j - alpha_i - c_ij)^+ is dual feasible for the strong-linking recourse LP
at every y in [0, 1]^I, so the cut
    theta_s >= sum_j d_sj v_j - sum_i (u_i alpha_i + sum_j d_sj beta_ij) y_i
is globally valid; with the LP's optimal duals it is tight at ybar. For a closed facility the cut value at ybar does
not depend on alpha_i, so alpha_i is chosen to minimize its coefficient u_i a + sum_j d_sj (v_j - c_ij - a)^+ over
a >= 0 (closed-facility lifting, as in the scale-up reference; the minimum is at 0 or a breakpoint v_j - c_ij).
Every master design is evaluated exactly (all S LPs), giving the upper bound; the master's proved bound is the lower
bound. The loop stops when no attainable objective value (a multiple of 0.001 / S, see `objective_step`) lies
strictly between them; if the master repeats a design, its tight cuts force the bounds to meet.
Shortcuts for the screen: `multi_cut=False` adds one aggregated cut sum_s p_s (cut_s) per iteration (the textbook
L-shaped method; exact but slower); `expected_value=True` runs the method on the mean-demand scenario and evaluates
that design exactly (not exact for the stochastic problem).
"""
from __future__ import annotations

import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.bd.scfl.a_gen_data import SCFLInstance
from src.bd.scfl.a_gold_monolithic import certified, lattice_bound, objective_step
from src.utils.reference_runtime import ReferenceRuntime


class SCFLBenders:
    def __init__(self, *, time_limit: float = 60.0, multi_cut: bool = True, expected_value: bool = False) -> None:
        self.runtime = ReferenceRuntime(time_limit)
        self.multi_cut = multi_cut
        self.expected_value = expected_value

    def _transport(self, inst: SCFLInstance):
        I, J = range(inst.n_facilities), range(inst.n_customers)
        lp = gp.Model("scfl_recourse")
        lp.Params.OutputFlag = 0
        x = lp.addVars(I, J, lb=0.0)
        dem = lp.addConstrs(x.sum("*", j) == 0.0 for j in J)
        cap = lp.addConstrs(x.sum(i, "*") <= 0.0 for i in I)
        lp.setObjective(gp.quicksum(inst.unit_cost[i][j] * x[i, j] for i in I for j in J), GRB.MINIMIZE)
        return lp, dem, cap

    def _evaluate(self, inst, sub, y, demands):
        """Exact recourse values and the lifted optimality cut (constant, coefficient per facility) per scenario."""
        lp, dem, cap = sub
        I, J = range(inst.n_facilities), range(inst.n_customers)
        for i in I:
            cap[i].RHS = inst.capacity[i] * y[i]
        values, cuts = [], []
        for d in demands:
            for j in J:
                dem[j].RHS = d[j]
            self.runtime.optimize(lp, role="subproblem")
            if lp.Status != GRB.OPTIMAL:
                raise RuntimeError(f"recourse LP not optimal (status={lp.Status}); the design violates the cover row")
            v = [dem[j].Pi for j in J]
            coef = []
            for i in I:
                w = [v[j] - inst.unit_cost[i][j] for j in J]
                if y[i]:
                    a = max(0.0, -cap[i].Pi)
                    coef.append(inst.capacity[i] * a + sum(d[j] * max(0.0, w[j] - a) for j in J))
                else:
                    coef.append(min(inst.capacity[i] * a + sum(d[j] * max(0.0, w[j] - a) for j in J) for a in [0.0, *(t for t in w if t > 0)]))
            values.append(float(lp.ObjVal))
            cuts.append((sum(d[j] * v[j] for j in J), coef))
        return values, cuts

    def run(self, instance_name: str, *, started_at: float | None = None) -> dict[str, Any]:
        self.runtime.start(started_at)
        started = time.perf_counter()
        inst = SCFLInstance.from_json_file(instance_name)
        I, J, S = range(inst.n_facilities), range(inst.n_customers), range(inst.n_scenarios)
        if self.expected_value:
            demands, probs = [[sum(inst.probability[s] * inst.demand[s][j] for s in S) for j in J]], [1.0]
        else:
            demands, probs = inst.demand, inst.probability
        K = range(len(demands)) if self.multi_cut else range(1)
        sub = self._transport(inst)
        step = objective_step(inst)

        master = gp.Model("scfl_master")
        master.Params.OutputFlag = 0
        master.Params.MIPGap, master.Params.MIPGapAbs = 0.0, 0.0
        y = master.addVars(I, vtype=GRB.BINARY, name="y")
        cheapest = [sum(d[j] * min(inst.unit_cost[i][j] for i in I) for j in J) for d in demands]
        floor = cheapest if self.multi_cut else [sum(p * c for p, c in zip(probs, cheapest))]
        theta = master.addVars(K, lb=[floor[k] for k in K], name="theta")
        master.addConstr(gp.quicksum(inst.capacity[i] * y[i] for i in I) >= inst.max_demand, name="cover")
        weights = probs if self.multi_cut else [1.0]
        master.setObjective(gp.quicksum(inst.fixed_cost[i] * y[i] for i in I) + gp.quicksum(weights[k] * theta[k] for k in K), GRB.MINIMIZE)

        def add_cuts(values, cuts):
            """Add the cut of every theta whose current master value is below its exact recourse value."""
            added = 0
            if self.multi_cut:
                groups = [(k, [(1.0, cuts[k])], values[k]) for k in K]
            else:
                groups = [(0, [(probs[s], cuts[s]) for s in range(len(cuts))], sum(p * q for p, q in zip(probs, values)))]
            for k, parts, value in groups:
                if master.SolCount and theta[k].X >= value - 1e-9 * max(1.0, abs(value)):
                    continue
                const = sum(p * c for p, (c, _) in parts)
                master.addConstr(theta[k] >= const - gp.quicksum(sum(p * coef[i] for p, (_, coef) in parts) * y[i] for i in I))
                added += 1
            return added

        def value_of(design, values):
            return sum(inst.fixed_cost[i] * design[i] for i in I) + sum(p * q for p, q in zip(probs, values))

        best, ub, lb, iters, termination = None, float("inf"), -float("inf"), 0, "optimal"
        evaluated: set[tuple[int, ...]] = set()
        design = [1] * inst.n_facilities  # initial design: everything open (satisfies the cover row)
        try:
            values, cuts = self._evaluate(inst, sub, design, demands)
            best, ub = design, value_of(design, values)
            evaluated.add(tuple(design))
            add_cuts(values, cuts)
            while True:
                iters += 1
                self.runtime.optimize(master, role="master", allow_incumbent=False)
                if master.Status != GRB.OPTIMAL:
                    raise RuntimeError(f"master not optimal (status={master.Status})")
                lb = max(lb, float(master.ObjBound))
                if certified(ub, lb, step):
                    break
                design = [int(y[i].X > 0.5) for i in I]
                values, cuts = self._evaluate(inst, sub, design, demands)
                if value_of(design, values) < ub:
                    best, ub = design, value_of(design, values)
                repeat = tuple(design) in evaluated
                evaluated.add(tuple(design))
                if not add_cuts(values, cuts) and repeat:
                    # No violated cut at a known design: the master value equals the design value up to tolerance.
                    termination = "stalled"
                    break
        except Exception as exc:
            if "budget" not in str(exc).lower() and "time limit" not in str(exc).lower():
                raise
            termination = "time_limit"
        if best is None:
            raise RuntimeError("Benders found no feasible design")
        if self.expected_value:
            # Exact stochastic value of the expected-value design; no stochastic lower bound.
            values, _ = self._evaluate(inst, sub, best, inst.demand)
            ub = sum(inst.fixed_cost[i] * best[i] for i in I) + sum(p * q for p, q in zip(inst.probability, values))
            lb, termination = None, "expected_value_design"
        proved = lb is not None and lb > -float("inf") and certified(ub, lb, step)
        if proved:
            lb = lattice_bound(lb, step)  # the bound snapped to its lattice value, which closes the gap
        result = {"best_objective": ub, "global_lb": lb if lb is not None and lb > -float("inf") else None, "global_ub": ub,
                  "y": best, "iters": iters, "n_cuts": master.NumConstrs - 1}
        result = self.runtime.finish(result, target_gap=0.0, termination_reason=termination)
        # The lattice certificate, not the shared relative-gap rule, decides the proof on every exit.
        result.update(status="optimal" if proved else "feasible", proven_optimal=proved)
        result["total_time"] = time.perf_counter() - started
        return result
