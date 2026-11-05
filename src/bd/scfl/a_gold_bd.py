"""Multi-cut L-shaped (Benders) reference for SCFL: an outer loop over an integer master.

Master:  min f.y + sum_s p_s theta_s  over y in {0, 1}^I with the (15d) cover in the strengthened form
         sum_i min(u_i, D) y_i >= D, D = max_s D_s (the same binary designs; it keeps the strong recourse LP feasible
         at every fractional root point),
         theta_s >= sum_j d_sj min_i c_ij (every unit at its cheapest cost) and all optimality cuts so far.
Subproblem per scenario at a binary design ybar: the transportation LP over all facilities with capacity rows
sum_j x_ij <= u_i ybar_i (closed facilities get capacity 0), duals v_j (demand rows) and alpha_i >= 0 (capacity).
Every (v, alpha >= 0) with beta_ij = (v_j - alpha_i - c_ij)^+ is dual feasible for the strong-linking recourse LP
at every y in [0, 1]^I, so the cut
    theta_s >= sum_j d_sj v_j - sum_i (u_i alpha_i + sum_j d_sj beta_ij) y_i
is globally valid; with the LP's optimal duals it is tight at ybar. For a closed facility the cut value at ybar does
not depend on alpha_i, so alpha_i is chosen to minimize its coefficient u_i a + sum_j d_sj (v_j - c_ij - a)^+ over
a >= 0 (closed-facility lifting, as in the scale-up reference; the minimum is at 0 or a breakpoint v_j - c_ij).
Root phase (McDaniel and Devine 1977, "A modified Benders' partitioning algorithm for mixed integer programming",
Management Science 24(3)): before the integer loop the master is solved with y relaxed to [0, 1] and cut at its
fractional solutions until its value stalls (after 10 rounds, relative improvement below 1e-7 over 5 rounds), no cut is
violated, the strong LP is infeasible at the point (also at y + 1e-6), or 40% of the budget is used (every root
master and subproblem solve is capped at that deadline; a root solve that hits it ends the phase, and the integer loop
continues with the cuts found so far). There the
subproblem is the strong-linking recourse LP (capacity rows sum_j x_ij <= u_i ybar_i and
x_ij <= d_sj ybar_i), and its optimal duals (v free, alpha <= 0, beta <= 0) give the globally valid cut
    theta_s >= sum_j d_sj v_j + sum_i (u_i alpha_i + sum_j d_sj beta_ij) y_i,
tight at ybar; the relaxed master value is a valid lower bound. These are the extensive form's own linking rows, so
the root phase targets the extensive form's LP bound (Bodur et al.'s "BEN" root) and reaches it when it converges
before the stall and budget exits; on the CAP data that bound is close to the optimum.
Every master design is evaluated exactly (all S LPs), giving the upper bound; the master's proved bound is the lower
bound. The loop stops when `a_gold_monolithic.certificate` accepts (no attainable value on the 1/(80 S) lattice
strictly between bound and incumbent, else the 1e-9 relative fallback); if the master repeats a design, its tight
cuts force the bounds to meet. The published objective is the independent verifier's value of the best design.
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
from src.bd.scfl.a_gold_monolithic import agree, certificate, certified, design_value, lattice_bound, objective_step
from src.utils.reference_runtime import ReferenceLimit, ReferenceRuntime


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

    def _strong(self, inst: SCFLInstance):
        """Recourse LP with the strong-linking rows, for fractional master points."""
        I, J = range(inst.n_facilities), range(inst.n_customers)
        lp = gp.Model("scfl_recourse_strong")
        lp.Params.OutputFlag = 0
        x = lp.addVars(I, J, lb=0.0)
        dem = lp.addConstrs(x.sum("*", j) == 0.0 for j in J)
        cap = lp.addConstrs(x.sum(i, "*") <= 0.0 for i in I)
        link = lp.addConstrs(x[i, j] <= 0.0 for i in I for j in J)
        lp.setObjective(gp.quicksum(inst.unit_cost[i][j] * x[i, j] for i in I for j in J), GRB.MINIMIZE)
        return lp, dem, cap, link

    def _evaluate_fractional(self, inst, strong, y, demands, *, solve):
        """Recourse values and strong-linking dual cuts (constant, coefficient per facility) at a fractional y;
        None when a solve is not optimal or `solve` reports the root deadline."""
        lp, dem, cap, link = strong
        I, J = range(inst.n_facilities), range(inst.n_customers)
        lp.setAttr("RHS", [cap[i] for i in I], [inst.capacity[i] * y[i] for i in I])
        keys = [(i, j) for i in I for j in J]
        rows = [link[k] for k in keys]
        values, cuts = [], []
        for d in demands:
            lp.setAttr("RHS", [dem[j] for j in J], [float(d[j]) for j in J])
            lp.setAttr("RHS", rows, [d[j] * y[i] for i, j in keys])
            if not solve(lp, "subproblem"):
                return None
            if lp.Status != GRB.OPTIMAL:
                return None  # e.g. a cover row met only within the master's tolerance: no cut from this point
            v = lp.getAttr("Pi", [dem[j] for j in J])
            alpha = lp.getAttr("Pi", [cap[i] for i in I])
            beta = lp.getAttr("Pi", rows)
            coef = [-(inst.capacity[i] * min(0.0, alpha[i]) + sum(d[j] * min(0.0, beta[i * len(J) + j]) for j in J)) for i in I]
            values.append(float(lp.ObjVal))
            cuts.append((sum(d[j] * v[j] for j in J), coef))
        return values, cuts

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
        master.addConstr(gp.quicksum(min(inst.capacity[i], inst.max_demand) * y[i] for i in I) >= inst.max_demand, name="cover")
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

        best, ub, lb, iters, termination, root_rounds = None, float("inf"), -float("inf"), 0, "optimal", 0
        evaluated: set[tuple[int, ...]] = set()
        design = [1] * inst.n_facilities  # initial design: everything open (satisfies the cover row)
        try:
            values, cuts = self._evaluate(inst, sub, design, demands)
            best, ub = design, value_of(design, values)
            evaluated.add(tuple(design))
            add_cuts(values, cuts)
            # Root phase on the LP relaxation of the master (see the module docstring).
            strong = self._strong(inst)
            master.setAttr("VType", [y[i] for i in I], [GRB.CONTINUOUS] * len(I))
            history: list[float] = []
            deadline = 0.4 * self.runtime.time_limit

            def root_solve(model, role):
                """Solve within the root deadline; False when the deadline has passed."""
                left = deadline - self.runtime.elapsed()
                if left <= 0:
                    return False
                model.Params.TimeLimit = left
                self.runtime.optimize(model, role=role)
                return True

            try:
                while self.runtime.elapsed() < deadline:
                    if not root_solve(master, "master_lp"):
                        break
                    if master.Status != GRB.OPTIMAL:
                        raise RuntimeError(f"master LP not optimal (status={master.Status})")
                    history.append(float(master.ObjVal))
                    lb = max(lb, history[-1])  # a relaxation of the problem: a valid lower bound
                    ybar = [y[i].X for i in I]
                    evaluated_lp = self._evaluate_fractional(inst, strong, ybar, demands, solve=root_solve)
                    if evaluated_lp is None:  # cuts are valid everywhere: retry at a point with slightly more capacity
                        evaluated_lp = self._evaluate_fractional(inst, strong, [min(1.0, v + 1e-6) for v in ybar], demands, solve=root_solve)
                    if evaluated_lp is None:
                        break
                    root_rounds += 1
                    if not add_cuts(*evaluated_lp) or (len(history) > 10 and history[-1] - history[-6] <= 1e-7 * max(1.0, abs(history[-1]))):
                        break
            except ReferenceLimit:
                pass  # a root solve reached the root deadline (or the shared budget, which the integer loop re-checks)
            finally:
                master.setAttr("VType", [y[i] for i in I], [GRB.BINARY] * len(I))
                master.Params.TimeLimit = GRB.INFINITY  # the runtime caps every later solve by the shared budget
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
        value = None
        if self.expected_value:
            # Exact stochastic value of the expected-value design; no stochastic lower bound.
            try:
                values, _ = self._evaluate(inst, sub, best, inst.demand)
                ub = sum(inst.fixed_cost[i] * best[i] for i in I) + sum(p * q for p, q in zip(inst.probability, values))
            except ReferenceLimit:  # no LP budget left: the verifier's solver-free value below is the published one
                ub = value = design_value(inst, best)
            lb, termination = None, "expected_value_design"
        if value is None:
            value = design_value(inst, best)  # independent recomputation of the published design value, on every exit
        if not agree(value, ub, step):
            raise RuntimeError(f"recourse LPs ({ub}) and the verifier ({value}) disagree on design {best}")
        ub = value
        kind = certificate(ub, lb, step) if lb is not None and lb > -float("inf") else None
        proved = kind is not None
        if kind == "lattice":
            lb = lattice_bound(lb, step)  # the bound's lattice value, which closes the gap
        result = {"best_objective": ub, "global_lb": lb if lb is not None and lb > -float("inf") else None, "global_ub": ub,
                  "y": best, "iters": iters, "root_rounds": root_rounds,
                  "n_cuts": master.NumConstrs - 1}
        result = self.runtime.finish(result, target_gap=0.0, termination_reason=termination)
        # The family certificate, not the shared relative-gap rule, decides the proof on every exit; the shared fields
        # are rewritten to describe it (the shared rule's own verdict is kept as runtime_rule_optimal).
        result.update(runtime_rule_optimal=result.get("proven_optimal"), status="optimal" if proved else "feasible",
                      proven_optimal=proved, certificate=kind, objective_step=step, target_gap=1e-9, target_gap_met=proved,
                      optimality_tolerance=(f"lattice step {step}" if kind == "lattice" else 1e-9))
        result["total_time"] = time.perf_counter() - started
        return result
