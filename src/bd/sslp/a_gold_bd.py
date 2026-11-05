"""Integer L-shaped reference (Laporte-Louveaux) for SSLP, plus the LP-recourse shortcut used in the screen.

Master: min c'y + sum_s p_s theta_s,  theta_s >= L_s (valid lower bound on scenario s's recourse).
At a master point y_bar, each scenario gives
  * an LP cut   theta_s >= Q_s^LP(y_bar) + sum_j u pi_j (y_j - y_bar_j)   (Q_s >= Q_s^LP, which is convex in y),
  * an integer cut theta_s >= Q_s(y_bar) - (Q_s(y_bar) - L_s) * delta(y, y_bar),
    delta = sum_{j: y_bar_j = 1} (1 - y_j) + sum_{j: y_bar_j = 0} y_j  (Hamming distance; exact at y_bar).
The shortcut (`integer_cuts=False`) is the classic L-shaped method on the LP relaxation of the recourse:
it converges on LP cuts alone and then evaluates the integer recourse exactly at the first-stage points it
visited, returning the best one. Its lower bound is only the LP-recourse bound.
"""
from __future__ import annotations

import math
import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.bd.sslp.a_gen_data import SSLPInstance
from src.bd.sslp.a_gold_monolithic import add_recourse
from src.utils.reference_runtime import ReferenceRuntime, relative_gap


class SSLPIntegerLShaped:
    def __init__(self, *, time_limit: float = 60.0, max_iters: int = 10_000, integer_cuts: bool = True, verbose: bool = False) -> None:
        self.runtime = ReferenceRuntime(time_limit)
        self.max_iters = max_iters
        self.integer_cuts = integer_cuts
        self.verbose = verbose
        self._cache: dict[tuple[tuple[int, ...], int, bool], tuple[float, Any]] = {}

    def load_instance(self, instance_name: str) -> SSLPInstance:
        return SSLPInstance.from_json_file(instance_name)

    def recourse(self, inst: SSLPInstance, s: int, y_bar: tuple[int, ...], *, binary: bool):
        """Exact recourse value of scenario s at y_bar; for the LP relaxation also the capacity duals."""
        key = (y_bar, s, binary)
        if key not in self._cache:
            model = gp.Model("sslp_recourse")
            model.Params.OutputFlag = 0
            z, _, cap, cost = add_recourse(model, inst, s, y_bar, binary=binary)
            model.setObjective(cost, GRB.MINIMIZE)
            if binary:
                model.Params.MIPGap = 0.0
                model.Params.MIPGapAbs = 1e-9
            self.runtime.optimize(model, role="recourse_mip" if binary else "recourse_lp", allow_incumbent=False)
            if model.Status != GRB.OPTIMAL:
                raise RuntimeError(f"Recourse not optimal (status={model.Status})")
            extra = [{"client": i, "site": j} for (i, j), var in z.items() if var.X > 0.5] if binary else [cap[j].Pi for j in range(inst.n_sites)]
            self._cache[key] = (float(model.ObjVal), extra)
        return self._cache[key]

    def evaluate(self, inst: SSLPInstance, y_bar: tuple[int, ...]) -> tuple[float, list[dict]]:
        total = float(sum(inst.open_cost[j] for j in range(inst.n_sites) if y_bar[j]))
        scenarios = []
        for s in range(inst.n_scenarios):
            q, rows = self.recourse(inst, s, y_bar, binary=True)
            total += inst.probability[s] * q
            scenarios.append({"scenario": s, "assignments": rows})
        return total, scenarios

    def run(self, instance_name: str) -> dict[str, Any]:
        self.runtime.start()
        self._cache.clear()
        inst = self.load_instance(instance_name)
        J, S = range(inst.n_sites), range(inst.n_scenarios)
        L = [inst.recourse_lower_bound(s) for s in S]
        master = gp.Model("sslp_master")
        master.Params.OutputFlag = 0
        master.Params.MIPGap = 0.0
        y = master.addVars(J, vtype=GRB.BINARY, name="y")
        theta = master.addVars(S, lb=L, name="theta")
        master.addConstr(y.sum() <= inst.max_open)
        master.setObjective(gp.quicksum(inst.open_cost[j] * y[j] for j in J) + gp.quicksum(inst.probability[s] * theta[s] for s in S), GRB.MINIMIZE)
        lb, ub, best, records, visited = -math.inf, math.inf, None, [], []
        termination = "max_iters"
        started = time.perf_counter()
        try:
            for it in range(1, self.max_iters + 1):
                self.runtime.optimize(master, role="master", allow_incumbent=False)
                lb = max(lb, float(master.ObjBound))
                y_bar = tuple(int(y[j].X > 0.5) for j in J)
                if y_bar not in visited:
                    visited.append(y_bar)
                added = 0
                lp_value = sum(inst.open_cost[j] for j in J if y_bar[j])
                for s in S:
                    q_lp, pi = self.recourse(inst, s, y_bar, binary=False)
                    lp_value += inst.probability[s] * q_lp
                    if q_lp > theta[s].X + 1e-6:
                        master.addConstr(theta[s] >= q_lp + gp.quicksum(inst.capacity * pi[j] * (y[j] - y_bar[j]) for j in J))
                        added += 1
                if self.integer_cuts:
                    value, scenarios = self.evaluate(inst, y_bar)
                    if value < ub:
                        ub, best = value, (y_bar, scenarios)
                    for s in S:
                        q = self.recourse(inst, s, y_bar, binary=True)[0]
                        if q > theta[s].X + 1e-6:
                            dist = gp.quicksum(1 - y[j] for j in J if y_bar[j]) + gp.quicksum(y[j] for j in J if not y_bar[j])
                            master.addConstr(theta[s] >= q - (q - L[s]) * dist)
                            added += 1
                records.append({"iteration": it, "elapsed_time": time.perf_counter() - started, "global_lb": lb, "global_ub": ub, "cuts_added": added})
                if self.verbose:
                    print(f"[sslp {it}] lb={lb:.3f} ub={ub:.3f} added={added}")
                if self.integer_cuts and ub - lb <= 1e-6 * max(1.0, abs(ub)):
                    termination = "optimal"
                    break
                if added == 0:
                    termination = "lp_converged" if not self.integer_cuts else "no_violated_cut"
                    break
        except Exception as exc:
            if "time limit" not in str(exc).lower() and "exhausted" not in str(exc).lower():
                raise
            termination = "time_limit"
        if not self.integer_cuts:
            # Shortcut: evaluate the integer recourse at the visited first-stage points and keep the best.
            for y_bar in reversed(visited):
                try:
                    value, scenarios = self.evaluate(inst, y_bar)
                except Exception:
                    break
                if value < ub:
                    ub, best = value, (y_bar, scenarios)
        if best is None:
            raise RuntimeError("SSLP stopped without a complete plan")
        return {
            "best_objective": ub, "global_lb": lb if math.isfinite(lb) else None, "global_ub": ub,
            "gap": relative_gap(lb, ub) if self.integer_cuts else None, "termination_reason": termination,
            "iters": len(records), "total_time": time.perf_counter() - started, "iter_records": records,
            "open_sites": [j for j in J if best[0][j]], "scenarios": best[1], "visited_points": len(visited),
        }
