"""Engineered column-generation reference for large GAP (scale-up study; the paper's references are unchanged).

Set-partitioning master over agent patterns (minimization):
    min sum_p c_p lam_p  s.t.  sum_{p covers i} lam_p = 1 (dual u_i),  sum_{p of agent k} lam_p <= 1 (dual v_k <= 0).
Pricing per agent is a 0-1 knapsack with integer weights, solved exactly by a vectorized DP.

- Initial incumbent: regret greedy plus shift/swap local search; its patterns seed the master.
- Wentges dual smoothing; on a mispricing step the true duals are priced.
- Lagrangian bound L(u) = sum_i u_i + sum_k min(0, min_x sum_i (c_ik - u_i) x_i) is valid for any u,
  so a valid lower bound is available at every iteration (rounded up for integer costs).
- Primal side (price-and-branch): the restricted master is solved as a MIP over all generated
  columns at geometric time checkpoints and once more with the remaining time after the root closes.
  This is a heuristic for incumbents; the bound comes only from the Lagrangian/LP side.
"""
from __future__ import annotations

import math
import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.cg.gap.a_gen_data import GAPInstance


def arrays(inst: GAPInstance) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cost (m, n), resource (m, n), capacity (m,)"""
    m, n = len(inst.machines), len(inst.jobs)
    cost, res = np.zeros((m, n)), np.zeros((m, n), dtype=np.int64)
    for job in inst.jobs:
        for a in job.assignments:
            cost[a.machine_id, job.id], res[a.machine_id, job.id] = a.cost, a.resource
    cap = np.array([mc.capacity for mc in sorted(inst.machines, key=lambda x: x.id)], dtype=np.int64)
    return cost, res, cap


def knapsack(profit: np.ndarray, weight: np.ndarray, cap: int) -> tuple[float, np.ndarray]:
    """Exact 0-1 knapsack max sum profit x, sum weight x <= cap, over items with positive profit."""
    # No profit tolerance: dropping slightly positive items would understate the knapsack value and
    # could overstate a Lagrangian lower bound.
    items = np.nonzero((profit > 0) & (weight <= cap))[0]
    dp = np.zeros(cap + 1)
    take = np.zeros((len(items), cap + 1), dtype=bool)
    for t, i in enumerate(items):
        w, p = int(weight[i]), profit[i]
        cand = dp[:cap + 1 - w] + p  # a copy, so the update below reads the previous layer
        better = cand > dp[w:]
        take[t, w:] = better
        np.copyto(dp[w:], cand, where=better)
    x = np.zeros(len(profit), dtype=bool)
    c = int(np.argmax(dp))
    for t in range(len(items) - 1, -1, -1):
        if take[t, c]:
            x[items[t]] = True
            c -= int(weight[items[t]])
    return float(dp.max()), x


def certified_bound(lag: float, u: np.ndarray, integral: bool) -> float:
    """Lagrangian value turned into a reported lower bound, with an allowance for floating-point error
    in the n-term sums (relative 1e-9 per term) before rounding up for integer costs."""
    allowance = 1e-6 + 1e-9 * len(u) * max(1.0, float(np.abs(u).max()))
    return math.ceil(lag - allowance) if integral else lag - allowance


def evaluate(assign: np.ndarray, cost: np.ndarray, res: np.ndarray, cap: np.ndarray) -> float | None:
    n = len(assign)
    load = np.bincount(assign, weights=res[assign, np.arange(n)], minlength=len(cap))
    return float(cost[assign, np.arange(n)].sum()) if np.all(load <= cap) else None


def greedy(cost: np.ndarray, res: np.ndarray, cap: np.ndarray) -> np.ndarray | None:
    """Regret greedy on a cost/resource desirability, then shift and swap descent."""
    m, n = cost.shape
    left, assign = cap.astype(float).copy(), np.full(n, -1)
    score = cost + 0.5 * res * (cost.mean() / max(1.0, res.mean()))
    todo = set(range(n))
    while todo:
        best_i, best_k, best_regret = -1, -1, -math.inf
        for i in todo:
            feas = np.nonzero(res[:, i] <= left)[0]
            if not len(feas):
                return None
            s = np.sort(score[feas, i])
            regret = (s[1] - s[0]) if len(s) > 1 else math.inf
            if regret > best_regret:
                best_i, best_k, best_regret = i, int(feas[np.argmin(score[feas, i])]), regret
        assign[best_i] = best_k
        left[best_k] -= res[best_k, best_i]
        todo.remove(best_i)
    return local_search(assign, cost, res, cap)


def local_search(assign: np.ndarray, cost: np.ndarray, res: np.ndarray, cap: np.ndarray, rounds: int = 50) -> np.ndarray:
    m, n = cost.shape
    idx = np.arange(n)
    load = np.bincount(assign, weights=res[assign, idx], minlength=m)
    for _ in range(rounds):
        improved = False
        # shifts: move job i to agent k if it fits and is cheaper
        gain = cost[assign, idx][None, :] - cost
        fits = res + load[:, None] <= cap[:, None]
        gain[~fits] = -np.inf
        for i in np.argsort(-gain.max(axis=0)):
            k = int(np.argmax(gain[:, i]))
            if gain[k, i] <= 1e-9 or res[k, i] + load[k] > cap[k]:
                continue
            load[assign[i]] -= res[assign[i], i]
            load[k] += res[k, i]
            assign[i] = k
            improved = True
        if not improved:
            break
    return assign


def repair(xs: list[np.ndarray], cost: np.ndarray, res: np.ndarray, cap: np.ndarray) -> np.ndarray | None:
    """Turn per-agent knapsack solutions into an assignment: keep each job on its cheapest selecting agent, drop
    the rest, then insert uncovered jobs by regret where they fit."""
    m, n = cost.shape
    sel = np.stack(xs)
    masked = np.where(sel, cost, np.inf)
    assign = np.where(sel.any(axis=0), masked.argmin(axis=0), -1)
    load = np.zeros(m)
    for i in np.nonzero(assign >= 0)[0]:
        load[assign[i]] += res[assign[i], i]
    todo = list(np.nonzero(assign < 0)[0])
    while todo:
        best = None
        for i in todo:
            feas = np.nonzero(res[:, i] + load <= cap)[0]
            if not len(feas):
                return None
            c = np.sort(cost[feas, i])
            regret = c[1] - c[0] if len(c) > 1 else math.inf
            if best is None or regret > best[0]:
                best = (regret, i, int(feas[np.argmin(cost[feas, i])]))
        _, i, k = best
        assign[i] = k
        load[k] += res[k, i]
        todo.remove(i)
    return assign


def subgradient(cost: np.ndarray, res: np.ndarray, cap: np.ndarray, deadline: float, iters: int, on_point) -> tuple[np.ndarray, float]:
    """Polyak subgradient ascent on the Lagrangian dual of the assignment constraints.

    Returns the best multipliers and bound. Every point's knapsack solutions are passed to on_point(u, xs)
    so the caller can keep columns and repair incumbents.
    """
    m, n = cost.shape
    u = np.sort(cost, axis=0)[1].astype(float) if m > 1 else cost[0].astype(float)
    best_u, best_l, step, stall = u.copy(), -math.inf, 2.0, 0
    target = None
    for _ in range(iters):
        if time.perf_counter() >= deadline:
            break
        xs, lag = [], u.sum()
        for k in range(m):
            val, x = knapsack(u - cost[k], res[k], int(cap[k]))
            lag -= val
            xs.append(x)
        upper = on_point(u, xs)
        if lag > best_l + 1e-9:
            best_l, best_u, stall = lag, u.copy(), 0
        else:
            stall += 1
            if stall >= 20:
                step, stall = step / 2, 0
        g = 1.0 - np.stack(xs).sum(axis=0)
        if not g.any():
            break
        target = upper if upper is not None else best_l + 0.05 * abs(best_l) + 1.0
        u = u + step * (target - lag) / float(g @ g) * g
    return best_u, best_l


class GAPColumnGenerationXL:
    """Engineered CG / price-and-branch reference; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, smoothing: float = 0.5, verbose: bool = False, on_incumbent=None, on_bound=None, lp_method: int = 0,
                 soft_mem_gb: float | None = None) -> None:
        self.soft_mem_gb = soft_mem_gb  # Gurobi SoftMemLimit for graceful termination before the job's memory cap
        self.lp_method = lp_method  # Gurobi Method for the restricted master LP
        self.on_incumbent = on_incumbent or (lambda assign: None)  # receives the job -> agent list
        self.on_bound = on_bound or (lambda bound: None)  # receives every valid lower bound
        self.time_limit = time_limit
        self.threads = threads
        self.smoothing = smoothing
        self.verbose = verbose
        self.trace: list[dict[str, Any]] = []

    def _log(self, t0: float, inc: float | None, bnd: float | None, stage: str) -> None:
        if bnd is not None and math.isfinite(bnd):
            self.on_bound(bnd)
        self.trace.append({"t": time.perf_counter() - t0, "incumbent": inc, "bound": bnd, "stage": stage})

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        cost, res, cap = arrays(GAPInstance.from_json_file(data_path))
        m, n = cost.shape
        integral = bool(np.all(cost == np.round(cost)))

        best_assign = greedy(cost, res, cap)
        best_obj = evaluate(best_assign, cost, res, cap) if best_assign is not None else None
        if best_obj is not None:
            self.on_incumbent(best_assign.tolist())
        self._log(t0, best_obj, None, "greedy")

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 1 if self.verbose else 0)
        env.setParam("Threads", self.threads)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()
        rmp = gp.Model(env=env)
        rmp.Params.Method = self.lp_method
        # Artificial cost: large enough to be driven out at the LP optimum of GAP-type instances, small enough
        # to keep early duals informative. Lower bounds never depend on it (they come from Lagrangian pricing).
        big = 10.0 * float(cost.max()) + 1.0
        art = [rmp.addVar(obj=big, name=f"art{i}") for i in range(n)]  # keeps the master feasible
        # Covering rows: with nonnegative costs a job can always be dropped from a pattern, so the covering LP
        # has the partitioning LP value, and its duals (u >= 0) are less degenerate.
        cover = [rmp.addLConstr(gp.LinExpr([(1.0, art[i])]), GRB.GREATER_EQUAL, 1.0) for i in range(n)]
        conv = [rmp.addLConstr(gp.LinExpr(), GRB.LESS_EQUAL, 1.0) for _ in range(m)]
        columns: list[tuple[int, np.ndarray]] = []
        seen: set[tuple[int, bytes]] = set()
        lam: list[gp.Var] = []

        def add_column(k: int, x: np.ndarray) -> bool:
            key = (k, np.packbits(x).tobytes())
            if key in seen or not x.any():
                return False
            seen.add(key)
            jobs = np.nonzero(x)[0]
            col = gp.Column([1.0] * len(jobs) + [1.0], [cover[int(i)] for i in jobs] + [conv[k]])
            lam.append(rmp.addVar(obj=float(cost[k, jobs].sum()), column=col))
            columns.append((k, x.copy()))
            return True

        if best_assign is not None:
            for k in range(m):
                add_column(k, best_assign == k)

        self.state = {"bound": -math.inf, "center": None, "iters": 0, "lp_seconds": 0.0, "pricing_seconds": 0.0}

        def column_generation(blocked_agents: set[int], covered: np.ndarray, root: bool, max_iters: int) -> bool:
            """Run CG at the current node; returns True only if the master LP is solved for the current columns.

            At the root, every pricing round also yields a Lagrangian lower bound (smoothed duals).
            Inside a dive, blocked agents are not priced and covered jobs are excluded.
            """
            for _ in range(max_iters):
                if time.perf_counter() >= deadline:
                    return False
                self.state["iters"] += 1
                rmp.Params.TimeLimit = max(0.01, deadline - time.perf_counter())
                lp_started = time.perf_counter()
                rmp.optimize()
                self.state["lp_seconds"] += time.perf_counter() - lp_started
                if rmp.Status != GRB.OPTIMAL:
                    return False
                pricing_started = time.perf_counter()
                u_true = np.array(rmp.getAttr("Pi", cover))
                v_true = np.array(rmp.getAttr("Pi", conv))
                center = self.state["center"] if root else None
                for u in ([self.smoothing * center + (1 - self.smoothing) * u_true] if center is not None else []) + [u_true]:
                    lag, added = u.sum(), 0
                    for k in range(m):
                        if k in blocked_agents:
                            continue
                        profit = u - cost[k]
                        profit[covered] = -1.0
                        val, x = knapsack(profit, res[k], int(cap[k]))
                        lag -= val  # min_x sum (c - u) x = -max_x sum (u - c) x
                        rc = float((cost[k] - u_true)[x].sum()) - v_true[k]
                        if rc < -1e-7 and add_column(k, x):
                            added += 1
                    if root:
                        bound = certified_bound(lag, u, integral)
                        if bound > self.state["bound"]:
                            self.state["bound"], self.state["center"] = bound, u.copy()
                    if added:
                        break
                self.state["pricing_seconds"] += time.perf_counter() - pricing_started
                if root:
                    self._log(t0, best["obj"], self.state["bound"], "cg_root")
                if not added:
                    return True
            # Iteration limit: the last round added columns, so re-solve before exposing solution values.
            if time.perf_counter() >= deadline:
                return False
            rmp.Params.TimeLimit = max(0.01, deadline - time.perf_counter())
            rmp.optimize()
            return rmp.Status == GRB.OPTIMAL

        best = {"obj": best_obj, "assign": best_assign}

        def try_assignment(assign: np.ndarray, stage: str) -> None:
            assign = local_search(assign, cost, res, cap)
            obj = evaluate(assign, cost, res, cap)  # recomputed from the assignment
            if obj is not None and (best["obj"] is None or obj < best["obj"] - 1e-9):
                best["obj"], best["assign"] = obj, assign
                self.on_incumbent(assign.tolist())
                self._log(t0, obj, self.state["bound"], stage)

        rng = np.random.default_rng(0)

        def dive(randomize: bool) -> None:
            """Fix one master column to 1 per node, removing its agent and jobs, and re-run CG."""
            blocked, covered, fixed = set(), np.zeros(n, dtype=bool), []
            changed = []
            try:
                dive_steps(blocked, covered, fixed, changed, randomize)
            finally:
                for v in changed:
                    v.LB, v.UB = 0.0, 1.0
                rmp.update()

        def dive_steps(blocked, covered, fixed, changed, randomize):
            while time.perf_counter() < deadline:
                if not column_generation(blocked, covered, False, 200):
                    break
                values = np.array([v.X for v in lam])
                if np.array(rmp.getAttr("X", art)).max() > 1e-6:
                    break  # infeasible node: the dive failed
                frac = [(val, j) for j, val in enumerate(values) if val > 1e-6 and columns[j][0] not in blocked]
                if not frac:
                    break
                if all(val > 1 - 1e-6 for val, _ in frac):
                    assign = np.full(n, -1)
                    for j, (k, x) in enumerate(columns):
                        if values[j] > 0.5:
                            assign[x] = k
                    if np.all(assign >= 0):
                        try_assignment(assign, "dive")
                    break
                frac.sort(reverse=True)
                pick = frac[int(rng.integers(0, min(3, len(frac))))][1] if randomize else frac[0][1]
                k, x = columns[pick]
                lam[pick].LB = 1.0
                changed.append(lam[pick])
                blocked.add(k)
                covered |= x
                fixed.append(pick)
                # Columns that overlap the fixed jobs, or belong to the fixed agent, are excluded.
                for j, (k2, x2) in enumerate(columns):
                    if j != pick and lam[j].UB > 0 and (k2 == k or (x2 & x).any()):
                        lam[j].UB = 0.0
                        changed.append(lam[j])

        # Warm start: subgradient ascent on the Lagrangian dual. It gives the CG stability center and a
        # bound, its knapsack solutions become master columns, and their repairs become incumbents.
        def on_point(u, xs):
            for k, x in enumerate(xs):
                add_column(k, x)
            assign = repair(xs, cost, res, cap)
            if assign is not None:
                try_assignment(assign, "subgradient")
                for k in range(m):
                    add_column(k, best["assign"] == k)
            return best["obj"]

        u_best, l_best = subgradient(cost, res, cap, t0 + 0.1 * self.time_limit, 300, on_point)
        self.state["center"] = u_best
        self.state["bound"] = certified_bound(l_best, u_best, integral)
        self._log(t0, best["obj"], self.state["bound"], "subgradient")
        warm_seconds = time.perf_counter() - t0
        root_done = column_generation(set(), np.zeros(n, dtype=bool), True, 10**9)
        root_seconds = time.perf_counter() - t0
        dives = 0
        mip_limit = 30.0
        while time.perf_counter() < deadline:
            if best["obj"] is not None and integral and best["obj"] - self.state["bound"] < 1 - 1e-9:
                break
            dive(randomize=dives > 0)
            dives += 1
            best["obj"], best["assign"] = self._restricted_mip(rmp, columns, n, cost, res, cap, best["obj"], best["assign"], min(mip_limit, deadline - time.perf_counter()), t0, self.state["bound"], integral)
            mip_limit = min(600.0, 2 * mip_limit)
        env.dispose()
        best_obj, best_assign, best_bound = best["obj"], best["assign"], self.state["bound"]
        gap = (best_obj - best_bound) / abs(best_obj) if best_obj else None
        self._log(t0, best_obj, best_bound, "final")
        return {
            "best_objective": best_obj, "global_lb": best_bound, "global_ub": best_obj, "gap": gap,
            "termination_reason": "optimal" if gap is not None and gap <= 1e-9 else "time_limit",
            "root_converged": root_done, "warm_start_seconds": warm_seconds, "root_seconds": root_seconds, "dives": dives,
            "iters": self.state["iters"], "columns": len(columns),
            "lp_seconds": self.state["lp_seconds"], "pricing_seconds": self.state["pricing_seconds"],
            "assignment": best_assign.tolist() if best_assign is not None else None,
            "total_seconds": time.perf_counter() - t0, "trace": self.trace,
        }

    def _restricted_mip(self, rmp, columns, n, cost, res, cap, best_obj, best_assign, limit, t0, bound, integral):
        if limit < 1.0:
            return best_obj, best_assign
        rmp.update()  # copy() sees only applied changes, e.g. bounds restored after a dive
        mip = rmp.copy()
        mvars = mip.getVars()
        for v in mvars[n:]:
            v.VType = GRB.BINARY
        for v in mvars[:n]:
            v.UB = 0.0
        if best_obj is not None:
            mip.Params.Cutoff = best_obj - (0.5 if integral else 1e-6)
        mip.Params.TimeLimit = limit
        cols = mvars[n:]

        def publish(cb, where):
            # Publish every restricted-master solution when found, not only after the solve.
            if where == GRB.Callback.MIPSOL:
                vals = cb.cbGetSolution(cols)
                assign = np.full(n, -1)
                for v, (k, x) in zip(vals, columns):
                    if v > 0.5:
                        assign[x] = k
                if np.all(assign >= 0):
                    self.on_incumbent(assign.tolist())

        mip.optimize(publish)
        if mip.SolCount:
            assign = np.full(n, -1)
            for v, (k, x) in zip(mvars[n:], columns):
                if v.X > 0.5:
                    assign[x] = k
            if np.all(assign >= 0):
                assign = local_search(assign, cost, res, cap)
                obj = evaluate(assign, cost, res, cap)  # recomputed from the assignment
                if obj is not None and (best_obj is None or obj < best_obj):
                    best_obj, best_assign = obj, assign
                    self.on_incumbent(assign.tolist())
                    self._log(t0, best_obj, bound, "restricted_mip")
        mip.dispose()
        return best_obj, best_assign
