"""Branch-and-price for nurse rostering.

Master: each nurse takes one roster (convexity row, dual mu_n); coverage rows
sum_rosters [roster works s on d] + shortage[d][s] >= demand[d][s] (duals pi[d][s] >= 0).
Pricing per nurse: exact DP over days with state (last shift, consecutive working days, days worked,
nights), reduced cost sum_d (cost[n][d][s] - pi[d][s]) - mu_n.
Branching on y[n, d, s] = sum of rosters of nurse n working s on day d: one child forces it, the other forbids it.
Costs are integers, so a node's bound is ceil(LP - eps).
"""
from __future__ import annotations

import math
import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.cg.nrp.a_gen_data import NRPInstance
from src.utils.reference_runtime import ReferenceRuntime


EPS = 1e-6


def best_roster(inst: NRPInstance, n: int, price: list[list[float]], forced: dict[int, int], banned: set[tuple[int, int]]) -> tuple[float, tuple[int, ...]] | None:
    """Minimum-price feasible roster for nurse n under branching restrictions (price[d][s] per assignment)."""
    succ = {tuple(p) for p in inst.forbidden_successions}
    states: dict[tuple[int, int, int, int], tuple[float, tuple[int, ...]]] = {(-1, 0, 0, 0): (0.0, ())}
    for d in range(inst.n_days):
        nxt: dict[tuple[int, int, int, int], tuple[float, tuple[int, ...]]] = {}
        options = [forced[d]] if d in forced else [-1, 0, 1, 2]
        for (last, run, worked, nights), (val, path) in states.items():
            for s in options:
                if (d, s) in banned or (d > 0 and (last, s) in succ):
                    continue
                if s < 0:
                    key, v = (-1, 0, worked, nights), val
                else:
                    if run + 1 > inst.max_consecutive or worked + 1 > inst.max_days[n] or (s == 2 and nights + 1 > inst.max_nights[n]):
                        continue
                    key, v = (s, run + 1, worked + 1, nights + (s == 2)), val + price[d][s]
                if key not in nxt or v < nxt[key][0] - 1e-12:
                    nxt[key] = (v, path + (s,))
        states = nxt
    finals = [(v, p) for (last, run, worked, nights), (v, p) in states.items() if worked >= inst.min_days[n]]
    return min(finals) if finals else None


def roster_cost(inst: NRPInstance, n: int, roster: tuple[int, ...]) -> float:
    return float(sum(inst.cost[n][d][s] for d, s in enumerate(roster) if s >= 0))


class NRPBranchAndPrice:
    def __init__(self, *, time_limit: float = 60.0, root_only: bool = False) -> None:
        self.runtime = ReferenceRuntime(time_limit)
        self.root_only = root_only

    def _lp(self, inst, pool, forced, banned):
        N, D = range(inst.n_nurses), range(inst.n_days)
        while True:
            cols = [(n, r) for n, rs in pool.items() for r in rs if all(r[d] == s for (m, d), s in forced.items() if m == n) and not any(r[d] == s for (m, d, s) in banned if m == n)]
            lp = gp.Model("nrp_rmp")
            lp.Params.OutputFlag = 0
            lam = lp.addVars(len(cols), lb=0.0)
            art = lp.addVars(N, lb=0.0)
            short = lp.addVars(D, range(3), lb=0.0)
            conv = [lp.addConstr(gp.quicksum(lam[k] for k, (m, _) in enumerate(cols) if m == n) + art[n] == 1) for n in N]
            cover = {(d, s): lp.addConstr(gp.quicksum(lam[k] for k, (_, r) in enumerate(cols) if r[d] == s) + short[d, s] >= inst.demand[d][s]) for d in D for s in range(3)}
            lp.setObjective(gp.quicksum(roster_cost(inst, n, r) * lam[k] for k, (n, r) in enumerate(cols)) + inst.shortage_penalty * short.sum() + 1e5 * art.sum(), GRB.MINIMIZE)
            self.runtime.optimize(lp, role="restricted_master", allow_incumbent=False)
            added = 0
            for n in N:
                self.runtime.check()
                price = [[inst.cost[n][d][s] - cover[d, s].Pi for s in range(3)] for d in D]
                f = {d: s for (m, d), s in forced.items() if m == n}
                b = {(d, s) for (m, d, s) in banned if m == n}
                res = best_roster(inst, n, price, f, b)
                if res is not None and res[0] - conv[n].Pi < -EPS and res[1] not in pool[n]:
                    pool[n].append(res[1])
                    added += 1
            if not added:
                if any(art[n].X > EPS for n in N):
                    return None, {}
                return float(lp.ObjVal), {cols[k]: lam[k].X for k in range(len(cols)) if lam[k].X > EPS}

    def _integer_master(self, inst, pool, limit):
        N, D = range(inst.n_nurses), range(inst.n_days)
        cols = [(n, r) for n, rs in pool.items() for r in rs]
        ip = gp.Model("nrp_integer_master")
        ip.Params.OutputFlag = 0
        ip.Params.Threads = 1
        ip.Params.TimeLimit = max(0.5, min(limit, self.runtime.remaining()))
        lam = ip.addVars(len(cols), vtype=GRB.BINARY)
        short = ip.addVars(D, range(3), lb=0.0)
        for n in N:
            ip.addConstr(gp.quicksum(lam[k] for k, (m, _) in enumerate(cols) if m == n) == 1)
        for d in D:
            for s in range(3):
                ip.addConstr(gp.quicksum(lam[k] for k, (_, r) in enumerate(cols) if r[d] == s) + short[d, s] >= inst.demand[d][s])
        ip.setObjective(gp.quicksum(roster_cost(inst, n, r) * lam[k] for k, (n, r) in enumerate(cols)) + inst.shortage_penalty * short.sum(), GRB.MINIMIZE)
        ip.optimize()
        if ip.SolCount == 0:
            return None, math.inf
        chosen = {n: r for k, (n, r) in enumerate(cols) if lam[k].X > 0.5}
        return [list(chosen[n]) for n in N], float(round(ip.ObjVal))

    def run(self, instance_name: str) -> dict[str, Any]:
        self.runtime.start()
        started = time.perf_counter()
        inst = NRPInstance.from_json_file(instance_name)
        N = range(inst.n_nurses)
        pool = {n: [best_roster(inst, n, [[inst.cost[n][d][s] for s in range(3)] for d in range(inst.n_days)], {}, set())[1],
                    best_roster(inst, n, [[-1.0] * 3 for _ in range(inst.n_days)], {}, set())[1]] for n in N}
        best, ub = None, math.inf
        stack, root_bound, nodes, termination = [({}, frozenset())], None, 0, "optimal"
        try:
            while stack:
                forced, banned = stack.pop()
                nodes += 1
                lp, lam = self._lp(inst, pool, forced, banned)
                if lp is None:
                    continue
                bound = math.ceil(lp - EPS)
                root_bound = bound if root_bound is None else root_bound
                if bound >= ub:
                    continue
                rosters, value = self._integer_master(inst, pool, 2.0 if not self.root_only else self.runtime.remaining())
                if rosters is not None and value < ub:
                    best, ub = rosters, value
                if self.root_only or bound >= ub:
                    continue
                y: dict[tuple[int, int, int], float] = {}
                for (n, r), v in lam.items():
                    for d, s in enumerate(r):
                        if s >= 0:
                            y[n, d, s] = y.get((n, d, s), 0.0) + v
                frac = [(abs(v - 0.5), key) for key, v in y.items() if EPS < v < 1 - EPS]
                if not frac:
                    continue  # integral LP: the integer master above already found it
                n, d, s = min(frac)[1]
                stack.append((forced, banned | {(n, d, s)}))
                stack.append(({**forced, (n, d): s}, banned))
        except Exception as exc:
            if "budget" not in str(exc).lower() and "time limit" not in str(exc).lower():
                raise
            termination = "time_limit"
        proved = (termination == "optimal" and not stack and not self.root_only) or (root_bound is not None and ub <= root_bound)
        return {
            "best_objective": ub, "global_lb": ub if proved else root_bound, "global_ub": ub, "root_bound": root_bound,
            "termination_reason": "optimal" if proved else ("root_only" if self.root_only else termination),
            "iters": nodes, "n_columns": sum(len(v) for v in pool.values()), "total_time": time.perf_counter() - started, "rosters": best,
        }
