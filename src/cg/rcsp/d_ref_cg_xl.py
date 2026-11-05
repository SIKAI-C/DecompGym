"""Engineered column-generation reference for the RCSP (scale-up study).

Set-covering master over duties j (cost c_j = duty time + 8 h):  min sum_j c_j lam_j  s.t.  sum_{j covers t} lam_j >= 1.
Pricing (pricing.DutyPricer): labelling on the acyclic task-connection graph with the resources of van Rossum (2026,
Appendix A) - reduced cost, duty length, time without break, break flag - exact dominance, a reduced-cost completion
bound recomputed per round and static duty-time / time-to-break bounds; all crew bases in one pass.

Lower bounds. For any duals pi >= 0 and an exact pricing round with minimum reduced cost r* (over all duties of all
crew bases), L(pi) = sum_t pi_t + kappa * min(0, r* - tol) is a valid bound (Lagrangian relaxation of the covering
rows; an optimal LP solution uses at most kappa = UB / (cheapest duty) duties). Every exact round publishes it (rounded
up: costs are integral); at convergence it equals the root LP value.

Column generation: heuristic pricing first (a beam of the cheapest labels per task and crew base, several hundred
candidate duties per round, of which a diverse subset is added: a duty is taken only if at least half of its tasks are
not yet covered by the duties taken earlier in the round); an exact round when the beam finds nothing or every
`exact_every` rounds; Wentges dual smoothing around the duals of the best Lagrangian bound (mispricing falls back to
the master duals); periodic removal of columns that stayed nonbasic with large reduced cost.

Primal side (every plan is checked with plan_cost, redundant duties are dropped, and each improvement is published):
the shared greedy initializer; restricted-master MIPs over the column pool (short ones during the root, longer ones
later, with the incumbent as MIP start); diving (van Rossum's main primal driver): fix the duties at >= 0.95 in the
master LP (else the largest fractional one) to 1, re-run up to 8 rounds of heuristic column generation (beam of 20
labels; dual simplex), repeat until the master LP is integral, abandoning a dive whose LP value reaches the incumbent
(one cut by its time budget is completed greedily from the pool); after the root, dives alternate with
large-neighbourhood dives that first fix a random quarter of the incumbent's duties; reduced-cost duty enumeration
(exact set-covering MIP over all duties that can improve the incumbent) once the gap is below `enum_gap`.
No branching: unless the enumeration closes the gap, the published bound is the root (Lagrangian) bound.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.cg.rcsp.a_data import RCSPInstance, load_rcsp
from src.cg.rcsp.b_mip_xl import duty_check, plan_cost
from src.cg.rcsp.pricing import DutyPricer, greedy_cover

INF = float("inf")


def drop_redundant(duties: list[dict[str, Any]], costs: list[float], n: int) -> list[dict[str, Any]]:
    """Remove duties whose tasks are all covered by the other duties (most expensive first)."""
    cnt = np.zeros(n, dtype=np.int64)
    for d in duties:
        np.add.at(cnt, d["tasks"], 1)
    keep = [True] * len(duties)
    for k in sorted(range(len(duties)), key=lambda k: -costs[k]):
        t = duties[k]["tasks"]
        if np.all(cnt[t] >= 2):
            keep[k] = False
            np.add.at(cnt, t, -1)
    return [d for d, k in zip(duties, keep) if k]


class RCSPColumnGenerationXL:
    """Engineered column generation (price-and-dive) reference for the RCSP; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0, beam: int = 6,
                 max_cols: int | None = None, alpha: float = 0.5, exact_every: int = 0, tol: float = 1e-4,
                 barrier: bool = True, rel: float = 0.5, enum_gap: float = 5000.0, enum_cols: int = 200_000,
                 dive_fix: float = 0.95, dive_rounds: int = 8, dive_beam: int | None = 20, dive_exact: bool = False,
                 verbose: bool = False) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.on_incumbent = on_incumbent or (lambda duties: None)  # receives [{"depot", "tasks"}] plans
        self.on_bound = on_bound or (lambda bound: None)  # receives every new valid global lower bound
        self.beam, self.max_cols, self.alpha, self.exact_every, self.tol = beam, max_cols, alpha, exact_every, tol
        self.rel = rel  # pricing prunes labels that cannot reach rel * (best reduced cost found so far)
        self.enum_gap, self.enum_cols = enum_gap, enum_cols  # reduced-cost duty enumeration trigger and size cap
        self.dive_fix, self.dive_rounds, self.dive_beam, self.dive_exact = dive_fix, dive_rounds, dive_beam, dive_exact
        self.barrier = barrier  # root master LP by barrier without crossover (central duals), else primal simplex
        self.verbose = verbose
        self.trace: list[dict[str, Any]] = []

    # ------------------------------------------------------------ setup

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        return self.solve(load_rcsp(data_path))

    def left(self) -> float:
        return self.deadline - time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def log(self, stage: str, **kw) -> None:
        self.trace.append({"t": round(self.elapsed(), 3), "ub": self.ub if self.ub < INF else None,
                           "lb": self.lb if self.lb > -INF else None, "stage": stage, **kw})
        if self.verbose:
            print(self.trace[-1], flush=True)

    def solve(self, inst: RCSPInstance) -> dict[str, Any]:
        if not hasattr(self, "t0"):
            self.t0 = time.perf_counter()
            self.deadline = self.t0 + self.time_limit
        self.inst = inst
        self.n = inst.n
        self.rng = np.random.default_rng(self.seed)
        self.ub, self.lb = INF, -INF
        self.best: list[dict[str, Any]] | None = None
        self.best_pi: np.ndarray | None = None
        self.best_lval = -INF
        self.enum_tried_ub = INF
        self.center, self.center_val = None, -INF
        self.diving = False
        self.stats = {"cg_rounds": 0, "exact_rounds": 0, "heuristic_rounds": 0, "lp_seconds": 0.0, "dives": 0,
                      "lns_dives": 0, "mip_runs": 0, "mip_seconds": 0.0, "removed_columns": 0, "root_converged": False,
                      "root_lp": None, "root_seconds": None, "lp_solves": 0,
                      "enumerations": 0, "enumerated_duties": None, "enumeration_proof": False}
        self.pricer = DutyPricer(inst)
        self.cmin = float(np.min(self.pricer.src_cost[np.isfinite(self.pricer.src_cost)]))
        self.K = self.max_cols or int(min(400, max(40, inst.n // 4)))
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", self.threads)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()
        self.env = env
        reason = "time_limit"
        try:
            init = greedy_cover(self.pricer, self.deadline)
            if init is None:
                reason = "infeasible" if self.left() > 0 else "time_limit"
            else:
                self.offer(init, "greedy")
                self.build_master(init)
                self.root()
                if not self.closed():
                    self.improve()
                if self.closed():
                    reason = "optimal"
        except gp.GurobiError as err:
            if err.errno != GRB.Error.OUT_OF_MEMORY and "memory" not in str(err).lower():
                raise
            reason = "memory_limit"
        out = self.result(reason)
        env.dispose()
        return out

    def result(self, reason: str) -> dict[str, Any]:
        ub = self.ub if self.ub < INF else None
        lb = self.lb if self.lb > -INF else None
        return {"best_objective": ub, "global_lb": lb, "global_ub": ub,
                "gap": (ub - lb) / abs(ub) if ub is not None and lb is not None and ub else None,
                "termination_reason": reason, "duties": self.best, "columns": len(getattr(self, "keys", [])),
                **self.stats, "pricing_seconds": self.pricer.stats["seconds"], "total_seconds": self.elapsed(),
                "trace": self.trace}

    def closed(self) -> bool:
        return self.ub < INF and self.lb >= self.ub - 1.0 + 1e-6

    # ------------------------------------------------------------ incumbents and bounds

    def offer(self, duties: list[dict[str, Any]], stage: str) -> bool:
        costs = []
        for d in duties:
            c, _ = duty_check(self.inst, d["depot"], d["tasks"])
            costs.append(c if c is not None else INF)
        if any(c == INF for c in costs):
            return False
        duties = drop_redundant(duties, costs, self.n)
        obj = plan_cost(self.inst, duties)
        if obj is None or obj >= self.ub - 1e-9:
            return False
        self.ub, self.best = obj, duties
        self.on_incumbent(duties)
        self.log(stage)
        if hasattr(self, "master"):
            for d in duties:
                self.add_column(d["depot"], d["tasks"])
        return True

    def lagrangian(self, pi: np.ndarray, min_rc: float, stage: str) -> float:
        kappa = self.ub / self.cmin
        value = float(pi.sum()) + kappa * (min(0.0, min_rc) - self.tol)
        if value > self.best_lval:
            self.best_lval, self.best_pi = value, pi.copy()
        bound = float(math.ceil(value - 1e-7 * max(1.0, abs(value))))
        if self.ub < INF:
            bound = min(bound, self.ub)
        if bound > self.lb:
            self.lb = bound
            self.on_bound(bound)
            self.log(stage)
        return value

    # ------------------------------------------------------------ master

    def build_master(self, init: list[dict[str, Any]]) -> None:
        m = gp.Model(env=self.env)
        self.root_method(m)
        self.master = m
        self.cover = [m.addLConstr(gp.LinExpr(), GRB.GREATER_EQUAL, 1.0) for _ in range(self.n)]
        self.vars: list[gp.Var] = []
        self.keys: list[tuple[int, tuple[int, ...]]] = []
        self.cost: list[float] = []
        self.age: list[int] = []
        self.index: dict[tuple[int, tuple[int, ...]], int] = {}
        for d in init:
            self.add_column(d["depot"], d["tasks"])

    def root_method(self, m: gp.Model) -> None:
        if self.barrier:
            m.Params.Method, m.Params.Crossover, m.Params.BarConvTol = 2, 0, 1e-6
        else:
            m.Params.Method, m.Params.Crossover = 0, -1

    def add_column(self, depot: int, tasks: list[int] | tuple[int, ...]) -> bool:
        key = (int(depot), tuple(int(t) for t in tasks))
        if key in self.index:
            return False
        cost, _ = duty_check(self.inst, key[0], list(key[1]))
        if cost is None:
            return False
        v = self.master.addVar(lb=0.0, obj=cost, column=gp.Column([1.0] * len(key[1]), [self.cover[t] for t in key[1]]))
        self.index[key] = len(self.vars)
        self.vars.append(v)
        self.keys.append(key)
        self.cost.append(cost)
        self.age.append(0)
        return True

    def solve_lp(self) -> bool:
        t = time.perf_counter()
        self.master.Params.TimeLimit = max(0.01, self.left())
        self.master.optimize()
        self.stats["lp_seconds"] += time.perf_counter() - t
        self.stats["lp_solves"] += 1
        return self.master.Status == GRB.OPTIMAL

    def duals(self) -> np.ndarray:
        return np.maximum(np.array(self.master.getAttr("Pi", self.cover)), 0.0)

    def select(self, cols: list[tuple[float, int, list[int]]], limit: int) -> list[tuple[int, list[int]]]:
        """Diverse subset: a duty is taken if at least half of its tasks are not covered by the duties taken before."""
        used = np.zeros(self.n, dtype=bool)
        out = []
        for rc, d, tasks in cols:
            if (d, tuple(tasks)) in self.index:
                continue
            if (~used[tasks]).sum() * 2 >= len(tasks):
                used[tasks] = True
                out.append((d, tasks))
                if len(out) >= limit:
                    break
        return out

    def cleanup(self) -> None:
        """Drop columns nonbasic for 40 rounds with reduced cost above 1% of the cheapest duty, when the pool is big."""
        if len(self.vars) < 30 * self.n or self.diving:
            return  # (never during a dive: it holds column indices)
        x = self.master.getAttr("X", self.vars)
        rc = self.master.getAttr("RC", self.vars)
        lbs = self.master.getAttr("LB", self.vars)
        incumbent = {(d["depot"], tuple(d["tasks"])) for d in (self.best or [])}
        drop = [k for k in range(len(self.vars)) if self.age[k] > 40 and x[k] < 1e-9 and rc[k] > 0.01 * self.cmin
                and lbs[k] == 0.0 and self.keys[k] not in incumbent]
        if len(drop) < len(self.vars) // 5:
            return
        dropset = set(drop)
        self.master.remove([self.vars[k] for k in drop])
        keep = [k for k in range(len(self.vars)) if k not in dropset]
        self.vars = [self.vars[k] for k in keep]
        self.keys = [self.keys[k] for k in keep]
        self.cost = [self.cost[k] for k in keep]
        self.age = [self.age[k] for k in keep]
        self.index = {key: k for k, key in enumerate(self.keys)}
        self.stats["removed_columns"] += len(drop)

    def update_ages(self) -> None:
        x = self.master.getAttr("X", self.vars)
        for k, v in enumerate(x):
            self.age[k] = 0 if v > 1e-9 else self.age[k] + 1

    # ------------------------------------------------------------ column generation

    def column_generation(self, stage: str, until: float, stabilize: bool = True, exact_allowed: bool = True,
                          max_rounds: int | None = None) -> bool:
        """Column generation on the current master (with any fixings); True when an exact round at the master duals
        proved that no negative duty remains."""
        # The stability center (duals of the best Lagrangian value) persists across calls on the unfixed master.
        center, best_val = (self.center, self.center_val) if stabilize else (None, -INF)
        rounds, since_exact = 0, 0
        force_rmp = False
        while time.perf_counter() < until and self.left() > 0:
            if not self.solve_lp():
                return False
            rounds += 1
            self.stats["cg_rounds"] += 1
            pi_rmp = self.duals()
            self.update_ages()
            if rounds % 10 == 0:
                self.cleanup()
            use_center = stabilize and center is not None and not force_rmp
            pi = self.alpha * center + (1.0 - self.alpha) * pi_rmp if use_center else pi_rmp
            force_rmp = False
            exact = exact_allowed and since_exact >= self.exact_every
            res = None
            if not exact:
                self.stats["heuristic_rounds"] += 1
                res = self.pricer.price(pi, exact=False, beam=self.beam, max_cols=3 * self.K, deadline=until, tol=self.tol,
                                        rel=self.rel)
                since_exact += 1
                if not res["cols"]:
                    if not exact_allowed:
                        if use_center:
                            force_rmp = True
                            continue
                        return False
                    exact = True
            if exact:
                self.stats["exact_rounds"] += 1
                since_exact = 0
                res = self.pricer.price(pi, exact=True, max_cols=3 * self.K, deadline=until, tol=self.tol, rel=self.rel)
                if res["complete"]:
                    val = self.lagrangian(pi, float(res["min_rc"].min()), stage)
                    if val > best_val:
                        best_val, center = val, pi.copy()
                        if stabilize:
                            self.center, self.center_val = center, best_val
                if not res["cols"]:
                    if not res["complete"]:
                        return False
                    if use_center:
                        force_rmp = True  # mispricing: price at the master duals next
                        continue
                    return True
            added = sum(self.add_column(d, t) for d, t in self.select(res["cols"], self.K))
            if added == 0:
                # Every proposed duty is already in the master (floating-point tolerances).
                if use_center:
                    force_rmp = True
                elif exact:
                    return bool(res["complete"])
                elif exact_allowed:
                    since_exact = self.exact_every
                else:
                    return False
            if max_rounds is not None and rounds >= max_rounds:
                return False
        return False

    def root(self) -> None:
        t = time.perf_counter()
        root_until = self.t0 + 0.6 * self.time_limit
        # Short restricted-master MIPs during the root (every max(10 s, 5% of T), each min(max(2 s, 1% of T), 20 s))
        # give early incumbents without starving column generation.
        gap = max(10.0, 0.05 * self.time_limit)
        next_mip = time.perf_counter() + gap
        converged = False
        while time.perf_counter() < root_until and self.left() > 0:
            slice_end = min(root_until, next_mip)
            converged = self.column_generation("root", slice_end)
            if converged or time.perf_counter() >= root_until:
                break
            self.restricted_mip(min(max(2.0, 0.01 * self.time_limit), 20.0), "root_mip")
            next_mip = time.perf_counter() + gap
        self.stats["root_converged"] = converged
        self.stats["root_seconds"] = time.perf_counter() - t
        if self.left() > 0 and self.solve_lp():
            self.stats["root_lp"] = float(self.master.ObjVal)
        self.log("root_done", converged=converged)

    def raise_bound(self, bound: float, stage: str) -> None:
        bound = min(bound, self.ub)
        if bound > self.lb:
            self.lb = bound
            self.on_bound(bound)
            self.log(stage)

    def try_enumeration(self, stage: str) -> None:
        """Reduced-cost duty enumeration (van Rossum 2026 uses it when the gap is small): with duals pi of Lagrangian
        value Lval, every plan of cost <= UB - 1 uses only duties of reduced cost <= UB - 1 - Lval (all reduced costs
        are >= the exact minimum minus tol, and a plan has at most kappa duties). If all of them can be listed, the
        set-covering MIP over them is exact: its optimum is the optimum (or the incumbent is optimal), and its bound,
        capped by the incumbent, is a valid global bound."""
        if self.best_pi is None or self.ub == INF or self.closed() or self.ub >= self.enum_tried_ub:
            return
        thr = self.ub - 1.0 - self.best_lval + 1e-6
        # Small gap: up to enum_cols duties; tiny instances (<= 100 tasks): a quick check that the duty set is small.
        if thr > self.enum_gap and self.n > 100:
            return
        cap, labels = (self.enum_cols, 4_000_000) if thr <= self.enum_gap else (5_000, 200_000)
        self.enum_tried_ub = self.ub
        self.stats["enumerations"] += 1
        res = self.pricer.price(self.best_pi, enumerate_all=True, threshold=thr, max_cols=cap + 1,
                                max_labels=labels, deadline=self.deadline)
        if not res["complete"] or len(res["cols"]) > cap:
            self.log(stage + "_too_large", labels=res["labels"])
            return
        self.stats["enumerated_duties"] = len(res["cols"])
        m = gp.Model(env=self.env)
        cover = [gp.LinExpr() for _ in range(self.n)]
        duties, xs = [], []
        for _, d, tasks in res["cols"]:
            cost, _ = duty_check(self.inst, d, tasks)
            if cost is None:
                continue
            v = m.addVar(vtype=GRB.BINARY, obj=cost)
            for t in tasks:
                cover[t].addTerms(1.0, v)
            duties.append({"depot": int(d), "tasks": [int(t) for t in tasks]})
            xs.append(v)
        for t in range(self.n):
            m.addLConstr(cover[t], GRB.GREATER_EQUAL, 1.0)
        m.Params.Cutoff = self.ub - 0.5
        m.Params.TimeLimit = max(0.01, self.left())
        # Costs are integral: an absolute gap below one unit (and no relative tolerance) makes OPTIMAL a real proof.
        m.Params.MIPGap = 0.0
        m.Params.MIPGapAbs = 0.5
        m.optimize()
        if m.SolCount:
            x = m.getAttr("X", xs)
            self.offer([d for d, v in zip(duties, x) if v > 0.5], stage)
        if m.Status in (GRB.INFEASIBLE, GRB.CUTOFF, GRB.INF_OR_UNBD):
            # No plan below UB - 0.5 among the enumerated duties (all duties that can improve on UB): UB is optimal.
            self.stats["enumeration_proof"] = True
            self.raise_bound(self.ub, stage)
        elif m.Status == GRB.OPTIMAL:
            self.stats["enumeration_proof"] = True
            self.raise_bound(min(self.ub, float(math.ceil(m.ObjBound - 1e-6))), stage)
        elif m.Status == GRB.TIME_LIMIT and math.isfinite(m.ObjBound) and m.ObjBound > -1e90:
            self.raise_bound(float(math.ceil(m.ObjBound - 1e-6)), stage)
        m.dispose()

    # ------------------------------------------------------------ primal heuristics

    def plan_from_lp(self) -> list[dict[str, Any]]:
        x = self.master.getAttr("X", self.vars)
        return [{"depot": self.keys[k][0], "tasks": list(self.keys[k][1])} for k, v in enumerate(x) if v > 0.5]

    def restricted_mip(self, limit: float, stage: str) -> None:
        limit = min(limit, self.left())
        if limit <= 0.05 or not self.vars:
            return
        t = time.perf_counter()
        self.master.update()
        mip = self.master.copy()
        mv = mip.getVars()
        for v in mv:
            v.VType = GRB.BINARY
            v.LB = 0.0
        keys = list(self.keys)
        if self.best:
            inc = {(d["depot"], tuple(d["tasks"])) for d in self.best}
            for v, key in zip(mv, keys):
                v.Start = 1.0 if key in inc else 0.0
        mip.Params.TimeLimit = limit
        mip.Params.Method, mip.Params.Crossover = -1, -1
        if self.ub < INF:
            mip.Params.Cutoff = self.ub - 0.5

        def decode(values) -> list[dict[str, Any]]:
            return [{"depot": d, "tasks": list(ts)} for v, (d, ts) in zip(values, keys) if v > 0.5]

        found = []

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                found.append(decode(model.cbGetSolution(mv)))

        mip.optimize(callback)
        if mip.SolCount:
            found.append(decode(mip.getAttr("X", mv)))
        mip.dispose()
        for plan in found:
            self.offer(plan, stage)
        self.stats["mip_runs"] += 1
        self.stats["mip_seconds"] += time.perf_counter() - t

    def complete_greedily(self, chosen: list[int]) -> list[dict[str, Any]]:
        """The chosen pool columns plus a greedy cover of the remaining tasks by pool columns (cost per new task)."""
        covered = np.zeros(self.n, dtype=bool)
        plan = []
        for k in chosen:
            covered[list(self.keys[k][1])] = True
            plan.append(k)
        if not covered.all():
            tasks = [np.array(key[1]) for key in self.keys]
            cost = np.array(self.cost)
            while not covered.all():
                gain = np.array([(~covered[t]).sum() for t in tasks], dtype=float)
                ratio = np.where(gain > 0, cost / np.maximum(gain, 1.0), INF)
                k = int(np.argmin(ratio))
                if not np.isfinite(ratio[k]):
                    break
                covered[tasks[k]] = True
                plan.append(k)
        return [{"depot": self.keys[k][0], "tasks": list(self.keys[k][1])} for k in plan]

    def dive(self, stage: str, until: float, fixed: list[int] | None = None) -> None:
        """Depth-first dive: fix duties to 1 (those at 0.75 or more in the master LP, else the largest fractional one)
        and re-run a few rounds of heuristic column generation, until the master LP is integral. A dive whose LP value
        reaches the incumbent is abandoned; one cut by its time budget is completed greedily from the pool."""
        fixed_vars = []
        fixed_idx: list[int] = []
        self.diving = True
        self.master.Params.Method = 1  # dual simplex after bound changes (about 3x faster than primal here)
        self.master.Params.Crossover = -1  # dives need basic solutions to detect integrality
        for k in fixed or []:
            self.vars[k].LB = 1.0
            fixed_vars.append(self.vars[k])
            fixed_idx.append(k)
        try:
            while True:
                if time.perf_counter() >= until or self.left() <= 0:
                    if fixed_idx and self.left() > 0:
                        self.offer(self.complete_greedily(fixed_idx), stage + "_greedy")
                    return
                beam = self.beam
                if self.dive_beam:
                    self.beam = self.dive_beam
                try:
                    self.column_generation(stage, until, stabilize=False, exact_allowed=self.dive_exact, max_rounds=self.dive_rounds)
                finally:
                    self.beam = beam
                if self.left() <= 0 or not self.solve_lp():
                    return
                if self.master.ObjVal >= self.ub - 1.0 + 1e-6:
                    return
                x = np.array(self.master.getAttr("X", self.vars))
                frac = np.nonzero((x > 1e-6) & (x < 1 - 1e-6))[0]
                if len(frac) == 0:
                    self.offer(self.plan_from_lp(), stage)
                    return
                lbs = np.array(self.master.getAttr("LB", self.vars))
                pick = np.nonzero((x >= self.dive_fix) & (lbs < 0.5))[0]
                if len(pick) == 0:
                    pick = [int(frac[np.argmax(x[frac] + 1e-3 * self.rng.random(len(frac)))])]
                for k in pick:
                    self.vars[int(k)].LB = 1.0
                    fixed_vars.append(self.vars[int(k)])
                    fixed_idx.append(int(k))
        finally:
            self.diving = False
            for v in fixed_vars:
                v.LB = 0.0
            self.root_method(self.master)
            self.master.update()

    def improve(self) -> None:
        cycle = 0
        while self.left() > 0.5 and not self.closed():
            cycle += 1
            # The first dive may use all the remaining time (it is the main primal driver); later ones a quarter of T.
            budget = self.left() if cycle == 1 else max(5.0, min(0.25 * self.time_limit, self.left()))
            until = time.perf_counter() + budget
            if cycle % 3 == 1:
                self.stats["dives"] += 1
                self.dive("dive", until)
            else:
                self.stats["lns_dives"] += 1
                inc = [self.index.get((d["depot"], tuple(d["tasks"]))) for d in (self.best or [])]
                inc = [k for k in inc if k is not None]
                pick = list(self.rng.choice(inc, size=max(1, len(inc) // 4), replace=False)) if inc else []
                self.dive("lns_dive", until, fixed=[int(k) for k in pick])
            if self.left() > 1.0:
                self.restricted_mip(min(max(5.0, 0.05 * self.time_limit), 120.0), "mip")
            if self.left() > 1.0 and self.stats["root_converged"]:
                self.try_enumeration("enumeration")
            # Back at the root master (no fixings): a few stabilized exact-capable rounds keep the bound and pool fresh.
            if self.left() > 1.0 and not self.stats["root_converged"]:
                self.stats["root_converged"] = self.column_generation("root_more", time.perf_counter() + min(0.1 * self.time_limit, self.left()))
