"""Engineered branch-and-price reference for the VRPTW, distance objective (scale-up study).

Master (integer tenths), over ng-routes r with cost c_r (truncated distances):
    min sum_r c_r lam_r + M (artificials)
    s.t. sum_r a_ir lam_r >= 1                 (pi_i >= 0; covering relaxes partitioning; a_ir counts repeated visits)
         sum_r lam_r <= k_hi                    (sigma_hi <= 0; k_hi = VEHICLES, or a branching bound)
         sum_r lam_r >= k_lo                    (sigma_lo >= 0; k_lo = ceil(q(N) / Q), or a branching bound)
with one artificial per row at a cost M above any plan, so every node master is feasible (a node whose converged LP
needs an artificial gets a bound above every plan and is pruned).

Pricing (pricing.TWPricer): ng-route labelling (ng = 8) with time windows and capacity, time buckets, a dominance
surface over (node, ng-memory, load), completion bounds over a time grid (time and capacity in exact rounds), and
memory bits of unreachable customers cleared. Rounds escalate when one finds nothing: heuristic (12 near-neighbour
arcs, a beam of 600 labels per bucket), wider heuristic (24 arcs, beam 2400), exact; up to `max_cols` columns per
round. After a node's first exact round, another exact round (for its bound) comes every `exact_every` rounds while
exact rounds have used at most 30% of the node's time; one exact round may take at most a quarter of the remaining
time (cut short it is incomplete: no bound). A node whose heuristics find nothing while two exact rounds in a row
cannot finish is branched on as it is ("stalled"; it keeps its last valid bound). Dual stabilization: Wentges
smoothing, the pricing duals are alpha * center + (1 - alpha) * LP duals, the center being the duals of the best
Lagrangian bound at the node; a round whose columns are not negative at the LP duals (mispricing) is repeated at the
LP duals.

Lower bounds. Every complete exact round at duals (pi, sigma) gives a Lagrangian bound: a plan is a partition of the
customers by k elementary routes, k_lo <= k <= k_hi, so its cost is
    sum_i pi_i + k sigma + sum_r rc_r  >=  sum_i pi_i + min over k in {k_lo, k_hi} of k (sigma + min(r*, 0)),
r* the exact minimum reduced cost over all ng-routes of the node (a superset of its elementary routes); it is rounded
up to the 0.1 grid after a floating-point allowance. At the root it is global; at a node it holds inside the node.

Search. Best-first branch-and-price (ties: deeper first): on the number of vehicles when sum lam is fractional, then on
the most fractional arc flow x_ij (x_ij = 0: arc removed; x_ij = 1: every other arc out of i and into j removed, the
depot excepted). Columns violating a node's rules get an upper bound of 0; pricing sees the removed arcs as infinite.
Primal side: the shared basic initializer (Solomon I1), and, alternating between pricing rounds, restricted-master
MIPs over the elementary columns (set partitioning, at most K routes, warm start and cutoff from the incumbent; a
longer one after the root) and price-and-fix dives (fix the largest LP column, remove its customers, a few heuristic
pricing rounds, repeat; a dive out of budget routes its uncovered customers with the I1 construction); integral node
LPs; and a local search (relocate, or-opt, swap, 2-opt*) on every plan, whose routes join the column pool. Every plan
is checked by route_cost before it is published. Not implemented: subset-row or rounded capacity cuts.
"""
from __future__ import annotations

import inspect
import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.cg.vrptw.a_data import VRPTWInstance, load_vrptw
from src.cg.vrptw.b_mip_xl import route_cost_tenths
from src.cg.vrptw.heuristic import LocalSearch, basic_initializer
from src.cg.vrptw.pricing import TWPricer

INF = float("inf")


class VRPTWColumnGenerationXL:
    """Engineered branch-and-price reference for the VRPTW; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0, ng: int = 8,
                 near: int = 12, beam: int = 600, max_cols: int = 60, exact_every: int = 20, alpha: float = 0.5,
                 dive_every: int = 10, verbose: bool = False) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.on_incumbent = on_incumbent or (lambda routes: None)  # receives routes as lists of customers 1..n
        try:
            params = inspect.signature(self.on_incumbent).parameters
            self.stage_kw = "stage" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())
        except (TypeError, ValueError):
            self.stage_kw = False
        self.on_bound = on_bound or (lambda bound: None)  # receives every new valid global lower bound
        self.ng, self.near, self.beam, self.max_cols = ng, near, beam, max_cols
        self.exact_every, self.alpha = exact_every, alpha
        self.dive_every = dive_every  # a price-and-fix dive after the root and after every dive_every-th node (0: never)
        self.max_labels = 400_000
        self.verbose = verbose
        self.trace: list[dict[str, Any]] = []

    # ------------------------------------------------------------ setup

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        return self.solve(load_vrptw(data_path))

    def left(self) -> float:
        return self.deadline - time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def log(self, stage: str) -> None:
        self.trace.append({"t": round(self.elapsed(), 3), "incumbent": self.ub / 10 if self.ub < INF else None,
                           "bound": self.lb / 10 if self.lb > -INF else None, "stage": stage})
        if self.verbose:
            print(self.trace[-1], flush=True)

    def solve(self, inst: VRPTWInstance) -> dict[str, Any]:
        if not hasattr(self, "t0"):
            self.t0 = time.perf_counter()
            self.deadline = self.t0 + self.time_limit
        self.inst = inst
        n = inst.n
        self.n = n
        self.K = int(inst.vehicles)
        self.Klo = min(self.K, inst.min_vehicles)
        self.ub, self.lb = INF, -INF
        self.best_plan: list[list[int]] | None = None
        self.first_incumbent: float | None = None
        self.stats = {"cg_iters": 0, "exact_rounds": 0, "heuristic_rounds": 0, "mispricings": 0, "lp_seconds": 0.0,
                      "pricing_seconds": 0.0, "exact_seconds": 0.0, "mip_runs": 0, "mip_seconds": 0.0, "ls_seconds": 0.0,
                      "nodes": 0, "labels": 0, "removed_columns": 0, "dives": 0, "dive_seconds": 0.0, "stalled_nodes": 0, "stuck_nodes": 0, "root_converged": False,
                      "root_bound": None, "root_lp": None, "root_seconds": None, "max_depth": 0, "incomplete_exact": 0}
        self.pricer = TWPricer(inst, ng=self.ng, near=self.near)
        self.ls = LocalSearch(inst)
        self.frontier: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.searching = False
        self.root_lb = self.trivial_bound()
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", self.threads)
        env.setParam("Seed", self.seed)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()
        self.env = env
        self.memory_limit = False
        try:
            self.build_master()
            self.construct()
            self.search()
        except gp.GurobiError as err:
            if err.errno != GRB.Error.OUT_OF_MEMORY and "memory" not in str(err).lower():
                raise
            self.memory_limit = True
        out = self.result("memory_limit" if self.memory_limit else "optimal" if self.closed() else "time_limit")
        env.dispose()
        return out

    def closed(self) -> bool:
        return self.ub < INF and self.lb >= self.ub

    def prunable(self, bound: float) -> bool:
        return bound >= self.ub  # integral costs (tenths): no strictly better plan below a bound >= ub

    def result(self, reason: str) -> dict[str, Any]:
        ub = self.ub / 10 if self.ub < INF else None
        lb = self.lb / 10 if self.lb > -INF else None
        routes = [list(r) for r in self.best_plan] if self.best_plan else None
        return {"best_objective": ub, "global_lb": lb, "global_ub": ub,
                "gap": (ub - lb) / abs(ub) if ub is not None and lb is not None and ub else None,
                "termination_reason": reason, "routes": routes, "columns": len(self.cols) if hasattr(self, "cols") else 0,
                "first_incumbent_seconds": self.first_incumbent, **self.stats, "total_seconds": self.elapsed(),
                "trace": self.trace}

    # ------------------------------------------------------------ bounds

    @staticmethod
    def certify(value: float, scale: float) -> int:
        """A computed bound (tenths) made safe against floating-point error, then rounded up (integral costs)."""
        return int(math.ceil(value - 1e-7 * max(1.0, scale)))

    def trivial_bound(self) -> int:
        """Before any pricing: every customer leaves on one arc and at least k_lo routes leave the depot."""
        pr = self.pricer
        out_min = np.minimum(np.where(np.isfinite(pr.C), pr.C, INF).min(axis=1), pr.c_in)
        dep = np.sort(pr.c_out[np.isfinite(pr.c_out)])[:self.Klo]
        val = float(out_min.sum() + dep.sum())
        return self.certify(val, val) if math.isfinite(val) else 0

    def publish_bound(self, stage: str) -> None:
        if self.searching:
            nodes = self.frontier + ([self.current] if self.current is not None else [])
            lb = min((x["lb"] for x in nodes), default=INF)
        else:
            lb = self.root_lb
        if self.ub < INF:
            lb = min(lb, self.ub)
        if lb > self.lb and math.isfinite(lb):
            self.lb = lb
            self.on_bound(lb / 10)
            self.log(stage)

    def lagrangian(self, pi: np.ndarray, sigma: float, min_rc: float, node: dict) -> float:
        r = min(min_rc, -1e-6)
        val = float(pi.sum()) + min(node["klo"] * (sigma + r), node["khi"] * (sigma + r))
        scale = float(np.abs(pi).sum() + node["khi"] * (abs(sigma) + abs(r)))
        return self.certify(val, scale)

    # ------------------------------------------------------------ master

    def build_master(self) -> None:
        n = self.n
        md = gp.Model(env=self.env)
        md.Params.Method = 0
        d = self.inst.dist
        self.big_m = 1.0 + 2.0 * float(d[0, 1:].sum() + d[1:, 0].sum() + d.max(axis=1).sum())
        self.cover = [md.addLConstr(gp.LinExpr(), GRB.GREATER_EQUAL, 1.0) for _ in range(n)]
        self.fleet_hi = md.addLConstr(gp.LinExpr(), GRB.LESS_EQUAL, float(self.K))
        self.fleet_lo = md.addLConstr(gp.LinExpr(), GRB.GREATER_EQUAL, float(self.Klo))
        for row in self.cover:
            md.addVar(lb=0.0, obj=self.big_m, column=gp.Column([1.0], [row]))
        md.addVar(lb=0.0, obj=self.big_m, column=gp.Column([-1.0], [self.fleet_hi]))
        md.addVar(lb=0.0, obj=self.big_m, column=gp.Column([1.0], [self.fleet_lo]))
        self.master = md
        self.cols: list[dict[str, Any]] = []
        self.col_index: dict[tuple[int, ...], int] = {}
        self.forb: frozenset = frozenset()
        for i in range(n):
            if self.pricer.out_ok[i] and self.pricer.in_ok[i]:
                self.add_column([i])

    def path_cost(self, path: list[int]) -> int:
        pr = self.pricer
        return int(pr.d_out[path[0]] + sum(int(pr.d[a, b]) for a, b in zip(path, path[1:])) + pr.d_in[path[-1]])

    def arcs_of(self, path: list[int]) -> list[tuple[int, int]]:
        n = self.n
        return [(n, path[0])] + list(zip(path, path[1:])) + [(path[-1], n)]

    def add_column(self, path: list[int]) -> bool:
        """Adds an ng-route (internal customers 0..n-1) if new and feasible (capacity and windows)."""
        path = [int(i) for i in path]
        key = tuple(path)
        if key in self.col_index or not path or not self.pricer.feasible(path):
            return False
        cnt: dict[int, int] = {}
        for i in path:
            cnt[i] = cnt.get(i, 0) + 1
        arcs = self.arcs_of(path)
        rows = [self.cover[i] for i in cnt] + [self.fleet_hi, self.fleet_lo]
        coef = [float(v) for v in cnt.values()] + [1.0, 1.0]
        ub = 0.0 if self.forb and any(a in self.forb for a in arcs) else INF
        cost = self.path_cost(path)
        var = self.master.addVar(lb=0.0, ub=ub, obj=float(cost), column=gp.Column(coef, rows))
        self.col_index[key] = len(self.cols)
        self.cols.append({"path": path, "cost": cost, "elem": len(cnt) == len(path), "var": var, "arcs": arcs})
        return True

    def set_node(self, node: dict) -> None:
        self.forb = node["forb"]
        self.fleet_hi.RHS = float(node["khi"])
        self.fleet_lo.RHS = float(node["klo"])
        F = self.forb
        ubs = [0.0 if F and any(a in F for a in c["arcs"]) else INF for c in self.cols]
        self.master.setAttr("UB", [c["var"] for c in self.cols], ubs)
        pr = self.pricer
        Cc, c_out, c_in = pr.C.copy(), pr.c_out.copy(), pr.c_in.copy()
        n = self.n
        for (a, b) in F:
            if a == n:
                c_out[b] = INF
            elif b == n:
                c_in[a] = INF
            else:
                Cc[a, b] = INF
        self.node_arcs = (Cc, c_out, c_in)

    def solve_lp(self) -> str:
        if self.left() <= 0:
            return "time"
        t = time.perf_counter()
        self.master.Params.TimeLimit = max(1e-3, self.left())
        self.master.optimize()
        self.stats["lp_seconds"] += time.perf_counter() - t
        st = self.master.Status
        if st == GRB.MEM_LIMIT:
            raise gp.GurobiError(GRB.Error.OUT_OF_MEMORY, "memory limit")
        return "optimal" if st == GRB.OPTIMAL else "infeasible" if st in (GRB.INFEASIBLE, GRB.INF_OR_UNBD) else "time"

    def duals(self) -> tuple[np.ndarray, float]:
        pi = np.maximum(np.array(self.master.getAttr("Pi", self.cover)), 0.0)
        sigma = min(float(self.fleet_hi.Pi), 0.0) + max(float(self.fleet_lo.Pi), 0.0)
        return pi, sigma

    def snapshot(self) -> None:
        vars_ = [c["var"] for c in self.cols]
        self.snap = {"x": np.array(self.master.getAttr("X", vars_)), "rc": np.array(self.master.getAttr("RC", vars_)),
                     "ncols": len(self.cols), "lp": float(self.master.ObjVal)}

    def compact(self) -> None:
        """Drops the columns with the largest reduced costs at the last LP once the pool is large (singletons and
        incumbent routes stay); pricing regenerates any column a later LP needs."""
        k = len(self.cols)
        limit = max(15000, 75 * self.n)
        snap = getattr(self, "snap", None)
        if k <= limit or snap is None:
            return
        k0 = snap["ncols"]
        keep = np.zeros(k, dtype=bool)
        keep[np.argsort(snap["rc"], kind="stable")[:limit // 2]] = True
        keep[:k0] |= snap["x"] > 1e-9
        keep[k0:] = True
        keep |= np.array([len(c["path"]) == 1 for c in self.cols])
        if self.best_plan:
            for r in self.best_plan:
                t = self.col_index.get(tuple(i - 1 for i in r))
                if t is not None:
                    keep[t] = True
        drop = np.flatnonzero(~keep)
        self.master.remove([self.cols[t]["var"] for t in drop])
        self.cols = [self.cols[t] for t in np.flatnonzero(keep)]
        self.col_index = {tuple(c["path"]): t for t, c in enumerate(self.cols)}
        self.stats["removed_columns"] += len(drop)
        self.snap = None
        self.master.update()

    # ------------------------------------------------------------ column generation

    def price(self, pi: np.ndarray, sigma: float, exact: bool, level: int = 0) -> dict[str, Any]:
        """level 0: heuristic (near neighbours, beam); level 1: wider heuristic (2 near, 4 x beam); exact=True: exact."""
        t = time.perf_counter()
        Cc, c_out, c_in = self.node_arcs
        # An exact round may use at most a quarter of the remaining time (cut short, it is incomplete: no bound).
        deadline = min(self.deadline, t + max(10.0, 0.25 * self.left())) if exact else self.deadline
        res = self.pricer.price(pi, -sigma, Cc, c_out, c_in, exact=exact, beam=self.beam * (4 if level else 1),
                                max_cols=self.max_cols, deadline=deadline, max_labels=self.max_labels, wide=level > 0)
        dt = time.perf_counter() - t
        self.stats["labels"] += res["labels"]
        self.stats["pricing_seconds"] += dt
        self.stats["exact_rounds" if exact else "heuristic_rounds"] += 1
        if exact:
            self.stats["exact_seconds"] += dt
            if not res["complete"]:
                self.stats["incomplete_exact"] += 1
        return res

    def rc_at(self, path: list[int], pi: np.ndarray, sigma: float) -> float:
        return self.path_cost(path) - float(sum(pi[i] for i in path)) - sigma

    def column_generation(self, node: dict, stage: str) -> str:
        """CG at a node. Returns "converged", "pruned" or "time". Pricing escalates when a round finds nothing:
        heuristic, wider heuristic, exact; besides, an exact round (for its Lagrangian bound) every `exact_every`
        rounds while exact rounds have used at most 30% of the node's time."""
        level = 0
        since_exact = 0
        t_node = time.perf_counter()
        exact_time = 0.0
        exact_count = 0
        stalls = 0
        center = None  # (pi, sigma, bound) of the best Lagrangian bound at this node
        while self.left() > 0:
            if self.primal_due():
                self.primal_step()
            st = self.solve_lp()
            if st != "optimal":
                return "pruned" if st == "infeasible" else "time"
            self.stats["cg_iters"] += 1
            lpv = float(self.master.ObjVal)
            node["lp"] = lpv
            pi, sigma = self.duals()
            # Periodic exact rounds (for their bounds) only after a first exact round at this node, which comes from
            # escalation when the heuristics stall (early duals can make exact labelling explode).
            periodic = (exact_count > 0 and since_exact >= self.exact_every
                        and exact_time <= 0.3 * (time.perf_counter() - t_node))
            exact = level >= 2 or periodic
            tries = []
            if center is not None and self.alpha > 0:
                tries.append((center[0] * self.alpha + pi * (1 - self.alpha), center[1] * self.alpha + sigma * (1 - self.alpha)))
            tries.append((pi, sigma))
            new: list[list[int]] = []
            complete_any = complete_plain = False
            incomplete = False
            t_round = time.perf_counter()
            for k, (pp, ss) in enumerate(tries):
                res = self.price(pp, ss, exact, level=min(level, 1))
                if exact and res["complete"]:
                    complete_any = True
                    L = self.lagrangian(pp, ss, res["min_rc"], node)
                    if center is None or L > center[2]:
                        center = (pp.copy(), float(ss), L)
                    if L > node["lb"]:
                        node["lb"] = L
                        if node["id"] == 0:
                            self.root_lb = max(self.root_lb, L)
                        self.publish_bound(stage)
                elif exact:
                    incomplete = True
                new = [path for _, path in res["routes"]
                       if tuple(path) not in self.col_index and self.rc_at(path, pi, sigma) < -1e-6]
                if new:
                    break
                if k + 1 < len(tries):
                    self.stats["mispricings"] += 1
                else:
                    complete_plain = exact and res["complete"]
            if exact:
                exact_time += time.perf_counter() - t_round
                exact_count += 1
            if self.closed() or self.prunable(node["lb"]):
                return "pruned"
            if (complete_plain and not new) or (complete_any and node["lb"] >= self.certify(lpv, abs(lpv))):
                # Converged, or the bound already matches the rounded LP value (more columns cannot raise it here).
                self.snapshot()
                for path in new:
                    self.add_column(path)
                return "converged"
            added = sum(self.add_column(path) for path in new)
            if exact:
                since_exact = 0
                if incomplete and not added:
                    self.max_labels = min(2 * self.max_labels, 1_600_000)  # bounded growth (memory)
                    stalls += 1
                    if stalls >= 2:
                        # The heuristics find nothing and exact labelling cannot finish: branch on this LP (the node
                        # keeps its last valid bound) rather than spend the budget on more incomplete rounds.
                        node["stalled"] = True
                        self.stats["stalled_nodes"] += 1
                        self.snapshot()
                        return "converged"
            else:
                since_exact += 1
            level = 0 if added else min(level + 1, 2)
        return "time"

    # ------------------------------------------------------------ primal

    def offer(self, routes: list[list[int]], stage: str, polish: bool = True) -> bool:
        """Checks a plan (customers 1..n) with route_cost_tenths and publishes it if it improves; polishes it by local
        search; the routes join the column pool."""
        routes = [[int(i) for i in r] for r in routes if len(r)]
        obj = route_cost_tenths(self.inst, routes)
        if obj is None:
            return False
        better = obj < self.ub
        if better:
            self.set_incumbent(obj, routes, stage)
        if polish and self.left() > 0:
            t = time.perf_counter()
            improved = self.ls.improve(routes, self.deadline)
            self.stats["ls_seconds"] += time.perf_counter() - t
            obj2 = route_cost_tenths(self.inst, improved)
            if obj2 is not None:
                if obj2 < self.ub:
                    self.set_incumbent(obj2, improved, stage + "+ls")
                    better = True
                for r in improved:
                    self.add_column([i - 1 for i in r])
        for r in routes:
            self.add_column([i - 1 for i in r])
        return better

    def set_incumbent(self, obj: int, routes: list[list[int]], stage: str) -> None:
        self.ub, self.best_plan = obj, [list(r) for r in routes]
        if self.first_incumbent is None:
            self.first_incumbent = self.elapsed()
        if self.stage_kw:
            self.on_incumbent([list(r) for r in routes], stage=stage)
        else:
            self.on_incumbent([list(r) for r in routes])
        self.log(stage)
        self.publish_bound(stage)

    def construct(self) -> None:
        try:
            init = basic_initializer(self.inst)
        except ValueError:
            return
        self.offer(init, "initializer")

    def primal_due(self) -> bool:
        return time.perf_counter() >= self.next_primal

    def primal_step(self) -> None:
        """Between pricing rounds: alternately a restricted-master MIP and (while a node is being solved) a dive."""
        self.primal_steps += 1
        if self.dive_every and self.current is not None and self.primal_steps % 2 == 0:
            self.dive(self.current, max(2.0, min(60.0, 0.05 * self.time_limit)))
        else:
            lp = self.current.get("lp") if self.current is not None else None
            far = self.ub < INF and lp is not None and self.ub > 1.03 * lp
            self.restricted_mip(self.mip_limit * (2 if far else 1), "restricted_mip")
        self.next_primal = time.perf_counter() + max(self.primal_gap, 0.1 * self.elapsed())

    def restricted_mip(self, limit: float, stage: str) -> None:
        """Set-partitioning integer master over the elementary columns (at most K routes); integer solutions are
        published from the callback (checked and polished by offer)."""
        limit = min(limit, self.left())
        if limit <= 0.2:
            return
        t = time.perf_counter()
        self.stats["mip_runs"] += 1
        cols = [k for k, c in enumerate(self.cols) if c["elem"]]
        md = gp.Model(env=self.env)
        md.Params.TimeLimit = limit
        md.Params.MIPFocus = 1
        lam = [md.addVar(vtype=GRB.BINARY, obj=float(self.cols[k]["cost"])) for k in cols]
        rows = [gp.LinExpr() for _ in range(self.n)]
        for v, k in zip(lam, cols):
            for i in self.cols[k]["path"]:
                rows[i].addTerms(1.0, v)
        for i in range(self.n):
            md.addLConstr(rows[i], GRB.EQUAL, 1.0)
        md.addLConstr(gp.quicksum(lam), GRB.LESS_EQUAL, float(self.K))
        if self.best_plan:
            md.Params.Cutoff = self.ub - 0.5
            pos = {k: v for v, k in enumerate(cols)}
            starts = [pos.get(self.col_index.get(tuple(i - 1 for i in r), -1)) for r in self.best_plan]
            if all(s_ is not None for s_ in starts):
                for v in lam:
                    v.Start = 0.0
                for s_ in starts:
                    lam[s_].Start = 1.0
        found = []

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                vals = model.cbGetSolution(lam)
                plan = [[i + 1 for i in self.cols[cols[v]]["path"]] for v, val in enumerate(vals) if val > 0.5]
                found.append(plan)
                self.offer(plan, stage)

        md.optimize(callback)
        if md.Status == GRB.MEM_LIMIT:
            self.memory_limit = True
        md.dispose()
        self.stats["mip_seconds"] += time.perf_counter() - t

    def dive(self, node: dict, budget: float) -> None:
        """Price-and-fix dive from the node's master: repeatedly fix to 1 the elementary column with the largest LP
        value (and every column at 1), remove its customers from every other column and from pricing, and
        re-optimize with a few heuristic pricing rounds; an integral LP solution gives a plan. The node's master is
        restored afterwards; the columns found stay in the pool."""
        end = min(self.deadline, time.perf_counter() + budget)
        t_start = time.perf_counter()
        self.stats["dives"] += 1
        fixed: list[int] = []
        covered: set[int] = set()
        Cc, c_out, c_in = (a.copy() for a in self.node_arcs)
        saved = self.node_arcs
        try:
            while time.perf_counter() < end:
                for rnd in range(5):
                    st = self.solve_lp()
                    if st != "optimal":
                        return
                    if rnd == 4 or time.perf_counter() > end:
                        break
                    pi, sigma = self.duals()
                    self.node_arcs = (Cc, c_out, c_in)
                    res = self.price(pi, sigma, exact=False)
                    new = [p for _, p in res["routes"] if tuple(p) not in self.col_index and self.rc_at(p, pi, sigma) < -1e-6]
                    if not new:
                        break
                    for p in new:
                        self.add_column(p)
                if self.master.ObjVal >= self.ub - 0.5:
                    return  # the dive cannot improve the incumbent any more
                vars_ = [c["var"] for c in self.cols]
                x = np.array(self.master.getAttr("X", vars_))
                pos = [t for t in np.flatnonzero(x > 1e-6) if t not in fixed]
                if all(x[t] > 1 - 1e-6 and self.cols[t]["elem"] for t in pos):
                    plan = [[i + 1 for i in self.cols[t]["path"]] for t in fixed + pos]
                    self.offer(plan, "dive")
                    return
                cand = [t for t in pos if self.cols[t]["elem"]]
                if not cand:
                    return
                top = max(cand, key=lambda t: x[t])
                pick = {top} | {t for t in cand if x[t] > 0.9}
                pick = {t for t in pick if not covered & set(self.cols[t]["path"])}
                chosen: set[int] = set()
                for t in sorted(pick, key=lambda t: -x[t]):  # disjoint columns only
                    if not chosen & set(self.cols[t]["path"]):
                        chosen |= set(self.cols[t]["path"])
                    else:
                        pick.discard(t)
                for t in pick:
                    fixed.append(int(t))
                    self.cols[t]["var"].LB = 1.0
                    covered |= set(self.cols[t]["path"])
                cov = sorted(covered)
                Cc[:, cov] = INF
                Cc[cov, :] = INF
                c_out[cov] = INF
                c_in[cov] = INF
                fixed_set = set(fixed)
                bad = [c["var"] for t, c in enumerate(self.cols) if t not in fixed_set and any(i in covered for i in c["path"])]
                if bad:
                    self.master.setAttr("UB", bad, [0.0] * len(bad))
                if len(covered) == self.n:
                    plan = [[i + 1 for i in self.cols[t]["path"]] for t in fixed]
                    self.offer(plan, "dive")
                    return
            # Out of budget: the uncovered customers are routed by the basic construction.
            if fixed and len(covered) < self.n:
                rest = [i + 1 for i in range(self.n) if i not in covered]
                try:
                    tail = basic_initializer(self.inst, customers=rest)
                except ValueError:
                    tail = None
                if tail is not None:
                    self.offer([[i + 1 for i in self.cols[t]["path"]] for t in fixed] + tail, "dive_completed")
        finally:
            for t in fixed:
                if t < len(self.cols):
                    self.cols[t]["var"].LB = 0.0
            self.node_arcs = saved
            self.set_node(node)
            self.stats["dive_seconds"] += time.perf_counter() - t_start

    # ------------------------------------------------------------ search

    def branch(self, node: dict) -> list[dict] | None:
        """Children of a converged node: on the number of vehicles if fractional, then on the most fractional arc.
        None: the LP solution is integral (offered as a plan)."""
        x = self.snap["x"]
        pos = np.flatnonzero(x > 1e-9)
        base = {"depth": node["depth"] + 1, "lb": node["lb"], "forb": node["forb"], "klo": node["klo"], "khi": node["khi"]}
        total = float(x[pos].sum())
        if abs(total - round(total)) > 1e-6:
            k = int(math.floor(total))
            kids = []
            if k >= node["klo"]:
                kids.append(dict(base, khi=k))
            if k + 1 <= node["khi"]:
                kids.append(dict(base, klo=k + 1))
            return kids
        flow: dict[tuple[int, int], float] = {}
        for t in pos:
            for a in self.cols[t]["arcs"]:
                flow[a] = flow.get(a, 0.0) + x[t]
        frac = [(abs(v - 0.5), a) for a, v in flow.items() if 1e-6 < v < 1 - 1e-6]
        if frac:
            _, (i, j) = min(frac)
            n = self.n
            one = set(node["forb"])
            if i != n:
                one |= {(i, k) for k in range(n + 1) if k != j and k != i}
            if j != n:
                one |= {(k, j) for k in range(n + 1) if k != i and k != j}
            return [dict(base, forb=node["forb"] | {(i, j)}), dict(base, forb=frozenset(one))]
        # Integral arc flows: the routes they form (an LP solution of value node["lp"]).
        if all(abs(x[t] - 1.0) < 1e-6 for t in pos) and all(self.cols[t]["elem"] for t in pos):
            plan = [[i + 1 for i in self.cols[t]["path"]] for t in pos]
            self.offer(plan, "integral_node", polish=False)
            if not node.get("stalled"):  # a converged LP value bounds the node; a stalled LP's does not
                node["lb"] = max(node["lb"], self.certify(self.snap["lp"], abs(self.snap["lp"])))
        return None

    def search(self) -> None:
        T = self.time_limit
        self.mip_limit = max(3.0, min(30.0, 0.02 * T))
        self.primal_gap = max(5.0, min(90.0, 0.05 * T))
        self.next_primal = time.perf_counter() + self.primal_gap
        self.primal_steps = 0
        self.snap = None
        self.frontier = [{"forb": frozenset(), "klo": self.Klo, "khi": self.K, "depth": 0, "id": 0, "lb": self.root_lb}]
        self.searching = True
        self.publish_bound("trivial")
        count = 0
        while self.left() > 0 and not self.closed():
            live = [x for x in self.frontier if not x.get("stuck")]
            if not live:
                break
            node = min(live, key=lambda x: (x["lb"], -x["depth"]))
            self.frontier.remove(node)
            self.current = node
            if self.prunable(node["lb"]):
                self.current = None
                self.publish_bound("prune")
                continue
            self.stats["nodes"] += 1
            self.stats["max_depth"] = max(self.stats["max_depth"], node["depth"])
            self.set_node(node)
            root = node["id"] == 0
            status = self.column_generation(node, "root" if root else "node")
            if root:
                self.stats["root_converged"] = status == "converged" and not node.get("stalled")
                self.stats["root_lp"] = node.get("lp", 0.0) / 10 if node.get("lp") is not None else None
                self.stats["root_bound"] = self.lb / 10 if self.lb > -INF else None
                self.stats["root_seconds"] = self.elapsed()
                self.log("root_done")
            if status == "time":
                self.frontier.append(node)
                break
            if status == "pruned":
                self.current = None
                self.publish_bound("prune")
                continue
            kids = self.branch(node)
            if kids is None:
                if not self.prunable(node["lb"]):
                    node["stuck"] = True
                    self.stats["stuck_nodes"] += 1
                    self.frontier.append(node)
            else:
                for kid in kids:
                    count += 1
                    kid["id"] = count
                    self.frontier.append(kid)
            self.current = None
            self.publish_bound("branch")
            if self.dive_every and (root or self.stats["nodes"] % self.dive_every == 0) and self.left() > 1:
                self.dive(node, max(2.0, min(60.0, 0.05 * T)))
            self.compact()
            if root:
                self.restricted_mip(2 * self.mip_limit, "restricted_mip_root")
            elif self.primal_due():
                self.primal_step()
        self.current = None
        while self.left() > 0.5 and not self.closed():
            before = self.ub
            self.restricted_mip(self.left(), "restricted_mip_final")
            if self.ub >= before:
                break
