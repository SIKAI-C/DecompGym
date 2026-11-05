"""Engineered branch-and-price reference for the PDPTW (scale-up study; the paper's references are unchanged).

Master (set covering over routes k, cost c_k = route distance):
    min sum_k c_k lam_k + BIG a
    s.t. sum_{k serves r} lam_k + a >= 1      (pi_r >= 0)
         sum_k lam_k <= hi                     (mu <= 0; hi = K, the fleet, at the root)
         sum_k lam_k >= lo                     (nu >= 0; lo = 0 at the root)
The artificial column a covers every request and stays outside the fleet rows, so every node master is feasible; it
only enlarges the master, so every bound below stays valid.

Pricing (pricing.PDPPricer): heuristic rounds first (elementary; successors limited to the best `near` pickups,
dominance without the unreachable sets, at most `beam` labels per node), adding up to `max_cols` routes per round; an
exact round when a heuristic round finds nothing or after `exact_every` heuristic rounds, within `exact_budget`
seconds: labelling over ng-routes (memories of the `ng` requests nearest to each node, a relaxation of the elementary
routes; elementary when ng >= the number of requests) with open-request sets. ng-routes that pick a request up twice
enter the LP master with that coefficient but never a restricted MIP. When an exact round cannot finish and finds
nothing, the heuristic pricing is widened and exact rounds become rarer.

Stabilization: Wentges smoothing of the exact-round duals around the duals that gave the node's best Lagrangian bound
(alpha = 0.5); a mispriced round (no route negative at the master duals) is repeated at the master duals.

Lower bounds: for any duals pi >= 0, mu <= 0, nu >= 0 and a complete exact round with minimum reduced cost r* (route
constant -mu-nu), every plan of a node uses between lo and hi routes, so
    L = sum_r pi_r + mu hi + nu lo + hi min(0, r*)
is valid for the node (Lagrangian relaxation of the covering and fleet rows; r* is the minimum over all ng-routes
allowed by the node's arc rules, a superset of its elementary routes). It is rounded up after a floating-point
allowance (integral costs). No bound comes from an exact round cut short by its budget (exact_budget = max(5 s,
0.1 T)). Children inherit their
parent's bound; the global bound is the minimum over the open nodes.

Branching: best-first on node bounds (ties: deeper first): on the number of routes when sum lam is fractional, then on
the most fractional arc flow x_ij (x_ij = 0: the arc is removed; x_ij = 1: the other arcs out of i and into j are
removed, depot ends excepted).

Primal side (every plan is checked with plan_cost before it is published; the publication carries its stage): the
shared basic initializer (pricing.insertion_plan), then local search on every new incumbent (relocation of each
request to its best position in any route, its own included; emptied routes disappear) whose routes join the
column pool; alternately, between pricing rounds, restricted-master MIPs (set partitioning over the elementary pool,
fleet row, incumbent as start, cutoff below it) and dives (fix the columns at >= 0.95 or the largest one, ban
overlapping columns, re-price heuristically, repeat to an integral master); a final restricted MIP; integral node LP
solutions.
"""
from __future__ import annotations

import inspect
import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.cg.pdptw.a_data import PDPTWInstance, feasible_arcs, load_pdptw, plan_cost, route_check
from src.cg.pdptw.pricing import PDPPricer, best_insertion, insertion_plan

INF = float("inf")


class LocalSearch:
    """Request relocation: remove a request (its pickup and delivery) from its route and insert it at its cheapest
    feasible position in any route (its own included) when that is cheaper; passes over all requests until a pass
    finds no improving move."""

    def __init__(self, inst: PDPTWInstance, c: list[list[int]]) -> None:
        self.inst, self.c = inst, c

    def improve(self, routes: list[list[int]], deadline: float) -> list[list[int]]:
        inst, c = self.inst, self.c
        routes = [list(r) for r in routes if r]
        cost = [route_check(inst, c, r)[0] for r in routes]
        if any(x is None for x in cost):
            return routes
        improved = True
        while improved and time.perf_counter() < deadline:
            improved = False
            where_now = {v: a for a in range(len(routes)) for v in routes[a]}
            for p in [v for r in routes for v in r if inst.is_pickup[v]]:
                if time.perf_counter() > deadline:
                    break
                a = where_now[p]
                d = inst.partner(p)
                rem = [v for v in routes[a] if v != p and v != d]
                rem_cost = route_check(inst, c, rem)[0] if rem else 0
                if rem_cost is None:
                    continue
                gain = cost[a] - rem_cost
                best, where, new = gain, None, None
                for b in range(len(routes)):
                    if b != a and not routes[b]:
                        continue
                    delta, cand = best_insertion(inst, c, rem if b == a else routes[b], p, d, best)
                    if cand is not None and delta < best:
                        best, where, new = delta, b, cand
                if where is None:
                    continue
                if where == a:
                    routes[a], cost[a] = new, rem_cost + best
                else:
                    routes[a], cost[a] = rem, rem_cost
                    routes[where], cost[where] = new, cost[where] + best
                    where_now[p] = where_now[d] = where
                improved = True
        return [r for r in routes if r]


class PDPTWColumnGenerationXL:
    """Engineered branch-and-price reference for the PDPTW; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0, max_cols: int = 60,
                 near: int = 12, beam: int = 40, alpha: float = 0.5, exact_every: int = 10, tol: float = 1e-6,
                 max_beam: int = 2000, ng: int | None = 10, verbose: bool = False) -> None:
        # ng: exact rounds price over ng-routes with neighbourhoods of this size (None: elementary routes)
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.on_incumbent = on_incumbent or (lambda routes, stage=None: None)  # receives [[node, ...], ...] plans
        try:
            params = inspect.signature(self.on_incumbent).parameters
            self.hook_takes_stage = "stage" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())
        except (TypeError, ValueError):
            self.hook_takes_stage = False
        self.on_bound = on_bound or (lambda bound: None)  # receives every new valid global lower bound
        self.max_cols, self.near, self.beam, self.alpha, self.exact_every, self.tol = max_cols, near, beam, alpha, exact_every, tol
        self.max_beam, self.ng = max_beam, ng
        self.verbose = verbose
        self.trace: list[dict[str, Any]] = []

    # ------------------------------------------------------------ setup

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        return self.solve(load_pdptw(data_path))

    def left(self) -> float:
        return self.deadline - time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def log(self, stage: str) -> None:
        self.trace.append({"t": round(self.elapsed(), 3), "incumbent": self.ub if self.ub < INF else None,
                           "bound": self.lb if self.lb > -INF else None, "stage": stage})
        if self.verbose:
            print(self.trace[-1], flush=True)

    def solve(self, inst: PDPTWInstance) -> dict[str, Any]:
        if not hasattr(self, "t0"):
            self.t0 = time.perf_counter()
            self.deadline = self.t0 + self.time_limit
        self.inst = inst
        self.n, self.N, self.K = inst.n, inst.N, inst.vehicles
        self.dist = inst.distances()
        self.c = self.dist.tolist()
        self.arcs = feasible_arcs(inst, self.dist)
        self.pred = [[int(i) for i in np.nonzero(self.arcs[:, j])[0]] for j in range(self.N + 1)]
        self.succ = [[int(j) for j in np.nonzero(self.arcs[i])[0]] for i in range(self.N + 1)]
        self.pricer = PDPPricer(inst, self.dist, self.arcs)
        self.ls = LocalSearch(inst, self.c)
        self.ub, self.lb = INF, -INF
        self.best_plan: list[list[int]] | None = None
        self.first_t: float | None = None
        self.cols: list[dict[str, Any]] = []
        self.col_index: dict[tuple[int, ...], int] = {}
        self.forbid: frozenset = frozenset()
        self.dive_mask = 0
        self.frontier: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.searching = False
        self.stats = {"cg_iters": 0, "exact_rounds": 0, "heuristic_rounds": 0, "mispriced": 0, "lp_seconds": 0.0,
                      "pricing_seconds": 0.0, "exact_seconds": 0.0, "labels": 0, "mip_runs": 0, "mip_seconds": 0.0,
                      "ls_seconds": 0.0, "nodes": 0, "max_depth": 0, "stuck_nodes": 0, "removed_columns": 0,
                      "root_converged": False, "root_bound": None, "root_lp": None, "root_seconds": None,
                      "incomplete_exact": 0, "widened": 0, "dives": 0, "dive_seconds": 0.0}
        self.exact_budget = max(5.0, 0.1 * self.time_limit)
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", self.threads)
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
        # "stuck": the search stopped before the deadline because only stuck nodes remained (bound kept valid).
        out = self.result("memory_limit" if self.memory_limit else "optimal" if self.closed()
                          else "time_limit" if self.left() <= 0.5 else "stuck")
        env.dispose()
        return out

    def closed(self) -> bool:
        return self.ub < INF and self.lb >= self.ub - (1.0 - 1e-6)

    def prunable(self, bound: float) -> bool:
        return bound >= self.ub - (1.0 - 1e-6)

    def result(self, reason: str) -> dict[str, Any]:
        ub = self.ub if self.ub < INF else None
        lb = self.lb if self.lb > -INF else None
        self.stats["pricing_labels"] = self.pricer.stats["labels"]
        return {"best_objective": ub, "global_lb": lb, "global_ub": ub,
                "gap": (ub - lb) / abs(ub) if ub is not None and lb is not None and ub else None,
                "termination_reason": reason, "routes": [list(r) for r in self.best_plan] if self.best_plan else None,
                "columns": len(self.cols), "first_incumbent_seconds": self.first_t, **self.stats,
                "total_seconds": self.elapsed(), "trace": self.trace}

    # ------------------------------------------------------------ bounds

    def node_bound(self, node: dict) -> float:
        return node["lb"]

    def publish_bound(self, stage: str) -> None:
        nodes = self.frontier + ([self.current] if self.current is not None else [])
        if nodes:
            lb = min(x["lb"] for x in nodes)
        elif self.searching:
            lb = self.ub  # the tree is exhausted: the incumbent is optimal (or there is no plan)
        else:
            return
        if self.ub < INF:
            lb = min(lb, self.ub)
        if math.isfinite(lb) and lb > self.lb + 1e-9:
            self.lb = lb
            self.on_bound(lb)
            self.log(stage)

    def lagrangian(self, pi: np.ndarray, mu: float, nu: float, min_rc: float, node: dict) -> bool:
        """Node bound from a complete exact round at these duals (see the module docstring). True if it improved."""
        hi, lo = node["hi"], node["lo"]
        r = min(0.0, float(min_rc))
        value = float(pi.sum()) + mu * hi + nu * lo + hi * r
        scale = float(np.abs(pi).sum()) + abs(mu) * hi + abs(nu) * lo + hi * abs(r) + 1.0
        b = float(math.ceil(value - 1e-7 * scale))
        if b > node["lb"]:
            node["lb"] = b
            self.publish_bound("lagrangian")
            return True
        return False

    # ------------------------------------------------------------ master

    def build_master(self) -> None:
        n = self.n
        md = gp.Model(env=self.env)
        md.Params.Method = 0
        self.cover = [md.addLConstr(gp.LinExpr(), GRB.GREATER_EQUAL, 1.0) for _ in range(n)]
        self.fleet = md.addLConstr(gp.LinExpr(), GRB.LESS_EQUAL, float(self.K))
        self.vlb = md.addLConstr(gp.LinExpr(), GRB.GREATER_EQUAL, 0.0)
        singles = [route_check(self.inst, self.c, [p, d])[0] for p, d in zip(self.inst.pickups, self.inst.deliveries)]
        self.big = 10.0 * float(sum(x for x in singles if x is not None)) + 1.0
        self.art = md.addVar(lb=0.0, obj=self.big, column=gp.Column([1.0] * n, self.cover))
        self.master = md
        for p, d in zip(self.inst.pickups, self.inst.deliveries):
            self.add_column([p, d])

    def column_cost(self, key: tuple[int, ...]) -> tuple[int | None, bool]:
        """(distance, elementary) of a route; an elementary route must pass route_check, an ng-route (a request picked
        up twice, produced by ng pricing) is costed by its arcs and only enters the LP master."""
        if len(set(key)) == len(key):
            return route_check(self.inst, self.c, list(key))[0], True
        return sum(self.c[a][b] for a, b in zip((0,) + key, key + (0,))), False

    def add_column(self, route: list[int]) -> bool:
        key = tuple(int(v) for v in route)
        if key in self.col_index:
            return False
        cost, elem = self.column_cost(key)
        if cost is None:
            return False
        req = self.inst.request_of
        cnt: dict[int, int] = {}
        for v in key:
            if self.inst.is_pickup[v]:
                cnt[req[v]] = cnt.get(req[v], 0) + 1
        reqs = sorted(cnt)
        arcs = list(zip((0,) + key, key + (0,)))
        mask = 0
        for r in reqs:
            mask |= 1 << r
        forbidden = (bool(self.forbid) and any(a in self.forbid for a in arcs)) or bool(mask & self.dive_mask)
        var = self.master.addVar(lb=0.0, ub=0.0 if forbidden else INF, obj=float(cost),
                                 column=gp.Column([float(cnt[r]) for r in reqs] + [1.0, 1.0], [self.cover[r] for r in reqs] + [self.fleet, self.vlb]))
        self.col_index[key] = len(self.cols)
        self.cols.append({"route": key, "cost": cost, "reqs": reqs, "arcs": arcs, "var": var, "elem": elem, "mask": mask})
        return True

    def set_node(self, node: dict) -> None:
        self.forbid = node["forbid"]
        self.fleet.RHS = float(node["hi"])
        self.vlb.RHS = float(node["lo"])
        f = self.forbid
        ubs = [0.0 if f and any(a in f for a in col["arcs"]) else INF for col in self.cols]
        if self.cols:
            self.master.setAttr("UB", [col["var"] for col in self.cols], ubs)

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

    def duals(self) -> tuple[np.ndarray, float, float]:
        pi = np.maximum(np.array(self.master.getAttr("Pi", self.cover)), 0.0)
        return pi, min(float(self.fleet.Pi), 0.0), max(float(self.vlb.Pi), 0.0)

    def compact(self) -> bool:
        """Right after an optimal master solve: drops the columns at zero with the largest reduced costs once the pool
        is large (singletons and the incumbent's routes stay); exact pricing regenerates any column a later LP
        needs. True if columns were removed (the master must then be solved again)."""
        k = len(self.cols)
        limit = max(8000, 150 * self.n)
        if k <= limit:
            return False
        vars_ = [c["var"] for c in self.cols]
        rc = np.array(self.master.getAttr("RC", vars_))
        x = np.array(self.master.getAttr("X", vars_))
        keep = np.zeros(k, dtype=bool)
        keep[np.argsort(rc, kind="stable")[:limit // 2]] = True
        keep |= x > 1e-9
        keep |= np.array([len(c["route"]) == 2 for c in self.cols])
        for r in self.best_plan or []:
            t = self.col_index.get(tuple(r))
            if t is not None:
                keep[t] = True
        drop = np.nonzero(~keep)[0]
        self.master.remove([self.cols[t]["var"] for t in drop])
        self.cols = [self.cols[t] for t in np.nonzero(keep)[0]]
        self.col_index = {c["route"]: t for t, c in enumerate(self.cols)}
        self.stats["removed_columns"] += len(drop)
        self.master.update()
        return len(drop) > 0

    # ------------------------------------------------------------ column generation

    def add_routes(self, routes: list[tuple[float, list[int]]], pi: np.ndarray, const: float) -> int:
        """Adds the routes whose reduced cost at the master duals (pi, route constant const) is negative."""
        added = 0
        req = self.inst.request_of
        for _, route in routes:
            cost, _ = self.column_cost(tuple(route))
            if cost is None:
                continue
            rc = cost + const - sum(pi[req[v]] for v in route if self.inst.is_pickup[v])
            if rc < -self.tol:
                added += self.add_column(route)
        return added

    def price(self, pi: np.ndarray, const: float, exact: bool) -> dict[str, Any]:
        # An exact round gets at most exact_budget seconds; an incomplete round gives columns but no bound.
        deadline = min(self.deadline, time.perf_counter() + self.exact_budget) if exact else self.deadline
        res = self.pricer.price(pi, const, exact=exact, max_cols=self.max_cols, deadline=deadline,
                                forbidden=self.forbid if self.forbid else None, near=self.near, beam=self.beam,
                                tol=self.tol, ng=self.ng if exact else None)
        self.stats["pricing_seconds"] += res["seconds"]
        self.stats["labels"] += res["labels"]
        self.stats["exact_rounds" if exact else "heuristic_rounds"] += 1
        if exact:
            self.stats["exact_seconds"] += res["seconds"]
        return res

    def column_generation(self, node: dict) -> str:
        """CG at a node. Returns "converged", "pruned", "stuck" (exact pricing cannot complete within its budget and
        the widest heuristic pricing finds nothing) or "time". When an exact round is incomplete and finds nothing,
        the heuristic pricing is widened (beam x 2, near + 4) before the node is given up."""
        heur_fail = False
        since_exact = 0
        center = None
        while self.left() > 0:
            if time.perf_counter() >= self.next_primal:
                self.primal_step()
            st = self.solve_lp()
            if st == "infeasible":
                return "pruned"
            if st != "optimal":
                return "time"
            if self.compact():
                continue
            self.stats["cg_iters"] += 1
            node["lp"] = float(self.master.ObjVal)
            pi, mu, nu = self.duals()
            const = -mu - nu
            if not heur_fail and since_exact < self.exact_every:
                res = self.price(pi, const, exact=False)
                since_exact += 1
                if self.add_routes(res["routes"], pi, const):
                    continue
                heur_fail = True
            since_exact = 0
            heur_fail = False
            duals = [(pi, mu, nu)]
            if center is not None and self.alpha > 0:
                a = self.alpha
                duals.insert(0, (a * center[0] + (1 - a) * pi, a * center[1] + (1 - a) * mu, a * center[2] + (1 - a) * nu))
            added, converged = 0, False
            for k, (spi, smu, snu) in enumerate(duals):
                res = self.price(spi, -smu - snu, exact=True)
                if not res["complete"]:
                    self.stats["incomplete_exact"] += 1
                    added += self.add_routes(res["routes"], pi, const)
                    if self.left() <= 0:
                        return "time"
                    if added:
                        break
                    self.exact_every *= 2  # exact rounds are expensive here: price exactly less often
                    if self.beam >= self.max_beam:
                        return "stuck"
                    self.beam, self.near = min(self.max_beam, 2 * self.beam), self.near + 4
                    self.stats["widened"] += 1
                    break
                if self.lagrangian(spi, smu, snu, res["min_rc"], node):
                    center = (spi, smu, snu)
                elif center is None:
                    center = (spi, smu, snu)
                added += self.add_routes(res["routes"], pi, const)
                if self.closed() or self.prunable(node["lb"]):
                    return "pruned"
                if added:
                    break
                if k + 1 < len(duals):
                    self.stats["mispriced"] += 1
                else:
                    converged = True
            if converged:
                return "converged"
        return "time"

    # ------------------------------------------------------------ primal

    def publish(self, plan: list[list[int]], obj: int, stage: str) -> None:
        self.ub, self.best_plan = obj, [list(r) for r in plan]
        if self.first_t is None:
            self.first_t = self.elapsed()
        routes = [list(r) for r in plan]
        self.on_incumbent(routes, stage=stage) if self.hook_takes_stage else self.on_incumbent(routes)
        self.log(stage)
        self.publish_bound(stage)

    def offer(self, plan: list[list[int]], stage: str, polish: bool = True) -> bool:
        """Checks a plan with plan_cost (exact integer arithmetic), publishes it if it improves, then polishes it."""
        plan = [[int(v) for v in r] for r in plan if len(r)]
        obj = plan_cost(self.inst, self.c, plan)
        if obj is None:
            return False
        better = obj < self.ub
        if better:
            self.publish(plan, obj, stage)
        for r in plan:
            self.add_column(r)
        if better and polish and self.left() > 0:
            t = time.perf_counter()
            improved = self.ls.improve(plan, self.deadline)
            self.stats["ls_seconds"] += time.perf_counter() - t
            obj2 = plan_cost(self.inst, self.c, improved)
            if obj2 is not None:
                for r in improved:
                    self.add_column(r)
                if obj2 < self.ub:
                    self.publish(improved, obj2, stage + "+ls")
        return better

    def construct(self) -> None:
        plan = insertion_plan(self.inst, self.c)
        if plan:
            self.offer(plan, "initializer")

    def primal_step(self) -> None:
        """Alternately a restricted-master MIP over the pool and a dive."""
        self.primal_steps += 1
        if self.primal_steps % 2 == 1:
            self.restricted_mip(self.mip_limit, "restricted_mip")
        else:
            self.dive(self.mip_limit)
        self.next_primal = time.perf_counter() + max(self.primal_gap, 0.1 * self.elapsed())

    def dive(self, budget: float, stage: str = "dive") -> None:
        """Diving in the master of the current node: solve the LP, run up to three heuristic pricing rounds, fix the
        elementary columns at >= 0.95 (else the largest fractional one) to 1 and ban every column sharing a request
        with them; repeat until the LP solution is integral (offered as a plan) or its value reaches the incumbent.
        The node's master is restored afterwards."""
        end = min(self.deadline, time.perf_counter() + budget)
        node = self.current or {"forbid": frozenset(), "lo": 0, "hi": self.K}
        fixed: list[int] = []
        t = time.perf_counter()
        self.stats["dives"] += 1
        try:
            rounds = 0
            while time.perf_counter() < end:
                if self.solve_lp() != "optimal" or self.master.ObjVal >= self.ub - 0.5:
                    break
                vars_ = [c["var"] for c in self.cols]
                x = np.array(self.master.getAttr("X", vars_))
                pos = np.nonzero(x > 1e-6)[0]
                if float(self.art.X) < 1e-6 and all(abs(x[k] - 1.0) < 1e-6 and self.cols[k]["elem"] for k in pos):
                    self.offer([list(self.cols[k]["route"]) for k in pos], stage)
                    break
                if rounds < 3:
                    pi, mu, nu = self.duals()
                    res = self.pricer.price(pi, -mu - nu, exact=False, max_cols=self.max_cols, deadline=end,
                                            forbidden=self.forbid if self.forbid else None, near=self.near, beam=self.beam)
                    self.stats["pricing_seconds"] += res["seconds"]
                    rounds += 1
                    if self.add_routes(res["routes"], pi, -mu - nu):
                        continue
                rounds = 0
                cand = [k for k in pos if self.cols[k]["elem"] and k not in fixed]
                if not cand:
                    break
                pick = [k for k in cand if x[k] >= 0.95] or [max(cand, key=lambda k: (x[k], -self.cols[k]["cost"]))]
                for k in pick:
                    if self.cols[k]["mask"] & self.dive_mask:
                        continue
                    self.cols[k]["var"].LB = 1.0
                    fixed.append(k)
                    self.dive_mask |= self.cols[k]["mask"]
                ban = [c["var"] for k, c in enumerate(self.cols) if k not in fixed and c["mask"] & self.dive_mask]
                if ban:
                    self.master.setAttr("UB", ban, [0.0] * len(ban))
        finally:
            self.dive_mask = 0
            for k in fixed:
                if k < len(self.cols):
                    self.cols[k]["var"].LB = 0.0
            self.set_node(node)
            self.stats["dive_seconds"] += time.perf_counter() - t

    def restricted_mip(self, limit: float, stage: str) -> None:
        """Set-partitioning MIP over the whole column pool (fleet row; incumbent as start, cutoff below it)."""
        limit = min(limit, self.left())
        if limit <= 0.2:
            return
        t = time.perf_counter()
        self.stats["mip_runs"] += 1
        md = gp.Model(env=self.env)
        md.Params.TimeLimit = limit
        md.Params.MIPFocus = 1
        md.Params.Seed = self.seed
        rows = [md.addLConstr(gp.LinExpr(), GRB.EQUAL, 1.0) for _ in range(self.n)]
        fleet = md.addLConstr(gp.LinExpr(), GRB.LESS_EQUAL, float(self.K))
        pool = [col for col in self.cols if col["elem"]]
        lam = [md.addVar(vtype=GRB.BINARY, obj=float(col["cost"]),
                         column=gp.Column([1.0] * (len(col["reqs"]) + 1), [rows[r] for r in col["reqs"]] + [fleet]))
               for col in pool]
        if self.best_plan and self.ub < INF:
            md.Params.Cutoff = self.ub - 0.5
            where = {col["route"]: k for k, col in enumerate(pool)}
            pos = [where.get(tuple(r)) for r in self.best_plan]
            if all(p is not None for p in pos):
                for v in lam:
                    v.Start = 0.0
                for p in pos:
                    lam[p].Start = 1.0
        cols = [col["route"] for col in pool]

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                vals = model.cbGetSolution(lam)
                plan = [list(cols[k]) for k, v in enumerate(vals) if v > 0.5]
                obj = plan_cost(self.inst, self.c, plan)
                if obj is not None and obj < self.ub:
                    self.publish(plan, obj, stage)
                    self.pending = plan

        self.pending = None
        md.optimize(callback)
        if md.Status == GRB.MEM_LIMIT:
            self.memory_limit = True
        md.dispose()
        self.stats["mip_seconds"] += time.perf_counter() - t
        if self.pending is not None and self.left() > 0:
            # Local search on the new incumbent; its routes join the pool.
            t2 = time.perf_counter()
            improved = self.ls.improve(self.best_plan, self.deadline)
            self.stats["ls_seconds"] += time.perf_counter() - t2
            obj2 = plan_cost(self.inst, self.c, improved)
            if obj2 is not None:
                for r in improved:
                    self.add_column(r)
                if obj2 < self.ub:
                    self.publish(improved, obj2, stage + "+ls")

    # ------------------------------------------------------------ search

    def branch(self, node: dict) -> list[dict] | None:
        """Children of a converged node, or None if the LP solution has nothing fractional to branch on (an integral
        LP solution is offered as a plan)."""
        vars_ = [c["var"] for c in self.cols]
        x = np.array(self.master.getAttr("X", vars_))
        total = float(x.sum())
        base = {"depth": node["depth"] + 1, "lb": node["lb"], "lo": node["lo"], "hi": node["hi"], "forbid": node["forbid"],
                "rules": node["rules"]}
        if abs(total - round(total)) > 1e-6:
            fl, ce = math.floor(total), math.ceil(total)
            kids = []
            if fl >= node["lo"]:
                kids.append(dict(base, hi=fl, rules=node["rules"] + (("routes<=", fl),)))
            if ce <= node["hi"]:
                kids.append(dict(base, lo=ce, rules=node["rules"] + (("routes>=", ce),)))
            return kids
        flow: dict[tuple[int, int], float] = {}
        for k in np.nonzero(x > 1e-9)[0]:
            for a in self.cols[k]["arcs"]:
                flow[a] = flow.get(a, 0.0) + float(x[k])
        frac = [(abs(f - 0.5), a) for a, f in flow.items() if 1e-6 < f < 1 - 1e-6]
        if frac:
            _, (i, j) = min(frac)
            zero = node["forbid"] | {(i, j)}
            one = set(node["forbid"])
            if i:
                one |= {(i, k) for k in self.succ[i] if k != j}
            if j:
                one |= {(k, j) for k in self.pred[j] if k != i}
            return [dict(base, forbid=frozenset(one), rules=node["rules"] + (("arc", i, j, 1),)),
                    dict(base, forbid=frozenset(zero), rules=node["rules"] + (("arc", i, j, 0),))]
        pos = np.nonzero(x > 1e-9)[0]
        if float(self.art.X) < 1e-6 and all(abs(x[k] - 1.0) < 1e-6 and self.cols[k]["elem"] for k in pos):
            self.offer([list(self.cols[k]["route"]) for k in pos], "integral_node")
        return None

    def search(self) -> None:
        T = self.time_limit
        self.mip_limit = max(3.0, min(60.0, 0.03 * T))
        self.primal_gap = max(3.0, min(60.0, 0.03 * T))
        self.next_primal = time.perf_counter() + self.primal_gap
        self.primal_steps = 0
        self.searching = True
        root = {"id": 0, "depth": 0, "lb": -INF, "lo": 0, "hi": self.K, "forbid": frozenset(), "rules": ()}
        self.frontier = [root]
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
            status = self.column_generation(node)
            is_root = node["id"] == 0
            if is_root:
                self.stats["root_converged"] = status == "converged"
                self.stats["root_lp"] = node.get("lp")
                self.stats["root_bound"] = node["lb"] if node["lb"] > -INF else None
                self.stats["root_seconds"] = self.elapsed()
                self.log("root")
            if status == "time":
                self.frontier.append(node)
                break
            if status == "pruned":
                self.current = None
                self.publish_bound("prune")
                continue
            if status == "stuck":
                node["stuck"] = True
                self.stats["stuck_nodes"] += 1
                self.frontier.append(node)
                self.current = None
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
            if is_root or time.perf_counter() >= self.next_primal:
                self.primal_step()
        self.current = None
        while self.left() > 0.5 and not self.closed():
            before = self.ub
            self.restricted_mip(self.left(), "restricted_mip_final")
            if self.ub >= before:
                break
