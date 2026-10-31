"""Engineered branch-and-price reference for bin packing with conflicts (scale-up study).

Every branch-and-price node is itself a BPPC instance (Sadykov and Vanderbeck 2013, Ryan-Foster branching through the
conflict graph): a "same" decision merges two items into one super-item (weights added, conflicts united), a
"differ" decision adds a conflict between them. Node master (set covering over the node's super-items; bins are
closed under taking subsets, so covering and partitioning have the same LP optimum and integer bins):
    min sum_S lam_S  s.t.  sum_{S containing g} lam_S >= 1   (pi_g >= 0)   every super-item g.
Columns live in one global pool as sets of original items; a pool bin enters a node's master iff it holds every
member of a merged super-item or none, and no differ pair (every such bin is a feasible bin of the node).

Pricing (pricing.py): greedy heuristic rounds first (several seeds, add / swap local search, up to max_cols bins per
round); an exact round when a heuristic round finds no improving bin, and every 10 rounds. The exact oracle is
SmallBinOracle (enumeration) when the bins of four or more items are few, as in every triplet instance, and otherwise
the clique-row KPC MIP with a time cap per round (initially max(1 s, 0.5% of T), doubled whenever a capped round
finds nothing; a capped round still yields Gurobi's bound on the pricing optimum, so its Farley bound stays valid).
The MIP oracle of a node uses the root's edge clique cover mapped to super-items (a clique stays a clique under
merging) plus the differ pairs.

Stabilization: Wentges smoothing; pricing at alpha * center + (1 - alpha) * master duals, the center being the duals
of the best Farley bound at the node; when no bin is improving at the master duals (mispricing) the round is repeated
at the master duals. Convergence needs a complete exact round at the master duals that finds nothing improving.

Lower bounds. Every exact round (also one cut short: the oracle's bound is still an upper bound v on the pricing
optimum) gives Farley's bound sum_g pi_g / max(1, v) at the duals pi it priced: pi / v is dual feasible for the full
node master (pi >= 0, every bin has value <= v). It is rounded up (integral bin count, 1e-6 safety). Early
termination: once the node bound reaches ceil(z_RMP), more columns cannot raise the integer bound, so the node is
pruned or branched. The global bound is the minimum over the open nodes (stuck nodes included).

Branching: Ryan-Foster on the super-item pair whose joint coverage in the master solution is closest to 1/2; the
"same" child first; best-first search on the bound, ties deeper first (a depth-first dive on equal bounds).

Primal side: the shared basic initializer (first fit decreasing with conflicts); after the root, the main primal
effort (up to 25% of T for the pure dive, 35% with the discrepancy dives): a diving heuristic with limited discrepancy
search (DH-LDS of Sadykov and Vanderbeck: fix the bin with the largest LP value, or every bin at >= 0.99, ban its
items from pricing, re-optimize, repeat; then dives that force the first d choices of the pure dive and forbid choice
d, d < 3; each dive step prices heuristically, with exact rounds when the heuristic finds nothing: one capped MIP
round, or up to two enumeration rounds) and a restricted-master MIP over the node's columns (incumbent as start,
cutoff below it); between pricing rounds, alternately dives and restricted MIPs. A dive cut short is completed by
first fit on the remaining items. Every plan is trimmed to a partition and re-checked with bin_count before it is
published.

termination_reason: "optimal" (bound = incumbent), "time_limit", "memory_limit", or "stuck" (stopped before the
deadline with only stuck nodes left: nodes whose converged master has no fractional pair and no integral solution).
"""
from __future__ import annotations

import inspect
import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.cg.bppc.a_data import BPPCInstance, load_bppc
from src.cg.bppc.b_mip_xl import bin_count, public
from src.cg.bppc.heuristic import basic_initializer, edge_clique_cover, first_fit, greedy_clique, trim_cover
from src.cg.bppc.pricing import KPCOracle, SmallBinOracle, greedy_bins, large_bins

INF = float("inf")


class NodeView:
    """A node as a BPPC instance over super-items (see the module docstring)."""

    def __init__(self, bp: "BPPCBranchAndPrice", node: dict) -> None:
        n = bp.n
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in node["same"]:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        roots = sorted({find(i) for i in range(n)})
        index = {r: g for g, r in enumerate(roots)}
        self.gid = np.array([index[find(i)] for i in range(n)], dtype=np.int64)
        m = len(roots)
        self.m = m
        self.members: list[list[int]] = [[] for _ in range(m)]
        for i in range(n):
            self.members[self.gid[i]].append(i)
        self.W = np.bincount(self.gid, weights=bp.w, minlength=m).astype(np.int64)
        self.feasible = bool((self.W <= bp.C).all())
        if m == n:
            adj = bp.adj.copy()
        else:
            M = np.zeros((n, m), dtype=np.float32)
            M[np.arange(n), self.gid] = 1.0
            S = M.T @ bp.adj.astype(np.float32) @ M
            if (np.diag(S) > 0.5).any():
                self.feasible = False  # a merged super-item holds a conflicting pair
            adj = S > 0.5
        for a, b in node["differ"]:
            g, h = int(self.gid[a]), int(self.gid[b])
            if g == h:
                self.feasible = False
            adj[g, h] = adj[h, g] = True
        np.fill_diagonal(adj, False)
        self.adj = adj
        self.merged = [sum(1 << i for i in mem) for mem in self.members if len(mem) > 1]
        self.differ = [(1 << a, 1 << b) for a, b in node["differ"]]
        cl = set()
        for K in bp.cliques:
            gs = tuple(sorted({int(self.gid[i]) for i in K}))
            if len(gs) >= 2:
                cl.add(gs)
        for a, b in node["differ"]:
            g, h = int(self.gid[a]), int(self.gid[b])
            if g != h:
                cl.add((min(g, h), max(g, h)))
        self.cliques = [list(K) for K in cl]

    def valid(self, mask: int) -> bool:
        for gm in self.merged:
            x = mask & gm
            if x and x != gm:
                return False
        return not any(mask & a and mask & b for a, b in self.differ)

    def to_super(self, key) -> np.ndarray:
        return np.unique(self.gid[np.asarray(key, dtype=np.int64)])

    def expand(self, groups) -> tuple[int, ...]:
        return tuple(sorted(i for g in groups for i in self.members[int(g)]))


class BPPCBranchAndPrice:
    """Engineered branch-and-price reference for BPPC; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 heuristic_pricing: bool = True, max_cols: int = 20, alpha: float = 0.5, dive: bool = True,
                 restricted_mip: bool = True, early_termination: bool = True, initializer: bool = True,
                 lds_depth: int = 3, dive_exact: bool = True, tol: float = 1e-6, verbose: bool = False) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.on_incumbent = on_incumbent or (lambda bins: None)  # receives bins as lists of items 1..n
        try:  # hooks that accept a `stage` keyword also receive the incumbent's provenance
            params = inspect.signature(self.on_incumbent).parameters
            self.hook_takes_stage = "stage" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())
        except (TypeError, ValueError):
            self.hook_takes_stage = False
        self.on_bound = on_bound or (lambda bound: None)
        self.heuristic_pricing, self.max_cols, self.alpha = heuristic_pricing, max_cols, alpha
        self.diving, self.use_rmip, self.early = dive, restricted_mip, early_termination
        self.initializer, self.lds_depth, self.tol, self.verbose = initializer, lds_depth, tol, verbose
        self.dive_exact = dive_exact
        self.trace: list[dict[str, Any]] = []

    # ------------------------------------------------------------ setup

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        return self.solve(load_bppc(data_path))

    def left(self) -> float:
        return self.deadline - time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def log(self, stage: str) -> None:
        self.trace.append({"t": round(self.elapsed(), 3), "incumbent": self.ub if self.ub < INF else None,
                           "bound": self.lb if self.lb > -INF else None, "stage": stage})
        if self.verbose:
            print(self.trace[-1], flush=True)

    def solve(self, inst: BPPCInstance) -> dict[str, Any]:
        if not hasattr(self, "t0"):
            self.t0 = time.perf_counter()
            self.deadline = self.t0 + self.time_limit
        self.inst, self.n = inst, inst.n
        self.w, self.C, self.adj = inst.weights.astype(np.int64), int(inst.capacity), inst.adj
        self.ub, self.lb = INF, -INF
        self.best_bins: list[list[int]] | None = None
        self.pool: dict[tuple[int, ...], dict[str, int]] = {}
        self.stamp = 0
        self.stats = {"cg_iters": 0, "exact_rounds": 0, "heuristic_rounds": 0, "incomplete_exact": 0, "mispricings": 0,
                      "early_terminations": 0, "lp_seconds": 0.0, "pricing_seconds": 0.0, "exact_seconds": 0.0,
                      "mip_runs": 0, "mip_seconds": 0.0, "dives": 0, "dive_seconds": 0.0, "nodes": 0, "max_depth": 0,
                      "stuck_nodes": 0, "infeasible_nodes": 0, "root_converged": False, "root_bound": None,
                      "root_lp": None, "root_seconds": None, "first_incumbent_seconds": None, "cliques": 0}
        self.frontier: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.memory_limit = False
        self.exact_cap = max(1.0, 0.005 * self.time_limit)
        self.root = {"id": 0, "depth": 0, "lb": -INF, "same": (), "differ": (), "order": 0}
        self.frontier = [self.root]
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", self.threads)
        env.setParam("Seed", self.seed)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()
        self.env = env
        self.lp = self.oracle = None
        try:
            self.cliques = edge_clique_cover(self.adj)
            self.stats["cliques"] = len(self.cliques)
            self.trivial_bound()
            if self.initializer:
                self.offer(basic_initializer(inst), "initializer")
            self.search()
        except gp.GurobiError as err:
            if err.errno != GRB.Error.OUT_OF_MEMORY and "memory" not in str(err).lower():
                raise
            self.memory_limit = True
        finally:
            self.dispose_node()
        reason = ("memory_limit" if self.memory_limit else "optimal" if self.closed()
                  else "time_limit" if self.left() <= 0.5 else "stuck")
        out = self.result(reason)
        env.dispose()
        return out

    def closed(self) -> bool:
        return self.ub < INF and self.lb >= self.ub

    def prunable(self, bound: float) -> bool:
        return bound >= self.ub  # integral objective: no strictly better plan below a bound >= ub

    def result(self, reason: str) -> dict[str, Any]:
        ub = float(self.ub) if self.ub < INF else None
        lb = float(self.lb) if self.lb > -INF else None
        return {"best_objective": ub, "global_lb": lb, "global_ub": ub,
                "gap": (ub - lb) / ub if ub and lb is not None else None, "termination_reason": reason,
                "bins": public(self.best_bins) if self.best_bins else None, "pool_columns": len(self.pool),
                **self.stats, "total_seconds": self.elapsed(), "trace": self.trace}

    # ------------------------------------------------------------ bounds and incumbents

    def publish_bound(self, stage: str) -> None:
        nodes = list(self.frontier) + ([self.current] if self.current is not None else [])
        lb = min((x["lb"] for x in nodes), default=INF)
        if self.ub < INF:
            lb = min(lb, self.ub)
        if math.isfinite(lb) and lb > self.lb:
            self.lb = lb
            self.on_bound(float(lb))
            self.log(stage)

    def trivial_bound(self) -> None:
        """ceil(w(N) / C) and the size of a greedy maximal clique of the conflict graph."""
        self.root["lb"] = float(max(self.inst.weight_bound, len(greedy_clique(self.adj, self.w))))
        self.publish_bound("trivial")

    def offer(self, bins: list[list[int]], stage: str) -> bool:
        """Trims a covering plan (internal items) to a partition, checks it with bin_count and publishes it if it
        improves; its bins join the pool."""
        bins = trim_cover(bins)
        obj = bin_count(self.inst, public(bins))
        if obj is None:
            return False
        for b in bins:
            self.pool_add(b)
        if obj >= self.ub or time.perf_counter() >= self.deadline:  # never publish after the deadline
            return False
        self.ub, self.best_bins = float(obj), bins
        if self.stats["first_incumbent_seconds"] is None:
            self.stats["first_incumbent_seconds"] = self.elapsed()
        if self.hook_takes_stage:
            self.on_incumbent(public(bins), stage=stage)
        else:
            self.on_incumbent(public(bins))
        self.log(stage)
        self.publish_bound(stage)
        return True

    # ------------------------------------------------------------ column pool and node master

    def pool_add(self, items) -> tuple[int, ...] | None:
        key = tuple(sorted(int(i) for i in items))
        if not key:
            return None
        if key not in self.pool:
            self.pool[key] = {"mask": sum(1 << i for i in key), "stamp": self.stamp}
        return key

    def dispose_node(self) -> None:
        for obj in (self.lp, self.oracle):
            if obj is not None:
                obj.dispose()
        self.lp = self.oracle = None

    def open_node(self, nv: NodeView) -> None:
        """The node's master LP over the valid pool bins (the most recently useful ones if there are many) and its
        exact pricing oracle."""
        self.dispose_node()
        md = gp.Model(env=self.env)
        md.Params.Method = 0  # primal simplex: the basis stays primal feasible when columns are added
        self.rows = [md.addLConstr(gp.LinExpr(), GRB.GREATER_EQUAL, 1.0) for _ in range(nv.m)]
        self.lp, self.nv = md, nv
        self.lp_cols: list[dict[str, Any]] = []
        self.lp_index: dict[tuple[int, ...], int] = {}
        for g in range(nv.m):
            self.lp_column(self.pool_add(nv.members[g]), np.array([g]))
        valid = [k for k, e in self.pool.items() if k not in self.lp_index and nv.valid(e["mask"])]
        cap = max(5000, 20 * nv.m)
        if len(valid) > cap:
            valid = sorted(valid, key=lambda k: -self.pool[k]["stamp"])[:cap]
        for k in valid:
            self.lp_column(k, nv.to_super(k))
        # Exact oracle: enumeration when bins of four or more items are few (triplet instances), else the
        # clique-row KPC MIP.
        self.fast_exact = large_bins(nv.W, self.C, nv.adj) is not None
        self.oracle = (SmallBinOracle(nv.W, self.C, nv.adj) if self.fast_exact
                       else KPCOracle(nv.W, self.C, nv.adj, nv.cliques, self.env, tol=self.tol))

    def lp_column(self, key: tuple[int, ...], groups: np.ndarray) -> bool:
        if key in self.lp_index:
            return False
        var = self.lp.addVar(lb=0.0, obj=1.0, column=gp.Column([1.0] * len(groups), [self.rows[g] for g in groups]))
        self.lp_index[key] = len(self.lp_cols)
        self.lp_cols.append({"key": key, "groups": np.asarray(groups, dtype=np.int64), "var": var})
        return True

    def add_groups(self, groups) -> bool:
        """A priced bin (super-items of the current node) joins the pool and the node master."""
        groups = np.unique(np.asarray(groups, dtype=np.int64))
        return self.lp_column(self.pool_add(self.nv.expand(groups)), groups)

    def solve_lp(self) -> str:
        if self.left() <= 0:
            return "time"
        t = time.perf_counter()
        self.lp.Params.TimeLimit = max(1e-3, self.left())
        self.lp.optimize()
        self.stats["lp_seconds"] += time.perf_counter() - t
        st = self.lp.Status
        if st == GRB.MEM_LIMIT:
            raise gp.GurobiError(GRB.Error.OUT_OF_MEMORY, "memory limit")
        return "optimal" if st == GRB.OPTIMAL else "time"

    def lp_values(self) -> np.ndarray:
        return np.array(self.lp.getAttr("X", [c["var"] for c in self.lp_cols]))

    def duals(self) -> np.ndarray:
        return np.maximum(np.array(self.lp.getAttr("Pi", self.rows)), 0.0)

    # ------------------------------------------------------------ pricing

    def price(self, pi: np.ndarray, exact: bool, banned: np.ndarray | None = None) -> dict[str, Any]:
        t = time.perf_counter()
        thr = 1.0 + self.tol
        if exact:
            p = pi if banned is None else np.where(banned, 0.0, pi)
            # Exact rounds have a time cap (Gurobi's bound keeps a capped round's Farley bound valid); a capped round
            # that finds no improving bin doubles the cap, so convergence is still reached.
            cap = self.deadline if self.fast_exact else min(self.deadline, t + self.exact_cap)
            res = self.oracle.solve(p, threshold=thr, max_cols=self.max_cols, deadline=cap)
            if not res["complete"] and not res["cols"]:
                self.exact_cap = min(2 * self.exact_cap, max(self.time_limit, 1.0))
            self.stats["exact_rounds"] += 1
            self.stats["incomplete_exact"] += not res["complete"]
            self.stats["exact_seconds"] += time.perf_counter() - t
        else:
            res = {"cols": greedy_bins(pi, self.nv.W, self.C, self.nv.adj, threshold=thr, max_cols=self.max_cols,
                                       banned=banned), "complete": False, "bound": None}
            self.stats["heuristic_rounds"] += 1
        self.stats["pricing_seconds"] += time.perf_counter() - t
        return res

    def farley(self, node: dict, pi: np.ndarray, res: dict, stage: str) -> None:
        """Farley's bound at the priced duals pi from an exact round's upper bound on the pricing optimum."""
        if res.get("bound") is None:
            return
        v = max(1.0, float(res["bound"]) * (1 + 1e-9) + 1e-9)
        val = float(pi.sum()) / v
        if val > node.get("center_val", -INF):
            node["center_val"] = val
            self.center = pi.copy()
        lb = float(math.ceil(val - 1e-6))
        if lb > node["lb"]:
            node["lb"] = lb
            self.publish_bound(stage)

    def improving(self, cols: list, pi: np.ndarray) -> list:
        thr = 1.0 + self.tol
        return [g for _, g in cols if float(pi[np.asarray(g)].sum()) > thr
                and self.nv.expand(g) not in self.lp_index]

    def column_generation(self, node: dict, stage: str, early: bool = True) -> str:
        """Column generation at the current node: "converged", "early" (the bound reached ceil(z_RMP)),
        "pruned" or "time"."""
        heuristic_fail, since_exact = False, 0
        self.center = None
        node["center_val"] = -INF
        while self.left() > 0:
            if self.primal_due():
                self.primal_step(node)
            if self.solve_lp() != "optimal":
                return "time"
            self.stats["cg_iters"] += 1
            self.stamp += 1
            z = float(self.lp.ObjVal)
            node["lp"] = z
            x = self.lp_values()
            for k in np.flatnonzero(x > 1e-9).tolist():
                self.pool[self.lp_cols[k]["key"]]["stamp"] = self.stamp
            pi = self.duals()
            exact = not self.heuristic_pricing or heuristic_fail or since_exact >= 10
            sep = pi if self.center is None or self.alpha <= 0 else self.alpha * self.center + (1 - self.alpha) * pi
            res = self.price(sep, exact)
            if exact:
                self.farley(node, sep, res, stage)
            cands = [g for _, g in res["cols"]]
            neg = self.improving(res["cols"], pi)
            complete_at_pi = exact and res["complete"] and sep is pi
            if sep is not pi and not neg:  # mispricing: repeat the round at the master duals
                self.stats["mispricings"] += 1
                res = self.price(pi, exact)
                if exact:
                    self.farley(node, pi, res, stage)
                cands += [g for _, g in res["cols"]]
                neg = self.improving(res["cols"], pi)
                complete_at_pi = exact and res["complete"]
            for g in cands:
                self.add_groups(g)
            if exact:
                since_exact, heuristic_fail = 0, False
            else:
                since_exact += 1
                heuristic_fail = not neg
            if self.closed() or self.prunable(node["lb"]):
                return "pruned"
            if early and self.early and node["lb"] >= math.ceil(z - 1e-6):
                self.stats["early_terminations"] += 1
                return "early"
            if complete_at_pi and not neg:
                return "converged"
        return "time"

    # ------------------------------------------------------------ primal heuristics

    def primal_due(self) -> bool:
        return (self.diving or self.use_rmip) and time.perf_counter() >= self.next_primal

    def primal_step(self, node: dict) -> None:
        """Between pricing rounds: alternately a dive and a restricted-master MIP."""
        self.primal_steps += 1
        if self.diving and (self.primal_steps % 2 == 0 or not self.use_rmip):
            self.dive(self.dive_budget, "dive")
        elif self.use_rmip:
            self.restricted_mip(self.mip_limit, "restricted_mip")
        self.next_primal = time.perf_counter() + max(self.primal_gap, 0.1 * self.elapsed())

    def complete_partial(self, chosen: list[tuple[int, ...]], stage: str) -> None:
        """A partial dive (fixed bins) completed by first fit decreasing on the remaining items."""
        if not chosen:
            return
        bins = trim_cover([list(k) for k in chosen])
        done = np.zeros(self.n, dtype=bool)
        for b in bins:
            done[b] = True
        rest = np.flatnonzero(~done)
        if len(rest):
            sub = BPPCInstance("rest", self.C, self.w[rest], self.adj[np.ix_(rest, rest)])
            order = np.lexsort((np.arange(len(rest)), -sub.weights))
            bins += [[int(rest[i]) for i in b] for b in first_fit(sub, order)]
        self.offer(bins, stage)

    def dive(self, budget: float, stage: str, forced: tuple = (), tabu: frozenset = frozenset()) -> list:
        """Diving on the current node master (see the module docstring); returns the sequence of fixed bins. The
        fixings are undone afterwards; generated columns stay."""
        until = min(self.deadline, time.perf_counter() + budget)
        t = time.perf_counter()
        self.stats["dives"] += 1
        nv = self.nv
        fixed: list[gp.Var] = []
        blocked: list[gp.Var] = []
        banned = np.zeros(nv.m, dtype=bool)
        chosen: list[tuple[int, ...]] = []
        finished = False

        def fix(k: int) -> None:
            col = self.lp_cols[k]
            col["var"].LB = 1.0
            fixed.append(col["var"])
            banned[col["groups"]] = True
            chosen.append(col["key"])

        self.lp.Params.Method = 1  # bound changes: dual simplex
        try:
            for key in tabu:
                k = self.lp_index.get(key)
                if k is not None and len(self.lp_cols[k]["groups"]) > 1:
                    self.lp_cols[k]["var"].UB = 0.0
                    blocked.append(self.lp_cols[k]["var"])
            for key in forced:
                k = self.lp_index.get(key)
                if k is None or banned[self.lp_cols[k]["groups"]].any():
                    return chosen
                fix(k)
            while time.perf_counter() < until:
                exact_left = 2 if self.fast_exact else 1
                for _ in range(12 if self.fast_exact else 6):  # heuristic rounds; exact ones when they find nothing
                    if self.solve_lp() != "optimal" or time.perf_counter() >= until:
                        return chosen
                    pi = self.duals()
                    res = self.price(pi, exact=False, banned=banned)
                    if not self.improving(res["cols"], pi) and self.dive_exact and exact_left > 0:
                        exact_left -= 1
                        res = self.price(pi, exact=True, banned=banned)
                    if sum(self.add_groups(g) for _, g in res["cols"]) == 0:
                        break
                if self.solve_lp() != "optimal":
                    return chosen
                if self.prunable(math.ceil(float(self.lp.ObjVal) - 1e-6)):
                    finished = True
                    return chosen
                x = self.lp_values()
                pos = np.flatnonzero(x > 1e-6)
                if (np.abs(x[pos] - 1.0) < 1e-6).all():
                    self.offer([list(self.lp_cols[k]["key"]) for k in pos], stage)
                    finished = True
                    return chosen
                cand = [k for k in pos.tolist() if self.lp_cols[k]["var"].LB < 0.5
                        and not banned[self.lp_cols[k]["groups"]].any()]
                if not cand:
                    return chosen
                sel = [k for k in cand if x[k] >= 0.99] or [max(cand, key=lambda k: x[k])]
                for k in sorted(sel, key=lambda k: -x[k]):
                    if not banned[self.lp_cols[k]["groups"]].any():
                        fix(k)
        finally:
            for v in fixed:
                v.LB = 0.0
            for v in blocked:
                v.UB = INF
            self.lp.Params.Method = 0
            if not finished:
                self.complete_partial(chosen, stage + "_ff")
            self.stats["dive_seconds"] += time.perf_counter() - t
        return chosen

    def lds(self, budget: float, stage: str, first_budget: float | None = None) -> None:
        """Limited discrepancy search around the pure dive: dive d forces the first d choices of the pure dive and
        forbids its choice d (d < lds_depth)."""
        until = time.perf_counter() + budget
        first = self.dive(budget if first_budget is None else first_budget, stage)
        for d in range(min(self.lds_depth, len(first))):
            if time.perf_counter() >= until or self.closed():
                break
            self.dive(until - time.perf_counter(), stage + "_lds", forced=tuple(first[:d]), tabu=frozenset([first[d]]))

    def restricted_mip(self, limit: float, stage: str, keys: list | None = None) -> None:
        """Integer covering master over the given bins (default: the node master's bins and the incumbent's), with
        the incumbent as MIP start and a cutoff below it; every solution is trimmed and offered."""
        limit = min(limit, self.left())
        if limit <= 0.2:
            return
        t = time.perf_counter()
        self.stats["mip_runs"] += 1
        keys = list(keys if keys is not None else [c["key"] for c in self.lp_cols])
        inc = [tuple(sorted(b)) for b in (self.best_bins or [])]
        have = set(keys)
        keys += [k for k in inc if k not in have]
        md = gp.Model(env=self.env)
        md.Params.TimeLimit = limit
        md.Params.MIPFocus = 1
        lam = list(md.addVars(len(keys), vtype=GRB.BINARY, obj=1.0).values())
        rows = [gp.LinExpr() for _ in range(self.n)]
        for v, k in zip(lam, keys):
            for i in k:
                rows[i].addTerms(1.0, v)
        for i in range(self.n):
            md.addLConstr(rows[i], GRB.GREATER_EQUAL, 1.0)
        if self.ub < INF:
            md.Params.Cutoff = self.ub - 0.5
            start = set(inc)
            for v, k in zip(lam, keys):
                v.Start = 1.0 if k in start else 0.0
        deadline = time.perf_counter() + limit

        def callback(model, where):
            if time.perf_counter() > deadline or time.perf_counter() > self.deadline:
                model.terminate()
                return
            if where == GRB.Callback.MIPSOL:
                vals = model.cbGetSolution(lam)
                self.offer([list(k) for k, val in zip(keys, vals) if val > 0.5], stage)

        md.optimize(callback)
        if md.Status == GRB.MEM_LIMIT:
            self.memory_limit = True
        md.dispose()
        self.stats["mip_seconds"] += time.perf_counter() - t

    # ------------------------------------------------------------ search

    def branch(self, node: dict, x: np.ndarray) -> list[dict] | None:
        """Ryan-Foster children on the super-item pair with joint coverage closest to 1/2; None if no pair is
        fractional."""
        nv = self.nv
        S = np.zeros((nv.m, nv.m))
        for k in np.flatnonzero(x > 1e-9).tolist():
            g = self.lp_cols[k]["groups"]
            S[np.ix_(g, g)] += x[k]
        T = np.triu(S, 1)
        frac = np.where((T > 1e-6) & (T < 1 - 1e-6), np.abs(T - 0.5), INF)
        if not np.isfinite(frac).any():
            return None
        g, h = np.unravel_index(np.argmin(frac), frac.shape)
        a, b = nv.members[int(g)][0], nv.members[int(h)][0]
        base = {"depth": node["depth"] + 1, "lb": node["lb"], "same": node["same"], "differ": node["differ"]}
        return [dict(base, same=node["same"] + ((a, b),), order=0),
                dict(base, differ=node["differ"] + ((a, b),), order=1)]

    def search(self) -> None:
        """Best-first branch-and-price (ties: deeper first, then the "same" child), primal steps between rounds."""
        T = self.time_limit
        self.mip_limit = max(3.0, min(60.0, 0.03 * T))
        self.primal_gap = max(3.0, min(60.0, 0.03 * T))
        self.dive_budget = max(3.0, min(120.0, 0.05 * T))
        self.primal_steps = 0
        self.next_primal = time.perf_counter() + self.primal_gap
        count = 0
        while self.left() > 0 and not self.closed():
            live = [x for x in self.frontier if not x.get("stuck")]
            if not live:
                break
            node = min(live, key=lambda x: (x["lb"], -x["depth"], x["order"], x["id"]))
            self.frontier.remove(node)
            self.current = node
            if self.prunable(node["lb"]):
                self.current = None
                self.publish_bound("prune")
                continue
            nv = NodeView(self, node)
            if not nv.feasible:
                self.stats["infeasible_nodes"] += 1
                self.current = None
                self.publish_bound("infeasible_node")
                continue
            self.stats["nodes"] += 1
            self.stats["max_depth"] = max(self.stats["max_depth"], node["depth"])
            root = node["id"] == 0
            self.open_node(nv)
            status = self.column_generation(node, "root" if root else "node")
            if root:
                self.stats["root_converged"] = status == "converged"
                self.stats["root_lp"] = node.get("lp")
                self.stats["root_bound"] = node["lb"]
                self.stats["root_seconds"] = self.elapsed()
            kids = None
            if status in ("converged", "early"):
                # Re-solve first: the last pricing round may have added columns after the last LP solve.
                status = status if self.solve_lp() == "optimal" else "time"
            if status in ("converged", "early"):
                x = self.lp_values()
                kids = self.branch(node, x)
                if kids is None and status == "early":  # no fractional pair yet: finish the node's LP
                    status = self.column_generation(node, "node_full", early=False)
                    if status in ("converged", "early"):
                        status = status if self.solve_lp() == "optimal" else "time"
                    if status in ("converged", "early"):
                        x = self.lp_values()
                        kids = self.branch(node, x)
                if kids is None and status == "converged":
                    pos = np.flatnonzero(x > 1e-9)
                    if (np.abs(x[pos] - 1.0) < 1e-6).all():  # integral master optimum: the node is solved
                        self.offer([list(self.lp_cols[k]["key"]) for k in pos], "integral_node")
                        node["lb"] = max(node["lb"], float(math.ceil(float(self.lp.ObjVal) - 1e-6)))
            if status == "time":
                self.frontier.append(node)
                break
            if status != "pruned" and not self.prunable(node["lb"]):
                if kids is None:
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
            if root and not self.closed() and self.left() > 0:
                if self.diving:  # the main primal effort: a long pure dive, then the discrepancy dives
                    self.lds(max(2 * self.dive_budget, 0.35 * self.time_limit), "dive",
                             first_budget=max(self.dive_budget, 0.25 * self.time_limit))
                if self.use_rmip and not self.closed():
                    self.restricted_mip(self.mip_limit, "restricted_mip")
                self.next_primal = time.perf_counter() + max(self.primal_gap, 0.1 * self.elapsed())
            elif self.primal_due() and not self.closed():
                self.primal_step(node)
        self.current = None
        if self.use_rmip:
            while self.left() > 0.5 and not self.closed():
                before = self.ub
                keys = sorted(self.pool, key=lambda k: -self.pool[k]["stamp"])[:max(5000, 20 * self.n)]
                self.restricted_mip(self.left(), "restricted_mip_final", keys=keys)
                if self.ub >= before:
                    break
