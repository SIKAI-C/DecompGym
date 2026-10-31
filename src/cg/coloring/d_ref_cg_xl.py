"""Engineered branch-and-price reference for vertex coloring (scale-up study).

Master (Mehrotra and Trick 1996): set covering over independent sets S of G,
    min sum_S lam_S   s.t.  sum_{S contains v} lam_S >= 1  (pi_v >= 0) for every vertex v,   lam >= 0.
Covering relaxes partitioning without changing the integer optimum (subsets of independent sets are independent);
an integer covering is turned into a coloring by giving every vertex the first chosen set containing it.

Branching (Ryan and Foster, as graph modifications; Mehrotra and Trick 1996, Malaguti, Monaci and Toth 2011): a node
carries "same" pairs (u, v): contract u and v into one vertex, and "differ" pairs: add the edge uv. The node graph B
has one vertex per group of contracted vertices; its independent sets, expanded to the original vertices, are exactly
the columns compatible with the node (they contain both or neither vertex of every same pair and never both of a
differ pair); incompatible columns get an upper bound of 0. Each group's own set is always a compatible column, so
every node master is feasible. The branching pair (two non-adjacent groups g, h) maximizes
min(sum of lam over columns containing both, sum over columns containing exactly one) at the node's LP solution, so
both children cut it off; the "same" child is explored first, nodes best-first on the bound (ties: deeper first).

Pricing: a maximum-weight independent set of B for the group weights (sum of pi over the group). Heuristic rounds
first (greedy constructions under several weight orders and seeds, (1,1)-swaps, maximal extension; up to max_cols
columns per round); an exact round when a heuristic round finds nothing or every exact_every rounds: a Gurobi MIP over
the clique rows of the root edge clique cover (mapped through the contractions; a clique never contains a contracted
pair because contracted vertices are non-adjacent) plus the differ edges, with Cutoff 1 + tol; it stops early once
it has found improving sets, and its best bound w_ub is kept.

Bounds (Farley 1990; the Lagrangian bound of the set-covering master): for every pi >= 0 and every upper bound w_ub on
the maximum weight of an independent set of B, every coloring in the node uses at least sum pi / max(1, w_ub) colors
(its classes split sum pi). Every exact round contributes this bound (made safe against solver tolerances and rounded
up); the initial bound is the size of a greedy clique. Early termination: column generation stops at a node as soon
as its bound reaches ceil(restricted-master value), since the LP bound cannot round higher. The global bound is the
minimum over the open nodes.

Stabilization: Wentges smoothing in exact rounds: pricing at alpha * center + (1 - alpha) * master duals (center = the
duals with the best Farley bound at the node); a mispricing (no set improving at the master duals) repeats the round
at the master duals.

Primal side: the shared basic initializer (DSATUR); an MMT-style start (Malaguti, Monaci and Toth 2008/2011): TabuCol
on k - 1 colors for a share of the budget (option tabu), whose colorings and conflict-free classes met during the
search (harvested every 5000 iterations, made maximal, pool capped at max_pool columns) seed the master, followed by
a set-covering MIP over that pool; then, between pricing rounds, cyclically a restricted-master MIP over all columns
(cutoff below the incumbent), a dive (fix the columns at >= 0.99, else the largest, re-optimize with heuristic
pricing, until the LP is integral) and a resumed TabuCol attempt at one color fewer than the incumbent; a dive and a
restricted MIP right after the root. Every coloring is checked with
color_count before it is published; its classes (made maximal) join the master.
"""
from __future__ import annotations

import inspect
import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from src.cg.coloring.a_data import ColoringInstance, load_col
from src.cg.coloring.b_mip_xl import color_count
from src.cg.coloring.c_naive_cg import farley_bound
from src.cg.coloring.heuristic import (TabuCol, canonical, classes_of, coloring_from_cover, dsatur, edge_clique_cover,
                                       greedy_clique, maximal_extension, weighted_greedy_sets)

INF = float("inf")


class ColoringBranchAndPrice:
    """Engineered branch-and-price reference for vertex coloring; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 tabu: bool = True, tabu_share: float = 0.1, alpha: float = 0.5, heuristic_pricing: bool = True,
                 max_cols: int = 30, exact_every: int = 25, dive: bool = True, rmp: bool = True, tol: float = 1e-6,
                 verbose: bool = False) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.on_incumbent = on_incumbent or (lambda coloring: None)  # receives a list of n labels
        try:  # hooks that accept a `stage` keyword also receive the incumbent's provenance
            params = inspect.signature(self.on_incumbent).parameters
            self.hook_takes_stage = "stage" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())
        except (TypeError, ValueError):
            self.hook_takes_stage = False
        self.on_bound = on_bound or (lambda bound: None)
        self.tabu, self.tabu_share, self.alpha = tabu, tabu_share, alpha
        self.heuristic_pricing, self.max_cols, self.exact_every = heuristic_pricing, max_cols, exact_every
        self.diving, self.use_rmp, self.tol, self.verbose = dive, rmp, tol, verbose
        self.exact_cap = max(10.0, 0.05 * time_limit)  # seconds per exact pricing solve (a capped round still bounds)
        self.max_pool = 30000  # columns, beyond which TabuCol classes are no longer harvested
        self.trace: list[dict[str, Any]] = []

    # ------------------------------------------------------------ setup

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        return self.solve(load_col(data_path))

    def left(self) -> float:
        return self.deadline - time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def log(self, stage: str) -> None:
        self.trace.append({"t": round(self.elapsed(), 3), "incumbent": self.ub if self.ub < INF else None,
                           "bound": self.lb if self.lb > -INF else None, "stage": stage})
        if self.verbose:
            print(self.trace[-1], flush=True)

    def solve(self, inst: ColoringInstance) -> dict[str, Any]:
        if not hasattr(self, "t0"):
            self.t0 = time.perf_counter()
            self.deadline = self.t0 + self.time_limit
        self.inst = inst
        self.n = n = inst.n
        self.A = inst.adjacency
        self.ub, self.lb = INF, -INF
        self.best: list[int] | None = None
        self.rng = np.random.default_rng(self.seed)
        self.stats = {"cg_iters": 0, "exact_rounds": 0, "heuristic_rounds": 0, "mispricings": 0, "early_stops": 0,
                      "lp_seconds": 0.0, "pricing_seconds": 0.0, "exact_seconds": 0.0, "mip_runs": 0, "mip_seconds": 0.0,
                      "dives": 0, "dive_seconds": 0.0, "tabu_runs": 0, "tabu_seconds": 0.0, "tabu_iterations": 0,
                      "nodes": 0, "stuck_nodes": 0, "max_depth": 0, "root_converged": False, "root_status": None,
                      "root_bound": None, "root_lp": None, "root_seconds": None, "clique": None, "cover_cliques": None,
                      "harvested": 0, "first_incumbent_seconds": None}
        self.frontier: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.memory_limit = False
        self.root = {"id": 0, "depth": 0, "lb": -INF, "same": (), "diff": ()}
        self.frontier = [self.root]
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", self.threads)
        env.setParam("Seed", self.seed)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()
        self.env = env
        T = self.time_limit
        self.mip_limit = max(3.0, min(60.0, 0.03 * T))
        self.primal_gap = max(3.0, min(60.0, 0.03 * T))
        self.dive_budget = max(3.0, min(90.0, 0.05 * T))
        self.tabu_step = max(2.0, min(60.0, 0.03 * T))
        self.harvest_seen: set[bytes] = set()
        try:
            self.build_master()
            self.construct()
            self.search()
        except gp.GurobiError as err:
            if err.errno != GRB.Error.OUT_OF_MEMORY and "memory" not in str(err).lower():
                raise
            self.memory_limit = True
        # "stuck": the search ended before the deadline with only stuck nodes left (the bound stays valid).
        out = self.result("memory_limit" if self.memory_limit else "optimal" if self.closed()
                          else "time_limit" if self.left() <= 0.5 else "stuck")
        env.dispose()
        return out

    def closed(self) -> bool:
        return self.ub < INF and self.lb >= self.ub

    def prunable(self, bound: float) -> bool:
        return bound >= self.ub

    def result(self, reason: str) -> dict[str, Any]:
        ub = self.ub if self.ub < INF else None
        lb = self.lb if self.lb > -INF else None
        return {"best_objective": ub, "global_lb": lb, "termination_reason": reason, "coloring": self.best,
                "columns": len(self.vars), **self.stats, "total_seconds": self.elapsed(), "trace": self.trace}

    # ------------------------------------------------------------ bounds

    def publish_bound(self, stage: str) -> None:
        nodes = list(self.frontier) + ([self.current] if self.current is not None else [])
        lb = min((x["lb"] for x in nodes), default=self.ub)
        lb = min(lb, self.ub)
        if math.isfinite(lb) and lb > self.lb:
            self.lb = lb
            self.on_bound(int(lb))
            self.log(stage)

    def node_bound(self, node: dict, value: int, stage: str) -> None:
        if value > node["lb"]:
            node["lb"] = value
            self.publish_bound(stage)

    # ------------------------------------------------------------ master and node graphs

    def build_master(self) -> None:
        n = self.n
        md = gp.Model(env=self.env)
        self.cover = [md.addLConstr(gp.LinExpr(), GRB.GREATER_EQUAL, 1.0) for _ in range(n)]
        md.update()
        self.master = md
        self.colmat = np.zeros((1024, n), dtype=bool)
        self.vars: list[gp.Var] = []
        self.col_index: dict[bytes, int] = {}
        self.rules = {"same": (), "diff": ()}
        self.gid = np.arange(n)
        self.G = n
        self.B = self.A
        self.Bdeg = self.A.sum(axis=1)
        self.pricing_model = None
        self.root_cover = edge_clique_cover(self.A)
        self.stats["cover_cliques"] = len(self.root_cover)
        for v in range(n):
            m = np.zeros(n, dtype=bool)
            m[v] = True
            self.add_column(m)

    def compatible(self, M: np.ndarray) -> np.ndarray:
        """Rows of M (column masks) compatible with the current node's rules."""
        ok = np.ones(len(M), dtype=bool)
        for u, v in self.rules["same"]:
            ok &= M[:, u] == M[:, v]
        for u, v in self.rules["diff"]:
            ok &= ~(M[:, u] & M[:, v])
        return ok

    def add_column(self, mask: np.ndarray) -> bool:
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return False
        key = mask.tobytes()
        if key in self.col_index:
            return False
        idx = np.nonzero(mask)[0]
        k = len(self.vars)
        if k >= len(self.colmat):
            self.colmat = np.vstack([self.colmat, np.zeros_like(self.colmat)])
        self.colmat[k] = mask
        ub = INF if self.compatible(mask[None, :])[0] else 0.0
        self.vars.append(self.master.addVar(lb=0.0, ub=ub, obj=1.0, column=gp.Column([1.0] * len(idx), [self.cover[v] for v in idx.tolist()])))
        self.col_index[key] = k
        return True

    def set_node(self, node: dict) -> bool:
        """Installs a node: groups of contracted vertices, the node graph, column bounds, group columns. False if the
        rules are contradictory."""
        n = self.n
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for u, v in node["same"]:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
        roots = np.array([find(v) for v in range(n)])
        _, gid = np.unique(roots, return_inverse=True)
        G = int(gid.max()) + 1
        M = np.zeros((G, n), dtype=np.float32)
        M[gid, np.arange(n)] = 1.0
        B = (M @ self.A.astype(np.float32) @ M.T) > 0.5
        for u, v in node["diff"]:
            B[gid[u], gid[v]] = B[gid[v], gid[u]] = True
        if B[np.arange(G), np.arange(G)].any():
            return False
        self.rules = {"same": tuple(node["same"]), "diff": tuple(node["diff"])}
        self.gid, self.G, self.B = gid, G, B
        self.Bdeg = B.sum(axis=1)
        k = len(self.vars)
        ok = self.compatible(self.colmat[:k])
        self.master.setAttr("UB", self.vars, np.where(ok, INF, 0.0).tolist())
        for g in range(G):
            self.add_column(gid == g)
        if self.pricing_model is not None:
            self.pricing_model[0].dispose()
            self.pricing_model = None
        self.center = None
        node["center_val"] = -INF
        return True

    def expand(self, S: np.ndarray) -> np.ndarray:
        """A set of node-graph vertices (bool over groups) as a column mask over the original vertices."""
        return S[self.gid]

    # ------------------------------------------------------------ LP

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

    def duals(self) -> np.ndarray:
        return np.maximum(np.array(self.master.getAttr("Pi", self.cover)), 0.0)

    def group_weights(self, pi: np.ndarray) -> np.ndarray:
        return np.bincount(self.gid, weights=pi, minlength=self.G)

    def snapshot(self) -> None:
        self.snap = np.array(self.master.getAttr("X", self.vars))

    # ------------------------------------------------------------ pricing

    def heuristic_price(self, pi: np.ndarray, tries: int = 12) -> list[np.ndarray]:
        t = time.perf_counter()
        w = self.group_weights(pi)
        sets = weighted_greedy_sets(self.B, w, self.rng, tries, 1.0 + self.tol, self.Bdeg)[:self.max_cols]
        self.stats["heuristic_rounds"] += 1
        self.stats["pricing_seconds"] += time.perf_counter() - t
        return [self.expand(S) for S in sets]

    def build_pricing(self) -> None:
        """MWIS model of the node graph: clique rows of the root cover mapped to groups, plus the differ edges."""
        G, gid = self.G, self.gid
        rows, cols = [], []
        t = 0
        seen: set[bytes] = set()
        for K in self.root_cover:
            g = np.unique(gid[K])
            key = g.tobytes()
            if len(g) < 2 or key in seen:
                continue
            seen.add(key)
            rows.append(np.full(len(g), t))
            cols.append(g)
            t += 1
        for u, v in self.rules["diff"]:
            rows.append(np.array([t, t]))
            cols.append(np.array([gid[u], gid[v]]))
            t += 1
        pm = gp.Model(env=self.env)
        y = pm.addMVar(G, vtype=GRB.BINARY)
        if t:
            R = sp.csr_matrix((np.ones(sum(len(c) for c in cols)), (np.concatenate(rows), np.concatenate(cols))), shape=(t, G))
            pm.addConstr(R @ y <= 1.0)
        pm.ModelSense = GRB.MAXIMIZE
        pm.update()
        self.pricing_model = (pm, y.tolist())

    def exact_price(self, pi: np.ndarray, stop_after: float = 1.0) -> dict[str, Any]:
        """Exact MWIS round at duals pi: improving sets found (weight > 1 + tol) and an upper bound w_ub on the maximum
        weight (None if the solve ended without a bound)."""
        t = time.perf_counter()
        if self.pricing_model is None:
            self.build_pricing()
        pm, ys = self.pricing_model
        w = self.group_weights(pi)
        pm.setAttr("Obj", ys, w.tolist())
        cutoff = 1.0 + self.tol
        pm.Params.Cutoff = cutoff
        pm.Params.TimeLimit = max(1e-3, min(self.left(), self.exact_cap))
        found: dict[bytes, np.ndarray] = {}
        start = time.perf_counter()
        deadline = self.deadline

        def callback(m, where):
            if where == GRB.Callback.MIPSOL:
                x = np.array(m.cbGetSolution(ys)) > 0.5
                if float(w[x].sum()) > cutoff:
                    S = maximal_extension(self.B, x, w)
                    found.setdefault(S.tobytes(), S)
                if len(found) >= self.max_cols or (found and time.perf_counter() - start >= stop_after):
                    m.terminate()
            elif time.perf_counter() > deadline:
                m.terminate()

        pm.optimize(callback)
        st = pm.Status
        if pm.SolCount:
            x = np.array(pm.getAttr("X", ys)) > 0.5
            if float(w[x].sum()) > cutoff:
                S = maximal_extension(self.B, x, w)
                found.setdefault(S.tobytes(), S)
        if st == GRB.CUTOFF or (st == GRB.OPTIMAL and not found):
            w_ub = cutoff if st == GRB.CUTOFF else max(float(pm.ObjBound), float(pm.ObjVal) if pm.SolCount else 0.0)
            complete = True
        elif st in (GRB.OPTIMAL, GRB.INTERRUPTED, GRB.TIME_LIMIT) and math.isfinite(pm.ObjBound) and abs(pm.ObjBound) < 1e20:
            w_ub, complete = float(pm.ObjBound), st == GRB.OPTIMAL
        else:
            w_ub, complete = None, False
        dt = time.perf_counter() - t
        self.stats["exact_rounds"] += 1
        self.stats["pricing_seconds"] += dt
        self.stats["exact_seconds"] += dt
        return {"sets": [self.expand(S) for S in found.values()], "w_ub": w_ub, "complete": complete}

    def smoothed(self, pi: np.ndarray) -> np.ndarray:
        if self.center is None or self.alpha <= 0:
            return pi
        return self.alpha * self.center + (1.0 - self.alpha) * pi

    def farley(self, node: dict, pi: np.ndarray, res: dict, stage: str) -> None:
        if res["w_ub"] is None:
            return
        val = farley_bound(float(pi.sum()), res["w_ub"])
        if val > node.get("center_val", -INF):
            node["center_val"] = val
            self.center = pi
        self.node_bound(node, val, stage)

    def improving(self, sets: list[np.ndarray], pi: np.ndarray) -> list[np.ndarray]:
        """Sets not in the master whose weight at pi exceeds 1 + tol (negative reduced cost)."""
        return [S for S in sets if S.tobytes() not in self.col_index and float(pi[S].sum()) > 1.0 + self.tol]

    def column_generation(self, node: dict, stage: str) -> str:
        """CG at a node. Returns "converged", "early" (bound = ceil(master value)), "stalled" (capped exact pricing found
        nothing), "pruned" or "time"."""
        heuristic_fail = False
        since_exact = 0
        while self.left() > 0:
            if self.primal_due():
                self.primal_step()
                if self.closed() or self.prunable(node["lb"]):
                    return "pruned"
            st = self.solve_lp()
            if st != "optimal":
                return "pruned" if st == "infeasible" else "time"
            self.stats["cg_iters"] += 1
            z = float(self.master.ObjVal)
            node["lp"] = z
            if node["lb"] >= math.ceil(z - 1e-6):
                self.snapshot()
                self.stats["early_stops"] += 1
                return "early"
            if self.prunable(node["lb"]):
                return "pruned"
            pi = self.duals()
            exact = not self.heuristic_pricing or heuristic_fail or since_exact >= self.exact_every
            if not exact:
                added = sum(self.add_column(S) for S in self.heuristic_price(pi))
                since_exact += 1
                heuristic_fail = added == 0
                continue
            sep = self.smoothed(pi)
            res = self.exact_price(sep)
            self.farley(node, sep, res, stage)
            cands = list(res["sets"])
            neg = self.improving(res["sets"], pi)
            complete = res["complete"]
            if sep is not pi and not neg:
                self.stats["mispricings"] += 1
                res = self.exact_price(pi)
                self.farley(node, pi, res, stage)
                cands += res["sets"]
                neg = self.improving(res["sets"], pi)
                complete = res["complete"]
            added = sum(self.add_column(S) for S in cands)
            since_exact = 0
            heuristic_fail = False
            if self.closed() or self.prunable(node["lb"]):
                return "pruned"
            if not neg:
                # Converged (complete exact round) or stalled (the capped exact round found nothing improving): the
                # node's bound is the Farley bound; branching on the current master solution is valid either way.
                if self.left() <= 0:
                    return "time"
                if added and self.solve_lp() != "optimal":  # columns improving only at the smoothed duals
                    return "time"
                self.snapshot()
                return "converged" if complete else "stalled"
        return "time"

    # ------------------------------------------------------------ primal

    def offer(self, coloring, stage: str) -> bool:
        if coloring is None:
            return False
        col = canonical(coloring)
        obj = color_count(self.inst, col)
        if obj is None:
            return False
        for S in classes_of(col):
            m = np.zeros(self.n, dtype=bool)
            m[S] = True
            self.add_column(maximal_extension(self.A, m))
        if obj < self.ub:
            self.ub, self.best = float(obj), col
            if self.stats["first_incumbent_seconds"] is None:
                self.stats["first_incumbent_seconds"] = self.elapsed()
            if self.hook_takes_stage:
                self.on_incumbent(list(col), stage=stage)
            else:
                self.on_incumbent(list(col))
            self.log(stage)
            self.publish_bound(stage)
            return True
        return False

    def construct(self) -> None:
        """Shared initializer (DSATUR), the clique bound, then the TabuCol phase (option tabu)."""
        self.offer(dsatur(self.A), "initializer")
        clique = greedy_clique(self.A) if self.n else []
        self.stats["clique"] = len(clique)
        self.root["lb"] = len(clique)
        self.publish_bound("clique")
        self.tabucol = TabuCol(self.A, seed=self.seed)
        if self.tabu and not self.closed():
            self.run_tabu(self.tabu_share * self.time_limit)
            # Set-covering phase over the harvested classes (MMT): may combine them into fewer colors.
            self.restricted_mip(self.mip_limit, "restricted_mip_pool")

    def run_tabu(self, budget: float) -> None:
        if self.best is None or budget <= 0 or self.closed():
            return
        t = time.perf_counter()
        until = min(self.deadline, t + budget)
        self.stats["tabu_runs"] += 1
        before = self.tabucol.iterations
        self.tabucol.reduce(np.array(self.best), until, on_coloring=lambda c: self.offer(c, "tabucol"),
                            lower=int(max(self.lb, 1)), harvest=self.harvest, resume=True)
        self.stats["tabu_iterations"] += self.tabucol.iterations - before
        self.stats["tabu_seconds"] += time.perf_counter() - t

    def harvest(self, masks: np.ndarray) -> None:
        """Conflict-free color classes met by TabuCol join the master as maximal independent sets (pool capped)."""
        for m in masks:
            if len(self.vars) >= self.max_pool:
                return
            key = m.tobytes()
            if key in self.harvest_seen:
                continue
            self.harvest_seen.add(key)
            self.stats["harvested"] += self.add_column(maximal_extension(self.A, m))

    def primal_due(self) -> bool:
        return time.perf_counter() >= self.next_primal

    def primal_step(self) -> None:
        """Between pricing rounds, cyclically: restricted-master MIP, dive, TabuCol."""
        steps = [s for s, on in (("rmp", self.use_rmp), ("dive", self.diving), ("tabu", self.tabu)) if on]
        if steps:
            step = steps[self.primal_steps % len(steps)]
            self.primal_steps += 1
            if step == "rmp":
                self.restricted_mip(self.mip_limit, "restricted_mip")
            elif step == "dive":
                self.dive(self.dive_budget, "dive")
            else:
                self.run_tabu(self.tabu_step)
        self.next_primal = time.perf_counter() + max(self.primal_gap, 0.1 * self.elapsed())

    def restricted_mip(self, limit: float, stage: str) -> None:
        """Integer covering master over all columns, solutions with fewer colors than the incumbent only."""
        limit = min(limit, self.left())
        if limit <= 0.2 or not self.use_rmp:
            return
        t = time.perf_counter()
        self.stats["mip_runs"] += 1
        K = len(self.vars)
        md = gp.Model(env=self.env)
        md.Params.TimeLimit = limit
        md.Params.MIPFocus = 1
        lam = md.addMVar(K, vtype=GRB.BINARY, obj=1.0)
        ks, vs = np.nonzero(self.colmat[:K])
        md.addConstr(sp.csr_matrix((np.ones(len(ks)), (vs, ks)), shape=(self.n, K)) @ lam >= 1.0)
        if self.ub < INF:
            md.Params.Cutoff = self.ub - 0.5
        lams = lam.tolist()
        deadline = time.perf_counter() + limit
        cols = self.colmat[:K]

        def decode(vals) -> np.ndarray | None:
            return coloring_from_cover(self.n, [cols[k] for k in np.nonzero(np.asarray(vals) > 0.5)[0]])

        def callback(model, where):
            if time.perf_counter() > deadline or time.perf_counter() > self.deadline:
                model.terminate()
                return
            if where == GRB.Callback.MIPSOL:
                self.offer(decode(model.cbGetSolution(lams)), stage)

        md.optimize(callback)
        if md.Status == GRB.MEM_LIMIT:
            self.memory_limit = True
        md.dispose()
        self.stats["mip_seconds"] += time.perf_counter() - t

    def dive(self, budget: float, stage: str) -> None:
        """Fix the columns at >= 0.99 in the LP (else the largest fractional one), re-optimize with up to 5 heuristic
        pricing rounds, and repeat until the LP solution is integral (offered as a coloring) or its rounded value
        reaches the incumbent. The fixings are undone afterwards; generated columns stay."""
        if not self.diving:
            return
        until = min(self.deadline, time.perf_counter() + budget)
        t = time.perf_counter()
        self.stats["dives"] += 1
        fixed: list[gp.Var] = []
        isfixed: set[int] = set()
        try:
            while time.perf_counter() < until:
                for _ in range(5):
                    if self.solve_lp() != "optimal" or time.perf_counter() >= until:
                        break
                    if sum(self.add_column(S) for S in self.heuristic_price(self.duals(), tries=6)) == 0:
                        break
                if self.solve_lp() != "optimal" or math.ceil(self.master.ObjVal - 1e-6) >= self.ub:
                    return
                x = np.array(self.master.getAttr("X", self.vars))
                pos = np.nonzero(x > 1e-6)[0]
                if (np.abs(x[pos] - 1.0) < 1e-6).all():
                    self.offer(coloring_from_cover(self.n, [self.colmat[k] for k in pos]), stage)
                    return
                cand = [int(k) for k in pos if int(k) not in isfixed]
                if not cand:
                    return
                sel = [k for k in cand if x[k] >= 0.99] or [max(cand, key=lambda k: x[k])]
                for k in sel:
                    self.vars[k].LB = 1.0
                    fixed.append(self.vars[k])
                    isfixed.add(k)
        finally:
            for v in fixed:
                v.LB = 0.0
            self.stats["dive_seconds"] += time.perf_counter() - t

    # ------------------------------------------------------------ search

    def branch(self, node: dict) -> list[dict] | None:
        """Children of a node on the most balanced pair of non-adjacent groups; None if no pair separates the LP
        solution (its support is then offered as a coloring)."""
        x = self.snap
        pos = np.nonzero(x > 1e-6)[0]
        lam = x[pos]
        if (np.abs(lam - 1.0) < 1e-6).all():
            self.offer(coloring_from_cover(self.n, [self.colmat[k] for k in pos]), "integral_node")
            return None
        G, gid = self.G, self.gid
        rep = np.zeros(G, dtype=np.int64)
        rep[gid[::-1]] = np.arange(self.n)[::-1]  # first vertex of every group
        X = self.colmat[pos][:, rep].astype(np.float64)  # support x groups
        both = X.T @ (lam[:, None] * X)
        cov = np.diag(both).copy()
        one = cov[:, None] + cov[None, :] - 2.0 * both
        score = np.minimum(both, one)
        score[self.B] = -1.0
        np.fill_diagonal(score, -1.0)
        g, h = np.unravel_index(np.argmax(score), score.shape)
        if score[g, h] <= 1e-6:
            order = pos[np.argsort(-lam, kind="stable")]
            self.offer(coloring_from_cover(self.n, [self.colmat[k] for k in order]), "integral_node")
            return None
        u, v = int(rep[min(g, h)]), int(rep[max(g, h)])
        base = {"depth": node["depth"] + 1, "lb": node["lb"]}
        return [dict(base, same=node["same"] + ((u, v),), diff=node["diff"]),
                dict(base, same=node["same"], diff=node["diff"] + ((u, v),))]

    def search(self) -> None:
        self.primal_steps = 0
        self.next_primal = time.perf_counter() + self.primal_gap
        self.snap = None
        count = 0
        while self.left() > 0 and not self.closed():
            live = [x for x in self.frontier if not x.get("stuck")]
            if not live:
                break
            node = min(live, key=lambda x: (x["lb"], -x["depth"], x["id"]))
            self.frontier.remove(node)
            self.current = node
            if self.prunable(node["lb"]):
                self.current = None
                self.publish_bound("prune")
                continue
            self.stats["nodes"] += 1
            self.stats["max_depth"] = max(self.stats["max_depth"], node["depth"])
            if not self.set_node(node):
                self.current = None
                self.publish_bound("infeasible_node")
                continue
            root = node["id"] == 0
            status = self.column_generation(node, "root" if root else "node")
            if root:
                self.stats["root_status"] = status
                self.stats["root_converged"] = status in ("converged", "early")  # "stalled": capped exact pricing
                self.stats["root_lp"] = node.get("lp")
                self.stats["root_bound"] = node["lb"]
                self.stats["root_seconds"] = self.elapsed()
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
            if root and not self.closed():
                self.dive(self.dive_budget, "dive")
                self.restricted_mip(self.mip_limit, "restricted_mip")
                self.next_primal = time.perf_counter() + max(self.primal_gap, 0.1 * self.elapsed())
            elif self.primal_due() and not self.closed():
                self.primal_step()
        self.current = None
        self.publish_bound("end")
        while self.left() > 0.5 and not self.closed():
            before = self.ub
            self.restricted_mip(self.left(), "restricted_mip_final")
            if self.tabu and self.left() > 0.5:
                self.run_tabu(self.left())
            if self.ub >= before:
                break
