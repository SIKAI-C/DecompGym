"""Engineered branch-and-price-and-cut reference for the CVRP (scale-up study; adapted from src/bd/clrp/d_ref_cg_xl.py
to one uncapacitated depot without opening or vehicle costs).

Master (set covering over routes r; c_r = rounded route distance; a_ir = visits of customer i by r):
    min sum_r c_r lam_r
    s.t. sum_r a_ir lam_r >= 1                              (pi_i >= 0)     every customer
         sum_r lam_r >= K = ceil(q(N) / Q)                   (sigma >= 0)
         sum_r cross_U(r) lam_r >= 2 ceil(q(U) / Q)         (beta_U >= 0)   rounded capacity cuts (robust: they only
                                                                             change arc costs), separated by greedy
                                                                             growth on the LP edge flows
         sum_r use_e(r) lam_r >= 1 (or >= 0 when inactive)  (eta_e >= 0)    edge-branching rows
Covering relaxes partitioning, so every bound below is valid for the CVRP; integer covering plans are repaired into
partitions (heuristic.repair_cover) and every plan is re-checked with route_cost before it is published.

Pricing: src.bd.clrp's NGPricer on one depot (ng-route labelling with the capacity resource, exact dominance,
completion bounds from q-routes with ng-consistent 2-cycle elimination). Heuristic rounds first (near-neighbour arcs,
label beam, up to max_cols columns per round), exact rounds when a heuristic round finds nothing new or every 15
rounds. Columns are identified by their visiting sequence up to reversal (cut coefficients depend on the order).

Stabilization: Wentges smoothing. Pricing uses alpha * (center) + (1 - alpha) * (master duals), the center being the
duals of the best Lagrangian bound at the node; when no column has a negative reduced cost at the master duals
(mispricing), the round is repeated at the master duals. Convergence is declared only by an exact round at the master
duals that adds nothing.

Lower bounds. Every complete exact round gives, at the duals D it priced (smoothed or not), a Lagrangian bound valid
for all plans in the node: dualize the master rows at D (all inequalities valid for every feasible plan; edge rows
only inside their node) and keep sum_r q_r lam_r = q(N) (true of every partition into routes), so the route part is at
least q(N) * min(0, min_r rc_r / q_r); alternatively (at most n routes) n * min(0, min_r rc_r). Both minima are exact
under ng-route pricing (the dominance rule preserves them; the pricing tolerance enters as -tol). The larger of the two
is kept, made safe against floating-point error and rounded up (distances are integral). The route set is all
ng-routes, a superset of the elementary ones. The global bound is the minimum over the open nodes.

Search: best-first branch-and-price on customer edges (most fractional x_e = sum_r use_e(r) lam_r; the "x_e >= 1"
branch adds an edge row with an artificial column at a prohibitive cost, the "x_e = 0" branch forbids the arc in
pricing); root: up to 50 RCC rounds, other nodes: 3.
Primal side: the shared basic initializer (heuristic.basic_initializer) polished by local search; between pricing
rounds, alternately restricted-master MIPs over the elementary columns (incumbent as MIP start, cutoff below it) and
diving (fix the columns with the largest LP values, remove their customers from pricing, re-optimize with up to 6
heuristic pricing rounds, repeat until the LP is integral); a dive and a restricted MIP right after the root
converges. Every improving plan is polished by the inter-route local search (heuristic.CVRPLocalSearch) and its
routes join the master.
"""
from __future__ import annotations

import inspect
import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.clrp.c_heuristic import two_opt
from src.bd.clrp.d_ref_cg_xl import NGPricer, separate_rcc
from src.cg.cvrp.a_data import CVRPInstance, load_vrp
from src.cg.cvrp.b_mip_xl import route_cost
from src.cg.cvrp.heuristic import CVRPLocalSearch, basic_initializer, repair_cover, single_depot_view

INF = float("inf")


class CVRPColumnGenerationXL:
    """Engineered branch-and-price-and-cut reference for the CVRP; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0, ng: int = 8,
                 near: int = 12, beam: int = 3000, max_cols: int = 60, heuristic_pricing: bool = True,
                 alpha: float = 0.5, rcc: bool = True, dive: bool = True, tol: float = 1e-6, verbose: bool = False) -> None:
        self.time_limit = time_limit
        self.threads = threads
        self.on_incumbent = on_incumbent or (lambda routes: None)  # receives routes as lists of customers 1..n
        try:  # hooks that accept a `stage` keyword also receive the incumbent's provenance
            params = inspect.signature(self.on_incumbent).parameters
            self.hook_takes_stage = "stage" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())
        except (TypeError, ValueError):
            self.hook_takes_stage = False
        self.on_bound = on_bound or (lambda bound: None)  # receives every new valid global lower bound
        self.soft_mem_gb = soft_mem_gb
        self.seed = seed
        self.ng, self.near, self.beam, self.max_cols = ng, near, beam, max_cols
        self.heuristic_pricing = heuristic_pricing
        self.alpha = alpha
        self.rcc = rcc
        self.diving = dive
        self.tol = tol
        self.max_labels = 400_000  # per depth in exact pricing; doubled whenever a capped round finds no column
        self.verbose = verbose
        self.trace: list[dict[str, Any]] = []

    # ------------------------------------------------------------ setup

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        return self.solve(load_vrp(data_path))

    def left(self) -> float:
        return self.deadline - time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def log(self, stage: str) -> None:
        self.trace.append({"t": round(self.elapsed(), 3), "incumbent": self.ub if self.ub < INF else None,
                           "bound": self.lb if self.lb > -INF else None, "stage": stage})
        if self.verbose:
            print(self.trace[-1], flush=True)

    def solve(self, inst: CVRPInstance) -> dict[str, Any]:
        if not hasattr(self, "t0"):
            self.t0 = time.perf_counter()
            self.deadline = self.t0 + self.time_limit
        self.inst = inst
        n = inst.n
        self.n = n
        self.dist = inst.distances()
        self.cost = self.dist.astype(float)
        self.c = self.dist.tolist()
        self.q = inst.demand[1:].astype(np.int64)  # internal customer k is customer k + 1 of the instance
        self.Q = int(inst.capacity)
        self.qtot = float(self.q.sum())
        self.K = inst.min_vehicles
        self.ub, self.lb = INF, -INF
        self.best_routes: list[list[int]] | None = None
        self.cols: list[dict[str, Any]] = []
        self.stats = {"cg_iters": 0, "exact_rounds": 0, "heuristic_rounds": 0, "mispricings": 0, "lp_seconds": 0.0,
                      "pricing_seconds": 0.0, "exact_seconds": 0.0, "mip_runs": 0, "mip_seconds": 0.0, "dives": 0,
                      "dive_seconds": 0.0, "ls_seconds": 0.0, "nodes": 0, "labels": 0, "cuts": 0, "cut_rounds": 0,
                      "removed_columns": 0, "stuck_nodes": 0, "root_converged": False, "root_bound": None,
                      "root_lp": None, "root_seconds": None, "max_depth": 0, "first_incumbent_seconds": None, "ng": None}
        self.frontier: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.memory_limit = False
        if (self.q > self.Q).any():
            return self.result("infeasible")
        ng = self.ng
        while n * (1 << max(0, ng - 1)) * (self.Q + 1) * 8 > 2.5e8 and ng > 3:
            ng -= 1  # memory of the pricer's dominance surface (customers x ng-masks x loads)
        self.stats["ng"] = ng
        self.pricer = NGPricer(single_depot_view(inst), self.cost, ng=ng, near=self.near)
        self.ls = CVRPLocalSearch(inst, self.dist)
        self.root = {"depth": 0, "id": 0, "lb": -INF, "rules": ()}
        self.frontier = [self.root]
        self.trivial_bound()
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", self.threads)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()
        self.env = env
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
        return self.ub < INF and self.lb >= self.ub - (1.0 - 1e-6)

    def prunable(self, bound: float) -> bool:
        return bound >= self.ub - (1.0 - 1e-6)

    def result(self, reason: str) -> dict[str, Any]:
        ub = self.ub if self.ub < INF else None
        lb = self.lb if self.lb > -INF else None
        return {"best_objective": ub, "global_lb": lb, "global_ub": ub,
                "gap": (ub - lb) / abs(ub) if ub is not None and lb is not None and ub else None,
                "termination_reason": reason, "routes": self.public(self.best_routes) if self.best_routes else None,
                "columns": len(self.cols), **self.stats, "total_seconds": self.elapsed(), "trace": self.trace}

    @staticmethod
    def public(routes: list[list[int]]) -> list[list[int]]:
        return [[int(i) + 1 for i in r] for r in routes if r]

    # ------------------------------------------------------------ bounds

    def certify(self, value: float, scale: float) -> float:
        """A computed bound made safe against floating-point error (relative 1e-7 of the summed magnitudes), then
        rounded up (distances are integral)."""
        return float(math.ceil(value - 1e-7 * max(1.0, scale)))

    def publish_bound(self, stage: str) -> None:
        nodes = list(self.frontier) + ([self.current] if self.current is not None else [])
        lb = min((x["lb"] for x in nodes), default=INF)
        if self.ub < INF:
            lb = min(lb, self.ub)
        if lb > self.lb + 1e-9 and math.isfinite(lb):
            self.lb = lb
            self.on_bound(lb)
            self.log(stage)

    def trivial_bound(self) -> None:
        """Before any pricing: every customer's cheapest incident edge (each route edge touches at most two customers
        and every customer has two edge incidences, so the sum is at most the plan's distance)."""
        full = self.cost.copy()
        np.fill_diagonal(full, INF)
        v = float(full[1:, :].min(axis=1).sum())
        self.root["lb"] = self.certify(v, v)
        self.publish_bound("trivial")

    def lagrangian(self, D: dict[str, Any], res: dict[str, Any], node: dict, stage: str) -> None:
        """Lagrangian bound at duals D from a complete exact pricing round res (see the module docstring)."""
        tol = self.tol
        const = float(D["pi"].sum() + D["sigma"] * self.K + D["beta"] @ self.cut_rhs_arr() + D["eta"] @ D["eta_rhs"])
        L = const + self.n * min(float(res["min_rc"]), -tol)  # at most n routes
        ratio = 0.0
        if self.q.min() > 0 and math.isfinite(float(res["min_ratio"])):  # the load fill needs positive demands
            ratio = min(float(res["min_ratio"]), -tol / float(self.q.min()))
            L = max(L, const + self.qtot * ratio)
        scale = float(np.abs(D["pi"]).sum() + D["sigma"] * self.K + np.abs(D["beta"]).sum() * 2 * self.n
                      + np.abs(D["eta"]).sum() + self.qtot * abs(ratio) + abs(L))
        val = self.certify(L, scale)
        if val > node.get("center_val", -INF):
            node["center_val"] = val
            self.center = D
        if val > node["lb"]:
            node["lb"] = val
            self.publish_bound(stage)

    # ------------------------------------------------------------ master

    def build_master(self) -> None:
        n = self.n
        md = gp.Model(env=self.env)
        md.Params.Method = 0
        self.cover = [md.addLConstr(gp.LinExpr(), GRB.GREATER_EQUAL, 1.0) for _ in range(n)]
        self.veh = md.addLConstr(gp.LinExpr(), GRB.GREATER_EQUAL, float(self.K))
        self.master = md
        self.big_m = 4.0 * float(self.cost.max()) * (n + 1)
        self.col_index: dict[tuple, int] = {}
        self.seq = np.full((1024, 16), n, dtype=np.int64)  # column node sequences padded with the depot (= n)
        self.cut_rows: list[gp.Constr] = []
        self.cut_mem = np.zeros((0, n + 1), dtype=bool)  # membership (depot column always False)
        self.cut_rhs: list[float] = []
        self.erows: dict[tuple[int, int], gp.Constr] = {}
        self.rules: dict[str, set] = {"e0": set()}
        self.center = None
        for i in range(n):  # singletons keep every node's master feasible
            self.add_column([i])

    def cut_rhs_arr(self) -> np.ndarray:
        return np.array(self.cut_rhs, dtype=float)

    def route_travel(self, path: list[int]) -> float:
        c = self.c
        s = [0] + [i + 1 for i in path] + [0]
        return float(sum(c[a][b] for a, b in zip(s, s[1:])))

    def forbidden(self, path: list[int]) -> bool:
        e0 = self.rules["e0"]
        return bool(e0) and any((min(a, b), max(a, b)) in e0 for a, b in zip(path, path[1:]))

    @staticmethod
    def key(path: list[int]) -> tuple:
        t = tuple(path)
        return min(t, t[::-1])

    def add_column(self, path: list[int], polish: bool = False) -> bool:
        path = [int(i) for i in path]
        if not path:
            return False
        elem = len(set(path)) == len(path)
        if polish and elem and len(path) >= 3:
            path = [x - 1 for x in two_opt(0, [i + 1 for i in path], self.cost)]
        key = self.key(path)
        if key in self.col_index:
            return False
        load = float(self.q[path].sum())
        if load > self.Q:
            return False
        cnt: dict[int, int] = {}
        for i in path:
            cnt[i] = cnt.get(i, 0) + 1
        rows = [self.cover[i] for i in cnt] + [self.veh]
        coef = [float(v) for v in cnt.values()] + [1.0]
        k = len(self.cols)
        if k >= len(self.seq):
            self.seq = np.vstack([self.seq, np.full_like(self.seq, self.n)])
        if len(path) + 2 > self.seq.shape[1]:
            self.seq = np.hstack([self.seq, np.full((len(self.seq), len(path) + 2 - self.seq.shape[1]), self.n, dtype=np.int64)])
        self.seq[k] = self.n
        self.seq[k, 1:len(path) + 1] = path
        if len(self.cut_rows):
            ins = self.cut_mem[:, self.seq[k]]
            cross = (ins[:, 1:] != ins[:, :-1]).sum(axis=1)
            for t in np.nonzero(cross)[0]:
                rows.append(self.cut_rows[t])
                coef.append(float(cross[t]))
        if self.erows:
            uses: dict[tuple[int, int], int] = {}
            for a, b in zip(path, path[1:]):
                e = (min(a, b), max(a, b))
                if e in self.erows:
                    uses[e] = uses.get(e, 0) + 1
            for e, u in uses.items():
                rows.append(self.erows[e])
                coef.append(float(u))
        var = self.master.addVar(lb=0.0, ub=0.0 if self.forbidden(path) else INF, obj=self.route_travel(path),
                                 column=gp.Column(coef, rows))
        self.col_index[key] = k
        self.cols.append({"path": path, "cost": self.route_travel(path), "load": load, "elem": elem, "var": var})
        return True

    def edge_count(self, i: int, j: int) -> np.ndarray:
        S = self.seq[:len(self.cols)]
        return (((S[:, :-1] == i) & (S[:, 1:] == j)) | ((S[:, :-1] == j) & (S[:, 1:] == i))).sum(axis=1)

    def set_node(self, node: dict) -> None:
        e0, e1 = set(), set()
        for _, i, j, v in node.get("rules", ()):
            (e1 if v else e0).add((i, j))
        self.rules = {"e0": e0}
        for e, row in self.erows.items():
            row.RHS = 1.0 if e in e1 else 0.0
        k = len(self.cols)
        bad = np.zeros(k, dtype=bool)
        for i, j in e0:
            bad |= self.edge_count(i, j) > 0
        self.master.setAttr("UB", [c["var"] for c in self.cols], np.where(bad, 0.0, INF).tolist())
        self.center = None

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

    def duals(self) -> dict[str, Any]:
        md = self.master
        ekeys = list(self.erows)
        return {"pi": np.maximum(np.array(md.getAttr("Pi", self.cover)), 0.0),
                "sigma": max(float(self.veh.Pi), 0.0),
                "beta": np.maximum(np.array(md.getAttr("Pi", self.cut_rows)), 0.0) if self.cut_rows else np.zeros(0),
                "eta": np.maximum(np.array([self.erows[e].Pi for e in ekeys]), 0.0) if ekeys else np.zeros(0),
                "eta_rhs": np.array([self.erows[e].RHS for e in ekeys]) if ekeys else np.zeros(0),
                "ekeys": ekeys}

    def smoothed(self, D: dict[str, Any]) -> dict[str, Any]:
        """Wentges smoothing toward the node's stability center (rows added after the center get 0 there)."""
        c = self.center
        if c is None or self.alpha <= 0:
            return D
        a = self.alpha
        pad = lambda v, k: np.concatenate([v, np.zeros(k - len(v))]) if len(v) < k else v[:k]
        return {"pi": a * c["pi"] + (1 - a) * D["pi"], "sigma": a * c["sigma"] + (1 - a) * D["sigma"],
                "beta": a * pad(c["beta"], len(D["beta"])) + (1 - a) * D["beta"],
                "eta": a * pad(c["eta"], len(D["eta"])) + (1 - a) * D["eta"],
                "eta_rhs": D["eta_rhs"], "ekeys": D["ekeys"]}

    def compact(self) -> None:
        """Drops the columns with the largest reduced costs at the last LP once the master is large (singletons and
        incumbent routes stay); exact pricing regenerates any column the LP needs later."""
        k = len(self.cols)
        limit = max(6000, 60 * self.n)
        snap = self.snap
        if k <= limit or snap is None:
            return
        k0 = snap["ncols"]
        keep = np.zeros(k, dtype=bool)
        keep[np.argsort(snap["rc"], kind="stable")[:limit // 2]] = True
        keep[:k0] |= snap["x"] > 1e-9
        keep[k0:] = True
        keep |= (self.seq[:k] < self.n).sum(axis=1) == 1
        if self.best_routes:
            for p in self.best_routes:
                t = self.col_index.get(self.key(p))
                if t is not None:
                    keep[t] = True
        drop = np.nonzero(~keep)[0]
        self.master.remove([self.cols[t]["var"] for t in drop])
        idx = np.nonzero(keep)[0]
        self.cols = [self.cols[t] for t in idx]
        self.seq[:len(idx)] = self.seq[idx]
        self.seq[len(idx):] = self.n
        self.col_index = {self.key(c["path"]): t for t, c in enumerate(self.cols)}
        self.stats["removed_columns"] += len(drop)
        self.snap = None
        self.master.update()

    # ------------------------------------------------------------ pricing and cuts

    def arc_costs(self, D: dict[str, Any], banned: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Customer arc costs and depot arc costs with the cut and edge-row duals subtracted; forbidden edges and
        banned customers (diving) at +inf."""
        n = self.n
        Cc = self.pricer.C.copy()
        c_out, c_in = self.pricer.c_out[0].copy(), self.pricer.c_in[0].copy()
        beta = D["beta"]
        if len(beta) and beta.max() > 0:
            Mb = self.cut_mem[:len(beta), :n].astype(float)
            s = Mb.T @ beta
            Cc -= s[:, None] + s[None, :] - 2.0 * ((Mb.T * beta) @ Mb)
            c_out -= s
            c_in -= s
        for (i, j), eta in zip(D["ekeys"], D["eta"]):
            Cc[i, j] -= eta
            Cc[j, i] -= eta
        for i, j in self.rules["e0"]:
            Cc[i, j] = Cc[j, i] = INF
        if banned is not None and banned.any():
            c_out[banned] = INF
            Cc[:, banned] = INF
        np.fill_diagonal(Cc, INF)
        return Cc, c_out, c_in

    def price(self, D: dict[str, Any], exact: bool, banned: np.ndarray | None = None) -> dict[str, Any]:
        t = time.perf_counter()
        Cc, c_out, c_in = self.arc_costs(D, banned)
        res = self.pricer.price(D["pi"], -float(D["sigma"]), Cc, c_out, c_in, exact=exact, beam=self.beam,
                                max_cols=self.max_cols, deadline=self.deadline, max_labels=self.max_labels, tol=self.tol)
        res["arcs"] = (Cc, c_out, c_in, D["pi"], -float(D["sigma"]))
        dt = time.perf_counter() - t
        self.stats["labels"] += res["labels"]
        self.stats["pricing_seconds"] += dt
        self.stats["exact_rounds" if exact else "heuristic_rounds"] += 1
        if exact:
            self.stats["exact_seconds"] += dt
        return res

    @staticmethod
    def reduced_cost(path: list[int], arcs: tuple) -> float:
        Cc, c_out, c_in, p, const = arcs
        v = const + c_out[path[0]] - p[path[0]] + c_in[path[-1]]
        for a, b in zip(path, path[1:]):
            v += Cc[a, b] - p[b]
        return float(v)

    def new_negative(self, routes: list, arcs: tuple) -> list[list[int]]:
        """Routes not yet in the master whose reduced cost at the given duals is negative."""
        return [p for _, p in routes if self.key(p) not in self.col_index and self.reduced_cost(p, arcs) < -self.tol]

    def snapshot(self) -> None:
        vars_ = [c["var"] for c in self.cols]
        self.snap = {"x": np.array(self.master.getAttr("X", vars_)), "rc": np.array(self.master.getAttr("RC", vars_)),
                     "ncols": len(self.cols)}

    def flows(self) -> tuple[np.ndarray, np.ndarray]:
        """LP customer-customer edge flows X (symmetric) and the depot-edge flow of every customer."""
        n = self.n
        x = self.snap["x"]
        X = np.zeros((n + 1, n + 1))
        for t in np.nonzero(x > 1e-9)[0]:
            s = [n] + self.cols[t]["path"] + [n]
            for a, b in zip(s, s[1:]):
                X[a, b] += x[t]
                X[b, a] += x[t]
        return X[:n, :n], X[:n, n]

    def add_cuts(self) -> int:
        X, dep = self.flows()
        deg = X.sum(axis=1) + dep
        found = separate_rcc(X, deg, self.q, self.Q, max_size=min(self.n, 30))
        if not found:
            return 0
        room = max(0, 4 * self.n + 200 - len(self.cut_rows))
        found = found[:min(room, max(10, self.n // 4))]
        if not found:
            return 0
        k = len(self.cols)
        S = self.seq[:k]
        vars_ = [c["var"] for c in self.cols]
        mems = []
        for _, U in found:
            ext = np.append(U, False)
            ins = ext[S]
            cross = (ins[:, 1:] != ins[:, :-1]).sum(axis=1)
            nz = np.nonzero(cross)[0]
            rhs = 2.0 * math.ceil(float(self.q[U].sum()) / self.Q - 1e-9)
            row = self.master.addLConstr(gp.LinExpr(cross[nz].astype(float).tolist(), [vars_[t] for t in nz]), GRB.GREATER_EQUAL, rhs)
            self.cut_rows.append(row)
            self.cut_rhs.append(rhs)
            mems.append(ext)
        self.cut_mem = np.vstack([self.cut_mem] + [m_[None, :] for m_ in mems])
        self.stats["cuts"] += len(found)
        self.stats["cut_rounds"] += 1
        return len(found)

    def column_generation(self, node: dict, stage: str, cut_rounds: int) -> str:
        """CG at a node with RCC rounds after each convergence. Returns "converged", "pruned" or "time"."""
        heuristic_fail = False
        since_exact = 0
        rounds = 0
        last_lp = -INF
        while self.left() > 0:
            if self.primal_due():
                self.primal_step()
            st = self.solve_lp()
            if st != "optimal":
                return "pruned" if st == "infeasible" else "time"
            self.stats["cg_iters"] += 1
            node["lp"] = float(self.master.ObjVal)
            rmp = self.duals()
            exact = not self.heuristic_pricing or heuristic_fail or since_exact >= 15
            sep = self.smoothed(rmp)
            res = self.price(sep, exact)
            if exact and res["complete"]:
                self.lagrangian(sep, res, node, stage)
            cands = [p for _, p in res["routes"]]
            if sep is rmp:
                neg = self.new_negative(res["routes"], res["arcs"])
            else:
                rmp_arcs = self.arc_costs(rmp) + (rmp["pi"], -float(rmp["sigma"]))
                neg = self.new_negative(res["routes"], rmp_arcs)
                if not neg:  # mispricing: repeat the round at the master duals
                    self.stats["mispricings"] += 1
                    res = self.price(rmp, exact)
                    if exact and res["complete"]:
                        self.lagrangian(rmp, res, node, stage)
                    cands += [p for _, p in res["routes"]]
                    neg = self.new_negative(res["routes"], res["arcs"])
            added = sum(self.add_column(p) for p in cands)
            if exact:
                since_exact = 0
                heuristic_fail = False
                if not res["complete"] and added == 0:
                    self.max_labels = min(2 * self.max_labels, 1_600_000)  # bounded growth (memory)
            else:
                since_exact += 1
                heuristic_fail = not neg
            if self.closed() or self.prunable(node["lb"]):
                return "pruned"
            if exact and res["complete"] and not neg:
                if added:  # columns that were only negative at the smoothed duals: the basis stays optimal
                    if self.solve_lp() != "optimal":
                        return "time"
                    node["lp"] = float(self.master.ObjVal)
                self.snapshot()
                if not self.rcc or rounds >= cut_rounds or node["lp"] < last_lp + 1e-5 * abs(node["lp"]):
                    return "converged"
                last_lp = node["lp"]
                rounds += 1
                if self.add_cuts() == 0:
                    return "converged"
                self.center = None  # new rows: restart the smoothing from the master duals
                node["center_val"] = -INF
        return "time"

    # ------------------------------------------------------------ primal

    def offer(self, routes: list[list[int]], stage: str, polish: bool = True) -> bool:
        """Checks a plan (internal customers 0..n-1) with route_cost and publishes it if it improves."""
        routes = [[int(i) for i in r] for r in routes if len(r)]
        obj = route_cost(self.inst, self.public(routes), self.dist)
        if obj is None:
            return False
        better = obj < self.ub
        if better:
            self.publish(routes, float(obj), stage)
        if polish and self.left() > 0:
            t = time.perf_counter()
            improved = self.ls.improve(self.public(routes), self.deadline)
            self.stats["ls_seconds"] += time.perf_counter() - t
            obj2 = route_cost(self.inst, improved, self.dist)
            internal = [[i - 1 for i in r] for r in improved]
            if obj2 is not None and obj2 < self.ub:
                self.publish(internal, float(obj2), stage + "+ls")
                better = True
            if obj2 is not None:
                for p in internal:
                    self.add_column(p, polish=True)
        for p in routes:
            self.add_column(p, polish=True)
        return better

    def publish(self, routes: list[list[int]], obj: float, stage: str) -> None:
        self.ub, self.best_routes = obj, routes
        if self.stats["first_incumbent_seconds"] is None:
            self.stats["first_incumbent_seconds"] = self.elapsed()
        if self.hook_takes_stage:
            self.on_incumbent(self.public(routes), stage=stage)
        else:
            self.on_incumbent(self.public(routes))
        self.log(stage)
        self.publish_bound(stage)

    def construct(self) -> None:
        """The shared basic initializer (savings + 2-opt), then local search."""
        self.offer([[i - 1 for i in r] for r in basic_initializer(self.inst, self.dist)], "initializer")

    def primal_due(self) -> bool:
        return time.perf_counter() >= self.next_primal

    def primal_step(self) -> None:
        """Between pricing rounds: alternately a restricted-master MIP and a dive."""
        self.primal_steps += 1
        if self.diving and self.primal_steps % 2 == 0:
            self.dive(self.dive_budget, "dive")
        else:
            self.restricted_mip(self.mip_limit, "restricted_mip")
        self.next_primal = time.perf_counter() + max(self.primal_gap, 0.1 * self.elapsed())

    def restricted_mip(self, limit: float, stage: str) -> None:
        """Integer covering master over the elementary columns (incumbent as MIP start, cutoff below it); every
        solution is repaired to a partition and offered from the callback."""
        limit = min(limit, self.left())
        if limit <= 0.2:
            return
        t = time.perf_counter()
        self.stats["mip_runs"] += 1
        cols = [k for k, col in enumerate(self.cols) if col["elem"]]
        md = gp.Model(env=self.env)
        md.Params.TimeLimit = limit
        md.Params.MIPFocus = 1
        lam = [md.addVar(vtype=GRB.BINARY, obj=self.cols[k]["cost"]) for k in cols]
        cover = [gp.LinExpr() for _ in range(self.n)]
        for v, k in enumerate(cols):
            for i in self.cols[k]["path"]:
                cover[i].addTerms(1.0, lam[v])
        for i in range(self.n):
            md.addLConstr(cover[i], GRB.GREATER_EQUAL, 1.0)
        if self.best_routes and self.ub < INF:
            md.Params.Cutoff = self.ub - 0.5
            pos = {k: v for v, k in enumerate(cols)}
            starts = [pos.get(self.col_index.get(self.key(p), -1)) for p in self.best_routes]
            if all(s_ is not None for s_ in starts):
                for v in range(len(cols)):
                    lam[v].Start = 0.0
                for s_ in starts:
                    lam[s_].Start = 1.0
        found = []
        deadline = time.perf_counter() + limit

        def callback(model, where):
            if time.perf_counter() > deadline:
                model.terminate()
                return
            if where == GRB.Callback.MIPSOL:
                vals = model.cbGetSolution(lam)
                plan = repair_cover([[i + 1 for i in self.cols[cols[v]]["path"]] for v, val in enumerate(vals) if val > 0.5], self.dist)
                found.append(plan)
                self.offer([[i - 1 for i in r] for r in plan], stage)

        md.optimize(callback)
        if md.Status == GRB.MEM_LIMIT:
            self.memory_limit = True
        md.dispose()
        self.stats["mip_seconds"] += time.perf_counter() - t

    def dive(self, budget: float, stage: str) -> None:
        """Diving heuristic on the current master (node rules kept): fix to 1 the elementary columns at >= 0.99 in the
        LP (else the largest fractional one) whose customers are still free, ban their customers from pricing,
        re-optimize with up to 6 heuristic pricing rounds, and repeat until the LP solution is integral (offered as a
        plan) or the LP value reaches the incumbent. The fixings are undone afterwards; generated columns stay."""
        until = min(self.deadline, time.perf_counter() + budget)
        t = time.perf_counter()
        self.stats["dives"] += 1
        fixed: list[gp.Var] = []
        banned = np.zeros(self.n, dtype=bool)
        try:
            while time.perf_counter() < until:
                for _ in range(6):
                    st = self.solve_lp()
                    if st != "optimal" or time.perf_counter() >= until:
                        break
                    res = self.price(self.duals(), exact=False, banned=banned)
                    if sum(self.add_column(p) for _, p in res["routes"]) == 0:
                        break
                if self.solve_lp() != "optimal" or self.prunable(math.ceil(self.master.ObjVal - 1e-6)):
                    return
                x = np.array(self.master.getAttr("X", [c["var"] for c in self.cols]))
                pos = np.nonzero(x > 1e-6)[0]
                if all(abs(x[k] - 1.0) < 1e-6 and self.cols[k]["elem"] for k in pos):
                    plan = repair_cover([[i + 1 for i in self.cols[k]["path"]] for k in pos], self.dist)
                    self.offer([[i - 1 for i in r] for r in plan], stage)
                    return
                cand = [k for k in pos if self.cols[k]["elem"] and self.cols[k]["var"].LB < 0.5
                        and not banned[self.cols[k]["path"]].any()]
                if not cand:
                    return
                sel = [k for k in cand if x[k] >= 0.99] or [max(cand, key=lambda k: x[k])]
                for k in sorted(sel, key=lambda k: -x[k]):
                    path = self.cols[k]["path"]
                    if banned[path].any():
                        continue
                    self.cols[k]["var"].LB = 1.0
                    fixed.append(self.cols[k]["var"])
                    banned[path] = True
        finally:
            for v in fixed:
                v.LB = 0.0
            self.stats["dive_seconds"] += time.perf_counter() - t

    # ------------------------------------------------------------ search

    def branch(self, node: dict) -> list[dict] | None:
        """Children of a converged node on the most fractional customer edge; None if no customer edge is fractional
        (an integral LP solution is then offered as a plan)."""
        base = {"depth": node["depth"] + 1, "lb": node["lb"], "rules": tuple(node.get("rules", ()))}
        X, _ = self.flows()
        up = np.triu(X, 1)
        frac = np.abs(up - 0.5)
        frac[(up < 1e-4) | (up > 1 - 1e-4)] = INF  # an edge fixed to 1 may carry more than 1: never re-branched
        if np.isfinite(frac).any():
            i, j = np.unravel_index(np.argmin(frac), frac.shape)
            e = (int(min(i, j)), int(max(i, j)))
            if e not in self.erows:
                cnt = self.edge_count(*e)
                nz = np.nonzero(cnt)[0]
                row = self.master.addLConstr(gp.LinExpr(cnt[nz].astype(float).tolist(), [self.cols[t]["var"] for t in nz]), GRB.GREATER_EQUAL, 0.0)
                # Artificial column at a cost above any plan: a node whose edge rows cannot be met gets an LP value,
                # hence a Lagrangian bound, above the incumbent and is pruned.
                self.master.addVar(lb=0.0, obj=self.big_m, column=gp.Column([1.0], [row]))
                self.erows[e] = row
            return [dict(base, rules=base["rules"] + (("e",) + e + (v,),)) for v in (1, 0)]
        x = self.snap["x"]
        pos = np.nonzero(x > 1e-9)[0]
        if all(abs(x[t] - 1.0) < 1e-6 for t in pos) and all(self.cols[t]["elem"] for t in pos):
            plan = repair_cover([[i + 1 for i in self.cols[t]["path"]] for t in pos], self.dist)
            self.offer([[i - 1 for i in r] for r in plan], "integral_node")
            node["lb"] = max(node["lb"], self.certify(node["lp"], abs(node["lp"])))
        return None

    def search(self) -> None:
        """Best-first branch-and-price (bound ties: deeper first), with primal steps between pricing rounds."""
        T = self.time_limit
        self.mip_limit = max(3.0, min(60.0, 0.03 * T))
        self.primal_gap = max(3.0, min(60.0, 0.03 * T))
        self.dive_budget = max(3.0, min(90.0, 0.05 * T))
        self.primal_steps = 0
        self.next_primal = time.perf_counter() + self.primal_gap
        self.snap = None
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
            status = self.column_generation(node, "root" if root else "node", cut_rounds=50 if root else 3)
            if root:
                self.stats["root_converged"] = status == "converged"
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
            kids = self.branch(node)  # reads the LP snapshot, so it comes before clean-up and primal steps
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
            if root:
                if self.diving:
                    self.dive(self.dive_budget, "dive")
                self.restricted_mip(self.mip_limit, "restricted_mip")
                self.next_primal = time.perf_counter() + max(self.primal_gap, 0.1 * self.elapsed())
            elif self.primal_due():
                self.primal_step()
            self.compact()
        self.current = None
        while self.left() > 0.5 and not self.closed():
            before = self.ub
            self.restricted_mip(self.left(), "restricted_mip_final")
            if self.ub >= before:
                break
