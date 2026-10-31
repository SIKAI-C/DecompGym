"""Engineered Benders (L-shaped) reference for the SCFL: multi-cut branch-and-Benders-cut with partial decomposition
(scale-up study).

Master: min f.y + sum_s p_s theta_s over y in {0, 1}^I with one recourse variable per scenario (multi-cut) and
  - theta_s >= sum_j d_sj min_i c_ij (every customer served at its cheapest unit cost);
  - the (15d) cover sum_i u_i y_i >= max_s D_s and its truncated version sum_i min(u_i, D_max) y_i >= D_max;
  - partial decomposition with artificial scenarios (Crainic, Hewitt, Maggioni and Rei 2021): the scenarios are
    clustered (k-means, f_heuristic.cluster_scenarios; at most `clusters` and at most `partial_budget` flow variables
    in the master, e.g. 10 clusters at 50 x 50, 2 at 200 x 200); for each cluster g the master keeps the recourse of the
    cluster-mean scenario explicitly, with flows xbar_g (demand rows, capacity rows u_i y_i, strong linking
    xbar_gij <= dbar_gj y_i) and the row sum_{s in g} p_s theta_s >= P_g c.xbar_g, valid by Jensen's inequality
    (Q_s(y) is convex in the demands for fixed y).
Cuts (c_naive_bd): theta_s >= d_s.v - sum_i (u_i alpha_i + sum_j d_sj (v_j - alpha_i - c_ij)^+) y_i, from
  - fractional points: the recourse LP with strong linking bounds x_ij <= d_sj y_i and penalized unmet demand
    (b_mip_xl.RecourseLP(penalty=True); a relaxation, so cuts are valid everywhere and no feasibility cuts are needed);
  - binary designs: the exact transportation LPs, with closed-facility lifting (c_naive_bd.lift_closed): every
    closed facility's coefficient minimized given (v, alpha_open), a Pareto-type strengthening that keeps the cut
    tight at the design.

Stages, all within one wall-clock budget:
1. The shared basic initializer (identical to the naive control), evaluated exactly; its cuts seed the master.
2. Root cutting-plane loop on the master LP (multi-cut, in-out stabilization: separation at lam y_lp + (1 - lam) core,
   core moved toward the separation points; lam -> 1 when no violated cut or the bound stalls); non-binding cuts are
   dropped at the end (cut management).
3. LP-guided designs (thresholds on y_lp, repaired to satisfy (15d)), evaluated exactly, their cuts added; then a
   time-capped surrogate-screened local search (f_heuristic.DesignSearch) from the best design.
4. Gurobi branch-and-Benders-cut on the master (one tree, lazy constraints): at every integer solution the design is
   evaluated exactly (cached), published, and the violated scenario cuts are added lazily; root-node fractional
   points get user cuts. A new design within `ls_window` of the incumbent gets a short local-search polish (capped
   at a share of the branch-and-cut time); improved designs are injected into the tree. The master bound is
   published through the MIP callback.
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from src.bd.scfl.a_data import SCFLInstance, load_instance
from src.bd.scfl.b_mip_xl import MIP_GAP, RecourseLP, add_scenario_blocks, is_memory_error, make_env, open_list
from src.bd.scfl.c_naive_bd import INF, certify, cuts_from_duals, greedy_design, lift_closed
from src.bd.scfl.f_heuristic import DesignSearch, cluster_scenarios, repair, surrogate_size


class SCFLBendersXL:
    """Engineered multi-cut branch-and-Benders-cut reference; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 clusters: int = 10, partial: bool = True, partial_budget: int = 100_000, root_fraction: float = 0.25,
                 root_rounds: int = 200,
                 inout_lambda: float = 0.3, lifting: bool = True, local_search: bool = True, ls_fraction: float = 0.1,
                 ls_window: float = 0.01, node_cut_rounds: int = 3, verbose: bool = False) -> None:
        self.time_limit = time_limit
        self.threads = threads
        self.on_incumbent = on_incumbent or (lambda design, stage=None: None)  # receives the open-facility list
        self.on_bound = on_bound or (lambda bound: None)
        self.soft_mem_gb = soft_mem_gb
        self.seed = seed
        self.clusters, self.partial, self.partial_budget = clusters, partial, partial_budget
        self.root_fraction, self.root_rounds, self.inout_lambda = root_fraction, root_rounds, inout_lambda
        self.lifting, self.local_search, self.ls_fraction, self.ls_window = lifting, local_search, ls_fraction, ls_window
        self.node_cut_rounds = node_cut_rounds
        self.verbose = verbose
        self.trace: list[dict[str, Any]] = []

    # ------------------------------------------------------------ bookkeeping

    def left(self) -> float:
        return self.deadline - time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def log(self, stage: str) -> None:
        self.trace.append({"t": round(self.elapsed(), 3), "incumbent": self.ub if self.ub < INF else None,
                           "bound": self.lb if self.lb > -INF else None, "stage": stage})
        if self.verbose:
            print(self.trace[-1], flush=True)

    def publish_bound(self, value: float, stage: str) -> None:
        value = certify(value)
        if value > self.lb and self.left() > 0:
            self.lb = value
            self.on_bound(value)
            self.log(stage)

    def offer(self, y: np.ndarray, value: float, stage: str, q: np.ndarray | None = None) -> bool:
        if value < self.ub - 1e-9 * max(1.0, abs(value)) and self.left() > 0:
            self.ub, self.best, self.best_q = value, y.copy(), q
            self.core = 0.5 + 0.5 * y
            self.pending = stage != "branch_and_cut"
            self.on_incumbent(open_list(y), stage=stage)
            self.log(stage)
            return True
        return False

    def closed(self) -> bool:
        return self.ub < INF and self.lb >= self.ub - 1e-6 * max(1.0, abs(self.ub))

    # ------------------------------------------------------------ run

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        return self.solve(load_instance(data_path))

    def solve(self, inst: SCFLInstance) -> dict[str, Any]:
        if not hasattr(self, "t0"):
            self.t0 = time.perf_counter()
            self.deadline = self.t0 + self.time_limit
        self.inst = inst
        self.ub, self.lb = INF, -INF
        self.best: np.ndarray | None = None
        self.best_q: np.ndarray | None = None
        self.core = np.ones(inst.I)
        self.pending = False
        self.stats: dict[str, Any] = {
            "root_rounds": 0, "cuts": 0, "root_cuts_total": 0, "root_cuts_kept": 0, "lazy_cuts": 0, "user_cuts": 0,
            "mipsol_calls": 0, "evaluations": 0, "injected": 0, "ls_calls": 0, "ls_seconds": 0.0, "master_seconds": 0.0,
            "sub_seconds": 0.0, "sep_solves": 0, "root_seconds": 0.0, "root_bound": None, "node_count": 0.0,
            "bc_status": None, "clusters": 0, "lifted_coefficients": 0}
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        self.env = env
        self.ev = self.sep = self.sur = self.pd = self.master = None
        reason = "time_limit"
        try:
            reason = self.search()
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        finally:
            for obj in (self.ev, self.sep, self.sur, self.pd, self.master):
                if obj is not None:
                    obj.dispose()
            env.dispose()
        ub = self.ub if self.ub < INF else None
        lb = self.lb if self.lb > -INF else None
        return {"best_objective": ub, "global_lb": lb, "global_ub": ub,
                "gap": (ub - lb) / abs(ub) if ub is not None and lb is not None and ub else None,
                "termination_reason": reason, "budget_exhausted": self.elapsed() >= self.time_limit - 1.0,
                "design": open_list(self.best) if self.best is not None else None,
                "stats": self.stats, **self.stats, "total_seconds": self.elapsed(), "trace": self.trace}

    def search(self) -> str:
        inst = self.inst
        self.lower = (inst.demand * inst.cost.min(axis=0)[None, :]).sum(axis=1)
        self.publish_bound(float(inst.prob @ self.lower), "trivial")
        self.ev = RecourseLP(inst, self.env)
        self.sep = RecourseLP(inst, self.env, penalty=True)
        # Master clusters within a flow-variable budget (the cluster block is a small extensive form); the local
        # search surrogate keeps `clusters` artificial scenarios in its own LP.
        self.labels, self.red = cluster_scenarios(inst, surrogate_size(inst, self.clusters, self.partial_budget))
        self.pd = RecourseLP(self.red, self.env)
        _, sur_red = cluster_scenarios(inst, surrogate_size(inst, self.clusters))
        self.sur = RecourseLP(sur_red, self.env)
        self.stats["clusters"] = self.red.S if self.partial else 0
        self.stats["surrogate_clusters"] = sur_red.S
        self.build_master()
        y0 = greedy_design(inst)
        res = self.evaluate(y0)
        if res is None:
            return "time_limit"
        self.offer(y0, res["value"], "initializer", res["q"])
        self.add_cuts(res["const"], res["coef"], np.arange(inst.S))
        if self.left() <= 0:
            return "time_limit"
        y_lp = self.root_loop()
        if self.closed():
            return "optimal"
        if y_lp is not None and self.left() > 0:
            self.lp_designs(y_lp)
        if self.local_search and self.left() > 0 and self.best is not None:
            self.polish(self.best, self.ub, self.ls_fraction * self.left(), "local_search", seed_cuts=True)
        if self.closed():
            return "optimal"
        if self.left() <= 0.01:
            return "time_limit"
        return self.branch_and_cut()

    # ------------------------------------------------------------ evaluation and cuts

    def evaluate(self, y: np.ndarray) -> dict[str, Any] | None:
        """Exact recourse of a binary design with its (lifted) scenario cuts; None if out of time."""
        t = time.perf_counter()
        res = self.ev.evaluate(y, self.deadline, duals=True)
        self.stats["sub_seconds"] += time.perf_counter() - t
        if res["status"] != "optimal":
            return None
        self.stats["evaluations"] += 1
        const, coef = cuts_from_duals(self.inst, np.arange(self.inst.S), res["V"], res["A"])
        if self.lifting:
            for s in range(self.inst.S):
                new = lift_closed(self.inst, s, res["V"][s], coef[s], y)
                self.stats["lifted_coefficients"] += int(np.sum(new < coef[s] - 1e-9))
                coef[s] = new
        return {"value": res["value"], "q": res["q"], "const": const, "coef": coef}

    def build_master(self) -> None:
        inst = self.inst
        I, J, S = inst.I, inst.J, inst.S
        md = gp.Model(env=self.env)
        md.Params.OutputFlag = 0
        md.Params.Seed = self.seed
        self.master = md
        self.y = md.addMVar(I, lb=0.0, ub=1.0, name="y")
        self.theta = md.addMVar(S, lb=self.lower, name="theta")
        md.setObjective(inst.fixed @ self.y + inst.prob @ self.theta, GRB.MINIMIZE)
        dmax = inst.max_demand
        md.addConstr(inst.capacity @ self.y >= dmax, name="cover")
        if np.any(inst.capacity > dmax):
            md.addConstr(np.minimum(inst.capacity, dmax) @ self.y >= dmax, name="cover_trunc")
        self.xbar = None
        if self.partial:
            red = self.red
            G = red.S
            self.xbar, _ = add_scenario_blocks(md, self.y, red.demand, inst.capacity, strong=True)
            # sum_{s in g} p_s theta_s >= P_g c.xbar_g for every cluster g
            W = sp.csr_matrix((inst.prob, (self.labels, np.arange(S))), shape=(G, S))
            R = sp.csr_matrix((np.repeat(red.prob, I * J) * np.tile(inst.cost.ravel(), G),
                               (np.repeat(np.arange(G), I * J), np.arange(G * I * J))), shape=(G, G * I * J))
            md.addConstr(W @ self.theta - R @ self.xbar >= 0.0, name="cluster")
        md.update()
        self.yv = self.y.tolist()
        self.thv = self.theta.tolist()
        self.cuts: list[gp.Constr] = []

    def add_cuts(self, const: np.ndarray, coef: np.ndarray, scen: np.ndarray) -> None:
        """Master rows theta_s + coef . y >= const (outside callbacks)."""
        if not len(scen):
            return
        C = np.where(np.abs(coef) > 1e-12, coef, 0.0)
        Cy = sp.csr_matrix(C)
        Et = sp.csr_matrix((np.ones(len(scen)), (np.arange(len(scen)), scen)), shape=(len(scen), self.inst.S))
        rows = self.master.addConstr(Cy @ self.y + Et @ self.theta >= const)
        self.master.update()
        self.cuts.extend(rows.tolist())
        self.stats["cuts"] += len(scen)

    def cut_expr(self, s: int, coef: np.ndarray) -> gp.LinExpr:
        nz = np.nonzero(np.abs(coef) > 1e-12)[0]
        expr = gp.LinExpr(coef[nz].tolist(), [self.yv[k] for k in nz])
        expr.add(self.thv[s])
        return expr

    def separate(self, point: np.ndarray, deadline: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Cuts of every scenario at a fractional point (penalized recourse with strong linking)."""
        inst = self.inst
        V = np.zeros((inst.S, inst.J))
        A = np.zeros((inst.S, inst.I))
        t = time.perf_counter()
        try:
            for s in self.sep.order:
                left = min(deadline, self.deadline) - time.perf_counter()
                if left <= 0:
                    return None
                res = self.sep.solve(int(s), point, left, vub=True)
                self.stats["sep_solves"] += 1
                if res["status"] != "optimal":
                    return None
                V[s], A[s] = res["v"], res["alpha"]
        finally:
            self.stats["sub_seconds"] += time.perf_counter() - t
        return cuts_from_duals(inst, np.arange(inst.S), V, A)

    # ------------------------------------------------------------ root

    def root_loop(self) -> np.ndarray | None:
        md = self.master
        t_root = time.perf_counter()
        root_deadline = t_root + self.root_fraction * max(0.0, self.left())
        core = self.core.copy()
        lam = self.inout_lambda
        best_bound, stalled = -INF, 0
        n_before = len(self.cuts)
        y_lp = None
        while self.stats["root_rounds"] < self.root_rounds and time.perf_counter() < root_deadline and self.left() > 0:
            t = time.perf_counter()
            md.Params.TimeLimit = max(0.01, self.left())
            md.optimize()
            self.stats["master_seconds"] += time.perf_counter() - t
            if md.Status != GRB.OPTIMAL:
                break
            self.stats["root_rounds"] += 1
            bound = float(md.ObjVal)
            self.publish_bound(bound, "root_lp")
            stalled = stalled + 1 if bound <= best_bound + 1e-6 * max(1.0, abs(bound)) else 0
            best_bound = max(best_bound, bound)
            if self.closed():
                break
            y_lp = np.clip(np.asarray(self.y.X), 0.0, 1.0)
            th = np.asarray(self.theta.X)
            point = lam * y_lp + (1.0 - lam) * core
            cuts = self.separate(point, root_deadline)
            if cuts is None:
                break
            const, coef = cuts
            viol = const - coef @ y_lp - th
            add = np.nonzero(viol > 1e-6 * np.maximum(1.0, np.abs(const)))[0]
            self.add_cuts(const[add], coef[add], add)
            core = 0.5 * core + 0.5 * point
            if len(add) == 0 or stalled >= 3:
                if lam < 1.0:
                    lam, stalled = 1.0, 0  # finish with Kelley cuts at the LP point
                    continue
                break
        self.stats["root_cuts_total"] = len(self.cuts) - n_before
        if self.left() > 0:
            md.Params.TimeLimit = max(0.01, self.left())
            md.optimize()
            if md.Status == GRB.OPTIMAL:
                self.publish_bound(float(md.ObjVal), "root_lp")
                y_lp = np.clip(np.asarray(self.y.X), 0.0, 1.0)
                # Cut management: drop the non-binding Benders cuts so the branch-and-cut master stays small.
                loose = [c for c in self.cuts if c.Slack < -1e-6 * max(1.0, abs(c.RHS))]
                if loose:
                    md.remove(loose)
                    gone = {id(c) for c in loose}
                    self.cuts = [c for c in self.cuts if id(c) not in gone]
                    md.update()
        self.stats["root_cuts_kept"] = len(self.cuts)
        self.stats["root_bound"] = self.lb
        self.stats["root_seconds"] = time.perf_counter() - t_root
        return y_lp

    # ------------------------------------------------------------ primal side

    def lp_designs(self, y_lp: np.ndarray) -> None:
        inst = self.inst
        rank = np.lexsort((np.arange(inst.I), -y_lp))
        seen: set[bytes] = set()
        for tau in (0.5, 0.25, 0.1, 0.01):
            if self.left() <= 0:
                return
            y = repair(inst, y_lp >= tau, rank)
            if y.tobytes() in seen:
                continue
            seen.add(y.tobytes())
            res = self.evaluate(y)
            if res is None:
                return
            self.offer(y, res["value"], "lp_rounding", res["q"])
            self.add_cuts(res["const"], res["coef"], np.arange(inst.S))

    def polish(self, y: np.ndarray, value: float, seconds: float, stage: str, seed_cuts: bool = False) -> None:
        """Time-capped surrogate-screened local search from y; publishes improvements; optionally seeds the master
        with the cuts of the final design (outside callbacks only)."""
        t = time.perf_counter()
        end = min(self.deadline, t + seconds)
        ds = DesignSearch(self.inst, self.ev, self.sur, lambda z, res: self.offer(z, res["value"], stage, res["q"]), seed=self.seed)
        z, v = ds.descend(y.copy(), value, end)
        self.stats["ls_calls"] += 1
        self.stats["ls_seconds"] += time.perf_counter() - t
        if seed_cuts and self.left() > 0 and self.best is not None:
            res = self.evaluate(self.best)
            if res is not None:
                self.add_cuts(res["const"], res["coef"], np.arange(self.inst.S))

    def master_start(self, y: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
        """A complete master solution for a design: y, theta = q, and the cluster-mean flows (None if out of time)."""
        X = None
        if self.partial:
            res = self.pd.evaluate(y, self.deadline, flows=True)
            if res["status"] != "optimal":
                return None
            X = np.stack([res["X"][g] for g in range(self.red.S)])
        return y.astype(float), q.copy(), X

    # ------------------------------------------------------------ branch-and-cut

    def branch_and_cut(self) -> str:
        inst, md = self.inst, self.master
        self.y.VType = GRB.BINARY
        md.Params.LazyConstraints = 1
        md.Params.PreCrush = 1
        md.Params.MIPGap = MIP_GAP
        if self.best is not None and self.best_q is not None:
            start = self.master_start(self.best, self.best_q)
            if start is not None:
                self.y.Start, self.theta.Start = start[0], start[1]
                if start[2] is not None:
                    self.xbar.Start = start[2].ravel()
            self.pending = False
        cache: OrderedDict[bytes, dict[str, Any]] = OrderedDict()
        polished: set[bytes] = set()
        node_rounds = [0]
        bc_start = time.perf_counter()
        xbflat = self.xbar.tolist() if self.xbar is not None else None

        def lookup(yy: np.ndarray) -> dict[str, Any] | None:
            key = yy.tobytes()
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            res = self.evaluate(yy)
            if res is None:
                return None
            cache[key] = res
            while len(cache) > 64:
                cache.popitem(last=False)
            return dict(res, fresh=True)

        errors: list[BaseException] = []

        def callback(cb: gp.Model, where: int) -> None:
            # gurobipy prints and ignores exceptions raised in a callback, which would accept an unchecked design.
            try:
                body(cb, where)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                cb.terminate()

        def body(cb: gp.Model, where: int) -> None:
            if self.left() <= 0:
                cb.terminate()
                return
            if where == GRB.Callback.MIPSOL:
                self.stats["mipsol_calls"] += 1
                yy = np.asarray(cb.cbGetSolution(self.yv)) > 0.5
                th = np.asarray(cb.cbGetSolution(self.thv))
                res = lookup(yy)
                if res is None:
                    cb.terminate()  # out of time
                    return
                if res.get("fresh"):
                    self.offer(yy, res["value"], "branch_and_cut", res["q"])
                    key = yy.tobytes()
                    bc_time = time.perf_counter() - bc_start
                    if (self.local_search and key not in polished and res["value"] <= self.ub * (1.0 + self.ls_window)
                            and self.stats["ls_seconds"] <= 0.25 * bc_time + 5.0):
                        polished.add(key)
                        self.polish(yy, res["value"], min(30.0, 0.05 * max(0.0, self.left())), "bc_local_search")
                viol = res["const"] - res["coef"] @ yy - th
                for s in np.nonzero(viol > 1e-6 * np.maximum(1.0, np.abs(res["const"])))[0]:
                    cb.cbLazy(self.cut_expr(int(s), res["coef"][s]) >= float(res["const"][s]))
                    self.stats["lazy_cuts"] += 1
            elif where == GRB.Callback.MIPNODE:
                if cb.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                    return
                if self.pending and self.best is not None and self.best_q is not None:
                    start = self.master_start(self.best, self.best_q)
                    if start is not None:
                        cb.cbSetSolution(self.yv, start[0].tolist())
                        cb.cbSetSolution(self.thv, start[1].tolist())
                        if xbflat is not None:
                            cb.cbSetSolution(xbflat, start[2].ravel().tolist())
                        cb.cbUseSolution()
                        self.stats["injected"] += 1
                    self.pending = False
                if cb.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0 and node_rounds[0] < self.node_cut_rounds:
                    node_rounds[0] += 1
                    y_node = np.clip(np.asarray(cb.cbGetNodeRel(self.yv)), 0.0, 1.0)
                    th = np.asarray(cb.cbGetNodeRel(self.thv))
                    cuts = self.separate(y_node, self.deadline)
                    if cuts is not None:
                        const, coef = cuts
                        viol = const - coef @ y_node - th
                        for s in np.nonzero(viol > 1e-4 * np.maximum(1.0, np.abs(const)))[0]:
                            cb.cbCut(self.cut_expr(int(s), coef[s]) >= float(const[s]))
                            self.stats["user_cuts"] += 1
            elif where == GRB.Callback.MIP:
                bnd = cb.cbGet(GRB.Callback.MIP_OBJBND)
                if bnd < GRB.INFINITY:
                    self.publish_bound(float(bnd), "branch_and_cut")

        t = time.perf_counter()
        md.Params.TimeLimit = max(0.01, self.left())
        md.optimize(callback)
        self.stats["master_seconds"] += time.perf_counter() - t
        if errors:
            raise errors[0]
        self.stats["node_count"] = float(md.NodeCount)
        self.stats["bc_status"] = int(md.Status)
        if md.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.MEM_LIMIT):
            self.publish_bound(float(md.ObjBound), "final")
        if md.Status == GRB.OPTIMAL:
            return "optimal"
        if md.Status == GRB.MEM_LIMIT:
            return "memory_limit"
        if md.Status == GRB.INFEASIBLE:
            raise RuntimeError("Benders master infeasible: the instance has no feasible design or a cut is invalid")
        if md.Status == GRB.TIME_LIMIT or (md.Status == GRB.INTERRUPTED and self.left() <= 0.5):
            return "time_limit"
        return "stuck" if md.Status == GRB.INTERRUPTED else f"status_{md.Status}"
