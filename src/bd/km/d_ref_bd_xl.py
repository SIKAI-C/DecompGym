"""Engineered Benders reference for the k-median (scale-up study), after Duran-Mateluna, Ales and Elloumi (2023, EJOR):
disaggregated per-customer cuts from the closed-form oracle (c_naive_bd_xl.Oracle), a two-phase scheme, and
branch-and-Benders-cut.

Master (classical y-space, see c_naive_bd_xl): min sum_j theta_j, sum_i y_i = k, y in [0, 1]^m (Phase 1) or
{0, 1}^m (Phase 2), theta_j >= 0, plus per-customer cuts theta_j >= v - sum_i (v - D_ij)^+ y_i.

Stages, all within one wall-clock budget:
1. Shared basic initializer (greedy add, identical to the naive controls); its n cuts seed the master (the naive LP
   master starts from the same cuts, aggregated).
2. Phase 1, LP master (cutting planes, up to `root_fraction` of the budget): dual simplex re-solves; in-out
   stabilization (Ben-Ameur and Neto 2007; Fischetti, Ljubic and Sinnl 2017): separation at
   lam * y_lp + (1 - lam) * core with core <- (core + separation point) / 2, lam -> 1 when no cut is violated at the
   LP point or the bound stalls; every violated per-customer cut is added; the rounding heuristic of Duran-Mateluna
   et al. (open the k largest y_lp) runs every round. Phase 1 ends when the LP master value equals sum_j Q_j(y_lp)
   (the LP bound of the classical formulation). Cut management for Phase 2: per customer, the cuts with v above the
   largest v among its binding cuts are dropped (Duran-Mateluna et al. Section 3.4.2); the cut log keeps every cut.
3. Swap local search (f_heuristic_xl.SwapSearch, capped at `ls_fraction` of the remaining time) from the best design.
4. Reduced-cost fixing with the Phase 1 LP and the incumbent (integral distances: any solution with value > UB - 1
   cannot improve).
5. Phase 2, Gurobi branch-and-Benders-cut on the integer master (MIPFocus 3, branch up first): lazy cuts at every
   integer solution (exact, closed form), user cuts at fractional nodes (capped share of the time), new incumbents
   polished by a short swap descent and injected. The tree bound is published as a solver-reported diagnostic
   (min(UB, ObjBound), valid with the fixings).

Outputs for the certificate checker: the cut log (time, customer, v) of every cut generated (Phases 1 and 2), the
Phase 1 LP point and value (the primal side of the LP* interval).
"""
from __future__ import annotations

import math
import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from src.bd.km.b_mip_xl import MIP_GAP, design_value, open_list
from src.bd.km.c_naive_bd_xl import INF, Oracle, _Base, greedy_design
from src.bd.km.f_heuristic_xl import SwapSearch
from src.bd.scfl.b_mip_xl import is_memory_error, make_env


class KMBendersXL(_Base):
    """Engineered Benders reference; see the module docstring."""

    def __init__(self, *, root_fraction: float = 0.7, root_rounds: int = 1000, inout_lambda: float = 0.3,
                 stabilize: bool = True, ls_fraction: float = 0.1, rc_fixing: bool = True, branch: bool = True,
                 user_cut_share: float = 0.2, polish_share: float = 0.2, **kw) -> None:
        super().__init__(**kw)
        self.root_fraction, self.root_rounds, self.inout_lambda = root_fraction, root_rounds, inout_lambda
        self.stabilize, self.ls_fraction, self.rc_fixing, self.branch = stabilize, ls_fraction, rc_fixing, branch
        self.user_cut_share, self.polish_share = user_cut_share, polish_share

    # ------------------------------------------------------------ cuts

    def add_cuts(self, js: np.ndarray, v: np.ndarray, cnt: np.ndarray) -> None:
        """Master rows theta_j + sum (v - D_ij) y_i >= v for the customers js (v, cnt aligned with js)."""
        if not len(js):
            return
        A = self.oracle.cut_matrix(js, v, cnt)
        E = sp.csr_matrix((np.ones(len(js)), (np.arange(len(js)), js)), shape=(len(js), self.inst.n))
        rows = self.md.addConstr(A @ self.y + E @ self.th >= v)
        self.md.update()
        self.rows.extend(rows.tolist())
        self.row_j.extend(int(j) for j in js)
        self.row_v.extend(float(x) for x in v)
        self.log_cuts(js, v)

    def log_cuts(self, js: np.ndarray, v: np.ndarray) -> None:
        t = self.elapsed()
        self.cut_t.append(np.full(len(js), t))
        self.cut_j.append(np.asarray(js, dtype=np.int32))
        self.cut_vv.append(np.asarray(v, dtype=float))
        self.stats["cuts"] += len(js)

    def publish_active(self, force: bool = False) -> None:
        """Certificate version: the master cuts with positive LP duals (largest first) within the size limit; they
        reproduce the current LP-master value."""
        from src.bd.km.g_certificate_xl import fit_budget, singles, sorted_counts
        if not self.rows or not self.cert_due(force):
            return
        try:
            pi = np.asarray(self.md.getAttr("Pi", self.rows))
        except gp.GurobiError:
            return
        act = np.nonzero(pi > 1e-12)[0]
        act = act[np.argsort(-pi[act], kind="stable")]
        rj, rv = np.asarray(self.row_j)[act], np.asarray(self.row_v)[act]
        keep = fit_budget(sorted_counts(self.D, self.oracle.sc), rj, rv, self.cert_limit)
        self.emit(singles(rj[keep], rv[keep]))
        self.stats["certificate_versions"] = self._cert_seq

    def cut_values(self, v: np.ndarray, cnt: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Value at y of every customer's cut (j, v_j)."""
        sc = self.oracle.sc
        Y = y[sc.order]
        mask = np.arange(sc.L)[None, :] < cnt[:, None]
        return v - (np.where(mask, v[:, None] - sc.ds, 0.0) * Y).sum(axis=1)

    # ------------------------------------------------------------ run

    def run(self, data_path: str) -> dict[str, Any]:
        self.start(data_path)
        self.cut_t, self.cut_j, self.cut_vv = [], [], []
        self.stats: dict[str, Any] = {
            "cuts": 0, "root_rounds": 0, "phase1_converged": False, "phase1_seconds": None, "lp_value": None,
            "root_cuts_kept": 0, "fixed_zero": 0, "fixed_one": 0, "lazy_cuts": 0, "user_cuts": 0, "mipsol_calls": 0,
            "node_count": 0.0, "bc_status": None, "ls_seconds": 0.0, "user_cut_seconds": 0.0, "injected": 0,
            "master_seconds": 0.0, "bound_kind": "solver_reported (Phase 1 LP master, then branch-and-cut tree)"}
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        self.md = None
        reason = "time_limit"
        try:
            reason = self.search(env)
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        finally:
            if self.md is not None:
                self.md.dispose()
            env.dispose()
        ub = self.ub if self.ub < INF else None
        lb = self.lb if self.lb > -INF else None
        log = {"kind": "single",
               "t": np.concatenate(self.cut_t) if self.cut_t else np.zeros(0),
               "j": np.concatenate(self.cut_j) if self.cut_j else np.zeros(0, dtype=np.int32),
               "v": np.concatenate(self.cut_vv) if self.cut_vv else np.zeros(0)}
        return {"best_objective": ub, "global_lb": lb, "termination_reason": reason,
                "design": open_list(self.best) if self.best is not None else None, "stats": self.stats, **self.stats,
                "total_seconds": self.elapsed(), "trace": self.trace, "cut_log": log}

    def search(self, env: gp.Env) -> str:
        inst, D = self.inst, self.D
        self.oracle = oracle = Oracle(D, inst.k)
        self.ss = SwapSearch(D, inst.k, seed=self.seed)
        y0 = greedy_design(D, inst.k)
        self.offer(y0, design_value(D, open_list(y0)), "initializer")
        md = gp.Model(env=env)
        self.md = md
        md.Params.OutputFlag = 0
        md.Params.Method = 1
        self.y = md.addMVar(inst.m, lb=0.0, ub=1.0, name="y")
        self.th = md.addMVar(inst.n, lb=0.0, name="theta")
        md.addConstr(self.y.sum() == inst.k, name="card")
        md.setObjective(self.th.sum(), GRB.MINIMIZE)
        md.update()
        self.yv, self.thv = self.y.tolist(), self.th.tolist()
        self.rows: list[gp.Constr] = []
        self.row_j: list[int] = []
        self.row_v: list[float] = []
        v, Q, cnt = oracle.separate(y0.astype(float))
        js = np.arange(inst.n)
        self.add_cuts(js, v, cnt)
        y_lp = self.phase1(y0.astype(float))
        if self.left() <= 0:
            return "time_limit"
        if self.best is not None and self.ls_fraction > 0:
            t = time.perf_counter()
            z, zv = self.ss.descend(self.best, min(self.deadline, t + self.ls_fraction * self.left()),
                                    lambda yy, val: self.offer(yy, val, "local_search"))
            self.stats["ls_seconds"] += time.perf_counter() - t
        if not self.branch or self.left() <= 0.01:
            return "phase1_converged" if self.stats["phase1_converged"] else "time_limit"
        if self.closed():
            return "optimal"
        return self.branch_and_cut(y_lp)

    # ------------------------------------------------------------ Phase 1

    def phase1(self, core: np.ndarray) -> np.ndarray | None:
        inst, md, oracle = self.inst, self.md, self.oracle
        t1 = time.perf_counter()
        end = t1 + self.root_fraction * max(0.0, self.left())
        lam = self.inout_lambda if self.stabilize else 1.0
        best, stall, y_lp = -INF, 0, None
        tried: set[bytes] = set()
        while self.stats["root_rounds"] < self.root_rounds and time.perf_counter() < end and self.left() > 0:
            t = time.perf_counter()
            md.Params.TimeLimit = max(0.01, min(self.left(), end - t))
            md.optimize()
            self.stats["master_seconds"] += time.perf_counter() - t
            if md.Status != GRB.OPTIMAL:
                break
            self.stats["root_rounds"] += 1
            lb_mp = float(md.ObjVal)
            self.publish_bound(lb_mp, "phase1_lp")
            if self.cert_due():
                self.publish_active()
            stall = stall + 1 if lb_mp <= best + 1e-7 * max(1.0, abs(lb_mp)) else 0
            best = max(best, lb_mp)
            y_lp = np.clip(np.asarray(self.y.X), 0.0, 1.0)
            th = np.asarray(self.th.X)
            top = np.zeros(inst.m, dtype=bool)
            top[np.argsort(-y_lp, kind="stable")[:inst.k]] = True
            key = top.tobytes()
            if key not in tried:
                tried.add(key)
                self.offer(top, design_value(self.D, open_list(top)), "lp_rounding")
            point = y_lp if lam >= 1.0 else lam * y_lp + (1.0 - lam) * core
            v, Q, cnt = oracle.separate(point)
            if lam >= 1.0:
                ub_mp = float(Q.sum())
                if ub_mp - lb_mp <= 1e-7 * max(1.0, abs(lb_mp)):
                    self.stats["phase1_converged"] = True
                    break
                at = Q
            else:
                at = self.cut_values(v, cnt, y_lp)
            viol = at - th
            add = np.nonzero(viol > 1e-7 * np.maximum(1.0, np.abs(v)))[0]
            if len(add) == 0 or (stall >= 5 and lam < 1.0):
                if lam < 1.0:
                    lam, stall = 1.0, 0
                    continue
                self.stats["phase1_converged"] = True  # no violated cut at the LP point
                break
            self.add_cuts(add, v[add], cnt[add])
            core = 0.5 * core + 0.5 * point
        if md.Status == GRB.OPTIMAL and self.left() > 0:
            self.publish_active(force=True)
        self.stats["phase1_seconds"] = time.perf_counter() - t1
        self.stats["phase1_end"] = self.elapsed()
        if y_lp is not None:
            self.stats["lp_value"] = best
            self.stats["final_lp_point"] = y_lp.tolist()
        # Cut management for Phase 2 (the cut log keeps every cut); re-solve first (cuts may have been added after the
        # last solve; warm-started, instantaneous when nothing changed).
        if y_lp is not None and self.left() > 0:
            md.Params.TimeLimit = max(0.01, self.left())
            md.optimize()
        if y_lp is not None and self.left() > 0 and md.Status == GRB.OPTIMAL:
            slack = np.array([c.Slack for c in self.rows])
            rj, rv = np.array(self.row_j), np.array(self.row_v)
            binding = np.abs(slack) <= 1e-6 * np.maximum(1.0, np.abs(rv))
            hmax = np.full(inst.n, -INF)
            np.maximum.at(hmax, rj[binding], rv[binding])
            drop = rv > hmax[rj] + 1e-12
            if drop.any():
                md.remove([c for c, d in zip(self.rows, drop) if d])
                keep = ~drop
                self.rows = [c for c, kp in zip(self.rows, keep) if kp]
                self.row_j = rj[keep].tolist()
                self.row_v = rv[keep].tolist()
                md.update()
            self.stats["root_cuts_kept"] = len(self.rows)
        return y_lp

    # ------------------------------------------------------------ Phase 2

    def branch_and_cut(self, y_lp: np.ndarray | None) -> str:
        inst, md, D, oracle = self.inst, self.md, self.D, self.oracle
        if self.rc_fixing and y_lp is not None and self.ub < INF and self.left() > 0:
            md.Params.TimeLimit = max(0.01, self.left())
            md.optimize()
            if md.Status == GRB.OPTIMAL:
                lb, rc = float(md.ObjVal), np.asarray(self.y.RC)
                yv = np.asarray(self.y.X)
                thr = self.ub - 1.0 + 1e-6 if self.integral else self.ub * (1 + 1e-9) + 1e-9
                z0 = (yv <= 1e-9) & (lb + rc > thr)
                z1 = (yv >= 1 - 1e-9) & (lb - rc > thr)
                ub_ = np.ones(inst.m)
                lb_ = np.zeros(inst.m)
                ub_[z0] = 0.0
                lb_[z1] = 1.0
                if int(z1.sum()) <= inst.k and inst.m - int(z0.sum()) >= inst.k:
                    self.y.UB, self.y.LB = ub_, lb_
                    self.stats["fixed_zero"], self.stats["fixed_one"] = int(z0.sum()), int(z1.sum())
        self.y.VType = GRB.BINARY
        md.Params.LazyConstraints = 1
        md.Params.PreCrush = 1
        if self.integral:
            md.Params.MIPGap = 0.0
            md.Params.MIPGapAbs = 0.999
        else:
            md.Params.MIPGap = MIP_GAP
        md.Params.MIPFocus = 3
        md.Params.BranchDir = 1
        md.Params.Method = 1
        if self.best is not None:
            self.y.Start = self.best.astype(float)
            self.th.Start = D[self.best].min(axis=0)
        pending: list[np.ndarray | None] = [None]
        seen: set[bytes] = set()
        bc0 = time.perf_counter()
        errors: list[BaseException] = []

        def polish(yy: np.ndarray, value: float) -> None:
            spent = time.perf_counter() - bc0
            if self.stats["ls_seconds"] > self.polish_share * spent + 2.0:
                return
            t = time.perf_counter()
            z, zv = self.ss.descend(yy, min(self.deadline, t + min(10.0, 0.05 * max(0.0, self.left()))))
            self.stats["ls_seconds"] += time.perf_counter() - t
            if self.offer(z, zv, "bc_local_search"):
                pending[0] = z.copy()

        def body(cb: gp.Model, where: int) -> None:
            if self.left() <= 0:
                cb.terminate()
                return
            if where == GRB.Callback.MIPSOL:
                self.stats["mipsol_calls"] += 1
                yy = np.asarray(cb.cbGetSolution(self.yv)) > 0.5
                if int(yy.sum()) != inst.k:
                    return
                th = np.asarray(cb.cbGetSolution(self.thv))
                c = D[yy].min(axis=0)
                key = yy.tobytes()
                if key not in seen:
                    seen.add(key)
                    value = design_value(D, open_list(yy))
                    self.offer(yy, value, "branch_and_cut")
                    if value <= self.ub * 1.01 + 1e-9:
                        polish(yy, value)
                viol = np.nonzero(c - th > 1e-6 * np.maximum(1.0, c))[0]
                if len(viol):
                    v = c  # at a binary design the optimal dual is the distance to the nearest open facility
                    for j in viol:
                        idx, coef = oracle.coef_dense(int(j), float(v[j]))
                        expr = gp.LinExpr(coef.tolist(), [self.yv[i] for i in idx])
                        expr.add(self.thv[int(j)])
                        cb.cbLazy(expr >= float(v[j]))
                    self.log_cuts(viol, v[viol])
                    self.stats["lazy_cuts"] += len(viol)
            elif where == GRB.Callback.MIPNODE:
                if cb.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                    return
                if pending[0] is not None:
                    z = pending[0]
                    pending[0] = None
                    cb.cbSetSolution(self.yv, z.astype(float).tolist())
                    cb.cbSetSolution(self.thv, D[z].min(axis=0).tolist())
                    cb.cbUseSolution()
                    self.stats["injected"] += 1
                spent = time.perf_counter() - bc0
                if self.stats["user_cut_seconds"] > self.user_cut_share * spent + 1.0:
                    return
                t = time.perf_counter()
                yn = np.clip(np.asarray(cb.cbGetNodeRel(self.yv)), 0.0, 1.0)
                thn = np.asarray(cb.cbGetNodeRel(self.thv))
                v, Q, cnt = oracle.separate(yn)
                at = self.cut_values(v, cnt, yn)
                viol = np.nonzero(at - thn > 1e-4 * np.maximum(1.0, np.abs(v)))[0]
                if len(viol) > inst.n // 2:
                    viol = viol[np.argsort(-(at - thn)[viol])[: inst.n // 2]]
                for j in viol:
                    idx, coef = oracle.coef_dense(int(j), float(v[j]))
                    expr = gp.LinExpr(coef.tolist(), [self.yv[i] for i in idx])
                    expr.add(self.thv[int(j)])
                    cb.cbCut(expr >= float(v[j]))
                if len(viol):
                    self.log_cuts(viol, v[viol])
                    self.stats["user_cuts"] += len(viol)
                self.stats["user_cut_seconds"] += time.perf_counter() - t
            elif where == GRB.Callback.MIP:
                bnd = cb.cbGet(GRB.Callback.MIP_OBJBND)
                if bnd < GRB.INFINITY:
                    self.publish_bound(min(float(bnd), self.ub), "branch_and_cut")

        def callback(cb: gp.Model, where: int) -> None:
            # gurobipy prints and ignores exceptions raised in a callback, which would accept an unchecked design.
            try:
                body(cb, where)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                cb.terminate()

        t = time.perf_counter()
        md.Params.TimeLimit = max(0.01, self.left())
        md.optimize(callback)
        self.stats["master_seconds"] += time.perf_counter() - t
        if errors:
            raise errors[0]
        self.stats["node_count"] = float(md.NodeCount)
        self.stats["bc_status"] = int(md.Status)
        if md.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED) and math.isfinite(md.ObjBound):
            self.publish_bound(min(float(md.ObjBound), self.ub), "final")
        if md.Status == GRB.INFEASIBLE:
            # every design better than the incumbent was fixed away: the incumbent is optimal
            if self.ub < INF and (self.stats["fixed_zero"] or self.stats["fixed_one"]):
                self.publish_bound(self.ub, "final")
                return "optimal"
            raise RuntimeError("Benders master infeasible: a cut is invalid")
        if self.closed():
            return "optimal"
        if md.Status == GRB.OPTIMAL:
            return "optimal_gap_not_closed"
        if md.Status == GRB.TIME_LIMIT or (md.Status == GRB.INTERRUPTED and self.left() <= 0.5):
            return "time_limit"
        return f"status_{md.Status}"
