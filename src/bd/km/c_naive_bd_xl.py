"""k-median (scale-up): the per-customer recourse oracle shared by every Benders method, the shared basic
initializer, and the naive Benders controls.

View (explicit, classical y-space). Master variables y in [0, 1]^m with sum_i y_i = k, and one recourse value
theta_j per customer (subproblem ID j, trusted weight 1 in the objective sum_j theta_j). Customer j's subproblem is
    Q_j(y) = min sum_i D_ij x_ij  s.t.  sum_i x_ij = 1,  0 <= x_ij <= y_i     (x_ij <= 1 implied),
with dual  max v - sum_i w_i y_i  s.t.  v - w_i <= D_ij,  w_i >= 0.  Every v with the cheapest completion
w_i = (v - D_ij)^+ is dual feasible, so
    theta_j >= v - sum_i (v - D_ij)^+ y_i                                                     (cut (v; j))
is valid for every y in [0, 1]^m with sum y >= 1 (weak duality); this is the y-space form of the radius-formulation
cut of Duran-Mateluna, Ales and Elloumi (2023, Corollary 1): v = D_j^{k+1} gives their cut (20). A cut is therefore
exported as (j, v) (or, aggregated, as one v per customer of a set of unique IDs); g_certificate_xl re-derives it.

Oracle (closed form, Duran-Mateluna et al. Algorithm 2): at a point y (fractional or binary) the LP subproblem fills
the facilities in increasing distance; v_j = the distance of the facility at which the cumulative sum of y first
reaches 1 is an optimal dual and Q_j(y) = v_j - sum_i y_i (v_j - D_ij)^+. At a binary design v_j is the distance to
the nearest open facility. `Oracle.separate` evaluates all customers at once on the sorted structure (SortedCosts).

Shared basic initializer (`greedy_design`, every method that needs a start): the classical greedy add heuristic
(open the facility with the largest cost reduction, k times; the first is the 1-median), implemented with incremental
gain updates.

NAIVE BENDERS, bound side (`KMNaiveLPBenders`, the executable control of protocol Section 20.6 for the certified
bound): unstabilized Kelley cutting planes on the LP master min theta s.t. sum y = k, y in [0, 1]^m, theta >= 0 and
the cuts; ONE aggregated cut per iteration, theta >= sum_j (v_j - sum_i (v_j - D_ij)^+ y_i), from the same oracle
at the master's LP point; initialized with the aggregated cut at the shared initializer's design (the engineered
method's first cuts are the same, disaggregated). Stops when the master value equals sum_j Q_j(y) (the LP bound).
Its only incumbent is the initializer's design.

NAIVE BENDERS, integer master (`KMNaiveBenders`, diagnostic only): the same with y binary, the master re-solved to
optimality every iteration (the textbook outer loop), every master design evaluated and published.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from src.bd.km.a_data_xl import SortedCosts, integral_costs, load_instance
from src.bd.km.b_mip_xl import design_value, gap_closed, open_list
from src.bd.scfl.b_mip_xl import is_memory_error, make_env

INF = float("inf")


def certify_float(bound: float) -> float:
    """A solver-reported bound made safe against floating-point error (diagnostic bounds only; certified bounds
    come from g_certificate_xl)."""
    return bound - 1e-9 * max(1.0, abs(bound)) - 1e-9


# ---------------------------------------------------------------- oracle

class Oracle:
    def __init__(self, D: np.ndarray, k: int, sc: SortedCosts | None = None) -> None:
        self.D = D
        self.k = k
        self.sc = sc or SortedCosts(D, k)
        self.n, self.m = D.shape[1], D.shape[0]
        self.calls = 0
        self.seconds = 0.0

    def separate(self, y: np.ndarray, tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(v, Q, cnt): optimal dual v_j, subproblem value Q_j(y) and the number of facilities strictly closer than
        v_j (the cut's support: the first cnt_j sorted positions)."""
        t = time.perf_counter()
        sc = self.sc
        Y = np.asarray(y, dtype=float)[sc.order]
        C = np.cumsum(Y, axis=1)
        hit = C >= 1.0 - tol
        crit = np.argmax(hit, axis=1)
        crit[~hit[:, -1]] = sc.L - 1  # cannot happen for sum y = k, y <= 1 (kept for safety)
        rows = np.arange(sc.n)
        v = sc.ds[rows, crit]
        below = sc.ds < v[:, None]
        cnt = below.sum(axis=1)
        gap = np.where(below, v[:, None] - sc.ds, 0.0)
        Q = v - (gap * Y).sum(axis=1)
        self.calls += 1
        self.seconds += time.perf_counter() - t
        return v, Q, cnt

    def cut_matrix(self, js: np.ndarray, v: np.ndarray, cnt: np.ndarray) -> sp.csr_matrix:
        """Coefficients (v_r - D_ij) on y of the cuts (js[r], v[r]); v and cnt are aligned with js (row r)."""
        sc = self.sc
        js = np.asarray(js, dtype=np.int64)
        c = np.asarray(cnt, dtype=np.int64)
        vv = np.asarray(v, dtype=float)
        r = np.repeat(np.arange(len(js)), c)
        t = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
        jj = js[r]
        cols = sc.order[jj, t]
        vals = vv[r] - sc.ds[jj, t]
        return sp.csr_matrix((vals, (r, cols)), shape=(len(js), self.m))

    def coef_dense(self, j: int, v: float) -> tuple[np.ndarray, np.ndarray]:
        """(facility indices, coefficients) of the cut (j, v)."""
        c = int(np.searchsorted(self.sc.ds[j], v, side="left"))
        idx = self.sc.order[j, :c]
        return idx, v - self.sc.ds[j, :c]

    def aggregate(self, v: np.ndarray, cnt: np.ndarray) -> tuple[float, np.ndarray]:
        """(constant, coefficient vector) of the aggregated cut sum_j theta_j >= sum_j v_j - coef . y."""
        A = self.cut_matrix(np.arange(self.n), v, cnt)
        return float(math.fsum(v.tolist())), np.asarray(A.sum(axis=0)).ravel()

    def evaluate(self, y: np.ndarray) -> float:
        _, Q, _ = self.separate(y)
        return float(Q.sum())


# ---------------------------------------------------------------- shared initializer

def greedy_design(D: np.ndarray, k: int) -> np.ndarray:
    """Greedy add (incremental gains); returns a boolean design with exactly k open facilities."""
    m, n = D.shape
    y = np.zeros(m, dtype=bool)
    i0 = int(np.argmin(D.sum(axis=1)))
    y[i0] = True
    cur = D[i0].copy()
    gain = np.maximum(cur[None, :] - D, 0.0).sum(axis=1)
    for _ in range(k - 1):
        g = np.where(y, -INF, gain)
        i = int(np.argmax(g))
        y[i] = True
        ch = np.nonzero(D[i] < cur)[0]
        if len(ch):
            old, new = cur[ch], D[i, ch]
            sub = D[:, ch]
            gain -= (np.maximum(old[None, :] - sub, 0.0) - np.maximum(new[None, :] - sub, 0.0)).sum(axis=1)
            cur[ch] = new
    return y


# ---------------------------------------------------------------- naive LP-master Benders (bound side)

class _Base:
    """Shared bookkeeping: incumbents and solver-reported bounds are published immediately; certificate versions
    (km-cert-v1, within the frozen expanded-size limit) through on_certificate, at most every `cert_interval`
    seconds plus at termination."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, on_certificate: Callable | None = None, soft_mem_gb: float | None = None,
                 seed: int = 0, verbose: bool = False, cert_interval: float = 5.0, cert_limit: int = 2_000_000) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed, self.verbose = time_limit, threads, soft_mem_gb, seed, verbose
        self.on_incumbent = on_incumbent or (lambda design, stage=None: None)
        self.on_bound = on_bound or (lambda bound: None)
        self.on_certificate = on_certificate or (lambda cert: None)
        self.cert_interval, self.cert_limit = cert_interval, cert_limit
        self.trace: list[dict[str, Any]] = []
        self._last_cert = -INF
        self._cert_seq = 0

    def left(self) -> float:
        return self.deadline - time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def log(self, stage: str, **kw) -> None:
        self.trace.append({"t": round(self.elapsed(), 3), "incumbent": self.ub if self.ub < INF else None,
                           "bound": self.lb if self.lb > -INF else None, "stage": stage, **kw})
        if self.verbose:
            print(self.trace[-1], flush=True)

    def publish_bound(self, value: float, stage: str) -> None:
        if self.left() > 0:
            self.lb_raw = max(self.lb_raw, value)
        value = certify_float(value)
        if value > self.lb and self.left() > 0:
            self.lb = value
            self.on_bound(value)
            self.log(stage)

    def offer(self, y: np.ndarray, value: float, stage: str) -> bool:
        if value < self.ub - 1e-9 * max(1.0, abs(value)) and self.left() > 0:
            self.ub, self.best = value, np.asarray(y, dtype=bool).copy()
            self.on_incumbent(open_list(self.best), stage=stage)
            self.log(stage)
            return True
        return False

    def closed(self) -> bool:
        """Unit-gap closure for integral costs (no relative shortcut; Codex review 46 #5), else relative 1e-6; on the
        solver-reported bound (b_mip_xl.gap_closed)."""
        return self.ub < INF and self.lb_raw > -INF and gap_closed(self.ub, self.lb_raw, self.integral)

    def cert_due(self, force: bool = False) -> bool:
        return self.left() > 0 and (force or time.perf_counter() - self._last_cert >= self.cert_interval)

    def emit(self, cert) -> None:
        if self.left() > 0:
            self._cert_seq += 1
            cert.version = self._cert_seq
            self.on_certificate(cert)
            self._last_cert = time.perf_counter()

    def start(self, data_path: str) -> None:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        self.inst = load_instance(data_path)
        self.D = self.inst.distances()
        self.integral = integral_costs(self.D)
        self.ub, self.lb, self.lb_raw, self.best = INF, -INF, -INF, None
        self.cut_t: list[float] = []      # aggregated cuts: time, v vector (ids = all customers)
        self.cut_v: list[np.ndarray] = []
        self.cut_size: list[int] = []     # expanded size of each aggregated cut (frozen counting rule)

    def agg_size(self, oracle: Oracle, v: np.ndarray) -> int:
        from src.bd.km.g_certificate_xl import sorted_counts
        return int(len(v) + sorted_counts(self.D, oracle.sc)(np.arange(len(v)), v).sum())

    def publish_aggregated(self, priority: list[int], force: bool = False) -> None:
        """Certificate version: the aggregated cuts in priority order that fit the size limit."""
        from src.bd.km.g_certificate_xl import aggregated
        if not self.cert_due(force) or not self.cut_v:
            return
        keep, size = [], 0
        for c in priority:
            if size + self.cut_size[c] <= self.cert_limit:
                keep.append(c)
                size += self.cut_size[c]
        self.emit(aggregated([self.cut_v[c] for c in keep], self.inst.n))

    def finish(self, reason: str, extra: dict) -> dict[str, Any]:
        ub = self.ub if self.ub < INF else None
        lb = self.lb if self.lb > -INF else None
        return {"best_objective": ub, "global_lb": lb, "termination_reason": reason,
                "design": open_list(self.best) if self.best is not None else None,
                "total_seconds": self.elapsed(), "trace": self.trace,
                "cut_log": {"kind": "aggregated", "t": self.cut_t, "v": [v.tolist() for v in self.cut_v]}, **extra}


class KMNaiveLPBenders(_Base):
    """Naive LP-master Benders (bound side); see the module docstring. Certificate versions: the aggregated cuts with
    positive master duals (largest first) that fit the size limit."""

    def run(self, data_path: str) -> dict[str, Any]:
        self.start(data_path)
        inst, D = self.inst, self.D
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        stats = {"iterations": 0, "master_seconds": 0.0, "lp_converged": False, "bound_kind": "solver_reported_lp_master"}
        reason = "time_limit"
        try:
            oracle = Oracle(D, inst.k)
            y0 = greedy_design(D, inst.k)
            self.offer(y0, design_value(D, open_list(y0)), "initializer")
            md = gp.Model(env=env)
            md.Params.OutputFlag = 0
            md.Params.Method = 1
            y = md.addMVar(inst.m, lb=0.0, ub=1.0, name="y")
            th = md.addVar(lb=0.0, name="theta")
            md.addConstr(y.sum() == inst.k)
            md.setObjective(th, GRB.MINIMIZE)
            rows: list[gp.Constr] = []
            point = y0.astype(float)
            th_val = -INF
            while self.left() > 0:
                v, Q, cnt = oracle.separate(point)
                total = float(Q.sum())
                if total <= th_val + 1e-9 * max(1.0, abs(total)):
                    stats["lp_converged"] = True
                    reason = "lp_converged"
                    break
                const, coef = oracle.aggregate(v, cnt)
                rows.append(md.addLConstr(gp.LinExpr(coef.tolist(), y.tolist()) + th, GRB.GREATER_EQUAL, const))
                self.cut_t.append(self.elapsed())
                self.cut_v.append(v.copy())
                self.cut_size.append(int(len(v) + cnt.sum()) if np.all(v <= oracle.sc.ds[:, -1]) else self.agg_size(oracle, v))
                if self.left() <= 0:
                    break
                t = time.perf_counter()
                md.Params.TimeLimit = max(0.01, self.left())
                md.optimize()
                stats["master_seconds"] += time.perf_counter() - t
                if md.Status != GRB.OPTIMAL:
                    reason = "time_limit" if md.Status == GRB.TIME_LIMIT else f"master_status_{md.Status}"
                    break
                stats["iterations"] += 1
                self.publish_bound(float(md.ObjVal), "lp_master")
                if self.cert_due():
                    pi = np.asarray(md.getAttr("Pi", rows))
                    self.publish_aggregated([int(c) for c in np.argsort(-pi, kind="stable") if pi[c] > 1e-12])
                point = np.clip(np.asarray(y.X), 0.0, 1.0)
                th_val = float(th.X)
            if md.Status == GRB.OPTIMAL and rows and self.left() > 0:
                pi = np.asarray(md.getAttr("Pi", rows))
                self.publish_aggregated([int(c) for c in np.argsort(-pi, kind="stable") if pi[c] > 1e-12], force=True)
            stats["final_lp_point"] = point.tolist() if stats["lp_converged"] else None
            md.dispose()
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        finally:
            env.dispose()
        return self.finish(reason, stats)


# ---------------------------------------------------------------- naive integer-master Benders (diagnostic)

class KMNaiveBenders(_Base):
    """Naive textbook outer-loop Benders with an integer master (diagnostic); see the module docstring. Integral
    costs: MIPGap 0 with absolute gap 0.999 for the master, and 'optimal' only at unit-gap closure."""

    def run(self, data_path: str) -> dict[str, Any]:
        self.start(data_path)
        inst, D = self.inst, self.D
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        stats = {"iterations": 0, "master_seconds": 0.0, "bound_kind": "solver_reported_integer_master"}
        reason = "time_limit"
        try:
            oracle = Oracle(D, inst.k)
            md = gp.Model(env=env)
            md.Params.OutputFlag = 0
            md.Params.MIPGap = 0.0
            if self.integral:
                md.Params.MIPGapAbs = 0.999
            y = md.addMVar(inst.m, vtype=GRB.BINARY, name="y")
            th = md.addVar(lb=0.0, name="theta")
            md.addConstr(y.sum() == inst.k)
            md.setObjective(th, GRB.MINIMIZE)
            ybar = greedy_design(D, inst.k)
            th_val = -INF
            while self.left() > 0:
                v, Q, cnt = oracle.separate(ybar.astype(float))
                self.offer(ybar, design_value(D, open_list(ybar)), "master_design" if stats["iterations"] else "initializer")
                if self.closed():
                    reason = "optimal"
                    break
                total = float(Q.sum())
                if total <= th_val + 1e-9 * max(1.0, abs(total)):
                    reason = "stalled_gap_not_closed"  # cut not violated but the gap is open (numerics)
                    break
                const, coef = oracle.aggregate(v, cnt)
                md.addLConstr(gp.LinExpr(coef.tolist(), y.tolist()) + th, GRB.GREATER_EQUAL, const)
                self.cut_t.append(self.elapsed())
                self.cut_v.append(v.copy())
                self.cut_size.append(int(len(v) + cnt.sum()))
                self.publish_aggregated(list(range(len(self.cut_v) - 1, -1, -1)))
                if self.left() <= 0:
                    break
                t = time.perf_counter()
                md.Params.TimeLimit = max(0.01, self.left())
                md.optimize()
                stats["master_seconds"] += time.perf_counter() - t
                stats["iterations"] += 1
                if md.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and md.SolCount and math.isfinite(md.ObjBound):
                    self.publish_bound(float(md.ObjBound), "integer_master")
                if self.closed():
                    reason = "optimal"
                    break
                if md.Status != GRB.OPTIMAL:
                    reason = "time_limit" if md.Status == GRB.TIME_LIMIT else f"master_status_{md.Status}"
                    break
                ybar = np.asarray(y.X) > 0.5
                th_val = float(th.X)
            self.publish_aggregated(list(range(len(self.cut_v) - 1, -1, -1)), force=True)
            md.dispose()
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        finally:
            env.dispose()
        return self.finish(reason, stats)
