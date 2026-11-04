"""Two-stage stochastic CFL at scale, bound side (protocol v4, Sections 3 and 6): the arms of the bound-side screen.
Phoenix only. View and certificate format: src/bd/scfl/h_view_xl.py (scfl-v1, scfl-cert-c1).

Oracle (shared by every Benders arm and the pool): b_mip_xl.RecourseLP(penalty=True) at a point y with the strong
linking bounds x_ij <= d_sj y_i (vub=True), warm-started dual simplex, scenarios in order of total demand. Its duals
(v, alpha) with the cheapest completion beta = (v - alpha - c)^+ are dual feasible for the view's recourse (the
penalty only adds v_j <= M_j), so every cut is valid everywhere:
    theta_s >= d_s.v - sum_i (u_i alpha_i + sum_j d_sj beta_ij) y_i                 (c_naive_bd.cuts_from_duals).
Shared initializer: c_naive_bd.greedy_design; every Benders master starts from the lower-bound cuts
(v_j = min_i c_ij, alpha = 0) and the initializer's cuts.

Arms (time_limit, threads, on_incumbent(open list, stage), on_bound(float), on_certificate(bytes)):
  SCFLEngLP       engineered L-shaped, LP phase: multi-cut master (one theta per scenario), in-out stabilization
                  (core = all-ones), all violated scenario cuts per round, aging purge of long non-binding cuts;
                  certificate = per scenario the master-dual combination of its cuts (one term per scenario, the
                  implied completion of the combined (v, alpha); reproduces the master value);
  SCFLNaiveLP     naive LP-master L-shaped (unstabilized Kelley, ONE aggregated cut per iteration, same oracle and
                  initializer); certificate = the master-dual combination of its aggregated cuts (one multi-term cut);
  SCFLCompactLP   extensive-form LP (barrier, crossover off, or dual simplex), 1 thread, Gurobi MemLimit;
                  certificate = per scenario the extensive form's (demand, capacity) duals / p_s (Pi or BarPi);
  SCFLJensen      decomposition-free aggregation control: the extensive-form LP of G cluster-mean scenarios
                  (k-means, f_heuristic.cluster_scenarios; G = 10, 20, 40, ...), each solve's cluster duals published
                  as one cut per scenario of the cluster (valid: the dual feasible set does not depend on the demands);
  SCFLStatic      'empty', 'oneshot_greedy' (exact duals at the initializer's design), 'oneshot_saa' (at the design of
                  an SAA extensive MIP on 10 sampled scenarios);
  SCFLPool        no-feedback pool: oracle cuts at the initializer, all-ones, random designs and random fractional
                  points (seed 20261103) until the size limit;
  SCFLLagrangian  subgradient Lagrangian relaxing the demand rows (diagnostic): per (s, i) a continuous knapsack;
                  snapshots published as one term (s, u_s, alpha_s) per scenario (the checker master then equals L(u)).
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from src.bd.scfl.a_data import SCFLInstance, load_instance
from src.bd.scfl.b_mip_xl import RecourseLP, build_extensive_form, is_memory_error, make_env, open_list
from src.bd.scfl.c_naive_bd import cuts_from_duals, greedy_design
from src.bd.scfl.h_view_xl import cert_bytes, term_sizes

INF = float("inf")
CERT_LIMIT = 2_000_000


# ---------------------------------------------------------------- shared

def lower_duals(inst: SCFLInstance) -> tuple[np.ndarray, np.ndarray]:
    """(v, alpha) of the lower-bound cut of every scenario: v_j = min_i c_ij, alpha = 0."""
    return np.tile(inst.cost.min(axis=0), (inst.S, 1)), np.zeros((inst.S, inst.I))


class _Base:
    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, on_certificate: Callable | None = None, soft_mem_gb: float | None = None,
                 seed: int = 0, verbose: bool = False, cert_interval: float = 10.0, cert_limit: int = CERT_LIMIT,
                 mem_limit_gb: float | None = None) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed, self.verbose = time_limit, threads, soft_mem_gb, seed, verbose
        self.mem_limit_gb = mem_limit_gb
        self.on_incumbent = on_incumbent or (lambda design, stage=None: None)
        self.on_bound = on_bound or (lambda bound: None)
        self.on_certificate = on_certificate or (lambda cert: None)
        self.cert_interval, self.cert_limit = cert_interval, cert_limit
        self._last_cert, self._cert_seq = -INF, 0
        self.trace: list[dict[str, Any]] = []

    def start(self, data_path: str) -> None:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        self.inst = load_instance(data_path)
        self.ub, self.lb, self.best = INF, -INF, None

    def left(self) -> float:
        return self.deadline - time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def offer(self, y: np.ndarray, value: float | None, stage: str) -> bool:
        if value is not None and value < self.ub - 1e-9 * max(1.0, abs(value)) and self.left() > 0:
            self.ub, self.best = value, np.asarray(y, dtype=bool).copy()
            self.on_incumbent(open_list(self.best), stage=stage)
            return True
        return False

    def publish_bound(self, value: float, stage: str) -> None:
        value = value - 1e-9 * max(1.0, abs(value))
        if value > self.lb and self.left() > 0:
            self.lb = value
            self.on_bound(value)
            self.trace.append({"t": round(self.elapsed(), 3), "bound": value, "stage": stage})

    def emit(self, data_fn: Callable[[int], bytes], force: bool = False) -> bool:
        now = time.perf_counter()
        if now >= self.deadline or (not force and now - self._last_cert < self.cert_interval):
            return False
        self._cert_seq += 1
        self.on_certificate(data_fn(self._cert_seq))
        self._last_cert = time.perf_counter()
        return True

    def single_cuts(self, scen: np.ndarray, V: np.ndarray, A: np.ndarray, priority: np.ndarray | None = None):
        """One single-term cut per given scenario, in priority order, within the size limit."""
        order = np.arange(len(scen)) if priority is None else np.asarray(priority)
        sizes = term_sizes(self.inst.cost, V[order], A[order])
        keep = order[np.nonzero(np.cumsum(sizes) <= self.cert_limit)[0]]
        return [[(int(scen[k]), V[k], A[k])] for k in keep]

    def separate_all(self, sep: RecourseLP, point: np.ndarray, deadline: float):
        """Penalized strong-linking oracle at `point` for every scenario: (V, A, Q) or None at the deadline."""
        inst = self.inst
        V = np.zeros((inst.S, inst.J))
        A = np.zeros((inst.S, inst.I))
        Q = np.zeros(inst.S)
        for s in sep.order:
            left = min(deadline, self.deadline) - time.perf_counter()
            if left <= 0:
                return None
            res = sep.solve(int(s), point, left, vub=True)
            if res["status"] != "optimal":
                return None
            V[s], A[s], Q[s] = res["v"], res["alpha"], res["obj"]
        return V, A, Q

    def design_duals(self, ev: RecourseLP, y: np.ndarray):
        """Exact recourse of a binary design with duals: (value, V, A) or None (time or infeasible design)."""
        res = ev.evaluate(y, self.deadline, duals=True)
        if res["status"] != "optimal":
            return None
        return res["value"], res["V"], res["A"]


# ---------------------------------------------------------------- engineered LP phase

class _CutStore:
    """Master cut rows with their (scenario, v, alpha), growable numpy storage."""

    def __init__(self, I: int, J: int) -> None:
        self.I, self.J = I, J
        self.cap = 0
        self.n = 0
        self.s = np.zeros(0, dtype=np.int64)
        self.V = np.zeros((0, J))
        self.A = np.zeros((0, I))
        self.age = np.zeros(0, dtype=np.int64)
        self.alive = np.zeros(0, dtype=bool)
        self.lower = np.zeros(0, dtype=bool)
        self.rows: list = []

    def add(self, s, V, A, rows, lower: bool = False) -> None:
        k = len(s)
        while self.n + k > self.cap:
            new = max(1024, 2 * self.cap)
            for name, shape in (("s", ()), ("V", (self.J,)), ("A", (self.I,)), ("age", ()), ("alive", ()), ("lower", ())):
                old = getattr(self, name)
                arr = np.zeros((new,) + shape, dtype=old.dtype)
                arr[:self.n] = old[:self.n]
                setattr(self, name, arr)
            self.cap = new
        sl = slice(self.n, self.n + k)
        self.s[sl], self.V[sl], self.A[sl] = s, V, A
        self.age[sl] = 0
        self.alive[sl] = True
        self.lower[sl] = lower
        self.rows.extend(rows)
        self.n += k


class SCFLEngLP(_Base):
    """Engineered multi-cut L-shaped LP phase (module docstring)."""

    def __init__(self, *, inout_lambda: float = 0.3, stabilize: bool = True, conv_tol: float = 1e-8, purge_age: int = 10,
                 max_rounds: int = 100000, jensen_seed: int = 0, **kw) -> None:
        super().__init__(**kw)
        self.inout_lambda, self.stabilize, self.conv_tol = inout_lambda, stabilize, conv_tol
        self.purge_age, self.max_rounds = purge_age, max_rounds
        # jensen_seed = G > 0: solve the extensive-form LP of G cluster-mean scenarios first (barrier); its cluster
        # duals seed one cut per scenario and its y is the stabilization core.
        self.jensen_seed = jensen_seed

    def add_cuts(self, scen: np.ndarray, V: np.ndarray, A: np.ndarray, lower: bool = False) -> None:
        if not len(scen):
            return
        inst = self.inst
        const, coef = cuts_from_duals(inst, scen, V, A)
        C = sp.csr_matrix(np.where(np.abs(coef) > 1e-12, coef, 0.0))
        E = sp.csr_matrix((np.ones(len(scen)), (np.arange(len(scen)), scen)), shape=(len(scen), inst.S))
        rows = self.md.addConstr(C @ self.y + E @ self.th >= const)
        self.md.update()
        self.store.add(scen, V, A, rows.tolist(), lower)
        self.stats["cuts"] += len(scen)

    def publish(self, force: bool = False) -> None:
        if not (force or time.perf_counter() - self._last_cert >= self.cert_interval):
            return
        st = self.store
        idx = np.nonzero(st.alive[:st.n])[0]
        try:
            pi = np.maximum(np.asarray(self.md.getAttr("Pi", [st.rows[k] for k in idx])), 0.0)
        except gp.GurobiError:
            return
        pos = pi > 1e-12
        idx, pi = idx[pos], pi[pos]
        S = self.inst.S
        w = np.bincount(st.s[idx], weights=pi, minlength=S)
        Vb = np.zeros((S, self.inst.J))
        Ab = np.zeros((S, self.inst.I))
        np.add.at(Vb, st.s[idx], pi[:, None] * st.V[idx])
        np.add.at(Ab, st.s[idx], pi[:, None] * st.A[idx])
        has = np.nonzero(w > 0)[0]
        Vb, Ab = Vb[has] / w[has, None], Ab[has] / w[has, None]
        cuts = self.single_cuts(has, Vb, Ab, priority=np.argsort(-w[has], kind="stable"))
        self.emit(lambda k: cert_bytes(cuts, version=k), force=True)
        self.stats["certificate_versions"] = self._cert_seq

    def purge(self) -> None:
        st = self.store
        idx = np.nonzero(st.alive[:st.n])[0]
        if not len(idx):
            return
        rows = [st.rows[k] for k in idx]
        slack = np.asarray(self.md.getAttr("Slack", rows))
        pi = np.asarray(self.md.getAttr("Pi", rows))
        loose = (slack < -1e-7) & (pi <= 1e-12)
        st.age[idx] = np.where(loose, st.age[idx] + 1, 0)
        old = idx[(st.age[idx] >= self.purge_age) & ~st.lower[idx]]
        if len(old):
            self.md.remove([st.rows[k] for k in old])
            st.alive[old] = False
            for k in old:
                st.rows[k] = None
            self.md.update()
            self.stats["purged"] += len(old)

    def run(self, data_path: str) -> dict[str, Any]:
        self.start(data_path)
        inst = self.inst
        self.stats: dict[str, Any] = {"cuts": 0, "purged": 0, "rounds": 0, "converged": False, "master_seconds": 0.0,
                                      "sub_seconds": 0.0, "lp_value": None}
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        self.md, self.lp_point, reason = None, None, "time_limit"
        sep = ev = None
        try:
            sep = RecourseLP(inst, env, penalty=True)
            ev = RecourseLP(inst, env)
            y0 = greedy_design(inst)
            t = time.perf_counter()
            r0 = self.design_duals(ev, y0)
            self.stats["sub_seconds"] += time.perf_counter() - t
            if r0 is None:
                raise TimeoutError
            self.offer(y0, r0[0], "initializer")
            md = gp.Model(env=env)
            self.md = md
            md.Params.OutputFlag = 0
            md.Params.Method = 1
            self.y = md.addMVar(inst.I, lb=0.0, ub=1.0, name="y")
            self.th = md.addMVar(inst.S, lb=0.0, name="theta")
            md.addConstr(inst.capacity @ self.y >= inst.max_demand, name="cover")
            md.setObjective(inst.fixed @ self.y + inst.prob @ self.th, GRB.MINIMIZE)
            self.store = _CutStore(inst.I, inst.J)
            all_s = np.arange(inst.S)
            Vl, Al = lower_duals(inst)
            self.add_cuts(all_s, Vl, Al, lower=True)     # lower-bound cuts: never purged
            self.add_cuts(all_s, r0[1], r0[2])
            core = np.ones(inst.I)
            if self.jensen_seed > 0:
                from src.bd.scfl.f_heuristic import cluster_scenarios
                t = time.perf_counter()
                labels, red = cluster_scenarios(inst, min(self.jensen_seed, inst.S))
                rm, ri = build_ef_lp(inst.I, inst.J, inst.fixed, inst.capacity, inst.cost, red.demand, red.prob, env)
                try:
                    rm.Params.Method = 2
                    rm.Params.Crossover = 0
                    rm.Params.TimeLimit = max(0.01, 0.25 * self.left())
                    rm.optimize()
                    if rm.Status == GRB.OPTIMAL:
                        Vg, Ag = ef_duals(ri, "Pi", red.S, inst.I, inst.J, red.prob)
                        core = np.clip(np.asarray(ri["y"].X, dtype=float), 0.0, 1.0)
                        self.stats["jensen_value"] = float(rm.ObjVal)
                finally:
                    rm.dispose()
                self.stats["jensen_seconds"] = time.perf_counter() - t
                if "jensen_value" in self.stats:
                    self.add_cuts(all_s, Vg[labels], Ag[labels])
            reason = self.loop(sep, core)
        except TimeoutError:
            reason = "time_limit"
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        finally:
            for obj in (sep, ev):
                if obj is not None:
                    obj.dispose()
            if self.md is not None:
                self.md.dispose()
            env.dispose()
        return {"best_objective": self.ub if self.ub < INF else None, "global_lb": self.lb if self.lb > -INF else None,
                "termination_reason": reason, "design": open_list(self.best) if self.best is not None else None,
                "lp_point": self.lp_point, "stats": self.stats, "total_seconds": self.elapsed(), "trace": self.trace[-200:]}

    def loop(self, sep: RecourseLP, core: np.ndarray) -> str:
        md, inst = self.md, self.inst
        lam = self.inout_lambda if self.stabilize else 1.0
        best, stall = -INF, 0
        reason = "time_limit"
        while self.left() > 0 and self.stats["rounds"] < self.max_rounds:
            t = time.perf_counter()
            md.Params.TimeLimit = max(0.01, self.left())
            md.optimize()
            self.stats["master_seconds"] += time.perf_counter() - t
            if md.Status != GRB.OPTIMAL:
                reason = "time_limit" if md.Status == GRB.TIME_LIMIT else f"master_status_{md.Status}"
                break
            self.stats["rounds"] += 1
            lb_mp = float(md.ObjVal)
            self.publish_bound(lb_mp, "lp_master")
            self.publish()
            stall = stall + 1 if lb_mp <= best + 1e-9 * max(1.0, abs(lb_mp)) else 0
            best = max(best, lb_mp)
            y_lp = np.clip(np.asarray(self.y.X), 0.0, 1.0)
            th = np.asarray(self.th.X)
            if self.purge_age > 0:
                self.purge()          # invalidates the master solution: read it first
            self.stats["lp_value"], self.lp_point = lb_mp, y_lp.tolist()
            point = y_lp if lam >= 1.0 else lam * y_lp + (1.0 - lam) * core
            t = time.perf_counter()
            res = self.separate_all(sep, point, self.deadline)
            self.stats["sub_seconds"] += time.perf_counter() - t
            if res is None:
                break
            V, A, Q = res
            const, coef = cuts_from_duals(inst, np.arange(inst.S), V, A)
            if lam >= 1.0:
                ub_mp = float(inst.fixed @ y_lp + inst.prob @ Q)
                self.stats["last_ub_mp"] = ub_mp
                if ub_mp - lb_mp <= self.conv_tol * max(1.0, abs(lb_mp)):
                    self.stats["converged"] = True
                    reason = "lp_converged"
                    break
            viol = const - coef @ y_lp - th
            add = np.nonzero(viol > 1e-9 * np.maximum(1.0, np.abs(const)))[0]
            if len(add) == 0 or (stall >= 3 and lam < 1.0):
                if lam < 1.0:
                    lam, stall = 1.0, 0
                    continue
                self.stats["converged"] = True
                reason = "lp_converged_no_violated_cut"
                break
            self.add_cuts(add, V[add], A[add])
            core = 0.5 * core + 0.5 * point
        if self.left() > 0:
            md.Params.TimeLimit = max(0.01, self.left())
            md.optimize()
            if md.Status == GRB.OPTIMAL:
                self.publish_bound(float(md.ObjVal), "lp_master")
                self.publish(force=True)
        return reason


# ---------------------------------------------------------------- naive LP-master L-shaped

class SCFLNaiveLP(_Base):
    """Naive LP-master L-shaped (module docstring)."""

    def run(self, data_path: str) -> dict[str, Any]:
        self.start(data_path)
        inst = self.inst
        stats: dict[str, Any] = {"iterations": 0, "master_seconds": 0.0, "sub_seconds": 0.0, "converged": False}
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        reason, point = "time_limit", None
        sep = ev = None
        try:
            sep = RecourseLP(inst, env, penalty=True)
            ev = RecourseLP(inst, env)
            y0 = greedy_design(inst)
            r0 = self.design_duals(ev, y0)
            if r0 is None:
                raise TimeoutError
            self.offer(y0, r0[0], "initializer")
            md = gp.Model(env=env)
            md.Params.OutputFlag = 0
            md.Params.Method = 1
            y = md.addMVar(inst.I, lb=0.0, ub=1.0, name="y")
            th = md.addVar(lb=0.0, name="theta")
            md.addConstr(inst.capacity @ y >= inst.max_demand)
            md.setObjective(inst.fixed @ y + th, GRB.MINIMIZE)
            rows: list = []
            store: list[tuple[np.ndarray, np.ndarray]] = []
            all_s = np.arange(inst.S)

            def add(V, A):
                const, coef = cuts_from_duals(inst, all_s, V, A)
                c = float(inst.prob @ const)
                a = inst.prob @ coef
                rows.append(md.addLConstr(gp.LinExpr(a.tolist(), y.tolist()) + th, GRB.GREATER_EQUAL, c))
                store.append((V.copy(), A.copy()))

            add(*lower_duals(inst))
            add(r0[1], r0[2])
            point = y0.astype(float)
            while self.left() > 0:
                t = time.perf_counter()
                md.Params.TimeLimit = max(0.01, self.left())
                md.optimize()
                stats["master_seconds"] += time.perf_counter() - t
                if md.Status != GRB.OPTIMAL:
                    reason = "time_limit" if md.Status == GRB.TIME_LIMIT else f"master_status_{md.Status}"
                    break
                stats["iterations"] += 1
                self.publish_bound(float(md.ObjVal), "lp_master")
                self.publish(np.asarray(md.getAttr("Pi", rows)), store)
                point = np.clip(np.asarray(y.X), 0.0, 1.0)
                th_val = float(th.X)
                t = time.perf_counter()
                res = self.separate_all(sep, point, self.deadline)
                stats["sub_seconds"] += time.perf_counter() - t
                if res is None:
                    break
                V, A, Q = res
                if float(inst.prob @ Q) <= th_val + 1e-9 * max(1.0, abs(th_val)):
                    stats["converged"] = True
                    reason = "lp_converged"
                    break
                add(V, A)
            if md.Status == GRB.OPTIMAL and self.left() > 0:
                self.publish(np.asarray(md.getAttr("Pi", rows)), store, force=True)
            stats["cuts"] = len(rows)
            md.dispose()
        except TimeoutError:
            reason = "time_limit"
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        finally:
            for obj in (sep, ev):
                if obj is not None:
                    obj.dispose()
            env.dispose()
        return {"best_objective": self.ub if self.ub < INF else None, "global_lb": self.lb if self.lb > -INF else None,
                "termination_reason": reason, "design": open_list(self.best) if self.best is not None else None,
                "lp_point": point.tolist() if stats["converged"] and point is not None else None, "stats": stats,
                "total_seconds": self.elapsed(), "trace": self.trace[-200:]}

    def publish(self, lam: np.ndarray, store: list, force: bool = False) -> None:
        """One multi-term cut: the master-dual combination of the aggregated cuts (per scenario (v, alpha) averaged
        with weights lam; implied completion), terms dropped (largest first) only if it exceeds the size limit."""
        if not (force or time.perf_counter() - self._last_cert >= self.cert_interval):
            return
        lam = np.maximum(lam, 0.0)
        pos = np.nonzero(lam > 1e-12)[0]
        if not len(pos):
            return
        w = lam[pos] / lam[pos].sum()
        Vb = sum(wk * store[k][0] for wk, k in zip(w, pos))
        Ab = sum(wk * store[k][1] for wk, k in zip(w, pos))
        sizes = term_sizes(self.inst.cost, Vb, Ab)
        order = np.argsort(sizes, kind="stable")
        keep = np.sort(order[np.nonzero(np.cumsum(sizes[order]) <= self.cert_limit)[0]])
        cut = [(int(s), Vb[s], Ab[s]) for s in keep]
        self.emit(lambda k: cert_bytes([cut], version=k), force=True)


# ---------------------------------------------------------------- extensive form (compact LP) and Jensen control

def build_ef_lp(I: int, J: int, fixed: np.ndarray, capacity: np.ndarray, cost: np.ndarray, demand: np.ndarray,
                prob: np.ndarray, env: gp.Env) -> tuple[gp.Model, dict]:
    """Strong-linking extensive-form LP (y continuous) with handles to the demand and capacity rows."""
    S = demand.shape[0]
    N = S * I * J
    md = gp.Model(env=env)
    y = md.addMVar(I, lb=0.0, ub=1.0, obj=fixed, name="y")
    ub = np.broadcast_to(demand[:, None, :], (S, I, J)).ravel()
    x = md.addMVar(N, lb=0.0, ub=ub, obj=(prob[:, None, None] * cost[None, :, :]).ravel())
    md.ModelSense = GRB.MINIMIZE
    k = np.arange(N)
    s_of, i_of, j_of = k // (I * J), (k // J) % I, k % J
    ones = np.ones(N)
    A_d = sp.csr_matrix((ones, (s_of * J + j_of, k)), shape=(S * J, N))
    dem = md.addConstr(A_d @ x == demand.ravel())
    del A_d
    A_c = sp.csr_matrix((ones, (s_of * I + i_of, k)), shape=(S * I, N))
    B_c = sp.csr_matrix((np.tile(capacity, S), (np.arange(S * I), np.tile(np.arange(I), S))), shape=(S * I, I))
    cap = md.addConstr(A_c @ x - B_c @ y <= 0.0)
    del A_c, B_c
    keep = np.nonzero(ub > 0)[0]
    rows = np.arange(len(keep))
    E = sp.csr_matrix((np.ones(len(keep)), (rows, keep)), shape=(len(keep), N))
    L = sp.csr_matrix((ub[keep], (rows, i_of[keep])), shape=(len(keep), I))
    md.addConstr(E @ x - L @ y <= 0.0)
    del E, L, keep, rows, k, s_of, i_of, j_of, ones
    cover = md.addConstr(capacity @ y >= float(demand.sum(axis=1).max()), name="cover")
    md.update()
    return md, {"y": y, "dem": dem, "cap": cap, "cover": cover, "num_vars": int(md.NumVars), "num_constrs": int(md.NumConstrs)}


def ef_duals(info: dict, attr: str, S: int, I: int, J: int, prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-scenario (v, alpha) from extensive-form row duals (scaled back by the probabilities)."""
    pd = np.asarray(getattr(info["dem"], attr), dtype=float).reshape(S, J)
    pc = np.asarray(getattr(info["cap"], attr), dtype=float).reshape(S, I)
    V = pd / prob[:, None]
    A = np.maximum(-pc / prob[:, None], 0.0)
    V = np.where(np.isfinite(V), V, 0.0)
    A = np.where(np.isfinite(A), A, 0.0)
    return V, A


class SCFLCompactLP(_Base):
    """Decomposition-free bound control: the strong-linking extensive-form LP (module docstring)."""

    def __init__(self, *, method: int = 2, crossover: int = 0, publish_margin: float = 30.0, **kw) -> None:
        super().__init__(**kw)
        self.method, self.crossover, self.publish_margin = method, crossover, publish_margin

    def run(self, data_path: str) -> dict[str, Any]:
        self.start(data_path)
        inst = self.inst
        env = make_env(self.threads, self.soft_mem_gb, self.seed, self.mem_limit_gb)
        out: dict[str, Any] = {"method": self.method, "crossover": self.crossover, "termination_reason": "deadline_during_build"}
        model = None
        try:
            model, info = build_ef_lp(inst.I, inst.J, inst.fixed, inst.capacity, inst.cost, inst.demand, inst.prob, env)
            out.update(build_seconds=self.elapsed(), num_vars=info["num_vars"], num_constrs=info["num_constrs"])
            margin = min(self.publish_margin if self.method == 2 else 5.0, 0.1 * self.time_limit)
            if self.left() - margin > 0.01:
                model.Params.TimeLimit = self.left() - margin
                model.Params.Method = self.method
                model.Params.Crossover = self.crossover
                spx: list[tuple[float, float]] = []

                def callback(cb, where):
                    if where == GRB.Callback.SIMPLEX and self.method == 1 and self.left() > 0:
                        if cb.cbGet(GRB.Callback.SPX_ISPERT) == 0 and cb.cbGet(GRB.Callback.SPX_DUALINF) <= 1e-9:
                            val = float(cb.cbGet(GRB.Callback.SPX_OBJVAL))
                            if math.isfinite(val) and (not spx or val > spx[-1][1]):
                                spx.append((round(self.elapsed(), 3), val))
                                self.on_bound(val)

                model.optimize(callback)
                st = int(model.Status)
                out["gurobi_status"] = st
                out["simplex_bounds"] = spx[-100:]
                VA = yv = None
                if st in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                    try:
                        VA = ef_duals(info, "Pi", inst.S, inst.I, inst.J, inst.prob)
                        yv = np.asarray(info["y"].X, dtype=float)
                        out["certificate_source"] = "Pi"
                    except gp.GurobiError:
                        VA = None
                if VA is None and self.method == 2:
                    try:
                        VA = ef_duals(info, "BarPi", inst.S, inst.I, inst.J, inst.prob)
                        yv = np.asarray(info["y"].BarX, dtype=float)
                        out["certificate_source"] = "BarPi"
                    except (gp.GurobiError, AttributeError):
                        VA = None
                if VA is not None and self.left() > 0:
                    cuts = self.single_cuts(np.arange(inst.S), VA[0], VA[1])
                    self.emit(lambda k: cert_bytes(cuts, version=k), force=True)
                    out["certificate_seconds"] = self.elapsed()
                    out["certificate_terms"] = len(cuts)
                    out["lp_point"] = yv.tolist() if yv is not None else None
                if st == GRB.OPTIMAL and self.left() > 0:
                    out["termination_reason"] = "optimal"
                    out["lp_value"] = float(model.ObjVal)
                    self.publish_bound(float(model.ObjVal), "compact_lp")
                else:
                    out["termination_reason"] = {GRB.TIME_LIMIT: "time_limit", GRB.INTERRUPTED: "time_limit",
                                                 GRB.MEM_LIMIT: "memory_limit"}.get(st, f"status_{st}")
                out["barrier_iterations"] = int(model.BarIterCount) if self.method == 2 else None
        except (gp.GurobiError, MemoryError) as err:
            if isinstance(err, gp.GurobiError) and not is_memory_error(err):
                raise
            out["termination_reason"] = "memory_limit" + ("" if model is not None else "_during_build")
            out["error"] = str(err)[:300]
        finally:
            if model is not None:
                model.dispose()
            env.dispose()
        out["total_seconds"] = self.elapsed()
        return out


class SCFLJensen(_Base):
    """Decomposition-free aggregation control (module docstring): G = g0, 2 g0, 4 g0, ... cluster-mean scenarios
    while G <= S / 2 and time remains; each solved reduced LP publishes its cluster duals, one cut per scenario."""

    def __init__(self, *, g0: int = 10, method: int = 2, **kw) -> None:
        super().__init__(**kw)
        self.g0, self.method = g0, method

    def run(self, data_path: str) -> dict[str, Any]:
        from src.bd.scfl.f_heuristic import cluster_scenarios
        self.start(data_path)
        inst = self.inst
        env = make_env(self.threads, self.soft_mem_gb, self.seed, self.mem_limit_gb)
        log = []
        G = self.g0
        reason = "time_limit"
        try:
            while G <= max(self.g0, inst.S // 2) and self.left() > 1.0:
                t = time.perf_counter()
                labels, red = cluster_scenarios(inst, G)
                model, info = build_ef_lp(inst.I, inst.J, inst.fixed, inst.capacity, inst.cost, red.demand, red.prob, env)
                try:
                    model.Params.TimeLimit = max(0.01, self.left() - 2.0)
                    model.Params.Method = self.method
                    model.Params.Crossover = 0
                    model.optimize()
                    ok = model.Status == GRB.OPTIMAL
                    entry = {"G": int(red.S), "status": int(model.Status), "seconds": round(time.perf_counter() - t, 2)}
                    if ok:
                        entry["jensen_value"] = float(model.ObjVal)
                        Vg, Ag = ef_duals(info, "Pi", red.S, inst.I, inst.J, red.prob)
                finally:
                    model.dispose()
                log.append(entry)
                if not ok:
                    break
                if self.left() > 0:
                    V, A = Vg[labels], Ag[labels]
                    cuts = self.single_cuts(np.arange(inst.S), V, A)
                    self.emit(lambda k: cert_bytes(cuts, version=k), force=True)
                    entry["terms"] = len(cuts)
                    entry["published_at"] = round(self.elapsed(), 2)
                if red.S >= inst.S // 2:
                    reason = "largest_G_done"
                    break
                G *= 2
        except (gp.GurobiError, MemoryError) as err:
            if isinstance(err, gp.GurobiError) and not is_memory_error(err):
                raise
            reason = "memory_limit"
        finally:
            env.dispose()
        return {"termination_reason": reason, "solves": log, "total_seconds": self.elapsed()}


# ---------------------------------------------------------------- trivial-certificate controls

class SCFLStatic(_Base):
    """'empty', 'oneshot_greedy', 'oneshot_saa' (module docstring)."""

    def __init__(self, *, kind: str = "empty", saa_scenarios: int = 10, **kw) -> None:
        super().__init__(**kw)
        self.kind, self.saa_scenarios = kind, saa_scenarios

    def run(self, data_path: str) -> dict[str, Any]:
        self.start(data_path)
        inst = self.inst
        info: dict[str, Any] = {"kind": self.kind}
        if self.kind == "empty":
            self.emit(lambda k: cert_bytes([], version=k), force=True)
            return {"termination_reason": "published", **info, "total_seconds": self.elapsed()}
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        ev = None
        try:
            ev = RecourseLP(inst, env)
            if self.kind == "oneshot_greedy":
                y = greedy_design(inst)
            elif self.kind == "oneshot_saa":
                from src.bd.scfl.f_heuristic import repair
                rng = np.random.default_rng(self.seed)
                idx = np.sort(rng.choice(inst.S, size=min(self.saa_scenarios, inst.S), replace=False))
                model, mi = build_extensive_form(inst, env, strong=True, scenarios=idx)
                try:
                    model.Params.TimeLimit = max(0.01, 0.5 * self.left())
                    model.Params.MIPGap = 1e-4
                    model.optimize()
                    y = np.asarray(mi["y"].X) > 0.5 if model.SolCount else greedy_design(inst)
                finally:
                    model.dispose()
                dbar = inst.mean_demand
                rank = np.argsort(inst.fixed / inst.capacity + inst.cost @ dbar / max(float(dbar.sum()), 1e-12), kind="stable")
                y = repair(inst, y, rank)
                info["saa_seconds"] = round(self.elapsed(), 2)
            else:
                raise ValueError(self.kind)
            r = self.design_duals(ev, y)
            if r is not None:
                self.offer(y, r[0], self.kind)
                cuts = self.single_cuts(np.arange(inst.S), r[1], r[2])
                self.emit(lambda k: cert_bytes(cuts, version=k), force=True)
                info["terms"] = len(cuts)
        finally:
            if ev is not None:
                ev.dispose()
            env.dispose()
        return {"termination_reason": "published", **info, "best_objective": self.ub if self.ub < INF else None,
                "design": open_list(self.best) if self.best is not None else None, "total_seconds": self.elapsed()}


class SCFLPool(_Base):
    """No-feedback cut pool (frozen recipe, seed 20261103; module docstring)."""

    def __init__(self, *, pool_seed: int = 20261103, **kw) -> None:
        super().__init__(**kw)
        self.pool_seed = pool_seed

    def run(self, data_path: str) -> dict[str, Any]:
        self.start(data_path)
        inst = self.inst
        I = inst.I
        rng = np.random.default_rng(self.pool_seed)
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        sep = None
        terms: list = []
        size, points, full = 0, 0, False
        try:
            sep = RecourseLP(inst, env, penalty=True)
            y0 = greedy_design(inst)
            k = int(y0.sum())
            t = 0
            while not full and self.left() > 0:
                if t == 0:
                    point = y0.astype(float)
                elif t == 1:
                    point = np.ones(I)
                elif t % 2 == 0:
                    point = np.zeros(I)
                    point[rng.choice(I, k, replace=False)] = 1.0
                    for i in rng.permutation(I):
                        if inst.capacity @ point >= inst.max_demand:
                            break
                        point[i] = 1.0
                else:
                    w = rng.dirichlet(np.ones(I))
                    point = np.minimum(1.0, k * w / w.sum())
                    if inst.capacity @ point < inst.max_demand:
                        point = np.minimum(1.0, point * (inst.max_demand / max(inst.capacity @ point, 1e-9)) * 1.05)
                t += 1
                res = self.separate_all(sep, point, self.deadline)
                if res is None:
                    break
                V, A, _ = res
                points += 1
                sz = term_sizes(inst.cost, V, A)
                for s in range(inst.S):
                    if size + sz[s] > self.cert_limit:
                        full = True
                        break
                    terms.append((s, V[s], A[s]))
                    size += int(sz[s])
                cuts = [[tm] for tm in terms]
                self.emit(lambda kk: cert_bytes(cuts, version=kk), force=True)
        finally:
            if sep is not None:
                sep.dispose()
            env.dispose()
        return {"termination_reason": "pool_full" if full else "time_limit", "points": points, "terms": len(terms),
                "size": size, "total_seconds": self.elapsed()}


class SCFLLagrangian(_Base):
    """Subgradient Lagrangian relaxing the demand rows (diagnostic; itself a decomposition). For multipliers
    u (S x J), per (s, i) the continuous knapsack K_si = min {sum_j (c_ij - u_sj) x_j : sum_j x_j <= u_i,
    0 <= x_j <= d_sj}, g_i = f_i + sum_s p_s K_si and
        L(u) = sum_s p_s d_s.u_s + min {g.y : u.y >= D_max, y in [0, 1]^I}     (fractional covering knapsack).
    Polyak steps against the best known design value; start: the exact duals at the initializer's design."""

    def __init__(self, *, snapshot_ratio: float = 2 ** 0.5, **kw) -> None:
        super().__init__(**kw)
        self.snapshot_ratio = snapshot_ratio

    def evaluate(self, u: np.ndarray):
        inst = self.inst
        S, I, J = inst.S, inst.I, inst.J
        L = float(inst.prob @ (inst.demand * u).sum(axis=1))
        g = inst.fixed.astype(float).copy()
        X = np.zeros((S, I, J))
        alpha = np.zeros((S, I))
        for a in range(0, S, 256):
            b = min(S, a + 256)
            R = inst.cost[None, :, :] - u[a:b, None, :]                   # (k, I, J)
            order = np.argsort(R, axis=2, kind="stable")
            Rs = np.take_along_axis(R, order, axis=2)
            ds = np.take_along_axis(np.broadcast_to(inst.demand[a:b, None, :], R.shape), order, axis=2)
            dn = np.where(Rs < 0, ds, 0.0)
            before = np.cumsum(dn, axis=2) - dn
            cap = inst.capacity[None, :, None]
            take = np.clip(cap - before, 0.0, dn)
            K = (Rs * take).sum(axis=2)                                    # (k, I)
            g += inst.prob[a:b] @ K
            full = (before + dn > cap) & (dn > 0)                          # items where the capacity binds
            crit = np.where(full.any(axis=2), np.argmax(full, axis=2), -1)
            rc = np.take_along_axis(Rs, np.maximum(crit, 0)[:, :, None], axis=2)[:, :, 0]
            alpha[a:b] = np.where(crit >= 0, np.maximum(-rc, 0.0), 0.0)
            Xs = np.zeros_like(take)
            np.put_along_axis(Xs, order, take, axis=2)
            X[a:b] = Xs
        # min g.y s.t. cap.y >= Dmax, y in [0, 1]
        y = (g < 0).astype(float)
        cov = float(inst.capacity @ y)
        if cov < inst.max_demand:
            rest = np.nonzero(g >= 0)[0]
            rest = rest[np.argsort(g[rest] / inst.capacity[rest], kind="stable")]
            for i in rest:
                need = inst.max_demand - cov
                if need <= 0:
                    break
                y[i] = min(1.0, need / inst.capacity[i])
                cov += y[i] * inst.capacity[i]
        L += float(g @ y)
        return L, y, X, alpha

    def run(self, data_path: str) -> dict[str, Any]:
        self.start(data_path)
        inst = self.inst
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        ev = None
        it, snaps, reason = 0, 0, "time_limit"
        best_L = -INF
        bounds: list = []
        try:
            ev = RecourseLP(inst, env)
            y0 = greedy_design(inst)
            r0 = self.design_duals(ev, y0)
            if r0 is None:
                raise TimeoutError
            self.offer(y0, r0[0], "initializer")
            u = r0[1].copy()
            best_u, best_a = u.copy(), np.zeros((inst.S, inst.I))
            step, stall, next_snap = 2.0, 0, 1.0
            while self.left() > 0:
                L, y, X, alpha = self.evaluate(u)
                it += 1
                if L > best_L + 1e-12 * max(1.0, abs(L)):
                    best_L, best_u, best_a, stall = L, u.copy(), alpha.copy(), 0
                    self.on_bound(L - 1e-9 * max(1.0, abs(L)))
                    bounds.append((round(self.elapsed(), 3), L))
                else:
                    stall += 1
                    if stall >= 20:
                        step, stall = max(step / 2.0, 1e-8), 0
                if self.elapsed() >= next_snap:
                    bu, ba = best_u.copy(), best_a.copy()
                    cuts = self.single_cuts(np.arange(inst.S), bu, ba)
                    self.emit(lambda k: cert_bytes(cuts, version=k), force=True)
                    snaps += 1
                    while next_snap <= self.elapsed():
                        next_snap *= self.snapshot_ratio
                G = inst.prob[:, None] * (inst.demand - np.einsum("i,sij->sj", y, X))
                gg = float((G * G).sum())
                if gg == 0.0:
                    reason = "subgradient_zero"
                    break
                gap = max(self.ub - L, 1e-6 * max(1.0, abs(L)))
                u = u + (step * gap / gg) * G
            if best_L > -INF and self.left() > 0:
                cuts = self.single_cuts(np.arange(inst.S), best_u, best_a)
                self.emit(lambda k: cert_bytes(cuts, version=k), force=True)
        except TimeoutError:
            reason = "time_limit"
        finally:
            if ev is not None:
                ev.dispose()
            env.dispose()
        return {"termination_reason": reason, "iterations": it, "snapshots": snaps,
                "global_lb": best_L if best_L > -INF else None, "best_objective": self.ub if self.ub < INF else None,
                "total_seconds": self.elapsed(), "bound_trace": bounds[-100:]}


# ---------------------------------------------------------------- exact LP_hi (protocol 3.4)

def exact_master_value(path: str, inst: SCFLInstance, y_float, bits: int = 20) -> tuple[Any, dict]:
    """Exact objective of an exactly feasible extensive-form LP point near y_float: y rounded UP to the grid 2^-bits
    (keeps the cover row and only enlarges capacities), then per scenario a vertex of the strong-linking recourse LP
    (dual simplex); on this grid the scaled data are integral, so the vertex flows are multiples of 2^-bits: they are
    rounded and checked exactly in integers (demand equalities, capacities, strong linking, x >= 0). The value is
    computed with the instance's exact unit costs (decimal text). Returns (Fraction or None, info)."""
    from fractions import Fraction
    from src.bd.scfl.h_view_xl import load_exact
    obj = load_exact(path)
    if obj.get("probability") is not None:
        return None, {"reason": "non-uniform probabilities not supported"}
    K = 1 << bits
    I, J, S = inst.I, inst.J, inst.S
    y = np.clip(np.asarray(y_float, dtype=float), 0.0, 1.0)
    yk = np.minimum(np.ceil(y * K - 1e-9), K).astype(np.int64)
    cap = inst.capacity.astype(np.int64)
    dmax = int(inst.demand.sum(axis=1).max())
    if int(cap @ yk) < dmax * K:
        return None, {"reason": "cover"}
    yg = yk.astype(float) / K
    costs = [Fraction(v) if not isinstance(v, Fraction) else v for row in obj["unit_cost"] for v in row]
    den = math.lcm(*{q.denominator for q in costs})
    Cn = [q.numerator * (den // q.denominator) for q in costs]
    fixed = [Fraction(v) for v in obj["fixed_cost"]]
    env = make_env(1)
    rec = RecourseLP(inst, env)
    acc = 0
    bad = 0
    try:
        for s in range(S):
            res = rec.solve(s, yg, 3600.0, vub=True, flows=True)
            if res["status"] != "optimal":
                return None, {"reason": f"scenario {s}: {res['status']}"}
            X = np.rint(res["flows"] * K).astype(np.int64)
            d = inst.demand[s].astype(np.int64)
            if (X < 0).any() or (X.sum(axis=0) != d * K).any() or (X.sum(axis=1) > cap * yk).any() \
                    or (X > d[None, :] * yk[:, None]).any():
                bad += 1
                return None, {"reason": f"scenario {s}: rounded flows not exactly feasible"}
            acc += sum(map(int.__mul__, Cn, X.ravel().tolist()))
    finally:
        rec.dispose()
        env.dispose()
    val = sum((f * Fraction(int(k), K) for f, k in zip(fixed, yk.tolist())), Fraction(0)) + Fraction(acc, den * K * S)
    return val, {"bits": bits, "grid_shift": float(np.abs(yg - y).max())}
