"""Engineered Benders reference for TSSO at scale (scale-up study).

Structure exploited: the recourse separates by product and scenario and the first stage has no coupling rows, so
the master separates by product: min sum_i [F_i.y_i + sum_s p_s theta_is]. Every product gets its own small master
(T + S columns), and all cuts are disaggregated (one per subproblem (i, s), multi-cut).

Phases, within one wall-clock budget:
1. Shared initializer (identical to the naive controls), evaluated exactly (b_mip_xl.DesignEvaluator); its S cuts per
   product seed the masters and its value is the first incumbent. Products are handled in blocks of at most as many
   products as recourse LPs fit in `model_budget_gb` (the same budget as the naive controls; one block up to
   60 x 30 x 2000).
2. LP phase (the certified bound): per product an in-out stabilized cutting-plane loop on the LP master
   (Ben-Ameur and Neto 2007; Fischetti, Ljubic and Sinnl 2017): separation at lam y_lp + (1 - lam) core with the
   core moved halfway toward each separation point; lam -> 1 (Kelley) when no separated cut is violated at the LP
   point or the product's bound stalls for 3 rounds. Separation first tries the product's dual pool: every dual
   vector found so far is dual feasible for every scenario of the product (the scenarios differ only in the demand
   right-hand side), so the best pooled cut of each scenario at the separation point costs one matrix product
   (cut sharing across scenarios, as in stochastic decomposition / SDDP); the pool round is kept when its cuts'
   weighted violation is at least `pool_ratio` times the product's current gap, otherwise one recourse LP per
   product (b_mip_xl.ProductRecourse, warm-started dual simplex) gives the exact cuts of all S scenarios and new
   pool vectors (at most `pool_max` kept). Only cuts violated at the master's LP point are added; when a master holds more than `keep_rows` * S rows, the non-binding ones are dropped (every cut stays in
   the log). Products are processed round robin; a product stops when its master value meets the best LP value
   F_i.y + p.q_i seen at its separation points (relative tolerance `lp_tol`), i.e. at its share of LP*, and the
   phase ends after `lp_fraction` of the time limit if not all products have converged. The sum of
   the master values is published as the (solver-reported) bound; every cut is logged with its time for the
   certificate checker (g_certificate_xl).
3. Primal: per-product rounding of the LP solution (thresholds 0.5, 0.25, 0.75), then surrogate-screened product
   local search (f_heuristic_xl.ProductSearch) for a share `ls_fraction` of the remaining time.
4. Integer phase (diagnostic bound, more designs): per-product branch-and-Benders-cut on the product master with y
   binary (one Gurobi tree per product, lazy cuts from the exact recourse at every integer solution, the incumbent
   as MIP start), products ordered by their gap with equal shares of the remaining time and a second pass for the
   unfinished ones; a finished product's master is freed. Its bound (sum of the products' MIP bounds) is
   solver-reported and kept apart (`int_trace`); its lazy cuts are not logged (the certificate is the LP master).
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from src.bd.tsso.a_data_xl import initial_design, load_instance
from src.bd.tsso.b_mip_xl import MIP_GAP, DesignEvaluator, cut_coefficients, is_memory_error, make_env
from src.bd.tsso.f_heuristic_xl import ProductSearch
from src.bd.tsso.g_certificate_xl import CutLog

INF = float("inf")


class TSSOBendersXL:
    """Engineered multi-cut, product-separated, in-out stabilized Benders; see the module docstring."""

    def __init__(self, *, time_limit: float = 900.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 inout_lambda: float = 0.3, lp_tol: float = 1e-8, lp_fraction: float = 0.7, stabilize: bool = True,
                 local_search: bool = True, ls_fraction: float = 0.15, integer: bool = True, verbose: bool = False,
                 keep_rows: int = 6, pool: bool = True, pool_max: int = 2000, pool_ratio: float = 0.1,
                 model_budget_gb: float = 8.0, **_: Any) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.on_incumbent = on_incumbent or (lambda y, stage=None: None)
        self.on_bound = on_bound or (lambda bound: None)
        self.inout_lambda, self.lp_tol, self.lp_fraction, self.stabilize = inout_lambda, lp_tol, lp_fraction, stabilize
        self.local_search, self.ls_fraction, self.integer, self.verbose = local_search, ls_fraction, integer, verbose
        self.keep_rows = keep_rows
        self.pool, self.pool_max, self.pool_ratio, self.model_budget_gb = pool, pool_max, pool_ratio, model_budget_gb
        self.trace: list[dict[str, Any]] = []
        self.int_trace: list[tuple[float, float]] = []
        self.cutlog: CutLog | None = None

    # ------------------------------------------------------------ bookkeeping

    def left(self) -> float:
        return self.deadline - time.perf_counter()

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def log(self, stage: str, **kw: Any) -> None:
        self.trace.append({"t": round(self.now(), 3), "incumbent": float(self.v.sum()) if np.all(np.isfinite(self.v)) else None,
                           "bound": self.lb if self.lb > -INF else None, "stage": stage, **kw})
        if self.verbose:
            print(self.trace[-1], flush=True)

    def publish_bound(self, value: float, stage: str) -> None:
        if value > self.lb and self.left() > 0:
            self.lb = value
            self.on_bound(value)
            self.log(stage)

    def offer(self, i: int, yi: np.ndarray, vi: float, stage: str) -> bool:
        if vi < self.v[i] - 1e-9 * max(1.0, abs(vi)) and self.left() > 0:
            self.y[i], self.v[i] = np.asarray(yi, dtype=bool), vi
            if np.all(np.isfinite(self.v)):
                self.on_incumbent(self.y.copy(), stage=stage)
                self.log(stage)
            return True
        return False

    # ------------------------------------------------------------ run

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        inst = self.inst = load_instance(data_path)
        I, T = inst.I, inst.T
        self.cutlog = CutLog(inst)
        self.y = np.zeros((I, T), dtype=bool)
        self.v = np.full(I, INF)
        self.lb = -INF
        self.stats: dict[str, Any] = {"lp_rounds": 0, "cuts": 0, "lazy_cuts": 0, "sep_seconds": 0.0, "master_seconds": 0.0,
                                      "lp_converged_products": 0, "lp_seconds": None, "lp_bound": None, "lp_ub": None,
                                      "int_bound": None, "bc_optimal_products": 0, "ls_seconds": 0.0}
        self.env = make_env(self.threads, self.soft_mem_gb, self.seed)
        self.ev = DesignEvaluator(inst, self.env, model_budget_gb=self.model_budget_gb)
        self.masters: list[dict[str, Any]] = []
        reason = "time_limit"
        try:
            reason = self.search()
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        finally:
            for m in self.masters:
                if m["model"] is not None:
                    m["model"].dispose()
            self.ev.close()
            self.env.dispose()
        ok = bool(np.all(np.isfinite(self.v)))
        y_lp = np.array([m["y_lp"] for m in self.masters]) if len(self.masters) == I else None
        return {"best_objective": float(self.v.sum()) if ok else None, "global_lb": self.lb if self.lb > -INF else None,
                "termination_reason": reason, "budget_exhausted": self.now() >= self.time_limit - 1.0,
                "design": self.y.astype(int).tolist() if ok else None,
                "y_lp": y_lp.tolist() if y_lp is not None else None, **self.stats, "total_seconds": self.now(),
                "trace": self.trace, "int_trace": self.int_trace}

    def search(self) -> str:
        inst = self.inst
        I, S = inst.I, inst.S
        self.publish_bound(0.0, "trivial")
        y0 = initial_design(inst)
        # products in blocks of at most `max_models` (the recourse LPs alive at once): initialize, then LP phase
        block = max(1, min(I, self.ev.max_models))
        self.stats["blocks"] = (I + block - 1) // block
        end = self.t0 + self.lp_fraction * self.time_limit
        for a in range(0, I, block):
            prods = list(range(a, min(I, a + block)))
            for i in prods:
                res = self.ev.product(i, y0[i], self.deadline, duals=True)
                if res is None:
                    return "time_limit"
                self.offer(i, y0[i], res["value"], "initializer")
                self.build_master(i, y0[i])
                const, coef = cut_coefficients(inst, i, np.arange(S), res["pi"], res["mu"])
                self.add_cuts(i, np.arange(S), const, coef, res["pi"], res["mu"])
                self.pool_add(i, res["pi"], res["mu"])
            if self.left() <= 0:
                return "time_limit"
            self.lp_phase(prods, end)
        self.lp_summary()
        if self.left() <= 0:
            return "time_limit"
        self.primal_phase()
        if self.integer and self.left() > 0.5:
            return self.integer_phase()
        return "time_limit" if self.left() <= 0 else "finished"

    # ------------------------------------------------------------ masters and cuts

    def build_master(self, i: int, y0: np.ndarray) -> None:
        inst = self.inst
        T, S = inst.T, inst.S
        md = gp.Model(env=self.env)
        md.Params.OutputFlag = 0
        md.Params.Method = 1
        md.Params.FeasibilityTol = 1e-9
        md.Params.OptimalityTol = 1e-9
        md.Params.Seed = self.seed
        U = inst.total_demand[i] * float((inst.c_reg[i] + inst.c_ot[i] + inst.h[i] + inst.b[i]).sum())
        y = md.addMVar(T, lb=0.0, ub=1.0)
        th = md.addMVar(S, lb=0.0, ub=U)
        md.setObjective(inst.F[i] @ y + inst.prob @ th, GRB.MINIMIZE)
        md.update()
        self.masters.append({"model": md, "y": y, "th": th, "yv": y.tolist(), "thv": th.tolist(), "lb": 0.0, "ub": INF,
                             "core": 0.5 * y0.astype(float) + 0.25, "lam": self.inout_lambda if self.stabilize else 1.0,
                             "stall": 0, "done": False, "y_lp": y0.astype(float), "rounds": 0, "pool_rounds": 0,
                             "exact_rounds": 0, "pool_pi": np.zeros((0, T)), "pool_mu": np.zeros((0, T)), "pool_keys": {}})

    def add_cuts(self, i: int, scen: np.ndarray, const: np.ndarray, coef: np.ndarray, pi: np.ndarray, mu: np.ndarray) -> None:
        if not len(scen):
            return
        m = self.masters[i]
        n = len(scen)
        E = sp.csr_matrix((np.ones(n), (np.arange(n), scen)), shape=(n, self.inst.S))
        m["model"].addConstr(E @ m["th"] + sp.csr_matrix(coef) @ m["y"] >= const)
        self.cutlog.add_multi(i * self.inst.S + np.asarray(scen), pi, mu, self.now())
        self.stats["cuts"] += n

    # ------------------------------------------------------------ LP phase

    def lp_phase(self, prods: list[int], end: float) -> None:
        """Round robin over the products of one block until they converge, the LP-phase end or the deadline."""
        while self.left() > 0 and time.perf_counter() < end:
            active = [i for i in prods if not self.masters[i]["done"]]
            if not active:
                break
            for i in active:
                if self.left() <= 0:
                    return
                self.lp_round(i)
            self.stats["lp_rounds"] += 1
            lb = float(sum(m["lb"] for m in self.masters))
            self.publish_bound(lb, "lp_master")
            self.log("lp_round", active=len(active))

    def lp_summary(self) -> None:
        I = self.inst.I
        done = len(self.masters) == I and all(m["done"] for m in self.masters)
        self.stats["lp_converged_products"] = int(sum(m["done"] for m in self.masters))
        self.stats["lp_bound"] = float(sum(m["lb"] for m in self.masters))
        self.stats["lp_ub"] = float(sum(m["ub"] for m in self.masters)) if len(self.masters) == I else None
        self.stats["pool_rounds"] = int(sum(m["pool_rounds"] for m in self.masters))
        self.stats["exact_rounds"] = int(sum(m["exact_rounds"] for m in self.masters))
        self.stats["pool_size"] = int(sum(len(m["pool_pi"]) for m in self.masters))
        if done:
            self.stats["lp_seconds"] = self.now()

    # ------------------------------------------------------------ dual pool

    def pool_add(self, i: int, pi: np.ndarray, mu: np.ndarray) -> None:
        """Distinct dual vectors of product i (dual feasible for every scenario of the product: the scenarios differ
        only in the demand right-hand side)."""
        if not self.pool:
            return
        m = self.masters[i]
        rows = np.unique(np.concatenate([pi, mu], axis=1), axis=0)
        keys = m["pool_keys"]
        new = [r for r in rows if r.tobytes() not in keys]
        if not new:
            return
        new = np.asarray(new)
        for r in new:
            keys[r.tobytes()] = True
        T = self.inst.T
        m["pool_pi"] = np.concatenate([m["pool_pi"], new[:, :T]])
        m["pool_mu"] = np.concatenate([m["pool_mu"], new[:, T:]])
        extra = len(m["pool_pi"]) - self.pool_max
        if extra > 0:  # keep the most recent vectors
            for r in np.concatenate([m["pool_pi"][:extra], m["pool_mu"][:extra]], axis=1):
                keys.pop(r.tobytes(), None)
            m["pool_pi"], m["pool_mu"] = m["pool_pi"][extra:], m["pool_mu"][extra:]

    def pool_cuts(self, i: int, point: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For every scenario the best pooled cut at `point`: const, coef (S, T) and its duals."""
        inst, m = self.inst, self.masters[i]
        V = m["pool_pi"] @ inst.demand[i]                  # (P, S)
        MK = m["pool_mu"] * inst.K[i][None, :]             # (P, T)
        j = np.argmax(V - (MK @ point)[:, None], axis=0)   # (S,)
        S = inst.S
        return V[j, np.arange(S)], MK[j], m["pool_pi"][j], m["pool_mu"][j]

    def converged(self, m: dict[str, Any]) -> bool:
        return m["ub"] < INF and m["ub"] - m["lb"] <= self.lp_tol * max(1.0, abs(m["ub"]))

    def lp_round(self, i: int) -> None:
        inst, m = self.inst, self.masters[i]
        S = inst.S
        md = m["model"]
        t = time.perf_counter()
        md.Params.TimeLimit = max(0.01, self.left())
        md.optimize()
        self.stats["master_seconds"] += time.perf_counter() - t
        if md.Status != GRB.OPTIMAL:
            return
        m["rounds"] += 1
        bound = float(md.ObjVal)
        m["stall"] = m["stall"] + 1 if bound <= m["lb"] + 1e-7 * max(1.0, abs(bound)) else 0
        m["lb"] = max(m["lb"], bound)
        y_lp = np.clip(np.asarray(m["y"].X), 0.0, 1.0)
        th = np.asarray(m["th"].X)
        m["y_lp"] = y_lp
        if md.NumConstrs > self.keep_rows * S:  # cut management (after reading the solution): drop non-binding rows
            cons = md.getConstrs()
            slack = np.asarray(md.getAttr("Slack", cons))
            loose = [c for c, sl in zip(cons, slack) if sl < -1e-7]
            if loose:
                md.remove(loose)
                md.update()
                self.stats["removed_cuts"] = self.stats.get("removed_cuts", 0) + len(loose)
        if self.converged(m):
            m["done"] = True
            return
        if m["stall"] >= 3 and m["lam"] < 1.0:
            m["lam"] = 1.0
        point = m["lam"] * y_lp + (1.0 - m["lam"]) * m["core"]
        if self.pool and m["ub"] < INF and len(m["pool_pi"]):
            t = time.perf_counter()
            const, coef, pi, mu = self.pool_cuts(i, point)
            viol = const - coef @ y_lp - th
            add = np.nonzero(viol > 1e-9 * np.maximum(1.0, np.abs(const)))[0]
            gain = float(inst.prob[add] @ viol[add])
            self.stats["pool_seconds"] = self.stats.get("pool_seconds", 0.0) + time.perf_counter() - t
            if len(add) and gain >= self.pool_ratio * (m["ub"] - m["lb"]):
                self.add_cuts(i, add, const[add], coef[add], pi[add], mu[add])
                m["core"] = 0.5 * m["core"] + 0.5 * point
                m["pool_rounds"] += 1
                return
        m["exact_rounds"] += 1
        t = time.perf_counter()
        res = self.ev.lp(i).solve(point, self.left(), duals=True)
        self.stats["sep_seconds"] += time.perf_counter() - t
        if res["status"] != "optimal":
            return
        m["ub"] = min(m["ub"], float(inst.F[i] @ point + inst.prob @ res["q"]))
        const, coef = cut_coefficients(inst, i, np.arange(S), res["pi"], res["mu"])
        viol = const - coef @ y_lp - th
        add = np.nonzero(viol > 1e-9 * np.maximum(1.0, np.abs(const)))[0]
        self.add_cuts(i, add, const[add], coef[add], res["pi"][add], res["mu"][add])
        self.pool_add(i, res["pi"], res["mu"])
        m["core"] = 0.5 * m["core"] + 0.5 * point
        if not len(add) and m["lam"] < 1.0:
            m["lam"] = 1.0  # the stabilized point separated nothing: Kelley at the LP point next
        if self.converged(m):
            m["done"] = True

    # ------------------------------------------------------------ primal phase

    def primal_phase(self) -> None:
        inst = self.inst
        for i in range(inst.I):
            yl = self.masters[i]["y_lp"]
            seen = set()
            for tau in (0.5, 0.25, 0.75):
                z = yl >= tau
                key = z.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                res = self.ev.product(i, z, self.deadline)
                if res is None:
                    return
                self.offer(i, z, res["value"], "lp_rounding")
        if self.local_search and self.left() > 0:
            t = time.perf_counter()
            end = t + self.ls_fraction * self.left()
            self.ps = ProductSearch(inst, self.ev, lambda i, yi, vi: self.offer(i, yi, vi, "local_search"), seed=self.seed)
            try:
                for i in np.argsort(-(self.v - np.array([m["lb"] for m in self.masters]))):
                    if time.perf_counter() >= end:
                        break
                    self.ps.descend(int(i), self.y[i].copy(), float(self.v[i]), end)
            finally:
                self.ps.close()
            self.stats["ls_seconds"] += time.perf_counter() - t

    # ------------------------------------------------------------ integer phase

    def integer_phase(self) -> str:
        inst = self.inst
        I = inst.I
        bounds = np.array([m["lb"] for m in self.masters])
        finished = np.zeros(I, dtype=bool)
        for m in self.masters:
            m["y"].VType = GRB.BINARY
            md = m["model"]
            md.Params.LazyConstraints = 1
            md.Params.MIPGap = MIP_GAP
            md.update()
        for _ in range(2):
            order = [int(i) for i in np.argsort(-(self.v - bounds)) if not finished[i]]
            for n_done, i in enumerate(order):
                if self.left() <= 0.5:
                    break
                share = self.left() / (len(order) - n_done)
                st, bnd = self.branch_and_cut(i, share, bounds)
                if bnd is not None:
                    bounds[i] = max(bounds[i], bnd)
                if st == GRB.OPTIMAL:
                    finished[i] = True
                self.int_trace.append((round(self.now(), 3), float(bounds.sum())))
            if finished.all() or self.left() <= 0.5:
                break
        self.stats["int_bound"] = float(bounds.sum())
        self.stats["bc_optimal_products"] = int(finished.sum())
        return "optimal" if finished.all() else "time_limit"

    def branch_and_cut(self, i: int, seconds: float, bounds: np.ndarray) -> tuple[int, float | None]:
        inst, m = self.inst, self.masters[i]
        S = inst.S
        md = m["model"]
        res = self.ev.product(i, self.y[i], self.deadline, duals=True)
        if res is None:
            return -1, None
        m["y"].Start = self.y[i].astype(float)
        m["th"].Start = res["q"]
        errors: list[BaseException] = []
        yv, thv = m["yv"], m["thv"]

        def callback(cb, where):
            try:
                if self.left() <= 0:
                    cb.terminate()
                    return
                if where == GRB.Callback.MIPSOL:
                    yy = np.asarray(cb.cbGetSolution(yv)) > 0.5
                    th = np.asarray(cb.cbGetSolution(thv))
                    r = self.ev.product(i, yy, self.deadline, duals=True)
                    if r is None:
                        cb.terminate()
                        return
                    self.offer(i, yy, r["value"], "branch_and_cut")
                    const, coef = cut_coefficients(inst, i, np.arange(S), r["pi"], r["mu"])
                    viol = const - coef @ yy - th
                    add = np.nonzero(viol > 1e-7 * np.maximum(1.0, np.abs(const)))[0]
                    for s in add:
                        nz = np.nonzero(coef[s])[0]
                        cb.cbLazy(gp.LinExpr(coef[s][nz].tolist(), [yv[t] for t in nz]) + thv[s] >= float(const[s]))
                    self.stats["lazy_cuts"] += len(add)  # not logged: the certificate is the LP-master bound
            except BaseException as exc:  # noqa: BLE001 - gurobipy would swallow it
                errors.append(exc)
                cb.terminate()

        t = time.perf_counter()
        md.Params.TimeLimit = max(0.01, min(seconds, self.left()))
        md.optimize(callback)
        self.stats["master_seconds"] += time.perf_counter() - t
        if errors:
            raise errors[0]
        st = int(md.Status)
        bnd = float(md.ObjBound) if st in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED) and math.isfinite(md.ObjBound) else None
        if st == GRB.OPTIMAL:  # finished: free the tree and the master
            md.dispose()
            m["model"] = None
        return st, bnd
