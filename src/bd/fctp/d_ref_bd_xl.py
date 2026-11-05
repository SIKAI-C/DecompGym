"""Engineered Benders reference for the FCTP (scale-up study; the paper's FCTP files are unchanged).

Master: min f.y + theta over y in {0, 1}^{m x n} with one recourse variable (the transportation recourse couples all
suppliers, so it does not split by customer or supplier: a per-customer split is only valid for fixed supplier
prices u, i.e. a Lagrangian relaxation that is never stronger than the aggregate cut, so cuts stay aggregated).
Cuts (c_naive_bd): theta >= s.u + d.v - sum m_ij (u_i + v_j - c_ij)^+ y_ij with the cheapest dual completion, and
Farkas cuts sum m_ij (u_i + v_j)^+ y_ij >= s.u + d.v (a no-good cut sum_{ybar_ij = 0} y_ij >= 1 if a ray does not
separate numerically).

Master-only valid inequalities: customer and supplier capacity covers sum_i m_ij y_ij >= d_j, sum_j m_ij y_ij >= s_i,
their cardinality versions (fewest arcs whose m_ij reach the requirement), theta >= sum_j d_j min_i c_ij, and Hall
cut-sets: for a supplier set I and a customer set T with deficit D(T) - S(I) > 0 (T's open neighbours inside I cannot
cover it), every feasible design opens arcs from outside I into T carrying the deficit:
    sum_{i not in I, j in T} min(m_ij, deficit) y_ij >= deficit,  its cardinality version and its MIR rounding.
Violated sets come from a max-flow min-cut (scipy) on the design (integer points: exact; fractional points: capacities
m_ij y_ij scaled by 1000 and rounded down, which only affects which sets are tried, not validity); the singleton covers
are also MIR-rounded at fractional points. At infeasible integer designs these combinatorial cuts replace the Farkas
cut (which is the weaker version with coefficients m_ij); the Farkas cut remains the fallback.

Stages, all within one wall-clock budget:
1. The shared basic initializer (Balinski's linearization, identical to the naive control): its LP value is the
   LP-relaxation bound, and its duals (u, v) give a single "LP phase" cut that lifts the master LP to that bound
   immediately (f_ij >= m_ij (u_i + v_j - c_ij)^+ by LP dual feasibility). Dynamic slope scaling (Kim and Pardalos
   1999) then proposes designs; every design is evaluated exactly, published, and seeds the master with its cut.
2. Basis local search (f_heuristic_xl) on the best plan (to a local optimum; with ils_seconds > 0, iterated local
   search for 10 * ils_seconds instead: the "engineered + ILS" variant used to test whether the decomposition adds
   anything on top of the heuristic).
3. Root cutting-plane loop on the master LP with in-out stabilization (separation at lam y_lp + (1 - lam) core,
   core moved toward feasible separation points; lam -> 1 when no violated cut or the bound stalls); non-binding
   Benders cuts are dropped at the end (cut management).
4. LP-guided designs (thresholds on y_lp, the recourse support at y_lp), evaluated, best one polished by local search.
5. Gurobi branch-and-Benders-cut on the master (one tree, lazy constraints): at every integer solution the
   transportation LP of the design is solved; the cleaned plan is published; a violated optimality cut is added as a
   Magnanti-Wong Pareto-optimal cut (auxiliary LP below; core point 0.5 * (best design + all-ones design), a convex
   combination of two feasible designs, so the MW LP is bounded; plain cut if the LP fails) or Hall cut-sets / a
   Farkas cut. A new plan within `ls_window` of the incumbent gets a time-capped local-search descent (ILS with
   ils_seconds > 0); improved plans are injected back into the tree. Root-node fractional points get Benders and
   cut-set user cuts. The master bound is published through the MIP callback.

Magnanti-Wong LP (primal form, one extra column eta): for a design ybar with recourse value T and core point y0,
    min c.x - T eta  s.t.  sum_j x_ij - s_i eta = s_i,  sum_i x_ij - d_j eta = d_j,  x_ij - m_ij ybar_ij eta <= m_ij y0_ij,
    x, eta >= 0;
its duals (u, v) maximize the cut value at y0 among the cuts that are tight at ybar.
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.fctp.a_data_xl import FCTPInstance, load_fctp
from src.bd.fctp.b_mip_xl import TransportLP, make_env, matrix_to_flows, plan_cost
from src.bd.fctp.c_naive_bd import INF, certify, cut_coefficients, cut_constant, initial_plan, is_memory_error
from src.bd.fctp.f_heuristic_xl import BasisLS, iterated_local_search, slope_scaling
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow


def cardinality(values: np.ndarray, need: float) -> int:
    """Fewest values (largest first) summing to at least `need` (len + 1 if impossible)."""
    s = np.cumsum(np.sort(values)[::-1])
    hit = np.nonzero(s >= need - 1e-9)[0]
    return int(hit[0]) + 1 if len(hit) else len(values) + 1


def mir_row(coef: np.ndarray, rhs: float, y: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    """Most violated mixed-integer rounding of sum coef_a y_a >= rhs (coef >= 0, y binary) at the point y: for a divisor
    lam with beta = rhs / lam fractional (f = frac(beta)),
        sum_a (floor(coef_a / lam) + min(1, frac(coef_a / lam) / f)) y_a >= ceil(beta).
    Divisors: the coefficients of the (at most 8) most fractional y_a and the largest coefficient."""
    if not len(coef):
        return None
    frac = (y > 1e-6) & (y < 1.0 - 1e-6)
    most = np.nonzero(frac)[0][np.argsort(np.abs(y[frac] - 0.5), kind="stable")[:8]]
    best = None
    for lam in np.unique(np.concatenate([coef[most], [coef.max()]])):
        if lam <= 1e-9:
            continue
        beta = rhs / lam
        fl = math.floor(beta + 1e-9)
        f = beta - fl
        if f < 1e-6 or f > 1.0 - 1e-6:
            continue  # integral (up to rounding): the rounded right-hand side would not be safe
        a = coef / lam
        fa = np.clip(a - np.floor(a + 1e-12), 0.0, 1.0)
        g = np.floor(a + 1e-12) + np.minimum(1.0, fa / f)
        r = float(fl + 1)
        viol = (r - g @ y) / max(1.0, r)
        if best is None or viol > best[2]:
            best = (g, r, viol)
    return best


def source_side(supply: np.ndarray, demand: np.ndarray, arc_cap: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Max flow source -> suppliers (supply) -> customers (arc_cap) -> sink (demand) with integer capacities; returns the
    source side of a minimum cut (suppliers, customers) and the flow value."""
    m, n = arc_cap.shape
    N = m + n + 2
    ai, aj = np.nonzero(arc_cap > 0)
    rows = np.concatenate([np.zeros(m, dtype=np.int64), 1 + ai, 1 + m + np.arange(n)])
    cols = np.concatenate([1 + np.arange(m), 1 + m + aj, np.full(n, N - 1)])
    data = np.concatenate([supply, arc_cap[ai, aj], demand]).astype(np.int32)
    G = csr_matrix((data, (rows, cols)), shape=(N, N))
    res = maximum_flow(G, 0, N - 1)
    R = G.toarray().astype(np.int64) - res.flow.toarray().astype(np.int64)
    seen = np.zeros(N, dtype=bool)
    seen[0] = True
    stack = [0]
    while stack:
        a = stack.pop()
        for b in np.nonzero((R[a] > 0) & ~seen)[0]:
            seen[b] = True
            stack.append(int(b))
    return seen[1:1 + m], seen[1 + m:1 + m + n], int(res.flow_value)


class ParetoLP:
    """Magnanti-Wong auxiliary LP in primal form (see the module docstring); persistent, eta coefficients updated."""

    def __init__(self, inst: FCTPInstance, env: gp.Env) -> None:
        self.inst = inst
        m, n = inst.m, inst.n
        md = gp.Model(env=env)
        md.Params.OutputFlag = 0
        md.Params.FeasibilityTol = 1e-9
        md.Params.OptimalityTol = 1e-9
        self.model = md
        self.x = md.addMVar((m, n), lb=0.0, name="x")
        self.eta = md.addMVar(1, lb=0.0, name="eta")
        self.srows = md.addConstr(self.x.sum(axis=1) - inst.supply[:, None] @ self.eta == inst.supply)
        self.drows = md.addConstr(self.x.sum(axis=0) - inst.demand[:, None] @ self.eta == inst.demand)
        self.link = md.addConstr(self.x.reshape(-1) <= np.zeros(m * n))
        md.setObjective((inst.tcost * self.x).sum(), GRB.MINIMIZE)
        md.update()
        self.link_rows = self.link.tolist()
        self.eta_var = self.eta.tolist()[0]
        self.coef = np.zeros(m * n)
        self.capf = inst.cap.ravel()
        self.solves = 0
        self.failures = 0

    def duals(self, ybar: np.ndarray, T: float, y0: np.ndarray, time_limit: float) -> tuple[np.ndarray, np.ndarray] | None:
        coef = -self.capf * np.asarray(ybar, dtype=float).ravel()
        for k in np.nonzero(coef != self.coef)[0]:
            self.model.chgCoeff(self.link_rows[k], self.eta_var, float(coef[k]))
        self.coef = coef
        self.link.RHS = self.capf * np.asarray(y0, dtype=float).ravel()
        self.eta_var.Obj = -T
        self.model.Params.TimeLimit = max(0.01, time_limit)
        self.model.optimize()
        self.solves += 1
        if self.model.Status != GRB.OPTIMAL:
            self.failures += 1
            return None
        return np.asarray(self.srows.Pi, dtype=float), np.asarray(self.drows.Pi, dtype=float)

    def dispose(self) -> None:
        self.model.dispose()


class FCTPBendersXL:
    """Engineered Benders branch-and-cut reference for the FCTP; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 ss_steps: int = 30, ss_fraction: float = 0.05, root_fraction: float = 0.15, root_rounds: int = 300,
                 inout_lambda: float = 0.3, local_search: bool = True, ls_seconds: float = 2.0, ls_window: float = 0.02,
                 pareto: bool = True, cardinality_rows: bool = True, node_cut_rounds: int = 20, cutsets: bool = True,
                 ils_seconds: float = 0.0, verbose: bool = False) -> None:
        self.time_limit = time_limit
        self.threads = threads
        self.on_incumbent = on_incumbent or (lambda flows, stage=None: None)  # sparse flows [[i, j, q], ...]
        self.on_bound = on_bound or (lambda bound: None)
        self.soft_mem_gb = soft_mem_gb
        self.seed = seed
        self.ss_steps, self.ss_fraction = ss_steps, ss_fraction
        self.root_fraction, self.root_rounds, self.inout_lambda = root_fraction, root_rounds, inout_lambda
        self.local_search, self.ls_seconds, self.ls_window = local_search, ls_seconds, ls_window
        self.pareto, self.cardinality_rows, self.node_cut_rounds = pareto, cardinality_rows, node_cut_rounds
        self.cutsets, self.ils_seconds = cutsets, ils_seconds
        self.polish_seconds = ils_seconds if ils_seconds > 0 else ls_seconds
        self.rng = np.random.default_rng(seed)
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
        if self.inst.integral:
            value = float(math.ceil(value))
        if value > self.lb and self.left() > 0:
            self.lb = value
            self.on_bound(value)
            self.log(stage)

    def offer(self, X: np.ndarray, stage: str) -> bool:
        cost = plan_cost(self.inst, X)
        if cost < self.ub - 1e-9 and self.left() > 0:
            self.ub, self.best = cost, X.copy()
            self.core = 0.5 + 0.5 * (X > 0)
            self.pending = True
            self.on_incumbent(matrix_to_flows(X), stage=stage)
            self.log(stage)
            return True
        return False

    def closed(self) -> bool:
        return self.ub < INF and self.lb >= self.ub - 1e-6 * max(1.0, abs(self.ub))

    # ------------------------------------------------------------ run

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        return self.solve(load_fctp(data_path))

    def solve(self, inst: FCTPInstance) -> dict[str, Any]:
        if not hasattr(self, "t0"):
            self.t0 = time.perf_counter()
            self.deadline = self.t0 + self.time_limit
        self.inst = inst
        self.ub, self.lb = INF, -INF
        self.best: np.ndarray | None = None
        self.core = np.ones((inst.m, inst.n))  # the all-arcs design (feasible); later 0.5 * (best + ones)
        self.pending = False
        self.stats: dict[str, Any] = {
            "root_rounds": 0, "ss_lps": 0, "cuts": 0, "optimality_cuts": 0, "pareto_cuts": 0, "feasibility_cuts": 0,
            "nogood_cuts": 0, "lazy_cuts": 0, "user_cuts": 0, "mipsol_calls": 0, "injected": 0, "evaluations": 0,
            "ls_calls": 0, "ls_moves": 0, "ls_seconds": 0.0, "master_seconds": 0.0, "sub_seconds": 0.0,
            "heuristic_seconds": 0.0, "root_seconds": 0.0, "root_bound": None, "lp_bound": None, "root_cuts_total": 0,
            "root_cuts_kept": 0, "node_count": 0.0, "bc_status": None, "cutset_rows": 0, "mir_rows": 0, "ils_iterations": 0}
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        self.env = env
        self.ev: TransportLP | None = None
        self.mw: ParetoLP | None = None
        reason = "time_limit"
        try:
            reason = self.search()
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        finally:
            for obj in (self.ev, self.mw, getattr(self, "master", None)):
                if obj is not None:
                    obj.dispose()
            env.dispose()
        ub = self.ub if self.ub < INF else None
        lb = self.lb if self.lb > -INF else None
        return {"best_objective": ub, "global_lb": lb, "global_ub": ub,
                "gap": (ub - lb) / abs(ub) if ub is not None and lb is not None and ub else None,
                "termination_reason": reason, "budget_exhausted": self.elapsed() >= self.time_limit - 1.0,
                "flows": matrix_to_flows(self.best) if self.best is not None else None,
                "stats": self.stats, **self.stats, "total_seconds": self.elapsed(), "trace": self.trace}

    def search(self) -> str:
        inst = self.inst
        self.publish_bound(0.0, "trivial")
        self.ev = TransportLP(inst, self.env)
        if self.pareto:
            self.mw = ParetoLP(inst, self.env)
        t = time.perf_counter()
        init = initial_plan(inst, self.env, self.left())
        if init is None:
            return "time_limit"
        self.stats["lp_bound"] = init["lp_value"]
        self.publish_bound(init["lp_value"], "lp_relaxation")
        self.offer(init["X"], "initializer")
        self.build_master()
        # The LP phase cut: Balinski's duals lift the master LP to the LP-relaxation bound.
        self.add_cut(cut_constant(inst, init["u"], init["v"]), cut_coefficients(inst, init["u"], init["v"], False), False)
        self.evaluate(init["X"] > 0, "initializer", seed_cut=True)
        if self.closed():
            return "optimal"
        # 1. Slope scaling.
        self.stats["ss_lps"] = slope_scaling(inst, self.env, init["X"], min(self.deadline, time.perf_counter() + self.ss_fraction * max(0.0, self.left())),
                                             self.ss_steps, lambda d: self.evaluate(d, "slope_scaling", seed_cut=True))
        # 2. Local search on the best plan.
        if self.local_search and self.best is not None and self.left() > 0:
            X = self.polish(self.best, "local_search", 10 * self.polish_seconds)
            self.evaluate(X > 0, "local_search", seed_cut=True)
        self.stats["heuristic_seconds"] += time.perf_counter() - t
        if self.closed():
            return "optimal"
        # 3. Root loop; 4. LP-guided designs.
        if self.left() > 0:
            y_lp, flows_lp = self.root_loop()
            if self.closed():
                return "optimal"
            if y_lp is not None and self.left() > 0:
                t = time.perf_counter()
                self.lp_heuristics(y_lp, flows_lp)
                self.stats["heuristic_seconds"] += time.perf_counter() - t
        if self.closed():
            return "optimal"
        if self.left() <= 0.01:
            return "time_limit"
        # 5. Branch-and-Benders-cut.
        return self.branch_and_cut()

    # ------------------------------------------------------------ master and cuts

    def build_master(self) -> None:
        inst = self.inst
        m, n = inst.m, inst.n
        md = gp.Model(env=self.env)
        md.Params.OutputFlag = 0
        md.Params.Seed = self.seed
        self.master = md
        cap = inst.cap
        self.y = md.addMVar((m, n), lb=0.0, ub=1.0, name="y")
        self.theta = md.addVar(lb=float(inst.demand @ inst.tcost.min(axis=0)), name="theta")
        md.setObjective((inst.fcost * self.y).sum() + self.theta, GRB.MINIMIZE)
        md.addConstr((cap * self.y).sum(axis=0) >= inst.demand, name="cover_d")
        md.addConstr((cap * self.y).sum(axis=1) >= inst.supply, name="cover_s")
        if self.cardinality_rows:
            for j in range(n):
                r = cardinality(cap[:, j], inst.demand[j])
                if r > 1:
                    md.addConstr(self.y[:, j].sum() >= r)
            for i in range(m):
                r = cardinality(cap[i, :], inst.supply[i])
                if r > 1:
                    md.addConstr(self.y[i, :].sum() >= r)
        self.yflat = [v for row in self.y.tolist() for v in row]
        self.cuts: list[gp.Constr] = []

    def add_cut(self, const: float, h: np.ndarray, feasibility: bool, cb: gp.Model | None = None, user: bool = False) -> None:
        hf = h.ravel()
        nz = np.nonzero(hf > 1e-12)[0]
        expr = gp.LinExpr(hf[nz].tolist(), [self.yflat[k] for k in nz])
        if not feasibility:
            expr.add(self.theta)
        row = expr >= const
        if cb is None:
            self.cuts.append(self.master.addConstr(row))
        elif user:
            cb.cbCut(row)
            self.stats["user_cuts"] += 1
        else:
            cb.cbLazy(row)
            self.stats["lazy_cuts"] += 1
        self.stats["cuts"] += 1
        self.stats["feasibility_cuts" if feasibility else "optimality_cuts"] += 1

    def add_nogood(self, ybar: np.ndarray, cb: gp.Model | None = None) -> None:
        """Some arc outside an infeasible design must open (every subset of an infeasible design is infeasible)."""
        closed = np.nonzero(~ybar.ravel())[0]
        row = gp.LinExpr([1.0] * len(closed), [self.yflat[k] for k in closed]) >= 1
        if cb is None:
            self.master.addConstr(row)
        else:
            cb.cbLazy(row)
            self.stats["lazy_cuts"] += 1
        self.stats["nogood_cuts"] += 1

    def design_cut(self, ybar: np.ndarray, res: dict[str, Any]) -> tuple[float, np.ndarray]:
        """Optimality cut tight at the integer design ybar: Magnanti-Wong if enabled and successful, else plain."""
        inst = self.inst
        T = res["obj"]
        if self.mw is not None and self.left() > 0:
            t = time.perf_counter()
            uv = self.mw.duals(ybar, T, self.core, self.left())
            self.stats["sub_seconds"] += time.perf_counter() - t
            if uv is not None:
                const, h = cut_constant(inst, *uv), cut_coefficients(inst, *uv, False)
                if const - float((h * ybar).sum()) >= T - 1e-6 * max(1.0, abs(T)):
                    self.stats["pareto_cuts"] += 1
                    return const, h
        return cut_constant(inst, res["u"], res["v"]), cut_coefficients(inst, res["u"], res["v"], False)

    def feasibility_cut(self, ybar: np.ndarray, res: dict[str, Any], cb: gp.Model | None = None) -> None:
        inst = self.inst
        if self.cutsets and inst.integral:
            # Open arcs uncapacitated (m_ij never binds), so the cut is a pure Hall set: no open arc crosses it.
            sup, cus, flow = source_side(inst.supply, inst.demand, np.where(ybar, inst.supply.sum() + 1, 0.0))
            rows = self.cutset_rows(sup, cus, None)
            if rows and flow < inst.supply.sum():
                self.add_rows(rows, cb, lazy=True)
                return
        const, h = cut_constant(inst, res["u"], res["v"]), cut_coefficients(inst, res["u"], res["v"], True)
        if const - float((h * ybar).sum()) > 1e-7 * max(1.0, abs(const)):
            self.add_cut(const, h, True, cb)
        else:
            self.add_nogood(ybar, cb)

    def cutset_rows(self, sup_S: np.ndarray, cus_S: np.ndarray, y: np.ndarray | None) -> list[tuple[np.ndarray, np.ndarray, float, str]]:
        """Hall cut-set rows for the cut (suppliers I = not sup_S, customers T = not cus_S): arcs from sup_S into T must
        carry deficit = D(T) - S(I); capacity, cardinality and (with a point y) the most violated MIR row."""
        inst = self.inst
        T, I = ~cus_S, ~sup_S
        deficit = float(inst.demand[T].sum() - inst.supply[I].sum())
        if deficit <= 1e-9:
            return []
        idx = np.nonzero((sup_S[:, None] & T[None, :]).ravel())[0]
        if not len(idx):
            raise RuntimeError("Hall set without crossing arcs: the instance would be infeasible")
        coef = np.minimum(inst.cap.ravel()[idx], deficit)
        rows = [(idx, coef, deficit, "cap")]
        r = cardinality(coef, deficit)
        if r > 1:
            rows.append((idx, np.ones(len(idx)), float(r), "card"))
        if y is not None:
            mir = mir_row(coef, deficit, y.ravel()[idx])
            if mir is not None and mir[2] > 1e-4:
                rows.append((idx, mir[0], mir[1], "mir"))
        return rows

    def add_rows(self, rows: list, cb: gp.Model | None = None, lazy: bool = False) -> int:
        for idx, coef, rhs, kind in rows:
            expr = gp.LinExpr(coef.tolist(), [self.yflat[k] for k in idx])
            if cb is None:
                self.master.addConstr(expr >= rhs)
            elif lazy:
                cb.cbLazy(expr >= rhs)
                self.stats["lazy_cuts"] += 1
            else:
                cb.cbCut(expr >= rhs)
                self.stats["user_cuts"] += 1
            self.stats["mir_rows" if kind == "mir" else "cutset_rows"] += 1
        return len(rows)

    def separate_rows(self, y: np.ndarray) -> list:
        """Violated master rows at a fractional point: MIR roundings of the singleton covers and Hall cut-sets from a
        scaled max-flow on the capacities m_ij y_ij."""
        if not self.cutsets:
            return []
        inst = self.inst
        cap = inst.cap
        out = []
        m, n = inst.m, inst.n
        for j in range(n):
            mir = mir_row(cap[:, j], float(inst.demand[j]), y[:, j])
            if mir is not None and mir[2] > 1e-3:
                out.append((np.arange(m) * n + j, mir[0], mir[1], "mir"))
        for i in range(m):
            mir = mir_row(cap[i, :], float(inst.supply[i]), y[i, :])
            if mir is not None and mir[2] > 1e-3:
                out.append((i * n + np.arange(n), mir[0], mir[1], "mir"))
        if inst.integral:
            scale = 1000
            sup, cus, flow = source_side(scale * inst.supply, scale * inst.demand, np.floor(scale * cap * y))
            if flow < scale * inst.supply.sum():
                for idx, coef, rhs, kind in self.cutset_rows(sup, cus, y):
                    if (rhs - coef @ y.ravel()[idx]) / max(1.0, rhs) > 1e-4:
                        out.append((idx, coef, rhs, kind))
        return out

    # ------------------------------------------------------------ primal side

    def evaluate(self, design: np.ndarray, stage: str, seed_cut: bool) -> dict[str, Any]:
        """Exact evaluation of a binary design (outside callbacks): publish the cleaned plan; optionally seed the master."""
        design = np.asarray(design, dtype=bool)
        t = time.perf_counter()
        res = self.ev.evaluate(design, max(0.01, self.left()))
        self.stats["sub_seconds"] += time.perf_counter() - t
        self.stats["evaluations"] += 1
        if res["status"] == "optimal":
            self.offer(res["X"], stage)
            if seed_cut:
                self.add_cut(*self.design_cut(design, res), False)
        elif res["status"] == "infeasible" and seed_cut:
            self.feasibility_cut(design, res)
        return res

    def polish(self, X: np.ndarray, stage: str, seconds: float) -> np.ndarray:
        """Basis local-search descent from the plan X (time-capped); publishes an improvement."""
        t = time.perf_counter()
        ls = BasisLS(self.inst, X)
        end = min(self.deadline, t + seconds)
        if self.ils_seconds > 0:
            st = iterated_local_search(ls, end, self.rng, lambda Z: self.offer(Z, stage))
            self.stats["ils_iterations"] += st["ils_iterations"]
        else:
            ls.descend(end)
        Y = ls.matrix()
        self.stats["ls_calls"] += 1
        self.stats["ls_moves"] += ls.moves
        self.stats["ls_seconds"] += time.perf_counter() - t
        self.offer(Y, stage)
        return Y

    def lp_heuristics(self, y_lp: np.ndarray, flows_lp: np.ndarray | None) -> None:
        cands = ([flows_lp > 1e-9] if flows_lp is not None else []) + [y_lp >= tau for tau in (0.5, 0.25, 0.1, 0.02)]
        seen: set[bytes] = set()
        best = None
        for design in cands:
            if self.left() <= 0:
                return
            if design.tobytes() in seen:
                continue
            seen.add(design.tobytes())
            res = self.evaluate(design, "lp_rounding", seed_cut=True)
            if res["status"] == "optimal" and (best is None or res["cost"] < best[0]):
                best = (res["cost"], res["X"])
        if best is not None and self.local_search and self.left() > 0:
            X = self.polish(best[1], "lp_local_search", 5 * self.polish_seconds)
            self.evaluate(X > 0, "lp_local_search", seed_cut=True)

    # ------------------------------------------------------------ root

    def root_loop(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        inst, md = self.inst, self.master
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
            th = float(self.theta.X)
            extra = self.add_rows(self.separate_rows(y_lp))
            sep = lam * y_lp + (1.0 - lam) * core
            t = time.perf_counter()
            res = self.ev.solve(inst.cap * sep, min(self.left(), max(0.01, root_deadline - time.perf_counter())), flows=False)
            self.stats["sub_seconds"] += time.perf_counter() - t
            added = extra
            if res["status"] == "optimal":
                const, h = cut_constant(inst, res["u"], res["v"]), cut_coefficients(inst, res["u"], res["v"], False)
                if const - float((h * y_lp).sum()) - th > 1e-6 * max(1.0, abs(const)):
                    self.add_cut(const, h, False)
                    added = 1
                core = 0.5 * core + 0.5 * sep
            elif res["status"] == "infeasible":
                const, h = cut_constant(inst, res["u"], res["v"]), cut_coefficients(inst, res["u"], res["v"], True)
                if const - float((h * y_lp).sum()) > 1e-6 * max(1.0, abs(const)):
                    self.add_cut(const, h, True)
                    added = 1
            else:
                break  # out of time
            if added == 0 or stalled >= 5:
                if lam < 1.0:
                    lam, stalled = 1.0, 0  # finish with Kelley cuts at the LP point
                    continue
                break
        self.stats["root_cuts_total"] = len(self.cuts) - n_before
        flows_lp = None
        if self.left() > 0:
            md.Params.TimeLimit = max(0.01, self.left())
            md.optimize()
            if md.Status == GRB.OPTIMAL:
                self.publish_bound(float(md.ObjVal), "root_lp")
                y_lp = np.clip(np.asarray(self.y.X), 0.0, 1.0)
                # Cut management: drop the non-binding Benders cuts so the branch-and-cut master stays small.
                loose = [c for c in self.cuts if c.Slack < -1e-6 * max(1.0, abs(c.RHS))]
                md.remove(loose)
                gone = {id(c) for c in loose}
                self.cuts = [c for c in self.cuts if id(c) not in gone]
                res = self.ev.solve(inst.cap * y_lp, self.left())
                if res["status"] == "optimal":
                    flows_lp = res["flows"]
        self.stats["root_cuts_kept"] = len(self.cuts)
        self.stats["root_bound"] = self.lb
        self.stats["root_seconds"] = time.perf_counter() - t_root
        return y_lp, flows_lp

    # ------------------------------------------------------------ branch-and-cut

    def branch_and_cut(self) -> str:
        inst, md = self.inst, self.master
        self.y.VType = GRB.BINARY
        md.Params.LazyConstraints = 1
        md.Params.PreCrush = 1
        md.Params.MIPGap = 0.0
        if inst.integral:
            md.Params.MIPGapAbs = 0.99
        if self.best is not None:
            self.y.Start = (self.best > 0).astype(float)
            self.theta.Start = float((inst.tcost * self.best).sum())
            self.pending = False
        cache: OrderedDict[bytes, dict[str, Any]] = OrderedDict()
        polished: set[bytes] = set()
        node_rounds = [0]
        bc_start = time.perf_counter()

        def evaluate(yy: np.ndarray) -> dict[str, Any]:
            key = yy.tobytes()
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            t = time.perf_counter()
            res = self.ev.evaluate(yy, max(0.01, self.left()))
            self.stats["sub_seconds"] += time.perf_counter() - t
            self.stats["evaluations"] += 1
            if res["status"] in ("optimal", "infeasible"):
                cache[key] = {k: v for k, v in res.items() if k not in ("flows",)}
                while len(cache) > 500:
                    cache.popitem(last=False)
                res = dict(res, fresh=True)
            return res

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
                yy = np.asarray(cb.cbGetSolution(self.yflat)).reshape(inst.m, inst.n) > 0.5
                th = float(cb.cbGetSolution(self.theta))
                res = evaluate(yy)
                if res["status"] == "optimal":
                    if res.get("fresh"):
                        self.offer(res["X"], "branch_and_cut")
                        key = (res["X"] > 0).tobytes()
                        bc_time = time.perf_counter() - bc_start
                        if (self.local_search and key not in polished and res["cost"] <= self.ub * (1.0 + self.ls_window)
                                and self.stats["ls_seconds"] <= 0.5 * bc_time + 10 * self.polish_seconds):
                            polished.add(key)
                            self.polish(res["X"], "bc_local_search", self.polish_seconds)
                    if th < res["obj"] - 1e-6 * max(1.0, abs(res["obj"])):
                        self.add_cut(*self.design_cut(yy, res), False, cb)
                elif res["status"] == "infeasible":
                    self.feasibility_cut(yy, res, cb)
                elif res["status"] == "time_limit":
                    cb.terminate()
                else:
                    raise RuntimeError(f"recourse status {res['status']} at an integer master solution")
            elif where == GRB.Callback.MIPNODE:
                if cb.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                    return
                if self.pending and self.best is not None:
                    cb.cbSetSolution(self.yflat, (self.best > 0).astype(float).ravel().tolist())
                    cb.cbSetSolution(self.theta, float((inst.tcost * self.best).sum()))
                    cb.cbUseSolution()
                    self.pending = False
                    self.stats["injected"] += 1
                if cb.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0 and node_rounds[0] < self.node_cut_rounds:
                    node_rounds[0] += 1
                    y_node = np.clip(np.asarray(cb.cbGetNodeRel(self.yflat)).reshape(inst.m, inst.n), 0.0, 1.0)
                    th = float(cb.cbGetNodeRel(self.theta))
                    t = time.perf_counter()
                    res = self.ev.solve(inst.cap * y_node, max(0.01, self.left()), flows=False)
                    self.stats["sub_seconds"] += time.perf_counter() - t
                    if res["status"] == "optimal":
                        const, h = cut_constant(inst, res["u"], res["v"]), cut_coefficients(inst, res["u"], res["v"], False)
                        if const - float((h * y_node).sum()) - th > 1e-4 * max(1.0, abs(const)):
                            self.add_cut(const, h, False, cb, user=True)
                    self.add_rows(self.separate_rows(y_node), cb, lazy=False)
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
        return "interrupted" if md.Status == GRB.INTERRUPTED else f"status_{md.Status}"
