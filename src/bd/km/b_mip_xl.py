"""k-median (scale-up): the solution contract, exact design evaluation, the compact MIP baselines and the compact LP
control.

Solutions (published by every method, stored by the probe runner) are designs: the sorted list of the k open
facilities. The value of a design is sum_j min_{i open} D_ij (evaluated exactly: integral distances sum exactly in
float64; `design_value`), the output contract is that of e_verifier_xl.

Compact formulations (y_i = 1 if facility i is open; F_j = the L = m - k + 1 facilities nearest to customer j, see
a_data_xl.SortedCosts: restricting customer j to F_j is exact for integer designs and leaves the LP relaxation
unchanged, so both formulations below have the LP bound of the classical formulation):
  classical  min sum_j sum_{i in F_j} D_ij x_ij  s.t.  sum_{i in F_j} x_ij = 1 (j),  x_ij <= y_i,  sum_i y_i = k,
             x >= 0, y binary (`reduce=False` keeps all m x n pairs);
  radius     (Elloumi 2010, the covering form of Garcia, Labbe and Marin 2011): for customer j with distinct
             distances D_j^1 < ... < D_j^K over F_j, z_j^t = 1 iff no open facility within D_j^t:
             min sum_j [D_j^1 + sum_{t<K} (D_j^{t+1} - D_j^t) z_j^t]
             s.t. z_j^1 + sum_{i: D_ij = D_j^1} y_i >= 1,   z_j^t + sum_{i: D_ij = D_j^t} y_i >= z_j^{t-1} (t >= 2),
                  sum_i y_i = k,  z >= 0,  y binary.
             Both have O(sum_j |F_j|) nonzeros. The radius model's master variables z are not in y-space; its dual
             is not exported as a certificate (the y-space certificate of the compact LP comes from the classical
             model's assignment-row duals, see KMCompactLP).
MIP gap: relative 1e-6 and, for integral distances, absolute 0.999 (integer objective), so `optimal` means proved.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Iterable

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from src.bd.km.a_data_xl import KMInstance, SortedCosts, integral_costs, load_instance
from src.bd.scfl.b_mip_xl import is_memory_error, make_env

MIP_GAP = 1e-6


# ---------------------------------------------------------------- contract and evaluation

def normalize_design(design: Iterable) -> list[int]:
    return sorted({int(i) for i in design})


def check_design(inst: KMInstance, design: Iterable) -> list[int]:
    d = normalize_design(design)
    if len(d) != inst.k or (d and (d[0] < 0 or d[-1] >= inst.m)):
        raise ValueError(f"a design must open exactly k={inst.k} distinct facilities in [0, {inst.m})")
    return d


def design_costs(D: np.ndarray, design: Iterable) -> tuple[np.ndarray, np.ndarray]:
    """(per-customer cost, index of the cheapest open facility) of a design (ties: lowest facility index)."""
    idx = np.asarray(normalize_design(design), dtype=np.int64)
    sub = D[idx]
    a = np.argmin(sub, axis=0)
    return sub[a, np.arange(D.shape[1])], idx[a]


def design_value(D: np.ndarray, design: Iterable) -> float:
    c, _ = design_costs(D, design)
    return float(math.fsum(c.tolist())) if not integral_costs(c) else float(c.sum())


def open_list(y: np.ndarray) -> list[int]:
    return [int(i) for i in np.nonzero(np.asarray(y) > 0.5)[0]]


def km_output(inst: KMInstance, design: Iterable, D: np.ndarray | None = None, status: str = "feasible",
              method: str = "benders_decomposition") -> dict:
    """Output contract of e_verifier_xl: y (0/1 per facility), assign (cheapest open facility per customer),
    objective_value = the design value."""
    d = check_design(inst, design)
    D = inst.distances() if D is None else D
    _, a = design_costs(D, d)
    y = np.zeros(inst.m, dtype=int)
    y[d] = 1
    return {"status": status, "method": method, "objective_value": design_value(D, d), "y": y.tolist(),
            "assign": [int(i) for i in a]}


class DesignEvaluator:
    """Cached exact design values (the runner's deferred evaluation)."""

    def __init__(self, inst: KMInstance, D: np.ndarray | None = None) -> None:
        self.inst = inst
        self.D = inst.distances() if D is None else D
        self.cache: dict[tuple[int, ...], float | None] = {}

    def objective(self, design: Iterable) -> float | None:
        key = tuple(normalize_design(design))
        if key not in self.cache:
            try:
                check_design(self.inst, key)
                self.cache[key] = design_value(self.D, key)
            except ValueError:
                self.cache[key] = None
        return self.cache[key]


# ---------------------------------------------------------------- compact models

def _pairs(sc: SortedCosts, reduce: bool, D: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(facility, customer, cost) of the x variables, customer-major."""
    n = sc.n
    if reduce:
        fi = sc.order.ravel().astype(np.int64)
        cj = np.repeat(np.arange(n), sc.L)
        return fi, cj, sc.ds.ravel()
    m = D.shape[0]
    fi = np.tile(np.arange(m), n)
    cj = np.repeat(np.arange(n), m)
    return fi, cj, np.ascontiguousarray(D.T).ravel()


def build_classical(inst: KMInstance, D: np.ndarray, env: gp.Env | None = None, relax: bool = False,
                    reduce: bool = True, sc: SortedCosts | None = None) -> tuple[gp.Model, dict[str, Any]]:
    sc = sc or SortedCosts(D, inst.k)
    fi, cj, cost = _pairs(sc, reduce, D)
    N = len(fi)
    md = gp.Model(env=env) if env else gp.Model()
    y = md.addMVar(inst.m, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS if relax else GRB.BINARY, name="y")
    x = md.addMVar(N, lb=0.0, ub=1.0, obj=cost, name="x")
    md.ModelSense = GRB.MINIMIZE
    A = sp.csr_matrix((np.ones(N), (cj, np.arange(N))), shape=(inst.n, N))
    assign = md.addConstr(A @ x == np.ones(inst.n), name="assign")
    B = sp.csr_matrix((np.ones(N), (np.arange(N), fi)), shape=(N, inst.m))
    md.addConstr(x - B @ y <= 0.0, name="link")
    card = md.addConstr(y.sum() == inst.k, name="card")
    md.update()
    return md, {"y": y, "x": x, "assign": assign, "card": card, "num_vars": int(md.NumVars),
                "num_constrs": int(md.NumConstrs), "pairs": N}


def build_radius(inst: KMInstance, D: np.ndarray, env: gp.Env | None = None, relax: bool = False,
                 sc: SortedCosts | None = None) -> tuple[gp.Model, dict[str, Any]]:
    sc = sc or SortedCosts(D, inst.k)
    n, L = sc.n, sc.L
    ds, order = sc.ds, sc.order
    new = np.zeros((n, L), dtype=bool)
    new[:, 1:] = ds[:, 1:] != ds[:, :-1]
    lev = np.cumsum(new, axis=1)                 # 0-based level of every position
    K = lev[:, -1] + 1                           # number of distinct levels per customer
    nz = K - 1                                   # z variables (levels 1..K-1) per customer
    off = np.concatenate([[0], np.cumsum(nz)])
    R = int(off[-1])
    # level values: first position of each level
    starts = np.nonzero(np.concatenate([np.ones((n, 1), dtype=bool), new[:, 1:]], axis=1))
    lv = np.zeros((n, L))
    lv[starts[0], lev[starts]] = ds[starts]
    jj = np.repeat(np.arange(n), nz)
    tt = np.arange(R) - off[jj]                  # 0-based level index of each z
    zcost = lv[jj, tt + 1] - lv[jj, tt]
    md = gp.Model(env=env) if env else gp.Model()
    y = md.addMVar(inst.m, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS if relax else GRB.BINARY, name="y")
    z = md.addMVar(R, lb=0.0, obj=zcost, name="z")
    md.ModelSense = GRB.MINIMIZE
    md.ObjCon = float(sc.ds[:, 0].sum())
    keep = lev < (K - 1)[:, None]                # positions in levels that have a row
    pj, pt = np.nonzero(keep)
    rows_y = off[pj] + lev[pj, pt]
    Ay = sp.csr_matrix((np.ones(len(pj)), (rows_y, order[pj, pt])), shape=(R, inst.m))
    sub = tt > 0
    rz = np.concatenate([np.arange(R), np.nonzero(sub)[0]])
    cz = np.concatenate([np.arange(R), np.nonzero(sub)[0] - 1])
    vz = np.concatenate([np.ones(R), -np.ones(int(sub.sum()))])
    Az = sp.csr_matrix((vz, (rz, cz)), shape=(R, R))
    rhs = (tt == 0).astype(float)
    cover = md.addConstr(Ay @ y + Az @ z >= rhs, name="cover")
    card = md.addConstr(y.sum() == inst.k, name="card")
    md.update()
    # y-space translation of the dual (protocol 3.1: only y-space cuts are certified): for customer j with rows,
    # v_j = D_j^1 + pi(first row of j) is a dual of its classical subproblem (with w_i = pi(level of i)); customers
    # with a single level have Q_j = D_j^1 (v_j = D_j^1).
    return md, {"y": y, "z": z, "card": card, "cover": cover, "first_row": off[:-1].astype(np.int64), "has_row": nz > 0,
                "d1": sc.ds[:, 0].copy(), "num_vars": int(md.NumVars), "num_constrs": int(md.NumConstrs), "levels": R}


def _set_gap(md: gp.Model, integral: bool) -> None:
    """Integral costs: MIPGap = 0 and absolute gap 0.999 (unit-gap closure is the only stop); otherwise relative
    MIP_GAP. 'optimal' is reported only after the gap check in the caller (Codex review 46 #5)."""
    if integral:
        md.Params.MIPGap = 0.0
        md.Params.MIPGapAbs = 0.999
    else:
        md.Params.MIPGap = MIP_GAP


def gap_closed(ub: float, bound: float, integral: bool) -> bool:
    if not (math.isfinite(ub) and math.isfinite(bound)):
        return False
    if integral:
        return ub - bound < 1.0 - 1e-9
    return ub - bound <= MIP_GAP * max(1.0, abs(ub))


class KMCompactMIP:
    """Compact MIP (radius or classical) with the interface of the decomposition methods: every new incumbent's
    design is published strictly before the deadline (scored by its exact value after the clock stops); the MIP
    bound (solver-reported, integer side) through on_bound."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 formulation: str = "radius", reduce: bool = True, method: int | None = None,
                 mip_focus: int | None = None, mem_limit_gb: float | None = None) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.formulation, self.reduce, self.method, self.mip_focus = formulation, reduce, method, mip_focus
        self.mem_limit_gb = mem_limit_gb
        self.on_incumbent = on_incumbent or (lambda design, stage=None: None)
        self.on_bound = on_bound or (lambda bound: None)

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        inst = load_instance(data_path)
        D = inst.distances()
        integral = integral_costs(D)
        env = make_env(self.threads, self.soft_mem_gb, self.seed, self.mem_limit_gb)
        base = {"formulation": self.formulation, "reduce": self.reduce, "method": self.method, "mip_focus": self.mip_focus}
        best = {"obj": math.inf, "design": None, "bound": -math.inf, "first": None, "published": 0, "stop": False}
        try:
            if self.formulation == "radius":
                model, info = build_radius(inst, D, env)
            else:
                model, info = build_classical(inst, D, env, reduce=self.reduce)
        except gp.GurobiError as err:
            env.dispose()
            if not is_memory_error(err):
                raise
            return {"best_objective": None, "global_lb": None, "termination_reason": "memory_limit_during_build",
                    "design": None, "error": str(err), "build_seconds": time.perf_counter() - t0,
                    "total_seconds": time.perf_counter() - t0, **base}
        except MemoryError as err:
            env.dispose()
            return {"best_objective": None, "global_lb": None, "termination_reason": "memory_limit_during_build",
                    "design": None, "error": f"MemoryError: {err}", "build_seconds": time.perf_counter() - t0,
                    "total_seconds": time.perf_counter() - t0, **base}
        build = time.perf_counter() - t0
        yl = info["y"].tolist()
        seen: set[tuple[int, ...]] = set()

        def offer(values, obj: float) -> None:
            if time.perf_counter() >= deadline or not obj < best["obj"] - 1e-9 * max(1.0, abs(obj)):
                return
            design = open_list(np.asarray(values))
            if len(design) != inst.k:
                return
            best.update(obj=obj, design=design)
            key = tuple(design)
            if key not in seen and time.perf_counter() < deadline:
                seen.add(key)
                best["published"] += 1
                if best["first"] is None:
                    best["first"] = time.perf_counter() - t0
                self.on_incumbent(design, stage="mip")

        def bound(value: float) -> None:
            if math.isfinite(value) and abs(value) < 1e90 and time.perf_counter() < deadline and value > best["bound"]:
                best["bound"] = value
                self.on_bound(value)

        errors: list[BaseException] = []

        def callback(cb, where):
            try:
                if time.perf_counter() >= deadline:
                    best["stop"] = True
                    cb.terminate()
                    return
                if where == GRB.Callback.MIPSOL:
                    offer(cb.cbGetSolution(yl), float(cb.cbGet(GRB.Callback.MIPSOL_OBJ)))
                elif where == GRB.Callback.MIP:
                    bound(float(cb.cbGet(GRB.Callback.MIP_OBJBND)))
            except BaseException as exc:  # noqa: BLE001 - gurobipy would swallow it
                errors.append(exc)
                cb.terminate()

        reason, status = "deadline_during_build", None
        try:
            if deadline - time.perf_counter() > 0.01:
                model.Params.TimeLimit = deadline - time.perf_counter()
                _set_gap(model, integral)
                if self.method is not None:
                    model.Params.Method = self.method
                if self.mip_focus is not None:
                    model.Params.MIPFocus = self.mip_focus
                model.optimize(callback)
                if errors:
                    raise errors[0]
                status = int(model.Status)
                if model.SolCount:
                    offer(model.getAttr("X", yl), float(model.ObjVal))
                if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.MEM_LIMIT):
                    bound(float(model.ObjBound))
                if status == GRB.OPTIMAL and best["design"] is not None and gap_closed(best["obj"], float(model.ObjBound), integral):
                    reason = "optimal"
                elif status == GRB.OPTIMAL:
                    reason = "optimal_gap_not_closed"
                elif status == GRB.TIME_LIMIT or (status == GRB.INTERRUPTED and best["stop"]):
                    reason = "time_limit"
                else:
                    reason = {GRB.INTERRUPTED: "interrupted", GRB.MEM_LIMIT: "memory_limit"}.get(status, f"status_{status}")
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        out = {"best_objective": best["obj"] if best["design"] is not None else None,
               "global_lb": best["bound"] if best["bound"] > -math.inf else None, "bound_kind": "solver_reported_mip",
               "termination_reason": reason, "gurobi_status": status, "design": best["design"],
               "build_seconds": build, "first_incumbent_seconds": best["first"], "num_vars": info["num_vars"],
               "num_constrs": info["num_constrs"], "node_count": float(model.NodeCount) if status is not None else 0.0,
               "published_designs": best["published"], "total_seconds": time.perf_counter() - t0, **base}
        model.dispose()
        env.dispose()
        return out


class KMCompactLP:
    """Decomposition-free bound control: the LP relaxation of the compact model, formulation 'classical' (reduced)
    or 'radius', in one thread under the memory limit, by barrier (method=2, Crossover off) or dual simplex
    (method=1). Certificates (published through on_certificate as km-cert-v1, within the frozen size limit):
    - a vector v of per-customer y-space duals: classical = the assignment-row duals; radius = D_j^1 + the dual of
      customer j's first cover row (see build_radius); published as the cuts (j, v_j) and (j, min(v_j, delta_j))
      (g_certificate_xl.Checker.dual_vector_certificate), clipped cuts first;
    - from a complete solve (OPTIMAL: Pi), a numerically troubled one (SUBOPTIMAL: Pi), or an incomplete barrier
      solve (time, iteration or memory limit, numerical stop: BarPi of the last iterate, Codex review 46 #1). The
      barrier's Gurobi TimeLimit is time_limit - publish_margin so that the iterate can be published before the
      supervisor's hard kill; an incomplete dual simplex solve exposes no dual vector (its dual-feasible,
      unperturbed iterates' objectives are recorded as solver-reported mid-solve bounds in `simplex_bounds`).
    The solver-reported LP value is published through on_bound only when the solve ends optimal."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, on_certificate: Callable | None = None, soft_mem_gb: float | None = None,
                 seed: int = 0, method: int = 2, crossover: int = 0, reduce: bool = True, formulation: str = "classical",
                 mem_limit_gb: float | None = None, publish_margin: float = 30.0, cert_limit: int = 2_000_000,
                 params: dict | None = None) -> None:
        self.params = dict(params or {})  # extra Gurobi parameters (tests: e.g. BarIterLimit for an incomplete barrier)
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.method, self.crossover, self.reduce, self.mem_limit_gb = method, crossover, reduce, mem_limit_gb
        self.formulation, self.publish_margin, self.cert_limit = formulation, publish_margin, cert_limit
        self.on_incumbent = on_incumbent or (lambda design, stage=None: None)
        self.on_bound = on_bound or (lambda bound: None)
        self.on_certificate = on_certificate or (lambda cert: None)

    def duals(self, model: gp.Model, info: dict, attr: str) -> np.ndarray:
        if self.formulation == "radius":
            pi = np.asarray(getattr(info["cover"], attr), dtype=float)
            v = info["d1"].copy()
            has = info["has_row"]
            v[has] = info["d1"][has] + pi[info["first_row"][has]]
            return v
        return np.asarray(getattr(info["assign"], attr), dtype=float)

    def publish(self, inst: KMInstance, D: np.ndarray, sc: SortedCosts, v: np.ndarray, tag: str) -> int:
        from src.bd.km.g_certificate_xl import fit_budget, singles, sorted_counts
        if not np.all(np.isfinite(v)):
            v = np.where(np.isfinite(v), v, sc.d1)
        delta = sc.ds[:, -1]
        j = np.r_[np.arange(inst.n), np.arange(inst.n)]
        vv = np.r_[np.minimum(v, delta), v]
        keep = fit_budget(sorted_counts(D, sc), j, vv, self.cert_limit)
        self.on_certificate(singles(j[keep], vv[keep], version=tag))
        return len(keep)

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        inst = load_instance(data_path)
        D = inst.distances()
        sc = SortedCosts(D, inst.k)
        env = make_env(self.threads, self.soft_mem_gb, self.seed, self.mem_limit_gb)
        base = {"method": self.method, "crossover": self.crossover, "reduce": self.reduce, "formulation": self.formulation}
        try:
            if self.formulation == "radius":
                model, info = build_radius(inst, D, env, relax=True, sc=sc)
            else:
                model, info = build_classical(inst, D, env, relax=True, reduce=self.reduce, sc=sc)
        except (gp.GurobiError, MemoryError) as err:
            env.dispose()
            if isinstance(err, gp.GurobiError) and not is_memory_error(err):
                raise
            return {"best_objective": None, "global_lb": None, "termination_reason": "memory_limit_during_build",
                    "error": str(err), "build_seconds": time.perf_counter() - t0, "total_seconds": time.perf_counter() - t0,
                    "certificate_v": None, **base}
        build = time.perf_counter() - t0
        out = {"best_objective": None, "global_lb": None, "termination_reason": "deadline_during_build", "build_seconds": build,
               "num_vars": info["num_vars"], "num_constrs": info["num_constrs"], "certificate_v": None,
               "certificate_seconds": None, "certificate_source": None, **base}
        margin = min(self.publish_margin if self.method == 2 else 5.0, 0.1 * self.time_limit)
        try:
            if deadline - margin - time.perf_counter() > 0.01:
                model.Params.TimeLimit = deadline - margin - time.perf_counter()
                model.Params.Method = self.method
                model.Params.Crossover = self.crossover
                for key, val in self.params.items():
                    model.setParam(key, val)
                spx: list[tuple[float, float]] = []
                out["simplex_bounds"] = spx

                def callback(cb, where):
                    if where == GRB.Callback.SIMPLEX and self.method == 1 and time.perf_counter() < deadline:
                        if (cb.cbGet(GRB.Callback.SPX_ISPERT) == 0 and cb.cbGet(GRB.Callback.SPX_DUALINF) <= 1e-9):
                            val = float(cb.cbGet(GRB.Callback.SPX_OBJVAL))
                            if math.isfinite(val) and (not spx or val > spx[-1][1]):
                                spx.append((round(time.perf_counter() - t0, 3), val))
                                self.on_bound(val)  # solver-reported mid-solve bound (diagnostic, not certified)

                model.optimize(callback)
                st = int(model.Status)
                out["gurobi_status"] = st
                v = y = None
                if st in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                    try:
                        v, y = self.duals(model, info, "Pi"), np.asarray(info["y"].X, dtype=float)
                        out["certificate_source"] = "Pi"
                    except gp.GurobiError:
                        v = None
                if v is None and self.method == 2:
                    try:
                        v, y = self.duals(model, info, "BarPi"), np.asarray(info["y"].BarX, dtype=float)
                        out["certificate_source"] = "BarPi"
                    except (gp.GurobiError, AttributeError):
                        v = None
                if v is not None and time.perf_counter() < deadline:
                    out["certificate_v"] = v.tolist()
                    out["lp_point"] = y.tolist() if y is not None else None
                    out["certificate_cuts"] = self.publish(inst, D, sc, v, out["certificate_source"])
                    out["certificate_seconds"] = time.perf_counter() - t0
                if st == GRB.OPTIMAL and time.perf_counter() < deadline:
                    out["termination_reason"] = "optimal"
                    out["lp_value"] = float(model.ObjVal)
                    out["global_lb"] = float(model.ObjVal)
                    self.on_bound(float(model.ObjVal))
                else:
                    out["termination_reason"] = {GRB.TIME_LIMIT: "time_limit", GRB.INTERRUPTED: "time_limit",
                                                 GRB.MEM_LIMIT: "memory_limit"}.get(st, f"status_{st}")
                out["barrier_iterations"] = int(model.BarIterCount) if self.method == 2 else None
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            out["termination_reason"] = "memory_limit"
        out["total_seconds"] = time.perf_counter() - t0
        model.dispose()
        env.dispose()
        return out
