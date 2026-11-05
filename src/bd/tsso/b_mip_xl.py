"""TSSO at scale: exact recourse, design evaluation, the compact extensive form, its MIP and the LP-relaxation
control (scale-up study).

Solutions published by every method are designs y in {0, 1}^(I x T). The value of a design is
F.y + sum_(i,s) p_s Q_is(y_i) (a_data_xl), computed exactly with `ProductRecourse`; the runner/probe evaluates
publications after the clock stops, and the independent verifier (e_verifier_xl) recomputes values with its own
dynamic program.

ProductRecourse: one persistent LP per product holding the recourse blocks of all (or a subset of) its scenarios:
per scenario 4T columns [xr, xo, inv, back] and the T balance rows; the capacity xr_t <= K_it y_t is a column bound,
so a right-hand side never changes between designs and Gurobi re-solves by warm-started dual simplex (the first,
cold solve uses primal simplex without presolve, about 3x faster on these block LPs). Its duals give
the optimality cut of every scenario: pi = Pi(balance rows) and mu_t = max(0, pi_t - c_reg_t) (the bound dual of
xr_t <= K_it y_t, i.e. the cheapest completion); the cut theta_is >= pi.d_is - sum_t mu_t K_it y_t is tight at the
point where the LP was solved and valid everywhere on [0, 1]^T (exactly checked later by g_certificate_xl).

Extensive form (`build_extensive_form`): the paper's monolithic model (a_gold_monolithic) built from sparse
matrices: y (I x T), and per (i, s, t) the four recourse columns, the balance rows and the rows xr - K y <= 0.
`relax=True` gives its LP relaxation (the decomposition-free bound control, `TSSOLPControl`).
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from src.bd.tsso.a_data_xl import TSSOXLInstance, design_key, load_instance

MIP_GAP = 1e-6


def make_env(threads: int = 1, soft_mem_gb: float | None = None, seed: int = 0, mem_limit_gb: float | None = None) -> gp.Env:
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", threads)
    env.setParam("Seed", seed)
    if soft_mem_gb:
        env.setParam("SoftMemLimit", soft_mem_gb)
    if mem_limit_gb:
        env.setParam("MemLimit", mem_limit_gb)
    env.start()
    return env


def is_memory_error(err: gp.GurobiError) -> bool:
    return err.errno == GRB.Error.OUT_OF_MEMORY or "memory" in str(err).lower()


def as_design(inst: TSSOXLInstance, y: Any) -> np.ndarray:
    """A design as a boolean (I, T) array (from nested lists, a flat list of open (i, t) pairs or an array)."""
    arr = np.asarray(y)
    if arr.shape == (inst.I, inst.T):
        return arr > 0.5
    raise ValueError(f"design must have shape ({inst.I}, {inst.T})")


def chain_matrices(n_blocks: int, T: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Balance-row coefficients of inv and back for n_blocks consecutive chains of T periods:
    -inv_t + inv_t-1 and +back_t - back_t-1 (rows and columns in block-major order)."""
    N = n_blocks * T
    k = np.arange(N)
    prev = k[(k % T) != 0]
    rows = np.concatenate([k, prev])
    cols = np.concatenate([k, prev - 1])
    A_inv = sp.csr_matrix((np.concatenate([-np.ones(N), np.ones(len(prev))]), (rows, cols)), shape=(N, N))
    A_back = sp.csr_matrix((np.concatenate([np.ones(N), -np.ones(len(prev))]), (rows, cols)), shape=(N, N))
    return A_inv, A_back


# ---------------------------------------------------------------- recourse

class ProductRecourse:
    """Recourse LP of product i over the scenarios `scen` (default all), see the module docstring."""

    def __init__(self, inst: TSSOXLInstance, i: int, env: gp.Env, scen: np.ndarray | None = None) -> None:
        self.inst, self.i = inst, i
        T = inst.T
        self.scen = np.arange(inst.S) if scen is None else np.asarray(scen, dtype=int)
        n = len(self.scen)
        self.n = n
        md = gp.Model(env=env)
        md.Params.OutputFlag = 0
        md.Params.Method = 0  # the first (cold) solve by primal simplex, later ones by warm-started dual simplex
        md.Params.Presolve = 0
        md.Params.FeasibilityTol = 1e-9
        md.Params.OptimalityTol = 1e-9
        self.model = md
        tile = lambda a: np.tile(a, n)  # noqa: E731
        self.xr = md.addMVar(n * T, lb=0.0, ub=np.tile(inst.K[i], n), obj=tile(inst.c_reg[i]))
        self.xo = md.addMVar(n * T, lb=0.0, obj=tile(inst.c_ot[i]))
        self.inv = md.addMVar(n * T, lb=0.0, obj=tile(inst.h[i]))
        self.back = md.addMVar(n * T, lb=0.0, obj=tile(inst.b[i]))
        A_inv, A_back = chain_matrices(n, T)
        d = inst.demand[i][:, self.scen].T.ravel()  # (n * T) in (scenario, t) order
        self.bal = md.addConstr(self.xr + self.xo + A_inv @ self.inv + A_back @ self.back == d)
        md.update()
        self.cost = np.stack([inst.c_reg[i], inst.c_ot[i], inst.h[i], inst.b[i]])  # (4, T)
        self.d = inst.demand[i][:, self.scen].T  # (n, T)
        self._key: bytes | None = np.ones(T).tobytes()
        self.solves = 0
        self.seconds = 0.0

    def solve(self, y: np.ndarray, time_limit: float = 1e9, duals: bool = True) -> dict[str, Any]:
        """The recourse at y in [0, 1]^T: q (n,) per scenario and, with duals, pi and mu (n, T)."""
        t0 = time.perf_counter()
        inst, i, T, n = self.inst, self.i, self.inst.T, self.n
        y = np.clip(np.asarray(y, dtype=float), 0.0, 1.0)
        key = y.tobytes()
        if key != self._key:
            self.xr.UB = np.tile(inst.K[i] * y, n)
            self._key = key
        md = self.model
        md.Params.TimeLimit = max(0.01, time_limit)
        md.optimize()
        if self.solves == 0:
            md.Params.Method = 1
        self.solves += 1
        out: dict[str, Any]
        if md.Status == GRB.OPTIMAL:
            X = np.stack([np.asarray(v.X).reshape(n, T) for v in (self.xr, self.xo, self.inv, self.back)], axis=1)
            q = np.einsum("nkt,kt->n", X, self.cost)
            out = {"status": "optimal", "q": q}
            if duals:
                pi = np.asarray(self.bal.Pi, dtype=float).reshape(n, T)
                out["pi"] = pi
                out["mu"] = np.maximum(pi - inst.c_reg[i][None, :], 0.0)
        elif md.Status == GRB.TIME_LIMIT:
            out = {"status": "time_limit"}
        else:
            out = {"status": f"status_{md.Status}"}
        self.seconds += time.perf_counter() - t0
        return out

    def dispose(self) -> None:
        self.model.dispose()


def cut_coefficients(inst: TSSOXLInstance, i: int, scen: np.ndarray, pi: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Float cuts theta_is >= const - coef . y_i from dual-feasible (pi, mu): const = pi . d_is, coef = mu * K_i."""
    d = inst.demand[i][:, scen].T
    return (pi * d).sum(axis=1), mu * inst.K[i][None, :]


class DesignEvaluator:
    """Exact design values, product by product (the recourse separates by product), with a per-product value cache
    and at most `max_models` recourse LPs alive (least recently used ones are disposed; about 60 KB per scenario
    each, so `model_budget_gb` bounds their memory)."""

    def __init__(self, inst: TSSOXLInstance, env: gp.Env | None = None, max_models: int | None = None,
                 model_budget_gb: float = 4.0) -> None:
        self.inst = inst
        self.own = env is None
        self.env = make_env() if env is None else env
        self.lps: dict[int, ProductRecourse] = {}
        self.cache: dict[tuple[int, bytes], float] = {}
        self.evaluations = 0
        per_model = 64e3 * inst.S * inst.T / 30.0
        self.max_models = max_models or max(2, int(model_budget_gb * 1e9 / per_model))
        self.builds = 0

    def lp(self, i: int) -> ProductRecourse:
        if i in self.lps:
            self.lps[i] = self.lps.pop(i)  # most recently used last
        else:
            while len(self.lps) >= self.max_models:
                old = next(iter(self.lps))
                self.lps.pop(old).dispose()
            self.lps[i] = ProductRecourse(self.inst, i, self.env)
            self.builds += 1
        return self.lps[i]

    def product(self, i: int, y_i: np.ndarray, deadline: float = math.inf, duals: bool = False) -> dict[str, Any] | None:
        """{value: F_i.y_i + p.q, q, (pi, mu)} for a binary y_i; None when out of time."""
        y_i = np.asarray(y_i) > 0.5
        key = (i, design_key(y_i))
        if key in self.cache and not duals:
            return {"value": self.cache[key]}
        left = deadline - time.perf_counter()
        if left <= 0:
            return None
        res = self.lp(i).solve(y_i.astype(float), left, duals=duals)
        if res["status"] != "optimal":
            if res["status"] == "time_limit":
                return None
            raise RuntimeError(f"recourse of product {i}: {res['status']}")
        self.evaluations += 1
        value = float(self.inst.F[i] @ y_i + self.inst.prob @ res["q"])
        self.cache[key] = value
        return dict(res, value=value)

    def value(self, y: np.ndarray, deadline: float = math.inf) -> float | None:
        y = as_design(self.inst, y)
        total = 0.0
        for i in range(self.inst.I):
            res = self.product(i, y[i], deadline)
            if res is None:
                return None
            total += res["value"]
        return total

    def close(self) -> None:
        for lp in self.lps.values():
            lp.dispose()
        self.lps.clear()
        if self.own:
            self.env.dispose()


def design_value(inst: TSSOXLInstance, y: np.ndarray) -> float:
    ev = DesignEvaluator(inst)
    try:
        v = ev.value(y)
    finally:
        ev.close()
    if v is None:
        raise RuntimeError("evaluation timed out")
    return v


def tsso_output(inst: TSSOXLInstance, y: np.ndarray, status: str = "feasible", method: str = "benders_decomposition",
                value: float | None = None) -> dict:
    """Output contract of e_verifier_xl: y (I x T, 0/1) and objective_value (the exact design value)."""
    y = as_design(inst, y)
    return {"status": status, "method": method, "objective_value": design_value(inst, y) if value is None else value,
            "y": y.astype(int).tolist()}


# ---------------------------------------------------------------- extensive form

def build_extensive_form(inst: TSSOXLInstance, env: gp.Env | None = None, relax: bool = False,
                         scen: np.ndarray | None = None) -> tuple[gp.Model, dict[str, Any]]:
    """The paper's monolithic model over all scenarios (or a subset with renormalized probabilities: the SAA
    control). Columns per (i, s, t) in block order k = (i * n + s) * T + t."""
    I, T = inst.I, inst.T
    idx = np.arange(inst.S) if scen is None else np.asarray(scen, dtype=int)
    n = len(idx)
    p = inst.prob[idx] / inst.prob[idx].sum()
    N = I * n * T
    md = gp.Model(env=env) if env is not None else gp.Model()
    y = md.addMVar(I * T, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS if relax else GRB.BINARY, obj=inst.F.ravel())
    w = np.repeat(np.tile(p, I), T)  # weight of each (i, s, t) column

    def per_col(a: np.ndarray) -> np.ndarray:  # (I, T) -> (N,) in (i, s, t) order
        return np.repeat(a[:, None, :], n, axis=1).ravel()

    xr = md.addMVar(N, lb=0.0, obj=w * per_col(inst.c_reg))
    xo = md.addMVar(N, lb=0.0, obj=w * per_col(inst.c_ot))
    inv = md.addMVar(N, lb=0.0, obj=w * per_col(inst.h))
    back = md.addMVar(N, lb=0.0, obj=w * per_col(inst.b))
    A_inv, A_back = chain_matrices(I * n, T)
    d = inst.demand[:, :, idx].transpose(0, 2, 1).ravel()
    bal = md.addConstr(xr + xo + A_inv @ inv + A_back @ back == d)
    k = np.arange(N)
    col = (k // (n * T)) * T + (k % T)
    B = sp.csr_matrix((-per_col(inst.K), (k, col)), shape=(N, I * T))
    cap = md.addConstr(xr + B @ y <= 0.0)
    md.ModelSense = GRB.MINIMIZE
    md.update()
    return md, {"y": y, "bal": bal, "cap": cap, "scen": idx, "p": p, "n": n,
                "num_vars": int(md.NumVars), "num_constrs": int(md.NumConstrs)}


def ef_duals(inst: TSSOXLInstance, info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-subproblem dual vectors from an extensive-form LP solution: the scenario-s rows' duals divided by p_s.
    Returns subproblem IDs (I * S,), pi and mu (I * S, T) (mu = -Pi(cap) / p_s clipped at 0)."""
    I, T, n = inst.I, inst.T, info["n"]
    p = info["p"]
    Pb = np.asarray(info["bal"].Pi, dtype=float).reshape(I, n, T)
    Pc = np.asarray(info["cap"].Pi, dtype=float).reshape(I, n, T)
    pi = Pb / p[None, :, None]
    mu = np.maximum(-Pc / p[None, :, None], 0.0)
    sub = (np.arange(I)[:, None] * inst.S + info["scen"][None, :]).ravel()
    return sub, pi.reshape(I * n, T), mu.reshape(I * n, T)


class TSSOExtensiveMIP:
    """The extensive-form MIP (paper formulation) with the solver interface of the decomposition methods: every new
    MIP incumbent's design is published (deduplicated) strictly before the deadline; bounds are Gurobi's MIP bound
    (solver-reported). Its scored value is the design's exact value, computed after the clock stops."""

    def __init__(self, *, time_limit: float = 900.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 method: int | None = None, mip_focus: int | None = None, mem_limit_gb: float | None = 14.0, **_: Any) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.method, self.mip_focus, self.mem_limit_gb = method, mip_focus, mem_limit_gb
        self.on_incumbent = on_incumbent or (lambda y, stage=None: None)
        self.on_bound = on_bound or (lambda bound: None)

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        inst = load_instance(data_path)
        env = make_env(self.threads, self.soft_mem_gb, self.seed, self.mem_limit_gb)
        best = {"obj": math.inf, "y": None, "bound": -math.inf, "first": None, "published": 0, "deadline_stop": False}
        base = {"method_param": self.method, "mip_focus": self.mip_focus, "I": inst.I, "T": inst.T, "S": inst.S}
        try:
            model, info = build_extensive_form(inst, env)
        except gp.GurobiError as err:
            env.dispose()
            if not is_memory_error(err):
                raise
            return {"best_objective": None, "global_lb": None, "termination_reason": "memory_limit_during_build",
                    "design": None, "error": str(err), "build_seconds": time.perf_counter() - t0,
                    "total_seconds": time.perf_counter() - t0, **base}
        build = time.perf_counter() - t0
        yl = info["y"].tolist()
        seen: set[bytes] = set()

        def offer(values, obj: float) -> None:
            if time.perf_counter() >= deadline or not obj < best["obj"] - 1e-9 * max(1.0, abs(obj)):
                return
            y = np.asarray(values).reshape(inst.I, inst.T) > 0.5
            best.update(obj=obj, y=y)
            key = design_key(y)
            if key not in seen and time.perf_counter() < deadline:
                seen.add(key)
                best["published"] += 1
                if best["first"] is None:
                    best["first"] = time.perf_counter() - t0
                self.on_incumbent(y, stage="mip")

        def bound(value: float) -> None:
            if math.isfinite(value) and abs(value) < 1e90 and time.perf_counter() < deadline and value > best["bound"]:
                best["bound"] = value
                self.on_bound(value)

        errors: list[BaseException] = []

        def callback(cb, where):
            try:
                if time.perf_counter() >= deadline:
                    best["deadline_stop"] = True
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
                model.Params.MIPGap = MIP_GAP
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
                if status == GRB.OPTIMAL:
                    reason = "optimal"
                elif status == GRB.TIME_LIMIT or (status == GRB.INTERRUPTED and best["deadline_stop"]):
                    reason = "time_limit"
                else:
                    reason = {GRB.INTERRUPTED: "interrupted", GRB.MEM_LIMIT: "memory_limit"}.get(status, f"status_{status}")
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        out = {"best_objective": best["obj"] if best["y"] is not None else None,
               "global_lb": best["bound"] if best["bound"] > -math.inf else None,
               "termination_reason": reason, "gurobi_status": status,
               "budget_exhausted": time.perf_counter() - t0 >= self.time_limit - 1.0,
               "design": best["y"].astype(int).tolist() if best["y"] is not None else None,
               "build_seconds": build, "first_incumbent_seconds": best["first"], "num_vars": info["num_vars"],
               "num_constrs": info["num_constrs"], "node_count": float(model.NodeCount) if status is not None else 0.0,
               "published_designs": best["published"], "total_seconds": time.perf_counter() - t0, **base}
        model.dispose()
        env.dispose()
        return out


def product_instance(inst: TSSOXLInstance, i: int) -> TSSOXLInstance:
    """The one-product instance of product i (the problem separates by product)."""
    sl = slice(i, i + 1)
    return TSSOXLInstance(F=inst.F[sl], K=inst.K[sl], c_reg=inst.c_reg[sl], c_ot=inst.c_ot[sl], h=inst.h[sl], b=inst.b[sl],
                          demand=inst.demand[sl], prob=inst.prob, I0=inst.I0[sl], B0=inst.B0[sl], name=f"{inst.name}_p{i}")


class TSSOLPControl:
    """Decomposition-free bound control: the LP relaxation of the compact extensive form, one thread, within the
    time limit and memory, by barrier (method=2, no crossover) or dual simplex (method=1). Its value is
    solver-reported; at completion its duals are exported as one cut per subproblem (`cutlog`) so g_certificate_xl
    certifies the same bound exactly, and its y is kept (`y_lp`) for the exact upper end of LP*. During a dual
    simplex solve the objective of every dual-feasible, unperturbed iterate is recorded (`simplex_bounds`,
    solver-reported valid bounds of an incomplete solve); barrier iterates give no valid bound (recorded only as a
    diagnostic, `barrier_trace`). No primal design.
    split=True: the same LP solved product by product (one extensive-form LP per product, the problem separates by
    product): a decomposition-free control that exploits only the block structure, not Benders. Its bound is the sum
    of the finished products' LP values (the others count 0, valid since all costs are nonnegative), and every
    finished product's duals are exported at once."""

    def __init__(self, *, time_limit: float = 900.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 mem_limit_gb: float | None = 14.0, method: int = 2, crossover: int = 0, split: bool = False, **_: Any) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.mem_limit_gb, self.method, self.crossover, self.split = mem_limit_gb, method, crossover, split
        self.on_bound = on_bound or (lambda bound: None)
        self.cutlog = None

    def run(self, data_path: str) -> dict[str, Any]:
        from src.bd.tsso.g_certificate_xl import CutLog
        t0 = self.t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        inst = load_instance(data_path)
        self.cutlog = CutLog(inst)
        env = make_env(self.threads, self.soft_mem_gb, self.seed, self.mem_limit_gb)
        out: dict[str, Any] = {"best_objective": None, "global_lb": None, "I": inst.I, "T": inst.T, "S": inst.S,
                               "lp_method": self.method, "y_lp": None}
        spx: list[tuple[float, float]] = []
        bar: list[tuple[float, float, float, float]] = []
        best = {"spx": -math.inf}

        def callback(cb, where):
            now = time.perf_counter()
            if now >= deadline:
                cb.terminate()
                return
            if where == GRB.Callback.SIMPLEX:
                if cb.cbGet(GRB.Callback.SPX_DUALINF) == 0.0 and cb.cbGet(GRB.Callback.SPX_ISPERT) == 0:
                    v = float(cb.cbGet(GRB.Callback.SPX_OBJVAL))
                    if math.isfinite(v) and v > best["spx"]:
                        best["spx"] = v
                        spx.append((round(now - t0, 3), v))
                        self.on_bound(v)
            elif where == GRB.Callback.BARRIER:
                bar.append((round(now - t0, 3), float(cb.cbGet(GRB.Callback.BARRIER_PRIMOBJ)),
                            float(cb.cbGet(GRB.Callback.BARRIER_DUALOBJ)), float(cb.cbGet(GRB.Callback.BARRIER_DUALINF))))

        model = None
        if self.split:
            try:
                return self._run_split(inst, env, t0, deadline, out)
            finally:
                env.dispose()
        try:
            model, info = build_extensive_form(inst, env, relax=True)
            out.update(build_seconds=time.perf_counter() - t0, num_vars=info["num_vars"], num_constrs=info["num_constrs"])
            left = deadline - time.perf_counter()
            if left <= 0.01:
                out["termination_reason"] = "deadline_during_build"
            else:
                model.Params.Method = self.method
                model.Params.Crossover = self.crossover
                model.Params.TimeLimit = left
                model.optimize(callback)
                st = int(model.Status)
                out["gurobi_status"] = st
                out["iterations"] = float(model.IterCount)
                out["barrier_iterations"] = int(model.BarIterCount)
                if st == GRB.OPTIMAL and time.perf_counter() < deadline:
                    out["global_lb"] = float(model.ObjVal)
                    out["lp_seconds"] = time.perf_counter() - t0
                    self.on_bound(float(model.ObjVal))
                    sub, pi, mu = ef_duals(inst, info)
                    self.cutlog.add_multi(sub, pi, mu, time.perf_counter() - t0)
                    out["y_lp"] = np.clip(np.asarray(info["y"].X), 0.0, 1.0).reshape(inst.I, inst.T).tolist()
                    out["termination_reason"] = "optimal"
                else:
                    out["termination_reason"] = "time_limit" if st in (GRB.TIME_LIMIT, GRB.INTERRUPTED) else f"status_{st}"
                    if best["spx"] > -math.inf:
                        out["global_lb"] = best["spx"]
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            out["termination_reason"] = "memory_limit"
            out["error"] = str(err)
        finally:
            if model is not None:
                model.dispose()
            env.dispose()
        out["simplex_bounds"] = spx[-50:]
        out["barrier_trace"] = bar[-50:]
        out["total_seconds"] = time.perf_counter() - t0
        return out

    def _run_split(self, inst: TSSOXLInstance, env: gp.Env, t0: float, deadline: float, out: dict[str, Any]) -> dict[str, Any]:
        total, y = 0.0, np.zeros((inst.I, inst.T))
        done = 0
        out["termination_reason"] = "time_limit"
        try:
            for i in range(inst.I):
                left = deadline - time.perf_counter()
                if left <= 0.01:
                    break
                sub = product_instance(inst, i)
                model, info = build_extensive_form(sub, env, relax=True)
                try:
                    model.Params.Method = self.method
                    model.Params.Crossover = self.crossover
                    model.Params.TimeLimit = max(0.01, deadline - time.perf_counter())
                    model.optimize()
                    if model.Status != GRB.OPTIMAL or time.perf_counter() >= deadline:
                        break
                    total += float(model.ObjVal)
                    s_ids, pi, mu = ef_duals(sub, info)
                    self.cutlog.add_multi(i * inst.S + s_ids, pi, mu, time.perf_counter() - t0)
                    y[i] = np.clip(np.asarray(info["y"].X), 0.0, 1.0)
                    done += 1
                    self.on_bound(total)
                finally:
                    model.dispose()
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            out["termination_reason"] = "memory_limit"
            out["error"] = str(err)
        out["products_done"] = done
        out["global_lb"] = total if done else None
        if done == inst.I:
            out["termination_reason"] = "optimal"
            out["lp_seconds"] = time.perf_counter() - t0
            out["y_lp"] = y.tolist()
        out["total_seconds"] = time.perf_counter() - t0
        return out
