"""SCFL extensive-form MIP, the exact recourse evaluator and the solution contract (scale-up study).

Solutions (published by every method, stored by the runner) are designs: the sorted list of open facilities. The
value of a design y is f.y + sum_s p_s Q_s(y) with the exact recourse Q_s (one transportation LP per scenario);
a design violating (15d) is infeasible (None). Every method publishes designs only; evaluating one costs S LPs, so
the runner evaluates publications after the clock stops (deferred evaluation, as for MCND), and the independent
verifier (e_verifier) recomputes the value with its own min-cost-flow code.

Recourse LP (`RecourseLP`, one persistent model reused for every scenario and design, warm-started by dual simplex
after right-hand-side and bound changes):
    Q_s(y) = min sum c_ij x_ij + sum M_j z_j
             s.t. sum_i x_ij + z_j = d_sj   (v_j),   sum_j x_ij <= u_i y_i   (-alpha_i <= 0),
                  0 <= x_ij <= d_sj y_i  (strong linking, only at fractional y; redundant at binary y),
                  z_j >= 0 only with penalty=True (unmet demand at cost M_j).
With M_j larger than any dual value of the transportation problem (M_j = (2 min(I, J) + 1) max_ij c_ij + 1) the
penalized LP equals Q_s at every feasible binary design and is a relaxation everywhere, so its cuts are always valid.
Every dual (v, alpha) with alpha >= 0 (v <= M with the penalty) and the cheapest completion
beta_ij = (v_j - alpha_i - c_ij)^+ gives the optimality cut
    theta_s >= sum_j d_sj v_j - sum_i (u_i alpha_i + sum_j d_sj beta_ij) y_i,
tight at the point where the duals are optimal (the cut algebra lives in c_naive_bd).

Extensive form (`build_extensive_form`, Bodur et al. 2017 eq. (15) with strong linking):
    min f.y + sum_s p_s sum_ij c_ij x_sij
    s.t. sum_i x_sij = d_sj,  sum_j x_sij <= u_i y_i,  x_sij <= d_sj y_i (strong=True),  sum_i u_i y_i >= max_s D_s,
         x >= 0, y binary.
It is exact: for a fixed binary y the remaining LP separates into the scenario recourse LPs (the strong rows are
redundant there), so a design is admitted iff (15d) holds and the optimal objective is the design value (tested
against the verifier on small instances). The strong linking rows are the standard strengthening of the weak model
(15) (whose LP gap is 8-25% on these instances, Bodur et al. Table 1); `strong=False` keeps the weak model for probes.
MIPGap is 1e-6 (relative), so an `optimal` label means proved within 1e-6.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Iterable

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from src.bd.scfl.a_data import SCFLInstance, load_instance

MIP_GAP = 1e-6


# ---------------------------------------------------------------- solution contract

def normalize_design(design: Iterable) -> list[int]:
    return sorted({int(i) for i in design})


def design_vector(inst: SCFLInstance, design: Iterable) -> np.ndarray:
    y = np.zeros(inst.I, dtype=bool)
    idx = normalize_design(design)
    if idx and (idx[0] < 0 or idx[-1] >= inst.I):
        raise ValueError(f"facility index out of range: {idx}")
    y[idx] = True
    return y


def open_list(y: np.ndarray) -> list[int]:
    return [int(i) for i in np.nonzero(np.asarray(y) > 0.5)[0]]


def penalty_costs(inst: SCFLInstance) -> np.ndarray:
    """Unmet-demand costs M_j above every dual value of a transportation problem on this instance."""
    return np.full(inst.J, (2 * min(inst.I, inst.J) + 1) * float(inst.cost.max()) + 1.0)


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


# ---------------------------------------------------------------- recourse LP

class RecourseLP:
    """Persistent scenario LP (see the module docstring). `solve` handles one scenario at a point y (binary or
    fractional); `evaluate` returns the exact recourse of a binary design over all scenarios."""

    def __init__(self, inst: SCFLInstance, env: gp.Env, penalty: bool = False) -> None:
        self.inst = inst
        I, J = inst.I, inst.J
        md = gp.Model(env=env)
        md.Params.OutputFlag = 0
        md.Params.Method = 1  # dual simplex: right-hand sides and bounds change between solves
        md.Params.FeasibilityTol = 1e-9
        md.Params.OptimalityTol = 1e-9
        md.Params.InfUnbdInfo = 1
        self.model = md
        self.x = md.addMVar((I, J), lb=0.0, ub=0.0, name="x")
        self.penalty = penalty
        if penalty:
            self.M = penalty_costs(inst)
            self.z = md.addMVar(J, lb=0.0, name="z")
            self.drows = md.addConstr(self.x.sum(axis=0) + self.z == np.zeros(J), name="d")
            md.setObjective((inst.cost * self.x).sum() + self.M @ self.z, GRB.MINIMIZE)
        else:
            self.M = None
            self.drows = md.addConstr(self.x.sum(axis=0) == np.zeros(J), name="d")
            md.setObjective((inst.cost * self.x).sum(), GRB.MINIMIZE)
        self.crows = md.addConstr(self.x.sum(axis=1) <= np.zeros(I), name="c")
        md.update()
        self._ub_key: bytes | None = None
        self._cap_key: bytes | None = None
        # Scenarios in order of total demand: consecutive solves differ little (fewer dual simplex pivots).
        self.order = np.argsort(inst.total_demand, kind="stable")
        self.solves = 0
        self.seconds = 0.0

    def _set_point(self, y: np.ndarray, d: np.ndarray, vub: bool) -> None:
        cap = self.inst.capacity * y
        key = cap.tobytes()
        if key != self._cap_key:
            self.crows.RHS = cap
            self._cap_key = key
        if vub:
            self.x.UB = y[:, None] * d[None, :]
            self._ub_key = None
        else:
            key = (y > 0.5).tobytes()
            if key != self._ub_key:
                self.x.UB = np.where(y[:, None] > 0.5, GRB.INFINITY, 0.0) * np.ones((1, self.inst.J))
                self._ub_key = key
        self.drows.RHS = d

    def solve(self, s: int, y: np.ndarray, time_limit: float, vub: bool = False, flows: bool = False) -> dict[str, Any]:
        """Scenario s at the point y. vub=True imposes x_ij <= d_sj y_i (needed for strong cuts at fractional y)."""
        t = time.perf_counter()
        y = np.asarray(y, dtype=float)
        self._set_point(y, self.inst.demand[s], vub)
        md = self.model
        md.Params.TimeLimit = max(0.01, time_limit)
        md.optimize()
        st = md.Status
        if st == GRB.INF_OR_UNBD:
            md.Params.DualReductions = 0
            md.optimize()
            md.Params.DualReductions = 1
            st = md.Status
        if st == GRB.OPTIMAL:
            out = {"status": "optimal", "obj": float(md.ObjVal), "v": np.asarray(self.drows.Pi, dtype=float),
                   "alpha": np.maximum(-np.asarray(self.crows.Pi, dtype=float), 0.0)}
            if flows:
                out["flows"] = np.asarray(self.x.X, dtype=float)
        elif st == GRB.INFEASIBLE:
            out = {"status": "infeasible", "v": -np.asarray(self.drows.FarkasDual, dtype=float),
                   "alpha": np.maximum(np.asarray(self.crows.FarkasDual, dtype=float), 0.0)}
        elif st == GRB.TIME_LIMIT:
            out = {"status": "time_limit"}
        else:
            out = {"status": f"status_{st}"}
        self.solves += 1
        self.seconds += time.perf_counter() - t
        return out

    def evaluate(self, y: np.ndarray, deadline: float, duals: bool = False, flows: bool = False,
                 scenarios: np.ndarray | None = None) -> dict[str, Any]:
        """Exact recourse of a binary design over all scenarios (or a subset): q (S,), value f.y + p.q, optionally
        the duals V (S, J), A (S, I) and the flows (per scenario, (I, J)). Status `infeasible` for a design that
        violates (15d) (no LP is solved), `time_limit` when the deadline passes first."""
        inst = self.inst
        y = np.asarray(y, dtype=float) > 0.5
        if not inst.feasible(y):
            return {"status": "infeasible"}
        yf = y.astype(float)
        idx = self.order if scenarios is None else np.asarray(scenarios)
        q = np.zeros(inst.S)
        V = np.zeros((inst.S, inst.J)) if duals else None
        A = np.zeros((inst.S, inst.I)) if duals else None
        X: dict[int, np.ndarray] = {}
        for s in idx:
            left = deadline - time.perf_counter()
            if left <= 0:
                return {"status": "time_limit"}
            res = self.solve(int(s), yf, left, vub=False, flows=flows)
            if res["status"] != "optimal":
                if res["status"] == "infeasible":
                    raise RuntimeError("recourse infeasible at a design satisfying (15d)")
                return {"status": res["status"]}
            if self.penalty and res["obj"] > 0 and np.any(np.asarray(self.z.X) > 1e-7):
                raise RuntimeError("penalty used at a feasible design: M is too small")
            q[s] = res["obj"]
            if duals:
                V[s], A[s] = res["v"], res["alpha"]
            if flows:
                X[int(s)] = res["flows"]
        out = {"status": "optimal", "q": q, "y": y}
        if scenarios is None:
            out["value"] = float(inst.fixed @ yf + inst.prob @ q)
        if duals:
            out["V"], out["A"] = V, A
        if flows:
            out["X"] = X
        return out

    def dispose(self) -> None:
        self.model.dispose()


def clean_flows(inst: SCFLInstance, s: int, X: np.ndarray) -> np.ndarray:
    """LP flows made exact: rounded to integers when the data are integral and the rounding stays feasible."""
    X = np.where(np.abs(X) < 1e-9, 0.0, np.maximum(X, 0.0))
    d = inst.demand[s]
    if np.all(d == np.round(d)) and np.all(inst.capacity == np.round(inst.capacity)):
        R = np.round(X)
        if np.all(R.sum(axis=0) == d) and np.all(R.sum(axis=1) <= inst.capacity + 1e-9):
            return R
    return X


def design_value(inst: SCFLInstance, design: Iterable, env: gp.Env | None = None) -> float | None:
    """Exact value of a design (None if it violates (15d))."""
    y = design_vector(inst, design)
    if not inst.feasible(y):
        return None
    own = env is None
    env = make_env() if own else env
    try:
        lp = RecourseLP(inst, env)
        res = lp.evaluate(y, time.perf_counter() + 1e9)
        lp.dispose()
    finally:
        if own:
            env.dispose()
    if res["status"] != "optimal":
        raise RuntimeError(f"recourse evaluation ended with status {res['status']}")
    return res["value"]


class DesignEvaluator:
    """Cached exact design values (the runner's deferred evaluation and the probes)."""

    def __init__(self, inst: SCFLInstance) -> None:
        self.inst = inst
        self.env = make_env()
        self.lp = RecourseLP(inst, self.env)
        self.cache: dict[tuple[int, ...], float | None] = {}

    def objective(self, design: Iterable) -> float | None:
        key = tuple(normalize_design(design))
        if key not in self.cache:
            y = design_vector(self.inst, key)
            res = self.lp.evaluate(y, time.perf_counter() + 1e9)
            self.cache[key] = res["value"] if res["status"] == "optimal" else None
        return self.cache[key]

    def close(self) -> None:
        self.lp.dispose()
        self.env.dispose()


def scfl_output(inst: SCFLInstance, design: Iterable, status: str = "feasible", method: str = "benders_decomposition",
                flows: bool = False, env: gp.Env | None = None) -> dict:
    """Output contract of e_verifier: y (0/1 per facility), objective_value = exact design value, optionally the
    optimal flows of every scenario as sparse [[s, i, j, q], ...]."""
    y = design_vector(inst, design)
    out = {"status": status, "method": method, "objective_value": None, "y": y.astype(int).tolist()}
    if not inst.feasible(y):
        return out
    own = env is None
    env = make_env() if own else env
    try:
        lp = RecourseLP(inst, env)
        res = lp.evaluate(y, time.perf_counter() + 1e9, flows=flows)
        lp.dispose()
    finally:
        if own:
            env.dispose()
    if res["status"] != "optimal":
        raise RuntimeError(f"recourse evaluation ended with status {res['status']}")
    out["objective_value"] = res["value"]
    if flows:
        rows = []
        for s in range(inst.S):
            Xs = clean_flows(inst, s, res["X"][s])
            for i, j in zip(*np.nonzero(Xs > 0)):
                q = float(Xs[i, j])
                rows.append([s, int(i), int(j), int(q) if q.is_integer() else q])
        out["flows"] = rows
    return out


# ---------------------------------------------------------------- extensive form

def add_scenario_blocks(model: gp.Model, y: gp.MVar, demand: np.ndarray, capacity: np.ndarray, strong: bool = True,
                        obj: np.ndarray | None = None) -> tuple[gp.MVar, int]:
    """Flows x (one 1-D MVar in (s, i, j) order, 0 <= x_sij <= d_sj) for the scenarios of `demand` (S x J) with the
    rows sum_i x_sij = d_sj, sum_j x_sij <= u_i y_i and, if strong, x_sij <= d_sj y_i (only where d_sj > 0).
    Built from sparse matrices without per-element names (millions of rows at S = 1500). Returns x and the number of
    strong-linking rows."""
    S, J = demand.shape
    I = len(capacity)
    N = S * I * J
    ub = np.broadcast_to(demand[:, None, :], (S, I, J)).ravel()
    x = model.addMVar(N, lb=0.0, ub=ub, obj=0.0 if obj is None else obj)
    k = np.arange(N)
    s_of, i_of, j_of = k // (I * J), (k // J) % I, k % J
    ones = np.ones(N)
    A_d = sp.csr_matrix((ones, (s_of * J + j_of, k)), shape=(S * J, N))
    model.addConstr(A_d @ x == demand.ravel())
    A_c = sp.csr_matrix((ones, (s_of * I + i_of, k)), shape=(S * I, N))
    B_c = sp.csr_matrix((np.tile(capacity, S), (np.arange(S * I), np.tile(np.arange(I), S))), shape=(S * I, I))
    model.addConstr(A_c @ x - B_c @ y <= 0.0)
    n_link = 0
    if strong:
        keep = np.nonzero(ub > 0)[0]
        n_link = len(keep)
        rows = np.arange(n_link)
        E = sp.csr_matrix((np.ones(n_link), (rows, keep)), shape=(n_link, N))
        L = sp.csr_matrix((ub[keep], (rows, i_of[keep])), shape=(n_link, I))
        model.addConstr(E @ x - L @ y <= 0.0)
    return x, n_link


def build_extensive_form(inst: SCFLInstance, env: gp.Env | None = None, strong: bool = True,
                         scenarios: np.ndarray | None = None) -> tuple[gp.Model, dict[str, Any]]:
    """The extensive form over all scenarios (or a subset with renormalized probabilities: the SAA control)."""
    idx = np.arange(inst.S) if scenarios is None else np.asarray(scenarios, dtype=int)
    d = inst.demand[idx]
    p = inst.prob[idx] / inst.prob[idx].sum()
    model = gp.Model(env=env) if env else gp.Model()
    y = model.addMVar(inst.I, vtype=GRB.BINARY, obj=inst.fixed, name="y")
    x, n_link = add_scenario_blocks(model, y, d, inst.capacity, strong,
                                    obj=(p[:, None, None] * inst.cost[None, :, :]).ravel())
    model.ModelSense = GRB.MINIMIZE
    model.addConstr(inst.capacity @ y >= float(d.sum(axis=1).max()), name="cover")
    model.update()
    return model, {"y": y, "x": x, "num_vars": int(model.NumVars), "num_constrs": int(model.NumConstrs),
                   "link_rows": n_link, "scenarios": len(idx)}


class SCFLExtensiveMIP:
    """The extensive-form MIP with the solver interface of the decomposition references. Every new MIP incumbent's
    design is published (deduplicated) strictly before the deadline; the scored value of a design is its exact value,
    computed after the clock stops (it is never above the MIP's own objective, whose flows are one feasible
    recourse). Bounds are Gurobi's MIP bound. `strong`, `method` (root LP algorithm) and `mip_focus` are the probed
    configuration knobs."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 strong: bool = True, method: int | None = None, mip_focus: int | None = None,
                 mem_limit_gb: float | None = None) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.strong, self.method, self.mip_focus, self.mem_limit_gb = strong, method, mip_focus, mem_limit_gb
        self.on_incumbent = on_incumbent or (lambda design: None)  # receives the open-facility list
        self.on_bound = on_bound or (lambda bound: None)

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        inst = load_instance(data_path)
        env = make_env(self.threads, self.soft_mem_gb, self.seed, self.mem_limit_gb)
        best = {"mip_obj": math.inf, "design": None, "bound": -math.inf, "first": None, "published": 0,
                "deadline_stop": False}
        seen: set[tuple[int, ...]] = set()
        base = {"strong": self.strong, "method": self.method, "mip_focus": self.mip_focus, "scenarios": inst.S}
        try:
            model, info = build_extensive_form(inst, env, strong=self.strong)
        except gp.GurobiError as err:
            env.dispose()
            if not is_memory_error(err):
                raise
            return {"best_objective": None, "global_lb": None, "termination_reason": "memory_limit_during_build",
                    "gurobi_status": None, "budget_exhausted": False, "design": None, "error": str(err),
                    "build_seconds": time.perf_counter() - t0, "total_seconds": time.perf_counter() - t0, **base}
        build = time.perf_counter() - t0
        yl = info["y"].tolist()

        def offer(values, obj: float) -> None:
            if time.perf_counter() >= deadline or not obj < best["mip_obj"] - 1e-9 * max(1.0, abs(obj)):
                return
            design = open_list(np.asarray(values))
            best.update(mip_obj=obj, design=design)
            key = tuple(design)
            if key not in seen and time.perf_counter() < deadline:
                seen.add(key)
                best["published"] += 1
                if best["first"] is None:
                    best["first"] = time.perf_counter() - t0
                self.on_incumbent(design)

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
                # The actual Gurobi status is kept; only TIME_LIMIT or our own deadline stop is a timeout.
                if status == GRB.OPTIMAL:
                    reason = "optimal"
                elif status == GRB.TIME_LIMIT or (status == GRB.INTERRUPTED and best["deadline_stop"]):
                    reason = "time_limit"
                else:
                    reason = {GRB.INTERRUPTED: "interrupted", GRB.MEM_LIMIT: "memory_limit",
                              GRB.INFEASIBLE: "infeasible"}.get(status, f"status_{status}")
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        out = {"best_objective": best["mip_obj"] if best["design"] is not None else None,
               "global_lb": best["bound"] if best["bound"] > -math.inf else None,
               "termination_reason": reason, "gurobi_status": status, "deadline_callback_stop": best["deadline_stop"],
               "budget_exhausted": time.perf_counter() - t0 >= self.time_limit - 1.0, "design": best["design"],
               "build_seconds": build, "first_incumbent_seconds": best["first"], "num_vars": info["num_vars"],
               "num_constrs": info["num_constrs"], "link_rows": info["link_rows"],
               "node_count": float(model.NodeCount) if status is not None else 0.0,
               "published_designs": best["published"], "total_seconds": time.perf_counter() - t0, **base}
        model.dispose()
        env.dispose()
        return out
