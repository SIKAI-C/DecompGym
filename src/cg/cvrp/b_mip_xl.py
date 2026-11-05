"""CVRP route evaluator and compact MIP (scale-up study).

Evaluator. `route_cost(inst, routes)` takes routes as lists of customer numbers 1..n (a_data's numbering, the order of
visits, the depot implicit at both ends), checks that every customer is visited exactly once, that every route is
nonempty and within the vehicle capacity, and returns the total rounded distance (TSPLIB nint), or None. It reproduces
the CVRPLIB best-known costs of the bundled .sol files exactly (tests/test_scaleup_cvrp.py).

Compact MIP (default formulation "flow"): the directed two-index vehicle-flow model with a single-commodity load
flow (Gavish and Graves; the formulation of src/bd/clrp/b_mip_xl.py on one depot), strengthened by rounded capacity
inequalities (RCC) separated as Gurobi user cuts at the root:
    min sum_a c_a x_a
    x(delta+(i)) = x(delta-(i)) = 1 and f(delta-(i)) - f(delta+(i)) = q_i for every customer i;  x(delta+(0)) >= K;
    q_v x_uv <= f_uv <= (Q - q_u) x_uv (Q x_0v on depot arcs; f_u0 = 0);  x binary
(arcs between customers whose joint demand exceeds Q are left out; K = ceil(q(N) / Q)). The model is complete, so
Gurobi's own heuristics produce feasible routes; the root cuts x(A(S)) <= |S| - k(S) or x(delta+(S)) >= k(S)
(whichever is shorter; k(S) = max(1, ceil(q(S) / Q))) close most of the gap left by the flow relaxation, whose LP bound
only implies the fractional capacity inequalities.
Alternative formulation "two_index": the undirected two-index model x(delta(i)) = 2, x(delta(0)) >= 2K, x_ij binary,
x_0j in {0, 1, 2}, with all RCCs x(delta(S)) >= 2 k(S) lazy (every integer candidate is split into the components of
its customer edges; a component detached from the depot or over capacity gets its violated RCC) and separated as user
cuts at every node (Laporte and Nobert; Lysgaard, Letchford and Eglese 2004). Its bound is stronger, but Gurobi finds
almost no incumbents under lazy RCCs (local 40 s probe, X-n101-k25: two-index +226% (only the trivial solution) and
bound -6.5%; flow +6.6% and bound -4.4% versus the best-known 27 591), so the flow model is the default.
Fractional RCC separation (both models): support-graph components, then greedy set growth from every customer
(src.bd.clrp's separate_rcc) on the symmetric edge values. Every integer solution is decoded into routes and
re-checked with route_cost before it is published.
Size ("flow"): about 2 n (n + 1) variables (x and f) and as many linking rows (45 000 variables at X-n153-k22).
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.cg.cvrp.a_data import CVRPInstance, load_vrp


# ---------------------------------------------------------------- evaluator

def normalize_routes(routes: list) -> list[list[int]]:
    return [[int(i) for i in r] for r in routes]


def route_issues(inst: CVRPInstance, routes: list, dist: np.ndarray | None = None) -> tuple[int | None, list[str]]:
    """(total distance, []) for a feasible plan, (None, issues) otherwise."""
    n, Q, q = inst.n, inst.capacity, inst.demand
    issues: list[str] = []
    seen = np.zeros(n + 1, dtype=np.int64)
    total = 0
    d = inst.distances() if dist is None else dist
    for k, r in enumerate(routes):
        if not isinstance(r, (list, tuple)) or not r:
            issues.append(f"route {k}: empty or not a list")
            continue
        if not all(isinstance(i, (int, np.integer)) and not isinstance(i, bool) and 1 <= i <= n for i in r):
            issues.append(f"route {k}: customer numbers must be integers in 1..{n}")
            continue
        load = int(q[list(r)].sum())
        if load > Q:
            issues.append(f"route {k}: load {load} exceeds capacity {Q}")
        np.add.at(seen, list(r), 1)
        seq = [0] + [int(i) for i in r] + [0]
        total += int(sum(d[a, b] for a, b in zip(seq, seq[1:])))
    bad = np.nonzero(seen[1:] != 1)[0] + 1
    if len(bad):
        issues.append(f"{len(bad)} customers not visited exactly once (e.g. {bad[:5].tolist()})")
    return (None, issues) if issues else (total, [])


def route_cost(inst: CVRPInstance, routes: list, dist: np.ndarray | None = None) -> int | None:
    """Total rounded distance of a feasible plan, or None (see route_issues)."""
    return route_issues(inst, routes, dist)[0]


def cvrp_output(inst: CVRPInstance, routes: list, status: str = "feasible", method: str = "decomposition") -> dict[str, Any]:
    """The output contract checked by e_verifier.verify_solution."""
    routes = normalize_routes(routes)
    return {"status": status, "method": method, "objective_value": route_cost(inst, routes), "routes": routes}


# ---------------------------------------------------------------- compact MIP

def build_cvrp_mip(inst: CVRPInstance, dist: np.ndarray, env: gp.Env | None = None,
                   formulation: str = "flow") -> tuple[gp.Model, dict[str, Any]]:
    """formulation "flow": directed arcs (u, v) with load flows; "two_index": undirected edges (i < j). Customer pairs
    whose joint demand exceeds Q are left out."""
    n, Q, q = inst.n, inst.capacity, inst.demand
    model = gp.Model(env=env) if env else gp.Model()
    if formulation == "flow":
        U, V = np.nonzero(~np.eye(n + 1, dtype=bool))
        keep = (U == 0) | (V == 0) | (q[U] + q[V] <= Q)
        U, V = U[keep], V[keep]
        x = model.addMVar(len(U), vtype=GRB.BINARY, obj=dist[U, V].astype(float))
        fub = np.where(V == 0, 0.0, np.where(U == 0, float(Q), (Q - q[U]).astype(float)))
        f = model.addMVar(len(U), lb=0.0, ub=fub)
        model.addConstr(f <= fub * x)
        model.addConstr(f >= q[V].astype(float) * x)
        xs, fs = x.tolist(), f.tolist()
        out = [[] for _ in range(n + 1)]
        inn = [[] for _ in range(n + 1)]
        for a, (u, v) in enumerate(zip(U.tolist(), V.tolist())):
            out[u].append(a)
            inn[v].append(a)
        for i in range(1, n + 1):
            model.addLConstr(gp.LinExpr([1.0] * len(out[i]), [xs[a] for a in out[i]]), GRB.EQUAL, 1.0)
            model.addLConstr(gp.LinExpr([1.0] * len(inn[i]), [xs[a] for a in inn[i]]), GRB.EQUAL, 1.0)
            model.addLConstr(gp.LinExpr([1.0] * len(inn[i]) + [-1.0] * len(out[i]), [fs[a] for a in inn[i]] + [fs[a] for a in out[i]]),
                             GRB.EQUAL, float(q[i]))
        model.addLConstr(gp.LinExpr([1.0] * len(out[0]), [xs[a] for a in out[0]]), GRB.GREATER_EQUAL, float(inst.min_vehicles))
    elif formulation == "two_index":
        U, V = np.triu_indices(n + 1, 1)
        keep = (U == 0) | (q[U] + q[V] <= Q)
        U, V = U[keep], V[keep]
        x = model.addMVar(len(U), lb=0.0, ub=np.where(U == 0, 2.0, 1.0), vtype=GRB.INTEGER, obj=dist[U, V].astype(float))
        xs = x.tolist()
        inc = [[] for _ in range(n + 1)]
        for a, (u, v) in enumerate(zip(U.tolist(), V.tolist())):
            inc[u].append(a)
            inc[v].append(a)
        for i in range(1, n + 1):
            model.addLConstr(gp.LinExpr([1.0] * len(inc[i]), [xs[a] for a in inc[i]]), GRB.EQUAL, 2.0)
        model.addLConstr(gp.LinExpr([1.0] * len(inc[0]), [xs[a] for a in inc[0]]), GRB.GREATER_EQUAL, 2.0 * inst.min_vehicles)
    else:
        raise ValueError(f"unknown formulation {formulation!r}")
    model.ModelSense = GRB.MINIMIZE
    model.update()
    return model, {"x": xs, "U": U, "V": V, "formulation": formulation, "num_vars": model.NumVars}


def edge_matrix(n: int, U: np.ndarray, V: np.ndarray, xv: np.ndarray) -> np.ndarray:
    """Symmetric (n + 1) x (n + 1) edge values (x_uv + x_vu for directed arcs)."""
    Y = np.zeros((n + 1, n + 1))
    np.add.at(Y, (U, V), xv)
    return Y + Y.T


def components(Y: np.ndarray, eps: float) -> list[list[int]]:
    """Connected components of the customers 1..n over the customer edges with value > eps (isolated customers are
    singleton components)."""
    n = len(Y) - 1
    parent = list(range(n + 1))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    I, J = np.nonzero(np.triu(Y[1:, 1:], 1) > eps)
    for a, b in zip((I + 1).tolist(), (J + 1).tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, list[int]] = {}
    for i in range(1, n + 1):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def integer_check(inst: CVRPInstance, info: dict[str, Any], xv: np.ndarray) -> tuple[list[list[int]] | None, list[np.ndarray]]:
    """For an integral x: (routes, []) if it is a set of feasible routes, else (None, violated customer sets): the
    customer-edge components detached from the depot or over capacity."""
    n, Q, q = inst.n, inst.capacity, inst.demand
    U, V = info["U"], info["V"]
    xr = np.rint(xv)
    Y = edge_matrix(n, U, V, xr)
    bad = []
    comps = components(Y, 0.5)
    for S in comps:
        if Y[0, S].sum() < 0.5 or q[S].sum() > Q:
            m = np.zeros(n + 1, dtype=bool)
            m[S] = True
            bad.append(m)
    if bad:
        return None, bad
    routes = []
    if info["formulation"] == "flow":
        succ = {int(u): int(v) for u, v in zip(U[xr > 0.5], V[xr > 0.5])}
        for v in sorted(v for u, v in zip(U[xr > 0.5], V[xr > 0.5]) if u == 0):
            r = []
            while v != 0 and len(r) <= n:
                r.append(int(v))
                v = succ[v]
            routes.append(r)
        return routes, []
    nb: dict[int, list[int]] = {}
    for a, b in zip(*np.nonzero(np.triu(Y[1:, 1:], 1) > 0.5)):
        nb.setdefault(int(a) + 1, []).append(int(b) + 1)
        nb.setdefault(int(b) + 1, []).append(int(a) + 1)
    for S in comps:
        if len(S) == 1:
            routes.append([S[0]])
            continue
        r, prev = [next(i for i in S if Y[0, i] > 0.5)], 0
        while True:
            nxt = [j for j in nb.get(r[-1], []) if j != prev]
            if not nxt:
                break
            prev = r[-1]
            r.append(nxt[0])
        routes.append(r)
    return routes, []


def separate_fractional(inst: CVRPInstance, Y: np.ndarray, max_cuts: int = 50, max_size: int = 30,
                        min_viol: float = 0.05) -> list[np.ndarray]:
    """RCCs violated by fractional edge values Y: support-graph components, then greedy growth from every customer
    (src.bd.clrp's separate_rcc). Returns customer-set masks over 0..n, most violated first."""
    from src.bd.clrp.d_ref_cg_xl import separate_rcc
    n, Q, q = inst.n, inst.capacity, inst.demand
    found: dict[bytes, tuple[float, np.ndarray]] = {}
    deg = Y[1:].sum(axis=1)
    for S in components(Y, 1e-6):
        m = np.zeros(n + 1, dtype=bool)
        m[S] = True
        cross = float(deg[np.array(S) - 1].sum() - Y[np.ix_(S, S)].sum())
        viol = 2.0 * max(1, math.ceil(q[S].sum() / Q - 1e-9)) - cross
        if viol > min_viol:
            found[m.tobytes()] = (viol, m)
    for viol, U in separate_rcc(Y[1:, 1:], deg, q[1:], Q, max_size=min(n, max_size), min_viol=min_viol):
        m = np.concatenate([[False], U])
        found.setdefault(m.tobytes(), (viol, m))
    return [m for _, m in sorted(found.values(), key=lambda t: -t[0])[:max_cuts]]


def rcc_expr(inst: CVRPInstance, info: dict[str, Any], S: np.ndarray) -> tuple[gp.LinExpr, str, float]:
    """The RCC of customer set S (bool mask over 0..n) in its shorter equivalent form: two-index x(E(S)) <= |S| - k or
    x(delta(S)) >= 2k; directed x(A(S)) <= |S| - k or x(delta+(S)) >= k (k = max(1, ceil(q(S) / Q)))."""
    U, V, xs = info["U"], info["V"], info["x"]
    k = max(1, math.ceil(int(inst.demand[S].sum()) / inst.capacity - 1e-9))  # >= 1: every set meets the depot
    inner = np.nonzero(S[U] & S[V])[0]
    if info["formulation"] == "flow":
        cut, rhs = np.nonzero(S[U] & ~S[V])[0], float(k)
    else:
        cut, rhs = np.nonzero(S[U] != S[V])[0], float(2 * k)
    if len(inner) <= len(cut):
        return gp.LinExpr([1.0] * len(inner), [xs[a] for a in inner]), GRB.LESS_EQUAL, float(S.sum() - k)
    return gp.LinExpr([1.0] * len(cut), [xs[a] for a in cut]), GRB.GREATER_EQUAL, rhs


class CVRPCompactMIP:
    """The compact MIP with the solver interface of the decomposition references (incumbents are decoded and checked
    with route_cost; bounds are Gurobi's best bounds, rounded up: distances are integral)."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 formulation: str = "flow", node_cuts: bool | None = None, start: list[list[int]] | None = None) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.formulation = formulation
        # RCC user cuts at every node (default for the two-index model, whose bound relies on them) or at the root only
        # (default for the flow model, which is complete without them).
        self.node_cuts = (formulation == "two_index") if node_cuts is None else node_cuts
        self.start = start  # optional MIP start (routes); none by default
        self.on_incumbent = on_incumbent or (lambda routes: None)
        self.on_bound = on_bound or (lambda bound: None)

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        inst = load_vrp(data_path)
        dist = inst.distances()
        n = inst.n
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", self.threads)
        env.setParam("Seed", self.seed)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()
        model, info = build_cvrp_mip(inst, dist, env, self.formulation)
        U, V, xs = info["U"], info["V"], info["x"]
        # Lazy RCCs: always for the two-index model; for the flow model only if some customer has zero demand (the
        # load flow then no longer excludes subtours through such customers).
        lazy = self.formulation == "two_index" or bool((inst.demand[1:] == 0).any())
        build = time.perf_counter() - t0
        best: dict[str, Any] = {"obj": math.inf, "routes": None, "bound": -math.inf, "lazy": 0, "cuts": 0, "first": None}

        def offer(routes: list[list[int]]) -> None:
            obj = route_cost(inst, routes, dist)
            if obj is not None and obj < best["obj"]:
                best.update(obj=obj, routes=routes)
                if best["first"] is None:
                    best["first"] = time.perf_counter() - t0
                self.on_incumbent(routes)

        def bound(value: float) -> None:
            if math.isfinite(value) and value > -1e90:
                b = float(math.ceil(value - 1e-6 * max(1.0, abs(value))))
                if best["obj"] < math.inf:
                    b = min(b, float(best["obj"]))
                if b > best["bound"]:
                    best["bound"] = b
                    self.on_bound(b)

        def callback(m, where):
            if time.perf_counter() > deadline:
                m.terminate()  # Gurobi's TimeLimit can overshoot on large models
                return
            if where == GRB.Callback.MIPSOL:
                routes, bad = integer_check(inst, info, np.array(m.cbGetSolution(xs)))
                for S in bad if lazy else ():
                    m.cbLazy(*rcc_expr(inst, info, S))
                    best["lazy"] += 1
                if routes is not None:
                    offer(routes)
            elif where == GRB.Callback.MIPNODE:
                if m.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                    return
                if not self.node_cuts and m.cbGet(GRB.Callback.MIPNODE_NODCNT) > 0:
                    return
                Y = edge_matrix(n, U, V, np.array(m.cbGetNodeRel(xs)))
                for S in separate_fractional(inst, Y):
                    m.cbCut(*rcc_expr(inst, info, S))
                    best["cuts"] += 1
            elif where == GRB.Callback.MIP:
                bound(m.cbGet(GRB.Callback.MIP_OBJBND))

        if self.start:
            arc = {(int(u), int(v)): a for a, (u, v) in enumerate(zip(U.tolist(), V.tolist()))}
            val = np.zeros(len(U))
            for r in self.start:
                seq = [0] + list(r) + [0]
                for a, b in zip(seq, seq[1:]):
                    val[arc[(a, b)] if self.formulation == "flow" else arc[(min(a, b), max(a, b))]] += 1
            model.setAttr("Start", xs, val.tolist())
        reason = "time_limit"
        if deadline - time.perf_counter() > 0.01:
            model.Params.TimeLimit = deadline - time.perf_counter()
            model.Params.LazyConstraints = int(lazy)
            model.Params.PreCrush = 1
            model.optimize(callback)
            if model.SolCount:
                routes, _ = integer_check(inst, info, np.array(model.getAttr("X", xs)))
                if routes is not None:
                    offer(routes)
            if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.MEM_LIMIT):
                bound(float(model.ObjBound))
            reason = {GRB.OPTIMAL: "optimal", GRB.MEM_LIMIT: "memory_limit"}.get(model.Status, "time_limit")
        out = {"best_objective": best["obj"] if best["routes"] else None,
               "global_lb": best["bound"] if best["bound"] > -math.inf else None, "termination_reason": reason,
               "routes": best["routes"], "formulation": self.formulation, "build_seconds": build,
               "num_vars": info["num_vars"], "num_constrs": model.NumConstrs, "lazy_cuts": best["lazy"],
               "user_cuts": best["cuts"], "first_incumbent_seconds": best["first"], "node_count": float(model.NodeCount),
               "total_seconds": time.perf_counter() - t0}
        model.dispose()
        env.dispose()
        return out
