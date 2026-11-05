"""Vertex-coloring evaluator and compact MIP (scale-up study).

Evaluator. `color_count(inst, coloring)` takes a list of n integer labels (vertex v of the file is entry v - 1),
checks that adjacent vertices have different labels and returns the number of distinct labels, or None.

Compact MIP: the assignment model with symmetry breaking and clique strengthening (Mendez-Diaz and Zabala 2006;
"ASS-S" in Jabrayilov and Mutzel 2018), on H = the DSATUR number of colors (the shared basic initializer, which is
also its MIP start and its first published incumbent):
    min sum_c w_c
    sum_c x_vc = 1                                  every vertex v
    sum_{v in K} x_vc <= w_c                         every clique K of an edge clique cover (heuristic.edge_clique_cover)
                                                     and every color c (x_vc <= w_c for isolated vertices)
    sum_v x_vc >= w_c;  w_c >= w_{c+1}               used colors are nonempty and come first
    x_{q_i, i} = 1, w_i = 1                           the vertices q_0, q_1, ... of a greedy clique fix the first colors
    x, w binary.
The clique rows imply the edge rows x_uc + x_vc <= w_c and are fewer and tighter on dense graphs; fixing a clique
removes most of the color-permutation symmetry and makes the LP bound at least its size. Every integer solution is
decoded (vertex -> its color), re-checked with color_count and published; bounds are Gurobi's best bounds rounded up.
Size: n H + H binaries, n + H (|cover| + 2) rows.
Alternative formulation "pop2" (build_pop2_mip): the hybrid partial-ordering model of Jabrayilov and Mutzel (2018)
with the same clique rows, H and start (no clique fixing: its symmetry breaking makes a designated vertex take the
largest color instead).
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from src.cg.coloring.a_data import ColoringInstance, load_col
from src.cg.coloring.heuristic import canonical, dsatur, edge_clique_cover, greedy_clique


# ---------------------------------------------------------------- evaluator

def normalize_coloring(coloring) -> list[int]:
    return [int(c) for c in coloring]


def coloring_issues(inst: ColoringInstance, coloring) -> tuple[int | None, list[str]]:
    """(number of colors, []) for a proper coloring, (None, issues) otherwise."""
    if not isinstance(coloring, (list, tuple, np.ndarray)):
        return None, ["coloring must be a list of labels"]
    if len(coloring) != inst.n:
        return None, [f"coloring has {len(coloring)} labels, {inst.n} vertices"]
    if not all(isinstance(c, (int, np.integer)) and not isinstance(c, (bool, np.bool_)) for c in coloring):
        return None, ["labels must be integers"]
    col = np.asarray([int(c) for c in coloring], dtype=np.int64)
    issues = []
    if inst.m:
        bad = np.nonzero(col[inst.edges[:, 0]] == col[inst.edges[:, 1]])[0]
        if len(bad):
            u, v = inst.edges[bad[0]].tolist()
            issues.append(f"{len(bad)} edges with equal labels (e.g. vertices {u + 1} and {v + 1})")
    return (None, issues) if issues else (len(set(col.tolist())), [])


def color_count(inst: ColoringInstance, coloring) -> int | None:
    """Number of colors of a proper coloring, or None (see coloring_issues)."""
    return coloring_issues(inst, coloring)[0]


def coloring_output(inst: ColoringInstance, coloring, status: str = "feasible", method: str = "decomposition") -> dict[str, Any]:
    """The output contract checked by e_verifier.verify_solution."""
    coloring = normalize_coloring(coloring)
    return {"status": status, "method": method, "objective_value": color_count(inst, coloring), "coloring": coloring}


# ---------------------------------------------------------------- compact MIP

def build_coloring_mip(inst: ColoringInstance, H: int, clique: list[int], cover: list[np.ndarray],
                       env: gp.Env | None = None) -> tuple[gp.Model, dict[str, Any]]:
    n = inst.n
    model = gp.Model(env=env) if env else gp.Model()
    x = model.addMVar(n * H, vtype=GRB.BINARY)  # x[v * H + c]
    w = model.addMVar(H, vtype=GRB.BINARY, obj=1.0)
    I_H = sp.identity(H, format="csr")
    model.addConstr(sp.kron(sp.identity(n, format="csr"), np.ones((1, H)), format="csr") @ x == 1.0)
    T = len(cover)
    if T:
        rows = np.concatenate([np.full(len(K), t) for t, K in enumerate(cover)])
        cols = np.concatenate(cover)
        Kinc = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(T, n))
        model.addConstr(sp.kron(Kinc, I_H, format="csr") @ x - sp.kron(np.ones((T, 1)), I_H, format="csr") @ w <= 0.0)
    iso = np.nonzero(inst.degree == 0)[0]
    if len(iso):
        Iinc = sp.csr_matrix((np.ones(len(iso)), (np.arange(len(iso)), iso)), shape=(len(iso), n))
        model.addConstr(sp.kron(Iinc, I_H, format="csr") @ x - sp.kron(np.ones((len(iso), 1)), I_H, format="csr") @ w <= 0.0)
    model.addConstr(sp.kron(np.ones((1, n)), I_H, format="csr") @ x - w >= 0.0)
    if H > 1:
        model.addConstr(w[:-1] - w[1:] >= 0.0)
    lb = np.zeros(n * H)
    for i, q in enumerate(clique):
        lb[q * H + i] = 1.0
    x.LB = lb
    wl = np.zeros(H)
    wl[:len(clique)] = 1.0
    w.LB = wl
    model.ModelSense = GRB.MINIMIZE
    model.update()
    return model, {"x": x, "w": w, "H": H, "cliques": T, "num_vars": model.NumVars}


def build_pop2_mip(inst: ColoringInstance, H: int, q: int, cover: list[np.ndarray],
                   env: gp.Env | None = None) -> tuple[gp.Model, dict[str, Any]]:
    """The hybrid partial-ordering model POP2 (Jabrayilov and Mutzel 2018) with clique rows: y_vi = 1 iff the color of
    v exceeds i, x_vi = 1 iff it equals i (colors 0..H-1); the designated vertex q takes the largest color:
        min 1 + sum_i y_qi   s.t.  y_v,H-1 = 0;  y_vi >= y_v,i+1;  x_v0 + y_v0 = 1;  x_vi = y_v,i-1 - y_vi (i >= 1);
        sum_{v in K} x_vi <= 1 (cover cliques K);  y_qi >= y_vi;  x, y binary."""
    n = inst.n
    model = gp.Model(env=env) if env else gp.Model()
    x = model.addMVar(n * H, vtype=GRB.BINARY)
    yub = np.ones(n * H)
    yub[H - 1::H] = 0.0
    obj = np.zeros(n * H)
    obj[q * H:(q + 1) * H] = 1.0
    y = model.addMVar(n * H, vtype=GRB.BINARY, ub=yub, obj=obj)
    model.ObjCon = 1.0
    I_n = sp.identity(n, format="csr")
    if H > 1:
        D = sp.diags([np.ones(H - 1), -np.ones(H - 1)], [0, 1], shape=(H - 1, H), format="csr")
        model.addConstr(sp.kron(I_n, D, format="csr") @ y >= 0.0)
        S = sp.csr_matrix((np.ones(H - 1), (np.arange(H - 1), np.arange(1, H))), shape=(H - 1, H))
        model.addConstr(sp.kron(I_n, S, format="csr") @ x - sp.kron(I_n, D, format="csr") @ y == 0.0)
    e0 = sp.csr_matrix(([1.0], ([0], [0])), shape=(1, H))
    model.addConstr(sp.kron(I_n, e0, format="csr") @ x + sp.kron(I_n, e0, format="csr") @ y == 1.0)
    T = len(cover)
    if T:
        rows = np.concatenate([np.full(len(K), t) for t, K in enumerate(cover)])
        cols = np.concatenate(cover)
        Kinc = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(T, n))
        model.addConstr(sp.kron(Kinc, sp.identity(H, format="csr"), format="csr") @ x <= 1.0)
    others = np.array([v for v in range(n) if v != q], dtype=np.int64)
    if len(others):
        sel = sp.csr_matrix((np.ones(len(others)), (np.arange(len(others)), others)), shape=(len(others), n))
        model.addConstr(sp.kron(np.ones((len(others), 1)), sp.identity(H, format="csr"), format="csr") @ y[q * H:(q + 1) * H]
                        - sp.kron(sel, sp.identity(H, format="csr"), format="csr") @ y >= 0.0)
    model.ModelSense = GRB.MINIMIZE
    model.update()
    return model, {"x": x, "y": y, "H": H, "q": q, "cliques": T, "num_vars": model.NumVars}


def start_labels(coloring: np.ndarray, clique: list[int]) -> np.ndarray:
    """A coloring relabeled so that clique vertex q_i has label i (the clique's labels are distinct)."""
    col = np.array(canonical(coloring), dtype=np.int64)
    perm: dict[int, int] = {int(col[q]): i for i, q in enumerate(clique)}
    nxt = len(clique)
    for c in sorted(set(col.tolist())):
        if c not in perm:
            perm[c] = nxt
            nxt += 1
    return np.array([perm[int(c)] for c in col], dtype=np.int64)


class ColoringCompactMIP:
    """The compact MIP with the solver interface of the decomposition references."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 formulation: str = "ass") -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        if formulation not in ("ass", "pop2"):
            raise ValueError(f"unknown formulation {formulation!r}")
        self.formulation = formulation  # "ass": assignment model (default); "pop2": partial-ordering hybrid
        self.on_incumbent = on_incumbent or (lambda coloring: None)
        self.on_bound = on_bound or (lambda bound: None)

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        inst = load_col(data_path)
        n = inst.n
        A = inst.adjacency
        best: dict[str, Any] = {"obj": math.inf, "coloring": None, "bound": -math.inf, "first": None}

        def offer(coloring) -> None:
            col = canonical(coloring)
            obj = color_count(inst, col)
            if obj is not None and obj < best["obj"]:
                best.update(obj=obj, coloring=col)
                if best["first"] is None:
                    best["first"] = time.perf_counter() - t0
                self.on_incumbent(col)

        def bound(value: float) -> None:
            if math.isfinite(value) and value > -1e90:
                b = float(math.ceil(value - 1e-6))
                if best["obj"] < math.inf:
                    b = min(b, float(best["obj"]))
                if b > best["bound"]:
                    best["bound"] = b
                    self.on_bound(b)

        init = dsatur(A)
        offer(init)
        H = int(init.max()) + 1 if n else 0
        clique = greedy_clique(A) if n else []
        bound(float(len(clique)))
        cover = edge_clique_cover(A)
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", self.threads)
        env.setParam("Seed", self.seed)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()
        if self.formulation == "ass":
            model, info = build_coloring_mip(inst, H, clique, cover, env)
            x = info["x"]
            start = start_labels(init, clique)
            info["w"].Start = np.ones(H)
        else:
            q = int(np.argmax(inst.degree))
            model, info = build_pop2_mip(inst, H, q, cover, env)
            x = info["x"]
            start = np.array(canonical(init), dtype=np.int64)
            cq = int(start[q])  # swap labels so that q has the largest one
            start = np.where(start == cq, H - 1, np.where(start == H - 1, cq, start))
            info["y"].Start = (start[:, None] > np.arange(H)[None, :]).astype(float).reshape(-1)
        xs = np.zeros(n * H)
        xs[np.arange(n) * H + start] = 1.0
        x.Start = xs
        xl = x.tolist()
        build = time.perf_counter() - t0

        def decode(vals) -> np.ndarray:
            return np.asarray(vals).reshape(n, H).argmax(axis=1)

        def callback(m, where):
            if time.perf_counter() > deadline:
                m.terminate()
                return
            if where == GRB.Callback.MIPSOL:
                offer(decode(m.cbGetSolution(xl)))
            elif where == GRB.Callback.MIP:
                bound(m.cbGet(GRB.Callback.MIP_OBJBND))

        reason = "time_limit"
        if deadline - time.perf_counter() > 0.01:
            model.Params.TimeLimit = deadline - time.perf_counter()
            model.optimize(callback)
            if model.SolCount:
                offer(decode(model.getAttr("X", xl)))
            if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.MEM_LIMIT):
                bound(float(model.ObjBound))
            # The actual Gurobi status is kept; only a timeout (Gurobi's or the callback's at the deadline) is
            # "time_limit".
            st = model.Status
            if st == GRB.INTERRUPTED and time.perf_counter() >= deadline - 1.0:
                reason = "time_limit"
            else:
                reason = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit", GRB.MEM_LIMIT: "memory_limit",
                          GRB.INTERRUPTED: "interrupted", GRB.INFEASIBLE: "infeasible"}.get(st, f"status_{st}")
        if best["obj"] < math.inf and best["bound"] >= best["obj"]:
            reason = "optimal"
        out = {"best_objective": best["obj"] if best["coloring"] is not None else None,
               "global_lb": best["bound"] if best["bound"] > -math.inf else None, "termination_reason": reason,
               "coloring": best["coloring"], "formulation": self.formulation, "H": H, "clique_size": len(clique),
               "cover_cliques": info["cliques"],
               "build_seconds": build, "num_vars": info["num_vars"], "num_constrs": model.NumConstrs,
               "first_incumbent_seconds": best["first"], "node_count": float(model.NodeCount),
               "total_seconds": time.perf_counter() - t0}
        model.dispose()
        env.dispose()
        return out
