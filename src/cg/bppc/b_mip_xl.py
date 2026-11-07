"""BPPC packing evaluator, output contract and compact MIP (scale-up study).

Evaluator. `bin_count(inst, bins)` takes bins as lists of item numbers 1..n (the file numbering) and returns the
number of bins if every item is packed exactly once, every bin is nonempty, within the capacity and free of
conflicting pairs; otherwise None (`packing_issues` lists the reasons).

Compact MIP: the assignment model with U bins (U = the shared basic initializer's bin count, an upper bound; the
initializer's plan is not given to the solver unless start=True, a diagnostic arm):
    min sum_b y_b
    sum_{b in B_i} x_ib = 1                                   every item i
    sum_i w_i x_ib <= C y_b                                   every bin b
    sum_{i in K} x_ib <= y_b                                  every clique K of an edge clique cover, every bin b
    x_ib <= y_b                                               items without conflicts
    y_b >= y_{b+1};  y_b = 1 for b <= max(q, ceil(w(N) / C));  x, y binary.
Symmetry breaking (exact; canonical_bins gives every packing with at most U bins its unique representation): the
items of a greedy maximal clique K0 = (k_1..k_q) of the conflict graph are fixed to bins 1..q; the other items are
ordered by non-increasing weight (ties: lower index) at positions q+1..n, and the item at position p may use bins
1..min(p, U) only (B_i). In a packing, the bins holding K0 items take labels 1..q, the others labels q+1, q+2, ... in
the order of their first item, so the bin of the item at position p has a label <= p and the used labels are 1..k.
The clique rows (Muritiba et al. 2010 use one row x_ib + x_jb <= y_b per edge and bin) are the same constraints
aggregated over cliques; they imply every edge row, so the model is exact (tests/large_scale/test_bppc.py fixes random
packings into it and compares with the independent verifier).
MIPGap 0 and MIPGapAbs 0.99 (integral objective), so OPTIMAL is a proof; bounds are Gurobi's best bounds rounded up.
Every integer solution is decoded and re-checked with bin_count before it is published; the actual Gurobi status is
kept (a deadline stop from the callback is reported as "time_limit").
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from src.cg.bppc.a_data import BPPCInstance, load_bppc
from src.cg.bppc.heuristic import basic_initializer, edge_clique_cover, greedy_clique


# ---------------------------------------------------------------- evaluator and output contract

def normalize_bins(bins: list) -> list[list[int]]:
    return [[int(i) for i in b] for b in bins]


def packing_issues(inst: BPPCInstance, bins: list) -> tuple[int | None, list[str]]:
    """(number of bins, []) for a feasible packing (items 1..n), (None, issues) otherwise."""
    n, C, w, adj = inst.n, inst.capacity, inst.weights, inst.adj
    issues: list[str] = []
    if not isinstance(bins, (list, tuple)):
        return None, ["bins must be a list"]
    seen = np.zeros(n + 1, dtype=np.int64)
    for k, b in enumerate(bins):
        if not isinstance(b, (list, tuple)) or not b:
            issues.append(f"bin {k}: empty or not a list")
            continue
        if not all(isinstance(i, (int, np.integer)) and not isinstance(i, bool) and 1 <= i <= n for i in b):
            issues.append(f"bin {k}: item numbers must be integers in 1..{n}")
            continue
        idx = np.array([int(i) - 1 for i in b])
        load = int(w[idx].sum())
        if load > C:
            issues.append(f"bin {k}: load {load} exceeds capacity {C}")
        if adj[np.ix_(idx, idx)].any():
            I, J = np.nonzero(np.triu(adj[np.ix_(idx, idx)], 1))
            issues.append(f"bin {k}: conflicting items {int(idx[I[0]]) + 1} and {int(idx[J[0]]) + 1}")
        np.add.at(seen, idx + 1, 1)
    bad = np.flatnonzero(seen[1:] != 1) + 1
    if len(bad):
        issues.append(f"{len(bad)} items not packed exactly once (e.g. {bad[:5].tolist()})")
    return (None, issues) if issues else (len(bins), [])


def bin_count(inst: BPPCInstance, bins: list) -> int | None:
    return packing_issues(inst, bins)[0]


def bppc_output(inst: BPPCInstance, bins: list, status: str = "feasible", method: str = "column_generation") -> dict[str, Any]:
    """The output contract checked by e_verifier.verify_solution."""
    bins = normalize_bins(bins)
    return {"status": status, "method": method, "objective_value": bin_count(inst, bins), "bins": bins}


def public(bins: list[list[int]]) -> list[list[int]]:
    """Internal item indices 0..n-1 -> item numbers 1..n."""
    return [[int(i) + 1 for i in b] for b in bins if len(b)]


# ---------------------------------------------------------------- compact MIP

def mip_layout(inst: BPPCInstance, U: int, clique: list[int] | None = None) -> dict[str, Any]:
    """Item positions and allowed bins (0-based labels) of the symmetry-broken assignment model."""
    n = inst.n
    K0 = list(greedy_clique(inst.adj, inst.weights) if clique is None else clique)
    q = len(K0)
    if q > U:
        raise ValueError("the clique is larger than the bin bound U")
    rest = [i for i in np.lexsort((np.arange(n), -inst.weights)).tolist() if i not in set(K0)]
    order = K0 + rest
    pos = np.empty(n, dtype=np.int64)
    pos[np.array(order, dtype=np.int64)] = np.arange(n)
    allowed = [[int(pos[i])] if pos[i] < q else list(range(min(int(pos[i]) + 1, U))) for i in range(n)]
    return {"K0": K0, "q": q, "order": order, "pos": pos, "allowed": allowed, "U": U,
            "lb0": max(q, inst.weight_bound)}


def canonical_bins(layout: dict[str, Any], bins: list[list[int]]) -> dict[int, int] | None:
    """The unique label (0-based bin index) of every item (internal indices) in the model's representation of a
    packing, or None if it has more than U bins. Bins holding a K0 item take that item's position; the others are
    numbered q, q+1, ... in the order of their first item's position."""
    pos, q, U = layout["pos"], layout["q"], layout["U"]
    if len(bins) > U:
        return None
    label: dict[int, int] = {}
    free = []
    for b in bins:
        k0 = [int(pos[i]) for i in b if pos[i] < q]
        if len(k0) > 1:
            return None  # two clique items together: not a feasible packing
        if k0:
            for i in b:
                label[int(i)] = k0[0]
        else:
            free.append(b)
    for t, b in enumerate(sorted(free, key=lambda b: min(int(pos[i]) for i in b))):
        for i in b:
            label[int(i)] = q + t
    return label


def build_bppc_mip(inst: BPPCInstance, U: int, env: gp.Env | None = None, layout: dict[str, Any] | None = None,
                   cliques: list[list[int]] | None = None) -> tuple[gp.Model, dict[str, Any]]:
    n, C, w = inst.n, inst.capacity, inst.weights
    lay = mip_layout(inst, U) if layout is None else layout
    cover = edge_clique_cover(inst.adj) if cliques is None else cliques
    allowed = lay["allowed"]
    I = np.concatenate([np.full(len(a), i, dtype=np.int64) for i, a in enumerate(allowed)])
    B = np.concatenate([np.array(a, dtype=np.int64) for a in allowed])
    nx = len(I)
    col = {(int(i), int(b)): k for k, (i, b) in enumerate(zip(I.tolist(), B.tolist()))}
    model = gp.Model(env=env) if env else gp.Model()
    yfix = np.zeros(U)
    yfix[:min(U, lay["lb0"])] = 1.0
    v = model.addMVar(nx + U, vtype=GRB.BINARY, lb=np.concatenate([np.zeros(nx), yfix]),
                      obj=np.concatenate([np.zeros(nx), np.ones(U)]))
    R, Cc, Vv, rhs, sense = [], [], [], [], []

    def block(r_, c_, v_) -> None:
        R.append(np.asarray(r_, dtype=np.int64)), Cc.append(np.asarray(c_, dtype=np.int64)), Vv.append(np.asarray(v_, dtype=float))

    r = 0
    # assignment: sum_b x_ib = 1
    block(I, np.arange(nx), np.ones(nx))
    rhs.append(np.ones(n)), sense.append(np.full(n, "="))
    r += n
    # capacity: sum_i w_i x_ib - C y_b <= 0
    block(r + B, np.arange(nx), w[I])
    block(r + np.arange(U), nx + np.arange(U), np.full(U, -float(C)))
    rhs.append(np.zeros(U)), sense.append(np.full(U, "<"))
    r += U
    # clique rows: sum_{i in K} x_ib - y_b <= 0 for every bin b that some member of K may use
    length = np.array([len(a) for a in allowed], dtype=np.int64)
    offset = np.concatenate([[0], np.cumsum(length)[:-1]])
    for K in cover:
        kk = np.concatenate([offset[i] + np.arange(length[i]) for i in K])
        bb = B[kk]
        ub, inv = np.unique(bb, return_inverse=True)
        block(r + inv, kk, np.ones(len(kk)))
        block(r + np.arange(len(ub)), nx + ub, -np.ones(len(ub)))
        rhs.append(np.zeros(len(ub))), sense.append(np.full(len(ub), "<"))
        r += len(ub)
    # links x_ib - y_b <= 0 for items without conflicts (the clique rows imply them for the others)
    link = np.flatnonzero(np.isin(I, np.flatnonzero(~inst.adj.any(axis=1))))
    if len(link):
        block(r + np.arange(len(link)), link, np.ones(len(link)))
        block(r + np.arange(len(link)), nx + B[link], -np.ones(len(link)))
        rhs.append(np.zeros(len(link))), sense.append(np.full(len(link), "<"))
        r += len(link)
    # contiguity: y_b - y_{b+1} >= 0
    if U > 1:
        block(r + np.arange(U - 1), nx + np.arange(U - 1), np.ones(U - 1))
        block(r + np.arange(U - 1), nx + np.arange(1, U), -np.ones(U - 1))
        rhs.append(np.zeros(U - 1)), sense.append(np.full(U - 1, ">"))
        r += U - 1
    A = sp.csr_matrix((np.concatenate(Vv), (np.concatenate(R), np.concatenate(Cc))), shape=(r, nx + U))
    model.addMConstr(A, v, np.concatenate(sense), np.concatenate(rhs))
    model.ModelSense = GRB.MINIMIZE
    model.update()
    return model, {"v": v, "I": I, "B": B, "nx": nx, "U": U, "layout": lay, "cliques": len(cover), "col": col,
                   "num_vars": int(model.NumVars), "num_constrs": int(model.NumConstrs), "num_nz": int(A.nnz)}


def decode(info: dict[str, Any], values: np.ndarray) -> list[list[int]]:
    """Bins (internal indices) of an integer solution: the items whose x_ib is set, grouped by b."""
    nx, I, B = info["nx"], info["I"], info["B"]
    on = np.flatnonzero(np.asarray(values[:nx]) > 0.5)
    groups: dict[int, list[int]] = {}
    for k in on.tolist():
        groups.setdefault(int(B[k]), []).append(int(I[k]))
    return [groups[b] for b in sorted(groups)]


class BPPCCompactMIP:
    """The compact MIP with the solver interface of the decomposition references (see the module docstring)."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 start: bool = False) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.start = start
        self.on_incumbent = on_incumbent or (lambda bins: None)
        self.on_bound = on_bound or (lambda bound: None)

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        inst = load_bppc(data_path)
        best: dict[str, Any] = {"obj": math.inf, "bins": None, "bound": -math.inf, "first": None, "rejected": 0,
                                "hit_deadline": False}

        def offer(bins: list[list[int]]) -> None:
            pub = public(bins)
            obj = bin_count(inst, pub)
            if obj is None:
                best["rejected"] += 1
                return
            if obj < best["obj"] and time.perf_counter() < deadline:  # never publish after the deadline
                best.update(obj=obj, bins=pub)
                if best["first"] is None:
                    best["first"] = time.perf_counter() - t0
                self.on_incumbent(pub)

        def bound(value: float) -> None:
            if math.isfinite(value) and value > -1e90:
                b = float(math.ceil(value - 1e-6))
                if best["obj"] < math.inf:
                    b = min(b, float(best["obj"]))
                if b > best["bound"]:
                    best["bound"] = b
                    self.on_bound(b)

        init = basic_initializer(inst)
        U = len(init)
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", self.threads)
        env.setParam("Seed", self.seed)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()
        out: dict[str, Any] = {"U": U}
        try:
            model, info = build_bppc_mip(inst, U, env)
        except gp.GurobiError as err:
            if err.errno != GRB.Error.OUT_OF_MEMORY and "memory" not in str(err).lower():
                raise
            env.dispose()
            return {"best_objective": None, "global_lb": None, "termination_reason": "memory_limit", "gurobi_status": None,
                    "budget_exhausted": time.perf_counter() - t0 >= self.time_limit - 1.0, "bins": None,
                    "error": str(err), "total_seconds": time.perf_counter() - t0, **out}
        build = time.perf_counter() - t0
        v = info["v"]
        bound(float(info["layout"]["lb0"]))
        if self.start:
            offer(init)
            lab = canonical_bins(info["layout"], init)
            x0 = np.zeros(info["nx"] + U)
            for i, b in lab.items():
                x0[info["col"][(i, b)]] = 1.0
            x0[info["nx"]:info["nx"] + len(init)] = 1.0
            v.Start = x0

        def callback(m, where):
            if time.perf_counter() > deadline:
                best["hit_deadline"] = True
                m.terminate()  # Gurobi's TimeLimit can overshoot on large models
                return
            if where == GRB.Callback.MIPSOL:
                offer(decode(info, np.array(m.cbGetSolution(v.tolist()))))
            elif where == GRB.Callback.MIP:
                bound(m.cbGet(GRB.Callback.MIP_OBJBND))

        reason, status = "deadline_during_build", None
        if deadline - time.perf_counter() > 0.01:
            model.Params.TimeLimit = deadline - time.perf_counter()
            model.Params.MIPGap = 0.0
            model.Params.MIPGapAbs = 0.99  # integral objective: OPTIMAL certifies optimality
            model.optimize(callback)
            status = int(model.Status)
            if model.SolCount:
                offer(decode(info, np.array(model.getAttr("X", v.tolist()))))
            if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.MEM_LIMIT):
                bound(float(model.ObjBound))
            reason = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit", GRB.MEM_LIMIT: "memory_limit",
                      GRB.INFEASIBLE: "infeasible"}.get(status, f"status_{status}")
            if status == GRB.INTERRUPTED:
                reason = "time_limit" if best["hit_deadline"] else "interrupted"
        out.update({"best_objective": best["obj"] if best["bins"] else None,
                    "global_lb": best["bound"] if best["bound"] > -math.inf else None, "termination_reason": reason,
                    "gurobi_status": status, "budget_exhausted": time.perf_counter() - t0 >= self.time_limit - 1.0,
                    "bins": best["bins"], "build_seconds": build, "first_incumbent_seconds": best["first"],
                    "rejected_solutions": best["rejected"], "num_vars": info["num_vars"],
                    "num_constrs": info["num_constrs"], "num_nz": info["num_nz"], "cliques": info["cliques"],
                    "clique_K0": info["layout"]["q"], "lb0": info["layout"]["lb0"],
                    "node_count": float(model.NodeCount) if status is not None else 0.0,
                    "total_seconds": time.perf_counter() - t0})
        model.dispose()
        env.dispose()
        return out
