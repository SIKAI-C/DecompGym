"""Pricing for BPPC column generation (scale-up study): the knapsack problem with conflicts (KPC)
    max sum_i p_i z_i  s.t.  sum_i w_i z_i <= C,  z_i + z_j <= 1 for every conflict (i, j),  z binary,
over the items of a branch-and-price node (merged items and added conflicts included); a bin (column) prices out
when its value exceeds 1 (reduced cost 1 - value < 0).

KPCOracle is the exact oracle shared by the naive and the engineered column generation: a Gurobi MIP built once per
conflict graph, with the conflicts written as clique rows sum_{i in K} z_i <= 1 over an edge clique cover (they imply
every edge row). Items with p_i <= tol are fixed to 0 (they cannot raise the value). MIPGap 0; Cutoff at the
threshold (1 + tol), so the search looks only for improving bins and, when there is none, ends with the proof that
the optimum is at most the threshold. Every returned bin is re-checked (capacity, conflicts). `bound` is a valid upper
bound on the optimum whenever it is not None (Gurobi's best bound, or the threshold after a cutoff), also when the
round is cut short (`complete` False).

SmallBinOracle is the engineered reference's exact pricing by enumeration when the bins of four or more items
are few (every triplet instance).
greedy_bins is the engineered reference's heuristic pricing: greedy fills in profit-per-weight order from several
seed items, each improved by add and one-for-one swap moves.
"""
from __future__ import annotations

import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp


def bin_ok(items: np.ndarray, weights: np.ndarray, capacity: int, adj: np.ndarray) -> bool:
    return len(items) > 0 and int(weights[items].sum()) <= capacity and not adj[np.ix_(items, items)].any()


class KPCOracle:
    def __init__(self, weights: np.ndarray, capacity: int, adj: np.ndarray, cliques: list[list[int]],
                 env: gp.Env, tol: float = 1e-6) -> None:
        self.w = np.asarray(weights, dtype=np.int64)
        self.C = int(capacity)
        self.adj = adj
        self.tol = tol
        m = len(self.w)
        md = gp.Model(env=env)
        md.Params.OutputFlag = 0
        md.Params.MIPGap = 0.0
        z = md.addMVar(m, vtype=GRB.BINARY)
        md.addConstr(self.w.astype(float) @ z <= float(self.C))
        cl = [K for K in cliques if len(K) >= 2]
        if cl:
            rows = np.concatenate([np.full(len(K), r) for r, K in enumerate(cl)])
            cols = np.concatenate([np.asarray(K, dtype=np.int64) for K in cl])
            A = sp.csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(cl), m))
            md.addMConstr(A, z, "<", np.ones(len(cl)))
        md.ModelSense = GRB.MAXIMIZE
        md.update()
        self.model, self.z, self.m = md, z, m
        self.calls, self.seconds = 0, 0.0

    def solve(self, p: np.ndarray, *, threshold: float | None = None, max_cols: int = 1,
              deadline: float | None = None, pool_mode: int = 0) -> dict:
        t = time.perf_counter()
        thr = 1.0 + self.tol if threshold is None else threshold
        p = np.asarray(p, dtype=float)
        out = {"cols": [], "complete": False, "value": 0.0, "bound": None, "status": None, "seconds": 0.0}
        left = None if deadline is None else deadline - time.perf_counter()
        if left is not None and left <= 1e-3:
            return out
        md, z = self.model, self.z
        pos = p > self.tol
        if not pos.any() or float(np.sort(p[pos])[::-1].sum()) <= thr:
            # Even all positive-profit items together cannot exceed the threshold.
            out.update(complete=True, bound=float(p[pos].sum()) if pos.any() else 0.0, seconds=time.perf_counter() - t)
            return out
        z.UB = pos.astype(float)
        z.Obj = np.where(pos, p, 0.0)
        md.Params.Cutoff = thr
        md.Params.PoolSolutions = max(1, int(max_cols))
        md.Params.PoolSearchMode = pool_mode if max_cols > 1 else 0
        md.Params.TimeLimit = max(1e-3, left) if left is not None else GRB.INFINITY
        md.optimize()
        st = int(md.Status)
        out["status"] = st
        if st in (GRB.CUTOFF, GRB.INFEASIBLE):  # no bin of value above the threshold
            out.update(complete=True, bound=thr)
        else:
            out["complete"] = st == GRB.OPTIMAL
            try:
                bd = float(md.ObjBound)
                out["bound"] = bd if np.isfinite(bd) and bd < 1e90 else None
            except gp.GurobiError:
                out["bound"] = None
            if out["complete"] and out["bound"] is not None:
                out["bound"] = max(out["bound"], float(md.ObjVal))
            found = {}
            for s in range(md.SolCount):
                md.Params.SolutionNumber = s
                items = np.flatnonzero(np.asarray(z.Xn) > 0.5)
                val = float(p[items].sum())
                if val > thr and bin_ok(items, self.w, self.C, self.adj):
                    found.setdefault(tuple(items.tolist()), val)
            out["cols"] = sorted(((v, list(k)) for k, v in found.items()), key=lambda c: -c[0])
            if out["cols"]:
                out["value"] = out["cols"][0][0]
        self.calls += 1
        out["seconds"] = time.perf_counter() - t
        self.seconds += out["seconds"]
        return out

    def dispose(self) -> None:
        self.model.dispose()


def max_items(weights: np.ndarray, capacity: int) -> int:
    """The largest number of items any bin can hold (the lightest items first)."""
    return int(np.searchsorted(np.cumsum(np.sort(np.asarray(weights))), capacity, side="right"))


def large_bins(weights: np.ndarray, capacity: int, adj: np.ndarray, max_light: int = 20,
               max_bins: int = 20000) -> list[np.ndarray] | None:
    """Every feasible bin of four or more items, or None if they are too many to list. Such a bin holds only "light"
    items i with w_i + (three smallest other weights) <= C (each item of a bin of >= 4 items satisfies it), so the
    bins are enumerated over the light items (at most max_light of them)."""
    w = np.asarray(weights, dtype=np.int64)
    n = len(w)
    if n < 4:
        return []
    srt = np.sort(w)
    light = []
    for i in range(n):
        others = srt[:4] if w[i] > srt[3] else np.delete(srt[:4], np.searchsorted(srt[:4], w[i]))
        if w[i] + int(others[:3].sum()) <= capacity:
            light.append(i)
    if len(light) > max_light:
        return None
    out: list[np.ndarray] = []

    def grow(items: list[int], start: int, load: int) -> bool:
        if len(items) >= 4:
            out.append(np.array(items, dtype=np.int64))
            if len(out) > max_bins:
                return False
        for t in range(start, len(light)):
            j = light[t]
            if load + w[j] <= capacity and not adj[j, items].any():
                if not grow(items + [j], t + 1, load + int(w[j])):
                    return False
        return True

    return out if grow([], 0, 0) else None


class SmallBinOracle:
    """Exact pricing by enumeration when the bins of four or more items are few (large_bins is not None; e.g. every
    triplet instance: weights in [250, 500], capacity 1000, where only items of weight 250 can form a fourth-item bin):
    the best single, the best pair and, for every item a, the best pair (b, c) after a, vectorized over the
    positive-profit items sorted by profit (items a whose bound p_a + p_{a+1} + p_{a+2} cannot beat the max_cols-th
    best value found, or the threshold, end the scan), and every listed bin of four or more items. Same result format
    as KPCOracle; `bound` is the exact optimum (or a valid bound above it when the scan stopped early)."""

    def __init__(self, weights: np.ndarray, capacity: int, adj: np.ndarray) -> None:
        self.w = np.asarray(weights, dtype=np.int64)
        self.C = int(capacity)
        self.ok = ~adj
        np.fill_diagonal(self.ok, False)
        big = large_bins(self.w, self.C, adj)
        if big is None:
            raise ValueError("SmallBinOracle needs few bins of four or more items")
        self.big = big
        if big:
            width = max(len(b) for b in big)
            self.big_idx = np.full((len(big), width), len(self.w), dtype=np.int64)  # padded with a dummy item
            for r, b in enumerate(big):
                self.big_idx[r, :len(b)] = b
        self.calls, self.seconds = 0, 0.0

    def solve(self, p: np.ndarray, *, threshold: float | None = None, max_cols: int = 1,
              deadline: float | None = None, banned: np.ndarray | None = None, **_) -> dict:
        t = time.perf_counter()
        thr = 1.0 + 1e-6 if threshold is None else threshold
        p = np.asarray(p, dtype=float)
        pos = p > 1e-9
        if banned is not None:
            pos &= ~banned
        cand = np.flatnonzero(pos)
        cand = cand[np.lexsort((cand, -p[cand]))]
        m = len(cand)
        found: dict[tuple, float] = {}
        best = [0.0]  # current best values (sorted descending, at most max_cols)

        def push(val: float, items) -> None:
            key = tuple(sorted(int(i) for i in items))
            if key not in found:
                found[key] = val
                best.append(val)
                best.sort(reverse=True)
                del best[max(1, max_cols):]

        out = {"cols": [], "complete": True, "value": 0.0, "bound": None, "status": "enumeration", "seconds": 0.0}
        tail = -np.inf  # upper bound on the triples left unscanned
        if m:
            q, w, ok = p[cand], self.w[cand], self.ok[np.ix_(cand, cand)]
            push(float(q[0]), [cand[0]])
            if m >= 2:
                V2 = np.where(np.triu(ok & (w[:, None] + w[None, :] <= self.C), 1), q[:, None] + q[None, :], -np.inf)
                rows = np.argmax(V2, axis=1)
                vals = V2[np.arange(m), rows]
                for a in np.argsort(-vals)[:max(1, max_cols)].tolist():
                    if np.isfinite(vals[a]):
                        push(float(vals[a]), [cand[a], cand[rows[a]]])
                if m >= 3:
                    for a in range(m - 2):
                        if q[a] + q[a + 1] + q[a + 2] <= max(thr, best[-1] if len(best) >= max_cols else thr) + 1e-12:
                            tail = float(q[a] + q[a + 1] + q[a + 2])
                            break
                        if deadline is not None and time.perf_counter() > deadline:
                            out["complete"] = False
                            break
                        sub = V2[a + 1:, a + 1:]
                        okb = ok[a, a + 1:]
                        mask = okb[:, None] & okb[None, :] & (w[a + 1:][:, None] + w[a + 1:][None, :] <= self.C - w[a])
                        vv = np.where(mask, sub, -np.inf)
                        k = int(np.argmax(vv))
                        v = vv.flat[k]
                        if np.isfinite(v):
                            b, c = divmod(k, vv.shape[1])
                            push(float(q[a] + v), [cand[a], cand[a + 1 + b], cand[a + 1 + c]])
        if self.big and out["complete"]:
            pe = np.append(np.where(pos, p, 0.0), 0.0)
            bad = np.append(~pos, False)
            vals = pe[self.big_idx].sum(axis=1)
            vals[bad[self.big_idx].any(axis=1)] = -np.inf  # a bin with a banned or non-positive item: skip
            for r in np.argsort(-vals)[:max(1, max_cols)].tolist():
                if np.isfinite(vals[r]):
                    push(float(vals[r]), self.big[r].tolist())
        top = max(found.values(), default=0.0)
        if out["complete"]:
            out["bound"] = max(top, tail)  # the optimum, or a valid bound above it when the scan stopped early
        out["cols"] = sorted(((v, list(k)) for k, v in found.items() if v > thr), key=lambda c: -c[0])[:max(1, max_cols)]
        out["value"] = out["cols"][0][0] if out["cols"] else 0.0
        self.calls += 1
        out["seconds"] = time.perf_counter() - t
        self.seconds += out["seconds"]
        return out

    def dispose(self) -> None:
        pass


def improve_bin(items: list[int], p: np.ndarray, weights: np.ndarray, capacity: int, adj: np.ndarray,
                ok: np.ndarray, rounds: int = 8) -> list[int]:
    """Local search on a bin: add the most profitable item that fits, or replace one item by the most profitable
    item that fits once it is removed (no conflicts with the rest), while the value increases."""
    items = list(items)
    for _ in range(rounds):
        load = int(weights[items].sum()) if items else 0
        blocked = adj[items].any(axis=0) if items else np.zeros(len(p), dtype=bool)
        free = ok & ~blocked
        free[items] = False
        fit = np.flatnonzero(free & (weights <= capacity - load))
        best_gain, move = 1e-9, None
        if len(fit):
            j = int(fit[np.argmax(p[fit])])
            best_gain, move = p[j], (None, j)
        for r in range(len(items)):
            rest = items[:r] + items[r + 1:]
            room = capacity - (load - int(weights[items[r]]))
            fr = ok & ~(adj[rest].any(axis=0) if rest else np.zeros(len(p), dtype=bool))
            fr[items] = False
            c = np.flatnonzero(fr & (weights <= room))
            if len(c):
                j = int(c[np.argmax(p[c])])
                gain = p[j] - p[items[r]]
                if gain > best_gain + 1e-12:
                    best_gain, move = gain, (r, j)
        if move is None or best_gain <= 1e-9:
            break
        r, j = move
        if r is None:
            items.append(j)
        else:
            items[r] = j
    return items


def greedy_bins(p: np.ndarray, weights: np.ndarray, capacity: int, adj: np.ndarray, *, threshold: float,
                seeds: int = 12, max_cols: int = 30, banned: np.ndarray | None = None,
                local_search: bool = True) -> list[tuple[float, list[int]]]:
    """Heuristic pricing: for each of the `seeds` items with the best profit-per-weight ratio, the `seeds` most
    profitable items and the empty seed, start a bin with it and add the remaining items in ratio order whenever
    they fit and conflict with nothing in the bin; then improve the bin by add / one-for-one swap moves
    (improve_bin). Returns distinct bins of value above the threshold, best first."""
    p = np.asarray(p, dtype=float)
    weights = np.asarray(weights)
    ok = p > 1e-9
    if banned is not None:
        ok &= ~banned
    cand = np.flatnonzero(ok)
    if not len(cand):
        return []
    order = cand[np.lexsort((cand, -p[cand], -(p[cand] / weights[cand])))]
    by_profit = cand[np.lexsort((cand, -p[cand]))]
    found: dict[tuple, float] = {}
    starts = [None] + list(dict.fromkeys(order[:seeds].tolist() + by_profit[:seeds].tolist()))
    for s in starts:
        load = 0
        items: list[int] = []
        free = ok.copy()
        if s is not None:
            items.append(s)
            load = int(weights[s])
            free &= ~adj[s]
            free[s] = False
        for i in order.tolist():
            if free[i] and load + weights[i] <= capacity:
                items.append(i)
                load += int(weights[i])
                free &= ~adj[i]
                free[i] = False
        if local_search:
            items = improve_bin(items, p, weights, capacity, adj, ok)
        val = float(p[items].sum())
        if val > threshold:
            found.setdefault(tuple(sorted(items)), val)
    return sorted(((v, list(k)) for k, v in found.items()), key=lambda c: -c[0])[:max_cols]
