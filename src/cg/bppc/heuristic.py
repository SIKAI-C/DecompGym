"""Shared BPPC helpers (scale-up study): the basic initializer shared by every method that uses one, the trimming of
a covering plan into a partition, and the conflict-graph cliques used by the compact MIP and the pricing oracle.

Bins are lists of internal item indices 0..n-1 (a_data's numbering minus one); public plans use 1..n.
"""
from __future__ import annotations

import numpy as np

from src.cg.bppc.a_data import BPPCInstance


def first_fit(inst: BPPCInstance, order: np.ndarray) -> list[list[int]]:
    """First fit in the given item order: each item goes into the first open bin with room and no conflicting item,
    otherwise into a new bin."""
    n, C, w, adj = inst.n, inst.capacity, inst.weights, inst.adj
    load = np.zeros(n, dtype=np.int64)
    blocked = np.zeros((n, n), dtype=bool)  # blocked[b, i]: bin b holds an item in conflict with i
    bins: list[list[int]] = []
    for i in order.tolist():
        nb = len(bins)
        ok = np.flatnonzero((load[:nb] + w[i] <= C) & ~blocked[:nb, i])
        b = int(ok[0]) if len(ok) else nb
        if b == nb:
            bins.append([])
        bins[b].append(i)
        load[b] += w[i]
        blocked[b] |= adj[i]
    return bins


def basic_initializer(inst: BPPCInstance) -> list[list[int]]:
    """The shared basic initializer: first fit decreasing with conflicts, items by non-increasing weight and,
    separately, by non-increasing conflict degree (ties: heavier, then lower index); the plan with fewer bins (the
    weight order on ties). Deterministic."""
    idx = np.arange(inst.n)
    deg = inst.adj.sum(axis=1)
    by_weight = np.lexsort((idx, -inst.weights))
    by_degree = np.lexsort((idx, -inst.weights, -deg))
    a, b = first_fit(inst, by_weight), first_fit(inst, by_degree)
    return a if len(a) <= len(b) else b


def trim_cover(bins: list[list[int]]) -> list[list[int]]:
    """A covering plan made a partition: every item stays only in the first bin that holds it (subsets of feasible
    bins are feasible); empty bins are dropped."""
    seen: set[int] = set()
    out = []
    for b in bins:
        kept = [int(i) for i in b if i not in seen]
        seen.update(kept)
        if kept:
            out.append(kept)
    return out


def greedy_clique(adj: np.ndarray, weights: np.ndarray | None = None) -> list[int]:
    """A maximal clique grown greedily from the vertex of largest degree, always adding the candidate with the most
    neighbours among the remaining candidates (ties: heavier, then lower index)."""
    n = len(adj)
    if n == 0:
        return []
    deg = adj.sum(axis=1)
    w = np.zeros(n) if weights is None else np.asarray(weights, dtype=float)
    best: list[int] = []
    for start in np.lexsort((np.arange(n), -w, -deg))[:min(n, 10)].tolist():
        K = [start]
        cand = adj[start].copy()
        while cand.any():
            ci = np.flatnonzero(cand)
            score = (adj[ci] & cand).sum(axis=1)
            j = int(ci[np.lexsort((ci, -w[ci], -score))[0]])
            K.append(j)
            cand &= adj[j]
        if len(K) > len(best):
            best = K
    return best


def edge_clique_cover(adj: np.ndarray) -> list[list[int]]:
    """Cliques (size >= 2) covering every edge of the graph: for every uncovered edge (u, v), u taken by decreasing
    degree, the clique {u, v} is grown greedily by the common neighbour with the most uncovered edges into the clique
    (ties: larger degree). Every clique inequality sum_{i in K} z_i <= 1 is valid, and together they imply z_i + z_j
    <= 1 for every edge."""
    n = len(adj)
    U = np.triu(adj, 1)
    U = U | U.T
    deg = adj.sum(axis=1)
    cliques: list[list[int]] = []
    for u in np.argsort(-deg, kind="stable").tolist():
        while U[u].any():
            nbr = np.flatnonzero(U[u])
            v = int(nbr[np.argmax(deg[nbr])])
            K = [u, v]
            cand = adj[u] & adj[v]
            gain = U[:, u].astype(np.int64) + U[:, v]  # uncovered edges from every vertex into the clique
            while cand.any():
                ci = np.flatnonzero(cand)
                j = int(ci[np.lexsort((-deg[ci], -gain[ci]))[0]])
                K.append(j)
                cand &= adj[j]
                gain += U[:, j]
            ix = np.array(K)
            U[np.ix_(ix, ix)] = False
            cliques.append(sorted(K))
    return cliques
