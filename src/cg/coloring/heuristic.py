"""Shared vertex-coloring primal and bound helpers (scale-up study).

- dsatur: Brelaz's DSATUR (1979), deterministic (largest saturation, then largest degree, then lowest index; smallest
  feasible color). It is the basic initializer shared by the compact MIP (its color bound H and MIP start), the naive
  and the engineered column generation.
- greedy_clique: a clique grown greedily from every vertex (largest remaining candidate degree first); its size is a
  valid lower bound on the chromatic number.
- tabucol: TabuCol (Hertz and de Werra 1987, with the tenure rule of Galinier and Hao 1999) on k colors, vectorized
  over the conflicting vertices; used only by the engineered reference (MMT-style column source and incumbent polish).
- edge_clique_cover: cliques covering every edge, for the strengthened conflict rows of the MIP and of the engineered
  exact pricing.
- independent-set helpers: classes of a coloring, colorings from covers, maximal extension.
Colorings are integer arrays of length n (vertex v gets label coloring[v]); the objective is the number of labels.
"""
from __future__ import annotations

import time

import numpy as np


def canonical(coloring) -> list[int]:
    """Labels renumbered 0..k-1 in order of first appearance."""
    seen: dict[int, int] = {}
    return [seen.setdefault(int(c), len(seen)) for c in coloring]


def classes_of(coloring) -> list[np.ndarray]:
    col = np.asarray(coloring)
    return [np.nonzero(col == c)[0] for c in sorted(set(col.tolist()))]


def is_proper(A: np.ndarray, coloring) -> bool:
    col = np.asarray(coloring)
    return not (A & (col[:, None] == col[None, :])).any()


def dsatur(A: np.ndarray, order_key: np.ndarray | None = None) -> np.ndarray:
    """DSATUR coloring of the graph with adjacency A (labels 0..k-1)."""
    n = len(A)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    deg = A.sum(axis=1).astype(np.int64) if order_key is None else np.asarray(order_key, dtype=np.int64)
    nbrs = [np.nonzero(A[v])[0] for v in range(n)]
    forb = np.zeros((n, n + 1), dtype=bool)
    sat = np.zeros(n, dtype=np.int64)
    color = np.full(n, -1, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    # Lexicographic key (saturation, degree, -index) as one integer.
    base = int(deg.max()) + 1
    for _ in range(n):
        key = np.where(done, -1, (sat * base + deg) * (n + 1) + (n - np.arange(n)))
        v = int(np.argmax(key))
        c = int(np.argmin(forb[v]))  # first False
        color[v] = c
        done[v] = True
        nb = nbrs[v]
        new = nb[~forb[nb, c]]
        sat[new] += 1
        forb[nb, c] = True
    return color


def greedy_clique(A: np.ndarray, starts: int | None = 100) -> list[int]:
    """Largest of the cliques grown greedily from (up to `starts`) vertices in decreasing degree order: at each step
    the candidate (adjacent to the whole clique) with the most candidate neighbours joins."""
    n = len(A)
    if n == 0:
        return []
    deg = A.sum(axis=1)
    order = np.argsort(-deg, kind="stable")
    if starts is not None:
        order = order[:starts]
    Af = A.astype(np.float32)
    best: list[int] = [int(order[0])]
    for s in order.tolist():
        if deg[s] + 1 <= len(best):
            continue
        clique = [s]
        cand = A[s].copy()
        while cand.any():
            idx = np.nonzero(cand)[0]
            inner = Af[np.ix_(idx, idx)].sum(axis=1)
            v = int(idx[np.argmax(inner)])
            clique.append(v)
            cand &= A[v]
            cand[v] = False
            if len(clique) + int(cand.sum()) <= len(best):
                break
        if len(clique) > len(best):
            best = clique
    return sorted(best)


def edge_clique_cover(A: np.ndarray) -> list[np.ndarray]:
    """Cliques (each of size >= 2) covering every edge of A: for every vertex v in order and every edge (v, u) not yet
    covered, the clique {v, u} is extended greedily by common neighbours, preferring vertices with the most uncovered
    edges into the clique. Isolated vertices are in no clique."""
    n = len(A)
    unc = A.copy()  # symmetric "uncovered" matrix
    deg = A.sum(axis=1).astype(np.int64)
    cliques: list[np.ndarray] = []
    for v in range(n):
        while unc[v].any():
            u = int(np.nonzero(unc[v])[0][0])
            K = [v, u]
            cand = A[v] & A[u]
            score = unc[v].astype(np.int32) + unc[u].astype(np.int32)
            while cand.any():
                idx = np.nonzero(cand)[0]
                w = int(idx[np.argmax(score[idx] * (n + 1) + deg[idx])])
                K.append(w)
                cand &= A[w]
                cand[w] = False
                score += unc[w]
            K = np.array(sorted(K), dtype=np.int64)
            sub = np.ix_(K, K)
            unc[sub] = False
            cliques.append(K)
    return cliques


def maximal_extension(A: np.ndarray, S: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """An independent set (bool mask) extended to a maximal one: free vertices join in decreasing weight (or degree)
    order."""
    S = S.copy()
    blocked = S | A[S].any(axis=0)
    if blocked.all():
        return S
    key = A.sum(axis=1).astype(float) if weight is None else np.asarray(weight, dtype=float)
    for v in np.argsort(-key, kind="stable").tolist():
        if not blocked[v]:
            S[v] = True
            blocked |= A[v]
            blocked[v] = True
    return S


def coloring_from_cover(n: int, sets: list[np.ndarray]) -> np.ndarray | None:
    """A coloring from a list of independent sets (index arrays or masks) covering every vertex: every vertex takes
    the first set containing it. None if some vertex is uncovered."""
    col = np.full(n, -1, dtype=np.int64)
    for k, S in enumerate(sets):
        idx = np.nonzero(S)[0] if np.asarray(S).dtype == bool else np.asarray(S, dtype=np.int64)
        free = idx[col[idx] < 0]
        col[free] = k
    return None if (col < 0).any() else np.array(canonical(col), dtype=np.int64)


class TabuCol:
    """TabuCol for a fixed number of colors k: minimizes the number of conflicting edges by single-vertex recolorings
    of conflicting vertices; the reverse move is tabu for L + 0.6 * |conflicting vertices| iterations (L uniform in
    0..9); aspiration when a move reaches a new best conflict count."""

    def __init__(self, A: np.ndarray, seed: int = 0) -> None:
        self.A = A
        self.n = len(A)
        self.Ai = A.astype(np.int32)
        self.nbrs = [np.nonzero(A[v])[0] for v in range(self.n)]
        self.rng = np.random.default_rng(seed)
        self.iterations = 0

    def solve(self, col: np.ndarray, k: int, deadline: float, max_iters: int | None = None, harvest=None,
              harvest_every: int = 5000, resume: bool = False) -> tuple[np.ndarray, int]:
        """(best assignment, its conflict count) from the start assignment col (labels in 0..k-1). With resume, a
        search interrupted by the deadline at the same k continues from its last assignment and tabu list. harvest
        (optional) receives, every harvest_every iterations, the conflict-free color classes of the current
        assignment (bool masks; independent sets), e.g. as columns for a set-covering master (Malaguti, Monaci and
        Toth 2008)."""
        n, rng = self.n, self.rng
        col = np.asarray(col, dtype=np.int64).copy()
        if k <= 1:  # no recoloring move exists
            f = int((self.A & (col[:, None] == col[None, :])).sum()) // 2
            return col, f
        state = getattr(self, "state", None)
        if resume and state is not None and state[0] == k:
            _, col, tabu, it = state
            col = col.copy()
        else:
            tabu = np.zeros((n, k), dtype=np.int64)
            it = 0
        onehot = np.zeros((n, k), dtype=np.int32)
        onehot[np.arange(n), col] = 1
        gamma = self.Ai @ onehot  # gamma[v, c] = neighbours of v with color c
        own = gamma[np.arange(n), col]
        f = int(own.sum()) // 2
        best_f, best = f, col.copy()
        start_it = it
        limit = it + (max_iters if max_iters is not None else 1 << 62)
        while f > 0 and it < limit:
            if it % 256 == 0 and time.perf_counter() >= deadline:
                break
            it += 1
            if harvest is not None and it % harvest_every == 0:
                bad = np.zeros(k, dtype=bool)
                bad[col[own > 0]] = True
                good = np.nonzero(~bad)[0]
                if len(good):
                    harvest(col[None, :] == good[:, None])
            conf = np.nonzero(own > 0)[0]
            G = gamma[conf]
            delta = G - own[conf][:, None]
            delta[np.arange(len(conf)), col[conf]] = 1 << 30
            allowed = (tabu[conf] <= it) | (f + delta < best_f)
            d = np.where(allowed, delta, 1 << 30)
            dmin = d.min()
            if dmin >= 1 << 29:  # every move tabu: take the best one anyway
                d = delta
                dmin = d.min()
            cand = np.flatnonzero(d == dmin)
            pick = int(cand[rng.integers(len(cand))]) if len(cand) > 1 else int(cand[0])
            i, c = divmod(pick, k)
            v = int(conf[i])
            old = int(col[v])
            nb = self.nbrs[v]
            gamma[nb, old] -= 1
            gamma[nb, c] += 1
            col[v] = c
            own[nb] = gamma[nb, col[nb]]
            own[v] = gamma[v, c]
            f += int(dmin)
            tabu[v, old] = it + int(rng.integers(10)) + int(0.6 * len(conf))
            if f < best_f:
                best_f, best = f, col.copy()
        self.iterations += it - start_it
        self.state = (k, col, tabu, it) if f > 0 else None
        return best, best_f

    def reduce(self, coloring: np.ndarray, deadline: float, on_coloring=None, max_iters: int | None = None,
               lower: int = 1, harvest=None, resume: bool = False) -> np.ndarray:
        """Repeatedly tries k - 1 colors from the best proper coloring (the last color class is spread over the other
        colors with the fewest conflicts) until a try fails or the deadline passes; every proper coloring found is
        passed to on_coloring. With resume, an interrupted try at the same k continues where it stopped. Returns the
        best proper coloring."""
        best = np.array(canonical(coloring), dtype=np.int64)
        while time.perf_counter() < deadline:
            k = int(best.max()) + 1 - 1
            if k < max(1, lower):
                break
            start = best.copy()
            moved = np.nonzero(start == k)[0]
            for v in moved.tolist():
                cnt = np.bincount(start[self.nbrs[v]][start[self.nbrs[v]] < k], minlength=k)
                start[v] = int(np.argmin(cnt))
            sol, f = self.solve(start, k, deadline, max_iters, harvest=harvest, resume=resume)
            if f > 0:
                break
            best = np.array(canonical(sol), dtype=np.int64)
            if on_coloring is not None:
                on_coloring(best)
        return best


def weighted_greedy_sets(B: np.ndarray, w: np.ndarray, rng: np.random.Generator, tries: int, threshold: float,
                         deg: np.ndarray | None = None) -> list[np.ndarray]:
    """Heuristic maximum-weight independent sets of the graph B for vertex weights w: greedy constructions under
    several orders (weight, weight / (degree + 1), perturbed weights, and seeded at heavy vertices), each improved by
    (1,1)-swaps and made maximal. Returns the distinct sets (bool masks) of weight > threshold, heaviest first."""
    N = len(B)
    if N == 0:
        return []
    deg = B.sum(axis=1) if deg is None else deg
    pos = w > 1e-12
    orders = [np.argsort(-w, kind="stable"), np.argsort(-(w / (deg + 1.0)), kind="stable")]
    heavy = np.argsort(-w, kind="stable")[:max(1, tries // 2)]
    for t in range(tries):
        noise = w * rng.uniform(0.6, 1.4, size=N)
        orders.append(np.argsort(-noise, kind="stable"))
    out: dict[bytes, tuple[float, np.ndarray]] = {}
    seeds: list[int | None] = [None] * len(orders) + [int(s) for s in heavy.tolist()]
    orders += [orders[0]] * len(heavy)
    for order, seed in zip(orders, seeds):
        S = np.zeros(N, dtype=bool)
        blocked = np.zeros(N, dtype=bool)
        if seed is not None:
            S[seed] = True
            blocked |= B[seed]
            blocked[seed] = True
        for v in order.tolist():
            if not blocked[v] and pos[v]:
                S[v] = True
                blocked |= B[v]
                blocked[v] = True
        S = swap_improve(B, w, S)
        S = maximal_extension(B, S, w)
        val = float(w[S].sum())
        if val > threshold:
            out.setdefault(S.tobytes(), (val, S))
    return [S for _, S in sorted(out.values(), key=lambda t: -t[0])]


def swap_improve(B: np.ndarray, w: np.ndarray, S: np.ndarray, max_steps: int = 40) -> np.ndarray:
    """(1,1)-swaps: a vertex outside S with exactly one neighbour in S replaces it when heavier; free vertices join."""
    S = S.copy()
    for _ in range(max_steps):
        cnt = B[:, S].sum(axis=1) if S.any() else np.zeros(len(B), dtype=np.int64)
        free = (~S) & (cnt == 0) & (w > 1e-12)
        if free.any():
            v = int(np.nonzero(free)[0][np.argmax(w[free])])
            S[v] = True
            continue
        one = np.nonzero((~S) & (cnt == 1))[0]
        if len(one) == 0:
            break
        partner = np.argmax(B[one] & S[None, :], axis=1)  # the unique neighbour in S
        gain = w[one] - w[partner]
        j = int(np.argmax(gain))
        if gain[j] <= 1e-9:
            break
        S[partner[j]] = False
        S[one[j]] = True
    return S
