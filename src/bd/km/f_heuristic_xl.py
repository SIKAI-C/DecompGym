"""k-median (scale-up): decomposition-free controls and the swap local search shared with the engineered Benders.

- `SwapSearch`: best-improvement swap (interchange) local search in the fast-interchange form of Whitaker (1983) /
  Resende and Werneck (2003): with d1_j, d2_j the distances to the nearest and second-nearest open facility and
  c1_j the nearest one, inserting a closed facility i and removing an open r changes the cost by
      delta(i, r) = -sum_j (d1_j - D_ij)^+ + sum_{j: c1_j = r} (min(D_ij, d2_j) - d1_j)^+ ,
  evaluated for all pairs at once in O(m n) per move (one m x n pass and a grouped sum over the customers of each
  open facility). Iterated local search: random q-swap kicks (q in {1, 2, 3}), descent, keep the best.
- `KMLocalSearch` (primal control, no decomposition, no bound): the shared initializer (c_naive_bd_xl.greedy_design),
  descent, then ILS to the deadline; every new best design is published immediately.
- `KMLagrangian` (bound diagnostic): relax the assignment rows sum_i x_ij = 1 of the classical formulation with
  multipliers u_j:
      L(u) = sum_j u_j + min_{|S| = k} sum_{i in S} rho_i,   rho_i = -sum_j (u_j - D_ij)^+ ,
  (the k smallest rho_i), maximized by subgradient steps (Polyak step lam (UB - L) / |g|^2, g_j = 1 - #{i in S:
  D_ij < u_j}, lam halved after 30 non-improving steps; start u = the initializer's assignment costs). The
  Lagrangian problem has the integrality property, so max L(u) is the LP bound. u is also a y-space certificate:
  the cuts (j, u_j) give a master LP whose value is at least L(u) (g_certificate_xl); snapshots of the best u are
  kept on a geometric time grid for deferred certification. The Lagrangian's designs S are evaluated and published
  (a Lagrangian heuristic, no local search). With pool=True every snapshot publishes the cumulative multiplier pool
  (all snapshots so far, most recent first, within the size limit) as a certificate version.
- Trivial-certificate controls (Codex reviews 42 #4, 46 #1), published through the same channel and checker:
  `KMStaticCertificate` ('empty', 'analytical' = analytical_cuts, 'oneshot_greedy'), the one-shot certificate of the
  local-search control's best design (KMLocalSearch), and the deterministic no-feedback pool `KMNoFeedbackPool`.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from src.bd.km.a_data_xl import load_instance
from src.bd.km.b_mip_xl import design_value, open_list
from src.bd.km.c_naive_bd_xl import INF, certify_float, greedy_design


class SwapSearch:
    def __init__(self, D: np.ndarray, k: int, seed: int = 0) -> None:
        self.D = D
        self.m, self.n = D.shape
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.moves = 0
        self.seconds = 0.0

    def state(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.nonzero(y)[0]
        sub = self.D[idx]
        cols = np.arange(self.n)
        if len(idx) == 1:
            return np.full(self.n, idx[0]), sub[0].copy(), np.full(self.n, INF)
        p2 = np.argpartition(sub, 1, axis=0)[:2]
        a, b = sub[p2[0], cols], sub[p2[1], cols]
        first = np.where((a < b) | ((a == b) & (idx[p2[0]] < idx[p2[1]])), p2[0], p2[1])
        return idx[first], np.minimum(a, b), np.maximum(a, b)

    def best_swap(self, y: np.ndarray, c1: np.ndarray, d1: np.ndarray, d2: np.ndarray) -> tuple[float, int, int]:
        D = self.D
        G = np.maximum(d1[None, :] - D, 0.0).sum(axis=1)
        M = np.maximum(np.minimum(D, d2[None, :]) - d1[None, :], 0.0)
        order = np.argsort(c1, kind="stable")
        cs = c1[order]
        starts = np.nonzero(np.r_[True, cs[1:] != cs[:-1]])[0]
        Lsum = np.add.reduceat(M[:, order], starts, axis=1)
        opened = np.nonzero(y)[0]
        loss = np.zeros((self.m, len(opened)))
        pos = np.searchsorted(opened, cs[starts])
        loss[:, pos] = Lsum
        delta = loss - G[:, None]
        delta[y] = INF
        flat = int(np.argmin(delta))
        i, r = divmod(flat, len(opened))
        return float(delta[i, r]), i, int(opened[r])

    def descend(self, y: np.ndarray, deadline: float, publish: Callable | None = None) -> tuple[np.ndarray, float]:
        """Best-improvement descent from y (bool design) until a local optimum or the deadline."""
        t = time.perf_counter()
        y = y.copy()
        c1, d1, d2 = self.state(y)
        value = float(d1.sum())
        while time.perf_counter() < deadline:
            delta, i, r = self.best_swap(y, c1, d1, d2)
            if not delta < -1e-9 * max(1.0, abs(value)):
                break
            y[i], y[r] = True, False
            c1, d1, d2 = self.state(y)
            value = float(d1.sum())
            self.moves += 1
            if publish is not None:
                publish(y, value)
        self.seconds += time.perf_counter() - t
        return y, value

    def kick(self, y: np.ndarray) -> np.ndarray:
        y = y.copy()
        opened, closed = np.nonzero(y)[0], np.nonzero(~y)[0]
        q = int(self.rng.integers(1, min(3, len(opened), len(closed)) + 1)) if len(closed) else 0
        if q:
            y[self.rng.choice(opened, q, replace=False)] = False
            y[self.rng.choice(closed, q, replace=False)] = True
        return y

    def ils(self, y: np.ndarray, value: float, deadline: float, publish: Callable | None = None,
            on_kick: Callable | None = None) -> tuple[np.ndarray, float, int]:
        best, bval, kicks = y.copy(), value, 0
        while time.perf_counter() < deadline and self.m > self.k:
            z, zv = self.descend(self.kick(best), deadline)
            kicks += 1
            if on_kick is not None:
                on_kick()
            if zv < bval - 1e-9 * max(1.0, abs(bval)):
                best, bval = z, zv
                if publish is not None:
                    publish(best, bval)
        return best, bval, kicks


def fit_singles(D: np.ndarray, sc, cert, limit: int):
    """The longest prefix of a single-cut certificate within the frozen expanded-size limit."""
    from src.bd.km.g_certificate_xl import fit_budget, singles, sorted_counts
    keep = fit_budget(sorted_counts(D, sc), cert.single_j, cert.single_v, limit)
    return singles(cert.single_j[keep], cert.single_v[keep])


class _Control:
    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, on_certificate: Callable | None = None, soft_mem_gb: float | None = None,
                 seed: int = 0, verbose: bool = False, cert_interval: float = 10.0, cert_limit: int = 2_000_000) -> None:
        self.time_limit, self.seed, self.verbose = time_limit, seed, verbose
        self.on_incumbent = on_incumbent or (lambda design, stage=None: None)
        self.on_bound = on_bound or (lambda bound: None)
        self.on_certificate = on_certificate or (lambda cert: None)
        self.cert_interval, self.cert_limit = cert_interval, cert_limit
        self._last_cert, self._cert_seq, self._cert_key = -INF, 0, None

    def offer(self, y: np.ndarray, value: float, stage: str) -> None:
        if value < self.ub - 1e-9 * max(1.0, abs(value)) and time.perf_counter() < self.deadline:
            self.ub, self.best = value, y.copy()
            self.on_incumbent(open_list(y), stage=stage)

    def emit(self, cert, key=None, force: bool = False) -> bool:
        """Publish a certificate version (rate-limited; skipped if `key` equals the last published one)."""
        now = time.perf_counter()
        if now >= self.deadline or (key is not None and key == self._cert_key):
            return False
        if not force and now - self._last_cert < self.cert_interval:
            return False
        self._cert_seq += 1
        cert.version = self._cert_seq
        self.on_certificate(cert)
        self._last_cert, self._cert_key = now, key
        return True


class KMLocalSearch(_Control):
    """Heuristic-only primal control; see the module docstring. It also publishes the one-shot certificate of its
    current best design (the 'one-shot ILS' trivial-certificate control), at most every cert_interval seconds."""

    def maybe_publish(self, D: np.ndarray) -> None:
        if self.best is not None:
            from src.bd.km.g_certificate_xl import one_shot
            key = self.best.tobytes()
            if key != self._cert_key and time.perf_counter() - self._last_cert >= self.cert_interval:
                self.emit(fit_singles(D, self.sc, one_shot(D, open_list(self.best)), self.cert_limit), key)

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        inst = load_instance(data_path)
        D = inst.distances()
        from src.bd.km.a_data_xl import SortedCosts
        self.sc = SortedCosts(D, inst.k)
        self.ub, self.best = INF, None
        y = greedy_design(D, inst.k)
        self.offer(y, design_value(D, open_list(y)), "initializer")
        ss = SwapSearch(D, inst.k, seed=self.seed)
        y, v = ss.descend(y, self.deadline, lambda z, val: self.offer(z, val, "descent"))
        self.maybe_publish(D)
        t_local = time.perf_counter() - self.t0
        y, v, kicks = ss.ils(y, v, self.deadline, lambda z, val: self.offer(z, val, "ils"), lambda: self.maybe_publish(D))
        return {"best_objective": self.ub if self.ub < INF else None, "global_lb": None, "termination_reason": "time_limit",
                "design": open_list(self.best) if self.best is not None else None, "moves": ss.moves, "kicks": kicks,
                "first_local_optimum_seconds": t_local, "total_seconds": time.perf_counter() - self.t0}


class KMLagrangian(_Control):
    """Subgradient Lagrangian bound diagnostic; see the module docstring."""

    def __init__(self, *, snapshot_ratio: float = 2 ** 0.5, eval_every: int = 10, pool: bool = True, **kw) -> None:
        super().__init__(**kw)
        self.snapshot_ratio, self.eval_every, self.pool = snapshot_ratio, eval_every, pool

    def publish_pool(self, D: np.ndarray, sc, snaps: list) -> None:
        """Cumulative multiplier pool (Codex review 46 #4): the cuts (j, min(u_j, delta_j)) and (j, u_j) of every
        snapshot u so far, most recent first, within the frozen size limit (pool=False: the latest snapshot only)."""
        from src.bd.km.g_certificate_xl import fit_budget, singles, sorted_counts
        n = D.shape[1]
        delta = sc.ds[:, -1]
        use = snaps[::-1] if self.pool else snaps[-1:]
        js, vs = [], []
        for _, u, _ in use:
            u = np.asarray(u, dtype=float)
            js += [np.arange(n), np.arange(n)]
            vs += [np.minimum(u, delta), u]
        j, v = np.concatenate(js), np.concatenate(vs)
        keep = fit_budget(sorted_counts(D, sc), j, v, self.cert_limit)
        self.emit(singles(j[keep], v[keep]), force=True)

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        inst = load_instance(data_path)
        D = inst.distances()
        k = inst.k
        self.ub, self.best = INF, None
        from src.bd.km.a_data_xl import SortedCosts
        sc = SortedCosts(D, k)
        y = greedy_design(D, k)
        c = D[y].min(axis=0)
        self.offer(y, float(c.sum()), "initializer")
        u = c.copy()
        best_L, best_u = -INF, u.copy()
        lam, stall, it = 2.0, 0, 0
        snaps: list[tuple[float, list[float], float]] = []
        next_snap = 0.25
        bounds: list[tuple[float, float]] = []
        reason = "time_limit"
        last = (0.0, best_u, best_L)
        while True:
            now = time.perf_counter()
            if now >= self.deadline:
                break
            P = np.maximum(u[None, :] - D, 0.0)
            rho = -P.sum(axis=1)
            S = np.argpartition(rho, k - 1)[:k] if k < inst.m else np.arange(inst.m)
            L = float(u.sum() + rho[S].sum())
            it += 1
            if L > best_L + 1e-12 * max(1.0, abs(L)):
                best_L, best_u, stall = L, u.copy(), 0
                if time.perf_counter() < self.deadline:
                    self.on_bound(certify_float(L))
                    bounds.append((round(time.perf_counter() - self.t0, 3), L))
            else:
                stall += 1
                if stall >= 30:
                    lam, stall = max(lam / 2.0, 1e-8), 0
            el = time.perf_counter() - self.t0
            last = (el, best_u, best_L)
            if el >= next_snap:
                snaps.append((el, best_u.tolist(), best_L))
                while next_snap <= el:
                    next_snap *= self.snapshot_ratio
                self.publish_pool(D, sc, snaps)
            g = 1.0 - (P[S] > 0).sum(axis=0)
            if it % self.eval_every == 1:
                ys = np.zeros(inst.m, dtype=bool)
                ys[S] = True
                self.offer(ys, design_value(D, S), "lagrangian")
            gg = float(g @ g)
            if gg == 0.0:
                reason = "subgradient_zero"
                break
            gap = max(self.ub - L, 1e-6 * max(1.0, abs(L)))
            u = u + (lam * gap / gg) * g
        el = time.perf_counter() - self.t0
        if best_L > -INF and (not snaps or last[0] > snaps[-1][0]):
            snaps.append((last[0], last[1].tolist(), last[2]))
            if time.perf_counter() < self.deadline:   # early end (zero subgradient): publish the final pool too
                self.publish_pool(D, sc, snaps)
        return {"best_objective": self.ub if self.ub < INF else None, "global_lb": best_L if best_L > -INF else None,
                "bound_kind": "solver_reported_lagrangian", "termination_reason": reason,
                "design": open_list(self.best) if self.best is not None else None, "iterations": it, "final_step": lam,
                "total_seconds": el, "bound_trace": bounds[-200:],
                "u_snapshots": [{"t": t, "v": v, "L": Lv} for t, v, Lv in snaps]}


# ---------------------------------------------------------------- trivial-certificate controls (Codex review 42 #4, 46 #1)

def analytical_cuts(D: np.ndarray, sc, limit: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Frozen analytical cut set: for every customer the radius cuts (j, v) at the distinct distance levels of its R
    nearest facilities (v = its r-th smallest distance, r = 1..R, the first occurrence of each level), R the largest
    common rank whose expanded size (1 + #{i: D_ij < v} = r per cut) fits the limit. No optimization, no point of
    any master: the truncated radius formulation, written as y-space cuts. Returns (j, v, R)."""
    n, L = sc.n, sc.L
    js, vs, size, R = [], [], 0, 0
    for r in range(1, L + 1):
        new = np.ones(n, dtype=bool) if r == 1 else sc.ds[:, r - 1] != sc.ds[:, r - 2]
        add = int(new.sum()) * r
        if size + add > limit:
            break
        js.append(np.nonzero(new)[0])
        vs.append(sc.ds[new, r - 1])
        size += add
        R = r
    if not js:
        return np.zeros(0, dtype=np.int64), np.zeros(0), 0
    return np.concatenate(js), np.concatenate(vs), R


def capped_simplex(w: np.ndarray, k: float) -> np.ndarray:
    """y in [0, 1]^m with sum y = k proportional to w > 0 where not capped."""
    y = k * w / w.sum()
    for _ in range(200):
        over = y > 1.0
        if not over.any():
            break
        excess = float((y[over] - 1.0).sum())
        y[over] = 1.0
        free = y < 1.0
        y[free] += excess * y[free] / y[free].sum()
    return np.minimum(y, 1.0)


class KMStaticCertificate(_Control):
    """Trivial-certificate controls published once at start: kind 'empty' (no cuts), 'analytical'
    (analytical_cuts), 'oneshot_greedy' (one cut per customer at the shared initializer's design)."""

    def __init__(self, *, kind: str = "empty", **kw) -> None:
        super().__init__(**kw)
        self.kind = kind

    def run(self, data_path: str) -> dict[str, Any]:
        from src.bd.km.a_data_xl import SortedCosts
        from src.bd.km.g_certificate_xl import Certificate, one_shot, singles
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        inst = load_instance(data_path)
        D = inst.distances()
        self.ub, self.best = INF, None
        info: dict[str, Any] = {"kind": self.kind}
        if self.kind == "empty":
            cert = Certificate()
        elif self.kind == "analytical":
            j, v, R = analytical_cuts(D, SortedCosts(D, inst.k), self.cert_limit)
            cert = singles(j, v)
            info.update(rank_R=R, cuts=len(j))
        elif self.kind == "oneshot_greedy":
            y = greedy_design(D, inst.k)
            self.offer(y, design_value(D, open_list(y)), "initializer")
            cert = fit_singles(D, SortedCosts(D, inst.k), one_shot(D, open_list(y)), self.cert_limit)
        else:
            raise ValueError(f"unknown static certificate {self.kind}")
        self.emit(cert, force=True)
        info["publish_seconds"] = time.perf_counter() - self.t0
        return {"best_objective": self.ub if self.ub < INF else None, "global_lb": None, "termination_reason": "published",
                "design": open_list(self.best) if self.best is not None else None, "total_seconds": time.perf_counter() - self.t0,
                **info}


class KMNoFeedbackPool(_Control):
    """Deterministic no-feedback cut pool (frozen recipe, seed 20261103): cuts from the closed-form oracle at points
    chosen WITHOUT any master solution: (1) the shared initializer's design, (2) the uniform point y = k/m,
    then alternately (3) a random design (k facilities, numpy default_rng(seed).choice) and (4) a random fractional
    point (k * Dirichlet(1, ..., 1), capped to [0, 1] with the excess redistributed). New (j, v) cuts are added in
    customer order until the frozen size limit is reached; a version is published after each point (at most every
    cert_interval seconds) and when the pool is full."""

    def __init__(self, *, pool_seed: int = 20261103, **kw) -> None:
        kw.setdefault("cert_interval", 5.0)
        super().__init__(**kw)
        self.pool_seed = pool_seed

    def run(self, data_path: str) -> dict[str, Any]:
        from src.bd.km.c_naive_bd_xl import Oracle
        from src.bd.km.g_certificate_xl import singles
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        inst = load_instance(data_path)
        D = inst.distances()
        m, n, k = inst.m, inst.n, inst.k
        self.ub, self.best = INF, None
        oracle = Oracle(D, k)
        rng = np.random.default_rng(self.pool_seed)
        seen: set[tuple[int, float]] = set()
        js: list[int] = []
        vs: list[float] = []
        size, points, full = 0, 0, False
        y0 = greedy_design(D, k)
        self.offer(y0, design_value(D, open_list(y0)), "initializer")
        t = 0
        while not full and time.perf_counter() < self.deadline:
            if t == 0:
                point = y0.astype(float)
            elif t == 1:
                point = np.full(m, k / m)
            elif t % 2 == 0:
                point = np.zeros(m)
                point[rng.choice(m, k, replace=False)] = 1.0
            else:
                point = capped_simplex(rng.dirichlet(np.ones(m)), k)
            t += 1
            v, _, cnt = oracle.separate(point)
            points += 1
            for j in range(n):
                key = (j, float(v[j]))
                if key in seen:
                    continue
                c = 1 + int(cnt[j])
                if size + c > self.cert_limit:
                    full = True
                    break
                seen.add(key)
                js.append(j)
                vs.append(float(v[j]))
                size += c
            self.emit(singles(js, vs), force=full)
        if not full:
            self.emit(singles(js, vs), force=True)
        return {"best_objective": self.ub if self.ub < INF else None, "global_lb": None,
                "termination_reason": "pool_full" if full else "time_limit", "design": open_list(self.best),
                "points": points, "cuts": len(js), "expanded_size": size, "total_seconds": time.perf_counter() - self.t0}
