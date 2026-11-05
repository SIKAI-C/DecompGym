"""Ruin-and-recreate routing engine without decomposition (scale-up study, protocol v4 decomposition-free controls).

A slack-induction-by-string-removals (SISR) style simulated annealing in the spirit of Christiaens and Vanden Berghe
(2020, Transportation Science 54(2)), written for one or several depots so that the CVRP control (one uncapacitated
depot) and the CLRP control (capacitated depots with opening costs) share it. Python 3 + NumPy only: no LP, no MIP,
no column generation.

Model. Nodes 0..m-1 are depots, m..m+n-1 customers (customer k is node m + k); C is the (m+n) x (m+n) cost matrix.
A plan is a list of routes [depot, [customer nodes in visiting order]]; its cost is
    sum over routes (F + C[d, first] + ... + C[last, d]) + sum over USED depots O_d,
subject to the vehicle capacity Q per route and the depot capacity W_d (sum of the loads of its routes). Every route
uses a depot of the allowed set (all depots unless restricted).

One iteration (parameters as in the paper where it states them: c_bar = 10 removed customers on average, L_max = 10,
blink rate 0.01, recreate orders random / demand / far / close with weights 4 / 4 / 2 / 1):
  ruin     l_s_max = min(L_max, mean route cardinality), k_s_max = 4 c_bar / (1 + l_s_max) - 1, k_s = floor(U(1,
           k_s_max + 1)) routes are ruined around a random seed customer, visiting its neighbours by increasing cost:
           from each touched route a string of l_t = floor(U(1, min(|t|, l_s_max) + 1)) consecutive customers holding
           the neighbour is removed, or (probability 1/2) a split string: a window of l_t + k customers of which k
           consecutive ones are kept (k = 1, grown while U < 1/2; this split rule is our own simplification);
  recreate removed customers are reinserted one by one at their cheapest feasible position (existing edge or a new
           route at an allowed depot, which pays F and, for an unused depot, O_d), each candidate position skipped
           with probability `blink`;
  accept   simulated annealing on the full plan cost with the temperature decreasing geometrically in elapsed time
           from T0 to Tf over the run (new < current - T ln U).
The best plan is optionally polished by a caller-provided local search (rate-limited) and reported through a
callback. All costs are recomputed from the matrix after every recreate (no drift).
"""
from __future__ import annotations

import math
import time
from typing import Callable, Sequence

import numpy as np

INF = float("inf")
Route = list  # [depot, [customer nodes]]


class RoutingModel:
    """The data of one multi-depot routing problem (see the module docstring)."""

    def __init__(self, C: np.ndarray, q: Sequence[float], Q: float, m: int, W: Sequence[float] | None = None,
                 F: float = 0.0, O: Sequence[float] | None = None) -> None:
        self.C = np.asarray(C, dtype=float)
        self.Cl = self.C.tolist()
        self.m, self.n = int(m), len(q)
        if self.C.shape != (self.m + self.n, self.m + self.n):
            raise ValueError("C must be (m + n) x (m + n)")
        self.q = [0.0] * self.m + [float(x) for x in q]          # demand per node (0 at depots)
        self.Q = float(Q)
        self.W = [INF] * self.m if W is None else [float(x) for x in W]
        self.F = float(F)
        self.O = [0.0] * self.m if O is None else [float(x) for x in O]
        if any(qi > self.Q for qi in self.q):
            raise ValueError("a customer demand exceeds the vehicle capacity")
        cc = self.C[self.m:, self.m:].copy()
        np.fill_diagonal(cc, -INF)                                  # each customer first in its own list
        self.adj = (np.argsort(cc, axis=1, kind="stable") + self.m).tolist()
        self.depot_dist = self.C[:self.m, self.m:].min(axis=0)      # per customer: cost to its nearest depot

    def path_cost(self, d: int, path: Sequence[int]) -> float:
        C = self.Cl
        prev, s = d, 0.0
        for v in path:
            s += C[prev][v]
            prev = v
        return s + C[prev][d]

    def cost(self, routes: Sequence[Route]) -> float:
        used, total = set(), 0.0
        for d, p in routes:
            if p:
                total += self.F + self.path_cost(d, p)
                used.add(d)
        return total + sum(self.O[d] for d in used)

    def feasible(self, routes: Sequence[Route], allowed: set[int] | None = None) -> bool:
        seen = [0] * (self.m + self.n)
        dload = [0.0] * self.m
        for d, p in routes:
            if not 0 <= d < self.m or (allowed is not None and p and d not in allowed):
                return False
            load = 0.0
            for v in p:
                if not self.m <= v < self.m + self.n:
                    return False
                seen[v] += 1
                load += self.q[v]
            if load > self.Q + 1e-9:
                return False
            dload[d] += load
        return all(s == 1 for s in seen[self.m:]) and all(dl <= w + 1e-9 for dl, w in zip(dload, self.W))


def copy_routes(routes: Sequence[Route]) -> list[Route]:
    return [[d, list(p)] for d, p in routes if p]


def insert_all(model: RoutingModel, routes: list[Route], todo: Sequence[int], allowed: Sequence[int],
               rng: np.random.Generator | None = None, blink: float = 0.0) -> list[Route] | None:
    """Cheapest feasible insertion of the customer nodes `todo`, in the given order, into `routes` (modified in place
    and returned); None if some customer has no feasible position (vehicle or depot capacity)."""
    m, C, Cl, q, Q, W = model.m, model.C, model.Cl, model.q, model.Q, model.W
    A: list[int] = []
    B: list[int] = []
    R: list[int] = []
    for r, (d, p) in enumerate(routes):
        seq = [d] + p + [d]
        A += seq[:-1]
        B += seq[1:]
        R += [r] * (len(seq) - 1)
    rload = [sum(q[v] for v in p) for _, p in routes]
    rdep = [d for d, _ in routes]
    dload = [0.0] * m
    dused = [0] * m
    for d, l in zip(rdep, rload):
        dload[d] += l
        dused[d] += 1
    for c in todo:
        qc = q[c]
        best, where = INF, None
        if A:
            a, b = np.asarray(A), np.asarray(B)
            delta = C[a, c] + C[c, b] - C[a, b]
            rl, rd = np.asarray(rload)[R], np.asarray(rdep)[R]
            ok = (rl + qc <= Q + 1e-9) & (np.asarray(dload)[rd] + qc <= np.asarray(W)[rd] + 1e-9)
            if blink > 0.0 and rng is not None:
                ok &= rng.random(len(A)) >= blink
            if ok.any():
                e = int(np.argmin(np.where(ok, delta, INF)))
                best, where = float(delta[e]), ("edge", e)
        for d in allowed:
            if dload[d] + qc > W[d] + 1e-9:
                continue
            if blink > 0.0 and rng is not None and rng.random() < blink:
                continue
            v = model.F + Cl[d][c] + Cl[c][d] + (model.O[d] if dused[d] == 0 else 0.0)
            if v < best:
                best, where = v, ("new", d)
        if where is None:
            return None
        if where[0] == "edge":
            e = where[1]
            r = R[e]
            d, p = routes[r]
            # edge e of route r is the k-th edge of its sequence: position k in the path
            k = e - R.index(r)
            p.insert(k, c)
            A.insert(e + 1, c)
            B.insert(e, c)
            R.insert(e + 1, r)
            rload[r] += qc
            dload[d] += qc
        else:
            d = where[1]
            r = len(routes)
            routes.append([d, [c]])
            A += [d, c]
            B += [c, d]
            R += [r, r]
            rload.append(qc)
            rdep.append(d)
            dload[d] += qc
            dused[d] += 1
    return routes


class RuinRecreate:
    """The SISR-style simulated annealing of the module docstring."""

    def __init__(self, model: RoutingModel, seed: int = 0, c_bar: float = 10.0, l_max: float = 10.0,
                 blink: float = 0.01, split_grow: float = 0.5, T0: float = 100.0, Tf: float = 1.0) -> None:
        self.model, self.rng = model, np.random.default_rng(seed)
        self.c_bar, self.l_max, self.blink, self.split_grow = c_bar, l_max, blink, split_grow
        self.T0, self.Tf = T0, Tf
        self.iterations = 0
        self.accepted = 0

    def ruin(self, routes: list[Route]) -> tuple[list[Route], list[int]]:
        rng, model = self.rng, self.model
        routes = copy_routes(routes)
        where = {}
        for r, (_, p) in enumerate(routes):
            for v in p:
                where[v] = r
        avg = model.n / max(1, len(routes))
        ls_max = min(self.l_max, avg)
        ks_max = 4.0 * self.c_bar / (1.0 + ls_max) - 1.0
        ks = int(rng.random() * ks_max) + 1
        seed = model.m + int(rng.integers(model.n))
        removed: list[int] = []
        gone: set[int] = set()
        ruined: set[int] = set()
        for c in model.adj[seed - model.m]:
            if len(ruined) >= ks:
                break
            if c in gone:
                continue
            r = where[c]
            if r in ruined:
                continue
            path = routes[r][1]
            L = len(path)
            lt = int(rng.random() * min(L, ls_max)) + 1
            pos = path.index(c)
            if L <= lt or rng.random() < 0.5:   # string
                start = int(rng.integers(max(0, pos - lt + 1), min(pos, L - lt) + 1))
                rem = path[start:start + lt]
                keep = path[:start] + path[start + lt:]
            else:                                # split string: a window of lt + k, k consecutive customers kept
                k = 1
                while lt + k < L and rng.random() < self.split_grow:
                    k += 1
                w = lt + k
                start = int(rng.integers(max(0, pos - w + 1), min(pos, L - w) + 1))
                off = int(rng.integers(0, lt + 1))
                window = path[start:start + w]
                rem = window[:off] + window[off + k:]
                keep = path[:start] + window[off:off + k] + path[start + w:]
            routes[r][1] = keep
            removed += rem
            gone.update(rem)
            ruined.add(r)
        return [rt for rt in routes if rt[1]], removed

    def order(self, removed: list[int]) -> list[int]:
        rng, model = self.rng, self.model
        u = rng.random() * 11.0
        if u < 4.0:
            rng.shuffle(removed)
            return removed
        if u < 8.0:
            return sorted(removed, key=lambda v: -model.q[v])
        if u < 10.0:
            return sorted(removed, key=lambda v: -model.depot_dist[v - model.m])
        return sorted(removed, key=lambda v: model.depot_dist[v - model.m])

    def run(self, start: Sequence[Route], deadline: float, *, allowed: Sequence[int] | None = None,
            on_best: Callable[[list[Route], float], None] | None = None,
            polish: Callable[[list[Route], float], list[Route]] | None = None, polish_every: float = 2.0,
            max_iterations: int | None = None, tick: Callable[[], None] | None = None) -> tuple[list[Route], float]:
        """Anneal from `start` until `deadline` (time.perf_counter) or `max_iterations`; returns the best plan and its
        cost. `on_best(plan, cost)` is called for every new best plan (before and, if it improves, after the optional
        polish); `tick()` once per
        iteration (e.g. to flush a rate-limited publication)."""
        model = self.model
        allowed = list(range(model.m)) if allowed is None else sorted(set(allowed))
        allowed_set = set(allowed)
        cur = copy_routes(start)
        if not model.feasible(cur, allowed_set):
            raise ValueError("the start plan is infeasible or uses a depot outside the allowed set")
        cur_cost = model.cost(cur)
        best, best_cost = copy_routes(cur), cur_cost
        t_start = time.perf_counter()
        span = max(1e-6, deadline - t_start)
        last_polish = -INF
        it = 0
        while True:
            now = time.perf_counter()
            if now >= deadline or (max_iterations is not None and it >= max_iterations):
                break
            it += 1
            if tick is not None:
                tick()
            frac = min(1.0, (now - t_start) / span)
            temp = self.T0 * (self.Tf / self.T0) ** frac
            partial, removed = self.ruin(cur)
            new = insert_all(model, partial, self.order(removed), allowed, self.rng, self.blink)
            if new is None:
                continue
            new_cost = model.cost(new)
            if new_cost < cur_cost - temp * math.log(max(self.rng.random(), 1e-300)):
                cur, cur_cost = new, new_cost
                self.accepted += 1
                if cur_cost < best_cost - 1e-9:
                    # reported BEFORE the optional polish, so a polish overrunning the deadline loses nothing
                    best, best_cost = copy_routes(cur), cur_cost
                    if on_best is not None:
                        on_best(copy_routes(best), best_cost)
                    if polish is not None and now - last_polish >= polish_every:
                        last_polish = now
                        pol = polish(copy_routes(cur), deadline)
                        if pol is not None and model.feasible(pol, allowed_set) and model.cost(pol) < cur_cost - 1e-9:
                            cur, cur_cost = copy_routes(pol), model.cost(pol)
                            best, best_cost = copy_routes(cur), cur_cost
                            if on_best is not None:
                                on_best(copy_routes(best), best_cost)
        self.iterations += it
        return best, best_cost
