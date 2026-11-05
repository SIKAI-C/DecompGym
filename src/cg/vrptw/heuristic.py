"""Shared VRPTW primal helpers (scale-up study): the basic initializer shared by the naive and the engineered column
generation, and the engineered reference's local search. Routes are lists of customer numbers 1..n; all times and
distances are a_data's integer tenths, and every plan is re-checked by a_data.plan_issues before it is published.
"""
from __future__ import annotations

import time

import numpy as np

from src.cg.vrptw.a_data import VRPTWInstance, route_schedule


def basic_initializer(inst: VRPTWInstance, mu: float = 1.0, lam: float = 1.0, alpha1: float = 0.5,
                      customers: list[int] | None = None) -> list[list[int]]:
    """Solomon's (1987) sequential insertion heuristic I1, deterministic: each route starts from the unrouted customer
    farthest from the depot; then the unrouted customer u maximizing lam * d(0, u) - c1(u) is inserted at its best
    feasible position, c1 = alpha1 (d(a, u) + d(u, b) - mu d(a, b)) + (1 - alpha1) (push-forward of the start at b),
    until no customer fits; then a new route starts. customers: route only these (default: all)."""
    n, Q = inst.n, int(inst.capacity)
    d = inst.dist.astype(np.int64)
    e, l, s, q = inst.ready.astype(np.int64), inst.due.astype(np.int64), inst.service.astype(np.int64), inst.demand.astype(np.int64)
    unrouted = np.zeros(n + 1, dtype=bool)
    unrouted[1:] = True if customers is None else False
    if customers is not None:
        unrouted[list(customers)] = True
    unrouted[0] = False
    routes: list[list[int]] = []
    while unrouted.any():
        cand = np.flatnonzero(unrouted)
        order = cand[np.lexsort((cand, -d[0, cand]))]
        seed = next((int(u) for u in order if route_schedule(inst, [int(u)]) is not None and q[u] <= Q), None)
        if seed is None:
            raise ValueError("a customer cannot be served by any route")
        route = [seed]
        unrouted[seed] = False
        load = int(q[seed])
        while True:
            U = np.flatnonzero(unrouted & (q + load <= Q))
            if not len(U):
                break
            a = np.array([0] + route)
            b = np.array(route + [0])
            # Service starts along the route (depot departure = e0) and latest starts keeping the suffix feasible.
            start = np.zeros(len(route) + 2, dtype=np.int64)
            start[0] = e[0]
            seq = [0] + route + [0]
            for k in range(1, len(seq)):
                start[k] = max(start[k - 1] + s[seq[k - 1]] + d[seq[k - 1], seq[k]], e[seq[k]] if k < len(seq) - 1 else 0)
            latest = np.zeros(len(seq), dtype=np.int64)
            latest[-1] = l[0]
            for k in range(len(seq) - 2, 0, -1):
                latest[k] = min(l[seq[k]], latest[k + 1] - s[seq[k]] - d[seq[k], seq[k + 1]])
            sa, sb, lb_ = start[:-1], start[1:], latest[1:]
            arr = sa[None, :] + s[a][None, :] + d[a[None, :], U[:, None]]
            su = np.maximum(e[U][:, None], arr)
            nb_ = su + s[U][:, None] + d[U[:, None], b[None, :]]
            nb_ = np.where(b[None, :] == 0, nb_, np.maximum(e[b][None, :], nb_))
            ok = (su <= l[U][:, None]) & (nb_ <= lb_[None, :])
            if not ok.any():
                break
            c11 = d[a[None, :], U[:, None]] + d[U[:, None], b[None, :]] - mu * d[a, b][None, :]
            c12 = nb_ - sb[None, :]
            c1 = np.where(ok, alpha1 * c11 + (1.0 - alpha1) * c12, np.inf)
            pos = c1.argmin(axis=1)
            best = c1[np.arange(len(U)), pos]
            c2 = np.where(np.isfinite(best), lam * d[0, U] - best, -np.inf)
            t = int(np.argmax(c2))
            u = int(U[t])
            route.insert(int(pos[t]), u)
            unrouted[u] = False
            load += int(q[u])
        routes.append(route)
    return routes


class LocalSearch:
    """Relocate (one customer, or a segment of 2-3: or-opt; inside a route or between routes), swap and 2-opt* (tail
    exchange), all granular (moves only towards a customer's `near` closest customers), best improvement per
    customer, repeated until no move improves. Every candidate route is evaluated exactly (integer tenths, windows
    and capacity)."""

    def __init__(self, inst: VRPTWInstance, near: int = 12) -> None:
        self.inst = inst
        self.n, self.Q = inst.n, int(inst.capacity)
        self.q = inst.demand.astype(int).tolist()
        d = inst.dist.astype(float).copy()
        np.fill_diagonal(d, np.inf)
        d[0, :] = np.inf
        d[:, 0] = np.inf
        rows = np.argsort(d, axis=1, kind="stable")[:, :max(1, min(near, inst.n - 1))].tolist()
        self.near = [[v for v in row if v != 0 and v != u] for u, row in enumerate(rows)]

    def length(self, route: list[int]) -> int | None:
        if not route:
            return 0
        if sum(self.q[i] for i in route) > self.Q:
            return None
        sch = route_schedule(self.inst, route)
        return None if sch is None else sch[0]

    def improve(self, routes: list[list[int]], deadline: float) -> list[list[int]]:
        R = [list(r) for r in routes if r]
        cost = [self.length(r) for r in R]
        if any(c is None for c in cost):
            return R
        where = {}
        for k, r in enumerate(R):
            for i in r:
                where[i] = k
        improved = True
        while improved and time.perf_counter() < deadline:
            improved = False
            for u in range(1, self.n + 1):
                if time.perf_counter() > deadline:
                    break
                ra = where[u]
                A = R[ra]
                pa = A.index(u)
                A_wo = A[:pa] + A[pa + 1:]
                c_wo = self.length(A_wo)
                best = None  # (delta, new routes {index: route})
                cands = {where[v] for v in self.near[u]}
                for v in self.near[u]:
                    rb = where[v]
                    B = R[rb]
                    pb = B.index(v)
                    # Relocate u next to v (before or after), inside the same route or into another one.
                    if rb == ra:
                        base = A_wo
                        pv = base.index(v)
                        for pos in (pv, pv + 1):
                            nr = base[:pos] + [u] + base[pos:]
                            if nr == A:
                                continue
                            c = self.length(nr)
                            if c is not None and (best is None or c - cost[ra] < best[0]):
                                if c - cost[ra] < 0:
                                    best = (c - cost[ra], {ra: nr})
                    elif c_wo is not None:
                        for pos in (pb, pb + 1):
                            nr = B[:pos] + [u] + B[pos:]
                            c = self.length(nr)
                            if c is not None:
                                delta = c_wo + c - cost[ra] - cost[rb]
                                if delta < 0 and (best is None or delta < best[0]):
                                    best = (delta, {ra: A_wo, rb: nr})
                    if rb != ra:
                        # Swap u and v.
                        na = A[:pa] + [v] + A[pa + 1:]
                        nb = B[:pb] + [u] + B[pb + 1:]
                        ca, cb = self.length(na), self.length(nb)
                        if ca is not None and cb is not None:
                            delta = ca + cb - cost[ra] - cost[rb]
                            if delta < 0 and (best is None or delta < best[0]):
                                best = (delta, {ra: na, rb: nb})
                        # 2-opt*: u -> v (A keeps its head up to u and takes B's tail from v), and v's predecessor
                        # continues with A's tail.
                        na = A[:pa + 1] + B[pb:]
                        nb = B[:pb] + A[pa + 1:]
                        ca, cb = self.length(na), self.length(nb)
                        if ca is not None and cb is not None:
                            delta = ca + cb - cost[ra] - cost[rb]
                            if delta < 0 and (best is None or delta < best[0]):
                                best = (delta, {ra: na, rb: nb})
                del cands
                # Or-opt: the segment of 2 or 3 customers starting at u moves next to a near customer v.
                for size in (2, 3):
                    if pa + size > len(A):
                        break
                    seg = A[pa:pa + size]
                    rest = A[:pa] + A[pa + size:]
                    c_rest = self.length(rest)
                    if c_rest is None:
                        continue
                    for v in self.near[u]:
                        if v in seg:
                            continue
                        rb = where[v]
                        base = rest if rb == ra else R[rb]
                        pv = base.index(v)
                        for pos in (pv, pv + 1):
                            nr = base[:pos] + seg + base[pos:]
                            c = self.length(nr)
                            if c is None:
                                continue
                            delta = c - cost[ra] if rb == ra else c_rest + c - cost[ra] - cost[rb]
                            if delta < 0 and (best is None or delta < best[0]):
                                best = (delta, {ra: nr} if rb == ra else {ra: rest, rb: nr})
                if best is None:
                    continue
                for k, r in best[1].items():
                    R[k] = r
                    cost[k] = self.length(r)
                    for i in r:
                        where[i] = k
                improved = True
            if improved:
                keep = [k for k, r in enumerate(R) if r]
                if len(keep) < len(R):
                    R = [R[k] for k in keep]
                    cost = [cost[k] for k in keep]
                    where = {i: k for k, r in enumerate(R) for i in r}
        return [r for r in R if r]
