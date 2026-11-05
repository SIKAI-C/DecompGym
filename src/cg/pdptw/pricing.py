"""Shared PDPTW pricing oracle and basic initializer (scale-up study).

Pricing problem (elementary shortest path with pickup and delivery, time windows and capacity; Ropke and Cordeau 2009):
find routes 0 -> ... -> 0 of minimum reduced cost  const + sum of arc costs - sum of the prizes of the requests picked
up (a request's covering dual sits on its pickup), where a delivery may be visited only while its request is open
(picked up, not yet delivered) and a route may return to the depot only with no open request.

Labels: (node, start time, reduced cost, open set O, unreachable set U, load). U holds the requests already picked up
and those whose pickup can no longer be served in time (thresholds include the rounding slack of
`a_data.insertion_slack`, so a request marked unreachable really is). Labels are processed in nondecreasing start
time (every extension is at least as late as its parent), so a label is compared only with the labels already kept
at its node, all of which start no later. L1 dominates L2 at the same node if rc1 <= rc2, U1 subset of U2 and either
O1 = O2 or (subset mode) O1 subset of O2 where every request in O2 \\ O1 has a delivery whose omission never makes a
path later (insertion slack 0) and rc1 + sum of their skip slacks (`a_data.skip_slack`, at most one unit each)
<= rc2: L1 can follow any completion of L2, skipping those deliveries, at no greater time or reduced cost. Subset mode
needs the static arc set (forbidden arcs break the skipping argument); with forbidden arcs use strict=True.

Subset dominance is checked for open sets missing one or two such requests (a safe restriction).

ng mode (ng=k, used by the engineered reference's exact rounds): the label keeps an ng-memory M of requests instead of
U (after visiting node j, M = (M & NG(j)) | {its request if j is a pickup}, NG(j) = the request of j and the k - 1
requests with the nearest pickups); a pickup is forbidden while its request is in M or open. This prices over
ng-routes, a superset of the elementary routes (a route may pick a request up again after delivering it and
forgetting it), so its minimum reduced cost never exceeds the elementary one. Dominance: O1 = O2, rc1 <= rc2 and every
request in M1 but not in M2 is unreachable for L2 (subset dominance on O is not used: skipping a delivery changes the
memories).

A label is discarded when its reduced cost plus a lower bound on its completion (the return distance c_j0 minus the
total skip slack, minus the nonnegative prizes of the requests it may still pick up) is not negative, so the minimum
reduced cost is exact whenever it is negative.

exact=True: all arcs, exact dominance; the round is complete unless the deadline or the label cap stops it, and then
`min_rc` is the minimum reduced cost over all routes (the oracle is exact). exact=False (heuristic): pickup successors
limited to the `near` best by arc cost minus prize, dominance ignores U, at most `beam` labels kept per node.
"""
from __future__ import annotations

import heapq
import time
from bisect import bisect_left, insort
from typing import Any

import numpy as np

from src.cg.pdptw.a_data import PDPTWInstance, feasible_arcs, insertion_slack, route_check, skip_slack

INF = float("inf")


class PDPPricer:
    def __init__(self, inst: PDPTWInstance, dist: np.ndarray, arcs: np.ndarray | None = None) -> None:
        self.inst = inst
        N, n = inst.N, inst.n
        self.N, self.n, self.Q = N, n, int(inst.capacity)
        self.c = dist.astype(np.int64).tolist()
        self.e = [int(v) for v in inst.ready]
        self.l = [int(v) for v in inst.due]
        self.s = [int(v) for v in inst.service]
        self.req = list(inst.request_of)
        self.isP = list(inst.is_pickup)
        self.q = list(inst.q)
        self.P, self.D = list(inst.pickups), list(inst.deliveries)
        self.arcs = feasible_arcs(inst, dist) if arcs is None else arcs
        self.succ = [[int(j) for j in np.nonzero(self.arcs[i])[0]] for i in range(N + 1)]
        self.ret = [bool(self.arcs[i, 0]) for i in range(N + 1)]
        # Pickup successors (latest departure l_j - c_ij, j, c_ij), latest first: a scan stops at the first one that is
        # too late. Delivery arcs as boolean rows (deliveries are extended only for open requests).
        self.succP = [sorted(((self.l[j] - self.c[i][j], j, self.c[i][j]) for j in self.succ[i] if j and self.isP[j]),
                             reverse=True) for i in range(N + 1)]
        self.okD = [[bool(self.arcs[i, j]) and j > 0 and not self.isP[j] for j in range(N + 1)] for i in range(N + 1)]
        delta = insertion_slack(inst, dist)
        self.slack = int(delta.sum())
        gam = skip_slack(dist, self.D)
        self.skippable = 0
        for r in range(n):
            if delta[self.D[r]] == 0:
                self.skippable |= 1 << r
        self.gamma = [int(gam[self.D[r]]) for r in range(n)]
        # Completion bound: any path from j back to the depot costs at least c_j0 minus the total skip slack.
        big_gamma = int(skip_slack(dist).sum())
        self.ret_lb = [max(0, int(dist[j, 0]) - big_gamma) for j in range(N + 1)]
        c, e, l, s, sl = self.c, self.e, self.l, self.s, self.slack
        # Latest start of service at the pickup of r that still allows its delivery (with the rounding slack).
        latest_p = [min(l[self.P[r]], l[self.D[r]] - s[self.P[r]] - c[self.P[r]][self.D[r]] + sl) for r in range(n)]
        # Pickup r is unreachable from a label at node i starting at tau when tau > th[i][r].
        self.ths: list[list[int]] = []
        self.order: list[list[int]] = []
        self._ng: dict[int, list[int]] = {}
        self.masks: list[list[int]] = []
        for i in range(N + 1):
            pairs = sorted((latest_p[r] - s[i] - c[i][self.P[r]] + sl, r) for r in range(n))
            self.ths.append([t for t, _ in pairs])
            self.order.append([r for _, r in pairs])
            acc, m = [0], 0
            for _, r in pairs:
                m |= 1 << r
                acc.append(m)
            self.masks.append(acc)
        # Open request r is still deliverable from a label at node j starting at tau only if tau <= dl[j][r].
        self.dl = [[l[self.D[r]] - s[j] - c[j][self.D[r]] + sl for r in range(n)] for j in range(N + 1)]
        self.stats = {"rounds": 0, "exact_rounds": 0, "labels": 0, "seconds": 0.0}

    def unreachable(self, i: int, tau: int) -> int:
        return self.masks[i][bisect_left(self.ths[i], tau)]

    def ng_sets(self, k: int) -> list[int]:
        """ng-neighbourhoods as request bitmasks: for node j, its own request and the k - 1 requests whose pickups are
        nearest to j (the depot's is empty)."""
        if k not in self._ng:
            out = [0]
            for j in range(1, self.N + 1):
                near = sorted(range(self.n), key=lambda r: (self.c[j][self.P[r]], r))
                m = 1 << self.req[j]
                for r in near:
                    if m.bit_count() >= k:
                        break
                    m |= 1 << r
                out.append(m)
            self._ng[k] = out
        return self._ng[k]

    def price(self, prize, const: float, *, exact: bool = True, max_cols: int = 50, deadline: float = INF,
              forbidden: set[tuple[int, int]] | None = None, strict: bool = False, max_labels: int = 3_000_000,
              near: int = 10, beam: int = 30, tol: float = 1e-6, ng: int | None = None) -> dict[str, Any]:
        """Routes of negative reduced cost (most negative first, one per client set) at the prizes (one per request)
        and the route constant. See the module docstring for the modes. ng=k prices over ng-routes (a relaxation of
        the elementary routes; their routes may pick a request up twice)."""
        t_start = time.perf_counter()
        N, Q, n = self.N, self.Q, self.n
        c, e, l, s = self.c, self.e, self.l, self.s
        req, isP, q, dl, ths, masks, ret = self.req, self.isP, self.q, self.dl, self.ths, self.masks, self.ret
        ret_lb = self.ret_lb
        prize = [float(x) for x in prize]
        gain = [max(0.0, x) for x in prize]
        succP, okD, Dn = self.succP, self.okD, self.D
        if forbidden:
            succP = [[x for x in succP[i] if (i, x[1]) not in forbidden] for i in range(N + 1)]
            okD = [list(row) for row in okD]
            for (i, j) in forbidden:
                okD[i][j] = False
            strict = True
        if not exact:
            lim = []
            for i in range(N + 1):
                pick = sorted(succP[i], key=lambda x: x[2] - prize[req[x[1]]])[:near]
                lim.append(sorted(pick, reverse=True))
            succP = lim
        use_ng = ng is not None and ng < n
        if use_ng:
            NG = self.ng_sets(ng)
            strict = True  # skipping deliveries changes ng-memories: open sets must be equal
            # Prize still reachable from node j at time tau: total minus the prefix of the threshold order.
            cum = []
            for j in range(N + 1):
                acc, tot = [0.0], 0.0
                for r in self.order[j]:
                    tot += gain[r]
                    acc.append(tot)
                cum.append(acc)
            total_gain = sum(gain)
        skippable, gamma = (0 if strict else self.skippable), self.gamma
        check_F = exact
        l0 = l[0]
        lab_node: list[int] = [0]
        lab_par: list[int] = [-1]
        U0 = self.unreachable(0, e[0])
        fp0 = sum(gain[r] for r in range(n) if not U0 >> r & 1)
        heap = [(e[0], float(const), 0, 0, 0, 0 if use_ng else U0, 0, fp0)]
        kept: list[dict[int, list[tuple[float, int]]]] = [dict() for _ in range(N + 1)]
        nkept = [0] * (N + 1)
        found: list[tuple[float, int]] = []
        min_rc = 0.0
        complete = True
        pops = 0
        total_kept = 0
        while heap:
            t, rc, idx, i, O, F, load, fp = heapq.heappop(heap)
            pops += 1
            if pops & 1023 == 0 and time.perf_counter() > deadline:
                complete = False
                break
            if i:
                bucket = kept[i]
                if not exact and nkept[i] >= beam:
                    continue
                dom = False
                lst = bucket.get(O)  # kept labels (rc, F) sorted by reduced cost
                if lst:
                    if not check_F:
                        dom = lst[0][0] <= rc
                    else:
                        # ng memories: what L1 forbids and L2 allows must be unreachable for L2.
                        free = ~(F | masks[i][bisect_left(ths[i], t)]) if use_ng else ~F
                        for rc1, F1 in lst:
                            if rc1 > rc:
                                break
                            if not (F1 & free):
                                dom = True
                                break
                if not dom and O & skippable:
                    # Subset dominance over open sets missing one or two skippable requests (a safe restriction).
                    sub = O & skippable
                    bits = []
                    while sub:
                        b = sub & -sub
                        bits.append(b)
                        sub ^= b
                    cands = [(O ^ b, gamma[b.bit_length() - 1]) for b in bits]
                    if len(bits) <= 12:
                        cands += [(O ^ b1 ^ b2, gamma[b1.bit_length() - 1] + gamma[b2.bit_length() - 1])
                                  for k1, b1 in enumerate(bits) for b2 in bits[k1 + 1:]]
                    for Op, sl in cands:
                        lst = bucket.get(Op)
                        if lst:
                            for rc1, F1 in lst:
                                if rc1 + sl > rc:
                                    break
                                if not check_F or not (F1 & ~F):
                                    dom = True
                                    break
                            if dom:
                                break
                if dom:
                    continue
                own = bucket.get(O)
                if own is None:
                    bucket[O] = [(rc, F)]
                else:
                    insort(own, (rc, F))
                nkept[i] += 1
                total_kept += 1
                if total_kept > max_labels:
                    complete = False
                    break
                if O == 0:
                    if ret[i] and (not forbidden or (i, 0) not in forbidden) and t + s[i] + c[i][0] <= l0:
                        rr = rc + c[i][0]
                        if rr < min_rc:
                            min_rc = rr
                        if rr < -tol:
                            found.append((rr, idx))
            dep = t + s[i]
            ci = c[i]
            blocked = (F | O) if use_ng else F
            okDi = okD[i]
            # Deliveries of the open requests, then the pickups that can still be reached in time.
            cand = []
            x = O
            while x:
                b = x & -x
                x ^= b
                j = Dn[b.bit_length() - 1]
                if okDi[j]:
                    cand.append((j, ci[j]))
            for lat, j, cij in succP[i]:
                if lat < dep:
                    break
                if not blocked & (1 << req[j]):
                    cand.append((j, cij))
            for j, cij in cand:
                arr = dep + cij
                if arr > l[j]:
                    continue
                if arr < e[j]:
                    arr = e[j]
                r = req[j]
                bit = 1 << r
                if isP[j]:
                    nload = load + q[r]
                    if nload > Q:
                        continue
                    nO = O | bit
                    nrc = rc + cij - prize[r]
                else:
                    nO = O ^ bit
                    nload = load - q[r]
                    nrc = rc + cij
                x = nO
                dlj = dl[j]
                ok = True
                while x:
                    b = x & -x
                    if arr > dlj[b.bit_length() - 1]:
                        ok = False
                        break
                    x ^= b
                if not ok:
                    continue
                kj = bisect_left(ths[j], arr)
                if use_ng:
                    nF = (F & NG[j]) | (bit if isP[j] else 0)
                    # Completion bound: prizes of the reachable requests that are neither open nor in memory.
                    unr = masks[j][kj]
                    nfp = total_gain - cum[j][kj]
                    x = (nF | nO) & ~unr
                    while x:
                        b = x & -x
                        nfp -= gain[b.bit_length() - 1]
                        x ^= b
                else:
                    nF = F | masks[j][kj] | (bit if isP[j] else 0)
                    # Completion bound: the prizes still collectible (requests outside U) and the way back.
                    nfp = fp
                    x = nF & ~F
                    while x:
                        b = x & -x
                        nfp -= gain[b.bit_length() - 1]
                        x ^= b
                if nrc + ret_lb[j] - nfp >= 0.0:
                    continue
                lab_node.append(j)
                lab_par.append(idx)
                heapq.heappush(heap, (arr, nrc, len(lab_node) - 1, j, nO, nF, nload, nfp))
        routes = []
        seen = set()
        for rr, idx in sorted(found):
            path = []
            x = idx
            while x > 0:
                path.append(lab_node[x])
                x = lab_par[x]
            path.reverse()
            key = frozenset(path)
            if key in seen:
                continue
            seen.add(key)
            routes.append((rr, path))
            if len(routes) >= max_cols:
                break
        dt = time.perf_counter() - t_start
        self.stats["rounds"] += 1
        self.stats["exact_rounds"] += int(exact)
        self.stats["labels"] += total_kept
        self.stats["seconds"] += dt
        return {"routes": routes, "min_rc": min_rc, "complete": complete and exact, "labels": total_kept,
                "seconds": dt}


# ---------------------------------------------------------------- shared basic initializer

def _lists(inst: PDPTWInstance) -> tuple[list[int], list[int], list[int], list[int]]:
    """Ready, due, service and demand as Python lists (cached on the instance)."""
    cache = inst.__dict__.get("_lists")
    if cache is None:
        cache = ([int(v) for v in inst.ready], [int(v) for v in inst.due], [int(v) for v in inst.service],
                 [int(v) for v in inst.demand])
        inst.__dict__["_lists"] = cache
    return cache


def best_insertion(inst: PDPTWInstance, dist, route: list[int], p: int, d: int, cutoff: float = INF) -> tuple[float, list[int] | None]:
    """Cheapest feasible insertion of request (p, d) into a feasible route: the pickup before position a, the delivery
    before position b >= a (positions in the original route, len(route) = at the end), scanned in (a, b) order with
    strict improvement; returns (added distance, new route) or (cutoff, None). Incremental: the state after the pickup
    is carried along b, and the part after the delivery is checked only until it starts no later than in the
    original route (from there on the original route's feasibility carries over, with the same loads)."""
    c = dist
    e, l, s, dem = _lists(inst)
    Q = inst.capacity
    L = len(route)
    S = [0] * L
    W = [0] * L
    t, prev, load = e[0], 0, 0
    for k, v in enumerate(route):
        t = max(t + s[prev] + c[prev][v], e[v])
        S[k] = t
        load += dem[v]
        W[k] = load
        prev = v
    qp = dem[p]
    best, best_route = cutoff, None
    for a in range(L + 1):
        pa = route[a - 1] if a else 0
        na = route[a] if a < L else 0
        add_p = c[pa][p] + c[p][na] - c[pa][na]
        if add_p >= best:
            continue
        load_cur = (W[a - 1] if a else 0) + qp
        if load_cur > Q:
            continue
        t_cur = max((S[a - 1] if a else e[0]) + s[pa] + c[pa][p], e[p])
        if t_cur > l[p]:
            continue
        cur = p
        for b in range(a, L + 1):
            nb = route[b] if b < L else 0
            delta = add_p + c[cur][d] + c[d][nb] - c[cur][nb]
            if delta < best:
                td = max(t_cur + s[cur] + c[cur][d], e[d])
                if td <= l[d]:
                    t, pv, ok = td, d, True
                    for k in range(b, L + 1):
                        if k == L:
                            ok = t + s[pv] + c[pv][0] <= l[0]
                            break
                        v = route[k]
                        t = max(t + s[pv] + c[pv][v], e[v])
                        if t > l[v]:
                            ok = False
                            break
                        if t <= S[k]:
                            break
                        pv = v
                    if ok:
                        best = delta
                        best_route = route[:a] + [p] + route[a:b] + [d] + route[b:]
            if b == L:
                break
            v = route[b]
            t_cur = max(t_cur + s[cur] + c[cur][v], e[v])
            load_cur += dem[v]
            if t_cur > l[v] or load_cur > Q:
                break
            cur = v
    return best, best_route


def insertion_plan(inst: PDPTWInstance, dist) -> list[list[int]] | None:
    """Deterministic sequential cheapest insertion: requests by increasing pickup due time; each goes to the cheapest
    feasible position over the current routes, or opens a new route. None if some request fits nowhere."""
    c = dist.tolist() if isinstance(dist, np.ndarray) else dist
    routes: list[list[int]] = []
    order = sorted(range(inst.n), key=lambda r: (int(inst.due[inst.pickups[r]]), int(inst.ready[inst.pickups[r]]), r))
    for r in order:
        p, d = inst.pickups[r], inst.deliveries[r]
        best, where, new = INF, None, None
        for k, route in enumerate(routes):
            delta, cand = best_insertion(inst, c, route, p, d, best)
            if cand is not None and delta < best:
                best, where, new = delta, k, cand
        if where is None:
            cost, _ = route_check(inst, c, [p, d])
            if cost is None:
                return None
            routes.append([p, d])
        else:
            routes[where] = new
    return routes
