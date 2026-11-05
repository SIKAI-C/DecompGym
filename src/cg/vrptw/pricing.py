"""ng-route pricing with time windows and capacity for the VRPTW (scale-up study), shared by the naive and the
engineered column generation.

Problem: over all ng-routes (Baldacci, Mingozzi and Roberti 2011) depot -> customers -> depot that meet the time
windows (waiting allowed) and the capacity, minimize  const + c_out[j1] - p[j1] + sum (Cc[a, b] - p[b]) + c_in[jk].
Customers are 0..n-1 here (customer i + 1 of the instance); times and distances are integer tenths.

Preprocessing (exact): tightened windows E_j (earliest possible service start over all paths from the depot) and L_j
(latest service start from which the depot can still be reached), by monotone fixpoint iterations; an arc (i, j) is
kept only if E_i + s_i + d_ij <= L_j and q_i + q_j <= Q; tau = s_i + d_ij.

Labelling: forward, mono-directional, processed by time buckets of width Delta = min tau over the arcs (> 0), so an
extension always lands in a later bucket. A label (node, time, load, ng-memory, cost) is dropped if
- a label of an earlier bucket at the same node has cost <= its cost, load <= its load and memory subset of its memory
  (its time is smaller by construction): a dense dominance surface over (node, memory, load cell) with subset-minimum
  and prefix-minimum closures (loads in cells of width lw, stored labels rounded up and queries rounded down, so it
  never claims a false dominance);
- a label of the same bucket at the same node dominates it in (cost, time, load, memory) (pairwise);
- its completion bound cannot reach a negative reduced cost: B[j, g] is a backward bound over a time grid (step w <=
  Delta; label times rounded down, which only relaxes) of the cheapest completion from j back to the depot, relaxing
  elementarity, the ng-memory and the capacity, so it underestimates every ng-route completion; exact rounds use
  B2[j, g, c], which also keeps the remaining capacity (in cells read at their top value, which only relaxes). Both
  are computed from a precomputed list of (arc, grid point) events in decreasing time blocks.
ng-memory bits of customers that are no longer reachable in time (t + shortest tau-path to x > L_x) are cleared;
that removes no feasible extension, keeps the dominance rule valid (see `price`), and lets more labels dominate.

exact=True extends to every customer and reports complete=True unless the deadline or a resource guard cut the search;
then the minimum reduced cost over all ng-routes is min_rc when it is below -tol (and >= -tol otherwise).
exact=False extends only to the `near` (wide=True: 2 near) closest feasible successors and keeps a beam of labels
per bucket.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.cg.vrptw.a_data import VRPTWInstance

INF = float("inf")


def time_bounds(inst: VRPTWInstance) -> dict[str, Any]:
    """Exact window tightening and arc elimination (customers 0..n-1, integer tenths).
    E_j: earliest possible service start at j over all paths from the depot, the fixpoint of
         E_j = max(e_j, min(e_0 + d_0j, min_i E_i + s_i + d_ij)) reached from above (Bellman-Ford; after k rounds it
         is the minimum over paths of at most k + 1 arcs, and cycles never help since waiting only delays);
    L_j: latest service start at j from which the depot can still be reached in time, the fixpoint of
         L_j = min(l_j, max(l_0 - s_j - d_j0, max_k L_k - s_j - d_jk)) reached from below;
    every feasible route starts service at j inside [E_j, L_j]. Arc (i, j) is kept iff E_i + s_i + d_ij <= L_j and
    q_i + q_j <= Q; depot arcs iff E_j <= L_j (out) and E_j <= l_0 - s_j - d_j0 (in); delta = min s_i + d_ij over the
    kept customer arcs."""
    n, Q = inst.n, int(inst.capacity)
    q = inst.demand[1:].astype(np.int64)
    d = inst.dist.astype(np.int64)
    dc, d_out, d_in = d[1:, 1:], d[0, 1:], d[1:, 0]
    e0, l0 = int(inst.ready[0]), int(inst.due[0])
    e, l, s = inst.ready[1:].astype(np.int64), inst.due[1:].astype(np.int64), inst.service[1:].astype(np.int64)
    tau = s[:, None] + dc
    big = np.where(np.eye(n, dtype=bool), 10 ** 12, 0)
    E = np.maximum(e, e0 + d_out)
    for _ in range(n + 1):
        new = np.maximum(e, np.minimum(e0 + d_out, (E[:, None] + tau + big).min(axis=0)))
        if np.array_equal(new, E):
            break
        E = new
    ret_by = l0 - s - d_in
    L = np.minimum(l, ret_by)
    for _ in range(n + 1):
        new = np.minimum(l, np.maximum(ret_by, (L[None, :] - tau - big).max(axis=1)))
        if np.array_equal(new, L):
            break
        L = new
    live = (E <= L) & (q <= Q)  # customers some route can serve
    ok = (E[:, None] + tau <= L[None, :]) & (q[:, None] + q[None, :] <= Q) & live[:, None] & live[None, :]
    np.fill_diagonal(ok, False)
    taus = tau[ok]
    return {"E": E, "L": L, "arc_ok": ok, "out_ok": (E <= L) & (q <= Q), "in_ok": (E <= ret_by) & (E <= L) & (q <= Q),
            "ret_by": ret_by, "tau": tau, "delta": int(taus.min()) if len(taus) else max(1, l0 - e0)}


class TWPricer:
    def __init__(self, inst: VRPTWInstance, ng: int = 8, near: int = 12, grid_points: int = 400, load_cells: int = 20,
                 max_events: int = 6_000_000) -> None:
        n = inst.n
        self.n, self.Q = n, int(inst.capacity)
        self.q = inst.demand[1:].astype(np.int64)
        d = inst.dist.astype(np.int64)
        self.d = d[1:, 1:].copy()
        self.d_out = d[0, 1:].copy()
        self.d_in = d[1:, 0].copy()
        self.e0, self.l0 = int(inst.ready[0]), int(inst.due[0])
        self.e = inst.ready[1:].astype(np.int64)
        self.l = inst.due[1:].astype(np.int64)
        self.s = inst.service[1:].astype(np.int64)
        tb = time_bounds(inst)
        self.tau, self.E, self.L, self.ret_by = tb["tau"], tb["E"], tb["L"], tb["ret_by"]
        ok = tb["arc_ok"]
        self.arc_ok, self.out_ok, self.in_ok = ok, tb["out_ok"], tb["in_ok"]
        self.C = np.where(ok, self.d.astype(float), INF)
        self.c_out = np.where(self.out_ok, self.d_out.astype(float), INF)
        self.c_in = np.where(self.in_ok, self.d_in.astype(float), INF)
        self.delta = tb["delta"]
        if self.delta < 1:
            raise ValueError("labelling by time buckets needs s_i + d_ij >= 0.1 on every arc")
        # Load cells of the dominance surface; completion-bound grid and events.
        self.lw = max(1, -(-self.Q // load_cells))
        self.LC = -(-self.Q // self.lw) + 1
        H = max(1, self.l0 - self.e0)
        w = max(1, min(self.delta, -(-H // grid_points)))
        while not self._completion_events(w, max_events if w < self.delta else 1 << 62):
            w = min(self.delta, 2 * w)  # coarser grid when the event list would be too large
        # ng-neighbourhoods: the ng - 1 closest customers linked by a feasible arc in either direction.
        k = max(0, min(ng - 1, n - 1))
        prox = np.where(ok | ok.T, self.d.astype(float), 1e18)
        np.fill_diagonal(prox, INF)  # never chosen: k <= n - 1
        nb = np.argsort(prox, axis=1, kind="stable")[:, :k]
        self.k, self.nm = k, 1 << k
        bitpos = np.full((n, n), -1, dtype=np.int64)
        if k:
            bitpos[np.arange(n)[:, None], nb] = np.arange(k)[None, :]
        self.bitpos = bitpos
        self.inN = bitpos >= 0
        self.nb = nb
        # conv[(i * n + j) * nm + mask]: memory after i -> j in j's coordinates (i itself if i is in N(j)).
        base = np.where(bitpos.T >= 0, np.left_shift(1, np.maximum(bitpos.T, 0)), 0)
        if k:
            bp = bitpos[np.arange(n)[None, :, None], nb[:, None, :]]  # [i, j, t] = position of nb[i, t] in N(j)
            P = np.where(bp >= 0, np.left_shift(1, np.maximum(bp, 0)), 0).reshape(n * n, k)
            bits = (np.arange(self.nm)[:, None] >> np.arange(k)[None, :]) & 1
            conv = base.reshape(n * n, 1) + P @ bits.T
        else:
            conv = base.reshape(n * n, 1)
        self.conv = conv.astype(np.int32).reshape(-1)
        # Reachability: latest[j, t] = latest start at j from which nb[j, t] can still be served (shortest tau paths).
        sp = np.where(ok, self.tau, np.iinfo(np.int64).max // 4).astype(np.int64)
        np.fill_diagonal(sp, 0)
        for m in range(n):
            np.minimum(sp, sp[:, m:m + 1] + sp[m:m + 1, :], out=sp)
        self.latest = (self.L[nb] - sp[np.arange(n)[:, None], nb]) if k else np.zeros((n, 0), dtype=np.int64)
        self.pow2 = (1 << np.arange(k)).astype(np.int64)
        # Heuristic successor lists (padded with the node itself, whose arc cost is infinite).
        order_d = np.argsort(np.where(ok, self.d.astype(float), INF), axis=1, kind="stable")
        self.near_lists = []
        for width in (near, 2 * near):
            cand = order_d[:, :max(1, min(width, n - 1))]
            self.near_lists.append(np.where(ok[np.arange(n)[:, None], cand], cand, np.arange(n)[:, None]))
        self.near = self.near_lists[0]
        # Exact successor lists (CSR), each sorted by th = L_j - tau_ij descending: a label at i with time t can only
        # extend to the prefix with th >= t (one global searchsorted on i * 2^40 - th finds its length).
        si, sj = np.nonzero(ok)
        th = self.L[sj] - self.tau[si, sj]
        order = np.lexsort((-th, si))
        si, sj, th = si[order], sj[order], th[order]
        self.succ = sj.astype(np.int64)
        self.succ_key = si.astype(np.int64) * (1 << 40) - th
        self.succ_ptr = np.searchsorted(si, np.arange(n + 1)).astype(np.int64)

    # ------------------------------------------------------------ completion bound

    def _completion_events(self, w: int, max_events: int) -> bool:
        """Precomputes the (arc, grid point) pairs of the completion bound at grid step w (False if too many).
        B[j, g] is needed only for floor((E_j - e0) / w) <= g <= floor((L_j - e0) / w) (every label at j starts
        service in [E_j, L_j]); arc (j, k) contributes at g if T_g + tau_jk <= L_k, reading
        B[k, floor((max(E_k, T_g + tau_jk) - e0) / w)] (a real start at k is >= E_k and >= t + tau_jk >= T_g + tau_jk,
        and the bound at an earlier grid time is a lower bound since completions only get dearer later)."""
        e0, n = self.e0, self.n
        G = (max(1, self.l0 - e0)) // w + 1
        gl = np.maximum(0, (self.E - e0) // w)
        gh = np.minimum(G - 1, (self.L - e0) // w)
        aj, ak = np.nonzero(self.arc_ok)
        at = self.tau[aj, ak]
        lo = gl[aj]
        hi = np.minimum(gh[aj], (self.L[ak] - at - e0) // w)
        cnt = np.maximum(0, hi - lo + 1)
        total = int(cnt.sum())
        if total > max_events:
            return False
        arc = np.repeat(np.arange(len(aj)), cnt)
        g = np.repeat(lo, cnt) + (np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        arr = np.maximum(self.E[ak[arc]], e0 + w * g + at[arc])
        idx = (arr - e0) // w
        bs = max(1, self.delta // w)
        blk = g // bs
        # Blocks in decreasing order; inside a block, entries sorted by target cell (j, g).
        key = aj[arc] * G + g
        order = np.lexsort((key, -blk))
        arc, key, idx, blk = arc[order], key[order], idx[order], blk[order]
        self.w, self.G = w, G
        self.ev_arc = arc
        self.ev_src = (ak[arc] * G + idx).astype(np.int64)
        self.ev_aj, self.ev_ak = aj, ak
        bounds = np.flatnonzero(np.r_[True, blk[1:] != blk[:-1], True]) if total else np.zeros(1, dtype=np.int64)
        self.ev_blocks = []
        for s0, s1 in zip(bounds[:-1], bounds[1:]):
            kk = key[s0:s1]
            st = np.flatnonzero(np.r_[True, kk[1:] != kk[:-1]])
            self.ev_blocks.append((int(s0), int(s1), st, kk[st]))
        # Direct returns, on the needed range only.
        T = e0 + w * np.arange(G)
        gg = np.arange(G)[None, :]
        self.ret_ok = (T[None, :] <= self.ret_by[:, None]) & (gg >= gl[:, None]) & (gg <= gh[:, None])
        self.n_events = total
        # Capacity-aware variant (exact rounds): remaining capacity in cells of width lw, each read at its top value
        # (c + 1) lw - 1 >= the true remaining capacity; visiting k leaves cell ((c + 1) lw - 1 - q_k) // lw.
        self.LCb = self.Q // self.lw + 1
        top = (np.arange(self.LCb) + 1) * self.lw - 1
        self.arc_cp = (top[None, :] - self.q[ak][:, None]) // self.lw
        self.arc_cp[top[None, :] - self.q[ak][:, None] < 0] = -1
        return True

    def completion(self, Cc: np.ndarray, p: np.ndarray, c_in: np.ndarray) -> np.ndarray:
        """B[j, g]: lower bound on the reduced cost of completing a path whose service at j starts at a time t with
        floor((t - e0) / w) = g (the prize of j already counted) back to the depot, relaxing elementarity, the
        ng-memory and the capacity (only read for g in j's range)."""
        n, G = self.n, self.G
        B = np.where(self.ret_ok & np.isfinite(c_in)[:, None], c_in[:, None], INF)
        Bf = B.reshape(-1)
        ca = Cc[self.ev_aj, self.ev_ak] - p[self.ev_ak]
        for s0, s1, st, cells in self.ev_blocks:
            V = ca[self.ev_arc[s0:s1]] + Bf[self.ev_src[s0:s1]]
            red = np.minimum.reduceat(V, st)
            Bf[cells] = np.minimum(Bf[cells], red)
        return B

    def completion_cap(self, Cc: np.ndarray, p: np.ndarray, c_in: np.ndarray) -> np.ndarray:
        """B2[j, g, c]: as `completion`, but also respecting the capacity: c is the cell of the remaining capacity
        (read at the cell's top, which only relaxes), so a completion only visits customers that still fit."""
        LCb = self.LCb
        base = np.where(self.ret_ok & np.isfinite(c_in)[:, None], c_in[:, None], INF)
        B2 = np.repeat(base[:, :, None], LCb, axis=2)
        Bf = B2.reshape(-1)
        ca = Cc[self.ev_aj, self.ev_ak] - p[self.ev_ak]
        lc = np.arange(LCb)[None, :]
        for s0, s1, st, cells in self.ev_blocks:
            arcs = self.ev_arc[s0:s1]
            cp = self.arc_cp[arcs]
            V = np.where(cp >= 0, ca[arcs][:, None] + Bf[self.ev_src[s0:s1][:, None] * LCb + np.maximum(cp, 0)], INF)
            red = np.minimum.reduceat(V, st, axis=0)
            tgt = cells[:, None] * LCb + lc
            Bf[tgt] = np.minimum(Bf[tgt], red)
        return B2

    # ------------------------------------------------------------ labelling

    def price(self, p: np.ndarray, const: float, Cc: np.ndarray, c_out: np.ndarray, c_in: np.ndarray, *, exact: bool,
              beam: int = 0, max_cols: int = 30, deadline: float = INF, tol: float = 1e-6, max_labels: int = 200_000, wide: bool = False,
              cap_bound: bool = True,
              max_extensions: int = 3_000_000, pair_cap: int = 4_000_000) -> dict[str, Any]:
        """Labelling at prizes p, route constant const, customer arc costs Cc (inf = forbidden), depot arc costs c_out
        and c_in. Returns {"routes": [(reduced cost, path)], "min_rc", "complete", "labels", "buckets"}.

        Dominance with cleared memory bits is valid: if L1 (time t1 <= t2) has M1 & R(t1) subset of M2 & R(t2), where
        R(t) is the set still reachable at time t (R(t2) subset of R(t1)), then every customer x that an extension of
        L2 visits is in R(t2) and not in M2, hence not in M1 & R(t1), and x in R(t1), so x is not in M1."""
        n, Q, q, nm, k = self.n, self.Q, self.q, self.nm, self.k
        LC, lw, w, e0 = self.LC, self.lw, self.w, self.e0
        Cc = np.where(self.arc_ok, Cc, INF)
        c_out = np.where(self.out_ok, c_out, INF)
        c_in = np.where(self.in_ok, c_in, INF)
        G = self.G
        if exact and cap_bound:
            Bf2 = self.completion_cap(Cc, p, c_in).reshape(-1)
            LCb = self.LCb

            def bnd(j, t, ld):
                return Bf2[(j * G + np.minimum((t - e0) // w, G - 1)) * LCb + (Q - ld) // lw]
        else:
            Bf1 = self.completion(Cc, p, c_in).reshape(-1)

            def bnd(j, t, ld):
                return Bf1[j * G + np.minimum((t - e0) // w, G - 1)]
        D = np.full(n * nm * LC, INF)
        width = self.delta
        nbk = (self.l0 - e0) // width + 1
        pending: list[list[tuple]] = [[] for _ in range(nbk)]
        # Initial labels.
        j0 = np.flatnonzero(np.isfinite(c_out))
        t0 = np.maximum(self.e[j0], e0 + self.d_out[j0])
        c0 = const + c_out[j0] - p[j0]
        keep = (t0 <= self.L[j0]) & (c0 + bnd(j0, t0, q[j0]) < -tol)
        j0, t0, c0 = j0[keep], t0[keep], c0[keep]
        self._push(pending, j0, t0, q[j0], np.zeros(len(j0), dtype=np.int64), c0, np.full(len(j0), -1, dtype=np.int64))
        g_node: list[np.ndarray] = []
        g_parent: list[np.ndarray] = []
        nlab = 0
        found_rc: list[np.ndarray] = []
        found_lid: list[np.ndarray] = []
        min_rc = 0.0
        complete = True
        done = 0
        for b in range(nbk):
            if not pending[b]:
                continue
            if time.perf_counter() > deadline:
                complete = False
                break
            node, tt, ld, mask, cost, par = (np.concatenate([x[i] for x in pending[b]]) for i in range(6))
            pending[b] = []
            done += 1
            # Dominance by labels of earlier buckets (smaller times).
            alive = cost < D[(node * nm + mask) * LC + ld // lw]
            node, tt, ld, mask, cost, par = node[alive], tt[alive], ld[alive], mask[alive], cost[alive], par[alive]
            if not len(node):
                continue
            # Dominance inside the bucket.
            keep = self._bucket_dominance(node, tt, ld, mask, cost, pair_cap)
            node, tt, ld, mask, cost, par = node[keep], tt[keep], ld[keep], mask[keep], cost[keep], par[keep]
            cap = max_labels if exact else beam
            if cap and len(node) > cap:
                if exact:
                    complete = False
                score = cost + bnd(node, tt, ld)
                top = np.argpartition(score, cap - 1)[:cap]
                node, tt, ld, mask, cost, par = node[top], tt[top], ld[top], mask[top], cost[top], par[top]
            lid = np.arange(nlab, nlab + len(node))
            nlab += len(node)
            g_node.append(node)
            g_parent.append(par)
            # Closures.
            rc = cost + c_in[node]
            closable = tt <= self.ret_by[node]
            neg = closable & (rc < -tol)
            if neg.any():
                found_rc.append(rc[neg])
                found_lid.append(lid[neg])
                min_rc = min(min_rc, float(rc[neg].min()))
            # Surface update.
            self._insert(D, node, mask, ld, cost)
            # Extensions.
            L = len(node)
            near_arr = self.near_lists[1 if wide else 0]
            wdt = n if exact else near_arr.shape[1]
            step = max(1, 1_000_000 // max(1, wdt))
            parts = []
            total = 0
            for s0 in range(0, L, step):
                if s0 and time.perf_counter() > deadline:
                    complete = False
                    break
                sl = slice(s0, min(L, s0 + step))
                nd, ms, l0_, t_, cs, li = node[sl], mask[sl], ld[sl], tt[sl], cost[sl], lid[sl]
                if exact:
                    cnt = np.searchsorted(self.succ_key, nd * (1 << 40) - t_, side="right") - self.succ_ptr[nd]
                    tot = int(cnt.sum())
                    if not tot:
                        continue
                    r_ = np.repeat(np.arange(len(nd)), cnt)
                    jj = self.succ[np.repeat(self.succ_ptr[nd], cnt) + (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt))]
                    ni = nd[r_]
                    cj = cs[r_] + Cc[ni, jj] - p[jj]
                    tj = np.maximum(self.e[jj], t_[r_] + self.tau[ni, jj])
                    lj = l0_[r_] + q[jj]
                    bp = self.bitpos[ni, jj]
                    ok = np.isfinite(cj) & (lj <= Q) & (tj <= self.L[jj]) & ((bp < 0) | (((ms[r_] >> np.maximum(bp, 0)) & 1) == 0))
                    if not ok.all():
                        r_, jj, tj, lj, cj = r_[ok], jj[ok], tj[ok], lj[ok], cj[ok]
                    if not len(r_):
                        continue
                else:
                    J = near_arr[nd]
                    cc = cs[:, None] + Cc[nd[:, None], J] - p[J]
                    bp = self.bitpos[nd[:, None], J]
                    tn = np.maximum(self.e[J], t_[:, None] + self.tau[nd[:, None], J])
                    ln = l0_[:, None] + q[J]
                    ok = np.isfinite(cc) & (ln <= Q) & (tn <= self.L[J])
                    ok &= (bp < 0) | (((ms[:, None] >> np.maximum(bp, 0)) & 1) == 0)
                    r_, c_ = np.nonzero(ok)
                    if not len(r_):
                        continue
                    jj, tj, lj, cj = J[r_, c_], tn[r_, c_], ln[r_, c_], cc[r_, c_]
                good = cj + bnd(jj, tj, lj) < -tol
                r_, jj, tj, lj, cj = r_[good], jj[good], tj[good], lj[good], cj[good]
                if not len(r_):
                    continue
                mj = self.conv[(nd[r_].astype(np.int64) * n + jj) * nm + ms[r_]].astype(np.int64)
                if k:
                    mj &= ((tj[:, None] <= self.latest[jj]) * self.pow2[None, :]).sum(axis=1)
                good = cj < D[(jj * nm + mj) * LC + lj // lw]
                if not good.all():
                    r_, jj, tj, lj, cj, mj = r_[good], jj[good], tj[good], lj[good], cj[good], mj[good]
                parts.append((jj, tj, lj, mj, cj, li[r_]))
                total += len(jj)
                if total > max_extensions:
                    # Memory guard: keep the most promising half; the round no longer certifies a bound.
                    complete = False
                    a = [np.concatenate([x[i] for x in parts]) for i in range(6)]
                    score = a[4] + bnd(a[0], a[1], a[2])
                    top = np.argpartition(score, max_extensions // 2 - 1)[:max_extensions // 2]
                    parts = [tuple(x[top] for x in a)]
                    total = len(top)
            for jj, tj, lj, mj, cj, pr in parts:
                self._push(pending, jj, tj, lj, mj, cj, pr)
        routes: list[tuple[float, list[int]]] = []
        if found_rc:
            rcs = np.concatenate(found_rc)
            lids = np.concatenate(found_lid)
            gn = np.concatenate(g_node)
            gpar = np.concatenate(g_parent)
            seen: set[tuple[int, ...]] = set()
            for t in np.argsort(rcs, kind="stable"):
                path = []
                x = int(lids[t])
                while x >= 0:
                    path.append(int(gn[x]))
                    x = int(gpar[x])
                path.reverse()
                key = tuple(sorted(path))
                if key in seen:
                    continue
                seen.add(key)
                routes.append((float(rcs[t]), path))
                if len(routes) >= max_cols:
                    break
        return {"routes": routes, "min_rc": min_rc, "complete": complete and exact, "labels": nlab, "buckets": done}

    def _push(self, pending, node, tt, ld, mask, cost, par) -> None:
        if not len(node):
            return
        bk = (tt - self.e0) // self.delta
        order = np.argsort(bk, kind="stable")
        bk = bk[order]
        cuts = np.flatnonzero(np.r_[True, bk[1:] != bk[:-1], True])
        arrs = [x[order] for x in (node, tt, ld, mask, cost, par)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            pending[int(bk[a])].append(tuple(x[a:b] for x in arrs))

    def _bucket_dominance(self, node, tt, ld, mask, cost, pair_cap: int, dominators: int = 32) -> np.ndarray:
        """Labels of one bucket not dominated by another label of the bucket at the same node (cost, time, load,
        memory); ties keep one copy. Only the `dominators` cheapest labels of each node are tried as dominating
        labels, and only against costlier ones (sorted by cost), which can only keep extra labels."""
        N = len(node)
        keep = np.ones(N, dtype=bool)
        if N < 2:
            return keep
        order = np.lexsort((tt, cost, node))
        nd, t_, l_, m_ = node[order], tt[order], ld[order], mask[order]
        starts = np.flatnonzero(np.r_[True, nd[1:] != nd[:-1]])
        gid = np.cumsum(np.r_[True, nd[1:] != nd[:-1]]) - 1
        pos = np.minimum(np.arange(N) - starts[gid], dominators)
        dominated = np.zeros(N, dtype=bool)
        # Chunks of labels whose pair counts stay below pair_cap.
        csum = np.cumsum(pos)
        lo = 0
        while lo < N:
            hi = int(np.searchsorted(csum, (csum[lo - 1] if lo else 0) + pair_cap, side="right"))
            hi = max(hi, lo + 1)
            pp = pos[lo:hi]
            P = int(pp.sum())
            if P:
                bidx = np.repeat(np.arange(lo, hi), pp)
                off = np.arange(P) - np.repeat(np.cumsum(pp) - pp, pp)
                aidx = np.repeat(starts[gid[lo:hi]], pp) + off
                dom = (t_[aidx] <= t_[bidx]) & (l_[aidx] <= l_[bidx]) & ((m_[aidx] & ~m_[bidx]) == 0)
                dominated[bidx[dom]] = True
            lo = hi
        keep[order[dominated]] = False
        return keep

    def _insert(self, D, node, mask, ld, cost) -> None:
        nm, LC, lw = self.nm, self.LC, self.lw
        cell_l = -(-ld // lw)
        present = np.zeros(self.n, dtype=bool)
        present[node] = True
        uj = np.flatnonzero(present)
        where = np.zeros(self.n, dtype=np.int64)
        where[uj] = np.arange(len(uj))
        inv = where[node]
        Dn = np.full((len(uj), nm, LC), INF)
        np.minimum.at(Dn, (inv, mask, cell_l), cost)
        self._zeta(Dn)
        np.minimum.accumulate(Dn, axis=2, out=Dn)
        blk = D.reshape(self.n, nm, LC)
        blk[uj] = np.minimum(blk[uj], Dn)

    def _zeta(self, Dn: np.ndarray) -> None:
        """In place: Dn[j, mask, l] = min over submasks of mask."""
        if not self.k:
            return
        shape = Dn.shape
        X = Dn.reshape((shape[0],) + (2,) * self.k + (shape[2],))
        for t in range(self.k):
            ax = 1 + (self.k - 1 - t)
            lo = [slice(None)] * X.ndim
            hi = [slice(None)] * X.ndim
            lo[ax], hi[ax] = 0, 1
            np.minimum(X[tuple(hi)], X[tuple(lo)], out=X[tuple(hi)])

    # ------------------------------------------------------------ helpers

    def route_rc(self, path: list[int], p: np.ndarray, const: float, Cc: np.ndarray, c_out: np.ndarray, c_in: np.ndarray) -> float:
        return float(const + c_out[path[0]] - p[path[0]] + sum(Cc[a, b] - p[b] for a, b in zip(path, path[1:])) + c_in[path[-1]])

    def feasible(self, path: list[int]) -> bool:
        """Capacity and time windows (original windows) of a path of internal customers 0..n-1."""
        if int(self.q[path].sum()) > self.Q:
            return False
        t = max(self.e[path[0]], self.e0 + self.d_out[path[0]])
        if t > self.l[path[0]]:
            return False
        for a, b in zip(path, path[1:]):
            t = max(self.e[b], t + self.tau[a, b])
            if t > self.l[b]:
                return False
        return t + self.s[path[-1]] + self.d_in[path[-1]] <= self.l0
