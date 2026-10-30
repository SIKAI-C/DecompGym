"""Engineered branch-and-price reference for the CLRP (scale-up study; the paper's references are unchanged).

Set-covering master over depot routes r (depot d(r), cost = travel + vehicle cost F), with depot variables y:
    min sum_r c_r lam_r + sum_d O_d y_d
    s.t. sum_r a_ir lam_r >= 1                    (pi_i >= 0; covering is exact: ceil'd distances keep the triangle inequality)
         sum_{r at d} q_r lam_r <= W_d y_d       (mu_d <= 0)
         sum_{r at d} a_ir lam_r <= y_d          (rho_id <= 0, one row per customer and depot)
         sum_r lam_r >= ceil(sum q / Q)          (sigma >= 0)
         rounded capacity cuts x(delta(U)) >= 2 ceil(q(U) / Q) on route edges (beta_U >= 0, separated by greedy growth)
         branching rows x_e >= 1 on customer edges (eta_e >= 0)
         sum_d W_d y_d >= sum q,  sum_d y_d >= (fewest depots whose capacity covers the demand),  0 <= y <= 1.
Cut and edge-row duals only change arc costs, so pricing keeps its structure. Pricing per depot is an ng-route
shortest path with a capacity resource (neighbourhoods of the ng nearest customers), solved by depth-synchronous
labeling vectorized with numpy. Dominance is exact (same node, smaller load, subset of the ng-memory, smaller cost); a
backward q-route bound with ng-consistent 2-cycle elimination prunes labels that cannot reach a negative reduced cost.
Columns are identified by their visiting sequence, since cut coefficients depend on the order.

Lower bounds. Every exactly priced round gives, for every depot set S, a Lagrangian bound valid for all plans whose
open depots are exactly S: the master rows are dualized at the current duals, the depot binaries are kept, and the
routes keep sum_r q_r lam_r = sum q and sum_{r at d} q_r lam_r <= W_d y_d, so the route part is a fractional fill
of the demand into the open depots at each depot's most negative reduced cost per unit of load (exact under ng-route
pricing, preserved by the dominance rule). best[S] is the maximum over all rounds. The route set of the relaxation is
all ng-routes, a superset of the elementary routes, so every bound is valid; bounds are rounded up only after a
floating-point allowance.

Search. Best-first branch-and-price: first on depots (most fractional y; a node's bound is min best[S] over the sets
it allows), then, with all depots fixed, on customer-depot assignments and customer edges; at those nodes the
Lagrangian bound holds only inside the node and is kept with it. The global bound is the minimum over the open nodes.
Primal side: the heuristic control's savings construction for the cheapest depot set and its neighbours, then, between
pricing rounds, constructions for the most promising untried depot sets and restricted-master MIPs over the
elementary columns (all depots, or the depots of the incumbent or of the lowest-bound set fixed). Every plan is
checked by route_cost and polished by inter-route local search before it is published.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.clrp.a_data import CLRPInstance, load_prodhon
from src.bd.clrp.b_mip_xl import route_cost
from src.bd.clrp.c_heuristic import plan_for, two_opt

INF = float("inf")


# ---------------------------------------------------------------- pricing

class NGPricer:
    """ng-route pricing for one CLRP instance. Customers are 0..n-1 here (node m + i in the travel matrix)."""

    def __init__(self, inst: CLRPInstance, cost: np.ndarray, ng: int = 8, near: int = 12) -> None:
        m, n = inst.m, inst.n
        self.m, self.n, self.Q = m, n, int(inst.vehicle_capacity)
        self.q = inst.demand.astype(np.int64)
        self.C = cost[m:, m:].astype(float).copy()
        np.fill_diagonal(self.C, INF)
        self.c_out = cost[:m, m:].astype(float)  # depot -> customer
        self.c_in = cost[m:, :m].T.astype(float).copy()  # customer -> depot, indexed [d, i]
        k = max(0, min(ng - 1, n - 1))
        self.k = k
        self.nm = 1 << k  # ng-memory masks over each customer's k nearest other customers
        nb = np.argsort(self.C, axis=1, kind="stable")[:, :k]
        bitpos = np.full((n, n), -1, dtype=np.int64)
        if k:
            bitpos[np.arange(n)[:, None], nb] = np.arange(k)[None, :]
        self.bitpos = bitpos  # bitpos[i, j]: position of j in N(i) \ {i}, or -1
        self.inN = bitpos >= 0
        # conv[i, j, mask]: ng-memory after the move i -> j, in j's coordinates.
        base = np.where(bitpos.T >= 0, np.left_shift(1, np.maximum(bitpos.T, 0)), 0)  # i in N(j)
        if k:
            bp = bitpos[np.arange(n)[None, :, None], nb[:, None, :]]  # [i, j, t] = position of nb[i, t] in N(j)
            P = np.where(bp >= 0, np.left_shift(1, np.maximum(bp, 0)), 0).reshape(n * n, k)
            bits = (np.arange(self.nm)[:, None] >> np.arange(k)[None, :]) & 1  # (nm, k)
            conv = base.reshape(n * n, 1) + P @ bits.T
        else:
            conv = base.reshape(n * n, 1)
        self.conv = conv.astype(np.int64).reshape(-1)  # flat index (i * n + j) * nm + mask
        nn = min(near, n - 1)
        self.near = np.argsort(self.C, axis=1, kind="stable")[:, :max(nn, 1)] if n > 1 else np.zeros((n, 1), dtype=np.int64)

    def completion(self, Cc: np.ndarray, p: np.ndarray, c_in: np.ndarray) -> np.ndarray:
        """B[j, r]: lower bound on the reduced cost of completing a path at customer j (its prize already counted)
        back to the depot with at most r units of remaining capacity. q-routes with 2-cycle elimination where an
        ng-route forbids it (j -> k -> j with j in N(k)), so it underestimates every ng-route completion."""
        n, Q, q = self.n, self.Q, self.q
        ar = np.arange(n)
        B1 = np.full((n, Q + 1), INF)
        B2 = np.full((n, Q + 1), INF)
        S1 = np.full((n, Q + 1), -1, dtype=np.int64)
        cp = Cc - p[None, :]
        # Every demand is at least qmin, so the levels r0 .. r0 + qmin - 1 only read levels below r0: one block per step.
        step = max(1, int(q.min()))
        for r0 in range(0, Q + 1, step):
            R = np.arange(r0, min(Q + 1, r0 + step))
            rr = R[None, :] - q[:, None]  # (k, r)
            valid = rr >= 0
            rrc = np.maximum(rr, 0)
            b1 = np.where(valid, B1[ar[:, None], rrc], INF)
            V = cp[:, :, None] + b1[None, :, :]  # V[j, k, r]
            s1 = np.where(valid, S1[ar[:, None], rrc], -1)
            ks, rs = np.nonzero(s1 >= 0)
            if len(ks):
                js = s1[ks, rs]
                fix = self.inN[ks, js]
                ks, rs, js = ks[fix], rs[fix], js[fix]
                V[js, ks, rs] = cp[js, ks] + B2[ks, rrc[ks, rs]]
            a1 = np.argmin(V, axis=1)  # (j, r)
            v1 = np.take_along_axis(V, a1[:, None, :], axis=1)[:, 0, :]
            np.put_along_axis(V, a1[:, None, :], INF, axis=1)
            v2 = V.min(axis=1)
            home = c_in[:, None] < v1
            B1[:, R] = np.where(home, c_in[:, None], v1)
            S1[:, R] = np.where(home, -1, a1)
            B2[:, R] = np.where(home, v1, np.minimum(v2, c_in[:, None]))
        return B1

    def price(self, p: np.ndarray, const: float, Cc: np.ndarray, c_out: np.ndarray, c_in: np.ndarray, *, exact: bool,
              beam: int = 0, max_cols: int = 30, deadline: float = INF, tol: float = 1e-6, max_labels: int = 400_000) -> dict[str, Any]:
        """Labeling for one depot: customer prizes p, customer arc costs Cc, depot -> customer costs c_out and
        customer -> depot costs c_in (inf = forbidden; a customer the depot may not serve has c_out and its Cc column
        at inf), route constant const (vehicle cost minus constant duals). exact=True: all arcs, no beam; the minimum
        reduced cost and the minimum of min(0, reduced cost / load) over all ng-routes are returned (complete=True
        unless the deadline cut the search). exact=False: near-neighbour arcs and a label beam. Both modes prune
        labels whose completion bound cannot reach a negative reduced cost. An exact round whose label count at one
        depth exceeds max_labels continues with a beam of that size and is reported incomplete (no bound from it)."""
        n, Q, q, nm = self.n, self.Q, self.q, self.nm
        B = self.completion(Cc, p, c_in)
        Bf = B.reshape(-1)
        W = Q + 1
        D = np.full(n * nm * W, INF)  # dominance surface: min cost over labels at (j, mask' subset of mask, load' <= load)
        start = const + c_out - p
        j0 = np.nonzero(np.isfinite(start) & (q <= Q))[0]
        cost0 = start[j0]
        load0 = q[j0]
        keep = cost0 + Bf[j0 * W + (Q - load0)] < -tol
        node, load, cost = j0[keep], load0[keep], cost0[keep]
        mask = np.zeros(len(node), dtype=np.int64)
        D[(node * nm + mask) * W + load] = cost
        self._close(D, np.unique(node), W)
        g_node, g_parent = [node], [np.full(len(node), -1, dtype=np.int64)]
        lid = np.arange(len(node))
        nlab = len(node)
        found_rc, found_lid, found_load = [], [], []
        min_ratio, min_rc = 0.0, 0.0
        complete = True
        depth = 0
        while len(node):
            depth += 1
            rc = cost + c_in[node]
            neg = rc < -tol
            if neg.any():
                found_rc.append(rc[neg]); found_lid.append(lid[neg]); found_load.append(load[neg])
                min_ratio = min(min_ratio, float((rc[neg] / load[neg]).min()))
                min_rc = min(min_rc, float(rc[neg].min()))
            if time.perf_counter() > deadline:
                complete = False
                break
            # Extensions.
            parts = []
            L = len(node)
            width = n if exact else self.near.shape[1]
            step = max(1, 1_000_000 // max(1, width))
            for s in range(0, L, step):
                sl = slice(s, min(L, s + step))
                nd, ms, ld0, cs = node[sl], mask[sl], load[sl], cost[sl]
                if exact:
                    J = np.broadcast_to(np.arange(n)[None, :], (len(nd), n))
                    cc = cs[:, None] + Cc[nd] - p[None, :]
                    bp = self.bitpos[nd]
                else:
                    J = self.near[nd]
                    cc = cs[:, None] + Cc[nd[:, None], J] - p[J]
                    bp = self.bitpos[nd[:, None], J]
                ld = ld0[:, None] + q[J]
                ok = (ld <= Q) & np.isfinite(cc)
                ok &= (bp < 0) | (((ms[:, None] >> np.maximum(bp, 0)) & 1) == 0)
                ok &= cc + Bf[J * W + np.where(ok, Q - ld, 0)] < -tol
                li, col = np.nonzero(ok)
                if len(li):
                    jj = J[li, col]
                    parts.append((li + s, jj, cc[li, col], ld[li, col]))
            if not parts:
                break
            li = np.concatenate([x[0] for x in parts])
            jj = np.concatenate([x[1] for x in parts])
            cc = np.concatenate([x[2] for x in parts])
            ld = np.concatenate([x[3] for x in parts])
            nmask = self.conv[(node[li] * n + jj) * nm + mask[li]]
            cell = (jj * nm + nmask) * W + ld
            alive = cc < D[cell]
            li, jj, cc, ld, nmask, cell = li[alive], jj[alive], cc[alive], ld[alive], nmask[alive], cell[alive]
            if not len(li):
                break
            order = np.lexsort((cc, cell))
            first = np.ones(len(order), dtype=bool)
            first[1:] = cell[order][1:] != cell[order][:-1]
            sel = order[first]
            li, jj, cc, ld, nmask, cell = li[sel], jj[sel], cc[sel], ld[sel], nmask[sel], cell[sel]
            # Mutual dominance among the new labels and update of the surface.
            uj, inv = np.unique(jj, return_inverse=True)
            Dn = np.full((len(uj), nm, W), INF)
            Dn[inv, nmask, ld] = cc
            self._zeta(Dn)
            np.minimum.accumulate(Dn, axis=2, out=Dn)
            keep = cc <= Dn[inv, nmask, ld]
            blk = D.reshape(n, nm, W)
            blk[uj] = np.minimum(blk[uj], Dn)
            li, jj, cc, ld, nmask = li[keep], jj[keep], cc[keep], ld[keep], nmask[keep]
            cap_ = beam if not exact else max_labels
            if cap_ and len(li) > cap_:
                complete = complete and not exact
                score = cc + Bf[jj * W + (Q - ld)]
                top = np.argpartition(score, cap_ - 1)[:cap_]
                li, jj, cc, ld, nmask = li[top], jj[top], cc[top], ld[top], nmask[top]
            g_node.append(jj)
            g_parent.append(lid[li])
            lid = np.arange(nlab, nlab + len(jj))
            nlab += len(jj)
            node, load, cost, mask = jj, ld, cc, nmask
        routes = []
        if found_rc:
            rcs = np.concatenate(found_rc)
            lids = np.concatenate(found_lid)
            gn = np.concatenate(g_node)
            gp_ = np.concatenate(g_parent)
            seen = set()
            for t in np.argsort(rcs, kind="stable"):
                path = []
                x = int(lids[t])
                while x >= 0:
                    path.append(int(gn[x]))
                    x = int(gp_[x])
                path.reverse()
                key = tuple(sorted(path))
                if key in seen:
                    continue
                seen.add(key)
                routes.append((float(rcs[t]), path))
                if len(routes) >= max_cols:
                    break
        return {"routes": routes, "min_rc": min_rc, "min_ratio": min_ratio, "complete": complete and exact, "labels": nlab, "depth": depth}

    def _zeta(self, Dn: np.ndarray) -> None:
        """In place: Dn[j, mask, l] = min over submasks of mask (memory subset dominance)."""
        if not self.k:
            return
        shape = Dn.shape
        X = Dn.reshape((shape[0],) + (2,) * self.k + (shape[2],))
        for t in range(self.k):
            ax = 1 + (self.k - 1 - t)  # bit t of the mask is axis (k - t) of the reshaped array (C order)
            lo = [slice(None)] * X.ndim
            hi = [slice(None)] * X.ndim
            lo[ax], hi[ax] = 0, 1
            np.minimum(X[tuple(hi)], X[tuple(lo)], out=X[tuple(hi)])

    def _close(self, D: np.ndarray, nodes: np.ndarray, W: int) -> None:
        blk = D.reshape(self.n, self.nm, W)
        sub = blk[nodes]
        self._zeta(sub)
        np.minimum.accumulate(sub, axis=2, out=sub)
        blk[nodes] = sub


# ---------------------------------------------------------------- primal helpers

def plan_routes(plan: list[tuple[int, list[int]]]) -> list[dict[str, Any]]:
    return [{"depot": int(d), "customers": [int(i) for i in path]} for d, path in plan if path]


class LocalSearch:
    """Inter-route relocate and swap over all used depots (granular: only routes holding one of a customer's near
    neighbours), 2-opt* tail exchanges between routes (also of different depots), route-to-depot reassignment, and
    2-opt inside each changed route. Route and depot capacities are kept; moves that empty a route or a depot gain its
    vehicle or opening cost; no depot is opened."""

    def __init__(self, inst: CLRPInstance, cost: np.ndarray, near: int = 15) -> None:
        self.m, self.n, self.Q = inst.m, inst.n, inst.vehicle_capacity
        self.F = float(inst.vehicle_cost)
        self.O = inst.opening_cost.tolist()
        self.q = inst.demand.astype(int).tolist()
        self.W = inst.depot_capacity.tolist()
        self.c = cost.tolist()
        self.cost = cost
        C = cost[inst.m:, inst.m:].astype(float).copy()
        np.fill_diagonal(C, INF)
        self.near = np.argsort(C, axis=1, kind="stable")[:, :min(near, max(1, inst.n - 1))].tolist()

    def length(self, d: int, path: list[int]) -> float:
        c, m = self.c, self.m
        seq = [d] + [m + i for i in path] + [d]
        return sum(c[a][b] for a, b in zip(seq, seq[1:]))

    def improve(self, plan: list[tuple[int, list[int]]], deadline: float) -> list[tuple[int, list[int]]]:
        m, c, q, Q, W = self.m, self.c, self.q, self.Q, self.W
        routes = [[d, list(p)] for d, p in plan if p]
        load = [sum(q[i] for i in p) for _, p in routes]
        dload = [0.0] * m
        for (d, _), l in zip(routes, load):
            dload[d] += l
        where = {}
        for r, (_, p) in enumerate(routes):
            for i in p:
                where[i] = r

        def seq(r):
            d, p = routes[r]
            return [d] + [m + i for i in p] + [d]

        def reopt(r):
            d, p = routes[r]
            if len(p) >= 3:
                routes[r][1] = [x - m for x in two_opt(d, [m + i for i in p], self.cost)]

        improved = True
        while improved and time.perf_counter() < deadline:
            improved = False
            for i in range(self.n):
                ra = where[i]
                da, pa = routes[ra]
                sa = seq(ra)
                pos = pa.index(i) + 1
                gain = c[sa[pos - 1]][m + i] + c[m + i][sa[pos + 1]] - c[sa[pos - 1]][sa[pos + 1]]
                gain_move = gain + (self.F if len(pa) == 1 else 0.0)
                best = None
                cands = {where[j] for j in self.near[i]}
                for rb in cands:
                    if rb == ra:
                        continue
                    db, pb = routes[rb]
                    if load[rb] + q[i] > Q or (db != da and dload[db] + q[i] > W[db]):
                        continue
                    sb = seq(rb)
                    g = gain_move + (self.O[da] if db != da and dload[da] == q[i] else 0.0)
                    for k in range(len(sb) - 1):
                        delta = c[sb[k]][m + i] + c[m + i][sb[k + 1]] - c[sb[k]][sb[k + 1]] - g
                        if delta < -1e-9 and (best is None or delta < best[0]):
                            best = (delta, "move", rb, k)
                    # Swap i with a near customer j of route rb.
                    for kj, j in enumerate(pb):
                        if q[j] - q[i] + load[ra] > Q or q[i] - q[j] + load[rb] > Q:
                            continue
                        if da != db and (dload[da] - q[i] + q[j] > W[da] or dload[db] - q[j] + q[i] > W[db]):
                            continue
                        a1, b1 = sa[pos - 1], sa[pos + 1]
                        a2, b2 = sb[kj], sb[kj + 2]
                        delta = (c[a1][m + j] + c[m + j][b1] - c[a1][m + i] - c[m + i][b1]
                                 + c[a2][m + i] + c[m + i][b2] - c[a2][m + j] - c[m + j][b2])
                        if delta < -1e-9 and (best is None or delta < best[0]):
                            best = (delta, "swap", rb, kj)
                if best is None:
                    continue
                _, kind, rb, k = best
                db = routes[rb][0]
                if kind == "move":
                    routes[ra][1].remove(i)
                    routes[rb][1].insert(k, i)
                    load[ra] -= q[i]; load[rb] += q[i]
                    dload[da] -= q[i]; dload[db] += q[i]
                    where[i] = rb
                else:
                    j = routes[rb][1][k]
                    pa_ = routes[ra][1]
                    pa_[pa_.index(i)] = j
                    routes[rb][1][k] = i
                    load[ra] += q[j] - q[i]; load[rb] += q[i] - q[j]
                    dload[da] += q[j] - q[i]; dload[db] += q[i] - q[j]
                    where[i], where[j] = rb, ra
                reopt(ra); reopt(rb)
                improved = True
            # 2-opt*: route a keeps its head up to i and takes the tail of route b from its near customer j on.
            for i in range(self.n):
                if time.perf_counter() > deadline:
                    break
                ra = where[i]
                da, pa = routes[ra]
                p = pa.index(i)
                for j in self.near[i]:
                    rb = where[j]
                    if rb == ra:
                        continue
                    db, pb = routes[rb]
                    s_ = pb.index(j)
                    na, nb = pa[:p + 1] + pb[s_:], pb[:s_] + pa[p + 1:]
                    la, lb = sum(q[x] for x in na), sum(q[x] for x in nb)
                    if la > Q or lb > Q:
                        continue
                    if da != db and (dload[da] - load[ra] + la > W[da] or dload[db] - load[rb] + lb > W[db]):
                        continue
                    old = self.length(da, pa) + self.length(db, pb)
                    new = self.length(da, na) + (self.length(db, nb) if nb else -self.F)
                    if new < old - 1e-9:
                        routes[ra][1], routes[rb][1] = na, nb
                        dload[da] += la - load[ra]
                        dload[db] += lb - load[rb]
                        load[ra], load[rb] = la, lb
                        for x in na:
                            where[x] = ra
                        for x in nb:
                            where[x] = rb
                        reopt(ra); reopt(rb)
                        improved = True
                        break
            # Route-to-depot reassignment (keeps the depot set: only depots already used).
            used = sorted({d for d, p in routes if p})
            for r, (d, p) in enumerate(routes):
                if not p:
                    continue
                f, l = m + p[0], m + p[-1]
                base = c[d][f] + c[l][d] + (self.O[d] if dload[d] == load[r] else 0.0)
                for e in used:
                    if e == d or dload[e] <= 0 or dload[e] + load[r] > W[e]:
                        continue  # only depots that still serve customers (an emptied depot is closed)
                    alt = min(c[e][f] + c[l][e], c[e][l] + c[f][e])
                    if alt < base - 1e-9:
                        if c[e][l] + c[f][e] < c[e][f] + c[l][e]:
                            routes[r][1] = p[::-1]
                        routes[r][0] = e
                        dload[d] -= load[r]; dload[e] += load[r]
                        improved = True
                        break
        return [(d, p) for d, p in routes if p]


def repair_cover(plan: list[tuple[int, list[int]]], c: list[list[float]], m: int) -> list[tuple[int, list[int]]]:
    """A covering plan made a partition: each repeated customer stays only where removing it saves least
    (removal never increases a route's cost: ceil'd Euclidean costs satisfy the triangle inequality)."""
    routes = [[d, list(p)] for d, p in plan]
    count: dict[int, int] = {}
    for _, p in routes:
        for i in p:
            count[i] = count.get(i, 0) + 1
    for i, k in count.items():
        while k > 1:
            best = None
            for r, (d, p) in enumerate(routes):
                if i in p:
                    s = [d] + [m + x for x in p] + [d]
                    t = p.index(i) + 1
                    save = c[s[t - 1]][s[t]] + c[s[t]][s[t + 1]] - c[s[t - 1]][s[t + 1]]
                    if best is None or save > best[0]:
                        best = (save, r)
            routes[best[1]][1].remove(i)
            k -= 1
    return [(d, p) for d, p in routes if p]


# ---------------------------------------------------------------- cuts

def separate_rcc(X: np.ndarray, deg: np.ndarray, q: np.ndarray, Q: float, max_size: int, min_viol: float = 0.05) -> list[tuple[float, np.ndarray]]:
    """Rounded capacity cuts x(delta(U)) >= 2 ceil(q(U) / Q) violated by LP edge flows. X: symmetric customer-customer
    flows; deg: total flow at each customer (depot edges included). Greedy growth from every customer at once: U takes
    the outside customer with the largest flow into U while one exists. Returns (violation, membership), most violated
    first, without duplicates."""
    n = len(q)
    rows = np.arange(n)
    inU = np.zeros((n, n), dtype=bool)
    inU[rows, rows] = True
    conn = X.copy()
    delta = deg.astype(float).copy()
    load = q.astype(float).copy()
    alive = np.ones(n, dtype=bool)
    found: dict[bytes, tuple[float, np.ndarray]] = {}
    for _ in range(max_size):
        viol = 2.0 * np.ceil(load / Q - 1e-9) - delta
        for r in np.nonzero(alive & (viol > min_viol))[0]:
            key = np.packbits(inU[r]).tobytes()
            if key not in found:
                found[key] = (float(viol[r]), inU[r].copy())
        cand = np.where(inU, -1.0, conn)
        j = cand.argmax(axis=1)
        alive &= cand[rows, j] > 1e-6
        a = np.nonzero(alive)[0]
        if not len(a):
            break
        ja = j[a]
        delta[a] += deg[ja] - 2.0 * conn[a, ja]
        conn[a] += X[ja]
        load[a] += q[ja]
        inU[a, ja] = True
    return sorted(found.values(), key=lambda t: -t[0])


# ---------------------------------------------------------------- reference

class CLRPColumnGenerationXL:
    """Engineered branch-and-price reference for the CLRP; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0, ng: int = 8,
                 near: int = 12, beam: int = 3000, max_cols: int = 25, heuristic_pricing: bool | None = None,
                 rcc: bool = True, verbose: bool = False) -> None:
        self.time_limit = time_limit
        self.threads = threads
        self.on_incumbent = on_incumbent or (lambda routes: None)  # receives [{"depot", "customers"}] plans
        self.on_bound = on_bound or (lambda bound: None)  # receives every new valid global lower bound
        self.soft_mem_gb = soft_mem_gb
        self.seed = seed
        self.ng, self.near, self.beam, self.max_cols = ng, near, beam, max_cols
        # None: near-neighbour beam pricing between exact rounds only when routes can be long (Q / min demand > 8):
        # with short routes exact pricing is as fast per round (measured on Prodhon "a" vs "b" instances).
        self.heuristic_pricing = heuristic_pricing
        self.max_labels = 400_000  # per depth in exact pricing; doubled whenever a capped round finds no column
        self.rcc = rcc
        self.verbose = verbose
        self.trace: list[dict[str, Any]] = []

    # ------------------------------------------------------------ setup

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        return self.solve(load_prodhon(data_path))

    def left(self) -> float:
        return self.deadline - time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def log(self, stage: str) -> None:
        self.trace.append({"t": round(self.elapsed(), 3), "incumbent": self.ub if self.ub < INF else None,
                           "bound": self.lb if self.lb > -INF else None, "stage": stage})
        if self.verbose:
            print(self.trace[-1], flush=True)

    def solve(self, inst: CLRPInstance) -> dict[str, Any]:
        if not hasattr(self, "t0"):
            self.t0 = time.perf_counter()
            self.deadline = self.t0 + self.time_limit
        self.inst = inst
        m, n = inst.m, inst.n
        self.m, self.n = m, n
        self.cost = inst.travel("ceil")
        self.c = self.cost.tolist()
        self.q = inst.demand.astype(np.int64)
        self.Q = int(inst.vehicle_capacity)
        self.W = inst.depot_capacity.astype(float)
        self.O = inst.opening_cost.astype(float)
        self.F = float(inst.vehicle_cost)
        self.qtot = float(self.q.sum())
        self.K = int(math.ceil(self.qtot / self.Q - 1e-9))
        self.integral = bool(np.all(self.cost == np.round(self.cost)) and np.all(self.O == np.round(self.O)) and self.F == round(self.F))
        self.ub, self.lb = INF, -INF
        self.best_plan: list[tuple[int, list[int]]] | None = None
        self.cols: list[dict[str, Any]] = []
        self.stats = {"cg_iters": 0, "exact_rounds": 0, "heuristic_rounds": 0, "lp_seconds": 0.0, "pricing_seconds": 0.0,
                      "exact_seconds": 0.0, "mip_runs": 0, "mip_seconds": 0.0, "ls_seconds": 0.0, "nodes": 0, "labels": 0,
                      "cuts": 0, "cut_rounds": 0, "removed_columns": 0, "stuck_nodes": 0, "root_converged": False,
                      "root_bound": None, "root_lp": None, "root_seconds": None, "max_depth": 0}
        if any(self.q > self.Q) or self.W.sum() < self.qtot:
            return self.result("infeasible")
        if m > 16:
            raise ValueError("depot-set enumeration supports at most 16 depots")
        # Depot sets that can hold the demand, as membership rows; best[s] bounds every plan opening exactly set s.
        sets = [s for s in range(1, 1 << m) if sum(self.W[d] for d in range(m) if s >> d & 1) >= self.qtot - 1e-9]
        self.sets = np.array(sets, dtype=np.int64)
        self.set_pos = {s: k for k, s in enumerate(sets)}
        self.mem = ((self.sets[:, None] >> np.arange(m)[None, :]) & 1).astype(bool)
        self.set_open = self.mem @ self.O
        self.best = np.full(len(sets), -INF)
        self.kmin = int(self.mem.sum(axis=1).min())
        self.pricer = NGPricer(inst, self.cost, ng=self.ng, near=self.near)
        if self.heuristic_pricing is None:
            self.heuristic_pricing = self.Q // int(self.q.min()) > 8
        self.ls = LocalSearch(inst, self.cost)
        self.frontier: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.trivial_bound()
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", self.threads)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()
        self.env = env
        self.memory_limit = False
        try:
            self.build_master()
            self.construct()
            self.search()
        except gp.GurobiError as err:
            if err.errno != GRB.Error.OUT_OF_MEMORY and "memory" not in str(err).lower():
                raise
            self.memory_limit = True
        out = self.result("memory_limit" if self.memory_limit else "optimal" if self.closed() else "time_limit")
        env.dispose()
        return out

    def closed(self) -> bool:
        return self.ub < INF and self.lb >= self.ub - (1.0 - 1e-6 if self.integral else 1e-6 * abs(self.ub))

    def prunable(self, bound: float) -> bool:
        return bound >= self.ub - (1.0 - 1e-6 if self.integral else 1e-9 * abs(self.ub))

    def result(self, reason: str) -> dict[str, Any]:
        ub = self.ub if self.ub < INF else None
        lb = self.lb if self.lb > -INF else None
        return {"best_objective": ub, "global_lb": lb, "global_ub": ub,
                "gap": (ub - lb) / abs(ub) if ub is not None and lb is not None and ub else None,
                "termination_reason": reason, "routes": plan_routes(self.best_plan) if self.best_plan else None,
                "columns": len(self.cols), **self.stats, "total_seconds": self.elapsed(), "trace": self.trace}

    # ------------------------------------------------------------ bounds

    def certify(self, value: float, scale: float) -> float:
        """A computed bound made safe against floating-point error (relative 1e-7 of the summed magnitudes),
        then rounded up when every cost is integral."""
        allowance = 1e-7 * max(1.0, scale)
        return math.ceil(value - allowance) if self.integral else value - allowance

    def node_bound(self, node: dict) -> float:
        if node.get("S") is not None:
            return max(node["lb"], float(self.best[node["S"]]))
        cons = self.consistent(node)
        return float(self.best[cons].min()) if cons.any() else INF

    def consistent(self, node: dict) -> np.ndarray:
        ok = np.ones(len(self.sets), dtype=bool)
        for d in node["open"]:
            ok &= self.mem[:, d]
        for d in node["closed"]:
            ok &= ~self.mem[:, d]
        return ok

    def publish_bound(self, stage: str) -> None:
        nodes = [x for x in self.frontier] + ([self.current] if self.current is not None else [])
        lb = min((self.node_bound(x) for x in nodes), default=INF) if nodes or self.searching else float(self.best.min())
        if self.ub < INF:
            lb = min(lb, self.ub)
        if lb > self.lb + 1e-9 and math.isfinite(lb):
            self.lb = lb
            self.on_bound(lb)
            self.log(stage)

    def trivial_bound(self) -> None:
        """Before any pricing: per depot set, its opening cost, K vehicles and every customer's cheapest incident edge
        (each customer is entered and left once and an edge touches at most two customers; a singleton route may use
        its depot edge twice, so the two cheapest distinct edges would not be valid)."""
        self.searching = False
        full = self.cost.copy()
        np.fill_diagonal(full, INF)
        base = self.F * self.K + float(full[self.m:, :].min(axis=1).sum())
        self.best = np.maximum(self.best, np.array([self.certify(v + base, v + base) for v in self.set_open]))
        self.publish_bound("trivial")

    def lagrangian(self, duals: dict[str, Any], ratio: np.ndarray, priced: np.ndarray, node: dict, stage: str) -> None:
        """Lagrangian bounds at the duals for every depot set of priced depots (see the module docstring). ratio[d]
        is the exact minimum of min(0, reduced cost / load) over the ng-routes of depot d at this node. At a node
        with route-level branching decisions the bound holds only inside the node and goes to node["lb"]."""
        pi, mu, rho, sigma = duals["pi"], duals["mu"], duals["rho"], duals["sigma"]
        const = float(pi.sum() + sigma * self.K + duals["beta"] @ self.cut_rhs_arr() + duals["eta"] @ duals["eta_rhs"])
        ytil = self.O + mu * self.W + rho.sum(axis=0)
        r = np.minimum(np.where(priced, ratio, 0.0), -1e-6 / float(self.q.min()))
        ok = ~(self.mem & ~priced[None, :]).any(axis=1)
        if node.get("rules"):
            ok &= np.arange(len(self.sets)) == node["S"]
        order = np.argsort(r, kind="stable")
        Wo = np.where(self.mem[:, order], self.W[order][None, :], 0.0)
        before = np.cumsum(Wo, axis=1) - Wo
        amount = np.clip(self.qtot - before, 0.0, Wo)
        L = const + self.mem @ ytil + amount @ r[order]
        scale = float(np.abs(pi).sum() + sigma * self.K + np.abs(duals["beta"]).sum() * 2 * self.K + np.abs(duals["eta"]).sum()
                      + np.abs(ytil).sum() + self.qtot * np.abs(r).max() + self.set_open.max())
        idx = np.nonzero(ok)[0]
        vals = np.array([self.certify(v, scale) for v in L[idx]])
        if node.get("rules"):
            node["lb"] = max(node["lb"], float(vals[0]))
        else:
            self.best[idx] = np.maximum(self.best[idx], vals)
        self.publish_bound(stage)

    # ------------------------------------------------------------ master

    def build_master(self) -> None:
        m, n = self.m, self.n
        md = gp.Model(env=self.env)
        md.Params.Method = 0
        self.y = [md.addVar(lb=0.0, ub=1.0, obj=float(self.O[d])) for d in range(m)]
        self.cover = [md.addLConstr(gp.LinExpr(), GRB.GREATER_EQUAL, 1.0) for _ in range(n)]
        self.cap = [md.addLConstr(gp.LinExpr(-self.W[d], self.y[d]), GRB.LESS_EQUAL, 0.0) for d in range(m)]
        self.link = [[md.addLConstr(gp.LinExpr(-1.0, self.y[d]), GRB.LESS_EQUAL, 0.0) for d in range(m)] for _ in range(n)]
        self.veh = md.addLConstr(gp.LinExpr(), GRB.GREATER_EQUAL, float(self.K))
        md.addLConstr(gp.LinExpr(self.W.tolist(), self.y), GRB.GREATER_EQUAL, self.qtot)
        md.addLConstr(gp.quicksum(self.y), GRB.GREATER_EQUAL, float(self.kmin))
        self.link_flat = [self.link[i][d] for i in range(n) for d in range(m)]
        self.master = md
        self.big_m = 2.0 * float(self.O.sum() + n * (self.F + 2.0 * self.cost.max()))
        self.col_index: dict[tuple, int] = {}
        self.L = self.Q // int(self.q.min()) + 2  # longest route (depot ends included)
        self.seq = np.full((1024, self.L), n, dtype=np.int64)  # column node sequences, depot = n
        self.col_depot = np.zeros(1024, dtype=np.int64)
        self.cut_rows: list[gp.Constr] = []
        self.cut_mem = np.zeros((0, n + 1), dtype=bool)  # membership (depot column always False)
        self.cut_rhs: list[float] = []
        self.erows: dict[tuple[int, int], gp.Constr] = {}
        self.rules: dict[str, Any] = {"closed": set(), "z0": set(), "z1": {}, "e0": set()}
        for d in range(m):  # singletons keep every node's master feasible
            for i in range(n):
                self.add_column(d, [i])

    def cut_rhs_arr(self) -> np.ndarray:
        return np.array(self.cut_rhs, dtype=float)

    def route_travel(self, d: int, path: list[int]) -> float:
        c, m = self.c, self.m
        s = [d] + [m + i for i in path] + [d]
        return float(sum(c[a][b] for a, b in zip(s, s[1:])))

    def forbidden(self, d: int, path: list[int]) -> bool:
        R = self.rules
        if d in R["closed"]:
            return True
        for i in path:
            if (i, d) in R["z0"] or R["z1"].get(i, d) != d:
                return True
        if R["e0"]:
            for a, b in zip(path, path[1:]):
                if (min(a, b), max(a, b)) in R["e0"]:
                    return True
        return False

    @staticmethod
    def key(d: int, path: list[int]) -> tuple:
        """Columns are identified by depot and visiting sequence (up to reversal): cut and edge-row coefficients
        depend on the order, so two orders of one customer set are different columns."""
        t = tuple(path)
        return (d, min(t, t[::-1]))

    def add_column(self, d: int, path: list[int], polish: bool = False) -> bool:
        path = [int(i) for i in path]
        elem = len(set(path)) == len(path)
        if polish and elem and len(path) >= 3:
            path = [x - self.m for x in two_opt(d, [self.m + i for i in path], self.cost)]
        cost = self.route_travel(d, path) + self.F
        key = self.key(d, path)
        if key in self.col_index:
            return False
        load = float(self.q[path].sum())
        if load > self.Q or len(path) + 2 > self.L:
            return False
        cnt: dict[int, int] = {}
        for i in path:
            cnt[i] = cnt.get(i, 0) + 1
        rows = [self.cover[i] for i in cnt] + [self.cap[d]] + [self.link[i][d] for i in cnt] + [self.veh]
        coef = [float(v) for v in cnt.values()] + [load] + [float(v) for v in cnt.values()] + [1.0]
        k = len(self.cols)
        if k >= len(self.seq):
            self.seq = np.vstack([self.seq, np.full_like(self.seq, self.n)])
            self.col_depot = np.concatenate([self.col_depot, np.zeros_like(self.col_depot)])
        self.seq[k] = self.n
        self.seq[k, 1:len(path) + 1] = path
        self.col_depot[k] = d
        if len(self.cut_rows):
            ins = self.cut_mem[:, self.seq[k]]
            cross = (ins[:, 1:] != ins[:, :-1]).sum(axis=1)
            for t in np.nonzero(cross)[0]:
                rows.append(self.cut_rows[t])
                coef.append(float(cross[t]))
        for a, b in zip(path, path[1:]):
            e = self.erows.get((min(a, b), max(a, b)))
            if e is not None:
                rows.append(e)
                coef.append(1.0)
        var = self.master.addVar(lb=0.0, ub=0.0 if self.forbidden(d, path) else INF, obj=cost, column=gp.Column(coef, rows))
        self.col_index[key] = k
        self.cols.append({"depot": d, "path": path, "cost": cost, "load": load, "elem": elem, "var": var})
        return True

    def edge_count(self, i: int, j: int) -> np.ndarray:
        S = self.seq[:len(self.cols)]
        return (((S[:, :-1] == i) & (S[:, 1:] == j)) | ((S[:, :-1] == j) & (S[:, 1:] == i))).sum(axis=1)

    def contains(self, i: int) -> np.ndarray:
        return (self.seq[:len(self.cols)] == i).any(axis=1)

    def set_node(self, node: dict) -> None:
        R = {"closed": set(node["closed"]), "z0": set(), "z1": {}, "e0": set()}
        e1 = set()
        for rule in node.get("rules", ()):
            if rule[0] == "z":
                _, i, d, v = rule
                if v:
                    R["z1"][i] = d
                else:
                    R["z0"].add((i, d))
            else:
                _, i, j, v = rule
                (e1.add if v else R["e0"].add)((i, j))
        self.rules = R
        for d in range(self.m):
            self.y[d].LB = 1.0 if d in node["open"] else 0.0
            self.y[d].UB = 0.0 if d in R["closed"] else 1.0
        for e, row in self.erows.items():
            row.RHS = 1.0 if e in e1 else 0.0
        k = len(self.cols)
        dep = self.col_depot[:k]
        bad = np.isin(dep, list(R["closed"])) if R["closed"] else np.zeros(k, dtype=bool)
        for i, d in R["z0"]:
            bad |= self.contains(i) & (dep == d)
        for i, d in R["z1"].items():
            bad |= self.contains(i) & (dep != d)
        for i, j in R["e0"]:
            bad |= self.edge_count(i, j) > 0
        self.master.setAttr("UB", [c["var"] for c in self.cols], np.where(bad, 0.0, INF).tolist())

    def solve_lp(self) -> str:
        if self.left() <= 0:
            return "time"
        t = time.perf_counter()
        self.master.Params.TimeLimit = max(1e-3, self.left())
        self.master.optimize()
        self.stats["lp_seconds"] += time.perf_counter() - t
        st = self.master.Status
        if st == GRB.MEM_LIMIT:
            raise gp.GurobiError(GRB.Error.OUT_OF_MEMORY, "memory limit")
        return "optimal" if st == GRB.OPTIMAL else "infeasible" if st in (GRB.INFEASIBLE, GRB.INF_OR_UNBD) else "time"

    def duals(self) -> dict[str, Any]:
        md = self.master
        ekeys = list(self.erows)
        return {"pi": np.maximum(np.array(md.getAttr("Pi", self.cover)), 0.0),
                "mu": np.minimum(np.array(md.getAttr("Pi", self.cap)), 0.0),
                "rho": np.minimum(np.array(md.getAttr("Pi", self.link_flat)), 0.0).reshape(self.n, self.m),
                "sigma": max(float(self.veh.Pi), 0.0),
                "beta": np.maximum(np.array(md.getAttr("Pi", self.cut_rows)), 0.0) if self.cut_rows else np.zeros(0),
                "eta": np.maximum(np.array([self.erows[e].Pi for e in ekeys]), 0.0) if ekeys else np.zeros(0),
                "eta_rhs": np.array([self.erows[e].RHS for e in ekeys]) if ekeys else np.zeros(0),
                "ekeys": ekeys}

    def compact(self) -> None:
        """Drops the columns with the largest reduced costs at the last LP once the master is large (singletons and
        incumbent routes stay); exact pricing regenerates any column the LP needs later."""
        k = len(self.cols)
        limit = max(6000, 60 * self.n)
        snap = self.snap
        if k <= limit or snap is None:
            return
        k0 = snap["ncols"]
        keep = np.zeros(k, dtype=bool)
        keep[np.argsort(snap["rc"], kind="stable")[:limit // 2]] = True
        keep[:k0] |= snap["x"] > 1e-9
        keep[k0:] = True  # added after the snapshot
        keep |= (self.seq[:k] < self.n).sum(axis=1) == 1
        if self.best_plan:
            for d, p in self.best_plan:
                t = self.col_index.get(self.key(d, p))
                if t is not None:
                    keep[t] = True
        drop = np.nonzero(~keep)[0]
        self.master.remove([self.cols[t]["var"] for t in drop])
        idx = np.nonzero(keep)[0]
        self.cols = [self.cols[t] for t in idx]
        self.seq[:len(idx)] = self.seq[idx]
        self.col_depot[:len(idx)] = self.col_depot[idx]
        self.col_index = {self.key(c["depot"], c["path"]): t for t, c in enumerate(self.cols)}
        self.stats["removed_columns"] += len(drop)
        self.snap = None
        self.master.update()

    # ------------------------------------------------------------ column generation and cuts

    def arc_costs(self, duals: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Customer arc costs and depot arc costs (m, n) with the cut and edge-row duals subtracted."""
        n = self.n
        Cc = self.pricer.C.copy()
        c_out, c_in = self.pricer.c_out.copy(), self.pricer.c_in.copy()
        beta = duals["beta"]
        if len(beta) and beta.max() > 0:
            Mb = self.cut_mem[:, :n].astype(float)
            s = Mb.T @ beta
            Cc -= s[:, None] + s[None, :] - 2.0 * ((Mb.T * beta) @ Mb)
            c_out -= s[None, :]
            c_in -= s[None, :]
        for (i, j), eta in zip(duals["ekeys"], duals["eta"]):
            Cc[i, j] -= eta
            Cc[j, i] -= eta
        for i, j in self.rules["e0"]:
            Cc[i, j] = Cc[j, i] = INF
        np.fill_diagonal(Cc, INF)
        return Cc, c_out, c_in

    def price_round(self, duals: dict[str, Any], depots: list[int], exact: bool) -> tuple[int, np.ndarray, bool]:
        """Prices the depots at the duals; returns (columns added, min ratio per depot, complete exact round)."""
        t = time.perf_counter()
        ratio = np.zeros(self.m)
        complete = True
        added = 0
        Cc, c_out, c_in = self.arc_costs(duals)
        R = self.rules
        for d in depots:
            if self.left() <= 0:
                complete = False
                break
            p = duals["pi"] + duals["rho"][:, d] + duals["mu"][d] * self.q
            out = c_out[d]
            Cd = Cc
            ban = [i for i, e in R["z0"] if e == d] + [i for i, e in R["z1"].items() if e != d]
            if ban:
                out = out.copy()
                out[ban] = INF
                Cd = Cc.copy()
                Cd[:, ban] = INF
            res = self.pricer.price(p, self.F - duals["sigma"], Cd, out, c_in[d], exact=exact, beam=self.beam,
                                    max_cols=self.max_cols, deadline=self.deadline, max_labels=self.max_labels)
            self.stats["labels"] += res["labels"]
            ratio[d] = res["min_ratio"]
            complete &= res["complete"]
            for _, path in res["routes"]:
                added += self.add_column(d, path)
        dt = time.perf_counter() - t
        self.stats["pricing_seconds"] += dt
        self.stats["exact_rounds" if exact else "heuristic_rounds"] += 1
        if exact:
            self.stats["exact_seconds"] += dt
        return added, ratio, complete and exact

    def snapshot(self) -> None:
        """The LP solution of the last master solve, kept for branching and column clean-up (later column or row
        additions would otherwise hide it)."""
        vars_ = [c["var"] for c in self.cols]
        self.snap = {"x": np.array(self.master.getAttr("X", vars_)), "rc": np.array(self.master.getAttr("RC", vars_)),
                     "y": [v.X for v in self.y], "ncols": len(self.cols)}

    def lp_solution(self) -> tuple[np.ndarray, list[int]]:
        x = self.snap["x"]
        return x, np.nonzero(x > 1e-9)[0].tolist()

    def flows(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """LP customer-customer edge flows X (symmetric), depot-edge flow per customer, and z[i, d] assignment."""
        n = self.n
        x, pos = self.lp_solution()
        X = np.zeros((n + 1, n + 1))
        z = np.zeros((n, self.m))
        for t in pos:
            col, v = self.cols[t], x[t]
            s = [n] + col["path"] + [n]
            for a, b in zip(s, s[1:]):
                X[a, b] += v
                X[b, a] += v
            for i in col["path"]:
                z[i, col["depot"]] += v
        return X[:n, :n], X[:n, n], z

    def add_cuts(self) -> int:
        X, dep, _ = self.flows()
        deg = X.sum(axis=1) + dep
        found = separate_rcc(X, deg, self.q, self.Q, max_size=min(self.n, 30))
        if not found:
            return 0
        room = max(0, 40 * self.n // 10 + 200 - len(self.cut_rows))
        found = found[:min(room, max(10, self.n // 4))]
        if not found:
            return 0
        k = len(self.cols)
        S = self.seq[:k]
        vars_ = [c["var"] for c in self.cols]
        mems = []
        for _, U in found:
            ext = np.append(U, False)
            ins = ext[S]
            cross = (ins[:, 1:] != ins[:, :-1]).sum(axis=1)
            nz = np.nonzero(cross)[0]
            rhs = 2.0 * math.ceil(float(self.q[U].sum()) / self.Q - 1e-9)
            row = self.master.addLConstr(gp.LinExpr(cross[nz].astype(float).tolist(), [vars_[t] for t in nz]), GRB.GREATER_EQUAL, rhs)
            self.cut_rows.append(row)
            self.cut_rhs.append(rhs)
            mems.append(ext)
        self.cut_mem = np.vstack([self.cut_mem] + [m_[None, :] for m_ in mems])
        self.stats["cuts"] += len(found)
        self.stats["cut_rounds"] += 1
        return len(found)

    def column_generation(self, node: dict, stage: str, cut_rounds: int) -> str:
        """CG at a node with RCC rounds after each convergence. Returns "converged", "pruned" or "time"."""
        depots = [d for d in range(self.m) if d not in node["closed"]]
        priced = np.zeros(self.m, dtype=bool)
        priced[depots] = True
        heuristic_fail = False
        since_exact = 0
        rounds = 0
        last_lp = -INF
        while self.left() > 0:
            if self.primal_due():
                self.primal_step()
            st = self.solve_lp()
            if st != "optimal":
                return "pruned" if st == "infeasible" else "time"  # an infeasible node master: no plan in the node
            self.stats["cg_iters"] += 1
            node["lp"] = float(self.master.ObjVal)
            duals = self.duals()
            exact = not self.heuristic_pricing or heuristic_fail or since_exact >= 15
            added, ratio, complete = self.price_round(duals, depots, exact=exact)
            if exact:
                since_exact = 0
                heuristic_fail = False
                if complete:
                    self.lagrangian(duals, ratio, priced, node, stage)
                elif added == 0:
                    self.max_labels *= 2
            else:
                since_exact += 1
                heuristic_fail = added == 0
            if self.closed() or self.prunable(self.node_bound(node)):
                return "pruned"
            if complete and added == 0:
                self.snapshot()
                if not self.rcc or rounds >= cut_rounds or node["lp"] < last_lp + 1e-5 * abs(node["lp"]):
                    return "converged"
                last_lp = node["lp"]
                rounds += 1
                if self.add_cuts() == 0:
                    return "converged"
        return "time"

    # ------------------------------------------------------------ primal

    def offer(self, plan: list[tuple[int, list[int]]], stage: str, polish: bool = True) -> bool:
        """Checks a plan with route_cost (the independent checker) and publishes it if it improves."""
        plan = [(int(d), [int(i) for i in p]) for d, p in plan if len(p)]
        obj = route_cost(self.inst, self.cost, plan_routes(plan))
        if obj is None:
            return False
        better = obj < self.ub - 1e-9
        if better:
            self.ub, self.best_plan = obj, plan
            self.on_incumbent(plan_routes(plan))
            self.log(stage)
            self.publish_bound(stage)
        if polish and self.left() > 0:
            t = time.perf_counter()
            improved = self.ls.improve(plan, self.deadline)
            self.stats["ls_seconds"] += time.perf_counter() - t
            obj2 = route_cost(self.inst, self.cost, plan_routes(improved))
            if obj2 is not None and obj2 < self.ub - 1e-9:
                self.ub, self.best_plan = obj2, improved
                self.on_incumbent(plan_routes(improved))
                self.log(stage + "+ls")
                self.publish_bound(stage)
                better = True
            if obj2 is not None:
                for d, p in improved:
                    self.add_column(d, p, polish=True)
        for d, p in plan:
            self.add_column(d, p, polish=True)
        return better

    def try_sets(self, sets: list[int], stage: str) -> None:
        for s in sets:
            if self.left() <= 0:
                return
            plan = plan_for(self.inst, self.cost, [d for d in range(self.m) if s >> d & 1])
            if plan:
                self.offer([(r["depot"], list(r["customers"])) for r in plan], stage)

    def construct(self) -> None:
        """The control's savings / 2-opt / relocate construction for the cheapest-per-capacity depot set and its
        add / drop / swap neighbours, each polished by the inter-route local search."""
        order = sorted(range(self.m), key=lambda d: self.O[d] / self.W[d])
        s, cap = 0, 0.0
        for d in order:
            s |= 1 << d
            cap += self.W[d]
            if cap >= self.qtot:
                break
        self.tried: set[int] = {s}
        self.try_sets([s], "construct")
        until = time.perf_counter() + max(0.5, min(10.0, 0.02 * self.time_limit))
        moves = [s | 1 << d for d in range(self.m) if not s >> d & 1] + [s & ~(1 << d) for d in range(self.m) if s >> d & 1]
        moves += [(s & ~(1 << a)) | 1 << b for a in range(self.m) if s >> a & 1 for b in range(self.m) if not s >> b & 1]
        for mv in moves:
            if time.perf_counter() > until:
                break
            if mv in self.set_pos and mv not in self.tried:
                self.tried.add(mv)
                self.try_sets([mv], "construct")

    def promising_sets(self, k: int, fresh: bool = True) -> list[int]:
        """Depot sets with the smallest bounds (ties: cheaper opening), untried ones only if fresh."""
        out = []
        for idx in np.lexsort((self.set_open, self.best)):
            s = int(self.sets[idx])
            if self.prunable(self.best[idx]):
                break
            if not fresh or s not in self.tried:
                out.append(s)
                if fresh:
                    self.tried.add(s)
                if len(out) >= k:
                    break
        return out

    def primal_due(self) -> bool:
        return time.perf_counter() >= self.next_primal

    def primal_step(self) -> None:
        """Between pricing rounds: a construction for the best untried depot set, then alternately the global restricted
        MIP (first) and a restricted MIP with the depots fixed, to the incumbent's set or the set with the smallest
        bound in turn."""
        self.try_sets(self.promising_sets(1), "lagrangian_set")
        self.primal_steps += 1
        if self.primal_steps % 2 == 1:
            self.restricted_mip(self.mip_limit, "restricted_mip")
        else:
            sets = self.promising_sets(1, fresh=False)
            if self.best_plan and (self.primal_steps % 4 == 2 or not sets):
                sets = [sum(1 << d for d in {d for d, _ in self.best_plan})]
            if sets:
                self.restricted_mip(min(self.mip_limit, max(2.0, 0.5 * self.mip_limit)), "restricted_mip_set", fixed=sets[0])
        self.next_primal = time.perf_counter() + max(self.primal_gap, 0.1 * self.elapsed())

    def restricted_mip(self, limit: float, stage: str, fixed: int | None = None) -> None:
        """Integer master over the elementary columns (covering rows; a solution is repaired to a partition and checked
        by route_cost before it is published from the callback). fixed: a depot set whose depots are all open and the
        only ones used."""
        limit = min(limit, self.left())
        if limit <= 0.2:
            return
        t = time.perf_counter()
        self.stats["mip_runs"] += 1
        cols = [k for k, col in enumerate(self.cols) if col["elem"] and (fixed is None or fixed >> col["depot"] & 1)]
        md = gp.Model(env=self.env)
        md.Params.TimeLimit = limit
        md.Params.MIPFocus = 1
        if fixed is None:
            y = [md.addVar(vtype=GRB.BINARY, obj=float(self.O[d])) for d in range(self.m)]
        else:
            y = [float(fixed >> d & 1) for d in range(self.m)]
            md.ObjCon = float(sum(self.O[d] for d in range(self.m) if fixed >> d & 1))
        lam = [md.addVar(vtype=GRB.BINARY, obj=self.cols[k]["cost"]) for k in cols]
        cover = [gp.LinExpr() for _ in range(self.n)]
        link = {}
        load = [gp.LinExpr() for _ in range(self.m)]
        for v, k in enumerate(cols):
            col = self.cols[k]
            d = col["depot"]
            for i in col["path"]:
                cover[i].addTerms(1.0, lam[v])
                if fixed is None:
                    link.setdefault((i, d), gp.LinExpr()).addTerms(1.0, lam[v])
            load[d].addTerms(col["load"], lam[v])
        for i in range(self.n):
            md.addLConstr(cover[i], GRB.GREATER_EQUAL, 1.0)
        for (i, d), expr in link.items():
            md.addLConstr(expr - y[d], GRB.LESS_EQUAL, 0.0)
        for d in range(self.m):
            if load[d].size():
                md.addLConstr(load[d] - self.W[d] * y[d], GRB.LESS_EQUAL, 0.0)
        if fixed is None:
            md.addLConstr(gp.LinExpr(self.W.tolist(), y), GRB.GREATER_EQUAL, self.qtot)
        if self.best_plan and self.ub < INF:
            md.Params.Cutoff = self.ub - (1.0 - 1e-6 if self.integral else 1e-6)
            pos = {k: v for v, k in enumerate(cols)}
            starts = [pos.get(self.col_index.get(self.key(d, p), -1)) for d, p in self.best_plan]
            if all(s_ is not None for s_ in starts):
                for v in range(len(cols)):
                    lam[v].Start = 0.0
                for s_ in starts:
                    lam[s_].Start = 1.0
                if fixed is None:
                    used = {d for d, _ in self.best_plan}
                    for d in range(self.m):
                        y[d].Start = 1.0 if d in used else 0.0
        found = []

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                vals = model.cbGetSolution(lam)
                plan = repair_cover([(self.cols[cols[v]]["depot"], list(self.cols[cols[v]]["path"])) for v, val in enumerate(vals) if val > 0.5], self.c, self.m)
                found.append(plan)
                self.offer(plan, stage)  # local search is cheap (milliseconds) and often closes depots

        md.optimize(callback)
        if md.Status == GRB.MEM_LIMIT:
            self.memory_limit = True
        md.dispose()
        self.stats["mip_seconds"] += time.perf_counter() - t
        if found and self.best_plan and self.left() > 0:
            self.offer(self.best_plan, stage)  # local search on the incumbent; its routes join the pool

    # ------------------------------------------------------------ search

    def branch(self, node: dict) -> list[dict] | None:
        """Children of a converged node: on the most fractional free depot, then (all depots fixed) on the most
        fractional customer-depot assignment, then on the most fractional customer edge. None: nothing fractional
        (an integral LP solution is offered as a plan)."""
        free = [d for d in range(self.m) if d not in node["open"] and d not in node["closed"]]
        base = {"depth": node["depth"] + 1, "lb": node.get("lb", -INF), "rules": tuple(node.get("rules", ()))}
        if free:
            yv = self.snap["y"]
            d = min(free, key=lambda e: (abs(yv[e] - 0.5), -self.O[e]))
            kids = [dict(base, open=node["open"] | {d}, closed=node["closed"]), dict(base, open=node["open"], closed=node["closed"] | {d})]
            for kid in kids:
                if len(kid["open"]) + len(kid["closed"]) == self.m:
                    kid["S"] = self.set_pos.get(sum(1 << e for e in kid["open"]))
            return [kid for kid in kids if self.consistent(kid).any()]
        S = node["S"]
        X, _, z = self.flows()
        frac = np.abs(z - 0.5)
        frac[(z < 1e-4) | (z > 1 - 1e-4)] = INF
        if np.isfinite(frac).any():
            i, d = np.unravel_index(np.argmin(frac), frac.shape)
            return [dict(base, open=node["open"], closed=node["closed"], S=S, rules=base["rules"] + (("z", int(i), int(d), v),)) for v in (1, 0)]
        up = np.triu(X, 1)
        frac = np.abs(up - 0.5)
        frac[(up < 1e-4) | (up > 1 - 1e-4)] = INF  # an edge fixed to 1 may carry more than 1: never re-branched
        if np.isfinite(frac).any():
            i, j = np.unravel_index(np.argmin(frac), frac.shape)
            e = (int(min(i, j)), int(max(i, j)))
            if e not in self.erows:
                cnt = self.edge_count(*e)
                nz = np.nonzero(cnt)[0]
                row = self.master.addLConstr(gp.LinExpr(cnt[nz].astype(float).tolist(), [self.cols[t]["var"] for t in nz]), GRB.GREATER_EQUAL, 0.0)
                # Artificial column at a cost above any plan: a node whose edge rows cannot be met gets an LP value,
                # hence a Lagrangian bound, above the incumbent and is pruned.
                self.master.addVar(lb=0.0, obj=self.big_m, column=gp.Column([1.0], [row]))
                self.erows[e] = row
            return [dict(base, open=node["open"], closed=node["closed"], S=S, rules=base["rules"] + (("e",) + e + (v,),)) for v in (1, 0)]
        x, pos = self.lp_solution()
        if all(abs(x[t] - 1.0) < 1e-6 for t in pos) and all(self.cols[t]["elem"] for t in pos):
            self.offer(repair_cover([(self.cols[t]["depot"], list(self.cols[t]["path"])) for t in pos], self.c, self.m), "integral_node")
            node["lb"] = max(node.get("lb", -INF), self.certify(node["lp"], abs(node["lp"])))
        return None

    def search(self) -> None:
        """Best-first branch-and-price (bound ties: deeper first), with primal steps between pricing rounds."""
        T = self.time_limit
        self.mip_limit = max(3.0, min(60.0, 0.03 * T))
        self.primal_gap = max(3.0, min(60.0, 0.03 * T))
        self.primal_steps = 0
        self.next_primal = time.perf_counter() + self.primal_gap
        self.snap = None
        self.searching = True
        self.frontier = [{"open": frozenset(), "closed": frozenset(), "depth": 0, "id": 0, "lb": -INF, "rules": ()}]
        count = 0
        while self.left() > 0 and not self.closed():
            live = [x for x in self.frontier if not x.get("stuck")]
            if not live:
                break
            node = min(live, key=lambda x: (self.node_bound(x), -x["depth"]))
            self.frontier.remove(node)
            self.current = node
            if self.prunable(self.node_bound(node)):
                self.current = None
                self.publish_bound("prune")
                continue
            self.stats["nodes"] += 1
            self.stats["max_depth"] = max(self.stats["max_depth"], node["depth"])
            self.set_node(node)
            root = node["id"] == 0
            status = self.column_generation(node, "root" if root else "node", cut_rounds=50 if root else 3)
            if root:
                self.stats["root_converged"] = status == "converged"
                self.stats["root_lp"] = node.get("lp")
                self.stats["root_bound"] = self.lb
                self.stats["root_seconds"] = self.elapsed()
            if status == "time":
                self.frontier.append(node)
                break
            if status == "pruned":
                self.current = None
                self.publish_bound("prune")
                continue
            kids = self.branch(node)  # reads the LP snapshot, so it comes before clean-up and primal steps
            if kids is None:
                if not self.prunable(self.node_bound(node)):
                    node["stuck"] = True
                    self.stats["stuck_nodes"] += 1
                    self.frontier.append(node)
            else:
                for kid in kids:
                    count += 1
                    kid["id"] = count
                    self.frontier.append(kid)
            self.current = None
            self.publish_bound("branch")
            self.compact()
            if root or self.primal_due():
                self.primal_step()
        self.current = None
        while self.left() > 0.5 and not self.closed():
            before = self.ub
            self.restricted_mip(self.left(), "restricted_mip_final")
            if self.ub >= before:
                break
