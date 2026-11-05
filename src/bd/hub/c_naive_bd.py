"""Naive textbook Benders decomposition for the UMAHLP (scale-up control) and the cut machinery it shares with the
engineered reference (scale-up study; the paper's references are unchanged).

Recourse of a hub vector z (Contreras, Cordeau, Laporte 2011, Sec. 3): per pair (i, j), the path LP of the reduced
formulation of b_mip_xl,
    Q_ij(z) = min sum_p F_p x_p   s.t.  sum_p x_p = 1,   sum_{p uses h} x_p <= z_h,   x >= 0,
over the paths p of the pair (single-hub loops (h, h) and the kept two-hub edges {k, m}); its dual is
    max a - sum_h z_h u_h   s.t.  a - u_h <= F_hh,  a - u_k - u_m <= F_km,  u >= 0.
Closed-form dual family ("ownership duals"): fix any order of the hubs and let every path be owned by its later hub
(a loop by its own hub); let c_h be the cheapest path owned by h. For ANY value a, u_h = (a - c_h)^+ is dual feasible
(a path p owned by h has a - u_h - u_other <= a - (a - c_h) = c_h <= F_p, or a <= c_h <= F_p), so for binary z
    W_ij theta_ij >= W_ij (a - sum_h (a - c_h)^+ z_h)                                       (*)
is a valid optimality cut (the subproblem is always feasible because the master keeps sum_h z_h >= 1).
At a binary point with open set H1, put H1 first: min_{h in H1} c_h = C_ij(H1) (the cheapest open path) and the
closed hubs' c_h never undercut it, so a = C_ij(H1) gives a cut tight at the point with u = 0 on open hubs: a
complementary-slackness dual in which every closed hub fully pays for each path it would close (the naive control's
dual). CCL's Algorithm 2 is a stronger member of the same optimal face: closed hubs first get the lower bound
l_h = (a - c0_h)^+ (loop and edges to open hubs) and an edge between two closed hubs only adds half of its remaining
deficit to each end (`HubCuts.half_split`, used by the engineered reference; on AP it cuts the number of outer
iterations by 3-6x). Both are valid for every binary z (tests/test_scaleup_hub.py checks all subsets).
At a fractional point zhat put the hubs with the largest zhat first; maximizing (*) at zhat over a gives the critical
value of the UFL closed form (cumulate zhat over the hubs sorted by c until it reaches 1). `lp_cut` instead solves the
path LPs on the support exactly (Gurobi) and completes the dual by ownership.

The c values are vectorized: with G the leading group (open hubs, or the top of the zhat order) and the remaining hubs
in a static order,
    c_h (h not in G) = min( F_hh, min_{m in G} F_{hm}, g_h ),   g_h = min over kept edges to hubs earlier in the
                                                                  static order (precomputed once, n^3 numbers),
    c_h (h in G)     = min( F_hh, min_{m in G before h} F_{hm} )       (dense over G x G).
Using all edges (not only the kept ones) in min_{m in G} only lowers c, i.e. stays dual feasible (for |G| <= 8 the
kept-edge filter is applied exactly).

NAIVE BENDERS (the executable control of the scale-up protocol, Section 20.6): outer loop; the integer master
    min f z + theta,  theta >= 0,  sum z >= 1,  z binary,  plus all cuts so far
is re-solved by Gurobi to optimality (MIPGap 0) every iteration; the recourse is evaluated in closed form at the
master's z and ONE aggregate cut (*) summed over all pairs, with a = C_ij(H1) and the identity hub order, is added.
The master bound (ObjBound) is published every iteration; every evaluated hub set is published with its true cost.
No multi-cuts, Pareto selection, stabilization, reduction tests, heuristics or callbacks.
Initializer (shared with the engineered reference): greedy add (best single hub, then the hub with the largest
saving while the saving is positive).
"""
from __future__ import annotations

import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.hub.a_data import HubInstance, load_hub
from src.bd.hub.b_mip_xl import CHUNK, route_costs

INF = float("inf")


# ---------------------------------------------------------------- helpers

def make_env(threads: int, soft_mem_gb: float | None) -> gp.Env:
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", threads)
    if soft_mem_gb:
        env.setParam("SoftMemLimit", soft_mem_gb)
    env.start()
    return env


def certify(bound: float) -> float:
    """A computed lower bound made safe against floating-point error in cuts and master LPs."""
    return bound - 1e-7 * max(1.0, abs(bound)) - 1e-6


def is_memory_error(err: gp.GurobiError) -> bool:
    return err.errno == GRB.Error.OUT_OF_MEMORY or "memory" in str(err).lower()


def objective(inst: HubInstance, z: np.ndarray) -> tuple[float, np.ndarray]:
    """(true cost, per-origin routing cost) of a boolean hub vector."""
    C = route_costs(inst, z)
    per = (inst.w * C).sum(axis=1)
    return float(inst.f[z].sum() + per.sum()), per


# ---------------------------------------------------------------- shared initializer and add moves

def add_terms(inst: HubInstance, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A[i, h] = min_{m in H} (chi d_im + alpha d_mh), B[h, j] = min_{m in H} (alpha d_hm + delta d_mj), all h."""
    d = inst.d
    A = np.full((inst.n, inst.n), INF)
    B = np.full((inst.n, inst.n), INF)
    for m in H:
        A = np.minimum(A, inst.chi * d[:, m][:, None] + inst.alpha * d[m][None, :])
        B = np.minimum(B, inst.alpha * d[:, m][:, None] + inst.delta * d[m][None, :])
    return A, B


def add_savings(inst: HubInstance, H: np.ndarray, C: np.ndarray, hubs: np.ndarray) -> np.ndarray:
    """Routing saving sum_ij W_ij (C_ij - C_ij(H + h)) for every h in hubs (C = routing costs of H)."""
    d, W = inst.d, inst.w
    A, B = add_terms(inst, H)
    out = np.zeros(len(hubs))
    step = max(1, CHUNK // (inst.n * inst.n))
    for s in range(0, len(hubs), step):
        hb = hubs[s:s + step]
        G = np.minimum(inst.chi * d[:, hb][:, None, :] + np.minimum(inst.delta * d[hb].T[None, :, :], B[hb].T[None, :, :]),
                       A[:, hb][:, None, :] + inst.delta * d[hb].T[None, :, :])  # (i, j, h)
        out[s:s + step] = (W[:, :, None] * np.maximum(C[:, :, None] - G, 0.0)).sum(axis=(0, 1))
    return out


def single_hub_costs(inst: HubInstance) -> np.ndarray:
    """Cost of every single-hub solution: f_h + chi sum_i O_i d_ih + delta sum_j D_j d_hj."""
    O, D = inst.w.sum(axis=1), inst.w.sum(axis=0)
    return inst.f + inst.chi * (O @ inst.d) + inst.delta * (inst.d @ D)


def greedy_add(inst: HubInstance, deadline: float, cand: np.ndarray | None = None) -> np.ndarray:
    """The shared basic initializer: best single hub, then repeatedly the candidate with the largest positive net
    saving (routing saving minus fixed cost). Stops early at the deadline (the current set is still feasible)."""
    cand = np.ones(inst.n, dtype=bool) if cand is None else cand
    single = np.where(cand, single_hub_costs(inst), INF)
    z = np.zeros(inst.n, dtype=bool)
    z[int(np.argmin(single))] = True
    while time.perf_counter() < deadline:
        H = np.nonzero(z)[0]
        rest = np.nonzero(cand & ~z)[0]
        if not len(rest):
            break
        C = route_costs(inst, z)
        net = add_savings(inst, H, C, rest) - inst.f[rest]
        k = int(np.argmax(net))
        if net[k] <= 1e-9 * max(1.0, float((inst.w * C).sum())):
            break
        z[rest[k]] = True
    return z


# ---------------------------------------------------------------- cut machinery

class HubCuts:
    """Closed-form dual cuts aggregated per origin: theta_i >= const_i - coef_i . z.

    cand: hubs that may open (others are fixed closed: their paths are ignored and their coefficients are 0);
    rank: static hub order (smaller first), used for the hubs outside the leading group;
    half_size: at integer points the first half_size closed hubs of the order get CCL Algorithm 2 style half-split
    duals (`half_split`), the others ownership duals (0 = pure ownership, the naive control; n = all, the reference)."""

    def __init__(self, inst: HubInstance, cand: np.ndarray | None = None, rank: np.ndarray | None = None,
                 dense_group: int = 8, half_size: int = 0, deadline: float | None = None) -> None:
        self.inst = inst
        self.dense_group = dense_group  # leading groups up to this size get exact kept-edge filtering
        self.half_size = half_size      # closed hubs (first in the static order) with half-split duals at integer points
        n = inst.n
        self.cand = np.ones(n, dtype=bool) if cand is None else np.asarray(cand, dtype=bool).copy()
        self.Hc = np.nonzero(self.cand)[0]
        self.rank = np.arange(n) if rank is None else np.asarray(rank)
        self.pos = np.full(n, -1)
        self.pos[self.Hc] = np.arange(len(self.Hc))
        t = time.perf_counter()
        self.g = self.prefix_edge_min(deadline)
        self.seconds = time.perf_counter() - t
        # Per-origin lower bound: routing with every candidate hub open.
        self.lb_origin = (inst.w * route_costs(inst, self.cand)).sum(axis=1)

    # -------------------------------------------------- static part

    def prefix_edge_min(self, deadline: float | None = None) -> np.ndarray:
        """g[i, j, a] = min over KEPT edges {Hc[a], m} with m a candidate earlier in the static order (inf if none).
        Raises TimeoutError once the deadline passes (n^3 work: ~5 s at n = 200)."""
        inst, Hc = self.inst, self.Hc
        n, k = inst.n, len(Hc)
        d = inst.d
        perm = Hc[np.argsort(self.rank[Hc], kind="stable")]       # candidates in static order
        back = np.argsort(np.argsort(self.rank[Hc], kind="stable"))  # position in perm of Hc[a]
        g = np.empty((n, n, k))
        dP = d[np.ix_(perm, perm)]
        earlier = np.tril(np.ones((k, k), dtype=bool), -1)  # [a, b]: b before a
        for i in range(n):
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("cut machinery setup reached the deadline")
            L = inst.chi * d[i, perm][None, :] + inst.delta * d[perm, :].T          # (j, a) loops
            F = inst.chi * d[i, perm][None, :, None] + inst.alpha * dP[None, :, :] + inst.delta * d[perm, :].T[:, None, :]
            F = np.minimum(F, F.transpose(0, 2, 1))                                    # (j, a, b) undirected
            F = np.where(F < np.minimum(L[:, :, None], L[:, None, :]), F, INF)        # kept edges only (Property 1)
            F[i] = INF                                                                 # i == j: loops only (Property 2)
            F = np.where(earlier[None, :, :], F, INF)
            g[i] = F.min(axis=2)[:, back]
        return g

    # -------------------------------------------------- c values

    def c_values(self, I: np.ndarray, G: np.ndarray, with_c0: bool = False):
        """c[i, j, a] for origins I and candidate hubs Hc[a]; G = leading group (hub indices, in order).
        with_c0: also return c0 = min(loop, edges to G) (the paths a hub outside G would serve with G alone)."""
        inst, Hc = self.inst, self.Hc
        d, chi, al, de = inst.d, inst.chi, inst.alpha, inst.delta
        dI = d[np.ix_(I, Hc)]                                      # (b, a)
        loop = chi * dI[:, None, :] + de * d[Hc, :].T[None, :, :]  # (b, j, a)
        c0 = loop
        if len(G):
            dG = d[np.ix_(I, G)]                                   # (b, g)
            loopG = chi * dG[:, None, :] + de * d[G, :].T[None, :, :]  # (b, j, g)
            if len(G) <= self.dense_group:
                # Edges {h, m}, m in G, kept only if cheaper than both loops (dominated edges would lower c_h).
                F = np.minimum(chi * dI[:, None, :, None] + al * d[np.ix_(Hc, G)][None, None] + de * d[G, :].T[None, :, None, :],
                               chi * dG[:, None, None, :] + al * d[np.ix_(G, Hc)].T[None, None] + de * d[Hc, :].T[None, :, :, None])
                F = np.where(F < np.minimum(loop[:, :, :, None], loopG[:, :, None, :]), F, INF)
                c0 = np.minimum(c0, F.min(axis=3))
            else:
                A, B = add_terms(inst, G)
                c0 = np.minimum(c0, chi * dI[:, None, :] + B[Hc, :].T[None, :, :])
                c0 = np.minimum(c0, A[np.ix_(I, Hc)][:, None, :] + de * d[Hc, :].T[None, :, :])
        c = np.minimum(c0, self.g[I])
        if len(G):
            # Group hubs own only their loop and the kept edges to earlier group hubs.
            F = chi * dG[:, None, :, None] + al * d[np.ix_(G, G)][None, None, :, :] + de * d[G, :].T[None, :, None, :]
            F = np.minimum(F, F.transpose(0, 1, 3, 2))             # (b, j, a, b') undirected
            k = len(G)
            keep = (F < np.minimum(loopG[:, :, :, None], loopG[:, :, None, :])) & np.tril(np.ones((k, k), dtype=bool), -1)[None, None]
            c[:, :, self.pos[G]] = np.minimum(loopG, np.where(keep, F, INF).min(axis=3))
        return (c, c0) if with_c0 else c

    def half_split(self, I: np.ndarray, A: np.ndarray, a: np.ndarray, c0A: np.ndarray,
                   K: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        """u for the closed hubs A (all earlier in the static order than every other closed hub): CCL Algorithm 2 style,
        l_h = (a - c0_h)^+ covers the loop and the edges to open hubs, and every edge {h, m} inside A whose requirement
        a - F_hm exceeds l_h + l_m gets half of the deficit on each end (simultaneously):
            u_h = max(l_h, (l_h + psi_h) / 2),  psi_h = max_{m in A, m != h} (a - l_m - F_hm).
        Edges between A and later closed hubs are owned by the later hub (its c includes them through g)."""
        inst = self.inst
        d, chi, al, de = inst.d, inst.chi, inst.alpha, inst.delta
        l = np.maximum(a[:, :, None] - c0A, 0.0)                   # (b, j, k)
        v = a[:, :, None] - l                                      # v_m = min(a, c0_m)
        # psi_h = max_m (v_m - min(F_hm, F_mh)) = max(max_m (v_m - F_hm), max_m (v_m - F_mh)), F_hm = chi d_ih + al d_hm + de d_mj.
        # The m = h terms equal v_h - F_hh <= l_h and never bind, so they are not excluded.
        K1, aT = K if K is not None else self.half_terms(A)
        dIA = d[np.ix_(I, A)]                                              # (b, m)
        psi = np.empty_like(l)
        for t in range(len(I)):
            p1 = (v[t][:, None, :] - K1).max(axis=2) - chi * dIA[t][None, :]                       # (j, h)
            p2 = ((v[t] - chi * dIA[t][None, :])[:, None, :] - aT[None, :, :]).max(axis=2) - de * d[A, :].T
            psi[t] = np.maximum(p1, p2)
        return np.maximum(l, 0.5 * (l + psi))

    def half_terms(self, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Origin-independent parts of half_split: al d_hm + de d_mj (j, h, m) and al d_mh (h, m)."""
        d, al, de = self.inst.d, self.inst.alpha, self.inst.delta
        return al * d[np.ix_(A, A)][None, :, :] + de * d[A, :].T[:, None, :], al * d[np.ix_(A, A)].T

    def chunks(self, extra: int = 1):
        n = self.inst.n
        step = max(1, CHUNK // max(1, n * len(self.Hc) * extra))
        for s in range(0, n, step):
            yield np.arange(s, min(n, s + step))

    # -------------------------------------------------- cuts

    def binary_cut(self, z: np.ndarray, core: np.ndarray | None = None) -> dict[str, Any]:
        """Per-origin cuts tight at the binary hub vector z (open hubs must be candidates).
        core=None: a = C_ij(H1) (textbook). core given: a in [C, c_(2)] maximizing the ownership cut at the core point
        (Magnanti-Wong selection within this dual family; c_(2) = second smallest c of the open hubs).
        With half_size > 0 the first half_size closed hubs of the static order get CCL-style half-split duals
        (`half_split`) instead of ownership duals."""
        inst = self.inst
        n = inst.n
        H1 = np.nonzero(z)[0]
        if not len(H1) or np.any(~self.cand[H1]):
            raise ValueError("open hubs must be nonempty candidates")
        G = H1[np.argsort(self.rank[H1], kind="stable")]
        pg = self.pos[G]
        closed = self.Hc[~z[self.Hc]]
        A = closed[np.argsort(self.rank[closed], kind="stable")][:self.half_size]
        pa = self.pos[A]
        const = np.zeros(n)
        coef = np.zeros((n, n))
        Q = np.zeros(n)
        Cfull = np.zeros((n, n))
        extra = max(1, len(G), (len(A) ** 2) // max(1, len(self.Hc)) + 1)
        K = self.half_terms(A) if len(A) else None
        for I in self.chunks(extra=extra):
            c, c0 = self.c_values(I, G, with_c0=True)
            cg = c[:, :, pg]
            C = cg.min(axis=2)                                     # (b, j) routing cost = min over open paths
            Cfull[I] = C
            a = C
            if core is not None:
                c2 = np.partition(cg, 1, axis=2)[:, :, 1] if len(G) > 1 else np.full(C.shape, INF)
                a = np.clip(self.critical(c, core[self.Hc], fallback=C), C, c2)
            W = inst.w[I]
            u = np.maximum(a[:, :, None] - c, 0.0)
            if len(A):
                u[:, :, pa] = self.half_split(I, A, a, c0[:, :, pa], K)
            const[I] = (W * a).sum(axis=1)
            coef[np.ix_(I, self.Hc)] = (W[:, :, None] * u).sum(axis=1)
            Q[I] = (W * C).sum(axis=1)
        return {"const": const, "coef": coef, "Q": Q, "C": Cfull}

    def fractional_cut(self, zhat: np.ndarray, max_group: int = 40) -> dict[str, Any]:
        """Per-origin cuts at a fractional point: group = the max_group largest zhat (> 1e-6), a = critical value."""
        inst = self.inst
        n = inst.n
        zc = np.where(self.cand, np.clip(zhat, 0.0, 1.0), 0.0)
        supp = np.nonzero(zc > 1e-6)[0]
        G = supp[np.argsort(-zc[supp], kind="stable")][:max_group]
        const = np.zeros(n)
        coef = np.zeros((n, n))
        for I in self.chunks(extra=max(1, len(G))):
            c = self.c_values(I, G)
            a = self.critical(c, zc[self.Hc], fallback=None)
            W = inst.w[I]
            u = np.maximum(a[:, :, None] - c, 0.0)
            const[I] = (W * a).sum(axis=1)
            coef[np.ix_(I, self.Hc)] = (W[:, :, None] * u).sum(axis=1)
        return {"const": const, "coef": coef, "value": const - coef @ zc}

    def lp_cut(self, zhat: np.ndarray, env: gp.Env, max_support: int = 40, deadline: float | None = None,
               max_vars: int = 300_000) -> dict[str, Any] | None:
        """Exact LP separation on the support: per pair, the path LP restricted to the paths through the support G
        (the max_support largest zhat; paths through hubs with zhat = 0 carry no flow at zhat) is solved by Gurobi
        (pairs batched into block-diagonal LPs); its optimal dual (a, u_G) is completed for the hubs outside G by the
        ownership rule (G first), which keeps the dual feasible for every path. Returns None if the support cannot
        carry the unit flow (sum_G zhat < 1) or the deadline passes."""
        inst = self.inst
        n = inst.n
        d, chi, al, de = inst.d, inst.chi, inst.alpha, inst.delta
        zc = np.where(self.cand, np.clip(zhat, 0.0, 1.0), 0.0)
        supp = np.nonzero(zc > 1e-9)[0]
        G = supp[np.argsort(-zc[supp], kind="stable")][:max_support]
        g = len(G)
        if not g or zc[G].sum() < 1.0 - 1e-9:
            return None
        cap = zc[G]
        const = np.zeros(n)
        coef = np.zeros((n, n))
        dGG = d[np.ix_(G, G)]
        iu, ju = np.triu_indices(g, 1)
        I_all = np.arange(n)
        step = max(1, max_vars // (n * (g + max(1, len(iu)) // 4)))
        for s0 in range(0, n, step):
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            I = I_all[s0:s0 + step]
            b = len(I)
            P = b * n
            dG = d[np.ix_(I, G)]
            loopG = (chi * dG[:, None, :] + de * d[G, :].T[None, :, :]).reshape(P, g)
            F = chi * dG[:, None, :, None] + al * dGG[None, None] + de * d[G, :].T[None, :, None, :]
            Fe = np.minimum(F, F.transpose(0, 1, 3, 2)).reshape(P, g, g)[:, iu, ju]          # (P, E) undirected
            kept = Fe < np.minimum(loopG[:, iu], loopG[:, ju])
            kept[(I[:, None] == I_all[None, :]).reshape(-1)] = False                         # i == j: loops only
            pk, ek = np.nonzero(kept)                                                         # sorted by pair
            nv = len(pk)
            # Ragged sums as dummy-padded slot matrices (one dummy variable per slot, fixed at 0).
            cnt = np.bincount(pk, minlength=P)
            emax = int(cnt.max()) if nv else 0
            r = np.concatenate([pk * g + iu[ek], pk * g + ju[ek]])
            v = np.concatenate([np.arange(nv), np.arange(nv)])
            o = np.argsort(r, kind="stable")
            r, v = r[o], v[o]
            cnt2 = np.bincount(r, minlength=P * g)
            dmax = int(cnt2.max()) if nv else 0
            m = gp.Model(env=env)
            m.Params.OutputFlag = 0
            if deadline is not None:
                m.Params.TimeLimit = max(0.01, deadline - time.perf_counter())
            xl = m.addMVar((P, g), lb=0.0)
            ub = np.concatenate([np.full(nv, GRB.INFINITY), np.zeros(emax + dmax)])
            xe = m.addMVar(nv + emax + dmax, lb=0.0, ub=ub)
            m.setObjective((loopG * xl).sum() + Fe[pk, ek] @ xe[:nv], GRB.MINIMIZE)
            lhs_a = xl.sum(axis=1)
            lhs_l = xl.reshape(-1)
            if nv:
                A_idx = np.tile(nv + np.arange(emax), (P, 1))
                A_idx[pk, np.arange(nv) - np.repeat(np.cumsum(cnt) - cnt, cnt)] = np.arange(nv)
                lhs_a = lhs_a + xe[A_idx].sum(axis=1)
                L_idx = np.tile(nv + emax + np.arange(dmax), (P * g, 1))
                L_idx[r, np.arange(len(r)) - np.repeat(np.cumsum(cnt2) - cnt2, cnt2)] = v
                lhs_l = lhs_l + xe[L_idx].sum(axis=1)
            assign = m.addConstr(lhs_a == 1.0)
            link = m.addConstr(lhs_l <= np.tile(cap, P))
            m.optimize()
            if m.Status != GRB.OPTIMAL:
                m.dispose()
                return None
            a = np.asarray(assign.Pi).reshape(b, n)
            uG = np.maximum(-np.asarray(link.Pi), 0.0).reshape(b, n, g)
            m.dispose()
            c = self.c_values(I, G)
            u = np.maximum(a[:, :, None] - c, 0.0)
            u[:, :, self.pos[G]] = uG
            W = inst.w[I]
            const[I] = (W * a).sum(axis=1)
            coef[np.ix_(I, self.Hc)] = (W[:, :, None] * u).sum(axis=1)
        return {"const": const, "coef": coef, "value": const - coef @ zc}

    @staticmethod
    def critical(c: np.ndarray, weight: np.ndarray, fallback: np.ndarray | None) -> np.ndarray:
        """argmax_a  a - sum_h weight_h (a - c_h)^+  per pair: the c at which the cumulative weight (hubs sorted by c)
        reaches 1; if the weights never reach 1 the function increases forever: return fallback (or the largest c with
        positive weight)."""
        sup = np.nonzero(weight > 1e-12)[0]
        if not len(sup):
            return fallback if fallback is not None else c.min(axis=2)
        cs = c[:, :, sup]
        order = np.argsort(cs, axis=2, kind="stable")
        cum = np.cumsum(weight[sup][order], axis=2)
        hit = cum >= 1.0 - 1e-9
        idx = np.argmax(hit, axis=2)
        val = np.take_along_axis(np.take_along_axis(cs, order, axis=2), idx[:, :, None], axis=2)[:, :, 0]
        ok = hit.any(axis=2)
        if ok.all():
            return val
        alt = fallback if fallback is not None else cs.max(axis=2)
        return np.where(ok, val, alt)


# ---------------------------------------------------------------- naive Benders

class HubNaiveBenders:
    """Naive textbook Benders control; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0, verbose: bool = False) -> None:
        self.time_limit = time_limit
        self.threads = threads
        self.on_incumbent = on_incumbent or (lambda hubs: None)  # receives the sorted list of open hub indices
        self.on_bound = on_bound or (lambda bound: None)  # receives every new valid global lower bound
        self.soft_mem_gb = soft_mem_gb
        self.seed = seed
        self.verbose = verbose
        self.trace: list[dict[str, Any]] = []

    def left(self) -> float:
        return self.deadline - time.perf_counter()

    def log(self, stage: str) -> None:
        self.trace.append({"t": round(time.perf_counter() - self.t0, 3), "incumbent": self.ub if self.ub < INF else None,
                           "bound": self.lb if self.lb > -INF else None, "stage": stage})
        if self.verbose:
            print(self.trace[-1], flush=True)

    def publish_bound(self, value: float) -> None:
        value = certify(value)
        if value > self.lb:
            self.lb = value
            self.on_bound(value)
            self.log("bound")

    def publish(self, z: np.ndarray, cost: float) -> None:
        if cost < self.ub - 1e-9 * max(1.0, abs(cost)):
            self.ub, self.best = cost, z.copy()
            self.on_incumbent(np.nonzero(z)[0].tolist())
            self.log("incumbent")

    def closed(self) -> bool:
        return self.ub < INF and self.lb >= self.ub - 1e-6 * max(1.0, abs(self.ub))

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        inst = load_hub(data_path)
        self.ub, self.lb, self.best = INF, -INF, None
        self.stats = {"iterations": 0, "cuts": 0, "master_seconds": 0.0, "sub_seconds": 0.0, "init_seconds": 0.0,
                      "setup_seconds": 0.0}
        reason = "time_limit"
        env = make_env(self.threads, self.soft_mem_gb)
        try:
            reason = self.solve(inst, env)
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        env.dispose()
        ub = self.ub if self.ub < INF else None
        lb = self.lb if self.lb > -INF else None
        return {"best_objective": ub, "global_lb": lb, "global_ub": ub,
                "gap": (ub - lb) / abs(ub) if ub is not None and lb is not None and ub else None,
                "termination_reason": reason, "hubs": np.nonzero(self.best)[0].tolist() if self.best is not None else None,
                "stats": self.stats, **self.stats, "total_seconds": time.perf_counter() - self.t0, "trace": self.trace}

    def solve(self, inst: HubInstance, env: gp.Env) -> str:
        self.publish_bound(0.0)  # costs are nonnegative
        t = time.perf_counter()
        z_bar = greedy_add(inst, self.deadline)
        self.publish(z_bar, objective(inst, z_bar)[0])
        self.stats["init_seconds"] = time.perf_counter() - t
        if self.left() <= 0:
            return "time_limit"
        t = time.perf_counter()
        try:
            cuts = HubCuts(inst, deadline=self.deadline)
        except TimeoutError:
            return "time_limit"
        self.stats["setup_seconds"] = time.perf_counter() - t
        master = gp.Model(env=env)
        master.Params.OutputFlag = 0
        master.Params.MIPGap = 0.0
        master.Params.Seed = self.seed
        z = master.addMVar(inst.n, vtype=GRB.BINARY, name="z")
        theta = master.addVar(lb=0.0, name="theta")
        master.addConstr(z.sum() >= 1)
        master.setObjective(inst.f @ z + theta, GRB.MINIMIZE)
        theta_bar = -INF
        while True:
            if self.left() <= 0:
                return "time_limit"
            t = time.perf_counter()
            cut = cuts.binary_cut(z_bar)
            self.stats["sub_seconds"] += time.perf_counter() - t
            Q = float(cut["Q"].sum())
            self.publish(z_bar, float(inst.f[z_bar].sum()) + Q)
            const, h = float(cut["const"].sum()), cut["coef"].sum(axis=0)
            if const - h @ z_bar - theta_bar <= 1e-6 * max(1.0, abs(Q)):
                return "optimal"  # theta_bar >= Q(z_bar): the master optimum is attained by z_bar
            master.addConstr(theta + h @ z >= const)
            self.stats["cuts"] += 1
            if self.closed():
                return "optimal"
            if self.left() <= 0:
                return "time_limit"
            t = time.perf_counter()
            master.Params.TimeLimit = max(0.01, self.left())
            master.optimize()
            self.stats["master_seconds"] += time.perf_counter() - t
            self.stats["iterations"] += 1
            if master.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.MEM_LIMIT):
                try:
                    self.publish_bound(float(master.ObjBound))
                except gp.GurobiError:
                    pass  # no bound available (stopped before the root relaxation)
            if master.Status != GRB.OPTIMAL:
                return "memory_limit" if master.Status == GRB.MEM_LIMIT else "time_limit"
            if self.closed():
                return "optimal"
            z_bar = np.asarray(z.X) > 0.5
            theta_bar = float(theta.X)
