"""Duty pricing for the RCSP: a resource-constrained shortest path on the acyclic task-connection graph.

Labels follow van Rossum (2026, Appendix A): (reduced cost c, duty length w, time without break beta, break flag b).
Extension along a connection a = (i, j):
    c' = c + cost_a - pi_j,   w' = w + dur_a,
    (beta', b') = (tab_a, 1) if a is a break arc and w + tbb_a <= MAX_WORK (the break is taken whenever it may start
                  within 5.5 h of the duty start; a later break only lowers the time without break)
                  (beta + dur_a, b) otherwise,
feasible while w' <= MAX_DUTY and beta' <= MAX_WORK; a duty ends on a SINK at its own crew base with b = 1.
`beta` is stored as max(beta, w - (MAX_DUTY - MAX_WORK)): below that value the 5.5 h limit is implied by the 9.5 h
limit, so the normalization only merges equivalent labels. Dominance at a task (same crew base): c, w, beta smaller or
equal and b larger or equal. The graph is acyclic, so every path is an elementary duty.

Pruning (all valid for exact pricing): a label is dropped when c + LBC[base, task] >= threshold (LBC = the cheapest
reduced-cost completion to a SINK at the base, recomputed per call; on time-consistent instances it is time-bucketed:
the completion must reach the SINK by the first 15-minute grid point at or after the label's start + 9.5 h), when w + MINW[base, task] >
MAX_DUTY (MINW = the shortest completion in duty time), and, without a break yet, when w + MINB[task] > MAX_WORK
(MINB = the shortest time to the start of any break). With threshold = -tol every duty of reduced cost below -tol is
found or dominated by one that is, so the minimum reduced cost per base is exact (complete=True) unless a deadline or
the label cap interrupts the call. Heuristic mode keeps at most `beam` labels per (task, base) (cheapest first).

All crew bases are priced in one pass: labels carry their base and dominate only labels of the same base.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.cg.rcsp.a_data import RCSPInstance

INF = float("inf")


class DutyPricer:
    def __init__(self, inst: RCSPInstance) -> None:
        self.inst = inst
        n = inst.n
        self.n = n
        self.L, self.LB = float(inst.max_duty), float(inst.max_work)
        self.depots = list(inst.depots)
        D = len(self.depots)
        self.D = D
        dpos = {d: k for k, d in enumerate(self.depots)}
        self.dpos = dpos
        # Arcs grouped by head task (pull-based labeling), tails in topological order within a group.
        order = np.lexsort((inst.pos[inst.arc_tail], inst.arc_head))
        self.a_conn = inst.arc_conn[order]
        self.a_tail = inst.arc_tail[order].astype(np.int64)
        self.a_head = inst.arc_head[order].astype(np.int64)
        self.a_dur = inst.arc_dur[order].astype(float)
        self.a_cost = inst.arc_cost[order].astype(float)
        self.a_tbb = np.where(inst.arc_break[order], inst.arc_tbb[order], 0).astype(float)
        self.a_tab = np.where(inst.arc_break[order], inst.arc_tab[order], 0).astype(float)
        self.a_brk = inst.arc_break[order].copy()
        if np.any(self.a_brk & (self.a_tbb + self.a_tab > self.a_dur)):
            raise ValueError("break arc with timeBeforeBreak + timeAfterBreak > duration")
        self.in_ptr = np.searchsorted(self.a_head, np.arange(n + 1), side="left") if len(order) else np.zeros(n + 1, dtype=np.int64)
        # Arcs sorted by tail (backward DPs).
        out = np.argsort(self.a_tail, kind="stable")
        self.out_idx = out
        self.out_ptr = np.searchsorted(self.a_tail[out], np.arange(n + 1), side="left") if len(out) else np.zeros(n + 1, dtype=np.int64)
        # Crew-base matrices (inf where the connection is missing).
        self.src_cost = np.full((D, n), INF)
        self.src_dur = np.full((D, n), INF)
        for c, d, j in zip(inst.src_conn, inst.src_depot, inst.src_task):
            self.src_cost[dpos[int(d)], j] = float(inst.c_cost[c])
            self.src_dur[dpos[int(d)], j] = float(inst.c_dur[c])
        self.snk_cost = np.full((D, n), INF)
        self.snk_dur = np.full((D, n), INF)
        for c, i, d in zip(inst.snk_conn, inst.snk_task, inst.snk_depot):
            self.snk_cost[dpos[int(d)], i] = float(inst.c_cost[c])
            self.snk_dur[dpos[int(d)], i] = float(inst.c_dur[c])
        self.topo = inst.order
        self.MINW = self.backward(self.a_dur, self.snk_dur)
        # MINB[k]: shortest time from task k's end to the start of a break (on an out-arc of k or later).
        mb = np.full(n, INF)
        for k in self.topo[::-1]:
            idx = self.out_idx[self.out_ptr[k]:self.out_ptr[k + 1]]
            if len(idx):
                h = self.a_head[idx]
                v = np.where(self.a_brk[idx], self.a_tbb[idx], INF)
                mb[k] = min(v.min(), (self.a_dur[idx] + mb[h]).min())
        self.MINB = mb
        # Forward reachability per base (tasks a duty from the base can reach), intersected with "can return".
        reach = np.isfinite(self.src_cost).copy()
        for k in self.topo:
            a = slice(self.in_ptr[k], self.in_ptr[k + 1])
            if self.in_ptr[k + 1] > self.in_ptr[k]:
                reach[:, k] |= reach[:, self.a_tail[a]].any(axis=1)
        self.relevant = reach & np.isfinite(self.MINW)
        # Time-bucketed completion bound (time-consistent instances: a duty starting at S must end by S + MAX_DUTY):
        # grid points g_b every `grid_step` seconds; BT[base, k, b] = cheapest reduced-cost completion from task k whose
        # SINK is reached by g_b. A label with start S uses the first grid point >= S + MAX_DUTY (a relaxation).
        self.tc = bool(inst.time_consistent)
        if self.tc:
            step = self.grid_step = 900.0
            lo, hi = float(inst.end.min()), float(inst.end.max()) + float(np.nanmax(np.where(np.isfinite(self.snk_dur), self.snk_dur, 0.0)))
            self.grid = np.arange(lo, hi + step, step)
            self.task_end = inst.end.astype(float)
            ends = self.task_end[None, :, None] + self.snk_dur[:, :, None]
            self.sink_ok = ends <= self.grid[None, None, :]  # (D, n, G)
        self.stats = {"calls": 0, "labels": 0, "seconds": 0.0}

    # ------------------------------------------------------------------ helpers

    def backward(self, arc_val: np.ndarray, sink_val: np.ndarray) -> np.ndarray:
        """V[base, k] = min over completions from task k (after arriving there) of the summed arc values + sink value."""
        V = sink_val.copy()
        for k in self.topo[::-1]:
            p, q = self.out_ptr[k], self.out_ptr[k + 1]
            if q > p:
                idx = self.out_idx[p:q]
                cand = (arc_val[idx][None, :] + V[:, self.a_head[idx]]).min(axis=1)
                np.minimum(V[:, k], cand, out=V[:, k])
        return V

    def backward_timed(self, rc_arc: np.ndarray, snk_rc: np.ndarray) -> np.ndarray:
        V = np.where(self.sink_ok, snk_rc[:, :, None], INF)
        for k in self.topo[::-1]:
            p, q = self.out_ptr[k], self.out_ptr[k + 1]
            if q > p:
                idx = self.out_idx[p:q]
                cand = (rc_arc[idx][None, :, None] + V[:, self.a_head[idx], :]).min(axis=1)
                np.minimum(V[:, k, :], cand, out=V[:, k, :])
        return V

    def duty_tasks(self, lab: int) -> list[int]:
        path = []
        while lab >= 0:
            path.append(int(self.G_node[lab]))
            lab = int(self.G_pred[lab])
        return path[::-1]

    def _grow(self, need: int) -> None:
        cap = len(self.G_c)
        if need <= cap:
            return
        new = max(need, 2 * cap)
        for name in ("G_c", "G_w", "G_beta", "G_b", "G_dep", "G_pred", "G_node"):
            old = getattr(self, name)
            arr = np.empty(new, dtype=old.dtype)
            arr[:len(old)] = old
            setattr(self, name, arr)

    @staticmethod
    def _dominance(c: np.ndarray, w: np.ndarray, beta: np.ndarray, b: np.ndarray, dep: np.ndarray) -> np.ndarray:
        """Indices of the nondominated labels (one representative of identical labels). First, within each group of
        equal (base, b, w), the 2-D front over (beta, c) by a segmented running minimum; then a pairwise check over the survivors
        of the same base (a label can only be dominated by one earlier in (base, c, w, beta, -b) order)."""
        m = len(c)
        if m <= 1:
            return np.arange(m)
        o = np.lexsort((c, beta, w, -b, dep))
        cs = c[o]
        new = np.ones(m, dtype=bool)
        new[1:] = (dep[o][1:] != dep[o][:-1]) | (b[o][1:] != b[o][:-1]) | (w[o][1:] != w[o][:-1])
        g = np.cumsum(new) - 1
        K = 2.0 * (float(cs.max() - cs.min()) + 1.0)
        v = cs - g * K
        run = np.minimum.accumulate(v)
        keep1 = new.copy()
        keep1[1:] |= v[1:] < run[:-1] - 1e-6
        o = o[keep1]
        m = len(o)
        if m <= 1:
            return o
        o = o[np.lexsort((-b[o], beta[o], w[o], c[o], dep[o]))]
        c, w, beta, b, dep = c[o], w[o], beta[o], b[o], dep[o]
        # Candidate dominators of row j: the earlier rows of the same base (pairs built without a Python loop).
        gs = np.searchsorted(dep, dep, side="left")
        lens = np.arange(m) - gs
        dominated = np.zeros(m, dtype=bool)
        cap = 4_000_000
        s0 = 0
        while s0 < m:
            # rows [s0, e0) whose pair count stays under the cap (at least one row)
            cum = np.cumsum(lens[s0:])
            e0 = s0 + max(1, int(np.searchsorted(cum, cap, side="right")))
            ln = lens[s0:e0]
            tot = int(ln.sum())
            if tot:
                J = np.repeat(np.arange(s0, e0), ln)
                I = np.arange(tot) - np.repeat(np.cumsum(ln) - ln, ln) + np.repeat(gs[s0:e0], ln)
                pair = (c[I] <= c[J] + 1e-9) & (w[I] <= w[J]) & (beta[I] <= beta[J]) & (b[I] >= b[J])
                dominated[s0:e0] = np.bincount(J[pair] - s0, minlength=e0 - s0) > 0
            s0 = e0
        return o[~dominated]

    # ------------------------------------------------------------------ pricing

    def price(self, pi: np.ndarray, *, exact: bool = True, beam: int = 0, max_cols: int = 50, per_depot: int | None = None,
              deadline: float = INF, tol: float = 1e-6, max_labels: int = 3_000_000, bases: list[int] | None = None,
              threshold: float | None = None, rel: float = 0.0, enumerate_all: bool = False) -> dict[str, Any]:
        """Price duties for duals pi (>= 0, one per task). Returns
            {"cols": [(rc, base_station, [tasks])] sorted by rc (rc < -tol), "min_rc": array per base (exact when
             complete; 0 if no duty below -tol), "complete": bool, "labels": int}.
        exact=False keeps only the `beam` cheapest labels per (task, base). per_depot caps the columns per base.
        rel > 0: once a duty of reduced cost r < 0 is found, labels that cannot reach rel * r are pruned too (the
        overall minimum stays exact, so min(min_rc) and the Lagrangian bound do; per-base minima become upper bounds
        and only duties below the running threshold are returned).
        enumerate_all=True: no dominance, no beam; every duty of reduced cost below `threshold` is returned (complete
        unless the deadline or max_labels interrupts); used for reduced-cost duty enumeration."""
        t0 = time.perf_counter()
        self.stats["calls"] += 1
        n, D, L, LB = self.n, self.D, self.L, self.LB
        LmLB = L - LB
        thr = -tol if threshold is None else threshold
        fin_thr = thr if enumerate_all else -tol
        if enumerate_all:
            exact, rel = True, 0.0
        pi = np.asarray(pi, dtype=float)
        rc_arc = self.a_cost - pi[self.a_head]
        src_rc = self.src_cost - pi[None, :]
        active = np.ones(D, dtype=bool)
        if bases is not None:
            active[:] = False
            for d in bases:
                active[self.dpos[d]] = True
        snk_rc = np.where(active[:, None], self.snk_cost, INF)
        if self.tc:
            BT = self.backward_timed(rc_arc, snk_rc)
            LBC = BT[:, :, -1]
            G = len(self.grid)
        else:
            LBC = self.backward(rc_arc, snk_rc)
        # A crew base can only produce a negative duty from a source whose best completion is negative.
        start_ok = active[:, None] & np.isfinite(src_rc) & (src_rc + LBC < thr)
        cap = max(1 << 16, 8 * n)
        self.G_c = np.empty(cap)
        self.G_w = np.empty(cap)
        self.G_beta = np.empty(cap)
        self.G_b = np.empty(cap, dtype=np.int8)
        self.G_dep = np.empty(cap, dtype=np.int32)
        self.G_pred = np.empty(cap, dtype=np.int64)
        self.G_node = np.empty(cap, dtype=np.int32)
        off = np.zeros(n, dtype=np.int64)
        cnt = np.zeros(n, dtype=np.int64)
        top = 0
        cand_rc, cand_lab = [], []
        complete = True
        a_tail, a_dur, a_tbb, a_tab, a_brk = self.a_tail, self.a_dur, self.a_tbb, self.a_tab, self.a_brk
        for step, k in enumerate(self.topo):
            if step % 64 == 0 and time.perf_counter() > deadline:
                complete = False
                break
            p, q = self.in_ptr[k], self.in_ptr[k + 1]
            parts = None
            if q > p:
                ia = np.arange(p, q)
                lens = cnt[a_tail[ia]]
                sel = lens > 0
                if sel.any():
                    ia, lens = ia[sel], lens[sel]
                    starts = off[a_tail[ia]]
                    tot = int(lens.sum())
                    idx = np.arange(tot) + np.repeat(starts - (np.cumsum(lens) - lens), lens)
                    arc = np.repeat(ia, lens)
                    w0 = self.G_w[idx]
                    dur = a_dur[arc]
                    c = self.G_c[idx] + rc_arc[arc]
                    w = w0 + dur
                    reset = a_brk[arc] & (w0 + a_tbb[arc] <= LB)
                    beta = np.where(reset, a_tab[arc], self.G_beta[idx] + dur)
                    b = self.G_b[idx] | reset.astype(np.int8)
                    dep = self.G_dep[idx]
                    pred = idx
                    parts = [c, w, beta, b, dep, pred]
            srcs = np.nonzero(start_ok[:, k])[0]
            if len(srcs):
                sw = self.src_dur[srcs, k]
                s_parts = [src_rc[srcs, k], sw, sw.copy(), np.zeros(len(srcs), dtype=np.int8), srcs.astype(np.int32),
                           np.full(len(srcs), -1, dtype=np.int64)]
                parts = s_parts if parts is None else [np.concatenate([x, y]) for x, y in zip(parts, s_parts)]
            if parts is None:
                continue
            c, w, beta, b, dep, pred = parts
            beta = np.maximum(beta, w - LmLB)
            ok = (w <= L) & (beta <= LB) & (w + self.MINW[dep, k] <= L)
            ok &= (b == 1) | (w + self.MINB[k] <= LB)
            if self.tc:
                gi = np.minimum(np.ceil((self.task_end[k] - w + L - self.grid[0]) / self.grid_step).astype(np.int64), G - 1)
                gi = np.maximum(gi, 0)
                ok &= c + BT[dep, k, gi] < thr
            else:
                ok &= c + LBC[dep, k] < thr
            if not ok.all():
                c, w, beta, b, dep, pred = c[ok], w[ok], beta[ok], b[ok], dep[ok], pred[ok]
            m = len(c)
            if m == 0:
                continue
            if m > 1 and not enumerate_all:
                keep = self._dominance(c, w, beta, b, dep)
                if not exact and beam:
                    # Beam: the cheapest `beam` labels per base at this task.
                    kk = keep[np.lexsort((c[keep], dep[keep]))]
                    dd = dep[kk]
                    first = np.searchsorted(dd, dd, side="left")
                    kk = kk[np.arange(len(kk)) - first < beam]
                    keep = kk
                c, w, beta, b, dep, pred = c[keep], w[keep], beta[keep], b[keep], dep[keep], pred[keep]
                m = len(c)
            self._grow(top + m)
            sl = slice(top, top + m)
            self.G_c[sl], self.G_w[sl], self.G_beta[sl], self.G_b[sl] = c, w, beta, b
            self.G_dep[sl], self.G_pred[sl], self.G_node[sl] = dep, pred, k
            off[k], cnt[k] = top, m
            # Completions on a SINK at the label's own base.
            sd = self.snk_dur[dep, k]
            fin = (b == 1) & np.isfinite(sd)
            if fin.any():
                fin &= (w + sd <= L) & (beta + sd <= LB)
                rc = c + self.snk_cost[dep, k]
                fin &= rc < fin_thr
                if fin.any():
                    cand_rc.append(rc[fin])
                    cand_lab.append(np.arange(top, top + m)[fin])
                    if rel > 0.0:
                        thr = min(thr, rel * float(rc[fin].min()))
            top += m
            if top > max_labels:
                complete = False
                break
        min_rc = np.zeros(D)
        cols: list[tuple[float, int, list[int]]] = []
        if cand_rc:
            rc = np.concatenate(cand_rc)
            lab = np.concatenate(cand_lab)
            dep = self.G_dep[lab]
            np.minimum.at(min_rc, dep, rc)
            o = np.argsort(rc, kind="stable")
            per = np.zeros(D, dtype=np.int64)
            for t in o:
                d = int(dep[t])
                if per_depot is not None and per[d] >= per_depot:
                    continue
                per[d] += 1
                cols.append((float(rc[t]), self.depots[d], self.duty_tasks(int(lab[t]))))
                if len(cols) >= max_cols:
                    break
        self.stats["labels"] += top
        self.stats["seconds"] += time.perf_counter() - t0
        return {"cols": cols, "min_rc": min_rc, "complete": bool(complete and exact), "labels": top,
                "seconds": time.perf_counter() - t0}


def greedy_cover(pricer: DutyPricer, deadline: float = INF, beam: int = 8) -> list[dict[str, Any]] | None:
    """Shared basic initializer (naive and engineered CG): prize-collecting greedy set cover. Every still uncovered
    task gets a prize larger than any duty's cost, so a duty's reduced cost is its cost minus that prize times the
    number of uncovered tasks it covers; each round prices with a beam and accepts, cheapest reduced cost first, the
    duties whose prized tasks are all still uncovered. If the beam finds nothing, an exact round decides (None when
    some task cannot be covered by any feasible duty, or at the deadline)."""
    n = pricer.n
    prize = 2.0 * (float(np.nanmax(np.where(np.isfinite(pricer.src_cost), pricer.src_cost, 0.0))) + pricer.L + 1.0)
    covered = np.zeros(n, dtype=bool)
    duties: list[dict[str, Any]] = []
    while not covered.all():
        if time.perf_counter() > deadline:
            return None
        pi = np.where(covered, 0.0, prize)
        res = pricer.price(pi, exact=False, beam=beam, max_cols=400, deadline=deadline)
        if not res["cols"]:
            res = pricer.price(pi, exact=True, max_cols=50, deadline=deadline)
            if not res["cols"]:
                return None
        start = covered.copy()
        for _, d, tasks in res["cols"]:
            prized = [t for t in tasks if not start[t]]
            if prized and not covered[prized].any():
                covered[tasks] = True
                duties.append({"depot": int(d), "tasks": [int(t) for t in tasks]})
    return duties
