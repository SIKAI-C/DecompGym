"""Crew-indexed multi-commodity flow (MCF) MIP for railway crew scheduling: the exact compact baseline of the
scale-up study, as in van Rossum (2026), Appendix B, eqs. (9)-(14) (source text in
outputs/scaleup/references/vanrossum2026_mcf_appendixB_and_table6.txt).

Variables. Crew members C = {1..|C|}; y[c, a] binary for every crew member c and every connection a of the variant's
connection file (SOURCE, CONNECTION and SINK); z[c, a] binary for every break connection a in A_B (the CONNECTIONs
with timeBeforeBreak >= 0).

Times. The paper's arc times tau_lo(a) <= tau_hi(a) and break times beta_lo(a) (break start), beta_hi(a) (break end)
on our data (a_data's conventions; the instance must be time-consistent):
    SOURCE (d, s):      tau_lo = start(s)            tau_hi = end(s)
    CONNECTION (i, j):  tau_lo = end(i)              tau_hi = end(j)
    SINK (i, d):        tau_lo = end(i)              tau_hi = end(i) + duration (0 in the benchmark)
    break a = (i, j):   beta_lo = end(i) + timeBeforeBreak(a)        beta_hi = end(j) - timeAfterBreak(a)
so tau_hi - tau_lo is the connection's duration, the duty length is tau_hi(SINK) - tau_lo(SOURCE) =
end(last) - start(first), and for a break arc a of the duty before(a) = beta_lo(a) - tau_lo(SOURCE) =
(durations before a) + timeBeforeBreak(a) and after(a) = tau_hi(SINK) - beta_hi(a) = timeAfterBreak(a) + (durations
after a), the two quantities of b_mip_xl.duty_check. All times are taken relative to T0 = the earliest task start:
(9h) and (9i) then change by T0 times rows implied by (9c) and (9f), an equivalent rewrite with smaller coefficients.

Model (rows for every crew member c unless stated; L = max_duty = 34 200 s, L_B = max_work = 19 800 s):
 (9a) min  sum_c sum_a cost(a) y[c, a]
 (9b) sum_c sum_{a in A-(k)} y[c, a] >= 1                    every task k (A-(k): SOURCE and CONNECTION arcs into k)
 (9c) sum_{a in A-(k)} y[c, a] = sum_{a in A+(k)} y[c, a]     every task k (A+(k): CONNECTION and SINK arcs out of k)
 (9d) sum_{a in SRC} y[c, a] <= 1
 (9e) sum_a (tau_hi - tau_lo)(a) y[c, a] <= L sum_{SRC} y[c, a]
 (9f) sum_{a in A_B} z[c, a] = sum_{SRC} y[c, a]
 (9g) z[c, a] <= y[c, a]                                       every a in A_B
 (9h) sum_{A_B} beta_lo z[c, a] - sum_{SRC} tau_lo y[c, a] <= L_B sum_{SRC} y[c, a]
 (9i) sum_{SNK} tau_hi y[c, a] - sum_{A_B} beta_hi z[c, a] <= L_B sum_{SRC} y[c, a]
 multi-depot only: sum_{SOURCE at d} y[c, a] = sum_{SINK at d} y[c, a] for every crew base d (the paper presents the
 single-depot case, where this is implied by (9c); a duty returns to its own base).
Exactness: the task graph is acyclic (a_data), so (9c)-(9d) make y[c] one SOURCE -> CONNECTION* -> SINK path or
nothing; (9f)-(9g) put exactly one break on an active path; (9e), (9h), (9i) then say: length <= L and the chosen
break has before <= L_B and after <= L_B. A path is therefore a duty of the model iff some break arc of it satisfies
both limits and its length is at most L, which is duty_check's rule (tests: every path of the synthetic and the
boundary instances, forced on one crew member, is accepted iff duty_check accepts it).

Symmetry breaking (symmetry_breaking=True):
 (10) sum_{SRC} y[c, a] >= sum_{SRC} y[c+1, a]                 c = 1..|C|-1
 (11) sum_{a in A-(q_c)} y[c, a] = 1                           c = 1..|Q|
 Q = the largest Q_tau over the distinct task start times tau (first one on ties), Q_tau = active(tau) union
 active(tau + L + 1), active(t) = tasks with start <= t < end; ordered by (start, end, index). Two tasks of Q_tau
 overlap in time (CONNECTIONs go forward in time: start(j) >= end(i), checked) or are more than L apart, so no duty
 contains two of them. Arc fixing: for c <= |Q|, y[c, a] = 0 (and z[c, a] = 0) for every arc whose end or start time
 violates the duty length combined with q_c: tau_hi(a) - start(q_c) > L or end(q_c) - tau_lo(a) > L. Fixed variables
 are not created. If |Q| > |C| the instance has no plan with |C| duties; (11) is then added for c <= |C| only.

Valid inequalities (valid_inequalities=True), for every crew member and every distinct task start time tau
(K = number of distinct start times, as in the paper):
 (12) sum_{SRC: tau_lo <= tau} y + sum_{SNK: tau_hi > tau + L} y <= sum_{SRC} y
 (13) sum_{SRC: tau_lo <= tau} y + sum_{A_B: beta_lo > tau + L_B} z <= sum_{SRC} y
 (14) sum_{A_B: beta_hi <= tau} z + sum_{SNK: tau_hi > tau + L_B} y <= sum_{SRC} y
vi_form="chain" (default) writes them with cumulative chain variables per crew member and threshold, defined by
equalities, e.g. G[c, k] = sum_{SRC: tau_lo > tau_k} y[c, a] = G[c, k+1] + sum_{SRC: tau_k < tau_lo <= tau_k+1} y[c, a]
(G[c, 0] = all sources), H[c, k] = sum_{SNK: tau_hi > tau_k + L} y, J[c, k] = sum_{A_B: beta_lo > tau_k + L_B} z,
M[c, k] = sum_{SNK: tau_hi > tau_k + L_B} y, I[c, k] = sum_{A_B: beta_hi <= tau_k} z; then (12) is H <= G[., k],
(13) is J <= G[., k] and (14) is I + M <= G[., 0], and every y or z appears in at most two chain rows. The chain
variables are determined by (y, z), so the LP relaxation projected on (y, z) is the paper's (tests compare the LP
value with vi_form="explicit", the rows as printed); the explicit rows have up to |A_B| nonzeros each, about 10^8
in total on the 500-700-task instances.

|C| = the number of duties of the benchmark's best-known solution + 5 (the paper's choice, favourable to the MIP;
BKS_DUTIES is the "Duties" column of the benchmark's solutions/README.md, equal to the line counts of all 200 solution
files, see the tests), or n_crews.

Solver. Gurobi with default settings except Threads, Seed, SoftMemLimit and MIPGap = 0, MIPGapAbs = 0.5 (costs are
integral, so OPTIMAL is a proof); `method` sets the root relaxation algorithm (default -1: Gurobi's choice). The
build counts against the time limit, and the callback stops the solve at the deadline. Every integer solution
(MIPSOL) is decoded into one duty per active crew member and checked with plan_issues; it is published
(on_incumbent) only if feasible, and must then cost what Gurobi reports. A decoding failure, an infeasible duty or a
cost mismatch would be a bug of this exact model and is recorded in stats["decode_issues"] (tests require none).
Bounds: Gurobi's best bound in the MIP callback, rounded up (integral costs) and published (on_bound) when it
improves.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from src.cg.rcsp.a_data import CONNECTION, SINK, SOURCE, RCSPInstance, load_rcsp
from src.cg.rcsp.b_mip_xl import plan_issues


EXTRA_CREWS = 5  # |C| = duties of the best-known solution + 5 (van Rossum 2026, Section 5.6)
# Duties of the benchmark's best-known solutions, "name:single/multi" (solutions/README.md).
_BKS_TABLE = """
s2_t90:14/14 s2_t118:22/21 s2_t120:19/18 s2_t122:14/13 s2_t131:21/20 s3_t168:26/26 s3_t172:22/21 s3_t228:33/33
s3_t232:27/27 s3_t262:32/31 s3_t300:36/35 s3_t337:43/41 s4_t273:21/20 s4_t290:30/29 s4_t297:31/30 s4_t311:40/40
s4_t313:37/35 s4_t320:42/41 s5_t333:30/30 s5_t349:24/22 s5_t385:29/28 s5_t409:34/34 s5_t413:30/29 s6_t450:43/42
s6_t482:50/49 s5_t501:34/33 s5_t591:44/42 s6_t514:61/57 s6_t518:46/46 s6_t521:50/48 s6_t580:78/76 s7_t570:63/60
s7_t599:61/58 s7_t627:67/63 s7_t681:58/55 s7_t695:67/65 s7_t738:68/65 s8_t628:72/65 s8_t681:74/71 s8_t694:80/77
s8_t734:67/63 s8_t872:92/86 s9_t773:77/73 s9_t775:81/76 s9_t784:75/72 s9_t946:97/95 s9_t972:90/86 s9_t991:82/79
s10_t928:103/96 s11_t961:108/97 s9_t1020:97/89 s10_t1014:102/98 s10_t1016:101/98 s10_t1088:112/103 s11_t1078:116/111
s11_t1151:117/111 s11_t1204:114/107 s12_t1049:119/104 s12_t1459:131/128 s12_t1482:144/132 s12_t1498:136/125
s13_t1286:140/124 s13_t1351:120/112 s13_t1372:146/135 s13_t1532:154/146 s13_t1591:153/152 s13_t1598:149/139
s14_t1500:154/150 s14_t1511:157/144 s14_t1629:177/149 s14_t1755:176/170 s15_t1713:172/156 s15_t1745:177/164
s15_t1749:162/158 s17_t1955:204/186 s15_t2029:190/177 s16_t2052:203/187 s16_t2136:217/209 s16_t2219:227/195
s16_t2252:214/200 s17_t2090:212/197 s17_t2166:238/214 s17_t2183:244/219 s17_t2214:216/187 s17_t2242:256/229
s17_t2366:234/218 s18_t2395:263/249 s18_t2550:257/243 s18_t2595:234/223 s18_t2641:265/237 s18_t2648:234/218
s19_t2315:253/225 s19_t2484:261/226 s19_t2661:317/269 s19_t2702:247/233 s19_t2711:270/258 s19_t2744:259/243
s20_t2668:274/253 s20_t2921:310/291 s20_t3016:277/265
"""
BKS_DUTIES: dict[str, tuple[int, int]] = {
    name: (int(single), int(multi))
    for name, single, multi in (tok.replace(":", "/").split("/") for tok in _BKS_TABLE.split())}


def default_crews(inst: RCSPInstance) -> int:
    """|C| of the paper: duties of the best-known solution of this instance and variant + 5."""
    if inst.name not in BKS_DUTIES:
        raise ValueError(f"no best-known duty count for instance {inst.name!r}; pass n_crews")
    return BKS_DUTIES[inst.name][0 if inst.variant == "single" else 1] + EXTRA_CREWS


def arc_table(inst: RCSPInstance) -> dict[str, Any]:
    """Every connection of the variant as one arc list (SOURCEs, CONNECTIONs, SINKs) with the paper's times relative
    to T0 = the earliest task start (module docstring)."""
    if not inst.time_consistent:
        raise ValueError("the MCF model needs time-consistent connection durations (see a_data)")
    if np.any(inst.start[inst.arc_head] < inst.end[inst.arc_tail]):
        raise ValueError("a CONNECTION goes backwards in time; Appendix B's incompatible set needs start(j) >= end(i)")
    T0 = int(inst.start.min())
    st, en = inst.start.astype(np.int64) - T0, inst.end.astype(np.int64) - T0
    ns, nc, nk = len(inst.src_conn), len(inst.arc_conn), len(inst.snk_conn)
    kind = np.concatenate([np.full(ns, SOURCE), np.full(nc, CONNECTION), np.full(nk, SINK)]).astype(np.int64)
    conn = np.concatenate([inst.src_conn, inst.arc_conn, inst.snk_conn]).astype(np.int64)
    tail = np.concatenate([np.full(ns, -1), inst.arc_tail, inst.snk_task]).astype(np.int64)
    head = np.concatenate([inst.src_task, inst.arc_head, np.full(nk, -1)]).astype(np.int64)
    depot = np.concatenate([inst.src_depot, np.full(nc, -1), inst.snk_depot]).astype(np.int64)
    t_lo = np.concatenate([st[inst.src_task], en[inst.arc_tail], en[inst.snk_task]])
    t_hi = np.concatenate([en[inst.src_task], en[inst.arc_head], en[inst.snk_task] + inst.c_dur[inst.snk_conn]])
    if np.any(t_hi - t_lo != inst.c_dur[conn]):
        raise AssertionError("arc times do not reproduce the connection durations")
    brk = np.concatenate([np.zeros(ns, bool), inst.arc_break, np.zeros(nk, bool)])
    bpos = np.nonzero(brk)[0]
    b_lo = en[tail[bpos]] + inst.c_tbb[conn[bpos]]
    b_hi = en[head[bpos]] - inst.c_tab[conn[bpos]]
    return {"T0": T0, "kind": kind, "conn": conn, "tail": tail, "head": head, "depot": depot,
            "cost": inst.c_cost[conn].astype(float), "t_lo": t_lo, "t_hi": t_hi, "bpos": bpos, "b_lo": b_lo, "b_hi": b_hi}


def incompatible_set(inst: RCSPInstance) -> tuple[list[int], int | None]:
    """(Q, tau): the largest Q_tau = active(tau) | active(tau + L + 1) over the distinct task start times (Appendix B),
    ordered by (start, end, index)."""
    L = int(inst.max_duty)
    st, en = inst.start, inst.end
    best, best_tau = np.zeros(0, dtype=np.int64), None
    for tau in np.unique(st).tolist():
        far = tau + L + 1
        q = np.nonzero(((st <= tau) & (en > tau)) | ((st <= far) & (en > far)))[0]
        if len(q) > len(best):
            best, best_tau = q, int(tau)
    order = np.lexsort((best, en[best], st[best]))
    return [int(t) for t in best[order]], best_tau


class _Rows:
    """Constraint matrix as COO triplets, appended family by family (block name -> row range)."""

    def __init__(self) -> None:
        self.r: list[np.ndarray] = []
        self.c: list[np.ndarray] = []
        self.v: list[np.ndarray] = []
        self.sense: list[np.ndarray] = []
        self.rhs: list[np.ndarray] = []
        self.blocks: dict[str, tuple[int, int]] = {}
        self.nnz: dict[str, int] = {}
        self.m = 0

    def add(self, name: str, nrows: int, rows, cols, vals, sense: str, rhs) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        vals = np.broadcast_to(np.asarray(vals, dtype=float), rows.shape).astype(float)
        keep = vals != 0.0
        self.r.append(rows[keep] + self.m)
        self.c.append(cols[keep])
        self.v.append(vals[keep])
        self.sense.append(np.full(nrows, sense))
        self.rhs.append(np.broadcast_to(np.asarray(rhs, dtype=float), (nrows,)).astype(float))
        self.blocks[name] = (self.m, self.m + nrows)
        self.nnz[name] = int(keep.sum())
        self.m += nrows

    def matrix(self, ncols: int) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        r = np.concatenate(self.r) if self.r else np.zeros(0, dtype=np.int64)
        c = np.concatenate(self.c) if self.c else np.zeros(0, dtype=np.int64)
        v = np.concatenate(self.v) if self.v else np.zeros(0)
        A = sp.csr_matrix((v, (r, c)), shape=(self.m, ncols))
        return A, np.concatenate(self.sense), np.concatenate(self.rhs)


def _ranges(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(i, k) for every i and every integer k in lo[i]..hi[i] (empty when hi[i] < lo[i])."""
    cnt = np.maximum(hi - lo + 1, 0)
    idx = np.repeat(np.arange(len(lo)), cnt)
    off = np.concatenate([[0], np.cumsum(cnt)[:-1]]) if len(cnt) else np.zeros(0, dtype=np.int64)
    k = np.arange(int(cnt.sum())) + np.repeat(lo - off, cnt)
    return idx, k


def _chain(rows: _Rows, name: str, C: int, size: int, first: Callable[[np.ndarray, np.ndarray], np.ndarray],
           crew: np.ndarray, var: np.ndarray, bucket: np.ndarray, decreasing: bool) -> None:
    """Chain variables v[c, j] (ids first(c, j), j = 0..size-1) defined by
        v[c, j] = v[c, j+1] + sum_{members of crew c with bucket j} var    (decreasing; v[c, size] = 0), or
        v[c, j] = v[c, j-1] + sum_{members of crew c with bucket j} var    (increasing; v[c, -1] = 0);
    members whose bucket is outside 0..size-1 belong to no v."""
    cc = np.repeat(np.arange(C), size)
    jj = np.tile(np.arange(size), C)
    row = cc * size + jj
    nb = jj + 1 if decreasing else jj - 1
    ok = (nb >= 0) & (nb < size)
    mk = (bucket >= 0) & (bucket < size)
    rows.add(name, C * size,
             np.concatenate([row, row[ok], crew[mk] * size + bucket[mk]]),
             np.concatenate([first(cc, jj), first(cc[ok], nb[ok]), var[mk]]),
             np.concatenate([np.ones(len(row)), -np.ones(int(ok.sum())), -np.ones(int(mk.sum()))]), "=", 0.0)


def build_mcf(inst: RCSPInstance, n_crews: int, *, valid_inequalities: bool = True, symmetry_breaking: bool = True,
              vi_form: str = "chain", env: gp.Env | None = None, deadline: float = math.inf) -> tuple[gp.Model, dict[str, Any]]:
    """Build model (9)-(14) (module docstring). Variables: one MVar X = [y (Ny, binary) | z (Nz, binary) |
    chain variables (Nw, continuous)]; y[v] is crew yc[v] on arc ya[v] of arc_table, z[v] crew zc[v] on break zb[v]
    (arc bpos[zb[v]]). info["rows"] maps each constraint family to its row range."""
    if vi_form not in ("chain", "explicit"):
        raise ValueError("vi_form must be 'chain' or 'explicit'")
    C = int(n_crews)
    if C < 1:
        raise ValueError("n_crews must be positive")

    def check() -> None:
        if time.perf_counter() > deadline:
            raise TimeoutError("deadline while building the MCF model")

    t_start = time.perf_counter()
    n, L, LB = inst.n, int(inst.max_duty), int(inst.max_work)
    A = arc_table(inst)
    kind, tail, head, depot = A["kind"], A["tail"], A["head"], A["depot"]
    t_lo, t_hi, bpos, b_lo, b_hi = A["t_lo"], A["t_hi"], A["bpos"], A["b_lo"], A["b_hi"]
    m = len(kind)
    T0 = A["T0"]
    Q, tau_q = incompatible_set(inst) if symmetry_breaking else ([], None)
    nQ = min(len(Q), C)
    # Variables (fixed ones are not created).
    allowed = np.ones((C, m), dtype=bool)
    for c in range(nQ):
        q = Q[c]
        allowed[c] = (t_hi - (int(inst.start[q]) - T0) <= L) & ((int(inst.end[q]) - T0) - t_lo <= L)
    yc, ya = np.nonzero(allowed)
    Ny = len(yc)
    yidx = np.full((C, m), -1, dtype=np.int64)
    yidx[yc, ya] = np.arange(Ny)
    zc, zb = np.nonzero(allowed[:, bpos])
    Nz = len(zc)
    zv = Ny + np.arange(Nz)
    y_src = np.nonzero(kind[ya] == SOURCE)[0]
    y_snk = np.nonzero(kind[ya] == SINK)[0]
    y_in = np.nonzero(head[ya] >= 0)[0]
    y_out = np.nonzero(tail[ya] >= 0)[0]
    one = lambda k: np.ones(len(k))  # noqa: E731
    rows = _Rows()
    check()
    # (9b) covering.
    if np.any(np.bincount(head[ya[y_in]], minlength=n) == 0):
        raise ValueError("a task has no incoming connection")
    rows.add("cover_9b", n, head[ya[y_in]], y_in, 1.0, ">", 1.0)
    # (9c) flow balance per crew member and task (empty rows dropped).
    keys = np.concatenate([yc[y_in] * n + head[ya[y_in]], yc[y_out] * n + tail[ya[y_out]]])
    uk, r = np.unique(keys, return_inverse=True)
    rows.add("flow_9c", len(uk), r, np.concatenate([y_in, y_out]), np.concatenate([one(y_in), -one(y_out)]), "=", 0.0)
    check()
    # (9d) at most one SOURCE arc.
    rows.add("one_path_9d", C, yc[y_src], y_src, 1.0, "<", 1.0)
    # (9e) duty length.
    coef = (t_hi - t_lo)[ya].astype(float) - L * (kind[ya] == SOURCE)
    rows.add("length_9e", C, yc, np.arange(Ny), coef, "<", 0.0)
    # (9f) one break per active crew member; (9g) on an active arc.
    rows.add("one_break_9f", C, np.concatenate([zc, yc[y_src]]), np.concatenate([zv, y_src]),
             np.concatenate([one(zv), -one(y_src)]), "=", 0.0)
    ylink = yidx[zc, bpos[zb]]
    assert np.all(ylink >= 0), "a break variable without its arc variable"
    rows.add("break_on_path_9g", Nz, np.concatenate([np.arange(Nz), np.arange(Nz)]), np.concatenate([zv, ylink]),
             np.concatenate([one(zv), -one(zv)]), "<", 0.0)
    # (9h) work before the break, (9i) after it.
    rows.add("before_break_9h", C, np.concatenate([zc, yc[y_src]]), np.concatenate([zv, y_src]),
             np.concatenate([b_lo[zb].astype(float), -(t_lo[ya[y_src]] + LB).astype(float)]), "<", 0.0)
    rows.add("after_break_9i", C, np.concatenate([yc[y_snk], zc, yc[y_src]]), np.concatenate([y_snk, zv, y_src]),
             np.concatenate([t_hi[ya[y_snk]].astype(float), -b_hi[zb].astype(float), -LB * one(y_src)]), "<", 0.0)
    # Multi-depot: SOURCE and SINK at the same crew base.
    D = len(inst.depots)
    if D > 1:
        dep = np.asarray(inst.depots, dtype=np.int64)
        keys = np.concatenate([yc[y_src] * D + np.searchsorted(dep, depot[ya[y_src]]),
                               yc[y_snk] * D + np.searchsorted(dep, depot[ya[y_snk]])])
        uk, r = np.unique(keys, return_inverse=True)
        rows.add("depot_balance", len(uk), r, np.concatenate([y_src, y_snk]),
                 np.concatenate([one(y_src), -one(y_snk)]), "=", 0.0)
    check()
    # (10) crew ordering, (11) the incompatible tasks Q on the first |Q| crew members.
    if symmetry_breaking and C > 1:
        lo, hi = y_src[yc[y_src] < C - 1], y_src[yc[y_src] > 0]
        rows.add("order_10", C - 1, np.concatenate([yc[lo], yc[hi] - 1]), np.concatenate([lo, hi]),
                 np.concatenate([one(lo), -one(hi)]), ">", 0.0)
    if nQ:
        qa = np.asarray(Q[:nQ], dtype=np.int64)
        sel = y_in[yc[y_in] < nQ]
        sel = sel[head[ya[sel]] == qa[yc[sel]]]
        rows.add("assign_Q_11", nQ, yc[sel], sel, 1.0, "=", 1.0)
    # (12)-(14) at every distinct task start time.
    Nw, K, explicit_nnz = 0, 0, 0
    if valid_inequalities:
        T = np.unique(inst.start).astype(np.int64) - T0  # tau_1 < ... < tau_K
        K = len(T)
        p_src = np.searchsorted(T, t_lo[ya[y_src]], "left")      # SOURCE in G_k  iff k <= p   (tau_lo > tau_k)
        p_12 = np.searchsorted(T, t_hi[ya[y_snk]] - L, "left")   # SINK in (12)   iff k <= p   (tau_hi > tau_k + L)
        p_13 = np.searchsorted(T, b_lo[zb] - LB, "left")         # break in (13)  iff k <= p   (beta_lo > tau_k + L_B)
        p_14 = np.searchsorted(T, t_hi[ya[y_snk]] - LB, "left")  # SINK in (14)   iff k <= p   (tau_hi > tau_k + L_B)
        r_14 = np.searchsorted(T, b_hi[zb], "left") + 1          # break in (14)  iff k >= r   (beta_hi <= tau_k)
        explicit_nnz = int(2 * p_src.sum() + p_12.sum() + p_13.sum() + np.maximum(K - r_14 + 1, 0).sum() + p_14.sum()
                           + len(y_src) * K)
        if vi_form == "chain":
            S = 5 * K + 1
            W0 = Ny + Nz
            Nw = C * S
            G = lambda c, j: W0 + c * S + j  # noqa: E731   G[c, k], k = 0..K (k = 0: every SOURCE)
            H = lambda c, j: W0 + c * S + K + 1 + j  # noqa: E731   index j = k - 1
            J = lambda c, j: W0 + c * S + 2 * K + 1 + j  # noqa: E731
            M = lambda c, j: W0 + c * S + 3 * K + 1 + j  # noqa: E731
            I = lambda c, j: W0 + c * S + 4 * K + 1 + j  # noqa: E731, E741
            _chain(rows, "chain_G", C, K + 1, G, yc[y_src], y_src, p_src, True)
            _chain(rows, "chain_H", C, K, H, yc[y_snk], y_snk, p_12 - 1, True)
            _chain(rows, "chain_J", C, K, J, zc, zv, p_13 - 1, True)
            _chain(rows, "chain_M", C, K, M, yc[y_snk], y_snk, p_14 - 1, True)
            _chain(rows, "chain_I", C, K, I, zc, zv, r_14 - 1, False)
            check()
            cc = np.repeat(np.arange(C), K)
            jj = np.tile(np.arange(K), C)
            row = np.arange(C * K)
            rows.add("vi_12", C * K, np.concatenate([row, row]), np.concatenate([H(cc, jj), G(cc, jj + 1)]),
                     np.concatenate([np.ones(C * K), -np.ones(C * K)]), "<", 0.0)
            rows.add("vi_13", C * K, np.concatenate([row, row]), np.concatenate([J(cc, jj), G(cc, jj + 1)]),
                     np.concatenate([np.ones(C * K), -np.ones(C * K)]), "<", 0.0)
            rows.add("vi_14", C * K, np.concatenate([row, row, row]), np.concatenate([I(cc, jj), M(cc, jj), G(cc, 0 * jj)]),
                     np.concatenate([np.ones(2 * C * K), -np.ones(C * K)]), "<", 0.0)
        else:
            # The rows as printed (SOURCE terms combined), row (c, j) for tau_{j+1}.
            def part(var, crew, lo, hi, val):
                i, j = _ranges(lo, hi)
                return crew[i] * K + j, var[i], np.full(len(i), val)

            def add(name, *parts):
                rows.add(name, C * K, *(np.concatenate(x) for x in zip(*parts)), "<", 0.0)

            zero_s, zero_k, zero_z = (np.zeros(len(x), dtype=np.int64) for x in (y_src, y_snk, zv))
            src_gt = part(y_src, yc[y_src], zero_s, p_src - 1, -1.0)  # -SOURCEs with tau_lo > tau_k
            add("vi_12", src_gt, part(y_snk, yc[y_snk], zero_k, p_12 - 1, 1.0))
            add("vi_13", src_gt, part(zv, zc, zero_z, p_13 - 1, 1.0))
            add("vi_14", part(zv, zc, r_14 - 1, zero_z + K - 1, 1.0), part(y_snk, yc[y_snk], zero_k, p_14 - 1, 1.0),
                part(y_src, yc[y_src], zero_s, zero_s + K - 1, -1.0))
    check()
    N = Ny + Nz + Nw
    Amat, sense, rhs = rows.matrix(N)
    model = gp.Model(env=env) if env is not None else gp.Model()
    vtype = np.full(N, GRB.CONTINUOUS)
    vtype[:Ny + Nz] = GRB.BINARY
    ub = np.full(N, np.inf)
    ub[:Ny + Nz] = 1.0
    obj = np.zeros(N)
    obj[:Ny] = A["cost"][ya]
    X = model.addMVar(N, lb=0.0, ub=ub, obj=obj, vtype=vtype)
    model.addMConstr(Amat, X, sense, rhs)
    model.ModelSense = GRB.MINIMIZE
    model.update()
    if time.perf_counter() > deadline:
        model.dispose()
        raise TimeoutError("deadline while building the MCF model")
    size = {"n_crews": C, "tasks": n, "arcs": m, "break_arcs": len(bpos), "sources": int((kind == SOURCE).sum()),
            "sinks": int((kind == SINK).sum()), "crew_bases": D, "q_size": len(Q), "q_used": nQ, "q_tau": tau_q,
            "vi_thresholds": K, "vi_explicit_nnz": explicit_nnz, "y_vars": Ny, "z_vars": Nz, "chain_vars": Nw, "fixed_y": C * m - Ny,
            "fixed_z": C * len(bpos) - Nz, "rows": {k: hi - lo for k, (lo, hi) in rows.blocks.items()},
            "nnz_by_family": dict(rows.nnz), "num_vars": int(model.NumVars), "num_bin_vars": int(model.NumBinVars),
            "num_constrs": int(model.NumConstrs), "num_nz": int(model.NumNZs),
            "build_seconds": time.perf_counter() - t_start}
    info = {"inst": inst, "X": X, "YZ": X[:Ny + Nz], "Ny": Ny, "Nz": Nz, "Nw": Nw, "yc": yc, "ya": ya, "zc": zc,
            "zb": zb, "yidx": yidx, "arcs": A, "rows": dict(rows.blocks), "Q": Q, "tau_q": tau_q, "K": K, "size": size}
    return model, info


def decode_mcf(info: dict[str, Any], values) -> tuple[list[dict[str, Any]], list[str]]:
    """One duty per active crew member of an integer solution (values of info["YZ"] or of all variables), and the
    structural issues found (each would be a bug of the model: several SOURCE arcs, a branching or broken path,
    different crew bases at both ends, a break not on the path)."""
    v = np.asarray(values, dtype=float)
    Ny, Nz = info["Ny"], info["Nz"]
    yc, ya, zc, zb = info["yc"], info["ya"], info["zc"], info["zb"]
    A = info["arcs"]
    kind, tail, head, depot, bpos = A["kind"], A["tail"], A["head"], A["depot"], A["bpos"]
    pos = np.nonzero(v[:Ny] > 0.5)[0]
    zpos = np.nonzero(v[Ny:Ny + Nz] > 0.5)[0]
    zarc: dict[int, list[int]] = {}
    for c, b in zip(zc[zpos].tolist(), zb[zpos].tolist()):
        zarc.setdefault(c, []).append(int(bpos[b]))
    duties, issues = [], []
    crews, first = np.unique(yc[pos], return_index=True)
    bounds = np.concatenate([first, [len(pos)]])
    for c, lo, hi in zip(crews.tolist(), bounds[:-1].tolist(), bounds[1:].tolist()):
        arcs = ya[pos[lo:hi]]
        src = arcs[kind[arcs] == SOURCE]
        if len(src) != 1:
            issues.append(f"crew {c}: {len(src)} SOURCE arcs")
            continue
        nxt: dict[int, int] = {}
        for a in arcs[kind[arcs] != SOURCE].tolist():
            if int(tail[a]) in nxt:
                issues.append(f"crew {c}: two arcs leave task {int(tail[a])}")
            nxt[int(tail[a])] = a
        s = int(src[0])
        tasks, path, last = [int(head[s])], [s], None
        while len(path) <= len(arcs):
            a = nxt.get(tasks[-1])
            if a is None:
                break
            path.append(a)
            if kind[a] == SINK:
                last = a
                break
            tasks.append(int(head[a]))
        if last is None:
            issues.append(f"crew {c}: path stops at task {tasks[-1]} without a SINK")
            continue
        if len(path) != len(arcs):
            issues.append(f"crew {c}: {len(arcs) - len(path)} active arcs off the path")
        if depot[s] != depot[last]:
            issues.append(f"crew {c}: SOURCE at base {int(depot[s])}, SINK at base {int(depot[last])}")
        on = set(path)
        brk = zarc.get(c, [])
        if len(brk) != 1 or brk[0] not in on:
            issues.append(f"crew {c}: break arcs {brk} (expected one on the path)")
        duties.append({"depot": int(depot[s]), "tasks": tasks})
    extra = set(zarc) - set(crews.tolist())
    if extra:
        issues.append(f"breaks on inactive crew members {sorted(extra)}")
    return duties, issues


class RCSPCrewMCF:
    """The MCF model (9)-(14) with the solver interface of RCSPCompactMIP: run(spec) returns best_objective,
    global_lb, termination_reason ("optimal", "time_limit", "memory_limit"; "infeasible" if |C| is too small),
    duties and stats (model size, build seconds, |C|, |Q|, incumbent and bound history, decode issues)."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 n_crews: int | None = None, valid_inequalities: bool = True, symmetry_breaking: bool = True,
                 vi_form: str = "chain", method: int = -1) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.method = method  # root relaxation algorithm; -1 = Gurobi's default (the paper ran CPLEX with defaults)
        self.n_crews, self.valid_inequalities = n_crews, valid_inequalities
        self.symmetry_breaking, self.vi_form = symmetry_breaking, vi_form
        self.on_incumbent = on_incumbent or (lambda duties: None)
        self.on_bound = on_bound or (lambda bound: None)

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        inst = load_rcsp(data_path)
        C = int(self.n_crews) if self.n_crews is not None else default_crews(inst)
        stats: dict[str, Any] = {"n_crews": C, "n_crews_source": "given" if self.n_crews is not None else "bks_duties+5",
                                 "valid_inequalities": self.valid_inequalities, "symmetry_breaking": self.symmetry_breaking,
                                 "vi_form": self.vi_form, "method": self.method, "decode_issues": 0, "decode_issue_examples": [],
                                 "rejected_solutions": 0, "incumbents": [], "bounds": [], "first_incumbent_seconds": None,
                                 "root_bound": None, "root_end_seconds": None}
        best: dict[str, Any] = {"obj": math.inf, "duties": None, "bound": -math.inf}
        elapsed = lambda: time.perf_counter() - t0  # noqa: E731

        def finish(reason: str) -> dict[str, Any]:
            stats["total_seconds"] = elapsed()
            return {"best_objective": best["obj"] if best["duties"] is not None else None,
                    "global_lb": best["bound"] if best["bound"] > -math.inf else None,
                    "termination_reason": reason, "duties": best["duties"], "stats": stats}

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", self.threads)
        env.setParam("Seed", self.seed)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()
        model = None
        try:
            try:
                model, info = build_mcf(inst, C, valid_inequalities=self.valid_inequalities,
                                        symmetry_breaking=self.symmetry_breaking, vi_form=self.vi_form, env=env,
                                        deadline=deadline)
            except TimeoutError:
                stats["build_completed"] = False
                return finish("time_limit")
            stats.update(info["size"])
            stats["build_completed"] = True
            stats["q_tasks"] = info["Q"]
            YZ = info["YZ"]

            def bound(value: float) -> None:
                if not math.isfinite(value) or value <= -1e90:
                    return
                b = float(math.ceil(value - 1e-6))
                if best["duties"] is not None:
                    b = min(b, best["obj"])
                if b > best["bound"]:
                    best["bound"] = b
                    if len(stats["bounds"]) < 1000:  # history (the first 1000 improvements)
                        stats["bounds"].append((round(elapsed(), 3), b))
                    self.on_bound(b)

            def offer(values, solver_obj: float) -> None:
                duties, issues = decode_mcf(info, values)
                obj, why = plan_issues(inst, duties)
                if obj is not None and abs(obj - solver_obj) > 0.5:
                    issues.append(f"plan cost {obj} differs from the solver objective {solver_obj}")
                issues += why
                if issues:
                    stats["decode_issues"] += 1
                    if len(stats["decode_issue_examples"]) < 5:
                        stats["decode_issue_examples"].append(issues[:5])
                if obj is None or issues:
                    stats["rejected_solutions"] += 1
                    return
                if obj < best["obj"] - 1e-9:
                    best.update(obj=obj, duties=duties)
                    t = elapsed()
                    if stats["first_incumbent_seconds"] is None:
                        stats["first_incumbent_seconds"] = t
                    stats["incumbents"].append((round(t, 3), obj))
                    self.on_incumbent(duties)

            def callback(m, where) -> None:
                if time.perf_counter() > deadline:
                    m.terminate()  # TimeLimit can overshoot on large models; the build already used part of the time
                    return
                if where == GRB.Callback.MIPSOL:
                    offer(m.cbGetSolution(YZ), m.cbGet(GRB.Callback.MIPSOL_OBJ))
                elif where == GRB.Callback.MIP:
                    value = m.cbGet(GRB.Callback.MIP_OBJBND)
                    if m.cbGet(GRB.Callback.MIP_NODCNT) < 0.5:
                        stats["root_bound"] = value if value > -1e90 else None
                    elif stats["root_end_seconds"] is None:
                        stats["root_end_seconds"] = elapsed()
                    bound(value)

            left = deadline - time.perf_counter()
            if left <= 0.01:
                return finish("time_limit")
            model.Params.TimeLimit = left
            model.Params.MIPGap = 0.0
            model.Params.MIPGapAbs = 0.5  # integral costs: OPTIMAL proves the incumbent optimal
            model.Params.Method = self.method
            t_opt = time.perf_counter()
            model.optimize(callback)
            stats["solve_seconds"] = time.perf_counter() - t_opt
            status = model.Status
            stats["gurobi_status"] = int(status)
            stats["node_count"] = float(model.NodeCount)
            stats["gurobi_solutions"] = int(model.SolCount)
            if model.SolCount and time.perf_counter() <= deadline:
                offer(YZ.X, float(model.ObjVal))
            if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.MEM_LIMIT, GRB.INTERRUPTED):
                try:
                    stats["gurobi_bound"] = float(model.ObjBound)
                    bound(float(model.ObjBound))
                except (gp.GurobiError, AttributeError):
                    pass
            reason = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit", GRB.INTERRUPTED: "time_limit",
                      GRB.MEM_LIMIT: "memory_limit", GRB.INFEASIBLE: "infeasible",
                      GRB.INF_OR_UNBD: "infeasible"}.get(status, f"status_{status}")
            return finish(reason)
        except MemoryError:
            return finish("memory_limit")
        except gp.GurobiError as err:
            if err.errno != GRB.Error.OUT_OF_MEMORY and "memory" not in str(err).lower():
                raise
            return finish("memory_limit")
        finally:
            if model is not None:
                model.dispose()
            env.dispose()
