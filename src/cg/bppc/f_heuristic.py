"""Heuristic-only control for BPPC (scale-up study): an iterated local search without decomposition (no column
generation, no LP, no MIP), built from the standard bin-packing neighbourhoods (Falkenauer's fitness, minimum-slack
bin filling of Gupta-Ho / Fleszar-Hindi, the fixed-k penalty search of Capua et al. 2017 run as a TabuCol-style
tabu search), made conflict-aware.

- Start: the shared basic initializer (heuristic.basic_initializer, first fit decreasing with conflicts), published.
- Main loop, large-neighbourhood search (LNS): remove 2-6 random bins (half of the time also the least-loaded one) and
  repack their items bin by bin with minimum slack: each new bin takes the heaviest remaining item and the subset of
  the other remaining items, pairwise conflict-free and compatible with it, that fills the bin best (depth-first
  search with a weight bound and a node cap; it stops at zero slack). The repacking is accepted if it uses fewer bins,
  or as many bins with a fitness sum (load/C)^2 at least as large (favouring full and empty bins), or with
  probability 1% as many bins (diversification).
- Bin elimination by tabu repair, every `tabu_every` LNS iterations without a new best: empty one of the three
  least-loaded bins, reinsert its items where the penalty grows least, and run a tabu search on the k-bin
  assignment minimizing f = sum_b max(0, load_b - C) + gamma * (conflicting pairs inside bins), gamma = mean weight,
  over the neighbourhoods of Capua, Frota, Ochi and Vidal (2017, ILS): Relocate, Swap, Swap2vs1 and Swap1vs2 of the
  items of problematic bins, evaluated with their O(1) conflict array (TabuRepair.repair); TabuCol-style tenure and
  aspiration. Attempts alternate between this rich neighbourhood and a light one (Relocate and Swap only, more
  iterations per second): the rich one is better on many-items-per-bin instances, the light one on dense graphs.
  f = 0 is a feasible packing with one bin fewer. The repair gives up after `stall` iterations without a
  new best f (initially max(2000, 5 n), doubled after every failed attempt up to 100 n, reset after a success).
- Every improving packing is checked with bin_count and published; nothing is published at or after the deadline.
It stops when the bin count reaches the trivial bound max(greedy clique size, ceil(w(N) / C)) ("finished_early") or
at the deadline ("time_limit").
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np

from src.cg.bppc.a_data import BPPCInstance, load_bppc
from src.cg.bppc.b_mip_xl import bin_count, public
from src.cg.bppc.heuristic import basic_initializer, greedy_clique


class TabuRepair:
    """Penalty tabu search on k-bin assignments of one instance (see the module docstring)."""

    def __init__(self, inst: BPPCInstance, rng: np.random.Generator, sample: int = 3, check: bool = False) -> None:
        self.inst, self.rng, self.sample, self.check = inst, rng, sample, check
        self.log: list[dict] = []  # one record per repair (diagnostics)
        self.n, self.C = inst.n, int(inst.capacity)
        self.w = inst.weights.astype(np.int64)
        self.A = inst.adj.astype(np.int32)
        self.gamma = float(self.w.mean())
        self.iterations = 0

    def ex(self, load):
        return np.maximum(load - self.C, 0)

    def start(self, bins: list[list[int]], drop: int) -> np.ndarray:
        """A k-bin assignment: every bin of `bins` but bins[drop], whose items are reinserted heaviest first where the
        penalty grows least (ties: the fuller bin)."""
        keep = [b for t, b in enumerate(bins) if t != drop]
        k = len(keep)
        bin_of = np.empty(self.n, dtype=np.int64)
        for t, b in enumerate(keep):
            bin_of[b] = t
        load = np.array([int(self.w[b].sum()) for b in keep], dtype=np.int64)
        cnt = np.zeros((k, self.n), dtype=np.int32)
        for t, b in enumerate(keep):
            cnt[t] = self.A[b].sum(axis=0)
        for i in sorted(bins[drop], key=lambda i: (-self.w[i], i)):
            d = self.ex(load + self.w[i]) - self.ex(load) + self.gamma * cnt[:, i]
            m = d.min()
            cand = np.flatnonzero(d <= m + 1e-9)
            t = int(cand[np.argmax(load[cand])])
            bin_of[i] = t
            load[t] += self.w[i]
            cnt[t] += self.A[i]
        return bin_of

    def repair(self, bin_of: np.ndarray, k: int, deadline: float, stall: int, rich: bool = True) -> np.ndarray | None:
        """Tabu search from the assignment bin_of (bins 0..k-1); the feasible assignment, or None after `stall`
        iterations without a new best penalty or at the deadline.

        Penalty f = sum_b max(0, load_b - C) + gamma * (conflicting pairs in b). Every iteration evaluates, for every
        item of up to `sample` problematic bins a (f_b > 0): Relocate (the item to any other bin), Swap (with any
        item of another bin), Swap2vs1 (two items of a against one item of another bin) and Swap1vs2 (one item of a
        against two items of the same other bin; the pair list is built once per iteration) - the neighbourhoods of Capua et al. (2017) - using their O(1)
        conflict array cnt[b, i] (conflicts of item i with bin b), and applies the best admissible move: no moved
        item may return to a bin it left less than tenure = 10 U(0, 1) + 0.6 * (#items in problematic bins)
        iterations ago, unless the move reaches a new best f (aspiration); ties at random. f is recomputed exactly
        after every move. rich=False evaluates Relocate and Swap only (cheaper iterations)."""
        n, C, w, A, g, rng = self.n, self.C, self.w, self.A, self.gamma, self.rng
        bin_of = bin_of.copy()
        items = [[] for _ in range(k)]
        for i in range(n):
            items[bin_of[i]].append(i)
        load = np.array([int(w[l].sum()) if l else 0 for l in items], dtype=np.int64)
        cnt = np.zeros((k, n), dtype=np.int32)
        for t in range(k):
            if items[t]:
                cnt[t] = A[items[t]].sum(axis=0)
        conf = np.array([int(cnt[t, items[t]].sum()) // 2 if items[t] else 0 for t in range(k)], dtype=np.int64)
        pairs: list = [None] * k  # cached item pairs (J1, J2) of every bin
        ex = self.ex
        f = float(ex(load).sum() + g * conf.sum())
        best_f, last, f0 = f, 0, f
        tabu = np.zeros((n, k), dtype=np.int64)
        it = 0
        arange_n = np.arange(n)

        def record(ok: bool) -> None:
            self.log.append({"k": k, "f0": round(f0, 1), "best_f": round(best_f, 1), "iters": it, "ok": ok,
                             "f_end": round(f, 1)})

        def pairs_of(t: int):
            if pairs[t] is None:
                l = items[t]
                if len(l) < 2:
                    pairs[t] = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
                else:
                    I, J = np.triu_indices(len(l), 1)
                    arr = np.array(l, dtype=np.int64)
                    pairs[t] = (arr[I], arr[J])
            return pairs[t]

        def relocate(x: int, s: int, t: int, until: int) -> None:
            items[s].remove(x)
            conf[s] -= cnt[s, x]
            cnt[s] -= A[x]
            load[s] -= w[x]
            conf[t] += cnt[t, x]
            cnt[t] += A[x]
            load[t] += w[x]
            items[t].append(x)
            bin_of[x] = t
            pairs[s] = pairs[t] = None
            tabu[x, s] = until

        while True:
            if self.check:  # tests: incremental loads, conflict counts and pair counts equal their recomputation
                assert (load == np.bincount(bin_of, weights=w, minlength=k)).all()
                assert (cnt == np.stack([A[bin_of == t].sum(axis=0) for t in range(k)])).all()
                assert all(conf[t] == int(cnt[t, items[t]].sum()) // 2 for t in range(k))
            pen = ex(load) + g * conf
            if f <= 0.0:
                record(True)
                return bin_of
            it += 1
            self.iterations += 1
            if it - last > stall or time.perf_counter() >= deadline:
                record(False)
                return None
            prob = np.flatnonzero(pen > 0)
            nprob_items = sum(len(items[t]) for t in prob.tolist())
            if len(prob) > self.sample:
                prob = rng.choice(prob, self.sample, replace=False)
            Lb = load[bin_of]
            own = cnt[bin_of, arange_n]
            exL = ex(load)
            best = None  # (delta, move)
            multi = [t for t in range(k) if len(items[t]) >= 2] if rich else []  # item pairs, once per iteration
            if multi:
                GJ1 = np.concatenate([pairs_of(t)[0] for t in multi])
                GJ2 = np.concatenate([pairs_of(t)[1] for t in multi])
            else:
                GJ1 = GJ2 = np.zeros(0, dtype=np.int64)
            GPB = bin_of[GJ1]

            def consider(D: np.ndarray, tabu_mask: np.ndarray, make) -> None:
                nonlocal best
                if D.size == 0:
                    return
                D = np.where(tabu_mask & (f + D >= best_f - 1e-9), np.inf, D)
                flat = D.ravel()
                m = flat.min()
                if not np.isfinite(m) or (best is not None and m > best[0]):
                    return
                cand = np.flatnonzero(flat <= m + 1e-9)
                pick = int(cand[rng.integers(len(cand))])
                if best is None or m < best[0] or rng.random() < 0.5:
                    best = (float(m), make(np.unravel_index(pick, D.shape)))

            for a in prob.tolist():
                ia = np.array(items[a], dtype=np.int64)
                La = load[a]
                wi = w[ia]
                ca = cnt[a, ia]
                # Relocate i (in a) -> bin t
                D = (ex(La - wi) - ex(La) - g * ca)[:, None] + ex(load[None, :] + wi[:, None]) - exL[None, :] + g * cnt[:, ia].T
                D = D.astype(float)
                D[:, a] = np.inf
                consider(D, tabu[ia] > it, lambda ij, ia=ia, a=a: ("rel", int(ia[ij[0]]), a, int(ij[1])))
                # Swap i (in a) <-> j (any other bin)
                Aij = A[ia]
                D = (ex(La - wi[:, None] + w[None, :]) - ex(La) + ex(Lb[None, :] - w[None, :] + wi[:, None]) - ex(Lb)[None, :]
                     + g * ((cnt[a][None, :] - Aij) - ca[:, None] + (cnt[:, ia][bin_of].T - Aij) - own[None, :])).astype(float)
                D[:, bin_of == a] = np.inf
                consider(D, (tabu[ia][:, bin_of] > it) | (tabu[:, a] > it)[None, :],
                         lambda ij, ia=ia, a=a: ("swap", int(ia[ij[0]]), a, int(ij[1])))
                # Swap2vs1: {i1, i2} (in a) <-> j (any other bin)
                if rich and len(ia) >= 2:
                    P1, P2 = np.triu_indices(len(ia), 1)
                    i1, i2 = ia[P1], ia[P2]
                    w12 = w[i1] + w[i2]
                    a12 = A[i1, i2]
                    D = (ex(La - w12[:, None] + w[None, :]) - ex(La) + ex(Lb[None, :] - w[None, :] + w12[:, None]) - ex(Lb)[None, :]
                         + g * (-(cnt[a, i1] + cnt[a, i2] - a12)[:, None] + (cnt[a][None, :] - A[i1] - A[i2])
                                - own[None, :] + (cnt[:, i1][bin_of].T - A[i1]) + (cnt[:, i2][bin_of].T - A[i2]) + a12[:, None])).astype(float)
                    D[:, bin_of == a] = np.inf
                    consider(D, ((tabu[i1][:, bin_of] > it) | (tabu[i2][:, bin_of] > it) | (tabu[:, a] > it)[None, :]),
                             lambda ij, i1=i1, i2=i2, a=a: ("s21", (int(i1[ij[0]]), int(i2[ij[0]])), a, int(ij[1])))
                # Swap1vs2: i (in a) <-> {j1, j2} (both in one other bin)
                sel = GPB != a
                if rich and sel.any():
                    J1, J2 = GJ1[sel], GJ2[sel]
                    if len(J1):
                        PB = GPB[sel]
                        LPB = load[PB]
                        w12 = w[J1] + w[J2]
                        a12 = A[J1, J2]
                        Ai1, Ai2 = A[ia][:, J1], A[ia][:, J2]
                        D = (ex(La - wi[:, None] + w12[None, :]) - ex(La) + ex(LPB[None, :] - w12[None, :] + wi[:, None]) - ex(LPB)[None, :]
                             + g * (-(cnt[PB, J1] + cnt[PB, J2] - a12)[None, :] + (cnt[:, ia][PB].T - Ai1 - Ai2)
                                    - ca[:, None] + (cnt[a, J1] - Ai1) + (cnt[a, J2] - Ai2) + a12[None, :])).astype(float)
                        consider(D, (tabu[ia][:, PB] > it) | (tabu[J1, a] > it)[None, :] | (tabu[J2, a] > it)[None, :],
                                 lambda ij, ia=ia, a=a, J1=J1, J2=J2: ("s12", int(ia[ij[0]]), a, (int(J1[ij[1]]), int(J2[ij[1]]))))
            if best is None:
                continue
            delta, mv = best
            until = lambda: it + int(10 * rng.random()) + int(0.6 * nprob_items) + 1
            kind = mv[0]
            if kind == "rel":
                _, i, a, t = mv
                relocate(i, a, t, until())
            elif kind == "swap":
                _, i, a, j = mv
                t = int(bin_of[j])
                relocate(i, a, t, until())
                relocate(j, t, a, until())
            elif kind == "s21":
                _, (i1, i2), a, j = mv
                t = int(bin_of[j])
                relocate(i1, a, t, until())
                relocate(i2, a, t, until())
                relocate(j, t, a, until())
            else:
                _, i, a, (j1, j2) = mv
                t = int(bin_of[j1])
                relocate(i, a, t, until())
                relocate(j1, t, a, until())
                relocate(j2, t, a, until())
            f_new = float(ex(load).sum() + g * conf.sum())
            if self.check:
                assert abs(f_new - (f + delta)) < 1e-6 * max(1.0, abs(f_new)), (kind, f_new, f + delta)
            f = f_new
            if f < best_f - 1e-9:
                best_f, last = f, it


class MinSlackRepacker:
    """Minimum-slack repacking of an item pool into bins (see the module docstring)."""

    def __init__(self, inst: BPPCInstance, node_cap: int = 4000) -> None:
        self.w = [int(x) for x in inst.weights]
        self.C = int(inst.capacity)
        self.nb = [sum(1 << int(j) for j in np.flatnonzero(inst.adj[i])) for i in range(inst.n)]
        self.node_cap = node_cap

    def fill(self, i0: int, cand: list[int]) -> list[int]:
        """The subset of cand (sorted by non-increasing weight) that best fills a bin holding i0."""
        w, nb = self.w, self.nb
        cap = self.C - w[i0]
        cand = [j for j in cand if not (nb[i0] >> j) & 1 and w[j] <= cap]
        suffix = [0] * (len(cand) + 1)
        for t in range(len(cand) - 1, -1, -1):
            suffix[t] = suffix[t + 1] + w[cand[t]]
        best = [0, []]
        nodes = [0]

        def dfs(idx: int, load: int, chosen: list[int], forb: int) -> None:
            if load > best[0]:
                best[0], best[1] = load, list(chosen)
            for t in range(idx, len(cand)):
                if best[0] == cap or nodes[0] > self.node_cap or load + suffix[t] <= best[0]:
                    return
                j = cand[t]
                if (forb >> j) & 1 or load + w[j] > cap:
                    continue
                nodes[0] += 1
                chosen.append(j)
                dfs(t + 1, load + w[j], chosen, forb | nb[j])
                chosen.pop()

        dfs(0, 0, [], nb[i0])
        return best[1]

    def repack(self, pool: list[int]) -> list[list[int]]:
        rest = sorted(pool, key=lambda i: (-self.w[i], i))
        bins = []
        while rest:
            i0 = rest[0]
            sub = self.fill(i0, rest[1:])
            taken = set(sub)
            bins.append([i0] + sub)
            rest = [i for i in rest[1:] if i not in taken]
        return bins


class BPPCHeuristicControl:
    """Heuristic-only control (no decomposition); see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 sample: int = 3, stall: int | None = None, tabu_every: int = 1000) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.on_incumbent = on_incumbent or (lambda bins: None)
        self.on_bound = on_bound or (lambda bound: None)
        self.sample, self.stall, self.tabu_every = sample, stall, tabu_every

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        inst = load_bppc(data_path)
        rng = np.random.default_rng(self.seed)
        lb = max(inst.weight_bound, len(greedy_clique(inst.adj, inst.weights)))
        self.on_bound(float(lb))
        best: dict[str, Any] = {"obj": math.inf, "bins": None, "first": None, "t": None}
        stats = {"lns_iterations": 0, "lns_accepted": 0, "tabu_attempts": 0, "tabu_successes": 0}

        def offer(bins: list[list[int]], stage: str) -> bool:
            pub = public(bins)
            obj = bin_count(inst, pub)
            if obj is None or obj >= best["obj"] or time.perf_counter() >= deadline:
                return False
            best.update(obj=obj, bins=pub, t=time.perf_counter() - t0, stage=stage)
            if best["first"] is None:
                best["first"] = best["t"]
            self.on_incumbent(pub)
            return True

        w, C = inst.weights, inst.capacity
        cur = [list(b) for b in basic_initializer(inst)]
        offer(cur, "initializer")
        tabu = TabuRepair(inst, rng, sample=self.sample)
        rep = MinSlackRepacker(inst)
        base_stall = self.stall or max(2000, 5 * inst.n)
        stall = base_stall  # doubled after every failed bin elimination (up to 100 n), reset after a success
        loads = [int(w[b].sum()) for b in cur]
        it = last = 0
        while len(cur) > lb and time.perf_counter() < deadline:
            it += 1
            stats["lns_iterations"] += 1
            m = len(cur)
            r = int(min(m, rng.integers(2, 7)))
            D = set(rng.choice(m, r, replace=False).tolist())
            if rng.random() < 0.5:
                D.add(int(np.argmin(loads)))
            pool = [i for t in D for i in cur[t]]
            new = rep.repack(pool)
            old_fit = sum((loads[t] / C) ** 2 for t in D)
            new_loads = [int(w[b].sum()) for b in new]
            new_fit = sum((x / C) ** 2 for x in new_loads)
            if len(new) < len(D) or (len(new) == len(D) and (new_fit >= old_fit - 1e-12 or rng.random() < 0.01)):
                keep = [t for t in range(m) if t not in D]
                cur = [cur[t] for t in keep] + new
                loads = [loads[t] for t in keep] + new_loads
                stats["lns_accepted"] += 1
                if len(cur) < best["obj"]:
                    if not offer(cur, "lns"):
                        break
                    last = it
            if it - last >= self.tabu_every and len(cur) > lb and time.perf_counter() < deadline:
                last = it  # one bin-elimination attempt per stagnation period
                stats["tabu_attempts"] += 1
                order = np.argsort(loads, kind="stable")
                drop = int(order[rng.integers(0, min(3, len(cur)))])
                res = tabu.repair(tabu.start(cur, drop), len(cur) - 1, deadline, stall,
                                  rich=stats["tabu_attempts"] % 2 == 1)  # alternate rich and light neighbourhoods
                stall = base_stall if res is not None else min(2 * stall, max(base_stall, 100 * inst.n))
                if res is not None:
                    bins = [np.flatnonzero(res == t).tolist() for t in range(len(cur) - 1)]
                    cur = [b for b in bins if b]
                    loads = [int(w[b].sum()) for b in cur]
                    stats["tabu_successes"] += 1
                    if not offer(cur, "tabu"):
                        break
        reached = best["obj"] <= lb
        return {"best_objective": best["obj"] if best["bins"] else None, "global_lb": float(lb),
                "termination_reason": "finished_early" if reached else "time_limit", "bins": best["bins"],
                "first_incumbent_seconds": best["first"], "best_seconds": best["t"], "best_stage": best.get("stage"),
                **stats, "tabu_iterations": tabu.iterations, "repair_log_tail": tabu.log[-6:], "trivial_bound": lb,
                "total_seconds": time.perf_counter() - t0}
