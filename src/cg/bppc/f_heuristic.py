"""Heuristic-only control for BPPC (scale-up study): an iterated local search without decomposition (no column
generation, no LP, no MIP), built from the standard bin-packing neighbourhoods (Falkenauer's fitness, minimum-slack
bin filling of Gupta-Ho / Fleszar-Hindi, Alvim et al.'s fixed-k tabu repair), made conflict-aware.

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
  with relocate and swap moves of violated items (sampled), tabu tenure 10 U(0, 1) + 0.6 * #violated, aspiration;
  f = 0 is a feasible packing with one bin fewer. The repair gives up after `stall` iterations without a new best f
  (initially max(2000, 5 n), doubled after every failed attempt up to 100 n, reset after a success).
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

    def __init__(self, inst: BPPCInstance, rng: np.random.Generator, sample: int = 8, check: bool = False) -> None:
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

    def repair(self, bin_of: np.ndarray, k: int, deadline: float, stall: int) -> np.ndarray | None:
        """Tabu search from the assignment bin_of (bins 0..k-1); the feasible assignment, or None after `stall`
        iterations without a new best penalty or at the deadline."""
        n, C, w, A, g, rng = self.n, self.C, self.w, self.A, self.gamma, self.rng
        bin_of = bin_of.copy()
        load = np.bincount(bin_of, weights=w, minlength=k).astype(np.int64)
        cnt = np.zeros((k, n), dtype=np.int32)
        for i in range(n):
            cnt[bin_of[i]] += A[i]
        own = cnt[bin_of, np.arange(n)]
        f = float(self.ex(load).sum() + g * own.sum() / 2)
        best_f, last, f0 = f, 0, f
        tabu = np.zeros((n, k), dtype=np.int64)
        it = 0

        def record(ok: bool) -> None:
            own_ = cnt[bin_of, np.arange(n)]
            self.log.append({"k": k, "f0": round(f0, 1), "best_f": round(best_f, 1), "iters": it, "ok": ok,
                             "f_end": round(float(self.ex(load).sum() + g * own_.sum() / 2), 1)})

        while True:
            over = load[bin_of] > C
            own = cnt[bin_of, np.arange(n)]
            if self.check:  # tests: the tracked penalty, loads and conflict counts equal their recomputation
                exact = float(self.ex(load).sum() + g * own.sum() / 2)
                assert abs(exact - f) < 1e-6 * max(1.0, exact), (exact, f)
                assert (load == np.bincount(bin_of, weights=w, minlength=k)).all()
                assert (cnt == np.stack([A[bin_of == t].sum(axis=0) for t in range(k)])).all()
            if f < 0.5 * min(1.0, g) or it % 1000 == 0:  # resynchronize the penalty (no float drift)
                f = float(self.ex(load).sum() + g * own.sum() / 2)
                if f == 0.0:
                    record(True)
                    return bin_of
            it += 1
            self.iterations += 1
            if it - last > stall or time.perf_counter() >= deadline:
                record(False)
                return None
            viol = np.flatnonzero(over | (own > 0))
            if len(viol) > self.sample:
                viol = rng.choice(viol, self.sample, replace=False)
            best = None  # (delta, tiebreak, move)
            for i in viol.tolist():
                a = int(bin_of[i])
                la = load[a]
                rem = self.ex(la - w[i]) - self.ex(la) - g * cnt[a, i]
                # relocate i to bin b
                d = rem + self.ex(load + w[i]) - self.ex(load) + g * cnt[:, i]
                d = d.astype(float)
                d[a] = np.inf
                tabu_b = tabu[i] > it
                d[tabu_b & (f + d >= best_f - 1e-9)] = np.inf
                b = int(np.argmin(d + rng.random(k) * 1e-6))
                if np.isfinite(d[b]) and (best is None or d[b] < best[0]):
                    best = (float(d[b]), ("rel", i, a, b))
                # swap i with j of another bin
                bj = bin_of
                lb = load[bj]
                Aij = A[i]
                dsw = (self.ex(la - w[i] + w) - self.ex(la) + self.ex(lb - w + w[i]) - self.ex(lb)
                       + g * ((cnt[a] - Aij) - cnt[a, i] + (cnt[bj, i] - Aij) - cnt[bj, np.arange(n)])).astype(float)
                dsw[bj == a] = np.inf
                tabu_s = (tabu[i, bj] > it) | (tabu[np.arange(n), a] > it)
                dsw[tabu_s & (f + dsw >= best_f - 1e-9)] = np.inf
                j = int(np.argmin(dsw + rng.random(n) * 1e-6))
                if np.isfinite(dsw[j]) and (best is None or dsw[j] < best[0]):
                    best = (float(dsw[j]), ("swap", i, a, j))
            if best is None:
                continue
            delta, mv = best
            nviol = len(np.flatnonzero(over | (own > 0)))
            tenure = lambda: it + int(10 * rng.random()) + int(0.6 * nviol) + 1
            if mv[0] == "rel":
                _, i, a, b = mv
                bin_of[i] = b
                load[a] -= w[i]
                load[b] += w[i]
                cnt[a] -= A[i]
                cnt[b] += A[i]
                tabu[i, a] = tenure()
            else:
                _, i, a, j = mv
                b = int(bin_of[j])
                bin_of[i], bin_of[j] = b, a
                load[a] += w[j] - w[i]
                load[b] += w[i] - w[j]
                cnt[a] += A[j] - A[i]
                cnt[b] += A[i] - A[j]
                tabu[i, a] = tenure()
                tabu[j, b] = tenure()
            f += delta
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
                 sample: int = 8, stall: int | None = None, tabu_every: int = 3000) -> None:
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
                res = tabu.repair(tabu.start(cur, drop), len(cur) - 1, deadline, stall)
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
