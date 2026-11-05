"""UFL at scale: the trivial-certificate control added for Stage 1 (protocol v4 section 16: the k-median Stage 1 failed
because an analytically generated cut set reached the LP bound target; UFL has the same closed-form subproblem, so
its Stage 1 carries the same control). The other trivial-certificate controls (empty, one-shot at the shared
initializer's design, one-shot at the ILS control's best design, the no-feedback pool) are in g_bound_xl.

`analytical_cuts` is the k-median recipe (src/bd/km/f_heuristic_xl.analytical_cuts) transcribed to UFL: for every
customer j the closed-form cuts (j, v) of g_bound_xl,
    theta_j >= v - sum_i (v - D_ij)^+ y_i ,
at the distinct cost levels of its R cheapest facilities (v = its r-th smallest cost, r = 1..R, first occurrence of
each level), R the largest common rank whose expanded size fits the certificate limit. Without ties the cut at rank
r has expanded size 1 + #{i: D_ij < v} = r. It uses no optimization and no point of any master: the truncated
"radius" (Cornuejols-Nemhauser-Wolsey) reformulation of UFL written as y-space cuts. With every rank (R = m) the
master with these cuts is exactly the compact LP, so the control measures how much of the LP bound the cheapest
levels already carry.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.bd.ufl.g_bound_xl import CERT_LIMIT, INF, Sorted, _Base, cert_bytes


def analytical_cuts(srt: Sorted, limit: int = CERT_LIMIT) -> tuple[np.ndarray, np.ndarray, int]:
    """(j, v, R): the frozen analytical cut set (module docstring), customers in index order within each rank."""
    n, m = srt.n, srt.m
    js, vs, size, R = [], [], 0, 0
    for r in range(1, m + 1):
        new = np.ones(n, dtype=bool) if r == 1 else srt.ds[:, r - 1] != srt.ds[:, r - 2]
        add = int(new.sum()) * r
        if size + add > limit:
            break
        js.append(np.nonzero(new)[0])
        vs.append(srt.ds[new, r - 1])
        size += add
        R = r
    if not js:
        return np.zeros(0, dtype=np.int64), np.zeros(0), 0
    return np.concatenate(js), np.concatenate(vs), R


class UFLAnalytical(_Base):
    """Trivial-certificate control 'analytical': the analytical cut set published once at start (no incumbent)."""

    def run(self, data_path: str) -> dict[str, Any]:
        self.start(data_path)
        srt = Sorted(self.inst.D)
        j, v, R = analytical_cuts(srt, self.cert_limit)
        size = int(len(j) + srt.counts(j, v).sum()) if len(j) else 0
        if size > self.cert_limit:      # cannot happen (sizes are r per first-occurrence cut); kept as a guard
            raise AssertionError("analytical cut set exceeds the size limit")
        self.emit(lambda k: cert_bytes(j, v, version=k), force=True)
        return {"termination_reason": "published", "kind": "analytical", "rank_R": R, "cuts": int(len(j)),
                "expanded_size": size, "best_objective": self.ub if self.ub < INF else None,
                "total_seconds": time.perf_counter() - self.t0}
