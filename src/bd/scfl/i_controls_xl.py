"""Two-stage stochastic CFL at scale: trivial-certificate controls added for Stage 1 (protocol v4 sections 6 and 16).
The others (empty, one-shot at the shared initializer's design, one-shot at an SAA design, the no-feedback pool, the
Jensen aggregation control) are in g_bound_xl.

- `analytical_terms` / `SCFLAnalytical`: the closed-form cuts of the uncapacitated relaxation, the analogue of the
  k-median / UFL analytical cut set. With capacity multipliers alpha = 0, any v gives the dual-feasible term
  (v, alpha = 0, beta_ij = (v_j - c_ij)^+) of scenario s, i.e. the valid cut
      theta_s >= sum_j d_sj v_j - sum_i (sum_j d_sj (v_j - c_ij)^+) y_i ,
  which needs no LP solve. The frozen set: for rank r = 1..R, v^r_j = the r-th smallest unit cost of customer j
  (over the facilities, ties kept), one single-term cut (s, v^r, 0) for every scenario s, ranks in increasing order
  and scenarios in index order within a rank; R the largest rank whose cumulative expanded size (h_view_xl
  term_sizes, the canonical count) fits the certificate limit. No optimization and no point of any master.
- `SCFLHeurOneShot` ('one-shot cuts at a single heuristic y' for the heuristic control): the heuristic-only primal
  control f_heuristic.SCFLHeuristicControl (mode 'saa': SAA extensive forms on growing scenario samples, then
  drop/add/swap descent and ILS) runs until T minus a reserve (max(60 s, 10% of T)); its incumbents are published as
  they come. In the reserve, the exact recourse duals at its best design are published once as one cut per scenario
  (the one-shot certificate). The heuristic's primal value at T is therefore that of a slightly shorter run.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.bd.scfl.b_mip_xl import RecourseLP, make_env
from src.bd.scfl.g_bound_xl import _Base
from src.bd.scfl.h_view_xl import cert_bytes, term_sizes


def analytical_terms(cost: np.ndarray, S: int, limit: int) -> tuple[list[np.ndarray], int, int]:
    """(levels v^1..v^R, R, expanded size): see the module docstring (alpha = 0 for every term)."""
    I, J = cost.shape
    srt = np.sort(cost, axis=0, kind="stable")            # (I, J): the r-th smallest unit cost per customer
    zero = np.zeros((1, I))
    levels, size = [], 0
    for r in range(I):
        v = srt[r].copy()
        add = int(term_sizes(cost, v[None, :], zero)[0]) * S
        if size + add > limit:
            break
        levels.append(v)
        size += add
    return levels, len(levels), size


class SCFLAnalytical(_Base):
    """Trivial-certificate control 'analytical' (module docstring), published once at start; no incumbent."""

    def run(self, data_path: str) -> dict[str, Any]:
        self.start(data_path)
        inst = self.inst
        levels, R, size = analytical_terms(inst.cost, inst.S, self.cert_limit)
        zero = np.zeros(inst.I)
        cuts = [[(s, v, zero)] for v in levels for s in range(inst.S)]
        self.emit(lambda k: cert_bytes(cuts, version=k), force=True)
        return {"termination_reason": "published", "kind": "analytical", "rank_R": R, "terms": len(cuts),
                "expanded_size": size, "total_seconds": self.elapsed()}


class SCFLHeurOneShot(_Base):
    """Heuristic-only primal control with the one-shot certificate of its best design (module docstring)."""

    def __init__(self, *, mode: str = "saa", reserve_min: float = 60.0, reserve_fraction: float = 0.1, **kw) -> None:
        super().__init__(**kw)
        self.mode, self.reserve_min, self.reserve_fraction = mode, reserve_min, reserve_fraction

    def run(self, data_path: str) -> dict[str, Any]:
        from src.bd.scfl.f_heuristic import SCFLHeuristicControl
        self.start(data_path)
        inst = self.inst
        reserve = max(self.reserve_min, self.reserve_fraction * self.time_limit)
        latest: dict[str, Any] = {"design": None, "n": 0}

        def on_inc(design, stage=None):
            latest["design"] = [int(i) for i in design]   # the heuristic publishes strict improvements only
            latest["n"] += 1
            self.on_incumbent(latest["design"], stage=stage)

        res = SCFLHeuristicControl(time_limit=max(1.0, self.time_limit - reserve), threads=self.threads, on_incumbent=on_inc,
                                   soft_mem_gb=self.soft_mem_gb, seed=self.seed, mode=self.mode).run(data_path)
        info: dict[str, Any] = {"heuristic": {k: v for k, v in res.items() if k not in ("trace", "design")
                                              and not isinstance(v, (list, dict))},
                                "heuristic_best": res.get("best_objective"), "incumbents": latest["n"], "reserve": reserve}
        if latest["design"] is not None and self.left() > 0:
            y = np.zeros(inst.I, dtype=bool)
            y[latest["design"]] = True
            env = make_env(self.threads, self.soft_mem_gb, self.seed)
            ev = None
            try:
                ev = RecourseLP(inst, env)
                r = self.design_duals(ev, y)
            finally:
                if ev is not None:
                    ev.dispose()
                env.dispose()
            if r is not None:
                cuts = self.single_cuts(np.arange(inst.S), r[1], r[2])
                info["certificate_published"] = self.emit(lambda k: cert_bytes(cuts, version=k), force=True)
                info["terms"] = len(cuts)
                info["design_value"] = r[0]
        return {"termination_reason": "published" if info.get("certificate_published") else res.get("termination_reason"),
                "best_objective": res.get("best_objective"), "design": latest["design"], "total_seconds": self.elapsed(),
                **info}
