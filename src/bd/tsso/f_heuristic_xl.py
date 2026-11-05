"""Primal heuristics for TSSO at scale, shared by the engineered Benders reference and the decomposition-free
controls (scale-up study).

- Product local search (`ProductSearch`): the design value separates by product, value(y) = sum_i v_i(y_i) with
  v_i(y_i) = F_i.y_i + sum_s p_s Q_is(y_i), so moves act on one product: flip one period, or shift an activation
  to a neighbouring period (close t, open t +- 1 or t +- 2). Candidate moves are ranked by a surrogate (the same
  recourse on a fixed random subsample of `sample` scenarios with renormalized probabilities; a ranking only, not a
  bound) and the best `max_exact` are evaluated exactly (b_mip_xl.DesignEvaluator); the first exact improvement is
  taken. Iterated local search kicks a product by flipping 1-3 random periods and keeps the best.
- Heuristic-only control (`TSSOHeuristicControl`, no decomposition, no bound): the shared initializer, then
  mode="ls": descent and ILS over all products until the deadline (round robin);
  mode="saa": first sample-average extensive forms (the paper's compact model) on growing random scenario subsets
  (10, 20, 40, ... scenarios; each solve time-capped, all within `saa_fraction` of the time limit), every SAA design
  evaluated exactly, then the same descent and ILS from the best design.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import gurobipy as gp
import numpy as np

from src.bd.tsso.a_data_xl import TSSOXLInstance, initial_design, load_instance
from src.bd.tsso.b_mip_xl import DesignEvaluator, ProductRecourse, build_extensive_form, is_memory_error, make_env

INF = float("inf")


class ProductSearch:
    """Surrogate-screened local search per product (see the module docstring). `on_better(i, y_i, v_i)` is called
    for every exact improvement of product i."""

    def __init__(self, inst: TSSOXLInstance, ev: DesignEvaluator, on_better: Callable[[int, np.ndarray, float], None],
                 sample: int = 64, max_exact: int = 3, seed: int = 0) -> None:
        self.inst, self.ev, self.on_better = inst, ev, on_better
        self.max_exact = max_exact
        self.rng = np.random.default_rng(seed)
        S = inst.S
        self.sample = np.sort(self.rng.choice(S, size=min(S, sample), replace=False)) if S > sample else None
        if self.sample is not None:
            p = inst.prob[self.sample]
            self.sp = p / p.sum()
        self.sur: dict[int, ProductRecourse] = {}
        self.sur_cache: dict[tuple[int, bytes], float] = {}
        self.stats = {"exact": 0, "surrogate": 0, "moves": 0, "kicks": 0}

    def exact(self, i: int, y: np.ndarray, end: float) -> float | None:
        res = self.ev.product(i, y, end)
        if res is None:
            return None
        self.stats["exact"] += 1
        return res["value"]

    def surrogate(self, i: int, y: np.ndarray, end: float) -> float | None:
        if self.sample is None:
            return self.exact(i, y, end)
        key = (i, y.tobytes())
        if key not in self.sur_cache:
            if i not in self.sur:
                self.sur[i] = ProductRecourse(self.inst, i, self.ev.env, scen=self.sample)
            left = end - time.perf_counter()
            if left <= 0:
                return None
            res = self.sur[i].solve(y.astype(float), left, duals=False)
            if res["status"] != "optimal":
                return None
            self.stats["surrogate"] += 1
            self.sur_cache[key] = float(self.inst.F[i] @ y + self.sp @ res["q"])
        return self.sur_cache[key]

    def moves(self, y: np.ndarray) -> list[np.ndarray]:
        T = len(y)
        out = []
        for t in range(T):
            z = y.copy()
            z[t] = not z[t]
            out.append(z)
        for t in np.nonzero(y)[0]:
            for dt in (-2, -1, 1, 2):
                u = t + dt
                if 0 <= u < T and not y[u]:
                    z = y.copy()
                    z[t], z[u] = False, True
                    out.append(z)
        return out

    def step(self, i: int, y: np.ndarray, val: float, end: float) -> tuple[np.ndarray, float] | None:
        base = self.surrogate(i, y, end)
        if base is None:
            return None
        cands = []
        for z in self.moves(y):
            if time.perf_counter() >= end:
                return None
            s = self.surrogate(i, z, end)
            if s is None:
                return None
            cands.append((s - base, z))
        cands.sort(key=lambda c: c[0])
        for _, z in cands[:self.max_exact]:
            v = self.exact(i, z, end)
            if v is None:
                return None
            if v < val - 1e-9 * max(1.0, abs(val)):
                self.stats["moves"] += 1
                return z, v
        return None

    def descend(self, i: int, y: np.ndarray, val: float, end: float) -> tuple[np.ndarray, float]:
        while time.perf_counter() < end:
            nxt = self.step(i, y, val, end)
            if nxt is None:
                break
            y, val = nxt
            self.on_better(i, y, val)
        return y, val

    def kick(self, i: int, y: np.ndarray, val: float, end: float) -> tuple[np.ndarray, float]:
        """One ILS step from (y, val): returns the better of the two."""
        self.stats["kicks"] += 1
        z = y.copy()
        idx = self.rng.choice(len(y), size=int(self.rng.integers(1, 4)), replace=False)
        z[idx] = ~z[idx]
        v = self.exact(i, z, end)
        if v is None:
            return y, val
        if v < val - 1e-9 * max(1.0, abs(val)):
            self.on_better(i, z, v)
            y, val = z, v
        else:
            z2, v2 = self.descend_quiet(i, z, v, end)
            if v2 < val - 1e-9 * max(1.0, abs(val)):
                self.on_better(i, z2, v2)
                y, val = z2, v2
        return y, val

    def descend_quiet(self, i: int, y: np.ndarray, val: float, end: float) -> tuple[np.ndarray, float]:
        while time.perf_counter() < end:
            nxt = self.step(i, y, val, end)
            if nxt is None:
                break
            y, val = nxt
        return y, val

    def close(self) -> None:
        for lp in self.sur.values():
            lp.dispose()
        self.sur.clear()


class TSSOHeuristicControl:
    """Initializer -> (SAA rounds) -> per-product descent -> ILS to the deadline; no decomposition, no bound."""

    def __init__(self, *, time_limit: float = 900.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0, mode: str = "ls",
                 saa_fraction: float = 0.5, saa_start: int = 10, verbose: bool = False, **_: Any) -> None:
        if mode not in ("ls", "saa"):
            raise ValueError("mode must be 'ls' or 'saa'")
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.on_incumbent = on_incumbent or (lambda y, stage=None: None)
        self.mode, self.saa_fraction, self.saa_start, self.verbose = mode, saa_fraction, saa_start, verbose

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        inst = load_instance(data_path)
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        ev = DesignEvaluator(inst, env)
        I = inst.I
        y = np.zeros((inst.I, inst.T), dtype=bool)
        v = np.full(I, INF)
        trace: list[dict] = []
        stats: dict[str, Any] = {"saa_rounds": 0, "saa_sizes": [], "saa_values": []}

        def publish(stage: str) -> None:
            if time.perf_counter() < deadline and np.all(np.isfinite(v)):
                self.on_incumbent(y.copy(), stage=stage)
                trace.append({"t": round(time.perf_counter() - t0, 3), "incumbent": float(v.sum()), "stage": stage})
                if self.verbose:
                    print(trace[-1], flush=True)

        def better(i: int, yi: np.ndarray, vi: float) -> None:
            if vi < v[i] - 1e-9 * max(1.0, abs(vi)):
                y[i], v[i] = yi, vi
                publish("local_search")

        def take(z: np.ndarray, stage: str) -> None:
            vals = []
            for i in range(I):
                r = ev.product(i, z[i], deadline)
                if r is None:
                    return
                vals.append(r["value"])
            if np.all(np.isfinite(v)) and sum(vals) >= v.sum() - 1e-9 * max(1.0, abs(sum(vals))):
                return
            y[:], v[:] = z, vals
            publish(stage)

        ps = ProductSearch(inst, ev, better, seed=self.seed)
        reason = "time_limit"
        try:
            take(initial_design(inst), "initializer")
            if self.mode == "saa":
                self.saa(inst, env, ev, t0 + self.saa_fraction * self.time_limit, deadline, take, stats)
            for i in range(I):  # descent, product by product
                if time.perf_counter() >= deadline:
                    break
                ps.descend(i, y[i].copy(), float(v[i]), deadline)
            i = 0
            while time.perf_counter() < deadline:  # ILS round robin
                ps.kick(i, y[i].copy(), float(v[i]), deadline)
                i = (i + 1) % I
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        finally:
            ps.close()
            ev.close()
            env.dispose()
        stats.update(ps.stats)
        ok = bool(np.all(np.isfinite(v)))
        return {"best_objective": float(v.sum()) if ok else None, "global_lb": None, "termination_reason": reason,
                "design": y.astype(int).tolist() if ok else None, "mode": self.mode, **stats,
                "total_seconds": time.perf_counter() - t0, "trace": trace}

    def saa(self, inst: TSSOXLInstance, env: gp.Env, ev: DesignEvaluator, saa_end: float, deadline: float,
            take: Callable, stats: dict) -> None:
        rng = np.random.default_rng(self.seed)
        n = self.saa_start
        while n < inst.S and time.perf_counter() < saa_end:
            scen = np.sort(rng.choice(inst.S, size=n, replace=False))
            model, info = build_extensive_form(inst, env, scen=scen)
            try:
                model.Params.TimeLimit = max(0.01, min(saa_end, deadline) - time.perf_counter())
                model.Params.MIPGap = 1e-4
                model.optimize()
                z = np.asarray(info["y"].X).reshape(inst.I, inst.T) > 0.5 if model.SolCount else None
                stats["saa_values"].append(float(model.ObjVal) if model.SolCount else None)
            finally:
                model.dispose()
            stats["saa_rounds"] += 1
            stats["saa_sizes"].append(n)
            if z is not None:
                take(z, f"saa_{n}")
            n *= 2
