"""Primal heuristics for the SCFL shared by the engineered Benders reference and the heuristic-only controls
(scale-up study).

- Scenario clusters (`cluster_scenarios`): k-means on the demand vectors (deterministic start at total-demand
  quantiles, Lloyd iterations), giving G artificial scenarios with the cluster mean demands and probabilities.
  Because Q_s(y) is convex in the demand vector for fixed y, Jensen's inequality gives, for every y in [0, 1]^I,
      sum_{s in g} p_s Q_s(y) >= P_g Q(y, dbar_g),
  so f.y + sum_g P_g Q(y, dbar_g) is a lower bound on the value of y (the engineered master uses the same clusters
  as partial decomposition; here it is a cheap surrogate costing G LPs instead of S).
- Design local search (`DesignSearch`): drop / add / swap moves on the open set, kept feasible for (15d) (a swap
  closes a and opens one of the `swap_neighbors` closed facilities with the most similar unit-cost rows). Moves are
  ranked by the surrogate (moves whose surrogate is not below the incumbent value are pruned: it is a lower bound),
  and the best few are evaluated exactly (S recourse LPs each); the first exact improvement is taken. Iterated local search adds
  random kicks (close k open facilities, open k closed ones, repair (15d) greedily) and returns to the best design.
- Heuristic-only control (`SCFLHeuristicControl`, no decomposition, no bound): shared basic initializer, descent,
  ILS to the deadline. mode="saa" first solves sample-average extensive forms on growing random scenario subsets
  (10, 20, 40, ... scenarios; each EF capped in time), evaluates each SAA design exactly, then runs the same
  descent and ILS from the best design: the scenario-reduction heuristic an agent could write without Benders.
  With ev_first=True the expected-value problem (mean demands, one artificial scenario) is solved first.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.scfl.a_data import SCFLInstance, load_instance
from src.bd.scfl.b_mip_xl import RecourseLP, build_extensive_form, is_memory_error, make_env, open_list
from src.bd.scfl.c_naive_bd import greedy_design

INF = float("inf")


# ---------------------------------------------------------------- scenario clusters

def cluster_scenarios(inst: SCFLInstance, G: int, iterations: int = 20) -> tuple[np.ndarray, SCFLInstance]:
    """Labels (S,) and the cluster instance (G' <= G artificial scenarios: probability-weighted mean demands)."""
    S = inst.S
    G = max(1, min(G, S))
    D, p = inst.demand, inst.prob
    order = np.argsort(inst.total_demand, kind="stable")
    centers = D[order[((np.arange(G) + 0.5) * S / G).astype(int)]].astype(float)
    labels = np.zeros(S, dtype=int)
    for _ in range(iterations):
        dist = ((D[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(dist, axis=1)
        for g in range(len(centers)):
            m = new == g
            if m.any():
                centers[g] = p[m] @ D[m] / p[m].sum()
        if np.array_equal(new, labels):
            break
        labels = new
    used = np.unique(labels)
    remap = {int(g): k for k, g in enumerate(used)}
    labels = np.array([remap[int(g)] for g in labels])
    P = np.array([p[labels == k].sum() for k in range(len(used))])
    means = np.array([p[labels == k] @ D[labels == k] / P[k] for k in range(len(used))])
    red = SCFLInstance(capacity=inst.capacity, fixed=inst.fixed, cost=inst.cost, demand=means, prob=P / P.sum(),
                       name=f"{inst.name}_clusters{len(used)}", meta={"clusters_of": inst.name})
    return labels, red


def surrogate_size(inst: SCFLInstance, clusters: int, budget: int = 100_000) -> int:
    """Number of artificial scenarios for a surrogate or master block: at most `clusters`, and at most `budget` flow
    variables in total (10 at 50 x 50, 2 at 200 x 200)."""
    return max(1, min(clusters, budget // (inst.I * inst.J)))


# ---------------------------------------------------------------- design local search

def repair(inst: SCFLInstance, y: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """Open facilities in `rank` order until (15d) holds."""
    y = y.copy()
    for i in rank:
        if inst.feasible(y):
            break
        y[i] = True
    return y


class DesignSearch:
    """Surrogate-screened local search over designs (see the module docstring). `on_eval(y, res)` is called after
    every new exact evaluation (res from RecourseLP.evaluate, with duals if requested)."""

    def __init__(self, inst: SCFLInstance, ev: RecourseLP, sur: RecourseLP, on_eval: Callable[[np.ndarray, dict], None],
                 duals: bool = False, max_exact: int = 4, seed: int = 0, swap_neighbors: int = 10) -> None:
        self.inst, self.ev, self.sur, self.on_eval = inst, ev, sur, on_eval
        self.duals, self.max_exact = duals, max_exact
        # Swap partners: the closed facilities whose unit-cost rows are closest to the facility being closed.
        C = inst.cost
        dist = np.sqrt(np.maximum(((C[:, None, :] - C[None, :, :]) ** 2).sum(axis=2), 0.0)) if inst.I <= 2000 else None
        self.near = np.argsort(dist, axis=1, kind="stable")[:, 1:] if dist is not None else None
        self.swap_neighbors = swap_neighbors
        self.rng = np.random.default_rng(seed)
        self.exact: dict[bytes, float | None] = {}
        self.cheap: dict[bytes, float | None] = {}
        dbar = inst.mean_demand
        self.rank = np.argsort(inst.fixed / inst.capacity + inst.cost @ dbar / max(float(dbar.sum()), 1e-12), kind="stable")
        self.stats = {"exact_evaluations": 0, "surrogate_evaluations": 0, "moves": 0, "kicks": 0, "exact_seconds": 0.0,
                      "surrogate_seconds": 0.0}

    def value(self, y: np.ndarray, deadline: float) -> float | None:
        """Exact value (cached); None if infeasible or out of time (out of time is not cached)."""
        key = y.tobytes()
        if key in self.exact:
            return self.exact[key]
        if not self.inst.feasible(y):
            self.exact[key] = None
            return None
        t = time.perf_counter()
        res = self.ev.evaluate(y, deadline, duals=self.duals)
        self.stats["exact_seconds"] += time.perf_counter() - t
        if res["status"] != "optimal":
            return None
        self.stats["exact_evaluations"] += 1
        self.exact[key] = res["value"]
        self.on_eval(y, res)
        return res["value"]

    def surrogate(self, y: np.ndarray, deadline: float) -> float | None:
        key = y.tobytes()
        if key not in self.cheap:
            if not self.inst.feasible(y):
                self.cheap[key] = None
            else:
                t = time.perf_counter()
                res = self.sur.evaluate(y, deadline)
                self.stats["surrogate_seconds"] += time.perf_counter() - t
                if res["status"] != "optimal":
                    return None
                self.stats["surrogate_evaluations"] += 1
                self.cheap[key] = res["value"]
        return self.cheap[key]

    def neighbours(self, y: np.ndarray, kind: str) -> list[np.ndarray]:
        out = []
        on, off = np.nonzero(y)[0], np.nonzero(~y)[0]
        if kind == "add_drop":
            for i in on:
                z = y.copy(); z[i] = False
                if self.inst.feasible(z):
                    out.append(z)
            for i in off:
                z = y.copy(); z[i] = True
                out.append(z)
        else:
            for a in on:
                if self.near is not None and self.swap_neighbors and len(off) > self.swap_neighbors:
                    partners = [b for b in self.near[a] if not y[b]][:self.swap_neighbors]
                else:
                    partners = off
                for b in partners:
                    z = y.copy(); z[a] = False; z[b] = True
                    if self.inst.feasible(z):
                        out.append(z)
        return out

    def step(self, y: np.ndarray, val: float, end: float) -> tuple[np.ndarray, float] | None:
        """One improving move (or None at a local optimum / out of time)."""
        tol = 1e-9 * max(1.0, abs(val))
        for kind in ("add_drop", "swap"):
            cands = []
            for z in self.neighbours(y, kind):
                if time.perf_counter() >= end:
                    return None
                key = z.tobytes()
                if key in self.exact:  # already evaluated exactly: a candidate only if it improves
                    if self.exact[key] is not None and self.exact[key] < val - tol:
                        cands.append((-math.inf, z))
                    continue
                s = self.surrogate(z, end)
                if s is not None and s < val - tol:  # the surrogate is a lower bound: s >= val cannot improve
                    cands.append((s, z))
            cands.sort(key=lambda t: t[0])
            for s, z in cands[:self.max_exact]:
                if time.perf_counter() >= end:
                    return None
                v = self.value(z, end)
                if v is not None and v < val - 1e-9 * max(1.0, abs(val)):
                    self.stats["moves"] += 1
                    return z, v
        return None

    def descend(self, y: np.ndarray, val: float, end: float, on_better: Callable[[np.ndarray, float], None] | None = None) -> tuple[np.ndarray, float]:
        while time.perf_counter() < end:
            nxt = self.step(y, val, end)
            if nxt is None:
                break
            y, val = nxt
            if on_better is not None:
                on_better(y, val)
        return y, val

    def kick(self, y: np.ndarray, k: int) -> np.ndarray:
        z = y.copy()
        on, off = np.nonzero(z)[0], np.nonzero(~z)[0]
        if len(on):
            z[self.rng.choice(on, size=min(k, len(on)), replace=False)] = False
        if len(off):
            z[self.rng.choice(off, size=min(k, len(off)), replace=False)] = True
        return repair(self.inst, z, self.rank)

    def ils(self, y: np.ndarray, val: float, end: float, on_better: Callable[[np.ndarray, float], None]) -> tuple[np.ndarray, float]:
        best_y, best_v = self.descend(y, val, end, on_better)
        while time.perf_counter() < end:
            self.stats["kicks"] += 1
            z = self.kick(best_y, int(self.rng.integers(1, 4)))
            v = self.value(z, end)
            if v is None:
                continue
            if v < best_v - 1e-9 * max(1.0, abs(best_v)):
                best_y, best_v = z, v
                on_better(z, v)
            z, v = self.descend(z, v, end)
            if v < best_v - 1e-9 * max(1.0, abs(best_v)):
                best_y, best_v = z, v
                on_better(z, v)
        return best_y, best_v


# ---------------------------------------------------------------- heuristic-only control

class SCFLHeuristicControl:
    """Initializer -> (SAA rounds) -> descent -> iterated local search to the deadline; no decomposition, no bound.
    Every published design has been evaluated exactly."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0, mode: str = "ls",
                 clusters: int = 10, saa_fraction: float = 0.5, saa_start: int = 10, ev_first: bool = False,
                 verbose: bool = False) -> None:
        if mode not in ("ls", "saa"):
            raise ValueError("mode must be 'ls' or 'saa'")
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.on_incumbent = on_incumbent or (lambda design, stage=None: None)
        self.on_bound = on_bound or (lambda bound: None)
        self.mode, self.clusters, self.saa_fraction, self.saa_start = mode, clusters, saa_fraction, saa_start
        self.ev_first = ev_first
        self.verbose = verbose

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        inst = load_instance(data_path)
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        state = {"ub": INF, "y": None}
        trace: list[dict] = []
        stats: dict[str, Any] = {"saa_rounds": 0, "saa_sizes": []}

        def offer(y: np.ndarray, val: float, stage: str) -> None:
            if val < state["ub"] - 1e-9 * max(1.0, abs(val)) and time.perf_counter() < deadline:
                state.update(ub=val, y=y.copy())
                self.on_incumbent(open_list(y), stage=stage)
                trace.append({"t": round(time.perf_counter() - t0, 3), "incumbent": val, "stage": stage})
                if self.verbose:
                    print(trace[-1], flush=True)

        ev = RecourseLP(inst, env)
        _, red = cluster_scenarios(inst, surrogate_size(inst, self.clusters))
        sur = RecourseLP(red, env)
        ds = DesignSearch(inst, ev, sur, lambda y, res: None, seed=self.seed)
        reason = "time_limit"
        try:
            y0 = greedy_design(inst)
            v0 = ds.value(y0, deadline)
            if v0 is not None:
                offer(y0, v0, "initializer")
            if self.mode == "saa":
                self.saa(inst, env, ds, t0 + self.saa_fraction * self.time_limit, deadline, offer, stats)
            if state["y"] is not None and time.perf_counter() < deadline:
                ds.ils(state["y"], state["ub"], deadline, lambda y, v: offer(y, v, "local_search"))
            if time.perf_counter() < deadline - 1.0:
                reason = "finished_early"
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        finally:
            ev.dispose()
            sur.dispose()
            env.dispose()
        stats.update(ds.stats)
        return {"best_objective": state["ub"] if state["y"] is not None else None, "global_lb": None,
                "termination_reason": reason, "budget_exhausted": time.perf_counter() - t0 >= self.time_limit - 1.0,
                "design": open_list(state["y"]) if state["y"] is not None else None, "mode": self.mode, **stats,
                "total_seconds": time.perf_counter() - t0, "trace": trace}

    def saa(self, inst: SCFLInstance, env: gp.Env, ds: DesignSearch, saa_end: float, deadline: float,
            offer: Callable, stats: dict) -> None:
        rng = np.random.default_rng(self.seed)
        n = self.saa_start
        rounds = [("ev", None)] if self.ev_first else []
        while n < inst.S:
            rounds.append((f"saa_{n}", n))
            n *= 2
        for stage, n in rounds:
            if time.perf_counter() >= saa_end:
                break
            if n is None:  # the expected-value problem: one artificial scenario with the mean demands
                model, info = build_extensive_form(cluster_scenarios(inst, 1)[1], env, strong=True)
            else:
                model, info = build_extensive_form(inst, env, strong=True, scenarios=np.sort(rng.choice(inst.S, size=n, replace=False)))
            try:
                model.Params.TimeLimit = max(0.01, min(saa_end, deadline) - time.perf_counter())
                model.Params.MIPGap = 1e-4
                model.optimize()
                y = np.asarray(info["y"].X) > 0.5 if model.SolCount else None
            finally:
                model.dispose()
            stats["saa_rounds"] += 1
            stats["saa_sizes"].append(n or 0)
            if y is not None:
                y = repair(inst, y, ds.rank)
                v = ds.value(y, deadline)
                if v is not None:
                    offer(y, v, stage)
