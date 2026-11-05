"""Engineered column-generation reference for large GAP (scale-up study; the paper's references are unchanged).

Dantzig-Wolfe master over agent patterns (minimization):
    min sum_p c_p lam_p  s.t.  sum_{p covers i} lam_p >= 1 (dual u_i),  sum_{p of agent k} lam_p <= 1 (dual v_k <= 0).
Pricing per agent is a 0-1 knapsack with integer weights, solved exactly by a vectorized DP. The Lagrangian bound
L(u) = sum_i u_i + sum_k min(0, min_x sum_i (c_ik - u_i) x_i) is valid for any u, so every pricing round gives a
lower bound (rounded up for integer costs after a floating-point allowance).

Stages, on one deadline; every new incumbent and bound is published when found:
1. Capacity Lagrangian: Kelley cutting planes in a box trust region on the m multipliers mu of the capacity rows.
   Its value is the LP bound, and u_i = min_k c_ik + mu_k r_ik satisfies L(u) >= that value (first stability center).
2. Primal start: regret construction on the Lagrangian costs c_ik + mu_k r_ik, then a penalized shift / swap /
   ejection-chain search with adaptive capacity weights (strategic oscillation).
3. Lagrangian master: the volume algorithm on L(u), pricing every agent per step, alternating with improvement slices
   that pick, by measured gain per second, either neighborhood price-and-branch (a set of free jobs re-optimized over
   all agents with residual capacities: CG with knapsack pricing on a small master, then its integer restricted master)
   or a fresh construction from the current duals' knapsack ratios followed by the penalized search.
4. LP master (instances with at most lp_max_jobs jobs): the restricted master LP seeded with the latest pricing patterns
   and the incumbent, Wentges smoothing around the best Lagrangian point. If it converges within its share,
   price-and-branch follows: dives fix one master column per node and re-run CG, and restricted-master MIPs run,
   alternating with improvement slices. Otherwise stage 3 continues to the deadline.
At 1600 jobs (20 to 80 agents) the full LP master does not pay: its optimal basis needs about n patterns, and seeded with
about 1 500 pricing patterns plus the incumbent its value did not move in 26 iterations while each re-solve grew to
seconds of degenerate pivots. The bound then comes from the Lagrangian master, which reaches the LP bound in seconds.
"""
from __future__ import annotations

import collections
import math
import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.cg.gap.a_gen_data import GAPInstance


def arrays(inst: GAPInstance) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cost (m, n), resource (m, n), capacity (m,)"""
    m, n = len(inst.machines), len(inst.jobs)
    cost, res = np.zeros((m, n)), np.zeros((m, n), dtype=np.int64)
    for job in inst.jobs:
        for a in job.assignments:
            cost[a.machine_id, job.id], res[a.machine_id, job.id] = a.cost, a.resource
    cap = np.array([mc.capacity for mc in sorted(inst.machines, key=lambda x: x.id)], dtype=np.int64)
    return cost, res, cap


def knapsack(profit: np.ndarray, weight: np.ndarray, cap: int) -> tuple[float, np.ndarray]:
    """Exact 0-1 knapsack max sum profit x, sum weight x <= cap, over items with positive profit."""
    # No profit tolerance: dropping slightly positive items would understate the knapsack value and
    # could overstate a Lagrangian lower bound.
    items = np.nonzero((profit > 0) & (weight <= cap))[0]
    dp = np.zeros(cap + 1)
    take = np.zeros((len(items), cap + 1), dtype=bool)
    for t, i in enumerate(items):
        w, p = int(weight[i]), profit[i]
        cand = dp[:cap + 1 - w] + p  # a copy, so the update below reads the previous layer
        better = cand > dp[w:]
        take[t, w:] = better
        np.copyto(dp[w:], cand, where=better)
    x = np.zeros(len(profit), dtype=bool)
    c = int(np.argmax(dp))
    for t in range(len(items) - 1, -1, -1):
        if take[t, c]:
            x[items[t]] = True
            c -= int(weight[items[t]])
    return float(dp.max()), x


def certified_bound(lag: float, u: np.ndarray, integral: bool) -> float:
    """Lagrangian value turned into a reported lower bound, with an allowance for floating-point error
    in the n-term sums (relative 1e-9 per term) before rounding up for integer costs."""
    allowance = 1e-6 + 1e-9 * len(u) * max(1.0, float(np.abs(u).max()))
    return math.ceil(lag - allowance) if integral else lag - allowance


def evaluate(assign: np.ndarray, cost: np.ndarray, res: np.ndarray, cap: np.ndarray) -> float | None:
    n = len(assign)
    load = np.bincount(assign, weights=res[assign, np.arange(n)], minlength=len(cap))
    return float(cost[assign, np.arange(n)].sum()) if np.all(load <= cap) else None


def capacity_lagrangian(cost: np.ndarray, res: np.ndarray, cap: np.ndarray, env: gp.Env, deadline: float, on_value) -> np.ndarray:
    """Maximize g(mu) = sum_i min_k (c_ik + mu_k r_ik) - mu b over mu >= 0 (the Lagrangian dual of the capacity rows,
    whose optimum is the LP bound) by Kelley cutting planes in a box trust region; on_value(g, terms) receives every
    improved value with its summed terms. Returns the best multipliers."""
    m, n = cost.shape
    idx = np.arange(n)

    def value(mu):
        pen = cost + mu[:, None] * res
        a = pen.argmin(axis=0)
        return pen[a, idx], a

    def cut(a):
        load = np.bincount(a, weights=res[a, idx], minlength=m)
        model.addLConstr(gp.LinExpr([1.0] + list(cap - load), [theta] + mus), GRB.LESS_EQUAL, float(cost[a, idx].sum()))

    mu_hi = float(cost.max() - cost.min()) + 1.0  # above it an agent is never strictly cheapest
    # Typical cost-per-resource trade-off of a job: the start point and the initial box size.
    scale = min(mu_hi, max(1e-6, float(np.median((cost.max(0) - cost.min(0)) / np.maximum(1, res.max(0) - res.min(0))))))
    model = gp.Model(env=env)
    model.ModelSense = GRB.MAXIMIZE
    theta = model.addVar(lb=-GRB.INFINITY, obj=1.0)
    mus = [model.addVar(lb=0.0, ub=mu_hi) for _ in range(m)]
    center = np.full(m, scale)
    terms, a = value(center)
    best = float(terms.sum() - center @ cap)
    on_value(best, np.concatenate([terms, center * cap]))
    cut(a)
    delta, delta_min = scale, 1e-4 * scale
    while time.perf_counter() < deadline:
        model.setAttr("LB", mus, np.maximum(0.0, center - delta).tolist())
        model.setAttr("UB", mus, np.minimum(mu_hi, center + delta).tolist())
        model.Params.TimeLimit = max(0.01, deadline - time.perf_counter())
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            break
        trial = np.maximum(np.array(model.getAttr("X", mus)), 0.0)
        predicted = theta.X - best
        if predicted <= 1e-9 * (1.0 + abs(best)):
            break  # the model's maximum over a box around the center is the center: optimal (concavity)
        terms, a = value(trial)
        g = float(terms.sum() - trial @ cap)
        cut(a)
        if g >= best + 0.1 * predicted:  # serious step; widen the box if the step reached it
            if np.abs(trial - center).max() >= delta - 1e-9:
                delta *= 2.0
            center, best = trial, g
            on_value(best, np.concatenate([terms, center * cap]))
        elif g < best:
            delta = max(0.7 * delta, delta_min)
    model.dispose()
    return center


def knapsack_ratios(u: np.ndarray, cost: np.ndarray, res: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """Per-agent capacity price at duals u: the profit/weight ratio of the critical item of agent k's
    knapsack LP over profits u - c_k (zero when everything profitable fits)."""
    m = len(cap)
    mu = np.zeros(m)
    for k in range(m):
        profit = u - cost[k]
        items = np.nonzero(profit > 0)[0]
        ratio = profit[items] / np.maximum(res[k, items], 1e-9)
        order = np.argsort(-ratio)
        j = int(np.searchsorted(np.cumsum(res[k, items[order]]), cap[k], side="right"))
        mu[k] = ratio[order[j]] if j < len(order) else 0.0
    return mu


def regret_assign(score: np.ndarray, res: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """MTHG-style construction: jobs with the largest regret (second-best minus best score over agents where they
    still fit) get their best agent, at most one job per agent per round. A job that fits nowhere takes its
    best-score agent; the overload is left to the penalized search."""
    m, n = score.shape
    left = cap.astype(float).copy()
    assign = np.full(n, -1)
    todo = np.arange(n)
    while len(todo):
        s = np.where(res[:, todo] <= left[:, None], score[:, todo], np.inf)
        first = s.argmin(axis=0)
        two = np.partition(s, 1, axis=0)[:2] if m > 1 else np.vstack([s[0], np.full(len(todo), np.inf)])
        fits = np.isfinite(two[0])
        regret = np.full(len(todo), np.inf)  # jobs that fit nowhere are placed first
        regret[fits] = np.where(np.isfinite(two[1, fits]), two[1, fits] - two[0, fits], 1e300)
        placed, used = [], np.zeros(m, dtype=bool)
        for r in np.argsort(-regret, kind="stable")[:m]:
            i = todo[r]
            k = int(first[r]) if fits[r] else int(score[:, i].argmin())
            if not used[k]:
                used[k] = True
                assign[i] = k
                left[k] -= res[k, i]
                placed.append(r)
        todo = np.delete(todo, placed)
    return assign


def move_deltas(assign, load, cost, res, cap, w):
    """Penalized deltas (cost + sum_k w_k * overload_k) of all moves:
    shift[k, i] (job i to agent k), swap[i, j] (i and j exchange agents) and chain[i, j] (i to the agent of j,
    j to its best other agent dest[j], judged with the current loads; dest[j] = agent of i is left to the swap)."""
    m, n = cost.shape
    idx = np.arange(n)
    over = np.maximum(0.0, load - cap)
    ra, ca = res[assign, idx], cost[assign, idx]
    insert = cost - ca[None, :] + w[:, None] * (np.maximum(0.0, load[:, None] + res - cap[:, None]) - over[:, None])
    insert[assign, idx] = np.inf
    remove = w[assign] * (np.maximum(0.0, load[assign] - ra - cap[assign]) - over[assign])
    shift = insert + remove[None, :]
    c_to, r_to = cost[assign, :], res[assign, :]  # [j, i]: cost / resource of job i on the agent of job j
    wa, la, ba, oa = w[assign], load[assign], cap[assign], over[assign]
    same = assign[:, None] == assign[None, :]
    new_i = la[:, None] - ra[:, None] + r_to  # [i, j]: load of i's agent after the swap (loses i, gains j)
    swap = c_to.T + c_to - ca[:, None] - ca[None, :]
    swap += wa[:, None] * (np.maximum(0.0, new_i - ba[:, None]) - oa[:, None])
    swap += wa[None, :] * (np.maximum(0.0, new_i.T - ba[None, :]) - oa[None, :])
    swap[same] = np.inf
    dest = insert.argmin(axis=0)
    new_j = la[None, :] + r_to.T - ra[None, :]  # [i, j]: load of j's agent after gaining i and losing j
    chain = c_to.T - ca[:, None] + remove[:, None] + insert[dest, idx][None, :]
    chain += wa[None, :] * (np.maximum(0.0, new_j - ba[None, :]) - oa[None, :])
    chain[same | (dest[None, :] == assign[:, None])] = np.inf
    return shift, swap, chain, dest


def descend(assign: np.ndarray, cost, res, cap, w: np.ndarray, deadline: float) -> tuple[np.ndarray, np.ndarray]:
    """Steepest descent on the penalized cost; each round applies the best improving moves on disjoint agents."""
    m, n = cost.shape
    idx = np.arange(n)
    assign = assign.copy()
    load = np.bincount(assign, weights=res[assign, idx], minlength=m)
    while time.perf_counter() < deadline:
        shift, swap, chain, dest = move_deltas(assign, load, cost, res, cap, w)
        moves = []
        for kind, d in enumerate((shift, swap, chain)):
            top = np.argpartition(d, min(d.size - 1, 4 * m), axis=None)[:4 * m]
            moves += [(float(d.flat[f]), kind, *divmod(int(f), d.shape[1])) for f in top if d.flat[f] < -1e-9]
        if not moves:
            break
        used = np.zeros(m, dtype=bool)
        for _, kind, r, c in sorted(moves):
            agents = [r, assign[c]] if kind == 0 else [assign[r], assign[c]] if kind == 1 else [assign[r], assign[c], dest[c]]
            if used[agents].any():
                continue  # moves on disjoint agents have independent deltas
            used[agents] = True
            if kind == 0:  # shift job c to agent r
                load[assign[c]] -= res[assign[c], c]
                load[r] += res[r, c]
                assign[c] = r
            elif kind == 1:  # swap jobs r and c
                ki, kj = assign[r], assign[c]
                load[ki] += res[ki, c] - res[ki, r]
                load[kj] += res[kj, r] - res[kj, c]
                assign[r], assign[c] = kj, ki
            else:  # job r to the agent of job c, job c to dest[c]
                ki, kj, kd = assign[r], assign[c], dest[c]
                load[ki] -= res[ki, r]
                load[kj] += res[kj, r] - res[kj, c]
                load[kd] += res[kd, c]
                assign[r], assign[c] = kj, kd
    return assign, load


def penalty_search(assign, cost, res, cap, w0: np.ndarray, deadline: float, on_feasible, patience: int = 6) -> np.ndarray | None:
    """Strategic oscillation: descend on the penalized cost; raise the weights of overloaded agents until feasible,
    then lower all weights to cross the boundary again. Every improving feasible point goes to on_feasible.
    Stops after `patience` feasible rounds without improvement, when stuck infeasible at the largest weights,
    or at the deadline. Returns the best feasible."""
    n = len(assign)
    idx = np.arange(n)
    w, w_max = w0.copy(), 1e6 * max(1.0, float(w0.max()))
    best, best_obj, idle = None, math.inf, 0
    while time.perf_counter() < deadline and idle < patience:
        assign, load = descend(assign, cost, res, cap, w, deadline)
        over = load > cap
        if over.any():
            if w[over].min() >= w_max:
                break
            w[over] = np.minimum(1.5 * w[over], w_max)
            continue
        obj = float(cost[assign, idx].sum())
        if obj < best_obj - 1e-9:
            best, best_obj, idle = assign.copy(), obj, 0
            on_feasible(best)
        else:
            idle += 1
        w *= 0.9
    return best


def sub_gap(cost, res, cap, start: np.ndarray, u: np.ndarray, env: gp.Env, deadline: float, integral: bool) -> np.ndarray | None:
    """Price-and-branch on a small GAP (a neighborhood of a few agents): column generation with exact knapsack
    pricing, Wentges smoothing around u, seeded with the start patterns; then the integer restricted master with a
    cutoff below the start cost. Returns an improving assignment (agent positions) or None."""
    q, n = cost.shape
    model = gp.Model(env=env)
    art = [model.addVar(obj=10.0 * float(cost.max()) + 1.0) for _ in range(n)]
    cover = [model.addLConstr(gp.LinExpr([(1.0, art[i])]), GRB.GREATER_EQUAL, 1.0) for i in range(n)]
    conv = [model.addLConstr(gp.LinExpr(), GRB.LESS_EQUAL, 1.0) for _ in range(q)]
    columns, seen, lam = [], set(), []

    def add(k, x):
        key = (k, np.packbits(x).tobytes())
        if key in seen or not x.any():
            return False
        seen.add(key)
        jobs = np.nonzero(x)[0]
        lam.append(model.addVar(obj=float(cost[k, jobs].sum()), column=gp.Column([1.0] * (len(jobs) + 1), [cover[int(i)] for i in jobs] + [conv[k]])))
        columns.append((k, x))
        return True

    for k in range(q):
        add(k, start == k)
    center = u.copy()
    try:
        while time.perf_counter() < deadline:
            model.Params.TimeLimit = max(0.01, deadline - time.perf_counter())
            model.optimize()
            if model.Status != GRB.OPTIMAL:
                return None
            u_true, v_true = np.array(model.getAttr("Pi", cover)), np.array(model.getAttr("Pi", conv))
            best_lag, best_point, added = -math.inf, center, 0
            for point in (0.5 * center + 0.5 * u_true, u_true):
                lag = point.sum()
                for k in range(q):
                    val, x = knapsack(point - cost[k], res[k], int(cap[k]))
                    lag -= val
                    if float((cost[k] - u_true)[x].sum()) - v_true[k] < -1e-7 and add(k, x):
                        added += 1
                if lag > best_lag:
                    best_lag, best_point = lag, point
                if added:
                    break
            center = best_point
            if not added:
                break
        if time.perf_counter() >= deadline:
            return None
        for v in lam:
            v.VType = GRB.BINARY
        for v in art:
            v.UB = 0.0
        start_cost = float(cost[start, np.arange(n)].sum())
        model.Params.Cutoff = start_cost - (0.5 if integral else 1e-6)
        model.Params.TimeLimit = max(0.01, deadline - time.perf_counter())
        model.optimize()
        if not model.SolCount:
            return None
        assign = np.full(n, -1)
        for v, (k, x) in zip(lam, columns):
            if v.X > 0.5:
                assign[x] = k
        return assign if np.all(assign >= 0) else None
    finally:
        model.dispose()


class Volume:
    """Volume algorithm (Barahona and Anbil, 2000) on the Lagrangian dual of the assignment rows. Every step prices all
    agents exactly; the moving average xbar of the pricing solutions gives the direction and a Polyak-type step
    toward the target. Keeps the pricing solutions of the last steps as candidate master columns."""

    def __init__(self, cost: np.ndarray, res: np.ndarray, cap: np.ndarray, u: np.ndarray, keep: int) -> None:
        self.cost, self.res, self.cap = cost, res, cap
        self.u, self.best_u, self.best = u.astype(float).copy(), u.astype(float).copy(), -math.inf
        self.factor, self.stall, self.xbar, self.steps = 1.0, 0, None, 0
        self.recent: collections.deque = collections.deque(maxlen=keep)

    @property
    def converged(self) -> bool:
        return self.factor < 1e-4

    def step(self, target: float) -> tuple[float, np.ndarray]:
        """One pricing round at the current point; returns its Lagrangian value and the point."""
        m, n = self.cost.shape
        u, lag, xs = self.u, float(self.u.sum()), np.zeros((m, n), dtype=bool)
        for k in range(m):
            val, xs[k] = knapsack(u - self.cost[k], self.res[k], int(self.cap[k]))
            lag -= val
        self.recent.append(xs)
        self.steps += 1
        if lag > self.best + 1e-9:
            self.best, self.best_u, self.stall = lag, u.copy(), 0
        else:
            self.stall += 1
            if self.stall >= 10:
                self.factor, self.stall = 0.7 * self.factor, 0
        self.xbar = xs.astype(float) if self.xbar is None else 0.1 * xs + 0.9 * self.xbar
        d = 1.0 - self.xbar.sum(axis=0)
        gap = max(target - self.best, 1e-6 * max(1.0, abs(self.best)))
        self.u = self.best_u + self.factor * gap / max(float(d @ d), 1e-12) * d
        return lag, u


class GAPColumnGenerationXL:
    """Engineered CG / Lagrangian reference; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, smoothing: float = 0.5, verbose: bool = False, on_incumbent=None, on_bound=None, lp_method: int = 0,
                 soft_mem_gb: float | None = None, lp_share: float = 0.25, lp_max_jobs: int = 1000, lns_jobs: int = 100, seed: int = 0) -> None:
        self.soft_mem_gb = soft_mem_gb  # Gurobi SoftMemLimit for graceful termination before the job's memory cap
        self.lp_method = lp_method  # Gurobi Method for the restricted master LP
        self.on_incumbent = on_incumbent or (lambda assign: None)  # receives the job -> agent list
        self.on_bound = on_bound or (lambda bound: None)  # receives every valid lower bound
        self.time_limit = time_limit
        self.threads = threads
        self.smoothing = smoothing
        self.lp_share = lp_share  # cap on the root LP phase, as a fraction of the time limit
        # Above this many jobs the full master LP is not run: measured at 1600 jobs, its re-solves take seconds of
        # degenerate pivots and its value does not move in 30 iterations (the optimal basis needs ~n patterns).
        self.lp_max_jobs = lp_max_jobs
        self.lns_jobs = lns_jobs  # target number of jobs in a price-and-branch neighborhood
        self.seed = seed
        self.verbose = verbose
        self.trace: list[dict[str, Any]] = []

    def _log(self, t0: float, inc: float | None, bnd: float | None, stage: str) -> None:
        if bnd is not None and math.isfinite(bnd):
            self.on_bound(bnd)
        self.trace.append({"t": time.perf_counter() - t0, "incumbent": inc, "bound": bnd, "stage": stage})

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        T = self.time_limit
        deadline = t0 + T
        cost, res, cap = arrays(GAPInstance.from_json_file(data_path))
        m, n = cost.shape
        integral = bool(np.all(cost == np.round(cost)))
        rng = np.random.default_rng(self.seed)
        self.state = {"bound": -math.inf, "center": None, "iters": 0, "lp_seconds": 0.0, "pricing_seconds": 0.0, "volume_steps": 0,
                      "heuristic_rounds": 0, "lns_rounds": 0, "lns_improvements": 0, "lp_phase": "skipped"}
        best = {"obj": None, "assign": None}

        def now() -> float:
            return time.perf_counter()

        def closed() -> bool:
            return best["obj"] is not None and integral and best["obj"] - self.state["bound"] < 1 - 1e-9

        def new_bound(value: float, stage: str) -> None:
            if value > self.state["bound"]:
                self.state["bound"] = value
                self._log(t0, best["obj"], value, stage)

        def offer(assign: np.ndarray, stage: str) -> None:
            obj = evaluate(assign, cost, res, cap)  # recomputed from the assignment
            if obj is not None and (best["obj"] is None or obj < best["obj"] - 1e-9):
                best["obj"], best["assign"] = obj, assign.copy()
                self.on_incumbent(assign.tolist())
                self._log(t0, obj, self.state["bound"], stage)

        hard = np.full(m, 2.0 * float(cost.max() - cost.min()) + 1.0)  # no move may trade a unit of overload for cost

        def polish(assign: np.ndarray, stage: str) -> None:
            assign, _ = descend(assign, cost, res, cap, hard, deadline)
            offer(assign, stage)

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 1 if self.verbose else 0)
        env.setParam("Threads", self.threads)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()

        # 1. Capacity Lagrangian: LP-quality bound and multipliers within seconds.
        mu = capacity_lagrangian(cost, res, cap, env, min(deadline, t0 + max(1.0, min(10.0, 0.05 * T))),
                                 lambda g, terms: new_bound(certified_bound(g, terms, integral), "capacity_lagrangian"))
        u0 = (cost + mu[:, None] * res).min(axis=0)
        w0 = 2.0 * np.maximum(mu, 0.1 * max(float(mu.max()), 1e-3))  # initial overload weights ~ capacity prices

        # 2. Primal start (the search stops by patience once feasible; the long limit only covers reaching feasibility).
        search_time = max(2.0, min(30.0, 0.05 * T))
        start = regret_assign(cost + mu[:, None] * res, res, cap)
        penalty_search(start, cost, res, cap, w0, min(deadline, now() + max(search_time, 0.2 * T)), lambda a: offer(a, "construct"))
        self._log(t0, best["obj"], self.state["bound"], "start")

        # 3. Lagrangian master (volume) alternating with Lagrangian heuristics.
        vol = Volume(cost, res, cap, u0, keep=40)

        def volume_steps(until: float) -> None:
            started = now()
            while now() < min(until, deadline) and not vol.converged:
                base = max(vol.best, self.state["bound"])
                target = best["obj"] if best["obj"] is not None else base + 1e-3 * abs(base) + 1.0
                lag, u = vol.step(target)
                new_bound(certified_bound(lag, u, integral), "volume")
                if closed():
                    break
            self.state["pricing_seconds"] += now() - started
            self.state["volume_steps"] = vol.steps

        def heuristic_round(until: float) -> None:
            """Construction on Lagrangian costs from the current duals (or perturbed capacity prices), then search."""
            r = self.state["heuristic_rounds"]
            self.state["heuristic_rounds"] += 1
            prices = knapsack_ratios(vol.best_u, cost, res, cap) if r % 2 == 0 else mu * rng.uniform(0.97, 1.03, m)
            a = regret_assign(cost + prices[:, None] * res, res, cap)
            penalty_search(a, cost, res, cap, w0, min(until, deadline), lambda x: offer(x, "lagrangian_heuristic"))

        lagcost = cost + mu[:, None] * res
        idx = np.arange(n)

        def lns_round(until: float) -> None:
            """Re-optimize a set of free jobs over all agents with residual capacities by price-and-branch (sub_gap).
            The set mixes jobs with a large Lagrangian reduced cost, jobs on the agents those jobs prefer, and random
            jobs. (Agent-subset neighborhoods were tried: the penalized search leaves them already optimal.)"""
            self.state["lns_rounds"] += 1
            a = best["assign"]
            reduced = lagcost[a, idx] - lagcost.min(axis=0)
            size = min(n, self.lns_jobs)
            p = reduced + 0.1 * float(reduced.mean()) + 1e-9
            bad = rng.choice(n, size=size // 3, replace=False, p=p / p.sum())
            near = np.nonzero(np.isin(a, lagcost[:, bad].argmin(axis=0)))[0]
            jobs = np.unique(np.concatenate([bad, rng.choice(near, size=min(len(near), size // 3), replace=False),
                                             rng.choice(n, size=size - 2 * (size // 3), replace=False)]))
            fixed = np.ones(n, dtype=bool)
            fixed[jobs] = False
            residual = cap - np.bincount(a[fixed], weights=res[a[fixed], idx[fixed]], minlength=m).astype(np.int64)
            sub = sub_gap(cost[:, jobs], res[:, jobs], residual, a[jobs], vol.best_u[jobs], env, min(until, deadline), integral)
            if sub is not None:
                candidate = a.copy()
                candidate[jobs] = sub
                before = best["obj"]
                polish(candidate, "lns")
                self.state["lns_improvements"] += best["obj"] < before

        operators = {"lns": [0.0, 0.0], "construct": [0.0, 0.0]}  # objective gain, seconds

        def improvement_slice(until: float) -> None:
            """One slice of LNS rounds or one fresh construction, chosen in proportion to each operator's measured
            gain per second (prior: one unit per second). Neighborhood price-and-branch pays off on type E-like
            instances; on tightly packed type D instances fresh constructions are the ones that improve."""
            if best["assign"] is None:
                heuristic_round(until)
                return
            rate = {k: (g + 1.0) / (s + 1.0) for k, (g, s) in operators.items()}
            choice = "lns" if rng.random() < rate["lns"] / (rate["lns"] + rate["construct"]) else "construct"
            before, started = best["obj"], now()
            if choice == "construct":
                heuristic_round(until)
            while choice == "lns" and now() < min(until, deadline) and not closed():
                lns_round(min(until, now() + max(2.0, 0.01 * T)))
            operators[choice][0] += before - best["obj"]
            operators[choice][1] += now() - started

        def lagrangian_cycle(until: float, while_volume: bool) -> None:
            """Alternate volume steps and improvement slices; with while_volume, stop once the volume converged."""
            while now() < min(until, deadline) and not closed():
                if vol.converged and while_volume:
                    break
                if not vol.converged:
                    volume_steps(min(until, now() + search_time))
                if now() < min(until, deadline) and not closed():
                    improvement_slice(min(until, now() + search_time))

        lagrangian_cycle(t0 + 0.25 * T, while_volume=True)
        warm_seconds = now() - t0

        # 4. LP master: root CG from the Lagrangian center, seeded with the latest pricing patterns.
        rmp = gp.Model(env=env)
        rmp.Params.Method = self.lp_method
        # Artificial cost: large enough to be driven out at the LP optimum of GAP-type instances, small enough
        # to keep early duals informative. Lower bounds never depend on it (they come from Lagrangian pricing).
        big = 10.0 * float(cost.max()) + 1.0
        art = [rmp.addVar(obj=big, name=f"art{i}") for i in range(n)]  # keeps the master feasible
        # Covering rows: with nonnegative costs a job can always be dropped from a pattern, so the covering LP
        # has the partitioning LP value, and its duals (u >= 0) are less degenerate.
        cover = [rmp.addLConstr(gp.LinExpr([(1.0, art[i])]), GRB.GREATER_EQUAL, 1.0) for i in range(n)]
        conv = [rmp.addLConstr(gp.LinExpr(), GRB.LESS_EQUAL, 1.0) for _ in range(m)]
        columns: list[tuple[int, np.ndarray]] = []
        seen: set[tuple[int, bytes]] = set()
        lam: list[gp.Var] = []

        def add_column(k: int, x: np.ndarray) -> bool:
            key = (k, np.packbits(x).tobytes())
            if key in seen or not x.any():
                return False
            seen.add(key)
            jobs = np.nonzero(x)[0]
            col = gp.Column([1.0] * len(jobs) + [1.0], [cover[int(i)] for i in jobs] + [conv[k]])
            lam.append(rmp.addVar(obj=float(cost[k, jobs].sum()), column=col))
            columns.append((k, x.copy()))
            return True

        if n <= self.lp_max_jobs:
            if best["assign"] is not None:
                for k in range(m):
                    add_column(k, best["assign"] == k)
            for xs in vol.recent:
                for k in range(m):
                    add_column(k, xs[k])
        self.state["center"] = vol.best_u.copy()

        def column_generation(blocked_agents: set[int], covered: np.ndarray, root: bool, max_iters: int, stop_at: float) -> bool:
            """Run CG at the current node; returns True only if the master LP is solved for the current columns.

            At the root, every pricing round also yields a Lagrangian lower bound (smoothed duals), and the phase
            stops early when the LP gap (master value minus bound) does not shrink by 5% over 30 iterations.
            Inside a dive, blocked agents are not priced and covered jobs are excluded.
            """
            history = []
            for _ in range(max_iters):
                if now() >= min(stop_at, deadline):
                    return False
                self.state["iters"] += 1
                rmp.Params.TimeLimit = max(0.01, min(stop_at, deadline) - now())
                lp_started = now()
                rmp.optimize()
                self.state["lp_seconds"] += now() - lp_started
                if rmp.Status != GRB.OPTIMAL:
                    return False
                pricing_started = now()
                u_true = np.array(rmp.getAttr("Pi", cover))
                v_true = np.array(rmp.getAttr("Pi", conv))
                center = self.state["center"] if root else None
                for u in ([self.smoothing * center + (1 - self.smoothing) * u_true] if center is not None else []) + [u_true]:
                    lag, added = u.sum(), 0
                    for k in range(m):
                        if k in blocked_agents:
                            continue
                        profit = u - cost[k]
                        profit[covered] = -1.0
                        val, x = knapsack(profit, res[k], int(cap[k]))
                        lag -= val  # min_x sum (c - u) x = -max_x sum (u - c) x
                        rc = float((cost[k] - u_true)[x].sum()) - v_true[k]
                        if rc < -1e-7 and add_column(k, x):
                            added += 1
                    if root:
                        bound = certified_bound(lag, u, integral)
                        if bound > self.state["bound"]:
                            self.state["center"] = u.copy()
                        new_bound(bound, "cg_root")
                    if added:
                        break
                self.state["pricing_seconds"] += now() - pricing_started
                if not added:
                    return True
                if root:
                    history.append(rmp.ObjVal - self.state["bound"])
                    if len(history) > 30 and history[-1] > 0.95 * history[-31]:
                        self.state["lp_phase"] = "stalled"
                        return False
            # Iteration limit: the last round added columns, so re-solve before exposing solution values.
            if now() >= min(stop_at, deadline):
                return False
            rmp.Params.TimeLimit = max(0.01, min(stop_at, deadline) - now())
            rmp.optimize()
            return rmp.Status == GRB.OPTIMAL

        def dive(randomize: bool) -> None:
            """Fix one master column per node, removing its agent and jobs, and re-run CG."""
            blocked, covered, changed = set(), np.zeros(n, dtype=bool), []
            try:
                dive_steps(blocked, covered, changed, randomize)
            finally:
                for v in changed:
                    v.LB, v.UB = 0.0, 1.0
                rmp.update()

        def dive_steps(blocked, covered, changed, randomize):
            while now() < deadline:
                if not column_generation(blocked, covered, False, 200, deadline):
                    break
                values = np.array([v.X for v in lam])
                if np.array(rmp.getAttr("X", art)).max() > 1e-6:
                    break  # infeasible node: the dive failed
                frac = [(val, j) for j, val in enumerate(values) if val > 1e-6 and columns[j][0] not in blocked]
                if not frac:
                    break
                if all(val > 1 - 1e-6 for val, _ in frac):
                    assign = np.full(n, -1)
                    for j, (k, x) in enumerate(columns):
                        if values[j] > 0.5:
                            assign[x] = k
                    if np.all(assign >= 0):
                        polish(assign, "dive")
                    break
                frac.sort(reverse=True)
                pick = frac[int(rng.integers(0, min(3, len(frac))))][1] if randomize else frac[0][1]
                k, x = columns[pick]
                lam[pick].LB = 1.0
                changed.append(lam[pick])
                blocked.add(k)
                covered |= x
                # Columns that overlap the fixed jobs, or belong to the fixed agent, are excluded.
                for j, (k2, x2) in enumerate(columns):
                    if j != pick and lam[j].UB > 0 and (k2 == k or (x2 & x).any()):
                        lam[j].UB = 0.0
                        changed.append(lam[j])

        root_done = False
        if n > self.lp_max_jobs:
            self.state["lp_phase"] = "skipped_large"
        elif now() < deadline and not closed():
            self.state["lp_phase"] = "running"
            root_done = column_generation(set(), np.zeros(n, dtype=bool), True, 10**9, min(deadline, now() + self.lp_share * T))
            if root_done:
                self.state["lp_phase"] = "converged"
            elif self.state["lp_phase"] == "running":
                self.state["lp_phase"] = "time_share"
        root_seconds = now() - t0

        # 5. Price-and-branch when the root LP converged (dives and restricted-master MIPs, alternating with
        # neighborhood price-and-branch); otherwise the Lagrangian master and neighborhood price-and-branch continue.
        dives = 0
        mip_limit = 10.0  # restricted-master MIPs rarely beat the dives here; keep most of the time for dives
        while now() < deadline and not closed():
            if root_done:
                dive(randomize=dives > 0)
                dives += 1
                cutoff = best["obj"] - (0.5 if integral else 1e-6) if best["obj"] is not None else None
                self._restricted_mip(rmp, columns, n, min(mip_limit, 0.1 * (deadline - now())), cutoff, offer, polish)
                mip_limit = min(300.0, 2 * mip_limit)
                improvement_slice(now() + search_time)
            else:
                lagrangian_cycle(deadline, while_volume=False)
        env.dispose()
        best_obj, best_assign, best_bound = best["obj"], best["assign"], self.state["bound"]
        gap = (best_obj - best_bound) / abs(best_obj) if best_obj else None
        self._log(t0, best_obj, best_bound, "final")
        return {
            "best_objective": best_obj, "global_lb": best_bound, "global_ub": best_obj, "gap": gap,
            "termination_reason": "optimal" if closed() or (gap is not None and gap <= 1e-9) else "time_limit",
            "root_converged": root_done, "lp_phase": self.state["lp_phase"], "warm_start_seconds": warm_seconds, "root_seconds": root_seconds,
            "dives": dives, "iters": self.state["iters"], "columns": len(columns), "volume_steps": self.state["volume_steps"],
            "heuristic_rounds": self.state["heuristic_rounds"], "lns_rounds": self.state["lns_rounds"], "lns_improvements": self.state["lns_improvements"],
            "lp_seconds": self.state["lp_seconds"], "pricing_seconds": self.state["pricing_seconds"],
            "assignment": best_assign.tolist() if best_assign is not None else None,
            "total_seconds": time.perf_counter() - t0, "trace": self.trace,
        }

    def _restricted_mip(self, rmp, columns, n, limit, cutoff, offer, polish):
        if limit < 1.0:
            return
        rmp.update()  # copy() sees only applied changes, e.g. bounds restored after a dive
        mip = rmp.copy()
        mvars = mip.getVars()
        for v in mvars[n:]:
            v.VType = GRB.BINARY
        for v in mvars[:n]:
            v.UB = 0.0
        if cutoff is not None:
            mip.Params.Cutoff = cutoff
        mip.Params.TimeLimit = limit
        cols = mvars[n:]

        def decode(values) -> np.ndarray:
            assign = np.full(n, -1)
            for v, (k, x) in zip(values, columns):
                if v > 0.5:
                    assign[x] = k
            return assign

        def publish(cb, where):
            # Publish every restricted-master solution when found, not only after the solve.
            if where == GRB.Callback.MIPSOL:
                assign = decode(cb.cbGetSolution(cols))
                if np.all(assign >= 0):
                    offer(assign, "restricted_mip")

        mip.optimize(publish)
        if mip.SolCount:
            assign = decode([v.X for v in cols])
            if np.all(assign >= 0):
                polish(assign, "restricted_mip")
        mip.dispose()
