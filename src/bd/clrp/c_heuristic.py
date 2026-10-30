"""CLRP heuristic control (no decomposition): depot-set search around savings-based routing.

For a set of open depots, customers are assigned to depots by regret on travel distance within depot
capacities, each depot's customers are routed with Clarke-Wright savings, and routes are improved by
2-opt and relocate moves between routes of the same depot. An outer search adds, drops and swaps
depots (with a fixed seed), keeping the best verified plan.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from src.bd.clrp.a_data import CLRPInstance
from src.bd.clrp.b_mip_xl import route_cost


def _route_len(seq: list[int], c: np.ndarray) -> float:
    return float(sum(c[a, b] for a, b in zip(seq, seq[1:])))


def savings(depot: int, custs: list[int], c: np.ndarray, q: np.ndarray, Q: float, m: int, rng=None) -> list[list[int]]:
    """Clarke-Wright parallel savings; customers are node indices (>= m), q indexed by node - m.
    With `rng`, each saving is multiplied by a factor in [0.8, 1.2] (randomized savings)."""
    routes = {i: [i] for i in custs}
    where = {i: i for i in custs}
    load = {i: q[i - m] for i in custs}
    noise = (lambda: rng.uniform(0.8, 1.2)) if rng is not None else (lambda: 1.0)
    pairs = sorted(((noise() * (c[depot, i] + c[depot, j] - c[i, j]), i, j) for a, i in enumerate(custs) for j in custs[a + 1:]), reverse=True)
    for s, i, j in pairs:
        if s <= 0:
            break
        ri, rj = where[i], where[j]
        if ri == rj or load[ri] + load[rj] > Q:
            continue
        a, b = routes[ri], routes[rj]
        # Join only at route ends.
        if a[-1] == i and b[0] == j:
            new = a + b
        elif a[0] == i and b[-1] == j:
            new = b + a
        elif a[-1] == i and b[-1] == j:
            new = a + b[::-1]
        elif a[0] == i and b[0] == j:
            new = a[::-1] + b
        else:
            continue
        routes[ri] = new
        load[ri] += load[rj]
        del routes[rj], load[rj]
        for k in new:
            where[k] = ri
    return list(routes.values())


def two_opt(depot: int, r: list[int], c: np.ndarray) -> list[int]:
    seq = [depot] + r + [depot]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(seq) - 2):
            for j in range(i + 1, len(seq) - 1):
                if c[seq[i - 1], seq[j]] + c[seq[i], seq[j + 1]] < c[seq[i - 1], seq[i]] + c[seq[j], seq[j + 1]] - 1e-9:
                    seq[i:j + 1] = seq[i:j + 1][::-1]
                    improved = True
    return seq[1:-1]


def relocate(depot: int, routes: list[list[int]], c: np.ndarray, q: np.ndarray, Q: float, m: int) -> list[list[int]]:
    """First-improvement moves of one customer to the best position in another route of the same depot."""
    improved = True
    while improved:
        improved = False
        for a in range(len(routes)):
            for pos, v in enumerate(routes[a]):
                sa = [depot] + routes[a] + [depot]
                gain = c[sa[pos], v] + c[v, sa[pos + 2]] - c[sa[pos], sa[pos + 2]]
                for b in range(len(routes)):
                    if b == a or sum(q[k - m] for k in routes[b]) + q[v - m] > Q:
                        continue
                    sb = [depot] + routes[b] + [depot]
                    ins = [c[sb[k], v] + c[v, sb[k + 1]] - c[sb[k], sb[k + 1]] for k in range(len(sb) - 1)]
                    k = int(np.argmin(ins))
                    if ins[k] < gain - 1e-9:
                        routes[b].insert(k, v)
                        routes[a].pop(pos)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        routes = [r for r in routes if r]
    return routes


def plan_for(inst: CLRPInstance, c: np.ndarray, open_: list[int], rng=None) -> list[dict[str, Any]] | None:
    m, n, q = inst.m, inst.n, inst.demand
    left = {d: float(inst.depot_capacity[d]) for d in open_}
    groups: dict[int, list[int]] = {d: [] for d in open_}
    todo = list(range(m, m + n))
    while todo:
        best = None
        for i in todo:
            feas = sorted((c[d, i], d) for d in open_ if left[d] >= q[i - m])
            if not feas:
                return None
            regret = feas[1][0] - feas[0][0] if len(feas) > 1 else float("inf")
            if best is None or regret > best[0]:
                best = (regret, i, feas[0][1])
        _, i, d = best
        groups[d].append(i)
        left[d] -= q[i - m]
        todo.remove(i)
    routes = []
    for d, custs in groups.items():
        rs = [two_opt(d, r, c) for r in savings(d, custs, c, q, inst.vehicle_capacity, m, rng)]
        rs = [two_opt(d, r, c) for r in relocate(d, rs, c, q, inst.vehicle_capacity, m)]
        routes += [{"depot": d, "customers": [k - m for k in r]} for r in rs if r]
    return routes


def depot_search(inst: CLRPInstance, c: np.ndarray, deadline: float, seed: int, on_incumbent: Callable[[list], None]) -> tuple[float, list]:
    rng = np.random.default_rng(seed)
    m = inst.m
    cache: dict[tuple[int, ...], float] = {}

    def evaluate(open_: tuple[int, ...]):
        if open_ in cache or inst.depot_capacity[list(open_)].sum() < inst.demand.sum():
            return None
        plan = plan_for(inst, c, list(open_))
        cost = route_cost(inst, c, plan) if plan else None
        cache[open_] = cost if cost is not None else float("inf")
        return (cost, plan) if cost is not None else None

    # Start: open depots by opening cost per unit capacity until capacity covers demand.
    order = sorted(range(m), key=lambda d: inst.opening_cost[d] / inst.depot_capacity[d])
    start, cap = [], 0.0
    for d in order:
        start.append(d)
        cap += inst.depot_capacity[d]
        if cap >= 1.0 * inst.demand.sum():
            break
    best_open = tuple(sorted(start))
    res = evaluate(best_open)
    best_cost, best_plan = res if res else (float("inf"), None)
    if best_plan:
        on_incumbent(best_plan)
    while time.perf_counter() < deadline:
        moves = [tuple(sorted(set(best_open) | {d})) for d in range(m) if d not in best_open]
        moves += [tuple(sorted(set(best_open) - {d})) for d in best_open if len(best_open) > 1]
        moves += [tuple(sorted((set(best_open) - {a}) | {b})) for a in best_open for b in range(m) if b not in best_open]
        rng.shuffle(moves)
        improved = False
        for mv in moves:
            if time.perf_counter() >= deadline:
                break
            res = evaluate(mv)
            if res and res[0] < best_cost - 1e-9:
                best_cost, best_plan, best_open = res[0], res[1], mv
                on_incumbent(best_plan)
                improved = True
                break
        if not improved:
            # Iterated search: perturb the depot set (flip 1-3 depots) and route with randomized savings.
            while time.perf_counter() < deadline:
                cand = set(best_open)
                for d in rng.choice(m, size=int(rng.integers(1, min(3, m) + 1)), replace=False):
                    cand ^= {int(d)}
                cand = tuple(sorted(cand))
                if not cand or inst.depot_capacity[list(cand)].sum() < inst.demand.sum():
                    continue
                plan = plan_for(inst, c, list(cand), rng)
                cost = route_cost(inst, c, plan) if plan else None
                if cost is not None and cost < best_cost - 1e-9:
                    best_cost, best_plan, best_open = cost, plan, cand
                    on_incumbent(best_plan)
                    break
    return best_cost, best_plan
