"""Bounded minimum-cost exact-cover search for fixed-depot route recourse.

Whole routes partition customers, subject to per-depot fleet limits. Exhaustion
certifies an optimum or infeasibility; interruption retains only a safe bound.
"""

from __future__ import annotations

import math
import time

import numpy as np


def solve_exact_cover(data, y, seconds, alpha=None, beta=None, on_incumbent=None):
    start = time.monotonic()
    deadline = start + max(0.0, seconds)
    n, m = data["n"], data["m"]
    routes, costs, depots = data["routes"], data["route_cost"], data["route_depot"]
    if len(y) != m or any(v not in (0, 1) for v in y):
        raise ValueError("Fixed depot decisions must be binary")
    fleet = [int(v * o) for v, o in zip(data["max_vehicles"], y, strict=True)]
    k = max(map(len, routes))
    a = np.zeros(n) if alpha is None else np.asarray(alpha, dtype=float).copy()
    b = np.zeros(m) if beta is None else np.asarray(beta, dtype=float).copy()
    if a.shape != (n,) or b.shape != (m,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid dual dimensions or values")
    # Repair floating-point dual infeasibility conservatively: beta <= 0 and
    # sum(alpha over route) + beta[depot] <= route cost for every nonempty route.
    b = np.minimum(b, 0.0)
    violation = max(
        0.0, max(sum(a[i] for i in route) + b[j] - c for route, j, c in zip(routes, depots, costs, strict=True))
    )
    a -= violation + 1e-10
    route_alpha = [float(sum(a[i] for i in route)) for route in routes]
    reduced = [float(c - ar - b[j]) for c, ar, j in zip(costs, route_alpha, depots, strict=True)]
    root_bound = float(a.sum() + b @ fleet)
    guard = 1e-9 * max(1.0, float(abs(a).sum() + abs(b) @ fleet), n * max(costs))
    fallback_bound = max(0.0, root_bound - guard)
    by_customer, by_depot = [0] * n, [0] * m
    masks = [sum(1 << i for i in route) for route in routes]
    min_length = n - (sum(fleet) - 1) * k
    active = 0
    active_ids = []
    for r, (route, j) in enumerate(zip(routes, depots, strict=True)):
        if fleet[j] > 0 and len(route) >= min_length:
            bit = 1 << r
            active |= bit
            active_ids.append(r)
            by_depot[j] |= bit
            for i in route:
                by_customer[i] |= bit
    conflicts = {}
    for r in active_ids:
        value = 0
        for i in routes[r]:
            value |= by_customer[i]
        conflicts[r] = value
    nodes, bound_prunes = 0, 0
    best_cost, best_routes = math.inf, None

    def visit(remaining, available, remaining_fleet, slots, chosen, cost, bound):
        nonlocal nodes, bound_prunes, best_cost, best_routes
        nodes += 1
        if nodes % 128 == 0 and time.monotonic() >= deadline:
            raise TimeoutError
        if not remaining:
            if cost < best_cost:
                best_cost, best_routes = cost, chosen.copy()
                if on_incumbent:
                    on_incumbent(best_routes, best_cost)
                if time.monotonic() >= deadline:
                    raise TimeoutError
            return
        if max(cost, bound) - guard >= best_cost:
            bound_prunes += 1
            return
        customers = remaining.bit_count()
        if customers > slots * k:
            return
        candidates, count, rest = 0, len(routes) + 1, remaining
        while rest:
            bit = rest & -rest
            rest ^= bit
            choices = available & by_customer[bit.bit_length() - 1]
            number = choices.bit_count()
            if not number:
                return
            if number < count:
                count, candidates = number, choices
        ordered = []
        while candidates:
            bit = candidates & -candidates
            candidates ^= bit
            r = bit.bit_length() - 1
            if customers - len(routes[r]) <= (slots - 1) * k:
                ordered.append(r)
        # Low reduced-cost rows tend to produce good incumbents early.
        ordered.sort(key=lambda r: (reduced[r], r))
        for r in ordered:
            j = depots[r]
            next_fleet = remaining_fleet.copy()
            next_fleet[j] -= 1
            next_available = available & ~conflicts[r]
            if next_fleet[j] == 0:
                next_available &= ~by_depot[j]
            visit(
                remaining ^ masks[r],
                next_available,
                next_fleet,
                slots - 1,
                [*chosen, r],
                cost + costs[r],
                bound + reduced[r],
            )

    exhausted = False
    try:
        if time.monotonic() >= deadline:
            raise TimeoutError
        visit((1 << n) - 1, active, fleet, sum(fleet), [], 0.0, root_bound)
        exhausted = True
    except TimeoutError:
        pass
    lower = max(0.0, best_cost - guard) if exhausted and best_routes is not None else fallback_bound
    return {
        "status": "optimal" if exhausted and best_routes is not None else "infeasible" if exhausted else "time_limit",
        "routes": best_routes,
        "objective": best_cost if best_routes is not None else None,
        "lower_bound": lower,
        "nodes": nodes,
        "bound_prunes": bound_prunes,
        "exhausted": exhausted,
        "seconds": time.monotonic() - start,
        "numerical_guard": guard,
        "repaired_dual_bound": root_bound,
    }
