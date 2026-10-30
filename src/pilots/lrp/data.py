"""Deterministic route-pool instances and independent, solver-free checks."""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np


def shortest_route(depot, customers, coordinates):
    """Held-Karp on a small customer subset; exact Euclidean closed route."""
    customers = tuple(sorted(customers))
    points = np.asarray([depot, *[coordinates[i] for i in customers]], dtype=float)
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    states = {(1 << j, j): (float(distances[0, j + 1]), (j,)) for j in range(len(customers))}
    for mask in range(1, 1 << len(customers)):
        for last in range(len(customers)):
            if (mask, last) not in states:
                continue
            cost, order = states[mask, last]
            for nxt in range(len(customers)):
                if mask & (1 << nxt):
                    continue
                key = (mask | (1 << nxt), nxt)
                candidate = (cost + float(distances[last + 1, nxt + 1]), (*order, nxt))
                if key not in states or candidate < states[key]:
                    states[key] = candidate
    full = (1 << len(customers)) - 1
    cost, order = min(
        (states[full, j][0] + float(distances[j + 1, 0]), states[full, j][1]) for j in range(len(customers))
    )
    return cost, [customers[i] for i in order]


def generate(spec, seed):
    """Sample by route length, preserving singletons and a feasible partition.

    Full pools contain a cheapest tour for EVERY customer subset up to max_stops.
    With no route-dependent restrictions this retains every nondominated coverage
    choice in the stated stop-limited Euclidean model. Sampled pools are explicitly
    restricted-pool problems, not unrestricted LRP benchmarks.
    """
    m, n, k = (spec[key] for key in ("depots", "customers", "max_stops"))
    cap = spec["routes_per_depot"]
    for name, value in [("depots", m), ("customers", n), ("max_stops", k), ("routes_per_depot", cap)]:
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not 1 <= k <= min(n, 8) or m > 64 or n > 200 or cap > 10000:
        raise ValueError("Pilot limits: max_stops<=8, depots<=64, customers<=200, routes/depot<=10000")
    scale = float(spec.get("opening_cost_scale", 1.0))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("opening_cost_scale must be positive and finite")
    rng = np.random.default_rng(seed)
    xc, xd = rng.random((n, 2)), rng.uniform(-0.2, 1.2, (m, 2))
    opening = rng.integers(12, 35, m).astype(float) * scale
    order = rng.permutation(n).tolist()
    witness_subsets = [[] for _ in range(m)]
    for index, start in enumerate(range(0, n, k)):
        witness_subsets[index % m].append(tuple(sorted(order[start : start + k])))
    singletons = {(i,) for i in range(n)}
    if cap < max(len(singletons | set(subsets)) for subsets in witness_subsets):
        raise ValueError("Route cap must accommodate singletons and witness routes")
    # Some spare fleet is essential: a fully tight total fleet can force every
    # depot open and accidentally remove the location decision from the pilot.
    fleet = spec.get("vehicles_per_depot", 2 * math.ceil(n / (k * m)))
    if type(fleet) is not int or fleet < max(map(len, witness_subsets)):
        raise ValueError("Fleet is too small for the constructed witness")
    possible = sum(math.comb(n, size) for size in range(1, k + 1))
    complete = possible <= cap
    routes, depots, costs, witness_indices = [], [], [], []
    for depot in range(m):
        if complete:
            pool = {subset for size in range(1, k + 1) for subset in combinations(range(n), size)}
        else:
            pool = {(i,) for i in range(n)} | set(witness_subsets[depot])
            # Round-robin sizes avoid filling the cap with short routes first.
            sizes = [s for s in range(2, k + 1)]
            attempts = 0
            while len(pool) < cap and attempts < cap * 100:
                size = sizes[attempts % len(sizes)]
                pool.add(tuple(sorted(rng.choice(n, size, replace=False).tolist())))
                attempts += 1
            if len(pool) != cap:
                raise ValueError("Could not construct requested distinct route pool within generation budget")
        for subset in sorted(pool, key=lambda s: (len(s), s)):
            cost, sequence = shortest_route(xd[depot], subset, xc)
            if subset in witness_subsets[depot]:
                witness_indices.append(len(routes))
            routes.append(sequence)
            depots.append(depot)
            costs.append(cost)
    incidence = np.zeros((n, len(routes)), dtype=int)
    for r, sequence in enumerate(routes):
        incidence[sequence, r] = 1
    data = {
        "m": m,
        "n": n,
        "max_stops": k,
        "max_vehicles": [fleet] * m,
        "depot_cost": opening.tolist(),
        "Xd": xd.tolist(),
        "Xc": xc.tolist(),
        "routes": routes,
        "route_depot": depots,
        "route_cost": costs,
        "incidence": incidence.tolist(),
    }
    lam = [int(r in witness_indices) for r in range(len(routes))]
    y = [int(bool(s)) for s in witness_subsets]
    witness = {"y": y, "lambda": lam, "objective_value": float(opening @ y + np.asarray(costs) @ lam)}
    validate_data(data)
    assert verify(data, witness)["ok"]
    metadata = {
        "generator": "lrp-pilot-v1",
        "seed": seed,
        "specification": spec,
        "route_pool": "complete_stop_limited" if complete else "sampled_restricted",
        "possible_subsets_per_depot": possible,
        "routes": len(routes),
        "route_length_counts": {str(s): sum(len(r) == s for r in routes) for s in range(1, k + 1)},
        "witness": witness,
    }
    return data, metadata


def validate_data(data):
    """Reject malformed route/coverage data before handing it to a solver."""
    m, n, routes = data["m"], data["n"], data["routes"]
    if type(m) is not int or type(n) is not int or min(m, n) < 1 or not routes:
        raise ValueError("Invalid dimensions or empty route pool")
    for key, length in [("depot_cost", m), ("max_vehicles", m), ("route_cost", len(routes))]:
        array = np.asarray(data[key], dtype=float)
        if array.shape != (length,) or not np.isfinite(array).all() or (array < 0).any():
            raise ValueError(f"Invalid {key}")
    if any(float(v) != int(v) for v in data["max_vehicles"]):
        raise ValueError("Fleet counts must be integers")
    if len(data["route_depot"]) != len(routes):
        raise ValueError("Route/depot dimension mismatch")
    expected = np.zeros((n, len(routes)), dtype=int)
    for r, route in enumerate(routes):
        depot = data["route_depot"][r]
        if type(depot) is not int or not 0 <= depot < m:
            raise ValueError("Invalid depot index")
        if not route or len(route) > data["max_stops"] or len(route) != len(set(route)):
            raise ValueError("Invalid route length or repeated customer")
        if any(type(i) is not int or not 0 <= i < n for i in route):
            raise ValueError("Invalid customer index")
        expected[route, r] = 1
    if np.asarray(data["incidence"]).shape != expected.shape or not np.array_equal(expected, data["incidence"]):
        raise ValueError("Incidence disagrees with explicit routes")


def verify(data, output, tolerance=1e-6):
    """Recount customers/fleet from explicit routes, independently of solver matrices."""
    issues = []
    try:
        y = np.asarray(output["y"], dtype=float)
        lam = np.asarray(output["lambda"], dtype=float)
        objective = float(output["objective_value"])
        if y.shape != (data["m"],) or lam.shape != (len(data["routes"]),):
            raise ValueError("Decision dimensions mismatch")
        if not np.isfinite(y).all() or not np.isfinite(lam).all() or not math.isfinite(objective):
            raise ValueError("Nonfinite output")
        for name, values in [("y", y), ("lambda", lam)]:
            if ((values < -tolerance) | (values > 1 + tolerance)).any() or (
                abs(values - np.rint(values)) > tolerance
            ).any():
                issues.append(f"{name} must be binary")
        cover, fleet = np.zeros(data["n"]), np.zeros(data["m"])
        for r, sequence in enumerate(data["routes"]):
            for customer in sequence:
                cover[customer] += lam[r]
            fleet[data["route_depot"][r]] += lam[r]
        if (abs(cover - 1) > tolerance).any():
            issues.append("Every customer must be covered exactly once")
        if (fleet - np.asarray(data["max_vehicles"]) * y > tolerance).any():
            issues.append("Fleet or closed-depot constraint violated")
        if sum(y) < 1 - tolerance:
            issues.append("No open depot")
        recomputed = float(np.asarray(data["depot_cost"]) @ y + np.asarray(data["route_cost"]) @ lam)
        if not math.isclose(recomputed, objective, rel_tol=tolerance, abs_tol=tolerance):
            issues.append("Reported objective disagrees with decisions")
        return {"ok": not issues, "issues": issues, "recomputed_objective": recomputed}
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return {"ok": False, "issues": [str(exc)], "recomputed_objective": None}
