"""Correctness checks for the engineered CLRP reference (scale-up study) against brute force on tiny instances."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

gp = pytest.importorskip("gurobipy")

from src.cg.clrp.a_data import CLRPInstance, load_prodhon
from src.cg.clrp.b_mip_xl import route_cost
from src.cg.clrp.d_ref_cg_xl import CLRPColumnGenerationXL, NGPricer, separate_rcc


def random_instance(rng, n: int, m: int, Q: int, depot_cap: float | None = None) -> CLRPInstance:
    demand = rng.integers(5, 20, size=n).astype(float)
    cap = np.full(m, depot_cap if depot_cap is not None else 10.0 ** 6)
    return CLRPInstance(rng.uniform(0, 50, (m, 2)).round(1), rng.uniform(0, 50, (n, 2)).round(1), Q, cap, demand,
                        rng.integers(500, 3000, size=m).astype(float), 1000.0, 0)


def write_prodhon(inst: CLRPInstance, path) -> None:
    tok = [inst.n, inst.m, *inst.depots.ravel(), *inst.customers.ravel(), inst.vehicle_capacity, *inst.depot_capacity,
           *inst.demand, *inst.opening_cost, inst.vehicle_cost, inst.cost_type]
    path.write_text("\n".join(str(t) for t in tok) + "\n")


def ng_enumeration(pr: NGPricer, p, const, Cc, c_out, c_in):
    """Minimum reduced cost and minimum min(0, reduced cost / load) over all ng-routes, by depth-first search."""
    best_rc, best_ratio = 0.0, 0.0

    def rec(j, mem, load, cost):
        nonlocal best_rc, best_ratio
        rc = cost + c_in[j]
        best_rc = min(best_rc, rc)
        if rc < 0:
            best_ratio = min(best_ratio, rc / load)
        for k in range(pr.n):
            if k != j and np.isfinite(Cc[j, k]) and k not in mem and load + pr.q[k] <= pr.Q:
                rec(k, (mem & set(np.nonzero(pr.inN[k])[0])) | {k}, load + pr.q[k], cost + Cc[j, k] - p[k])

    for j in range(pr.n):
        if np.isfinite(c_out[j]) and pr.q[j] <= pr.Q:
            rec(j, {j}, pr.q[j], const + c_out[j] - p[j])
    return best_rc, best_ratio


def test_ng_pricing_matches_enumeration():
    rng = np.random.default_rng(0)
    for _ in range(25):
        n, Q, ng = int(rng.integers(4, 9)), int(rng.integers(20, 50)), int(rng.integers(1, 6))
        inst = random_instance(rng, n, 2, Q)
        pr = NGPricer(inst, inst.travel("ceil"), ng=ng, near=3)
        d = int(rng.integers(2))
        p = rng.uniform(0, 4000, n)
        const = 1000.0 - rng.uniform(0, 500)
        Cc = pr.C - rng.uniform(0, 300, (n, n))  # cut duals can make arc costs negative
        np.fill_diagonal(Cc, np.inf)
        Cc[rng.integers(n), rng.integers(n)] = np.inf
        c_out = np.where(rng.random(n) < 0.85, pr.c_out[d], np.inf)
        res = pr.price(p, const, Cc, c_out, pr.c_in[d], exact=True, max_cols=1000)
        bf_rc, bf_ratio = ng_enumeration(pr, p, const, Cc, c_out, pr.c_in[d])
        assert res["complete"]
        assert res["min_rc"] == pytest.approx(min(bf_rc, 0.0), abs=1e-6)
        assert res["min_ratio"] == pytest.approx(bf_ratio, abs=1e-6)
        for rc, path in res["routes"]:
            assert pr.q[path].sum() <= Q
            val = const + c_out[path[0]] - p[path[0]] + sum(Cc[a, b] - p[b] for a, b in zip(path, path[1:])) + pr.c_in[d, path[-1]]
            assert val == pytest.approx(rc, abs=1e-6)


def test_rcc_separation_reports_true_violations():
    rng = np.random.default_rng(1)
    n, Q = 12, 30
    q = rng.integers(5, 15, size=n)
    X = np.triu(rng.random((n, n)) * (rng.random((n, n)) < 0.3), 1)
    X = X + X.T
    deg = X.sum(axis=1) + rng.random(n)
    for viol, U in separate_rcc(X, deg, q, Q, max_size=n):
        cut = X[U][:, ~U].sum() + (deg[U] - X[U].sum(axis=1)).sum()
        assert viol == pytest.approx(2 * math.ceil(q[U].sum() / Q) - cut)
        assert viol > 0


def brute_force(inst: CLRPInstance, cost: np.ndarray) -> float:
    """Optimal CLRP value by enumerating set partitions into routes with depot labels (tiny instances only)."""
    n, m, Q, q = inst.n, inst.m, inst.vehicle_capacity, inst.demand
    route = {}
    for size in range(1, n + 1):
        for S in itertools.combinations(range(n), size):
            if q[list(S)].sum() > Q:
                continue
            for d in range(m):
                route[d, S] = min(cost[d, m + p[0]] + sum(cost[m + a, m + b] for a, b in zip(p, p[1:])) + cost[m + p[-1], d]
                                  for p in itertools.permutations(S)) + inst.vehicle_cost
    best = math.inf

    def rec(left: tuple, load: list, total: float) -> None:
        nonlocal best
        if not left:
            best = min(best, total + sum(inst.opening_cost[d] for d in range(m) if load[d] > 0))
            return
        first, rest = left[0], left[1:]
        for size in range(len(rest) + 1):
            for extra in itertools.combinations(rest, size):
                S = (first,) + extra
                if q[list(S)].sum() > Q:
                    continue
                for d in range(m):
                    if load[d] + q[list(S)].sum() <= inst.depot_capacity[d]:
                        load[d] += q[list(S)].sum()
                        rec(tuple(x for x in rest if x not in extra), load, total + route[d, S])
                        load[d] -= q[list(S)].sum()

    rec(tuple(range(n)), [0.0] * m, 0.0)
    return best


@pytest.mark.parametrize("seed", range(10))
def test_reference_bounds_and_incumbents_on_tiny_instances(tmp_path, seed):
    rng = np.random.default_rng(10 + seed)
    n, m = int(rng.integers(4, 8)), int(rng.integers(2, 4))
    inst = random_instance(rng, n, m, Q=int(rng.integers(25, 45)))
    inst.depot_capacity = np.ceil(inst.demand.sum() * rng.uniform(0.4, 0.8, size=m))
    if inst.depot_capacity.sum() < inst.demand.sum():
        inst.depot_capacity[:] = inst.demand.sum()
    write_prodhon(inst, tmp_path / "tiny.dat")
    inst = load_prodhon(tmp_path / "tiny.dat")
    cost = inst.travel("ceil")
    opt = brute_force(inst, cost)
    seen, bounds = [], []
    # Small ng-neighbourhoods on half of the seeds, so non-elementary columns enter the master.
    ref = CLRPColumnGenerationXL(time_limit=30, ng=3 if seed % 2 else 8,
                                 on_incumbent=lambda r: seen.append(route_cost(inst, cost, r)), on_bound=bounds.append)
    out = ref.run(str(tmp_path / "tiny.dat"))
    assert seen and all(v is not None for v in seen)
    assert all(b <= opt + 1e-6 for b in bounds)
    assert bounds == sorted(bounds)
    assert out["global_lb"] <= opt + 1e-6
    assert route_cost(inst, cost, out["routes"]) == pytest.approx(out["best_objective"])
    assert out["termination_reason"] == "optimal"
    assert out["best_objective"] == pytest.approx(opt)


@pytest.mark.parametrize("seed", range(10))
def test_root_only_control_never_reports_a_bound_above_the_optimum(tmp_path, seed):
    rng = np.random.default_rng(100 + seed)
    n, m = int(rng.integers(4, 8)), int(rng.integers(2, 4))
    inst = random_instance(rng, n, m, Q=int(rng.integers(25, 45)))
    inst.depot_capacity = np.ceil(inst.demand.sum() * rng.uniform(0.4, 0.8, size=m))
    if inst.depot_capacity.sum() < inst.demand.sum():
        inst.depot_capacity[:] = inst.demand.sum()
    write_prodhon(inst, tmp_path / "tiny.dat")
    inst = load_prodhon(tmp_path / "tiny.dat")
    cost = inst.travel("ceil")
    opt = brute_force(inst, cost)
    seen, bounds = [], []
    out = CLRPColumnGenerationXL(time_limit=30, root_only=True, on_incumbent=lambda r: seen.append(route_cost(inst, cost, r)),
                                 on_bound=bounds.append).run(str(tmp_path / "tiny.dat"))
    assert all(v is not None and v >= opt - 1e-6 for v in seen)
    assert all(b <= opt + 1e-6 for b in bounds)
    assert out["global_lb"] <= opt + 1e-6


@pytest.mark.parametrize("seed", range(8))
def test_naive_cg_control_bounds_and_plans(tmp_path, seed):
    from src.cg.clrp.e_naive_cg import CLRPNaiveCG

    rng = np.random.default_rng(200 + seed)
    n, m = int(rng.integers(4, 8)), int(rng.integers(2, 4))
    inst = random_instance(rng, n, m, Q=int(rng.integers(25, 45)))
    inst.depot_capacity = np.ceil(inst.demand.sum() * rng.uniform(0.4, 0.8, size=m))
    if inst.depot_capacity.sum() < inst.demand.sum():
        inst.depot_capacity[:] = inst.demand.sum()
    write_prodhon(inst, tmp_path / "tiny.dat")
    inst = load_prodhon(tmp_path / "tiny.dat")
    cost = inst.travel("ceil")
    opt = brute_force(inst, cost)
    seen, bounds = [], []
    out = CLRPNaiveCG(time_limit=20, on_incumbent=lambda r: seen.append(route_cost(inst, cost, r)), on_bound=bounds.append).run(str(tmp_path / "tiny.dat"))
    assert seen and all(v is not None and v >= opt - 1e-6 for v in seen)
    assert all(b <= opt + 1e-6 for b in bounds)
    assert out["best_objective"] >= opt - 1e-6


def test_naive_cg_discards_incomplete_pricing_rounds(tmp_path, monkeypatch):
    import src.cg.clrp.e_naive_cg as naive

    rng = np.random.default_rng(7)
    inst = random_instance(rng, 6, 2, Q=40)
    inst.depot_capacity[:] = inst.demand.sum()
    write_prodhon(inst, tmp_path / "tiny.dat")
    real = naive.NGPricer.price

    def truncated(self, *args, **kwargs):
        res = real(self, *args, **kwargs)
        return {**res, "complete": False}  # as if a resource guard had cut the round short

    monkeypatch.setattr(naive.NGPricer, "price", truncated)
    out = naive.CLRPNaiveCG(time_limit=10).run(str(tmp_path / "tiny.dat"))
    assert out["cg_stop_reason"] == "pricing_resource_limit"
    assert out["cg_iterations"] == 1 and not out["cg_converged"] and out["global_lb"] is None
