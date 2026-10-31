"""Correctness checks for the PDPTW scale-up family (Li & Lim instances, PyVRP's VRPLIB format and conventions).

Tiny random instances (at most 4 requests) are solved by brute force (every feasible route, evaluated with the
independent verifier, then the best partition into at most K routes); the pricing oracle, the compact MIP, the naive
CG and the engineered branch-and-price are checked against it. The bundled PyVRP solutions check the conventions.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.cg.pdptw import e_verifier as ver
from src.cg.pdptw.a_data import feasible_arcs, load_pdptw, plan_cost, read_sol, route_check
from src.cg.pdptw.pricing import PDPPricer, best_insertion, insertion_plan

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/scaleup/raw/cvrp/Instances-main/PDPTW"
SOLS = ["Li100/lc101", "Li100/lc105", "Li100/lr101", "Li100/lr205", "Li100/lrc101", "Li100/lrc207", "Li100/lc201",
        "Li200/lc1_2_1", "Li200/lr1_2_1", "Li200/lrc1_2_1"]


# ---------------------------------------------------------------- tiny instances and brute force

def write_vrp(path: Path, xy, rows, Q: int, K: int) -> None:
    """rows[i] = (demand, ready, due, service, pickup_id, delivery_id) with 1-based file ids; row 0 is the depot."""
    lines = ["NAME: tiny", "TYPE: PDPTW", f"DIMENSION: {len(xy)}", f"VEHICLES: {K}", f"CAPACITY: {Q}",
             "EDGE_WEIGHT_TYPE: EUC_2D", "NODE_COORD_SECTION"]
    lines += [f"{i + 1} {x} {y}" for i, (x, y) in enumerate(xy)]
    lines.append("PICKUP_AND_DELIVERY_SECTION")
    lines += [f"{i + 1} " + " ".join(str(v) for v in row) for i, row in enumerate(rows)]
    lines += ["DEPOT_SECTION", "1", "-1", "EOF"]
    path.write_text("\n".join(lines) + "\n")


def random_instance(rng, n: int, path: Path, K: int | None = None, H: int = 240) -> Path:
    N = 2 * n
    nodes = list(range(1, N + 1))
    rng.shuffle(nodes)  # pickups and deliveries interleaved in arbitrary file order
    xy = [(10, 10)] + [None] * N
    rows = [(0, 0, H, 0, 0, 0)] + [None] * N
    Q = int(rng.integers(12, 25))
    for r in range(n):
        p, d = nodes[2 * r], nodes[2 * r + 1]
        q = int(rng.integers(3, 12))
        xy[p] = (int(rng.integers(0, 21)), int(rng.integers(0, 21)))
        dummy = rng.random() < 0.3  # a Li & Lim style dummy request: zero-service pickup at its delivery's location
        xy[d] = xy[p] if dummy else (int(rng.integers(0, 21)), int(rng.integers(0, 21)))
        a = int(rng.integers(0, 90))
        w = int(rng.integers(10, 80))
        rows[p] = (q, a, a + w, 0 if dummy else int(rng.integers(1, 6)), 0, d + 1)
        rows[d] = (-q, a + int(rng.integers(0, 20)), min(H - 15, a + w + int(rng.integers(20, 90))), int(rng.integers(1, 6)), p + 1, 0)
    write_vrp(path, xy, rows, Q, K if K is not None else n)
    return path


def feasible_routes(path: Path) -> dict[tuple[int, ...], int]:
    """Every feasible route (node sequence) with its distance, by enumeration with the independent verifier."""
    d = ver.read_instance(path)
    reqs = sorted(d["delivery_of"].items())
    out = {}
    for k in range(1, len(reqs) + 1):
        for S in itertools.combinations(reqs, k):
            nodes = [v for pd in S for v in pd]
            for perm in itertools.permutations(nodes):
                pos = {v: i for i, v in enumerate(perm)}
                if any(pos[p] > pos[q] for p, q in S):
                    continue
                cost, _ = ver.route_issue(d, list(perm))
                if cost is not None:
                    out[perm] = cost
    return out


def brute_force(path: Path) -> float:
    d = ver.read_instance(path)
    reqs = sorted(d["delivery_of"])
    idx = {p: k for k, p in enumerate(reqs)}
    best_set: dict[int, int] = {}
    for route, cost in feasible_routes(path).items():
        m = sum(1 << idx[v] for v in route if v in idx)
        best_set[m] = min(best_set.get(m, math.inf), cost)
    full = (1 << len(reqs)) - 1
    best = math.inf

    def rec(left: int, used: int, total: int) -> None:
        nonlocal best
        if not left:
            best = min(best, total)
            return
        if used >= d["K"]:
            return
        low = left & -left
        sub = left
        while sub:
            if sub & low and sub in best_set:
                rec(left ^ sub, used + 1, total + best_set[sub])
            sub = (sub - 1) & left

    rec(full, 0, 0)
    return best


def check_output(path: Path, tmp_path: Path, routes, objective) -> None:
    out = tmp_path / "out.json"
    out.write_text(json.dumps({"status": "feasible", "method": "decomposition", "objective_value": objective, "routes": routes}))
    res = ver.verify_solution(path, out)
    assert res["ok"], res["issues"]


# ---------------------------------------------------------------- conventions (no solver)

@pytest.mark.parametrize("name", SOLS)
def test_bundled_solutions_reproduce_costs(name):
    vrp, sol = DATA / f"{name}.vrp", DATA / f"{name}.sol"
    if not vrp.exists():
        pytest.skip("PyVRP PDPTW instances not present")
    inst = load_pdptw(vrp)
    routes, cost = read_sol(sol)
    assert plan_cost(inst, inst.distances(), routes) == cost
    res = ver.verify_sol_file(vrp, sol)
    assert res["ok"] and res["objective_value"] == cost, res["issues"]
    # The arc elimination keeps every arc of a feasible solution.
    arcs = feasible_arcs(inst, inst.distances())
    for r in routes:
        seq = [0] + r + [0]
        assert all(arcs[a, b] for a, b in zip(seq, seq[1:]))


def test_verifier_rejects_broken_plans(tmp_path):
    vrp, sol = DATA / "Li100/lc101.vrp", DATA / "Li100/lc101.sol"
    if not vrp.exists():
        pytest.skip("PyVRP PDPTW instances not present")
    routes, cost = read_sol(sol)
    out = tmp_path / "o.json"

    def verdict(rs, obj, **extra):
        out.write_text(json.dumps({"status": "feasible", "method": "x", "objective_value": obj, "routes": rs, **extra}))
        return ver.verify_solution(vrp, out)

    assert verdict(routes, cost)["ok"]
    assert not verdict(routes, cost + 1)["ok"]
    assert not verdict(routes[:-1], cost)["ok"]
    inst = load_pdptw(vrp)
    r0 = list(routes[0])
    p = next(v for v in r0 if inst.is_pickup[v])
    dlv = inst.partner(p)
    swapped = [dlv if v == p else p if v == dlv else v for v in r0]
    assert not verdict([swapped] + routes[1:], cost)["ok"]
    split = [[v for v in r0 if v != dlv], [dlv]] + routes[1:]
    assert not verdict(split, cost)["ok"]
    out.write_text(json.dumps({"objective_value": cost, "routes": routes}))
    assert not ver.verify_solution(vrp, out)["ok"]  # status and method are required


def test_initializer_is_feasible():
    for name in ["Li100/lc101", "Li100/lr101", "Li100/lrc101"]:
        vrp = DATA / f"{name}.vrp"
        if not vrp.exists():
            pytest.skip("PyVRP PDPTW instances not present")
        inst = load_pdptw(vrp)
        plan = insertion_plan(inst, inst.distances())
        assert plan is not None and plan_cost(inst, inst.distances(), plan) is not None


def reference_insertion(inst, c, route, p, d, cutoff=math.inf):
    """Every insertion position checked from scratch with route_check (same scan order, strict improvement)."""
    base = route_check(inst, c, route)[0] if route else 0
    best, best_route = cutoff, None
    for a in range(len(route) + 1):
        for b in range(a, len(route) + 1):
            cand = route[:a] + [p] + route[a:b] + [d] + route[b:]
            cost, _ = route_check(inst, c, cand)
            if cost is not None and cost - base < best:
                best, best_route = cost - base, cand
    return best, best_route


def test_incremental_insertion_matches_reference():
    rng = np.random.default_rng(3)
    for name in ["Li100/lr104", "Li100/lc101", "Li100/lrc201", "Li100/lr205"]:
        vrp = DATA / f"{name}.vrp"
        if not vrp.exists():
            pytest.skip("PyVRP PDPTW instances not present")
        inst = load_pdptw(vrp)
        c = inst.distances().tolist()
        routes, _ = read_sol(DATA / f"{name}.sol")
        for _ in range(40):
            ra, rb = rng.choice(len(routes), size=2, replace=len(routes) < 2)
            p = int(rng.choice([v for v in routes[ra] if inst.is_pickup[v]]))
            d = inst.partner(p)
            target = [v for v in routes[rb] if v not in (p, d)]
            cutoff = math.inf if rng.random() < 0.7 else float(rng.integers(0, 30000))
            assert best_insertion(inst, c, target, p, d, cutoff) == reference_insertion(inst, c, target, p, d, cutoff)


def test_pricing_matches_enumeration(tmp_path):
    rng = np.random.default_rng(0)
    for trial in range(30):
        path = random_instance(rng, int(rng.integers(2, 5)), tmp_path / f"p{trial}.vrp")
        inst = load_pdptw(path)
        dist = inst.distances()
        routes = feasible_routes(path)
        arcs = feasible_arcs(inst, dist)
        for route in routes:  # arc elimination is valid
            seq = (0,) + route + (0,)
            assert all(arcs[a, b] for a, b in zip(seq, seq[1:]))
        pr = PDPPricer(inst, dist, arcs)
        prize = rng.uniform(0, 30000, inst.n)
        const = float(rng.uniform(0, 5000))
        forbidden = None
        if trial % 3 == 2:  # branching rules: some forbidden arcs (strict dominance)
            cand = [(int(a), int(b)) for a, b in zip(*np.nonzero(arcs))]
            forbidden = {cand[k] for k in rng.choice(len(cand), size=min(3, len(cand)), replace=False)}
        rc = lambda r: const + routes[r] - sum(prize[inst.request_of[v]] for v in r if inst.is_pickup[v])
        allowed = [r for r in routes if not forbidden or not any(a in forbidden for a in zip((0,) + r, r + (0,)))]
        bf = min([rc(r) for r in allowed], default=0.0)
        res = pr.price(prize, const, exact=True, max_cols=10_000, forbidden=forbidden)
        assert res["complete"]
        assert res["min_rc"] == pytest.approx(min(bf, 0.0), abs=1e-6)
        for val, path_ in res["routes"]:
            assert tuple(path_) in routes and tuple(path_) in allowed
            assert val == pytest.approx(rc(tuple(path_)), abs=1e-6)
        if bf < -1e-6:
            assert res["routes"] and res["routes"][0][0] == pytest.approx(bf, abs=1e-6)
        # ng-routes: a relaxation (never above the elementary minimum); with memories as large as the request set
        # they are the elementary routes.
        ngr = pr.price(prize, const, exact=True, max_cols=10_000, forbidden=forbidden, ng=2)
        assert ngr["complete"] and ngr["min_rc"] <= min(bf, 0.0) + 1e-6
        full = pr.price(prize, const, exact=True, max_cols=10_000, forbidden=forbidden, ng=inst.n)
        assert full["min_rc"] == pytest.approx(min(bf, 0.0), abs=1e-6)
        heur = pr.price(prize, const, exact=False, max_cols=50, forbidden=forbidden, near=2, beam=3)
        for val, path_ in heur["routes"]:
            assert tuple(path_) in allowed and val == pytest.approx(rc(tuple(path_)), abs=1e-6)


# ---------------------------------------------------------------- solvers on tiny instances (Gurobi)

def tiny_cases(tmp_path, count: int = 6):
    rng = np.random.default_rng(7)
    cases = []
    while len(cases) < count:
        n = int(rng.integers(2, 5))
        K = int(rng.integers(1, n + 1))
        path = random_instance(rng, n, tmp_path / f"t{len(cases)}_{int(rng.integers(1e9))}.vrp", K=K)
        opt = brute_force(path)
        if math.isfinite(opt):
            cases.append((path, opt))
    return cases


def run_method(cls, path, tmp_path, limit=20.0, **kw):
    pub, bounds = [], []
    res = cls(time_limit=limit, on_incumbent=lambda routes, *a, **k: pub.append([list(r) for r in routes]),
              on_bound=bounds.append, **kw).run(str(path))
    inst = load_pdptw(path)
    dist = inst.distances()
    for routes in pub:
        obj = plan_cost(inst, dist, routes)
        assert obj is not None
        check_output(path, tmp_path, routes, obj)
    return res, pub, bounds


def test_compact_mip_reaches_optimum(tmp_path):
    pytest.importorskip("gurobipy")
    from src.cg.pdptw.b_mip_xl import PDPTWCompactMIP
    for k, (path, opt) in enumerate(tiny_cases(tmp_path)):
        res, pub, bounds = run_method(PDPTWCompactMIP, path, tmp_path, warm_start=k % 2 == 1)
        assert res["termination_reason"] == "optimal"
        assert res["best_objective"] == opt
        assert all(b <= opt + 1e-6 for b in bounds)
        assert res["rejected_solutions"] == 0


def test_naive_cg_reaches_optimum(tmp_path):
    pytest.importorskip("gurobipy")
    from src.cg.pdptw.c_naive_cg import PDPTWNaiveCG
    for path, opt in tiny_cases(tmp_path):
        res, pub, bounds = run_method(PDPTWNaiveCG, path, tmp_path, limit=10.0)
        assert res["cg_converged"]
        assert res["best_objective"] == opt
        assert res["global_lb"] is not None and res["global_lb"] <= opt
        assert all(b <= opt for b in bounds)


def test_engineered_reaches_and_proves_optimum(tmp_path):
    pytest.importorskip("gurobipy")
    from src.cg.pdptw.d_ref_cg_xl import PDPTWColumnGenerationXL
    for path, opt in tiny_cases(tmp_path):
        res, pub, bounds = run_method(PDPTWColumnGenerationXL, path, tmp_path)
        assert res["best_objective"] == opt
        assert res["termination_reason"] == "optimal"
        assert all(b <= opt for b in bounds)
        assert res["global_lb"] == opt


# ---------------------------------------------------------------- MIP exactness on fixed plans (review 28)

def fixed_plan_feasible(path: Path, routes: list[list[int]]) -> bool:
    """Whether the compact MIP admits the given plan: every arc of the plan is fixed to 1, every other arc to 0."""
    import gurobipy as gp
    from gurobipy import GRB
    from src.cg.pdptw.b_mip_xl import build_pdptw_mip
    inst = load_pdptw(path)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model, info = build_pdptw_mip(inst, inst.distances(), env=env)
    used = {a for r in routes for a in zip([0] + r, r + [0])}
    if not used <= set(info["arcs"]):
        model.dispose()
        env.dispose()
        return False
    for a, var in info["x"].items():
        var.LB = var.UB = 1.0 if a in used else 0.0
    model.optimize()
    ok = model.Status == GRB.OPTIMAL
    model.dispose()
    env.dispose()
    return ok


def test_mip_admits_exactly_the_feasible_single_routes(tmp_path):
    """On tiny instances (two requests, incl. dummy pickups), every ordering of all nodes on one route is admitted by
    the MIP iff the independent verifier accepts it (time windows from the depot, precedence, capacity)."""
    pytest.importorskip("gurobipy")
    rng = np.random.default_rng(28)
    checked = disagreements = 0
    for k in range(12):
        path = random_instance(rng, 2, tmp_path / f"r{k}.vrp", K=1, H=int(rng.integers(120, 240)))
        d = ver.read_instance(path)
        inst = load_pdptw(path)
        assert build_precedence_exact(inst)
        for perm in itertools.permutations(range(1, inst.N + 1)):
            truth = ver.route_issue(d, list(perm))[0] is not None
            got = fixed_plan_feasible(path, [list(perm)])
            checked += 1
            disagreements += truth != got
    assert checked > 0 and disagreements == 0


def build_precedence_exact(inst) -> bool:
    import gurobipy as gp
    from src.cg.pdptw.b_mip_xl import build_pdptw_mip
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model, info = build_pdptw_mip(inst, inst.distances(), env=env)
    model.dispose()
    env.dispose()
    return info["precedence_exact"]


@pytest.mark.skipif(not (DATA / "Li400/lrc1_4_5.vrp").exists(), reason="Li400 data not present")
def test_review28_depot_start_counterexample_is_rejected():
    """Review 28: on lrc1_4_5 the route below cannot leave the depot early enough (arrival 266,006 > 266,000 at node 41);
    the bundled plan with that route substituted used to satisfy every MIP row."""
    pytest.importorskip("gurobipy")
    path = DATA / "Li400/lrc1_4_5.vrp"
    bad = [143, 283, 76, 296, 23, 228, 19, 41, 99, 135, 414, 399]
    d = ver.read_instance(path)
    assert ver.route_issue(d, bad)[0] is None
    routes, _ = read_sol(DATA / "Li400/lrc1_4_5.sol")
    rest = [[v for v in r if v not in set(bad)] for r in routes]
    plan = [r for r in rest if r] + [bad]
    assert sorted(v for r in plan for v in r) == list(range(1, load_pdptw(path).N + 1))
    assert not fixed_plan_feasible(path, plan)
    assert fixed_plan_feasible(path, [list(r) for r in routes])  # the bundled solution itself stays admissible
    assert build_precedence_exact(load_pdptw(path))
