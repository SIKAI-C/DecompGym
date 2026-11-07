"""VRPTW scale-up family: conventions against the bundled best-known solutions, pricing against ng-route enumeration,
and the compact MIP, the naive and the engineered column generation against brute force on tiny instances."""
from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.cg.vrptw.a_data import load_sol, load_vrptw, plan_issues
from src.cg.vrptw.e_verifier import verify_solution

RAW = Path(__file__).resolve().parents[2] / "outputs/scaleup/raw/cvrp/Instances-main/VRPTW"
SOL_FILES = ["Solomon/C101", "Solomon/R101", "Solomon/R105", "Solomon/RC105", "Solomon/R201", "Solomon/RC208",
             "GH200/R1_2_1", "GH200/RC1_2_1", "GH200/C1_2_1"]


# ---------------------------------------------------------------- instances

def write_vrptw(path: Path, depot, coords, demand, windows, service, Q: int, vehicles: int | None, uniform: bool) -> None:
    n = len(coords)
    lines = ["NAME : tiny", "TYPE : CVRPTW", f"DIMENSION : {n + 1}"]
    if vehicles is not None:
        lines.append(f"VEHICLES : {vehicles}")
    lines.append(f"CAPACITY : {Q}")
    if uniform:
        lines.append(f"SERVICE_TIME : {service[0]}")
    lines += ["EDGE_WEIGHT_TYPE : EUC_2D", "NODE_COORD_SECTION", f"1 {depot[0]} {depot[1]}"]
    lines += [f"{i + 2} {x} {y}" for i, (x, y) in enumerate(coords)]
    lines += ["DEMAND_SECTION", "1 0"] + [f"{i + 2} {q}" for i, q in enumerate(demand)]
    lines += ["TIME_WINDOW_SECTION", f"1 {windows[0][0]} {windows[0][1]}"]
    lines += [f"{i + 2} {a} {b}" for i, (a, b) in enumerate(windows[1:])]
    if not uniform:
        lines += ["SERVICE_TIME_SECTION", "1 0"] + [f"{i + 2} {s}" for i, s in enumerate(service)]
    lines += ["DEPOT_SECTION", "1", "-1", "EOF"]
    path.write_text("\n".join(lines) + "\n")


def random_instance(rng, path: Path, n: int, vehicles: int | None = None, wide: bool = False) -> Path:
    depot = (int(rng.integers(15, 26)), int(rng.integers(15, 26)))
    coords = [(int(rng.integers(0, 41)), int(rng.integers(0, 41))) for _ in range(n)]
    demand = [int(rng.integers(3, 11)) for _ in range(n)]
    Q = int(rng.integers(12, 31))
    uniform = bool(rng.integers(2))
    service = [int(rng.integers(3, 11))] * n if uniform else [int(rng.integers(3, 11)) for _ in range(n)]
    H = int(rng.integers(120, 220))
    windows = [(0, H)]
    for (x, y), s in zip(coords, service):
        d0 = math.dist((x, y), depot)
        lo, hi = math.ceil(d0), math.floor(H - s - d0) - 1
        c = int(rng.integers(lo, max(lo, hi) + 1))
        w = int(rng.integers(5, 100 if wide else 50))
        a, b = max(0, c - w // 2), min(max(lo, hi), c + w // 2)
        windows.append((a, max(a, b)))
    write_vrptw(path, depot, coords, demand, windows, service, Q, vehicles, uniform)
    return path


def brute_force(inst):
    """Optimal distance (tenths) and plan: all feasible elementary routes, then set partitioning by dynamic
    programming over customer subsets with at most `vehicles` routes. None if infeasible."""
    n, Q = inst.n, int(inst.capacity)
    d, e, l, s, q = inst.dist.tolist(), inst.ready.tolist(), inst.due.tolist(), inst.service.tolist(), inst.demand.tolist()
    best: dict[int, tuple[int, list[int]]] = {}

    def dfs(route, t, load, length, mask):
        last = route[-1]
        if t + s[last] + d[last][0] <= l[0]:
            c = length + d[last][0]
            if mask not in best or c < best[mask][0]:
                best[mask] = (c, list(route))
        for j in range(1, n + 1):
            if mask >> j & 1 or load + q[j] > Q:
                continue
            tj = max(t + s[last] + d[last][j], e[j])
            if tj <= l[j]:
                dfs(route + [j], tj, load + q[j], length + d[last][j], mask | 1 << j)

    for j in range(1, n + 1):
        t = max(e[0] + d[0][j], e[j])
        if t <= l[j] and q[j] <= Q:
            dfs([j], t, q[j], d[0][j], 1 << j)
    full = sum(1 << j for j in range(1, n + 1))
    f: dict[int, dict[int, tuple[int, list]]] = {0: {0: (0, [])}}
    for mask in range(2, full + 1, 2):
        if mask & ~full:
            continue
        low = mask & -mask
        rest = mask ^ low
        out: dict[int, tuple[int, list]] = {}
        sub = rest
        while True:
            r = sub | low
            if r in best and (mask ^ r) in f:
                c, route = best[r]
                for k, (c2, routes) in f[mask ^ r].items():
                    if k + 1 <= inst.vehicles and (k + 1 not in out or c + c2 < out[k + 1][0]):
                        out[k + 1] = (c + c2, routes + [route])
            if sub == 0:
                break
            sub = (sub - 1) & rest
        if out:
            f[mask] = out
    if full not in f:
        return None, None, best
    k, (c, routes) = min(f[full].items(), key=lambda kv: kv[1][0])
    return c, routes, best


def tiny_cases(tmp_path: Path, count: int, seed: int, sizes=(4, 8)):
    rng = np.random.default_rng(seed)
    out = []
    t = 0
    while len(out) < count:
        n = int(rng.integers(sizes[0], sizes[1] + 1))
        tight = bool(rng.integers(3) == 0)
        path = random_instance(rng, tmp_path / f"t{seed}_{t}.vrp", n, vehicles=None if not tight else max(1, n // 3),
                               wide=bool(rng.integers(2)))
        t += 1
        inst = load_vrptw(path)
        opt, plan, _ = brute_force(inst)
        if opt is None:
            continue
        out.append((path, inst, opt, plan))
    return out


def check_witness(path: Path, inst, routes, tmp_path: Path, status: str = "feasible") -> float:
    from src.cg.vrptw.b_mip_xl import vrptw_output
    out = tmp_path / "witness.json"
    out.write_text(json.dumps(vrptw_output(inst, routes, status=status, method="test")))
    res = verify_solution(path, out)
    assert res["ok"], res["issues"]
    return res["objective_value"]


# ---------------------------------------------------------------- conventions (no solver)

@pytest.mark.parametrize("name", SOL_FILES)
def test_bundled_solutions_reproduced_exactly(name, tmp_path):
    vrp, sol = RAW / f"{name}.vrp", RAW / f"{name}.sol"
    if not vrp.exists() or not sol.exists():
        pytest.skip("instance files not present")
    inst = load_vrptw(vrp)
    routes, cost = load_sol(sol)
    total, issues = plan_issues(inst, routes)
    assert not issues and total == cost
    assert len(routes) <= inst.vehicles
    out = tmp_path / "bks.json"
    out.write_text(json.dumps({"status": "feasible", "method": "bks", "objective_value": cost / 10, "routes": routes}))
    res = verify_solution(vrp, out)
    assert res["ok"], res["issues"]
    assert res["objective_value"] == cost / 10
    # Window tightening and arc elimination keep every arc and start time of the best-known plan.
    from src.cg.vrptw.pricing import time_bounds
    tb = time_bounds(inst)
    for r in routes:
        assert tb["out_ok"][r[0] - 1] and tb["in_ok"][r[-1] - 1]
        for a, b in zip(r, r[1:]):
            assert tb["arc_ok"][a - 1, b - 1]
        t = inst.ready[0]
        prev = 0
        for c in r:
            t = max(t + inst.service[prev] + inst.dist[prev, c], inst.ready[c])
            assert tb["E"][c - 1] <= t <= tb["L"][c - 1]
            prev = c


def test_verifier_rejects_bad_plans(tmp_path):
    # Depot (0, 0) open [0, 100]; customer 1 at (10, 0) open [0, 15]; customer 2 at (20, 0); service 5.
    path = tmp_path / "v.vrp"
    write_vrptw(path, (0, 0), [(10, 0), (20, 0), (0, 7)], [3, 3, 6], [(0, 100), (0, 15), (0, 100), (0, 100)], [5, 5, 5],
                Q=10, vehicles=2, uniform=True)
    inst = load_vrptw(path)
    good = {"status": "feasible", "method": "x", "objective_value": 54.0, "routes": [[1, 2], [3]]}
    total, issues = plan_issues(inst, good["routes"])
    assert not issues and total == 540
    out = tmp_path / "o.json"
    out.write_text(json.dumps(good))
    assert verify_solution(path, out)["ok"]
    bad_plans = [dict(good, objective_value=54.1), dict(good, status="unknown"), {k: v for k, v in good.items() if k != "method"},
                 dict(good, routes=[[1, 2]]), dict(good, routes=[[1, 2], [3], [3]]), dict(good, objective_value=float("nan")),
                 dict(good, routes=[[2, 1], [3]], objective_value=54.0),   # late at customer 1
                 dict(good, routes=[[1, 3], [2]], objective_value=round((100 + 122 + 70 + 200 + 200) / 10, 1)),  # load 9 ok
                 dict(good, routes=[[1, 2, 3]], objective_value=0.0),       # load 12 > 10
                 dict(good, routes=[[1], [2], [3]], objective_value=74.0)]  # 3 routes > 2 vehicles
    for k, bad in enumerate(bad_plans):
        out.write_text(json.dumps(bad))
        res = verify_solution(path, out)
        if k == 7:
            # Feasible plan: only its objective decides.
            assert res["ok"] == (abs(bad["objective_value"] - res["objective_value"]) <= 1e-6)
            continue
        assert not res["ok"], (k, res)
        if k >= 6:
            assert plan_issues(inst, bad["routes"])[0] is None


# ---------------------------------------------------------------- pricing

def ng_min_rc(pr, p, const, Cc, c_out, c_in):
    """Minimum reduced cost over all ng-routes by memoized recursion over (node, memory, load, time)."""
    n = pr.n
    N = [set(int(x) for x in pr.nb[k]) for k in range(n)]

    @functools.lru_cache(maxsize=None)
    def f(j, mem, load, t):
        best = c_in[j] if (t <= pr.ret_by[j] and np.isfinite(c_in[j])) else math.inf
        for k in range(n):
            if not np.isfinite(Cc[j, k]) or (mem >> k) & 1 or load + pr.q[k] > pr.Q:
                continue
            tk = max(int(pr.e[k]), t + int(pr.tau[j, k]))
            if tk > pr.l[k]:
                continue
            m2 = sum(1 << x for x in range(n) if (mem >> x) & 1 and x in N[k]) | (1 << k)
            best = min(best, Cc[j, k] - p[k] + f(k, m2, load + int(pr.q[k]), tk))
        return best

    out = math.inf
    for j in range(n):
        if np.isfinite(c_out[j]) and pr.q[j] <= pr.Q:
            t = max(int(pr.e[j]), pr.e0 + int(pr.d_out[j]))
            if t <= pr.l[j]:
                out = min(out, const + c_out[j] - p[j] + f(j, 1 << j, int(pr.q[j]), t))
    return out


def ng_feasible(pr, path) -> bool:
    mem = {path[0]}
    for a, b in zip(path, path[1:]):
        if b in mem:
            return False
        mem = (mem & set(int(x) for x in pr.nb[b])) | {b}
    return True


def test_pricing_matches_ng_enumeration(tmp_path):
    from src.cg.vrptw.pricing import TWPricer
    rng = np.random.default_rng(11)
    checked = negative = 0
    for t in range(60):
        n = int(rng.integers(4, 9))
        inst = load_vrptw(random_instance(rng, tmp_path / f"p{t}.vrp", n, wide=bool(rng.integers(2))))
        pr = TWPricer(inst, ng=int(rng.integers(1, 9)), near=3, grid_points=int(rng.integers(5, 400)),
                      load_cells=int(rng.integers(1, 30)), max_events=int(rng.choice([30, 10 ** 6])))
        p = rng.uniform(0, 600, n)
        const = float(rng.uniform(-50, 50))
        Cc = pr.C.copy()
        if rng.random() < 0.5:
            a, b = rng.integers(n, size=2)
            Cc[a, b] = np.inf
        c_out, c_in = pr.c_out.copy(), pr.c_in.copy()
        if rng.random() < 0.3:
            c_out[rng.integers(n)] = np.inf
        res = pr.price(p, const, Cc, c_out, c_in, exact=True, max_cols=1000)
        bf = ng_min_rc(pr, p, const, Cc, c_out, c_in)
        assert res["complete"]
        if bf < -1e-6:
            negative += 1
            assert res["min_rc"] == pytest.approx(bf, abs=1e-6)
            assert res["routes"] and res["routes"][0][0] == pytest.approx(bf, abs=1e-6)
        else:
            assert res["min_rc"] >= -1e-6 and not res["routes"]
        for rc, path in res["routes"]:
            assert pr.feasible(path) and ng_feasible(pr, path)
            assert pr.route_rc(path, p, const, Cc, c_out, c_in) == pytest.approx(rc, abs=1e-6)
        # The time-only completion bound (heuristic rounds) is valid too: exact labelling with it finds the same minimum.
        res2 = pr.price(p, const, Cc, c_out, c_in, exact=True, max_cols=1, cap_bound=False)
        assert res2["complete"] and res2["min_rc"] == pytest.approx(res["min_rc"], abs=1e-6)
        heur = pr.price(p, const, Cc, c_out, c_in, exact=False, beam=5, max_cols=50)
        for rc, path in heur["routes"]:
            assert rc < 0 and pr.feasible(path) and ng_feasible(pr, path)
            assert pr.route_rc(path, p, const, Cc, c_out, c_in) == pytest.approx(rc, abs=1e-6)
        checked += 1
    assert checked == 60 and negative >= 30


def test_time_bounds_keep_every_feasible_route(tmp_path):
    from src.cg.vrptw.pricing import time_bounds
    rng = np.random.default_rng(3)
    for t in range(15):
        inst = load_vrptw(random_instance(rng, tmp_path / f"b{t}.vrp", int(rng.integers(4, 8)), wide=True))
        tb = time_bounds(inst)
        _, _, routes = brute_force(inst)
        for _, r in routes.values():
            assert tb["out_ok"][r[0] - 1] and tb["in_ok"][r[-1] - 1]
            tt, prev = inst.ready[0], 0
            for c in r:
                if prev:
                    assert tb["arc_ok"][prev - 1, c - 1]
                tt = max(tt + inst.service[prev] + inst.dist[prev, c], inst.ready[c])
                assert tb["E"][c - 1] <= tt <= tb["L"][c - 1]
                prev = c


# ---------------------------------------------------------------- heuristics

def test_initializer_and_local_search(tmp_path):
    from src.cg.vrptw.heuristic import LocalSearch, basic_initializer
    import time
    rng = np.random.default_rng(7)
    for t in range(15):
        inst = load_vrptw(random_instance(rng, tmp_path / f"h{t}.vrp", int(rng.integers(4, 9))))
        try:
            plan = basic_initializer(inst)
        except ValueError:
            continue  # a customer no route can serve
        total, issues = plan_issues(inst, plan)
        if len(plan) > inst.vehicles:
            continue
        assert not issues
        better = LocalSearch(inst).improve(plan, time.perf_counter() + 5)
        total2, issues2 = plan_issues(inst, better)
        assert not issues2 and total2 <= total
    for name in ("Solomon/R101", "GH200/RC1_2_1"):
        vrp = RAW / f"{name}.vrp"
        if vrp.exists():
            inst = load_vrptw(vrp)
            plan = basic_initializer(inst)
            total, issues = plan_issues(inst, plan)
            assert not issues
            total2, issues2 = plan_issues(inst, LocalSearch(inst).improve(plan, time.perf_counter() + 10))
            assert not issues2 and total2 <= total


# ---------------------------------------------------------------- solvers vs brute force

gp = pytest.importorskip("gurobipy")


def test_compact_mip_matches_brute_force(tmp_path):
    from src.cg.vrptw.b_mip_xl import VRPTWCompactMIP
    for path, inst, opt, _ in tiny_cases(tmp_path, 12, seed=21):
        seen = []
        res = VRPTWCompactMIP(time_limit=60, on_incumbent=seen.append).run(str(path))
        assert res["termination_reason"] == "optimal"
        assert res["best_objective"] == pytest.approx(opt / 10, abs=1e-9)
        assert res["global_lb"] <= opt / 10 + 1e-9
        for routes in seen:
            check_witness(path, inst, routes, tmp_path)


def test_naive_cg_tiny(tmp_path):
    from src.cg.vrptw.c_naive_cg import VRPTWNaiveCG
    hits = 0
    cases = tiny_cases(tmp_path, 12, seed=31)
    for path, inst, opt, _ in cases:
        seen, bounds = [], []
        res = VRPTWNaiveCG(time_limit=20, on_incumbent=seen.append, on_bound=bounds.append).run(str(path))
        assert res["cg_converged"]
        assert res["best_objective"] is not None and res["best_objective"] >= opt / 10 - 1e-9
        assert all(b <= opt / 10 + 1e-9 for b in bounds) and bounds
        for routes in seen:
            check_witness(path, inst, routes, tmp_path)
        hits += res["best_objective"] == pytest.approx(opt / 10, abs=1e-9)
    print(f"naive CG reaches the optimum on {hits}/{len(cases)} tiny instances")
    assert hits >= len(cases) - 2


def test_engineered_branch_and_price_tiny(tmp_path):
    from src.cg.vrptw.d_ref_cg_xl import VRPTWColumnGenerationXL
    for path, inst, opt, _ in tiny_cases(tmp_path, 20, seed=41, sizes=(5, 9)):
        seen, bounds = [], []
        res = VRPTWColumnGenerationXL(time_limit=30, on_incumbent=seen.append, on_bound=bounds.append, max_cols=5).run(str(path))
        assert res["termination_reason"] == "optimal", res
        assert res["best_objective"] == pytest.approx(opt / 10, abs=1e-9)
        assert res["global_lb"] == pytest.approx(opt / 10, abs=1e-9)
        assert all(b <= opt / 10 + 1e-9 for b in bounds)
        for routes in seen:
            check_witness(path, inst, routes, tmp_path)


def test_engineered_without_smoothing_and_with_tight_fleet(tmp_path):
    from src.cg.vrptw.d_ref_cg_xl import VRPTWColumnGenerationXL
    rng = np.random.default_rng(51)
    done = 0
    t = 0
    while done < 6:
        n = int(rng.integers(5, 9))
        path = random_instance(rng, tmp_path / f"k{t}.vrp", n, vehicles=int(rng.integers(1, 4)), wide=True)
        t += 1
        inst = load_vrptw(path)
        opt, _, _ = brute_force(inst)
        if opt is None:
            continue
        for alpha in (0.0, 0.7):
            seen, bounds = [], []
            res = VRPTWColumnGenerationXL(time_limit=30, alpha=alpha, on_incumbent=seen.append, on_bound=bounds.append).run(str(path))
            assert res["termination_reason"] == "optimal" and res["best_objective"] == pytest.approx(opt / 10, abs=1e-9)
            assert all(b <= opt / 10 + 1e-9 for b in bounds)
            for routes in seen:
                assert len(routes) <= inst.vehicles
                check_witness(path, inst, routes, tmp_path)
        done += 1
