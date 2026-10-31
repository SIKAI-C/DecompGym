"""Correctness checks for the CVRP family (scale-up study): the CVRPLIB conventions and the evaluator against the
published best-known solutions of the X set, the independent verifier, and the compact MIP, the naive and the
engineered column generation against brute force on synthetic instances with at most 9 customers."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

gp = pytest.importorskip("gurobipy")

from src.cg.cvrp.a_data import load_sol, load_vrp
from src.cg.cvrp.b_mip_xl import CVRPCompactMIP, build_cvrp_mip, cvrp_output, integer_check, route_cost, route_issues
from src.cg.cvrp.c_naive_cg import CVRPNaiveCG
from src.cg.cvrp.d_ref_cg_xl import CVRPColumnGenerationXL
from src.cg.cvrp.e_verifier import verify_solution
from src.cg.cvrp.heuristic import CVRPLocalSearch, basic_initializer, repair_cover

RAW = Path(__file__).resolve().parents[1] / "outputs" / "scaleup" / "raw" / "cvrp" / "Instances-main" / "CVRP"
needs_raw = pytest.mark.skipif(not (RAW / "X-n101-k25.vrp").exists(), reason="CVRPLIB X instances not downloaded")
# CVRPLIB best-known costs and route counts (instance name: (cost, routes)); every one is a proven optimum. Several use
# more routes than the k of the name, so k is a minimum, not a fleet size.
PUBLISHED = {"X-n101-k25": (27591, 26), "X-n110-k13": (14971, 13), "X-n125-k30": (55539, 30),
             "X-n153-k22": (21220, 23), "X-n195-k51": (44225, 53)}


# ---------------------------------------------------------------- helpers

def write_vrp(path: Path, coords, demand, Q: int, depot: int = 1) -> Path:
    """A .vrp file in the CVRPLIB format; coords and demand are given in node order 1..n+1."""
    lines = [f"NAME : {path.stem}", 'COMMENT : "synthetic"', "TYPE : CVRP", f"DIMENSION : {len(coords)}",
             "EDGE_WEIGHT_TYPE : EUC_2D", f"CAPACITY : {Q}", "NODE_COORD_SECTION"]
    lines += [f"{k + 1}\t{x}\t{y}" for k, (x, y) in enumerate(coords)]
    lines += ["DEMAND_SECTION"] + [f"{k + 1}\t{d}" for k, d in enumerate(demand)]
    lines += ["DEPOT_SECTION", f"\t{depot}", "\t-1", "EOF"]
    path.write_text("\n".join(lines) + "\n")
    return path


def random_instance(tmp_path: Path, seed: int, n_range=(5, 9), name: str = "tiny") -> Path:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(*n_range))
    Q = int(rng.integers(10, 30))
    coords = rng.integers(0, 100, size=(n + 1, 2)).tolist()
    demand = [0] + rng.integers(1, Q // 2 + 1, size=n).tolist()
    return write_vrp(tmp_path / f"{name}{seed}.vrp", coords, demand, Q)


def brute_force(inst) -> float:
    """Optimal CVRP value: Held-Karp over every customer subset (cheapest route through it), then the cheapest
    partition into routes within the capacity by subset dynamic programming."""
    n, Q, q = inst.n, inst.capacity, inst.demand
    d = inst.distances()
    full = 1 << n
    dp = np.full((full, n), np.inf)
    for j in range(n):
        dp[1 << j, j] = d[0, j + 1]
    for S in range(1, full):
        for j in range(n):
            if not (S >> j) & 1 or not np.isfinite(dp[S, j]):
                continue
            for k in range(n):
                if not (S >> k) & 1:
                    T = S | 1 << k
                    dp[T, k] = min(dp[T, k], dp[S, j] + d[j + 1, k + 1])
    route = np.full(full, np.inf)
    for S in range(1, full):
        members = [j for j in range(n) if (S >> j) & 1]
        if sum(q[j + 1] for j in members) <= Q:
            route[S] = min(dp[S, j] + d[j + 1, 0] for j in members)
    best = np.full(full, np.inf)
    best[0] = 0.0
    for M in range(1, full):
        low = M & -M
        S = M
        while S:
            if S & low and np.isfinite(route[S]):
                best[M] = min(best[M], route[S] + best[M ^ S])
            S = (S - 1) & M
    return float(best[full - 1])


def run(cls, path: Path, inst, **kw):
    seen, bounds = [], []
    out = cls(time_limit=kw.pop("time_limit", 30), on_incumbent=lambda r, **_: seen.append(route_cost(inst, r)),
              on_bound=bounds.append, **kw).run(str(path))
    return out, seen, bounds


# ---------------------------------------------------------------- conventions, evaluator, verifier

@needs_raw
def test_evaluator_reproduces_every_published_x_solution(tmp_path):
    files = sorted(RAW.glob("X-n*.vrp"))
    assert files
    for f in files:
        inst = load_vrp(f)
        routes, cost = load_sol(f.with_suffix(".sol"))
        assert route_cost(inst, routes) == cost, f.name
        out = tmp_path / "out.json"
        out.write_text(json.dumps(cvrp_output(inst, routes)))
        rep = verify_solution(f, out)
        assert rep["ok"] and rep["objective_value"] == cost, (f.name, rep["issues"])


@needs_raw
@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_published_values_and_unlimited_fleet(name):
    inst = load_vrp(RAW / f"{name}.vrp")
    routes, cost = load_sol(RAW / f"{name}.sol")
    assert (cost, len(routes)) == PUBLISHED[name]
    assert inst.n == int(name.split("-n")[1].split("-")[0]) - 1
    assert inst.demand[0] == 0 and (inst.demand[1:] > 0).all()
    k = int(name.split("-k")[1])
    assert inst.min_vehicles <= k <= len(routes)  # k is a minimum; the best-known plan may use more routes


def test_loader_numbers_customers_in_file_order_around_the_depot(tmp_path):
    coords = [(10, 0), (0, 0), (20, 0), (0, 30)]
    demand = [3, 0, 4, 5]
    path = write_vrp(tmp_path / "dep2.vrp", coords, demand, 9, depot=2)
    inst = load_vrp(path)
    assert inst.n == 3 and inst.coords[0].tolist() == [0, 0]
    assert inst.demand.tolist() == [0, 3, 4, 5]  # customers 1, 2, 3 = nodes 1, 3, 4 of the file
    plan = [[1, 2], [3]]
    assert route_cost(inst, plan) == 10 + 10 + 20 + 30 + 30
    out = tmp_path / "out.json"
    out.write_text(json.dumps(cvrp_output(inst, plan)))
    assert verify_solution(path, out)["ok"]


def test_nearest_integer_rounding(tmp_path):
    path = write_vrp(tmp_path / "round.vrp", [(0, 0), (1, 1), (2, 1)], [0, 1, 1], 5)  # sqrt(2) -> 1, sqrt(5) -> 2
    inst = load_vrp(path)
    assert inst.distances().tolist() == [[0, 1, 2], [1, 0, 1], [2, 1, 0]]
    assert route_cost(inst, [[1, 2]]) == 1 + 1 + 2


def test_route_cost_and_verifier_reject_invalid_plans(tmp_path):
    path = write_vrp(tmp_path / "v.vrp", [(0, 0), (3, 4), (6, 8), (0, 5)], [0, 4, 4, 4], 8)
    inst = load_vrp(path)
    good = [[1, 2], [3]]
    assert route_cost(inst, good) == 5 + 5 + 10 + 5 + 5
    bad_plans = [[[1, 2]], [[1, 2], [3], [3]], [[1, 2, 3]], [[1, 2], []], [[1, 2], [4]], [[0, 1, 2], [3]], [[1, 2], [3.0]]]
    for plan in bad_plans:
        assert route_cost(inst, plan) is None, plan
        assert route_issues(inst, plan)[1]
    out = tmp_path / "out.json"

    def check(payload) -> bool:
        out.write_text(json.dumps(payload) if not isinstance(payload, str) else payload)
        return verify_solution(path, out)["ok"]

    base = {"status": "feasible", "method": "decomposition", "objective_value": 30, "routes": good}
    assert check(base)
    assert check({**base, "status": "optimal"})
    for plan in bad_plans[:5]:
        assert not check({**base, "routes": plan})
    assert not check({**base, "routes": [[0, 1, 2], [3]]})
    assert not check({**base, "routes": [[1, 2], [3.0]]})
    assert not check({**base, "routes": [[1, 2], [True]]})
    assert not check({**base, "objective_value": 29})
    assert not check({**base, "objective_value": True})
    assert not check({**base, "objective_value": "30"})
    assert not check('{"status": "feasible", "method": "x", "objective_value": NaN, "routes": [[1, 2], [3]]}')
    assert not check('{"status": "feasible", "method": "x", "objective_value": Infinity, "routes": [[1, 2], [3]]}')
    assert not check({k: v for k, v in base.items() if k != "method"})
    assert not check({k: v for k, v in base.items() if k != "status"})
    assert not check({**base, "status": "infeasible"})
    assert not check({k: v for k, v in base.items() if k != "routes"})
    assert not check("[1, 2]")
    assert not check("not json")


def test_verifier_imports_only_the_standard_library():
    tree = ast.parse((Path(__file__).resolve().parents[1] / "src" / "cg" / "cvrp" / "e_verifier.py").read_text())
    mods = {a.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names}
    mods |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert mods <= {"__future__", "json", "math", "pathlib", "typing"}


def test_initializer_repair_and_local_search_keep_plans_feasible(tmp_path):
    for seed in range(6):
        path = random_instance(tmp_path, 300 + seed, n_range=(12, 30))
        inst = load_vrp(path)
        dist = inst.distances()
        init = basic_initializer(inst, dist)
        c0 = route_cost(inst, init)
        assert c0 is not None
        better = CVRPLocalSearch(inst, dist).improve(init, deadline=float("inf"))
        assert route_cost(inst, better) is not None and route_cost(inst, better) <= c0
        # A covering plan (every route of the plan plus a copy of one customer elsewhere) is repaired to a partition.
        cover = [list(r) for r in init] + [[init[0][0]]]
        assert route_cost(inst, repair_cover(cover, dist)) is not None


# ---------------------------------------------------------------- compact MIP

@pytest.mark.parametrize("formulation", ["flow", "two_index"])
def test_integer_check_decodes_routes_and_finds_violations(tmp_path, formulation):
    path = write_vrp(tmp_path / "ic.vrp", [(0, 0), (10, 0), (20, 0), (0, 10), (0, 20), (5, 5)], [0, 3, 3, 3, 3, 3], 7)
    inst = load_vrp(path)
    model, info = build_cvrp_mip(inst, inst.distances(), formulation=formulation)
    U, V = info["U"], info["V"]
    pos = {(int(u), int(v)): a for a, (u, v) in enumerate(zip(U.tolist(), V.tolist()))}

    def vector(walks):
        x = np.zeros(len(U))
        for w in walks:
            for a, b in zip(w, w[1:]):
                x[pos[(a, b)] if formulation == "flow" else pos[(min(a, b), max(a, b))]] += 1
        return x

    routes, bad = integer_check(inst, info, vector([[0, 1, 2, 0], [0, 3, 4, 0], [0, 5, 0]]))
    assert not bad and sorted(map(sorted, routes)) == [[1, 2], [3, 4], [5]]
    assert route_cost(inst, routes) == route_cost(inst, [[1, 2], [3, 4], [5]])
    # A subtour 3-4-3 detached from the depot is reported (and so is customer 5, left without edges).
    routes, bad = integer_check(inst, info, vector([[0, 1, 2, 0], [3, 4, 3]]))
    assert routes is None and any(set(np.nonzero(S)[0]) == {3, 4} for S in bad)
    # A route over capacity (load 9 > 7) is reported as its customer set.
    routes, bad = integer_check(inst, info, vector([[0, 1, 2, 0], [0, 3, 4, 5, 0]]))
    assert routes is None and [set(np.nonzero(S)[0]) for S in bad] == [{3, 4, 5}]
    model.dispose()


@pytest.mark.parametrize("formulation", ["flow", "two_index"])
@pytest.mark.parametrize("seed", range(6))
def test_compact_mip_matches_brute_force(tmp_path, formulation, seed):
    path = random_instance(tmp_path, seed)
    inst = load_vrp(path)
    opt = brute_force(inst)
    out, seen, bounds = run(CVRPCompactMIP, path, inst, formulation=formulation)
    assert out["termination_reason"] == "optimal"
    assert out["best_objective"] == pytest.approx(opt)
    assert seen and all(v is not None and v >= opt - 1e-6 for v in seen)
    assert all(b <= opt + 1e-6 for b in bounds) and out["global_lb"] == pytest.approx(opt)
    assert route_cost(inst, out["routes"]) == out["best_objective"]


# ---------------------------------------------------------------- naive column generation

@pytest.mark.parametrize("seed", range(8))
def test_naive_cg_bounds_and_plans(tmp_path, seed):
    path = random_instance(tmp_path, 100 + seed)
    inst = load_vrp(path)
    opt = brute_force(inst)
    out, seen, bounds = run(CVRPNaiveCG, path, inst, time_limit=20)
    assert out["cg_converged"] and out["cg_stop_reason"] == "converged"
    assert seen and all(v is not None and v >= opt - 1e-6 for v in seen)
    assert len(bounds) == 1 and bounds[0] <= opt + 1e-6 and out["global_lb"] == bounds[0]
    assert out["best_objective"] >= opt - 1e-6 and out["rmp_status"] == gp.GRB.OPTIMAL
    assert route_cost(inst, out["routes"]) == out["best_objective"]


def test_naive_cg_reaches_the_optimum_on_most_tiny_instances(tmp_path):
    """The integer restricted master over the root columns is a heuristic (protocol: "root CG + integer RMP"), so the
    optimum is reached only when its routes were generated at the root: 6 of these 8 instances on Phoenix (Gurobi
    13.0.1); the RMP itself is solved to optimality on all of them (test above)."""
    hits = 0
    for seed in range(8):
        path = random_instance(tmp_path, 100 + seed)
        inst = load_vrp(path)
        out, _, _ = run(CVRPNaiveCG, path, inst, time_limit=20)
        hits += out["best_objective"] == pytest.approx(brute_force(inst))
    assert hits >= 5


def test_naive_cg_discards_incomplete_pricing_rounds(tmp_path, monkeypatch):
    import src.cg.cvrp.c_naive_cg as naive

    path = random_instance(tmp_path, 7)
    real = naive.NGPricer.price

    def truncated(self, *args, **kwargs):
        res = real(self, *args, **kwargs)
        return {**res, "complete": False}  # as if a resource guard had cut the round short

    monkeypatch.setattr(naive.NGPricer, "price", truncated)
    out = naive.CVRPNaiveCG(time_limit=10).run(str(path))
    assert out["cg_stop_reason"] == "pricing_resource_limit"
    assert out["cg_iterations"] == 1 and not out["cg_converged"] and out["global_lb"] is None
    assert out["best_objective"] is not None  # the shared initializer's plan


# ---------------------------------------------------------------- engineered branch-and-price

@pytest.mark.parametrize("seed", range(10))
def test_engineered_reference_matches_brute_force(tmp_path, seed):
    path = random_instance(tmp_path, 10 + seed)
    inst = load_vrp(path)
    opt = brute_force(inst)
    # Small ng-neighbourhoods on half of the seeds, so non-elementary columns enter the master.
    out, seen, bounds = run(CVRPColumnGenerationXL, path, inst, ng=3 if seed % 2 else 8)
    assert seen and all(v is not None and v >= opt - 1e-6 for v in seen)
    assert all(b <= opt + 1e-6 for b in bounds) and bounds == sorted(bounds)
    assert out["termination_reason"] == "optimal"
    assert out["best_objective"] == pytest.approx(opt) and out["global_lb"] <= opt + 1e-6
    assert route_cost(inst, out["routes"]) == out["best_objective"]


def test_engineered_branch_and_price_alone_proves_optimality(tmp_path, monkeypatch):
    """No initializer, restricted MIPs or dives: incumbents can only come from integral nodes, so the cuts, the edge
    branching, the node bounds and the pruning are exercised on their own (RCCs off on half of the instances, so that
    several of them branch)."""
    monkeypatch.setattr(CVRPColumnGenerationXL, "construct", lambda self: None)
    monkeypatch.setattr(CVRPColumnGenerationXL, "restricted_mip", lambda self, *a, **k: None)
    branched = 0
    for seed in range(10):
        path = random_instance(tmp_path, 40 + seed, n_range=(8, 11))
        inst = load_vrp(path)
        opt = brute_force(inst)
        out, seen, bounds = run(CVRPColumnGenerationXL, path, inst, dive=False, rcc=seed % 2 == 0,
                                heuristic_pricing=bool(seed % 3), alpha=0.0 if seed % 4 == 0 else 0.5)
        assert all(v is not None and v >= opt - 1e-6 for v in seen), seed
        assert all(b <= opt + 1e-6 for b in bounds) and bounds == sorted(bounds), seed
        assert out["termination_reason"] == "optimal" and out["best_objective"] == pytest.approx(opt), seed
        branched += out["nodes"] > 1
    assert branched >= 2


def test_master_reduced_costs_match_the_pricing_arc_costs(tmp_path, monkeypatch):
    """With cuts and edge-branching rows in the master, the reduced cost of every column computed from the pricing arc
    costs (cut and edge duals moved onto arcs) equals Gurobi's reduced cost: the pricing problem and the Lagrangian
    bounds use the master's true duals."""
    real = CVRPColumnGenerationXL.price
    seen = {"checks": 0, "with_cuts": 0, "with_edges": 0}

    def checked(self, D, exact, banned=None):
        if (banned is None or not banned.any()) and self.master.Status == gp.GRB.OPTIMAL:
            Cc, c_out, c_in = self.arc_costs(D)
            arcs = (Cc, c_out, c_in, D["pi"], -float(D["sigma"]))
            rc = self.master.getAttr("RC", [c["var"] for c in self.cols])
            for col, r in zip(self.cols, rc):
                mine = self.reduced_cost(col["path"], arcs)
                if math.isfinite(mine):
                    assert mine == pytest.approx(r, abs=1e-5 * max(1.0, abs(r)))
            seen["checks"] += 1
            seen["with_cuts"] += bool(len(D["beta"]))
            seen["with_edges"] += bool(len(D["eta"]))
        return real(self, D, exact, banned)

    monkeypatch.setattr(CVRPColumnGenerationXL, "price", checked)
    monkeypatch.setattr(CVRPColumnGenerationXL, "construct", lambda self: None)
    monkeypatch.setattr(CVRPColumnGenerationXL, "restricted_mip", lambda self, *a, **k: None)
    for seed in range(8):
        path = random_instance(tmp_path, 60 + seed, n_range=(10, 14))
        CVRPColumnGenerationXL(time_limit=10, alpha=0.0, dive=False, rcc=seed % 2 == 0).run(str(path))
    assert seen["checks"] > 0 and seen["with_cuts"] > 0 and seen["with_edges"] > 0


def test_engineered_reports_incumbent_provenance(tmp_path):
    path = random_instance(tmp_path, 3)
    stages = []
    CVRPColumnGenerationXL(time_limit=10, on_incumbent=lambda routes, stage=None: stages.append(stage)).run(str(path))
    assert stages and stages[0] == "initializer" and all(isinstance(s, str) for s in stages)
