"""Correctness checks for the bin packing with conflicts family (scale-up study): the literature file conventions, the
Sadykov-Vanderbeck-recipe generator, the evaluator and the independent verifier, the shared heuristics, the exact
pricing oracle against enumeration, the exactness of the compact MIP (packings fixed into it are accepted iff the
verifier accepts them), and the compact MIP, the naive and the engineered column generation against brute force on
tiny instances."""
from __future__ import annotations

import ast
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

gp = pytest.importorskip("gurobipy")
from gurobipy import GRB  # noqa: E402

from src.cg.bppc.a_data import (LIT_CLASSES, BPPCInstance, arbitrary_instance, format_bppc, load_bppc,  # noqa: E402
                                parse_bppc, write_bppc)
from src.cg.bppc.b_mip_xl import (BPPCCompactMIP, bin_count, bppc_output, build_bppc_mip, canonical_bins,  # noqa: E402
                                  mip_layout, packing_issues, public)
from src.cg.bppc.c_naive_cg import BPPCNaiveCG  # noqa: E402
from src.cg.bppc.d_ref_cg_xl import BPPCBranchAndPrice, NodeView  # noqa: E402
from src.cg.bppc.e_verifier import verify_solution  # noqa: E402
from src.cg.bppc.f_heuristic import BPPCHeuristicControl, TabuRepair  # noqa: E402
from src.cg.bppc.heuristic import basic_initializer, edge_clique_cover, first_fit, greedy_clique, trim_cover  # noqa: E402
from src.cg.bppc.pricing import KPCOracle, SmallBinOracle, greedy_bins, large_bins, max_items  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "outputs" / "scaleup" / "raw" / "bppc" / "Istanze"
needs_raw = pytest.mark.skipif(not (RAW / "BPPC_1_0_1.txt").exists(), reason="BPPC literature files not downloaded")


# ---------------------------------------------------------------- helpers

def make_instance(weights, C: int, edges, name: str = "tiny") -> BPPCInstance:
    n = len(weights)
    adj = np.zeros((n, n), dtype=bool)
    for i, j in edges:
        adj[i, j] = adj[j, i] = True
    return BPPCInstance(name, int(C), np.array(weights, dtype=np.int64), adj)


def random_instance(tmp_path: Path, seed: int, n_range=(5, 10), name: str = "tiny") -> tuple[Path, BPPCInstance]:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(*n_range))
    C = int(rng.integers(10, 25))
    w = rng.integers(1, C + 1, size=n)
    w = np.where(rng.random(n) < 0.7, np.minimum(w, C // 2 + 1), w)  # mostly small items, so bins hold several
    p = float(rng.uniform(0.0, 0.6))
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
    inst = make_instance(w.tolist(), C, edges, f"{name}{seed}")
    path = write_bppc(tmp_path / f"{name}{seed}.txt", inst)
    return path, load_bppc(path)


def pairs_instance(tmp_path: Path, seed: int) -> tuple[Path, BPPCInstance]:
    """Weights 5 or 6 and capacity 12: a bin holds at most two items, so the problem is a matching problem on the
    compatibility graph, whose covering LP is often fractional (odd cycles): the search has to branch."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(7, 11))
    w = rng.integers(5, 7, size=n)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.45]
    inst = make_instance(w.tolist(), 12, edges, f"pairs{seed}")
    path = write_bppc(tmp_path / f"pairs{seed}.txt", inst)
    return path, load_bppc(path)


def triplet_instance(tmp_path: Path, seed: int) -> tuple[Path, BPPCInstance]:
    """Falkenauer-like triplets (capacity 100, every third item completes its triple to 100) with a few random
    conflicts: at most three items per bin, so the engineered reference prices by enumeration."""
    rng = np.random.default_rng(seed)
    w = []
    for _ in range(3):
        a, b = int(rng.integers(26, 38)), int(rng.integers(26, 38))  # the third item weighs 26..48
        w += [a, b, 100 - a - b]
    edges = [(i, j) for i in range(9) for j in range(i + 1, 9) if rng.random() < 0.15]
    inst = make_instance(w, 100, edges, f"trip{seed}")
    path = write_bppc(tmp_path / f"trip{seed}.txt", inst)
    return path, load_bppc(path)


def feasible_sets(inst: BPPCInstance) -> tuple[list[bool], list[int]]:
    n = inst.n
    nb = [sum(1 << j for j in np.flatnonzero(inst.adj[i]).tolist()) for i in range(n)]
    full = 1 << n
    feas, wsum = [False] * full, [0] * full
    feas[0] = True
    for S in range(1, full):
        low = (S & -S).bit_length() - 1
        R = S ^ (1 << low)
        wsum[S] = wsum[R] + int(inst.weights[low])
        feas[S] = feas[R] and not (nb[low] & R) and wsum[S] <= inst.capacity
    return feas, wsum


def brute_force(inst: BPPCInstance) -> int:
    """Optimal number of bins: dynamic programming over item subsets (every partition block a feasible bin)."""
    n = inst.n
    feas, _ = feasible_sets(inst)
    full = 1 << n
    dp = [math.inf] * full
    dp[0] = 0
    for M in range(1, full):
        low = M & -M
        rest = M ^ low
        sub = rest
        best = math.inf
        while True:
            S = sub | low
            if feas[S] and dp[M ^ S] + 1 < best:
                best = dp[M ^ S] + 1
            if sub == 0:
                break
            sub = (sub - 1) & rest
        dp[M] = best
    return int(dp[full - 1])


def run(cls, path: Path, inst: BPPCInstance, **kw):
    seen, bounds = [], []
    out = cls(time_limit=kw.pop("time_limit", 30), on_incumbent=lambda b, **_: seen.append(bin_count(inst, b)),
              on_bound=bounds.append, **kw).run(str(path))
    return out, seen, bounds


def verified(path: Path, inst: BPPCInstance, bins, tmp_path: Path) -> float:
    out = tmp_path / "witness.json"
    out.write_text(json.dumps(bppc_output(inst, bins)))
    rep = verify_solution(path, out)
    assert rep["ok"], rep["issues"]
    return rep["objective_value"]


# ---------------------------------------------------------------- literature files and the generator

@needs_raw
def test_literature_files_follow_the_documented_conventions():
    files = sorted(RAW.glob("BPPC_*.txt"))
    assert len(files) == 800
    for c, (kind, n) in LIT_CLASSES.items():
        for k in (1, 7):
            weights = None
            dens = []
            for d in range(10):
                f = RAW / f"BPPC_{c}_{d}_{k}.txt"
                lines = f.read_text().split("\n")
                assert lines[0].split() == [str(n), "150" if kind == "u" else "1000"]
                for ln in lines[1:]:  # each conflict listed once, with the higher-numbered item
                    t = ln.split()
                    assert all(int(j) > int(t[0]) for j in t[2:])
                inst = load_bppc(f)
                assert inst.n == n and inst.capacity == (150 if kind == "u" else 1000)
                if weights is None:
                    weights = inst.weights
                assert (inst.weights == weights).all()  # the same items for every density label
                dens.append(inst.density)
            assert dens[0] == 0.0 and all(a <= b for a, b in zip(dens, dens[1:])) and dens[9] > 0.9
            if kind == "t":  # triplets fill n / 3 bins exactly
                assert int(weights.sum()) == n // 3 * 1000 and BPPCInstance("x", 1000, weights, np.zeros((n, n), bool)).weight_bound == n // 3
            else:
                assert weights.min() >= 20 and weights.max() <= 100


@needs_raw
def test_initializer_plans_on_literature_files_pass_the_verifier(tmp_path):
    for name in ("BPPC_1_3_1", "BPPC_4_9_2", "BPPC_6_5_3", "BPPC_8_0_4"):
        f = RAW / f"{name}.txt"
        inst = load_bppc(f)
        plan = public(basic_initializer(inst))
        assert bin_count(inst, plan) == len(plan)
        assert verified(f, inst, plan, tmp_path) == len(plan)
        assert len(plan) >= max(inst.weight_bound, len(greedy_clique(inst.adj)))


def test_parser_accepts_conflicts_listed_on_either_line_and_rejects_bad_files():
    a = parse_bppc("3 10\n1 4 2\n2 5\n3 6 1\n")
    assert a.n == 3 and a.capacity == 10 and a.weights.tolist() == [4, 5, 6]
    assert a.adj[0, 1] and a.adj[1, 0] and a.adj[0, 2] and a.adj[2, 0] and not a.adj[1, 2]
    b = parse_bppc(format_bppc(a))
    assert (b.adj == a.adj).all() and (b.weights == a.weights).all()
    for bad in ("3 10\n1 4\n2 5\n", "2 10\n1 4 1\n2 5\n", "2 10\n1 4 3\n2 5\n", "2 10\n1 11\n2 5\n", "2 10\n1 4\n1 5\n",
                "2 10\n1 0\n2 5\n"):
        with pytest.raises(ValueError):
            parse_bppc(bad)


def test_generator_follows_the_recipe_and_is_reproducible(tmp_path):
    a = arbitrary_instance("da", 120, 5, 3)
    b = arbitrary_instance("da", 120, 5, 3)
    assert format_bppc(a) == format_bppc(b)
    n = 120
    assert a.capacity == 10000 and a.weights.min() >= 500 and a.weights.max() <= 2500
    assert a.num_conflicts == math.ceil(0.5 * n * (n - 1) / 2)
    c = arbitrary_instance("da", 120, 7, 3)
    assert (c.weights == a.weights).all() and c.num_conflicts > a.num_conflicts  # same items, denser graph
    assert format_bppc(arbitrary_instance("da", 120, 5, 4)) != format_bppc(a)
    p = write_bppc(tmp_path / "da.txt", a)
    assert format_bppc(load_bppc(p)) == format_bppc(a)
    if (RAW / "BPPC_7_0_2.txt").exists():
        t = arbitrary_instance("ta", 249, 6, 2, RAW)
        assert (t.weights == load_bppc(RAW / "BPPC_7_0_2.txt").weights).all() and t.capacity == 1000
        assert t.num_conflicts == math.ceil(0.6 * 249 * 248 / 2)


# ---------------------------------------------------------------- evaluator and verifier

def test_evaluator_and_verifier_reject_invalid_plans(tmp_path):
    inst = make_instance([4, 5, 6, 3], 10, [(0, 1)])
    path = write_bppc(tmp_path / "v.txt", inst)
    good = [[1, 3], [2, 4]]
    assert bin_count(inst, good) == 2
    bad_plans = [[[1, 2], [3, 4]], [[1, 3], [2]], [[1, 3], [2, 4], [4]], [[1, 3, 4], [2]], [[1, 3], [2, 4], []],
                 [[1, 3], [2, 5]], [[0, 1, 3], [2, 4]], [[1, 3], [2, 4.0]], [[1, 3], [2, True]], [[1, 1, 3], [2, 4]]]
    for plan in bad_plans:
        assert bin_count(inst, plan) is None, plan
        assert packing_issues(inst, plan)[1]
    out = tmp_path / "out.json"

    def check(payload) -> bool:
        out.write_text(json.dumps(payload) if not isinstance(payload, str) else payload)
        return verify_solution(path, out)["ok"]

    base = {"status": "feasible", "method": "column_generation", "objective_value": 2, "bins": good}
    assert check(base) and check({**base, "status": "optimal"}) and check({**base, "objective_value": 2.0})
    for plan in bad_plans:
        assert not check({**base, "bins": plan}), plan
    assert not check({**base, "objective_value": 3})
    assert not check({**base, "objective_value": True})
    assert not check({**base, "objective_value": "2"})
    assert not check('{"status": "feasible", "method": "x", "objective_value": NaN, "bins": [[1, 3], [2, 4]]}')
    assert not check('{"status": "feasible", "method": "x", "objective_value": Infinity, "bins": [[1, 3], [2, 4]]}')
    assert not check({k: v for k, v in base.items() if k != "method"})
    assert not check({**base, "status": "infeasible"})
    assert not check({k: v for k, v in base.items() if k != "bins"})
    assert not check("[1, 2]")
    assert not check("not json")
    # A conflict listed only on the higher-numbered item's line still counts.
    path2 = tmp_path / "v2.txt"
    path2.write_text("2 10\n1 4\n2 5 1\n")
    out.write_text(json.dumps({"status": "feasible", "method": "x", "objective_value": 1, "bins": [[1, 2]]}))
    assert not verify_solution(path2, out)["ok"]


def test_verifier_imports_only_the_standard_library():
    tree = ast.parse((ROOT / "src" / "cg" / "bppc" / "e_verifier.py").read_text())
    mods = {a.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names}
    mods |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert mods <= {"__future__", "json", "math", "pathlib", "typing"}


# ---------------------------------------------------------------- heuristics and cliques

def test_initializer_trim_and_cliques(tmp_path):
    for seed in range(8):
        _, inst = random_instance(tmp_path, 500 + seed, n_range=(15, 40))
        plan = basic_initializer(inst)
        assert bin_count(inst, public(plan)) == len(plan)
        cover = [list(b) for b in plan] + [[plan[0][0]]] + [plan[-1][:1]]
        assert bin_count(inst, public(trim_cover(cover))) is not None
        cl = edge_clique_cover(inst.adj)
        covered = np.zeros_like(inst.adj)
        for K in cl:
            assert len(K) >= 2
            sub = inst.adj[np.ix_(K, K)]
            assert sub[~np.eye(len(K), dtype=bool)].all()  # a clique
            covered[np.ix_(K, K)] = True
        assert not (inst.adj & ~covered).any()  # every edge covered
        K0 = greedy_clique(inst.adj, inst.weights)
        assert inst.adj[np.ix_(K0, K0)][~np.eye(len(K0), dtype=bool)].all()


# ---------------------------------------------------------------- pricing

def test_pricing_oracle_and_greedy_match_enumeration(tmp_path):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    rng = np.random.default_rng(7)
    checked_pos = checked_none = 0
    for seed in range(12):
        _, inst = random_instance(tmp_path, 900 + seed, n_range=(6, 11))
        feas, _ = feasible_sets(inst)
        oracle = KPCOracle(inst.weights, inst.capacity, inst.adj, edge_clique_cover(inst.adj), env)
        for _ in range(6):
            p = np.where(rng.random(inst.n) < 0.8, rng.uniform(0, 0.7, inst.n), 0.0)
            best = max(sum(p[i] for i in range(inst.n) if S >> i & 1) for S in range(1 << inst.n) if feas[S])
            res = oracle.solve(p, max_cols=5)
            assert res["complete"] and res["bound"] is not None and res["bound"] >= best - 1e-6
            for val, items in res["cols"]:
                S = sum(1 << i for i in items)
                assert feas[S] and val == pytest.approx(p[items].sum()) and val > 1 + 1e-6
            if best > 1 + 1e-5:
                assert res["cols"] and res["cols"][0][0] == pytest.approx(best)
                checked_pos += 1
            else:
                assert not res["cols"]
                checked_none += 1
            for val, items in greedy_bins(p, inst.weights, inst.capacity, inst.adj, threshold=1 + 1e-6):
                assert feas[sum(1 << i for i in items)] and val <= best + 1e-9
        oracle.dispose()
    env.dispose()
    assert checked_pos > 5 and checked_none > 5


def test_small_bin_enumeration_matches_the_mip_oracle_and_enumeration(tmp_path):
    """Triplet-like instances (few bins of four or more items, all listed): the enumeration oracle returns the exact
    optimum and a valid bound, also with banned items and thresholds above the optimum."""
    rng = np.random.default_rng(3)
    for seed in range(20):
        n = int(rng.integers(6, 13))
        C = 100
        w = rng.integers(26, 51, size=n)
        if seed % 2:  # a few items of weight 25: bins of four (and five with lighter ones) become possible
            w[:int(rng.integers(4, 7))] = rng.integers(18, 26)
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < rng.uniform(0, 0.5)]
        inst = make_instance(w.tolist(), C, edges)
        big = large_bins(inst.weights, C, inst.adj)
        assert big is not None and (seed % 2 == 1 or max_items(inst.weights, C) <= 3)
        feas_all, _ = feasible_sets(inst)
        listed = {sum(1 << int(i) for i in b) for b in big}
        assert listed == {S for S in range(1 << n) if feas_all[S] and bin(S).count("1") >= 4}
        feas, _ = feasible_sets(inst)
        orc = SmallBinOracle(inst.weights, C, inst.adj)
        for trial in range(8):
            p = np.where(rng.random(n) < 0.85, rng.uniform(0, 0.6, n), 0.0)
            banned = rng.random(n) < 0.2 if trial % 2 else None
            pe = p if banned is None else np.where(banned, 0.0, p)
            best = max(sum(pe[i] for i in range(n) if S >> i & 1) for S in range(1 << n) if feas[S])
            thr = [1.0 + 1e-6, 0.3, 5.0][trial % 3]
            res = orc.solve(p, threshold=thr, max_cols=4, banned=banned)
            assert res["complete"] and res["bound"] >= best - 1e-9
            if best > thr + 1e-9:
                assert res["cols"] and res["cols"][0][0] == pytest.approx(best) and res["bound"] == pytest.approx(best)
            for val, items in res["cols"]:
                assert feas[sum(1 << i for i in items)] and val > thr and val == pytest.approx(pe[items].sum())
                assert banned is None or not banned[items].any()


# ---------------------------------------------------------------- compact MIP

def test_mip_accepts_a_packing_iff_the_verifier_accepts_it(tmp_path):
    """Exactness: every packing with at most U bins, fixed into the model through its canonical labels, is feasible
    for the model iff the independent verifier accepts it, with the bin count as objective."""
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    rng = np.random.default_rng(11)
    accepted = rejected = 0
    for seed in range(10):
        path, inst = random_instance(tmp_path, 700 + seed, n_range=(6, 10))
        U = inst.n
        model, info = build_bppc_mip(inst, U, env)
        lay, v, nx = info["layout"], info["v"], info["nx"]
        xs = v.tolist()
        for trial in range(25):
            k = int(rng.integers(1, inst.n + 1))
            lab = rng.integers(0, k, size=inst.n)
            if trial % 3 == 0:  # also start from feasible plans
                plan = basic_initializer(inst)
                lab = np.zeros(inst.n, dtype=int)
                for t, b in enumerate(plan):
                    lab[b] = t
            bins = [np.flatnonzero(lab == t).tolist() for t in range(lab.max() + 1)]
            bins = [b for b in bins if b]
            canon = canonical_bins(lay, bins)
            out = tmp_path / "p.json"
            out.write_text(json.dumps({"status": "feasible", "method": "t", "objective_value": len(bins), "bins": public(bins)}))
            ok = verify_solution(path, out)["ok"]
            if canon is None:  # two clique items share a bin: the packing is infeasible
                assert not ok
                rejected += 1
                continue
            assert all(canon[i] in lay["allowed"][i] for i in range(inst.n))  # the representation exists
            val = np.zeros(nx)
            for i, b in canon.items():
                val[info["col"][(i, b)]] = 1.0
            for kk in range(nx):
                xs[kk].LB = xs[kk].UB = val[kk]
            model.optimize()
            feasible = model.Status == GRB.OPTIMAL
            assert feasible == ok, (seed, trial, bins)
            if feasible:
                assert round(model.ObjVal) == len(bins)
                accepted += 1
            else:
                rejected += 1
        model.dispose()
    env.dispose()
    assert accepted >= 10 and rejected >= 10


@pytest.mark.parametrize("seed", range(6))
def test_compact_mip_matches_brute_force(tmp_path, seed):
    path, inst = random_instance(tmp_path, seed)
    opt = brute_force(inst)
    out, seen, bounds = run(BPPCCompactMIP, path, inst)
    assert out["termination_reason"] == "optimal" and out["gurobi_status"] == GRB.OPTIMAL
    assert not out["budget_exhausted"]
    assert out["best_objective"] == opt and out["global_lb"] == opt
    assert all(s is not None and s >= opt for s in seen)
    assert all(b <= opt for b in bounds) and bounds == sorted(bounds)
    assert verified(path, inst, out["bins"], tmp_path) == opt


def test_mip_reports_deadline_during_build(tmp_path):
    path, inst = random_instance(tmp_path, 3)
    out = BPPCCompactMIP(time_limit=1e-6).run(str(path))
    assert out["termination_reason"] == "deadline_during_build" and out["best_objective"] is None
    assert out["budget_exhausted"] and out["gurobi_status"] is None


def test_mip_start_arm_publishes_the_initializer(tmp_path):
    path, inst = random_instance(tmp_path, 4)
    out, seen, _ = run(BPPCCompactMIP, path, inst, start=True)
    assert seen[0] == len(basic_initializer(inst)) and out["best_objective"] == brute_force(inst)


# ---------------------------------------------------------------- naive column generation

@pytest.mark.parametrize("seed", range(6))
def test_naive_cg_bounds_and_plans(tmp_path, seed):
    path, inst = random_instance(tmp_path, 100 + seed)
    opt = brute_force(inst)
    out, seen, bounds = run(BPPCNaiveCG, path, inst, time_limit=20)
    assert out["cg_converged"] and out["cg_stop_reason"] == "converged"
    assert all(s is not None and s >= opt for s in seen)
    assert len(bounds) == 1 and bounds[0] <= opt and out["global_lb"] == bounds[0]
    assert out["best_objective"] >= opt and out["rmp_status"] == GRB.OPTIMAL
    assert out["termination_reason"] == "finished_early"
    assert verified(path, inst, out["bins"], tmp_path) == out["best_objective"]


def test_naive_cg_discards_incomplete_pricing_rounds(tmp_path, monkeypatch):
    import src.cg.bppc.c_naive_cg as naive

    path, inst = random_instance(tmp_path, 7, n_range=(8, 10))
    real = naive.KPCOracle.solve

    def truncated(self, *args, **kwargs):
        res = real(self, *args, **kwargs)
        return {**res, "complete": False}  # as if the deadline had cut the round short

    monkeypatch.setattr(naive.KPCOracle, "solve", truncated)
    out = naive.BPPCNaiveCG(time_limit=10).run(str(path))
    assert out["cg_stop_reason"] == "pricing_incomplete"
    assert out["cg_iterations"] == 1 and not out["cg_converged"] and out["global_lb"] is None
    assert out["best_objective"] is not None  # the shared initializer's plan


# ---------------------------------------------------------------- engineered branch-and-price

@pytest.mark.parametrize("seed", range(8))
def test_engineered_reference_matches_brute_force(tmp_path, seed):
    path, inst = random_instance(tmp_path, 10 + seed, n_range=(7, 11))
    opt = brute_force(inst)
    out, seen, bounds = run(BPPCBranchAndPrice, path, inst)
    assert out["termination_reason"] == "optimal"
    assert out["best_objective"] == opt and out["global_lb"] == opt
    assert all(s is not None and s >= opt for s in seen)
    assert all(b <= opt for b in bounds) and bounds == sorted(bounds)
    assert verified(path, inst, out["bins"], tmp_path) == opt


def test_branch_and_price_alone_proves_optimality(tmp_path):
    """No initializer, dives or restricted MIPs: incumbents come only from integral nodes, so the Ryan-Foster
    branching through merged items and added conflicts, the Farley bounds, the early termination and the pruning are
    exercised on their own (with and without smoothing, heuristic pricing and early termination)."""
    branched = 0
    for seed in range(18):
        path, inst = (random_instance(tmp_path, 40 + seed, n_range=(8, 11)) if seed < 4
                      else pairs_instance(tmp_path, 60 + seed) if seed < 12 else triplet_instance(tmp_path, 80 + seed))
        opt = brute_force(inst)
        out, seen, bounds = run(BPPCBranchAndPrice, path, inst, initializer=False, dive=False, restricted_mip=False,
                                heuristic_pricing=bool(seed % 3), alpha=0.0 if seed % 4 == 0 else 0.5,
                                early_termination=seed % 2 == 0)
        assert out["termination_reason"] == "optimal" and out["best_objective"] == opt, seed
        assert all(s is not None and s >= opt for s in seen), seed
        assert all(b <= opt for b in bounds) and bounds == sorted(bounds), seed
        branched += out["nodes"] > 1
    assert branched >= 3


def test_node_view_merges_items_and_adds_conflicts():
    inst = make_instance([3, 4, 5, 2, 6], 12, [(0, 2), (3, 4)])
    bp = BPPCBranchAndPrice()
    bp.n, bp.w, bp.C, bp.adj = inst.n, inst.weights, inst.capacity, inst.adj
    bp.cliques = edge_clique_cover(inst.adj)
    nv = NodeView(bp, {"same": ((0, 1),), "differ": ((1, 3),)})
    assert nv.feasible and nv.m == 4 and nv.members[0] == [0, 1] and nv.W.tolist() == [7, 5, 2, 6]
    g = {i: int(nv.gid[i]) for i in range(5)}
    assert nv.adj[g[0], g[2]] and nv.adj[g[0], g[3]] and nv.adj[g[3], g[4]]  # conflicts united, differ added
    assert not nv.adj[g[0], g[4]] and not nv.adj[g[2], g[3]]
    covered = np.zeros_like(nv.adj)
    for K in nv.cliques:
        assert nv.adj[np.ix_(K, K)][~np.eye(len(K), dtype=bool)].all()
        covered[np.ix_(K, K)] = True
    assert not (nv.adj & ~covered).any()
    mask = lambda items: sum(1 << i for i in items)
    assert nv.valid(mask([0, 1, 4])) and not nv.valid(mask([0, 4])) and not nv.valid(mask([0, 1, 3]))
    assert nv.expand(nv.to_super([0, 1, 4])) == (0, 1, 4)
    assert not NodeView(bp, {"same": ((0, 2),), "differ": ()}).feasible  # merging a conflicting pair
    assert not NodeView(bp, {"same": ((0, 1),), "differ": ((0, 1),)}).feasible


def test_engineered_labels_an_early_exit_with_only_stuck_nodes(tmp_path, monkeypatch):
    path = write_bppc(tmp_path / "odd.txt", make_instance([5, 5, 5], 10, []))  # LP 1.5, optimum 2
    monkeypatch.setattr(BPPCBranchAndPrice, "branch", lambda self, node, x: None)
    out = BPPCBranchAndPrice(time_limit=20, initializer=False, dive=False, restricted_mip=False).run(str(path))
    assert out["termination_reason"] == "stuck" and out["stuck_nodes"] == 1
    assert out["global_lb"] == 2 and out["best_objective"] is None


def test_engineered_reports_incumbent_provenance(tmp_path):
    path, _ = random_instance(tmp_path, 3, n_range=(9, 11))
    stages = []
    BPPCBranchAndPrice(time_limit=10, on_incumbent=lambda bins, stage=None: stages.append(stage)).run(str(path))
    assert stages and stages[0] == "initializer" and all(isinstance(s, str) for s in stages)


def test_engineered_never_publishes_after_the_deadline(tmp_path):
    path, inst = random_instance(tmp_path, 5, n_range=(9, 11))
    times = []
    bp = BPPCBranchAndPrice(time_limit=0.05, on_incumbent=lambda bins, **_: times.append(bp.elapsed()))
    out = bp.run(str(path))
    assert all(t < 0.05 + 2e-3 for t in times)  # the hook's clock reads a moment after the solver's check
    assert out["termination_reason"] in ("optimal", "time_limit")


# ---------------------------------------------------------------- heuristic-only control

def test_heuristic_control_publishes_feasible_packings_and_mostly_reaches_the_optimum(tmp_path):
    """The tabu-search control (no decomposition): every publication is a feasible, strictly improving packing; the
    trivial bound it publishes is valid; it reaches the brute-force optimum on most tiny instances (3 s each)."""
    hits = 0
    for seed in range(12):
        path, inst = (random_instance(tmp_path, 200 + seed, n_range=(7, 11)) if seed % 3 == 0
                      else pairs_instance(tmp_path, 300 + seed) if seed % 3 == 1 else triplet_instance(tmp_path, 400 + seed))
        opt = brute_force(inst)
        out, seen, bounds = run(BPPCHeuristicControl, path, inst, time_limit=3, seed=seed)
        assert seen and all(s is not None and s >= opt for s in seen), seed
        assert all(a > b for a, b in zip(seen, seen[1:])), seed
        assert out["best_objective"] == seen[-1] and verified(path, inst, out["bins"], tmp_path) == seen[-1]
        assert bounds == [out["global_lb"]] and bounds[0] <= opt
        assert out["termination_reason"] == ("finished_early" if seen[-1] == bounds[0] else "time_limit")
        hits += out["best_objective"] == opt
    assert hits >= 10


def test_heuristic_control_respects_the_deadline_and_stops_at_the_trivial_bound(tmp_path):
    path, inst = random_instance(tmp_path, 901, n_range=(80, 81))
    T, times = 1.5, []
    t0 = time.perf_counter()
    out = BPPCHeuristicControl(time_limit=T, on_incumbent=lambda b: times.append(time.perf_counter() - t0)).run(str(path))
    assert times and all(t < T + 0.05 for t in times)  # the hook's clock starts a moment before the solver's
    assert out["total_seconds"] < T + 0.5 and out["termination_reason"] in ("time_limit", "finished_early")
    # Perfect triplets without conflicts: the weight bound n / 3 is reached, so the control stops early.
    rng = np.random.default_rng(5)
    w = []
    for _ in range(8):
        a, b = int(rng.integers(26, 38)), int(rng.integers(26, 38))
        w += [a, b, 100 - a - b]
    path2 = write_bppc(tmp_path / "perfect.txt", make_instance(w, 100, []))
    out2 = BPPCHeuristicControl(time_limit=30).run(str(path2))
    assert out2["termination_reason"] == "finished_early" and out2["best_objective"] == 8
    assert out2["total_seconds"] < 30


def test_tabu_repair_tracks_the_penalty_exactly(tmp_path):
    """Self-check mode: after every move the tracked penalty, loads and conflict counts equal their recomputation;
    a repaired assignment is a feasible packing."""
    for seed in range(6):
        _, inst = random_instance(tmp_path, 950 + seed, n_range=(25, 40))
        rng = np.random.default_rng(seed)
        tr = TabuRepair(inst, rng, check=True)
        plan = basic_initializer(inst)
        if len(plan) < 2:
            continue
        drop = int(np.argmin([inst.weights[b].sum() for b in plan]))
        for rich in (True, False):  # Capua's neighbourhoods (with Swap2vs1 / Swap1vs2), and Relocate + Swap only
            res = tr.repair(tr.start(plan, drop), len(plan) - 1, time.perf_counter() + 5, stall=300, rich=rich)
            if res is not None:
                bins = [np.flatnonzero(res == t).tolist() for t in range(len(plan) - 1)]
                assert bin_count(inst, public([b for b in bins if b])) is not None
        assert tr.log and tr.iterations > 0


def test_pricing_bounds_keep_tiny_positive_profits():
    """Review 34: an oracle that drops tiny positive profits reports a bound below the true pricing optimum, and
    Farley's bound sum(pi) / bound then exceeds the optimum. Both exact oracles must bound the full profit vector."""
    import math
    gp = pytest.importorskip("gurobipy")
    from src.cg.bppc.heuristic import edge_clique_cover
    from src.cg.bppc.pricing import KPCOracle, SmallBinOracle
    # 21 unit items, capacity 21, no conflicts: one bin holds everything, so the packing optimum is 1.
    n = 21
    w, adj = np.ones(n, dtype=np.int64), np.zeros((n, n), dtype=bool)
    p = np.array([1.0] + [9e-7] * (n - 1))
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    res = KPCOracle(w, n, adj, edge_clique_cover(adj), env).solve(p)
    assert res["bound"] is not None and res["bound"] >= p.sum() - 1e-12
    assert math.ceil(p.sum() / max(1.0, res["bound"] * (1 + 1e-9) + 1e-9) - 1e-6) <= 1
    env.dispose()
    # Triplet-size bins for the enumeration oracle: capacity 3, unit items.
    q = np.array([1.0, 4e-10, 4e-10, 0.2, 0.1, 0.05])
    orc = SmallBinOracle(np.ones(6, dtype=np.int64), 3, np.zeros((6, 6), dtype=bool))
    res = orc.solve(q)
    best = max(q[list(c)].sum() for k in (1, 2, 3) for c in __import__("itertools").combinations(range(6), k))
    assert res["bound"] is not None and res["bound"] >= best - 1e-12
