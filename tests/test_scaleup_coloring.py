"""Correctness checks for the vertex-coloring family (scale-up study): the DIMACS loaders (ASCII and binary) against
each other, the evaluator and the independent verifier, the shared heuristics, and the compact MIP, the naive and the
engineered column generation against brute force on random graphs with at most 11 vertices."""
from __future__ import annotations

import ast
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

gp = pytest.importorskip("gurobipy")

from src.cg.coloring.a_data import from_edges, load_col, write_col
from src.cg.coloring.b_mip_xl import ColoringCompactMIP, color_count, coloring_issues, coloring_output, start_labels
from src.cg.coloring.c_naive_cg import ColoringNaiveCG, farley_bound
from src.cg.coloring.d_ref_cg_xl import ColoringBranchAndPrice
from src.cg.coloring.e_verifier import read_graph, verify_solution
from src.cg.coloring.heuristic import (TabuCol, canonical, coloring_from_cover, dsatur, edge_clique_cover, greedy_clique,
                                       is_proper, maximal_extension, weighted_greedy_sets)

RAW = Path(__file__).resolve().parents[1] / "outputs" / "scaleup" / "raw" / "coloring"
needs_raw = pytest.mark.skipif(not (RAW / "dimacs1993" / "DSJC125.1.col.b").exists(), reason="DIMACS coloring files not downloaded")


# ---------------------------------------------------------------- helpers

def random_graph(tmp_path: Path, seed: int, n_range=(5, 11), p_range=(0.25, 0.7), name: str = "g") -> Path:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(*n_range))
    p = float(rng.uniform(*p_range))
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    return write_col(tmp_path / f"{name}{seed}.col", from_edges(f"{name}{seed}", n, np.array(pairs, dtype=np.int64).reshape(-1, 2)))


def cycle(tmp_path: Path, n: int) -> Path:
    return write_col(tmp_path / f"c{n}.col", from_edges(f"c{n}", n, [(i, (i + 1) % n) for i in range(n)]))


def grotzsch(tmp_path: Path) -> Path:
    """Mycielski graph of C5 (11 vertices, chromatic number 4, fractional chromatic number 2.9)."""
    edges = [(i, (i + 1) % 5) for i in range(5)]
    edges += [(5 + i, (i + 1) % 5) for i in range(5)] + [(5 + i, (i - 1) % 5) for i in range(5)]
    edges += [(10, 5 + i) for i in range(5)]
    return write_col(tmp_path / "grotzsch.col", from_edges("grotzsch", 11, edges))


def chromatic_number(inst) -> int:
    """Exact chromatic number by dynamic programming over vertex subsets (minimum cover by independent sets)."""
    n = inst.n
    if n == 0:
        return 0
    nbr = [0] * n
    for u, v in inst.edges.tolist():
        nbr[u] |= 1 << v
        nbr[v] |= 1 << u
    full = 1 << n
    indep = [False] * full
    indep[0] = True
    for S in range(1, full):
        low = (S & -S).bit_length() - 1
        rest = S & ~(1 << low)
        indep[S] = indep[rest] and not (nbr[low] & rest)
    best = [0] + [n + 1] * (full - 1)
    for S in range(1, full):
        low = S & -S
        T = S
        b = n + 1
        while T:
            if T & low and indep[T]:
                b = min(b, best[S ^ T] + 1)
            T = (T - 1) & S
        best[S] = b
    return best[full - 1]


def max_weight_independent_set(inst, w: np.ndarray) -> float:
    A = inst.adjacency
    best = 0.0
    for bits in itertools.product([0, 1], repeat=inst.n):
        S = np.array(bits, dtype=bool)
        if not (A[np.ix_(S, S)]).any():
            best = max(best, float(w[S].sum()))
    return best


def run(cls, path: Path, inst, **kw):
    seen, bounds = [], []
    out = cls(time_limit=kw.pop("time_limit", 30), on_incumbent=lambda c, **_: seen.append(color_count(inst, c)),
              on_bound=bounds.append, **kw).run(str(path))
    return out, seen, bounds


def write_binary(path: Path, n: int, edges) -> Path:
    """A .col.b file in the DIMACS binary format (preamble length, preamble, lower-triangle rows MSB first)."""
    pre = f"c binary test\np edge {n} {len(edges)}\n"
    rows = []
    adj = {(max(u, v), min(u, v)) for u, v in edges}
    for i in range(n):
        row = bytearray((i + 8) // 8)
        for j in range(i + 1):
            if (i, j) in adj:
                row[j >> 3] |= 0x80 >> (j & 7)
        rows.append(bytes(row))
    path.write_bytes(f"{len(pre)}\n".encode() + pre.encode() + b"".join(rows))
    return path


# ---------------------------------------------------------------- formats, evaluator, verifier

@needs_raw
@pytest.mark.parametrize("name", ["DSJC125.1", "DSJC125.5", "DSJC250.5", "DSJR500.5", "DSJR500.1c"])
def test_binary_1993_files_equal_the_color02_ascii_files(name):
    a = load_col(RAW / "dimacs1993" / f"{name}.col.b")
    b = load_col(RAW / "color02" / f"{name}.col")
    assert a.n == b.n and np.array_equal(a.edges, b.edges)
    n, E = read_graph(RAW / "dimacs1993" / f"{name}.col.b")
    assert n == a.n and E == set(map(tuple, a.edges.tolist()))


@needs_raw
@pytest.mark.parametrize("name,n,m", [("DSJC125.5", 125, 3891), ("DSJC250.5", 250, 15668), ("DSJR500.5", 500, 58862),
                                      ("flat300_28_0", 300, 21695), ("le450_15c", 450, 16680), ("le450_25c", 450, 17343),
                                      ("queen8_8", 64, 728), ("queen16_16", 256, 6320), ("school1", 385, 19095)])
def test_raw_instance_sizes(name, n, m):
    f = RAW / "dimacs1993" / (f"{name}.col.b" if name.startswith(("DSJ", "flat")) else f"{name}.col")
    inst = load_col(f)
    assert (inst.n, inst.m) == (n, m)
    # The DSJ files declare every edge twice; the others once.
    assert inst.declared_edges in (m, 2 * m)


@needs_raw
def test_known_small_chromatic_numbers():
    assert chromatic_number(load_col(RAW / "dimacs1993" / "myciel3.col")) == 4
    q = load_col(RAW / "dimacs1993" / "queen5_5.col")
    assert len(greedy_clique(q.adjacency)) == 5 and is_proper(q.adjacency, dsatur(q.adjacency))


def test_ascii_loader_merges_duplicates_and_rejects_loops(tmp_path):
    p = tmp_path / "d.col"
    p.write_text("c dup\np col 4 5\ne 1 2\ne 2 1\ne 3 4\nn 1 7\ne 2 3\ne 3 2\n")
    inst = load_col(p)
    assert inst.n == 4 and inst.edges.tolist() == [[0, 1], [1, 2], [2, 3]] and inst.declared_edges == 5
    for bad in ["p edge 3 1\ne 1 1\n", "p edge 3 1\ne 1 4\n", "e 1 2\n", "p edge 2 1\nq 1 2\n"]:
        p.write_text(bad)
        with pytest.raises(ValueError):
            load_col(p)


def test_binary_loader_and_verifier_read_the_binary_format(tmp_path):
    rng = np.random.default_rng(5)
    for n in (1, 7, 8, 9, 17, 30):
        edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < 0.4]
        path = write_binary(tmp_path / f"b{n}.col.b", n, edges)
        inst = load_col(path)
        assert inst.n == n and set(map(tuple, inst.edges.tolist())) == set(edges)
        assert read_graph(path) == (n, set(edges))


def test_evaluator_and_verifier_reject_invalid_colorings(tmp_path):
    path = cycle(tmp_path, 5)
    inst = load_col(path)
    good = [0, 1, 0, 1, 2]
    assert color_count(inst, good) == 3 and color_count(inst, [7, 3, 7, 3, -1]) == 3
    bad = [[0, 1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, 1, 2, 3], [0, 1, 0, 1, 2.0], [0, 1, 0, 1, True], "01012"]
    for c in bad:
        assert color_count(inst, c) is None and coloring_issues(inst, c)[1]
    out = tmp_path / "out.json"

    def check(payload) -> bool:
        out.write_text(json.dumps(payload) if not isinstance(payload, str) else payload)
        return verify_solution(path, out)["ok"]

    base = coloring_output(inst, good)
    assert base["objective_value"] == 3 and check(base) and check({**base, "status": "optimal"})
    assert check({**base, "coloring": [7, 3, 7, 3, -1]})
    for c in bad[:5]:
        assert not check({**base, "coloring": c})
    assert not check({**base, "objective_value": 2})
    assert not check({**base, "objective_value": True})
    assert not check({**base, "objective_value": "3"})
    assert not check('{"status": "feasible", "method": "x", "objective_value": NaN, "coloring": [0, 1, 0, 1, 2]}')
    assert not check({k: v for k, v in base.items() if k != "method"})
    assert not check({**base, "status": "infeasible"})
    assert not check({k: v for k, v in base.items() if k != "coloring"})
    assert not check("[1, 2]")
    assert not check("not json")


def test_verifier_imports_only_the_standard_library():
    tree = ast.parse((Path(__file__).resolve().parents[1] / "src" / "cg" / "coloring" / "e_verifier.py").read_text())
    mods = {a.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names}
    mods |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert mods <= {"__future__", "json", "math", "pathlib", "typing"}


# ---------------------------------------------------------------- heuristics and bounds

def test_heuristics_on_random_graphs(tmp_path):
    for seed in range(12):
        inst = load_col(random_graph(tmp_path, 200 + seed, n_range=(6, 11)))
        A = inst.adjacency
        chi = chromatic_number(inst)
        col = dsatur(A)
        assert is_proper(A, col) and int(col.max()) + 1 >= chi
        K = greedy_clique(A)
        assert all(A[u, v] for u, v in itertools.combinations(K, 2)) and len(K) <= chi
        cover = edge_clique_cover(A)
        covered = np.zeros_like(A)
        for Q in cover:
            assert len(Q) >= 2 and all(A[u, v] for u, v in itertools.combinations(Q.tolist(), 2))
            covered[np.ix_(Q, Q)] = True
        assert not (A & ~covered).any()
        found = []
        best = TabuCol(A, seed=seed).reduce(col, deadline=float("inf"), on_coloring=found.append, max_iters=20000, lower=len(K))
        assert is_proper(A, best) and chi <= int(best.max()) + 1 <= int(col.max()) + 1
        assert all(is_proper(A, c) for c in found)
        sets = [np.nonzero(np.asarray(best) == c)[0] for c in set(best.tolist())]
        assert color_count(inst, coloring_from_cover(inst.n, sets[::-1])) == len(sets)
        w = np.random.default_rng(seed).random(inst.n)
        for S in weighted_greedy_sets(A, w, np.random.default_rng(seed), 6, 0.0):
            assert not A[np.ix_(S, S)].any()
            assert maximal_extension(A, S).sum() == S.sum()  # already maximal


def test_tabucol_harvest_and_resume(tmp_path):
    """Harvested classes are independent sets; a resumed search at the same k continues its own assignment."""
    inst = load_col(random_graph(tmp_path, 600, n_range=(30, 31), p_range=(0.5, 0.5)))
    A = inst.adjacency
    tc = TabuCol(A, seed=0)
    k = len(greedy_clique(A)) - 1  # fewer colors than a clique: never solved
    got = []
    start = np.arange(inst.n) % k
    best, f = tc.solve(start, k, float("inf"), max_iters=3000, harvest=got.extend, harvest_every=50)
    assert f > 0 and got and all(not A[np.ix_(m, m)].any() for m in got)
    assert tc.state[0] == k and tc.state[3] == 3000 and tc.iterations == 3000
    tc.solve(start, k, float("inf"), max_iters=500, resume=True)
    assert tc.state[3] == 3500 and tc.iterations == 3500


def test_tabucol_reaches_the_chromatic_number_on_small_graphs(tmp_path):
    hits = 0
    for seed in range(8):
        inst = load_col(random_graph(tmp_path, 400 + seed, n_range=(9, 12)))
        best = TabuCol(inst.adjacency, seed=0).reduce(dsatur(inst.adjacency), deadline=float("inf"), max_iters=50000)
        hits += int(best.max()) + 1 == chromatic_number(inst)
    assert hits >= 7


def test_farley_bound_is_valid_and_safe(tmp_path):
    assert farley_bound(15.0, 1.0) == 15 and farley_bound(14.5, 1.0) == 15 and farley_bound(15.000001, 1.0) == 15
    assert farley_bound(10.0, 2.0) == 5 and farley_bound(10.0, 0.5) == 10
    for seed in range(10):
        inst = load_col(random_graph(tmp_path, 500 + seed, n_range=(5, 10)))
        chi = chromatic_number(inst)
        for t in range(5):
            pi = np.random.default_rng(seed * 10 + t).random(inst.n)
            assert farley_bound(float(pi.sum()), max_weight_independent_set(inst, pi)) <= chi


def test_mip_start_labels_put_the_clique_first(tmp_path):
    inst = load_col(random_graph(tmp_path, 7, n_range=(9, 11), p_range=(0.5, 0.7)))
    K = greedy_clique(inst.adjacency)
    lab = start_labels(dsatur(inst.adjacency), K)
    assert [int(lab[q]) for q in K] == list(range(len(K))) and is_proper(inst.adjacency, lab)
    assert sorted(set(lab.tolist())) == list(range(int(lab.max()) + 1))


# ---------------------------------------------------------------- compact MIP

@pytest.mark.parametrize("formulation", ["ass", "pop2"])
@pytest.mark.parametrize("seed", range(8))
def test_compact_mip_matches_brute_force(tmp_path, seed, formulation):
    path = random_graph(tmp_path, seed)
    inst = load_col(path)
    chi = chromatic_number(inst)
    out, seen, bounds = run(ColoringCompactMIP, path, inst, formulation=formulation)
    assert out["termination_reason"] == "optimal" and out["best_objective"] == chi
    assert seen and all(v is not None and v >= chi for v in seen) and seen == sorted(seen, reverse=True)
    assert all(b <= chi for b in bounds) and out["global_lb"] == chi
    assert color_count(inst, out["coloring"]) == chi


@pytest.mark.parametrize("formulation", ["ass", "pop2"])
def test_compact_mip_on_graphs_needing_search(tmp_path, formulation):
    for path, chi in [(cycle(tmp_path, 7), 3), (grotzsch(tmp_path), 4)]:
        inst = load_col(path)
        out, seen, bounds = run(ColoringCompactMIP, path, inst, formulation=formulation)
        assert out["best_objective"] == chi and out["global_lb"] == chi and all(b <= chi for b in bounds)


# ---------------------------------------------------------------- naive column generation

@pytest.mark.parametrize("seed", range(8))
def test_naive_cg_bounds_and_colorings(tmp_path, seed):
    path = random_graph(tmp_path, 100 + seed)
    inst = load_col(path)
    chi = chromatic_number(inst)
    out, seen, bounds = run(ColoringNaiveCG, path, inst, time_limit=20)
    assert out["cg_converged"] and out["cg_stop_reason"] == "converged"
    assert seen and all(v is not None and v >= chi for v in seen)
    assert len(bounds) == 1 and bounds[0] <= chi and out["global_lb"] == bounds[0]
    assert out["best_objective"] >= chi and out["rmp_status"] == gp.GRB.OPTIMAL
    assert color_count(inst, out["coloring"]) == out["best_objective"]


def test_naive_cg_bound_is_the_fractional_chromatic_number(tmp_path):
    """C5: fractional chromatic number 5/2, so the converged bound is 3; the Grotzsch graph: 29/10, bound 3."""
    for path, bound in [(cycle(tmp_path, 5), 3), (grotzsch(tmp_path), 3)]:
        inst = load_col(path)
        out, _, bounds = run(ColoringNaiveCG, path, inst, time_limit=20)
        assert out["cg_converged"] and bounds == [bound]
        assert out["lp_value"] == pytest.approx(2.5 if inst.n == 5 else 2.9, abs=1e-6)


def test_naive_cg_discards_incomplete_pricing_rounds(tmp_path, monkeypatch):
    path = random_graph(tmp_path, 7, n_range=(9, 11))
    monkeypatch.setattr(ColoringNaiveCG, "price", staticmethod(lambda *a, **k: {"complete": False, "reason": "pricing_status_9"}))
    out = ColoringNaiveCG(time_limit=10).run(str(path))
    assert out["cg_stop_reason"] == "pricing_status_9"
    assert out["cg_iterations"] == 1 and not out["cg_converged"] and out["global_lb"] is None
    assert out["best_objective"] is not None  # the shared initializer's coloring (or the integer master's)


# ---------------------------------------------------------------- engineered branch-and-price

@pytest.mark.parametrize("seed", range(10))
def test_engineered_matches_brute_force(tmp_path, seed):
    path = random_graph(tmp_path, 10 + seed)
    inst = load_col(path)
    chi = chromatic_number(inst)
    out, seen, bounds = run(ColoringBranchAndPrice, path, inst, tabu=seed % 2 == 0, alpha=0.0 if seed % 3 == 0 else 0.5)
    assert seen and all(v is not None and v >= chi for v in seen)
    assert all(b <= chi for b in bounds) and bounds == sorted(bounds)
    assert out["termination_reason"] == "optimal" and out["best_objective"] == chi and out["global_lb"] == chi
    assert color_count(inst, out["coloring"]) == chi


def test_engineered_branch_and_price_alone_proves_optimality(tmp_path, monkeypatch):
    """No initializer, TabuCol, restricted MIPs or dives: incumbents come only from integral nodes, so the Ryan-Foster
    branching (contraction / added edge), the node graphs, the Farley bounds and the pruning are exercised alone."""

    def bare_construct(self):
        self.root["lb"] = len(greedy_clique(self.A))
        self.publish_bound("clique")

    monkeypatch.setattr(ColoringBranchAndPrice, "construct", bare_construct)
    branched = 0
    paths = [cycle(tmp_path, 5), cycle(tmp_path, 7), grotzsch(tmp_path)]
    paths += [random_graph(tmp_path, 40 + s, n_range=(8, 12), p_range=(0.3, 0.6)) for s in range(9)]
    for k, path in enumerate(paths):
        inst = load_col(path)
        chi = chromatic_number(inst)
        out, seen, bounds = run(ColoringBranchAndPrice, path, inst, tabu=False, dive=False, rmp=False,
                                heuristic_pricing=bool(k % 2), alpha=0.0 if k % 3 == 0 else 0.5)
        assert all(v is not None and v >= chi for v in seen), path.name
        assert all(b <= chi for b in bounds) and bounds == sorted(bounds), path.name
        assert out["termination_reason"] == "optimal" and out["best_objective"] == chi, path.name
        branched += out["nodes"] > 1
        if path.name == "grotzsch.col":  # chi = 4 > ceil(29/10): the proof needs branching
            assert out["nodes"] > 1
    assert branched >= 2


def test_engineered_node_columns_respect_the_branching_rules(tmp_path):
    """Pricing at a node with same/differ rules returns independent sets of G that contain both or neither vertex of
    every same pair and never both of a differ pair; incompatible columns are fixed to 0."""
    path = random_graph(tmp_path, 77, n_range=(10, 12), p_range=(0.3, 0.4))
    inst = load_col(path)
    A = inst.adjacency
    nonadj = [(u, v) for u in range(inst.n) for v in range(u + 1, inst.n) if not A[u, v]]
    same, diff = (nonadj[0],), (nonadj[-1],)
    bp = ColoringBranchAndPrice(time_limit=10, tabu=False)
    bp.t0, bp.deadline = 0.0, float("inf")
    bp.inst, bp.n, bp.A = inst, inst.n, A
    bp.ub, bp.lb, bp.best, bp.rng = math.inf, -math.inf, None, np.random.default_rng(0)
    bp.stats = {k: 0 for k in ("heuristic_rounds", "pricing_seconds", "exact_rounds", "exact_seconds")}
    bp.env = gp.Env(empty=True)
    bp.env.setParam("OutputFlag", 0)
    bp.env.start()
    bp.build_master()
    for S in bp.heuristic_price(np.ones(inst.n)) + [np.zeros(inst.n, dtype=bool)]:
        bp.add_column(S)
    node = {"id": 1, "depth": 1, "lb": 0, "same": same, "diff": diff}
    assert bp.set_node(node)
    pi = np.random.default_rng(1).random(inst.n) + 0.5
    sets = bp.heuristic_price(pi) + bp.exact_price(pi)["sets"]
    assert sets
    for S in sets:
        assert not A[np.ix_(S, S)].any()
        assert S[same[0][0]] == S[same[0][1]] and not (S[diff[0][0]] and S[diff[0][1]])
    bp.master.update()
    ub = np.array(bp.master.getAttr("UB", bp.vars))
    M = bp.colmat[:len(bp.vars)]
    ok = (M[:, same[0][0]] == M[:, same[0][1]]) & ~(M[:, diff[0][0]] & M[:, diff[0][1]])
    assert np.array_equal(ub > 0, ok)


def test_engineered_reports_incumbent_provenance(tmp_path):
    path = random_graph(tmp_path, 3, n_range=(9, 11))
    stages = []
    ColoringBranchAndPrice(time_limit=10, on_incumbent=lambda c, stage=None: stages.append(stage)).run(str(path))
    assert stages and stages[0] == "initializer" and all(isinstance(s, str) for s in stages)


def test_canonical_labels():
    assert canonical([5, 5, 2, 9, 2]) == [0, 0, 1, 2, 1]
