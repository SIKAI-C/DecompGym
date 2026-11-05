"""Literature-recipe generators of the small CG families bp, csp, gap and bppc (no optimizer calls: the GAP feasibility
MIP is stubbed)."""
from __future__ import annotations

import math

import pytest

from src.cg.bp.a_gen_data import SCHOLL_CLASS, generate_bp_instance
from src.cg.bp.a_gen_data import first_fit_decreasing_bins
from src.cg.bppc.a_gen_data import DENSITY, REGIMES, generate_bppc_instance, recipe_record as bppc_recipe
from src.cg.csp.a_gen_data import CUTGEN1_CELL, CutgenRandom, cutgen1_problem, generate_csp_instance
import src.cg.gap.a_gen_data as gap_gen
from src.cg.gap.a_gen_data import _greedy_feasible, draw_type_de, generate_gap_instance_with_recipe

SIZES = ("small", "medium", "large")


def test_cutgen1_port_reproduces_the_ejor_samples():
    # Gau and Waescher (1995), p. 576: seed 1994, m = 10, L = 1000, v1 = 0.375, v2 = 0.625, d = 50; problems 1 and 150.
    rng = CutgenRandom(1994)
    args = {"m": 10, "L": 1000, "v1": 0.375, "v2": 0.625, "dbar": 50}
    assert cutgen1_problem(rng, **args) == ([577, 570, 544, 539, 526, 512, 504, 459, 446, 378], [63, 64, 85, 84, 54, 20, 24, 31, 48, 27])
    for _ in range(2, 150):
        cutgen1_problem(rng, **args)
    assert cutgen1_problem(rng, **args) == ([570, 558, 543, 540, 513, 490, 476, 453, 434], [16, 94, 20, 99, 59, 48, 78, 35, 51])


@pytest.mark.parametrize("size", SIZES)
def test_csp_cutgen_cell(size):
    for seed in range(5):
        d = generate_csp_instance(seed, size)
        assert d == generate_csp_instance(seed, size)
        L, lengths = d["stock_size"], [it["size"] for it in d["item_types"]]
        assert L == CUTGEN1_CELL["L"] and lengths == sorted(set(lengths), reverse=True)
        assert all(CUTGEN1_CELL["v1"] * L <= x <= CUTGEN1_CELL["v2"] * L + 1 for x in lengths)
        from src.cg.csp.a_gen_data import SIZE_SCALING
        assert sum(it["demand"] for it in d["item_types"]) >= SIZE_SCALING[size]["m"] * CUTGEN1_CELL["dbar"]


@pytest.mark.parametrize("size", SIZES)
def test_bp_scholl_class(size):
    d = generate_bp_instance(3, size)
    lo, hi = SCHOLL_CLASS["weight_range"]
    assert d["bin_capacity"] == SCHOLL_CLASS["capacity"] and all(lo <= it["size"] <= hi for it in d["items"])


def test_ffd_bound():
    assert first_fit_decreasing_bins([60, 50, 40], [1, 1, 2], 100) == 2
    assert first_fit_decreasing_bins([70], [3], 100) == 3


def test_gap_type_formulas():
    import random
    a, c, b = draw_type_de(random.Random(1), "D", 200, 4)
    assert all(1 <= a[i][j] <= 100 and -10 <= c[i][j] + a[i][j] - 111 <= 10 for i in range(200) for j in range(4))
    assert b == [math.floor(0.8 * sum(a[i][j] for i in range(200)) / 4) for j in range(4)]
    a, c, b = draw_type_de(random.Random(2), "E", 200, 4)
    assert all(a[i][j] >= 1 and -11 < c[i][j] - 1000 / a[i][j] <= 0 for i in range(200) for j in range(4))
    assert b == [math.floor(0.8 * sum(a[i][j] for i in range(200)) / 4) for j in range(4)]


@pytest.mark.parametrize("size", SIZES)
def test_gap_instances_typed(size, monkeypatch):
    monkeypatch.setattr(gap_gen, "gap_feasible", lambda a, b: True)
    for seed in range(4):
        d, rec = generate_gap_instance_with_recipe(seed, size)
        assert rec["gap_type"] == ("D" if seed % 2 == 0 else "E") and rec["redraw_index"] == 0
        assert len(d["jobs"]) == rec["n_jobs"] == 5 * rec["n_machines"] and len(d["machines"]) == rec["n_machines"]
        assert "gap_type" not in d  # recipe metadata stays out of the agent's data


def test_gap_redraw_until_feasible(monkeypatch):
    calls = []
    monkeypatch.setattr(gap_gen, "gap_feasible", lambda a, b: calls.append(1) or len(calls) >= 3)
    d, rec = generate_gap_instance_with_recipe(7, "small")
    assert rec["redraw_index"] == 2
    a, c, b = draw_type_de(__import__("random").Random("gap-litgen:7:2"), "E", gap_gen.SIZE_SCALING["small"]["n"], gap_gen.SIZE_SCALING["small"]["m"])
    assert [m["capacity"] for m in d["machines"]] == b


def test_greedy_feasibility_is_sufficient_only():
    assert _greedy_feasible([[5, 5], [5, 5]], [5, 5])
    assert not _greedy_feasible([[6, 6]], [5, 5])


@pytest.mark.parametrize("size", SIZES)
def test_bppc_sadykov_vanderbeck(size):
    for seed in range(4):
        d = generate_bppc_instance(seed, size)
        rec = bppc_recipe(seed, size)
        regime = rec["regime"]
        assert regime == ("DA" if seed % 2 == 0 else "UA")
        lo, hi = REGIMES[regime]["weight_range"]
        n = d["n_items"]
        assert d["capacity"] == REGIMES[regime]["capacity"] and all(lo <= w <= hi for w in d["weights"])
        edges = {tuple(e) for e in d["conflicts"]}
        assert len(edges) == len(d["conflicts"]) == math.ceil(DENSITY[regime] * n * (n - 1) / 2 - 1e-9) == rec["n_conflicts"]
        assert all(a < b for a, b in edges)


def test_strict_suite_records_generation_failure(tmp_path):
    from src.utils.litgen_suite import generate_strict_suite

    folder = tmp_path / "bp_small_seed000"
    folder.mkdir()
    (folder / "meta.json").write_text("{}")

    def boom(seed, size, inst_dir):
        raise ValueError("no draw")

    with pytest.raises(RuntimeError, match="generation or prompt failed"):
        generate_strict_suite("bp", gen=boom, render_prompt=lambda d: "p", sizes=["small"], num_instances=1, out_root=tmp_path)
    assert not (folder / "meta.json").exists() and "no draw" in (folder / "rejection.json").read_text()
