"""CLRP literature-recipe generator (Prodhon / Prins et al. 2006): recipe rules and the published-file parser.

No optimizer is called here; the reference checks on the published 20-customer files are recorded in the
generation notes (they reproduce the published optima 54 793, 39 104, 48 908 and 37 542 with ceil costs).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.cg.clrp.a_gen_data import DEFAULT_CONFIG, SIZE_CONFIG, CLRPInstance, generate_clrp_instance, packs, read_prodhon

PRODHON = Path(__file__).resolve().parents[2] / "DecompBench/outputs/scaleup/raw/lrp/prodhon"


@pytest.mark.parametrize("size", list(SIZE_CONFIG))
@pytest.mark.parametrize("seed", [0, 1, 2, 100, 101, 102, 103])
def test_generator_follows_the_recipe(size, seed):
    inst = CLRPInstance.from_dict(generate_clrp_instance(seed, size))
    n, m = SIZE_CONFIG[size]["n_customers"], SIZE_CONFIG[size]["n_depots"]
    assert (inst.n_customers, inst.n_depots) == (n, m)
    assert inst.vehicle_capacity == (70 if seed % 2 == 0 else 150) and inst.vehicle_cost == 1000
    assert all(11 <= q <= 20 for q in inst.demand)
    total = sum(inst.demand)
    assert all(total / 3 <= w <= total / 2 * 1.2 + 1 for w in inst.depot_capacity)
    assert sum(sorted(inst.depot_capacity)[-1:]) < total  # one depot never suffices
    assert all(5000 <= o <= 15000 for o in inst.opening_cost)
    assert all(1 <= v <= 50 for v in inst.depot_x + inst.depot_y + inst.customer_x + inst.customer_y)
    squares = DEFAULT_CONFIG["clusters"][(0, 2, 3)[seed % 3]]
    for i in range(n):
        x0, x1, y0, y1 = squares[i % len(squares)]
        assert x0 <= inst.customer_x[i] <= x1 and y0 <= inst.customer_y[i] <= y1
    assert packs(inst.demand, inst.depot_capacity)


def test_generator_is_deterministic_and_sizes_grow():
    assert generate_clrp_instance(5, "medium") == generate_clrp_instance(5, "medium")
    sizes = [SIZE_CONFIG[s]["n_customers"] for s in ("small", "medium", "large")]
    assert sizes == sorted(sizes) and len(set(sizes)) == 3


def test_packing_test():
    assert packs([20, 20, 20, 20], [40, 40])
    assert not packs([30, 30, 30], [40, 40])


def test_costs_round_up():
    inst = CLRPInstance("t", 1, 1, 70, 1000, [0], [0], [70], [1], [1], [1], [11])
    assert inst.cost(0, 1) == 142  # ceil(141.42)
    assert inst.route_cost(0, [0]) == 1000 + 284


@pytest.mark.skipif(not PRODHON.is_dir(), reason="published Prodhon files not available locally")
def test_published_20_customer_files_parse():
    inst = read_prodhon(PRODHON / "coord20-5-1.dat")
    assert (inst.n_customers, inst.n_depots, inst.vehicle_capacity, inst.vehicle_cost) == (20, 5, 70, 1000)
    assert inst.depot_capacity == [140] * 5 and inst.opening_cost == [10841, 11961, 6091, 7570, 7497]
    assert (inst.depot_x[0], inst.depot_y[0], inst.customer_x[0], inst.customer_y[0], inst.demand[0]) == (6, 7, 20, 35, 17)
    assert read_prodhon(PRODHON / "coord20-5-1b.dat").vehicle_capacity == 150
