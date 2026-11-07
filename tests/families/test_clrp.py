"""CLRP (small scale): literature-recipe generator (Prodhon / Prins et al. 2006), the published-file parser, verifier and reference arguments.

No optimizer is called by the recipe tests; the reference checks on the published 20-customer files are recorded in the
generation notes (they reproduce the published optima 54 793, 39 104, 48 908 and 37 542 with ceil costs).
"""
from __future__ import annotations

import pytest

from src.cg.clrp.a_gen_data import DEFAULT_CONFIG, SIZE_CONFIG, CLRPInstance, generate_clrp_instance, packs, read_prodhon


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
    inst = CLRPInstance(1, 1, 70, 1000, [0], [0], [70], [1], [1], [1], [11])
    assert inst.cost(0, 1) == 142  # ceil(141.42)
    assert inst.route_cost(0, [0]) == 1000 + 284


def test_published_20_customer_files_parse():
    from tests.reference_data import ROOT, reference_path

    PRODHON = reference_path("DECOMPGYM_PRODHON_DIR", ROOT / "prodhon", "Prodhon CLRP instances (coord*.dat)")
    inst = read_prodhon(PRODHON / "coord20-5-1.dat")
    assert (inst.n_customers, inst.n_depots, inst.vehicle_capacity, inst.vehicle_cost) == (20, 5, 70, 1000)
    assert inst.depot_capacity == [140] * 5 and inst.opening_cost == [10841, 11961, 6091, 7570, 7497]
    assert (inst.depot_x[0], inst.depot_y[0], inst.customer_x[0], inst.customer_y[0], inst.demand[0]) == (6, 7, 20, 35, 17)
    assert read_prodhon(PRODHON / "coord20-5-1b.dat").vehicle_capacity == 150


def test_clrp_verifier_charges_opening_vehicle_and_ceil_travel():
    from src.cg.clrp.d_verifier import verify_solution

    data = {"n_customers": 2, "n_depots": 2, "vehicle_capacity": 10, "vehicle_cost": 7, "depot_x": [0, 9], "depot_y": [0, 9],
            "depot_capacity": [5, 20], "opening_cost": [100, 1000], "customer_x": [1, 0], "customer_y": [1, 1], "demand": [3, 4]}
    # Depot 0 holds only 5 < 7, so both customers on one route from depot 0 is infeasible.
    assert not verify_solution(data, {"solution": {"routes": [{"depot": 0, "customers": [0, 1]}]}})["ok"]
    # One route each: depot 0 -> c0 -> 0 = 2 * ceil(141.42) = 284; depot 1 -> c1 -> 1 = 2 * ceil(100 * sqrt(145)) = 2 * 1205.
    result = verify_solution(data, {"solution": {"routes": [{"depot": 0, "customers": [0]}, {"depot": 1, "customers": [1]}]}})
    assert result["ok"] and result["objective_value"] == 100 + 1000 + 2 * 7 + 284 + 2 * 1205
    assert not verify_solution(data, {"solution": {"routes": [{"depot": 2, "customers": [0, 1]}]}})["ok"]


def test_clrp_verifier_charges_a_used_depot_even_without_demand():
    # Codex review (routing families) finding 3: opening costs follow the routes, not the load.
    from src.cg.clrp.d_verifier import verify_solution

    data = {"n_customers": 1, "n_depots": 1, "vehicle_capacity": 10, "vehicle_cost": 7, "depot_x": [0], "depot_y": [0],
            "depot_capacity": [5], "opening_cost": [100], "customer_x": [1], "customer_y": [0], "demand": [0]}
    assert verify_solution(data, {"solution": {"routes": [{"depot": 0, "customers": [0]}]}})["objective_value"] == 307


@pytest.mark.parametrize("module,cls", [("src.cg.clrp.a_gold_cg", "CLRPColumnGeneration")])
def test_reference_rejects_an_empty_pricing_batch(module, cls):
    # Codex review (routing families) round 2: batch=0 would read as pricing convergence.
    import importlib

    with pytest.raises(ValueError, match="positive"):
        getattr(importlib.import_module(module), cls)(batch=0)
