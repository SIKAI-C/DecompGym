"""k-median (small scale): the TSPLIB-derived p-median recipe (parent subset, floor-Euclidean distances, p schedule)
regenerates the final instances bit for bit; solver-free."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.bd.km.a_gen_data import (TSP_P_VALUES, TSP_PARENT_SHA256, TSP_PARENTS, TSP_SIZE_CONFIG, floor_euclidean,
                                  generate_km_instance, read_tsplib_euc2d, tsp_schedule)
from tests.families._recipe import SEEDS, SIZES, fresh_matches_pins_and_stored


PINS = {  # sha256 of data.json of the final instances (2026-11-06, round 2)
    ("small", 0): "4845878397b14ea61c80aa410c696a114af9aa3d46cec13c870083d691153534",
    ("small", 1): "bca3e9665e0c39d1b4dd9ee25ad3a922bbccb0afa4ffb745382e44e2e540e8e4",
    ("small", 2): "9eb8bff14cac929f47186b95f912f9eda61192d73195718febd99bf675806216",
    ("medium", 0): "ba71e5780562e3170f79baa630b306eabc7e0491630a3608ed381d43b52306c1",
    ("medium", 1): "ef3de14269d148e0faa193417f0c023a99c7b9b155d852982b139539d8b18816",
    ("medium", 2): "6a2a2b2d73a59ff2576ae2d00450a22c85562d52a7c8f7bb70eec3252db3b762",
    ("large", 0): "447e33dee4aae1c3300b1453bb2007fe450100b438072aa15c9ce55f07a63bd0",
    ("large", 1): "308b495854973276d10c7ec8402120ff6e2a87e2eb6f08958bfdec2c77200c33",
    ("large", 2): "83e842d3ed73aea9a3cbd000b7dedd23e50cbba6239cf3ef1ea0ccc62f2b6523",
}


def test_floor_not_nint_distances():
    P = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0], [0.6, 0.0]])
    D = floor_euclidean(P, P)
    assert D[0, 1] == 5 and D[0, 2] == 1 and D[0, 3] == 0  # sqrt(2) = 1.41 -> 1; 0.6 -> 0 (TSPLIB nint would give 1)
    assert np.array_equal(D, D.T) and not np.any(np.diag(D))


def test_the_declared_schedule_and_parents():
    assert [tsp_schedule(seed, size) for size in SIZES for seed in SEEDS] == [
        ("rl1304", 5), ("fl1400", 10), ("u1432", 20), ("vm1748", 5), ("rl1304", 10), ("fl1400", 20), ("u1432", 5), ("vm1748", 10), ("rl1304", 20)]
    assert TSP_SIZE_CONFIG == {"small": {"n": 250}, "medium": {"n": 400}, "large": {"n": 525}} and TSP_P_VALUES == (5, 10, 20)
    for parent, dimension in zip(TSP_PARENTS, (1304, 1400, 1432, 1748)):
        assert read_tsplib_euc2d(parent).shape == (dimension, 2)  # the recorded sha256 is checked on read
    assert set(TSP_PARENT_SHA256) == set(TSP_PARENTS)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_fresh_instances_match_the_pins_and_the_final_data(tmp_path, size, seed):
    directory = fresh_matches_pins_and_stored(generate_km_instance, "km", size, seed, PINS, tmp_path)
    data, source = json.loads((directory / "data.json").read_text()), json.loads((directory / "source.json").read_text())
    parent, p = tsp_schedule(seed, size)
    n = TSP_SIZE_CONFIG[size]["n"]
    assert set(data) == {"m", "n", "k", "Xc", "Xf", "D"} and (data["m"], data["n"], data["k"]) == (n, n, p)
    assert (source["parent"], source["p"], source["parent_sha256"]) == (parent, p, TSP_PARENT_SHA256[parent])
    nodes = source["parent_nodes_1based"]
    assert len(nodes) == n and nodes == sorted(set(nodes))  # a subset without replacement, in file order
    expected = np.sort(np.random.default_rng(seed).choice(len(read_tsplib_euc2d(parent)), size=n, replace=False)) + 1
    assert nodes == expected.tolist()
    X = read_tsplib_euc2d(parent)[np.asarray(nodes) - 1]
    assert data["Xc"] == X.tolist() and data["Xf"] == data["Xc"]
    D = np.asarray(data["D"])
    assert np.array_equal(D, floor_euclidean(X, X))
    for i, j in ((0, 1), (1, n - 1), (n // 2, n // 3)):  # an independent scalar recomputation
        assert D[i, j] == math.floor(math.sqrt((X[i, 0] - X[j, 0]) ** 2 + (X[i, 1] - X[j, 1]) ** 2))


def test_pinned_values():
    data = generate_km_instance(0, "small")
    assert data["k"] == 5 and data["Xc"][:2] == [[15440.0, 9768.0], [15440.0, 14058.0]] and data["D"][0][:4] == [0.0, 4290.0, 1655.0, 2209.0]
