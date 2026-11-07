"""Cutting stock (small scale): the CUTGEN1 port and the generator's cell (no optimizer call)."""
from __future__ import annotations

import pytest

from src.cg.csp.a_gen_data import CUTGEN1_CELL, CutgenRandom, cutgen1_problem, generate_csp_instance


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
