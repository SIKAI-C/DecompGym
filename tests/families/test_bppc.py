"""Bin packing with conflicts (small scale): Sadykov-Vanderbeck classes."""
from __future__ import annotations

import math

import pytest

from src.cg.bppc.a_gen_data import DENSITY, REGIMES, generate_bppc_instance, recipe_record as bppc_recipe


SIZES = ("small", "medium", "large")


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
