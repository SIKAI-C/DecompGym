"""Generalized assignment (small scale): Chu-Beasley / Yagiura type D and E recipes (the feasibility MIP is stubbed)."""
from __future__ import annotations

import math

import pytest

import src.cg.gap.a_gen_data as gap_gen
from src.cg.gap.a_gen_data import _greedy_feasible, draw_type_de, generate_gap_instance_with_recipe


SIZES = ("small", "medium", "large")


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
