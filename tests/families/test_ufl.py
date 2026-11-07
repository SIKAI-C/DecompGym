"""UFL (small scale): the Koerkel-Ghosh recipe (MSVC rand stream, srand 123456, class and symmetry schedule, integer
truncation into the published ranges) regenerates the final instances bit for bit; solver-free."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.bd.ufl.a_gen_data import (KG_CLASSES, KG_COST_RANGE, KG_SIZE_CONFIG, KG_SRAND, MsvcRand, generate_ufl_instance,
                                   kg_schedule, kggen)
from tests.families._recipe import SEEDS, SIZES, fresh_matches_pins_and_stored


PINS = {  # sha256 of data.json of the final instances (2026-11-06, round 2)
    ("small", 0): "3eb44a3e8ccd70a1cc2f6a8a27c33512402de962fd57672fd22e3188f9ddfa95",
    ("small", 1): "1f6c3cb064ef6e9c867ef89c459aa0620f95c5250aa2f2e1de6575989f9b721d",
    ("small", 2): "4eda80018fa9a8b9468c8a7f60075830daebd622b55a7b134cefa5475c115dc2",
    ("medium", 0): "4848acc000ce70a2b1ae8e4c483e24d9b24cfc211cc6cf685ad92c5a6dc714e9",
    ("medium", 1): "d100b9a7bbc9532ffe4f43edc2b254dba2c5fe6d671d1b51daae9fdb3760465e",
    ("medium", 2): "5624df184e1c1ce08abee202187ed24bb143b4330a2485ca451c4d2075dbc99f",
    ("large", 0): "f35623e9e48037cc0d80dd373dc9abef5818cb6caa6af8385a25bc4398cee3ab",
    ("large", 1): "92f6d06685696cb0310b28a89ca68c05b02a77bd6a62d761c9f6839bd39ae666",
    ("large", 2): "ed6d8e61e1071ec835c7da10d8d5cb1d76638858bb0d0ede729f9b4ad953fff0",
}


def test_the_msvc_stream():
    rng = MsvcRand(KG_SRAND)
    assert [rng.rand() for _ in range(5)] == [9977, 22818, 10150, 16017, 7706]
    rng = MsvcRand(0)  # holdrand = 2531011 -> (2531011 >> 16) & 0x7fff = 38
    assert rng.rand() == 38


def test_the_kggen_draw_order_and_truncation():
    # Instance 1, asymmetric 2 x 2: F[0], C[0][0], C[0][1], F[1], C[1][0], C[1][1] from one srand(123456) stream.
    draws = MsvcRand(KG_SRAND)
    values = [draws.rand() for _ in range(12)]
    expect = [int(v * 1.0 / 32767 * (hi - lo) + lo) for v, (lo, hi) in zip(values, [(100, 200), (1000, 2000), (1000, 2000)] * 4)]
    F, C = kggen(2, 2, (100, 200), (1000, 2000), 1, symmetric=False)
    assert (F, C) == ([expect[0], expect[3]], [[expect[1], expect[2]], [expect[4], expect[5]]])
    # Instance 2 continues the same stream (no reseeding between instances).
    F2, C2 = kggen(2, 2, (100, 200), (1000, 2000), 2, symmetric=False)
    assert (F2, C2) == ([expect[6], expect[9]], [[expect[7], expect[8]], [expect[10], expect[11]]])
    # Symmetric: row i draws C[i][i..n-1] and mirrors them.
    Fs, Cs = kggen(2, 2, (100, 200), (1000, 2000), 1, symmetric=True)
    assert (Fs, Cs) == ([expect[0], expect[3]], [[expect[1], expect[2]], [expect[2], expect[4]]])
    with pytest.raises(ValueError):
        kggen(2, 3, (100, 200), (1000, 2000), 1, symmetric=True)


def test_the_declared_schedule():
    assert [kg_schedule(seed, size) for size in SIZES for seed in SEEDS] == [
        ("a", True), ("b", False), ("c", True), ("a", False), ("b", True), ("c", False), ("a", True), ("b", False), ("c", True)]
    assert KG_SIZE_CONFIG == {"small": {"n": 60}, "medium": {"n": 75}, "large": {"n": 100}}


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_fresh_instances_match_the_pins_and_the_final_data(tmp_path, size, seed):
    directory = fresh_matches_pins_and_stored(generate_ufl_instance, "ufl", size, seed, PINS, tmp_path)
    data, source = json.loads((directory / "data.json").read_text()), json.loads((directory / "source.json").read_text())
    n = KG_SIZE_CONFIG[size]["n"]
    fclass, symmetric = kg_schedule(seed, size)
    assert set(data) == {"m", "n", "f", "D"} and data["m"] == data["n"] == n  # no provenance in the agent-visible data
    f, D = np.asarray(data["f"]), np.asarray(data["D"])
    (flo, fhi), (clo, chi) = KG_CLASSES[fclass], KG_COST_RANGE
    assert np.all(f == np.floor(f)) and flo <= f.min() and f.max() <= fhi
    assert np.all(D == np.floor(D)) and clo <= D.min() and D.max() <= chi
    assert np.array_equal(D, D.T) == symmetric
    assert (source["opening_cost_class"], source["symmetric"], source["generator_instance_number"]) == (fclass.upper(), symmetric, seed + 1)
    F, C = kggen(n, n, KG_CLASSES[fclass], KG_COST_RANGE, seed + 1, symmetric=symmetric)
    assert f.tolist() == F and D.tolist() == C


def test_pinned_values():
    data = generate_ufl_instance(0, "small")
    assert data["f"][:3] == [130.0, 153.0, 152.0] and data["D"][0][:3] == [1696.0, 1309.0, 1488.0] and data["D"][1][:2] == [1309.0, 1835.0]
