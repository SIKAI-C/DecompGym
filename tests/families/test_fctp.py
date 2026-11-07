"""FCTP (small scale): the Roberti-Bartolini-Mingozzi (Data Sets 3-4) recipe (balanced integer supplies and demands,
fixed costs in [200, 800], c_ij = floor(theta f_ij (2n - 1) / sum a), m_ij = min(a_i, b_j)) regenerates the final
instances bit for bit; solver-free."""
from __future__ import annotations

from fractions import Fraction
import json
import math

import numpy as np
import pytest

from src.bd.fctp.a_gen_data import LIT_B, LIT_FIXED_RANGE, LIT_SIZE_CONFIG, LIT_THETAS, generate_fctp_instance
from tests.families._recipe import SEEDS, SIZES, fresh_matches_pins_and_stored


PINS = {  # sha256 of data.json of the final instances (2026-11-06, round 2)
    ("small", 0): "f92293c9f1a2356f284534f3300efa87a26ad36236868ebf767cc2ccbc708541",
    ("small", 1): "b1d715a45ddbb345d470680ede7f35e6933fec5c7f7872e56ee8cfdb6a1c95f6",
    ("small", 2): "2d88a7f19cf494592f4d773c184e1e8c88bd1fb02fd1cd2e9781a906b92930d8",
    ("medium", 0): "4c49a134655dd1df8172b0351a88650c7995964380d9aa7c8c00aa7c31e85280",
    ("medium", 1): "9e56b400f0e1cecd28556edc5fc1528e314247068546a803d374e3fb79032137",
    ("medium", 2): "d76812a4b7b83ba4bccf437f001b94b0978e086070d6ecaab5b297b5445a3cf8",
    ("large", 0): "baf7bb50a6a1c9ec7a2d5eda95a6469122cd4895aa3448e1d789c2ee3ec0a519",
    ("large", 1): "0368d50078305d7cb66a877099a8bd40a88f14d8fde5c365fe14ac8409abe9aa",
    ("large", 2): "c5d0c596d85fdd1963239a09bc643fd0f6ab30cd1b41abda768a57ad94cece44",
}


def redraw(seed: int, n: int):
    """The declared draw order: default_rng(seed), f row-major, then (a, b) pairs until sum a = sum b."""
    rng = np.random.default_rng(seed)
    fixed = rng.integers(200, 801, size=(n, n))
    draws = 0
    while True:
        draws += 1
        a, b = rng.integers(1, 21, size=n), rng.integers(1, 21, size=n)
        if a.sum() == b.sum():
            return fixed, a, b, draws


def test_the_recipe_constants():
    assert LIT_SIZE_CONFIG == {"small": {"n": 11}, "medium": {"n": 13}, "large": {"n": 14}}
    assert (LIT_FIXED_RANGE, LIT_B, LIT_THETAS) == ((200, 800), 20, (0.0, 0.2, 0.5))


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_fresh_instances_match_the_pins_and_the_final_data(tmp_path, size, seed):
    directory = fresh_matches_pins_and_stored(generate_fctp_instance, "fctp", size, seed, PINS, tmp_path)
    data, source = json.loads((directory / "data.json").read_text()), json.loads((directory / "source.json").read_text())
    n, theta = LIT_SIZE_CONFIG[size]["n"], LIT_THETAS[seed % 3]
    assert set(data) == {"m", "n", "supply", "demand", "unit_cost", "fixed_cost", "capacity"} and data["m"] == data["n"] == n
    a, b = data["supply"], data["demand"]
    f, c, cap = np.asarray(data["fixed_cost"]), np.asarray(data["unit_cost"]), np.asarray(data["capacity"])
    assert sum(a) == sum(b) and all(1 <= v <= LIT_B and v == int(v) for v in a + b)  # balanced integer draws in [1, B]
    assert np.all(f == np.floor(f)) and 200 <= f.min() and f.max() <= 800
    total = int(sum(a))
    for i in range(n):
        for j in range(n):
            # Exact rational arithmetic, independent of the generator's integer formulation.
            assert c[i, j] == math.floor(Fraction(theta).limit_denominator(10) * int(f[i, j]) * (2 * n - 1) / total)
            assert cap[i, j] == min(a[i], b[j])
    fixed, sa, sb, draws = redraw(seed, n)
    assert f.tolist() == fixed.tolist() and a == sa.tolist() and b == sb.tolist()
    assert (source["theta"], source["balanced"], source["balance_draws"]) == (theta, True, draws)


def test_pinned_values():
    data = generate_fctp_instance(2, "small")  # theta = 0.5: c = floor(0.5 * 703 * 21 / 82) = floor(90.01) = 90
    assert data["supply"][:4] == [13.0, 6.0, 6.0, 7.0] and sum(data["supply"]) == 82.0
    assert data["fixed_cost"][0][:3] == [703.0, 357.0, 265.0] and data["unit_cost"][0][:3] == [90.0, 45.0, 33.0]
    zero = generate_fctp_instance(0, "small")  # theta = 0.0: no variable cost
    assert zero["supply"][:4] == [6.0, 6.0, 17.0, 2.0] and not any(any(row) for row in zero["unit_cost"])
