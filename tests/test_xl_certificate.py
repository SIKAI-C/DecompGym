"""Exact Benders certificate checker (protocol v4 Section 3; release acceptance suite items of Section 9.8).

Pure-Python tests use an injected master solve (any multipliers give a valid bound, so fixed multipliers make the
expected value computable by hand); the Gurobi tests solve the master, the deterministic equivalent and small
LP-master Benders loops. All of them run on Phoenix only (no optimization on the laptop)."""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

try:
    import gurobipy as gp
except ImportError:  # pragma: no cover
    gp = None

from src.experiments import xl_certificate as xc
from src.experiments.xl_certificate import (BDView, CheckerLimits, DESolve, MasterSolve, NumberError, check_certificate,
                                            check_versions, de_dual_bound, de_primal_value, expand_term, lp_interval,
                                            ratio, recertify)

P54 = 2 ** 54


def one_sub_view(rows, xs, *, y=None, fs_rows=None, view_id="toy", weight=1, theta=None):
    sub = {"id": "s", "weight": weight, "vars": xs, "rows": rows}
    if theta:
        sub["theta_lo"], sub["theta_hi"] = theta
    return BDView.from_dict({"view": view_id, "first_stage": {"vars": y or [], "rows": fs_rows or []}, "subproblems": [sub]})


def cert(cuts, view="toy", version=1):
    return json.dumps({"view": view, "version": version, "cuts": cuts})


def opt(*terms):
    return {"kind": "opt", "terms": [{"s": s, "pi": pi} for s, pi in terms]}


def fixed_master(duals_for):
    """A master 'solver' returning given multipliers (a function of the row list)."""
    def solve(lo, hi, obj, rows, time_limit, limits):
        return MasterSolve("optimal", list(duals_for(rows)), None, 0.0)
    return solve


ONES = fixed_master(lambda rows: [1.0] * len(rows))


# ---------------------------------------------------------------- exact numbers

def test_ratio_is_exact_and_rejects_bad_values():
    assert ratio("0.1") == (1, 10)
    assert ratio(0.1) == (0.1).as_integer_ratio()
    assert ratio("-2.5e-3") == (-1, 400)
    assert ratio("17") == (17, 1)
    assert ratio(3) == (3, 1)
    for bad, why in [("NaN", "non_finite"), ("Infinity", "non_finite"), ("-inf", "non_finite"), (float("nan"), "non_finite"),
                     ("1e500", "out_of_range"), ("1" * 70, "token_too_long"), ("abc", "bad_number"), (True, "bad_number"),
                     (None, "bad_number"), ([1], "bad_number")]:
        with pytest.raises(NumberError) as exc:
            ratio(bad)
        assert exc.value.reason == why


def test_float_rounding_helpers():
    third = Fraction(1, 3)
    assert Fraction(xc.float_down(third)) <= third <= Fraction(xc.float_up(third))
    assert xc.float_down(Fraction(1, 2)) == 0.5 == xc.float_up(Fraction(1, 2))


# ---------------------------------------------------------------- review 42: float cancellation

CANCEL_ROWS = [
    {"id": "R1", "sense": "=", "rhs": 0, "a": {"x": P54, "w": -P54}},
    {"id": "R2", "sense": ">=", "rhs": 1, "a": {"x": 1}},
    {"id": "R3", "sense": "=", "rhs": 0, "a": {"x": -P54, "w": P54}},
]
CANCEL_VARS = [{"id": "x", "cost": 0, "lb": 0, "ub": 1}, {"id": "w", "cost": 0, "lb": 0, "ub": 1},
               {"id": "v", "cost": 1, "lb": 0, "ub": 1}]


def test_cancellation_example_is_exact():
    """Codex review 42: A^T pi = 2^54 + 1 - 2^54 is 0 in left-to-right float64 but 1 exactly. A float 'safe' cut
    claims Q >= 1 for a subproblem whose optimum is 0 (x = w = 1, v = 0); the exact cut is Q >= 0."""
    view = one_sub_view(CANCEL_ROWS, CANCEL_VARS)
    pi = {"R1": 1, "R2": 1, "R3": 1}
    # the naive float evaluation (what the exact checker must not do)
    cost = {"x": 0.0, "w": 0.0, "v": 1.0}
    rf = {}
    for var in ("x", "w", "v"):
        acc = 0.0
        for row in CANCEL_ROWS:
            acc += float(row["a"].get(var, 0)) * pi[row["id"]]
        rf[var] = cost[var] - acc
    naive = 1.0 + sum(min(0.0, r) for r in rf.values())
    assert naive == 1.0  # invalid: exceeds Q* = 0
    den, const, ycoef = expand_term(view.subproblems[0], {k: str(v) for k, v in pi.items()}, False)
    assert Fraction(const, den) == 0 and not ycoef
    res = check_certificate(view, cert([opt(("s", {"R1": "1", "R2": "1", "R3": "1"}))]), solve_master=ONES)
    assert res.cert_ok and res.lb == 0


# ---------------------------------------------------------------- review 43: min {x : 3x >= 1, 0 <= x <= 1}

THREE_ROWS = [{"id": "r", "sense": ">=", "rhs": 1, "a": {"x": 3}}]
THREE_VARS = [{"id": "x", "cost": 1, "lb": 0, "ub": 1}]


@pytest.mark.parametrize("pi,expected", [
    ("0.3333333333333333", Fraction(3333333333333333, 10**16)),
    (1 / 3, Fraction(3333333333333333, 10**16)),   # a JSON float token is its decimal text
    ("0.33333333333333337", 1 - 2 * Fraction("0.33333333333333337")),
    ("1000", Fraction(0)),        # forged: the exact cut is weak (theta >= -1999), the trivial bound wins
    ("0", Fraction(0)),
])
def test_three_x_certificates_never_exceed_one_third(pi, expected):
    view = one_sub_view(THREE_ROWS, THREE_VARS)
    res = check_certificate(view, json.dumps({"view": "toy", "cuts": [opt(("s", {"r": pi}))]}), solve_master=ONES)
    assert res.cert_ok
    assert res.lb == expected
    assert res.lb <= Fraction(1, 3)


def test_three_x_lp_interval_from_float_solution():
    """Review 43: the float64 dual 1/3 re-certifies to 6004799503160661/2^54 < 1/3; the float primal is infeasible
    (3 * fl(1/3) < 1) and is repaired to 1/3 exactly, so the interval is rigorous and contains LP* = 1/3."""
    view = one_sub_view(THREE_ROWS, THREE_VARS)
    lo = de_dual_bound(view, [1 / 3])
    assert lo == Fraction(6004799503160661, 18014398509481984)
    assert de_primal_value(view, [], [[Fraction(1 / 3)]])[0] is None
    out = lp_interval(view, solution=DESolve("optimal", [1 / 3], [], [[1 / 3]], 1 / 3))
    assert out["rigorous"]
    assert Fraction(out["lp_lo"]) == lo < Fraction(1, 3) == Fraction(out["lp_hi"])
    assert out["repair"].startswith("generic_")


# ---------------------------------------------------------------- data are exact rationals

def test_coefficient_rounding_uses_decimal_data():
    """Review 43: rhs 0.1 is 1/10, not the float 0.1000000000000000055...; with the float the bound would exceed
    LP* = 1/10 by 1/180143985094819840."""
    view = one_sub_view([{"id": "r", "sense": ">=", "rhs": "0.1", "a": {"x": 1}}], [{"id": "x", "cost": 1, "lb": 0, "ub": 1}])
    res = check_certificate(view, cert([opt(("s", {"r": "1"}))]), solve_master=ONES)
    assert res.lb == Fraction(1, 10) < Fraction(0.1)
    view2 = xc.ufl_view({"f": ["0.1", "0.2"], "D": [["1.1"], ["0.3"]]})
    assert view2.f[0] == Fraction(1, 10) and view2.subproblems[0].cost(0) == Fraction(11, 10)


def test_recertification_uses_original_rows_not_float_images(monkeypatch):
    """The master gets float images of the rows; the bound uses the exact rows. Perturbing the float images (as if
    read back from the solver) must not change the certified value."""
    view = one_sub_view([{"id": "r", "sense": ">=", "rhs": "0.1", "a": {"x": 1}}], [{"id": "x", "cost": 1, "lb": 0, "ub": 1}])

    def perturbed(lo, hi, obj, rows, time_limit, limits):
        for r in rows:
            r.fvals = [v * (1 + 1e-6) for v in r.fvals]
            r.frhs *= 1 + 1e-6
        return MasterSolve("optimal", [1.0] * len(rows))

    res = check_certificate(view, cert([opt(("s", {"r": "1"}))]), solve_master=perturbed)
    assert res.lb == Fraction(1, 10)


# ---------------------------------------------------------------- sign projection, duplicates, non-finite

def test_sign_violating_multipliers_are_clipped_and_reported():
    rows = [{"id": "g", "sense": ">=", "rhs": 1, "a": {"x": 1}}, {"id": "l", "sense": "<=", "rhs": 1, "a": {"x": 1}}]
    view = one_sub_view(rows, [{"id": "x", "cost": 2, "lb": 0, "ub": 5}])
    bad = check_certificate(view, cert([opt(("s", {"g": "-1", "l": "3"}))]), solve_master=ONES)
    assert bad.cert_ok and bad.stats["clipped_count"] == 2 and bad.stats["clipped_mass"] == "4"
    empty = check_certificate(view, cert([opt(("s", {}))]), solve_master=ONES)
    assert bad.lb == empty.lb
    good = check_certificate(view, cert([opt(("s", {"g": "2", "l": "-0"}))]), solve_master=ONES)
    assert good.lb == 2  # min 2x s.t. x >= 1: Q = 2


def two_sub_view():
    return BDView.from_dict({"view": "two", "first_stage": {"vars": [{"id": "y", "cost": 1}], "rows": []},
                             "subproblems": [{"id": sid, "weight": w, "vars": [{"id": "x", "cost": 1, "lb": 0, "ub": 1}],
                                              "rows": [{"id": "r", "sense": ">=", "rhs": 1, "a": {"x": 1}, "t": {"y": 1}}]}
                                             for sid, w in (("a", "0.5"), ("b", "0.5"))]})


def test_duplicate_subproblem_ids_reject_the_cut():
    view = two_sub_view()
    dup = {"kind": "opt", "terms": [{"s": "a", "pi": {"r": "1"}}, {"s": "a", "pi": {"r": "1"}}]}
    res = check_certificate(view, cert([dup], view="two"), solve_master=ONES)
    assert res.cert_ok and res.stats["rejected"] == {"duplicate_id": 1} and res.stats["accepted"] == 0
    assert res.lb == res.trivial
    # duplicate row keys in one term and duplicate "s" keys are ambiguous JSON: rejected too
    raw = '{"view": "two", "cuts": [{"kind": "opt", "terms": [{"s": "a", "pi": {"r": "1", "r": "2"}}]},' \
          ' {"kind": "opt", "terms": [{"s": "a", "s": "b", "pi": {"r": "1"}}]}]}'
    res2 = check_certificate(view, raw, solve_master=ONES)
    assert res2.stats["rejected"] == {"duplicate_row": 1, "malformed_term": 1}


def test_aggregated_cut_uses_view_weights():
    view = two_sub_view()
    agg = {"kind": "opt", "terms": [{"s": "a", "pi": {"r": "1"}}, {"s": "b", "pi": {"r": "1"}}]}
    rows_seen = []

    def capture(lo, hi, obj, rows, time_limit, limits):
        rows_seen.extend(rows)
        return MasterSolve("optimal", [1.0] * len(rows))

    res = check_certificate(view, cert([agg], view="two"), solve_master=capture)
    (row,) = rows_seen
    # 0.5 theta_a + 0.5 theta_b >= 0.5 (1 - y) + 0.5 (1 - y): theta coefficients are the view's weights
    exact_row = {c: Fraction(v, row.den) for c, v in zip(row.cols, row.nums)}
    assert exact_row == {0: Fraction(1), 1: Fraction(1, 2), 2: Fraction(1, 2)} and Fraction(row.rhs, row.den) == 1
    assert res.cert_ok


@pytest.mark.parametrize("value", ['NaN', 'Infinity', '-Infinity', '"nan"', '"inf"', '"1e999"', '"x1"'])
def test_non_finite_or_unparsable_values_reject_only_their_cut(value):
    view = one_sub_view(THREE_ROWS, THREE_VARS)
    raw = '{"view": "toy", "cuts": [{"kind": "opt", "terms": [{"s": "s", "pi": {"r": %s}}]},' \
          ' {"kind": "opt", "terms": [{"s": "s", "pi": {"r": "0.25"}}]}]}' % value
    res = check_certificate(view, raw, solve_master=ONES)
    assert res.cert_ok and res.stats["accepted"] == 1 and sum(res.stats["rejected"].values()) == 1
    assert res.lb == Fraction(1, 4)


# ---------------------------------------------------------------- limits and degenerate certificates

def test_oversized_certificates_give_the_trivial_bound():
    view = one_sub_view(THREE_ROWS, THREE_VARS)
    good = cert([opt(("s", {"r": "0.25"}))])
    for limits, why in [(CheckerLimits(max_bytes=40), "too_many_bytes"), (CheckerLimits(max_nonzeros=0), "too_many_nonzeros"),
                        (CheckerLimits(max_expanded=1), "too_much_expansion"),
                        (CheckerLimits(max_master_nonzeros=0), "too_many_master_nonzeros")]:
        res = check_certificate(view, good, limits=limits, solve_master=ONES)
        assert not res.cert_ok and res.reason == why and res.lb == res.trivial == 0
    flood = json.dumps({"view": "toy", "cuts": [[] for _ in range(2000)]})
    res = check_certificate(view, flood, limits=CheckerLimits(max_nonzeros=10), solve_master=ONES)
    assert res.reason == "too_many_values"


@pytest.mark.parametrize("payload,reason", [
    ("not json", "parse_error"), ("[1, 2]", "malformed"), ('{"view": "other", "cuts": []}', "view_mismatch"),
    ('{"view": "toy", "cuts": 3}', "malformed"), ("[" * 100000 + "]" * 100000, "parse_error"),
])
def test_malformed_certificates_fail_safely(payload, reason):
    view = one_sub_view(THREE_ROWS, THREE_VARS)
    res = check_certificate(view, payload, solve_master=ONES)
    assert not res.cert_ok and res.reason == reason and res.lb == res.trivial


def test_degenerate_certificates():
    view = one_sub_view(THREE_ROWS, THREE_VARS)
    for payload in ('{"view": "toy"}', '{"view": "toy", "cuts": []}'):
        res = check_certificate(view, payload, solve_master=ONES)
        assert res.cert_ok and res.lb == 0
    res = check_certificate(view, json.dumps({"view": "toy", "cuts": [
        {"kind": "opt", "terms": []}, {"kind": "bogus", "terms": [{"s": "s", "pi": {}}]}, {"terms": [{"s": "s", "pi": {}}]},
        {"kind": "opt", "terms": [{"s": "zz", "pi": {}}]}, {"kind": "opt", "terms": [{"s": "s", "pi": {"nope": 1}}]},
        {"kind": "opt", "terms": [{"s": "s", "pi": [1]}]}, 7, {"kind": "opt", "terms": [{"s": "s", "pi": {"r": "1e20"}}]}]}),
        solve_master=ONES)
    assert res.cert_ok and res.stats["accepted"] == 0
    assert res.stats["rejected"] == {"no_terms": 1, "bad_kind": 2, "unknown_id": 1, "unknown_row": 1, "malformed_term": 1,
                                     "malformed": 1, "magnitude": 1}
    assert check_certificate(view, None).reason == "absent"
    unreadable = check_certificate(view, Path("/nonexistent/cert.json"))   # an evaluator I/O failure: infrastructure
    assert unreadable.infrastructure == "certificate_unreadable" and not unreadable.cert_ok


def test_failed_or_infeasible_master_gives_the_trivial_bound():
    view = one_sub_view(THREE_ROWS, THREE_VARS, y=[{"id": "y", "cost": "-2"}])
    good = cert([opt(("s", {"r": "0.25"}))])
    for status, reason in [("infeasible", "master_infeasible"), ("numerical", "numerical"), ("time_limit", "time_limit")]:
        res = check_certificate(view, good, solve_master=lambda *a, st=status: MasterSolve(st))
        assert not res.cert_ok and res.reason == reason and res.lb == res.trivial == -2


def test_feasibility_cut_proving_infeasibility_for_every_y():
    """x >= 2 with x in [0, 1] is infeasible for every y: the exact feasibility cut is 0 >= 1, the master is
    infeasible, and the bound is the trivial one (never +inf)."""
    view = one_sub_view([{"id": "r", "sense": ">=", "rhs": 2, "a": {"x": 1}}], [{"id": "x", "cost": 1, "lb": 0, "ub": 1}])
    res = check_certificate(view, cert([{"kind": "feas", "terms": [{"s": "s", "pi": {"r": "1"}}]}]), solve_master=ONES)
    assert not res.cert_ok and res.reason == "master_infeasible" and res.lb == res.trivial


def test_feasibility_cut_value():
    """x >= 1 - y, x in [0, 1/2]: feasible iff y >= 1/2. The Farkas multiplier 1 gives 0 >= (1 - y) - 1/2."""
    view = one_sub_view([{"id": "r", "sense": ">=", "rhs": 1, "a": {"x": 1}, "t": {"y": 1}}],
                        [{"id": "x", "cost": 0, "lb": 0, "ub": "1/2"}], y=[{"id": "y", "cost": 1}])
    den, const, ycoef = expand_term(view.subproblems[0], {"r": "1"}, True)
    assert Fraction(const, den) == Fraction(1, 2) and {k: Fraction(v, den) for k, v in ycoef.items()} == {0: -1}
    res = check_certificate(view, cert([{"kind": "feas", "terms": [{"s": "s", "pi": {"r": "1"}}]}]), solve_master=ONES)
    assert res.cert_ok and res.lb == Fraction(1, 2)  # min y s.t. y >= 1/2


# ---------------------------------------------------------------- exact arithmetic against a Fraction reference

def reference_term(sp, pi, zero_cost):
    rows = sp.rows
    dr, dv = rows.den, sp.den
    pis = {}
    for rid, v in pi.items():
        i = rows.index[rid]
        val = Fraction(*ratio(v))
        s = rows.sense[i]
        if (s == xc.GE and val < 0) or (s == xc.LE and val > 0):
            val = Fraction(0)
        pis[i] = val
    r = [Fraction(0) if zero_cost else Fraction(c, dv) for c in sp.C]
    const = Fraction(0)
    ycoef = {}
    for i, p in pis.items():
        const += p * Fraction(rows.H[i], dr)
        for j, a in zip(*rows.A[i]):
            r[j] -= p * Fraction(a, dr)
        for k, t in zip(*rows.T[i]):
            ycoef[k] = ycoef.get(k, Fraction(0)) - p * Fraction(t, dr)
    for j in range(sp.nvars):
        lo, hi = Fraction(sp.L[j], dv), Fraction(sp.U[j], dv)
        const += min(r[j] * lo, r[j] * hi)
    return const, {k: v for k, v in ycoef.items() if v}


def random_view(rng, n_y=3, n_sub=3, n_x=4, n_rows=4):
    y = [{"id": f"y{k}", "cost": str(rng.randint(-3, 9)), "lb": 0, "ub": 1} for k in range(n_y)]
    subs = []
    for s in range(n_sub):
        xs = [{"id": f"x{j}", "cost": f"{rng.randint(-50, 50) / 10}", "lb": str(rng.choice([0, -1, "-0.5"])),
               "ub": str(rng.choice([1, 2, "2.5"]))} for j in range(n_x)]
        rows = []
        for i in range(n_rows):
            a = {f"x{j}": f"{rng.randint(-20, 20) / 4}" for j in rng.sample(range(n_x), rng.randint(1, n_x))}
            if i == 0 and rng.random() < 0.6:  # a uniform row (same coefficient on every variable): the shift path
                coef = rng.choice(["1", "-2", "0.5"])
                a = {f"x{j}": coef for j in range(n_x)}
            t = {f"y{k}": f"{rng.randint(-9, 9) / 3:.6f}" for k in rng.sample(range(n_y), rng.randint(0, n_y))}
            rows.append({"id": f"r{i}", "sense": rng.choice([">=", "<=", "="]), "rhs": f"{rng.randint(-10, 10) / 8}", "a": a, "t": t})
        subs.append({"id": f"s{s}", "weight": rng.choice(["1", "0.25", "1/3"]), "vars": xs, "rows": rows})
    return BDView.from_dict({"view": "rand", "first_stage": {"vars": y, "rows": []}, "subproblems": subs})


def random_pi(rng, rows):
    pick = rng.sample(rows.ids, rng.randint(0, len(rows.ids)))
    vals = [lambda: rng.uniform(-5, 5), lambda: f"{rng.randint(-10**6, 10**6)}e-{rng.randint(0, 12)}",
            lambda: rng.randint(-3, 3), lambda: 1 / rng.choice([3, 7, 10])]
    return {rid: rng.choice(vals)() for rid in pick}


def test_integer_scaled_cuts_match_fraction_reference():
    rng = random.Random(7)
    shifted = 0
    for _ in range(60):
        view = random_view(rng, n_x=rng.choice([1, 4, 7]))
        for sp in view.subproblems:
            shifted += bool(sp.uniform)
            for zero_cost in (False, True):
                pi = random_pi(rng, sp.rows)
                den, const, ycoef = expand_term(sp, pi, zero_cost)
                rc, ry = reference_term(sp, pi, zero_cost)
                assert Fraction(const, den) == rc
                assert {k: Fraction(v, den) for k, v in ycoef.items() if v} == ry
    assert shifted >= 20


def test_uniform_row_shift_on_ufl_view():
    """The UFL assignment row is a uniform row: its multiplier is applied in closed form (sorted costs and prefix
    sums); sparse link multipliers are corrected column by column. Compare with the Fraction reference."""
    rng = random.Random(9)
    m, n = 9, 4
    D = [[f"{rng.randint(-50, 400) / 10}" for _ in range(n)] for _ in range(m)]
    view = xc.ufl_view({"f": ["1"] * m, "D": D})
    assert view.subproblems[0].uniform == {0: 1}
    for sp in view.subproblems:
        for _ in range(30):
            pi = {"d": rng.choice([rng.uniform(-10, 50), f"{rng.randint(-500, 500) / 10}", 0])}
            for i in rng.sample(range(m), rng.randint(0, m)):
                pi[f"l{i}"] = rng.choice([rng.uniform(-30, 5), -rng.randint(0, 9), f"-{rng.randint(0, 99)}.5"])
            for zero_cost in (False, True):
                stats = xc._Stats()
                den, const, ycoef = expand_term(sp, pi, zero_cost, stats)
                rc, ry = reference_term(sp, pi, zero_cost)
                assert Fraction(const, den) == rc
                assert {k: Fraction(v, den) for k, v in ycoef.items()} == ry
                assert stats.expanded <= 3 * len(pi)   # no O(m) work for the assignment row


def reference_recert(obj, lo, hi, rows, duals):
    lam, e = xc.snap_duals(duals, [r.sense for r in rows])
    d = list(obj)
    b = Fraction(0)
    for r, l in zip(rows, lam):
        lq = Fraction(l, 2 ** e)
        b += lq * Fraction(r.rhs, r.den)
        for c, v in zip(r.cols, r.nums):
            d[c] -= lq * Fraction(v, r.den)
    return b + sum(min(dj * a, dj * h) for dj, a, h in zip(d, lo, hi))


def test_recertification_matches_reference_and_is_valid_for_any_multipliers():
    """Weak duality over the boxes: whatever the multipliers (wrong signs, huge, NaN), the re-certified bound equals
    the Fraction reference and never exceeds the objective of a feasible master point."""
    rng = random.Random(11)
    view = random_view(rng, n_y=2, n_sub=2, n_x=2, n_rows=2)
    obj, lo, hi = xc._master_columns(view)
    stats = xc._Stats()
    rows = []
    for _ in range(30):
        sp = rng.choice(view.subproblems)
        cut = {"kind": rng.choice(["opt", "feas"]), "terms": [{"s": sp.id, "pi": random_pi(rng, sp.rows)}]}
        try:
            row = xc.build_cut(view, cut, stats, CheckerLimits())
        except xc.CutError:
            continue
        if row is not None:
            rows.append(row)
    assert rows
    # a feasible master point: theta at its upper box satisfies every valid opt cut only if recourse is feasible;
    # use exhaustive grid points that satisfy the (exact) rows instead
    feas_vals = []
    grid = [Fraction(0), Fraction(1, 2), Fraction(1)]
    for ys in itertools.product(grid, repeat=view.n):
        for ths in itertools.product(*[[sp.theta_lo, sp.theta_hi] for sp in view.subproblems]):
            z = list(ys) + list(ths)
            if all(sum(Fraction(v, r.den) * z[c] for c, v in zip(r.cols, r.nums)) >= Fraction(r.rhs, r.den) for r in rows):
                feas_vals.append(sum(o * zz for o, zz in zip(obj, z)))
    for trial in range(25):
        duals = [rng.choice([rng.uniform(-3, 3), rng.uniform(0, 1e6), 0.0, float("nan"), 1e-300, -1e300]) for _ in rows]
        b = recertify(obj, lo, hi, rows, duals)
        assert b == reference_recert(obj, lo, hi, rows, duals)
        for v in feas_vals:
            assert b <= v


def test_view_json_round_trip(tmp_path):
    rng = random.Random(3)
    view = random_view(rng)
    path = tmp_path / "bd_view.json"
    path.write_text(json.dumps(view.to_dict()))
    again = BDView.from_json(path)
    assert again.trivial_bound() == view.trivial_bound()
    for a, b in zip(view.subproblems, again.subproblems):
        pi = random_pi(rng, a.rows)
        da, ca, ya = expand_term(a, pi, False)
        db, cb, yb = expand_term(b, pi, False)
        assert Fraction(ca, da) == Fraction(cb, db)
        assert {k: Fraction(v, da) for k, v in ya.items()} == {k: Fraction(v, db) for k, v in yb.items()}
        assert (a.theta_lo, a.theta_hi) == (b.theta_lo, b.theta_hi)


def test_theta_box_default_formula_and_override_flag():
    """Review 43 counterexample: Q(y) = 10 (1 - y) on y in [0, 1]; a theta box from one design (U = Q(1) = 0) would
    make the master value 100 > LP* = 10. The default formula gives [0, 10]; explicit tighter boxes are flagged."""
    rows = [{"id": "r", "sense": ">=", "rhs": 1, "a": {"x": 1}, "t": {"y": 1}}]
    xs = [{"id": "x", "cost": 10, "lb": 0, "ub": 1}]
    view = one_sub_view(rows, xs, y=[{"id": "y", "cost": 100}])
    sp = view.subproblems[0]
    assert (sp.theta_lo, sp.theta_hi) == (0, 10) and not view.theta_overrides()
    res = check_certificate(view, cert([opt(("s", {"r": "10"}))]), solve_master=fixed_master(lambda rows: [1.0]))
    assert res.lb == 10  # theta >= 10 - 10 y with y-cost 100: the bound at y = 0 with multiplier 1
    with pytest.raises(ValueError, match="theta_proof"):   # review 47: overrides need a proved, trusted view
        one_sub_view(rows, xs, y=[{"id": "y", "cost": 100}], theta=(0, 0))
    trusted = BDView.from_dict({"view": "toy", "theta_proof": "docs/family-note.md#theta", "first_stage": {"vars": [{"id": "y", "cost": 100}]},
                                "subproblems": [{"id": "s", "vars": xs, "rows": rows, "theta_lo": 0, "theta_hi": 10}]})
    assert not trusted.theta_overrides()   # the default box itself is not an override
    trusted = BDView.from_dict({"view": "toy", "theta_proof": "docs/family-note.md#theta", "first_stage": {"vars": [{"id": "y", "cost": 100}]},
                                "subproblems": [{"id": "s", "vars": xs, "rows": rows, "theta_lo": 0, "theta_hi": 9}]})
    assert trusted.theta_overrides() == ["s"] and trusted.to_dict()["theta_proof"] == "docs/family-note.md#theta"


# ---------------------------------------------------------------- checker budget

class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_checker_budget_exhaustion_and_final_first(tmp_path):
    view = one_sub_view(THREE_ROWS, THREE_VARS)
    clock = FakeClock()
    files, shas = {}, {}
    for k, pi in enumerate(["0.1", "0.2", "0.3"]):
        p = tmp_path / f"C{k}.json"
        p.write_text(cert([opt(("s", {"r": pi}))]))
        files[k] = p.name
        shas[k] = hashlib.sha256(p.read_bytes()).hexdigest()

    def slow_master(lo, hi, obj, rows, time_limit, limits):
        clock.now += 400.0
        return MasterSolve("optimal", [1.0] * len(rows))

    cps = [{"checkpoint": 25.0, "file": files[0], "sha256": shas[0], "t": 1.0, "seq": 1},
           {"checkpoint": 50.0, "file": files[1], "sha256": shas[1], "t": 30.0, "seq": 2},
           {"checkpoint": 75.0, "file": files[1], "sha256": shas[1], "t": 30.0, "seq": 2},
           {"checkpoint": 100.0, "file": files[2], "sha256": shas[2], "t": 80.0, "seq": 3}]
    out = check_versions(view, cps, store_dir=tmp_path, budget_seconds=600.0, solve_master=slow_master, clock=clock)
    by_cp = {d["checkpoint"]: d for d in out}
    assert by_cp[100.0]["cert_ok"] and by_cp[100.0]["lb"] == Fraction(3, 10)          # final first: gets the budget
    assert by_cp[25.0]["reason"] == "checker_budget_exhausted"                        # 200 s left, the master took 400
    assert by_cp[50.0]["reason"] == "checker_budget_exhausted"
    assert by_cp[75.0]["reason"] == "checker_budget_exhausted" and not by_cp[75.0]["cached"]
    assert [d["checkpoint"] for d in out] == [25.0, 50.0, 75.0, 100.0]
    # identical versions are checked once
    out2 = check_versions(view, cps[1:3], store_dir=tmp_path, budget_seconds=1e9, solve_master=ONES)
    assert [d["cached"] for d in out2] == [True, False]  # the T_f-position entry is checked first
    assert check_versions(view, [{"checkpoint": 1.0, "file": None, "sha256": None}])[0]["reason"] == "absent"


def test_deadline_in_the_past_is_a_time_limit():
    view = one_sub_view(THREE_ROWS, THREE_VARS)
    clock = FakeClock()
    clock.now = 10.0
    res = check_certificate(view, cert([opt(("s", {"r": "0.25"}))]), deadline=5.0, clock=clock, solve_master=ONES)
    assert not res.cert_ok and res.reason == "time_limit" and res.lb == res.trivial


# ---------------------------------------------------------------- Gurobi: master, LP* interval, families

needs_gurobi = pytest.mark.skipif(gp is None, reason="gurobipy not installed")


@needs_gurobi
def test_gurobi_master_three_x_and_cancellation():
    v3 = one_sub_view(THREE_ROWS, THREE_VARS)
    r = check_certificate(v3, cert([opt(("s", {"r": "0.3333333333333333"}))]))
    assert r.cert_ok and r.lb == Fraction(3333333333333333, 10**16)
    out = lp_interval(v3)
    assert out["rigorous"] and Fraction(out["lp_lo"]) <= Fraction(1, 3) == Fraction(out["lp_hi"])
    vc = one_sub_view(CANCEL_ROWS, CANCEL_VARS)
    r = check_certificate(vc, cert([opt(("s", {"R1": "1", "R2": "1", "R3": "1"}))]))
    assert r.cert_ok and r.lb == 0


@needs_gurobi
def test_gurobi_infeasible_first_stage_rows():
    view = one_sub_view(THREE_ROWS, THREE_VARS, y=[{"id": "y", "cost": 1}],
                        fs_rows=[{"id": "bad", "sense": ">=", "rhs": 2, "coefs": {"y": 1}}])
    res = check_certificate(view, cert([opt(("s", {"r": "0.25"}))]))
    assert not res.cert_ok and res.reason == "master_infeasible" and res.lb == res.trivial == 0


@needs_gurobi
def test_fractional_box_check_detects_codex_counterexample():
    """Review 43: Q(y) = min{-x : x <= y, x <= 1 - y, x >= 0} = -min(y, 1 - y). A box x <= 1/4 leaves Q unchanged at
    binary y but changes it at y = 1/2 (-1/2 -> -1/4); the fractional check must catch it."""
    rows = [{"id": "a", "sense": "<=", "rhs": 0, "a": {"x": 1}, "t": {"y": -1}},
            {"id": "b", "sense": "<=", "rhs": 1, "a": {"x": 1}, "t": {"y": 1}}]
    bad = one_sub_view(rows, [{"id": "x", "cost": -1, "lb": 0, "ub": "1/4", "natural_ub": None}], y=[{"id": "y", "cost": 0}])
    assert xc.fractional_box_check(bad, ys=[[0.0], [1.0]])["ok"]
    res = xc.fractional_box_check(bad, ys=[[0.5]])
    assert not res["ok"] and res["mismatches"][0]["boxed"][1] == pytest.approx(-0.25)
    good = one_sub_view(rows, [{"id": "x", "cost": -1, "lb": 0, "ub": 1, "natural_ub": None}], y=[{"id": "y", "cost": 0}])
    assert xc.fractional_box_check(good, ys=[[0.0], [0.5], [0.3], [1.0]])["ok"]


def forged_cuts(rng, view, count, scale=1.0):
    cuts = []
    for _ in range(count):
        k = rng.randint(1, min(3, view.S))
        terms = []
        for sp in rng.sample(view.subproblems, k):
            pi = random_pi(rng, sp.rows)
            terms.append({"s": sp.id, "pi": {r: (v * scale if isinstance(v, float) else v) for r, v in pi.items()}})
        cuts.append({"kind": rng.choice(["opt", "opt", "feas"]), "terms": terms})
    return cuts


@needs_gurobi
def test_forged_certificates_never_exceed_lp_star(tmp_path):
    """Random multipliers (wrong signs, huge, tiny, aggregated over random subproblems, opt and feas) never give a
    certified bound above LP* (the rigorous LP_hi of the reference interval)."""
    rng = random.Random(5)
    checked = 0
    views = [random_view(rng) for _ in range(12)]
    views += [xc.ufl_view(write_ufl(tmp_path, seed=3)[0]), xc.fctp_view(write_fctp(tmp_path, seed=4)[0])]
    for view in views:
        ref = lp_interval(view)
        if ref["status"] != "optimal" or ref["lp_hi"] is None:
            continue
        assert ref["rigorous"]
        hi = Fraction(ref["lp_hi"])
        for scale in (1.0, 1e4):
            res = check_certificate(view, json.dumps({"view": view.view_id, "cuts": forged_cuts(rng, view, 80, scale)}))
            assert res.lb >= res.trivial
            if res.cert_ok:
                assert res.lb <= hi
                checked += 1
    assert checked >= 6


# ------------------------------------------- UFL

def write_ufl(tmp_path, m=5, n=8, seed=1):
    rng = random.Random(seed)
    f = [f"{rng.randint(20, 60) / 2}" for _ in range(m)]
    D = [[f"{rng.randint(10, 400) / 10}" for _ in range(n)] for _ in range(m)]
    p = tmp_path / "data.json"
    p.write_text('{"m": %d, "n": %d, "f": [%s], "D": [%s]}' % (m, n, ", ".join(f), ", ".join("[" + ", ".join(r) + "]" for r in D)))
    return p, [float(v) for v in f], [[float(v) for v in r] for r in D]


def ufl_lp_benders(f, D, max_iter=300):
    from gurobipy import GRB
    m, n = len(f), len(D[0])
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)
    env.start()
    M = gp.Model(env=env)
    y = M.addVars(m, lb=0, ub=1, obj=f)
    th = M.addVars(n, lb=0, ub=[sum(D[i][j] for i in range(m)) for j in range(n)], obj=[1.0] * n)
    M.addConstr(gp.quicksum(y.values()) >= 1)
    cuts = []
    for _ in range(max_iter):
        M.optimize()
        assert M.Status == GRB.OPTIMAL
        yv = [y[i].X for i in range(m)]
        added = 0
        for j in range(n):
            order = sorted(range(m), key=lambda i: D[i][j])
            cum, ck = 0.0, D[order[-1]][j]
            for i in order:
                cum += yv[i]
                if cum >= 1 - 1e-12:
                    ck = D[i][j]
                    break
            val = ck - sum(max(ck - D[i][j], 0.0) * yv[i] for i in range(m))
            if val > th[j].X + 1e-9:
                M.addConstr(th[j] >= ck - gp.quicksum(max(ck - D[i][j], 0.0) * y[i] for i in range(m)))
                cuts.append({"kind": "opt", "terms": [{"s": f"c{j}", "pi": {"d": repr(ck), **{f"l{i}": repr(-(ck - D[i][j]))
                                                                                         for i in range(m) if ck > D[i][j]}}}]})
                added += 1
        if not added:
            break
    obj = M.ObjVal
    M.dispose()
    env.dispose()
    return cuts, obj


@needs_gurobi
def test_ufl_view_lp_interval_and_benders_certificate_reach_lp_star(tmp_path):
    path, f, D = write_ufl(tmp_path)
    view = xc.ufl_view(path)
    assert view.stats()["row_blocks"] == 1 and view.S == 8
    ref = lp_interval(view)
    assert ref["rigorous"] and ref["repair"] == "family"
    lo, hi = Fraction(ref["lp_lo"]), Fraction(ref["lp_hi"])
    assert lo <= hi and float(hi - lo) <= 1e-7 * max(1.0, float(hi))
    # the compact strong LP (as in scaleup_calibration.run_mip, relaxed) has the same value
    m, n = len(f), len(D[0])
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    C = gp.Model(env=env)
    yv = C.addVars(m, lb=0, ub=1, obj=f)
    xv = C.addVars(m, n, lb=0, ub=1, obj={(i, j): D[i][j] for i in range(m) for j in range(n)})
    C.addConstrs(xv.sum("*", j) == 1 for j in range(n))
    C.addConstrs(xv[i, j] <= yv[i] for i in range(m) for j in range(n))
    C.optimize()
    assert C.ObjVal == pytest.approx(float(hi), rel=1e-9)
    C.dispose()
    env.dispose()
    cuts, master_obj = ufl_lp_benders(f, D)
    res = check_certificate(view, json.dumps({"view": "ufl-v1", "version": 1, "cuts": cuts}))
    assert res.cert_ok and res.stats["rejected"] == {} and res.stats["clipped_count"] == 0
    assert res.lb <= hi
    assert float(hi - res.lb) <= 1e-7 * max(1.0, float(hi))
    assert master_obj == pytest.approx(float(hi), rel=1e-7)
    # the empty certificate: master without cuts = min f.y over sum y >= 1 plus sum L_j
    empty = check_certificate(view, '{"view": "ufl-v1", "cuts": []}')
    assert empty.cert_ok and empty.lb == min(Fraction(v) for v in view.f)
    assert xc.fractional_box_check(view, samples=3, subproblems=[0, 3, 7])["ok"]


# ------------------------------------------- FCTP

def write_fctp(tmp_path, m=3, n=4, seed=2):
    import numpy as np
    from src.bd.fctp.a_data_xl import random_instance, write_fctp as wf
    inst = random_instance(np.random.default_rng(seed), m, n, fixed_scale=2.0)
    p = tmp_path / "inst.FCTP"
    wf(inst, p)
    return p, inst


def fctp_lp_benders(view, inst, max_iter=400):
    from gurobipy import GRB
    m, n = inst.m, inst.n
    cap = [min(inst.supply[i], inst.demand[j]) for i in range(m) for j in range(n)]
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)
    env.start()
    M = gp.Model(env=env)
    sp = view.subproblems[0]
    z = M.addVars(m * n + 1, lb=[0.0] * (m * n) + [float(sp.theta_lo)], ub=[1.0] * (m * n) + [float(sp.theta_hi)],
                  obj=[float(v) for v in view.f] + [1.0])
    S = gp.Model(env=env)
    S.Params.InfUnbdInfo = 1
    S.Params.DualReductions = 0
    S.Params.FeasibilityTol = 1e-9
    S.Params.OptimalityTol = 1e-9
    x = S.addVars(m * n, lb=0.0, ub=cap, obj=[float(inst.tcost[i, j]) for i in range(m) for j in range(n)])
    rows = {}
    for i in range(m):
        rows[f"s{i}"] = S.addConstr(gp.quicksum(x[i * n + j] for j in range(n)) == float(inst.supply[i]))
    for j in range(n):
        rows[f"d{j}"] = S.addConstr(gp.quicksum(x[i * n + j] for i in range(m)) == float(inst.demand[j]))
    link = {k: S.addConstr(x[k] <= 0.0) for k in range(m * n)}
    for k, c in link.items():
        rows[f"l{k // n}_{k % n}"] = c
    cuts, feas = [], 0
    for _ in range(max_iter):
        M.optimize()
        assert M.Status == GRB.OPTIMAL
        yv = [z[k].X for k in range(m * n)]
        for k, c in link.items():
            c.RHS = cap[k] * yv[k]
        S.optimize()
        if S.Status == GRB.INFEASIBLE:
            cut = {"kind": "feas", "terms": [{"s": "t", "pi": {rid: repr(-c.FarkasDual) for rid, c in rows.items()}}]}
            feas += 1
        else:
            assert S.Status == GRB.OPTIMAL
            if S.ObjVal <= z[m * n].X + 1e-7 * max(1.0, abs(S.ObjVal)):
                break
            cut = {"kind": "opt", "terms": [{"s": "t", "pi": {rid: repr(c.Pi) for rid, c in rows.items()}}]}
        row = xc.build_cut(view, cut, xc._Stats(), CheckerLimits())
        M.addLConstr(gp.quicksum(v * z[c] for c, v in zip(row.cols, row.fvals)), ">", row.frhs)
        cuts.append(cut)
    obj = M.ObjVal
    for mod in (S, M):
        mod.dispose()
    env.dispose()
    return cuts, obj, feas


@needs_gurobi
def test_fctp_view_matches_compact_lp_and_benders_certificate_reaches_lp_star(tmp_path):
    from src.bd.fctp.a_data_xl import load_fctp
    from src.bd.fctp.b_mip_xl import build_fctp_mip
    path, _ = write_fctp(tmp_path)
    inst = load_fctp(path)
    view = xc.fctp_view(path)
    assert view.fs_rows == [] and view.S == 1
    ref = lp_interval(view)
    assert ref["rigorous"], ref
    lo, hi = Fraction(ref["lp_lo"]), Fraction(ref["lp_hi"])
    assert lo <= hi and float(hi - lo) <= 1e-7 * max(1.0, float(hi))
    model, _ = build_fctp_mip(inst)
    model.Params.OutputFlag = 0
    relaxed = model.relax()
    relaxed.optimize()
    assert relaxed.ObjVal == pytest.approx(float(hi), rel=1e-9)  # identical first-stage rows (none)
    cuts, master_obj, feas = fctp_lp_benders(view, inst)
    assert feas >= 1  # the transportation recourse is not complete: feasibility cuts were needed
    res = check_certificate(view, json.dumps({"view": "fctp-v1", "version": 3, "cuts": cuts}))
    assert res.cert_ok and res.stats["rejected"] == {} and res.version == "3"
    assert res.stats["feas_cuts"] == feas
    assert res.lb <= hi
    assert float(hi - res.lb) <= 1e-6 * max(1.0, float(hi)), (float(res.lb), float(hi), master_obj)
    assert xc.fractional_box_check(view, samples=4)["ok"]


# ---------------------------------------------------------------- Codex review 45 regressions

def test_review45_1_float_model_data_are_rejected():
    """Model coefficients must be canonical: json.loads turns D = 0.1 into 3602879701896397/2^55, which would certify
    a bound above the declared LP* = 1/10. Float-bearing data are rejected; exact-text loading gives 1/10."""
    with pytest.raises(NumberError) as exc:
        xc.ufl_view(json.loads('{"f": [0], "D": [[0.1]]}'))
    assert exc.value.reason == "float_not_canonical"
    with pytest.raises(NumberError):
        xc.exact(0.1)
    assert xc.exact(0.1, allow_float=True) == Fraction(0.1)
    with pytest.raises(NumberError):
        BDView.from_dict({"view": "v", "first_stage": {"vars": [{"id": "y", "cost": 0.5}], "rows": []},
                          "subproblems": [{"id": "s", "vars": [{"id": "x", "cost": 1, "lb": 0, "ub": 1}], "rows": []}]})
    view = xc.ufl_view(xc.exact_json('{"f": [0], "D": [[0.1]]}'))
    assert view.subproblems[0].cost(0) == Fraction(1, 10)
    res = check_certificate(view, '{"view": "ufl-v1", "cuts": [{"kind": "opt", "terms": [{"s": "c0", "pi": {"d": 0.1}}]}]}',
                            solve_master=ONES)
    assert res.cert_ok and res.lb == Fraction(1, 10)   # multipliers may be floats or decimals: the bound is still valid


def test_review45_4_time_limit_is_enforced_before_accepting_a_result():
    clock = FakeClock()

    def slow(lo, hi, obj, rows, time_limit, limits):
        clock.now += 2.0
        return MasterSolve("optimal", [0.0] * len(rows))

    view = one_sub_view(THREE_ROWS, THREE_VARS)
    for payload in ('{"view": "toy", "cuts": []}', cert([opt(("s", {"r": "0.25"}))])):
        res = check_certificate(view, payload, limits=CheckerLimits(max_seconds=1.0), solve_master=slow, clock=clock)
        assert not res.cert_ok and res.reason == "time_limit" and res.lb == res.trivial
    # recertify checks the deadline even when every multiplier is zero (rows and columns)
    obj, lo, hi = xc._master_columns(view)
    clock.now = 10.0
    with pytest.raises(xc._CertFailure):
        recertify(obj, lo, hi, [], [], deadline=5.0, clock=clock)


def test_review45_4_huge_clipped_mass_is_serialized_exactly():
    view = one_sub_view(THREE_ROWS, THREE_VARS)
    payload = '{"view": "toy", "cuts": [{"kind": "opt", "terms": [{"s": "s", "pi": {"r": "-1e400"}}]}]}'
    res = check_certificate(view, payload, solve_master=ONES)
    assert res.cert_ok and res.stats["clipped_mass"] == "1" + "0" * 400 and res.stats["clipped_mass_float"] is None
    json.dumps(res.to_dict(), allow_nan=False)
    failed = check_certificate(view, payload, limits=CheckerLimits(max_master_nonzeros=0), solve_master=ONES)
    assert not failed.cert_ok and failed.reason == "too_many_master_nonzeros" and failed.stats["clipped_mass"] == "1" + "0" * 400
    json.dumps(failed.to_dict(), allow_nan=False)


def test_review45_5_check_versions_verifies_receipt_hashes(tmp_path):
    view = one_sub_view(THREE_ROWS, THREE_VARS)
    p = tmp_path / "C1.json"
    p.write_text(cert([opt(("s", {"r": "0.25"}))]))
    good = hashlib.sha256(p.read_bytes()).hexdigest()
    base = {"checkpoint": 100.0, "file": "C1.json", "t": 1.0, "seq": 1}
    ok = check_versions(view, [base | {"sha256": good}], store_dir=tmp_path, solve_master=ONES)[0]
    assert ok["cert_ok"] and ok["infrastructure"] is None and ok["lb"] == Fraction(1, 4)
    bad = check_versions(view, [base | {"sha256": "0" * 64}], store_dir=tmp_path, solve_master=ONES)[0]
    assert not bad["cert_ok"] and bad["infrastructure"] == "hash_mismatch" and bad["lb"] == 0
    missing = check_versions(view, [base | {"sha256": good, "file": "nope.json"}], store_dir=tmp_path, solve_master=ONES)[0]
    assert missing["infrastructure"] == "stored_version_unreadable"
    pruned = check_versions(view, [base | {"sha256": good, "file": None}], store_dir=tmp_path, solve_master=ONES)[0]
    assert pruned["infrastructure"] == "stored_version_missing"
    absent = check_versions(view, [{"checkpoint": 100.0, "file": None, "sha256": None, "seq": None}])[0]
    assert absent["reason"] == "absent" and absent["infrastructure"] is None


# ---------------------------------------------------------------- Codex review 47 regressions

def test_review47_2_expansion_limit_counts_the_work_of_rejected_cuts():
    view = one_sub_view(THREE_ROWS, THREE_VARS)
    for pi in ("0.25", "1e100"):   # 1e100 is rejected for magnitude only after its expansion
        res = check_certificate(view, cert([opt(("s", {"r": pi}))]), limits=CheckerLimits(max_expanded=0), solve_master=ONES)
        assert not res.cert_ok and res.reason == "too_much_expansion" and res.lb == res.trivial


def _stored(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return {"file": name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}


def test_review47_2_reading_and_hashing_count_against_the_budget(tmp_path, monkeypatch):
    view = one_sub_view(THREE_ROWS, THREE_VARS)
    cp = {"checkpoint": 100.0, "t": 1.0, "seq": 1} | _stored(tmp_path, "C1.json", cert([opt(("s", {"r": "0.25"}))]))
    clock = FakeClock()
    real = Path.read_bytes

    def slow_read(self):
        clock.now += 10.0   # a 10 s read under a 1 s budget
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", slow_read)
    out = check_versions(view, [cp], store_dir=tmp_path, budget_seconds=1.0, solve_master=ONES, clock=clock)[0]
    assert not out["cert_ok"] and out["reason"] == "checker_budget_exhausted" and out["checker_seconds_used"] >= 10.0


def test_review47_3_every_distinct_stored_file_is_verified_before_cache_reuse(tmp_path):
    view = one_sub_view(THREE_ROWS, THREE_VARS)
    good = _stored(tmp_path, "good.json", cert([opt(("s", {"r": "0.25"}))]))
    (tmp_path / "corrupt.json").write_text("BROKEN" + " " * (good["bytes"] - 6))
    cps = [{"checkpoint": 25.0, "t": 1.0, "seq": 1, "file": "corrupt.json", "sha256": good["sha256"], "bytes": good["bytes"]},
           {"checkpoint": 100.0, "t": 2.0, "seq": 2} | good]
    out = check_versions(view, cps, store_dir=tmp_path, solve_master=ONES)
    assert out[1]["cert_ok"] and out[1]["infrastructure"] is None
    assert not out[0]["cert_ok"] and out[0]["infrastructure"] == "hash_mismatch" and not out[0]["cached"]
    (tmp_path / "short.json").write_text("x")
    short = check_versions(view, [cps[1] | {"file": "short.json"}], store_dir=tmp_path, solve_master=ONES)[0]
    assert short["infrastructure"] == "size_mismatch"


linux_fork = pytest.mark.skipif(sys.platform != "linux", reason="the supervised checker forks (Linux)")


@linux_fork
def test_review47_2_supervised_checker_matches_and_enforces_one_hard_deadline(tmp_path, monkeypatch):
    import time as _time

    view = one_sub_view(THREE_ROWS, THREE_VARS)
    cps = [{"checkpoint": 50.0, "t": 1.0, "seq": 1} | _stored(tmp_path, "C1.json", cert([opt(("s", {"r": "0.2"}))])),
           {"checkpoint": 100.0, "t": 2.0, "seq": 2} | _stored(tmp_path, "C2.json", cert([opt(("s", {"r": "0.25"}))]))]
    same = xc.check_versions_supervised(view, cps, store_dir=tmp_path, budget_seconds=60.0, solve_master=ONES)
    assert [d["lb"] for d in same] == [Fraction(1, 5), Fraction(1, 4)] and all(d["checker_process"] == "done" for d in same)

    def hang(*args):
        _time.sleep(60)
        return MasterSolve("optimal", [1.0])

    t0 = _time.monotonic()
    out = xc.check_versions_supervised(view, cps, store_dir=tmp_path, budget_seconds=1.0, solve_master=hang, init_solver=False)
    assert _time.monotonic() - t0 < 10.0
    assert [d["reason"] for d in out] == ["checker_budget_exhausted"] * 2 and all(d["lb"] == 0 for d in out)
    assert out[1]["checker_process"] == "killed"
    real = Path.read_bytes

    def slow_read(self):   # a hard 10 s read under a 1 s budget (inherited by the forked checker)
        _time.sleep(10)
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", slow_read)
    t0 = _time.monotonic()
    out = xc.check_versions_supervised(view, cps, store_dir=tmp_path, budget_seconds=1.0, solve_master=ONES, init_solver=False)
    assert _time.monotonic() - t0 < 6.0 and out[1]["reason"] == "checker_budget_exhausted" and not out[1]["cert_ok"]


# ---------------------------------------------------------------- Codex review 49 regressions

def test_review49_1_version_is_compact_metadata_and_certificates_are_strict_utf8():
    view = one_sub_view(THREE_ROWS, THREE_VARS)
    big = check_certificate(view, json.dumps({"view": "toy", "version": "x" * 32768, "cuts": []}), solve_master=ONES)
    assert big.cert_ok and big.version is None
    assert check_certificate(view, '{"view": "toy", "version": 12, "cuts": []}', solve_master=ONES).version == "12"
    text = cert([opt(("s", {"r": "0.25"}))])
    for data, why in [(text.encode("utf-16"), "not_utf8"), (text.encode("utf-16-be"), "parse_error"),
                      (text.encode("utf-32"), "not_utf8"), (text.encode("utf-8-sig"), "bom")]:
        res = check_certificate(view, data, solve_master=ONES)
        assert not res.cert_ok and res.reason == why, (why, res.reason)


def test_review49_3_master_solver_errors_and_memory_errors_in_process_are_infrastructure():
    view = one_sub_view(THREE_ROWS, THREE_VARS)
    res = check_certificate(view, cert([opt(("s", {"r": "0.25"}))]), solve_master=lambda *a: MasterSolve("error", detail="license"))
    assert not res.cert_ok and res.infrastructure == "master_solver_error"
    assert xc.failure_category(res.cert_ok, res.reason, res.infrastructure) == "infrastructure"

    def oom(*a):
        raise MemoryError

    store = Path(__import__("tempfile").mkdtemp())
    cp = {"checkpoint": 100.0, "t": 1.0, "seq": 1} | _stored(store, "C.json", cert([opt(("s", {"r": "0.25"}))]))
    out = check_versions(view, [cp], store_dir=store, solve_master=oom)[0]
    assert out["infrastructure"] == "memory_error_unattributed" and out["failure"]["category"] == "infrastructure"
    ok = check_versions(view, [cp], store_dir=store, solve_master=ONES)[0]
    assert ok["cert_ok"] and ok["failure"] is None


@linux_fork
def test_review49_1_ipc_never_blocks_past_the_deadline(tmp_path, monkeypatch):
    """Codex: a pause in the middle of a result frame (header sent, body delayed) must not let the parent block or
    accept the result after the deadline."""
    import os as _os
    import time as _time

    view = one_sub_view(THREE_ROWS, THREE_VARS)
    cp = {"checkpoint": 100.0, "t": 1.0, "seq": 1} | _stored(
        tmp_path, "C.json", json.dumps({"view": "toy", "version": "x" * 32768, "cuts": [opt(("s", {"r": "0.25"}))]}))
    real_write = _os.write

    def split_write(fd, data):
        data = bytes(data)
        if b'"type": "result"' in data:
            real_write(fd, data[:4])      # the frame header only
            _time.sleep(3.0)
            return 4
        return real_write(fd, data)

    monkeypatch.setattr(_os, "write", split_write)
    t0 = _time.monotonic()
    out = xc.check_versions_supervised(view, [cp], store_dir=tmp_path, budget_seconds=0.5, solve_master=ONES, init_solver=False)[0]
    assert _time.monotonic() - t0 < 2.0
    assert not out["cert_ok"] and out["reason"] == "checker_budget_exhausted" and out["checker_process"] == "killed"
    assert out["failure"]["category"] == "certificate_resource" and out["checker_setup"] is not None
    monkeypatch.setattr(_os, "write", real_write)
    ok = xc.check_versions_supervised(view, [cp], store_dir=tmp_path, budget_seconds=30.0, solve_master=ONES)[0]
    assert ok["cert_ok"] and ok["version"] is None and ok["lb"] == Fraction(1, 4)


@linux_fork
def test_review49_3_supervised_failure_attribution(tmp_path):
    import os as _os
    import signal as _signal

    view = one_sub_view(THREE_ROWS, THREE_VARS)
    cp = {"checkpoint": 100.0, "t": 1.0, "seq": 1} | _stored(tmp_path, "C.json", cert([opt(("s", {"r": "0.25"}))]))

    def bug(*a):
        raise RuntimeError("trusted evaluator bug")

    out = xc.check_versions_supervised(view, [cp], store_dir=tmp_path, budget_seconds=30.0, solve_master=bug)[0]
    assert out["infrastructure"] == "checker_error" and out["failure"] == {"stage": "master", "checkpoint": 0, "category": "infrastructure"}
    assert "trusted evaluator bug" in out["stats"]["child_error"]

    def die(*a):
        _os.kill(_os.getpid(), _signal.SIGKILL)

    out = xc.check_versions_supervised(view, [cp], store_dir=tmp_path, budget_seconds=30.0, solve_master=die)[0]
    assert out["infrastructure"] == "checker_died" and out["failure"]["category"] == "infrastructure"

    baseline = xc._vm_bytes()
    limit = 4 * baseline

    def hog(*a):
        bytearray(limit)   # above the frozen RLIMIT_AS: MemoryError while processing the certificate
        return MasterSolve("optimal", [1.0])

    out = xc.check_versions_supervised(view, [cp], store_dir=tmp_path, budget_seconds=30.0, solve_master=hog, memory_bytes=limit)[0]
    assert out["checker_setup"]["view_fits"] is True
    assert out["reason"] == "certificate_memory_limit" and out["infrastructure"] is None
    assert out["failure"] == {"stage": "master", "checkpoint": 0, "category": "certificate_resource"}

    def oom(*a):
        raise MemoryError

    out = xc.check_versions_supervised(view, [cp], store_dir=tmp_path, budget_seconds=30.0, solve_master=oom)[0]
    assert out["infrastructure"] == "memory_error_unattributed"   # no explicit frozen memory limit: not attributable


# ---------------------------------------------------------------- Codex review 50 regressions

def _fake_gurobipy(*, start_sleep: float = 0.0, start_error: bool = False, model_oom: bool = False):
    """A stand-in gurobipy: Env.start can stall (licensing) or fail with errno 10001; Model can fail with 10001."""
    import types as _types
    import time as _time

    class FakeGurobiError(Exception):
        def __init__(self, msg: str = "", errno: int | None = None) -> None:
            super().__init__(msg)
            self.errno = errno

    class Env:
        def __init__(self, *a, **kw) -> None:
            pass

        def setParam(self, *a) -> None:
            pass

        def start(self) -> None:
            if start_error:
                raise FakeGurobiError("Out of memory", 10001)
            _time.sleep(start_sleep)

        def dispose(self) -> None:
            pass

    def Model(*a, **kw):
        if model_oom:
            raise FakeGurobiError("Out of memory", 10001)
        raise FakeGurobiError("no model in the fake", 10009)

    mod = _types.ModuleType("gurobipy")
    mod.Env, mod.Model, mod.GurobiError, mod.GRB = Env, Model, FakeGurobiError, _types.SimpleNamespace()
    return mod


def test_review50_1_gurobi_memory_outcomes_go_through_the_gate_in_process(tmp_path, monkeypatch):
    view = one_sub_view(THREE_ROWS, THREE_VARS)
    cp = {"checkpoint": 100.0, "t": 1.0, "seq": 1} | _stored(tmp_path, "C.json", '{"view": "toy", "cuts": []}')
    monkeypatch.setitem(sys.modules, "gurobipy", _fake_gurobipy(model_oom=True))
    out = check_versions(view, [cp], store_dir=tmp_path, limits=CheckerLimits(gurobi_mem_gb=None))[0]
    assert out["infrastructure"] == "memory_error_unattributed" and out["failure"]["category"] == "infrastructure"
    assert out["stats"]["master_status"] == "memory_limit"


@linux_fork
def test_review50_1_and_2_supervised_memory_gate_and_trusted_initialization(tmp_path, monkeypatch):
    import time as _time

    view = one_sub_view(THREE_ROWS, THREE_VARS)
    cp = {"checkpoint": 100.0, "t": 1.0, "seq": 1} | _stored(tmp_path, "C.json", '{"view": "toy", "cuts": []}')   # empty, valid
    # Gurobi out-of-memory with both memory caps disabled: infrastructure (Codex review 50 #1)
    monkeypatch.setitem(sys.modules, "gurobipy", _fake_gurobipy(model_oom=True))
    out = xc.check_versions_supervised(view, [cp], store_dir=tmp_path, budget_seconds=30.0, limits=CheckerLimits(gurobi_mem_gb=None))[0]
    assert out["infrastructure"] == "memory_error_unattributed" and out["checker_setup"]["view_fits"] is False
    # ... and with an explicit frozen memory cap that the trusted setup fits well within: charged to the certificate
    limit = 4 * xc._vm_bytes()
    out = xc.check_versions_supervised(view, [cp], store_dir=tmp_path, budget_seconds=30.0, memory_bytes=limit)[0]
    assert out["reason"] == "certificate_memory_limit" and out["failure"]["category"] == "certificate_resource"
    # a licensing stall during trusted initialization is infrastructure, never the certificate's budget (Codex #2)
    monkeypatch.setitem(sys.modules, "gurobipy", _fake_gurobipy(start_sleep=5.0))
    t0 = _time.monotonic()
    out = xc.check_versions_supervised(view, [cp], store_dir=tmp_path, budget_seconds=0.2)[0]
    assert _time.monotonic() - t0 < 3.0
    assert out["infrastructure"] == "checker_setup_timeout" and out["checker_setup"] is None
    assert out["failure"] == {"stage": "setup", "checkpoint": 0, "category": "infrastructure"}
    monkeypatch.setitem(sys.modules, "gurobipy", _fake_gurobipy(start_error=True))
    out = xc.check_versions_supervised(view, [cp], store_dir=tmp_path, budget_seconds=30.0)[0]
    assert out["infrastructure"] == "checker_setup_error" and "Out of memory" in out["stats"]["child_error"]


@linux_fork
@needs_gurobi
def test_review50_2_real_trusted_initialization_is_reused_by_the_master(tmp_path):
    view = one_sub_view(THREE_ROWS, THREE_VARS)
    cps = [{"checkpoint": 50.0, "t": 1.0, "seq": 1} | _stored(tmp_path, "A.json", cert([opt(("s", {"r": "0.2"}))])),
           {"checkpoint": 100.0, "t": 2.0, "seq": 2} | _stored(tmp_path, "B.json", cert([opt(("s", {"r": "0.3333333333333333"}))]))]
    out = xc.check_versions_supervised(view, cps, store_dir=tmp_path, budget_seconds=120.0)
    assert out[0]["checker_setup"]["solver_initialized"] is True
    assert [d["lb"] for d in out] == [Fraction(1, 5), Fraction(3333333333333333, 10**16)] and all(d["cert_ok"] for d in out)


@linux_fork
def test_review50_4_poll_above_fd_1024_cleanup_and_structured_startup_failures(tmp_path, monkeypatch):
    import os as _os
    import resource as _resource
    import select as _select

    view = one_sub_view(THREE_ROWS, THREE_VARS)
    cp = {"checkpoint": 100.0, "t": 1.0, "seq": 1} | _stored(tmp_path, "C.json", cert([opt(("s", {"r": "0.25"}))]))
    soft, hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
    if hard != _resource.RLIM_INFINITY and hard < 1200:
        pytest.skip("cannot open 1,100 descriptors here")
    _resource.setrlimit(_resource.RLIMIT_NOFILE, (max(soft, 2048) if hard == _resource.RLIM_INFINITY else min(hard, 4096), hard))
    opened = []
    try:
        for _ in range(1100):
            opened.append(_os.open("/dev/null", _os.O_RDONLY))
        out = xc.check_versions_supervised(view, [cp], store_dir=tmp_path, budget_seconds=30.0, solve_master=ONES, init_solver=False)[0]
        assert out["cert_ok"] and out["lb"] == Fraction(1, 4)    # the pipe descriptor is above 1024: poll() handles it
    finally:
        for fd in opened:
            _os.close(fd)
        _resource.setrlimit(_resource.RLIMIT_NOFILE, (soft, hard))

    def no_fork():
        raise OSError("fork: resource temporarily unavailable")

    monkeypatch.setattr(_os, "fork", no_fork)
    out = xc.check_versions_supervised(view, [cp], store_dir=tmp_path, budget_seconds=30.0, solve_master=ONES)[0]
    assert out["infrastructure"] == "checker_startup_error" and out["checker_process"] == "startup_error"
    monkeypatch.undo()

    class BrokenPoll:
        def register(self, *a):
            pass

        def poll(self, *a):
            raise ValueError("injected IPC failure")

    monkeypatch.setattr(_select, "poll", BrokenPoll)
    out = xc.check_versions_supervised(view, [cp], store_dir=tmp_path, budget_seconds=30.0, solve_master=ONES, init_solver=False)[0]
    assert out["infrastructure"] == "checker_ipc_error" and "injected IPC failure" in out["stats"]["ipc_detail"]
    assert out["stats"]["child"] != "unknown"     # the child was killed and reaped


# ---------------------------------------------------------------- Codex review 52 regressions

def _late(status: str):
    import time as _time

    def solve(lo, hi, obj, rows, time_limit, limits):
        _time.sleep(0.04)   # returned after the 0.01 s per-check limit
        return MasterSolve(status, [1.0] * len(rows) if status == "optimal" else None)
    return solve


def test_review52_3_returned_solver_status_is_classified_before_the_time_fallback(tmp_path):
    view = one_sub_view(THREE_ROWS, THREE_VARS)
    cp = {"checkpoint": 100.0, "t": 1.0, "seq": 1} | _stored(tmp_path, "C.json", cert([opt(("s", {"r": "0.25"}))]))
    lim = CheckerLimits(max_seconds=0.01)
    late_mem = check_versions(view, [cp], store_dir=tmp_path, limits=lim, budget_seconds=2.0, solve_master=_late("memory_limit"))[0]
    assert late_mem["infrastructure"] == "memory_error_unattributed" and late_mem["failure"]["category"] == "infrastructure"
    late_err = check_versions(view, [cp], store_dir=tmp_path, limits=lim, budget_seconds=2.0, solve_master=_late("error"))[0]
    assert late_err["infrastructure"] == "master_solver_error"
    late_inf = check_versions(view, [cp], store_dir=tmp_path, limits=lim, budget_seconds=2.0, solve_master=_late("infeasible"))[0]
    assert late_inf["reason"] == "time_limit" and late_inf["failure"]["category"] == "certificate_resource"


@linux_fork
def test_review52_3_supervised_late_memory_status_without_a_cap_is_infrastructure(tmp_path):
    view = one_sub_view(THREE_ROWS, THREE_VARS)
    cp = {"checkpoint": 100.0, "t": 1.0, "seq": 1} | _stored(tmp_path, "C.json", cert([opt(("s", {"r": "0.25"}))]))
    out = xc.check_versions_supervised(view, [cp], store_dir=tmp_path, limits=CheckerLimits(max_seconds=0.01), budget_seconds=2.0,
                                       solve_master=_late("memory_limit"), init_solver=False)[0]
    assert out["infrastructure"] == "memory_error_unattributed" and out["reason"] == "infrastructure"


def test_review52_1_convert_hook_separates_rejections_from_evaluator_failures(tmp_path):
    view = one_sub_view(THREE_ROWS, THREE_VARS)
    cp = {"checkpoint": 100.0, "t": 1.0, "seq": 1} | _stored(tmp_path, "C.json", cert([opt(("s", {"r": "0.25"}))], view="FAMILY"))
    ok = check_versions(view, [cp], store_dir=tmp_path, solve_master=ONES, convert=lambda d: d.replace(b'"FAMILY"', b'"toy"'))[0]
    assert ok["cert_ok"] and ok["lb"] == Fraction(1, 4)

    def rejected(d):
        raise xc.CertificateRejected("parse_error")

    def oom(d):
        raise MemoryError

    def bug(d):
        raise RuntimeError("converter bug")

    r = check_versions(view, [cp], store_dir=tmp_path, solve_master=ONES, convert=rejected)[0]
    assert r["reason"] == "rejected:parse_error" and r["infrastructure"] is None and r["failure"]["category"] == "certificate"
    r = check_versions(view, [cp], store_dir=tmp_path, solve_master=ONES, convert=oom)[0]
    assert r["infrastructure"] == "memory_error_unattributed" and r["failure"]["stage"] == "convert"
    with pytest.raises(RuntimeError):   # in process an evaluator bug propagates; supervised it is infrastructure
        check_versions(view, [cp], store_dir=tmp_path, solve_master=ONES, convert=bug)


@linux_fork
def test_review52_1_supervised_converter_failure_is_infrastructure(tmp_path):
    view = one_sub_view(THREE_ROWS, THREE_VARS)
    cp = {"checkpoint": 100.0, "t": 1.0, "seq": 1} | _stored(tmp_path, "C.json", cert([opt(("s", {"r": "0.25"}))], view="FAMILY"))

    def bug(d):
        raise RuntimeError("converter bug")

    out = xc.check_versions_supervised(view, [cp], store_dir=tmp_path, solve_master=ONES, convert=bug, init_solver=False)[0]
    assert out["infrastructure"] == "checker_error" and out["failure"] == {"stage": "convert", "checkpoint": 0, "category": "infrastructure"}
