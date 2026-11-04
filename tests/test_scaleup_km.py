"""Correctness checks for the scale-up k-median family (data and distances, compact MIP/LP, oracle, naive and
engineered Benders, local search, Lagrangian, certificate checker, verifier) against brute force over all designs of
small generated instances. Solver tests: run on Phoenix only (no optimization on the laptop)."""
from __future__ import annotations

import functools
import itertools
import json
import math
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

gp = pytest.importorskip("gurobipy")
from gurobipy import GRB  # noqa: E402

from src.bd.km.a_data_xl import (KMInstance, SortedCosts, load_instance, random_instance, read_tsplib,  # noqa: E402
                                 tsplib_instance, write_instance)
from src.bd.km.b_mip_xl import (KMCompactLP, KMCompactMIP, build_classical, build_radius, design_value,  # noqa: E402
                                km_output)
from src.bd.km.c_naive_bd_xl import KMNaiveBenders, KMNaiveLPBenders, Oracle, greedy_design  # noqa: E402
from src.bd.km.d_ref_bd_xl import KMBendersXL  # noqa: E402
from src.bd.km.e_verifier_xl import distance, read_instance, verify_solution  # noqa: E402
from src.bd.km.f_heuristic_xl import KMLagrangian, KMLocalSearch, SwapSearch  # noqa: E402
from src.bd.km.g_certificate_xl import (Certificate, Checker, Limits, aggregated, exact_master_value,  # noqa: E402
                                        lp_interval, one_shot, parse_certificate, repair_point, singles, to_bytes,
                                        to_json)

SHAPES = [(6, 6, 2, True), (7, 9, 3, False), (8, 8, 3, True), (5, 7, 2, False), (8, 10, 4, False), (6, 6, 1, True),
          (7, 7, 6, True), (8, 6, 2, False)]


def tiny(seed: int, tmp_path: Path, integer: bool | None = None) -> tuple[KMInstance, str]:
    rng = np.random.default_rng(seed)
    m, n, k, same = SHAPES[seed % len(SHAPES)]
    integer = (seed % 3 != 2) if integer is None else integer
    inst = random_instance(rng, m, n, k, integer=integer, same_sites=same and m == n, span=8)
    path = tmp_path / f"km{seed}.json"
    write_instance(inst, path)
    return load_instance(path), str(path)


@functools.lru_cache(maxsize=None)
def _brute(path: str) -> tuple[float, tuple[int, ...]]:
    inst = load_instance(path)
    D = inst.distances()
    best = (math.inf, ())
    for S in itertools.combinations(range(inst.m), inst.k):
        v = design_value(D, S)
        if v < best[0]:
            best = (v, S)
    return best


@functools.lru_cache(maxsize=None)
def _lp(path: str) -> float:
    inst = load_instance(path)
    md, _ = build_classical(inst, inst.distances(), relax=True, reduce=False)
    md.Params.OutputFlag = 0
    md.optimize()
    assert md.Status == GRB.OPTIMAL
    v = float(md.ObjVal)
    md.dispose()
    return v


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


# ---------------------------------------------------------------- data

def test_tsplib_reader_and_distances(tmp_path):
    f = tmp_path / "toy.tsp"
    f.write_text("NAME : toy\nTYPE : TSP\nDIMENSION : 4\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n"
                 "1 0.0 0.0\n2 3.0e+00 4.0\n3 1.5 2.25\n4 1.10461e+01 7.0\nEOF\n")
    name, xy = read_tsplib(f)
    assert name == "toy" and xy.shape == (4, 2) and xy[3, 0] == 11.0461
    inst = tsplib_instance(f, 2)
    D = inst.distances()
    assert D[0, 1] == 5.0 and D[1, 0] == 5.0 and np.all(np.diag(D) == 0) and np.all(D == np.floor(D))
    path = tmp_path / "toy.json"
    write_instance(inst, path)
    raw = read_instance(path)
    for i in range(4):
        for j in range(4):
            assert distance(raw, i, j) == D[i, j]  # bit-identical stdlib distances


@pytest.mark.parametrize("integer", [True, False])
def test_json_roundtrip_and_bit_identical_distances(tmp_path, integer):
    inst, path = tiny(4, tmp_path, integer=integer)
    D = inst.distances()
    raw = read_instance(path)
    assert all(distance(raw, i, j) == D[i, j] for i in range(inst.m) for j in range(inst.n))
    again = load_instance(path)
    assert np.array_equal(again.distances(), D) and again.k == inst.k


def test_sorted_costs_prefix_covers_every_design(tmp_path):
    for seed in range(6):
        inst, _ = tiny(seed, tmp_path)
        D = inst.distances()
        sc = SortedCosts(D, inst.k)
        assert sc.order.shape == (inst.n, inst.m - inst.k + 1)
        assert np.all(np.diff(sc.ds, axis=1) >= 0)
        for S in itertools.combinations(range(inst.m), inst.k):
            # every design opens one of the L nearest facilities of every customer, at the true nearest distance
            assert np.all(np.isin(sc.order, S).any(axis=1))
            assert np.allclose(np.where(np.isin(sc.order, S), sc.ds, np.inf).min(axis=1), D[list(S)].min(axis=0))


# ---------------------------------------------------------------- compact models

@pytest.mark.parametrize("seed", range(8))
def test_compact_mips_match_brute_force(tmp_path, seed):
    inst, path = tiny(seed, tmp_path)
    opt, _ = _brute(path)
    for kw in ({"formulation": "radius"}, {"formulation": "classical"}, {"formulation": "classical", "reduce": False}):
        designs = []
        res = KMCompactMIP(time_limit=30, on_incumbent=lambda d, stage=None: designs.append(d), **kw).run(path)
        assert res["termination_reason"] == "optimal", kw
        assert close(design_value(inst.distances(), res["design"]), opt), kw
        assert res["global_lb"] <= opt + 1e-6 and designs


@pytest.mark.parametrize("seed", range(6))
def test_lp_relaxations_agree(tmp_path, seed):
    inst, path = tiny(seed, tmp_path)
    D = inst.distances()
    vals = []
    for build in (lambda: build_classical(inst, D, relax=True, reduce=True), lambda: build_radius(inst, D, relax=True)):
        md, _ = build()
        md.Params.OutputFlag = 0
        md.optimize()
        vals.append(float(md.ObjVal))
        md.dispose()
    lp = _lp(path)
    assert close(vals[0], lp) and close(vals[1], lp)
    assert lp <= _brute(path)[0] + 1e-6
    chk = Checker(inst)
    for method in (1, 2):
        res = KMCompactLP(time_limit=30, method=method).run(path)
        assert res["termination_reason"] == "optimal" and close(res["lp_value"], lp)
        cert = chk.certify(chk.dual_vector_certificate(res["certificate_v"]))
        assert cert["cert_ok"] and cert["bound"] <= lp + 1e-9 and close(cert["bound"], lp, 1e-6)
        iv = lp_interval(chk, [("lp", cert)], [("lp point", np.array(res["lp_point"]))])
        assert iv["rigorous"] and iv["lo_float"] <= lp + 1e-9 <= iv["hi_float"] + 1e-6 * max(1, lp)


# ---------------------------------------------------------------- oracle

def _sub_lp(D: np.ndarray, j: int, y: np.ndarray) -> float:
    md = gp.Model()
    md.Params.OutputFlag = 0
    x = md.addMVar(D.shape[0], lb=0.0, ub=y)
    md.addConstr(x.sum() == 1)
    md.setObjective(D[:, j] @ x, GRB.MINIMIZE)
    md.optimize()
    v = float(md.ObjVal)
    md.dispose()
    return v


@pytest.mark.parametrize("seed", range(6))
def test_oracle_matches_subproblem_lp(tmp_path, seed):
    inst, _ = tiny(seed, tmp_path)
    D = inst.distances()
    oracle = Oracle(D, inst.k)
    rng = np.random.default_rng(seed)
    for _ in range(4):
        y = rng.dirichlet(np.ones(inst.m)) * inst.k
        while y.max() > 1:  # project into [0, 1] with sum k
            ex = np.maximum(y - 1, 0).sum()
            y = np.minimum(y, 1)
            free = y < 1
            y[free] += ex * y[free] / y[free].sum()
        v, Q, cnt = oracle.separate(y)
        for j in range(inst.n):
            assert close(Q[j], _sub_lp(D, j, y), 1e-7)
            # the cut (j, v_j) is tight at y and valid at every design
            idx, coef = oracle.coef_dense(j, v[j])
            assert close(v[j] - coef @ y[idx], Q[j], 1e-7)
            for S in itertools.combinations(range(inst.m), inst.k):
                yy = np.zeros(inst.m)
                yy[list(S)] = 1
                assert v[j] - coef @ yy[idx] <= D[list(S), j].min() + 1e-9


# ---------------------------------------------------------------- heuristics

def test_swap_delta_formula_is_exact(tmp_path):
    for seed in range(6):
        inst, _ = tiny(seed, tmp_path)
        D = inst.distances()
        y = greedy_design(D, inst.k)
        assert y.sum() == inst.k
        ss = SwapSearch(D, inst.k)
        c1, d1, d2 = ss.state(y)
        delta, i, r = ss.best_swap(y, c1, d1, d2)
        if inst.k == inst.m:
            continue
        z = y.copy()
        z[i], z[r] = True, False
        assert close(D[z].min(axis=0).sum() - d1.sum(), delta, 1e-9)
        # best over all swaps
        best = min(D[np.r_[np.nonzero(y)[0][np.nonzero(y)[0] != rr], ii]].min(axis=0).sum()
                   for ii in np.nonzero(~y)[0] for rr in np.nonzero(y)[0]) - d1.sum()
        assert close(best, delta, 1e-9)


@pytest.mark.parametrize("seed", range(8))
def test_local_search_control(tmp_path, seed):
    inst, path = tiny(seed, tmp_path)
    opt, _ = _brute(path)
    vals = []
    res = KMLocalSearch(time_limit=1.5, on_incumbent=lambda d, stage=None: vals.append(design_value(inst.distances(), d))).run(path)
    assert close(res["best_objective"], opt) and close(min(vals), opt) and res["global_lb"] is None


@pytest.mark.parametrize("seed", range(6))
def test_lagrangian_bound_is_valid(tmp_path, seed):
    inst, path = tiny(seed, tmp_path)
    lp = _lp(path)
    res = KMLagrangian(time_limit=1.5).run(path)
    assert res["global_lb"] <= lp + 1e-7 and res["global_lb"] >= 0.8 * lp - 1e-6
    chk = Checker(inst)
    for s in res["u_snapshots"]:
        b = chk.certify(singles(np.arange(inst.n), s["v"]))["bound"]
        assert b <= lp + 1e-9 and b >= s["L"] - 1e-6 * max(1, abs(s["L"]))


# ---------------------------------------------------------------- Benders

@pytest.mark.parametrize("seed", range(8))
def test_naive_lp_benders_reaches_lp_bound(tmp_path, seed):
    inst, path = tiny(seed, tmp_path)
    lp = _lp(path)
    res = KMNaiveLPBenders(time_limit=30).run(path)
    assert res["termination_reason"] == "lp_converged" and res["global_lb"] <= lp + 1e-7
    assert close(res["global_lb"], lp, 1e-6)
    cert = Checker(inst).certify(aggregated(res["cut_log"]["v"], inst.n))
    assert cert["bound"] <= lp + 1e-9 and close(cert["bound"], lp, 1e-6)


@pytest.mark.parametrize("seed", range(8))
def test_naive_integer_benders_optimal(tmp_path, seed):
    inst, path = tiny(seed, tmp_path)
    opt, _ = _brute(path)
    res = KMNaiveBenders(time_limit=30).run(path)
    assert res["termination_reason"] == "optimal" and close(res["best_objective"], opt)
    assert res["global_lb"] <= opt + 1e-6


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("kw", [{}, {"rc_fixing": False, "stabilize": False}])
def test_engineered_benders_optimal_and_certified(tmp_path, seed, kw):
    inst, path = tiny(seed, tmp_path)
    opt, _ = _brute(path)
    lp = _lp(path)
    designs = []
    res = KMBendersXL(time_limit=30, on_incumbent=lambda d, stage=None: designs.append(d), **kw).run(path)
    assert res["termination_reason"] == "optimal", res["termination_reason"]
    assert close(res["best_objective"], opt) and res["global_lb"] <= opt + 1e-6
    assert res["phase1_converged"] and close(res["lp_value"], lp, 1e-6)
    log = res["cut_log"]
    chk = Checker(inst)
    cert = chk.certify(singles(log["j"], log["v"]))
    assert cert["bound"] <= lp + 1e-9 and close(cert["bound"], lp, 1e-6)
    iv = lp_interval(chk, [("eng", cert)], [("eng point", np.array(res["final_lp_point"]))])
    assert iv["rigorous"] and iv["lo_float"] <= lp + 1e-9 <= iv["hi_float"] + 2e-9 and iv["rel_width"] <= 1e-6


# ---------------------------------------------------------------- certificate checker

def test_certificate_trivial_controls(tmp_path):
    inst, path = tiny(2, tmp_path)
    D = inst.distances()
    chk = Checker(inst)
    lp = _lp(path)
    assert chk.certify(Certificate())["bound"] == 0.0
    b = chk.certify(one_shot(D, _brute(path)[1]))["bound"]
    assert 0.0 <= b <= lp + 1e-9


def test_forged_cuts_are_rejected_or_neutralized(tmp_path):
    inst, path = tiny(0, tmp_path)
    D = inst.distances()
    lp, opt = _lp(path), _brute(path)[0]
    chk = Checker(inst)
    j = 0
    huge = float(D[:, j].max() * 10 + 100)
    # (a) claims theta_j >= huge with w = 0 (dual infeasible: v - w_i > D_ij for every i)
    forged = Certificate(explicit=[(np.array([j]), np.array([huge]), [(np.zeros(0, dtype=np.int64), np.zeros(0))])])
    res = chk.certify(forged, reject_tol=1e-6)
    assert res["rejected"] and res["bound"] <= lp + 1e-9
    # (b) the same cut accepted without the tolerance: the Neumaier-Shcherbina correction makes it harmless
    res = chk.certify(forged, reject_tol=None)
    assert not res["rejected"] and res["bound"] <= lp + 1e-9
    # (c) a valid explicit dual (cheapest completion) is accepted and certified
    v = float(np.sort(D[:, j])[2])
    ix = np.nonzero(D[:, j] < v)[0]
    good = Certificate(explicit=[(np.array([j]), np.array([v]), [(ix, v - D[ix, j])])])
    res = chk.certify(good)
    assert not res["rejected"] and res["bound"] <= lp + 1e-9
    # (d) the valid cut with one coefficient lowered (too strong) is rejected
    if len(ix):
        bad = Certificate(explicit=[(np.array([j]), np.array([v]), [(ix, np.maximum(v - D[ix, j] - 1.0, 0.0))])])
        assert chk.certify(bad, reject_tol=1e-6)["rejected"]
        assert chk.certify(bad)["bound"] <= lp + 1e-9  # default: Neumaier-Shcherbina correction, as the canonical checker
    # (e) a forged full certificate that would prove more than the optimum if trusted
    fake = Certificate(explicit=[(np.arange(inst.n), np.full(inst.n, huge), [(np.zeros(0, dtype=np.int64), np.zeros(0))] * inst.n)])
    assert chk.certify(fake, reject_tol=None)["bound"] <= opt + 1e-9


def test_certificate_json_structure_checks(tmp_path):
    inst, _ = tiny(1, tmp_path)
    m, n = inst.m, inst.n
    obj = {"format": "km-cert-v1", "cuts": [
        {"ids": [0, 0], "v": [1.0, 1.0]},                      # repeated ID
        {"ids": [n], "v": [1.0]},                              # out of range
        {"ids": [1], "v": [float("nan")]},                     # non-finite
        {"ids": [1], "v": [1.0], "w": [[[m, 1.0]]]},           # facility out of range
        {"ids": [1], "v": [1.0], "w": [[[0, 1.0], [0, 2.0]]]},  # repeated facility
        {"ids": [2], "v": [3.0]},                              # fine
        {"ids": [0, 1], "v": [2.0, 2.0]}]}                     # fine (aggregated)
    cert, rejected = parse_certificate(json.loads(json.dumps(obj)), m, n)
    assert len(rejected) == 5 and len(cert.single_j) == 1 and len(cert.multi) == 1
    chk = Checker(inst)
    a = chk.certify(cert)["bound"]
    again, rej2 = parse_certificate(json.loads(json.dumps(to_json(cert))), m, n)
    assert not rej2 and chk.certify(again)["bound"] == a


def test_coefficient_rounding_never_trusted(tmp_path):
    """Non-integral data and duals (inexact float coefficients), and an LP model whose y-coefficients were
    deliberately weakened (stronger cuts in Gurobi's model): the exact re-certification from the original data
    stays a valid bound."""
    inst, path = tiny(5, tmp_path, integer=False)
    lp = _lp(path)
    chk = Checker(inst)
    res = KMLagrangian(time_limit=1.0).run(path)
    u = np.array(res["u_snapshots"][-1]["v"]) + 0.1  # non-representable shifts
    cert = singles(np.arange(inst.n), u)
    honest = chk.certify(cert)
    skewed = chk.certify(cert, lp_coef_scale=0.5)
    assert honest["bound"] <= lp + 1e-9 and skewed["bound"] <= lp + 1e-9
    assert skewed["lp_value"] is None or skewed["lp_value"] >= skewed["bound"] - 1e-9
    # the skewed model is stronger than the valid one: its LP value may exceed LP*, the certified bound never does
    assert honest["lp_value"] <= lp + 1e-6 * max(1.0, lp)


def test_theta_box_contains_fractional_subproblem_values(tmp_path):
    for seed in range(4):
        inst, _ = tiny(seed, tmp_path)
        D = inst.distances()
        chk = Checker(inst)
        rng = np.random.default_rng(seed)
        for _ in range(5):
            y = rng.random(inst.m)
            if y.sum() < 1:  # Q_j is finite iff sum y >= 1 (sum y = k >= 1 in the master)
                y = y / y.sum()
            for j in range(inst.n):
                q = _sub_lp(D, j, y)
                assert 0.0 - 1e-9 <= q <= chk.U_float[j] + 1e-9


def test_lp_interval_provenance_and_untrusted_lower(tmp_path):
    inst, path = tiny(3, tmp_path)
    chk = Checker(inst)
    lp = _lp(path)
    res = KMBendersXL(time_limit=30, branch=False).run(path)
    good = chk.certify(singles(res["cut_log"]["j"], res["cut_log"]["v"]))
    forged = {"cert_ok": False, "lb": str(Fraction(10**9))}          # a failed check never sets the lower end
    iv = lp_interval(chk, [("forged", forged), ("eng", good)], [("eng point", np.array(res["final_lp_point"]))])
    assert iv["lo_provenance"] == "eng" and iv["hi_provenance"] == "eng point" and iv["rigorous"]
    assert Fraction(iv["lo"]) <= Fraction(lp) + Fraction(1, 10**6) and Fraction(iv["hi"]) >= Fraction(iv["lo"])
    assert lp_interval(chk, [("forged", forged)], [])["lo"] is None


def test_lp_interval_repair_is_exact(tmp_path):
    inst, path = tiny(3, tmp_path)
    chk = Checker(inst)
    rng = np.random.default_rng(0)
    y = np.clip(rng.random(inst.m) * inst.k / inst.m * 1.3, 0, 1)
    Y = repair_point(y, inst.k)
    assert sum(Y, Fraction(0)) == inst.k and all(0 <= x <= 1 for x in Y)
    assert float(exact_master_value(chk, Y)) >= _lp(path) - 1e-9


# ---------------------------------------------------------------- verifier

def test_verifier(tmp_path):
    inst, path = tiny(2, tmp_path)
    opt, S = _brute(path)
    out = km_output(inst, S)
    p = tmp_path / "out.json"
    p.write_text(json.dumps(out))
    res = verify_solution(path, p)
    assert res["ok"] and close(res["objective_value"], opt) and close(res["assign_cost"], opt)
    for mutate in (lambda o: o.update(objective_value=o["objective_value"] + 1),
                   lambda o: o.update(y=[1] * inst.m),
                   lambda o: o.update(y=[0.5] * inst.m),
                   lambda o: o["assign"].__setitem__(0, [i for i in range(inst.m) if o["y"][i] == 0][0]),
                   lambda o: o.update(status="optimal?")):
        bad = json.loads(json.dumps(out))
        mutate(bad)
        p.write_text(json.dumps(bad))
        assert not verify_solution(path, p)["ok"]


# ---------------------------------------------------------------- review 46: hardened checker, controls, runner

def test_bounded_parsing_overflow_and_failure_semantics(tmp_path):
    inst, path = tiny(0, tmp_path)
    chk = Checker(inst)
    # v = 10**400 used to raise OverflowError: now the cut is rejected (magnitude), the certificate still checks
    cert, rej = parse_certificate({"format": "km-cert-v1", "cuts": [{"ids": [0], "v": [10**400]}, {"ids": [1], "v": [2.0]}]},
                                  inst.m, inst.n)
    assert len(rej) == 1 and "magnitude" in rej[0] and len(cert.single_j) == 1
    bad = (b'{"format":"km-cert-v1","view":"km-v1","cuts":[{"ids":[0],"v":[1e500]},{"ids":[1],"v":[NaN]},'
           b'{"ids":[2],"v":["' + b"1" * 80 + b'"]},{"ids":[3],"v":[3]}]}')
    r = chk.certify(bad)
    assert r["cert_ok"] and len(r["rejected"]) == 3
    for data, reason in ((b"{not json", "parse_error"), (b'{"format":"km-cert-v1","view":"ufl-v1","cuts":[]}', "view_mismatch"),
                         (b'{"format":"other","cuts":[]}', "format_mismatch")):
        r = chk.certify(data)
        assert not r["cert_ok"] and r["reason"] == reason and r["bound"] == 0.0
    r = chk.certify(to_bytes(one_shot(inst.distances(), [0])), limits=Limits(max_bytes=10))
    assert not r["cert_ok"] and r["reason"] == "too_many_bytes" and r["bound"] == 0.0
    r = chk.certify(one_shot(inst.distances(), [0]), deadline=0.0)          # deadline already passed
    assert not r["cert_ok"] and r["reason"] == "time_limit" and r["bound"] == 0.0
    assert chk.certify(None)["reason"] == "absent"


def test_size_limit_counts_the_expanded_form(tmp_path):
    inst, path = tiny(2, tmp_path)
    chk = Checker(inst)
    D = inst.distances()
    v = np.sort(D, axis=0)[-1] + 1.0                       # every facility is closer than v: expanded size m + 1 per cut
    cert = singles(np.arange(inst.n), v)
    size = chk.expanded_size(cert)
    assert size == inst.n * (inst.m + 1)
    assert chk.certify(cert, limits=Limits(max_nonzeros=size))["cert_ok"]
    r = chk.certify(cert, limits=Limits(max_nonzeros=size - 1))
    assert not r["cert_ok"] and r["reason"] == "too_many_nonzeros"


def _xc_or_skip():
    try:
        import src.experiments.xl_certificate as xc
        from src.bd.km.h_view_xl import cross_check, km_view
        assert xc.check_certificate is not None
        return cross_check, km_view
    except Exception as exc:  # noqa: BLE001 - maintained by another session
        pytest.skip(f"canonical checker unavailable: {exc}")


@pytest.mark.parametrize("seed", range(4))
def test_family_backend_agrees_with_canonical_checker(tmp_path, seed):
    cross_check, km_view = _xc_or_skip()
    inst, path = tiny(seed, tmp_path)
    chk = Checker(inst)
    view = km_view(inst)
    lp = _lp(path)
    eng = KMBendersXL(time_limit=30, branch=False).run(path)
    lag = KMLagrangian(time_limit=1.0).run(path)
    nlp = KMNaiveLPBenders(time_limit=30).run(path)
    D = inst.distances()
    j = 0
    v = float(np.sort(D[:, j])[2])
    ix = np.nonzero(D[:, j] < v)[0]
    certs = [singles(eng["cut_log"]["j"], eng["cut_log"]["v"]), chk.dual_vector_certificate(lag["u_snapshots"][-1]["v"]),
             aggregated(nlp["cut_log"]["v"], inst.n), one_shot(D, [0]), Certificate(),
             Certificate(explicit=[(np.array([j]), np.array([v]), [(ix, v - D[ix, j] - 0.5)])])]
    for cert in certs:
        res = cross_check(view, chk, cert)
        assert res["agree"], res
        assert res["canonical"]["cert_ok"] and Fraction(res["canonical"]["lb"]) <= Fraction(lp) + Fraction(1, 10**7)


def test_km_view_lp_interval_and_boxes(tmp_path):
    _, km_view = _xc_or_skip()
    import src.experiments.xl_certificate as xc
    inst, path = tiny(1, tmp_path)
    view = km_view(inst)
    lp = _lp(path)
    iv = xc.lp_interval(view, time_limit=30)
    assert iv["rigorous"] and iv["lp_lo_float"] <= lp + 1e-7 and iv["lp_hi_float"] >= lp - 1e-7
    from src.bd.km.f_heuristic_xl import capped_simplex
    ys = [np.full(inst.m, inst.k / inst.m), capped_simplex(np.random.default_rng(0).random(inst.m) + 0.1, inst.k)]
    box = xc.fractional_box_check(view, ys=ys, subproblems=list(range(min(3, inst.n))))
    assert box["ok"] and box["checked"] == 2 * min(3, inst.n)


def test_radius_lp_control_and_incomplete_barrier_multipliers(tmp_path):
    for seed in range(4):
        inst, path = tiny(seed, tmp_path)
        lp = _lp(path)
        chk = Checker(inst)
        for kw in ({"formulation": "radius", "method": 1}, {"formulation": "radius", "method": 2},
                   {"formulation": "classical", "method": 2, "params": {"BarIterLimit": 1}},
                   {"formulation": "radius", "method": 2, "params": {"BarIterLimit": 1}}):
            got = []
            res = KMCompactLP(time_limit=30, publish_margin=0.0, on_certificate=got.append, **kw).run(path)
            if "params" not in kw:
                assert res["termination_reason"] == "optimal" and close(res["lp_value"], lp)
            assert got, (kw, res["termination_reason"])
            r = chk.certify(got[-1])
            assert r["cert_ok"] and r["bound"] <= lp + 1e-9
            if "params" not in kw:
                assert close(r["bound"], lp, 1e-6)


def test_analytical_pool_and_oneshot_controls(tmp_path):
    from src.bd.km.f_heuristic_xl import KMNoFeedbackPool, KMStaticCertificate
    for seed in range(4):
        inst, path = tiny(seed, tmp_path)
        lp = _lp(path)
        chk = Checker(inst)
        for kind in ("empty", "analytical", "oneshot_greedy"):
            got = []
            KMStaticCertificate(kind=kind, time_limit=10, cert_limit=60, on_certificate=got.append).run(path)
            assert len(got) == 1 and chk.expanded_size(got[0]) <= 60
            r = chk.certify(got[0])
            assert r["cert_ok"] and r["bound"] <= lp + 1e-9
        runs = []
        for _ in range(2):
            got = []
            KMNoFeedbackPool(time_limit=5, cert_limit=80, on_certificate=got.append).run(path)
            assert got and chk.expanded_size(got[-1]) <= 80
            runs.append(got[-1])
            assert chk.certify(got[-1])["bound"] <= lp + 1e-9
        assert np.array_equal(runs[0].single_j, runs[1].single_j) and np.array_equal(runs[0].single_v, runs[1].single_v)


def test_lagrangian_cumulative_pool(tmp_path):
    for seed in range(3):
        inst, path = tiny(seed, tmp_path)
        lp = _lp(path)
        chk = Checker(inst)
        got = []
        res = KMLagrangian(time_limit=1.5, on_certificate=got.append).run(path)
        assert got
        pool = chk.certify(got[-1])["bound"]
        included = res["u_snapshots"][:len(got)]   # one publication per in-loop snapshot
        last = chk.certify(chk.dual_vector_certificate(included[-1]["v"]))["bound"]
        assert pool <= lp + 1e-9 and pool >= last - 1e-6 * max(1.0, abs(last))


def test_published_certificates_of_benders_arms(tmp_path):
    for seed in range(4):
        inst, path = tiny(seed, tmp_path)
        lp = _lp(path)
        chk = Checker(inst)
        for cls in (KMBendersXL, KMNaiveLPBenders):
            got = []
            cls(time_limit=30, on_certificate=got.append, cert_interval=0.0).run(path)
            assert got and [c.version for c in got] == sorted(c.version for c in got)
            r = chk.certify(got[-1])
            assert r["cert_ok"] and r["bound"] <= lp + 1e-9 and close(r["bound"], lp, 1e-6)


def test_unit_gap_closure_rule():
    from src.bd.km.b_mip_xl import gap_closed
    assert gap_closed(1_000_000.0, 999_999.5, True) and not gap_closed(1_000_000.0, 999_999.0, True)
    assert not gap_closed(3_000_000.0, 2_999_997.5, True)          # relative 1e-6 would have stopped here
    assert gap_closed(10.0, 10.0 - 5e-6, False) and not gap_closed(10.0, 9.9, False)


def _runner():
    import importlib.util
    spec = importlib.util.spec_from_file_location("km_stage1", Path(__file__).resolve().parents[1] / "scripts" / "phoenix" / "scaleup_km_stage1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_supervisor_hard_kills_and_persists(tmp_path):
    mod = _runner()
    inst, path = tiny(1, tmp_path)
    t = time.monotonic()
    records, run = mod.supervise(path, "_test_sleeper", 8.0, tmp_path / "sleeper")
    assert run["killed_at_T"] and time.monotonic() - t < 30
    kinds = [r["kind"] for r in records]
    assert "incumbent" in kinds and "certificate" in kinds and all(r["t"] < 8.0 for r in records)
    lines = (tmp_path / "sleeper" / "events.jsonl").read_text().splitlines()
    assert len(lines) == len(records)
    cert = next(r for r in records if r["kind"] == "certificate")
    assert (tmp_path / "sleeper" / "certs" / cert["file"]).exists()


def test_runner_task_end_to_end(tmp_path):
    _xc_or_skip()
    mod = _runner()
    inst, path = tiny(2, tmp_path)
    lp = _lp(path)
    row = mod.run_task(path, "eng", 20.0, tmp_path / "out", dense=True)
    cps = row["certificates"]["checkpoints"]
    assert row["incumbents"]["verified"] and cps and cps[-1]["cert_ok"] and row["certificates"]["family_agrees"]
    assert float(Fraction(cps[-1]["lb"])) <= lp + 1e-9 and close(float(Fraction(cps[-1]["lb"])), lp, 1e-6)


def test_frozen_acceptance_rules():
    mod = _runner()
    rules = {"eps_b": 1e-5, "eps_p": 1e-3, "naive_quality_gap": 0.005, "naive_speed_ratio": 3.0, "saved_fraction": 0.25,
             "control_speed_ratio": 2.0, "control_saved_seconds": 225.0}

    def arm(t_bound=None, target=False, bgap=0.1, gap=0.01):
        return {"t_bound": t_bound, "bound_target_T": target, "bgap_LP_T": bgap, "gap_best": gap, "near_best": gap <= 1e-3}

    base = {"eng": arm(60, True, 0.0, 0.0), "naive_lp": arm(None, False, 0.05, 0.05), "ils": arm(None, False, 0.3, 0.005),
            "naive_int": arm(None, False, 0.5, 0.05)}
    for c in mod.BOUND_CONTROLS:
        base.setdefault(c, arm(None, False, 0.3))
    assert mod.accept(base, "final_target", rules, 900.0)["pass"]
    assert mod.accept(base, "time_to_target", rules, 900.0)["pass"]
    slow = dict(base, lp_classical_bar=arm(600, True, 0.0))          # control reaches the target 10x later, 540 s saved
    assert mod.accept(slow, "time_to_target", rules, 900.0)["pass"]
    assert not mod.accept(slow, "final_target", rules, 900.0)["pass"]  # final-quality headline: any control success fails
    fast = dict(base, analytical=arm(60, True, 0.0))
    assert not mod.accept(fast, "time_to_target", rules, 900.0)["pass"]
    close_naive = dict(base, naive_lp=arm(120, True, 0.0))             # naive reaches the target only 2x later
    assert not mod.accept(close_naive, "final_target", rules, 900.0)["pass"]
    ils_good = dict(base, ils=arm(None, False, 0.3, 0.0005))           # primal control near_best
    assert not mod.accept(ils_good, "final_target", rules, 900.0)["pass"]
    assert not mod.accept({"eng": base["eng"]}, "final_target", rules, 900.0)["pass"]
