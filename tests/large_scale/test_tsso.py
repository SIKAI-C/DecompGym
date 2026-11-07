"""Correctness checks for the scale-up TSSO family (generator, recourse LP, extensive form, LP controls, naive and
engineered Benders, heuristic controls, verifier, certificate checker) against the paper's code, an independent
dynamic program and brute force on small generated instances."""
from __future__ import annotations

import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

gp = pytest.importorskip("gurobipy")
from gurobipy import GRB  # noqa: E402

from src.bd.tsso.a_data_xl import (VIEW, TSSOXLInstance, generate, initial_design, load_instance,  # noqa: E402
                                   write_instance)
from src.bd.tsso.b_mip_xl import (DesignEvaluator, ProductRecourse, TSSOExtensiveMIP, TSSOLPControl,  # noqa: E402
                                  build_extensive_form, cut_coefficients, make_env, tsso_output)
from src.bd.tsso.c_naive_bd import TSSONaiveBenders, TSSONaiveLPBenders  # noqa: E402
from src.bd.tsso.d_ref_bd_xl import TSSOBendersXL  # noqa: E402
from src.bd.tsso.e_verifier_xl import read_instance, recourse, verify_solution  # noqa: E402
from src.bd.tsso.f_heuristic_xl import TSSOHeuristicControl  # noqa: E402
from src.bd.tsso import g_certificate_xl as gc  # noqa: E402

SHAPES = [(2, 3, 2), (3, 4, 3), (2, 5, 4), (3, 3, 5), (2, 4, 3)]


def tiny(seed: int, tmp_path: Path) -> tuple[TSSOXLInstance, str]:
    P, T, S = SHAPES[seed % len(SHAPES)]
    inst = generate(P, T, S, seed)
    path = tmp_path / f"{inst.name}.json"
    write_instance(inst, path)
    return load_instance(path), str(path)


def brute_force(inst: TSSOXLInstance) -> tuple[float, np.ndarray]:
    """Optimum by enumeration, product by product (the value separates by product)."""
    ev = DesignEvaluator(inst)
    total, best = 0.0, np.zeros((inst.I, inst.T), dtype=bool)
    for i in range(inst.I):
        vals = []
        for bits in itertools.product([False, True], repeat=inst.T):
            z = np.array(bits)
            vals.append((ev.product(i, z)["value"], z))
        v, z = min(vals, key=lambda a: a[0])
        total += v
        best[i] = z
    ev.close()
    return total, best


def paper_lp(inst: TSSOXLInstance, i: int, s: int, y: np.ndarray, bounds: str | None = None) -> float:
    """The paper's recourse LP for one (product, scenario), explicit capacity rows (a_gold_bd style).
    bounds: None, 'total' (all variables <= D) or 'remaining' (production <= remaining demand, NOT implied)."""
    T = inst.T
    md = gp.Model(env=ENV)
    d = inst.demand[i, :, s]
    D = float(d.sum())
    ub = D if bounds == "total" else GRB.INFINITY
    xr = md.addVars(T, lb=0.0, ub=ub)
    xo = md.addVars(T, lb=0.0, ub=ub)
    inv = md.addVars(T, lb=0.0, ub=ub)
    back = md.addVars(T, lb=0.0, ub=ub)
    for t in range(T):
        prev = (inv[t - 1] - back[t - 1]) if t else 0.0
        md.addConstr(xr[t] + xo[t] - inv[t] + back[t] + prev == d[t])
        md.addConstr(xr[t] <= inst.K[i, t] * y[t])
        if bounds == "remaining":
            md.addConstr(xr[t] + xo[t] <= float(d[t:].sum()))
    md.setObjective(gp.quicksum(inst.c_reg[i, t] * xr[t] + inst.c_ot[i, t] * xo[t] + inst.h[i, t] * inv[t] + inst.b[i, t] * back[t]
                                for t in range(T)))
    md.Params.FeasibilityTol = 1e-9
    md.Params.OptimalityTol = 1e-9
    md.optimize()
    assert md.Status == GRB.OPTIMAL
    v = md.ObjVal
    md.dispose()
    return v


ENV = None


@pytest.fixture(scope="module", autouse=True)
def _env():
    global ENV
    ENV = make_env()
    yield
    ENV.dispose()


def lp_value(path: str) -> float:
    r = TSSOLPControl(time_limit=60, method=1).run(path)
    assert r["termination_reason"] == "optimal"
    return r["global_lb"]


# ---------------------------------------------------------------- data

def test_generator_is_the_paper_recipe(tmp_path):
    from src.bd.tsso.a_gen_data import TSSOGenerator, TSSOInstance
    inst = generate(4, 6, 7, 3)
    raw = TSSOGenerator(seed=3).generate_instance(n_products=4, n_periods=6, n_scenarios=7, seed=3)
    for k in ("F", "K", "c_reg", "c_ot", "h", "b", "demand", "prob", "I0", "B0"):
        assert np.array_equal(getattr(inst, k), np.asarray(getattr(raw, k), float)), k
    path = write_instance(inst, tmp_path / "x.json")
    back = load_instance(path)
    for k in ("F", "K", "c_reg", "c_ot", "h", "b", "demand", "prob"):
        assert np.array_equal(getattr(back, k), getattr(inst, k)), k
    paper = TSSOInstance.from_json_file(path)  # the paper's loader reads the XL file
    assert np.array_equal(np.asarray(paper.demand), inst.demand)
    assert VIEW["id"] == "tsso-xl-compact-v1"


# ---------------------------------------------------------------- recourse

@pytest.mark.parametrize("seed", range(4))
def test_recourse_matches_paper_lp_and_dp(seed, tmp_path):
    inst, path = tiny(seed, tmp_path)
    vinst = read_instance(path)
    rng = np.random.default_rng(seed)
    for _ in range(4):
        y = rng.random((inst.I, inst.T))
        if _ % 2 == 0:
            y = (y > 0.5).astype(float)
        for i in range(inst.I):
            lp = ProductRecourse(inst, i, ENV)
            res = lp.solve(y[i])
            lp.dispose()
            for s in range(inst.S):
                ref = paper_lp(inst, i, s, y[i])
                dp = recourse(vinst["c_reg"][i], vinst["c_ot"][i], vinst["h"][i], vinst["b"][i],
                              [vinst["K"][i][t] * y[i, t] for t in range(inst.T)], [vinst["d"][i][t][s] for t in range(inst.T)])
                assert res["q"][s] == pytest.approx(ref, rel=1e-8, abs=1e-7)
                assert dp == pytest.approx(ref, rel=1e-9, abs=1e-8)


def test_backlog_clearing_case_total_bound_implied_remaining_bound_not():
    """Production may clear earlier backlog: 'production <= remaining demand' cuts off the optimum, the total-horizon
    bound D does not (also at fractional y)."""
    inst = TSSOXLInstance(F=np.array([[1.0, 1.0]]), K=np.array([[0.0, 100.0]]), c_reg=np.array([[1.0, 1.0]]),
                          c_ot=np.array([[10.0, 10.0]]), h=np.array([[0.5, 0.5]]), b=np.array([[0.1, 5.0]]),
                          demand=np.array([[[10.0], [0.0]]]), prob=np.array([1.0]), I0=np.zeros(1), B0=np.zeros(1))
    inst.validate()
    for y in (np.array([1.0, 1.0]), np.array([0.3, 0.7])):
        free = paper_lp(inst, 0, 0, y)
        assert paper_lp(inst, 0, 0, y, "total") == pytest.approx(free, abs=1e-9)
        assert paper_lp(inst, 0, 0, y, "remaining") > free + 1.0
    assert paper_lp(inst, 0, 0, np.array([1.0, 1.0])) == pytest.approx(11.0)  # backlog 10 (1.0) + regular 10 (10.0)


@pytest.mark.parametrize("seed", range(3))
def test_boxes_preserve_recourse_at_fractional_y(seed, tmp_path):
    """Variable boxes [0, D] leave Q unchanged and theta boxes [L, U] contain Q on the whole fractional box."""
    inst, _ = tiny(seed, tmp_path)
    rng = np.random.default_rng(100 + seed)
    U = inst.total_demand * (inst.c_reg + inst.c_ot + inst.h + inst.b).sum(axis=1)[:, None]
    for _ in range(5):
        y = rng.random((inst.I, inst.T)) * rng.choice([0.0, 0.3, 1.0], size=(inst.I, inst.T))
        for i in range(inst.I):
            for s in range(inst.S):
                q = paper_lp(inst, i, s, y[i])
                assert paper_lp(inst, i, s, y[i], "total") == pytest.approx(q, rel=1e-9, abs=1e-8)
                assert 0.0 <= q <= U[i, s] + 1e-9


@pytest.mark.parametrize("seed", range(3))
def test_cuts_tight_at_point_and_valid_elsewhere(seed, tmp_path):
    inst, _ = tiny(seed, tmp_path)
    rng = np.random.default_rng(seed)
    for i in range(inst.I):
        lp = ProductRecourse(inst, i, ENV)
        y0 = rng.random(inst.T)
        res = lp.solve(y0)
        const, coef = cut_coefficients(inst, i, np.arange(inst.S), res["pi"], res["mu"])
        assert np.allclose(const - coef @ y0, res["q"], rtol=1e-8, atol=1e-7)
        for _ in range(6):
            y = rng.random(inst.T) if _ % 2 else (rng.random(inst.T) > 0.5).astype(float)
            q = lp.solve(y)["q"]
            assert np.all(const - coef @ y <= q + 1e-7 * np.maximum(1.0, np.abs(q)))
        lp.dispose()


# ---------------------------------------------------------------- extensive form and controls

@pytest.mark.parametrize("seed", range(3))
def test_extensive_mip_equals_paper_baseline_and_brute_force(seed, tmp_path):
    from src.bd.tsso.a_gold_monolithic import TSSOBaseline
    inst, path = tiny(seed, tmp_path)
    opt, _ = brute_force(inst)
    pubs = []
    r = TSSOExtensiveMIP(time_limit=60, on_incumbent=lambda y, stage=None: pubs.append(y)).run(path)
    assert r["termination_reason"] == "optimal"
    assert r["best_objective"] == pytest.approx(opt, rel=1e-6)
    ev = DesignEvaluator(inst)
    assert ev.value(np.asarray(r["design"])) == pytest.approx(opt, rel=1e-6)
    ev.close()
    paper = TSSOBaseline().run(path)
    assert paper["best_objective"] == pytest.approx(opt, rel=1e-6)
    assert r["global_lb"] <= opt + 1e-6 * opt


@pytest.mark.parametrize("seed", range(2))
def test_lp_controls_agree_and_bound_the_optimum(seed, tmp_path):
    inst, path = tiny(seed, tmp_path)
    opt, _ = brute_force(inst)
    a = TSSOLPControl(time_limit=60, method=1).run(path)
    b = TSSOLPControl(time_limit=60, method=2).run(path)
    split = TSSOLPControl(time_limit=60, method=2, split=True)
    c = split.run(path)
    assert a["termination_reason"] == b["termination_reason"] == c["termination_reason"] == "optimal"
    assert a["global_lb"] == pytest.approx(b["global_lb"], rel=1e-6)
    assert c["global_lb"] == pytest.approx(a["global_lb"], rel=1e-6)
    cert = gc.certify(inst, split.cutlog.arrays())
    assert a["global_lb"] - 1e-5 * abs(a["global_lb"]) <= cert["lb"] <= a["global_lb"] + 1e-9 * abs(a["global_lb"])
    assert a["global_lb"] <= opt + 1e-6
    md, info = build_extensive_form(inst, ENV, relax=True)
    md.optimize()
    assert md.ObjVal == pytest.approx(a["global_lb"], rel=1e-8)
    md.dispose()


# ---------------------------------------------------------------- Benders

@pytest.mark.parametrize("seed", range(3))
def test_engineered_reaches_lp_and_optimum(seed, tmp_path):
    inst, path = tiny(seed, tmp_path)
    opt, _ = brute_force(inst)
    lp = lp_value(path)
    for stab in (True, False):
        pubs, bounds = [], []
        eng = TSSOBendersXL(time_limit=60, stabilize=stab, on_incumbent=lambda y, stage=None: pubs.append(y.copy()),
                            on_bound=bounds.append)
        r = eng.run(path)
        assert r["lp_bound"] == pytest.approx(lp, rel=1e-6)
        assert r["lp_seconds"] is not None
        assert max(bounds) <= lp + 1e-6 * abs(lp)
        assert r["best_objective"] == pytest.approx(opt, rel=1e-6)
        assert r["termination_reason"] == "optimal"
        assert r["int_bound"] <= opt + 1e-6 * opt
        ev = DesignEvaluator(inst)
        assert ev.value(pubs[-1]) == pytest.approx(r["best_objective"], rel=1e-9)
        ev.close()
        assert eng.cutlog.n_cuts > 0


@pytest.mark.parametrize("seed", range(3))
def test_naive_controls(seed, tmp_path):
    inst, path = tiny(seed, tmp_path)
    opt, _ = brute_force(inst)
    lp = lp_value(path)
    r = TSSONaiveLPBenders(time_limit=60).run(path)
    assert r["termination_reason"] == "lp_optimal"
    assert r["global_lb"] == pytest.approx(lp, rel=1e-6)
    assert r["best_objective"] >= opt - 1e-6
    r = TSSONaiveBenders(time_limit=60).run(path)
    assert r["termination_reason"] == "optimal"
    assert r["best_objective"] == pytest.approx(opt, rel=1e-6)
    assert r["global_lb"] <= opt + 1e-6 * opt


@pytest.mark.parametrize("mode", ["ls", "saa"])
def test_heuristic_control(mode, tmp_path):
    inst, path = tiny(1, tmp_path)
    opt, _ = brute_force(inst)
    pubs = []
    r = TSSOHeuristicControl(time_limit=5, mode=mode, on_incumbent=lambda y, stage=None: pubs.append(y.copy())).run(path)
    assert r["best_objective"] >= opt - 1e-6
    assert r["best_objective"] <= opt * 1.02
    ev = DesignEvaluator(inst)
    assert ev.value(pubs[-1]) == pytest.approx(r["best_objective"], rel=1e-9)
    ev.close()


# ---------------------------------------------------------------- verifier

def test_verifier(tmp_path):
    inst, path = tiny(2, tmp_path)
    _, y = brute_force(inst)
    out = tsso_output(inst, y)
    p = tmp_path / "out.json"
    p.write_text(json.dumps(out))
    v = verify_solution(path, p)
    assert v["ok"], v
    assert v["objective_value"] == pytest.approx(out["objective_value"], rel=1e-9)
    for bad in ({**out, "objective_value": out["objective_value"] * 0.99}, {**out, "status": "error"},
                {**out, "y": out["y"][:-1]}, {**out, "y": [[0.5] * inst.T] * inst.I}, {**out, "method": ""}):
        p.write_text(json.dumps(bad))
        assert not verify_solution(path, p)["ok"]


# ---------------------------------------------------------------- certificate checker

def eng_cuts(path: str) -> tuple[dict, dict]:
    eng = TSSOBendersXL(time_limit=60, integer=False, local_search=False)
    r = eng.run(path)
    return eng.cutlog.arrays(), r


@pytest.mark.parametrize("seed", range(3))
def test_certify_engineered_cuts_reproduces_lp_star(seed, tmp_path):
    inst, path = tiny(seed, tmp_path)
    lp = lp_value(path)
    cuts, r = eng_cuts(path)
    c = gc.certify(inst, cuts)
    assert c["rejected_cuts"] == 0
    assert c["lb"] <= lp + 1e-9 * abs(lp)
    assert c["lb"] >= lp - 1e-6 * abs(lp)
    assert Fraction(c["lb"]) <= c["lb_exact"]
    f = gc.certify(inst, cuts, exact="fraction")
    assert f["lb_exact"] == c["lb_exact"]  # integer and Fraction arithmetic agree exactly
    up = gc.lp_upper(inst, np.asarray(r["y_lp"]))
    assert up["ub_exact"] >= c["lb_exact"]
    assert (up["ub"] - c["lb"]) <= 1e-6 * abs(lp)


@pytest.mark.parametrize("seed", range(2))
def test_certify_aggregated_and_lp_control_cuts(seed, tmp_path):
    inst, path = tiny(seed, tmp_path)
    lp = lp_value(path)
    naive = TSSONaiveLPBenders(time_limit=60)
    r = naive.run(path)
    c = gc.certify(inst, naive.cutlog.arrays())
    assert c["lb"] <= lp + 1e-9 * abs(lp) and c["lb"] >= r["global_lb"] - 1e-6 * abs(lp)
    assert gc.certify(inst, naive.cutlog.arrays(), exact="fraction")["lb_exact"] == c["lb_exact"]
    for method in (1, 2):
        ctl = TSSOLPControl(time_limit=60, method=method)
        ctl.run(path)
        c = gc.certify(inst, ctl.cutlog.arrays())
        assert lp - 1e-5 * abs(lp) <= c["lb"] <= lp + 1e-9 * abs(lp)


def test_certificate_controls_and_time_filter(tmp_path):
    inst, path = tiny(0, tmp_path)
    lp = lp_value(path)
    assert gc.certify(inst, gc.empty_cuts(inst.T))["lb"] == 0.0
    cuts, _ = eng_cuts(path)
    early = gc.cuts_upto(cuts, 1e-12)
    assert len(early["time"]) == 0
    first = gc.cuts_upto(cuts, float(np.min(cuts["time"])) + 1e-9)
    c1 = gc.certify(inst, first)
    assert 0.0 < c1["lb"] <= gc.certify(inst, cuts)["lb"] <= lp + 1e-9 * abs(lp)


@pytest.mark.parametrize("seed", range(3))
def test_forged_and_malformed_certificates(seed, tmp_path):
    """Infeasible (forged) duals only weaken cuts; malformed cuts are rejected; a failing master degrades to the
    trivial bound, never to +inf."""
    inst, path = tiny(seed, tmp_path)
    lp = lp_value(path)
    opt, _ = brute_force(inst)
    cuts, _ = eng_cuts(path)
    rng = np.random.default_rng(seed)
    for scale, noise in ((10.0, 0.0), (1.0, 5.0), (-3.0, 0.0), (1e6, 1.0)):
        forged = dict(cuts, pi=cuts["pi"] * scale + noise * rng.standard_normal(cuts["pi"].shape),
                      mu=cuts["mu"] * scale + noise * rng.standard_normal(cuts["mu"].shape))
        c = gc.certify(inst, forged)
        assert c["lb"] <= lp + 1e-9 * abs(lp), (scale, noise)
        assert c["lb"] <= opt
        assert math.isfinite(c["lb"])
    # a cut whose naive float evaluation (pi.d - mu K y with the claimed duals) would exceed Q: forged pi = 50
    n = inst.S
    sub = np.arange(n)
    forged = {"sub": sub, "cut": np.zeros(n, dtype=np.int64), "pi": np.full((n, inst.T), 50.0), "mu": np.zeros((n, inst.T)),
              "time": np.zeros(1)}
    c = gc.certify(inst, forged)
    assert c["lb"] <= lp and c["max_dual_infeasibility"] > 1.0
    # malformed: NaN, duplicate ID within a cut, ID out of range -> rejected
    bad = {"sub": np.array([0, 0, inst.I * inst.S, 1]), "cut": np.array([0, 0, 1, 2]), "pi": np.ones((4, inst.T)),
           "mu": np.zeros((4, inst.T)), "time": np.zeros(3)}
    bad["pi"][3, 0] = np.nan
    c = gc.certify(inst, bad)
    assert c["rejected_cuts"] == 3 and c["cuts_used"] == 0 and c["lb"] == 0.0
    # master failure -> trivial bound
    orig = gc._solve_master
    try:
        gc._solve_master = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        c = gc.certify(inst, cuts)
        assert c["lb"] == 0.0 and "error" in c
    finally:
        gc._solve_master = orig


def test_coefficient_rounding_is_exact(tmp_path):
    """Data and duals that are not short decimals: the exact certified bound equals the Fraction reference, is computed
    from the original data (not Gurobi's coefficients), and the float LP value is not trusted: duals with huge
    cancelling entries give a bound below LP*, while their float master value may be anything."""
    inst, path = tiny(1, tmp_path)
    inst.c_reg = inst.c_reg + 1.0 / 3.0
    inst.c_ot = inst.c_ot + 1.0 / 3.0
    inst.h = inst.h * (1.0 + 1e-13)
    inst.prob = np.full(inst.S, 1.0 / inst.S)
    inst.validate()
    write_instance(inst, Path(path))
    inst = load_instance(path)
    lp = lp_value(path)
    cuts, _ = eng_cuts(path)
    a = gc.certify(inst, cuts)
    b = gc.certify(inst, cuts, exact="fraction")
    assert a["lb_exact"] == b["lb_exact"]
    assert a["lb"] <= lp + 1e-9 * abs(lp)
    big = dict(cuts, pi=cuts["pi"] + 1e15 * (np.arange(inst.T) % 2 * 2 - 1)[None, :])
    c = gc.certify(inst, big)
    assert c["lb"] <= lp and c["lb_exact"] == gc.certify(inst, big, exact="fraction")["lb_exact"]


def test_json_certificate_roundtrip(tmp_path):
    inst, path = tiny(0, tmp_path)
    cuts, _ = eng_cuts(path)
    f = tmp_path / "certificate.json"
    gc.write_certificate(cuts, f)
    back, issues = gc.read_certificate(inst, f)
    assert not issues
    assert gc.certify(inst, back)["lb_exact"] == gc.certify(inst, cuts)["lb_exact"]
    obj = json.loads(f.read_text())
    obj["view"] = "other"
    f.write_text(json.dumps(obj))
    back, issues = gc.read_certificate(inst, f)
    assert issues and len(back["time"]) == 0


def test_initial_design_shape(tmp_path):
    inst, _ = tiny(3, tmp_path)
    y0 = initial_design(inst)
    assert y0.shape == (inst.I, inst.T) and y0.dtype == bool


def test_cutlog_pool_roundtrip(tmp_path):
    inst, path = tiny(2, tmp_path)
    eng = TSSOBendersXL(time_limit=60, integer=False, local_search=False)
    eng.run(path)
    log = eng.cutlog
    a = log.arrays()
    assert log.pool_rows <= log.terms and len(a["sub"]) == log.terms
    f = tmp_path / "cuts.npz"
    log.save(f)
    b = gc.load_cuts(f)
    for k in ("sub", "cut", "pi", "mu", "time"):
        assert np.array_equal(a[k], b[k]), k
    assert gc.certify(inst, b)["lb_exact"] == gc.certify(inst, a)["lb_exact"]


def test_dual_pool_and_model_cache(tmp_path):
    """Mid-size instance: the pooled cuts are used, the LP phase still reaches LP*, the certificate reproduces it, and
    an evaluator holding a single recourse LP gives the same values as an unlimited one."""
    inst = generate(3, 8, 60, 5)
    path = str(write_instance(inst, tmp_path / "mid.json"))
    inst = load_instance(path)
    lp = lp_value(path)
    eng = TSSOBendersXL(time_limit=120, integer=False, local_search=False, pool_ratio=0.0)
    r = eng.run(path)
    assert r["pool_rounds"] > 0 and r["exact_rounds"] > 0
    assert r["lp_bound"] == pytest.approx(lp, rel=1e-6)
    c = gc.certify(inst, eng.cutlog.arrays())
    assert lp - 1e-6 * abs(lp) <= c["lb"] <= lp + 1e-9 * abs(lp)
    eng = TSSOBendersXL(time_limit=120, integer=True, local_search=True, keep_rows=1, model_budget_gb=1e-9)
    r = eng.run(path)  # cut management on every round, two recourse LPs alive
    assert r["removed_cuts"] > 0 and r["blocks"] == 2  # max(2, budget) models alive: blocks of 2 products
    assert r["lp_bound"] == pytest.approx(lp, rel=1e-6)
    assert lp - 1e-6 * abs(lp) <= gc.certify(inst, eng.cutlog.arrays())["lb"] <= lp + 1e-9 * abs(lp)
    a, b = DesignEvaluator(inst), DesignEvaluator(inst, max_models=1)
    rng = np.random.default_rng(0)
    for _ in range(4):
        y = rng.random((inst.I, inst.T)) > 0.5
        assert a.value(y) == pytest.approx(b.value(y), rel=1e-12)
    assert b.builds > inst.I and len(b.lps) == 1
    a.close()
    b.close()
