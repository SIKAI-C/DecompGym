"""Bound-side probe code for UFL, stochastic CFL and FCTP at scale (src/bd/{ufl,scfl,fctp}/g_bound_xl.py,
src/bd/scfl/h_view_xl.py, scripts/phoenix/scaleup_bdbound_probe.py). Solver tests: Phoenix only."""
from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

gp = pytest.importorskip("gurobipy")

ROOT = Path(__file__).resolve().parents[1]
import src.experiments.xl_certificate as xc  # noqa: E402

REL = 1e-6


def _run(cls, path, **kw):
    certs = []
    res = cls(time_limit=kw.pop("time_limit", 30.0), threads=1, on_certificate=certs.append, **kw).run(str(path))
    return res, certs


def _close(a, b, rel=REL):
    return abs(float(a) - float(b)) <= rel * max(1.0, abs(float(b)))


# ---------------------------------------------------------------- UFL

@pytest.fixture(scope="module")
def ufl_case(tmp_path_factory):
    from src.bd.ufl import g_bound_xl as g
    d = tmp_path_factory.mktemp("ufl")
    obj = g.make_kg(14, "c", 3)
    p = d / f"{obj['name']}.json"
    g.write_instance(obj, p)
    view = xc.ufl_view(p)
    iv = xc.lp_interval(view, time_limit=60)
    assert iv["status"] == "optimal" and iv["rigorous"]
    inst = g.load_instance(p)
    return g, p, view, Fraction(iv["lp_lo"]), Fraction(iv["lp_hi"]), inst, g.Sorted(inst.D)


def _cert(g, view, inst, srt, data):
    gen = g.to_generic(data, inst, srt)
    assert g.expanded_size(data, srt) == sum(len(t["pi"]) for c in json.loads(gen)["cuts"] for t in c["terms"])
    return xc.check_certificate(view, gen)


def test_ufl_engineered_and_naive_reach_lp(ufl_case):
    g, p, view, lo, hi, inst, srt = ufl_case
    for cls, kw in ((g.UFLEngLP, {}), (g.UFLEngLP, {"stabilize": False}), (g.UFLNaiveLP, {})):
        res, certs = _run(cls, p, **kw)
        assert certs, cls
        r = _cert(g, view, inst, srt, certs[-1])
        assert r.cert_ok and r.lb <= hi
        assert _close(r.lb, lo), (cls, float(r.lb), float(lo))
        if res.get("lp_point") is not None:
            val, _ = g.exact_master_value(inst, srt, res["lp_point"])
            assert val >= lo and _close(val, hi, 1e-6)


def test_ufl_compact_lp_controls(ufl_case):
    g, p, view, lo, hi, inst, srt = ufl_case
    for method in (1, 2):
        res, certs = _run(g.UFLCompactLP, p, method=method)
        assert res["termination_reason"] == "optimal" and certs
        r = _cert(g, view, inst, srt, certs[-1])
        assert r.cert_ok and r.lb <= hi and _close(r.lb, lo, 1e-5)


def test_ufl_trivial_controls_and_lagrangian(ufl_case):
    g, p, view, lo, hi, inst, srt = ufl_case
    _, certs = _run(g.UFLStatic, p, kind="empty")
    r = _cert(g, view, inst, srt, certs[-1])
    assert r.cert_ok and view.trivial_bound() <= r.lb <= lo     # no cuts: the master over the first-stage rows only
    for cls, kw in ((g.UFLStatic, {"kind": "oneshot_greedy"}), (g.UFLPool, {"time_limit": 5.0})):
        _, certs = _run(cls, p, **kw)
        r = _cert(g, view, inst, srt, certs[-1])
        assert r.cert_ok and r.lb <= hi
    res, certs = _run(g.UFLLagrangian, p, time_limit=5.0)
    r = _cert(g, view, inst, srt, certs[-1])
    assert r.cert_ok and r.lb <= hi
    assert _close(r.lb, res["global_lb"], 1e-6)          # the checker master equals L(u)


def test_ufl_primal_arms(ufl_case):
    g, p, view, lo, hi, inst, srt = ufl_case
    for cls in (g.UFLCompactMIP, g.UFLNaiveInt, g.UFLILS, g.UFLEngPrimal):
        got = []
        cls(time_limit=5.0, threads=1, on_incumbent=lambda d, stage=None: got.append(d)).run(str(p))
        assert got
        assert g.design_value(inst, got[-1]) >= float(lo) - 1e-6


# ---------------------------------------------------------------- SCFL

@pytest.fixture(scope="module")
def scfl_case(tmp_path_factory):
    from src.bd.scfl import g_bound_xl as g
    from src.bd.scfl.a_data import random_instance, write_instance
    from src.bd.scfl.h_view_xl import scfl_view
    d = tmp_path_factory.mktemp("scfl")
    inst = random_instance(np.random.default_rng(5), 5, 7, 6)
    p = d / "rand.json"
    write_instance(inst, p)
    view = scfl_view(p)
    sol = xc.gurobi_de(view, time_limit=60)
    assert sol.status == "optimal"
    lo = xc.de_dual_bound(view, sol.duals)
    from src.bd.scfl.a_data import load_instance
    inst = load_instance(p)
    env = __import__("src.bd.scfl.b_mip_xl", fromlist=["make_env"]).make_env(1)
    md, info = g.build_ef_lp(inst.I, inst.J, inst.fixed, inst.capacity, inst.cost, inst.demand, inst.prob, env)
    md.Params.FeasibilityTol = 1e-9
    md.optimize()
    assert abs(md.ObjVal - sol.objective) <= 1e-6 * max(1.0, abs(sol.objective))   # view DE == compact LP
    md.dispose()
    env.dispose()
    return g, p, view, lo, sol.objective, inst


def _scert(view, inst, data):
    from src.bd.scfl.h_view_xl import expanded_size, to_generic
    gen = to_generic(data, inst.cost)
    assert expanded_size(data, inst.cost) == sum(len(t["pi"]) for c in json.loads(gen)["cuts"] for t in c["terms"])
    return xc.check_certificate(view, gen)


def test_scfl_benders_and_compact_reach_lp(scfl_case):
    g, p, view, lo, obj, inst = scfl_case
    for cls, kw in ((g.SCFLEngLP, {}), (g.SCFLEngLP, {"stabilize": False}), (g.SCFLNaiveLP, {}),
                    (g.SCFLCompactLP, {"method": 2}), (g.SCFLCompactLP, {"method": 1})):
        res, certs = _run(cls, p, **kw)
        assert certs, cls
        r = _scert(view, inst, certs[-1])
        assert r.cert_ok and float(r.lb) <= obj + 1e-6 * abs(obj)
        assert _close(r.lb, lo, 1e-6), (cls, kw, float(r.lb), float(lo))
        if res.get("lp_point") is not None and cls is not g.SCFLCompactLP:
            val, info = g.exact_master_value(str(p), inst, res["lp_point"])
            assert val is not None, info
            assert val >= lo and _close(val, obj, 1e-4)


def test_scfl_controls(scfl_case):
    g, p, view, lo, obj, inst = scfl_case
    _, certs = _run(g.SCFLStatic, p, kind="empty")
    r = _scert(view, inst, certs[-1])
    assert r.cert_ok and view.trivial_bound() <= r.lb <= lo
    for cls, kw in ((g.SCFLStatic, {"kind": "oneshot_greedy"}), (g.SCFLStatic, {"kind": "oneshot_saa", "saa_scenarios": 3}),
                    (g.SCFLPool, {"time_limit": 5.0}), (g.SCFLJensen, {"g0": 2})):
        _, certs = _run(cls, p, **kw)
        assert certs, (cls, kw)
        r = _scert(view, inst, certs[-1])
        assert r.cert_ok and r.lb <= lo + Fraction(1, 10**6)
    res, certs = _run(g.SCFLLagrangian, p, time_limit=5.0)
    r = _scert(view, inst, certs[-1])
    assert r.cert_ok and r.lb <= lo + Fraction(1, 10**6)
    assert _close(r.lb, res["global_lb"], 1e-6)          # the checker master equals L(u)


# ---------------------------------------------------------------- FCTP

def test_fctp_balinski_one_cut_is_lp(tmp_path):
    from src.bd.fctp import g_bound_xl as g
    base = ROOT / "outputs" / "scaleup" / "raw" / "fctp" / "glover" / "N3004.FCTP"
    if not base.exists():
        pytest.skip("Glover base instance not present")
    p = tmp_path / "gl.FCTP"
    p.write_text(g.make_glover_like(7, 9, 1, base))
    view = xc.fctp_view(p)
    iv = xc.lp_interval(view, time_limit=60)
    assert iv["status"] == "optimal"
    lo = Fraction(iv["lp_lo"])
    for kw in ({"balinski": True, "method": 1}, {"method": 1}, {"method": 2, "crossover": -1}):
        res, certs = _run(g.FCTPCompactLP, p, **kw)
        r = xc.check_certificate(view, g.to_generic(certs[-1], g.load_fctp(p).tcost))
        assert r.cert_ok and _close(r.lb, lo, 1e-7), kw
        if res.get("lp_hi_exact"):
            assert Fraction(res["lp_hi_exact"]) >= lo and _close(Fraction(res["lp_hi_exact"]), lo, 1e-7)


# ---------------------------------------------------------------- runner smoke (supervisor, hard kill, checkpoints)

def test_runner_task_smoke(ufl_case, tmp_path):
    g, p, view, lo, hi, inst, srt = ufl_case
    out = tmp_path / "out"
    cmd = [sys.executable, str(ROOT / "scripts" / "phoenix" / "scaleup_bdbound_probe.py"), "--task", "ufl", str(p),
           "eng,empty,lagr", "--T", "12", "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    r = json.loads((out / f"{p.stem}__eng" / "result.json").read_text())
    final = r["certificates"]["checkpoints"][-1]
    assert final["cert_ok"] and _close(Fraction(final["lb"]), lo)
    lag = json.loads((out / f"{p.stem}__lagr" / "result.json").read_text())
    assert lag["run"]["killed_at_T"] or lag["run"]["wall_seconds"] <= 12.5
    s = subprocess.run([sys.executable, str(ROOT / "scripts" / "phoenix" / "scaleup_bdbound_probe.py"), "--summarize", "ufl",
                        "--out", str(out)], capture_output=True, text=True, cwd=str(ROOT), timeout=600)
    assert s.returncode == 0, s.stderr[-3000:]
