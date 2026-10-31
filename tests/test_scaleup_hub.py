"""Correctness checks for the UMAHLP scale-up family (OR-Library AP loader, compact MIPs, cut machinery, naive and
engineered Benders, verifier) against brute force over all hub subsets and against the OR-Library solution files."""
from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import numpy as np
import pytest

gp = pytest.importorskip("gurobipy")

from src.bd.hub.a_data import (HubInstance, ap_aggregate, ap_instance, from_json, load_hub, parse_ap_nodes, read_raw,
                               write_instance)
from src.bd.hub.b_mip_xl import build_hub_mip, hub_cost, hub_output, route_costs
from src.bd.hub.c_naive_bd import HubCuts, HubNaiveBenders, greedy_add, make_env
from src.bd.hub.d_ref_bd_xl import HubBendersXL, local_search
from src.bd.hub.e_verifier import verify_solution

RAW = Path(__file__).resolve().parents[1] / "outputs" / "scaleup" / "raw" / "hub"
needs_raw = pytest.mark.skipif(not (RAW / "phub1.txt").exists(), reason="OR-Library hub files not downloaded")


def random_instance(rng, n: int) -> HubInstance:
    xy = rng.uniform(0, 100, size=(n, 2))
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    w = rng.uniform(0, 10, size=(n, n)) * (rng.uniform(size=(n, n)) < 0.8)
    return HubInstance(n=n, w=w, d=d, f=rng.uniform(50, 2500, size=n), chi=float(rng.choice([1.0, 3.0])),
                       alpha=float(rng.choice([0.2, 0.75])), delta=float(rng.choice([1.0, 2.0])), coords=xy)


def subsets(n: int):
    for bits in itertools.product([False, True], repeat=n):
        if any(bits):
            yield np.array(bits, dtype=bool)


def brute_force(inst: HubInstance) -> tuple[float, np.ndarray]:
    return min(((hub_cost(inst, z), z) for z in subsets(inst.n)), key=lambda t: t[0])


@pytest.fixture(scope="module")
def env():
    e = make_env(1, None)
    yield e
    e.dispose()


# ---------------------------------------------------------------- data

@needs_raw
def test_generate_port_matches_the_or_library_sample():
    sec = read_raw(RAW)
    agg = ap_aggregate(parse_ap_nodes(sec["APdata200"]), 20)
    sample = parse_ap_nodes(sec["20.3"])  # generate 20 3 < APdata200
    assert np.array_equal(agg["xy"], sample["xy"]) and np.array_equal(agg["w"], sample["w"])
    assert (sample["chi"], sample["alpha"], sample["delta"]) == (3.0, 0.75, 2.0)


@needs_raw
def test_umaphmp_optima_of_phub2_are_reproduced():
    """phub2 (multiple-allocation p-hub median, no fixed costs): our closed-form routing of the published hub sets
    reproduces every published objective, including the intra-node (i = j) flows."""
    sec = read_raw(RAW)
    base = parse_ap_nodes(sec["APdata200"])
    cases = re.findall(r"n=(\d+), p=(\d+) :\s*\nObjective\s*:\s*([\d.]+)\s*\nHubs\s*:\s*([\d, ]+)", sec["solutions.phub2"])
    assert len(cases) == 19  # n = 50, p = 2 has no objective in the file
    for n, p, obj, hubs in cases:
        inst = ap_instance(int(n), "L", RAW)
        h = [int(x) - 1 for x in hubs.replace(",", " ").split()]  # 1-based in the file
        assert len(h) == int(p)
        assert float((inst.w * route_costs(inst, h)).sum()) == pytest.approx(float(obj), abs=0.01)
    assert base["n"] == 200


@needs_raw
def test_or_library_single_allocation_files_bound_the_multiple_allocation_problem():
    """phub1 (CSAHLP: capacitated single allocation with fixed costs) and phub3 (USApHMP) are other variants: the
    UMAHLP optimum (here 10L, 10T by brute force) is at most the CSAHLP optimum with the same fixed costs, and the
    multiple-allocation routing of a published single-allocation hub set costs at most its published objective."""
    sec = read_raw(RAW)
    csahlp = dict(re.findall(r"Problem (\d+[LT][LT])\s*\nObjective:\s*([\d.]+)", sec["solutions.phub1"]))
    for kind in ("L", "T"):
        inst = ap_instance(10, kind, RAW)
        opt, _ = brute_force(inst)
        for cap in ("L", "T"):
            assert opt <= float(csahlp[f"10{kind}{cap}"]) + 1e-6
    assert brute_force(ap_instance(10, "L", RAW))[0] == pytest.approx(221032.73, abs=0.01)
    assert brute_force(ap_instance(10, "T", RAW))[0] == pytest.approx(257558.09, abs=0.01)
    for n, p, obj, alloc in re.findall(r"n=(\d+), p=(\d+) :\s*\nObjective\s*:\s*([\d.]+)\s*\nAllocation : ([\d, ]+)",
                                       sec["solutions.phub3"]):
        inst = ap_instance(int(n), "L", RAW)
        hubs = sorted({int(a) - 1 for a in alloc.replace(",", " ").split()})
        assert len(hubs) == int(p)
        assert float((inst.w * route_costs(inst, hubs)).sum()) <= float(obj) + 0.01


@needs_raw
def test_inherited_fixed_costs_and_files(tmp_path):
    a, b = ap_instance(50, "L", RAW), ap_instance(50, "L50", RAW)
    assert np.allclose(a.f, b.f)
    for n in (100, 200):
        inst = ap_instance(n, "T50", RAW)
        assert inst.source["nested_in_50"] and np.all(np.isin(inst.f, ap_instance(50, "T", RAW).f))
    with pytest.raises(ValueError):
        ap_instance(100, "L", RAW)
    write_instance(b, tmp_path / "ap50.json")
    back = load_hub(tmp_path / "ap50.json")
    for key in ("w", "d", "f"):
        assert np.array_equal(getattr(back, key), getattr(b, key))
    (tmp_path / "spec.json").write_text(json.dumps({"source": "AP", "n": 25, "fixed": "T", "raw_dir": str(RAW)}))
    spec = load_hub(tmp_path / "spec.json")
    assert np.array_equal(spec.w, ap_instance(25, "T", RAW).w)
    assert spec.d[0, 1] == pytest.approx(np.hypot(*(spec.coords[0] - spec.coords[1])) / 1000)


@needs_raw
def test_cab_loader():
    from src.bd.hub.a_data import cab_instance
    inst = cab_instance(25, 0.6, 150.0, RAW)
    assert inst.w[0, 1] == 6469.0 and inst.d[0, 1] == pytest.approx(576.9631) and inst.w[24, 16] == 166694.0
    assert np.allclose(inst.d, inst.d.T) and np.all(np.diag(inst.w) == 0)
    assert cab_instance(10, 0.6, 150.0, RAW).w.shape == (10, 10)


# ---------------------------------------------------------------- compact MIPs

@pytest.mark.parametrize("formulation", ["path", "flow"])
def test_compact_mips_match_brute_force(env, formulation):
    rng = np.random.default_rng(3)
    for _ in range(4):
        inst = random_instance(rng, int(rng.integers(3, 7)))
        opt, _ = brute_force(inst)
        m, v = build_hub_mip(inst, env, formulation=formulation)
        m.Params.OutputFlag = 0
        m.Params.MIPGap = 0.0
        m.optimize()
        assert m.Status == gp.GRB.OPTIMAL and m.ObjVal == pytest.approx(opt, rel=1e-6)
        assert hub_cost(inst, np.asarray(v["z"].X) > 0.5) == pytest.approx(opt, rel=1e-6)
        m.dispose()


@needs_raw
@pytest.mark.parametrize("formulation", ["path", "flow"])
def test_compact_mips_reproduce_a_p_hub_median_optimum(env, formulation):
    """With zero fixed costs and sum z = p the MIPs solve UMApHMP: phub2 n = 10, p = 2 has optimum 163603.94."""
    inst = ap_instance(10, "L", RAW)
    inst.f = np.zeros(inst.n)
    m, v = build_hub_mip(inst, env, formulation=formulation)
    m.addConstr(v["z"].sum() == 2)
    m.Params.OutputFlag = 0
    m.Params.MIPGap = 0.0
    m.optimize()
    assert m.ObjVal == pytest.approx(163603.94, abs=0.01)


# ---------------------------------------------------------------- cuts

def test_cuts_are_valid_and_tight(env):
    """Every cut family is valid for every hub subset of the candidates; integer-point cuts are tight at their point."""
    rng = np.random.default_rng(5)
    for trial in range(8):
        inst = random_instance(rng, int(rng.integers(3, 7)))
        n = inst.n
        cand = np.ones(n, dtype=bool)
        if trial % 2:
            cand[rng.choice(n, size=int(rng.integers(0, n - 1)), replace=False)] = False
        subs = [z for z in subsets(n) if not np.any(z & ~cand)]
        per = {z.tobytes(): (inst.w * route_costs(inst, z)).sum(axis=1) for z in subs}
        hc = HubCuts(inst, cand, rng.permutation(n), dense_group=int(rng.integers(0, 9)), half_size=int(rng.integers(0, n + 1)))
        cuts = []
        for z in subs[:: max(1, len(subs) // 6)]:
            for core in (None, rng.uniform(0, 1, n)):
                cut = hc.binary_cut(z, core)
                assert np.allclose(cut["Q"], per[z.tobytes()])
                assert np.allclose(cut["const"] - cut["coef"] @ z, per[z.tobytes()], rtol=1e-9, atol=1e-6)
                cuts.append(cut)
        for _ in range(3):
            zh = np.where(cand, rng.uniform(0, 1, n), 0.0)
            zh *= max(1.0, 1.2 / max(zh.sum(), 1e-9))
            cuts.append(hc.fractional_cut(zh, int(rng.integers(1, n + 1))))
            lc = hc.lp_cut(zh, env, int(rng.integers(1, n + 1)))
            if lc is not None:
                cuts.append(lc)
        for cut in cuts:
            for z in subs:
                assert np.all(cut["const"] - cut["coef"] @ z <= per[z.tobytes()] + 1e-6 * np.maximum(1.0, per[z.tobytes()]))


def test_lp_cut_equals_the_exact_path_lp(env):
    """At a fractional point the LP-separation cut evaluates to the path LP value (pairs solved one by one here)."""
    from src.bd.hub.b_mip_xl import loop_costs, surviving_edges
    rng = np.random.default_rng(7)
    inst = random_instance(rng, 6)
    n = inst.n
    zh = np.array([0.6, 0.5, 0.0, 0.3, 0.4, 0.2])
    lc = HubCuts(inst).lp_cut(zh, env, n)
    exact = 0.0
    for i in range(n):
        ej, ek, em, eF = surviving_edges(inst, i)
        L = loop_costs(inst, i)[0]
        for j in range(n):
            if inst.w[i, j] <= 0:
                continue
            sel = ej == j
            m = gp.Model(env=env)
            xl = m.addMVar(n)
            xe = m.addMVar(int(sel.sum()))
            m.addConstr(xl.sum() + xe.sum() == 1)
            for h in range(n):
                m.addConstr(xl[h] + gp.quicksum(xe[t] for t in np.nonzero((ek[sel] == h) | (em[sel] == h))[0]) <= zh[h])
            m.setObjective(L[j] @ xl + eF[sel] @ xe)
            m.optimize()
            exact += inst.w[i, j] * m.ObjVal
            m.dispose()
    assert lc["value"].sum() == pytest.approx(exact, rel=1e-7)


# ---------------------------------------------------------------- Benders

def run_and_check(cls, inst: HubInstance, tmp_path: Path, opt: float | None = None, **kw) -> dict:
    path = tmp_path / "inst.json"
    write_instance(inst, path)
    if opt is None:
        opt, _ = brute_force(inst)
    incumbents, bounds = [], []
    res = cls(time_limit=30.0, on_incumbent=incumbents.append, on_bound=bounds.append, **kw).run(str(path))
    assert res["termination_reason"] == "optimal"
    assert res["best_objective"] == pytest.approx(opt, rel=1e-7)
    assert all(b <= opt + 1e-6 * max(1.0, opt) for b in bounds) and bounds == sorted(bounds)
    assert res["global_lb"] <= opt + 1e-6 * max(1.0, opt)
    costs = []
    for k, hubs in enumerate(incumbents):
        out = tmp_path / f"out{k}.json"
        out.write_text(json.dumps(hub_output(inst, hubs, method="benders_decomposition")))
        check = verify_solution(path, out)
        assert check["ok"], check["issues"]
        costs.append(check["objective_value"])
    assert costs and costs == sorted(costs, reverse=True) and costs[-1] == pytest.approx(opt, rel=1e-7)
    return res


@pytest.mark.parametrize("cls", [HubNaiveBenders, HubBendersXL])
def test_benders_reach_the_brute_force_optimum(cls, tmp_path):
    rng = np.random.default_rng(11)
    for _ in range(8):
        run_and_check(cls, random_instance(rng, int(rng.integers(3, 9))), tmp_path)


@needs_raw
@pytest.mark.parametrize("cls", [HubNaiveBenders, HubBendersXL])
def test_benders_solve_ap10(cls, tmp_path):
    for kind in ("L", "T"):
        run_and_check(cls, ap_instance(10, kind, RAW), tmp_path)


@needs_raw
def test_engineered_matches_the_mip_on_ap25(env, tmp_path):
    """n = 25 is beyond brute force: the reference optimum is the compact MIP's; the run also exercises the reduction
    test (hubs closed for good) and the rebuilt cut machinery. (The naive control needs ~15 s here: not in the suite.)"""
    cls = HubBendersXL
    fixed = 0
    for kind in ("L", "T"):
        inst = ap_instance(25, kind, RAW)
        m, _ = build_hub_mip(inst, env, formulation="flow")
        m.Params.OutputFlag = 0
        m.Params.MIPGap = 0.0
        m.optimize()
        assert m.Status == gp.GRB.OPTIMAL
        res = run_and_check(cls, inst, tmp_path, opt=float(m.ObjVal))
        fixed += res["stats"]["fixed_hubs"]
        m.dispose()
    assert fixed > 0


def test_engineered_without_pareto_or_exact_lp(tmp_path):
    rng = np.random.default_rng(13)
    inst = random_instance(rng, 8)
    run_and_check(HubBendersXL, inst, tmp_path, pareto=False, exact_lp=False, half_size=2, reduction=False)


@needs_raw
@pytest.mark.parametrize("cls", [HubNaiveBenders, HubBendersXL])
def test_deadline_is_respected(cls, tmp_path):
    write_instance(ap_instance(100, "L50", RAW), tmp_path / "ap100.json")  # neither method finishes in 1 s
    res = cls(time_limit=1.0).run(str(tmp_path / "ap100.json"))
    assert res["total_seconds"] < 2.0
    assert res["best_objective"] is not None and res["termination_reason"] in ("time_limit", "optimal")


def test_heuristics_return_feasible_sets():
    rng = np.random.default_rng(17)
    inst = random_instance(rng, 8)
    z = greedy_add(inst, 1e18)
    z2, c2 = local_search(inst, z, np.ones(8, dtype=bool), 1e18)
    assert z.any() and z2.any() and c2 <= hub_cost(inst, z) + 1e-9 and c2 == pytest.approx(hub_cost(inst, z2))


# ---------------------------------------------------------------- verifier

def test_verifier_rejects_bad_outputs(tmp_path):
    inst = random_instance(np.random.default_rng(19), 5)
    path = tmp_path / "inst.json"
    write_instance(inst, path)
    good = hub_output(inst, [1, 3], method="x")
    out = tmp_path / "out.json"

    def check(obj):
        out.write_text(json.dumps(obj))
        return verify_solution(path, out)

    assert check(good)["ok"]
    assert check({**good, "objective_value": good["objective_value"] * 1.01})["ok"] is False
    for bad in ({**good, "hubs": []}, {**good, "hubs": [1, 1]}, {**good, "hubs": [5]}, {**good, "hubs": [True]},
                {**good, "hubs": "1"}, {**good, "status": "infeasible"}, {k: v for k, v in good.items() if k != "method"}):
        assert check(bad)["ok"] is False
    out.write_text(json.dumps(good).replace(str(good["objective_value"]), "Infinity"))
    assert verify_solution(path, out)["ok"] is False
    out.write_text(json.dumps(good).replace(str(good["objective_value"]), "NaN"))
    assert verify_solution(path, out)["ok"] is False
