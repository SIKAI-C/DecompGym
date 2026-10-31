"""Correctness checks for the scale-up SCFL family (loader and generator, recourse LP, extensive-form MIP, cuts,
naive and engineered Benders, heuristic controls, verifier) against brute force over all designs of small
generated instances."""
from __future__ import annotations

import functools
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

gp = pytest.importorskip("gurobipy")
from gurobipy import GRB  # noqa: E402

from src.bd.scfl.a_data import (SCFLInstance, deterministic_instance, load_instance, make_cst, make_stochastic,  # noqa: E402
                                random_instance, read_orlib_cap, write_instance)
from src.bd.scfl.b_mip_xl import (DesignEvaluator, RecourseLP, SCFLExtensiveMIP, build_extensive_form, design_value,  # noqa: E402
                                  make_env, open_list, scfl_output)
from src.bd.scfl.c_naive_bd import (NaiveRecourse, SCFLNaiveBenders, cut_from_duals, cuts_from_duals, greedy_design,  # noqa: E402
                                    lift_closed)
from src.bd.scfl.d_ref_bd_xl import SCFLBendersXL  # noqa: E402
from src.bd.scfl.e_verifier import read_instance, transport, verify_solution  # noqa: E402
from src.bd.scfl.f_heuristic import DesignSearch, SCFLHeuristicControl, cluster_scenarios  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "outputs" / "scaleup" / "raw" / "scfl" / "orlib"
SHAPES = [(3, 3, 2), (4, 3, 3), (3, 5, 2), (4, 4, 4), (5, 3, 2), (4, 5, 3), (3, 4, 1), (5, 4, 3)]


def tiny(seed: int, tmp_path: Path, uniform: bool = True) -> tuple[SCFLInstance, str]:
    rng = np.random.default_rng(seed)
    I, J, S = SHAPES[seed % len(SHAPES)]
    inst = random_instance(rng, I, J, S, tightness=[1.2, 1.6, 2.5][seed % 3], uniform=uniform, integer_costs=seed % 2 == 0)
    path = tmp_path / f"tiny{seed}.json"
    write_instance(inst, path)
    return load_instance(path), str(path)


@functools.lru_cache(maxsize=None)
def _designs_cached(path: str):
    """Every binary design with its exact LP recourse per scenario (None if (15d) fails)."""
    inst = load_instance(path)
    env = make_env()
    lp = RecourseLP(inst, env)
    out = []
    for bits in itertools.product([False, True], repeat=inst.I):
        y = np.array(bits, dtype=bool)
        res = lp.evaluate(y, time.perf_counter() + 60, duals=True)
        out.append((y, res if res["status"] == "optimal" else None))
    lp.dispose()
    env.dispose()
    return out


def all_designs(path: str):
    return _designs_cached(str(path))


def brute_force(path: str) -> float:
    return min(r["value"] for y, r in all_designs(path) if r is not None)


def write_out(tmp_path: Path, out: dict) -> Path:
    p = tmp_path / "out.json"
    p.write_text(json.dumps(out))
    return p


# ---------------------------------------------------------------- data

@pytest.mark.skipif(not (RAW / "cap111.txt").exists(), reason="OR-Library CAP files not downloaded")
def test_orlib_unit_costs_reproduce_the_deterministic_optima(tmp_path):
    """One-scenario EF of the deterministic CAP instance equals the OR-Library optimum (capopt.txt): checks the
    unit-cost conversion alloc_cost / demand."""
    opt = {}
    for line in (RAW / "capopt.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("cap") and parts[0][3:].isdigit():
            opt[parts[0]] = float(parts[1])
    env = make_env()
    for name in ("cap101", "cap111", "cap124", "cap131"):
        base = read_orlib_cap(RAW / f"{name}.txt")
        assert base["demand"].sum() == 58268
        inst = deterministic_instance(base)
        model, v = build_extensive_form(inst, env)
        model.Params.MIPGap = 1e-9
        model.optimize()
        assert model.Status == GRB.OPTIMAL
        assert model.ObjVal == pytest.approx(opt[name], rel=1e-8, abs=1e-2), name
        model.dispose()
    env.dispose()


@pytest.mark.skipif(not (RAW / "cap111.txt").exists(), reason="OR-Library CAP files not downloaded")
def test_generator_is_reproducible_and_roundtrips(tmp_path):
    base = read_orlib_cap(RAW / "cap121.txt")
    a = make_stochastic(base, 40, seed=3)
    b = make_stochastic(base, 40, seed=3)
    c = make_stochastic(base, 40, seed=4)
    assert np.array_equal(a.demand, b.demand) and not np.array_equal(a.demand, c.demand)
    assert a.demand.shape == (40, 50) and np.all(a.demand >= 0) and np.all(a.demand == np.round(a.demand))
    assert np.allclose(a.cost * base["demand"][None, :], base["alloc_cost"])
    assert a.name == "cap121_s40_g3" and a.prob.sum() == pytest.approx(1.0)
    h1 = write_instance(a, tmp_path / "a.json")
    h2 = write_instance(b, tmp_path / "b.json")
    assert h1 == h2
    back = load_instance(tmp_path / "a.json")
    for name in ("capacity", "fixed", "cost", "demand", "prob"):
        assert np.array_equal(getattr(a, name), getattr(back, name)), name
    d = read_instance(tmp_path / "a.json")
    assert d["c"][3][7] == a.cost[3, 7] and d["p"][0] == 1.0 / 40


def test_cst_generator(tmp_path):
    a, b = make_cst(20, 30, 5, 8, seed=2), make_cst(20, 30, 5, 8, seed=2)
    assert np.array_equal(a.demand, b.demand) and np.array_equal(a.fixed, b.fixed)
    assert (a.I, a.J, a.S) == (20, 30, 8) and a.name == "cst20x30_r5_s8_g2"
    assert np.all(a.capacity == np.ceil(a.capacity)) and np.all(a.demand == np.round(a.demand)) and np.all(a.demand >= 0)
    assert np.all(a.cost <= 10 * np.sqrt(2) + 1e-12) and a.feasible(np.ones(a.I))
    write_instance(a, tmp_path / "c.json")
    assert np.array_equal(load_instance(tmp_path / "c.json").cost, a.cost)


def test_random_instances_roundtrip_with_probabilities(tmp_path):
    inst, path = tiny(5, tmp_path, uniform=False)
    back = load_instance(path)
    assert np.array_equal(inst.prob, back.prob) and abs(back.prob.sum() - 1) < 1e-12
    assert read_instance(path)["p"] == back.prob.tolist()


# ---------------------------------------------------------------- recourse, verifier min-cost flow

@pytest.mark.parametrize("seed", range(8))
def test_verifier_transport_matches_the_lp(tmp_path, seed):
    inst, path = tiny(seed, tmp_path)
    d = read_instance(path)
    for y, res in all_designs(path):
        fac = [i for i in range(inst.I) if y[i]]
        for s in range(inst.S):
            got = transport(fac, d["u"], d["c"], d["d"][s])
            if res is None:
                assert not inst.feasible(y)
                assert got is None or inst.demand[s].sum() <= inst.capacity[y].sum()
                continue
            assert got is not None
            assert got[0] == pytest.approx(res["q"][s], rel=1e-9, abs=1e-7)
            # the flows are feasible and integral
            load = np.zeros(inst.I)
            recv = np.zeros(inst.J)
            for (i, j), q in got[1].items():
                assert y[i] and q > 0 and q == round(q)
                load[i] += q
                recv[j] += q
            assert np.all(load <= inst.capacity + 1e-9) and np.allclose(recv, inst.demand[s])


def test_verifier_transport_on_larger_random_problems():
    """Many overloaded facilities and long augmenting paths, fractional demands included."""
    rng = np.random.default_rng(7)
    env = make_env()
    for trial in range(40):
        I, J = int(rng.integers(3, 12)), int(rng.integers(5, 40))
        inst = random_instance(rng, I, J, 1, tightness=float(rng.uniform(1.0, 1.4)))
        if trial % 3 == 0:  # fractional demands
            inst.demand = inst.demand + rng.random(inst.demand.shape).round(3)
        y = np.ones(I, dtype=bool)
        lp = RecourseLP(inst, env)
        res = lp.evaluate(y, time.perf_counter() + 60)
        lp.dispose()
        got = transport(list(range(I)), inst.capacity.tolist(), inst.cost.tolist(), inst.demand[0].tolist())
        if res["status"] == "infeasible":
            assert got is None
            continue
        assert got is not None and got[0] == pytest.approx(res["q"][0], rel=1e-9, abs=1e-6), trial
    env.dispose()


@pytest.mark.parametrize("naive", [False, True])
def test_recourse_cuts_are_tight_at_designs(tmp_path, naive):
    env = make_env()
    for seed in range(6):
        inst, path = tiny(200 + seed, tmp_path)
        rec = NaiveRecourse(inst, env) if naive else RecourseLP(inst, env)
        for y, ref in all_designs(path):
            if ref is None:
                continue
            for s in range(inst.S):
                res = rec.solve(s, y.astype(float), 10) if naive else rec.solve(s, y.astype(float), 10)
                assert res["status"] == "optimal"
                assert res["obj"] == pytest.approx(ref["q"][s], abs=1e-6)
                const, coef = cut_from_duals(inst, s, res["v"], res["alpha"], False, res.get("beta"))
                assert const - coef @ y == pytest.approx(res["obj"], abs=1e-6)
                lifted = lift_closed(inst, s, res["v"], coef, y)
                assert np.all(lifted <= coef + 1e-9)
                assert const - lifted @ y == pytest.approx(res["obj"], abs=1e-6)
        rec.dispose()
    env.dispose()


def test_lifted_coefficient_is_the_minimum_over_alpha():
    rng = np.random.default_rng(4)
    inst = random_instance(rng, 4, 6, 2)
    for _ in range(200):
        v = rng.normal(30, 15, inst.J)
        coef = np.full(inst.I, 1e12)
        y = np.zeros(inst.I, dtype=bool)
        lifted = lift_closed(inst, 0, v, coef, y)
        d = inst.demand[0]
        for i in range(inst.I):
            w = v - inst.cost[i]
            cand = [0.0] + [a for a in w if a > 0]  # the convex piecewise-linear function has its kinks at w_j
            best = min(inst.capacity[i] * a + (np.maximum(w - a, 0) * d).sum() for a in cand)
            assert lifted[i] == pytest.approx(best, rel=1e-12, abs=1e-9)


def test_cuts_from_any_duals_are_valid(tmp_path):
    rng = np.random.default_rng(1)
    for seed in range(5):
        inst, path = tiny(100 + seed, tmp_path)
        designs = [(y, r) for y, r in all_designs(path) if r is not None]
        for _ in range(10):
            s = int(rng.integers(inst.S))
            v, alpha = rng.normal(20, 20, inst.J), np.abs(rng.normal(0, 10, inst.I))
            const, coef = cut_from_duals(inst, s, v, alpha)
            for y, r in designs:
                lifted = lift_closed(inst, s, v, coef, y)
                for y2, r2 in designs:  # a cut lifted at y is valid at every design
                    assert r2["q"][s] >= const - coef @ y2 - 1e-6
                    assert r2["q"][s] >= const - lifted @ y2 - 1e-6


def test_fractional_separation_cuts_are_valid_and_tight(tmp_path):
    rng = np.random.default_rng(2)
    env = make_env()
    for seed in range(6):
        inst, path = tiny(300 + seed, tmp_path)
        designs = [(y, r) for y, r in all_designs(path) if r is not None]
        sep = RecourseLP(inst, env, penalty=True)
        for _ in range(8):
            point = rng.random(inst.I)
            for s in range(inst.S):
                res = sep.solve(s, point, 10, vub=True)
                assert res["status"] == "optimal"
                const, coef = cut_from_duals(inst, s, res["v"], res["alpha"])
                assert const - coef @ point == pytest.approx(res["obj"], rel=1e-7, abs=1e-5)
                for y, r in designs:
                    assert r["q"][s] >= const - coef @ y - 1e-6
        # at binary feasible designs the penalized LP equals the true recourse
        for y, r in designs:
            for s in range(inst.S):
                assert sep.solve(s, y.astype(float), 10, vub=True)["obj"] == pytest.approx(r["q"][s], abs=1e-6)
        sep.dispose()
    env.dispose()


def test_vectorized_cuts_agree(tmp_path):
    inst, path = tiny(9, tmp_path)
    y, r = next((y, r) for y, r in all_designs(path) if r is not None)
    const, coef = cuts_from_duals(inst, np.arange(inst.S), r["V"], r["A"])
    for s in range(inst.S):
        c1, h1 = cut_from_duals(inst, s, r["V"][s], r["A"][s])
        assert const[s] == pytest.approx(c1) and np.allclose(coef[s], h1)


# ---------------------------------------------------------------- extensive form exactness

@pytest.mark.parametrize("strong", [True, False])
def test_extensive_form_admits_a_design_iff_feasible_with_the_verifier_value(tmp_path, strong):
    env = make_env()
    for seed in range(5):
        inst, path = tiny(500 + seed, tmp_path)
        model, v = build_extensive_form(inst, env, strong=strong)
        model.Params.MIPGap = 1e-9
        for y, r in all_designs(path):
            v["y"].LB = y.astype(float)
            v["y"].UB = y.astype(float)
            model.optimize()
            out = scfl_output(inst, open_list(y), env=env)
            if r is None:
                assert model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD)
                assert out["objective_value"] is None
                assert not verify_solution(path, write_out(tmp_path, dict(out, objective_value=0.0)))["ok"]
            else:
                assert model.Status == GRB.OPTIMAL
                ver = verify_solution(path, write_out(tmp_path, out))
                assert ver["ok"], ver["issues"]
                assert model.ObjVal == pytest.approx(ver["objective_value"], rel=1e-9, abs=1e-6)
                assert model.ObjVal == pytest.approx(r["value"], rel=1e-9, abs=1e-6)
        model.dispose()
    env.dispose()


def test_extensive_form_admits_fixed_flows_iff_the_verifier_accepts_them(tmp_path):
    env = make_env()
    rng = np.random.default_rng(8)
    inst, path = tiny(3, tmp_path)
    model, v = build_extensive_form(inst, env, strong=True)
    I, J, S = inst.I, inst.J, inst.S
    for y, r in all_designs(path):
        if r is None:
            continue
        good = scfl_output(inst, open_list(y), flows=True, env=env)
        for variant in range(3):
            flows = [list(e) for e in good["flows"]]
            if variant == 1 and flows:  # move one unit to a random facility (maybe closed or over capacity)
                e = flows[int(rng.integers(len(flows)))]
                e[3] -= 1
                flows.append([e[0], int(rng.integers(I)), e[2], 1])
            if variant == 2 and flows:  # under-serve one customer
                flows[0][3] -= 1
            X = np.zeros((S, I, J))
            for s, i, j, q in flows:
                X[s, i, j] += q
            ok_flows = (np.all(X >= -1e-9) and np.allclose(X.sum(axis=1), inst.demand, rtol=0, atol=1e-9)
                        and np.all(X.sum(axis=2) <= inst.capacity[None, :] * y[None, :] + 1e-9))
            cost = float(inst.fixed @ y + sum(inst.prob[s] * (inst.cost * X[s]).sum() for s in range(S)))
            out = dict(good, flows=flows)
            ver = verify_solution(path, write_out(tmp_path, out))
            v["y"].LB = v["y"].UB = y.astype(float)
            v["x"].LB = np.maximum(X.ravel(), 0)
            v["x"].UB = np.maximum(X.ravel(), 0)
            model.optimize()
            admitted = model.Status == GRB.OPTIMAL and np.all(X >= 0)
            assert admitted == bool(ok_flows) == ver["ok"], (variant, ver["issues"])
            if admitted:
                assert model.ObjVal == pytest.approx(cost, abs=1e-6)
                assert ver["flows_expected_cost"] == pytest.approx(cost - float(inst.fixed @ y), abs=1e-6)
            v["x"].LB = 0.0
            v["x"].UB = np.broadcast_to(inst.demand[:, None, :], (S, I, J)).ravel()
    model.dispose()
    env.dispose()


# ---------------------------------------------------------------- partial decomposition and surrogate

def test_cluster_rows_and_surrogate_never_cut_off_a_feasible_design(tmp_path):
    env = make_env()
    for seed in range(5):
        inst, path = tiny(600 + seed, tmp_path, uniform=seed % 2 == 0)
        labels, red = cluster_scenarios(inst, 2)
        assert red.prob.sum() == pytest.approx(1.0) and len(labels) == inst.S
        for g in range(red.S):
            m = labels == g
            assert np.allclose(red.demand[g], inst.prob[m] @ inst.demand[m] / inst.prob[m].sum())
        sur = RecourseLP(red, env)
        ref = SCFLBendersXL(clusters=2)
        ref.inst, ref.env, ref.labels, ref.red = inst, env, labels, red
        ref.lower = (inst.demand * inst.cost.min(axis=0)[None, :]).sum(axis=1)
        ref.build_master()
        md = ref.master
        for y, r in all_designs(path):
            if r is None:
                continue
            assert sur.evaluate(y, time.perf_counter() + 10)["value"] <= r["value"] + 1e-6  # Jensen
            ref.y.LB = ref.y.UB = y.astype(float)
            ref.theta.LB = ref.theta.UB = r["q"]
            md.optimize()
            assert md.Status == GRB.OPTIMAL, seed
            assert md.ObjVal == pytest.approx(r["value"], abs=1e-6)
        sur.dispose()
        md.dispose()
    env.dispose()


def test_design_search_reaches_a_local_optimum(tmp_path):
    env = make_env()
    inst, path = tiny(4, tmp_path)
    ev, (_, red) = RecourseLP(inst, env), cluster_scenarios(inst, 2)
    sur = RecourseLP(red, env)
    seen = []
    ds = DesignSearch(inst, ev, sur, lambda y, res: seen.append(res["value"]), max_exact=100)
    y0 = greedy_design(inst)
    v0 = ds.value(y0, time.perf_counter() + 10)
    y, v = ds.descend(y0, v0, time.perf_counter() + 10)
    vals = {yy.tobytes(): r["value"] for yy, r in all_designs(path) if r is not None}
    assert v == pytest.approx(vals[y.tobytes()]) and v <= v0
    for z in ds.neighbours(y, "add_drop") + ds.neighbours(y, "swap"):  # every exact-improving neighbour was screened out
        if z.tobytes() in ds.exact:
            assert ds.exact[z.tobytes()] >= v - 1e-6
    ev.dispose(); sur.dispose(); env.dispose()


# ---------------------------------------------------------------- methods vs brute force

def published_values(path: str, designs: list) -> list[float | None]:
    ev = DesignEvaluator(load_instance(path))
    vals = [ev.objective(dz) for dz in designs]
    ev.close()
    return vals


@pytest.mark.parametrize("seed", range(10))
def test_methods_find_the_optimum_with_valid_bounds(tmp_path, seed):
    inst, path = tiny(seed, tmp_path, uniform=seed % 3 != 0)
    opt = brute_force(path)
    variants = {"mip": SCFLExtensiveMIP, "mip_weak": functools.partial(SCFLExtensiveMIP, strong=False),
                "mip_barrier": functools.partial(SCFLExtensiveMIP, method=2), "naive": SCFLNaiveBenders,
                "xl": SCFLBendersXL, "xl_no_partial": functools.partial(SCFLBendersXL, partial=False),
                "xl_no_lifting_no_ls": functools.partial(SCFLBendersXL, lifting=False, local_search=False),
                "xl_branch_and_cut_only": functools.partial(SCFLBendersXL, root_rounds=0, local_search=False),
                "xl_one_cluster": functools.partial(SCFLBendersXL, clusters=1)}
    for name, cls in variants.items():
        seen, bounds = [], []
        res = cls(time_limit=30, on_incumbent=lambda dz, stage=None: seen.append(list(dz)), on_bound=bounds.append).run(path)
        assert res["termination_reason"] == "optimal", name
        assert res["global_lb"] <= opt + 1e-6 and all(b <= opt + 1e-6 for b in bounds), name
        assert bounds == sorted(bounds), name
        assert seen, name
        vals = published_values(path, seen)
        assert all(v is not None and v >= opt - 1e-6 for v in vals), name
        assert min(vals) == pytest.approx(opt, rel=2e-6, abs=1e-6), name  # MIPGap 1e-6
        if not name.startswith("mip"):
            assert vals == sorted(vals, reverse=True), name  # decomposition methods publish only exact improvements
            assert res["best_objective"] == pytest.approx(min(vals), abs=1e-6), name
        else:
            assert res["best_objective"] >= opt - 1e-6, name
        out = scfl_output(inst, seen[-1] if not name.startswith("mip") else res["design"], method=name, flows=seed % 2 == 0)
        assert verify_solution(path, write_out(tmp_path, out))["ok"], name
    for mode, ev_first in (("ls", False), ("saa", False), ("saa", True)):
        seen = []
        res = SCFLHeuristicControl(time_limit=1.0, mode=mode, saa_start=1, clusters=2, ev_first=ev_first,
                                   on_incumbent=lambda dz, stage=None: seen.append(list(dz))).run(path)
        vals = published_values(path, seen)
        assert seen and all(v >= opt - 1e-6 for v in vals) and vals == sorted(vals, reverse=True)
        assert res["global_lb"] is None and res["termination_reason"] in ("time_limit", "finished_early")


# ---------------------------------------------------------------- verifier and contract

def test_verifier_rejects_bad_outputs(tmp_path):
    inst, path = tiny(7, tmp_path)
    y = greedy_design(inst)
    good = scfl_output(inst, open_list(y), method="test", flows=True)

    def check(out):
        return verify_solution(path, write_out(tmp_path, out))

    assert check(good)["ok"]
    assert check({k: v for k, v in good.items() if k != "flows"})["ok"]
    assert check(good)["objective_value"] == pytest.approx(design_value(inst, open_list(y)))
    assert not check(good | {"objective_value": good["objective_value"] + 1})["ok"]
    assert not check(good | {"objective_value": float("nan")})["ok"]
    assert not check(good | {"objective_value": None})["ok"]
    assert not check({k: v for k, v in good.items() if k != "method"})["ok"]
    assert not check(good | {"status": "infeasible"})["ok"]
    assert not check(good | {"y": good["y"][:-1]})["ok"]
    assert not check(good | {"y": [2] + good["y"][1:]})["ok"]
    assert not check(good | {"y": [0] * inst.I, "objective_value": 0.0})["ok"]  # violates (15d)
    s, i, j, q = good["flows"][0]
    closed = next(k for k in range(inst.I) if not y[k]) if not y.all() else None
    if closed is not None:
        moved = [e for e in good["flows"] if e != [s, i, j, q]] + [[s, closed, j, q]]
        assert not check(good | {"flows": moved})["ok"]
    assert not check(good | {"flows": [e for e in good["flows"] if e != [s, i, j, q]]})["ok"]  # demand not met
    assert not check(good | {"flows": good["flows"] + [[s, i, j, 1e9]]})["ok"]  # over capacity
    assert not check(good | {"flows": good["flows"] + [[s, i, j, -0.5], [inst.S, 0, 0, 1]]})["ok"]
    # flows cheaper than the true optimum cannot exist: a fabricated cheaper (infeasible) plan is caught
    cheap = [e if e[0] != s else [e[0], e[1], e[2], 0.5 * e[3]] for e in good["flows"]]
    assert not check(good | {"flows": cheap})["ok"]


# ---------------------------------------------------------------- deadlines and labels on a CAP-based instance

@pytest.fixture(scope="module")
def cap_instance(tmp_path_factory):
    if not (RAW / "cap111.txt").exists():
        pytest.skip("OR-Library CAP files not downloaded")
    inst = make_stochastic(read_orlib_cap(RAW / "cap111.txt"), 250, seed=1)
    path = tmp_path_factory.mktemp("scfl") / "cap111_s250_g1.json"
    write_instance(inst, path)
    return str(path)


@pytest.mark.parametrize("cls", [SCFLExtensiveMIP, SCFLNaiveBenders, SCFLBendersXL,
                                 functools.partial(SCFLHeuristicControl, mode="ls"),
                                 functools.partial(SCFLHeuristicControl, mode="saa")])
def test_deadline_is_respected(cap_instance, cls):
    T = 10.0
    t0 = time.perf_counter()
    seen, bounds = [], []
    res = cls(time_limit=T, on_incumbent=lambda dz, stage=None: seen.append((time.perf_counter() - t0, list(dz))),
              on_bound=lambda b: bounds.append((time.perf_counter() - t0, b))).run(cap_instance)
    assert res["total_seconds"] < T + 3.0
    assert all(t < T for t, _ in seen) and all(t < T for t, _ in bounds)
    if res["termination_reason"] != "optimal":
        assert res["termination_reason"] in ("time_limit", "deadline_during_build") and res["budget_exhausted"]
    if seen:
        vals = published_values(cap_instance, [dz for _, dz in seen])
        assert all(v is not None for v in vals)
        if bounds:
            assert bounds[-1][1] <= min(vals) + 1e-6


def test_verifier_runs_on_a_cap_instance(cap_instance):
    inst = load_instance(cap_instance)
    y = greedy_design(inst)
    out = scfl_output(inst, open_list(y))
    p = Path(cap_instance).with_suffix(".out.json")
    p.write_text(json.dumps(out))
    t = time.perf_counter()
    ver = verify_solution(cap_instance, p)
    print(f"verifier: {time.perf_counter() - t:.1f} s for {inst.S} scenarios, {int(y.sum())} open facilities")
    assert ver["ok"], ver["issues"]


def test_mip_labels_a_build_overrun(cap_instance):
    res = SCFLExtensiveMIP(time_limit=1e-4).run(cap_instance)
    assert res["termination_reason"] == "deadline_during_build" and res["best_objective"] is None


def test_callback_errors_are_not_swallowed(tmp_path):
    """gurobipy ignores exceptions raised in callbacks; the reference must re-raise them, never accept an unchecked design."""
    _, path = tiny(4, tmp_path)

    class Failing(SCFLBendersXL):
        def closed(self):
            return False  # always reach branch-and-cut

        def branch_and_cut(self):
            def boom(*args, **kwargs):
                raise RuntimeError("recourse failure")
            self.ev.evaluate = boom
            return super().branch_and_cut()

    with pytest.raises(RuntimeError, match="recourse failure"):
        Failing(time_limit=30, root_rounds=0, local_search=False).run(path)
