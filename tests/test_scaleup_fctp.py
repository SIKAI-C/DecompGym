"""Correctness checks for the scale-up FCTP family (loader, compact MIP, cuts, naive and engineered Benders, basis
local search, heuristic control, verifier) against brute force over all designs of tiny generated instances."""
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

from src.bd.fctp.a_data_xl import FCTPInstance, load_fctp, parse_fctp, random_instance, write_fctp  # noqa: E402
from src.bd.fctp.b_mip_xl import (FCTPCompactMIP, TransportLP, build_fctp_mip, fctp_output, flow_cost, make_env,  # noqa: E402
                                  matrix_to_flows, normalize_flows, plan_cost)
from src.bd.fctp.c_naive_bd import FCTPNaiveBenders, NaiveRecourse, cut_coefficients, cut_constant, initial_plan  # noqa: E402
from src.bd.fctp.d_ref_bd_xl import FCTPBendersXL, ParetoLP, cardinality, mir_row, source_side  # noqa: E402
from src.bd.fctp.e_verifier_xl import verify_solution  # noqa: E402
from src.bd.fctp.f_heuristic_xl import BasisLS, FCTPHeuristicControl, iterated_local_search  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "outputs" / "scaleup" / "raw" / "fctp" / "glover"
SHAPES = [(2, 3), (3, 3), (2, 4), (3, 2), (3, 3), (2, 4), (3, 3), (4, 2)]


def tiny(seed: int, tmp_path: Path) -> tuple[FCTPInstance, str]:
    rng = np.random.default_rng(seed)
    m, n = SHAPES[seed % len(SHAPES)]
    inst = random_instance(rng, m, n, fixed_scale=[0.2, 1.0, 5.0, 20.0][seed % 4], total=int(rng.integers(20, 60)))
    path = tmp_path / f"tiny{seed}.FCTP"
    write_fctp(inst, path)
    return load_fctp(path), str(path)


@functools.lru_cache(maxsize=None)
def _designs_cached(key: str):
    inst = load_fctp(key)
    env = make_env()
    lp = TransportLP(inst, env)
    out = []
    for bits in itertools.product([False, True], repeat=inst.m * inst.n):
        y = np.array(bits, dtype=bool).reshape(inst.m, inst.n)
        res = lp.evaluate(y, 10)
        out.append((y, res))
    lp.dispose()
    env.dispose()
    return out


def all_designs(path: str):
    return _designs_cached(str(path))


def brute_force(inst: FCTPInstance, path: str) -> float:
    return min(float(inst.fcost[y].sum()) + r["obj"] for y, r in all_designs(path) if r["status"] == "optimal")


def write_out(tmp_path: Path, out: dict) -> Path:
    p = tmp_path / "out.json"
    p.write_text(json.dumps(out))
    return p


# ---------------------------------------------------------------- data

@pytest.mark.skipif(not RAW.exists(), reason="Glover instances not downloaded")
def test_loader_parses_every_glover_file():
    files = sorted(RAW.glob("*.FCTP"))
    assert len(files) == 132
    for f in files:
        inst = load_fctp(f)
        assert inst.integral and inst.supply.sum() == inst.demand.sum()
    n3709 = load_fctp(RAW / "N3709.FCTP")
    assert (n3709.m, n3709.n, n3709.supply[:3].tolist()) == (50, 100, [207.0, 1563.0, 1370.0])


def test_write_and_load_roundtrip(tmp_path):
    inst = random_instance(np.random.default_rng(0), 4, 5)
    write_fctp(inst, tmp_path / "a.FCTP")
    back = load_fctp(tmp_path / "a.FCTP")
    for name in ("supply", "demand", "tcost", "fcost"):
        assert np.array_equal(getattr(inst, name), getattr(back, name))
    with pytest.raises(ValueError):
        parse_fctp("2 2\n1 1\n1 2\n1 1 1 1\n1 1 1 1\n")  # unbalanced


# ---------------------------------------------------------------- cuts

def test_cuts_from_any_duals_are_valid(tmp_path):
    rng = np.random.default_rng(1)
    for seed in range(5):
        inst, path = tiny(100 + seed, tmp_path)
        designs = all_designs(path)
        for _ in range(8):
            u, v = rng.normal(0, 20, inst.m), rng.normal(0, 20, inst.n)
            for feas in (False, True):
                const, h = cut_constant(inst, u, v), cut_coefficients(inst, u, v, feas)
                for y, res in designs:
                    if res["status"] != "optimal":
                        continue
                    if feas:
                        assert (h * y).sum() >= const - 1e-6
                    else:
                        assert res["obj"] >= const - (h * y).sum() - 1e-6


@pytest.mark.parametrize("naive", [False, True])
def test_recourse_cuts_are_tight_and_farkas_cuts_separate(tmp_path, naive):
    env = make_env()
    for seed in range(6):
        inst, path = tiny(200 + seed, tmp_path)
        rec = NaiveRecourse(inst, env) if naive else TransportLP(inst, env)
        for y, ref in all_designs(path):
            res = rec.solve(y, 10) if naive else rec.solve(inst.cap * y, 10)
            assert res["status"] == ref["status"]
            feas = res["status"] == "infeasible"
            const = cut_constant(inst, res["u"], res["v"])
            h = cut_coefficients(inst, res["u"], res["v"], feas, res.get("w"))
            if feas:
                assert const - (h * y).sum() > 1e-9
            else:
                assert res["obj"] == pytest.approx(ref["obj"], abs=1e-6)
                assert const - (h * y).sum() == pytest.approx(res["obj"], abs=1e-6)  # tight at the design
        rec.dispose()
    env.dispose()


def test_pareto_cuts_are_valid_and_tight(tmp_path):
    env = make_env()
    rng = np.random.default_rng(3)
    for seed in range(5):
        inst, path = tiny(300 + seed, tmp_path)
        designs = all_designs(path)
        mw = ParetoLP(inst, env)
        feasible = [(y, r) for y, r in designs if r["status"] == "optimal"]
        for y, r in feasible[:: max(1, len(feasible) // 12)]:
            # A core point in the convex hull of feasible designs (the MW LP is then bounded).
            pick = rng.choice(len(feasible), size=3, replace=False)
            w = rng.dirichlet(np.ones(3))
            y0 = sum(wk * feasible[k][0].astype(float) for wk, k in zip(w, pick))
            uv = mw.duals(y, r["obj"], y0, 10)
            assert uv is not None
            const, h = cut_constant(inst, *uv), cut_coefficients(inst, *uv, False)
            assert const - (h * y).sum() == pytest.approx(r["obj"], abs=1e-6)
            for y2, r2 in feasible:
                assert r2["obj"] >= const - (h * y2).sum() - 1e-6
            # Pareto: at the core point it dominates the plain cut of the same design.
            pc, ph = cut_constant(inst, r["u"], r["v"]), cut_coefficients(inst, r["u"], r["v"], False)
            assert const - (h * y0).sum() >= pc - (ph * y0).sum() - 1e-6
        mw.dispose()
    env.dispose()


def test_cardinality_rows_are_valid(tmp_path):
    for seed in range(5):
        inst, path = tiny(400 + seed, tmp_path)
        cap = inst.cap
        rj = [cardinality(cap[:, j], inst.demand[j]) for j in range(inst.n)]
        ri = [cardinality(cap[i, :], inst.supply[i]) for i in range(inst.m)]
        for y, res in all_designs(path):
            if res["status"] == "optimal":
                assert all(y[:, j].sum() >= rj[j] for j in range(inst.n))
                assert all(y[i, :].sum() >= ri[i] for i in range(inst.m))
                assert np.all((cap * y).sum(axis=0) >= inst.demand) and np.all((cap * y).sum(axis=1) >= inst.supply)


def test_mir_rows_are_valid():
    rng = np.random.default_rng(11)
    for _ in range(300):
        k = int(rng.integers(1, 7))
        coef = rng.integers(1, 30, size=k).astype(float)
        rhs = float(rng.integers(1, int(coef.sum()) + 1))
        row = mir_row(coef, rhs, rng.random(k))
        if row is None:
            continue
        g, r, _ = row
        for bits in itertools.product([0, 1], repeat=k):
            yb = np.array(bits, dtype=float)
            if coef @ yb >= rhs:
                assert g @ yb >= r - 1e-9


def test_hall_cutsets_are_valid_and_cut_off_infeasible_designs(tmp_path):
    rng = np.random.default_rng(12)
    for seed in range(6):
        inst, path = tiny(800 + seed, tmp_path)
        ref = FCTPBendersXL()
        ref.inst = inst
        designs = all_designs(path)
        feasible = [y for y, r in designs if r["status"] == "optimal"]
        for y, r in designs:
            sup, cus, flow = source_side(inst.supply, inst.demand, np.where(y, inst.supply.sum() + 1, 0.0))
            assert (flow == inst.supply.sum()) == (r["status"] == "optimal")
            if r["status"] == "optimal":
                continue
            rows = ref.cutset_rows(sup, cus, rng.random((inst.m, inst.n)))
            assert rows
            for idx, coef, rhs, kind in rows:
                if kind != "mir":
                    assert coef @ y.ravel()[idx] < rhs - 1e-9  # the infeasible design is cut off
                for y2 in feasible:
                    assert coef @ y2.ravel()[idx] >= rhs - 1e-9
        for _ in range(20):  # random sets and points: every row is valid for every feasible design
            sup, cus = rng.random(inst.m) < 0.5, rng.random(inst.n) < 0.5
            for idx, coef, rhs, kind in ref.cutset_rows(sup, cus, rng.random((inst.m, inst.n))):
                for y2 in feasible:
                    assert coef @ y2.ravel()[idx] >= rhs - 1e-9


# ---------------------------------------------------------------- MIP exactness

def test_mip_admits_a_fixed_design_iff_its_recourse_is_feasible(tmp_path):
    env = make_env()
    for seed in range(4):
        inst, path = tiny(500 + seed, tmp_path)
        model, v = build_fctp_mip(inst, env)
        for y, res in all_designs(path):
            v["y"].LB = y.astype(float)
            v["y"].UB = y.astype(float)
            model.optimize()
            if res["status"] == "optimal":
                assert model.Status == GRB.OPTIMAL
                assert model.ObjVal == pytest.approx(inst.fcost[y].sum() + res["obj"], abs=1e-6)
            else:
                assert model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD)
        model.dispose()
    env.dispose()


def test_mip_admits_a_fixed_plan_iff_the_verifier_accepts_it(tmp_path):
    env = make_env()
    rng = np.random.default_rng(6)
    for seed in range(4):
        inst, path = tiny(600 + seed, tmp_path)
        model, v = build_fctp_mip(inst, env)
        plans = [r["X"] for y, r in all_designs(path) if r["status"] == "optimal"][::5]
        cands = []
        for X in plans[:15]:
            cands.append(X)
            bad = X.copy()
            i, j = int(rng.integers(inst.m)), int(rng.integers(inst.n))
            bad[i, j] += 0.5  # breaks a supply and a demand row
            cands.append(bad)
            if inst.m > 1 and inst.n > 1:  # a balanced 2x2 exchange keeps every row: feasible iff nonnegative
                sw = X.copy()
                sw[0, 0] += 1.0; sw[1, 1] += 1.0; sw[0, 1] -= 1.0; sw[1, 0] -= 1.0
                cands.append(sw)
        for X in cands:
            y = (np.abs(X) > 1e-9).astype(int)
            out = {"status": "feasible", "method": "t", "objective_value": float((inst.fcost * y).sum() + (inst.tcost * X).sum()),
                   "y": y.tolist(), "x": X.tolist()}
            ok = verify_solution(path, write_out(tmp_path, out))["ok"]
            if np.any(X < 0):
                feasible = False  # outside the MIP's variable domain x >= 0
            else:
                v["x"].LB = X
                v["x"].UB = X
                v["y"].LB = y.astype(float)
                v["y"].UB = y.astype(float)
                model.optimize()
                feasible = model.Status == GRB.OPTIMAL
            assert feasible == ok, (X, ok)
            v["x"].LB = 0.0
            v["x"].UB = inst.cap
        model.dispose()
    env.dispose()


# ---------------------------------------------------------------- local search

def test_basis_local_search_moves_are_exact_and_end_in_local_optima(tmp_path):
    for seed in range(8):
        inst, path = tiny(700 + seed, tmp_path)
        rng = np.random.default_rng(seed)
        init = initial_plan(inst, make_env(), 10)
        # A plan with a cycle in its support (uniform split) must be repaired into a basis without a cost increase.
        U = np.outer(inst.supply, inst.demand) / inst.supply.sum()
        for X0 in (init["X"], U):
            ls = BasisLS(inst, X0)
            assert ls.cost <= plan_cost(inst, X0) + 1e-6
            arcs = ls.tree_arcs()
            assert len(arcs) == inst.m + inst.n - 1 and len(set(ls._components())) == 1
            X = ls.matrix()
            assert flow_cost(inst, matrix_to_flows(X)) == pytest.approx(ls.cost, abs=1e-6)
            assert all(X[i, j] == 0 or (i, j) in set(arcs) for i in range(inst.m) for j in range(inst.n))
            for _ in range(6):
                g, i, j = ls.best_move()
                if i < 0:
                    break
                before = ls.cost
                change = ls.pivot(i, j)
                assert change == pytest.approx(g, abs=1e-6)
                assert ls.cost == pytest.approx(ls.exact_cost(), abs=1e-6) == pytest.approx(before + g, abs=1e-6)
                assert flow_cost(inst, matrix_to_flows(ls.matrix())) == pytest.approx(ls.cost, abs=1e-6)
            ls.descend(time.perf_counter() + 5)
            g, i, j = ls.best_move()
            assert i < 0 or g > -ls.eps
            # Brute check of the local optimum: every non-degenerate entering move, applied on a copy, does not improve.
            for a in range(inst.m):
                cand: list[int] = []
                ls.scan_root(a, cand)
                for b in set(cand):
                    snap = ls.snapshot()
                    ls.pivot(a, b)
                    assert ls.exact_cost() >= snap[3] - ls.eps
                    ls.restore(snap)
            ls.kick(rng, 3)
            assert ls.cost == pytest.approx(ls.exact_cost(), abs=1e-6)
            assert flow_cost(inst, matrix_to_flows(ls.matrix())) is not None


def test_iterated_local_search_publishes_only_improving_feasible_plans(tmp_path):
    inst, path = tiny(3, tmp_path)
    opt = brute_force(inst, path)
    seen = []
    ls = BasisLS(inst, initial_plan(inst, make_env(), 10)["X"])
    iterated_local_search(ls, time.perf_counter() + 1.0, np.random.default_rng(0), seen.append)
    costs = [flow_cost(inst, matrix_to_flows(X)) for X in seen]
    assert all(c is not None and c >= opt - 1e-6 for c in costs)
    assert costs == sorted(costs, reverse=True)


# ---------------------------------------------------------------- methods vs brute force

@pytest.mark.parametrize("seed", range(10))
def test_methods_find_the_optimum_with_valid_bounds(tmp_path, seed):
    inst, path = tiny(seed, tmp_path)
    opt = brute_force(inst, path)
    variants = {"mip": FCTPCompactMIP, "naive": FCTPNaiveBenders, "xl": FCTPBendersXL,
                "xl_no_ls_no_pareto": functools.partial(FCTPBendersXL, local_search=False, pareto=False),
                "xl_branch_and_cut_only": functools.partial(FCTPBendersXL, root_rounds=0, ss_steps=0, local_search=False),
                "xl_no_cutsets": functools.partial(FCTPBendersXL, cutsets=False),
                "xl_ils": functools.partial(FCTPBendersXL, ils_seconds=0.05),
                "xl_mip_focus": functools.partial(FCTPCompactMIP, mip_focus=1)}
    for name, cls in variants.items():
        seen, bounds = [], []
        res = cls(time_limit=30, on_incumbent=lambda fl, stage=None: seen.append(fl), on_bound=bounds.append).run(path)
        assert res["termination_reason"] == "optimal", name
        assert res["best_objective"] == pytest.approx(opt, abs=1e-6), name
        assert res["global_lb"] <= opt + 1e-6 and all(b <= opt + 1e-6 for b in bounds), name
        assert bounds == sorted(bounds), name
        assert seen, name
        costs = [flow_cost(inst, fl) for fl in seen]
        assert all(c is not None and c >= opt - 1e-6 for c in costs), name
        assert costs == sorted(costs, reverse=True), name
        for fl in seen[-2:]:  # published plans pass the independent verifier
            assert verify_solution(path, write_out(tmp_path, fctp_output(inst, fl, method=name)))["ok"], name
        assert flow_cost(inst, res["flows"]) == pytest.approx(res["best_objective"], abs=1e-6), name
    seen = []
    res = FCTPHeuristicControl(time_limit=1.0, on_incumbent=lambda fl, stage=None: seen.append(fl)).run(path)
    assert seen and all(flow_cost(inst, fl) >= opt - 1e-6 for fl in seen)
    assert res["global_lb"] is None and res["termination_reason"] in ("time_limit", "finished_early")


# ---------------------------------------------------------------- verifier and contract

def test_verifier_rejects_bad_outputs(tmp_path):
    inst, path = tiny(7, tmp_path)
    X = initial_plan(inst, make_env(), 10)["X"]
    good = fctp_output(inst, matrix_to_flows(X), method="test")

    def check(out):
        return verify_solution(path, write_out(tmp_path, out))

    assert check(good)["ok"]
    assert check(good)["objective_value"] == pytest.approx(flow_cost(inst, good["flows"]))
    dense = {k: v for k, v in good.items() if k != "flows"} | {"x": X.tolist()}
    assert check(dense)["ok"]
    assert not check(good | {"objective_value": good["objective_value"] + 1})["ok"]
    assert not check(good | {"objective_value": float("inf")})["ok"]
    assert not check(good | {"objective_value": None})["ok"]
    assert not check({k: v for k, v in good.items() if k != "method"})["ok"]
    assert not check(good | {"status": "infeasible"})["ok"]
    assert not check(good | {"y": [[2] * inst.n] * inst.m})["ok"]
    assert not check({k: v for k, v in good.items() if k != "flows"})["ok"]
    i, j, q = good["flows"][0]
    closed = [row[:] for row in good["y"]]
    closed[i][j] = 0
    assert not check(good | {"y": closed, "objective_value": good["objective_value"] - inst.fcost[i, j]})["ok"]
    short = [e for e in good["flows"] if e != [i, j, q]]
    assert not check(good | {"flows": short, "objective_value": good["objective_value"] - inst.tcost[i, j] * q})["ok"]
    assert not check(good | {"flows": good["flows"] + [[0, 0, -1.0], [0, 0, 1.0], [0, 1, -0.5]]})["ok"]
    # An extra open arc without flow is allowed when its fixed cost is claimed.
    y2 = [row[:] for row in good["y"]]
    a, b = next((a, b) for a in range(inst.m) for b in range(inst.n) if y2[a][b] == 0)
    y2[a][b] = 1
    assert check(good | {"y": y2, "objective_value": good["objective_value"] + inst.fcost[a, b]})["ok"]


def test_contract_helpers_agree():
    inst = random_instance(np.random.default_rng(9), 3, 4)
    X = initial_plan(inst, make_env(), 10)["X"]
    fl = matrix_to_flows(X)
    assert normalize_flows(fl + [[0, 0, 0]]) == fl
    assert normalize_flows([[0, 1, 2], [0, 1, 3]]) == [[0, 1, 5]]
    assert flow_cost(inst, fl) == pytest.approx(plan_cost(inst, X))
    assert flow_cost(inst, fl[1:]) is None
    assert flow_cost(inst, [[inst.m, 0, 1]] + fl) is None


# ---------------------------------------------------------------- deadlines and labels on a real instance

@pytest.mark.skipif(not (RAW / "N3709.FCTP").exists(), reason="Glover instances not downloaded")
@pytest.mark.parametrize("cls", [FCTPCompactMIP, FCTPNaiveBenders, FCTPBendersXL, FCTPHeuristicControl])
def test_deadline_is_respected(cls):
    path = str(RAW / "N3709.FCTP")
    inst = load_fctp(path)
    T = 6.0
    t0 = time.perf_counter()
    seen, bounds = [], []
    res = cls(time_limit=T, on_incumbent=lambda fl, stage=None: seen.append((time.perf_counter() - t0, fl)),
              on_bound=lambda b: bounds.append((time.perf_counter() - t0, b))).run(path)
    assert res["total_seconds"] < T + 3.0
    assert seen and all(t < T for t, _ in seen) and all(t < T for t, _ in bounds)
    costs = [flow_cost(inst, fl) for _, fl in seen]
    assert all(c is not None for c in costs) and costs == sorted(costs, reverse=True)
    assert res["best_objective"] == pytest.approx(costs[-1])
    if bounds:
        assert bounds[-1][1] <= costs[-1]
    assert res["termination_reason"] == "time_limit" and res["budget_exhausted"]


@pytest.mark.skipif(not (RAW / "N3709.FCTP").exists(), reason="Glover instances not downloaded")
def test_mip_labels_a_build_overrun():
    res = FCTPCompactMIP(time_limit=1e-4).run(str(RAW / "N3709.FCTP"))
    assert res["termination_reason"] == "deadline_during_build" and res["best_objective"] is None


def test_callback_errors_are_not_swallowed(tmp_path):
    """gurobipy ignores exceptions raised in callbacks; the reference must re-raise them, never accept an unchecked design."""
    _, path = tiny(4, tmp_path)

    class Failing(FCTPBendersXL):
        def closed(self):
            return False  # always reach branch-and-cut

        def branch_and_cut(self):
            def boom(*args, **kwargs):
                raise RuntimeError("recourse failure")
            self.ev.evaluate = boom
            return super().branch_and_cut()

    with pytest.raises(RuntimeError, match="recourse failure"):
        Failing(time_limit=30, root_rounds=0, ss_steps=0, local_search=False).run(path)
