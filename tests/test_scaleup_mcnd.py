"""Correctness checks for the MCND scale-up family (loader, compact MIP, naive and engineered Benders, verifier)
against brute force over all designs of tiny generated instances."""
from __future__ import annotations

import functools
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

gp = pytest.importorskip("gurobipy")

from src.bd.mcnd.a_data import MCNDInstance, load_dow, write_dow
from src.bd.mcnd.b_mip_xl import build_mcnd_mip, design_cost, design_output, solve_flows
from src.bd.mcnd.c_naive_bd import MCNDNaiveBenders, RecourseLP, arc_excess, cut_constant, make_env, plain_cut, strong_arc_coefficients
from src.bd.mcnd.d_ref_bd_xl import MCNDBendersXL, ReducedRecourse, mir_row
from src.bd.mcnd.e_verifier import verify_solution

RAW = Path(__file__).resolve().parents[1] / "outputs" / "scaleup" / "raw" / "mcnd"


def random_instance(rng, n: int, m: int, K: int, tight: bool) -> MCNDInstance:
    """A strongly connected random digraph (a Hamiltonian cycle plus random arcs) with K commodities."""
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    perm = rng.permutation(n)
    arcs = [(int(perm[i]), int(perm[(i + 1) % n])) for i in range(n)]
    rest = [p for p in pairs if p not in arcs]
    arcs += [rest[i] for i in rng.choice(len(rest), size=m - n, replace=False)]
    arcs = np.array(arcs)
    od = rng.choice(len(pairs), size=K, replace=False)
    demand = rng.integers(5, 30, size=K).astype(float)
    lo, hi = (0.3, 1.0) if tight else (0.8, 2.0)
    cap = rng.integers(int(demand.sum() * lo), int(demand.sum() * hi) + 1, size=m).astype(float)
    return MCNDInstance(n=n, tail=arcs[:, 0], head=arcs[:, 1], cost=rng.integers(1, 10, size=m).astype(float), cap=cap,
                        fixed=rng.integers(50, 600, size=m).astype(float), orig=np.array([pairs[i][0] for i in od]),
                        dest=np.array([pairs[i][1] for i in od]), demand=demand)


def feasible_instance(seed: int, tmp_path: Path) -> tuple[MCNDInstance, str]:
    rng = np.random.default_rng(seed)
    while True:
        n = int(rng.integers(3, 6))
        m = int(rng.integers(n, min(10, n * (n - 1)) + 1))
        inst = random_instance(rng, n, m, int(rng.integers(1, 5)), tight=bool(seed % 2))
        if design_cost(inst, np.ones(inst.m, dtype=bool)) is not None:
            break
    path = tmp_path / f"tiny{seed}.dow"
    write_dow(inst, path)
    return load_dow(path), str(path)


def all_designs(inst: MCNDInstance):
    for bits in itertools.product([False, True], repeat=inst.m):
        y = np.array(bits, dtype=bool)
        yield y, design_cost(inst, y)


def brute_force(inst: MCNDInstance) -> float:
    return min(c for _, c in all_designs(inst) if c is not None)


# ---------------------------------------------------------------- data

@pytest.mark.skipif(not (RAW / "C").exists(), reason="raw Canad instances not downloaded")
def test_loader_parses_every_raw_file():
    files = sorted((RAW / "C").glob("*.dow")) + sorted((RAW / "CPlus").glob("*.dow"))
    assert len(files) == 43
    for f in files:
        inst = load_dow(f)
        assert 20 <= inst.n <= 100 and 100 <= inst.m <= 700 and 10 <= inst.K <= 400
    c33 = load_dow(RAW / "C" / "c33.dow")
    assert (c33.n, c33.m, c33.K) == (20, 228, 39)


def test_write_and_load_roundtrip(tmp_path):
    inst = random_instance(np.random.default_rng(0), 5, 9, 3, tight=False)
    write_dow(inst, tmp_path / "a.dow")
    back = load_dow(tmp_path / "a.dow")
    for name in ("tail", "head", "cost", "cap", "fixed", "orig", "dest", "demand"):
        assert np.array_equal(getattr(inst, name), getattr(back, name))


# ---------------------------------------------------------------- cuts

def test_cuts_from_any_potentials_are_valid(tmp_path):
    """Every pi gives valid optimality and feasibility cuts (plain and strong coefficients) for all binary designs."""
    rng = np.random.default_rng(1)
    for seed in range(6):
        inst, _ = feasible_instance(100 + seed, tmp_path)
        designs = list(all_designs(inst))
        L = inst.link_bound()
        for _ in range(5):
            pi = rng.normal(0, 40, size=(inst.K, inst.n))
            for feas in (False, True):
                const = cut_constant(inst, pi).sum()
                G = arc_excess(inst, pi, 0.0 if feas else inst.cost)
                for h in (plain_cut(inst, pi, None, feas)[1], strong_arc_coefficients(G, inst.cap, L)):
                    for y, cost in designs:
                        if cost is None:
                            continue
                        flow = cost - inst.fixed[y].sum()
                        if feas:
                            assert h @ y >= const - 1e-6
                        else:
                            assert flow >= const - h @ y - 1e-6


def test_strong_coefficients_minimize_over_t():
    rng = np.random.default_rng(2)
    for _ in range(50):
        K, m = int(rng.integers(1, 6)), int(rng.integers(1, 5))
        G = np.maximum(rng.normal(0, 3, size=(K, m)), 0.0)
        u = rng.uniform(1, 20, size=m)
        L = np.minimum(rng.uniform(1, 15, size=(K, 1)), u[None, :])
        h = strong_arc_coefficients(G, u, L)
        for a in range(m):
            ts = np.concatenate([[0.0], G[:, a], np.linspace(0, G[:, a].max() + 1, 50)])
            vals = [u[a] * t + (L[:, a] * np.maximum(G[:, a] - t, 0)).sum() for t in ts]
            assert h[a] == pytest.approx(min(vals), abs=1e-9)


@pytest.mark.parametrize("strong", [False, True])
def test_recourse_cuts_are_tight_and_farkas_cuts_separate(tmp_path, strong):
    env = make_env(1, None)
    for seed in range(8):
        inst, _ = feasible_instance(200 + seed, tmp_path)
        designs = list(all_designs(inst))
        rec = RecourseLP(inst, env, strong=strong)
        L = np.where(rec.skip, 0.0, inst.link_bound()) if strong else None
        for y, cost in designs:
            res = rec.solve(y.astype(float), 10)
            feas = res["status"] == "infeasible"
            if strong:
                const = cut_constant(inst, res["pi"]).sum()
                h = strong_arc_coefficients(arc_excess(inst, res["pi"], 0.0 if feas else inst.cost, rec.skip), inst.cap, L)
            else:
                const, h = plain_cut(inst, res["pi"], res["alpha"], feas)
            if cost is None:
                assert feas
                assert const - h @ y > 1e-9  # the Farkas cut cuts the design off
            else:
                assert res["status"] == "optimal"
                assert res["obj"] == pytest.approx(cost - inst.fixed[y].sum(), abs=1e-6)
                assert const - h @ y == pytest.approx(res["obj"], abs=1e-6)  # tight at the design
        rec.dispose()
    env.dispose()


def test_reduced_recourse_matches_evaluator(tmp_path):
    env = make_env(1, None)
    for seed in range(5):
        inst, _ = feasible_instance(300 + seed, tmp_path)
        red = ReducedRecourse(inst, env)
        for y, cost in all_designs(inst):
            res = red.solve(y, 10)
            if cost is None:
                assert res["status"] in ("infeasible", "disconnected")
            else:
                assert res["status"] == "optimal"
                assert res["obj"] == pytest.approx(cost - inst.fixed[y].sum(), abs=1e-6)
    env.dispose()


def test_mir_rows_are_valid():
    rng = np.random.default_rng(3)
    for _ in range(300):
        k = int(rng.integers(1, 7))
        coef = rng.integers(1, 30, size=k).astype(float)
        rhs = float(rng.integers(1, int(coef.sum()) + 1))
        point = rng.random(k)
        row = mir_row(coef, rhs, point)
        if row is None:
            continue
        g, r, _ = row
        for bits in itertools.product([0, 1], repeat=k):
            yb = np.array(bits, dtype=float)
            if coef @ yb >= rhs:
                assert g @ yb >= r - 1e-9


def test_cutset_rows_are_valid_for_feasible_designs(tmp_path):
    rng = np.random.default_rng(4)
    for seed in range(6):
        inst, _ = feasible_instance(400 + seed, tmp_path)
        ref = MCNDBendersXL()
        ref.inst = inst
        feasible = [y for y, c in all_designs(inst) if c is not None]
        for _ in range(10):
            S = rng.random(inst.n) < 0.5
            if S.all() or not S.any():
                continue
            for arcs, coef, rhs, _ in ref.cutset_rows(S, rng.random(inst.m)):
                for y in feasible:
                    assert coef @ y[arcs] >= rhs - 1e-9


# ---------------------------------------------------------------- methods vs brute force

@pytest.mark.parametrize("seed", range(10))
def test_methods_find_the_optimum_with_valid_bounds(tmp_path, seed):
    inst, path = feasible_instance(seed, tmp_path)
    opt = brute_force(inst)
    model, _ = build_mcnd_mip(inst)
    model.Params.OutputFlag = 0
    model.optimize()
    assert model.ObjVal == pytest.approx(opt, abs=1e-6)
    variants = {"naive": MCNDNaiveBenders, "xl": MCNDBendersXL,
                "xl_branch_and_cut_only": functools.partial(MCNDBendersXL, root_rounds=0, ss_steps=0),
                "xl_aggregate_cuts_only": functools.partial(MCNDBendersXL, commodity_cuts=False, cutsets=False)}
    for name, cls in variants.items():
        seen, bounds = [], []
        res = cls(time_limit=30, on_incumbent=seen.append, on_bound=bounds.append).run(path)
        assert res["termination_reason"] == "optimal", name
        assert res["best_objective"] == pytest.approx(opt, abs=1e-6), name
        assert res["global_lb"] <= opt + 1e-6 and all(b <= opt + 1e-6 for b in bounds), name
        assert bounds == sorted(bounds), name
        assert seen, name
        for design in seen:  # every published incumbent passes the independent verifier
            out = design_output(inst, design, method=name)
            assert out is not None and out["objective_value"] >= opt - 1e-6
            (tmp_path / "out.json").write_text(json.dumps(out))
            assert verify_solution(path, tmp_path / "out.json")["ok"], name
        assert design_cost(inst, res["open"]) == pytest.approx(res["best_objective"], abs=1e-6)


# ---------------------------------------------------------------- verifier

def test_verifier_rejects_bad_outputs(tmp_path):
    inst, path = feasible_instance(7, tmp_path)
    y = np.ones(inst.m, dtype=bool)
    good = design_output(inst, y, method="test")
    f = tmp_path / "o.json"

    def check(out):
        f.write_text(json.dumps(out))
        return verify_solution(path, f)

    assert check(good)["ok"]
    dense = dict(good)
    flows = np.zeros((inst.K, inst.m))
    for k, a, v in good["flows"]:
        flows[k, a] = v
    dense.pop("flows")
    dense["x"] = flows.tolist()
    assert check(dense)["ok"]
    assert not check(good | {"objective_value": good["objective_value"] + 1})["ok"]
    assert not check(good | {"objective_value": float("inf")})["ok"]
    assert not check(good | {"objective_value": None})["ok"]
    assert not check({k: v for k, v in good.items() if k != "method"})["ok"]
    assert not check(good | {"status": "infeasible"})["ok"]
    assert not check(good | {"y": [2] * inst.m})["ok"]
    assert not check({k: v for k, v in good.items() if k != "flows"})["ok"]
    # Closing a used arc without rerouting violates its capacity.
    k, a, v = next(e for e in good["flows"] if e[2] > 1e-3)
    closed = list(good["y"])
    closed[a] = 0
    cost = good["objective_value"] - inst.fixed[a]
    assert not check(good | {"y": closed, "objective_value": cost})["ok"]
    # Dropping a flow entry breaks conservation.
    broken = [e for e in good["flows"] if e != [k, a, v]]
    cost = good["objective_value"] - inst.cost[a] * v
    assert not check(good | {"flows": broken, "objective_value": cost})["ok"]
    # Negative flows are rejected.
    assert not check(good | {"flows": good["flows"] + [[0, 0, -1.0], [0, 0, 1.0]] + [[0, 1, -0.5]]})["ok"]


# ---------------------------------------------------------------- deadlines on a real instance

@pytest.mark.skipif(not (RAW / "C" / "c33.dow").exists(), reason="raw Canad instances not downloaded")
@pytest.mark.parametrize("cls", [MCNDNaiveBenders, MCNDBendersXL])
def test_deadline_is_respected(cls):
    path = str(RAW / "C" / "c37.dow")
    inst = load_dow(path)
    seen, bounds = [], []
    res = cls(time_limit=3.0, on_incumbent=seen.append, on_bound=bounds.append).run(path)
    assert res["total_seconds"] < 4.5
    assert seen and res["best_objective"] is not None
    assert design_cost(inst, res["open"]) == pytest.approx(res["best_objective"], rel=1e-9)
    assert res["global_lb"] <= res["best_objective"]
    assert math.isfinite(res["global_lb"])


def test_callback_errors_are_not_swallowed(tmp_path):
    """gurobipy ignores exceptions raised in callbacks; the reference must re-raise them, never accept an unchecked design."""
    _, path = feasible_instance(4, tmp_path)  # a seed whose run reaches branch-and-cut (seed 3 closes at the root)

    class Failing(MCNDBendersXL):
        def branch_and_cut(self):
            def boom(*args, **kwargs):
                raise RuntimeError("recourse failure")
            self.reduced.solve = boom
            return super().branch_and_cut()

    with pytest.raises(RuntimeError, match="recourse failure"):
        Failing(time_limit=30, root_rounds=0, ss_steps=0).run(path)
