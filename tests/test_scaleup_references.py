"""Correctness checks for the scale-up references against brute force on tiny instances."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

gp = pytest.importorskip("gurobipy")

from src.bd.ufl.a_gen_data import UFLInstance
from src.bd.ufl.b_ref_bd_xl import UFLBendersXL, UFLCutSeparator, assignment_cost
from src.cg.gap.a_gen_data import gap_instance_from_arrays
from src.cg.gap.b_ref_cg_xl import GAPColumnGenerationXL, knapsack


def test_knapsack_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(1, 9))
        profit, weight, cap = rng.normal(size=n), rng.integers(1, 10, size=n), int(rng.integers(1, 30))
        best = max(sum(profit[i] for i in s) for r in range(n + 1) for s in itertools.combinations(range(n), r) if sum(weight[i] for i in s) <= cap)
        val, x = knapsack(profit, weight, cap)
        assert val == pytest.approx(max(best, 0.0))
        assert weight[x].sum() <= cap and profit[x].sum() == pytest.approx(val)


def test_ufl_cuts_are_valid_and_tight():
    rng = np.random.default_rng(2)
    D = rng.integers(1, 50, size=(5, 7)).astype(float)
    sep = UFLCutSeparator(D)
    for _ in range(30):
        y = rng.random(5)
        y[rng.integers(5)] = 1.0
        ck = sep.critical(y)
        for j in range(7):
            # LP value of customer j at y by greedy filling equals the cut at y (tightness).
            order, left, lp = np.argsort(D[:, j]), 1.0, 0.0
            for i in order:
                take = min(left, y[i])
                lp, left = lp + take * D[i, j], left - take
            assert sep.value(y)[j] == pytest.approx(lp)
            # The cut underestimates the LP value at every binary y (validity).
            for bits in itertools.product([0, 1], repeat=5):
                yb = np.array(bits, dtype=float)
                if yb.sum():
                    assert sep.rhs(ck, yb)[j] <= D[yb > 0.5, j].min() + 1e-9


def test_ufl_reference_finds_optimum(tmp_path):
    rng = np.random.default_rng(3)
    m, n = 6, 12
    inst = UFLInstance(m=m, n=n, Xc=np.zeros((n, 2)), Xf=np.zeros((m, 2)), f=rng.integers(5, 40, size=m).astype(float), D=rng.integers(1, 30, size=(m, n)).astype(float))
    inst.to_json_file(tmp_path / "data.json")
    opt = min(assignment_cost(inst.D, inst.f, np.array(b, dtype=float)) for b in itertools.product([0, 1], repeat=m) if any(b))
    res = UFLBendersXL(time_limit=30).run(str(tmp_path / "data.json"))
    assert res["best_objective"] == pytest.approx(opt)
    assert res["global_lb"] <= opt + 1e-6


def test_gap_reference_finds_optimum(tmp_path):
    rng = np.random.default_rng(4)
    m, n = 3, 8
    res_ = rng.integers(1, 10, size=(m, n))
    cost = (20 - res_ + rng.integers(-2, 3, size=(m, n))).astype(float)
    cap = (0.8 * res_.sum(axis=1) / m).astype(int) + 3
    gap_instance_from_arrays(cost.tolist(), res_.tolist(), cap.tolist()).to_json_file(tmp_path / "data.json")
    best = min((cost[list(a), range(n)].sum() for a in itertools.product(range(m), repeat=n)
                if all(res_[k, [i for i in range(n) if a[i] == k]].sum() <= cap[k] for k in range(m))), default=None)
    out = GAPColumnGenerationXL(time_limit=30).run(str(tmp_path / "data.json"))
    assert out["global_lb"] <= best + 1e-6
    assert out["best_objective"] is None or out["best_objective"] >= best - 1e-6


def test_recorder_rejects_late_publications_and_journals(tmp_path, monkeypatch):
    import json

    import src.experiments.scaleup_calibration as cal

    class Adapter:
        normalize = staticmethod(list)

        def objective(self, sol):
            return float(sum(sol))

    clock = {"t": 1.0}
    monkeypatch.setattr(cal, "elapsed", lambda: clock["t"])
    rec = cal.Recorder(Adapter(), 10.0, tmp_path / "j.jsonl")
    rec.solution([5, 5], "a")
    rec.bound(3.0, "a")
    clock["t"] = 10.0  # at the deadline: not eligible
    rec.solution([1], "late")
    rec.bound(9.0, "late")
    rec.journal.close()
    assert rec.best == 10.0 and rec.late == 1 and rec.bounds[-1]["bound"] == 3.0
    lines = [json.loads(line) for line in (tmp_path / "j.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in lines] == ["incumbent", "bound"]


def test_gap_bound_allowance_never_exceeds_true_value():
    from src.cg.gap.b_ref_cg_xl import certified_bound

    u = np.full(1600, 1 + 9e-10)
    # The exact Lagrangian value is 1600 (u.sum() minus a knapsack value of 1600 * 9e-10).
    lag = u.sum() - knapsack(u - 1.0, np.ones(1600, dtype=int), 1600)[0]
    assert certified_bound(lag, u, True) <= 1600


def test_replay_ignores_late_and_torn_records(tmp_path):
    import json

    import src.experiments.scaleup_calibration as cal

    class Adapter:
        def objective(self, sol):
            return float(sum(sol))

    lines = [json.dumps({"kind": "incumbent", "t": 1.0, "objective": 10.0, "solution": [10], "source": "a"}),
             json.dumps({"kind": "incumbent", "t": 2.0, "objective": 7.0, "solution": [3], "source": "a"}),  # objective mismatch: rejected
             json.dumps({"kind": "incumbent", "t": 11.0, "objective": 1.0, "solution": [1], "source": "a"}),  # after the deadline
             json.dumps({"kind": "bound", "t": 3.0, "bound": 2.0, "source": "a"}), '{"kind": "incum']  # torn final line
    (tmp_path / "j.jsonl").write_text("\n".join(lines))
    rec = cal.replay(tmp_path / "j.jsonl", Adapter(), 10.0)
    assert rec.best == 10.0 and len(rec.witnesses) == 1 and rec.bounds[-1]["bound"] == 2.0


def test_gap_lagrangian_bounds_and_neighborhood_search_are_valid():
    import time

    from src.cg.gap.b_ref_cg_xl import capacity_lagrangian, certified_bound, penalty_search, regret_assign, sub_gap

    rng = np.random.default_rng(5)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    for _ in range(6):
        m, n = 3, 7
        res_ = rng.integers(1, 10, size=(m, n))
        cost = (20 - res_ + rng.integers(-2, 3, size=(m, n))).astype(float)
        cap = (0.8 * res_.sum(axis=1) / m).astype(int) + 3
        feasible = [np.array(a) for a in itertools.product(range(m), repeat=n) if all(res_[k, np.array(a) == k].sum() <= cap[k] for k in range(m))]
        opt = min(cost[a, np.arange(n)].sum() for a in feasible)
        values = []
        mu = capacity_lagrangian(cost, res_, cap, env, time.perf_counter() + 5, lambda g, terms: values.append(certified_bound(g, terms, True)))
        assert max(values) <= opt
        found = []
        penalty_search(regret_assign(cost + mu[:, None] * res_, res_, cap), cost, res_, cap, np.full(m, 2.0), time.perf_counter() + 5, found.append)
        for a in found:
            assert all(res_[k, a == k].sum() <= cap[k] for k in range(m))
        start = max(feasible, key=lambda a: cost[a, np.arange(n)].sum())
        sub = sub_gap(cost, res_, cap, start, (cost + mu[:, None] * res_).min(axis=0), env, time.perf_counter() + 10, True)
        if sub is not None:
            assert all(res_[k, sub == k].sum() <= cap[k] for k in range(m))
            assert opt <= cost[sub, np.arange(n)].sum() < cost[start, np.arange(n)].sum()
    env.dispose()


def test_preview_keeps_structure_and_reports_lengths():
    from src.experiments.xl_inputs import preview

    data = {"m": 4, "f": [1, 2, 3, 4], "D": [[1, 2, 3, 4, 5]] * 4, "name": "x"}
    p = preview(data, keep=2)
    assert p["m"] == 4 and p["name"] == "x"
    assert p["f"] == [1, 2, "... (4 items in total)"]
    assert p["D"] == [[1, 2, "... (5 items in total)"], [1, 2, "... (5 items in total)"], "... (4 items in total)"]


def test_deferred_publications_keep_their_times_and_replay(tmp_path, monkeypatch):
    import json

    import src.experiments.scaleup_calibration as cal

    class Deferred:
        deferred = True
        normalize = staticmethod(lambda s: sorted(s))
        calls = 0

        def objective(self, sol):
            Deferred.calls += 1
            return None if 99 in sol else float(sum(sol))

    clock = {"t": 1.0}
    monkeypatch.setattr(cal, "elapsed", lambda: clock["t"])
    rec = cal.Recorder(Deferred(), 10.0, tmp_path / "j.jsonl")
    rec.solution([5, 5], "a")
    clock["t"] = 2.0
    rec.solution([99], "a")  # infeasible
    clock["t"] = 3.0
    rec.solution([1, 2], "a")
    assert Deferred.calls == 0 and not rec.witnesses  # nothing evaluated while the solver runs
    rec.evaluate_pending()
    rec.journal.close()
    assert [(w["t"], w["objective"]) for w in rec.witnesses] == [(1.0, 10.0), (3.0, 3.0)] and rec.infeasible == 1
    again = cal.replay(tmp_path / "j.jsonl", Deferred(), 10.0)
    assert [(w["t"], w["objective"]) for w in again.witnesses] == [(1.0, 10.0), (3.0, 3.0)]
