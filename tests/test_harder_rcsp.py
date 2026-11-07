"""Railway crew scheduling (small-scale RCSP): verifier, generator, pricing and reference checks."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.utils.output_contract import validate_output_contract

# Two stations, crew base 0. Task rows: [start, end, start station, end station, duration].
TASKS = [[21600, 25200, 0, 1, 3600], [27600, 31200, 1, 0, 3600], [32400, 36000, 0, 1, 3600], [36600, 40200, 1, 0, 3600]]
CONNECTIONS = [
    [0, "SOURCE", 0, 0, -1, -1, 3600, 32400],
    [1, "SOURCE", 0, 2, -1, -1, 3600, 32400],
    [2, "CONNECTION", 0, 1, 300, 3900, 6000, 6000],  # 40-min gap: a break connection
    [3, "CONNECTION", 1, 2, -1, -1, 4800, 4800],
    [4, "CONNECTION", 2, 3, -1, -1, 4200, 4200],
    [5, "SINK", 1, 0, -1, -1, 0, 0],
    [6, "SINK", 3, 0, -1, -1, 0, 0],
]
DATA = {"n_tasks": 4, "tasks": TASKS, "stations": [0, 1], "crew_bases": [0], "connections": CONNECTIONS}


def _out(duties, objective=0):
    return {"status": "feasible", "method": "column_generation", "objective_value": objective, "solution": {"duties": duties}}


def test_rcsp_output_contract():
    assert validate_output_contract("cg", "rcsp", {}, _out([{"depot": 0, "tasks": [0, 1]}])) == []
    assert validate_output_contract("cg", "rcsp", {}, _out([[0, 1]])) != []
    assert validate_output_contract("cg", "rcsp", {}, _out([{"depot": 0, "tasks": [0.5]}])) != []


def test_rcsp_verifier_costs_and_rules():
    from src.cg.rcsp.d_verifier import verify_solution

    # One duty over all four tasks: 32 400 + 6 000 + 4 800 + 4 200 + 0 (break on 0 -> 1).
    assert verify_solution(DATA, _out([{"depot": 0, "tasks": [0, 1, 2, 3]}])) == {"ok": True, "issues": [], "objective_value": 47400.0}
    # [2, 3] has no break connection.
    assert not verify_solution(DATA, _out([{"depot": 0, "tasks": [0, 1]}, {"depot": 0, "tasks": [2, 3]}]))["ok"]
    # Uncovered task, missing connection, wrong base.
    assert "not covered" in verify_solution(DATA, _out([{"depot": 0, "tasks": [0, 1]}]))["issues"][0]
    assert "no connection" in verify_solution(DATA, _out([{"depot": 0, "tasks": [0, 2, 3]}]))["issues"][0]
    assert "crew base" in verify_solution(DATA, _out([{"depot": 1, "tasks": [0, 1, 2, 3]}]))["issues"][0]


def test_rcsp_labour_rules_on_chains():
    from src.cg.rcsp.a_gen_data import duty_cost
    from src.cg.rcsp.d_verifier import chain_cost

    src = [0, "SOURCE", 0, 0, -1, -1, 3600, 32400]
    snk = [9, "SINK", 1, 0, -1, -1, 0, 0]
    ok = [src, [1, "CONNECTION", 0, 1, 300, 3900, 6000, 6000], snk]
    long = [src, [1, "CONNECTION", 0, 1, 300, 3900, 6000, 6000], [2, "CONNECTION", 1, 2, -1, -1, 25000, 25000], snk]
    late_break = [[0, "SOURCE", 0, 0, -1, -1, 19600, 48400], [1, "CONNECTION", 0, 1, 300, 3900, 6000, 6000], snk]
    long_after = [src, [1, "CONNECTION", 0, 1, 300, 3900, 6000, 6000], [2, "CONNECTION", 1, 2, -1, -1, 16000, 16000], snk]
    for check in (duty_cost, lambda c: chain_cost(c)[0]):
        assert check(ok) == 38400
        assert check(long) is None  # 34 600 s > 9.5 h
        assert check(late_break) is None  # break would start after 5.5 h
        assert check(long_after) is None  # 3 900 + 16 000 s after the break


@pytest.fixture(scope="module")
def generated():
    from src.cg.rcsp.a_gen_data import RCSPInstance, generate_rcsp_instance

    data = [generate_rcsp_instance(seed, size) for size in ("small", "medium") for seed in range(3)]
    # Only the model data reach the agent: no labels, parent name, window or original indices.
    assert all(set(d) == {"n_tasks", "tasks", "stations", "crew_bases", "connections"} for d in data)
    return [RCSPInstance.from_dict(d) for d in data]


def test_rcsp_generator_windows_coverage_and_parents(generated):
    from src.cg.rcsp.a_gen_data import DEV_PARENTS, EVAL_PARENTS, SIZE_CONFIG, RCSPGenerator, enumerate_duties

    for inst in generated:
        assert inst.n_tasks > 0
        assert {t for _, path, _ in enumerate_duties(inst) for t in path} == set(range(inst.n_tasks))
    for size, cfg in SIZE_CONFIG.items():
        for seed in (0, 1, 2, 100, 101):  # evaluation and development parents (generation only, no solve)
            inst, prov = RCSPGenerator().draw(seed, size)
            assert prov["benchmark"] in (DEV_PARENTS if seed >= 100 else EVAL_PARENTS)
            assert prov["variant"] == ("multi" if seed % 2 else "single") and (len(inst.crew_bases) == 1) == (prov["variant"] == "single")
            lo, hi = prov["window"]
            assert hi - lo == round(cfg["window_hours"] * 3600) - prov["window_shortened_seconds"]
            assert prov["window_shortened_seconds"] % 900 == 0 and prov["n_duties"] <= cfg["max_duties"]
            assert all(lo <= t[0] and t[1] <= hi for t in inst.tasks)
    assert not set(DEV_PARENTS) & set(EVAL_PARENTS)


def test_rcsp_labelling_matches_enumeration(generated):
    """Without dominance the labelling returns exactly the enumerated duties below a threshold, with their reduced
    costs; with dominance its minimum reduced cost per call is the true minimum."""
    from src.cg.rcsp.a_gen_data import enumerate_duties
    from src.cg.rcsp.a_gold_cg import DutyPricer

    rng = np.random.default_rng(0)
    for inst in generated:
        duties = enumerate_duties(inst)
        pricer = DutyPricer(inst)
        for _ in range(3):
            pi = rng.uniform(0, 40000, inst.n_tasks)
            rc = {(b, p): c - pi[list(p)].sum() for b, p, c in duties}
            thr = float(np.quantile(list(rc.values()), 0.7))
            got = {col: r for r, col in pricer.price(pi, thr, enumerate_all=True)}
            assert set(got) == {k for k, v in rc.items() if v <= thr}
            assert all(abs(got[k] - rc[k]) < 1e-6 for k in got)
            best = pricer.price(pi, 1e18)
            assert abs(best[0][0] - min(rc.values())) < 1e-6


def test_rcsp_reference_matches_the_mip(tmp_path):
    from src.cg.rcsp.a_gen_data import generate_rcsp_instance
    from src.cg.rcsp.a_gold_cg import RCSPColumnGeneration
    from src.cg.rcsp.a_gold_monolithic import RCSPBaseline
    from src.cg.rcsp.d_verifier import verify_solution

    pytest.importorskip("gurobipy")
    generate_rcsp_instance(102, "small", tmp_path)  # a development parent
    path = str(tmp_path / "data.json")
    mono = RCSPBaseline().run(path)
    ref = RCSPColumnGeneration(time_limit=30).run(path)
    assert ref["termination_reason"] == "optimal" and ref["global_lb"] == ref["best_objective"] == mono["best_objective"]
    assert verify_solution(path, _out(ref["duties"]))["objective_value"] == ref["best_objective"]
    assert verify_solution(path, _out(mono["master_metadata"]["duties"]))["objective_value"] == mono["best_objective"]


def test_rcsp_reference_rejects_an_empty_pricing_batch():
    from src.cg.rcsp.a_gold_cg import RCSPColumnGeneration

    with pytest.raises(ValueError):
        RCSPColumnGeneration(batch=0)


def test_rcsp_prompt_renders():
    from src.cg.rcsp.b_problem import render_prompt

    text = render_prompt(json.loads(json.dumps(DATA)))
    assert "34200" in text and "19800" in text and '"duties"' in text
