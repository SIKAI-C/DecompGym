"""Planning and scheduling with a cumulative resource (bd/lbbd_ps): generator, verifier, contract and reference."""
from __future__ import annotations

import itertools
import json

import pytest

from src.utils.output_contract import validate_output_contract


# Two facilities, three jobs, capacity 4; job 0 and job 1 (rate 3 each) cannot run together on facility 0.
DATA = {
    "instance_size": "small", "kind": "c", "n_jobs": 3, "n_facilities": 2, "capacity": [4, 4],
    "release": [0, 0, 1], "deadline": [4, 4, 4],
    "proc": [[2, 2, 1], [3, 3, 3]], "rate": [[3, 3, 1], [2, 2, 2]], "cost": [[1, 1, 1], [5, 5, 5]],
}


def _out(schedule, objective=None):
    return {"status": "feasible", "method": "benders_decomposition", "objective_value": objective, "solution": {"schedule": schedule}}


def test_generator_matches_the_scaleup_recipe():
    from src.bd.lbbd_ps.a_gen_data import generate_lbbd_ps_instance

    c = generate_lbbd_ps_instance(0, "medium")
    e = generate_lbbd_ps_instance(1, "large")
    assert (c["kind"], c["n_jobs"], c["n_facilities"]) == ("c", 15, 3)
    assert c["deadline"] == [33] * 15 and c["capacity"] == [10] * 3  # ceil(1.65 * 4 * 15 / 3) = 33
    assert all(3 - i <= p < 10 * (3 - i) for i in range(3) for p in c["proc"][i])
    assert all(10 * (i + 1) <= f < 20 * (i + 1) for i in range(3) for f in c["cost"][i])
    assert (e["kind"], e["deadline"][0]) == ("e", 33) and all(row == e["rate"][0] for row in e["rate"])
    assert generate_lbbd_ps_instance(1, "large") == e


def test_verifier_accepts_back_to_back_jobs_and_charges_assignment_costs():
    from src.bd.lbbd_ps.d_verifier import verify_solution

    plan = [{"job": 0, "facility": 0, "start": 0}, {"job": 1, "facility": 0, "start": 2}, {"job": 2, "facility": 0, "start": 1}]
    res = verify_solution(DATA, _out(plan))
    assert res["ok"] and res["objective_value"] == 3.0


@pytest.mark.parametrize("plan,reason", [
    ([{"job": 0, "facility": 0, "start": 0}, {"job": 1, "facility": 0, "start": 1}, {"job": 2, "facility": 1, "start": 1}], "exceeds capacity"),
    ([{"job": 0, "facility": 0, "start": 0}, {"job": 1, "facility": 0, "start": 2}, {"job": 2, "facility": 0, "start": 0}], "before its release"),
    ([{"job": 0, "facility": 1, "start": 2}, {"job": 1, "facility": 0, "start": 0}, {"job": 2, "facility": 0, "start": 3}], "after its deadline"),
    ([{"job": 0, "facility": 0, "start": 0.5}, {"job": 1, "facility": 0, "start": 2}, {"job": 2, "facility": 0, "start": 1}], "integer"),
    ([{"job": 0, "facility": 0, "start": 0}, {"job": 0, "facility": 0, "start": 2}, {"job": 2, "facility": 0, "start": 1}], "more than once"),
    ([{"job": 0, "facility": 0, "start": 0}, {"job": 2, "facility": 0, "start": 1}], "not scheduled"),
    ([{"job": 0, "facility": 0, "start": 0}, {"job": 1, "facility": 0, "start": 2}, {"job": 2, "facility": 2, "start": 1}], "out of range"),
    ([{"job": 0, "facility": 0, "start": 1e308}, {"job": 1, "facility": 0, "start": 2}, {"job": 2, "facility": 0, "start": 1}], "after its deadline"),
])
def test_verifier_rejects_infeasible_schedules(plan, reason):
    from src.bd.lbbd_ps.d_verifier import verify_solution

    res = verify_solution(DATA, _out(plan))
    assert not res["ok"] and reason in res["issues"][0]


def test_output_contract_requires_integer_schedule_entries():
    good = _out([{"job": 0, "facility": 0, "start": 0}], 1.0)
    assert validate_output_contract("bd", "lbbd_ps", DATA, good) == []
    assert validate_output_contract("bd", "lbbd_ps", DATA, _out([{"job": 0, "facility": 0, "start": 0.5}], 1.0)) != []
    assert validate_output_contract("bd", "lbbd_ps", DATA, {**good, "method": "column_generation"}) != []
    assert validate_output_contract("bd", "lbbd_ps", DATA, {**good, "solution": {"assignments": []}}) != []


def test_shared_verifier_rejects_a_wrong_claimed_objective(tmp_path):
    from src.experiments.verification import OutputVerifier

    plan = [{"job": 0, "facility": 0, "start": 0}, {"job": 1, "facility": 0, "start": 2}, {"job": 2, "facility": 0, "start": 1}]
    (tmp_path / "data.json").write_text(json.dumps(DATA))
    for claim, ok in ((3.0, True), (2.0, False)):
        (tmp_path / "output.json").write_text(json.dumps(_out(plan, claim)))
        assert OutputVerifier.verify_output("bd", "lbbd_ps", tmp_path / "data.json", tmp_path / "output.json", 1e-6)["ok"] is ok


def _brute_force(data):
    """Cheapest assignment whose facilities all have a feasible integer schedule (enumeration of starts)."""
    from src.bd.lbbd_ps.d_verifier import verify_solution

    n, m = data["n_jobs"], data["n_facilities"]
    best = None
    for assign in itertools.product(range(m), repeat=n):
        windows = [range(data["release"][j], data["deadline"][j] - data["proc"][assign[j]][j] + 1) for j in range(n)]
        for starts in itertools.product(*windows):
            plan = [{"job": j, "facility": assign[j], "start": starts[j]} for j in range(n)]
            res = verify_solution(data, _out(plan))
            if res["ok"]:
                best = res["objective_value"] if best is None else min(best, res["objective_value"])
                break
    return best


def test_reference_mip_and_shortcut_match_brute_force(tmp_path):
    from src.bd.lbbd_ps.a_gold_bd import PSBenders
    from src.bd.lbbd_ps.a_gold_monolithic import PSBaseline

    path = tmp_path / "data.json"
    path.write_text(json.dumps(DATA))
    opt = _brute_force(DATA)
    assert opt == 3.0  # everything on the cheap facility: 0 and 1 back to back, 2 alongside
    assert PSBaseline().run(str(path))["best_objective"] == opt
    ref = PSBenders(time_limit=20).run(str(path))
    assert ref["best_objective"] == opt and ref["termination_reason"] == "optimal" and ref["global_lb"] == opt
    tight = {**DATA, "deadline": [3, 3, 3]}  # 0 and 1 no longer fit together on facility 0 (core {0, 1})
    path.write_text(json.dumps(tight))
    opt = _brute_force(tight)
    assert PSBaseline().run(str(path))["best_objective"] == opt
    for res in (PSBenders(time_limit=20).run(str(path)), PSBenders(time_limit=20).run_outer_loop(str(path))):
        assert res["best_objective"] == opt and res["termination_reason"] == "optimal"


def test_deletion_filter_returns_an_irreducible_core():
    from src.bd.lbbd_ps.a_gen_data import PSInstance
    from src.bd.lbbd_ps.a_gold_bd import INFEASIBLE, FacilityOracle, bits
    from src.utils.reference_runtime import ReferenceRuntime

    inst = PSInstance(**{**DATA, "deadline": [3, 3, 3]})
    runtime = ReferenceRuntime(20)
    runtime.start()
    oracle = FacilityOracle(inst, runtime)
    core = oracle.core(0, 0b111)
    assert bits(core) == [0, 1]
    assert oracle.check(0, core)[0] == INFEASIBLE
    assert all(oracle.check(0, core & ~(1 << j))[0] != INFEASIBLE for j in bits(core))
