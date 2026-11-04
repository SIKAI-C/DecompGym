"""Small-scale SCFL family: verifier, output contract and the reference/monolithic agreement on a hand-solved instance."""
from __future__ import annotations

import json

import pytest

from src.experiments.verification import OutputVerifier
from src.utils.output_contract import validate_output_contract


# Both facilities are needed (capacity 10 each, scenario demand 12). Scenario 0: each customer from its cheap
# facility, 6 * 1 + 6 * 2 = 18; scenario 1: customer 0 takes 10 from facility 0 and 2 from facility 1, 10 + 8 = 18.
TINY = {
    "instance_size": "small", "n_facilities": 2, "n_customers": 2, "n_scenarios": 2,
    "capacity": [10, 10], "fixed_cost": [100.0, 50.0], "unit_cost": [[1.0, 5.0], [4.0, 2.0]],
    "demand": [[6, 6], [12, 0]], "probability": [0.5, 0.5],
}
TINY_OPT = 150.0 + 18.0


def _write(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


def _output(y, objective):
    return {"status": "optimal", "method": "benders_decomposition", "objective_value": objective, "y": y}


def test_scfl_verifier_recomputes_the_capacitated_recourse():
    from src.bd.scfl.d_verifier import verify_solution

    result = verify_solution(TINY, {"y": [1, 1]})
    assert result["ok"], result
    assert result["objective_value"] == pytest.approx(TINY_OPT)
    assert not verify_solution(TINY, {"y": [1, 0]})["ok"]  # open capacity 10 < demand 12
    assert not verify_solution(TINY, {"y": [1, 2]})["ok"]
    assert not verify_solution(TINY, {"y": [1]})["ok"]


def test_scfl_shared_verifier_and_contract(tmp_path):
    data = _write(tmp_path, "data.json", TINY)
    good = OutputVerifier.verify_output("bd", "scfl", data, _write(tmp_path, "good.json", _output([1, 1], TINY_OPT)), 1e-6)
    assert good["ok"], good
    bad = OutputVerifier.verify_output("bd", "scfl", data, _write(tmp_path, "bad.json", _output([1, 1], TINY_OPT - 1)), 1e-6)
    assert not bad["ok"] and any("objective mismatch" in issue for issue in bad["issues"])
    assert validate_output_contract("bd", "scfl", TINY, _output([1], TINY_OPT))
    assert not validate_output_contract("bd", "scfl", TINY, _output([1, 1], TINY_OPT))


def test_scfl_reference_output_uses_the_design():
    from src.utils.reference_output import reference_output

    mono = reference_output("bd", "scfl", TINY, {"best_objective": TINY_OPT, "master_metadata": {"y": [1, 1]}})
    assert mono["y"] == [1, 1] and mono["method"] == "benders_decomposition"


@pytest.mark.parametrize("kwargs", [{}, {"multi_cut": False}])
def test_scfl_reference_and_mip_agree_on_the_tiny_instance(tmp_path, kwargs):
    from src.bd.scfl.a_gold_bd import SCFLBenders
    from src.bd.scfl.a_gold_monolithic import SCFLBaseline

    path = str(_write(tmp_path, "data.json", TINY))
    mono = SCFLBaseline().run(path)
    ref = SCFLBenders(time_limit=20, **kwargs).run(path)
    assert mono["best_objective"] == pytest.approx(TINY_OPT)
    assert ref["best_objective"] == pytest.approx(TINY_OPT) and ref["y"] == [1, 1] and ref["status"] == "optimal"


# Codex review counterexample (2026-11-04): opening both facilities costs 10000.01995 and the master's first bound
# is already the optimum 10000.01990 (facility 0 alone), a relative gap of 5e-9. A relative stopping tolerance
# returned the worse design; the lattice certificate (values are multiples of 0.001 / S) must not.
NEAR_TIE = {
    "instance_size": "small", "n_facilities": 2, "n_customers": 1, "n_scenarios": 20,
    "capacity": [2, 1], "fixed_cost": [10000.0, 0.01], "unit_cost": [[0.199], [0.0]],
    "demand": [[2]] + [[0]] * 19, "probability": [1 / 20] * 20,
}


def test_scfl_lattice_certificate():
    from src.bd.scfl.a_gen_data import SCFLInstance
    from src.bd.scfl.a_gold_monolithic import certified, lattice_bound, objective_step

    step = objective_step(SCFLInstance(**NEAR_TIE))
    assert step == pytest.approx(0.001 / 20)
    assert not certified(10000.01995, 10000.0199, step)
    assert certified(10000.0199, 10000.0199 - 1e-12, step) and certified(10000.0199, 10000.0199 - 0.9 * step, step)
    assert lattice_bound(10000.0199 - 0.5 * step, step) == pytest.approx(10000.0199)
    with pytest.raises(ValueError):
        objective_step(SCFLInstance(**{**NEAR_TIE, "unit_cost": [[0.1995], [0.0]]}))


def test_scfl_reference_does_not_stop_on_a_near_tie(tmp_path):
    from src.bd.scfl.a_gold_bd import SCFLBenders
    from src.bd.scfl.a_gold_monolithic import SCFLBaseline

    path = str(_write(tmp_path, "data.json", NEAR_TIE))
    ref = SCFLBenders(time_limit=20).run(path)
    mono = SCFLBaseline().run(path)
    assert ref["y"] == [1, 0] and ref["best_objective"] == pytest.approx(10000.0199, abs=1e-9) and ref["status"] == "optimal"
    assert mono["best_objective"] == pytest.approx(10000.0199, abs=1e-9)
