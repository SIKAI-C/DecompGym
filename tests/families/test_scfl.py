"""Small-scale SCFL family: verifier, output contract and the reference/monolithic agreement on a hand-solved instance."""
from __future__ import annotations

import json

import pytest

from src.experiments.verification import OutputVerifier
from src.utils.output_contract import validate_output_contract


# Both facilities are needed (capacity 10 each, scenario demand 12). Scenario 0: each customer from its cheap
# facility, 6 * 1 + 6 * 2 = 18; scenario 1: customer 0 takes 10 from facility 0 and 2 from facility 1, 10 + 8 = 18.
TINY = {
    "n_facilities": 2, "n_customers": 2, "n_scenarios": 2,
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
# is already the optimum 10000.01990 (facility 0 alone), a relative gap of 5e-9. A loose relative stopping tolerance
# returned the worse design; the 1e-9 certificate must not.
NEAR_TIE = {
    "n_facilities": 2, "n_customers": 1, "n_scenarios": 20,
    "capacity": [2, 1], "fixed_cost": [10000.0, 0.01], "unit_cost": [[0.199], [0.0]],
    "demand": [[2]] + [[0]] * 19, "probability": [1 / 20] * 20,
}


def test_scfl_certificates():
    import math

    from src.bd.scfl.a_gen_data import SCFLInstance, generate_scfl_instance
    from src.bd.scfl.a_gold_monolithic import certificate, lattice_bound, objective_step

    step = objective_step(SCFLInstance(**NEAR_TIE))
    assert step == pytest.approx(0.001 / 20)
    assert certificate(10000.01995, 10000.0199, step) is None  # one lattice value between: not proved
    assert certificate(10000.0199, 10000.0199 - 1e-12, step) == "lattice"
    assert certificate(10000.0199, 10000.0199 - 0.4 * step, step) == "lattice"
    assert lattice_bound(10000.0199 + 0.005 * step, step) == pytest.approx(10000.0199)
    assert certificate(1e6, 1e6 - 9e-4, None) == "relative" and certificate(1e6, 1e6 - 2e-3, None) is None
    for inc, bnd in ((1e6, 1e6 + 100), (math.inf, 1.0), (1.0, math.nan), (math.nan, 1.0)):  # crossed or non-finite
        assert certificate(inc, bnd, step) is None and certificate(inc, bnd, None) is None
    assert certificate(1e6, 1e6 + 5e-4, None) == "relative"  # roundoff crossing within the tolerance
    assert certificate(1e6, 1e6 + 0.00075, 0.00025) == "relative"  # crossed by 3 steps: never a lattice proof
    assert objective_step(SCFLInstance(**{**NEAR_TIE, "unit_cost": [[0.1995], [0.0]]})) is None
    # CAP-derived instances: unit costs are multiples of 1/80 up to the double conversion, so the step is 1/(80 S).
    for size in ("small", "large"):
        d = generate_scfl_instance(0, size)
        assert objective_step(SCFLInstance(**d)) == pytest.approx(1 / (80 * d["n_scenarios"]))


def test_scfl_generator_follows_the_cap_recipe():
    import numpy as np

    from src.bd.scfl.a_data import read_orlib_cap
    from src.bd.scfl.a_gen_data import CAP_BASES, CAPACITATED_BASES, ORLIB_DIR, SIZE_CONFIG, generate_scfl_instance, load_cap_base

    assert len(CAP_BASES) == 16 and all(load_cap_base(b)["name"] == b for b in CAP_BASES)  # pinned hashes
    assert CAPACITATED_BASES == [f"cap{k}" for k in (111, 112, 113, 114, 121, 122, 123, 124)]
    for size in SIZE_CONFIG:
        for seed in (0, 1):
            d = generate_scfl_instance(seed, size)
            assert d == generate_scfl_instance(seed, size)
            S = SIZE_CONFIG[size]["n_scenarios"]
            assert (d["n_facilities"], d["n_customers"], d["n_scenarios"]) == (50, 50, S) and d["probability"] == [1.0 / S] * S
            base = read_orlib_cap(ORLIB_DIR / f"{CAPACITATED_BASES[int(np.random.default_rng([seed, 0x5CF1]).integers(8))]}.txt")
            assert np.array_equal(np.array(d["unit_cost"]), base["alloc_cost"] / base["demand"][None, :])  # unrounded
            assert d["fixed_cost"] == base["fixed"].tolist() and d["capacity"] == base["capacity"].tolist()
            rng = np.random.default_rng(seed)  # Bodur et al. stream: sigma, then the S x J normal demands
            sig = rng.uniform(0.1 * base["demand"], 0.3 * base["demand"])
            raw = rng.normal(base["demand"][None, :], sig[None, :], size=(S, 50))
            assert np.array_equal(np.array(d["demand"]), np.maximum(np.rint(raw), 0.0))


def test_scfl_interrupted_reference_is_not_labelled_optimal(tmp_path, monkeypatch):
    from src.bd.scfl import a_gold_bd

    path = str(_write(tmp_path, "data.json", NEAR_TIE))
    calls = {"n": 0}
    real = a_gold_bd.SCFLBenders._evaluate

    def evaluate_then_expire(self, *args):
        calls["n"] += 1
        if calls["n"] == 2:  # the budget runs out while evaluating the master's (optimal) design
            raise RuntimeError("time limit budget exhausted")
        return real(self, *args)

    monkeypatch.setattr(a_gold_bd.SCFLBenders, "_evaluate", evaluate_then_expire)
    ref = a_gold_bd.SCFLBenders(time_limit=20).run(path)
    assert ref["y"] == [1, 1] and ref["status"] == "feasible" and ref["proven_optimal"] is False


def test_scfl_reference_does_not_stop_on_a_near_tie(tmp_path):
    from src.bd.scfl.a_gold_bd import SCFLBenders
    from src.bd.scfl.a_gold_monolithic import SCFLBaseline

    path = str(_write(tmp_path, "data.json", NEAR_TIE))
    ref = SCFLBenders(time_limit=20).run(path)
    mono = SCFLBaseline().run(path)
    assert ref["y"] == [1, 0] and ref["best_objective"] == pytest.approx(10000.0199, abs=1e-9) and ref["status"] == "optimal"
    assert mono["best_objective"] == pytest.approx(10000.0199, abs=1e-9)


# Codex review counterexample (2026-11-05): with the plain cover u.y >= D the root LP picks y = 0.5, where the
# strong recourse x <= d y cannot serve the demand; the strengthened cover sum_i min(u_i, D) y_i >= D cannot.
OVERSIZED = {
    "n_facilities": 1, "n_customers": 1, "n_scenarios": 1,
    "capacity": [2], "fixed_cost": [1.0], "unit_cost": [[0.0]], "demand": [[1]], "probability": [1.0],
}


def test_scfl_reference_handles_a_facility_larger_than_the_demand(tmp_path):
    from src.bd.scfl.a_gold_bd import SCFLBenders

    ref = SCFLBenders(time_limit=20).run(str(_write(tmp_path, "data.json", OVERSIZED)))
    assert ref["y"] == [1] and ref["best_objective"] == pytest.approx(1.0) and ref["status"] == "optimal"
    assert ref["root_rounds"] >= 1  # the fractional root point admits a strong recourse solution


@pytest.mark.parametrize("stage", ["master", "subproblem"])
def test_scfl_root_timeout_hands_over_to_the_integer_loop(tmp_path, monkeypatch, stage):
    from src.bd.scfl import a_gold_bd
    from src.utils.reference_runtime import ReferenceLimit

    if stage == "subproblem":
        def expire(self, *args, **kwargs):
            raise ReferenceLimit("root deadline")
        monkeypatch.setattr(a_gold_bd.SCFLBenders, "_evaluate_fractional", expire)
    else:
        real = a_gold_bd.ReferenceRuntime.optimize

        def expire_master_lp(self, model, *, role, allow_incumbent=False):
            if role == "master_lp":
                raise ReferenceLimit("root deadline")
            return real(self, model, role=role, allow_incumbent=allow_incumbent)
        monkeypatch.setattr(a_gold_bd.ReferenceRuntime, "optimize", expire_master_lp)
    ref = a_gold_bd.SCFLBenders(time_limit=20).run(str(_write(tmp_path, "data.json", TINY)))
    assert ref["root_rounds"] == 0 and ref["iters"] >= 1
    assert ref["best_objective"] == pytest.approx(TINY_OPT) and ref["y"] == [1, 1] and ref["status"] == "optimal"


def test_scfl_expected_value_design_survives_an_exhausted_budget(tmp_path, monkeypatch):
    from src.bd.scfl import a_gold_bd
    from src.utils.reference_runtime import ReferenceLimit

    real = a_gold_bd.SCFLBenders._evaluate

    def expire_on_the_stochastic_evaluation(self, inst, sub, y, demands):
        if len(demands) > 1:
            raise ReferenceLimit("budget exhausted")
        return real(self, inst, sub, y, demands)

    monkeypatch.setattr(a_gold_bd.SCFLBenders, "_evaluate", expire_on_the_stochastic_evaluation)
    ref = a_gold_bd.SCFLBenders(time_limit=20, expected_value=True).run(str(_write(tmp_path, "data.json", TINY)))
    assert ref["best_objective"] == pytest.approx(TINY_OPT) and ref["global_lb"] is None and ref["status"] == "feasible"
