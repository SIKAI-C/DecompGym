"""Regression tests for the harder-family cohort (Codex review 4 findings)."""
from __future__ import annotations

import json

import pytest

from src.experiments.verification import OutputVerifier
from src.utils.output_contract import validate_output_contract


# Codex review 4, finding 4: arrival 118.00000000000001 > 118 in floating point rejected the optimal PDPTW route.
PDPTW_EDGE = {
    "instance_size": "small", "n_requests": 2, "capacity": 60,
    "x": [50.0, 51.0, 52.0, 53.0, 51.0], "y": [50.0, 51.0, 52.0, 53.0, 52.0],
    "demand": [0, 10, 10, -10, -10],
    "ready": [0.0, 83.0, 47.0, 96.0, 58.0], "due": [600.0, 143.0, 107.0, 156.0, 118.0],
    "service": [0.0, 10.0, 10.0, 10.0, 10.0],
}


def _write(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


def test_pdptw_verifier_accepts_route_meeting_a_deadline_exactly(tmp_path):
    from src.cg.pdptw.d_verifier import verify_solution

    result = verify_solution(PDPTW_EDGE, {"solution": {"routes": [[1, 2, 3, 4]]}})
    assert result["ok"], result
    assert result["objective_value"] == pytest.approx(8.6)


def test_pdptw_reference_finds_the_deadline_route(tmp_path):
    from src.cg.pdptw.a_gold_cg import PDPTWBranchAndPrice

    path = _write(tmp_path, "data.json", PDPTW_EDGE)
    result = PDPTWBranchAndPrice(time_limit=20).run(str(path))
    assert result["termination_reason"] == "optimal"
    assert result["best_objective"] == pytest.approx(8.6)


def test_routing_verifiers_respect_depot_opening_and_closing():
    from src.cg.vrptw.d_verifier import verify_solution

    data = {"n_customers": 1, "capacity": 10, "x": [0.0, 10.0], "y": [0.0, 0.0], "demand": [0, 1],
            "ready": [50.0, 0.0], "due": [100.0, 55.0], "service": [0.0, 0.0]}
    # Leaving at the depot's opening (50) reaches the customer at 60 > 55.
    assert not verify_solution(data, {"solution": {"routes": [[1]]}})["ok"]
    data["due"] = [65.0, 70.0]  # return at 70 > depot closing 65
    assert not verify_solution(data, {"solution": {"routes": [[1]]}})["ok"]


def test_lbbd_verifier_rejects_overflowing_starts():
    from src.bd.lbbd.d_verifier import verify_solution

    data = {"n_tasks": 1, "n_machines": 1, "release": [0], "due": [1], "weight": [2], "proc_time": [[1]], "proc_cost": [[1]]}
    assert not verify_solution(data, {"solution": {"assignments": [{"task_id": 0, "machine_id": 0, "start": 1e308}]}})["ok"]
    assert not verify_solution(data, {"solution": {"assignments": [{"task_id": 0, "machine_id": 0, "start": 10**400}]}})["ok"]


@pytest.mark.parametrize("suite,problem,solution", [
    ("bd", "lbbd", {"assignments": [{"task_id": 0, "machine_id": 0, "start": 0}]}),
    ("cg", "gcol", {"color_classes": [[0]]}),
    ("cg", "bppc", {"bins": [[0]]}),
    ("cg", "vrptw", {"routes": [[1]]}),
    ("cg", "pdptw", {"routes": [[1, 2]]}),
    ("cg", "nrp", {"rosters": [[-1, 0]]}),
])
def test_output_contract_knows_the_harder_families(suite, problem, solution):
    method = "benders_decomposition" if suite == "bd" else "column_generation"
    good = {"status": "feasible", "method": method, "objective_value": 1.0, "solution": solution}
    assert validate_output_contract(suite, problem, {}, good) == []
    assert validate_output_contract(suite, problem, {}, {**good, "method": "benders"}) != []
    assert validate_output_contract(suite, problem, {}, {**good, "status": "great"}) != []


def test_shared_verifier_rejects_a_wrong_claimed_objective(tmp_path):
    # Codex review 4, finding 3: a valid plan with a bogus objective must fail through the shared path.
    data = {"instance_size": "small", "n_vertices": 2, "edges": [[0, 1]]}
    output = {"status": "optimal", "method": "column_generation", "objective_value": -1234567, "solution": {"color_classes": [[0], [1]]}}
    result = OutputVerifier.verify_output("cg", "gcol", _write(tmp_path, "data.json", data), _write(tmp_path, "output.json", output), 1e-6)
    assert not result["ok"]
    assert any("objective mismatch" in issue for issue in result["issues"])


def test_nrp_integral_node_keeps_its_own_solution(monkeypatch, tmp_path):
    # Codex review 4, finding 5: a worse timed integer-master incumbent must not be certified when the LP is integral.
    from src.cg.nrp import a_gold_cg
    from src.cg.nrp.a_gen_data import generate_nrp_instance

    generate_nrp_instance(0, "small", tmp_path)
    solver = a_gold_cg.NRPBranchAndPrice(time_limit=30)
    # The timed integer master finds nothing: the optimum must come from the integral node LP itself.
    monkeypatch.setattr(solver, "_integer_master", lambda inst, pool, limit: (None, float("inf")))
    result = solver.run(str(tmp_path / "data.json"))
    from src.cg.nrp.a_gold_monolithic import NRPBaseline

    assert result["best_objective"] == NRPBaseline().run(str(tmp_path / "data.json"))["best_objective"]
    assert result["termination_reason"] == "optimal"
