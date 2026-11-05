"""Regression tests for the harder-family cohort (Codex review 4 findings)."""
from __future__ import annotations

import json

import pytest

from src.experiments.verification import OutputVerifier
from src.utils.output_contract import validate_output_contract


# Codex review 4, finding 4: arrival 118.00000000000001 > 118 in floating point rejected the optimal PDPTW route.
# Rewritten for the 1000x rounded-distance convention (2026-11-05): the route [1, 2, 3, 4] reaches node 4 at exactly 118064.
PDPTW_EDGE = {
    "instance_size": "small", "n_requests": 2, "capacity": 60,
    "x": [50, 51, 52, 53, 51], "y": [50, 51, 52, 53, 52],
    "demand": [0, 10, 10, -10, -10],
    "ready": [0, 83000, 47000, 96000, 58000], "due": [600000, 143000, 107000, 156000, 118064],
    "service": [0, 10000, 10000, 10000, 10000],
}


def _write(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


def test_pdptw_verifier_accepts_route_meeting_a_deadline_exactly(tmp_path):
    from src.cg.pdptw.d_verifier import verify_solution

    result = verify_solution(PDPTW_EDGE, {"solution": {"routes": [[1, 2, 3, 4]]}})
    assert result["ok"], result
    assert result["objective_value"] == 8714
    late = {**PDPTW_EDGE, "due": PDPTW_EDGE["due"][:4] + [118063]}
    assert not verify_solution(late, {"solution": {"routes": [[1, 2, 3, 4]]}})["ok"]


def test_pdptw_reference_finds_the_deadline_route(tmp_path):
    from src.cg.pdptw.a_gold_cg import PDPTWBranchAndPrice

    path = _write(tmp_path, "data.json", PDPTW_EDGE)
    result = PDPTWBranchAndPrice(time_limit=20).run(str(path))
    assert result["termination_reason"] == "optimal"
    assert result["best_objective"] == 8714


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


def test_certificates_must_be_ordered_meeting_and_match_the_objective():
    from src.utils.harder_suite import _certified

    assert _certified({"global_lb": 10, "global_ub": 10, "best_objective": 10})
    assert not _certified({"global_lb": 20, "global_ub": 10, "best_objective": 10})  # crossed
    assert not _certified({"global_lb": 0, "global_ub": 0, "best_objective": 10})  # bound/objective mismatch
    assert not _certified({"global_lb": None, "global_ub": 10, "best_objective": 10})


def test_vrptw_reference_does_not_certify_an_infeasible_seed(tmp_path):
    # Codex review 5, finding 2: depot opens at 50, the customer is 10 away and due at 55.
    from src.cg.vrptw.a_gold_cg import BranchAndPrice

    data = {"instance_size": "small", "n_customers": 1, "capacity": 10, "x": [0.0, 10.0], "y": [0.0, 0.0], "demand": [0, 1],
            "ready": [50.0, 0.0], "due": [100.0, 55.0], "service": [0.0, 0.0]}
    with pytest.raises(RuntimeError, match="no feasible routing plan"):
        BranchAndPrice(time_limit=10).run(str(_write(tmp_path, "data.json", data)))


def test_pdptw_shortcut_keeps_its_seed_plan(monkeypatch, tmp_path):
    # Codex review 5, finding 3: the final master finding nothing must not lose the verified seed plan.
    from src.cg.pdptw.a_gold_cg import PDPTWRootColumnGeneration

    solver = PDPTWRootColumnGeneration(time_limit=10)
    monkeypatch.setattr(solver, "_integer_master", lambda inst, pool, limit: (None, float("inf")))
    result = solver.run(str(_write(tmp_path, "data.json", PDPTW_EDGE)))
    assert result["routes"] == [[1, 3], [2, 4]]
    assert result["best_objective"] < float("inf")


def test_new_routing_families_output_contracts():
    good = {"status": "feasible", "method": "column_generation", "objective_value": 1.0}
    assert validate_output_contract("cg", "cvrp", {}, {**good, "solution": {"routes": [[1, 2]]}}) == []
    assert validate_output_contract("cg", "clrp", {}, {**good, "solution": {"routes": [{"depot": 0, "customers": [0, 1]}]}}) == []
    assert validate_output_contract("cg", "clrp", {}, {**good, "solution": {"routes": [[0, 1]]}}) != []
    assert validate_output_contract("cg", "clrp", {}, {**good, "solution": {"routes": [{"depot": 0, "customers": [0.5]}]}}) != []


def test_cvrp_verifier_uses_nint_distances_and_capacity():
    from src.cg.cvrp.d_verifier import verify_solution

    data = {"n_customers": 2, "capacity": 5, "x": [0, 3, 0], "y": [0, 4, 1], "demand": [0, 3, 3]}
    # 0 -> 1 is 5; 1 -> 2 is sqrt(18) = 4.24 -> 4; 2 -> 0 is 1.
    assert verify_solution(data, {"solution": {"routes": [[1], [2]]}}) == {"ok": True, "issues": [], "objective_value": 12.0}
    assert not verify_solution(data, {"solution": {"routes": [[1, 2]]}})["ok"]  # load 6 > 5
    assert not verify_solution(data, {"solution": {"routes": [[1]]}})["ok"]  # customer 2 unserved


def test_clrp_verifier_charges_opening_vehicle_and_ceil_travel():
    from src.cg.clrp.d_verifier import verify_solution

    data = {"n_customers": 2, "n_depots": 2, "vehicle_capacity": 10, "vehicle_cost": 7, "depot_x": [0, 9], "depot_y": [0, 9],
            "depot_capacity": [5, 20], "opening_cost": [100, 1000], "customer_x": [1, 0], "customer_y": [1, 1], "demand": [3, 4]}
    # Depot 0 holds only 5 < 7, so both customers on one route from depot 0 is infeasible.
    assert not verify_solution(data, {"solution": {"routes": [{"depot": 0, "customers": [0, 1]}]}})["ok"]
    # One route each: depot 0 -> c0 -> 0 = 2 * ceil(141.42) = 284; depot 1 -> c1 -> 1 = 2 * ceil(100 * sqrt(145)) = 2 * 1205.
    result = verify_solution(data, {"solution": {"routes": [{"depot": 0, "customers": [0]}, {"depot": 1, "customers": [1]}]}})
    assert result["ok"] and result["objective_value"] == 100 + 1000 + 2 * 7 + 284 + 2 * 1205
    assert not verify_solution(data, {"solution": {"routes": [{"depot": 2, "customers": [0, 1]}]}})["ok"]


def test_clrp_verifier_charges_a_used_depot_even_without_demand():
    # Codex review (routing families) finding 3: opening costs follow the routes, not the load.
    from src.cg.clrp.d_verifier import verify_solution

    data = {"n_customers": 1, "n_depots": 1, "vehicle_capacity": 10, "vehicle_cost": 7, "depot_x": [0], "depot_y": [0],
            "depot_capacity": [5], "opening_cost": [100], "customer_x": [1], "customer_y": [0], "demand": [0]}
    assert verify_solution(data, {"solution": {"routes": [{"depot": 0, "customers": [0]}]}})["objective_value"] == 307


def test_strict_suite_rejects_an_uncertified_reference(monkeypatch, tmp_path):
    # Codex review (routing families) finding 2: strict families need both certificates, and they must agree.
    from src.utils import harder_suite

    data = {"n_customers": 1}
    gen = lambda seed, size, out: (out.mkdir(parents=True, exist_ok=True), (out / "data.json").write_text("{}"), data)[2]
    results = {"monolithic": {"best_objective": 10.0, "output": {}}, "decomposition": {"best_objective": 10.0, "global_lb": 9.0, "global_ub": 10.0, "output": {}}}
    monkeypatch.setattr(harder_suite, "solve", lambda suite, problem, path, method, **kw: dict(results[method]))
    monkeypatch.setattr(harder_suite, "verify", lambda *args: {"ok": True})
    with pytest.raises(RuntimeError, match="matching certified reference"):
        harder_suite.generate_suite("cg", "cvrp", gen=gen, render_prompt=lambda d: "p", sizes=["small"], num_instances=1, out_root=tmp_path, strict=True)
    results["decomposition"]["global_lb"] = 10.0
    metas = harder_suite.generate_suite("cg", "cvrp", gen=gen, render_prompt=lambda d: "p", sizes=["small"], num_instances=1, out_root=tmp_path, strict=True)
    assert metas[0]["ground_truth_source"] == "monolithic_mip" and metas[0]["reference_proved"]


@pytest.mark.parametrize("module,cls", [("src.cg.cvrp.a_gold_cg", "CVRPColumnGeneration"), ("src.cg.clrp.a_gold_cg", "CLRPColumnGeneration")])
def test_routing_references_reject_an_empty_pricing_batch(module, cls):
    # Codex review (routing families) round 2: batch=0 would read as pricing convergence.
    import importlib

    with pytest.raises(ValueError, match="positive"):
        getattr(importlib.import_module(module), cls)(batch=0)
