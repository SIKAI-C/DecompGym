"""VRPTW (small scale): verifier depot windows and reference infeasibility regressions (Codex review 5)."""
from __future__ import annotations

import json

import pytest


def _write(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


def test_routing_verifiers_respect_depot_opening_and_closing():
    from src.cg.vrptw.d_verifier import verify_solution

    data = {"n_customers": 1, "capacity": 10, "x": [0.0, 10.0], "y": [0.0, 0.0], "demand": [0, 1],
            "ready": [50.0, 0.0], "due": [100.0, 55.0], "service": [0.0, 0.0]}
    # Leaving at the depot's opening (50) reaches the customer at 60 > 55.
    assert not verify_solution(data, {"solution": {"routes": [[1]]}})["ok"]
    data["due"] = [65.0, 70.0]  # return at 70 > depot closing 65
    assert not verify_solution(data, {"solution": {"routes": [[1]]}})["ok"]


def test_vrptw_reference_does_not_certify_an_infeasible_seed(tmp_path):
    # Codex review 5, finding 2: depot opens at 50, the customer is 10 away and due at 55.
    from src.cg.vrptw.a_gold_cg import BranchAndPrice

    data = {"n_customers": 1, "capacity": 10, "x": [0.0, 10.0], "y": [0.0, 0.0], "demand": [0, 1],
            "ready": [50.0, 0.0], "due": [100.0, 55.0], "service": [0.0, 0.0]}
    with pytest.raises(RuntimeError, match="no feasible routing plan"):
        BranchAndPrice(time_limit=10).run(str(_write(tmp_path, "data.json", data)))
