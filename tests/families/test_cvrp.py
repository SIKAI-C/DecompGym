"""CVRP (small scale): verifier distances and capacity, reference arguments."""
from __future__ import annotations

import pytest


def test_cvrp_verifier_uses_nint_distances_and_capacity():
    from src.cg.cvrp.d_verifier import verify_solution

    data = {"n_customers": 2, "capacity": 5, "x": [0, 3, 0], "y": [0, 4, 1], "demand": [0, 3, 3]}
    # 0 -> 1 is 5; 1 -> 2 is sqrt(18) = 4.24 -> 4; 2 -> 0 is 1.
    assert verify_solution(data, {"solution": {"routes": [[1], [2]]}}) == {"ok": True, "issues": [], "objective_value": 12.0}
    assert not verify_solution(data, {"solution": {"routes": [[1, 2]]}})["ok"]  # load 6 > 5
    assert not verify_solution(data, {"solution": {"routes": [[1]]}})["ok"]  # customer 2 unserved


@pytest.mark.parametrize("module,cls", [("src.cg.cvrp.a_gold_cg", "CVRPColumnGeneration")])
def test_reference_rejects_an_empty_pricing_batch(module, cls):
    # Codex review (routing families) round 2: batch=0 would read as pricing convergence.
    import importlib

    with pytest.raises(ValueError, match="positive"):
        getattr(importlib.import_module(module), cls)(batch=0)
