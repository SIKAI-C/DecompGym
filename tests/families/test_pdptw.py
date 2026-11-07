"""PDPTW (small scale): verifier, branch-and-price reference and shortcut regressions (Codex reviews 4 and 5)."""
from __future__ import annotations

import json


# Codex review 4, finding 4: arrival 118.00000000000001 > 118 in floating point rejected the optimal PDPTW route.
# Rewritten for the 1000x rounded-distance convention (2026-11-05): the route [1, 2, 3, 4] reaches node 4 at exactly 118064.
PDPTW_EDGE = {
    "n_requests": 2, "capacity": 60,
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


def test_pdptw_monolithic_records_its_solve_time(tmp_path):
    """Codex final review, finding 7: the monolithic gold records its MIP time (total_time, checked by the admission
    rule), Gurobi's runtime and its parameters; the objective and routes are unchanged."""
    import math

    from src.cg.pdptw.a_gold_monolithic import PDPTWBaseline
    from src.utils.reference_output import reference_output

    path = _write(tmp_path, "data.json", PDPTW_EDGE)
    result = PDPTWBaseline().run(str(path))
    assert (result["best_objective"], result["routes"]) == (8714.0, [[1, 2, 3, 4]])
    assert all(math.isfinite(result[key]) and result[key] >= 0 for key in ("total_time", "runtime", "end_to_end_seconds"))
    assert result["total_time"] <= result["end_to_end_seconds"] and result["params"]["Threads"] == 1
    assert reference_output("cg", "pdptw", PDPTW_EDGE, result)["status"] == "optimal"


def test_pdptw_shortcut_keeps_its_seed_plan(monkeypatch, tmp_path):
    # Codex review 5, finding 3: the final master finding nothing must not lose the verified seed plan.
    from src.cg.pdptw.a_gold_cg import PDPTWRootColumnGeneration

    solver = PDPTWRootColumnGeneration(time_limit=10)
    monkeypatch.setattr(solver, "_integer_master", lambda inst, pool, limit: (None, float("inf")))
    result = solver.run(str(_write(tmp_path, "data.json", PDPTW_EDGE)))
    assert result["routes"] == [[1, 3], [2, 4]]
    assert result["best_objective"] < float("inf")
