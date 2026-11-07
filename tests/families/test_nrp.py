"""Nurse rostering (small scale): branch-and-price certificate regression (Codex review 4)."""
from __future__ import annotations




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
