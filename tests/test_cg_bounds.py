"""A restricted master is not a global bound until pricing proves convergence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.reference_runtime import ReferenceLimit
from src.utils.base_cg_class import (
    BaseColumnGeneration,
    IntegerMasterSolution,
    MasterLPSolution,
    PricingResult,
)


class ToyCoveringCG(BaseColumnGeneration):
    """Cover two rows: singleton columns cost 10 each; a joint column costs 1."""

    def load_instance(self, name):
        return None

    def initial_patterns(self, inst):
        return ["row_1", "row_2"]

    def solve_restricted_master(self, inst, patterns):
        return MasterLPSolution(1.0 if "both_rows" in patterns else 20.0, None)

    def solve_pricing(self, inst, master, patterns):
        if "both_rows" in patterns:
            return PricingResult(None, 0.0)
        return PricingResult("both_rows", -19.0)

    def solve_integer_master(self, inst, patterns):
        if "both_rows" in patterns:
            return IntegerMasterSolution(1.0, {2: 1.0})
        return IntegerMasterSolution(20.0, {0: 1.0, 1: 1.0})


def test_iteration_limit_does_not_certify_restricted_master_bound():
    result = ToyCoveringCG(max_iters=1).run("unused")
    assert result["best_objective"] == result["global_ub"] == 20.0
    assert result["global_lb"] is None  # The full problem's optimum is 1, not 20.
    assert not result["lp_converged"]
    assert json.loads(json.dumps(result, default=vars, allow_nan=False))["global_lb"] is None


def test_pricing_can_certify_convergence_on_the_last_allowed_iteration():
    result = ToyCoveringCG(max_iters=2).run("unused")
    assert result["iters"] == 2
    assert result["global_lb"] == result["global_ub"] == 1.0
    assert result["lp_converged"]


@pytest.mark.parametrize("pattern", [None, "row_1"])
def test_stalled_pricing_does_not_certify_a_bound(pattern):
    class StalledCG(ToyCoveringCG):
        def solve_pricing(self, inst, master, patterns):
            return PricingResult(pattern, -19.0)

    result = StalledCG(max_iters=3).run("unused")
    assert result["iters"] == 1
    assert result["global_lb"] is None
    assert result["global_ub"] == 20.0
    assert not result["lp_converged"]


def test_real_bp_early_stop_preserves_solution_without_invalid_bound():
    gp = pytest.importorskip("gurobipy")
    from src.api import solve, verify

    data_path = Path(__file__).resolve().parents[1] / "data/cg/bp/bp_small_seed000/data.json"
    try:
        result = solve("cg", "bp", data_path, max_iters=1)
        baseline = solve("cg", "bp", data_path, method="monolithic")
    except gp.GurobiError as exc:
        if exc.errno == gp.GRB.Error.NO_LICENSE:
            pytest.skip("Gurobi license unavailable")
        raise

    # The restricted LP is about 8.36, above the full integer optimum of 6.
    assert baseline["best_objective"] == 6.0
    assert result["iter_records"][-1]["master_obj"] > baseline["best_objective"]
    assert result["global_lb"] is None
    assert result["global_ub"] == result["best_objective"]
    assert not result["lp_converged"]
    verified = verify("cg", "bp", data_path, result["output"])
    assert verified["ok"], verified["issues"]


def test_gap_pricing_time_limit_is_not_reported_as_convergence():
    gp = pytest.importorskip("gurobipy")
    from src.cg.gap.a_gold_cg import GAPColumnGeneration

    solver = GAPColumnGeneration(pricing_params={"TimeLimit": 0, "Threads": 1})
    data_path = Path(__file__).resolve().parents[1] / "data/cg/gap/gap_small_seed000/data.json"
    inst = solver.load_instance(str(data_path))
    solver._init_matrices(inst)
    master = MasterLPSolution(
        0.0,
        {"pi": [0.0] * len(inst.jobs), "sigma": [0.0] * len(inst.machines)},
    )
    try:
        with pytest.raises(ReferenceLimit, match="pricing reached its time limit"):
            solver.solve_pricing(inst, master, [])
    except gp.GurobiError as exc:
        if exc.errno == gp.GRB.Error.NO_LICENSE:
            pytest.skip("Gurobi license unavailable")
        raise
