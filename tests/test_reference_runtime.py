"""Budget expiry must not manufacture a certificate or mix incumbent plans."""
from types import SimpleNamespace

import pytest

from src.utils.base_bd_class import BaseBenders, MasterSolution, SubproblemResult
from src.utils.base_cg_class import BaseColumnGeneration, IntegerMasterSolution, MasterLPSolution, PricingResult
from src.utils.reference_runtime import NoReferenceIncumbent, ReferenceLimit, ReferenceRuntime


class InterruptedBenders(BaseBenders):
    def __init__(self, interrupt_at=2):
        super().__init__(max_iters=3)
        self.counter, self.interrupt_at = 0, interrupt_at

    def load_instance(self, name):
        return None

    def build_initial_master(self, inst):
        return None

    def solve_master(self, inst, master):
        self.counter += 1
        return MasterSolution(8.0, {"y": self.counter}, None, bound=float(self.counter))

    def solve_subproblems(self, inst, master):
        if self.counter == self.interrupt_at:
            raise ReferenceLimit("interrupted after some scenarios, before all recourse was available")
        return SubproblemResult([], [1], True, 9.0, metadata={"x": self.counter})

    def compute_iteration_bounds(self, *args):
        self.global_ub = min(self.global_ub, 10.0)
        return self.global_lb, self.global_ub

    def add_feasibility_cuts(self, *args):
        pass

    def add_optimality_cuts(self, *args):
        pass


def test_partial_benders_iteration_retains_complete_incumbent_and_new_valid_bound():
    result = InterruptedBenders().run("unused")
    assert result["final_vars"] == {"y": 1}
    assert result["sub_metadata"] == {"x": 1}
    assert result["best_objective"] == 10 and result["global_lb"] == 2
    assert result["incumbent_iteration"] == result["iters"] == 1
    assert result["termination_reason"] == "time_limit" and result["status"] == "feasible"


def test_no_complete_incumbent_never_exports_master_relaxation_as_solution():
    with pytest.raises(NoReferenceIncumbent):
        InterruptedBenders(interrupt_at=1).run("unused")


def test_ufl_keeps_incumbent_objective_separate_from_master_bound():
    from src.bd.ufl.b_gold_bd import UFLBenders

    solver = UFLBenders()
    master = MasterSolution(10.0, {}, None, bound=9.0)
    sub = SubproblemResult([], [], True, 0, metadata={"total_sub_cost": 5.0, "theta_sum": 4.0})
    assert solver.compute_iteration_bounds(None, master, sub) == (9.0, 11.0)


class InterruptedCG(BaseColumnGeneration):
    def load_instance(self, name):
        return None

    def initial_patterns(self, inst):
        return ["seed"]

    def solve_restricted_master(self, inst, patterns):
        return MasterLPSolution(10.0 if len(patterns) == 1 else 5.0, None)

    def solve_pricing(self, inst, master, patterns):
        if len(patterns) == 2:
            raise ReferenceLimit("later pricing incomplete")
        return PricingResult("new", -5.0)

    def solve_integer_master(self, inst, patterns):
        assert self.runtime.stage_reserve == 0
        return IntegerMasterSolution(6.0, {1: 1})


def test_partial_later_pricing_does_not_create_a_global_bound():
    result = InterruptedCG().run("unused")
    assert result["best_objective"] == 6.0
    assert result["global_lb"] is None and not result["lp_converged"]
    assert result["termination_reason"] == "time_limit" and not result["proven_optimal"]


def test_timing_includes_initialization_and_final_integer_master(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])

    class TimedCG(InterruptedCG):
        def initial_patterns(self, inst):
            clock[0] += 2
            return ["seed"]

        def solve_pricing(self, inst, master, patterns):
            clock[0] += 3
            return PricingResult(None, 0.0)

        def solve_integer_master(self, inst, patterns):
            clock[0] += 5
            return IntegerMasterSolution(10.0, {0: 1})

    result = TimedCG().run("unused")
    assert result["total_time"] == 10.0
    assert result["iter_records"][0].elapsed_time == 5.0


def fake_model(clock, seconds, *, status=2, solutions=1):
    model = SimpleNamespace(Params=SimpleNamespace(TimeLimit=999.0, Threads=8, MIPGap=0.1, MIPGapAbs=0.1), Status=status, SolCount=solutions, ModelName="test")
    model.optimize = lambda: clock.__setitem__(0, clock[0] + seconds)
    return model


def test_every_model_obeys_remaining_shared_budget_and_pricing_tolerance(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])
    runtime = ReferenceRuntime()
    runtime.start()
    first = fake_model(clock, 20)
    runtime.optimize(first, role="master")
    second = fake_model(clock, 1)
    runtime.optimize(second, role="pricing")
    assert first.Params.Threads == second.Params.Threads == 1
    assert first.Params.TimeLimit == 54 and second.Params.TimeLimit == 34
    assert second.Params.MIPGap == second.Params.MIPGapAbs == 0


@pytest.mark.parametrize("allow_incumbent,solutions,accepted", [(True, 1, True), (True, 0, False), (False, 1, False)])
def test_cutoff_preserves_admissible_integer_incumbent_only(monkeypatch, allow_incumbent, solutions, accepted):
    clock = [0.0]
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])
    runtime = ReferenceRuntime()
    runtime.start()
    model = fake_model(clock, 55, status=9, solutions=solutions)
    if accepted:
        runtime.optimize(model, role="integer_master", allow_incumbent=allow_incumbent)
    else:
        with pytest.raises(ReferenceLimit):
            runtime.optimize(model, role="pricing", allow_incumbent=allow_incumbent)
    with pytest.raises(ReferenceLimit):
        runtime.check()


def test_target_gap_is_not_a_global_optimality_claim():
    result = ReferenceRuntime().finish({"global_lb": 99.6, "global_ub": 100}, target_gap=0.005, termination_reason="target_gap")
    assert result["target_gap_met"] and not result["proven_optimal"] and result["status"] == "feasible"
    crossed = ReferenceRuntime().finish({"global_lb": 100.00005, "global_ub": 100}, target_gap=0.005, termination_reason="no_new_cuts")
    assert not crossed["target_gap_met"] and not crossed["proven_optimal"]


def test_nonoptimal_bp_incumbent_normalizes_surplus_bins_before_output():
    from src.cg.bp.a_gold_cg import _trim_surplus_bins
    from src.utils.reference_output import reference_output

    patterns = [[1, 0], [0, 1], [1, 1]]
    kept = _trim_surplus_bins(patterns, {0: 1, 1: 1, 2: 1}, [1, 1])
    assert kept == {0: 1, 1: 1}
    result = {"best_objective": sum(kept.values()), "master_metadata": {"patterns": [{"usage": value, "contents": [{"item_id": i, "count": n} for i, n in enumerate(patterns[p]) if n]} for p, value in kept.items()]}}
    data = {"bin_capacity": 5, "items": [{"id": 0, "size": 2}, {"id": 1, "size": 3}]}
    output = reference_output("cg", "bp", data, result)
    assert output["objective_value"] == len(output["solution"]["bins"]) == 2


def test_shift_pricing_certificate_accounts_for_small_negative_reduced_cost():
    from src.cg.shift.a_gold_cg import ShiftColumnGeneration

    solver = ShiftColumnGeneration()
    solver._all_patterns = [(0, 0)]
    pricing = solver.solve_pricing(None, MasterLPSolution(10, [1.0000005]), [(0, 0)])
    assert -solver.rc_tolerance < pricing.reduced_cost < 0
    assert pricing.metadata["lp_bound"] < 10
    assert pricing.metadata["lp_bound"] * 1.0000005 == pytest.approx(10)
