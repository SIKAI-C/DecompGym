"""Independent exhaustive oracles exercise cuts, certificates, and real HiGHS solves."""

from copy import deepcopy
from functools import lru_cache
from itertools import permutations, product
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.pilots.lrp.certify import certify
from src.pilots.lrp.data import generate, shortest_route, validate_data, verify
from src.pilots.lrp.exact_cover import solve_exact_cover
from src.pilots.lrp.run import campaign, validate_config
from src.pilots.lrp.solvers import Solver, point_cut, relative_gap


def oracle(data, y):
    """Enumerate exact partitions through bitmasks; no LP/MIP solver or incidence matrix."""
    routes = [
        (sum(1 << i for i in route), data["route_depot"][r], data["route_cost"][r])
        for r, route in enumerate(data["routes"])
    ]

    @lru_cache(None)
    def search(remaining, fleet):
        if not remaining:
            return 0.0
        first = remaining & -remaining
        best = math.inf
        for mask, depot, cost in routes:
            if y[depot] and fleet[depot] and mask & first and mask & remaining == mask:
                reduced = list(fleet)
                reduced[depot] -= 1
                best = min(best, cost + search(remaining ^ mask, tuple(reduced)))
        return best

    return search((1 << data["n"]) - 1, tuple(map(int, data["max_vehicles"])))


def gap_fixture(integer_infeasible=False):
    # Three pair routes cover an odd customer set fractionally at cost 1.5.
    # Whole routes need one pair plus one singleton, costing 3 (or 4 at depot 2).
    routes = [[0, 1], [1, 2], [0, 2], [0], [1], [2]]
    depots = [0, 0, 0, 1, 1, 1] if integer_infeasible else [0] * 6
    m = 2 if integer_infeasible else 1
    data = {
        "m": m,
        "n": 3,
        "max_stops": 2,
        "max_vehicles": [2, 3] if m == 2 else [3],
        "depot_cost": [1.0] * m,
        "routes": routes,
        "route_depot": depots,
        "route_cost": [1.0] * 3 + ([3.0] * 3 if m == 2 else [2.0] * 3),
        "incidence": [[int(i in r) for r in routes] for i in range(3)],
    }
    return data


def check_cuts(data, cuts):
    for y in product((0, 1), repeat=data["m"]):
        q = oracle(data, y)
        if not math.isfinite(q):
            continue
        for cut in cuts:
            rhs = cut["constant"] + np.dot(cut["coefficients"], y)
            assert rhs <= (q if cut["kind"].endswith("optimality") else 0) + 1e-6, (y, cut, q)


def test_route_dynamic_program_matches_brute_force():
    rng = np.random.default_rng(7)
    depot, coordinates = rng.random(2), rng.random((5, 2))

    def cost(order):
        points = np.asarray([depot, *coordinates[list(order)], depot])
        return np.linalg.norm(np.diff(points, axis=0), axis=1).sum()

    value, order = shortest_route(depot, range(5), coordinates)
    assert value == pytest.approx(min(cost(p) for p in permutations(range(5))))
    assert value == pytest.approx(cost(order))


@pytest.mark.parametrize("cap", [50, 200])
def test_generator_reproducible_complete_vs_sampled_and_feasible(cap):
    spec = {"depots": 3, "customers": 8, "max_stops": 3, "routes_per_depot": cap}
    first = generate(spec, 17)
    assert first == generate(spec, 17)
    data, metadata = first
    assert verify(data, metadata["witness"])["ok"]
    assert metadata["route_pool"] == ("sampled_restricted" if cap == 50 else "complete_stop_limited")
    assert all(metadata["route_length_counts"][str(k)] > 0 for k in (1, 2, 3))
    assert len(data["routes"]) == 3 * min(cap, 8 + 28 + 56)


def test_single_stop_pool_needs_only_singletons():
    data, metadata = generate({"depots": 2, "customers": 4, "max_stops": 1, "routes_per_depot": 4}, 0)
    assert len(data["routes"]) == 8
    assert verify(data, metadata["witness"])["ok"]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    "method",
    [
        "monolithic",
        "lp_benders",
        "integer_benders",
        "integer_benders_lp_first",
        "integer_benders_lp_first_repair",
        "integer_benders_exact_cover",
    ],
)
def test_real_solvers_against_exhaustive_partition_oracle(seed, method):
    data, metadata = generate({"depots": 3, "customers": 5, "max_stops": 3, "routes_per_depot": 30}, seed)
    optimum = min(np.dot(data["depot_cost"], y) + oracle(data, y) for y in product((0, 1), repeat=data["m"]))
    result = Solver(data, method, 10, witness=metadata["witness"]).run()
    assert result["verification"]["ok"]
    assert result["lower_bound"] <= optimum + 1e-6 <= result["upper_bound"] + 1e-6
    check_cuts(data, result["cuts"])
    if method != "lp_benders":
        assert result["status"] == "optimal"
        assert result["upper_bound"] == pytest.approx(optimum)
    lowers = [r["lower_bound"] for r in result["trace"]]
    uppers = [r["upper_bound"] for r in result["trace"] if r["upper_bound"] is not None]
    assert lowers == sorted(lowers)
    assert uppers == sorted(uppers, reverse=True)


def test_integer_cuts_close_real_integrality_gap_lp_cuts_leave_open():
    data = gap_fixture()
    lp = Solver(data, "lp_benders", 5).run()
    exact = Solver(data, "integer_benders", 5).run()
    assert lp["status"] == "feasible" and lp["termination_reason"] == "lp_stalled_with_integer_gap"
    assert lp["lower_bound"] == pytest.approx(2.5)
    assert lp["upper_bound"] == pytest.approx(4.0)
    assert exact["status"] == "optimal" and exact["lower_bound"] == pytest.approx(4.0)
    check_cuts(data, lp["cuts"] + exact["cuts"])


@pytest.mark.parametrize(
    "method",
    ["integer_benders", "integer_benders_lp_first", "integer_benders_lp_first_repair", "integer_benders_exact_cover"],
)
def test_integer_infeasibility_is_not_mistaken_for_lp_infeasibility(method):
    data = gap_fixture(integer_infeasible=True)
    exact = Solver(data, method, 5).run()
    assert exact["status"] == "optimal" and exact["upper_bound"] == pytest.approx(6.0)
    assert any(c["kind"] == "integer_feasibility" for c in exact["cuts"])
    check_cuts(data, exact["cuts"])


@pytest.mark.parametrize("method", ["integer_benders", "integer_benders_lp_first"])
def test_incomplete_integer_solve_never_becomes_an_integer_optimality_cut(monkeypatch, method):
    solver = Solver(gap_fixture(), method, 5)
    monkeypatch.setattr(solver, "recourse_integer", lambda _: SimpleNamespace(status=1, x=None, fun=None))
    result = solver.run()
    assert result["termination_reason"] == "integer_recourse_unresolved"
    assert result["status"] != "optimal"
    assert not any(c["kind"] == "integer_optimality" for c in result["cuts"])


def test_lp_first_reoptimizes_master_before_integer_recourse():
    data = gap_fixture()
    result = Solver(data, "integer_benders_lp_first", 5).run()
    roles = [solve["role"] for solve in result["solves"]]
    assert roles[:4] == ["master", "recourse_lp", "master", "recourse_lp"]
    assert roles[4] == "integer_recourse"
    assert any(row.get("integer_recourse_deferred") for row in result["trace"])
    assert result["status"] == "optimal" and result["upper_bound"] == pytest.approx(4.0)
    check_cuts(data, result["cuts"])


def test_repair_is_only_a_primal_heuristic_and_closes_unused_depots():
    data = gap_fixture()
    data.update(m=2, depot_cost=[1.0, 100.0], max_vehicles=[3, 3])
    solver = Solver(data, "integer_benders_lp_first_repair", 5)
    solver.lower = 2.5
    solver.repair_openings(np.array([1, 0]), 0)
    assert solver.lower == 2.5 and solver.cuts == []
    assert solver.incumbent["objective_value"] == pytest.approx(4.0)
    assert solver.incumbent["y"] == [1, 0]
    assert solver.solves[0]["role"] == "repair_integer_recourse"
    assert solver.solves[0]["time_limit_seconds"] <= 2.0


def test_repair_after_timeout_then_returns_to_original_depot_choice(monkeypatch):
    data = gap_fixture(integer_infeasible=True)
    solver = Solver(data, "integer_benders_lp_first_repair", 5)
    original = solver.recourse_integer
    calls = []

    def interrupted_first(y, seconds=None):
        calls.append((y.tolist(), seconds))
        if len(calls) == 1:
            solver.solves.append({"role": "integer_recourse", "status": 1})
            return SimpleNamespace(status=1, x=None, fun=None)
        return original(y, seconds)

    monkeypatch.setattr(solver, "recourse_integer", interrupted_first)
    result = solver.run()
    assert calls[:3] == [([1, 0], 2.0), ([1, 1], 2.0), ([1, 0], None)]
    assert result["status"] == "optimal" and result["upper_bound"] == pytest.approx(6.0)
    assert result["incumbent"]["source"] == "one_extra_depot_repair"
    check_cuts(data, result["cuts"])


def test_expired_budget_preserves_verified_witness():
    data, meta = generate({"depots": 2, "customers": 4, "max_stops": 2, "routes_per_depot": 20}, 0)
    solver = Solver(data, "integer_benders", 5, witness=meta["witness"])
    solver.deadline = 0
    result = solver.run()
    assert result["status"] == "feasible" and result["termination_reason"] == "time_limit"
    assert result["verification"]["ok"]


def test_pointwise_cut_valid_away_from_its_generating_plan():
    cut = point_cut([1, 0, 1], 7.0)
    for y in product((0, 1), repeat=3):
        rhs = cut["constant"] + np.dot(cut["coefficients"], y)
        assert rhs == 7 if y == (1, 0, 1) else rhs <= 0


@pytest.mark.parametrize("change", ["fractional", "nonfinite", "objective", "closed", "missing"])
def test_independent_verifier_rejects_bad_outputs(change):
    data, metadata = generate({"depots": 2, "customers": 4, "max_stops": 2, "routes_per_depot": 20}, 0)
    output = deepcopy(metadata["witness"])
    if change == "fractional":
        output["lambda"] = [0.5] * len(data["routes"])
    elif change == "nonfinite":
        output["y"][0] = float("nan")
    elif change == "objective":
        output["objective_value"] -= 1
    elif change == "closed":
        output["y"] = [0, 0]
    else:
        del output["lambda"]
    assert not verify(data, output)["ok"]


def test_data_incidence_is_checked_against_routes():
    data = gap_fixture()
    data["incidence"][0][0] = 0
    with pytest.raises(ValueError, match="Incidence"):
        validate_data(data)


@pytest.mark.parametrize("report_fails", [False, True])
def test_runner_freezes_inputs_runs_workers_and_refuses_overwrite(tmp_path, monkeypatch, report_fails):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "solver_seconds": 5,
                "cases": [{"name": "tiny", "depots": 2, "customers": 4, "max_stops": 2, "routes_per_depot": 20}],
            }
        )
    )
    output = tmp_path / "run"
    if report_fails:

        def broken_report(*_):
            raise RuntimeError("Injected report failure")

        monkeypatch.setattr("src.pilots.lrp.report.report", broken_report)
        with pytest.raises(RuntimeError, match="Injected report failure"):
            campaign(config, output)
        assert json.loads((output / "manifest.json").read_text())["run_state"] == "report_failed"
        return
    assert campaign(config, output) == 0
    results = json.loads((output / "results.json").read_text())
    assert len(results) == 3
    assert all(r["independent_verification"]["ok"] for r in results)
    assert all("solvers.py" in r["executed_source_sha256"] for r in results)
    assert json.loads((output / "manifest.json").read_text())["run_state"] == "complete"
    assert json.loads((output / "cross_method_checks.json").read_text())["bound_contradictions"] == []
    assert (output / "source_snapshot/src/pilots/lrp/solvers.py").is_file()
    with pytest.raises(FileExistsError):
        campaign(config, output)


def test_config_paths_budgets_and_crossed_bounds():
    with pytest.raises(ValueError):
        validate_config({"schema_version": 1, "cases": [{"name": "../bad"}]})
    with pytest.raises(ValueError):
        validate_config({"schema_version": 1, "cases": [{"name": "good"}], "solver_seconds": float("nan")})
    with pytest.raises(ValueError, match="Imported inputs"):
        validate_config({"schema_version": 1, "cases": [{"name": "good", "source": "data.json", "seeds": [0, 1]}]})
    assert relative_gap(11, 10) is None
    assert relative_gap(8, 10) == pytest.approx(0.2)


@pytest.mark.parametrize("seed", range(6))
def test_exact_cover_costs_and_dual_pruning_against_independent_oracle(seed):
    data, _ = generate({"depots": 3, "customers": 6, "max_stops": 3, "routes_per_depot": 30}, seed)
    for y in product((0, 1), repeat=3):
        expected = oracle(data, y)
        # Deliberately infeasible duals exercise conservative dual repair.
        result = solve_exact_cover(data, y, 5, alpha=np.full(6, 10.0), beta=np.full(3, 2.0))
        assert result["exhausted"]
        if math.isfinite(expected):
            assert result["status"] == "optimal"
            assert result["objective"] == pytest.approx(expected)
            assert result["lower_bound"] <= expected + 1e-9
            output = {
                "y": list(y),
                "lambda": [int(r in result["routes"]) for r in range(len(data["routes"]))],
                "objective_value": result["objective"] + np.dot(data["depot_cost"], y),
            }
            assert verify(data, output)["ok"]
        else:
            assert result["status"] == "infeasible" and result["routes"] is None


def test_exact_cover_timeout_retains_safe_bound_and_never_claims_infeasible(monkeypatch):
    data = gap_fixture()
    result = solve_exact_cover(data, [1], 0)
    assert result["status"] == "time_limit" and not result["exhausted"]
    assert result["lower_bound"] <= oracle(data, [1])
    clock = [0.0]
    monkeypatch.setattr("src.pilots.lrp.exact_cover.time.monotonic", lambda: clock[0])

    def interrupt_after_incumbent(*_):
        clock[0] = 2.0

    result = solve_exact_cover(data, [1], 1, on_incumbent=interrupt_after_incumbent)
    assert result["status"] == "time_limit" and result["routes"] is not None
    assert not result["exhausted"]
    assert result["lower_bound"] <= oracle(data, [1]) <= result["objective"]


def test_incomplete_exact_cover_does_not_generate_integer_cut(monkeypatch):
    original = solve_exact_cover
    monkeypatch.setattr("src.pilots.lrp.exact_cover.solve_exact_cover", lambda data, y, *_: original(data, y, 0))
    result = Solver(gap_fixture(), "integer_benders_exact_cover", 5).run()
    assert result["termination_reason"] == "integer_recourse_unresolved"
    assert not any(c["kind"].startswith("integer_") for c in result["cuts"])


def test_enumeration_certificate_never_uses_benders_master(monkeypatch):
    data = gap_fixture(integer_infeasible=True)
    solution = Solver(data, "integer_benders_exact_cover", 5).run()["incumbent"]

    def forbidden(*_):
        raise AssertionError("Enumeration must not use Benders master")

    monkeypatch.setattr(Solver, "master", forbidden)
    result = certify(data, solution, 5)
    assert result["status"] == "certified" and result["patterns_checked"] == 4
    assert result["lower_bound"] <= 6.0 and result["relative_gap"] <= 1e-7


def test_enumeration_audit_finds_better_solution_and_reports_timeout():
    data = gap_fixture()
    bad = {"y": [1], "lambda": [0, 0, 0, 1, 1, 1], "objective_value": 7.0}
    result = certify(data, bad, 5)
    assert result["status"] == "counterexample"
    assert result["better_solution"]["objective_value"] == pytest.approx(4.0)
    assert verify(data, result["better_solution"])["ok"]
    timeout = certify(data, bad, 0)
    assert timeout["status"] == "unresolved" and timeout["lower_bound"] is None
