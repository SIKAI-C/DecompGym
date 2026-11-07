"""src/01_gen.py validation follows the registered admission rules (src.utils.families, src.utils.dataset,
src.utils.instance_suite) and the admitted meta.json; no optimizer is imported here."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

from src.utils.dataset import instance_admission


ROOT = Path(__file__).resolve().parents[1]
CLI = importlib.import_module("src.01_gen")
PASSING = {"pass_simple_test", "reference_gap"}


def gold(objective, verified=True, lb=None):
    """A gold as src.utils.instance_suite writes it: verified with bounds [lb, objective] (closed by default) and a solve
    time, or failed (error, no verification)."""
    if not verified:
        return {"best_objective": objective, "error": "RuntimeError: x"}
    return {"best_objective": objective, "global_lb": objective if lb is None else lb, "global_ub": objective, "total_time": 1.0, "verification": {"ok": True}}


def stub(directory: Path, suite: str, mono, decomposed, meta: dict | None, *, mono_verified=None, decomposition_verified=None, decomposition_lb=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "data.json").write_text("{}")
    (directory / "prompt.txt").write_text("prompt")
    (directory / "gold_monolithic.json").write_text(json.dumps(gold(mono, mono is not None if mono_verified is None else mono_verified)))
    (directory / f"gold_{suite}.json").write_text(json.dumps(gold(decomposed, decomposed is not None if decomposition_verified is None else decomposition_verified, decomposition_lb)))
    if meta is not None:
        (directory / "meta.json").write_text(json.dumps(meta))
    return directory


def admitted(source: str, truth) -> dict:
    return {"ground_truth_source": source, "ground_truth_objective": truth}


@pytest.mark.parametrize("dataset", ["data", "data_harder", "outputs/litgen_data"])
@pytest.mark.parametrize("level", ["strict", "simple"])
def test_the_pinned_datasets_validate(dataset, level):
    if not (ROOT / dataset).is_dir():
        message = f"{dataset} is not present (outputs/ is not part of the repository)"
        if os.environ.get("DECOMPGYM_ALLOW_MISSING_REFERENCE_DATA") == "1":
            pytest.skip(message)
        pytest.fail(message)
    results = CLI.validate_all(ROOT / dataset, validation_level=level)
    assert results and [r["name"] for r in results if r["status"] not in PASSING] == []
    if dataset == "data_harder" and level == "strict":
        # The two BPPC instances admitted by the branch-and-price certificate (gold_monolithic.json has no optimum).
        certified = sorted(r["name"] for r in results if r["ground_truth_source"] == "decomposition_certificate")
        assert certified == ["bppc_medium_seed000", "bppc_medium_seed002"]
        assert all(r["status"] == "pass_simple_test" and r["mono_obj"] is None for r in results if r["name"] in certified)


def test_the_rule_is_the_rule_of_the_root_holding_the_instance(tmp_path):
    assert instance_admission(ROOT / "data/bd/ufl/ufl_small_seed000") == "family"  # the paper's UFL, not the KG recipe
    assert instance_admission(ROOT / "data_harder/cg/bppc/bppc_small_seed000") == "mip_or_certificate"
    assert instance_admission(ROOT / "data_harder/cg/gcol/gcol_small_seed000") == "mip_or_certificate"
    assert instance_admission(ROOT / "outputs/litgen_data/bd/ufl/ufl_small_seed000") == "strict"
    # A fresh reproduction root: the family's registered rule.
    assert instance_admission(tmp_path / "bd/ufl/ufl_small_seed000") == "strict"
    assert instance_admission(tmp_path / "cg/gcol/gcol_small_seed000") == "mip_or_certificate"
    assert instance_admission(tmp_path / "cg/shift/shift_small_seed000") == "family"
    assert instance_admission(tmp_path / "cg/hub/hub_small_seed000") is None


# bp: a strict family whose reference need not certify itself (reference_certified False); its reference bounds are open.
@pytest.mark.parametrize("case,mono,decomposed,meta,status", [
    ("admitted", 100.0, 100.0, admitted("monolithic_mip", 100.0), "pass_simple_test"),
    ("admitted_gap", 100.0, 100.5, admitted("monolithic_mip", 100.0), "reference_gap"),
    ("gap_above_rule", 100.0, 102.0, admitted("monolithic_mip", 100.0), "inadmissible_gap"),
    ("below_optimum", 100.0, 99.0, admitted("monolithic_mip", 100.0), "objective_mismatch"),
    ("truth_differs_from_mip", 100.0, 101.0, admitted("monolithic_mip", 101.0), "objective_mismatch"),
    # Matching golds without admitted metadata are not a pass (the former strict branch accepted them).
    ("no_source", 100.0, 100.0, {"ground_truth_objective": 100.0}, "not_admitted"),
    ("no_meta", 100.0, 100.0, None, "not_admitted"),
    ("certificate_is_not_strict", 100.0, 100.0, admitted("decomposition_certificate", 100.0), "not_admitted"),
    ("mip_only_is_not_strict", 100.0, None, admitted("monolithic_mip", 100.0), "missing_gold_cg"),
])
def test_strict_rule(tmp_path, case, mono, decomposed, meta, status):
    directory = stub(tmp_path / "cg" / "bp" / "bp_small_seed000", "cg", mono, decomposed, meta, decomposition_lb=99.0)
    result = CLI.validate_instance(directory)
    assert (result["admission"], result["status"]) == ("strict", status), result["issues"]
    if status not in PASSING:
        assert result["issues"]


@pytest.mark.parametrize("case,mono,decomposed,meta,status", [
    ("mip_only", 5.0, None, admitted("monolithic_mip", 5.0), "pass_simple_test"),
    ("certificate_only", None, 5.0, admitted("decomposition_certificate", 5.0), "pass_simple_test"),
    ("open_certificate", None, 5.0, admitted("decomposition_certificate", 5.0), "not_admitted"),
    ("certified_reference_differs", 5.0, 6.0, admitted("monolithic_mip", 5.0), "objective_mismatch"),
    ("certificate_contradicts_verified_mip", 5.0, 6.0, admitted("decomposition_certificate", 6.0), "objective_mismatch"),
    ("both", 5.0, 5.0, admitted("monolithic_mip", 5.0), "pass_simple_test"),
    ("certificate_differs", None, 6.0, admitted("decomposition_certificate", 5.0), "objective_mismatch"),
    ("neither", None, None, admitted("monolithic_mip", None), "missing_gold_mono"),
    ("no_source", 5.0, 5.0, {"ground_truth_objective": 5.0}, "not_admitted"),
])
def test_reused_families_rule(tmp_path, case, mono, decomposed, meta, status):
    directory = stub(tmp_path / "cg" / "gcol" / "gcol_small_seed000", "cg", mono, decomposed, meta, decomposition_lb=4.0 if case == "open_certificate" else None)
    result = CLI.validate_instance(directory)
    assert (result["admission"], result["status"]) == ("mip_or_certificate", status), result["issues"]


def test_simple_level_checks_the_admitted_source(tmp_path):
    good = stub(tmp_path / "bd" / "ufl" / "ufl_small_seed000", "bd", 1.0, 1.0, admitted("monolithic_mip", 1.0))
    assert CLI.validate_instance(good, "simple")["status"] == "pass_simple_test"
    bad = stub(tmp_path / "bd" / "ufl" / "ufl_small_seed001", "bd", 1.0, 1.0, {"ground_truth_objective": 1.0})
    assert CLI.validate_instance(bad, "simple")["status"] == "not_admitted"
    paper = stub(tmp_path / "bd" / "tsso" / "tsso_small_seed000", "bd", 1.0, 1.0, {"ground_truth_objective": 1.0})
    assert CLI.validate_instance(paper, "simple")["status"] == "pass_simple_test"


def test_an_unregistered_family_fails(tmp_path):
    directory = stub(tmp_path / "cg" / "hub" / "hub_small_seed000", "cg", 1.0, 1.0, admitted("monolithic_mip", 1.0))
    assert CLI.validate_instance(directory)["status"] == "unregistered_family"


def test_an_unverified_reference_is_discarded_by_the_reused_rule_and_rejected_by_the_strict_rule(tmp_path):
    # instance_suite._admit_mip_or_certificate drops a reference whose verification failed: MIP-only admission, even
    # with a finite objective below the optimum.
    reused = stub(tmp_path / "cg" / "gcol" / "gcol_small_seed000", "cg", 5.0, 4.0, admitted("monolithic_mip", 5.0), decomposition_verified=False)
    assert CLI.validate_instance(reused)["status"] == "pass_simple_test"
    strict = stub(tmp_path / "cg" / "cvrp" / "cvrp_small_seed000", "cg", 5.0, 5.0, admitted("monolithic_mip", 5.0), decomposition_verified=False)
    assert CLI.validate_instance(strict)["status"] == "not_admitted"
    unverified_mip = stub(tmp_path / "cg" / "cvrp" / "cvrp_small_seed001", "cg", 5.0, 5.0, admitted("monolithic_mip", 5.0), mono_verified=False)
    assert CLI.validate_instance(unverified_mip)["status"] == "not_admitted"
    certificate = stub(tmp_path / "cg" / "gcol" / "gcol_small_seed001", "cg", None, 5.0, admitted("decomposition_certificate", 5.0), decomposition_verified=False)
    assert CLI.validate_instance(certificate)["status"] == "not_admitted"


@pytest.mark.parametrize("source", [[], {}, 1])
@pytest.mark.parametrize("level", ["strict", "simple"])
def test_a_malformed_source_is_not_admitted(tmp_path, source, level):
    directory = stub(tmp_path / "cg" / "cvrp" / "cvrp_small_seed000", "cg", 5.0, 5.0, admitted(source, 5.0))
    assert CLI.validate_instance(directory, level)["status"] == "not_admitted"


def test_a_family_under_the_wrong_suite_is_unregistered(tmp_path):
    assert instance_admission(tmp_path / "cg/ufl/ufl_small_seed000") is None
    directory = stub(tmp_path / "cg" / "ufl" / "ufl_small_seed000", "cg", 5.0, 5.0, admitted("monolithic_mip", 5.0))
    assert CLI.validate_instance(directory)["status"] == "unregistered_family"


def test_the_registered_rules_use_the_family_tolerance(tmp_path):
    # 4.99995 is within the paper rule's historical 1e-4 but 1e-5 below the optimum, beyond the strict tol 1e-6.
    strict = stub(tmp_path / "cg" / "cvrp" / "cvrp_small_seed000", "cg", 5.0, 4.99995, admitted("monolithic_mip", 5.0))
    assert CLI.validate_instance(strict)["status"] == "objective_mismatch"
    within = stub(tmp_path / "cg" / "cvrp" / "cvrp_small_seed001", "cg", 5.0, 5.0 - 5e-7, admitted("monolithic_mip", 5.0))
    assert CLI.validate_instance(within)["status"] == "pass_simple_test"
    paper = stub(tmp_path / "cg" / "shift" / "shift_small_seed000", "cg", 5.0, 4.99995, {"ground_truth_objective": 5.0})
    assert CLI.validate_instance(paper)["status"] == "pass_simple_test"


def test_the_strict_rule_applies_the_family_certificate_requirement(tmp_path):
    # CVRP (reference_certified): admission needs a certified reference matching the MIP optimum.
    exact = stub(tmp_path / "cg" / "cvrp" / "cvrp_small_seed000", "cg", 100.0, 100.0, admitted("monolithic_mip", 100.0))
    assert CLI.validate_instance(exact)["status"] == "pass_simple_test"
    gap = stub(tmp_path / "cg" / "cvrp" / "cvrp_small_seed001", "cg", 100.0, 100.5, admitted("monolithic_mip", 100.0), decomposition_lb=99.0)
    assert CLI.validate_instance(gap)["status"] == "not_admitted"
    uncertified = stub(tmp_path / "cg" / "cvrp" / "cvrp_small_seed002", "cg", 100.0, 100.0, admitted("monolithic_mip", 100.0), decomposition_lb=99.0)
    assert CLI.validate_instance(uncertified)["status"] == "not_admitted"
    slow = stub(tmp_path / "cg" / "bp" / "bp_small_seed000", "cg", 100.0, 100.0, admitted("monolithic_mip", 100.0))
    reference = json.loads((slow / "gold_cg.json").read_text())
    (slow / "gold_cg.json").write_text(json.dumps({**reference, "total_time": 61.0}))
    assert CLI.validate_instance(slow)["status"] == "not_admitted"  # reference above the 60 s limit


@pytest.mark.parametrize("gold_name,field,value", [("gold_cg.json", "total_time", None), ("gold_cg.json", "total_time", "unknown"),
                                                   ("gold_monolithic.json", "total_time", "unknown"), ("gold_monolithic.json", "runtime", "x")])
def test_malformed_timing_fails_the_instance_not_the_batch(tmp_path, gold_name, field, value):
    bad = stub(tmp_path / "cg" / "bp" / "bp_small_seed000", "cg", 5.0, 5.0, admitted("monolithic_mip", 5.0))
    payload = json.loads((bad / gold_name).read_text())
    if field == "runtime":
        payload.pop("total_time")  # the MIP time falls back to runtime only without total_time
    (bad / gold_name).write_text(json.dumps({**payload, field: value}))
    stub(tmp_path / "cg" / "bp" / "bp_small_seed001", "cg", 5.0, 5.0, admitted("monolithic_mip", 5.0))
    results = {r["name"]: r["status"] for r in CLI.validate_all(tmp_path)}
    assert results["bp_small_seed001"] == "pass_simple_test"
    assert results["bp_small_seed000"] in {"invalid_reference", "not_admitted"}


@pytest.mark.parametrize("level", ["strict", "simple"])
def test_an_unreadable_meta_fails_the_instance_not_the_batch(tmp_path, level):
    bad = stub(tmp_path / "cg" / "bp" / "bp_small_seed000", "cg", 5.0, 5.0, None)
    (bad / "meta.json").mkdir()  # exists, but reading it raises IsADirectoryError (an OSError)
    stub(tmp_path / "cg" / "bp" / "bp_small_seed001", "cg", 5.0, 5.0, admitted("monolithic_mip", 5.0))
    results = {r["name"]: r for r in CLI.validate_all(tmp_path, validation_level=level)}
    assert results["bp_small_seed001"]["status"] == "pass_simple_test"
    assert results["bp_small_seed000"]["status"] not in PASSING and any("meta.json" in issue for issue in results["bp_small_seed000"]["issues"])


def test_a_failing_instance_never_aborts_the_batch(tmp_path, monkeypatch):
    stub(tmp_path / "cg" / "bp" / "bp_small_seed000", "cg", 5.0, 5.0, admitted("monolithic_mip", 5.0))
    stub(tmp_path / "cg" / "bp" / "bp_small_seed001", "cg", 5.0, 5.0, admitted("monolithic_mip", 5.0))
    original = CLI.validate_instance

    def flaky(path, validation_level="strict"):
        if path.name == "bp_small_seed000":
            raise OSError("read failed")
        return original(path, validation_level)

    monkeypatch.setattr(CLI, "validate_instance", flaky)
    results = {r["name"]: r["status"] for r in CLI.validate_all(tmp_path)}
    assert results == {"bp_small_seed000": "invalid_reference", "bp_small_seed001": "pass_simple_test"}
