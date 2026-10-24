"""Malformed JSON is a scored candidate failure, never a validator recovery loop."""
from copy import deepcopy
import importlib
import json
from types import SimpleNamespace

import pytest

from src.utils.output_contract import validate_output_contract
from src.experiments.verification import OutputVerifier, VerificationError


CASES = {
    "bp": ({"bin_capacity": 1, "items": [{"id": 0, "size": 1}]}, {"bins": [{"usage": 1, "contents": [{"item_id": 0, "count": 1}]}]}),
    "csp": ({"stock_size": 1, "item_types": [{"id": 0, "size": 1, "demand": 1}]}, {"patterns": [{"pattern_id": 0, "usage": 1, "cuts": [{"item_id": 0, "count": 1}]}]}),
    "gap": ({"machines": [{"id": 0, "capacity": 1}], "jobs": [{"id": 0, "assignments": [{"machine_id": 0, "resource": 1, "cost": 1}]}]}, {"assignments": [{"job_id": 0, "machine_id": 0}]}),
    "shift": ({"demand": [1], "shift_min_length": 1, "shift_max_length": 1}, {"shifts": [{"start": 0, "end": 0, "usage": 1}]}),
}


def fixture(problem):
    data, solution = deepcopy(CASES[problem])
    return data, {"status": "feasible", "method": "column_generation", "objective_value": 1, "solution": solution}


def verify(tmp_path, problem, data, output):
    (tmp_path / "data.json").write_text(json.dumps(data))
    (tmp_path / "output.json").write_text(json.dumps(output, indent=3, sort_keys=True))
    return OutputVerifier.verify_output("cg", problem, tmp_path / "data.json", tmp_path / "output.json", 1e-6)


@pytest.mark.parametrize("problem", CASES)
@pytest.mark.parametrize("malformed", [None, [], True, "invalid", 3])
def test_all_cg_entrypoints_reject_malformed_solution_containers(tmp_path, problem, malformed):
    data, output = fixture(problem)
    output["solution"] = malformed
    result = verify(tmp_path, problem, data, output)
    assert not result["ok"] and result["failure_kind"] == "output_contract"
    direct = importlib.import_module(f"src.cg.{problem}.d_verifier")
    assert not direct.verify_solution(data, output)["ok"]


@pytest.mark.parametrize("problem", CASES)
def test_required_structure_and_fields_are_enforced_without_formatting_constraints(tmp_path, problem):
    data, valid = fixture(problem)
    assert verify(tmp_path, problem, data, valid | {"extra": {"anything": 1}})["ok"]
    for field in valid:
        missing = deepcopy(valid)
        del missing[field]
        assert not verify(tmp_path, problem, data, missing)["ok"], field
    collection = next(iter(valid["solution"]))
    for malformed in [None, {}, "bad", [None], [True], [[]]]:
        output = deepcopy(valid)
        output["solution"][collection] = malformed
        assert not verify(tmp_path, problem, data, output)["ok"]
    entry = valid["solution"][collection][0]
    for field in entry:
        output = deepcopy(valid)
        del output["solution"][collection][0][field]
        assert not verify(tmp_path, problem, data, output)["ok"], field


@pytest.mark.parametrize("problem", CASES)
@pytest.mark.parametrize("bad", [0.5, True, "0", float("inf"), float("nan"), None])
def test_integer_identifiers_cannot_be_truncated_or_coerced(tmp_path, problem, bad):
    data, output = fixture(problem)
    if problem in {"bp", "csp"}:
        key, inner = ("bins", "contents") if problem == "bp" else ("patterns", "cuts")
        output["solution"][key][0][inner][0]["item_id"] = bad
    elif problem == "gap":
        output["solution"]["assignments"][0]["job_id"] = bad
    else:
        output["solution"]["shifts"][0]["start"] = bad
    assert not verify(tmp_path, problem, data, output)["ok"]
    direct = importlib.import_module(f"src.cg.{problem}.d_verifier")
    assert not direct.verify_solution(data, output)["ok"]


def test_zero_usage_does_not_hide_malformed_nested_fields(tmp_path):
    data, output = fixture("csp")
    output["solution"]["patterns"].append({"pattern_id": 1, "usage": 0, "cuts": [{"item_id": 0.5, "count": 1}]})
    assert not verify(tmp_path, "csp", data, output)["ok"]


def test_unexpected_verifier_defect_is_not_a_candidate_error(tmp_path, monkeypatch):
    data, output = fixture("bp")
    def crash(*args, **kwargs):
        raise AttributeError("trusted verifier defect")
    monkeypatch.setattr(OutputVerifier, "import_verifier", lambda *_: (SimpleNamespace(verify_solution=crash), None))
    with pytest.raises(VerificationError, match="trusted verifier defect"):
        verify(tmp_path, "bp", data, output)


@pytest.mark.parametrize("field", ["status", "method", "objective_value"])
@pytest.mark.parametrize("bad", [[], {}, True, None])
def test_scalar_contract_fields_reject_wrong_json_types(field, bad):
    data, output = fixture("bp")
    output[field] = bad
    assert validate_output_contract("cg", "bp", data, output)
