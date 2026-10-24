"""The SHIFT verifier reads only the documented start, end and usage fields."""
from src.cg.shift.d_verifier import verify_solution


DATA = {"demand": [1], "shift_min_length": 1, "shift_max_length": 1}


def output(shift):
    return {"status": "feasible", "objective_value": 1, "solution": {"shifts": [shift]}}


def test_documented_fields_pass():
    result = verify_solution(DATA, output({"start": 0, "end": 0, "usage": 1}))
    assert result["ok"], result


def test_conflicting_old_field_names_cannot_override_the_documented_plan():
    # Codex's reproduction: an impossible documented shift plus valid old-style fields used to pass.
    assert not verify_solution(DATA, output({"start": 99, "end": 98, "usage": 0, "start_slot": 0, "end_slot": 0, "count": 1}))["ok"]
    assert not verify_solution(DATA, output({"start_slot": 0, "end_slot": 0, "count": 1}))["ok"]
