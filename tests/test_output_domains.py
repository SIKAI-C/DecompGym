"""Final decision domains follow each problem's model-visible contract."""
import json

import pytest

from src.agent.output_verifier import OutputVerifier
from src.bd.lrp.b_problem import render_prompt


def verify(tmp_path, problem, data, output):
    (tmp_path / "data.json").write_text(json.dumps(data))
    (tmp_path / "output.json").write_text(json.dumps({"status": "feasible", "method": "benders_decomposition", **output}))
    return OutputVerifier.verify_output("bd", problem, tmp_path / "data.json", tmp_path / "output.json", 1e-6)


def test_ufl_fractional_assignment_fails_but_kmedian_retains_fractional_contract(tmp_path):
    data = {"m": 2, "n": 1, "f": [2, 3], "D": [[4], [1]], "k": 2}
    output = {"y": [1, 1], "x": [[0.5], [0.5]], "objective_value": 7.5}
    result = verify(tmp_path, "ufl", data, output)
    assert not result["ok"] and any("binary" in issue for issue in result["issues"])
    assert verify(tmp_path, "km", data, output | {"objective_value": 2.5})["ok"]
    assert verify(tmp_path, "ufl", data, output | {"x": [[0], [1]], "objective_value": 6})["ok"]


LRP_DATA = {"m": 1, "n": 1, "R": 1, "route_depot": [0], "incidence": [[1]], "max_vehicles": [2], "depot_cost": [2], "route_cost": [3], "routes": [[0]]}


@pytest.mark.parametrize("y", [[0.5], [float("nan")], [float("inf")], [], [1, 1]])
def test_lrp_rejects_nonbinary_nonfinite_and_wrong_length_depot_decisions(tmp_path, y):
    assert not verify(tmp_path, "lrp", LRP_DATA, {"y": y, "lambda": [1], "objective_value": 5})["ok"]


def test_lrp_prompt_domains_and_feasible_plan_agree(tmp_path):
    prompt = render_prompt(LRP_DATA)
    assert '"y": [0/1 list of length m indicating open depots]' in prompt
    assert "1 means open and 0 means closed" in prompt
    assert verify(tmp_path, "lrp", LRP_DATA, {"y": [1], "lambda": [1], "objective_value": 5})["ok"]
    for data, output in [
        (LRP_DATA, {"y": [1], "lambda": [0.5], "objective_value": 3.5}),
        (LRP_DATA, {"y": [1], "lambda": [0], "objective_value": 2}),
        (LRP_DATA | {"max_vehicles": [0]}, {"y": [1], "lambda": [1], "objective_value": 5}),
    ]:
        assert not verify(tmp_path, "lrp", data, output)["ok"]
