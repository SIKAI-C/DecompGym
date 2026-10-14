"""Reference data and incumbent-plan consistency checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.agent.output_verifier import OutputVerifier
from src.utils.base_bd_class import BaseBenders, MasterSolution, SubproblemResult
from src.utils.reference_output import reference_output


ROOT = Path(__file__).resolve().parents[1]
REFERENCES = sorted((ROOT / "data").glob("*/*/*/gold*.json"))


def test_educational_dataset_is_complete():
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    assert manifest["num_instances"] == 81
    assert len(REFERENCES) == 162
    for instance in manifest["instances"]:
        assert instance["complete"]
        path = ROOT / "data" / instance["suite"] / instance["problem"] / instance["instance"] / "data.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == instance["data_sha256"]


@pytest.mark.parametrize("path", REFERENCES, ids=lambda path: str(path.relative_to(ROOT / "data")))
def test_saved_reference_is_feasible_and_objective_consistent(path, tmp_path):
    suite, problem = path.relative_to(ROOT / "data").parts[:2]
    data = json.loads((path.parent / "data.json").read_text())
    result = json.loads(path.read_text())
    output = tmp_path / "output.json"
    output.write_text(json.dumps(reference_output(suite, problem, data, result)))
    verified = OutputVerifier.verify_reference_output(suite, problem, path.parent / "data.json", output, 1e-6)
    assert verified["ok"], verified["issues"]


def test_reference_validation_does_not_relax_the_agent_contract(tmp_path):
    from src.api import verify

    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps({"m": 1, "n": 1, "f": [2], "D": [[3]]}))
    output = {"objective_value": 5, "y": [1], "x": [[1]]}
    output_path = tmp_path / "output.json"
    output_path.write_text(json.dumps(output))
    assert verify("bd", "ufl", data_path, output)["ok"]
    strict = OutputVerifier.verify_output("bd", "ufl", data_path, output_path, 1e-6)
    assert not strict["ok"] and strict["failure_kind"] == "output_contract"
    output["objective_value"] = 4
    assert not verify("bd", "ufl", data_path, output)["ok"]


@pytest.mark.parametrize("final_state", ["worse", "infeasible", "no_integer_recourse"])
def test_benders_exports_best_incumbent_and_keeps_final_trace(final_state):
    class ToyBenders(BaseBenders):
        def __init__(self):
            super().__init__(max_iters=2)
            self.shared = {"decision": [0]}
        def load_instance(self, name):
            return None
        def build_initial_master(self, inst):
            return None
        def solve_master(self, inst, master):
            self.shared["decision"][0] += 1
            return MasterSolution(float(self.shared["decision"][0]), self.shared, None, {"label": self.shared["decision"][0]})
        def solve_subproblems(self, inst, master):
            first = master.vars["decision"][0] == 1
            return SubproblemResult([], [1], first or final_state != "infeasible", 1, 10 if first else 20, {"plan": master.vars["decision"][0]})
        def compute_iteration_bounds(self, inst, master, sub):
            self.global_lb = master.objective
            candidate = 10 if master.vars["decision"][0] == 1 else (20 if final_state == "worse" else float("inf"))
            self.global_ub = min(self.global_ub, candidate)
            return self.global_lb, self.global_ub
        def should_terminate(self, *args):
            return False
        def add_feasibility_cuts(self, *args):
            pass
        def add_optimality_cuts(self, *args):
            pass
    result = ToyBenders().run("unused")
    assert result["best_objective"] == 10
    assert result["final_vars"] == {"decision": [1]}
    assert result["master_metadata"] == {"label": 1}
    assert result["sub_metadata"] == {"plan": 1}
    assert result["incumbent_iteration"] == 1
    assert result["final_master_objective"] == 2
    assert result["global_lb"] == 2 and result["global_ub"] == 10
    assert len(result["iter_records"]) == 2
