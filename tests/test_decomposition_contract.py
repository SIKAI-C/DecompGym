"""Check the contract actually rendered and delivered, including reflection."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.algorithm_verifier import algorithm_review_prompt
from src.agent.decomposition_policy import COMMON_REQUIREMENTS, METHOD_HEADER, METHOD_REQUIREMENTS
from src.agent.output_contract import validate_output_contract
from src.agent.paired_runner import PairedRunner
from src.agent.prompt_builder import PromptBuilder


ROOT = Path(__file__).resolve().parents[1]
CASES = sorted((ROOT / "data").glob("*/*/*/data.json"))


@pytest.mark.parametrize("data_path", CASES, ids=lambda path: path.parent.name)
def test_all_fixed_instances_render_the_same_generation_reflection_judge_contract(data_path, tmp_path):
    suite, problem = data_path.relative_to(ROOT / "data").parts[:2]
    prompt = importlib.import_module(f"src.{suite}.{problem}.b_problem").render_prompt(json.loads(data_path.read_text()))
    (tmp_path / "prompt.txt").write_text(prompt)
    generation = PairedRunner.generation_prompt(SimpleNamespace(suite=suite, solver_timeout=60, solver_threads=1), tmp_path)
    reflection = PromptBuilder.build_reflection_prompt(generation)
    judge = algorithm_review_prompt(suite)
    # The published instance prompt is complete before any runner sees it.
    for text in (prompt, generation, reflection, judge):
        assert text.count(METHOD_HEADER) == 1
        assert METHOD_REQUIREMENTS[suite] in text
        assert all(rule in text for rule in COMMON_REQUIREMENTS)
        assert '# or "monolithic"' not in text
    other = "cg" if suite == "bd" else "bd"
    assert METHOD_REQUIREMENTS[other] not in judge
    assert "NumPy" in generation and "60 seconds" in generation
    assert "last subproblem solution" not in prompt
    assert generation.startswith(prompt.rstrip())
    assert (tmp_path / "prompt.txt").read_text() == prompt
    assert "read-only context for this editing call" in generation
    assert "separate runtime directory" in generation and "../../prompt.txt" not in generation


def test_new_fctp_contract_rejects_monolithic_label():
    data = {"m": 1, "n": 1}
    output = {"status": "feasible", "method": "benders_decomposition", "objective_value": 1, "y": [[1]], "x": [[1]]}
    assert not validate_output_contract("bd", "fctp", data, output)
    output["method"] = "monolithic"
    assert validate_output_contract("bd", "fctp", data, output) == ["method must be a method permitted by the problem prompt"]
