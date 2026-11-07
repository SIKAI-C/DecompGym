"""Check the contract actually rendered and delivered, including reflection."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.prompts.judge import algorithm_review_prompt
from src.agent.prompts.method import COMMON_REQUIREMENTS, METHOD_HEADER, METHOD_REQUIREMENTS
from src.utils.output_contract import validate_output_contract
from src.experiments.paired_runner import PairedRunner
from src.agent.prompts.builder import PromptBuilder


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


# The versioned contract: the paper's v2 text stays the default and byte-identical; v3 changes one Benders sentence.
from src.agent.prompts.method import CONTRACTS, DECOMPOSITION_POLICY, DECOMPOSITION_POLICY_V3, decomposition_contract, method_requirements
from src.agent.prompts.snapshot import judge_prompt_hashes, render_case_prompt, snapshot_prompt
from src.experiments.data_context import accepted_prompt_protocol, decomposition_contract_policy
from src.experiments.plan import load_plan
from src.utils.errors import ProtocolError
import hashlib

V2_BD_OPENING = "Use a master problem for the complicating decisions and recourse-cost estimates."
V3_BD_OPENING = "Use a master problem for the complicating decisions and, when applicable, recourse-cost estimates."
# Judge prompts recorded before contracts were versioned (algorithm_judge_prompt_sha256 of the v2 protocol).
V2_JUDGE_SHA256 = {"bd": "7f2202c7d047679b9771e58dbdfdff9fd8c0c4b12f9df268f52995b27fd1550f",
                   "cg": "733c78626f66c3bfd78b3b42b49b282ca9fbb2f2bf5bb5ac466cb8c81a7046e7"}
PAPER_BD_CASE = ROOT / "data/bd/ufl/ufl_small_seed000"
PAPER_CG_CASE = ROOT / "data/cg/csp/csp_small_seed000"


def test_v2_contract_is_the_default_and_byte_identical_to_the_paper():
    assert DECOMPOSITION_POLICY == "strict-decomposition-20261013-v2" and CONTRACTS[DECOMPOSITION_POLICY] is METHOD_REQUIREMENTS
    assert METHOD_REQUIREMENTS["bd"].startswith(V2_BD_OPENING)
    for suite in ("bd", "cg"):
        assert method_requirements(suite) == method_requirements(suite, DECOMPOSITION_POLICY)
        for contract in ((), (DECOMPOSITION_POLICY,)):
            assert hashlib.sha256(algorithm_review_prompt(suite, *contract).encode()).hexdigest() == V2_JUDGE_SHA256[suite]
    assert judge_prompt_hashes() == judge_prompt_hashes(DECOMPOSITION_POLICY) == V2_JUDGE_SHA256
    # The published paper prompt.txt is reproduced byte for byte by the default and the explicit v2 renderer.
    for case in (PAPER_BD_CASE, PAPER_CG_CASE):
        suite, problem = case.relative_to(ROOT / "data").parts[:2]
        data = json.loads((case / "data.json").read_text())
        stored = (case / "prompt.txt").read_text()
        assert render_case_prompt(suite, problem, data) == render_case_prompt(suite, problem, data, DECOMPOSITION_POLICY) == stored


def test_v3_contract_differs_only_in_the_recourse_cost_sentence():
    assert DECOMPOSITION_POLICY_V3 == "strict-decomposition-20261106-v3"
    assert method_requirements("bd", DECOMPOSITION_POLICY_V3) == method_requirements("bd").replace(V2_BD_OPENING, V3_BD_OPENING)
    assert method_requirements("bd", DECOMPOSITION_POLICY_V3) != method_requirements("bd")
    assert method_requirements("cg", DECOMPOSITION_POLICY_V3) == method_requirements("cg")
    assert algorithm_review_prompt("bd", DECOMPOSITION_POLICY_V3) == algorithm_review_prompt("bd").replace(V2_BD_OPENING, V3_BD_OPENING)
    assert algorithm_review_prompt("cg", DECOMPOSITION_POLICY_V3) == algorithm_review_prompt("cg")
    for case in (PAPER_BD_CASE, PAPER_CG_CASE):
        suite, problem = case.relative_to(ROOT / "data").parts[:2]
        stored = (case / "prompt.txt").read_text()
        rendered = render_case_prompt(suite, problem, json.loads((case / "data.json").read_text()), DECOMPOSITION_POLICY_V3)
        assert rendered == (stored.replace(V2_BD_OPENING, V3_BD_OPENING) if suite == "bd" else stored)
        assert rendered.count(METHOD_HEADER) == 1
        # Reflection is built from the generation prompt and so carries the same contract text.
        reflection = PromptBuilder.build_reflection_prompt(PromptBuilder.build_generation_prompt(rendered, solver_timeout=60, solver_threads=1))
        assert method_requirements(suite, DECOMPOSITION_POLICY_V3) in reflection
    # The renderer context is restored after the block.
    with decomposition_contract(DECOMPOSITION_POLICY_V3):
        assert method_requirements("bd") == method_requirements("bd", DECOMPOSITION_POLICY_V3)
    assert method_requirements("bd") == method_requirements("bd", DECOMPOSITION_POLICY)


def test_snapshot_prompt_renders_the_selected_contract(tmp_path):
    for contract in (DECOMPOSITION_POLICY, DECOMPOSITION_POLICY_V3):
        destination = tmp_path / contract
        destination.mkdir()
        for name in ("data.json", "prompt.txt"):
            (destination / name).write_bytes((PAPER_BD_CASE / name).read_bytes())
        snapshot_prompt(destination, "bd", "ufl", contract)
        assert method_requirements("bd", contract) in (destination / "prompt.txt").read_text()
        assert ((destination / "prompt.txt").read_bytes() == (PAPER_BD_CASE / "prompt.txt").read_bytes()) == (contract == DECOMPOSITION_POLICY)


@pytest.mark.parametrize("config,contract", [("paper.json", DECOMPOSITION_POLICY), ("harder_small_all.json", DECOMPOSITION_POLICY), ("small_scale.json", DECOMPOSITION_POLICY_V3)])
def test_config_selects_the_contract(config, contract):
    assert decomposition_contract_policy(load_plan(ROOT / "configs" / config)["prompt_protocol"]) == contract


def test_runner_judge_uses_the_selected_contract():
    assert PairedRunner(client=None, judge_client=None, solver_executor=None).decomposition_contract == DECOMPOSITION_POLICY
    assert PairedRunner(client=None, judge_client=None, solver_executor=None, decomposition_contract=DECOMPOSITION_POLICY_V3).decomposition_contract == DECOMPOSITION_POLICY_V3


def test_unknown_contract_is_rejected(tmp_path):
    unknown = "strict-decomposition-unknown"
    with pytest.raises(ProtocolError, match="Unknown decomposition contract"):
        method_requirements("bd", unknown)
    with pytest.raises(ProtocolError):
        algorithm_review_prompt("bd", unknown)
    with pytest.raises(ProtocolError):
        with decomposition_contract(unknown):
            pass
    with pytest.raises(ProtocolError):
        PairedRunner(client=None, judge_client=None, solver_executor=None, decomposition_contract=unknown)
    plan = load_plan(ROOT / "configs/small_scale.json")
    plan["prompt_protocol"] = dict(plan["prompt_protocol"], decomposition_contract=unknown)
    assert not accepted_prompt_protocol(plan["prompt_protocol"])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ProtocolError):
        load_plan(path)


def test_malformed_contract_values_are_rejected_cleanly():
    from src.agent.prompts.method import require_contract
    for value in ([], {}, None, 3):
        with pytest.raises(ProtocolError):
            require_contract(value)
        assert not accepted_prompt_protocol({**load_plan(ROOT / "configs/paper.json")["prompt_protocol"], "decomposition_contract": value})


def test_renderer_context_is_restored_after_an_exception_and_nesting():
    with pytest.raises(RuntimeError):
        with decomposition_contract(DECOMPOSITION_POLICY_V3):
            raise RuntimeError("render failure")
    assert method_requirements("bd") == method_requirements("bd", DECOMPOSITION_POLICY)
    with decomposition_contract(DECOMPOSITION_POLICY_V3):
        with decomposition_contract(DECOMPOSITION_POLICY):
            assert method_requirements("bd") == method_requirements("bd", DECOMPOSITION_POLICY)
        assert method_requirements("bd") == method_requirements("bd", DECOMPOSITION_POLICY_V3)


def test_dataset_prompts_are_rerendered_under_the_selected_contract(tmp_path):
    from src.agent.prompts.snapshot import render_dataset_prompts
    for case in (PAPER_BD_CASE, PAPER_CG_CASE):
        target = tmp_path / case.relative_to(ROOT / "data")
        target.mkdir(parents=True)
        for name in ("data.json", "prompt.txt"):
            (target / name).write_bytes((case / name).read_bytes())
    assert render_dataset_prompts(tmp_path) == {"contract": DECOMPOSITION_POLICY, "instances": 2, "rewritten": 0}
    assert render_dataset_prompts(tmp_path, DECOMPOSITION_POLICY_V3) == {"contract": DECOMPOSITION_POLICY_V3, "instances": 2, "rewritten": 1}
    bd = (tmp_path / PAPER_BD_CASE.relative_to(ROOT / "data") / "prompt.txt").read_text()
    assert bd == (PAPER_BD_CASE / "prompt.txt").read_text().replace(V2_BD_OPENING, V3_BD_OPENING)


@pytest.mark.parametrize("contract", [DECOMPOSITION_POLICY, DECOMPOSITION_POLICY_V3])
def test_runner_delivers_the_selected_contract_to_the_judge(tmp_path, monkeypatch, contract):
    from contextlib import nullcontext
    from host_solver import host_solver
    from test_native_paired import Client

    class PromptClient(Client):
        def edit_file(self, target, prompt, *, context, log_dir, step):
            self.prompts.append((step, prompt))
            return super().edit_file(target, prompt, context=context, log_dir=log_dir, step=step)

    monkeypatch.setattr("src.experiments.paired_runner.runtime_lock", lambda name: nullcontext())
    source = tmp_path / "data/bd/ufl/ufl_small_seed000"
    source.mkdir(parents=True)
    (source / "data.json").write_text(json.dumps({"m": 2, "n": 1, "f": [2, 3], "D": [[4], [1]]}))
    (source / "prompt.txt").write_text("Implement Benders decomposition.")
    (source / "meta.json").write_text('{"ground_truth_objective":4}')
    client = PromptClient()
    client.prompts = []
    PairedRunner(client=client, judge_client=client, model="fixture", suite="bd", problem=["ufl"], solver_executor=host_solver, decomposition_contract=contract).run_pair(source, tmp_path / "runs")
    judged = [prompt for step, prompt in client.prompts if step == "llm_verifier"]
    assert judged and all(prompt == algorithm_review_prompt("bd", contract) for prompt in judged)
    # A finished pair cannot be resumed under another contract.
    other = DECOMPOSITION_POLICY_V3 if contract == DECOMPOSITION_POLICY else DECOMPOSITION_POLICY
    with pytest.raises(ProtocolError, match="Case protocol changed"):
        PairedRunner(client=client, judge_client=client, model="fixture", suite="bd", problem=["ufl"], solver_executor=host_solver, decomposition_contract=other).run_pair(source, tmp_path / "runs")
