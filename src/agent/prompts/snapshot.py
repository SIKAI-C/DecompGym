"""Render the reviewed prompts without changing fixed data or historical prompts."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

from src.agent.prompts.judge import ALGORITHM_JUDGE, algorithm_review_prompt
from src.agent.prompts.method import DECOMPOSITION_POLICY, decomposition_contract
from src.utils.output_contract import OUTPUT_CONTRACT_POLICY
from src.agent.prompts.policy import REFLECTION_POLICY


PROMPT_PROTOCOL = {
    "decomposition_contract": DECOMPOSITION_POLICY,
    "problem_templates": "instance-bound-decomposition-prompts-20261013-v3",
    "runtime_location": "separate-edit-runtime-v1",
    "algorithm_judge": ALGORITHM_JUDGE,
    "output_contract": OUTPUT_CONTRACT_POLICY,
    "reflection_delivery": REFLECTION_POLICY,
}


def render_case_prompt(suite, problem, data, contract=DECOMPOSITION_POLICY):
    renderer = importlib.import_module(f"src.{suite}.{problem}.b_problem")
    with decomposition_contract(contract):
        return renderer.render_prompt(data).rstrip() + "\n"


def snapshot_prompt(destination: Path, suite: str, problem: str, contract=DECOMPOSITION_POLICY):
    original = (destination / "prompt.txt").read_bytes()
    (destination / "prompt.original.txt").write_bytes(original)
    prompt = render_case_prompt(suite, problem, json.loads((destination / "data.json").read_text()), contract)
    # The copy belongs to the new cohort; the source dataset is never changed.
    (destination / "prompt.txt").write_text(prompt)
    return {"policy": PROMPT_PROTOCOL["problem_templates"], "original_sha256": hashlib.sha256(original).hexdigest(), "effective_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


def render_dataset_prompts(data: Path, contract=DECOMPOSITION_POLICY):
    """Rewrite data/<suite>/<problem>/<instance>/prompt.txt under one contract (the dataset writers render v2).

    Prompt-only: data, references and metadata are untouched. Run it before pinning a dataset inventory for a config
    whose decomposition_contract is not the paper's. Returns the number of instances and of rewritten prompts."""
    total = changed = 0
    for path in sorted(Path(data).glob("*/*/*/data.json")):
        suite, problem = path.relative_to(data).parts[:2]
        prompt = render_case_prompt(suite, problem, json.loads(path.read_text()), contract)
        total += 1
        if (path.parent / "prompt.txt").read_text() != prompt:
            (path.parent / "prompt.txt").write_text(prompt)
            changed += 1
    return {"contract": contract, "instances": total, "rewritten": changed}


def judge_prompt_hashes(contract=DECOMPOSITION_POLICY, extensions=(), target=None):
    """sha256 of each suite's single-file judge prompt (with the campaign's audit target and extensions, if any)."""
    options = {"extensions": extensions} | ({"target": target} if target is not None else {})
    return {suite: hashlib.sha256(algorithm_review_prompt(suite, contract, **options).encode()).hexdigest() for suite in ("bd", "cg")}
