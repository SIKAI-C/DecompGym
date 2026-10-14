"""Render the reviewed prompts without changing fixed data or historical prompts."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

from src.agent.algorithm_verifier import algorithm_review_prompt
from src.agent.decomposition_policy import DECOMPOSITION_POLICY
from src.agent.output_contract import OUTPUT_CONTRACT_POLICY
from src.agent.prompt_policy import REFLECTION_POLICY


PROMPT_PROTOCOL = {
    "decomposition_contract": DECOMPOSITION_POLICY,
    "problem_templates": "instance-bound-decomposition-prompts-20261013-v3",
    "runtime_location": "separate-edit-runtime-v1",
    "algorithm_judge": "strict-decomposition-source-review-v2",
    "output_contract": OUTPUT_CONTRACT_POLICY,
    "reflection_delivery": REFLECTION_POLICY,
}


def render_case_prompt(suite, problem, data):
    renderer = importlib.import_module(f"src.{suite}.{problem}.b_problem")
    return renderer.render_prompt(data).rstrip() + "\n"


def snapshot_prompt(destination: Path, suite: str, problem: str):
    original = (destination / "prompt.txt").read_bytes()
    (destination / "prompt.original.txt").write_bytes(original)
    prompt = render_case_prompt(suite, problem, json.loads((destination / "data.json").read_text()))
    # The copy belongs to the new cohort; the source dataset is never changed.
    (destination / "prompt.txt").write_text(prompt)
    return {"policy": PROMPT_PROTOCOL["problem_templates"], "original_sha256": hashlib.sha256(original).hexdigest(), "effective_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


def judge_prompt_hashes():
    return {suite: hashlib.sha256(algorithm_review_prompt(suite).encode()).hexdigest() for suite in ("bd", "cg")}
