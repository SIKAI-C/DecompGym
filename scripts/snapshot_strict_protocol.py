"""Snapshot prospective prompts and reference source hashes without creating a campaign."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    controller, destination = args.controller.resolve(), args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(controller))
    from src.agent.algorithm_verifier import algorithm_review_prompt
    from src.agent.paired_runner import PairedRunner
    from src.agent.prompt_builder import PromptBuilder
    from src.prompt_snapshot import PROMPT_PROTOCOL
    from src.utils.reference_runtime import REFERENCE_POLICY

    sources = set(controller.glob("src/bd/*/*gold_bd.py")) | set(controller.glob("src/cg/*/*gold_cg.py")) | set(controller.glob("src/*/*/b_problem.py"))
    sources.update(controller / name for name in ("src/agent/decomposition_policy.py", "src/agent/algorithm_verifier.py", "src/agent/paired_runner.py", "src/agent/experiment_runner.py", "src/agent/prompt_builder.py", "src/agent/prompt_policy.py", "src/agent/output_contract.py", "src/prompt_snapshot.py", "src/utils/reference_runtime.py", "src/utils/base_bd_class.py", "src/utils/base_cg_class.py", "src/utils/reference_output.py", "src/api.py"))
    source_hashes = {}
    for path in sorted(sources):
        relative = path.relative_to(controller)
        target = destination / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source_hashes[str(relative)] = digest(path)
    judge_hashes = {}
    for suite in ("bd", "cg"):
        path = destination / f"judge_{suite}.txt"
        path.write_text(algorithm_review_prompt(suite))
        judge_hashes[suite] = digest(path)
    cases = []
    for path in sorted((REPO / "data").glob("*/*/*/data.json")):
        suite, problem, instance = path.relative_to(REPO / "data").parts[:3]
        directory = destination / "prompts" / suite / problem / instance
        directory.mkdir(parents=True)
        module = importlib.import_module(f"src.{suite}.{problem}.b_problem")
        (directory / "prompt.txt").write_text(module.render_prompt(json.loads(path.read_text())).rstrip() + "\n")
        generation = PairedRunner.generation_prompt(SimpleNamespace(suite=suite, solver_timeout=60, solver_threads=1), directory)
        (directory / "generation.txt").write_text(generation)
        (directory / "reflection.txt").write_text(PromptBuilder.build_reflection_prompt(generation))
        cases.append({"suite": suite, "problem": problem, "instance": instance, "data_sha256": digest(path), "sha256": {name: digest(directory / name) for name in ("prompt.txt", "generation.txt", "reflection.txt")}})
    assert len(cases) == 81
    plan = REPO / "docs/2026-10-13-strict-decomposition-experiment-plan.json"
    value = {"prompt_protocol": PROMPT_PROTOCOL, "reference_policy": REFERENCE_POLICY, "controller_source": str(controller), "source_sha256": source_hashes, "judge_prompt_sha256": judge_hashes, "plan_sha256": digest(plan), "cases": cases, "campaign_created": False, "model_calls": 0, "scope": "Prepared source and pre-workspace generation/reflection prompts; runtime workspace delivery instructions are unchanged."}
    value["snapshot_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"snapshot_sha256": value["snapshot_sha256"], "sources": len(source_hashes), "cases": len(cases), "campaign_created": False}, indent=2))


if __name__ == "__main__":
    main()
