"""Validate the explicit fresh-cohort plan without starting any model calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent.paired_runner import ProtocolError, identity
from src.agent.prompt_policy import FEEDBACK_POLICY, WORKSPACE_POLICY, REFLECTION_POLICY


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    if plan.get("schema_version") != 1 or plan.get("prompt_policies") != {"feedback": FEEDBACK_POLICY, "workspace": WORKSPACE_POLICY}:
        raise ProtocolError("Plan prompt policy differs from the current harness")
    queue = plan.get("model_queue", [])
    optional = plan.get("optional_queue", [])
    all_models = queue + optional
    if not queue or len({item["model"] for item in all_models}) != len(all_models):
        raise ProtocolError("Model queue is empty or contains duplicates")
    ranks = {"codex-cli": 0, "agy-cli": 1, "claude-code-cli": 2}
    previous = -1
    for item in queue:
        rank = ranks.get(item.get("backend"), -1)
        if rank < previous or rank < 0:
            raise ProtocolError("Required queue must be GPT, Gemini, then Claude Code")
        previous = rank
        if item["backend"] == "agy-cli" and not item["model"].startswith("gemini-"):
            raise ProtocolError("The required AGY queue is reserved for Gemini")
        if item["backend"] == "claude-code-cli" and item["model"] not in ("claude-opus-5-5", "claude-sonnet-5-5"):
            raise ProtocolError("Claude must use an explicitly pinned approved model ID")
    if any(item.get("model") != "gpt-oss-120b-medium" or item.get("backend") != "agy-cli" for item in optional):
        raise ProtocolError("Only GPT-OSS 120B is the optional last priority")
    if any(item["model"] == "gpt-6-astra" for item in all_models):
        raise ProtocolError("Astra is a historical reference, not a rerun")
    expected = {"workers": 2, "max_attempts": 10, "solver_timeout": 60, "solver_threads": 1,
                "solver_cooldown_seconds": 10, "agent_timeout": 900, "active_codex_invocation_limit": 2,
                "active_agy_invocation_limit": 1, "active_claude_invocation_limit": 1,
                "shared_first_attempt": True, "stop_after_success": True, "gap_threshold": 0.01,
                "judge_model": "gpt-6-astra", "judge_effort": "high"}
    if plan.get("execution") != expected:
        raise ProtocolError("Execution settings differ from the reviewed paired protocol")
    if not plan.get("dataset_inventory_sha256") or plan.get("reuse_generated_context") is not False:
        raise ProtocolError("A fresh cohort and pinned fixed dataset are required")
    if plan.get("reflection_policy") != REFLECTION_POLICY or plan.get("wrong_target_reflection_failure") != REFLECTION_POLICY:
        raise ProtocolError("Completed reflection omissions must consume their opportunity and advance")
    return plan


def model_queue(manifest):
    if "model_queue" in manifest:
        return manifest["model_queue"]
    return ([{"model": name, "backend": "codex-cli", "effort": manifest.get("generation_effort", "medium")} for name in manifest.get("gpt_order", [])]
            + [{"model": name, "backend": "agy-cli", "effort": "catalog-slug"} for name in manifest.get("agy_order", [])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    print(json.dumps({"plan_id": identity(plan), "required_models": plan["model_queue"], "planned_fresh_pairs": 81 * len(plan["model_queue"]),
                      "optional_models": plan["optional_queue"], "launch_performed": False}, indent=2))


if __name__ == "__main__":
    main()
