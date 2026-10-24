"""Validate an experiment config (for example configs/paper.json) without starting any model calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent.docker_client import CLI_VERSIONS
from src.agent.prompts.policy import FEEDBACK_POLICY, WORKSPACE_POLICY, REFLECTION_POLICY
from src.agent.prompts.snapshot import PROMPT_PROTOCOL
from src.utils.errors import ProtocolError
from src.utils.records import identity


# The paired protocol of the paper. A config may not change it; a different protocol needs code review.
EXECUTION = {"workers": 2, "max_attempts": 10, "solver_timeout": 60, "solver_threads": 1,
             "solver_cooldown_seconds": 10, "agent_timeout": 900, "active_codex_invocation_limit": 2,
             "active_agy_invocation_limit": 1, "active_claude_invocation_limit": 1,
             "shared_first_attempt": True, "stop_after_success": True, "gap_threshold": 0.01,
             "judge_model": "gpt-6-astra", "judge_effort": "high"}


def load_plan(path: Path) -> dict:
    plan = json.loads(Path(path).read_text())
    if plan.get("schema_version") != 2:
        raise ProtocolError("Unsupported config schema; expected schema_version 2")
    if plan.get("prompt_policies") != {"feedback": FEEDBACK_POLICY, "workspace": WORKSPACE_POLICY}:
        raise ProtocolError("Config prompt policy differs from the current harness")
    if plan.get("prompt_protocol") != PROMPT_PROTOCOL:
        raise ProtocolError("Config must freeze the reviewed problem, judge, and output contracts")
    if plan.get("reflection_policy") != REFLECTION_POLICY:
        raise ProtocolError("Completed reflection omissions must consume their opportunity and advance")
    queue = plan.get("model_queue", [])
    if not queue or len({item.get("model") for item in queue}) != len(queue):
        raise ProtocolError("Model queue is empty or contains duplicates")
    for item in queue:
        if not item.get("model") or item.get("backend") not in CLI_VERSIONS or not item.get("effort"):
            raise ProtocolError(f"Each model needs a model ID, a backend in {sorted(CLI_VERSIONS)} and an effort: {item}")
    if plan.get("execution") != EXECUTION:
        raise ProtocolError("Execution settings differ from the paired protocol")
    if not plan.get("dataset_inventory_sha256"):
        raise ProtocolError("The config must pin the fixed dataset inventory")
    return plan


def model_queue(manifest):
    return manifest["model_queue"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    plan = load_plan(args.config)
    print(json.dumps({"config_id": identity(plan), "models": plan["model_queue"], "planned_pairs": 81 * len(plan["model_queue"]), "launch_performed": False}, indent=2))


if __name__ == "__main__":
    main()
