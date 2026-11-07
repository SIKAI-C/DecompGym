"""Validate an experiment config (for example configs/paper.json) without starting any model calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent.docker_client import CLI_VERSIONS
from src.agent.prompts.policy import FEEDBACK_POLICY, WORKSPACE_POLICY, REFLECTION_POLICY
from src.experiments.data_context import accepted_prompt_protocol, context_delivery_policy
from src.experiments.execution_policy import PAIRED_EXECUTION, SMALL_SCALE_NATIVE_POLICY
from src.utils.errors import ProtocolError
from src.utils.families import FAMILIES, PAPER_FAMILIES, cohort_tracks
from src.utils.records import identity


# The paired protocol of the paper. A config may not change it; a different protocol needs code review.
EXECUTION = PAIRED_EXECUTION
# The reviewed execution settings of each execution policy (src.experiments.execution_policy). A config without an
# "execution_policy" key (every config so far) uses the paper's native policy.
EXECUTION_SETTINGS = {SMALL_SCALE_NATIVE_POLICY: EXECUTION}


def planned_cases(plan: dict) -> int:
    """The instances of the plan's cohort (the paper's fixed 81 without a cohort), from the families' track descriptors."""
    return sum(track.instance_count for track in cohort_tracks(plan.get("cohort", {}).get("families", PAPER_FAMILIES)).values())


def load_plan(path: Path) -> dict:
    plan = json.loads(Path(path).read_text())
    if plan.get("schema_version") != 2:
        raise ProtocolError("Unsupported config schema; expected schema_version 2")
    if plan.get("prompt_policies") != {"feedback": FEEDBACK_POLICY, "workspace": WORKSPACE_POLICY}:
        raise ProtocolError("Config prompt policy differs from the current harness")
    # The reviewed contracts, optionally with the recorded data-context setting (src.experiments.data_context).
    if not accepted_prompt_protocol(plan.get("prompt_protocol")):
        raise ProtocolError("Config must freeze the reviewed problem, judge, and output contracts")
    if "solver_output_policy" in plan:
        # Optional, recorded with the config (and so in the campaign protocol id); absent keeps the paper's 4 MiB readback.
        # The larger limit needs the mounted-files protocol: a pasted reflection context cannot carry such an output.json.
        from src.experiments.sandbox import OUTPUT_LIMITS
        if plan["solver_output_policy"] is None or plan["solver_output_policy"] not in OUTPUT_LIMITS:
            raise ProtocolError(f"Unknown solver output policy: {plan['solver_output_policy']!r}")
        if context_delivery_policy(plan["prompt_protocol"]) is None:
            raise ProtocolError("A larger solver output readback requires the mounted-files context delivery")
    if plan.get("reflection_policy") != REFLECTION_POLICY:
        raise ProtocolError("Completed reflection omissions must consume their opportunity and advance")
    queue = plan.get("model_queue", [])
    if not queue or len({item.get("model") for item in queue}) != len(queue):
        raise ProtocolError("Model queue is empty or contains duplicates")
    for item in queue:
        if not item.get("model") or item.get("backend") not in CLI_VERSIONS or not item.get("effort"):
            raise ProtocolError(f"Each model needs a model ID, a backend in {sorted(CLI_VERSIONS)} and an effort: {item}")
    policy = plan.get("execution_policy", SMALL_SCALE_NATIVE_POLICY)
    if not isinstance(policy, str) or policy not in EXECUTION_SETTINGS or plan.get("execution") != EXECUTION_SETTINGS[policy]:
        raise ProtocolError("Execution settings differ from the paired protocol")
    if not plan.get("dataset_inventory_sha256"):
        raise ProtocolError("The config must pin the fixed dataset inventory")
    if "cohort" in plan:
        families = plan["cohort"].get("families")
        if not isinstance(families, list) or not families or len(set(families)) != len(families) or any(name not in FAMILIES for name in families):
            raise ProtocolError(f"The cohort must list distinct registered problem families (src.utils.families): {families}")
    return plan


def model_queue(manifest):
    return manifest["model_queue"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = load_plan(args.config)
    print(json.dumps({"config_id": identity(plan), "models": plan["model_queue"], "planned_pairs": planned_cases(plan) * len(plan["model_queue"]), "launch_performed": False}, indent=2))


if __name__ == "__main__":
    main()
