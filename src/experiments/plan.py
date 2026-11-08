"""Validate an experiment config (for example configs/paper.json) without starting any model calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from src.agent.docker_client import CLI_VERSIONS
from src.agent.prompts.policy import FEEDBACK_POLICY, WORKSPACE_POLICY, REFLECTION_POLICY
from src.experiments.data_context import accepted_prompt_protocol, context_delivery_policy
from src.experiments.execution_policy import LANE_FIELDS, LANES_EXECUTION, PAIRED_EXECUTION, SMALL_SCALE_LANES_POLICY, SMALL_SCALE_NATIVE_POLICY
from src.utils.errors import ProtocolError
from src.utils.families import FAMILIES, PAPER_FAMILIES, cohort_tracks
from src.utils.records import identity


# The paired protocol of the paper. A config may not change it; a different protocol needs code review.
EXECUTION = PAIRED_EXECUTION
# The reviewed execution settings of each execution policy (src.experiments.execution_policy). A config without an
# "execution_policy" key (the paper and harder configs) uses the paper's native policy; configs/small_scale.json selects
# the lanes variant (model lanes side by side in one controller, the 2 GiB per-solver memory budget; execution_matches).
EXECUTION_SETTINGS = {SMALL_SCALE_NATIVE_POLICY: EXECUTION, SMALL_SCALE_LANES_POLICY: LANES_EXECUTION}
LANE_NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")  # also part of each lane worker's thread name (experiment-<lane>_<n>)


def execution_policy_id(manifest: dict) -> str:
    """The execution policy of a campaign manifest (its frozen plan) or of a plan."""
    plan = manifest.get("experiment_plan", manifest) or {}
    return plan.get("execution_policy", SMALL_SCALE_NATIVE_POLICY)


def expected_specs(manifest: dict) -> dict:
    """The non-default spec ids the campaign's runners record (PairedRunner.spec_record): the variant policy's id."""
    policy = execution_policy_id(manifest)
    return {} if policy == SMALL_SCALE_NATIVE_POLICY else {"execution_policy": policy}


def execution_matches(policy: str, execution) -> bool:
    """Whether a config's execution settings are the reviewed ones of `policy`. The lanes variant's lanes and total
    workers are the config's (LANE_FIELDS); lane_partition validates them against the model queue."""
    if policy != SMALL_SCALE_LANES_POLICY:
        return execution == EXECUTION_SETTINGS[policy]
    return isinstance(execution, dict) and set(LANE_FIELDS) <= set(execution) and {key: value for key, value in execution.items() if key not in LANE_FIELDS} == LANES_EXECUTION


def lane_partition(lanes: list | None, cohort: list, scope: list | None = None) -> list[tuple[dict, list[dict]]]:
    """[(lane, its models of the launch scope, in queue order)], or [] without lanes (the sequential run).

    The lanes ({"name", "models", "workers"}) partition the cohort's model queue exactly: every queued model in exactly
    one lane, each lane's models in queue order, distinct names, a positive worker count. After scoping (`scope`, the
    selected queue entries) a lane may be empty. A lane may hold several backends: its calls still pass the gate of each
    model's own backend."""
    if lanes is None:
        return []
    if not isinstance(lanes, list) or not lanes:
        raise ProtocolError("Lanes must be a non-empty list")
    order = {item["model"]: index for index, item in enumerate(cohort)}
    names, owner = set(), {}
    for lane in lanes:
        name, members, workers = (lane.get("name"), lane.get("models"), lane.get("workers")) if isinstance(lane, dict) else (None, None, None)
        if not isinstance(name, str) or not LANE_NAME.fullmatch(name) or name in names or set(lane) != {"name", "models", "workers"}:
            raise ProtocolError(f"Each lane is {{name, models, workers}} with a distinct lower-case hyphenated name: {lane!r}")
        if type(workers) is not int or workers < 1 or not isinstance(members, list) or not members:
            raise ProtocolError(f"A lane needs a positive worker count and at least one model: {lane}")
        for model in members:
            if model not in order or model in owner:
                raise ProtocolError(f"Lane model {model!r} is outside the model queue or belongs to more than one lane")
            owner[model] = name
        if [order[model] for model in members] != sorted(order[model] for model in members):
            raise ProtocolError(f"Lane {name} lists its models out of queue order")
        names.add(name)
    missing = [item["model"] for item in cohort if item["model"] not in owner]
    if missing:
        raise ProtocolError(f"Queued models without a lane: {missing}")
    selected = cohort if scope is None else scope
    return [(lane, [item for item in selected if owner[item["model"]] == lane["name"]]) for lane in lanes]


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
    if not isinstance(policy, str) or policy not in EXECUTION_SETTINGS or not execution_matches(policy, plan.get("execution")):
        raise ProtocolError("Execution settings differ from the paired protocol")
    if "lanes" in plan["execution"]:
        lane_partition(plan["execution"]["lanes"], queue)
        if sum(lane["workers"] for lane in plan["execution"]["lanes"]) != plan["execution"]["workers"]:
            raise ProtocolError("The lanes' workers must add up to the campaign's workers")
    if plan["execution"].get("solver_memory_policy") is not None and context_delivery_policy(plan["prompt_protocol"]) is None:
        raise ProtocolError("A memory budget is stated only in the mounted-files prompt")
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
