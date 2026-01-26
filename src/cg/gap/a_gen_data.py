# gap__gen_data.py
import argparse
from collections.abc import Callable
import json
from pathlib import Path
import random
from typing import Any, Dict, Tuple


SIZE_LABELS = ["small", "medium", "large"]

SIZE_SCALING = {
    "small": 0.7,
    "medium": 1.0,
    "large": 1.15,
}


GAP_PRODUCTION_CONFIG = {
    "n_jobs_range": (5, 8),
    "n_machines_range": (2, 3),
    "resource_min": 2,
    "resource_max": 8,
    "base_cost_min": 8.0,
    "base_cost_max": 18.0,
    "machine_bias_scale": 0.25,
    "capacity_tightness": 0.9,
    "job_name_prefix": "Order_",
    "machine_name_prefix": "Plant_",
}

GAP_IT_CONFIG = {
    "n_jobs_range": (5, 8),
    "n_machines_range": (2, 3),
    "resource_min": 1,
    "resource_max": 7,
    "base_cost_min": 6.0,
    "base_cost_max": 16.0,
    "machine_bias_scale": 0.3,
    "capacity_tightness": 0.9,
    "job_name_prefix": "Service_",
    "machine_name_prefix": "Server_",
}

GAP_STAFFING_CONFIG = {
    "n_jobs_range": (5, 8),
    "n_machines_range": (2, 3),
    "resource_min": 3,
    "resource_max": 9,
    "base_cost_min": 9.0,
    "base_cost_max": 20.0,
    "machine_bias_scale": 0.25,
    "capacity_tightness": 0.95,
    "job_name_prefix": "Project_",
    "machine_name_prefix": "Consultant_",
}


def _scale_range(rng: tuple[int, int], scale: float, minimum: int = 1) -> tuple[int, int]:
    low, high = rng
    new_low = max(minimum, int(round(low * scale)))
    new_high = max(new_low, int(round(high * scale)))
    return new_low, new_high


def _generate_gap_instance_from_config(
    seed: int,
    cfg: dict[str, Any],
    problem_type: str,
    size_label: str,
) -> dict[str, Any]:
    if size_label not in SIZE_SCALING:
        raise ValueError(f"Unknown size_label={size_label}")

    rng = random.Random(seed)
    scale = SIZE_SCALING[size_label]

    n_jobs = rng.randint(*_scale_range(cfg["n_jobs_range"], scale, minimum=2))
    n_machines = rng.randint(*_scale_range(cfg["n_machines_range"], scale, minimum=2))

    a = [[0] * n_machines for _ in range(n_jobs)]
    c = [[0.0] * n_machines for _ in range(n_jobs)]

    for i in range(n_jobs):
        base_cost = rng.uniform(cfg["base_cost_min"], cfg["base_cost_max"])
        for j in range(n_machines):
            a_ij = rng.randint(cfg["resource_min"], cfg["resource_max"])
            bias = (rng.random() - 0.5) * 2 * cfg["machine_bias_scale"] * base_cost
            a[i][j] = a_ij
            c[i][j] = round(base_cost + bias, 2)

    total_resource_per_machine = [sum(a[i][j] for i in range(n_jobs)) for j in range(n_machines)]
    capacities = []
    for j in range(n_machines):
        base = total_resource_per_machine[j] * cfg["capacity_tightness"]
        cap = max(
            max(a[i][j] for i in range(n_jobs)),
            int(base) + rng.randint(0, cfg["resource_max"]),
        )
        capacities.append(cap)

    machines = [
        {"id": j, "name": f"{cfg.get('machine_name_prefix', 'Machine_')}{j}", "capacity": int(capacities[j])}
        for j in range(n_machines)
    ]

    jobs = []
    for i in range(n_jobs):
        assignments = []
        for j in range(n_machines):
            assignments.append(
                {
                    "machine_id": j,
                    "resource": int(a[i][j]),
                    "cost": float(c[i][j]),
                }
            )
        jobs.append(
            {
                "id": i,
                "name": f"{cfg.get('job_name_prefix', 'Job_')}{i}",
                "assignments": assignments,
            }
        )

    data = {
        "problem_type": problem_type,
        "instance_size": size_label,
        "machines": machines,
        "jobs": jobs,
    }
    return data


def _describe_gap_instance_for_narrative(
    data: dict[str, Any],
    max_job_examples: int = 1,
    max_machine_examples: int = 1,
) -> str:
    """Build a short human-readable summary of the GAP instance for the business prompt.
    We summarize numbers of jobs/machines, capacity ranges, and ranges of
    resource consumption and costs, plus one concrete example job and machine.
    """
    machines = data["machines"]
    jobs = data["jobs"]

    n_jobs = len(jobs)
    n_machines = len(machines)

    capacities = [m["capacity"] for m in machines]
    cap_min, cap_max = min(capacities), max(capacities)

    resources = []
    costs = []
    for job in jobs:
        for assignment in job["assignments"]:
            resources.append(assignment["resource"])
            costs.append(assignment["cost"])

    res_min, res_max = min(resources), max(resources)
    cost_min, cost_max = min(costs), max(costs)

    # Example machine and job (if any)
    example_machine = machines[0] if machines else None
    example_job = jobs[0] if jobs else None

    example_parts = []
    if example_machine is not None:
        example_parts.append(
            f"for instance, resource {example_machine['name']} has capacity {example_machine['capacity']} units"
        )
    if example_job is not None:
        job_res = [a["resource"] for a in example_job["assignments"]]
        job_cost = [a["cost"] for a in example_job["assignments"]]
        ej_res_min, ej_res_max = min(job_res), max(job_res)
        ej_cost_min, ej_cost_max = min(job_cost), max(job_cost)
        example_parts.append(
            f"job {example_job['name']} would consume between "
            f"{ej_res_min} and {ej_res_max} units of capacity depending on which "
            f"resource handles it, with assignment costs ranging from "
            f"{ej_cost_min:.2f} to {ej_cost_max:.2f}"
        )

    example_str = ""
    if example_parts:
        example_str = " For example, " + "; ".join(example_parts) + "."

    summary = (
        f"In this planning run there are {n_jobs} jobs to assign and {n_machines} available resources. "
        f"Resource capacities range from {cap_min} to {cap_max} units. "
        f"A single job–resource assignment can consume between {res_min} and {res_max} units of capacity, "
        f"and assignment costs span roughly from {cost_min:.2f} to {cost_max:.2f} in the chosen cost units."
        f"{example_str}"
    )
    return summary


def _size_flavor_gap(problem_label: str, size_label: str) -> str:
    """Provide a short phrase about the scale and context of the operation,
    depending on the problem variant and instance size.
    """
    if problem_label == "gap_production":
        if size_label == "small":
            return "a small manufacturing network with a few plants serving regional customers"
        if size_label == "medium":
            return "a mid-sized manufacturing network with several plants sharing orders across regions"
        if size_label == "large":
            return "a large multi-plant manufacturing network supplying many product lines and markets"

    if problem_label == "gap_it_deployment":
        if size_label == "small":
            return "a small IT operations team managing a limited cluster of application servers"
        if size_label == "medium":
            return "a mid-size IT organization running a mix of business applications across several servers"
        if size_label == "large":
            return "a large-scale IT infrastructure team managing a fleet of servers and services in production"

    if problem_label == "gap_staffing":
        if size_label == "small":
            return "a boutique consulting firm with a handful of consultants and project engagements"
        if size_label == "medium":
            return "a mid-sized consulting firm with several teams working on parallel projects"
        if size_label == "large":
            return "a large consulting organization coordinating many consultants across numerous client projects"

    # Fallback generic description
    return "an organization that must assign jobs to limited-capacity resources while controlling cost"


def _gap_output_instructions() -> str:
    return (
        "To report your plan, write an output.json file with fields: "
        '"status", "objective_value", and "solution". '
        "Use solution.assignments as a list of {job_id, machine_id} pairs. "
        "Each job must appear exactly once, and the total resource usage on each machine "
        "cannot exceed its capacity. Set objective_value to the total assignment cost."
    )


def generate_production_instance_and_prompt(
    seed: int = 0,
    size: str = "medium",
) -> tuple[dict[str, Any], str]:
    """Production planning variant of the generalized assignment problem:
    assigning customer orders to production plants/lines.
    """
    data = _generate_gap_instance_from_config(
        seed,
        GAP_PRODUCTION_CONFIG,
        "gap_production",
        size_label=size,
    )

    scale_phrase = _size_flavor_gap("gap_production", size)
    instance_summary = _describe_gap_instance_for_narrative(data)

    prompt = (
        f"The setting is {scale_phrase}. The company receives a collection of customer orders that "
        "must be produced during the upcoming planning period. Each order can in principle be "
        "processed at any of the available plants, but the production time and cost depend on "
        "which plant takes the order.\n\n"
        "Each plant has a limited number of effective production hours or capacity units available "
        "for this period. Assigning a particular order to a particular plant consumes a known amount "
        "of that plant's capacity and incurs a plant-specific cost that reflects labor, setup, and "
        "logistics considerations. In this simplified model, an order cannot be split across multiple "
        "plants: it must be fully produced at exactly one plant.\n\n"
        f"{instance_summary}\n\n"
        "The planning question is to decide which plant should produce each order so that no plant "
        "exceeds its available capacity and the total cost of fulfilling all orders is minimized."
    )
    prompt = prompt + "\n\n" + _gap_output_instructions()

    return data, prompt


def generate_it_instance_and_prompt(
    seed: int = 0,
    size: str = "medium",
) -> tuple[dict[str, Any], str]:
    """IT deployment variant of the generalized assignment problem:
    placing applications/services onto servers.
    """
    data = _generate_gap_instance_from_config(
        seed,
        GAP_IT_CONFIG,
        "gap_it_deployment",
        size_label=size,
    )

    scale_phrase = _size_flavor_gap("gap_it_deployment", size)
    instance_summary = _describe_gap_instance_for_narrative(data)

    prompt = (
        f"The scenario involves {scale_phrase}. The operations team must decide where to host a set "
        "of applications and services during the next deployment cycle. Each application can be "
        "deployed on any of the available servers, but it consumes different amounts of CPU/memory "
        "depending on the server, and each deployment option has its own operating cost.\n\n"
        "Each server has a limited capacity budget (for example, in normalized CPU or memory units) "
        "that cannot be exceeded without violating reliability and performance targets. Assigning "
        "an application to a specific server consumes some of that server's capacity and incurs a "
        "server-specific cost capturing licensing, infrastructure, and performance overhead. "
        "In this planning model, each application must run on exactly one primary server; we do not "
        "consider replication or load balancing.\n\n"
        f"{instance_summary}\n\n"
        "The deployment question is to assign every application to a server so that no server's "
        "capacity is overloaded and the total operating cost across all assignments is minimized."
    )
    prompt = prompt + "\n\n" + _gap_output_instructions()

    return data, prompt


def generate_staffing_instance_and_prompt(
    seed: int = 0,
    size: str = "medium",
) -> tuple[dict[str, Any], str]:
    """Staffing variant of the generalized assignment problem:
    assigning projects to consultants.
    """
    data = _generate_gap_instance_from_config(
        seed,
        GAP_STAFFING_CONFIG,
        "gap_staffing",
        size_label=size,
    )

    scale_phrase = _size_flavor_gap("gap_staffing", size)
    instance_summary = _describe_gap_instance_for_narrative(data)

    prompt = (
        f"The problem comes from {scale_phrase}. At the start of a planning period, the firm has a "
        "portfolio of client projects that must be staffed. Each project can, in principle, be "
        "assigned to any consultant, but the number of hours required and the effective cost of "
        "the assignment depend on which consultant takes the project (due to differences in "
        "experience, billing rate, and travel needs).\n\n"
        "Each consultant has a limited number of hours available in the period. Assigning a project "
        "to a consultant consumes some of those available hours and incurs a cost that may reflect "
        "a combination of billing discounts, opportunity cost, and penalties for poor skill match. "
        "To keep accountability clear, each project must be owned by exactly one primary consultant "
        "in this simplified model.\n\n"
        f"{instance_summary}\n\n"
        "The staffing question is to decide which consultant should own each project so that no "
        "consultant is assigned more work than they can deliver and the total cost of the assignment "
        "plan is minimized."
    )
    prompt = prompt + "\n\n" + _gap_output_instructions()

    return data, prompt


PROBLEM_GENERATORS: dict[str, Callable[[int, str], tuple[dict[str, Any], str]]] = {
    "gap_production": generate_production_instance_and_prompt,
    "gap_it_deployment": generate_it_instance_and_prompt,
    "gap_staffing": generate_staffing_instance_and_prompt,
}


def write_instance(output_dir: Path, data: dict[str, Any], prompt: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "data.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    with (output_dir / "prompt.txt").open("w", encoding="utf-8") as f:
        f.write(prompt.strip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GAP benchmark data for different business narratives.")
    parser.add_argument(
        "--problem-type",
        choices=["all"] + sorted(PROBLEM_GENERATORS.keys()),
        default="all",
        help="Which business variant to generate (or 'all').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/cg/gap"),
        help="Base output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed (instances use seed + index).",
    )

    args = parser.parse_args()

    if args.problem_type == "all":
        problem_types = list(PROBLEM_GENERATORS.keys())
    else:
        problem_types = [args.problem_type]

    global_index = 0
    for problem_type in problem_types:
        gen_fn = PROBLEM_GENERATORS[problem_type]
        for size_label in SIZE_LABELS:
            seed = args.seed + global_index
            data, prompt = gen_fn(seed=seed, size=size_label)

            instance_dir = args.output_dir / problem_type / size_label
            write_instance(instance_dir, data, prompt)

            global_index += 1


if __name__ == "__main__":
    main()
