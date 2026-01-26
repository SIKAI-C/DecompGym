# shift__gen_data.py
import argparse
from collections.abc import Callable
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, Tuple


SIZE_LABELS = ["small", "medium", "large"]

SIZE_SCALING = {
    "small": 0.7,
    "medium": 1.0,
    "large": 1.1,
}


CALL_CENTER_CONFIG = {
    "T_range": (8, 12),
    "demand_min": 2,
    "demand_max": 6,
    "peak_multiplier": 1.6,
    "shift_min_length": 2,
    "shift_max_length": 5,
    "time_label_start": 8,
    "time_label_step": 1,
}

HOSPITAL_CONFIG = {
    "T_range": (10, 14),
    "demand_min": 3,
    "demand_max": 8,
    "peak_multiplier": 1.6,
    "shift_min_length": 3,
    "shift_max_length": 6,
    "time_label_start": 0,
    "time_label_step": 1,
}

RETAIL_CONFIG = {
    "T_range": (8, 12),
    "demand_min": 2,
    "demand_max": 6,
    "peak_multiplier": 1.5,
    "shift_min_length": 2,
    "shift_max_length": 5,
    "time_label_start": 8,
    "time_label_step": 1,
}

WAREHOUSE_CONFIG = {
    "T_range": (8, 12),
    "demand_min": 2,
    "demand_max": 7,
    "peak_multiplier": 1.5,
    "shift_min_length": 2,
    "shift_max_length": 5,
    "time_label_start": 6,
    "time_label_step": 1,
}


def _scale_value(value: int, scale: float, minimum: int = 1) -> int:
    return max(minimum, int(round(value * scale)))


def _generate_shift_instance_from_config(
    seed: int,
    cfg: dict[str, Any],
    problem_type: str,
    size_label: str,
) -> dict[str, Any]:
    if size_label not in SIZE_SCALING:
        raise ValueError(f"Unknown size_label={size_label}")

    rng = random.Random(seed)
    scale = SIZE_SCALING[size_label]

    T_low = _scale_value(cfg["T_range"][0], scale, minimum=6)
    T_high = _scale_value(cfg["T_range"][1], scale, minimum=T_low)
    T = rng.randint(T_low, T_high)

    demand: list[int] = []
    for t in range(T):
        base = rng.randint(cfg["demand_min"], cfg["demand_max"])
        if T // 3 <= t <= 2 * T // 3:
            base = int(math.ceil(base * cfg["peak_multiplier"]))
        demand.append(max(1, base))

    L_min = cfg["shift_min_length"]
    L_max = min(cfg["shift_max_length"], T)
    shift_cost = 1.0

    time_slots = []
    start_hour = cfg["time_label_start"]
    step = cfg["time_label_step"]
    for t in range(T):
        h1 = start_hour + t * step
        h2 = start_hour + (t + 1) * step
        label = f"{h1:02d}:00-{h2:02d}:00"
        time_slots.append({"id": t, "label": label})

    data = {
        "problem_type": problem_type,
        "instance_size": size_label,
        "time_slots": time_slots,
        "demand": demand,
        "shift_min_length": L_min,
        "shift_max_length": L_max,
        "shift_cost": shift_cost,
    }
    return data


def _describe_demand_profile(data: dict[str, Any]) -> str:
    """Build a short, human-readable summary of the demand profile
    using the generated time_slots and demand.
    """
    time_slots = data["time_slots"]
    demand = data["demand"]
    T = len(time_slots)

    min_d = min(demand)
    max_d = max(demand)
    avg_d = sum(demand) / T

    # Identify one high-demand and one low-demand slot
    peak_idx = max(range(T), key=lambda i: demand[i])
    low_idx = min(range(T), key=lambda i: demand[i])

    first_label = time_slots[0]["label"]
    last_label = time_slots[-1]["label"]
    peak_label = time_slots[peak_idx]["label"]
    low_label = time_slots[low_idx]["label"]

    summary = (
        f"The planning horizon is divided into {T} time slots, from {first_label} to {last_label}. "
        f"Staffing requirements fluctuate over the day: the minimum requirement is {min_d} workers, "
        f"the maximum requirement is {max_d} workers (around {peak_label}), "
        f"and on average about {avg_d:.1f} workers are needed per slot. "
        f"For comparison, demand is as low as {min_d} workers in periods such as {low_label}."
    )
    return summary


def _size_flavor_shift(problem_label: str, size: str) -> str:
    """Provide a short phrase about the scale/context of the operation,
    depending on instance size and business setting.
    """
    if size == "small":
        if problem_label == "shift_call_center":
            return "a small customer support center handling regional calls"
        if problem_label == "shift_hospital":
            return "a single hospital ward with a relatively small nursing team"
        if problem_label == "shift_retail":
            return "a neighborhood retail store with a limited number of associates on the floor"
        if problem_label == "shift_warehouse":
            return "a compact warehouse with a small crew of pickers"
    elif size == "medium":
        if problem_label == "shift_call_center":
            return "a mid-sized call center that handles customer inquiries throughout the day"
        if problem_label == "shift_hospital":
            return "a general hospital department with several nurses on duty each hour"
        if problem_label == "shift_retail":
            return "a medium-sized store that sees steady walk-in traffic across its opening hours"
        if problem_label == "shift_warehouse":
            return "a medium-volume warehouse that needs consistent staffing on picking lines"
    elif size == "large":
        if problem_label == "shift_call_center":
            return "a high-volume call center serving customers nationwide with multiple queues"
        if problem_label == "shift_hospital":
            return "a busy hospital unit operating around the clock with significant demand variability"
        if problem_label == "shift_retail":
            return "a large flagship retail store that experiences strong lunchtime and evening peaks"
        if problem_label == "shift_warehouse":
            return "a high-throughput distribution center feeding many downstream locations"

    return "an operation that must carefully plan staff coverage across the day"


def _build_shift_prompt(narrative: str, data: dict[str, Any]) -> str:
    T = len(data["time_slots"])
    L_min = data["shift_min_length"]
    L_max = data["shift_max_length"]

    demand_story = _describe_demand_profile(data)

    prompt = (
        f"{narrative}\n\n"
        f"{demand_story}\n\n"
        "In this setting, each feasible shift corresponds to a contiguous block of time slots "
        f"with length between {L_min} and {L_max} slots. A worker assigned to a given shift "
        "is assumed to be available for every time slot covered by that shift. The planner must "
        "decide how many workers to assign to each possible shift pattern so that, for every time slot, "
        "the number of workers on duty is at least the required demand.\n\n"
        'From a planning perspective, different shift patterns can be viewed as "columns" in a set-covering '
        "formulation: each column represents one possible contiguous shift, and the objective is to select a "
        "combination of shifts that covers all time-slot demands while minimizing the total number of shifts (or "
        "equivalently, the total staffing cost).\n\n"
        f"{_shift_output_instructions()}"
    )
    return prompt


def _shift_output_instructions() -> str:
    return (
        "To report your plan, write an output.json file with fields: "
        '"status", "objective_value", and "solution". '
        "Use solution.shifts as a list of {start_slot, end_slot, count} entries. "
        "Each shift must be a contiguous block whose length is within the allowed bounds, and the total number "
        "of workers covering each time slot must meet or exceed demand. "
        "Set objective_value to shift_cost multiplied by the total number of shifts assigned."
    )


def generate_call_center(seed: int = 0, size: str = "medium") -> tuple[dict[str, Any], str]:
    data = _generate_shift_instance_from_config(
        seed,
        CALL_CENTER_CONFIG,
        "shift_call_center",
        size,
    )
    scale_phrase = _size_flavor_shift("shift_call_center", size)

    narrative = (
        f"The scenario is {scale_phrase}. The center receives inbound calls throughout the day, "
        "and managers want to guarantee acceptable waiting times while keeping labor costs under control. "
        "Historical call-volume analysis has been translated into a per-slot requirement: for each time slot, "
        "there is a minimum number of agents that must be logged in to meet service-level targets.\n\n"
        "Each agent works a single contiguous shift during the planning horizon. Workforce management policies "
        "impose minimum and maximum shift lengths (e.g., to ensure reasonable working days and break patterns). "
        "If too few agents are on duty in a slot, service levels degrade; if too many agents are scheduled, the "
        "center pays for idle capacity. The goal is to construct a set of feasible shifts and assign agents to them "
        "such that all time-slot staffing requirements are satisfied with as few shifts as possible."
    )

    return data, _build_shift_prompt(narrative, data)


def generate_hospital(seed: int = 0, size: str = "medium") -> tuple[dict[str, Any], str]:
    data = _generate_shift_instance_from_config(
        seed,
        HOSPITAL_CONFIG,
        "shift_hospital",
        size,
    )
    scale_phrase = _size_flavor_shift("shift_hospital", size)

    narrative = (
        f"The setting is {scale_phrase}. The unit must ensure that enough nurses are on duty in each hour "
        "to safely care for patients and comply with staffing guidelines. For the planning horizon, the charge nurse "
        "has specified, for each hourly time slot, the minimum number of nurses that should be present on the ward.\n\n"
        "Each nurse is assigned to exactly one shift in this simplified model. Shifts are contiguous blocks of hours and "
        "must respect lower and upper bounds on shift length, reflecting contract rules and fatigue considerations. "
        "Understaffing in any hour is unacceptable from a safety and quality-of-care perspective, but excessive "
        "overstaffing drives up labor cost and pulls nurses away from other units. The task is to design a set of "
        "feasible shifts and choose how many nurses work each shift pattern so that the hourly staffing requirements "
        "are met with minimal total shifts."
    )

    return data, _build_shift_prompt(narrative, data)


def generate_retail(seed: int = 0, size: str = "medium") -> tuple[dict[str, Any], str]:
    data = _generate_shift_instance_from_config(
        seed,
        RETAIL_CONFIG,
        "shift_retail",
        size,
    )
    scale_phrase = _size_flavor_shift("shift_retail", size)

    narrative = (
        f"The problem comes from {scale_phrase}. Customer traffic varies across the day: mornings may be quiet, "
        "lunchtime and late afternoon are busier, and evenings taper off again. Store management translates expected "
        "footfall and service standards into a required number of sales associates for each time slot.\n\n"
        "Each associate works a single contiguous shift during the planning horizon, with minimum and maximum shift "
        "lengths dictated by labor rules and store policies. If too few associates are on the floor, customers wait "
        "longer for help and sales may be lost; if too many are scheduled, labor cost increases without improving "
        "service. The planner must pick a set of feasible shift patterns and decide how many associates to assign "
        "to each pattern so that all time slots are adequately staffed while keeping the number of shifts as low as possible."
    )

    return data, _build_shift_prompt(narrative, data)


def generate_warehouse(seed: int = 0, size: str = "medium") -> tuple[dict[str, Any], str]:
    data = _generate_shift_instance_from_config(
        seed,
        WAREHOUSE_CONFIG,
        "shift_warehouse",
        size,
    )
    scale_phrase = _size_flavor_shift("shift_warehouse", size)

    narrative = (
        f"The scenario involves {scale_phrase}. The warehouse experiences varying picking and packing workloads "
        "over the course of the day, driven by inbound receipts, order-cutoff times, and outbound truck departures. "
        "Operations planning has translated these workloads into a required number of warehouse workers for each time slot.\n\n"
        "Each worker is assigned to one contiguous shift, whose length must fall within allowed bounds capturing "
        "standard shift patterns and break policies. Understaffing in busy periods leads to backlog and missed "
        "departure windows, while overstaffing during quiet periods wastes labor and reduces productivity. "
        "The warehouse planner must assemble a set of feasible shifts and decide how many workers to assign to each "
        "shift pattern so that every time slot's staffing requirement is satisfied with the smallest possible number of shifts."
    )

    return data, _build_shift_prompt(narrative, data)


PROBLEM_GENERATORS: dict[str, Callable[[int, str], tuple[dict[str, Any], str]]] = {
    "shift_call_center": generate_call_center,
    "shift_hospital": generate_hospital,
    "shift_retail": generate_retail,
    "shift_warehouse": generate_warehouse,
}


def write_instance(output_dir: Path, data: dict[str, Any], prompt: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "data.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    with (output_dir / "prompt.txt").open("w", encoding="utf-8") as f:
        f.write(prompt.strip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate shift scheduling benchmark instances.")
    parser.add_argument(
        "--problem-type",
        choices=["all"] + sorted(PROBLEM_GENERATORS.keys()),
        default="all",
        help="Business narrative to generate (or 'all').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/cg/shift"),
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
