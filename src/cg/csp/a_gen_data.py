import argparse
from collections.abc import Callable
import json
from pathlib import Path
import random
from typing import Any, Dict, Tuple


# ---------------------------------------------------------------------------
# Base problem-specific configs (treated as "medium" baseline)
# ---------------------------------------------------------------------------

PAPER_MILL_CONFIG = {
    "n_item_types_range": (3, 6),
    "stock_size_range": (150, 250),  # roll width in cm
    "size_min_frac": 0.10,
    "size_max_frac": 0.60,
    "demand_range": (10, 60),
    "stock_unit": "cm",
    "item_name_prefix": "PaperProduct_",
}

STEEL_BAR_CONFIG = {
    "n_item_types_range": (4, 8),
    "stock_size_range": (6000, 12000),  # bar length in mm
    "size_min_frac": 0.10,
    "size_max_frac": 0.90,
    "demand_range": (15, 80),
    "stock_unit": "mm",
    "item_name_prefix": "CutLength_",
}

CABLE_SPOOL_CONFIG = {
    "n_item_types_range": (3, 6),
    "stock_size_range": (200, 500),  # spool length in m
    "size_min_frac": 0.05,
    "size_max_frac": 0.50,
    "demand_range": (20, 100),
    "stock_unit": "m",
    "item_name_prefix": "CableSegment_",
}

WOOD_PLANK_CONFIG = {
    "n_item_types_range": (3, 7),
    "stock_size_range": (240, 360),  # board length in cm
    "size_min_frac": 0.15,
    "size_max_frac": 0.75,
    "demand_range": (8, 50),
    "stock_unit": "cm",
    "item_name_prefix": "BoardPiece_",
}


# ---------------------------------------------------------------------------
# Size scaling (small / medium / large)
# ---------------------------------------------------------------------------

SIZE_LABELS = ["small", "medium", "large"]

SIZE_SCALING = {
    # Factors applied to "medium" base ranges
    "small": {
        "n_item_factor": 0.7,
        "stock_factor": 0.8,
        "demand_factor": 0.6,
    },
    "medium": {
        "n_item_factor": 1.0,
        "stock_factor": 1.0,
        "demand_factor": 1.0,
    },
    "large": {
        "n_item_factor": 1.4,
        "stock_factor": 1.2,
        "demand_factor": 1.3,
    },
}


def _scale_range(rng: tuple[int, int], factor: float, minimum: int = 1) -> tuple[int, int]:
    low, high = rng
    new_low = max(minimum, int(round(low * factor)))
    new_high = max(new_low, int(round(high * factor)))
    return new_low, new_high


def _describe_items_for_narrative(
    data: dict[str, Any],
    max_examples: int = 3,
) -> str:
    """Build a short human-readable summary of item types for the business prompt.
    Uses sizes and demands from data["item_types"] without leaking every detail.
    """
    items = data["item_types"]
    unit = data.get("stock_unit", "units")
    n_types = len(items)

    sizes = [it["size"] for it in items]
    demands = [it["demand"] for it in items]

    size_min, size_max = min(sizes), max(sizes)
    dem_min, dem_max = min(demands), max(demands)

    # A few concrete examples
    examples = []
    for it in items[:max_examples]:
        examples.append(f"{it['name']} (length {it['size']} {unit}, demand {it['demand']})")

    examples_str = "; ".join(examples)

    summary = (
        f"There are {n_types} distinct item types in this planning batch, "
        f"with lengths ranging from {size_min} to {size_max} {unit} and "
        f"order quantities between {dem_min} and {dem_max} units. "
        f"For example: {examples_str}."
    )

    return summary


def _size_flavor(problem_label: str, size: str) -> str:
    """Provide a short phrase about the scale of the operation depending on instance size."""
    if size == "small":
        if problem_label == "paper_mill":
            return "a small specialty paper converter serving a handful of regional customers"
        if problem_label == "steel_bar":
            return "a local steel service center supporting a limited number of construction projects"
        if problem_label == "cable_spool":
            return "a small electrical distributor serving a few nearby contractors"
        if problem_label == "wood_plank":
            return "a small furniture workshop producing custom pieces to order"
    elif size == "medium":
        if problem_label == "paper_mill":
            return "a mid-sized paper mill supplying packaging and printing customers across the region"
        if problem_label == "steel_bar":
            return "a regional steel fabrication shop serving multiple building sites in parallel"
        if problem_label == "cable_spool":
            return "a regional distributor with a steady stream of cable orders from several installers"
        if problem_label == "wood_plank":
            return "a medium-scale furniture factory producing standard product lines for retailers"
    elif size == "large":
        if problem_label == "paper_mill":
            return "a high-volume paper mill running around the clock for national brand customers"
        if problem_label == "steel_bar":
            return "a large steel service center that cuts high volumes of rebar for major infrastructure projects"
        if problem_label == "cable_spool":
            return "a large distribution hub supplying cable for many concurrent construction and industrial jobs"
        if problem_label == "wood_plank":
            return "a large furniture manufacturer supplying multiple product lines to big-box retailers"

    # Fallback
    return "a manufacturing operation that needs to plan cutting decisions efficiently"


def _csp_output_instructions() -> str:
    return (
        "To report your plan, write an output.json file with fields: "
        '"status", "objective_value", and "solution". '
        'Use solution.patterns as a list of cutting patterns; each pattern has an integer "usage" '
        'and a "cuts" list of {item_id, count} pairs. '
        "For every pattern, the total length cut (sum of size * count) cannot exceed the stock length. "
        "Across all patterns (accounting for usage), the total count for each item_id must meet its demand. "
        "Set objective_value to the total number of stock pieces used (sum of usage)."
    )


# ---------------------------------------------------------------------------
# Generic instance generator (with size label)
# ---------------------------------------------------------------------------


def _generate_instance_from_config(
    seed: int,
    cfg: dict[str, Any],
    problem_type: str,
    size_label: str = "medium",
) -> dict[str, Any]:
    """Generate a 1D cutting stock instance according to a domain-specific config
    and an instance size label (small / medium / large).

    Schema:
    {
      "problem_type": str,
      "instance_size": str,  # "small" | "medium" | "large"
      "stock_size": int,
      "stock_unit": str,
      "item_types": [
        {"id": int, "name": str, "size": int, "demand": int},
        ...
      ]
    }
    """
    if size_label not in SIZE_SCALING:
        raise ValueError(f"Unknown size_label={size_label}")

    scale = SIZE_SCALING[size_label]
    rng = random.Random(seed)

    # Scale ranges based on size_label
    n_item_types_range = _scale_range(
        cfg["n_item_types_range"],
        scale["n_item_factor"],
        minimum=2,
    )
    stock_size_range = _scale_range(
        cfg["stock_size_range"],
        scale["stock_factor"],
        minimum=10,
    )
    demand_range = _scale_range(
        cfg["demand_range"],
        scale["demand_factor"],
        minimum=1,
    )

    n_item_types = rng.randint(*n_item_types_range)
    stock_size = rng.randint(*stock_size_range)

    sizes = []
    for _ in range(n_item_types):
        min_size = max(1, int(cfg["size_min_frac"] * stock_size))
        max_size = max(min_size, int(cfg["size_max_frac"] * stock_size))
        size = rng.randint(min_size, max_size)
        size = min(size, stock_size)
        sizes.append(size)

    demands = [rng.randint(*demand_range) for _ in range(n_item_types)]

    item_name_prefix = cfg.get("item_name_prefix", "Item_")

    data = {
        "problem_type": problem_type,
        "instance_size": size_label,
        "stock_size": stock_size,
        "stock_unit": cfg.get("stock_unit", "units"),
        "item_types": [
            {
                "id": i,
                "name": f"{item_name_prefix}{i}",
                "size": sizes[i],
                "demand": demands[i],
            }
            for i in range(n_item_types)
        ],
    }
    return data


# ---------------------------------------------------------------------------
# Problem-type-specific generators (data + NL business prompt)
# ---------------------------------------------------------------------------


def generate_paper_mill_instance_and_prompt(
    seed: int = 0,
    size: str = "medium",
) -> tuple[dict[str, Any], str]:
    """Paper mill trim-loss / roll-cutting variant."""
    data = _generate_instance_from_config(
        seed,
        PAPER_MILL_CONFIG,
        "paper_mill",
        size_label=size,
    )
    W = data["stock_size"]
    unit = data["stock_unit"]
    item_summary = _describe_items_for_narrative(data)
    scale_phrase = _size_flavor("paper_mill", size)

    prompt = (
        f"The company is {scale_phrase}. "
        f"It runs a cutting line that slices large parent paper reels, each {W} {unit} wide, "
        "into narrower rolls for a mix of end products and intermediate grades. "
        "Customer orders specify the widths they need and the number of rolls of each width that must be shipped.\n\n"
        f"{item_summary}\n\n"
        "Each cutting pattern corresponds to one way of using a parent reel by cutting it into several narrower rolls. "
        "Any unused width on a reel after cutting is trim loss: it either has to be recycled or reprocessed, "
        "which is costly and environmentally undesirable. The production planner must decide how many parent reels to cut "
        "and which cutting patterns to use so that all width and quantity requirements are met.\n\n"
        "Management wants a plan that satisfies every order exactly while using as few parent reels as possible and "
        "minimizing total trim loss."
    )
    prompt = prompt + "\n\n" + _csp_output_instructions()
    return data, prompt


def generate_steel_bar_instance_and_prompt(
    seed: int = 0,
    size: str = "medium",
) -> tuple[dict[str, Any], str]:
    """Steel bar cutting variant (rebar / structural steel)."""
    data = _generate_instance_from_config(
        seed,
        STEEL_BAR_CONFIG,
        "steel_bar",
        size_label=size,
    )
    L = data["stock_size"]
    unit = data["stock_unit"]
    item_summary = _describe_items_for_narrative(data)
    scale_phrase = _size_flavor("steel_bar", size)

    prompt = (
        f"The setting is {scale_phrase}. The operation stocks standard steel bars, each {L} {unit} long, "
        "and cuts them into shorter lengths required by different construction projects. "
        "The demand list specifies, for each bar length, how many pieces are needed to match the reinforcement "
        "and structural design of current jobs.\n\n"
        f"{item_summary}\n\n"
        "When a standard bar is cut according to a particular pattern, several shorter bars are produced, "
        "and any leftover length becomes scrap. Scrap steel increases material cost and conflicts with the "
        "company's goals around waste reduction and sustainable construction. "
        "The planner must choose which cutting patterns to use and how many standard bars to assign to each pattern "
        "so that all length requirements are satisfied.\n\n"
        "The objective is to cover all requested lengths exactly while using as few standard bars as possible "
        "and minimizing total scrap generated."
    )
    prompt = prompt + "\n\n" + _csp_output_instructions()
    return data, prompt


def generate_cable_spool_instance_and_prompt(
    seed: int = 0,
    size: str = "medium",
) -> tuple[dict[str, Any], str]:
    """Cable spool cutting variant."""
    data = _generate_instance_from_config(
        seed,
        CABLE_SPOOL_CONFIG,
        "cable_spool",
        size_label=size,
    )
    L = data["stock_size"]
    unit = data["stock_unit"]
    item_summary = _describe_items_for_narrative(data)
    scale_phrase = _size_flavor("cable_spool", size)

    prompt = (
        f"The scenario involves {scale_phrase}. The distributor buys large spools of electrical cable, "
        f"each containing {L} {unit} of cable, and cuts them into standardized segment lengths requested "
        "by electricians and installers. Each segment type corresponds to a particular job configuration "
        "and has a required number of segments in the current planning horizon.\n\n"
        f"{item_summary}\n\n"
        "Cutting a spool according to a chosen pattern produces several cable segments. "
        "Any leftover cable on a spool after cutting is treated as waste or low-value remnant that is hard to reuse. "
        "The planner must decide how to cut the spools and how many spools to assign to each cutting pattern "
        "so that all segment requirements are satisfied.\n\n"
        "The company seeks a cutting plan that covers all requested segment lengths while keeping the number of spools "
        "used small and limiting leftover cable."
    )
    prompt = prompt + "\n\n" + _csp_output_instructions()
    return data, prompt


def generate_wood_plank_instance_and_prompt(
    seed: int = 0,
    size: str = "medium",
) -> tuple[dict[str, Any], str]:
    """Wooden board / plank cutting variant."""
    data = _generate_instance_from_config(
        seed,
        WOOD_PLANK_CONFIG,
        "wood_plank",
        size_label=size,
    )
    L = data["stock_size"]
    unit = data["stock_unit"]
    item_summary = _describe_items_for_narrative(data)
    scale_phrase = _size_flavor("wood_plank", size)

    prompt = (
        f"The problem comes from {scale_phrase}. The factory purchases long wooden boards, each {L} {unit} in length, "
        "and cuts them into shorter pieces required by different furniture components (legs, rails, panels, etc.). "
        "For the current production batch, the planning team has a bill of materials that specifies how many pieces "
        "of each length must be produced across several product designs.\n\n"
        f"{item_summary}\n\n"
        "Cutting a board according to a chosen pattern yields several pieces, and any leftover length at the end "
        "of the board is scrap. Scrap wood increases material cost and creates additional handling and disposal work. "
        "The production planner must decide which cutting patterns to use and how many boards to cut with each pattern "
        "so that all piece requirements are met.\n\n"
        "The factory wants a plan that satisfies the entire production plan while using as few boards as possible "
        "and reducing wood waste."
    )
    prompt = prompt + "\n\n" + _csp_output_instructions()
    return data, prompt


# Mapping from problem type name to generator
PROBLEM_GENERATORS: dict[str, Callable[[int, str], tuple[dict[str, Any], str]]] = {
    "paper_mill": generate_paper_mill_instance_and_prompt,
    "steel_bar": generate_steel_bar_instance_and_prompt,
    "cable_spool": generate_cable_spool_instance_and_prompt,
    "wood_plank": generate_wood_plank_instance_and_prompt,
}


# ---------------------------------------------------------------------------
# Utility: write one instance to disk
# ---------------------------------------------------------------------------


def write_instance(output_dir: Path, data: dict[str, Any], prompt: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "data.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    with (output_dir / "prompt.txt").open("w", encoding="utf-8") as f:
        f.write(prompt.strip() + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cutting stock benchmark data points for different business problem types."
    )
    parser.add_argument(
        "--problem-type",
        choices=["all"] + sorted(PROBLEM_GENERATORS.keys()),
        default="all",
        help="Which business variant to generate (or 'all').",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/cg/csp"), help="Base output directory.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed (instances use seed + index).")

    args = parser.parse_args()

    if args.problem_type == "all":
        problem_types = list(PROBLEM_GENERATORS.keys())
    else:
        problem_types = [args.problem_type]

    global_index = 0
    for problem_type in problem_types:
        gen_fn = PROBLEM_GENERATORS[problem_type]
        for size_label in SIZE_LABELS:  # small, medium, large
            seed = args.seed + global_index
            data, prompt = gen_fn(seed=seed, size=size_label)

            instance_dir = args.output_dir / problem_type / size_label
            write_instance(instance_dir, data, prompt)

            global_index += 1


if __name__ == "__main__":
    main()
