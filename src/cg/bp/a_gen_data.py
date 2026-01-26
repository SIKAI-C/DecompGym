# bp__gen_data.py
import argparse
from collections.abc import Callable
import json
from pathlib import Path
import random
from typing import Any, Dict, Tuple


# ---------------------------------------------------------------------------
# Base problem-specific configs (treated as "medium" baseline)
# ---------------------------------------------------------------------------

ECOM_BOX_CONFIG = {
    "n_items_range": (10, 20),
    "bin_capacity_range": (40, 80),  # total weight/volume per box
    "size_min_frac": 0.15,
    "size_max_frac": 0.70,
    "bin_unit": "kg",
    "item_name_prefix": "Parcel_",
}

CONTAINER_LOADING_CONFIG = {
    "n_items_range": (12, 25),
    "bin_capacity_range": (80, 160),  # total volume/weight per container
    "size_min_frac": 0.10,
    "size_max_frac": 0.60,
    "bin_unit": "volume_units",
    "item_name_prefix": "Shipment_",
}

TRUCK_LOADING_CONFIG = {
    "n_items_range": (10, 22),
    "bin_capacity_range": (60, 120),  # total load per truck
    "size_min_frac": 0.20,
    "size_max_frac": 0.75,
    "bin_unit": "load_units",
    "item_name_prefix": "Load_",
}

WAREHOUSE_BIN_CONFIG = {
    "n_items_range": (8, 18),
    "bin_capacity_range": (30, 70),  # storage capacity per bin
    "size_min_frac": 0.15,
    "size_max_frac": 0.65,
    "bin_unit": "volume_units",
    "item_name_prefix": "SKU_",
}


# ---------------------------------------------------------------------------
# Size scaling (small / medium / large)
# ---------------------------------------------------------------------------

SIZE_LABELS = ["small", "medium", "large"]

SIZE_SCALING = {
    # Factors applied to "medium" base ranges
    "small": {
        "n_item_factor": 0.7,
        "capacity_factor": 0.8,
    },
    "medium": {
        "n_item_factor": 1.0,
        "capacity_factor": 1.0,
    },
    "large": {
        "n_item_factor": 1.4,
        "capacity_factor": 1.2,
    },
}


def _scale_range(rng: tuple[int, int], factor: float, minimum: int = 1) -> tuple[int, int]:
    low, high = rng
    new_low = max(minimum, int(round(low * factor)))
    new_high = max(new_low, int(round(high * factor)))
    return new_low, new_high


def _describe_items_for_narrative_bp(
    data: dict[str, Any],
    max_examples: int = 3,
) -> str:
    """Build a short human-readable summary of items for the business prompt.
    Uses item sizes (and names) from data["items"].
    """
    items = data["items"]
    unit = data.get("bin_unit", "units")
    n_items = len(items)

    sizes = [it["size"] for it in items]
    size_min, size_max = min(sizes), max(sizes)

    examples = []
    for it in items[:max_examples]:
        examples.append(f"{it['name']} (size {it['size']} {unit})")
    examples_str = "; ".join(examples)

    summary = (
        f"There are {n_items} individual items that must be packed, "
        f"with sizes ranging from {size_min} to {size_max} {unit}. "
        f"For example: {examples_str}."
    )
    return summary


def _size_flavor_bp(problem_label: str, size: str) -> str:
    """Provide a short phrase about the scale/context of the operation
    depending on instance size.
    """
    if size == "small":
        if problem_label == "ecom_boxes":
            return "a small but fast-growing online shop that ships directly from its own warehouse"
        if problem_label == "container_loading":
            return "a local freight forwarder consolidating a modest number of export orders"
        if problem_label == "truck_loading":
            return "a regional distributor serving a few nearby cities with a small truck fleet"
        if problem_label == "warehouse_bins":
            return "a compact warehouse that stores inventory for a single retail channel"
    elif size == "medium":
        if problem_label == "ecom_boxes":
            return "a mid-sized e-commerce fulfillment center handling a steady daily order volume"
        if problem_label == "container_loading":
            return "a logistics provider managing regular container shipments for several customers"
        if problem_label == "truck_loading":
            return "a distribution hub coordinating deliveries to many supermarkets and stores"
        if problem_label == "warehouse_bins":
            return "a medium-sized warehouse that stocks a live catalog of products across categories"
    elif size == "large":
        if problem_label == "ecom_boxes":
            return "a high-volume e-commerce operation processing thousands of parcels per wave"
        if problem_label == "container_loading":
            return "a large consolidation center that fills multiple containers for international routes each day"
        if problem_label == "truck_loading":
            return "a large regional DC that dispatches a continuous stream of loaded trucks"
        if problem_label == "warehouse_bins":
            return "a high-density warehouse feeding multiple sales channels with fast-moving inventory"

    # Fallback
    return "a logistics operation that needs to pack items into limited-capacity containers efficiently"


def _bp_output_instructions() -> str:
    return (
        "To report your plan, write an output.json file with fields: "
        '"status", "objective_value", and "solution". '
        'Use solution.bins as a list of boxes, where each box has an "item_ids" list '
        "containing the item IDs from data.json placed in that box. "
        "Each item must appear exactly once across all boxes, and the total size of items "
        "in any box cannot exceed the box capacity. "
        "Set objective_value to the number of boxes used."
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
    """Generate a 1D bin packing instance according to a domain-specific config
    and an instance size label (small / medium / large).

    Schema:
    {
      "problem_type": str,
      "instance_size": str,  # "small" | "medium" | "large"
      "bin_capacity": int,
      "bin_unit": str,
      "items": [
        {"id": int, "name": str, "size": int},
        ...
      ]
    }
    """
    if size_label not in SIZE_SCALING:
        raise ValueError(f"Unknown size_label={size_label}")

    scale = SIZE_SCALING[size_label]
    rng = random.Random(seed)

    # Scale ranges based on size_label
    n_items_range = _scale_range(
        cfg["n_items_range"],
        scale["n_item_factor"],
        minimum=3,
    )
    bin_capacity_range = _scale_range(
        cfg["bin_capacity_range"],
        scale["capacity_factor"],
        minimum=10,
    )

    n_items = rng.randint(*n_items_range)
    bin_capacity = rng.randint(*bin_capacity_range)

    sizes = []
    for _ in range(n_items):
        min_size = max(1, int(cfg["size_min_frac"] * bin_capacity))
        max_size = max(min_size, int(cfg["size_max_frac"] * bin_capacity))
        size = rng.randint(min_size, max_size)
        size = min(size, bin_capacity)
        sizes.append(size)

    item_name_prefix = cfg.get("item_name_prefix", "Item_")

    data = {
        "problem_type": problem_type,
        "instance_size": size_label,
        "bin_capacity": bin_capacity,
        "bin_unit": cfg.get("bin_unit", "units"),
        "items": [
            {
                "id": i,
                "name": f"{item_name_prefix}{i}",
                "size": sizes[i],
            }
            for i in range(n_items)
        ],
    }
    return data


# ---------------------------------------------------------------------------
# Problem-type-specific generators (data + NL business prompt)
# ---------------------------------------------------------------------------


def generate_ecom_box_instance_and_prompt(
    seed: int = 0,
    size: str = "medium",
) -> tuple[dict[str, Any], str]:
    """E-commerce parcels into shipping boxes."""
    data = _generate_instance_from_config(
        seed,
        ECOM_BOX_CONFIG,
        "ecom_boxes",
        size_label=size,
    )
    C = data["bin_capacity"]
    unit = data["bin_unit"]
    n_items = len(data["items"])
    item_summary = _describe_items_for_narrative_bp(data)
    scale_phrase = _size_flavor_bp("ecom_boxes", size)

    prompt = (
        f"The setting is {scale_phrase}. In each planning run, the packing team receives a batch of "
        f"{n_items} customer parcels that all need to be shipped out in the next wave. "
        f"The operation uses a single standard shipping box type, and each box can hold up to {C} {unit} "
        "of parcel content in total.\n\n"
        f"{item_summary}\n\n"
        "Once a parcel is assigned to a box, it must fit within the remaining capacity of that box, and parcels "
        "cannot be split across multiple boxes. Unused space in a box represents wasted cardboard, filler material, "
        "and extra transportation cost. The packing coordinator wants to assign parcels to boxes in a way that "
        "covers the entire batch while using as few boxes as possible and keeping wasted capacity low."
    )
    prompt = prompt + "\n\n" + _bp_output_instructions()
    return data, prompt


def generate_container_loading_instance_and_prompt(
    seed: int = 0,
    size: str = "medium",
) -> tuple[dict[str, Any], str]:
    """Shipments into identical containers."""
    data = _generate_instance_from_config(
        seed,
        CONTAINER_LOADING_CONFIG,
        "container_loading",
        size_label=size,
    )
    C = data["bin_capacity"]
    unit = data["bin_unit"]
    n_items = len(data["items"])
    item_summary = _describe_items_for_narrative_bp(data)
    scale_phrase = _size_flavor_bp("container_loading", size)

    prompt = (
        f"The scenario involves {scale_phrase}. For a given departure, the planner must load "
        f"{n_items} shipment lots into identical containers. Each container has a capacity of {C} {unit}, "
        "representing the maximum volume or weight that can be loaded without incurring penalties or violating "
        "transport regulations.\n\n"
        f"{item_summary}\n\n"
        "Each shipment lot must be fully placed into exactly one container; shipments cannot be split across "
        "multiple containers in this simplified planning model. Containers that are only partially filled tie up "
        "equipment and drive up shipping costs per unit of cargo. The logistics planner wants to decide how many "
        "containers to use and which shipments to group together in each container, with the goal of fitting all "
        "shipments while minimizing the number of containers required."
    )
    prompt = prompt + "\n\n" + _bp_output_instructions()
    return data, prompt


def generate_truck_loading_instance_and_prompt(
    seed: int = 0,
    size: str = "medium",
) -> tuple[dict[str, Any], str]:
    """Loads into identical trucks."""
    data = _generate_instance_from_config(
        seed,
        TRUCK_LOADING_CONFIG,
        "truck_loading",
        size_label=size,
    )
    C = data["bin_capacity"]
    unit = data["bin_unit"]
    n_items = len(data["items"])
    item_summary = _describe_items_for_narrative_bp(data)
    scale_phrase = _size_flavor_bp("truck_loading", size)

    prompt = (
        f"The setting is {scale_phrase}. For an upcoming delivery wave, the dispatch team needs to assign "
        f"{n_items} palletized customer orders to a fleet of identical trucks. "
        f"Each truck has a loading capacity of {C} {unit}, representing a simplified combination of space and "
        "weight limits.\n\n"
        f"{item_summary}\n\n"
        "Each order must be loaded in full onto exactly one truck; orders cannot be split among multiple vehicles "
        "in this planning step. If too many trucks are used or if trucks run significantly under capacity, the cost "
        "per delivered unit rises and the carbon footprint per shipment increases. The dispatcher wants to group "
        "orders into truck loads so that all orders are delivered while using as few trucks as possible and avoiding "
        "overloaded vehicles."
    )
    prompt = prompt + "\n\n" + _bp_output_instructions()
    return data, prompt


def generate_warehouse_bins_instance_and_prompt(
    seed: int = 0,
    size: str = "medium",
) -> tuple[dict[str, Any], str]:
    """SKUs into storage bins."""
    data = _generate_instance_from_config(
        seed,
        WAREHOUSE_BIN_CONFIG,
        "warehouse_bins",
        size_label=size,
    )
    C = data["bin_capacity"]
    unit = data["bin_unit"]
    n_items = len(data["items"])
    item_summary = _describe_items_for_narrative_bp(data)
    scale_phrase = _size_flavor_bp("warehouse_bins", size)

    prompt = (
        f"The problem comes from {scale_phrase}. The warehouse uses standardized storage bins on shelving "
        f"racks, each with capacity {C} {unit}. A new assortment of {n_items} stock keeping units (SKUs) "
        "needs to be slotted into these bins for the next replenishment cycle.\n\n"
        f"{item_summary}\n\n"
        "Each SKU must be assigned to exactly one bin, and the total size of all SKUs placed in any bin cannot "
        "exceed that bin's capacity. If too many bins are opened, the warehouse consumes excess rack space and "
        "increases picking complexity. The warehouse manager therefore wants to pack all SKUs into as few bins "
        "as possible while respecting bin capacities."
    )
    prompt = prompt + "\n\n" + _bp_output_instructions()
    return data, prompt


# Mapping from problem type name to generator
PROBLEM_GENERATORS: dict[str, Callable[[int, str], tuple[dict[str, Any], str]]] = {
    "ecom_boxes": generate_ecom_box_instance_and_prompt,
    "container_loading": generate_container_loading_instance_and_prompt,
    "truck_loading": generate_truck_loading_instance_and_prompt,
    "warehouse_bins": generate_warehouse_bins_instance_and_prompt,
}


# ---------------------------------------------------------------------------
# Writing instances
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
        description="Generate bin packing benchmark data points for different business problem types."
    )
    parser.add_argument(
        "--problem-type",
        choices=["all"] + sorted(PROBLEM_GENERATORS.keys()),
        default="all",
        help="Which business variant to generate (or 'all').",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/cg/bp"), help="Base output directory.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed (instances use seed + index).")

    args = parser.parse_args()

    if args.problem_type == "all":
        problem_types = list(PROBLEM_GENERATORS.keys())  # all problem types
    else:
        problem_types = [args.problem_type]  # only the specified problem type(s)

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
