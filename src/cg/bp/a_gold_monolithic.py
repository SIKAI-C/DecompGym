# bp__gt_monolith.py
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB


def load_data(path: str = "data.json") -> tuple[int, list[int], list[int]]:
    """Load bin packing instance from data.json.

    Returns:
        bin_capacity: C
        sizes: list of item sizes (ordered by type)
        demands: list of demands per type

    """
    data = json.loads(Path(path).read_text())

    # Capacity key variants
    if "bin_capacity" in data:
        bin_capacity = int(data["bin_capacity"])
    elif "stock_size" in data:
        bin_capacity = int(data["stock_size"])
    elif "roll_width" in data:
        bin_capacity = int(data["roll_width"])
    else:
        raise KeyError("Expected one of 'bin_capacity', 'stock_size', or 'roll_width' in data.json")

    # Item schema variants
    if "item_types" in data:
        items = sorted(data["item_types"], key=lambda it: it["id"])
        # allow either "size" (BPP style) or "width" (CSP style)
        sizes = [int(it.get("size", it.get("width"))) for it in items]
        demands = [int(it.get("demand", 1)) for it in items]
    elif "items" in data:
        # Each item appears once; aggregate by size to get (size, demand)
        items = data["items"]
        size_counts: dict[int, int] = {}
        for it in items:
            s = int(it["size"])
            size_counts[s] = size_counts.get(s, 0) + 1
        sizes = sorted(size_counts.keys())
        demands = [size_counts[s] for s in sizes]
    else:
        raise KeyError("Expected 'item_types' or 'items' in data.json")

    return bin_capacity, sizes, demands


def solve_monolithic(bin_capacity: int, sizes: list[int], demands: list[int]) -> dict[str, Any]:
    """Monolithic MIP for bin packing (identical bins):

    - Bins b = 0..B-1, where B is an upper bound on #bins.
    - Variables:
        y_b = 1 if bin b is used (binary)
        x_{i,b} = number of items of type i assigned to bin b (integer >= 0)
    - Constraints:
        sum_i size_i * x_{i,b} <= bin_capacity * y_b   for each b
        sum_b x_{i,b} >= demand_i                      for each i
    - Objective: minimize sum_b y_b
    """
    n_items = len(sizes)
    # Very simple upper bound: each item type contributes demand_i items,
    # and each bin must hold at least one item.
    max_bins = sum(demands)

    model = gp.Model("bp_monolithic")
    model.setParam("OutputFlag", 0)

    bins = range(max_bins)
    items = range(n_items)

    y = model.addVars(bins, vtype=GRB.BINARY, name="y")
    x = model.addVars(items, bins, vtype=GRB.INTEGER, lb=0, name="x")

    # Capacity constraints
    for b in bins:
        model.addConstr(
            gp.quicksum(sizes[i] * x[i, b] for i in items) <= bin_capacity * y[b],
            name=f"cap_{b}",
        )

    # Demand constraints
    for i in items:
        model.addConstr(
            gp.quicksum(x[i, b] for b in bins) >= demands[i],
            name=f"demand_{i}",
        )

    # Objective: minimize number of bins
    model.setObjective(gp.quicksum(y[b] for b in bins), GRB.MINIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Monolithic MIP not optimal, status={model.Status}")

    obj = float(model.ObjVal)

    # Represent each used bin as a pattern with usage=1
    patterns = []
    pattern_id = 0
    for b in bins:
        if y[b].X > 0.5:
            contents = []
            for i in items:
                cnt = int(round(x[i, b].X))
                if cnt > 0:
                    contents.append({"item_id": i, "count": cnt})
            patterns.append(
                {
                    "pattern_id": pattern_id,
                    "usage": 1,
                    "contents": contents,
                }
            )
            pattern_id += 1

    sol = {
        "status": "optimal",
        "method": "bin_packing_monolithic_mip",
        "objective_value": obj,
        "solution": {"bins": patterns},
    }
    return sol


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute monolithic MIP ground truth for a bin packing instance.")
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Directory containing data.json (output written there too). Default: current directory.",
    )
    args = parser.parse_args()

    work_dir = Path(args.dir)
    data_path = work_dir / "data.json"
    out_path = work_dir / "ground_truth_monolithic.json"

    bin_capacity, sizes, demands = load_data(str(data_path))

    for s in sizes:
        if s > bin_capacity:
            raise ValueError(f"Item size {s} exceeds bin capacity {bin_capacity}; infeasible instance.")

    sol = solve_monolithic(bin_capacity, sizes, demands)
    out_path.write_text(json.dumps(sol, indent=2))
    print(f"[BP monolithic] objective_value={sol['objective_value']} -> {out_path}")


if __name__ == "__main__":
    main()
