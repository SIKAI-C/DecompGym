# GroundTruthMonolith.py
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB


def load_data(path: str = "data.json") -> tuple[int, list[int], list[int]]:
    """Load CSP instance from data.json.

    Returns:
        roll_width: W
        widths: list of widths (ordered by item id)
        demands: list of demands (ordered by item id)

    """
    data = json.loads(Path(path).read_text())
    if "roll_width" not in data:
        roll_width = int(data["stock_size"])
        items = sorted(data["item_types"], key=lambda it: it["id"])
        widths = [int(it["size"]) for it in items]
        demands = [int(it["demand"]) for it in items]
    else:
        roll_width = int(data["roll_width"])
        items = sorted(data["item_types"], key=lambda it: it["id"])
        widths = [int(it["width"]) for it in items]
        demands = [int(it["demand"]) for it in items]
    return roll_width, widths, demands


def solve_monolithic(roll_width: int, widths: list[int], demands: list[int]) -> dict[str, Any]:
    """Monolithic MIP for cutting stock:

    - Rolls r = 0..R-1, where R is an upper bound on #rolls.
    - Variables:
        y_r = 1 if roll r is used (binary)
        x_{i,r} = number of pieces of type i cut on roll r (integer >= 0)
    - Constraints:
        sum_i width_i * x_{i,r} <= roll_width * y_r   for each r
        sum_r x_{i,r} >= demand_i                     for each i
    - Objective: minimize sum_r y_r
    """
    n_items = len(widths)
    # Very simple upper bound: each roll carries at least 1 item
    max_rolls = sum(demands)

    model = gp.Model("csp_monolithic")
    model.setParam("OutputFlag", 0)

    rolls = range(max_rolls)
    items = range(n_items)

    y = model.addVars(rolls, vtype=GRB.BINARY, name="y")
    x = model.addVars(items, rolls, vtype=GRB.INTEGER, lb=0, name="x")

    # Capacity constraints
    for r in rolls:
        model.addConstr(
            gp.quicksum(widths[i] * x[i, r] for i in items) <= roll_width * y[r],
            name=f"cap_{r}",
        )

    # Demand constraints
    for i in items:
        model.addConstr(
            gp.quicksum(x[i, r] for r in rolls) >= demands[i],
            name=f"demand_{i}",
        )

    # Objective: minimize number of rolls
    model.setObjective(gp.quicksum(y[r] for r in rolls), GRB.MINIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Monolithic MIP not optimal, status={model.Status}")

    obj = float(model.ObjVal)

    # Represent each used roll as a pattern with usage=1
    patterns = []
    pattern_id = 0
    for r in rolls:
        if y[r].X > 0.5:
            cuts = []
            for i in items:
                cnt = int(round(x[i, r].X))
                if cnt > 0:
                    cuts.append({"item_id": i, "count": cnt})
            patterns.append({"pattern_id": pattern_id, "usage": 1, "cuts": cuts})
            pattern_id += 1

    sol = {
        "status": "optimal",
        "method": "monolithic_mip",
        "objective_value": obj,
        "solution": {"patterns": patterns},
    }
    return sol


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute monolithic MIP ground truth for a CSP instance.")
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

    roll_width, widths, demands = load_data(str(data_path))

    for w in widths:
        if w > roll_width:
            raise ValueError(f"Item width {w} exceeds roll width {roll_width}; infeasible instance.")

    sol = solve_monolithic(roll_width, widths, demands)
    out_path.write_text(json.dumps(sol, indent=2))
    print(f"Monolithic objective_value={sol['objective_value']} -> {out_path}")


if __name__ == "__main__":
    main()
