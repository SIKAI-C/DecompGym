# bp__gt_cg.py
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


def initial_patterns(bin_capacity: int, sizes: list[int]) -> list[list[int]]:
    """Single-type patterns: for each item type i, take as many copies as fit in one bin.
    Same idea as CSP; this yields a feasible initialization.
    """
    m = len(sizes)
    patterns: list[list[int]] = []
    for i in range(m):
        max_count = bin_capacity // sizes[i]
        if max_count > 0:
            pat = [0] * m
            pat[i] = max_count
            patterns.append(pat)
    if not patterns:
        raise RuntimeError("No initial patterns constructed; check data.")
    return patterns


def solve_master_lp(
    sizes: list[int],
    demands: list[int],
    patterns: list[list[int]],
) -> tuple[float, list[float]]:
    """Restricted master LP:

        min sum_p lambda_p
        s.t. sum_p a_i^p * lambda_p >= demand_i   for all i
             lambda_p >= 0

    Returns:
        objective_value, duals π_i for demand constraints (π_i >= 0).

    """
    m = len(sizes)
    P = range(len(patterns))
    items = range(m)

    model = gp.Model("bp_master_lp")
    model.setParam("OutputFlag", 0)

    lam = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0.0, name="lambda")

    demand_constrs = []
    for i in items:
        c = model.addConstr(
            gp.quicksum(patterns[p][i] * lam[p] for p in P) >= demands[i],
            name=f"demand_{i}",
        )
        demand_constrs.append(c)

    model.setObjective(gp.quicksum(lam[p] for p in P), GRB.MINIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Master LP not optimal, status={model.Status}")

    obj = float(model.ObjVal)

    duals = [c.Pi for c in demand_constrs]
    return obj, duals


def solve_pricing(
    bin_capacity: int,
    sizes: list[int],
    duals: list[float],
) -> tuple[list[int], float]:
    """Pricing subproblem: unbounded knapsack

        max sum_i dual_i * a_i
        s.t. sum_i size_i * a_i <= bin_capacity
             a_i integer >= 0
             sum_i a_i >= 1

    Reduced cost of the new column is:

        rc = 1 - Z_price

    We add a column if rc < 0.
    """
    m = len(sizes)
    items = range(m)

    model = gp.Model("bp_pricing")
    model.setParam("OutputFlag", 0)

    a = model.addVars(items, vtype=GRB.INTEGER, lb=0, name="a")

    model.addConstr(
        gp.quicksum(sizes[i] * a[i] for i in items) <= bin_capacity,
        name="capacity",
    )
    model.addConstr(
        gp.quicksum(a[i] for i in items) >= 1,
        name="nontrivial",
    )

    model.setObjective(
        gp.quicksum(duals[i] * a[i] for i in items),
        GRB.MAXIMIZE,
    )
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Pricing problem not optimal, status={model.Status}")

    Z_price = float(model.ObjVal)
    pattern = [int(round(a[i].X)) for i in items]
    reduced_cost = 1.0 - Z_price  # cost 1 per bin

    return pattern, reduced_cost


def solve_column_generation(
    bin_capacity: int,
    sizes: list[int],
    demands: list[int],
    rc_tol: float = 1e-6,
) -> dict[str, Any]:
    """Column generation + integer master:

    1) Start with initial patterns.
    2) While reduced_cost < 0:
         - Solve LP restricted master.
         - Get duals.
         - Solve knapsack pricing.
         - Add new pattern.
    3) Solve integer master over final pattern set.
    """
    m = len(sizes)
    items = range(m)

    patterns: list[list[int]] = initial_patterns(bin_capacity, sizes)

    # Column generation loop (LP-based)
    while True:
        _, duals = solve_master_lp(sizes, demands, patterns)
        pattern, rc = solve_pricing(bin_capacity, sizes, duals)

        if rc >= -rc_tol:
            # No improving pattern
            break

        if any(pattern) and pattern not in patterns:
            patterns.append(pattern)

    # Integer master over generated patterns
    P = range(len(patterns))

    model = gp.Model("bp_master_int")
    model.setParam("OutputFlag", 0)

    x = model.addVars(P, vtype=GRB.INTEGER, lb=0, name="x")

    for i in items:
        model.addConstr(
            gp.quicksum(patterns[p][i] * x[p] for p in P) >= demands[i],
            name=f"demand_{i}",
        )

    model.setObjective(gp.quicksum(x[p] for p in P), GRB.MINIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Integer master not optimal, status={model.Status}")

    obj = float(model.ObjVal)

    used_patterns = []
    for p in P:
        val = x[p].X
        if val > 1e-6:
            used_patterns.append(
                {
                    "pattern_id": p,
                    "usage": int(round(val)),
                    "contents": [{"item_id": i, "count": patterns[p][i]} for i in items if patterns[p][i] > 0],
                }
            )

    sol = {
        "status": "optimal",
        "method": "bin_packing_column_generation",
        "objective_value": obj,
        "solution": {"bins": used_patterns},
    }
    return sol


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute column-generation ground truth for a bin packing instance.")
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Directory containing data.json (output written there too). Default: current directory.",
    )
    args = parser.parse_args()

    work_dir = Path(args.dir)
    data_path = work_dir / "data.json"
    out_path = work_dir / "ground_truth_cg.json"

    bin_capacity, sizes, demands = load_data(str(data_path))

    for s in sizes:
        if s > bin_capacity:
            raise ValueError(f"Item size {s} exceeds bin capacity {bin_capacity}; infeasible instance.")

    sol = solve_column_generation(bin_capacity, sizes, demands)
    out_path.write_text(json.dumps(sol, indent=2))
    print(f"[BP CG] objective_value={sol['objective_value']} -> {out_path}")


if __name__ == "__main__":
    main()
