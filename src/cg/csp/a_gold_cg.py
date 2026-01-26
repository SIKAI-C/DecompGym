# GroundTruthCG.py
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


def initial_patterns(roll_width: int, widths: list[int]) -> list[list[int]]:
    """Single-type patterns: for each item i, take as many copies as fit on one roll.
    This is the standard feasible initialization. :contentReference[oaicite:13]{index=13}
    """
    m = len(widths)
    patterns: list[list[int]] = []
    for i in range(m):
        max_count = roll_width // widths[i]
        if max_count > 0:
            pat = [0] * m
            pat[i] = max_count
            patterns.append(pat)
    if not patterns:
        raise RuntimeError("No initial patterns constructed; check data.")
    return patterns


def solve_master_lp(
    widths: list[int],
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
    m = len(widths)
    P = range(len(patterns))
    items = range(m)

    model = gp.Model("csp_master_lp")
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

    # For a minimization model with >= constraints, Gurobi's Pi are the dual
    # prices π_i (non-negative). No sign flip needed.
    duals = [c.Pi for c in demand_constrs]

    return obj, duals


def solve_pricing(
    roll_width: int,
    widths: list[int],
    duals: list[float],
) -> tuple[list[int], float]:
    """Pricing subproblem: unbounded knapsack

        max sum_i dual_i * a_i
        s.t. sum_i width_i * a_i <= roll_width
             a_i integer >= 0
             sum_i a_i >= 1

    Reduced cost of the new column is:

        rc = 1 - Z_price

    as in your CSP section (equation (43)). :contentReference[oaicite:15]{index=15}
    """
    m = len(widths)
    items = range(m)

    model = gp.Model("csp_pricing")
    model.setParam("OutputFlag", 0)

    a = model.addVars(items, vtype=GRB.INTEGER, lb=0, name="a")

    model.addConstr(
        gp.quicksum(widths[i] * a[i] for i in items) <= roll_width,
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
    reduced_cost = 1.0 - Z_price  # cost 1 per roll

    return pattern, reduced_cost


def solve_column_generation(
    roll_width: int,
    widths: list[int],
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
    m = len(widths)
    items = range(m)

    patterns: list[list[int]] = initial_patterns(roll_width, widths)

    # Column generation loop (LP-based)
    while True:
        _, duals = solve_master_lp(widths, demands, patterns)
        pattern, rc = solve_pricing(roll_width, widths, duals)

        if rc >= -rc_tol:
            # No improving pattern
            break

        if any(pattern) and pattern not in patterns:
            patterns.append(pattern)

    # Integer master over generated patterns
    P = range(len(patterns))

    model = gp.Model("csp_master_int")
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
                    "cuts": [{"item_id": i, "count": patterns[p][i]} for i in items if patterns[p][i] > 0],
                }
            )

    sol = {
        "status": "optimal",
        "method": "column_generation",
        "objective_value": obj,
        "solution": {"patterns": used_patterns},
    }
    return sol


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute column-generation ground truth for a CSP instance.")
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

    roll_width, widths, demands = load_data(str(data_path))

    for w in widths:
        if w > roll_width:
            raise ValueError(f"Item width {w} exceeds roll width {roll_width}; infeasible instance.")

    sol = solve_column_generation(roll_width, widths, demands)
    out_path.write_text(json.dumps(sol, indent=2))
    print(f"CG objective_value={sol['objective_value']} -> {out_path}")


if __name__ == "__main__":
    main()
