# Ground truth monolithic solver for shift scheduling
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB


def load_shift_data(dir_path: Path) -> tuple[list[int], int, int, float]:
    data = json.loads((dir_path / "data.json").read_text())
    demand = [int(d) for d in data["demand"]]
    L_min = int(data["shift_min_length"])
    L_max = int(data["shift_max_length"])
    shift_cost = float(data.get("shift_cost", 1.0))
    return demand, L_min, L_max, shift_cost


def enumerate_shifts(T: int, L_min: int, L_max: int) -> list[tuple[int, int]]:
    shifts: list[tuple[int, int]] = []
    for s in range(T):
        for e in range(s + L_min - 1, min(T - 1, s + L_max - 1) + 1):
            shifts.append((s, e))
    return shifts


def solve_monolithic(
    demand: list[int],
    L_min: int,
    L_max: int,
    shift_cost: float,
) -> dict[str, Any]:
    T = len(demand)
    shifts = enumerate_shifts(T, L_min, L_max)
    P = range(len(shifts))

    model = gp.Model("shift_monolith")
    model.setParam("OutputFlag", 0)

    x = model.addVars(P, vtype=GRB.INTEGER, lb=0, name="x")

    for t in range(T):
        model.addConstr(
            gp.quicksum(x[p] for p in P if shifts[p][0] <= t <= shifts[p][1]) >= demand[t],
            name=f"cov_{t}",
        )

    model.setObjective(
        shift_cost * gp.quicksum(x[p] for p in P),
        GRB.MINIMIZE,
    )
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Monolithic shift model not optimal, status={model.Status}")

    solution_shifts = []
    for p in P:
        cnt = int(round(x[p].X))
        if cnt > 0:
            s, e = shifts[p]
            solution_shifts.append({"shift_id": p, "start_slot": s, "end_slot": e, "count": cnt})

    sol = {
        "status": "optimal",
        "method": "shift_monolithic_set_cover",
        "objective_value": float(model.ObjVal),
        "solution": {"shifts": solution_shifts},
    }
    return sol


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve shift scheduling instance monolithically.")
    parser.add_argument("--dir", type=Path, default=Path("."), help="Directory containing data.json")
    args = parser.parse_args()

    demand, L_min, L_max, shift_cost = load_shift_data(args.dir)
    sol = solve_monolithic(demand, L_min, L_max, shift_cost)

    out_path = args.dir / "ground_truth_monolithic.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()
