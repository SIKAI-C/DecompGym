# Ground truth column generation for shift scheduling
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


def initial_shifts(T: int, L_min: int) -> list[tuple[int, int]]:
    seen = set()
    shifts: list[tuple[int, int]] = []
    for t in range(T):
        e = min(T - 1, t + L_min - 1)
        s = e - (L_min - 1)
        pair = (s, e)
        if pair not in seen:
            seen.add(pair)
            shifts.append(pair)
    return shifts


def solve_master_lp(
    demand: list[int],
    shifts: list[tuple[int, int]],
    shift_cost: float,
) -> tuple[float, list[float]]:
    T = len(demand)
    P = range(len(shifts))

    model = gp.Model("shift_master_lp")
    model.setParam("OutputFlag", 0)

    x = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0.0, name="x")

    coverage_constrs = []
    for t in range(T):
        cstr = model.addConstr(
            gp.quicksum(x[p] for p in P if shifts[p][0] <= t <= shifts[p][1]) >= demand[t],
            name=f"cov_{t}",
        )
        coverage_constrs.append(cstr)

    model.setObjective(shift_cost * gp.quicksum(x[p] for p in P), GRB.MINIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Master LP not optimal, status={model.Status}")

    duals = [c.Pi for c in coverage_constrs]
    return float(model.ObjVal), duals


def pricing(
    duals: list[float],
    L_min: int,
    L_max: int,
    shift_cost: float,
    existing: set[tuple[int, int]],
    rc_tol: float = 1e-6,
) -> tuple[float, tuple[int, int] | None]:
    T = len(duals)
    prefix = [0.0] * (T + 1)
    for t in range(T):
        prefix[t + 1] = prefix[t] + duals[t]

    best_rc = 0.0
    best_shift = None
    for s in range(T):
        max_e = min(T - 1, s + L_max - 1)
        min_e = s + L_min - 1
        for e in range(min_e, max_e + 1):
            dual_sum = prefix[e + 1] - prefix[s]
            rc = shift_cost - dual_sum
            if rc < best_rc - rc_tol:
                candidate = (s, e)
                if candidate not in existing:
                    best_rc = rc
                    best_shift = candidate
    return best_rc, best_shift


def solve_column_generation(
    demand: list[int],
    L_min: int,
    L_max: int,
    shift_cost: float,
    rc_tol: float = 1e-6,
) -> dict[str, Any]:
    T = len(demand)
    shifts = initial_shifts(T, L_min)
    seen = set(shifts)

    while True:
        _, duals = solve_master_lp(demand, shifts, shift_cost)
        best_rc, best_shift = pricing(duals, L_min, L_max, shift_cost, seen, rc_tol=rc_tol)
        if best_rc >= -rc_tol or best_shift is None:
            break
        shifts.append(best_shift)
        seen.add(best_shift)

    P = range(len(shifts))
    model_int = gp.Model("shift_master_int")
    model_int.setParam("OutputFlag", 0)
    x = model_int.addVars(P, vtype=GRB.INTEGER, lb=0, name="x")

    for t in range(T):
        model_int.addConstr(
            gp.quicksum(x[p] for p in P if shifts[p][0] <= t <= shifts[p][1]) >= demand[t],
            name=f"cov_{t}",
        )

    model_int.setObjective(shift_cost * gp.quicksum(x[p] for p in P), GRB.MINIMIZE)
    model_int.optimize()

    if model_int.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Integer master not optimal, status={model_int.Status}")

    solution_shifts = []
    for p in P:
        cnt = int(round(x[p].X))
        if cnt > 0:
            s, e = shifts[p]
            solution_shifts.append({"shift_id": p, "start_slot": s, "end_slot": e, "count": cnt})

    sol = {
        "status": "optimal",
        "method": "shift_column_generation",
        "objective_value": float(model_int.ObjVal),
        "solution": {"shifts": solution_shifts},
    }
    return sol


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve shift scheduling instance via column generation.")
    parser.add_argument("--dir", type=Path, default=Path("."), help="Directory containing data.json")
    args = parser.parse_args()

    demand, L_min, L_max, shift_cost = load_shift_data(args.dir)
    sol = solve_column_generation(demand, L_min, L_max, shift_cost)

    out_path = args.dir / "ground_truth_cg.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()
