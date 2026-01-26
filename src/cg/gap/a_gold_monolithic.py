# Ground truth monolithic solver for GAP
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB


def load_gap_data(dir_path: Path) -> tuple[list[list[int]], list[list[float]], list[int]]:
    data_path = dir_path / "data.json"
    data = json.loads(data_path.read_text())

    machines = sorted(data["machines"], key=lambda m: m["id"])
    jobs = sorted(data["jobs"], key=lambda j: j["id"])

    b = [int(m["capacity"]) for m in machines]
    n_jobs = len(jobs)
    n_machines = len(machines)

    a = [[0] * n_machines for _ in range(n_jobs)]
    c = [[0.0] * n_machines for _ in range(n_jobs)]

    for job in jobs:
        i = int(job["id"])
        for assignment in sorted(job["assignments"], key=lambda x: x["machine_id"]):
            j = int(assignment["machine_id"])
            a[i][j] = int(assignment["resource"])
            c[i][j] = float(assignment["cost"])

    return a, c, b


def solve_monolithic(a: list[list[int]], c: list[list[float]], b: list[int]) -> dict[str, Any]:
    n_jobs = len(a)
    n_machines = len(b)
    jobs = range(n_jobs)
    machines = range(n_machines)

    model = gp.Model("gap_monolith")
    model.setParam("OutputFlag", 0)

    x = model.addVars(n_jobs, n_machines, vtype=GRB.BINARY, name="x")

    for i in jobs:
        model.addConstr(gp.quicksum(x[i, j] for j in machines) == 1, name=f"assign_{i}")

    for j in machines:
        model.addConstr(
            gp.quicksum(a[i][j] * x[i, j] for i in jobs) <= b[j],
            name=f"cap_{j}",
        )

    model.setObjective(
        gp.quicksum(c[i][j] * x[i, j] for i in jobs for j in machines),
        GRB.MINIMIZE,
    )
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Monolithic GAP model not optimal, status={model.Status}")

    assignments = []
    for i in jobs:
        for j in machines:
            if x[i, j].X > 0.5:
                assignments.append({"job_id": i, "machine_id": j})

    sol = {
        "status": "optimal",
        "method": "gap_monolithic_mip",
        "objective_value": float(model.ObjVal),
        "solution": {
            "assignments": assignments,
        },
    }
    return sol


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve GAP instance with monolithic MIP.")
    parser.add_argument("--dir", type=Path, default=Path("."), help="Directory containing data.json")
    args = parser.parse_args()

    a, c, b = load_gap_data(args.dir)
    sol = solve_monolithic(a, c, b)

    out_path = args.dir / "ground_truth_monolithic.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()
