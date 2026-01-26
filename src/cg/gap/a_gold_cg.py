# Ground truth column generation solver for GAP
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


def pattern_key(machine: int, jobs_vec: list[int]) -> tuple[int, tuple[int, ...]]:
    return machine, tuple(jobs_vec)


def initial_patterns(a: list[list[int]], b: list[int]) -> list[dict[str, Any]]:
    n_jobs = len(a)
    n_machines = len(b)
    patterns: list[dict[str, Any]] = []

    for j in range(n_machines):
        patterns.append({"machine": j, "jobs": [0] * n_jobs})  # empty pattern keeps feasibility
        # single-job patterns
        for i in range(n_jobs):
            if a[i][j] <= b[j]:
                jobs_vec = [0] * n_jobs
                jobs_vec[i] = 1
                patterns.append({"machine": j, "jobs": jobs_vec})
        # greedy packed pattern to enable feasibility when jobs > machines
        order = sorted(range(n_jobs), key=lambda i: a[i][j])
        load = 0
        greedy = [0] * n_jobs
        for i in order:
            if load + a[i][j] <= b[j]:
                greedy[i] = 1
                load += a[i][j]
        if any(greedy):
            patterns.append({"machine": j, "jobs": greedy})
    return patterns


def solve_master_lp(
    patterns: list[dict[str, Any]],
    a: list[list[int]],
    c: list[list[float]],
    b: list[int],
) -> tuple[float, tuple[list[float], list[float]]]:
    n_jobs = len(a)
    n_machines = len(b)
    P = range(len(patterns))
    jobs = range(n_jobs)
    machines = range(n_machines)

    model = gp.Model("gap_master_lp")
    model.setParam("OutputFlag", 0)

    lam = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0.0, name="lambda")

    job_constrs = []
    for i in jobs:
        cstr = model.addConstr(
            gp.quicksum(patterns[p]["jobs"][i] * lam[p] for p in P) == 1,
            name=f"job_{i}",
        )
        job_constrs.append(cstr)

    machine_constrs = []
    for j in machines:
        cstr = model.addConstr(
            gp.quicksum(lam[p] for p in P if patterns[p]["machine"] == j) == 1,
            name=f"machine_{j}",
        )
        machine_constrs.append(cstr)

    def cost_of_pattern(p_idx: int) -> float:
        pat = patterns[p_idx]
        m = pat["machine"]
        return sum(c[i][m] * pat["jobs"][i] for i in jobs)

    model.setObjective(
        gp.quicksum(cost_of_pattern(p) * lam[p] for p in P),
        GRB.MINIMIZE,
    )
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Master LP not optimal, status={model.Status}")

    pi = [cstr.Pi for cstr in job_constrs]
    sigma = [cstr.Pi for cstr in machine_constrs]
    return float(model.ObjVal), (pi, sigma)


def solve_pricing_for_machine(
    machine: int,
    a: list[list[int]],
    c: list[list[float]],
    b: list[int],
    pi: list[float],
    sigma_j: float,
    rc_tol: float = 1e-9,
) -> tuple[list[int], float]:
    n_jobs = len(a)
    jobs = range(n_jobs)

    model = gp.Model(f"gap_pricing_{machine}")
    model.setParam("OutputFlag", 0)

    z = model.addVars(jobs, vtype=GRB.BINARY, name="z")
    model.addConstr(
        gp.quicksum(a[i][machine] * z[i] for i in jobs) <= b[machine],
        name="capacity",
    )
    model.addConstr(gp.quicksum(z[i] for i in jobs) >= 1, name="nonempty")

    model.setObjective(
        gp.quicksum((pi[i] - c[i][machine]) * z[i] for i in jobs) + sigma_j,
        GRB.MAXIMIZE,
    )
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        return [0] * n_jobs, 0.0

    jobs_vec = [int(round(z[i].X)) for i in jobs]
    pattern_cost = sum(c[i][machine] * jobs_vec[i] for i in jobs)
    reduced_cost = pattern_cost - sum(pi[i] * jobs_vec[i] for i in jobs) - sigma_j

    if sum(jobs_vec) == 0:
        reduced_cost = 0.0
    if reduced_cost > -rc_tol:
        reduced_cost = max(reduced_cost, 0.0)
    return jobs_vec, reduced_cost


def solve_column_generation(
    a: list[list[int]],
    c: list[list[float]],
    b: list[int],
    rc_tol: float = 1e-6,
) -> dict[str, Any]:
    n_jobs = len(a)
    n_machines = len(b)
    jobs = range(n_jobs)
    machines = range(n_machines)

    patterns = initial_patterns(a, b)
    seen = {pattern_key(p["machine"], p["jobs"]) for p in patterns}

    while True:
        _, (pi, sigma) = solve_master_lp(patterns, a, c, b)
        best_rc = 0.0
        new_patterns: list[dict[str, Any]] = []

        for j in machines:
            jobs_vec, rc = solve_pricing_for_machine(j, a, c, b, pi, sigma[j], rc_tol=rc_tol)
            if rc < best_rc:
                best_rc = rc
            key = pattern_key(j, jobs_vec)
            if rc < -rc_tol and key not in seen:
                seen.add(key)
                new_patterns.append({"machine": j, "jobs": jobs_vec})

        if best_rc >= -rc_tol or not new_patterns:
            break

        patterns.extend(new_patterns)

    P = range(len(patterns))
    model_int = gp.Model("gap_master_int")
    model_int.setParam("OutputFlag", 0)
    lam = model_int.addVars(P, vtype=GRB.BINARY, name="lambda")

    for i in jobs:
        model_int.addConstr(
            gp.quicksum(patterns[p]["jobs"][i] * lam[p] for p in P) == 1,
            name=f"job_{i}",
        )
    for j in machines:
        model_int.addConstr(
            gp.quicksum(lam[p] for p in P if patterns[p]["machine"] == j) == 1,
            name=f"machine_{j}",
        )

    def cost_of_pattern(p_idx: int) -> float:
        pat = patterns[p_idx]
        m = pat["machine"]
        return sum(c[i][m] * pat["jobs"][i] for i in jobs)

    model_int.setObjective(
        gp.quicksum(cost_of_pattern(p) * lam[p] for p in P),
        GRB.MINIMIZE,
    )
    model_int.optimize()

    if model_int.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Integer master not optimal, status={model_int.Status}")

    assignments = []
    chosen_patterns = []
    for p in P:
        val = lam[p].X
        if val > 0.5:
            pat = patterns[p]
            chosen_patterns.append(
                {
                    "pattern_id": p,
                    "machine_id": pat["machine"],
                    "jobs": [i for i in jobs if pat["jobs"][i] == 1],
                    "usage": 1,
                }
            )
            for i in jobs:
                if pat["jobs"][i] == 1:
                    assignments.append({"job_id": i, "machine_id": pat["machine"]})

    sol = {
        "status": "optimal",
        "method": "gap_column_generation",
        "objective_value": float(model_int.ObjVal),
        "solution": {
            "assignments": assignments,
            "patterns": chosen_patterns,
        },
    }
    return sol


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve GAP instance with column generation.")
    parser.add_argument("--dir", type=Path, default=Path("."), help="Directory containing data.json")
    args = parser.parse_args()

    a, c, b = load_gap_data(args.dir)
    sol = solve_column_generation(a, c, b)

    out_path = args.dir / "ground_truth_cg.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()
