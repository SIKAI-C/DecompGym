import argparse
import json
from pathlib import Path
from typing import Any


def _load_gap_data(data: dict[str, Any]) -> tuple[list[list[int]], list[list[float]], list[int]]:
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


def _assignments_from_patterns(patterns: list[dict[str, Any]]) -> list[dict[str, int]]:
    assignments: list[dict[str, int]] = []
    for pat in patterns:
        if not isinstance(pat, dict):
            continue
        machine_id = pat.get("machine_id")
        jobs = pat.get("jobs", [])
        if machine_id is None or not isinstance(jobs, list):
            continue
        for job_id in jobs:
            assignments.append({"job_id": job_id, "machine_id": machine_id})
    return assignments


def verify_solution(data: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    try:
        a, c, b = _load_gap_data(data)
    except Exception as exc:
        return {"feasible": False, "objective_value": None, "message": f"data load error: {exc}"}

    n_jobs = len(a)
    n_machines = len(b)
    solution = output.get("solution", {})
    assignments = solution.get("assignments")
    if assignments is None:
        patterns = solution.get("patterns", [])
        assignments = _assignments_from_patterns(patterns) if isinstance(patterns, list) else None

    if not isinstance(assignments, list):
        return {"feasible": False, "objective_value": None, "message": "solution.assignments missing or not a list"}

    assigned = [-1] * n_jobs
    load = [0] * n_machines
    objective = 0.0

    for idx, assign in enumerate(assignments):
        if not isinstance(assign, dict):
            return {"feasible": False, "objective_value": None, "message": f"assignment {idx} not a dict"}
        job_id = assign.get("job_id")
        machine_id = assign.get("machine_id")
        try:
            job_id = int(job_id)
            machine_id = int(machine_id)
        except (TypeError, ValueError):
            return {"feasible": False, "objective_value": None, "message": f"assignment {idx} ids not int"}

        if job_id < 0 or job_id >= n_jobs:
            return {"feasible": False, "objective_value": None, "message": f"job_id out of range: {job_id}"}
        if machine_id < 0 or machine_id >= n_machines:
            return {"feasible": False, "objective_value": None, "message": f"machine_id out of range: {machine_id}"}
        if assigned[job_id] != -1:
            return {"feasible": False, "objective_value": None, "message": f"job {job_id} assigned multiple times"}

        assigned[job_id] = machine_id
        load[machine_id] += a[job_id][machine_id]
        objective += c[job_id][machine_id]

    missing = [i for i, m in enumerate(assigned) if m == -1]
    if missing:
        return {"feasible": False, "objective_value": None, "message": f"unassigned jobs: {missing}"}

    for j in range(n_machines):
        if load[j] > b[j]:
            return {"feasible": False, "objective_value": None, "message": f"capacity exceeded on machine {j}"}

    return {"feasible": True, "objective_value": float(objective), "message": "ok"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify GAP solution feasibility.")
    parser.add_argument("--dir", type=Path, default=Path("."), help="Directory with data.json and output.json")
    parser.add_argument("--output", type=str, default="output.json", help="Output file name")
    args = parser.parse_args()

    data = json.loads((args.dir / "data.json").read_text())
    output = json.loads((args.dir / args.output).read_text())
    result = verify_solution(data, output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
