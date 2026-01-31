import argparse
import json
from pathlib import Path
from typing import Any


INT_TOL = 1e-6


def _coerce_int(value: Any, field: str) -> tuple[int | None, str | None]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None, f"{field} is not numeric"
    if abs(val - round(val)) > INT_TOL:
        return None, f"{field} is not an integer: {value}"
    return round(val), None


def verify_solution(data: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    try:
        demand = [int(d) for d in data["demand"]]
        L_min = int(data["shift_min_length"])
        L_max = int(data["shift_max_length"])
        shift_cost = float(data.get("shift_cost", 1.0))
    except Exception as exc:
        return {"feasible": False, "objective_value": None, "message": f"data load error: {exc}"}

    solution = output.get("solution", {})
    shifts = solution.get("shifts")
    if not isinstance(shifts, list):
        return {"feasible": False, "objective_value": None, "message": "solution.shifts missing or not a list"}

    T = len(demand)
    coverage = [0] * T
    total_shifts = 0

    for idx, shift in enumerate(shifts):
        if not isinstance(shift, dict):
            return {"feasible": False, "objective_value": None, "message": f"shift {idx} not a dict"}

        start = shift.get("start_slot", shift.get("start"))
        end = shift.get("end_slot", shift.get("end"))
        try:
            start = int(start)
            end = int(end)
        except (TypeError, ValueError):
            return {"feasible": False, "objective_value": None, "message": f"shift {idx} start/end not int"}

        count_raw = shift.get("count", shift.get("usage", 1))
        count, err = _coerce_int(count_raw, f"shifts[{idx}].count")
        if err:
            return {"feasible": False, "objective_value": None, "message": err}
        if count < 0:
            return {"feasible": False, "objective_value": None, "message": f"shift {idx} has negative count"}
        if count == 0:
            continue

        if start < 0 or end < 0 or start >= T or end >= T:
            return {"feasible": False, "objective_value": None, "message": f"shift {idx} out of range"}
        if end < start:
            return {"feasible": False, "objective_value": None, "message": f"shift {idx} end before start"}

        length = end - start + 1
        if length < L_min or length > L_max:
            return {"feasible": False, "objective_value": None, "message": f"shift {idx} length outside bounds"}

        total_shifts += count
        for t in range(start, end + 1):
            coverage[t] += count

    for t, need in enumerate(demand):
        if coverage[t] < need:
            return {"feasible": False, "objective_value": None, "message": f"demand not met at slot {t}"}

    return {"feasible": True, "objective_value": float(shift_cost * total_shifts), "message": "ok"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify shift scheduling solution feasibility.")
    parser.add_argument("--dir", type=Path, default=Path("."), help="Directory with data.json and output.json")
    parser.add_argument("--output", type=str, default="output.json", help="Output file name")
    args = parser.parse_args()

    data = json.loads((args.dir / "data.json").read_text())
    output = json.loads((args.dir / args.output).read_text())
    result = verify_solution(data, output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
