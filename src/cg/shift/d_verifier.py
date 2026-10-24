import argparse
import json
from pathlib import Path
from typing import Any

from src.utils.output_contract import integer_value


def _coerce_int(value: Any, field: str) -> tuple[int | None, str | None]:
    if not integer_value(value):
        return None, f"{field} must be a finite integer JSON number: {value!r}"
    return int(value), None


def _load_json_if_path(obj: Any) -> dict[str, Any]:
    if isinstance(obj, (str, Path)):
        return json.loads(Path(obj).read_text())
    if isinstance(obj, dict):
        return obj
    raise TypeError("expected dict or path-like input")


def _result(ok: bool, message: str, objective_value: float | None = None) -> dict[str, Any]:
    return {
        "ok": ok,
        "issues": [] if ok else [message],
        "objective_value": objective_value,
    }


def verify_solution(data: Any, output: Any, tol: float | None = None) -> dict[str, Any]:
    _ = tol
    try:
        data_dict = _load_json_if_path(data)
        output_dict = _load_json_if_path(output)
        demand = [int(d) for d in data_dict["demand"]]
        L_min = int(data_dict["shift_min_length"])
        L_max = int(data_dict["shift_max_length"])
        shift_cost = float(data_dict.get("shift_cost", 1.0))
    except Exception as exc:
        return _result(False, f"data load error: {exc}")

    solution = output_dict.get("solution", {})
    if not isinstance(solution, dict):
        return _result(False, "solution must be an object")
    shifts = solution.get("shifts")
    if not isinstance(shifts, list):
        return _result(False, "solution.shifts missing or not a list")

    T = len(demand)
    coverage = [0] * T
    total_shifts = 0

    for idx, shift in enumerate(shifts):
        if not isinstance(shift, dict):
            return _result(False, f"shift {idx} not a dict")

        start = shift.get("start")
        end = shift.get("end")
        if not integer_value(start) or not integer_value(end):
            return _result(False, f"shift {idx} start/end must be finite integer JSON numbers")
        start, end = int(start), int(end)

        count_raw = shift.get("usage", 1)
        count, err = _coerce_int(count_raw, f"shifts[{idx}].usage")
        if err:
            return _result(False, err)
        if count < 0:
            return _result(False, f"shift {idx} has negative count")
        if count == 0:
            continue

        if start < 0 or end < 0 or start >= T or end >= T:
            return _result(False, f"shift {idx} out of range")
        if end < start:
            return _result(False, f"shift {idx} end before start")

        length = end - start + 1
        if length < L_min or length > L_max:
            return _result(False, f"shift {idx} length outside bounds")

        total_shifts += count
        for t in range(start, end + 1):
            coverage[t] += count

    for t, need in enumerate(demand):
        if coverage[t] < need:
            return _result(False, f"demand not met at slot {t}")

    return _result(True, "ok", float(shift_cost * total_shifts))


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
