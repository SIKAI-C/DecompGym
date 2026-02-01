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


def _load_csp_data(data: dict[str, Any]) -> tuple[int, list[int], list[int]]:
    roll_width = int(data.get("roll_width", data.get("stock_size")))
    items = sorted(data["item_types"], key=lambda it: it["id"])
    widths = [int(it.get("width", it.get("size"))) for it in items]
    demands = [int(it["demand"]) for it in items]
    return roll_width, widths, demands


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
        roll_width, widths, demands = _load_csp_data(data_dict)
    except Exception as exc:
        return _result(False, f"data load error: {exc}")

    solution = output_dict.get("solution", {})
    patterns = solution.get("patterns")
    if not isinstance(patterns, list):
        return _result(False, "solution.patterns missing or not a list")

    n_items = len(widths)
    totals = [0] * n_items
    total_usage = 0

    for idx, pattern in enumerate(patterns):
        if not isinstance(pattern, dict):
            return _result(False, f"pattern {idx} is not a dict")

        usage_raw = pattern.get("usage", 1)
        usage, err = _coerce_int(usage_raw, f"patterns[{idx}].usage")
        if err:
            return _result(False, err)
        if usage < 0:
            return _result(False, f"patterns[{idx}].usage < 0")
        if usage == 0:
            continue

        cuts = pattern.get("cuts", [])
        if not isinstance(cuts, list):
            return _result(False, f"patterns[{idx}].cuts not a list")

        width_sum = 0
        for cut in cuts:
            if not isinstance(cut, dict):
                return _result(False, f"patterns[{idx}].cuts entry not a dict")
            item_id = cut.get("item_id")
            if item_id is None:
                return _result(False, f"patterns[{idx}].cuts missing item_id")
            try:
                item_id = int(item_id)
            except (TypeError, ValueError):
                return _result(False, f"patterns[{idx}].cuts item_id not int")
            if item_id < 0 or item_id >= n_items:
                return _result(False, f"item_id out of range: {item_id}")

            count_raw = cut.get("count", 1)
            count, err = _coerce_int(count_raw, f"patterns[{idx}].cuts[{item_id}].count")
            if err:
                return _result(False, err)
            if count < 0:
                return _result(False, f"negative cut count for item {item_id}")
            if count == 0:
                continue

            width_sum += widths[item_id] * count
            totals[item_id] += usage * count

        if width_sum > roll_width + 1e-6:
            return _result(False, f"pattern {idx} exceeds roll width")

        total_usage += usage

    for i, demand in enumerate(demands):
        if totals[i] < demand:
            return _result(False, f"demand not met for item {i}")

    return _result(True, "ok", float(total_usage))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CSP solution feasibility.")
    parser.add_argument("--dir", type=Path, default=Path("."), help="Directory with data.json and output.json")
    parser.add_argument("--output", type=str, default="output.json", help="Output file name")
    args = parser.parse_args()

    data = json.loads((args.dir / "data.json").read_text())
    output = json.loads((args.dir / args.output).read_text())
    result = verify_solution(data, output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
