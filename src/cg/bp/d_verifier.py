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


def _load_bp_data(data: dict[str, Any]) -> dict[str, Any]:
    if "bin_capacity" in data:
        capacity = int(data["bin_capacity"])
    elif "stock_size" in data:
        capacity = int(data["stock_size"])
    elif "roll_width" in data:
        capacity = int(data["roll_width"])
    else:
        raise KeyError("missing bin capacity field")

    if "item_types" in data:
        items = sorted(data["item_types"], key=lambda it: it["id"])
        sizes = [int(it.get("size", it.get("width"))) for it in items]
        demands = [int(it.get("demand", 1)) for it in items]
        return {"mode": "types", "capacity": capacity, "sizes": sizes, "demands": demands}

    if "items" not in data:
        raise KeyError("missing items list")

    items = sorted(data["items"], key=lambda it: it["id"])
    sizes_by_id = {int(it["id"]): int(it["size"]) for it in items}
    return {
        "mode": "items",
        "capacity": capacity,
        "sizes_by_id": sizes_by_id,
    }


def _parse_bins(solution: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, str | None]:
    bins = solution.get("bins")
    if not isinstance(bins, list):
        return None, "solution.bins missing or not a list"

    parsed: list[dict[str, Any]] = []
    for idx, bin_entry in enumerate(bins):
        if isinstance(bin_entry, list):
            bin_entry = {"items": bin_entry}
        if not isinstance(bin_entry, dict):
            return None, f"bin {idx} not a dict"

        if "items" in bin_entry or "item_ids" in bin_entry:
            ids = bin_entry.get("items", bin_entry.get("item_ids"))
            if not isinstance(ids, list):
                return None, f"bin {idx} items not a list"
            if "usage" in bin_entry:
                usage_val, err = _coerce_int(bin_entry.get("usage"), f"bins[{idx}].usage")
                if err:
                    return None, err
                if usage_val != 1:
                    return None, f"bin {idx} uses usage with explicit item list"
            items: list[tuple[int, int]] = []
            for item_id in ids:
                try:
                    item_id = int(item_id)
                except (TypeError, ValueError):
                    return None, f"bin {idx} item id not int"
                items.append((item_id, 1))
            parsed.append({"usage": 1, "items": items, "mode": "items"})
            continue

        if "contents" in bin_entry:
            usage_raw = bin_entry.get("usage", 1)
            usage, err = _coerce_int(usage_raw, f"bins[{idx}].usage")
            if err:
                return None, err
            if usage < 0:
                return None, f"bins[{idx}].usage < 0"
            contents = bin_entry.get("contents")
            if not isinstance(contents, list):
                return None, f"bin {idx} contents not a list"
            items = []
            for entry in contents:
                if not isinstance(entry, dict):
                    return None, f"bin {idx} content not a dict"
                item_id = entry.get("item_id")
                if item_id is None:
                    return None, f"bin {idx} content missing item_id"
                try:
                    item_id = int(item_id)
                except (TypeError, ValueError):
                    return None, f"bin {idx} item_id not int"
                count_raw = entry.get("count", 1)
                count, err = _coerce_int(count_raw, f"bins[{idx}].contents[{item_id}].count")
                if err:
                    return None, err
                if count < 0:
                    return None, f"bin {idx} has negative count"
                if count == 0:
                    continue
                items.append((item_id, count))
            parsed.append({"usage": usage, "items": items, "mode": "contents"})
            continue

        return None, f"bin {idx} missing items or contents"

    return parsed, None


def _verify_with_types(
    bins: list[dict[str, Any]],
    capacity: int,
    sizes: list[int],
    demands: list[int],
) -> tuple[bool, str, float | None]:
    totals = [0] * len(sizes)
    total_usage = 0

    for idx, bin_entry in enumerate(bins):
        usage = bin_entry["usage"]
        if usage == 0:
            continue
        bin_size = 0
        for item_id, count in bin_entry["items"]:
            if item_id < 0 or item_id >= len(sizes):
                return False, f"item_id out of range: {item_id}", None
            if count < 0:
                return False, f"negative count for item {item_id}", None
            bin_size += sizes[item_id] * count
            totals[item_id] += count * usage
        if bin_size > capacity + 1e-6:
            return False, f"bin {idx} exceeds capacity", None
        total_usage += usage

    for i, demand in enumerate(demands):
        if totals[i] != demand:
            return False, f"demand mismatch for item type {i}", None

    return True, "ok", float(total_usage)


def _verify_with_items(
    bins: list[dict[str, Any]],
    capacity: int,
    sizes_by_id: dict[int, int],
) -> tuple[bool, str, float | None]:
    assigned = {item_id: 0 for item_id in sizes_by_id}
    total_usage = 0

    for idx, bin_entry in enumerate(bins):
        usage = bin_entry["usage"]
        if usage != 1:
            return False, "usage must be 1 when assigning specific item ids", None
        bin_size = 0
        for item_id, count in bin_entry["items"]:
            if item_id not in sizes_by_id:
                return False, f"item_id out of range: {item_id}", None
            if count != 1:
                return False, f"item {item_id} has count {count}, expected 1", None
            if assigned[item_id] >= 1:
                return False, f"item {item_id} assigned multiple times", None
            assigned[item_id] += 1
            bin_size += sizes_by_id[item_id]
        if bin_size > capacity + 1e-6:
            return False, f"bin {idx} exceeds capacity", None
        total_usage += 1

    missing = [i for i, v in assigned.items() if v != 1]
    if missing:
        return False, f"unassigned items: {sorted(missing)}", None

    return True, "ok", float(total_usage)


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
        bp_data = _load_bp_data(data_dict)
    except Exception as exc:
        return _result(False, f"data load error: {exc}")

    solution = output_dict.get("solution")
    if solution is None:
        return _result(False, "missing solution")
    bins, err = _parse_bins(solution)
    if err:
        return _result(False, err)
    if bins is None:
        return _result(False, "no bins found")

    capacity = bp_data["capacity"]
    if bp_data["mode"] == "types":
        ok, msg, obj = _verify_with_types(bins, capacity, bp_data["sizes"], bp_data["demands"])
        return _result(ok, msg, obj if ok else None)

    ok, msg, obj = _verify_with_items(bins, capacity, bp_data["sizes_by_id"])
    return _result(ok, msg, obj if ok else None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify bin packing solution feasibility.")
    parser.add_argument("--dir", type=Path, default=Path("."), help="Directory with data.json and output.json")
    parser.add_argument("--output", type=str, default="output.json", help="Output file name")
    args = parser.parse_args()

    data = json.loads((args.dir / "data.json").read_text())
    output = json.loads((args.dir / args.output).read_text())
    result = verify_solution(data, output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
