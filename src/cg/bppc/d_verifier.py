"""Independent checker for bin packing with conflicts: bins partition the items, fit the capacity, and hold no conflicting pair."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.output_contract import integer_value


def _load(obj: Any) -> dict[str, Any]:
    if isinstance(obj, (str, Path)):
        return json.loads(Path(obj).read_text())
    if isinstance(obj, dict):
        return obj
    raise TypeError("expected dict or path-like input")


def _result(ok: bool, message: str, objective_value: float | None = None) -> dict[str, Any]:
    return {"ok": ok, "issues": [] if ok else [message], "objective_value": objective_value}


def verify_solution(data: Any, output: Any, tol: float | None = None) -> dict[str, Any]:
    _ = tol
    try:
        data, output = _load(data), _load(output)
        n, cap, w = int(data["n_items"]), int(data["capacity"]), data["weights"]
        conflicts = {(min(a, b), max(a, b)) for a, b in data["conflicts"]}
    except Exception as exc:
        return _result(False, f"data load error: {exc}")
    solution = output.get("solution")
    if not isinstance(solution, dict) or not isinstance(solution.get("bins"), list):
        return _result(False, "solution.bins missing or not a list")
    seen: set[int] = set()
    for k, b in enumerate(solution["bins"]):
        if not isinstance(b, list) or not b or not all(integer_value(i) and 0 <= int(i) < n for i in b):
            return _result(False, f"bins[{k}] must be a non-empty list of item indices")
        items = [int(i) for i in b]
        if seen & set(items) or len(set(items)) != len(items):
            return _result(False, f"bins[{k}] repeats an item")
        seen |= set(items)
        if sum(w[i] for i in items) > cap:
            return _result(False, f"bins[{k}] exceeds the capacity")
        if any((min(a, c), max(a, c)) in conflicts for a in items for c in items if a < c):
            return _result(False, f"bins[{k}] holds a conflicting pair")
    if len(seen) != n:
        return _result(False, f"{n - len(seen)} item(s) not packed")
    return _result(True, "ok", float(len(solution["bins"])))
