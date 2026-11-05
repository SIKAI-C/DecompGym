"""Independent checker for graph coloring outputs: color classes must partition the vertices into independent sets."""
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
        n = int(data["n_vertices"])
        edges = {(min(u, v), max(u, v)) for u, v in data["edges"]}
    except Exception as exc:
        return _result(False, f"data load error: {exc}")
    solution = output.get("solution")
    if not isinstance(solution, dict) or not isinstance(solution.get("color_classes"), list):
        return _result(False, "solution.color_classes missing or not a list")
    color = {}
    for k, cls in enumerate(solution["color_classes"]):
        if not isinstance(cls, list) or not cls or not all(integer_value(v) and 0 <= int(v) < n for v in cls):
            return _result(False, f"color_classes[{k}] must be a non-empty list of vertex indices")
        for v in cls:
            if int(v) in color:
                return _result(False, f"vertex {v} colored twice")
            color[int(v)] = k
    if len(color) != n:
        return _result(False, f"{n - len(color)} vertex(es) uncolored")
    bad = next(((u, v) for u, v in edges if color[u] == color[v]), None)
    if bad:
        return _result(False, f"adjacent vertices {bad} share a color")
    return _result(True, "ok", float(len(solution["color_classes"])))
