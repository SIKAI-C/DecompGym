"""Check decision domains and recompute objectives independently of solver claims."""
from __future__ import annotations

import math
from typing import Any


def numbers(value: Any):
    if isinstance(value, list):
        for item in value:
            yield from numbers(item)
    else:
        yield float(value)


def check_bd_solution(problem: str, data: dict, output: dict, tol: float) -> dict[str, Any]:
    issues = []
    y = output["y"]
    y_values = list(numbers(y))
    if any(not math.isfinite(v) or min(abs(v), abs(v - 1)) > tol for v in y_values):
        issues.append("y must contain finite binary decisions")
    nonnegative = {"ufl": ["x"], "km": ["x"], "fctp": ["x"], "lrp": ["lambda"], "tsso": ["x_reg", "x_ot", "inv", "back"]}[problem]
    for name in nonnegative:
        values = list(numbers(output[name]))
        if any(not math.isfinite(v) or v < -tol for v in values):
            issues.append(f"{name} must contain finite nonnegative values")
        if name == "lambda" and any(abs(v - round(v)) > tol or v > 1 + tol for v in values):
            issues.append("lambda must contain binary route selections")
    if issues:
        return {"ok": False, "issues": issues}
    if problem in {"ufl", "km"}:
        objective = sum(float(data["D"][i][j]) * float(output["x"][i][j]) for i in range(data["m"]) for j in range(data["n"]))
        if problem == "ufl":
            objective += sum(float(f) * float(yi) for f, yi in zip(data["f"], y, strict=True))
        if sum(y_values) < 1 - tol:
            issues.append("at least one facility must be open")
    elif problem == "fctp":
        objective = sum(float(data["fixed_cost"][i][j]) * float(y[i][j]) + float(data["unit_cost"][i][j]) * float(output["x"][i][j]) for i in range(data["m"]) for j in range(data["n"]))
    elif problem == "lrp":
        objective = sum(float(c) * float(v) for c, v in zip(data["depot_cost"], y, strict=True)) + sum(float(c) * float(v) for c, v in zip(data["route_cost"], output["lambda"], strict=True))
        if sum(y_values) < 1 - tol:
            issues.append("at least one depot must be open")
    else:
        objective = sum(float(data["F"][i][t]) * float(y[i][t]) for i in range(data["n_products"]) for t in range(data["n_periods"]))
        for name, cost in [("x_reg", "c_reg"), ("x_ot", "c_ot"), ("inv", "h"), ("back", "b")]:
            objective += sum(float(data["prob"][s]) * float(data[cost][i][t]) * float(output[name][i][t][s]) for i in range(data["n_products"]) for t in range(data["n_periods"]) for s in range(data["n_scenarios"]))
    return {"ok": not issues, "issues": issues, "objective_value": objective}
