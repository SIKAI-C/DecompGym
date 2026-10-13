"""Validate the JSON structure required by the frozen problem prompts."""
from __future__ import annotations

import math


OUTPUT_CONTRACT_POLICY = "prompt-output-contract-v1"


def finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def integer_value(value):
    return finite_number(value) and (type(value) is int or value.is_integer())


def validate_output_contract(suite, problem, data, output):
    """Return candidate errors; extra keys and JSON formatting are immaterial."""
    issues = []
    if not isinstance(output, dict):
        return ["output.json must be an object"]

    def require(mapping, name, predicate, description, prefix=""):
        label = prefix + name
        if name not in mapping:
            issues.append(f"{label} is required")
            return False
        if not predicate(mapping[name]):
            issues.append(f"{label} must be {description}")
            return False
        return True

    require(output, "status", lambda v: isinstance(v, str) and v in {"optimal", "feasible", "infeasible", "error"}, "a documented status string")
    methods = {"column_generation"} if suite == "cg" else {"benders_decomposition"}
    if problem == "fctp":
        methods.add("monolithic")  # Explicitly permitted by this prompt; the algorithm judge remains separate.
    require(output, "method", lambda v: isinstance(v, str) and v in methods, "a method permitted by the problem prompt")
    require(output, "objective_value", finite_number, "a finite JSON number")

    def array(value, shape, label):
        if not shape:
            if not finite_number(value):
                issues.append(f"{label} must be a finite JSON number")
            return
        if not isinstance(value, list) or len(value) != shape[0]:
            issues.append(f"{label} must be an array of length {shape[0]}")
            return
        for idx, entry in enumerate(value):
            array(entry, shape[1:], f"{label}[{idx}]")

    if suite == "bd":
        if problem in {"ufl", "km", "fctp"}:
            m, n = data.get("m", 0), data.get("n", 0)
            shapes = {"y": (m, n) if problem == "fctp" else (m,), "x": (m, n)}
        elif problem == "lrp":
            shapes = {"y": (data.get("m", 0),), "lambda": (len(data.get("route_depot", [])),)}
        elif problem == "tsso":
            m, n, s = data.get("n_products", 0), data.get("n_periods", 0), data.get("n_scenarios", 0)
            shapes = {"y": (m, n), **{name: (m, n, s) for name in ("x_reg", "x_ot", "inv", "back")}}
        else:
            raise ValueError(f"Unknown BD problem: {problem}")
        for name, shape in shapes.items():
            if name not in output:
                issues.append(f"{name} is required")
            else:
                array(output[name], shape, name)
        return issues

    if suite != "cg":
        raise ValueError(f"Unknown suite: {suite}")
    if not require(output, "solution", lambda v: isinstance(v, dict), "an object"):
        return issues
    collection, fields, nested = {
        "bp": ("bins", ("usage",), ("contents", ("item_id", "count"))),
        "csp": ("patterns", ("pattern_id", "usage"), ("cuts", ("item_id", "count"))),
        "gap": ("assignments", ("job_id", "machine_id"), None),
        "shift": ("shifts", ("start", "end", "usage"), None),
    }[problem]
    solution = output["solution"]
    if not require(solution, collection, lambda v: isinstance(v, list), "an array", "solution."):
        return issues
    for idx, entry in enumerate(solution[collection]):
        label = f"solution.{collection}[{idx}]"
        if not isinstance(entry, dict):
            issues.append(f"{label} must be an object")
            continue
        for name in fields:
            require(entry, name, integer_value, "a finite integer JSON number", label + ".")
        if nested:
            nested_name, nested_fields = nested
            if not require(entry, nested_name, lambda v: isinstance(v, list), "an array", label + "."):
                continue
            for j, member in enumerate(entry[nested_name]):
                nested_label = f"{label}.{nested_name}[{j}]"
                if not isinstance(member, dict):
                    issues.append(f"{nested_label} must be an object")
                    continue
                for name in nested_fields:
                    require(member, name, integer_value, "a finite integer JSON number", nested_label + ".")
    return issues
