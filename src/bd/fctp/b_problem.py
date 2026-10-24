"""
Prompt renderer for the Fixed-Charge Transportation Problem (FCTP).
"""

from __future__ import annotations

from typing import Any

from src.agent.prompts.method import method_requirements


BASE_PROMPT_FCTP = """You are an expert operations research engineer and Python programmer.

You are given a fixed-charge transportation problem:
  First choose which factory-to-customer lanes to activate; each chosen lane costs a fixed fee and then allows up to a capacity of flow.
  After lanes are open, decide shipment quantities so every customer's demand is met without exceeding any factory's supply.
  The total cost combines fixed activation and per-unit shipping. This mirrors "decide infrastructure first, operate flow second."
  Parameters:
  - There are m = {m} supply nodes and n = {n} demand nodes.
  - Each (i, j) arc has a fixed activation cost, a unit shipping cost, and a capacity.
  - Decision variables: y[i,j] in {{0,1}} indicates whether arc (i,j) is opened; x[i,j] >= 0 is quantity shipped.

The data for this instance is in data.json with fields:
  - "supply": list length m
  - "demand": list length n
  - "unit_cost", "fixed_cost", "capacity": each an m x n matrix

Sample rows (full data is in the JSON file):
Supply vector:
{supply}
Demand vector:
{demand}
Fixed costs:
{fixed_cost}
Unit costs:
{unit_cost}

Task:
Assume gurobipy and NumPy are available; the Python standard library is also allowed. Do not depend on any files other than data.json. Write a Python program solver.py that:
1) Loads data.json.
2) Implements Benders decomposition according to the method requirements below.
3) Writes output.json with at least:
    {{
      "status": "optimal | feasible | infeasible | error",
      "method": "benders_decomposition",
      "objective_value": <float>,
      "y": [[...]],  # m x n
      "x": [[...]]   # m x n corresponding to the reported incumbent y and objective_value
    }}

{method_requirements}
"""


def _fmt_vector(name: str, vec: list[Any], limit: int = 5) -> str:
    if len(vec) <= limit:
        return f"  {name} = {vec}"
    return f"  {name} = {vec[:limit]} ... (see data.json)"


def _fmt_matrix(name: str, mat: list[list[Any]], limit: int = 4) -> str:
    lines: list[str] = []
    for i, row in enumerate(mat[:limit]):
        lines.append(f"  {name}[{i}] = {row}")
    if len(mat) > limit:
        lines.append("  ...")
    return "\n".join(lines)


def render_prompt(data: dict[str, Any]) -> str:
    m = data.get("m", 0)
    n = data.get("n", 0)
    return BASE_PROMPT_FCTP.format(
        m=m,
        n=n,
        supply=_fmt_vector("supply", data.get("supply", [])),
        demand=_fmt_vector("demand", data.get("demand", [])),
        fixed_cost=_fmt_matrix("fixed_cost", data.get("fixed_cost", [])),
        unit_cost=_fmt_matrix("unit_cost", data.get("unit_cost", [])),
        method_requirements=method_requirements("bd"),
    )
