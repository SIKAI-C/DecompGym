"""
Prompt template and renderer for the TSSO (two-stage stochastic production) task.
"""

from __future__ import annotations

from typing import Any

from src.agent.decomposition_policy import method_requirements


BASE_PROMPT_TSSO = """You are an expert operations research engineer and Python programmer.

You are given a two-stage stochastic production planning problem with capacity expansion decisions.
First, decide which production lines (regular capacity) to pay for across products and periods.
Later, when demand scenarios unfold, react by producing with regular or overtime capacity,
holding inventory, or backlogging unmet demand. The aim is to pick upfront capacities that keep the expected
total cost (activation + operations) low across scenarios.
The instance describes:
  - n_products = {n_products}
  - n_periods = {n_periods}
  - n_scenarios = {n_scenarios} with probabilities {probabilities}
  - First-stage binary variables y[i,t] decide whether capacity is installed.
  - Second-stage (recourse) variables for each scenario s: regular/OT production, inventory, backlog.

The JSON file data.json contains:
  - "F": fixed expansion cost matrix (n_products x n_periods)
  - "K": capacity multipliers
  - "c_reg", "c_ot": regular and overtime unit costs
  - "h", "b": inventory holding and backlog penalty costs
  - "demand": tensor (n_products x n_periods x n_scenarios)
  - "prob": scenario probabilities
  - "I0", "B0": initial inventory/backlog per product

Key slices (first few rows) to orient yourself:
Fixed cost F:
{F}
Capacity coefficient K:
{K}
Regular cost c_reg:
{c_reg}
Overtime cost c_ot:
{c_ot}
Holding cost h:
{h}
Backlog cost b:
{b}
Demand samples:
{demand}
Initial inventory I0:
{I0}
Initial backlog B0:
{B0}

Task:
Assume gurobipy and NumPy are available; the Python standard library is also allowed. Do not depend on any files other than data.json. Write a Python program solver.py that:
1) Loads data.json.
2) Implements Benders decomposition according to the method requirements below.
3) Writes output.json with at least:
    {{
    "status": "optimal | feasible | infeasible | error",
    "method": "benders_decomposition",
    "objective_value": <float>,
    "y": [[...]],          # shape (n_products, n_periods)
    "x_reg": [[[...]]],    # shape (n_products, n_periods, n_scenarios)
    "x_ot": [[[...]]],     # shape (n_products, n_periods, n_scenarios)
    "inv": [[[...]]],      # shape (n_products, n_periods, n_scenarios)
    "back": [[[...]]]      # shape (n_products, n_periods, n_scenarios)
    }}

{method_requirements}
"""


def _fmt_matrix(name: str, mat: list[list[Any]], max_rows: int = 4) -> str:
    if not mat:
        return f"{name}: []"
    lines = []
    rows = mat[:max_rows]
    for idx, row in enumerate(rows):
        lines.append(f"  {name}[{idx}] = {row}")
    if len(mat) > max_rows:
        lines.append("  ... full data is in data.json")
    return "\n".join(lines)


def _fmt_prob(prob: list[Any], max_len: int = 6) -> str:
    if len(prob) <= max_len:
        return ", ".join(str(p) for p in prob)
    return ", ".join(str(prob[i]) for i in range(max_len)) + ", ..."


def _fmt_demand(demand: list[list[list[Any]]], max_prods: int = 2, max_periods: int = 2) -> str:
    lines: list[str] = []
    for i, prod in enumerate(demand[:max_prods]):
        for t, values in enumerate(prod[:max_periods]):
            lines.append(f"  demand[{i},{t},:] = {values}")
    lines.append("  ...")
    return "\n".join(lines)


def _fmt_vector(name: str, vec: list[Any], max_len: int = 6) -> str:
    if not vec:
        return f"{name}: []"
    slice_ = vec[:max_len]
    suffix = " ..." if len(vec) > max_len else ""
    return f"{name}: {slice_}{suffix}"


def render_prompt(data: dict[str, Any]) -> str:
    n_products = data.get("n_products", 0)
    n_periods = data.get("n_periods", 0)
    n_scenarios = data.get("n_scenarios", 0)
    prob = data.get("prob", [])

    return BASE_PROMPT_TSSO.format(
        n_products=n_products,
        n_periods=n_periods,
        n_scenarios=n_scenarios,
        probabilities=_fmt_prob(prob),
        F=_fmt_matrix("F", data.get("F", [])),
        K=_fmt_matrix("K", data.get("K", [])),
        c_reg=_fmt_matrix("c_reg", data.get("c_reg", [])),
        c_ot=_fmt_matrix("c_ot", data.get("c_ot", [])),
        h=_fmt_matrix("h", data.get("h", [])),
        b=_fmt_matrix("b", data.get("b", [])),
        demand=_fmt_demand(data.get("demand", [])),
        I0=_fmt_vector("I0", data.get("I0", [])),
        B0=_fmt_vector("B0", data.get("B0", [])),
        method_requirements=method_requirements("bd"),
    )
