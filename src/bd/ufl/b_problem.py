"""
Prompt template and renderer for UFL (Benders decomposition) instances.
"""

from __future__ import annotations

from typing import Any

from src.agent.prompts.method import method_requirements


BASE_PROMPT_UFL = """You are an expert operations research engineer and Python programmer.

You are given an Uncapacitated Facility Location (UFL) problem. Opening a facility incurs a fixed
cost, and serving a customer from a facility incurs a distance-based shipping cost. The goal is to
decide which facilities to open and how to assign each customer so total open-plus-shipping cost is minimized.

- There are {m} (m) facilities and {n} (n) customers.
- Customer locations (Xc), one per line (customer j: x, y):
{Xc}
- Facility locations (Xf), one per line (facility i: x, y):
{Xf}
- Fixed opening costs (f), one per line:
{f}
- Shipping cost matrix (D): each line is for one facility i, listing cost to each customer j (i -> j):
{D}

All input data are stored in a JSON file named data.json.

Task:
Assume gurobipy and NumPy are available; the Python standard library is also allowed. Do not depend on any files other than data.json. Write a Python program solver.py that:
1) Reads the input data.json file.
2) Implements Benders decomposition according to the method requirements below.
3) Writes a JSON file output.json with at least:
   {{
     "status": "optimal | feasible | infeasible | error",
     "method": "benders_decomposition",
     "objective_value": <float>,
     "y": [0/1 list of length m indicating open facilities],
     "x": [[0/1 matrix of shape (m, n) indicating assignments]]
   }}

{method_requirements}
"""


def _fmt_customers(Xc: list[list[Any]]) -> str:
    lines = []
    for j, pt in enumerate(Xc):
        lines.append(f"  customer {j}: ({pt[0]}, {pt[1]})")
    return _truncate_block(lines)


def _fmt_facilities(Xf: list[list[Any]]) -> str:
    lines = []
    for i, pt in enumerate(Xf):
        lines.append(f"  facility {i}: ({pt[0]}, {pt[1]})")
    return _truncate_block(lines)


def _fmt_fixed_costs(f: list[Any]) -> str:
    lines = []
    for i, val in enumerate(f):
        lines.append(f"  facility {i}: fixed cost = {val}")
    return _truncate_block(lines)


def _fmt_distance_matrix(D: list[list[Any]]) -> str:
    lines = []
    for i, row in enumerate(D):
        entries = ", ".join(f"i={i}->cust {j}: {row[j]}" for j in range(len(row)))
        lines.append(f"  {entries}")
    return _truncate_block(lines)


def _truncate_block(lines: list[str]) -> str:
    if len(lines) <= 5:
        return "\n".join(lines)
    short_lines = lines[:5]
    short_lines.append("  ... you can find all the data in the data.json")
    return "\n".join(short_lines)


def render_prompt(data: dict[str, Any]) -> str:
    m = data.get("m", 0)
    n = data.get("n", 0)
    Xc = data.get("Xc", [])
    Xf = data.get("Xf", [])
    f = data.get("f", [])
    D = data.get("D", [])

    Xc_block = _fmt_customers(Xc)
    Xf_block = _fmt_facilities(Xf)
    f_block = _fmt_fixed_costs(f)
    D_block = _fmt_distance_matrix(D)

    return BASE_PROMPT_UFL.format(
        m=m,
        n=n,
        Xc=Xc_block,
        Xf=Xf_block,
        f=f_block,
        D=D_block,
        method_requirements=method_requirements("bd"),
    )
