"""
Prompt template and renderer for UFL (Benders decomposition) instances.
"""

from __future__ import annotations

from typing import Any, Dict, List


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

All data are stored in a JSON file named data.json in the current directory.

Task:
Write a Python program solver.py that:
1) Reads data.json from the current directory.
2) Solves the UFL using Benders decomposition with gurobipy.
3) Organizes the code in a class with clear functions for the master problem, subproblem, and the
   generation of feasibility and optimality cuts (as applicable).
4) Writes a JSON file output.json with at least:
   {{
     "status": "optimal | feasible | infeasible | error",
     "method": "benders_decomposition",
     "objective_value": <float>,
     "y": [0/1 list of length m indicating open facilities],
     "x": [[0/1 matrix of shape (m, n) indicating assignments]]
   }}

Assume you can use gurobipy. Do not require any command-line arguments or user input. Do not access
the internet or external files other than data.json.
"""


def _fmt_customers(Xc: List[List[Any]]) -> str:
    lines = []
    for j, pt in enumerate(Xc):
        lines.append(f"  customer {j}: ({pt[0]}, {pt[1]})")
    return _truncate_block(lines)


def _fmt_facilities(Xf: List[List[Any]]) -> str:
    lines = []
    for i, pt in enumerate(Xf):
        lines.append(f"  facility {i}: ({pt[0]}, {pt[1]})")
    return _truncate_block(lines)


def _fmt_fixed_costs(f: List[Any]) -> str:
    lines = []
    for i, val in enumerate(f):
        lines.append(f"  facility {i}: fixed cost = {val}")
    return _truncate_block(lines)


def _fmt_distance_matrix(D: List[List[Any]]) -> str:
    lines = []
    for i, row in enumerate(D):
        entries = ", ".join(f"i={i}->cust {j}: {row[j]}" for j in range(len(row)))
        lines.append(f"  {entries}")
    return _truncate_block(lines)


def _truncate_block(lines: List[str]) -> str:
    if len(lines) <= 5:
        return "\n".join(lines)
    short_lines = lines[:5]
    short_lines.append("  ... you can find all the data in the data.json")
    return "\n".join(short_lines)


def render_prompt(data: Dict[str, Any]) -> str:
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
    )

