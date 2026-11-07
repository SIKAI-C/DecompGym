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


# Koerkel-Ghosh instances (no coordinates; general, possibly asymmetric costs). Paper-format data keeps BASE_PROMPT_UFL.
BASE_PROMPT_UFL_KG = """You are an expert operations research engineer and Python programmer.

You are given an Uncapacitated Facility Location (UFL) problem. Opening a facility incurs a fixed
cost, and serving a customer from a facility incurs a connection cost. The goal is to decide which
facilities to open and how to assign each customer so total open-plus-connection cost is minimized.

- There are {m} (m) candidate facilities and {n} (n) customers. Facilities have no capacity limit.
- Connection costs are general integers (not derived from coordinates; they need not satisfy the triangle
  inequality{symmetry_note}).
- Fixed opening costs (f), one per line:
{f}
- Connection cost matrix (D): each line is for one facility i, listing the cost to serve each customer j (i -> j):
{D}

All input data are stored in a JSON file named data.json with fields "m", "n", "f" (length m) and "D" (m x n, row =
facility, column = customer).

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
    if "Xc" not in data:
        return _render_prompt_kg(data)
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


def _is_symmetric(D: list[list[Any]]) -> bool:
    """D is square and D[i][j] == D[j][i] for every pair (a property of the data itself)."""
    return all(len(row) == len(D) for row in D) and all(D[i][j] == D[j][i] for i in range(len(D)) for j in range(i))


def _render_prompt_kg(data: dict[str, Any]) -> str:
    note = "; in this instance D is symmetric, D[i][j] = D[j][i]" if _is_symmetric(data.get("D", [])) else "; in this instance D is not symmetric"
    return BASE_PROMPT_UFL_KG.format(
        m=data.get("m", 0),
        n=data.get("n", 0),
        symmetry_note=note,
        f=_fmt_fixed_costs(data.get("f", [])),
        D=_fmt_distance_matrix(data.get("D", [])),
        method_requirements=method_requirements("bd"),
    )
