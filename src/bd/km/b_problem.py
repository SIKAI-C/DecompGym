"""
Prompt template for the k-Median problem (Benders).
"""

from __future__ import annotations

import math
from typing import Any

from src.agent.prompts.method import method_requirements


BASE_PROMPT_KM = """You are an expert operations research engineer and Python programmer.

You are given a k-median facility location problem:
  - Pick exactly k = {k} service sites to open from m = {m} candidates serving n = {n} customers.
  - Once sites are chosen, each customer is assigned—potentially fractionally—to the open sites, paying an assignment cost that reflects distance or service quality (D[i][j]).
  - The core trade-off is which k sites to open so the total assignment cost to serve everyone is minimal.

The data for this instance lives in data.json with fields:
  - "Xc": n x 2 customer coordinates
  - "Xf": m x 2 facility coordinates
  - "D": m x n assignment costs
  - "m", "n", "k"

Samples:
Customer coords (first few):
{Xc}
Facility coords:
{Xf}
Cost rows:
{D}

Task:
Assume gurobipy and NumPy are available; the Python standard library is also allowed. Do not depend on any files other than data.json. Write a Python program solver.py that:
1) Loads data.json.
2) Implements Benders decomposition according to the method requirements below.
3) Writes output.json with at least:
    {{
      "status": "optimal | feasible | infeasible | error",
      "method": "benders_decomposition",
      "objective_value": <float>,
      "y": [0/1 list length m],
      "x": [[assignment matrix m x n]]
    }}

{method_requirements}
"""


# TSPLIB-derived p-median instances (sites = customers). Paper-format data keeps BASE_PROMPT_KM.
BASE_PROMPT_KM_TSP = """You are an expert operations research engineer and Python programmer.

You are given a k-median (p-median) facility location problem:
  - There are n = {n} nodes (points in the plane). Every node is a customer and also a candidate site (m = n = {m}).
  - Pick exactly k = {k} sites to open. Each customer is then assigned (possibly fractionally) to open sites, paying
    the assignment cost D[i][j] between site i and customer j.
  - D[i][j] = floor(sqrt((x_i - x_j)^2 + (y_i - y_j)^2)), the Euclidean distance rounded down to an integer, so
    D is symmetric with a zero diagonal (a node that is an open site serves itself at cost 0).
  - The goal is to choose the k open sites so that the total assignment cost of all customers is minimal.

The data for this instance lives in data.json with fields:
  - "Xc": n x 2 node coordinates (customers)
  - "Xf": m x 2 node coordinates (sites; identical to Xc)
  - "D": m x n assignment costs (row = site i, column = customer j)
  - "m", "n", "k"

Samples:
Node coords (first few):
{Xc}
Cost rows:
{D}

Task:
Assume gurobipy and NumPy are available; the Python standard library is also allowed. Do not depend on any files other than data.json. Write a Python program solver.py that:
1) Loads data.json.
2) Implements Benders decomposition according to the method requirements below.
3) Writes output.json with at least:
    {{
      "status": "optimal | feasible | infeasible | error",
      "method": "benders_decomposition",
      "objective_value": <float>,
      "y": [0/1 list length m],
      "x": [[assignment matrix m x n]]
    }}

{method_requirements}
"""


def _fmt_matrix(name: str, mat: list[list[Any]], max_rows: int = 4) -> str:
    lines: list[str] = []
    for i, row in enumerate(mat[:max_rows]):
        lines.append(f"  {name}[{i}] = {row}")
    if len(mat) > max_rows:
        lines.append("  ...")
    return "\n".join(lines)


def _sites_are_customers(data: dict[str, Any]) -> bool:
    """TSPLIB-derived p-median data, i.e. the facts BASE_PROMPT_KM_TSP states: every node is both a customer and a site
    (m = n, Xf = Xc) and D[i][j] = floor(sqrt(dx * dx + dy * dy)) (IEEE double, as in a_gen_data.floor_euclidean). The
    paper-format generator draws facilities and customers separately, so its data never satisfy this."""
    X, D = data.get("Xc") or [], data.get("D") or []
    if not X or data.get("m") != data.get("n") or X != data.get("Xf") or len(D) != len(X) or any(len(row) != len(X) for row in D):
        return False
    return all(D[i][j] == math.floor(math.sqrt((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1])))
               for i, p in enumerate(X) for j, q in enumerate(X))


def render_prompt(data: dict[str, Any]) -> str:
    if _sites_are_customers(data):
        return BASE_PROMPT_KM_TSP.format(
            m=data.get("m", 0),
            n=data.get("n", 0),
            k=data.get("k", 0),
            Xc=_fmt_matrix("Xc", data.get("Xc", [])),
            D=_fmt_matrix("D", data.get("D", [])),
            method_requirements=method_requirements("bd"),
        )
    return BASE_PROMPT_KM.format(
        m=data.get("m", 0),
        n=data.get("n", 0),
        k=data.get("k", 0),
        Xc=_fmt_matrix("Xc", data.get("Xc", [])),
        Xf=_fmt_matrix("Xf", data.get("Xf", [])),
        D=_fmt_matrix("D", data.get("D", [])),
        method_requirements=method_requirements("bd"),
    )
