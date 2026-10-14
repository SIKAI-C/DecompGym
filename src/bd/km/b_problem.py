"""
Prompt template for the k-Median problem (Benders).
"""

from __future__ import annotations

from typing import Any

from src.agent.decomposition_policy import method_requirements


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


def _fmt_matrix(name: str, mat: list[list[Any]], max_rows: int = 4) -> str:
    lines: list[str] = []
    for i, row in enumerate(mat[:max_rows]):
        lines.append(f"  {name}[{i}] = {row}")
    if len(mat) > max_rows:
        lines.append("  ...")
    return "\n".join(lines)


def render_prompt(data: dict[str, Any]) -> str:
    return BASE_PROMPT_KM.format(
        m=data.get("m", 0),
        n=data.get("n", 0),
        k=data.get("k", 0),
        Xc=_fmt_matrix("Xc", data.get("Xc", [])),
        Xf=_fmt_matrix("Xf", data.get("Xf", [])),
        D=_fmt_matrix("D", data.get("D", [])),
        method_requirements=method_requirements("bd"),
    )
