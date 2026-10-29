"""
Prompt renderer for vehicle routing with time windows instances.
"""

from __future__ import annotations

from typing import Any

from src.agent.prompts.method import method_requirements


BASE_PROMPT_VRPTW = """You are an expert operations research engineer and Python programmer.

You are given a capacitated vehicle routing problem with time windows (VRPTW):
  - Node 0 is the depot; nodes 1..n are customers with coordinates (x, y), demand, time window [ready, due] and service time.
  - Travel time and travel distance from i to j both equal floor(10 * euclidean_distance(i, j)) / 10.
  - Identical vehicles of capacity Q, unlimited in number. Each route starts and ends at the depot within the depot's window.
  - Every customer is visited exactly once. A vehicle may wait if it arrives before ready; service must start no later than due.
  - Objective: minimize total travel distance.

This instance has {n} customers and vehicle capacity {capacity}.

Input file data.json contains:
  - "n_customers", "capacity": int
  - "x", "y", "demand", "ready", "due", "service": lists of length n_customers + 1 (index 0 = depot)

Examples from this instance:
{examples}

Task:
Assume gurobipy and NumPy are available; the Python standard library is also allowed. Do not depend on any files other than data.json. Write a Python program solver.py that:
1) loads data.json,
2) Implements a Column Generation to solve the problem,
3) writes output.json with at least:
{{
  "status": "optimal | feasible | infeasible | error",
  "method": "column_generation",
  "objective_value": <total distance>,
  "solution": {{
    "routes": [[<customer>, ...], ...]
  }}
}}
Each route lists its customers in visiting order, without the depot. Every customer appears in exactly one route; each route must respect the capacity and all time windows.

{method_requirements}
"""


def render_prompt(data: dict[str, Any]) -> str:
    lines = [f"  - node {i}: x={data['x'][i]}, y={data['y'][i]}, demand={data['demand'][i]}, window=[{data['ready'][i]}, {data['due'][i]}], service={data['service'][i]}" for i in range(min(4, data["n_customers"] + 1))]
    return BASE_PROMPT_VRPTW.format(n=data["n_customers"], capacity=data["capacity"], examples="\n".join(lines) + "\n  - ...", method_requirements=method_requirements("cg"))
