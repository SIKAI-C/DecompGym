"""
Prompt renderer for pickup and delivery with time windows instances.
"""

from __future__ import annotations

from typing import Any

from src.agent.prompts.method import method_requirements


BASE_PROMPT_PDPTW = """You are an expert operations research engineer and Python programmer.

You are given a pickup and delivery problem with time windows (PDPTW):
  - Node 0 is the depot. Request k (k = 1..n) is picked up at node k and delivered at node n + k.
  - Each node has coordinates (x, y), a demand (+q at a pickup, -q at its delivery), a time window [ready, due] and a service time.
  - Travel time and travel distance from i to j both equal floor(10 * euclidean_distance(i, j)) / 10.
  - Identical vehicles of capacity Q, unlimited in number; each route starts and ends at the depot within the depot's window.
  - Each request is served by one vehicle, which visits its pickup before its delivery. The load must stay within [0, Q] at all times.
  - A vehicle may wait if it arrives before ready; service must start no later than due.
  - Objective: minimize total travel distance.

This instance has {n} requests ({nodes} nodes besides the depot) and vehicle capacity {capacity}.

Input file data.json contains:
  - "n_requests", "capacity": int
  - "x", "y", "demand", "ready", "due", "service": lists of length 2 * n_requests + 1 (index 0 = depot)

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
    "routes": [[<node>, ...], ...]
  }}
}}
Each route lists its nodes in visiting order, without the depot. Every node 1..2n appears in exactly one route, together with its paired node, pickup first.

{method_requirements}
"""


def render_prompt(data: dict[str, Any]) -> str:
    lines = [f"  - node {i}: x={data['x'][i]}, y={data['y'][i]}, demand={data['demand'][i]}, window=[{data['ready'][i]}, {data['due'][i]}], service={data['service'][i]}" for i in range(min(4, 2 * data["n_requests"] + 1))]
    return BASE_PROMPT_PDPTW.format(n=data["n_requests"], nodes=2 * data["n_requests"], capacity=data["capacity"], examples="\n".join(lines) + "\n  - ...", method_requirements=method_requirements("cg"))
