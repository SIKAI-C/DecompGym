"""
Prompt renderer for capacitated location-routing instances.
"""

from __future__ import annotations

from typing import Any

from src.agent.prompts.method import method_requirements


BASE_PROMPT_CLRP = """You are an expert operations research engineer and Python programmer.

You are given a capacitated location-routing problem (CLRP):
  - There are m candidate depots, each with coordinates, a capacity W_d and an opening cost O_d, and n customers, each with coordinates and a demand.
  - Identical vehicles of capacity Q, unlimited in number; every route used costs a fixed vehicle cost F.
  - Each route starts and ends at the same open depot. Every customer is visited exactly once; the total demand on a route must not exceed Q, and the total demand served from depot d must not exceed W_d.
  - Travel cost between two points equals ceil(100 * euclidean_distance).
  - Objective: minimize the opening costs of the depots used + F per route + total travel cost.

This instance has {m} candidate depots, {n} customers, vehicle capacity {capacity} and vehicle cost {vehicle_cost}.

Input file data.json contains:
  - "n_depots", "n_customers", "vehicle_capacity", "vehicle_cost": int
  - "depot_x", "depot_y", "depot_capacity", "opening_cost": lists of length n_depots (depots 0..n_depots-1)
  - "customer_x", "customer_y", "demand": lists of length n_customers (customers 0..n_customers-1)

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
  "objective_value": <total cost>,
  "solution": {{
    "routes": [{{"depot": <depot>, "customers": [<customer>, ...]}}, ...]
  }}
}}
Each route gives its depot and lists its customers in visiting order, without the depot. Every customer appears in exactly one route; a depot is open exactly when at least one route uses it.

{method_requirements}
"""


def render_prompt(data: dict[str, Any]) -> str:
    lines = [f"  - depot {d}: x={data['depot_x'][d]}, y={data['depot_y'][d]}, capacity={data['depot_capacity'][d]}, opening_cost={data['opening_cost'][d]}" for d in range(min(2, data["n_depots"]))]
    lines += [f"  - customer {i}: x={data['customer_x'][i]}, y={data['customer_y'][i]}, demand={data['demand'][i]}" for i in range(min(3, data["n_customers"]))]
    return BASE_PROMPT_CLRP.format(m=data["n_depots"], n=data["n_customers"], capacity=data["vehicle_capacity"], vehicle_cost=data["vehicle_cost"],
                                   examples="\n".join(lines) + "\n  - ...", method_requirements=method_requirements("cg"))
