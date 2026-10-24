"""
Prompt renderer for the enumerated-route Location Routing Problem.
"""

from __future__ import annotations

from typing import Any

from src.agent.prompts.method import method_requirements


BASE_PROMPT_LRP = """You are an expert operations research engineer and Python programmer.

You are given a Location Routing Problem (LRP) instance with pre-enumerated vehicle routes:
  First choose which depots to open (paying fixed costs). 
  Then operate vehicle routes that start/end at an open depot and collectively visit each customer exactly once. 
  Each route has a known travel cost and each depot may limit how many routes can depart. 
  The objective is to minimize opening costs plus travel while ensuring every customer is served.
  - m = {m} depots and n = {n} customers.
  - Each route r visits a subset of customers and starts/ends at the depot route_depot[r].
  - Opening depot j incurs cost depot_cost[j] and allows up to max_vehicles[j] active routes.
  - Selecting route r incurs route_cost[r].

The instance is stored in data.json with fields:
  - "routes": list of customer sequences for each route (length R)
  - "route_depot": list length R mapping each route to its depot
  - "route_cost": list length R
  - "max_vehicles": list length m
  - "depot_cost": list length m
  - "incidence": n x R matrix (1 if customer i is on route r)

Samples:
Depot costs: {depot_cost}
Max vehicles: {max_vehicles}
Example routes:
{routes}

Task:
Assume gurobipy and NumPy are available; the Python standard library is also allowed. Do not depend on any files other than data.json. Write a Python program solver.py that:
1) Reads data.json.
2) Implements Benders decomposition according to the method requirements below.
3) Writes output.json with at least:
    {{
      "status": "optimal | feasible | infeasible | error",
      "method": "benders_decomposition",
      "objective_value": <float>,
      "y": [0/1 list of length m indicating open depots],
      "lambda": [0/1 list of length R]  # feasible integer routing plan; LP relaxations may be used during cut generation
    }}
Each y element corresponds to one depot: 1 means open and 0 means closed.

{method_requirements}
"""


def _fmt_vector(name: str, vec: list[Any], limit: int = 6) -> str:
    if len(vec) <= limit:
        return f"  {name} = {vec}"
    return f"  {name} = {vec[:limit]} ... (see data.json)"


def _fmt_routes(routes: list[list[int]], route_depot: list[int], limit: int = 4) -> str:
    lines: list[str] = []
    for r, custs in enumerate(routes[:limit]):
        depot = route_depot[r] if r < len(route_depot) else "?"
        lines.append(f"  Route {r} (depot {depot}): {custs}")
    if len(routes) > limit:
        lines.append("  ...")
    return "\n".join(lines)


def render_prompt(data: dict[str, Any]) -> str:
    return BASE_PROMPT_LRP.format(
        m=data.get("m", 0),
        n=data.get("n", 0),
        depot_cost=_fmt_vector("depot_cost", data.get("depot_cost", [])),
        max_vehicles=_fmt_vector("max_vehicles", data.get("max_vehicles", [])),
        routes=_fmt_routes(data.get("routes", []), data.get("route_depot", [])),
        method_requirements=method_requirements("bd"),
    )
