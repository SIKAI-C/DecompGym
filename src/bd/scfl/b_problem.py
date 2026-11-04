"""
Prompt renderer for two-stage stochastic capacitated facility location instances.
"""

from __future__ import annotations

from typing import Any

from src.agent.prompts.method import method_requirements


BASE_PROMPT_SCFL = """You are an expert operations research engineer and Python programmer.

You are given a two-stage stochastic capacitated facility location problem (SCFL):
  - First stage: choose which facilities to open, y[i] in {{0, 1}}. Facility i has fixed opening cost f[i] and capacity u[i].
  - Then one of S demand scenarios occurs; scenario s has probability p[s] and demand d[s][j] for each customer j.
  - Second stage, in each scenario s: ship x[s][i][j] >= 0 units from open facilities to customers so that every customer's demand is met exactly (sum_i x[s][i][j] = d[s][j]) and no facility ships more than its capacity (sum_j x[s][i][j] <= u[i] * y[i]). Shipping one unit from i to j costs c[i][j]. Let Q_s(y) be the minimum shipping cost of scenario s.
  - The open capacity must cover every scenario: sum_i u[i] * y[i] >= max_s sum_j d[s][j] (exactly the condition for every scenario to be feasible).
  - Objective: minimize sum_i f[i] * y[i] + sum_s p[s] * Q_s(y).

This instance has {I} facilities, {J} customers and {S} equally likely scenarios.

Input file data.json contains:
  - "n_facilities", "n_customers", "n_scenarios": int
  - "capacity" (u, integers), "fixed_cost" (f): lists of length n_facilities
  - "unit_cost" (c): n_facilities x n_customers
  - "demand" (d, integers): n_scenarios x n_customers
  - "probability" (p): list of length n_scenarios

Examples from this instance:
{examples}

Task:
Assume gurobipy and NumPy are available; the Python standard library is also allowed. Do not depend on any files other than data.json. Write a Python program solver.py that:
1) loads data.json,
2) Implements Benders decomposition to solve the problem,
3) writes output.json with at least:
{{
  "status": "optimal | feasible | infeasible | error",
  "method": "benders_decomposition",
  "objective_value": <f.y + expected shipping cost>,
  "y": [<0 or 1 per facility>]
}}
The verifier recomputes every Q_s(y) exactly for the reported y; objective_value must equal f.y + sum_s p[s] * Q_s(y) within a relative tolerance of 1e-6.

{method_requirements}
"""


def render_prompt(data: dict[str, Any]) -> str:
    lines = [f"  - facility {i}: capacity={data['capacity'][i]}, fixed_cost={data['fixed_cost'][i]}, unit_cost[{i}][:4]={data['unit_cost'][i][:4]}" for i in range(min(3, data["n_facilities"]))]
    lines += [f"  - scenario {s}: probability={data['probability'][s]}, demand[{s}][:6]={data['demand'][s][:6]}" for s in range(min(2, data["n_scenarios"]))]
    return BASE_PROMPT_SCFL.format(I=data["n_facilities"], J=data["n_customers"], S=data["n_scenarios"], examples="\n".join(lines) + "\n  - ...",
                                   method_requirements=method_requirements("bd"))
