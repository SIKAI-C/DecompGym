"""
Prompt renderer for nurse rostering instances.
"""

from __future__ import annotations

from typing import Any

from src.agent.prompts.method import method_requirements


BASE_PROMPT_NRP = """You are an expert operations research engineer and Python programmer.

You are given a nurse rostering problem:
  - Nurses n = 0..N-1 over days d = 0..D-1. Each day a nurse works at most one shift s: 0 = Early, 1 = Late, 2 = Night, or is off.
  - Coverage: demand[d][s] nurses are wanted on shift s of day d. Each missing nurse costs shortage_penalty.
  - Assigning nurse n to shift s on day d costs cost[n][d][s].
  - Hard rules for each nurse n: the number of working days is within [min_days[n], max_days[n]]; at most max_nights[n] night shifts;
    at most max_consecutive consecutive working days; and no forbidden succession [a, b] (shift a on day d followed by shift b on day d+1).
  - Objective: minimize total assignment cost + shortage penalties.

This instance has {N} nurses and {D} days.

Input file data.json contains:
  - "n_nurses", "n_days", "max_consecutive", "shortage_penalty": int
  - "demand": D lists of 3 integers; "cost": N lists of D lists of 3 integers
  - "min_days", "max_days", "max_nights": lists of length N
  - "forbidden_successions": list of [a, b] pairs

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
  "objective_value": <total cost including shortage penalties>,
  "solution": {{
    "rosters": [[<shift or -1 for off>, ...], ...]
  }}
}}
Give one roster of length D per nurse, in nurse order; each roster must satisfy all of that nurse's hard rules.

{method_requirements}
"""


def render_prompt(data: dict[str, Any]) -> str:
    lines = [f"  - nurse {n}: working days in [{data['min_days'][n]}, {data['max_days'][n]}], at most {data['max_nights'][n]} nights" for n in range(min(3, data["n_nurses"]))]
    lines.append(f"  - demand on day 0 (Early, Late, Night): {data['demand'][0]}")
    return BASE_PROMPT_NRP.format(N=data["n_nurses"], D=data["n_days"], examples="\n".join(lines) + "\n  - ...", method_requirements=method_requirements("cg"))
