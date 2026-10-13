"""
Prompt renderer for shift scheduling (set covering) instances.
"""

from __future__ import annotations

from typing import Any


SIZE_FLAVOR = "an operation scheduling staff shifts to cover time-varying demand"

BASE_PROMPT_SHIFT = """You are an expert operations research engineer and Python programmer.

You are given a shift scheduling / set-covering problem:
  - Time horizon split into T slots, demand[t] workers needed in each slot.
  - A feasible shift is any contiguous block of slots with length between L_min and L_max.
  - Cost per shift = 1. Objective: minimize number of shifts while covering all demand.

Context: {scenario}.

Input data.json contains:
  - "time_slots": list of {{id, label}}
  - "demand": list of length T
  - "shift_min_length", "shift_max_length"

Sample demand (slot label : demand):
{demand_lines}

Task:
Assume gurobipy is available. Do not depend on any files other than data.json. Write a Python program solver.py that:
1) loads data.json,
2) Implements a Column Generation to solve the shift-cover problem,
3) writes output.json with at least:
{{
  "status": "optimal | feasible | infeasible | error",
  "method": "column_generation",
  "objective_value": <#shifts>,
  "solution": {{
    "shifts": [
      {{
        "start": <int>,
        "end": <int>,
        "usage": <int>
      }},
      ...
    ]  # end is inclusive index
  }}
}}
Each shift covers its contiguous slots fully; across all shifts (counting usage) every slot demand must be met.
"""


def _fmt_demand(time_slots: list[dict[str, Any]], demand: list[Any], limit: int = 6) -> str:
    lines = []
    for ts, d in list(zip(time_slots, demand, strict=False))[:limit]:
        lines.append(f"  - {ts.get('label', ts.get('id'))}: {d}")
    if len(time_slots) > limit:
        lines.append("  - ...")
    return "\n".join(lines)


def _scenario(size: str) -> str:
    _ = size
    return SIZE_FLAVOR


def render_prompt(data: dict[str, Any]) -> str:
    return BASE_PROMPT_SHIFT.format(
        scenario=_scenario(data.get("instance_size", "medium")),
        demand_lines=_fmt_demand(data.get("time_slots", []), data.get("demand", [])),
    )
