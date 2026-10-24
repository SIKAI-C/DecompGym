"""
Prompt renderer for bin packing (identical bins).
"""

from __future__ import annotations

from typing import Any

from src.agent.prompts.method import method_requirements


SIZE_FLAVOR = "a logistics provider packing items into identical bins for shipment"

BASE_PROMPT_BP = """You are an expert operations research engineer and Python programmer.

You are given a 1D bin-packing problem with identical bins:
  - Each bin has capacity C = {bin_capacity} {bin_unit}.
  - Items cannot be split. Each item must be placed into exactly one bin.
  - Objective: minimize the number of bins used.

Context: {scenario}.

The input file data.json provides:
  - "bin_capacity": integer
  - "bin_unit": unit label
  - "items": list of objects {{ "id": int, "name": str, "size": int }}

Examples from this instance:
{item_lines}

Task:
Assume gurobipy and NumPy are available; the Python standard library is also allowed. Do not depend on any files other than data.json. Write a Python program solver.py that:
1) loads data.json,
2) Implements a Column Generation to solve the bin-packing problem.
3) writes output.json with at least:
{{
  "status": "optimal | feasible | infeasible | error",
  "method": "column_generation",
  "objective_value": <#bins used as float>,
  "solution": {{
    "bins": [
      {{
        "usage": <int>,              # number of identical bins of this pattern
        "contents": [
          {{
            "item_id": <int>,
            "count": <int>
          }},
          ...
        ]
      }},
      ...
    ]
  }}
}}
  Each item id from data.json must appear exactly once across the bins (count=1 for single items).

{method_requirements}
"""


def _fmt_items(items: list[dict[str, Any]], limit: int = 4) -> str:
    lines = []
    for it in items[:limit]:
        lines.append(f"  - {it.get('name', 'item')} (id={it.get('id')}, size={it.get('size')})")
    if len(items) > limit:
        lines.append("  - ...")
    return "\n".join(lines)


def _scenario(size: str) -> str:
    _ = size
    return SIZE_FLAVOR


def render_prompt(data: dict[str, Any]) -> str:
    items = data.get("items", [])
    bin_capacity = data.get("bin_capacity", "?")
    bin_unit = data.get("bin_unit", "units")
    scenario = _scenario(data.get("instance_size", "medium"))

    return BASE_PROMPT_BP.format(
        bin_capacity=bin_capacity,
        bin_unit=bin_unit,
        scenario=scenario,
        item_lines=_fmt_items(items),
        method_requirements=method_requirements("cg"),
    )
