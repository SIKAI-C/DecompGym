"""
Prompt renderer for cutting stock (1D) instances.
"""

from __future__ import annotations

from typing import Any

from src.agent.prompts.method import method_requirements


SIZE_FLAVOR = "a manufacturing operation cutting stock material into ordered widths"

BASE_PROMPT_CSP = """You are an expert operations research engineer and Python programmer.

You are given a 1D cutting stock problem:
  - Stock length W = {stock_size} {stock_unit}.
  - Item types i=0..m-1, each with width size[i] and demand demand[i].
  - Objective: minimize number of stock pieces used.

Context: {scenario}.

Input file data.json contains:
  - "stock_size" (or "roll_width"), "stock_unit"
  - "item_types": list of {{ "id": int, "name": str, "size": int, "demand": int }}

Examples from this instance:
{item_lines}

Task:
Assume gurobipy and NumPy are available; the Python standard library is also allowed. Do not depend on any files other than data.json. Write a Python program solver.py that:
1) loads data.json,
2) Implements a Column Generation to solve the cutting stock problem,
3) writes output.json with at least:
{{
  "status": "optimal | feasible | infeasible | error",
  "method": "column_generation",
  "objective_value": <#stock pieces used>,
  "solution": {{
    "patterns": [
      {{
        "pattern_id": <int>,
        "usage": <int>,
        "cuts": [
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
Each pattern's total width must not exceed stock_size. Across all patterns (accounting for usage) every item demand must be met.

{method_requirements}
"""


def _fmt_items(items: list[dict[str, Any]], limit: int = 4) -> str:
    lines = []
    for it in items[:limit]:
        lines.append(
            f"  - {it.get('name', 'item')} (id={it.get('id')}, size={it.get('size')}, demand={it.get('demand')})"
        )
    if len(items) > limit:
        lines.append("  - ...")
    return "\n".join(lines)


def _scenario(size: str) -> str:
    _ = size
    return SIZE_FLAVOR


def render_prompt(data: dict[str, Any]) -> str:
    items = data.get("item_types", [])
    stock_size = data.get("stock_size", data.get("roll_width", "?"))
    stock_unit = data.get("stock_unit", "units")
    scenario = _scenario(data.get("instance_size", "medium"))

    return BASE_PROMPT_CSP.format(
        stock_size=stock_size,
        stock_unit=stock_unit,
        scenario=scenario,
        item_lines=_fmt_items(items),
        method_requirements=method_requirements("cg"),
    )
