"""
Prompt renderer for bin packing with conflicts instances.
"""

from __future__ import annotations

from typing import Any

from src.agent.prompts.method import method_requirements


BASE_PROMPT_BPPC = """You are an expert operations research engineer and Python programmer.

You are given a bin packing problem with conflicts:
  - Items i = 0..n-1 with integer weights weights[i]; identical bins of capacity C.
  - Some pairs of items conflict and may not be packed in the same bin.
  - Objective: minimize the number of bins used.

This instance has {n_items} items, bin capacity {capacity}, and {n_conflicts} conflicting pairs.

Input file data.json contains:
  - "n_items", "capacity": int
  - "weights": list of n_items integers
  - "conflicts": list of [i, j] pairs

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
  "objective_value": <number of bins>,
  "solution": {{
    "bins": [[<item>, ...], ...]
  }}
}}
Every item must be packed exactly once; each bin's total weight must not exceed the capacity, and no bin may hold a conflicting pair.

{method_requirements}
"""


def render_prompt(data: dict[str, Any]) -> str:
    examples = "\n".join(f"  - item {i}: weight {w}" for i, w in enumerate(data["weights"][:4])) + "\n  - ..."
    examples += "\n" + "\n".join(f"  - conflict {a} -- {b}" for a, b in data["conflicts"][:2])
    return BASE_PROMPT_BPPC.format(n_items=data["n_items"], capacity=data["capacity"], n_conflicts=len(data["conflicts"]), examples=examples, method_requirements=method_requirements("cg"))
