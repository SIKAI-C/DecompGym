"""
Prompt renderer for graph coloring instances.
"""

from __future__ import annotations

from typing import Any

from src.agent.prompts.method import method_requirements


BASE_PROMPT_GCOL = """You are an expert operations research engineer and Python programmer.

You are given a graph coloring problem:
  - Vertices v = 0..n-1 and undirected edges (u, v).
  - Assign each vertex a color so that the two endpoints of every edge have different colors.
  - Objective: minimize the number of colors used.

This instance has {n_vertices} vertices and {n_edges} edges.

Input file data.json contains:
  - "n_vertices": int
  - "edges": list of [u, v] pairs

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
  "objective_value": <number of colors>,
  "solution": {{
    "color_classes": [[<vertex>, ...], ...]
  }}
}}
The color classes must partition the vertices, and no class may contain both endpoints of an edge.

{method_requirements}
"""


def render_prompt(data: dict[str, Any]) -> str:
    examples = "\n".join(f"  - edge {e[0]} -- {e[1]}" for e in data["edges"][:4]) + ("\n  - ..." if len(data["edges"]) > 4 else "")
    return BASE_PROMPT_GCOL.format(n_vertices=data["n_vertices"], n_edges=len(data["edges"]), examples=examples, method_requirements=method_requirements("cg"))
