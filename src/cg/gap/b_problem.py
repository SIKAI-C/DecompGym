"""
Prompt renderer for the Generalized Assignment Problem.
"""

from __future__ import annotations

from typing import Any


SIZE_FLAVOR = "an operation assigning jobs to limited resources under capacity"

BASE_PROMPT_GAP = """You are an expert operations research engineer and Python programmer.

You are given a generalized assignment problem:
  - Jobs i must each be assigned to exactly one machine j.
  - Assigning job i to machine j consumes resource a[i,j] and costs c[i,j].
  - Each machine j has capacity b[j].
  - Objective: minimize total assignment cost.

Context: {scenario}.

Input data.json contains:
  - "machines": list of {{id, name, capacity}}
  - "jobs": list of {{id, name, assignments: [{{machine_id, resource, cost}}, ...]}}

Sample assignments:
{sample_lines}

Task:
Assume gurobipy is available. Do not depend on any files other than data.json. Write a Python program solver.py that:
1) loads data.json,
2) Implements a Column Generation to solve the GAP,
3) writes output.json with at least:
{{
  "status": "optimal | feasible | infeasible | error",
  "method": "column_generation",
  "objective_value": <float>,
  "solution": {{
    "assignments": [
      {{
        "job_id": i,
        "machine_id": j
      }},
      ...
    ]
  }}
}}
All jobs must be assigned exactly once and no machine capacity may be exceeded.
"""


def _fmt_assignments(jobs: list[dict[str, Any]], limit_jobs: int = 3, limit_machines: int = 3) -> str:
    lines = []
    for job in jobs[:limit_jobs]:
        job_id = job.get("id")
        assigns = job.get("assignments", [])[:limit_machines]
        desc = ", ".join(
            f"to {a.get('machine_id')} uses {a.get('resource')} cost {a.get('cost')}"  # type: ignore[arg-type]
            for a in assigns
        )
        lines.append(f"  - job {job_id}: {desc}")
    if len(jobs) > limit_jobs:
        lines.append("  - ...")
    return "\n".join(lines)


def _scenario(size: str) -> str:
    _ = size
    return SIZE_FLAVOR


def render_prompt(data: dict[str, Any]) -> str:
    return BASE_PROMPT_GAP.format(
        scenario=_scenario(data.get("instance_size", "medium")),
        sample_lines=_fmt_assignments(data.get("jobs", [])),
    )
