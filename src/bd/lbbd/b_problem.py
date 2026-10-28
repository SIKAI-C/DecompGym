"""
Prompt renderer for task assignment + single-machine scheduling (logic-based Benders) instances.
"""

from __future__ import annotations

from typing import Any

from src.agent.prompts.method import method_requirements


BASE_PROMPT_LBBD = """You are an expert operations research engineer and Python programmer.

You are given a task assignment and scheduling problem:
  - Tasks j = 0..n-1 must each be assigned to exactly one of the machines i = 0..m-1 and scheduled on it.
  - Each machine processes at most one task at a time, without preemption.
  - Task j cannot start before its release time release[j]. On machine i it takes proc_time[i][j] time units
    and costs proc_cost[i][j].
  - Task j is due at due[j]; if it completes at time C_j, its tardiness is T_j = max(0, C_j - due[j]),
    penalized with weight weight[j]. Due dates are soft: a task may finish late.
  - Objective: minimize sum of processing costs of the chosen assignments + sum_j weight[j] * T_j.

This instance has {n_tasks} tasks and {n_machines} machines.

Input file data.json contains:
  - "n_tasks", "n_machines": int
  - "release", "due", "weight": lists of length n_tasks (integers)
  - "proc_time", "proc_cost": lists of n_machines lists of length n_tasks (integers), indexed [machine][task]

Examples from this instance:
{task_lines}

Task:
Assume gurobipy and NumPy are available; the Python standard library is also allowed. Do not depend on any files other than data.json. Write a Python program solver.py that:
1) loads data.json,
2) Implements a Benders decomposition to solve the problem,
3) writes output.json with at least:
{{
  "status": "optimal | feasible | infeasible | error",
  "method": "benders",
  "objective_value": <total processing cost + total weighted tardiness>,
  "solution": {{
    "assignments": [
      {{
        "task_id": <int>,
        "machine_id": <int>,
        "start": <start time, a number>
      }},
      ...
    ]
  }}
}}
Every task must appear exactly once. Start times must respect release times, and tasks on the same machine must not overlap.

{method_requirements}
"""


def _fmt_tasks(data: dict[str, Any], limit: int = 4) -> str:
    lines = []
    for j in range(min(limit, data["n_tasks"])):
        times = ", ".join(str(data["proc_time"][i][j]) for i in range(data["n_machines"]))
        costs = ", ".join(str(data["proc_cost"][i][j]) for i in range(data["n_machines"]))
        lines.append(f"  - task {j}: release={data['release'][j]}, due={data['due'][j]}, weight={data['weight'][j]}, proc_time by machine=[{times}], proc_cost by machine=[{costs}]")
    if data["n_tasks"] > limit:
        lines.append("  - ...")
    return "\n".join(lines)


def render_prompt(data: dict[str, Any]) -> str:
    return BASE_PROMPT_LBBD.format(
        n_tasks=data["n_tasks"],
        n_machines=data["n_machines"],
        task_lines=_fmt_tasks(data),
        method_requirements=method_requirements("bd"),
    )
