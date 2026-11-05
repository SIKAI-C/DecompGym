"""
Prompt renderer for railway crew scheduling instances.
"""

from __future__ import annotations

from typing import Any

from src.agent.prompts.method import method_requirements


BASE_PROMPT_RCSP = """You are an expert operations research engineer and Python programmer.

You are given a railway crew scheduling problem (RCSP):
  - Tasks are train trips with a start and end time (seconds from midnight) and a start and end station. Crews start and end their duties at crew bases (stations).
  - The connection graph lists the allowed moves. A SOURCE connection (base b -> task j) starts a duty at base b with task j; a CONNECTION (task i -> task j) lets the same crew perform task j after task i; a SINK connection (task i -> base b) ends a duty after task i at base b. Only listed connections may be used.
  - A duty is a path SOURCE -> CONNECTION* -> SINK whose SOURCE and SINK use the same crew base. It is given by its base and its task sequence; its connections are SOURCE (base, first task), one CONNECTION per consecutive pair of tasks, and SINK (last task, base).
  - Duty length = the sum of the durations of its connections; it must be at most 34200 seconds (9.5 hours).
  - Meal break: some connection a of the duty must be a break connection (timeBeforeBreak(a) >= 0; -1 means no break is possible on a) with
      (sum of the durations of the connections before a) + timeBeforeBreak(a) <= 19800 seconds (5.5 hours), and
      timeAfterBreak(a) + (sum of the durations of the connections after a, the SINK included) <= 19800 seconds.
  - Duty cost = the sum of the costs of its connections (each SOURCE cost already includes a fixed charge of 28800 per duty).
  - Every task must be covered by at least one duty (covering a task more than once is allowed).
  - Objective: minimize the total cost of the duties.

This instance has {n} tasks, crew bases {bases} and {m} connections ({variant}-depot variant).

Input file data.json contains:
  - "n_tasks": int; "tasks": list of n_tasks rows [startTime, endTime, startStation, endStation, duration] (tasks 0..n_tasks-1)
  - "crew_bases": list of station indices a duty may start and end at; "stations": all station indices
  - "connections": list of rows [index, type, startBlock, endBlock, timeBeforeBreak, timeAfterBreak, duration, cost], where type is "SOURCE" (startBlock = base station, endBlock = task), "CONNECTION" (task -> task) or "SINK" (startBlock = task, endBlock = base station); there is at most one connection per (type, startBlock, endBlock)
  - "variant", "instance_size", "source": descriptive only (the instance is a time window of a synthetic benchmark day; times are seconds from midnight)

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
  "objective_value": <total cost>,
  "solution": {{
    "duties": [{{"depot": <crew base station>, "tasks": [<task>, ...]}}, ...]
  }}
}}
Each duty gives its crew base and its tasks in performing order. Every task appears in at least one duty, and every duty must satisfy the rules above.

{method_requirements}
"""


def render_prompt(data: dict[str, Any]) -> str:
    lines = [f"  - task {k}: {data['tasks'][k]}" for k in range(min(3, data["n_tasks"]))]
    for kind in ("SOURCE", "CONNECTION", "SINK"):
        rows = [c for c in data["connections"] if c[1] == kind]
        brk = [c for c in rows if c[4] >= 0]
        lines += [f"  - connection {c}" for c in (rows[:1] + brk[:1])]
    return BASE_PROMPT_RCSP.format(n=data["n_tasks"], bases=data["crew_bases"], m=len(data["connections"]), variant=data["variant"],
                                   examples="\n".join(lines) + "\n  - ...", method_requirements=method_requirements("cg"))
