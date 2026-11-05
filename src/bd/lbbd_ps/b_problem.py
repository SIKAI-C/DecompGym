"""
Prompt renderer for planning and scheduling with a cumulative resource (logic-based Benders) instances.
"""

from __future__ import annotations

from typing import Any

from src.agent.prompts.method import method_requirements


BASE_PROMPT_LBBD_PS = """You are an expert operations research engineer and Python programmer.

You are given a planning and scheduling problem with a cumulative resource:
  - Jobs j = 0..n-1 must each be assigned to exactly one of the facilities i = 0..m-1 and scheduled on it.
  - On facility i, job j takes proc[i][j] time units, consumes rate[i][j] units of the facility's resource while it
    runs, and costs cost[i][j].
  - Job j starts at an integer time s_j with release[j] <= s_j and s_j + proc[i][j] <= deadline[j] on its facility i.
  - Job j occupies the half-open interval [s_j, s_j + proc[i][j]). At every time, the rates of the jobs in process on
    facility i sum to at most capacity[i] (several jobs may run at once; a job ending at t and a job starting at t
    do not overlap).
  - Objective: minimize the total assignment cost sum_j cost[i(j)][j]. Start times only need to be feasible.

This instance has {n_jobs} jobs and {n_facilities} facilities with capacities {capacity}.

Input file data.json contains:
  - "n_jobs", "n_facilities": int
  - "capacity": list of length n_facilities
  - "release", "deadline": lists of length n_jobs
  - "proc", "rate", "cost": lists of n_facilities lists of length n_jobs, indexed [facility][job]
  All values are integers.

Examples from this instance:
{job_lines}

Task:
Assume gurobipy and NumPy are available; the Python standard library is also allowed. Do not depend on any files other than data.json. Write a Python program solver.py that:
1) loads data.json,
2) Implements a Benders decomposition to solve the problem,
3) writes output.json with at least:
{{
  "status": "optimal | feasible | infeasible | error",
  "method": "benders_decomposition",
  "objective_value": <total assignment cost>,
  "solution": {{
    "schedule": [
      {{
        "job": <int>,
        "facility": <int>,
        "start": <integer start time>
      }},
      ...
    ]
  }}
}}
Every job must appear exactly once, with an integer start time that respects its release date, its deadline and the facility's resource capacity.

{method_requirements}
"""


def _fmt_jobs(data: dict[str, Any], limit: int = 4) -> str:
    lines = []
    for j in range(min(limit, data["n_jobs"])):
        per = "; ".join(f"facility {i}: proc={data['proc'][i][j]}, rate={data['rate'][i][j]}, cost={data['cost'][i][j]}" for i in range(data["n_facilities"]))
        lines.append(f"  - job {j}: release={data['release'][j]}, deadline={data['deadline'][j]}; {per}")
    if data["n_jobs"] > limit:
        lines.append("  - ...")
    return "\n".join(lines)


def render_prompt(data: dict[str, Any]) -> str:
    return BASE_PROMPT_LBBD_PS.format(
        n_jobs=data["n_jobs"],
        n_facilities=data["n_facilities"],
        capacity=data["capacity"],
        job_lines=_fmt_jobs(data),
        method_requirements=method_requirements("bd"),
    )
