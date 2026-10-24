Only edit the existing solver.py in the current working directory. All task context is supplied below.

Do not inspect other directories, use external tools or services, or search for reference implementations.

Do not execute the generated solver. The harness executes it later in its documented runtime directory.

Keep prose paragraphs and list items on one physical line; rely on editor soft wrapping.

You are an expert operations research engineer and Python programmer.

You are given a 1D cutting stock problem:
  - Stock length W = 247 cm.
  - Item types i=0..m-1, each with width size[i] and demand demand[i].
  - Objective: minimize number of stock pieces used.

Context: a manufacturing operation cutting stock material into ordered widths.

Input file data.json contains:
  - "stock_size" (or "roll_width"), "stock_unit"
  - "item_types": list of { "id": int, "name": str, "size": int, "demand": int }

Examples from this instance:
  - PaperProduct_0 (id=0, size=137, demand=41)
  - PaperProduct_1 (id=1, size=77, demand=35)
  - PaperProduct_2 (id=2, size=29, demand=60)
  - PaperProduct_3 (id=3, size=57, demand=29)
  - ...

Task:
Assume gurobipy and NumPy are available; the Python standard library is also allowed. Do not depend on any files other than data.json. Write a Python program solver.py that:
1) loads data.json,
2) Implements a Column Generation to solve the cutting stock problem,
3) writes output.json with at least:
{
  "status": "optimal | feasible | infeasible | error",
  "method": "column_generation",
  "objective_value": <#stock pieces used>,
  "solution": {
    "patterns": [
      {
        "pattern_id": <int>,
        "usage": <int>,
        "cuts": [
          {
            "item_id": <int>,
            "count": <int>
          },
          ...
        ]
      },
      ...
    ]
  }
}
Each pattern's total width must not exceed stock_size. Across all patterns (accounting for usage) every item demand must be met.

Decomposition method requirements:

Use a restricted master problem and pricing subproblems. Use master dual or reduced-cost information to generate columns, add useful columns to the restricted master, and reoptimize iteratively. A final integer solve of the restricted master over the accumulated columns is allowed. Construct the reported solution from this column-generation procedure.

Do not build and solve a separate monolithic optimization model of the full original problem, including for an initial incumbent, initialization, a warm start, a fallback, the reported solution, or an optimality certificate. This prohibition includes a full-original-model feasibility solve with a zero objective and applies to every reachable solution-producing path, including conditional fallback paths, even when decomposition also runs.

Heuristic initialization, directly constructed seed columns, phase-I restricted masters with artificial variables and pricing, valid initial bounds or cuts, and warm-starting a legitimate decomposition master are allowed. A permitted initialization must not obtain its solution from a separate full-original-model solve. The restriction concerns that separate full model, not legitimate decomposition masters, fixed-decision subproblems, pricing problems, or integer restricted masters.

Implement the information feedback required by the requested method. There is no minimum iteration count and no requirement that every instance add a cut or column, or improve the initial incumbent: legitimate immediate convergence or termination is allowed. Report mutually consistent decisions and objective values from the same incumbent. Use status feasible when returning an incumbent without a global optimality proof; optimality of an integer restricted master alone does not prove global optimality.

The data below are provided as read-only context for this editing call.
The generated solver.py will be executed later in a separate runtime directory, where it must read ../../data.json and write output.json in that runtime directory.

Implement a standalone solver.py using gurobipy, NumPy, and the Python standard library. Write output.json using the documented contract. Do not import repository implementations or load reference solutions.
The harness limits the entire solver.py execution to 60 seconds and numerical solvers to 1 thread(s). Design your algorithm and termination handling within this budget.

--- BEGIN CONTEXT: csp_medium_seed000/data.json ---
{
  "instance_size": "medium",
  "stock_size": 247,
  "stock_unit": "cm",
  "item_types": [
    {
      "id": 0,
      "name": "PaperProduct_0",
      "size": 137,
      "demand": 41
    },
    {
      "id": 1,
      "name": "PaperProduct_1",
      "size": 77,
      "demand": 35
    },
    {
      "id": 2,
      "name": "PaperProduct_2",
      "size": 29,
      "demand": 60
    },
    {
      "id": 3,
      "name": "PaperProduct_3",
      "size": 57,
      "demand": 29
    },
    {
      "id": 4,
      "name": "PaperProduct_4",
      "size": 147,
      "demand": 40
    },
    {
      "id": 5,
      "name": "PaperProduct_5",
      "size": 89,
      "demand": 32
    }
  ]
}
--- END CONTEXT: csp_medium_seed000/data.json ---

--- BEGIN CONTEXT: attempt_01/solver.py ---
import json
import math
import time

import gurobipy as gp
from gurobipy import GRB


def main():
    t_start = time.time()
    time_budget = 55.0

    with open("../../data.json", "r") as f:
        data = json.load(f)

    stock_size = data.get("stock_size", data.get("roll_width"))
    items = data["item_types"]
    m = len(items)
    sizes = [int(it["size"]) for it in items]
    demands = [int(it["demand"]) for it in items]
    ids = [int(it["id"]) for it in items]

    status = "optimal"

    try:
        # ---- Initial patterns: one-item patterns (as many copies as fit) ----
        patterns = []  # each: list of counts per item index
        for i in range(m):
            cnt = stock_size // sizes[i]
            if cnt < 1:
                # item does not fit in stock at all -> infeasible
                raise ValueError(f"Item {ids[i]} with size {sizes[i]} exceeds stock size {stock_size}")
            col = [0] * m
            col[i] = cnt
            patterns.append(col)

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", 1)
        env.start()

        # ---- Column generation loop ----
        max_iters = 200
        for it in range(max_iters):
            if time.time() - t_start > time_budget:
                break

            # Build / solve restricted master LP relaxation
            rmp = gp.Model(env=env)
            rmp.setParam("OutputFlag", 0)
            rmp.setParam("Threads", 1)
            x_vars = []
            for p_idx, col in enumerate(patterns):
                x = rmp.addVar(obj=1.0, lb=0.0, vtype=GRB.CONTINUOUS, name=f"x_{p_idx}")
                x_vars.append(x)
            constrs = []
            for i in range(m):
                expr = gp.LinExpr()
                for p_idx, col in enumerate(patterns):
                    if col[i] > 0:
                        expr += col[i] * x_vars[p_idx]
                c = rmp.addConstr(expr >= demands[i], name=f"demand_{i}")
                constrs.append(c)
            rmp.setParam("TimeLimit", max(1.0, time_budget - (time.time() - t_start)))
            rmp.optimize()

            if rmp.Status != GRB.OPTIMAL:
                # fallback: stop generating, proceed with what we have
                break

            duals = [c.Pi for c in constrs]

            # ---- Pricing subproblem: knapsack maximizing dual value ----
            sub = gp.Model(env=env)
            sub.setParam("OutputFlag", 0)
            sub.setParam("Threads", 1)
            a_vars = []
            for i in range(m):
                ub = stock_size // sizes[i]
                a = sub.addVar(lb=0, ub=ub, vtype=GRB.INTEGER, name=f"a_{i}")
                a_vars.append(a)
            sub.addConstr(gp.quicksum(sizes[i] * a_vars[i] for i in range(m)) <= stock_size)
            sub.setObjective(gp.quicksum(duals[i] * a_vars[i] for i in range(m)), GRB.MAXIMIZE)
            sub.setParam("TimeLimit", max(1.0, time_budget - (time.time() - t_start)))
            sub.optimize()

            if sub.Status != GRB.OPTIMAL and sub.Status != GRB.TIME_LIMIT:
                break

            best_val = sub.ObjVal if sub.SolCount > 0 else -math.inf

            if best_val <= 1.0 + 1e-6:
                # no improving column found -> optimal for LP relaxation
                break

            new_col = [int(round(a_vars[i].X)) for i in range(m)]
            if all(c == 0 for c in new_col):
                break

            # avoid adding duplicate columns repeatedly
            if new_col in patterns:
                break

            patterns.append(new_col)

        # ---- Final integer solve of restricted master over accumulated columns ----
        final = gp.Model(env=env)
        final.setParam("OutputFlag", 0)
        final.setParam("Threads", 1)
        remaining = max(1.0, time_budget - (time.time() - t_start))
        final.setParam("TimeLimit", remaining)

        y_vars = []
        for p_idx, col in enumerate(patterns):
            y = final.addVar(obj=1.0, lb=0.0, vtype=GRB.INTEGER, name=f"y_{p_idx}")
            y_vars.append(y)
        for i in range(m):
            expr = gp.LinExpr()
            for p_idx, col in enumerate(patterns):
                if col[i] > 0:
                    expr += col[i] * y_vars[p_idx]
            final.addConstr(expr >= demands[i], name=f"fdemand_{i}")

        final.optimize()

        if final.SolCount == 0:
            status = "error"
            result = {
                "status": status,
                "method": "column_generation",
                "objective_value": None,
                "solution": {"patterns": []},
            }
            with open("output.json", "w") as f:
                json.dump(result, f, indent=2)
            return

        if final.Status == GRB.OPTIMAL:
            status = "optimal"
        else:
            status = "feasible"

        out_patterns = []
        pid = 0
        for p_idx, col in enumerate(patterns):
            usage = int(round(y_vars[p_idx].X))
            if usage <= 0:
                continue
            cuts = []
            for i in range(m):
                if col[i] > 0:
                    cuts.append({"item_id": ids[i], "count": col[i]})
            out_patterns.append({
                "pattern_id": pid,
                "usage": usage,
                "cuts": cuts,
            })
            pid += 1

        objective_value = sum(p["usage"] for p in out_patterns)

        result = {
            "status": status,
            "method": "column_generation",
            "objective_value": objective_value,
            "solution": {"patterns": out_patterns},
        }
        with open("output.json", "w") as f:
            json.dump(result, f, indent=2)

    except Exception as e:
        result = {
            "status": "error",
            "method": "column_generation",
            "objective_value": None,
            "solution": {"patterns": []},
            "error": str(e),
        }
        with open("output.json", "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

--- END CONTEXT: attempt_01/solver.py ---

--- BEGIN CONTEXT: attempt_01/debug.md ---
# Debug Reflection — attempt_01

## Diagnosis
The verifier failed the attempt on a single concrete defect: the final status-reporting logic conflates "integer restricted master solved to optimality" with "global optimality." The line `if final.Status == GRB.OPTIMAL: status = "optimal"` reports `optimal` whenever the final integer solve over the accumulated columns terminates optimally, but this only proves optimality of the restricted master (a subset of all possible cutting patterns), not of the original cutting-stock problem. Since the LP relaxation's column generation loop stops as soon as no column has reduced cost > 1 (i.e., LP-optimal for the restricted set), and the integer master is then solved separately, there is no global lower bound (e.g., ceil(LP relaxation objective) compared against the integer incumbent, or proof that no further columns could improve the integer solution) ever computed or checked. So the reported "optimal" status is unsupported — everything else in the algorithm (seed patterns, dual-based knapsack pricing, iterative column addition, final integer solve) was judged structurally sound and compliant with the decomposition requirements.

## Root cause
Status assignment used local (restricted master) optimality as a proxy for global optimality without any certificate.

## Concrete changes for next attempt
1. Always report `status = "feasible"` for the incumbent produced by the final integer restricted-master solve, unless a genuine global optimality certificate is constructed (e.g., comparing the final integer objective to a valid lower bound such as ceil(final LP relaxation bound) AND confirming the pricing loop terminated because no improving column exists at the LP optimum, not merely because of a time/iteration cutoff).
2. Simplest compliant fix: unconditionally set `status = "feasible"` when returning any incumbent from the integer restricted master, since optimality of that integer program does not prove global optimality of the original cutting-stock problem (as stated explicitly in the task requirements). Only use `"optimal"` if the column-generation loop provably converged (no improving column exists, i.e., pricing objective <= 1 + tol) AND the final integer master's objective equals the LP relaxation's rounded-up bound — otherwise default to `"feasible"`.
3. Keep all other logic (seed patterns, duals from master, knapsack pricing subproblem, final integer solve over accumulated columns) unchanged — it already satisfies the decomposition requirements and avoids any monolithic full-model solve.
4. Double check: if `sub.Status == GRB.TIME_LIMIT` is used to break the loop (not a proven no-improving-column condition), that path must also report `"feasible"`, not `"optimal"`.

--- END CONTEXT: attempt_01/debug.md ---