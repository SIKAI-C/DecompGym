import json
import math
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np


def main():
    t_start = time.time()
    with open("../../data.json", "r") as f:
        data = json.load(f)

    W = data.get("stock_size", data.get("roll_width"))
    items = data["item_types"]
    m = len(items)
    sizes = [it["size"] for it in items]
    demands = [it["demand"] for it in items]
    ids = [it["id"] for it in items]

    result = {
        "status": "error",
        "method": "column_generation",
        "objective_value": None,
        "solution": {"patterns": []},
    }

    try:
        # Initial patterns: one item type per pattern, as many as fit in stock.
        patterns = []
        for i in range(m):
            cnt = W // sizes[i]
            if cnt < 1:
                cnt = 1  # should not happen if size <= W
            pat = [0] * m
            pat[i] = int(cnt)
            patterns.append(pat)

        time_limit = 60.0

        master_env = gp.Env(empty=True)
        master_env.setParam("OutputFlag", 0)
        master_env.setParam("Threads", 1)
        master_env.start()

        max_iters = 200
        for it_num in range(max_iters):
            if time.time() - t_start > time_limit - 10:
                break

            # Build and solve LP relaxation of restricted master.
            rm = gp.Model(env=master_env)
            rm.setParam("OutputFlag", 0)
            rm.setParam("Threads", 1)
            x_vars = []
            for p_idx, pat in enumerate(patterns):
                x = rm.addVar(lb=0.0, obj=1.0, vtype=GRB.CONTINUOUS, name=f"x{p_idx}")
                x_vars.append(x)
            constrs = []
            for i in range(m):
                coeffs = [patterns[p_idx][i] for p_idx in range(len(patterns))]
                expr = gp.LinExpr()
                for p_idx, c in enumerate(coeffs):
                    if c != 0:
                        expr += c * x_vars[p_idx]
                con = rm.addConstr(expr >= demands[i], name=f"demand_{i}")
                constrs.append(con)
            rm.ModelSense = GRB.MINIMIZE
            rm.optimize()

            if rm.status != GRB.OPTIMAL:
                break

            duals = [constrs[i].Pi for i in range(m)]

            # Pricing subproblem: knapsack to find pattern with max reduced cost.
            # maximize sum(duals[i]*y_i) - 1 subject to sum(size_i * y_i) <= W
            sp = gp.Model(env=master_env)
            sp.setParam("OutputFlag", 0)
            sp.setParam("Threads", 1)
            y_vars = []
            for i in range(m):
                ub = W // sizes[i]
                y = sp.addVar(lb=0, ub=int(ub), vtype=GRB.INTEGER, name=f"y{i}")
                y_vars.append(y)
            sp.addConstr(gp.quicksum(sizes[i] * y_vars[i] for i in range(m)) <= W)
            sp.setObjective(gp.quicksum(duals[i] * y_vars[i] for i in range(m)), GRB.MAXIMIZE)
            sp.optimize()

            if sp.status != GRB.OPTIMAL:
                break

            best_val = sp.ObjVal
            if best_val <= 1.0 + 1e-6:
                # No column with negative reduced cost found; converged.
                break

            new_pat = [int(round(y_vars[i].X)) for i in range(m)]
            # Avoid adding duplicate pattern (safety against stalling).
            if new_pat in patterns:
                break
            patterns.append(new_pat)

        # Final integer solve of restricted master over accumulated columns.
        rm_int = gp.Model(env=master_env)
        rm_int.setParam("OutputFlag", 0)
        rm_int.setParam("Threads", 1)
        remaining = max(1.0, time_limit - (time.time() - t_start) - 2.0)
        rm_int.setParam("TimeLimit", remaining)

        x_vars = []
        for p_idx, pat in enumerate(patterns):
            x = rm_int.addVar(lb=0, obj=1.0, vtype=GRB.INTEGER, name=f"x{p_idx}")
            x_vars.append(x)
        for i in range(m):
            expr = gp.LinExpr()
            for p_idx, pat in enumerate(patterns):
                if pat[i] != 0:
                    expr += pat[i] * x_vars[p_idx]
            rm_int.addConstr(expr >= demands[i], name=f"demand_{i}")
        rm_int.ModelSense = GRB.MINIMIZE
        rm_int.optimize()

        if rm_int.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and rm_int.SolCount > 0:
            usage_list = []
            for p_idx, pat in enumerate(patterns):
                usage = int(round(x_vars[p_idx].X))
                if usage > 0:
                    usage_list.append((pat, usage))

            out_patterns = []
            for pid, (pat, usage) in enumerate(usage_list):
                cuts = []
                for i in range(m):
                    if pat[i] > 0:
                        cuts.append({"item_id": ids[i], "count": int(pat[i])})
                out_patterns.append({
                    "pattern_id": pid,
                    "usage": usage,
                    "cuts": cuts,
                })

            obj_val = sum(u for _, u in usage_list)
            status = "optimal" if rm_int.status == GRB.OPTIMAL else "feasible"

            result["status"] = status
            result["objective_value"] = obj_val
            result["solution"]["patterns"] = out_patterns
        else:
            result["status"] = "infeasible" if rm_int.status == GRB.INFEASIBLE else "error"

    except Exception as e:
        result["status"] = "error"
        result["objective_value"] = None
        result["solution"] = {"patterns": []}
        result["error"] = str(e)

    with open("output.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
