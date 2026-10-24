import json
import time
import math

import gurobipy as gp
from gurobipy import GRB


def main():
    start_time = time.time()
    time_budget = 55.0

    with open("../../data.json", "r") as f:
        data = json.load(f)

    stock_size = data.get("stock_size", data.get("roll_width"))
    items = data["item_types"]
    n = len(items)
    sizes = [it["size"] for it in items]
    demands = [it["demand"] for it in items]
    ids = [it["id"] for it in items]

    status = "error"
    objective_value = None
    patterns_out = []

    try:
        # ---- Initial patterns: one item type repeated as many times as fits ----
        patterns = []
        for i in range(n):
            cnt = stock_size // sizes[i]
            if cnt < 1:
                cnt = 1  # should not happen if size <= stock_size
            col = [0] * n
            col[i] = cnt
            patterns.append(col)

        # ---- Column generation loop ----
        max_iters = 200
        it_count = 0
        while True:
            it_count += 1
            if time.time() - start_time > time_budget:
                break

            # Build and solve restricted master LP (relaxation)
            master = gp.Model("master_lp")
            master.Params.OutputFlag = 0
            master.Params.Threads = 1

            x_vars = []
            for p_idx, col in enumerate(patterns):
                x = master.addVar(lb=0.0, obj=1.0, name=f"x_{p_idx}")
                x_vars.append(x)

            constrs = []
            for i in range(n):
                expr = gp.LinExpr()
                for p_idx, col in enumerate(patterns):
                    if col[i] > 0:
                        expr += col[i] * x_vars[p_idx]
                c = master.addConstr(expr >= demands[i], name=f"demand_{i}")
                constrs.append(c)

            master.ModelSense = GRB.MINIMIZE
            master.optimize()

            if master.Status != GRB.OPTIMAL:
                # cannot proceed with pricing without duals
                break

            duals = [c.Pi for c in constrs]

            # ---- Pricing subproblem: knapsack ----
            sub = gp.Model("pricing")
            sub.Params.OutputFlag = 0
            sub.Params.Threads = 1
            sub.Params.TimeLimit = max(1.0, time_budget - (time.time() - start_time))

            a_vars = []
            for i in range(n):
                ub_i = stock_size // sizes[i]
                a = sub.addVar(lb=0, ub=ub_i, vtype=GRB.INTEGER, name=f"a_{i}")
                a_vars.append(a)

            sub.addConstr(gp.quicksum(sizes[i] * a_vars[i] for i in range(n)) <= stock_size)
            sub.setObjective(gp.quicksum(duals[i] * a_vars[i] for i in range(n)), GRB.MAXIMIZE)
            sub.optimize()

            if sub.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or sub.SolCount == 0:
                break

            best_val = sub.ObjVal
            reduced_cost = 1.0 - best_val

            if reduced_cost < -1e-6:
                new_col = [int(round(a_vars[i].X)) for i in range(n)]
                # avoid duplicate / zero columns
                if sum(new_col) > 0 and new_col not in patterns:
                    patterns.append(new_col)
                else:
                    # no new useful column -> stop
                    break
            else:
                # no improving column found -> converged
                break

            if it_count >= max_iters:
                break

        # ---- Final integer solve of restricted master over accumulated columns ----
        master_int = gp.Model("master_ip")
        master_int.Params.OutputFlag = 0
        master_int.Params.Threads = 1
        remaining_time = max(1.0, time_budget - (time.time() - start_time))
        master_int.Params.TimeLimit = remaining_time

        x_int_vars = []
        for p_idx, col in enumerate(patterns):
            x = master_int.addVar(lb=0, vtype=GRB.INTEGER, obj=1.0, name=f"x_{p_idx}")
            x_int_vars.append(x)

        for i in range(n):
            expr = gp.LinExpr()
            for p_idx, col in enumerate(patterns):
                if col[i] > 0:
                    expr += col[i] * x_int_vars[p_idx]
            master_int.addConstr(expr >= demands[i], name=f"demand_{i}")

        master_int.ModelSense = GRB.MINIMIZE
        master_int.optimize()

        if master_int.SolCount > 0:
            objective_value = int(round(master_int.ObjVal))
            if master_int.Status == GRB.OPTIMAL:
                status = "optimal"
            else:
                status = "feasible"

            for p_idx, col in enumerate(patterns):
                usage = int(round(x_int_vars[p_idx].X))
                if usage > 0:
                    cuts = []
                    for i in range(n):
                        if col[i] > 0:
                            cuts.append({"item_id": ids[i], "count": col[i]})
                    patterns_out.append({
                        "pattern_id": p_idx,
                        "usage": usage,
                        "cuts": cuts,
                    })
        else:
            status = "infeasible"
            objective_value = None

    except Exception:
        status = "error"
        objective_value = None
        patterns_out = []

    output = {
        "status": status,
        "method": "column_generation",
        "objective_value": objective_value,
        "solution": {
            "patterns": patterns_out,
        },
    }

    with open("output.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
