import json
import math
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def load_data():
    with open("../../data.json", "r") as f:
        data = json.load(f)
    W = data.get("stock_size", data.get("roll_width"))
    items = data["item_types"]
    ids = [it["id"] for it in items]
    sizes = [it["size"] for it in items]
    demands = [it["demand"] for it in items]
    return W, ids, sizes, demands


def solve_knapsack_pricing(duals, sizes, W):
    n = len(sizes)
    m = gp.Model("pricing")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    x = m.addVars(n, vtype=GRB.INTEGER, lb=0, name="x")
    m.addConstr(gp.quicksum(sizes[i] * x[i] for i in range(n)) <= W)
    m.setObjective(gp.quicksum(duals[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        return None, None
    counts = [int(round(x[i].X)) for i in range(n)]
    return counts, m.ObjVal


def main():
    try:
        W, ids, sizes, demands = load_data()
        n = len(sizes)

        # Initial patterns: one item type per pattern (homogeneous), as many as fit.
        patterns = []
        for i in range(n):
            cnt = W // sizes[i]
            if cnt < 1:
                cnt = 1  # shouldn't happen if item fits; fallback safety
            col = [0] * n
            col[i] = int(cnt)
            patterns.append(col)

        max_iters = 200
        time_budget_hit = False

        for it in range(max_iters):
            n_patterns = len(patterns)
            master = gp.Model("master_rmp")
            master.Params.OutputFlag = 0
            master.Params.Threads = 1
            lam = master.addVars(n_patterns, lb=0.0, vtype=GRB.CONTINUOUS, name="lam")
            cons = []
            for i in range(n):
                c = master.addConstr(
                    gp.quicksum(patterns[p][i] * lam[p] for p in range(n_patterns)) >= demands[i],
                    name=f"demand_{i}"
                )
                cons.append(c)
            master.setObjective(gp.quicksum(lam[p] for p in range(n_patterns)), GRB.MINIMIZE)
            master.optimize()

            if master.Status != GRB.OPTIMAL:
                break

            duals = [cons[i].Pi for i in range(n)]

            new_col, price_obj = solve_knapsack_pricing(duals, sizes, W)
            if new_col is None:
                break

            if price_obj <= 1.0 + 1e-6:
                break

            if new_col in patterns:
                break

            patterns.append(new_col)

        # Final integer solve of restricted master over accumulated columns
        n_patterns = len(patterns)
        master_int = gp.Model("master_int")
        master_int.Params.OutputFlag = 0
        master_int.Params.Threads = 1
        master_int.Params.TimeLimit = 20
        y = master_int.addVars(n_patterns, lb=0.0, vtype=GRB.INTEGER, name="y")
        for i in range(n):
            master_int.addConstr(
                gp.quicksum(patterns[p][i] * y[p] for p in range(n_patterns)) >= demands[i]
            )
        master_int.setObjective(gp.quicksum(y[p] for p in range(n_patterns)), GRB.MINIMIZE)
        master_int.optimize()

        if master_int.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) or master_int.SolCount == 0:
            output = {
                "status": "error",
                "method": "column_generation",
                "objective_value": None,
                "solution": {"patterns": []}
            }
            with open("output.json", "w") as f:
                json.dump(output, f, indent=2)
            return

        status_str = "optimal" if master_int.Status == GRB.OPTIMAL else "feasible"

        result_patterns = []
        pid = 0
        for p in range(n_patterns):
            usage = int(round(y[p].X))
            if usage <= 0:
                continue
            cuts = []
            for i in range(n):
                if patterns[p][i] > 0:
                    cuts.append({"item_id": ids[i], "count": int(patterns[p][i])})
            result_patterns.append({
                "pattern_id": pid,
                "usage": usage,
                "cuts": cuts
            })
            pid += 1

        objective_value = sum(rp["usage"] for rp in result_patterns)

        output = {
            "status": status_str,
            "method": "column_generation",
            "objective_value": objective_value,
            "solution": {
                "patterns": result_patterns
            }
        }
        with open("output.json", "w") as f:
            json.dump(output, f, indent=2)

    except Exception as e:
        output = {
            "status": "error",
            "method": "column_generation",
            "objective_value": None,
            "solution": {"patterns": []},
            "error": str(e)
        }
        with open("output.json", "w") as f:
            json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
