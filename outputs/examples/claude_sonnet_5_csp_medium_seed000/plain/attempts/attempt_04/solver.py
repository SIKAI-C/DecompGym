import json
import math
import gurobipy as gp
from gurobipy import GRB


def load_data():
    with open("../../data.json", "r") as f:
        data = json.load(f)
    stock_size = data.get("stock_size", data.get("roll_width"))
    items = data["item_types"]
    return stock_size, items


def solve_knapsack_pricing(sizes, duals, W, time_limit):
    m = len(sizes)
    model = gp.Model("pricing")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit
    x = model.addVars(m, vtype=GRB.INTEGER, lb=0, name="x")
    model.setObjective(gp.quicksum(duals[i] * x[i] for i in range(m)), GRB.MAXIMIZE)
    model.addConstr(gp.quicksum(sizes[i] * x[i] for i in range(m)) <= W)
    model.optimize()
    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        return None, None
    counts = [int(round(x[i].X)) for i in range(m)]
    obj = model.ObjVal
    return counts, obj


def main():
    try:
        W, items = load_data()
        m = len(items)
        sizes = [it["size"] for it in items]
        demands = [it["demand"] for it in items]
        ids = [it["id"] for it in items]

        # Initial patterns: one item type per pattern (as many as fit)
        patterns = []
        for i in range(m):
            cnt = W // sizes[i]
            if cnt < 1:
                cnt = 1  # shouldn't happen if item fits; guard
            vec = [0] * m
            vec[i] = int(cnt)
            patterns.append(vec)

        master = gp.Model("master")
        master.Params.OutputFlag = 0
        master.Params.Threads = 1

        pattern_vars = []
        for p_idx, vec in enumerate(patterns):
            v = master.addVar(obj=1.0, lb=0.0, vtype=GRB.CONTINUOUS, name=f"p{p_idx}")
            pattern_vars.append(v)

        constrs = []
        for i in range(m):
            constrs.append(master.addConstr(
                gp.quicksum(patterns[p][i] * pattern_vars[p] for p in range(len(patterns))) >= demands[i],
                name=f"demand_{i}"
            ))

        master.setObjective(gp.quicksum(pattern_vars), GRB.MINIMIZE)
        master.update()

        import time
        start_time = time.time()
        time_budget = 55.0

        max_iters = 200
        for it in range(max_iters):
            if time.time() - start_time > time_budget:
                break
            master.optimize()
            if master.Status != GRB.OPTIMAL:
                break
            duals = [constrs[i].Pi for i in range(m)]

            remaining = time_budget - (time.time() - start_time)
            if remaining < 1.0:
                break
            counts, obj = solve_knapsack_pricing(sizes, duals, W, min(10.0, remaining))
            if counts is None:
                break

            if obj <= 1.0 + 1e-6:
                break

            new_vec = counts
            col = gp.Column(coeffs=[new_vec[i] for i in range(m)], constrs=constrs)
            v = master.addVar(obj=1.0, lb=0.0, vtype=GRB.CONTINUOUS, column=col, name=f"p{len(patterns)}")
            pattern_vars.append(v)
            patterns.append(new_vec)

        # Final integer solve of restricted master
        for v in pattern_vars:
            v.VType = GRB.INTEGER
        remaining = time_budget - (time.time() - start_time)
        master.Params.TimeLimit = max(1.0, remaining)
        master.optimize()

        if master.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and master.SolCount > 0:
            status = "optimal" if master.Status == GRB.OPTIMAL else "feasible"
            out_patterns = []
            total = 0
            for p_idx, v in enumerate(pattern_vars):
                usage = int(round(v.X))
                if usage <= 0:
                    continue
                cuts = []
                for i in range(m):
                    c = patterns[p_idx][i]
                    if c > 0:
                        cuts.append({"item_id": ids[i], "count": int(c)})
                out_patterns.append({
                    "pattern_id": p_idx,
                    "usage": usage,
                    "cuts": cuts
                })
                total += usage

            result = {
                "status": status,
                "method": "column_generation",
                "objective_value": total,
                "solution": {
                    "patterns": out_patterns
                }
            }
        else:
            result = {
                "status": "infeasible",
                "method": "column_generation",
                "objective_value": None,
                "solution": {"patterns": []}
            }

        with open("output.json", "w") as f:
            json.dump(result, f, indent=2)

    except Exception as e:
        result = {
            "status": "error",
            "method": "column_generation",
            "objective_value": None,
            "solution": {"patterns": []},
            "error": str(e)
        }
        with open("output.json", "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
