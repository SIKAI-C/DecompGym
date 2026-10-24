import json
import time
import os
import gurobipy as gp
from gurobipy import GRB

START = time.time()
TIME_BUDGET = 55.0


def time_left():
    return TIME_BUDGET - (time.time() - START)


def load_data():
    data_path = os.path.join("..", "..", "data.json")
    with open(data_path, "r") as f:
        return json.load(f)


def solve_pricing(sizes, duals, W):
    n = len(sizes)
    model = gp.Model("pricing")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    tl = max(1.0, time_left())
    model.Params.TimeLimit = min(tl, 10.0)
    a = model.addVars(n, vtype=GRB.INTEGER, lb=0, name="a")
    model.setObjective(gp.quicksum(duals[i] * a[i] for i in range(n)), GRB.MAXIMIZE)
    model.addConstr(gp.quicksum(sizes[i] * a[i] for i in range(n)) <= W)
    model.optimize()
    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        return None, None
    if model.SolCount == 0:
        return None, None
    counts = [int(round(a[i].X)) for i in range(n)]
    obj = model.ObjVal
    return counts, obj


def main():
    data = load_data()
    W = data.get("stock_size", data.get("roll_width"))
    items = data["item_types"]
    m = len(items)
    sizes = [it["size"] for it in items]
    demands = [it["demand"] for it in items]
    ids = [it["id"] for it in items]

    # Seed patterns: one item type per pattern, as many copies as fit.
    patterns = []
    for i in range(m):
        cnt = [0] * m
        cnt[i] = W // sizes[i]
        if cnt[i] < 1:
            cnt[i] = 1  # item wider than stock edge case, still need a column
        patterns.append(cnt)

    try:
        master = gp.Model("master")
        master.Params.OutputFlag = 0
        master.Params.Threads = 1

        x_vars = []
        for p_idx, pat in enumerate(patterns):
            x_vars.append(master.addVar(lb=0.0, obj=1.0, vtype=GRB.CONTINUOUS, name=f"x_{p_idx}"))

        master.update()

        constrs = []
        for i in range(m):
            expr = gp.quicksum(patterns[p_idx][i] * x_vars[p_idx] for p_idx in range(len(patterns)))
            c = master.addConstr(expr >= demands[i], name=f"demand_{i}")
            constrs.append(c)

        master.ModelSense = GRB.MINIMIZE
        master.optimize()

        max_iters = 200
        it = 0
        while it < max_iters and time_left() > 5.0:
            it += 1
            if master.Status != GRB.OPTIMAL:
                break
            duals = [constrs[i].Pi for i in range(m)]
            counts, price_obj = solve_pricing(sizes, duals, W)
            if counts is None:
                break
            reduced_cost = 1.0 - price_obj
            if reduced_cost < -1e-6:
                new_col = gp.Column(
                    [counts[i] for i in range(m) if counts[i] != 0],
                    [constrs[i] for i in range(m) if counts[i] != 0],
                )
                new_var = master.addVar(lb=0.0, obj=1.0, vtype=GRB.CONTINUOUS, column=new_col, name=f"x_{len(patterns)}")
                patterns.append(counts)
                x_vars.append(new_var)
                master.optimize()
            else:
                break

        # Final integer solve of restricted master over accumulated columns.
        for v in x_vars:
            v.VType = GRB.INTEGER
        master.Params.TimeLimit = max(1.0, time_left() - 1.0)
        master.optimize()

        status_map = {
            GRB.OPTIMAL: "optimal",
            GRB.TIME_LIMIT: "feasible",
            GRB.SUBOPTIMAL: "feasible",
        }

        if master.SolCount == 0:
            result = {
                "status": "infeasible" if master.Status == GRB.INFEASIBLE else "error",
                "method": "column_generation",
                "objective_value": None,
                "solution": {"patterns": []},
            }
        else:
            status = status_map.get(master.Status, "feasible")
            obj_val = master.ObjVal
            out_patterns = []
            pid = 0
            for p_idx, pat in enumerate(patterns):
                usage = x_vars[p_idx].X
                usage_int = int(round(usage))
                if usage_int <= 0:
                    continue
                cuts = []
                for i in range(m):
                    if pat[i] > 0:
                        cuts.append({"item_id": ids[i], "count": pat[i]})
                out_patterns.append({
                    "pattern_id": pid,
                    "usage": usage_int,
                    "cuts": cuts,
                })
                pid += 1

            result = {
                "status": status,
                "method": "column_generation",
                "objective_value": int(round(obj_val)),
                "solution": {"patterns": out_patterns},
            }

        with open("output.json", "w") as f:
            json.dump(result, f, indent=2)

    except gp.GurobiError as e:
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
