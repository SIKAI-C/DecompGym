import json
import os
import time
import math

import gurobipy as gp
from gurobipy import GRB

START = time.time()
TIME_BUDGET = 55.0


def time_left():
    return TIME_BUDGET - (time.time() - START)


def load_data():
    path = os.path.join("..", "..", "data.json")
    with open(path, "r") as f:
        data = json.load(f)
    return data


def main():
    data = load_data()
    W = data.get("stock_size", data.get("roll_width"))
    items = data["item_types"]
    m = len(items)
    sizes = [it["size"] for it in items]
    demands = [it["demand"] for it in items]
    ids = [it["id"] for it in items]

    # Initial patterns: one pattern per item, as many copies of that item as fit.
    patterns = []
    for i in range(m):
        cnt = int(W // sizes[i])
        cnt = max(cnt, 1)
        pat = [0] * m
        pat[i] = cnt
        patterns.append(pat)

    master_env = gp.Env(empty=True)
    master_env.setParam("OutputFlag", 0)
    master_env.setParam("Threads", 1)
    master_env.start()

    max_iters = 200

    for iteration in range(max_iters):
        if time_left() < 5.0:
            break

        rmp = gp.Model(env=master_env)
        rmp.setParam("OutputFlag", 0)
        rmp.setParam("Threads", 1)

        n_pat = len(patterns)
        x = []
        for p in range(n_pat):
            x.append(rmp.addVar(lb=0.0, obj=1.0, vtype=GRB.CONTINUOUS, name=f"x{p}"))

        cons = []
        for i in range(m):
            expr = gp.LinExpr()
            for p in range(n_pat):
                coeff = patterns[p][i]
                if coeff > 0:
                    expr += coeff * x[p]
            c = rmp.addConstr(expr >= demands[i], name=f"dem{i}")
            cons.append(c)

        rmp.ModelSense = GRB.MINIMIZE
        remaining = max(time_left() - 1.0, 1.0)
        rmp.setParam("TimeLimit", remaining)
        rmp.optimize()

        if rmp.status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
            break

        try:
            duals = [cons[i].Pi for i in range(m)]
        except Exception:
            break

        # Pricing subproblem: knapsack to maximize sum(dual_i * a_i) s.t. sum(size_i*a_i) <= W
        if time_left() < 3.0:
            break

        sub = gp.Model(env=master_env)
        sub.setParam("OutputFlag", 0)
        sub.setParam("Threads", 1)
        a = []
        for i in range(m):
            ub = int(W // sizes[i])
            a.append(sub.addVar(lb=0, ub=ub, vtype=GRB.INTEGER, name=f"a{i}"))
        sub.addConstr(gp.quicksum(sizes[i] * a[i] for i in range(m)) <= W)
        sub.setObjective(gp.quicksum(duals[i] * a[i] for i in range(m)), GRB.MAXIMIZE)
        sub.setParam("TimeLimit", max(time_left() - 1.0, 1.0))
        sub.optimize()

        if sub.status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
            break

        best_val = sub.ObjVal
        reduced_cost = 1.0 - best_val

        if reduced_cost < -1e-6:
            new_pat = [int(round(a[i].X)) for i in range(m)]
            if any(c > 0 for c in new_pat):
                patterns.append(new_pat)
            else:
                break
        else:
            break

    # Final integer solve of restricted master with accumulated patterns.
    n_pat = len(patterns)
    final = gp.Model(env=master_env)
    final.setParam("OutputFlag", 0)
    final.setParam("Threads", 1)
    final.setParam("TimeLimit", max(time_left() - 1.0, 1.0))

    y = []
    for p in range(n_pat):
        y.append(final.addVar(lb=0, vtype=GRB.INTEGER, obj=1.0, name=f"y{p}"))

    for i in range(m):
        expr = gp.LinExpr()
        for p in range(n_pat):
            coeff = patterns[p][i]
            if coeff > 0:
                expr += coeff * y[p]
        final.addConstr(expr >= demands[i])

    final.ModelSense = GRB.MINIMIZE
    final.optimize()

    status_map = {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "feasible",
        GRB.SUBOPTIMAL: "feasible",
    }

    if final.SolCount > 0:
        status = status_map.get(final.status, "feasible")
        obj_val = 0
        out_patterns = []
        pid = 0
        for p in range(n_pat):
            usage = int(round(y[p].X))
            if usage > 0:
                cuts = []
                for i in range(m):
                    if patterns[p][i] > 0:
                        cuts.append({"item_id": ids[i], "count": int(patterns[p][i])})
                out_patterns.append({"pattern_id": pid, "usage": usage, "cuts": cuts})
                pid += 1
                obj_val += usage
        result = {
            "status": status,
            "method": "column_generation",
            "objective_value": obj_val,
            "solution": {"patterns": out_patterns},
        }
    else:
        result = {
            "status": "error",
            "method": "column_generation",
            "objective_value": None,
            "solution": {"patterns": []},
        }

    with open("output.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    try:
        main()
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
