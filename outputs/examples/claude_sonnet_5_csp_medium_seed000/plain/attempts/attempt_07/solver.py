import json
import math

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


def solve_knapsack_pricing(W, sizes, duals, time_limit=5.0):
    n = len(sizes)
    m = gp.Model("pricing")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = time_limit
    x = m.addVars(n, vtype=GRB.INTEGER, lb=0, name="x")
    m.addConstr(gp.quicksum(sizes[i] * x[i] for i in range(n)) <= W)
    m.setObjective(gp.quicksum(duals[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        return None, None
    counts = [int(round(x[i].X)) for i in range(n)]
    return counts, m.ObjVal


def build_initial_patterns(W, sizes):
    n = len(sizes)
    patterns = []
    for i in range(n):
        cap = W // sizes[i]
        if cap < 1:
            cap = 1
        counts = [0] * n
        counts[i] = int(cap)
        patterns.append(counts)
    return patterns


def main():
    W, ids, sizes, demands = load_data()
    n = len(sizes)

    patterns = build_initial_patterns(W, sizes)

    max_iters = 200
    import time
    start_time = time.time()
    time_budget = 50.0

    master = None
    for iteration in range(max_iters):
        if time.time() - start_time > time_budget:
            break

        master = gp.Model("master")
        master.Params.OutputFlag = 0
        master.Params.Threads = 1
        vars_list = []
        for p_idx, pat in enumerate(patterns):
            v = master.addVar(obj=1.0, lb=0.0, vtype=GRB.CONTINUOUS, name=f"p{p_idx}")
            vars_list.append(v)

        constrs = []
        for i in range(n):
            expr = gp.quicksum(patterns[p_idx][i] * vars_list[p_idx] for p_idx in range(len(patterns)))
            c = master.addConstr(expr >= demands[i], name=f"demand_{i}")
            constrs.append(c)

        master.setObjective(gp.quicksum(vars_list), GRB.MINIMIZE)
        master.optimize()

        if master.Status != GRB.OPTIMAL:
            break

        duals = [constrs[i].Pi for i in range(n)]

        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time <= 0.5:
            break
        pt_limit = min(5.0, max(0.5, remaining_time / 2))
        counts, obj_val = solve_knapsack_pricing(W, sizes, duals, time_limit=pt_limit)

        if counts is None or obj_val is None:
            break

        if obj_val <= 1.0 + 1e-6:
            break

        if sum(c * sizes[i] for i, c in enumerate(counts)) > W + 1e-6:
            break

        if all(c == 0 for c in counts):
            break

        if counts in patterns:
            break

        patterns.append(counts)

    # Final integer solve of restricted master
    final_model = gp.Model("final_master")
    final_model.Params.OutputFlag = 0
    final_model.Params.Threads = 1
    remaining_time = max(1.0, time_budget - (time.time() - start_time))
    final_model.Params.TimeLimit = remaining_time

    vars_list = []
    for p_idx, pat in enumerate(patterns):
        v = final_model.addVar(obj=1.0, lb=0.0, vtype=GRB.INTEGER, name=f"p{p_idx}")
        vars_list.append(v)

    constrs = []
    for i in range(n):
        expr = gp.quicksum(patterns[p_idx][i] * vars_list[p_idx] for p_idx in range(len(patterns)))
        c = final_model.addConstr(expr >= demands[i], name=f"demand_{i}")
        constrs.append(c)

    final_model.setObjective(gp.quicksum(vars_list), GRB.MINIMIZE)
    final_model.optimize()

    status_map_optimal = (GRB.OPTIMAL,)
    status_map_feasible = (GRB.TIME_LIMIT, GRB.SUBOPTIMAL, GRB.INTERRUPTED)

    result = {}
    if final_model.Status in status_map_optimal or final_model.Status in status_map_feasible:
        if final_model.SolCount > 0:
            usage_list = []
            for p_idx, v in enumerate(vars_list):
                u = int(round(v.X))
                if u > 0:
                    usage_list.append((p_idx, u))

            pattern_entries = []
            pid = 0
            for p_idx, usage in usage_list:
                cuts = []
                for i in range(n):
                    if patterns[p_idx][i] > 0:
                        cuts.append({"item_id": ids[i], "count": int(patterns[p_idx][i])})
                pattern_entries.append({
                    "pattern_id": pid,
                    "usage": usage,
                    "cuts": cuts,
                })
                pid += 1

            objective_value = sum(u for _, u in usage_list)
            status = "optimal" if final_model.Status in status_map_optimal else "feasible"

            result = {
                "status": status,
                "method": "column_generation",
                "objective_value": objective_value,
                "solution": {
                    "patterns": pattern_entries
                }
            }
        else:
            result = {
                "status": "error",
                "method": "column_generation",
                "objective_value": None,
                "solution": {"patterns": []}
            }
    elif final_model.Status == GRB.INFEASIBLE:
        result = {
            "status": "infeasible",
            "method": "column_generation",
            "objective_value": None,
            "solution": {"patterns": []}
        }
    else:
        result = {
            "status": "error",
            "method": "column_generation",
            "objective_value": None,
            "solution": {"patterns": []}
        }

    with open("output.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
