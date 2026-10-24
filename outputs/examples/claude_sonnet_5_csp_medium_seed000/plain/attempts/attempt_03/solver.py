import json
import time

import gurobipy as gp
from gurobipy import GRB

START = time.time()
TIME_LIMIT = 55.0


def remaining_time():
    return max(1.0, TIME_LIMIT - (time.time() - START))


def load_data():
    with open("../../data.json", "r") as f:
        data = json.load(f)
    stock_size = data.get("stock_size", data.get("roll_width"))
    items = data["item_types"]
    ids = [it["id"] for it in items]
    sizes = [it["size"] for it in items]
    demands = [it["demand"] for it in items]
    return stock_size, ids, sizes, demands


def build_initial_patterns(stock_size, sizes):
    patterns = []
    n = len(sizes)
    for i in range(n):
        cnt = stock_size // sizes[i]
        if cnt < 1:
            cnt = 1
        pattern = [0] * n
        pattern[i] = int(cnt)
        patterns.append(pattern)
    return patterns


def solve_pricing(stock_size, sizes, duals, time_limit):
    n = len(sizes)
    sub = gp.Model("pricing")
    sub.Params.OutputFlag = 0
    sub.Params.Threads = 1
    sub.Params.TimeLimit = time_limit
    x = sub.addVars(n, vtype=GRB.INTEGER, lb=0, name="x")
    sub.addConstr(gp.quicksum(sizes[i] * x[i] for i in range(n)) <= stock_size)
    sub.setObjective(1 - gp.quicksum(duals[i] * x[i] for i in range(n)), GRB.MINIMIZE)
    sub.optimize()
    if sub.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        return None, None
    if sub.SolCount == 0:
        return None, None
    reduced_cost = sub.ObjVal
    pattern = [int(round(x[i].X)) for i in range(n)]
    return reduced_cost, pattern


def main():
    try:
        stock_size, ids, sizes, demands = load_data()
        n = len(sizes)

        patterns = build_initial_patterns(stock_size, sizes)

        max_outer_iters = 200
        it = 0
        while it < max_outer_iters and remaining_time() > 3.0:
            it += 1

            master = gp.Model("master_lp")
            master.Params.OutputFlag = 0
            master.Params.Threads = 1
            master.Params.TimeLimit = remaining_time()

            num_patterns = len(patterns)
            use_vars = master.addVars(num_patterns, lb=0.0, name="use")
            constrs = []
            for i in range(n):
                c = master.addConstr(
                    gp.quicksum(patterns[p][i] * use_vars[p] for p in range(num_patterns)) >= demands[i],
                    name=f"demand_{i}",
                )
                constrs.append(c)
            master.setObjective(gp.quicksum(use_vars[p] for p in range(num_patterns)), GRB.MINIMIZE)
            master.optimize()

            if master.Status != GRB.OPTIMAL:
                break

            duals = [constrs[i].Pi for i in range(n)]

            sub_time = min(10.0, remaining_time())
            reduced_cost, new_pattern = solve_pricing(stock_size, sizes, duals, sub_time)

            if reduced_cost is None:
                break

            if reduced_cost < -1e-6:
                if new_pattern in patterns:
                    break
                patterns.append(new_pattern)
            else:
                break

        # Final integer solve of restricted master over accumulated columns
        master_int = gp.Model("master_int")
        master_int.Params.OutputFlag = 0
        master_int.Params.Threads = 1
        master_int.Params.TimeLimit = remaining_time()

        num_patterns = len(patterns)
        use_vars = master_int.addVars(num_patterns, lb=0, vtype=GRB.INTEGER, name="use")
        for i in range(n):
            master_int.addConstr(
                gp.quicksum(patterns[p][i] * use_vars[p] for p in range(num_patterns)) >= demands[i],
                name=f"demand_{i}",
            )
        master_int.setObjective(gp.quicksum(use_vars[p] for p in range(num_patterns)), GRB.MINIMIZE)
        master_int.optimize()

        if master_int.SolCount == 0:
            result = {
                "status": "infeasible" if master_int.Status == GRB.INFEASIBLE else "error",
                "method": "column_generation",
                "objective_value": None,
                "solution": {"patterns": []},
            }
            with open("output.json", "w") as f:
                json.dump(result, f, indent=2)
            return

        status = "optimal" if master_int.Status == GRB.OPTIMAL else "feasible"

        pattern_list = []
        pid = 0
        for p in range(num_patterns):
            usage = int(round(use_vars[p].X))
            if usage <= 0:
                continue
            cuts = []
            for i in range(n):
                if patterns[p][i] > 0:
                    cuts.append({"item_id": ids[i], "count": int(patterns[p][i])})
            pattern_list.append({"pattern_id": pid, "usage": usage, "cuts": cuts})
            pid += 1

        objective_value = sum(pat["usage"] for pat in pattern_list)

        result = {
            "status": status,
            "method": "column_generation",
            "objective_value": objective_value,
            "solution": {"patterns": pattern_list},
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
