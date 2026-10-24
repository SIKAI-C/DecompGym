import json
import time
import math
import gurobipy as gp
from gurobipy import GRB


def main():
    t_start = time.time()
    time_limit = 55.0

    with open("../../data.json", "r") as f:
        data = json.load(f)

    W = data.get("stock_size", data.get("roll_width"))
    items = data["item_types"]
    m = len(items)
    sizes = [it["size"] for it in items]
    demands = [it["demand"] for it in items]
    ids = [it["id"] for it in items]

    status = "optimal"

    # ---- Initial columns: one pattern per item type (max copies that fit) ----
    patterns = []  # each pattern: list of counts length m
    for i in range(m):
        cnt = W // sizes[i]
        if cnt < 1:
            cnt = 1  # degenerate, shouldn't happen since size <= W assumed
        pat = [0] * m
        pat[i] = int(cnt)
        patterns.append(pat)

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)
    env.start()

    try:
        max_iters = 200
        for _iteration in range(max_iters):
            if time.time() - t_start > time_limit:
                status = "feasible"
                break

            # ---- Build / solve restricted master LP relaxation ----
            rmp = gp.Model("rmp", env=env)
            rmp.setParam("OutputFlag", 0)
            rmp.setParam("Threads", 1)

            n_pat = len(patterns)
            x = []
            for p in range(n_pat):
                x.append(rmp.addVar(lb=0.0, obj=1.0, vtype=GRB.CONTINUOUS, name=f"x_{p}"))

            cons = []
            for i in range(m):
                expr = gp.LinExpr()
                for p in range(n_pat):
                    c = patterns[p][i]
                    if c:
                        expr += c * x[p]
                cons.append(rmp.addConstr(expr >= demands[i], name=f"demand_{i}"))

            rmp.modelSense = GRB.MINIMIZE
            rmp.optimize()

            if rmp.status != GRB.OPTIMAL:
                status = "feasible"
                break

            duals = [cons[i].Pi for i in range(m)]

            # ---- Pricing subproblem: knapsack ----
            remaining = time_limit - (time.time() - t_start)
            if remaining <= 0:
                status = "feasible"
                break

            pricing = gp.Model("pricing", env=env)
            pricing.setParam("OutputFlag", 0)
            pricing.setParam("Threads", 1)
            pricing.setParam("TimeLimit", max(1.0, min(remaining, 10.0)))

            a = []
            for i in range(m):
                ub = W // sizes[i]
                a.append(pricing.addVar(lb=0, ub=ub, vtype=GRB.INTEGER, name=f"a_{i}"))

            pricing.addConstr(gp.quicksum(sizes[i] * a[i] for i in range(m)) <= W)
            pricing.setObjective(gp.quicksum(duals[i] * a[i] for i in range(m)), GRB.MAXIMIZE)
            pricing.optimize()

            if pricing.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
                status = "feasible"
                break

            best_val = pricing.ObjVal if pricing.SolCount > 0 else -math.inf

            if best_val <= 1.0 + 1e-6:
                # no improving column found; LP relaxation is optimal
                break

            new_pat = [int(round(a[i].X)) for i in range(m)]

            # avoid adding duplicate pattern (numerical safety)
            if new_pat in patterns:
                break

            patterns.append(new_pat)

        # ---- Final integer solve of restricted master over accumulated columns ----
        n_pat = len(patterns)
        final_model = gp.Model("rmp_int", env=env)
        final_model.setParam("OutputFlag", 0)
        final_model.setParam("Threads", 1)
        remaining = max(1.0, time_limit - (time.time() - t_start))
        final_model.setParam("TimeLimit", remaining)

        x = []
        for p in range(n_pat):
            x.append(final_model.addVar(lb=0, vtype=GRB.INTEGER, obj=1.0, name=f"x_{p}"))

        for i in range(m):
            expr = gp.LinExpr()
            for p in range(n_pat):
                c = patterns[p][i]
                if c:
                    expr += c * x[p]
            final_model.addConstr(expr >= demands[i], name=f"demand_{i}")

        final_model.modelSense = GRB.MINIMIZE
        final_model.optimize()

        if final_model.SolCount == 0:
            result = {
                "status": "error",
                "method": "column_generation",
                "objective_value": None,
                "solution": {"patterns": []},
            }
            with open("output.json", "w") as f:
                json.dump(result, f, indent=2)
            return

        if final_model.status == GRB.OPTIMAL and status == "optimal":
            final_status = "optimal"
        else:
            final_status = "feasible"

        out_patterns = []
        pid = 0
        total_usage = 0
        for p in range(n_pat):
            usage = int(round(x[p].X))
            if usage <= 0:
                continue
            cuts = []
            for i in range(m):
                if patterns[p][i] > 0:
                    cuts.append({"item_id": ids[i], "count": patterns[p][i]})
            out_patterns.append({"pattern_id": pid, "usage": usage, "cuts": cuts})
            pid += 1
            total_usage += usage

        result = {
            "status": final_status,
            "method": "column_generation",
            "objective_value": total_usage,
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
