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
