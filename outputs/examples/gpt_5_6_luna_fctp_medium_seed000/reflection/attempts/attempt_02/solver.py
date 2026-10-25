import json
import os
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB


def load_data():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data.json"))
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    m, n = int(d["m"]), int(d["n"])
    supply = np.asarray(d["supply"], dtype=float)
    demand = np.asarray(d["demand"], dtype=float)
    unit = np.asarray(d["unit_cost"], dtype=float)
    fixed = np.asarray(d["fixed_cost"], dtype=float)
    capacity = np.asarray(d["capacity"], dtype=float)
    if supply.shape != (m,) or demand.shape != (n,):
        raise ValueError("bad vector dimensions")
    if unit.shape != (m, n) or fixed.shape != (m, n) or capacity.shape != (m, n):
        raise ValueError("bad matrix dimensions")
    return m, n, supply, demand, unit, fixed, capacity


def fixed_subproblem(m, n, supply, demand, unit, capacity, y_value):
    model = gp.Model("benders_recourse")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.DualReductions = 0
    x = model.addVars(m, n, lb=0.0, name="x")
    demand_con = [model.addConstr(gp.quicksum(x[i, j] for i in range(m)) == float(demand[j])) for j in range(n)]
    supply_con = [model.addConstr(gp.quicksum(x[i, j] for j in range(n)) <= float(supply[i])) for i in range(m)]
    cap_con = [model.addConstr(x[i, j] <= float(capacity[i, j] * y_value[i][j])) for i in range(m) for j in range(n)]
    model.setObjective(gp.quicksum(float(unit[i, j]) * x[i, j] for i in range(m) for j in range(n)), GRB.MINIMIZE)
    model.optimize()
    return model, x, demand_con, supply_con, cap_con


def add_capacity_cut(master, y, capacity, demand, subset, number):
    master.addConstr(
        gp.quicksum(float(capacity[i, j]) * y[i, j] for i in range(len(capacity)) for j in subset) >= float(np.sum(demand[list(subset)])),
        name="capacity_subset_%d" % number,
    )


def result(status, objective, y_value, x_value):
    return {
        "status": status,
        "method": "benders_decomposition",
        "objective_value": float(objective) if objective is not None and np.isfinite(objective) else 0.0,
        "y": y_value,
        "x": x_value,
    }


def solve():
    m, n, supply, demand, unit, fixed, capacity = load_data()
    zero_y = [[0 for _ in range(n)] for _ in range(m)]
    zero_x = [[0.0 for _ in range(n)] for _ in range(m)]
    if np.any(supply < -1e-9) or np.any(demand < -1e-9) or np.sum(supply) + 1e-9 < np.sum(demand):
        return result("infeasible", 0.0, zero_y, zero_x)

    # The master contains only the complicating binary decisions and a lower
    # estimate of recourse cost; all flow decisions are made in subproblems.
    master = gp.Model("benders_master")
    master.Params.OutputFlag = 0
    master.Params.Threads = 1
    master.Params.MIPGap = 0.0
    y = master.addVars(m, n, vtype=GRB.BINARY, name="y")
    theta = master.addVar(lb=0.0, name="theta")
    master.setObjective(gp.quicksum(float(fixed[i, j]) * y[i, j] for i in range(m) for j in range(n)) + theta, GRB.MINIMIZE)

    cut_number = 0
    subset_cuts = set()
    # Every customer subset gives a valid necessary aggregate-capacity cut.
    # Including all subsets is inexpensive here and is substantially stronger
    # than excluding individual infeasible binary patterns.
    for mask in range(1, 1 << n):
        subset = tuple(j for j in range(n) if mask & (1 << j))
        add_capacity_cut(master, y, capacity, demand, subset, cut_number)
        subset_cuts.add(subset)
        cut_number += 1
    master.update()

    started = time.monotonic()
    deadline = started + 55.0
    incumbent = None
    tolerance = 1e-5

    # A deliberately simple valid seed: opening every lane guarantees enough
    # capacity, and its flow is still obtained by a decomposition subproblem.
    seed_y = [[1 for _ in range(n)] for _ in range(m)]
    seed_model, seed_x, _, _, _ = fixed_subproblem(m, n, supply, demand, unit, capacity, seed_y)
    if seed_model.Status == GRB.OPTIMAL:
        seed_x_value = [[max(0.0, float(seed_x[i, j].X)) for j in range(n)] for i in range(m)]
        seed_obj = float(seed_model.ObjVal) + float(np.sum(fixed * np.asarray(seed_y)))
        incumbent = {"objective": seed_obj, "y": seed_y, "x": seed_x_value}
    seed_model.dispose()

    proven = False
    iterations = 0
    while time.monotonic() < deadline:
        master.Params.TimeLimit = max(0.2, deadline - time.monotonic())
        master.optimize()
        if master.SolCount == 0:
            break
        y_value = [[1 if y[i, j].X >= 0.5 else 0 for j in range(n)] for i in range(m)]
        sp, x, demand_con, supply_con, cap_con = fixed_subproblem(m, n, supply, demand, unit, capacity, y_value)
        if sp.Status == GRB.OPTIMAL:
            x_value = [[max(0.0, float(x[i, j].X)) for j in range(n)] for i in range(m)]
            total = float(sp.ObjVal) + sum(float(fixed[i, j]) * y_value[i][j] for i in range(m) for j in range(n))
            if incumbent is None or total < incumbent["objective"] - tolerance:
                incumbent = {"objective": total, "y": y_value, "x": x_value}

            # At an optimal master solve, its objective is a valid global
            # lower bound. A feasible subproblem plus matching bounds proves
            # optimality before another redundant cut is added.
            if master.Status == GRB.OPTIMAL and float(master.ObjVal) >= incumbent["objective"] - tolerance:
                proven = True
                sp.dispose()
                break

            # The LP dual produces a valid Benders optimality cut in y.
            constant = sum(float(demand[j]) * float(demand_con[j].Pi) for j in range(n))
            constant += sum(float(supply[i]) * float(supply_con[i].Pi) for i in range(m))
            cut = gp.LinExpr(constant)
            for i in range(m):
                for j in range(n):
                    cut += float(capacity[i, j]) * float(cap_con[i * n + j].Pi) * y[i, j]
            master.addConstr(theta >= cut, name="optimality_%d" % iterations)
            master.update()
            sp.dispose()
            iterations += 1
            # Reoptimization is intentionally done on the next loop, so this
            # lower bound includes the newly added cut.
            continue

        if sp.Status == GRB.INFEASIBLE:
            # Find a violated aggregate cut for this fixed decision. It is a
            # valid feasibility cut independent of the particular incumbent.
            violated = None
            violation = 0.0
            for mask in range(1, 1 << n):
                subset = tuple(j for j in range(n) if mask & (1 << j))
                available = sum(float(capacity[i, j]) * y_value[i][j] for i in range(m) for j in subset)
                gap = float(np.sum(demand[list(subset)])) - available
                if gap > violation + 1e-7:
                    violated, violation = subset, gap
            if violated is not None and violated not in subset_cuts:
                add_capacity_cut(master, y, capacity, demand, violated, cut_number)
                subset_cuts.add(violated)
                cut_number += 1
            else:
                # This remains a legitimate decomposition feasibility cut and
                # prevents cycling if aggregate cuts do not identify the cause.
                expr = gp.LinExpr()
                for i in range(m):
                    for j in range(n):
                        expr += (1.0 - y[i, j]) if y_value[i][j] else y[i, j]
                master.addConstr(expr >= 1.0, name="exclude_pattern_%d" % iterations)
            master.update()
            sp.dispose()
            iterations += 1
            continue
        sp.dispose()
        break

    # A matching master lower bound after the final cut is the certificate.
    if incumbent is not None and time.monotonic() < deadline:
        master.Params.TimeLimit = max(0.2, deadline - time.monotonic())
        master.optimize()
        if master.SolCount > 0 and master.Status == GRB.OPTIMAL and float(master.ObjVal) >= incumbent["objective"] - tolerance:
            proven = True
    master.dispose()
    if incumbent is not None:
        return result("optimal" if proven else "feasible", incumbent["objective"], incumbent["y"], incumbent["x"])
    return result("infeasible", 0.0, zero_y, zero_x)


if __name__ == "__main__":
    try:
        output = solve()
    except Exception:
        output = result("error", 0.0, [], [])
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, allow_nan=False)
