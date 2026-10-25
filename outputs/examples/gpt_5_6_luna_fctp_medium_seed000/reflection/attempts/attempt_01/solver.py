import json
import os
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB


def load_instance():
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data.json"))
    with open(data_path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    m = int(data["m"])
    n = int(data["n"])
    supply = np.asarray(data["supply"], dtype=float)
    demand = np.asarray(data["demand"], dtype=float)
    unit_cost = np.asarray(data["unit_cost"], dtype=float)
    fixed_cost = np.asarray(data["fixed_cost"], dtype=float)
    capacity = np.asarray(data["capacity"], dtype=float)
    if supply.shape != (m,) or demand.shape != (n,):
        raise ValueError("invalid supply or demand dimensions")
    for array in (unit_cost, fixed_cost, capacity):
        if array.shape != (m, n):
            raise ValueError("invalid matrix dimensions")
    return m, n, supply, demand, unit_cost, fixed_cost, capacity


def make_subproblem(m, n, supply, demand, unit_cost, capacity, y_value):
    model = gp.Model("transportation_subproblem")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.DualReductions = 0
    x = model.addVars(m, n, lb=0.0, name="x")
    demand_rows = [
        model.addConstr(gp.quicksum(x[i, j] for i in range(m)) == float(demand[j]), name="demand_%d" % j)
        for j in range(n)
    ]
    supply_rows = [
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) <= float(supply[i]), name="supply_%d" % i)
        for i in range(m)
    ]
    capacity_rows = [
        model.addConstr(x[i, j] <= float(capacity[i, j] * y_value[i][j]), name="capacity_%d_%d" % (i, j))
        for i in range(m) for j in range(n)
    ]
    model.setObjective(gp.quicksum(unit_cost[i, j] * x[i, j] for i in range(m) for j in range(n)), GRB.MINIMIZE)
    model.optimize()
    return model, x, demand_rows, supply_rows, capacity_rows


def add_nogood(master, y, y_value):
    expression = gp.LinExpr()
    for i in range(len(y_value)):
        for j in range(len(y_value[i])):
            if y_value[i][j] > 0.5:
                expression += 1.0 - y[i, j]
            else:
                expression += y[i, j]
    master.addConstr(expression >= 1.0, name="infeasible_pattern_%d" % master.NumConstrs)


def write_result(result):
    with open("output.json", "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


def solve():
    m, n, supply, demand, unit_cost, fixed_cost, capacity = load_instance()
    if np.any(supply < -1e-9) or np.any(demand < -1e-9):
        raise ValueError("negative supply or demand")
    if float(np.sum(supply)) + 1e-9 < float(np.sum(demand)):
        return {
            "status": "infeasible",
            "method": "benders_decomposition",
            "objective_value": None,
            "y": [[0 for _ in range(n)] for _ in range(m)],
            "x": [[0.0 for _ in range(n)] for _ in range(m)],
        }

    master = gp.Model("benders_master")
    master.Params.OutputFlag = 0
    master.Params.Threads = 1
    master.Params.MIPGap = 0.0
    y = master.addVars(m, n, vtype=GRB.BINARY, name="y")
    theta = master.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="theta")
    master.setObjective(
        gp.quicksum(float(fixed_cost[i, j]) * y[i, j] for i in range(m) for j in range(n)) + theta,
        GRB.MINIMIZE,
    )
    master.update()

    started = time.monotonic()
    time_limit = 54.0
    incumbent = None
    iterations = 0
    tolerance = 1e-6
    proven_optimal = False
    while time.monotonic() - started < time_limit:
        remaining = max(0.2, time_limit - (time.monotonic() - started))
        master.Params.TimeLimit = remaining
        master.optimize()
        if master.SolCount == 0:
            break
        y_value = [[1 if y[i, j].X >= 0.5 else 0 for j in range(n)] for i in range(m)]
        fixed_value = sum(float(fixed_cost[i, j]) * y_value[i][j] for i in range(m) for j in range(n))

        subproblem, x, demand_rows, supply_rows, capacity_rows = make_subproblem(
            m, n, supply, demand, unit_cost, capacity, y_value
        )
        if subproblem.Status == GRB.OPTIMAL:
            x_value = [[max(0.0, float(x[i, j].X)) for j in range(n)] for i in range(m)]
            recourse_value = float(subproblem.ObjVal)
            total_value = fixed_value + recourse_value
            if incumbent is None or total_value < incumbent["objective_value"] - tolerance:
                incumbent = {"objective_value": total_value, "y": y_value, "x": x_value}

            dual_constant = sum(float(demand[j]) * float(demand_rows[j].Pi) for j in range(n))
            dual_constant += sum(float(supply[i]) * float(supply_rows[i].Pi) for i in range(m))
            cut = gp.LinExpr()
            cut.addConstant(float(dual_constant))
            for i in range(m):
                for j in range(n):
                    cut += float(capacity[i, j] * capacity_rows[i * n + j].Pi) * y[i, j]
            master.addConstr(theta >= cut, name="optimality_cut_%d" % iterations)
            master.update()
            master_value = float(master.ObjVal)
            if abs(master_value - total_value) <= 5e-5:
                proven_optimal = True
                break
        elif subproblem.Status == GRB.INFEASIBLE:
            add_nogood(master, y, y_value)
            master.update()
        else:
            break
        iterations += 1
        subproblem.dispose()

    master.dispose()
    if incumbent is not None:
        status = "optimal" if proven_optimal else "feasible"
        return {
            "status": status,
            "method": "benders_decomposition",
            "objective_value": float(incumbent["objective_value"]),
            "y": incumbent["y"],
            "x": incumbent["x"],
        }
    return {
        "status": "infeasible",
        "method": "benders_decomposition",
        "objective_value": None,
        "y": [[0 for _ in range(n)] for _ in range(m)],
        "x": [[0.0 for _ in range(n)] for _ in range(m)],
    }


if __name__ == "__main__":
    try:
        write_result(solve())
    except Exception:
        write_result({
            "status": "error",
            "method": "benders_decomposition",
            "objective_value": None,
            "y": [],
            "x": [],
        })
