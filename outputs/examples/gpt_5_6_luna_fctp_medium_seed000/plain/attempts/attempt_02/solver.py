import json
import math
import os

import gurobipy as gp
import numpy as np


def load_instance():
    # The harness runs this file from a directory two levels below data.json.
    path = os.path.join("..", "..", "data.json")
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    m = int(data["m"])
    n = int(data["n"])
    supply = np.asarray(data["supply"], dtype=float)
    demand = np.asarray(data["demand"], dtype=float)
    unit = np.asarray(data["unit_cost"], dtype=float)
    fixed = np.asarray(data["fixed_cost"], dtype=float)
    capacity = np.asarray(data["capacity"], dtype=float)
    return m, n, supply, demand, unit, fixed, capacity


def solve_subproblem(y_value, supply, demand, unit, capacity):
    """Solve the recourse LP for one fixed binary activation pattern."""
    m, n = unit.shape
    model = gp.Model("benders_recourse")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    x = model.addMVar((m, n), lb=0.0, name="x")
    model.addConstr(x.sum(axis=1) <= supply, name="supply")
    model.addConstr(x.sum(axis=0) == demand, name="demand")
    model.addConstr(x <= capacity * y_value, name="lane_capacity")
    model.setObjective((unit * x).sum(), gp.GRB.MINIMIZE)
    model.optimize()
    if model.Status == gp.GRB.OPTIMAL:
        return True, float(model.ObjVal), np.asarray(x.X, dtype=float)
    return False, None, None


def add_no_good(model, y, pattern, name):
    # This excludes exactly one infeasible integer master assignment.
    expression = gp.LinExpr()
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            expression += (1.0 - float(pattern[i, j])) * y[i, j]
            expression += float(pattern[i, j]) * (1.0 - y[i, j])
    model.addConstr(expression >= 1.0, name=name)


def add_logic_optimality_cut(model, y, theta, pattern, recourse, name):
    # The RHS is recourse only at this exact pattern and is zero elsewhere.
    # Since all shipping costs are nonnegative, theta >= 0 makes this valid.
    distance = gp.LinExpr()
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            distance += (1.0 - float(pattern[i, j])) * y[i, j]
            distance += float(pattern[i, j]) * (1.0 - y[i, j])
    model.addConstr(theta >= float(recourse) * (1.0 - distance), name=name)


def main():
    m, n, supply, demand, unit, fixed, capacity = load_instance()
    master = gp.Model("benders_master")
    master.Params.OutputFlag = 0
    master.Params.Threads = 1
    y = master.addMVar((m, n), vtype=gp.GRB.BINARY, name="y")
    theta = master.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name="theta")

    # Necessary customer-wise capacity cuts provide useful initial structure.
    for j in range(n):
        coefficients = np.minimum(capacity[:, j], demand[j])
        master.addConstr((coefficients * y[:, j]).sum() >= demand[j], name="customer_capacity_%d" % j)
    master.setObjective((fixed * y).sum() + theta, gp.GRB.MINIMIZE)

    incumbent_obj = math.inf
    incumbent_y = None
    incumbent_x = None
    incumbent_recourse = None
    max_iterations = 10000
    tolerance = 1.0e-7
    proven_optimal = False

    for iteration in range(max_iterations):
        master.optimize()
        if master.Status == gp.GRB.INFEASIBLE:
            break
        if master.Status != gp.GRB.OPTIMAL:
            raise RuntimeError("master optimization did not reach optimality")

        pattern = np.rint(np.asarray(y.X, dtype=float)).astype(int)
        lower_bound = float(master.ObjVal)
        feasible, recourse, x_value = solve_subproblem(pattern, supply, demand, unit, capacity)

        if not feasible:
            add_no_good(master, y, pattern, "feasibility_%d" % iteration)
            continue

        total = float((fixed * pattern).sum() + recourse)
        if total < incumbent_obj - tolerance:
            incumbent_obj = total
            incumbent_y = pattern.copy()
            incumbent_x = x_value.copy()
            incumbent_recourse = recourse
        add_logic_optimality_cut(master, y, theta, pattern, recourse, "optimality_%d" % iteration)

        # The master is a valid lower-bounding relaxation. Once its bound
        # reaches the best decomposed incumbent, global optimality is proved.
        if incumbent_y is not None and lower_bound >= incumbent_obj - tolerance:
            proven_optimal = True
            break

    if incumbent_y is None:
        return {
            "status": "infeasible",
            "method": "benders_decomposition",
            "objective_value": None,
            "y": None,
            "x": None,
        }

    # Recompute from the retained subproblem solution for a consistent report.
    reported_obj = float((fixed * incumbent_y).sum() + incumbent_recourse)
    return {
        "status": "optimal" if proven_optimal else "feasible",
        "method": "benders_decomposition",
        "objective_value": reported_obj,
        "y": incumbent_y.astype(int).tolist(),
        "x": incumbent_x.tolist(),
    }


def write_error(exc):
    return {
        "status": "error",
        "method": "benders_decomposition",
        "objective_value": None,
        "y": None,
        "x": None,
        "error": str(exc),
    }


if __name__ == "__main__":
    try:
        result = main()
    except Exception as exc:
        result = write_error(exc)
    with open("output.json", "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
