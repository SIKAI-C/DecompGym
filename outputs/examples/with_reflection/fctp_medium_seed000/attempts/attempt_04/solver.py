import json
import math
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB


TOL = 1.0e-7
TIME_LIMIT = 50.0


def quiet_model(name):
    model = gp.Model(name)
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = 10.0
    return model


def load_data():
    base = Path(__file__).resolve().parent
    with (base / "../../data.json").open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    with (base / "../../prompt.txt").open("r", encoding="utf-8"):
        pass
    return data


def transportation(data, y_value):
    m, n = data["m"], data["n"]
    model = quiet_model("benders_subproblem")
    x = [[model.addVar(lb=0.0, ub=data["capacity"][i][j] * y_value[i][j], name=f"x_{i}_{j}") for j in range(n)] for i in range(m)]
    for j in range(n):
        model.addConstr(gp.quicksum(x[i][j] for i in range(m)) == data["demand"][j])
    for i in range(m):
        model.addConstr(gp.quicksum(x[i][j] for j in range(n)) <= data["supply"][i])
    model.setObjective(gp.quicksum(data["unit_cost"][i][j] * x[i][j] for i in range(m) for j in range(n)), GRB.MINIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        return None
    return model.ObjVal, [[max(0.0, x[i][j].X) for j in range(n)] for i in range(m)]


def dual_cut(data, y_value):
    """Return a dual optimality cut represented as constant plus coefficients times y."""
    m, n = data["m"], data["n"]
    model = quiet_model("benders_dual")
    pi = [model.addVar(lb=-GRB.INFINITY, name=f"pi_{j}") for j in range(n)]
    alpha = [model.addVar(lb=-GRB.INFINITY, ub=0.0, name=f"alpha_{i}") for i in range(m)]
    beta = [[model.addVar(lb=-GRB.INFINITY, ub=0.0, name=f"beta_{i}_{j}") for j in range(n)] for i in range(m)]
    for i in range(m):
        for j in range(n):
            model.addConstr(pi[j] + alpha[i] + beta[i][j] <= data["unit_cost"][i][j])
    objective = gp.quicksum(data["demand"][j] * pi[j] for j in range(n))
    objective += gp.quicksum(data["supply"][i] * alpha[i] for i in range(m))
    objective += gp.quicksum(data["capacity"][i][j] * y_value[i][j] * beta[i][j] for i in range(m) for j in range(n))
    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        return None
    constant = sum(data["demand"][j] * pi[j].X for j in range(n)) + sum(data["supply"][i] * alpha[i].X for i in range(m))
    coefficients = [[data["capacity"][i][j] * beta[i][j].X for j in range(n)] for i in range(m)]
    return constant, coefficients, model.ObjVal


def valid_solution(data, y_value, x_value):
    if y_value is None or x_value is None:
        return False
    m, n = data["m"], data["n"]
    for i in range(m):
        if sum(x_value[i][j] for j in range(n)) > data["supply"][i] + 1.0e-5:
            return False
        for j in range(n):
            if y_value[i][j] not in (0.0, 1.0) or x_value[i][j] < -1.0e-6 or x_value[i][j] > data["capacity"][i][j] * y_value[i][j] + 1.0e-5:
                return False
    return all(abs(sum(x_value[i][j] for i in range(m)) - data["demand"][j]) <= 1.0e-5 for j in range(n))


def output(status, objective, y_value, x_value):
    with Path("output.json").open("w", encoding="utf-8") as handle:
        json.dump({"status": status, "method": "benders_decomposition", "objective_value": objective, "y": y_value, "x": x_value}, handle, indent=2)


def main():
    data = None
    try:
        data = load_data()
        m, n = data["m"], data["n"]
        zero = [[0.0 for _ in range(n)] for _ in range(m)]
        all_open = [[1.0 for _ in range(n)] for _ in range(m)]
        seed = transportation(data, all_open)
        if seed is None:
            output("infeasible", None, zero, zero)
            return
        seed_cost, seed_x = seed
        best_value = sum(data["fixed_cost"][i][j] for i in range(m) for j in range(n)) + seed_cost
        best_y, best_x = all_open, seed_x

        master = quiet_model("benders_master")
        y = [[master.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}") for j in range(n)] for i in range(m)]
        theta = master.addVar(lb=0.0, name="theta")
        fixed = gp.quicksum(data["fixed_cost"][i][j] * y[i][j] for i in range(m) for j in range(n))
        master.setObjective(fixed + theta, GRB.MINIMIZE)
        started = time.monotonic()
        proven = False
        for _ in range(250):
            if time.monotonic() - started >= TIME_LIMIT:
                break
            master.optimize()
            if master.Status != GRB.OPTIMAL:
                break
            y_value = [[1.0 if y[i][j].X > 0.5 else 0.0 for j in range(n)] for i in range(m)]
            sub = transportation(data, y_value)
            if sub is None:
                # A no-good cut is a valid Benders feasibility cut for this binary pattern.
                master.addConstr(gp.quicksum((1.0 - y[i][j]) if y_value[i][j] > 0.5 else y[i][j] for i in range(m) for j in range(n)) >= 1.0)
                continue
            sub_cost, x_value = sub
            candidate = sum(data["fixed_cost"][i][j] * y_value[i][j] for i in range(m) for j in range(n)) + sub_cost
            if valid_solution(data, y_value, x_value) and candidate < best_value - TOL:
                best_value, best_y, best_x = candidate, y_value, x_value
            cut = dual_cut(data, y_value)
            if cut is None:
                break
            constant, coefficients, _ = cut
            master.addConstr(theta >= constant + gp.quicksum(coefficients[i][j] * y[i][j] for i in range(m) for j in range(n)))
            if master.ObjBound >= best_value - 1.0e-6:
                proven = True
                break
        if valid_solution(data, best_y, best_x):
            output("optimal" if proven else "feasible", float(best_value), best_y, best_x)
        else:
            output("error", None, zero, zero)
    except Exception:
        m = data["m"] if data is not None else 0
        n = data["n"] if data is not None else 0
        output("error", None, [[0.0] * n for _ in range(m)], [[0.0] * n for _ in range(m)])


if __name__ == "__main__":
    main()
