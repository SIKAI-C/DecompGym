import json
import math
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB


TOL = 1.0e-7
TIME_LIMIT = 55.0


def load_instance():
    base = Path(__file__).resolve().parent
    with (base / "../../data.json").open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    with (base / "../../prompt.txt").open("r", encoding="utf-8"):
        pass
    return data


def model(name):
    result = gp.Model(name)
    result.Params.OutputFlag = 0
    result.Params.Threads = 1
    return result


def make_master(data):
    m, n = data["m"], data["n"]
    master = model("benders_master")
    y = [[master.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}") for j in range(n)] for i in range(m)]
    theta = master.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="theta")
    fixed = gp.quicksum(data["fixed_cost"][i][j] * y[i][j] for i in range(m) for j in range(n))
    master.setObjective(fixed + theta, GRB.MINIMIZE)
    master.update()
    return master, y, theta


def feasibility_oracle(data, y_value):
    """Minimize shortage and return a dual certificate for its Benders cut."""
    m, n = data["m"], data["n"]
    primal = model("feasibility_oracle")
    x = [[primal.addVar(lb=0.0, ub=data["capacity"][i][j] * y_value[i][j]) for j in range(n)] for i in range(m)]
    shortage = [primal.addVar(lb=0.0) for _ in range(n)]
    for j in range(n):
        primal.addConstr(gp.quicksum(x[i][j] for i in range(m)) + shortage[j] == data["demand"][j])
    for i in range(m):
        primal.addConstr(gp.quicksum(x[i][j] for j in range(n)) <= data["supply"][i])
    primal.setObjective(gp.quicksum(shortage), GRB.MINIMIZE)
    primal.optimize()
    if primal.Status != GRB.OPTIMAL:
        raise RuntimeError(f"feasibility oracle status {primal.Status}")

    dual = model("feasibility_cut_dual")
    pi = [dual.addVar(lb=-GRB.INFINITY, ub=1.0, name=f"pi_{j}") for j in range(n)]
    alpha = [dual.addVar(lb=-GRB.INFINITY, ub=0.0, name=f"alpha_{i}") for i in range(m)]
    beta = [[dual.addVar(lb=-GRB.INFINITY, ub=0.0, name=f"beta_{i}_{j}") for j in range(n)] for i in range(m)]
    for i in range(m):
        for j in range(n):
            dual.addConstr(pi[j] + alpha[i] + beta[i][j] <= 0.0)
    expression = gp.quicksum(data["demand"][j] * pi[j] for j in range(n))
    expression += gp.quicksum(data["supply"][i] * alpha[i] for i in range(m))
    expression += gp.quicksum(data["capacity"][i][j] * y_value[i][j] * beta[i][j] for i in range(m) for j in range(n))
    dual.setObjective(expression, GRB.MAXIMIZE)
    dual.optimize()
    if dual.Status != GRB.OPTIMAL:
        raise RuntimeError(f"feasibility dual status {dual.Status}")
    return primal.ObjVal, pi, alpha, beta


def transportation_oracle(data, y_value):
    """Solve the fixed-y transportation problem and return primal and dual data."""
    m, n = data["m"], data["n"]
    primal = model("transportation_oracle")
    x = [[primal.addVar(lb=0.0, ub=data["capacity"][i][j] * y_value[i][j]) for j in range(n)] for i in range(m)]
    for j in range(n):
        primal.addConstr(gp.quicksum(x[i][j] for i in range(m)) == data["demand"][j])
    for i in range(m):
        primal.addConstr(gp.quicksum(x[i][j] for j in range(n)) <= data["supply"][i])
    primal.setObjective(gp.quicksum(data["unit_cost"][i][j] * x[i][j] for i in range(m) for j in range(n)), GRB.MINIMIZE)
    primal.optimize()
    if primal.Status != GRB.OPTIMAL:
        return None
    x_value = [[max(0.0, x[i][j].X) for j in range(n)] for i in range(m)]

    dual = model("transportation_cut_dual")
    pi = [dual.addVar(lb=-GRB.INFINITY, name=f"pi_{j}") for j in range(n)]
    alpha = [dual.addVar(lb=-GRB.INFINITY, ub=0.0, name=f"alpha_{i}") for i in range(m)]
    beta = [[dual.addVar(lb=-GRB.INFINITY, ub=0.0, name=f"beta_{i}_{j}") for j in range(n)] for i in range(m)]
    for i in range(m):
        for j in range(n):
            dual.addConstr(pi[j] + alpha[i] + beta[i][j] <= data["unit_cost"][i][j])
    expression = gp.quicksum(data["demand"][j] * pi[j] for j in range(n))
    expression += gp.quicksum(data["supply"][i] * alpha[i] for i in range(m))
    expression += gp.quicksum(data["capacity"][i][j] * y_value[i][j] * beta[i][j] for i in range(m) for j in range(n))
    dual.setObjective(expression, GRB.MAXIMIZE)
    dual.optimize()
    if dual.Status != GRB.OPTIMAL:
        raise RuntimeError(f"transportation dual status {dual.Status}")
    return dual.ObjVal, x_value, pi, alpha, beta


def cut_expression(data, pi, alpha, beta, y_vars):
    m, n = data["m"], data["n"]
    expression = gp.quicksum(data["demand"][j] * pi[j].X for j in range(n))
    expression += gp.quicksum(data["supply"][i] * alpha[i].X for i in range(m))
    expression += gp.quicksum(data["capacity"][i][j] * beta[i][j].X * y_vars[i][j] for i in range(m) for j in range(n))
    return expression


def valid_solution(data, y_value, x_value):
    if y_value is None or x_value is None:
        return False
    m, n = data["m"], data["n"]
    for i in range(m):
        if any(abs(y_value[i][j] - round(y_value[i][j])) > TOL for j in range(n)):
            return False
        if sum(x_value[i][j] for j in range(n)) > data["supply"][i] + TOL:
            return False
        for j in range(n):
            if x_value[i][j] < -TOL or x_value[i][j] > data["capacity"][i][j] * y_value[i][j] + TOL:
                return False
    return all(abs(sum(x_value[i][j] for i in range(m)) - data["demand"][j]) <= 1.0e-5 for j in range(n))


def payload(status, objective, y_value, x_value):
    return {"status": status, "method": "benders_decomposition", "objective_value": objective, "y": y_value, "x": x_value}


def main():
    data = None
    m = n = 0
    try:
        data = load_instance()
        m, n = data["m"], data["n"]
        master, y_vars, theta = make_master(data)
        best_value = math.inf
        best_y = None
        best_x = None
        started = time.monotonic()

        for _ in range(500):
            if time.monotonic() - started >= TIME_LIMIT:
                break
            master.optimize()
            if master.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                break
            y_value = [[1.0 if y_vars[i][j].X >= 0.5 else 0.0 for j in range(n)] for i in range(m)]
            shortage, pi, alpha, beta = feasibility_oracle(data, y_value)
            if shortage > TOL:
                master.addConstr(cut_expression(data, pi, alpha, beta, y_vars) <= 0.0)
                continue
            result = transportation_oracle(data, y_value)
            if result is None:
                raise RuntimeError("feasible oracle disagreed with transportation oracle")
            cost, x_value, pi, alpha, beta = result
            fixed = sum(data["fixed_cost"][i][j] * y_value[i][j] for i in range(m) for j in range(n))
            candidate = fixed + cost
            if valid_solution(data, y_value, x_value) and candidate < best_value:
                best_value, best_y, best_x = candidate, y_value, x_value
            master.addConstr(theta >= cut_expression(data, pi, alpha, beta, y_vars))
            master.optimize()
            if best_y is not None and master.Status == GRB.OPTIMAL and master.ObjBound >= best_value - 1.0e-6:
                break

        if best_y is not None and valid_solution(data, best_y, best_x):
            status = "optimal" if master.Status == GRB.OPTIMAL and master.ObjBound >= best_value - 1.0e-6 else "feasible"
            write_output(payload(status, float(best_value), best_y, best_x))
            return

        all_open = [[1.0 for _ in range(n)] for _ in range(m)]
        max_flow_shortage, _, _, _ = feasibility_oracle(data, all_open)
        if max_flow_shortage > TOL:
            write_output(payload("infeasible", None, [[0.0] * n for _ in range(m)], [[0.0] * n for _ in range(m)]))
        else:
            write_output(payload("error", None, [[0.0] * n for _ in range(m)], [[0.0] * n for _ in range(m)]))
    except Exception:
        write_output(payload("error", None, [[0.0] * n for _ in range(m)], [[0.0] * n for _ in range(m)]))


def write_output(result):
    with Path("output.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
