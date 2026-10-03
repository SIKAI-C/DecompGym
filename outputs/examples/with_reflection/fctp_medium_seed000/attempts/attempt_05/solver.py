import json
import math
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB


EPS = 1.0e-7
DEADLINE = 52.0


def quiet_model(name, remaining=None):
    model = gp.Model(name)
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    if remaining is not None:
        model.Params.TimeLimit = max(0.05, remaining)
    return model


def load_data():
    base = Path(__file__).resolve().parent
    with (base / "../../data.json").open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    with (base / "../../prompt.txt").open("r", encoding="utf-8"):
        pass
    return data


def transportation(data, y_value, remaining=None):
    m, n = data["m"], data["n"]
    model = quiet_model("benders_subproblem", remaining)
    x = [[model.addVar(lb=0.0, ub=data["capacity"][i][j] * y_value[i][j]) for j in range(n)] for i in range(m)]
    for j in range(n):
        model.addConstr(gp.quicksum(x[i][j] for i in range(m)) == data["demand"][j])
    for i in range(m):
        model.addConstr(gp.quicksum(x[i][j] for j in range(n)) <= data["supply"][i])
    model.setObjective(gp.quicksum(data["unit_cost"][i][j] * x[i][j] for i in range(m) for j in range(n)), GRB.MINIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        return None
    return float(model.ObjVal), [[max(0.0, x[i][j].X) for j in range(n)] for i in range(m)]


def dual_cut(data, y_value, feasibility, remaining=None):
    m, n = data["m"], data["n"]
    model = quiet_model("benders_dual", remaining)
    pi = [model.addVar(lb=-GRB.INFINITY, ub=1.0 if feasibility else GRB.INFINITY) for _ in range(n)]
    alpha = [model.addVar(lb=-GRB.INFINITY, ub=0.0) for _ in range(m)]
    beta = [[model.addVar(lb=-GRB.INFINITY, ub=0.0) for _ in range(n)] for _ in range(m)]
    rhs = 0.0 if feasibility else None
    for i in range(m):
        for j in range(n):
            bound = 0.0 if feasibility else data["unit_cost"][i][j]
            model.addConstr(pi[j] + alpha[i] + beta[i][j] <= bound)
    objective = gp.quicksum(data["demand"][j] * pi[j] for j in range(n))
    objective += gp.quicksum(data["supply"][i] * alpha[i] for i in range(m))
    objective += gp.quicksum(data["capacity"][i][j] * y_value[i][j] * beta[i][j] for i in range(m) for j in range(n))
    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        return None
    constant = sum(data["demand"][j] * pi[j].X for j in range(n)) + sum(data["supply"][i] * alpha[i].X for i in range(m))
    coefficients = [[data["capacity"][i][j] * beta[i][j].X for j in range(n)] for i in range(m)]
    value = constant + sum(coefficients[i][j] * y_value[i][j] for i in range(m) for j in range(n))
    return constant, coefficients, value


def valid(data, y_value, x_value):
    if y_value is None or x_value is None:
        return False
    m, n = data["m"], data["n"]
    for i in range(m):
        if sum(x_value[i]) > data["supply"][i] + 1.0e-5:
            return False
        for j in range(n):
            if y_value[i][j] not in (0.0, 1.0) or x_value[i][j] < -1.0e-6 or x_value[i][j] > data["capacity"][i][j] * y_value[i][j] + 1.0e-5:
                return False
    return all(abs(sum(x_value[i][j] for i in range(m)) - data["demand"][j]) <= 1.0e-5 for j in range(n))


def write_result(status, objective, y_value, x_value):
    with Path("output.json").open("w", encoding="utf-8") as handle:
        json.dump({"status": status, "method": "benders_decomposition", "objective_value": objective, "y": y_value, "x": x_value}, handle, indent=2)


def main():
    data = None
    best_y = best_x = None
    best_value = math.inf
    m = n = 0
    try:
        data = load_data()
        m, n = data["m"], data["n"]
        started = time.monotonic()
        all_open = [[1.0 for _ in range(n)] for _ in range(m)]
        seed = transportation(data, all_open, DEADLINE)
        if seed is None:
            write_result("infeasible", None, [[0.0] * n for _ in range(m)], [[0.0] * n for _ in range(m)])
            return

        def consider(y_value, result):
            nonlocal best_y, best_x, best_value
            if result is None:
                return
            cost, x_value = result
            fixed = sum(data["fixed_cost"][i][j] * y_value[i][j] for i in range(m) for j in range(n))
            if valid(data, y_value, x_value) and fixed + cost < best_value:
                best_value = fixed + cost
                best_y = [row[:] for row in y_value]
                best_x = [row[:] for row in x_value]

        consider(all_open, seed)
        # A short greedy closing pass gives Benders a useful sparse incumbent immediately.
        lanes = sorted(((data["fixed_cost"][i][j], i, j) for i in range(m) for j in range(n)), reverse=True)
        incumbent_y = [row[:] for row in all_open]
        for _, i, j in lanes:
            if time.monotonic() - started >= 8.0:
                break
            trial = [row[:] for row in incumbent_y]
            trial[i][j] = 0.0
            result = transportation(data, trial, max(0.1, 8.0 - (time.monotonic() - started)))
            if result is not None:
                incumbent_y = trial
                consider(trial, result)

        master = quiet_model("benders_master", DEADLINE)
        y = [[master.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}") for j in range(n)] for i in range(m)]
        theta = master.addVar(lb=0.0, name="theta")
        master.setObjective(gp.quicksum(data["fixed_cost"][i][j] * y[i][j] for i in range(m) for j in range(n)) + theta, GRB.MINIMIZE)
        # These necessary cuts eliminate the most damaging low-capacity master points.
        for j in range(n):
            master.addConstr(gp.quicksum(data["capacity"][i][j] * y[i][j] for i in range(m)) >= data["demand"][j])
        master.update()
        proven = False
        for _ in range(300):
            remaining = DEADLINE - (time.monotonic() - started)
            if remaining <= 0.5:
                break
            master.Params.TimeLimit = remaining
            master.optimize()
            if master.Status != GRB.OPTIMAL:
                break
            y_value = [[1.0 if y[i][j].X > 0.5 else 0.0 for j in range(n)] for i in range(m)]
            result = transportation(data, y_value, max(0.1, remaining - 0.1))
            if result is None:
                cut = dual_cut(data, y_value, True, max(0.1, remaining - 0.2))
                if cut is None:
                    # This is weaker than a ray cut but remains valid for binary master variables.
                    master.addConstr(gp.quicksum((1.0 - y[i][j]) if y_value[i][j] else y[i][j] for i in range(m) for j in range(n)) >= 1.0)
                else:
                    constant, coefficients, value = cut
                    if value > EPS:
                        master.addConstr(constant + gp.quicksum(coefficients[i][j] * y[i][j] for i in range(m) for j in range(n)) <= 0.0)
                    else:
                        master.addConstr(gp.quicksum((1.0 - y[i][j]) if y_value[i][j] else y[i][j] for i in range(m) for j in range(n)) >= 1.0)
                continue
            consider(y_value, result)
            cut = dual_cut(data, y_value, False, max(0.1, remaining - 0.2))
            if cut is None:
                break
            constant, coefficients, _ = cut
            master.addConstr(theta >= constant + gp.quicksum(coefficients[i][j] * y[i][j] for i in range(m) for j in range(n)))
            master.optimize()
            if master.Status == GRB.OPTIMAL and best_y is not None and master.ObjBound >= best_value - 1.0e-6:
                proven = True
                break

        if best_y is not None and valid(data, best_y, best_x):
            write_result("optimal" if proven else "feasible", float(best_value), best_y, best_x)
        else:
            write_result("error", None, [[0.0] * n for _ in range(m)], [[0.0] * n for _ in range(m)])
    except Exception:
        if best_y is not None and valid(data, best_y, best_x):
            write_result("feasible", float(best_value), best_y, best_x)
        else:
            write_result("error", None, [[0.0] * n for _ in range(m)], [[0.0] * n for _ in range(m)])


if __name__ == "__main__":
    main()
