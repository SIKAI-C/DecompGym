import json
import math
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB


TOL = 1.0e-7


def load_instance():
    base = Path(__file__).resolve().parent
    with (base / "../../data.json").open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    # The prompt is part of the documented input layout; read it without depending on its contents.
    with (base / "../../prompt.txt").open("r", encoding="utf-8"):
        pass
    return data


def make_master(data):
    m, n = data["m"], data["n"]
    model = gp.Model("benders_master")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    y = [[model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}") for j in range(n)] for i in range(m)]
    theta = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="theta")
    model.setObjective(
        gp.quicksum(data["fixed_cost"][i][j] * y[i][j] for i in range(m) for j in range(n)) + theta,
        GRB.MINIMIZE,
    )
    model.update()
    return model, y, theta


def dual_model(data, y_value, shortage):
    m, n = data["m"], data["n"]
    model = gp.Model("benders_dual")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    pi = [model.addVar(lb=-GRB.INFINITY, name=f"pi_{j}") for j in range(n)]
    alpha = [model.addVar(ub=0.0, name=f"alpha_{i}") for i in range(m)]
    beta = [[model.addVar(ub=0.0, name=f"beta_{i}_{j}") for j in range(n)] for i in range(m)]
    for i in range(m):
        for j in range(n):
            model.addConstr(pi[j] + alpha[i] + beta[i][j] <= data["unit_cost"][i][j])
    for j in range(n):
        if shortage:
            model.addConstr(pi[j] <= 1.0)
    expression = gp.quicksum(data["demand"][j] * pi[j] for j in range(n))
    expression += gp.quicksum(data["supply"][i] * alpha[i] for i in range(m))
    expression += gp.quicksum(
        data["capacity"][i][j] * y_value[i][j] * beta[i][j] for i in range(m) for j in range(n)
    )
    model.setObjective(expression, GRB.MAXIMIZE)
    model.update()
    return model, pi, alpha, beta


def dual_expression(data, pi, alpha, beta, y_vars):
    m, n = data["m"], data["n"]
    expression = gp.LinExpr()
    expression += gp.quicksum(data["demand"][j] * pi[j].X for j in range(n))
    expression += gp.quicksum(data["supply"][i] * alpha[i].X for i in range(m))
    expression += gp.quicksum(
        data["capacity"][i][j] * beta[i][j].X * y_vars[i][j] for i in range(m) for j in range(n)
    )
    return expression


def solve_primal(data, y_value):
    m, n = data["m"], data["n"]
    model = gp.Model("benders_primal")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    x = [[model.addVar(lb=0.0, ub=data["capacity"][i][j] * y_value[i][j], name=f"x_{i}_{j}") for j in range(n)] for i in range(m)]
    for j in range(n):
        model.addConstr(gp.quicksum(x[i][j] for i in range(m)) == data["demand"][j])
    for i in range(m):
        model.addConstr(gp.quicksum(x[i][j] for j in range(n)) <= data["supply"][i])
    model.setObjective(gp.quicksum(data["unit_cost"][i][j] * x[i][j] for i in range(m) for j in range(n)), GRB.MINIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        return None
    return [[x[i][j].X for j in range(n)] for i in range(m)]


def write_output(payload):
    with Path("output.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    try:
        data = load_instance()
        m, n = data["m"], data["n"]
        master, y_vars, theta = make_master(data)
        last_y = [[0.0 for _ in range(n)] for _ in range(m)]
        last_x = [[0.0 for _ in range(n)] for _ in range(m)]
        incumbent = math.inf
        max_iterations = 300

        for _ in range(max_iterations):
            master.optimize()
            if master.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                status = "infeasible" if master.Status == GRB.INFEASIBLE else "error"
                write_output({"status": status, "method": "benders_decomposition", "objective_value": None, "y": last_y, "x": last_x})
                return
            y_value = [[1.0 if y_vars[i][j].X > 0.5 else 0.0 for j in range(n)] for i in range(m)]
            last_y = y_value

            feasibility_dual, pi, alpha, beta = dual_model(data, y_value, shortage=True)
            feasibility_dual.optimize()
            if feasibility_dual.Status != GRB.OPTIMAL:
                raise RuntimeError("feasibility dual did not solve to optimality")
            shortage_bound = feasibility_dual.ObjVal
            if shortage_bound > TOL:
                master.addConstr(dual_expression(data, pi, alpha, beta, y_vars) <= 0.0)
                continue

            last_x = solve_primal(data, y_value)
            if last_x is None:
                raise RuntimeError("primal subproblem failed after feasibility check")
            cost_dual, pi, alpha, beta = dual_model(data, y_value, shortage=False)
            cost_dual.optimize()
            if cost_dual.Status != GRB.OPTIMAL:
                raise RuntimeError("optimality dual did not solve to optimality")
            subproblem_cost = cost_dual.ObjVal
            master.addConstr(theta >= dual_expression(data, pi, alpha, beta, y_vars))
            candidate = sum(data["fixed_cost"][i][j] * y_value[i][j] for i in range(m) for j in range(n)) + subproblem_cost
            incumbent = min(incumbent, candidate)
            master.optimize()
            if master.Status == GRB.OPTIMAL and abs(master.ObjVal - incumbent) <= 1.0e-6:
                write_output({"status": "optimal", "method": "benders_decomposition", "objective_value": float(incumbent), "y": last_y, "x": last_x})
                return

        objective = None if not math.isfinite(incumbent) else float(incumbent)
        write_output({"status": "feasible" if objective is not None else "infeasible", "method": "benders_decomposition", "objective_value": objective, "y": last_y, "x": last_x})
    except Exception:
        write_output({"status": "error", "method": "benders_decomposition", "objective_value": None, "y": [], "x": []})


if __name__ == "__main__":
    main()
