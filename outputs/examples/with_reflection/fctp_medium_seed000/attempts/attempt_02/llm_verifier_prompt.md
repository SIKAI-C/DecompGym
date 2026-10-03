Edit only alg_verifier.md. All task context is supplied below.

Do not inspect other directories, use external tools or services, or search for reference implementations.

Do not execute the generated solver. The harness executes it later in its documented runtime directory.

Keep prose paragraphs and list items on one physical line; rely on editor soft wrapping.

Inspect the supplied solver.py and determine whether it implements Benders decomposition. Check the master/subproblem split, dual or reduced-cost information, and iterative cut/column updates. A monolithic optimization model alone does not pass. Do not execute or modify solver.py. Write alg_verifier.md with exactly a first line VERDICT: PASS or VERDICT: FAIL, followed by NOTES: and concise evidence. Feasibility is checked separately.

--- BEGIN CONTEXT: attempt_02/solver.py ---
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
    with (base / "../../prompt.txt").open("r", encoding="utf-8"):
        pass
    return data


def quiet_model(name):
    model = gp.Model(name)
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    return model


def make_master(data):
    m, n = data["m"], data["n"]
    model = quiet_model("benders_master")
    y = [[model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}") for j in range(n)] for i in range(m)]
    theta = model.addVar(lb=0.0, name="theta")
    fixed = gp.quicksum(data["fixed_cost"][i][j] * y[i][j] for i in range(m) for j in range(n))
    model.setObjective(fixed + theta, GRB.MINIMIZE)
    model.update()
    return model, y, theta


def shortage_oracle(data, y_value):
    """Return the minimum unmet demand and a dual ray/certificate for its cut."""
    m, n = data["m"], data["n"]
    primal = quiet_model("benders_feasibility_primal")
    x = [[primal.addVar(lb=0.0, ub=data["capacity"][i][j] * y_value[i][j]) for j in range(n)] for i in range(m)]
    shortage = [primal.addVar(lb=0.0) for _ in range(n)]
    for j in range(n):
        primal.addConstr(gp.quicksum(x[i][j] for i in range(m)) + shortage[j] == data["demand"][j])
    for i in range(m):
        primal.addConstr(gp.quicksum(x[i][j] for j in range(n)) <= data["supply"][i])
    primal.setObjective(gp.quicksum(shortage), GRB.MINIMIZE)
    primal.optimize()
    if primal.Status != GRB.OPTIMAL:
        raise RuntimeError("feasibility oracle failed")
    value = primal.ObjVal

    dual = quiet_model("benders_feasibility_dual")
    pi = [dual.addVar(lb=-GRB.INFINITY, ub=1.0, name=f"pi_{j}") for j in range(n)]
    alpha = [dual.addVar(lb=-GRB.INFINITY, ub=0.0, name=f"alpha_{i}") for i in range(m)]
    beta = [[dual.addVar(lb=-GRB.INFINITY, ub=0.0, name=f"beta_{i}_{j}") for j in range(n)] for i in range(m)]
    for i in range(m):
        for j in range(n):
            dual.addConstr(pi[j] + alpha[i] + beta[i][j] <= 0.0)
    objective = gp.quicksum(data["demand"][j] * pi[j] for j in range(n))
    objective += gp.quicksum(data["supply"][i] * alpha[i] for i in range(m))
    objective += gp.quicksum(data["capacity"][i][j] * y_value[i][j] * beta[i][j] for i in range(m) for j in range(n))
    dual.setObjective(objective, GRB.MAXIMIZE)
    dual.optimize()
    if dual.Status != GRB.OPTIMAL:
        raise RuntimeError("feasibility dual failed")
    return value, pi, alpha, beta


def cost_oracle(data, y_value):
    m, n = data["m"], data["n"]
    primal = quiet_model("benders_cost_primal")
    x = [[primal.addVar(lb=0.0, ub=data["capacity"][i][j] * y_value[i][j]) for j in range(n)] for i in range(m)]
    for j in range(n):
        primal.addConstr(gp.quicksum(x[i][j] for i in range(m)) == data["demand"][j])
    for i in range(m):
        primal.addConstr(gp.quicksum(x[i][j] for j in range(n)) <= data["supply"][i])
    primal.setObjective(gp.quicksum(data["unit_cost"][i][j] * x[i][j] for i in range(m) for j in range(n)), GRB.MINIMIZE)
    primal.optimize()
    if primal.Status != GRB.OPTIMAL:
        raise RuntimeError("cost oracle failed")
    x_value = [[x[i][j].X for j in range(n)] for i in range(m)]

    dual = quiet_model("benders_optimality_dual")
    pi = [dual.addVar(lb=-GRB.INFINITY, name=f"pi_{j}") for j in range(n)]
    alpha = [dual.addVar(lb=-GRB.INFINITY, ub=0.0, name=f"alpha_{i}") for i in range(m)]
    beta = [[dual.addVar(lb=-GRB.INFINITY, ub=0.0, name=f"beta_{i}_{j}") for j in range(n)] for i in range(m)]
    for i in range(m):
        for j in range(n):
            dual.addConstr(pi[j] + alpha[i] + beta[i][j] <= data["unit_cost"][i][j])
    objective = gp.quicksum(data["demand"][j] * pi[j] for j in range(n))
    objective += gp.quicksum(data["supply"][i] * alpha[i] for i in range(m))
    objective += gp.quicksum(data["capacity"][i][j] * y_value[i][j] * beta[i][j] for i in range(m) for j in range(n))
    dual.setObjective(objective, GRB.MAXIMIZE)
    dual.optimize()
    if dual.Status != GRB.OPTIMAL:
        raise RuntimeError("optimality dual failed")
    return dual.ObjVal, x_value, pi, alpha, beta


def cut_expression(data, pi, alpha, beta, y_vars):
    m, n = data["m"], data["n"]
    expression = gp.quicksum(data["demand"][j] * pi[j].X for j in range(n))
    expression += gp.quicksum(data["supply"][i] * alpha[i].X for i in range(m))
    expression += gp.quicksum(data["capacity"][i][j] * beta[i][j].X * y_vars[i][j] for i in range(m) for j in range(n))
    return expression


def write_output(payload):
    with Path("output.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    data = None
    try:
        data = load_instance()
        m, n = data["m"], data["n"]
        master, y_vars, theta = make_master(data)
        best_value = math.inf
        best_y = None
        best_x = None
        for _ in range(500):
            master.optimize()
            if master.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                break
            y_value = [[1.0 if y_vars[i][j].X >= 0.5 else 0.0 for j in range(n)] for i in range(m)]
            shortage, pi, alpha, beta = shortage_oracle(data, y_value)
            if shortage > TOL:
                master.addConstr(cut_expression(data, pi, alpha, beta, y_vars) <= 0.0)
                continue
            cost, x_value, pi, alpha, beta = cost_oracle(data, y_value)
            fixed = sum(data["fixed_cost"][i][j] * y_value[i][j] for i in range(m) for j in range(n))
            candidate = fixed + cost
            if candidate < best_value:
                best_value, best_y, best_x = candidate, y_value, x_value
            master.addConstr(theta >= cut_expression(data, pi, alpha, beta, y_vars))
            master.optimize()
            if best_y is not None and master.Status == GRB.OPTIMAL and master.ObjBound >= best_value - 1.0e-6:
                break
        if best_y is not None:
            write_output({"status": "optimal" if master.Status == GRB.OPTIMAL and master.ObjBound >= best_value - 1.0e-6 else "feasible", "method": "benders_decomposition", "objective_value": float(best_value), "y": best_y, "x": best_x})
        else:
            write_output({"status": "infeasible", "method": "benders_decomposition", "objective_value": None, "y": [[0.0] * n for _ in range(m)], "x": [[0.0] * n for _ in range(m)]})
    except Exception:
        m = data["m"] if data else 0
        n = data["n"] if data else 0
        write_output({"status": "error", "method": "benders_decomposition", "objective_value": None, "y": [[0.0] * n for _ in range(m)], "x": [[0.0] * n for _ in range(m)]})


if __name__ == "__main__":
    main()

--- END CONTEXT: attempt_02/solver.py ---