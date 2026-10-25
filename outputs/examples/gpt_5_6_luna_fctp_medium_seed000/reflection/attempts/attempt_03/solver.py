import json
import os
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB


def load_data():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data.json"))
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    m, n = int(data["m"]), int(data["n"])
    supply = np.asarray(data["supply"], dtype=float)
    demand = np.asarray(data["demand"], dtype=float)
    unit = np.asarray(data["unit_cost"], dtype=float)
    fixed = np.asarray(data["fixed_cost"], dtype=float)
    capacity = np.asarray(data["capacity"], dtype=float)
    if supply.shape != (m,) or demand.shape != (n,) or any(a.shape != (m, n) for a in (unit, fixed, capacity)):
        raise ValueError("bad data dimensions")
    return m, n, supply, demand, unit, fixed, capacity


def recourse(m, n, supply, demand, unit, capacity, y_value):
    model = gp.Model("fixed_y_transportation")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.DualReductions = 0
    x = model.addVars(m, n, lb=0.0, name="x")
    demand_rows = [model.addConstr(gp.quicksum(x[i, j] for i in range(m)) == float(demand[j])) for j in range(n)]
    supply_rows = [model.addConstr(gp.quicksum(x[i, j] for j in range(n)) <= float(supply[i])) for i in range(m)]
    capacity_rows = [model.addConstr(x[i, j] <= float(capacity[i, j] * y_value[i][j])) for i in range(m) for j in range(n)]
    model.setObjective(gp.quicksum(float(unit[i, j]) * x[i, j] for i in range(m) for j in range(n)), GRB.MINIMIZE)
    model.optimize()
    return model, x, demand_rows, supply_rows, capacity_rows


def total_cost(fixed, y_value, variable_cost):
    return float(variable_cost) + sum(float(fixed[i, j]) * y_value[i][j] for i in range(len(y_value)) for j in range(len(y_value[i])))


def evaluate(m, n, supply, demand, unit, fixed, capacity, y_value):
    model, x, dr, sr, cr = recourse(m, n, supply, demand, unit, capacity, y_value)
    if model.Status != GRB.OPTIMAL:
        model.dispose()
        return None
    return {"objective": total_cost(fixed, y_value, model.ObjVal),
            "y": [row[:] for row in y_value],
            "x": [[max(0.0, float(x[i, j].X)) for j in range(n)] for i in range(m)],
            "model": model, "demand_rows": dr, "supply_rows": sr, "capacity_rows": cr}


def add_flow_cuts(master, y, m, n, supply, demand, capacity):
    # These are all source/sink cut inequalities for the capacitated flow network.
    # For supplier set A and customer set B on the source side:
    # cap(A, customers\B)*y >= demand(customers\B)-supply(suppliers\A).
    number = 0
    for amask in range(1 << m):
        outside_supply = sum(float(supply[i]) for i in range(m) if not (amask & (1 << i)))
        for bmask in range(1 << n):
            rhs = sum(float(demand[j]) for j in range(n) if not (bmask & (1 << j))) - outside_supply
            if rhs <= 1e-9:
                continue
            expr = gp.quicksum(float(capacity[i, j]) * y[i, j]
                               for i in range(m) if amask & (1 << i)
                               for j in range(n) if not (bmask & (1 << j)))
            master.addConstr(expr >= rhs, name="flow_feas_%d" % number)
            number += 1


def make_output(status, objective, y_value, x_value):
    return {"status": status, "method": "benders_decomposition", "objective_value": float(objective), "y": y_value, "x": x_value}


def solve():
    m, n, supply, demand, unit, fixed, capacity = load_data()
    zero_y = [[0 for _ in range(n)] for _ in range(m)]
    zero_x = [[0.0 for _ in range(n)] for _ in range(m)]
    if np.any(supply < -1e-9) or np.any(demand < -1e-9) or float(np.sum(supply)) + 1e-9 < float(np.sum(demand)):
        return make_output("infeasible", 0.0, zero_y, zero_x)

    started = time.monotonic()
    deadline = started + 55.0
    all_y = [[1 for _ in range(n)] for _ in range(m)]
    incumbent = evaluate(m, n, supply, demand, unit, fixed, capacity, all_y)
    if incumbent is None:
        return make_output("infeasible", 0.0, zero_y, zero_x)

    # Greedily close lanes, retaining only candidates whose fixed-y recourse is feasible.
    current_y = [row[:] for row in all_y]
    improved = True
    while improved and time.monotonic() < deadline - 12.0:
        improved = False
        candidates = sorted(((float(fixed[i, j]), i, j) for i in range(m) for j in range(n) if current_y[i][j]), reverse=True)
        for _, i, j in candidates:
            trial = [row[:] for row in current_y]
            trial[i][j] = 0
            candidate = evaluate(m, n, supply, demand, unit, fixed, capacity, trial)
            if candidate is not None and candidate["objective"] < incumbent["objective"] - 1e-7:
                current_y = trial
                incumbent["model"].dispose()
                incumbent = candidate
                improved = True
            elif candidate is not None:
                candidate["model"].dispose()
        # A second pass can expose removals that were not profitable in the first pass.
    incumbent["model"].dispose()

    master = gp.Model("benders_master")
    master.Params.OutputFlag = 0
    master.Params.Threads = 1
    master.Params.MIPGap = 0.0
    master.Params.DualReductions = 0
    y = master.addVars(m, n, vtype=GRB.BINARY, name="y")
    theta = master.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="theta")
    master.setObjective(gp.quicksum(float(fixed[i, j]) * y[i, j] for i in range(m) for j in range(n)) + theta, GRB.MINIMIZE)
    add_flow_cuts(master, y, m, n, supply, demand, capacity)
    master.update()

    proven = False
    iterations = 0
    while time.monotonic() < deadline:
        master.Params.TimeLimit = max(0.2, deadline - time.monotonic())
        master.optimize()
        if master.SolCount == 0:
            break
        y_value = [[1 if y[i, j].X > 0.5 else 0 for j in range(n)] for i in range(m)]
        sp, x, dr, sr, cr = recourse(m, n, supply, demand, unit, capacity, y_value)
        if sp.Status != GRB.OPTIMAL:
            sp.dispose()
            break
        x_value = [[max(0.0, float(x[i, j].X)) for j in range(n)] for i in range(m)]
        total = total_cost(fixed, y_value, sp.ObjVal)
        if total < incumbent["objective"] - 1e-7:
            incumbent = {"objective": total, "y": y_value, "x": x_value}
        constant = sum(float(demand[j]) * float(dr[j].Pi) for j in range(n)) + sum(float(supply[i]) * float(sr[i].Pi) for i in range(m))
        cut = gp.LinExpr(constant)
        for i in range(m):
            for j in range(n):
                cut += float(capacity[i, j]) * float(cr[i * n + j].Pi) * y[i, j]
        master.addConstr(theta >= cut, name="optimality_%d" % iterations)
        master.update()
        sp.dispose()
        iterations += 1
        # Reoptimize after adding the cut before checking the lower bound.
        if time.monotonic() >= deadline:
            break
        master.Params.TimeLimit = max(0.2, deadline - time.monotonic())
        master.optimize()
        if master.Status == GRB.OPTIMAL and master.SolCount > 0 and float(master.ObjVal) >= incumbent["objective"] - 1e-5:
            proven = True
            break
    master.dispose()
    return make_output("optimal" if proven else "feasible", incumbent["objective"], incumbent["y"], incumbent["x"])


if __name__ == "__main__":
    try:
        answer = solve()
    except Exception:
        answer = {"status": "error", "method": "benders_decomposition", "objective_value": 0.0, "y": [], "x": []}
    with open("output.json", "w", encoding="utf-8") as stream:
        json.dump(answer, stream, indent=2, allow_nan=False)
