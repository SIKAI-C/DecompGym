import json
import time

import gurobipy as gp
from gurobipy import GRB

start = time.time()
data = json.load(open("../../data.json"))
m, n, f, D = data["m"], data["n"], data["f"], data["D"]

# Master: open decisions y and one recourse-cost estimate theta_j per customer.
master = gp.Model("master")
master.Params.OutputFlag = 0
master.Params.Threads = 1
y = master.addVars(m, vtype=GRB.BINARY, name="y")
theta = master.addVars(n, lb=0.0, name="theta")
master.addConstr(y.sum() >= 1)
master.setObjective(gp.quicksum(f[i] * y[i] for i in range(m)) + theta.sum(), GRB.MINIMIZE)


def subproblem(open_set):
    """Fixed-y assignment: each customer goes to its cheapest open facility; returns cost and assignment."""
    assignment = [min(open_set, key=lambda i: D[i][j]) for j in range(n)]
    return sum(D[assignment[j]][j] for j in range(n)), assignment


best, incumbent = float("inf"), None
for iteration in range(200):
    master.optimize()
    lower = master.ObjVal
    y_hat = [int(round(y[i].X)) for i in range(m)]
    open_set = [i for i in range(m) if y_hat[i]]
    cost, assignment = subproblem(open_set)
    upper = sum(f[i] for i in open_set) + cost
    if upper < best:
        best, incumbent = upper, (y_hat, assignment)
    if best - lower <= 1e-6 * max(1.0, abs(best)) or time.time() - start > 50:
        break
    # Benders optimality cut per customer from the fixed-y subproblem (UFL closed-form duals).
    for j in range(n):
        c_star = D[assignment[j]][j]
        master.addConstr(theta[j] >= c_star - gp.quicksum(max(0.0, c_star - D[i][j]) * y[i] for i in range(m)))

y_best, assignment = incumbent
x = [[1 if assignment[j] == i else 0 for j in range(n)] for i in range(m)]
status = "optimal" if best - lower <= 1e-6 * max(1.0, abs(best)) else "feasible"
json.dump({"status": status, "method": "benders_decomposition", "objective_value": float(best), "y": y_best, "x": x}, open("output.json", "w"))
