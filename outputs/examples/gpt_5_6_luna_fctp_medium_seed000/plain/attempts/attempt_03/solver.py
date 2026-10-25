import json
import math
import os
import time

import gurobipy as gp
import numpy as np


def load_instance():
    # The harness runs this file from its documented runtime directory.
    path = os.path.join(os.getcwd(), "..", "..", "data.json")
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


def max_flow_cut(y, supply, demand, capacity, tol=1.0e-9):
    """Return a source-side min cut for the capacitated network induced by y."""
    m, n = y.shape
    source = 0
    first_factory = 1
    first_customer = first_factory + m
    sink = first_customer + n
    size = sink + 1
    graph = [[] for _ in range(size)]

    def add_edge(a, b, cap):
        graph[a].append([b, float(cap), len(graph[b])])
        graph[b].append([a, 0.0, len(graph[a]) - 1])

    for i in range(m):
        add_edge(source, first_factory + i, supply[i])
    for i in range(m):
        for j in range(n):
            if y[i, j] > 0.5 and capacity[i, j] > tol:
                add_edge(first_factory + i, first_customer + j, capacity[i, j])
    for j in range(n):
        add_edge(first_customer + j, sink, demand[j])

    flow = 0.0
    while True:
        level = [-1] * size
        level[source] = 0
        queue = [source]
        for v in queue:
            for w, cap, _ in graph[v]:
                if cap > tol and level[w] < 0:
                    level[w] = level[v] + 1
                    queue.append(w)
        if level[sink] < 0:
            break
        it = [0] * size

        def send(v, amount):
            if v == sink:
                return amount
            while it[v] < len(graph[v]):
                edge = graph[v][it[v]]
                w, cap, rev = edge
                if cap > tol and level[w] == level[v] + 1:
                    pushed = send(w, min(amount, cap))
                    if pushed > tol:
                        edge[1] -= pushed
                        graph[w][rev][1] += pushed
                        return pushed
                it[v] += 1
            return 0.0

        while True:
            pushed = send(source, float("inf"))
            if pushed <= tol:
                break
            flow += pushed

    reachable = [False] * size
    reachable[source] = True
    queue = [source]
    for v in queue:
        for w, cap, _ in graph[v]:
            if cap > tol and not reachable[w]:
                reachable[w] = True
                queue.append(w)

    # The cut is represented as constant capacity plus coefficients of y.
    constant = 0.0
    coeff = np.zeros((m, n), dtype=float)
    for i in range(m):
        if reachable[source] and not reachable[first_factory + i]:
            constant += supply[i]
    for j in range(n):
        if reachable[first_customer + j] and not reachable[sink]:
            constant += demand[j]
    for i in range(m):
        if reachable[first_factory + i]:
            for j in range(n):
                if not reachable[first_customer + j]:
                    coeff[i, j] = capacity[i, j]
    return flow, constant, coeff


def solve_subproblem(y, supply, demand, unit, capacity):
    m, n = y.shape
    model = gp.Model("fixed_decision_recourse")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    x = model.addMVar((m, n), lb=0.0, name="x")
    demand_con = []
    supply_con = []
    cap_con = []
    for j in range(n):
        demand_con.append(model.addConstr(gp.quicksum(x[i, j] for i in range(m)) == float(demand[j])))
    for i in range(m):
        supply_con.append(model.addConstr(gp.quicksum(x[i, j] for j in range(n)) <= float(supply[i])))
    for i in range(m):
        for j in range(n):
            cap_con.append(model.addConstr(x[i, j] <= float(capacity[i, j] * y[i, j])))
    model.setObjective(gp.quicksum(float(unit[i, j]) * x[i, j] for i in range(m) for j in range(n)), gp.GRB.MINIMIZE)
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        return None
    dual_demand = np.array([c.Pi for c in demand_con], dtype=float)
    dual_supply = np.array([c.Pi for c in supply_con], dtype=float)
    dual_capacity = np.array([[cap_con[i * n + j].Pi for j in range(n)] for i in range(m)], dtype=float)
    return np.asarray(x.X, dtype=float), float(model.ObjVal), dual_demand, dual_supply, dual_capacity


def write_output(result):
    with open("output.json", "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


def main():
    m, n, supply, demand, unit, fixed, capacity = load_instance()
    master = gp.Model("benders_master")
    master.Params.OutputFlag = 0
    master.Params.Threads = 1
    yvar = master.addMVar((m, n), vtype=gp.GRB.BINARY, name="y")
    theta = master.addVar(lb=0.0, name="theta")
    master.setObjective(gp.quicksum(float(fixed[i, j]) * yvar[i, j] for i in range(m) for j in range(n)) + theta, gp.GRB.MINIMIZE)

    incumbent_obj = float("inf")
    incumbent_y = None
    incumbent_x = None
    start = time.monotonic()
    max_iterations = 300

    for _ in range(max_iterations):
        if time.monotonic() - start > 52.0:
            break
        master.optimize()
        if master.Status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
            break
        candidate_y = np.rint(np.asarray(yvar.X, dtype=float)).astype(int)
        flow, constant, coeff = max_flow_cut(candidate_y, supply, demand, capacity)
        if flow < float(np.sum(demand)) - 1.0e-7:
            lhs = float(constant) + gp.quicksum(float(coeff[i, j]) * yvar[i, j] for i in range(m) for j in range(n) if coeff[i, j] != 0.0)
            master.addConstr(lhs >= float(np.sum(demand)), name="feasibility_cut")
            continue

        recourse = solve_subproblem(candidate_y, supply, demand, unit, capacity)
        if recourse is None:
            # This should be unreachable after the max-flow test, but gives a safe
            # decomposition-only exclusion if a numerical edge case occurs.
            lhs = gp.quicksum((1.0 - 2.0 * int(candidate_y[i, j])) * yvar[i, j] for i in range(m) for j in range(n))
            rhs = float(np.sum(candidate_y)) - 1.0
            master.addConstr(lhs <= rhs, name="numerical_exclusion")
            continue

        x_value, recourse_value, pi_demand, pi_supply, pi_capacity = recourse
        total = float(np.sum(fixed * candidate_y) + recourse_value)
        if total < incumbent_obj - 1.0e-8:
            incumbent_obj = total
            incumbent_y = candidate_y.copy()
            incumbent_x = x_value.copy()

        # Every dual solution of the fixed-y LP is feasible for the recourse
        # dual, so this globally valid cut supplies the master cost estimate.
        cut_rhs = float(np.dot(demand, pi_demand) + np.dot(supply, pi_supply))
        cut_rhs += gp.quicksum(float(capacity[i, j] * pi_capacity[i, j]) * yvar[i, j] for i in range(m) for j in range(n))
        master.addConstr(theta >= cut_rhs, name="optimality_cut")

        if incumbent_y is not None and master.ObjBound >= incumbent_obj - 1.0e-7:
            break

    if incumbent_y is None:
        write_output({"status": "infeasible", "method": "benders_decomposition", "objective_value": None, "y": [], "x": []})
        return

    proven = master.Status == gp.GRB.OPTIMAL and master.ObjBound >= incumbent_obj - 1.0e-7
    result = {
        "status": "optimal" if proven else "feasible",
        "method": "benders_decomposition",
        "objective_value": float(incumbent_obj),
        "y": incumbent_y.astype(int).tolist(),
        "x": incumbent_x.tolist(),
    }
    write_output(result)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        write_output({"status": "error", "method": "benders_decomposition", "objective_value": None, "y": [], "x": []})
