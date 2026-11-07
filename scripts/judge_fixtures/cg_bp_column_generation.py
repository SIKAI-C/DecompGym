import json
import math
import time

import gurobipy as gp
from gurobipy import GRB

start = time.time()
data = json.load(open("../../data.json"))
capacity = data["bin_capacity"]
items = data["items"]
size = [item["size"] for item in items]
k = len(items)

# Initial columns: one item per bin.
patterns = [[1 if i == j else 0 for i in range(k)] for j in range(k)]


def price(duals):
    """0-1 knapsack by dynamic programming: the pattern with the largest dual value that fits one bin."""
    best = [(0.0, [])] * (capacity + 1)
    for i in range(k):
        for c in range(capacity, size[i] - 1, -1):
            value, chosen = best[c - size[i]]
            if value + duals[i] > best[c][0]:
                best[c] = (value + duals[i], chosen + [i])
    value, chosen = max(best, key=lambda entry: entry[0])
    return value, [1 if i in chosen else 0 for i in range(k)]


def restricted_master(relax):
    model = gp.Model("rmp")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    lam = model.addVars(len(patterns), lb=0.0, vtype=GRB.CONTINUOUS if relax else GRB.INTEGER)
    cover = model.addConstrs(gp.quicksum(patterns[p][i] * lam[p] for p in range(len(patterns))) == 1 for i in range(k))  # each item packed exactly once
    model.setObjective(lam.sum(), GRB.MINIMIZE)
    model.optimize()
    return model, lam, cover


converged = False
while time.time() - start < 45:
    model, lam, cover = restricted_master(relax=True)
    duals = [cover[i].Pi for i in range(k)]
    value, column = price(duals)
    if value <= 1 + 1e-9:
        converged = True  # no column with negative reduced cost: the master LP value is the LP bound
        break
    patterns.append(column)

lp_bound = model.ObjVal
model, lam, _ = restricted_master(relax=False)  # integer restricted master over the generated columns
bins = [{"usage": int(round(lam[p].X)), "contents": [{"item_id": items[i]["id"], "count": patterns[p][i]} for i in range(k) if patterns[p][i]]}
        for p in range(len(patterns)) if lam[p].X > 0.5]
objective = float(sum(entry["usage"] for entry in bins))
status = "optimal" if converged and objective <= math.ceil(lp_bound - 1e-9) + 1e-9 else "feasible"
json.dump({"status": status, "method": "column_generation", "objective_value": objective, "solution": {"bins": bins}}, open("output.json", "w"))
