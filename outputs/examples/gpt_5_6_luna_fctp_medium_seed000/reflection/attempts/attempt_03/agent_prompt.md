Only edit the existing solver.py in the current working directory. All task context is supplied below.

Do not inspect other directories, use external tools or services, or search for reference implementations.

Do not execute the generated solver. The harness executes it later in its documented runtime directory.

Keep prose paragraphs and list items on one physical line; rely on editor soft wrapping.

You are an expert operations research engineer and Python programmer.

You are given a fixed-charge transportation problem:
  First choose which factory-to-customer lanes to activate; each chosen lane costs a fixed fee and then allows up to a capacity of flow.
  After lanes are open, decide shipment quantities so every customer's demand is met without exceeding any factory's supply.
  The total cost combines fixed activation and per-unit shipping. This mirrors "decide infrastructure first, operate flow second."
  Parameters:
  - There are m = 4 supply nodes and n = 5 demand nodes.
  - Each (i, j) arc has a fixed activation cost, a unit shipping cost, and a capacity.
  - Decision variables: y[i,j] in {0,1} indicates whether arc (i,j) is opened; x[i,j] >= 0 is quantity shipped.

The data for this instance is in data.json with fields:
  - "supply": list length m
  - "demand": list length n
  - "unit_cost", "fixed_cost", "capacity": each an m x n matrix

Sample rows (full data is in the JSON file):
Supply vector:
  supply = [32.0, 28.0, 25.0, 20.0]
Demand vector:
  demand = [14.0, 10.0, 11.0, 10.0, 60.0]
Fixed costs:
  fixed_cost[0] = [21.0, 20.0, 12.0, 14.0, 24.0]
  fixed_cost[1] = [21.0, 24.0, 12.0, 18.0, 24.0]
  fixed_cost[2] = [18.0, 21.0, 18.0, 19.0, 12.0]
  fixed_cost[3] = [22.0, 7.0, 16.0, 19.0, 21.0]
Unit costs:
  unit_cost[0] = [9.214800195499496, 6.459721981904619, 7.565469048855985, 5.892624923188806, 9.415651814089914]
  unit_cost[1] = [8.342681987093789, 1.0246465015313329, 8.716638489288124, 1.3022701777491792, 7.566899017869496]
  unit_cost[2] = [2.580900585423031, 8.768610301148978, 5.8731509822418255, 3.697407014836463, 4.804184990778926]
  unit_cost[3] = [1.2548770403091667, 2.1185494884960754, 7.035619732242673, 6.824705604168251, 6.538466003331285]

Task:
Assume gurobipy and NumPy are available; the Python standard library is also allowed. Do not depend on any files other than data.json. Write a Python program solver.py that:
1) Loads data.json.
2) Implements Benders decomposition according to the method requirements below.
3) Writes output.json with at least:
    {
      "status": "optimal | feasible | infeasible | error",
      "method": "benders_decomposition",
      "objective_value": <float>,
      "y": [[...]],  # m x n
      "x": [[...]]   # m x n corresponding to the reported incumbent y and objective_value
    }

Decomposition method requirements:

Use a master problem for the complicating decisions and recourse-cost estimates. Solve subproblems with the master decisions fixed, and use subproblem information to generate feasibility or optimality cuts and update subsequent master solves. Classical, integer, and logic-based Benders decomposition are allowed; LP duals are not required for integer or logic-based variants. Solving the master and fixed-decision subproblems as LPs or MIPs is allowed. Construct the reported solution from this decomposition, retaining a feasible incumbent when appropriate.

Do not build and solve a separate monolithic optimization model of the full original problem, including for an initial incumbent, initialization, a warm start, a fallback, the reported solution, or an optimality certificate. This prohibition includes a full-original-model feasibility solve with a zero objective and applies to every reachable solution-producing path, including conditional fallback paths, even when decomposition also runs.

Heuristic initialization, directly constructed seed columns, phase-I restricted masters with artificial variables and pricing, valid initial bounds or cuts, and warm-starting a legitimate decomposition master are allowed. A permitted initialization must not obtain its solution from a separate full-original-model solve. The restriction concerns that separate full model, not legitimate decomposition masters, fixed-decision subproblems, pricing problems, or integer restricted masters.

Implement the information feedback required by the requested method. There is no minimum iteration count and no requirement that every instance add a cut or column, or improve the initial incumbent: legitimate immediate convergence or termination is allowed. Report mutually consistent decisions and objective values from the same incumbent. Use status feasible when returning an incumbent without a global optimality proof; optimality of an integer restricted master alone does not prove global optimality.

The data below are provided as read-only context for this editing call.
The generated solver.py will be executed later in a separate runtime directory, where it must read ../../data.json and write output.json in that runtime directory.

Implement a standalone solver.py using gurobipy, NumPy, and the Python standard library. Write output.json using the documented contract. Do not import repository implementations or load reference solutions.
The harness limits the entire solver.py execution to 60 seconds and numerical solvers to 1 thread(s). Design your algorithm and termination handling within this budget.

--- BEGIN CONTEXT: fctp_medium_seed000/data.json ---
{
  "m": 4,
  "n": 5,
  "supply": [
    32.0,
    28.0,
    25.0,
    20.0
  ],
  "demand": [
    14.0,
    10.0,
    11.0,
    10.0,
    60.0
  ],
  "unit_cost": [
    [
      9.214800195499496,
      6.459721981904619,
      7.565469048855985,
      5.892624923188806,
      9.415651814089914
    ],
    [
      8.342681987093789,
      1.0246465015313329,
      8.716638489288124,
      1.3022701777491792,
      7.566899017869496
    ],
    [
      2.580900585423031,
      8.768610301148978,
      5.8731509822418255,
      3.697407014836463,
      4.804184990778926
    ],
    [
      1.2548770403091667,
      2.1185494884960754,
      7.035619732242673,
      6.824705604168251,
      6.538466003331285
    ]
  ],
  "fixed_cost": [
    [
      21.0,
      20.0,
      12.0,
      14.0,
      24.0
    ],
    [
      21.0,
      24.0,
      12.0,
      18.0,
      24.0
    ],
    [
      18.0,
      21.0,
      18.0,
      19.0,
      12.0
    ],
    [
      22.0,
      7.0,
      16.0,
      19.0,
      21.0
    ]
  ],
  "capacity": [
    [
      32.0,
      32.0,
      32.0,
      32.0,
      60.0
    ],
    [
      28.0,
      28.0,
      28.0,
      28.0,
      60.0
    ],
    [
      25.0,
      25.0,
      25.0,
      25.0,
      60.0
    ],
    [
      20.0,
      20.0,
      20.0,
      20.0,
      60.0
    ]
  ]
}
--- END CONTEXT: fctp_medium_seed000/data.json ---

--- BEGIN CONTEXT: attempt_01/solver.py ---
import json
import os
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB


def load_instance():
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data.json"))
    with open(data_path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    m = int(data["m"])
    n = int(data["n"])
    supply = np.asarray(data["supply"], dtype=float)
    demand = np.asarray(data["demand"], dtype=float)
    unit_cost = np.asarray(data["unit_cost"], dtype=float)
    fixed_cost = np.asarray(data["fixed_cost"], dtype=float)
    capacity = np.asarray(data["capacity"], dtype=float)
    if supply.shape != (m,) or demand.shape != (n,):
        raise ValueError("invalid supply or demand dimensions")
    for array in (unit_cost, fixed_cost, capacity):
        if array.shape != (m, n):
            raise ValueError("invalid matrix dimensions")
    return m, n, supply, demand, unit_cost, fixed_cost, capacity


def make_subproblem(m, n, supply, demand, unit_cost, capacity, y_value):
    model = gp.Model("transportation_subproblem")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.DualReductions = 0
    x = model.addVars(m, n, lb=0.0, name="x")
    demand_rows = [
        model.addConstr(gp.quicksum(x[i, j] for i in range(m)) == float(demand[j]), name="demand_%d" % j)
        for j in range(n)
    ]
    supply_rows = [
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) <= float(supply[i]), name="supply_%d" % i)
        for i in range(m)
    ]
    capacity_rows = [
        model.addConstr(x[i, j] <= float(capacity[i, j] * y_value[i][j]), name="capacity_%d_%d" % (i, j))
        for i in range(m) for j in range(n)
    ]
    model.setObjective(gp.quicksum(unit_cost[i, j] * x[i, j] for i in range(m) for j in range(n)), GRB.MINIMIZE)
    model.optimize()
    return model, x, demand_rows, supply_rows, capacity_rows


def add_nogood(master, y, y_value):
    expression = gp.LinExpr()
    for i in range(len(y_value)):
        for j in range(len(y_value[i])):
            if y_value[i][j] > 0.5:
                expression += 1.0 - y[i, j]
            else:
                expression += y[i, j]
    master.addConstr(expression >= 1.0, name="infeasible_pattern_%d" % master.NumConstrs)


def write_result(result):
    with open("output.json", "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


def solve():
    m, n, supply, demand, unit_cost, fixed_cost, capacity = load_instance()
    if np.any(supply < -1e-9) or np.any(demand < -1e-9):
        raise ValueError("negative supply or demand")
    if float(np.sum(supply)) + 1e-9 < float(np.sum(demand)):
        return {
            "status": "infeasible",
            "method": "benders_decomposition",
            "objective_value": None,
            "y": [[0 for _ in range(n)] for _ in range(m)],
            "x": [[0.0 for _ in range(n)] for _ in range(m)],
        }

    master = gp.Model("benders_master")
    master.Params.OutputFlag = 0
    master.Params.Threads = 1
    master.Params.MIPGap = 0.0
    y = master.addVars(m, n, vtype=GRB.BINARY, name="y")
    theta = master.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="theta")
    master.setObjective(
        gp.quicksum(float(fixed_cost[i, j]) * y[i, j] for i in range(m) for j in range(n)) + theta,
        GRB.MINIMIZE,
    )
    master.update()

    started = time.monotonic()
    time_limit = 54.0
    incumbent = None
    iterations = 0
    tolerance = 1e-6
    proven_optimal = False
    while time.monotonic() - started < time_limit:
        remaining = max(0.2, time_limit - (time.monotonic() - started))
        master.Params.TimeLimit = remaining
        master.optimize()
        if master.SolCount == 0:
            break
        y_value = [[1 if y[i, j].X >= 0.5 else 0 for j in range(n)] for i in range(m)]
        fixed_value = sum(float(fixed_cost[i, j]) * y_value[i][j] for i in range(m) for j in range(n))

        subproblem, x, demand_rows, supply_rows, capacity_rows = make_subproblem(
            m, n, supply, demand, unit_cost, capacity, y_value
        )
        if subproblem.Status == GRB.OPTIMAL:
            x_value = [[max(0.0, float(x[i, j].X)) for j in range(n)] for i in range(m)]
            recourse_value = float(subproblem.ObjVal)
            total_value = fixed_value + recourse_value
            if incumbent is None or total_value < incumbent["objective_value"] - tolerance:
                incumbent = {"objective_value": total_value, "y": y_value, "x": x_value}

            dual_constant = sum(float(demand[j]) * float(demand_rows[j].Pi) for j in range(n))
            dual_constant += sum(float(supply[i]) * float(supply_rows[i].Pi) for i in range(m))
            cut = gp.LinExpr()
            cut.addConstant(float(dual_constant))
            for i in range(m):
                for j in range(n):
                    cut += float(capacity[i, j] * capacity_rows[i * n + j].Pi) * y[i, j]
            master.addConstr(theta >= cut, name="optimality_cut_%d" % iterations)
            master.update()
            master_value = float(master.ObjVal)
            if abs(master_value - total_value) <= 5e-5:
                proven_optimal = True
                break
        elif subproblem.Status == GRB.INFEASIBLE:
            add_nogood(master, y, y_value)
            master.update()
        else:
            break
        iterations += 1
        subproblem.dispose()

    master.dispose()
    if incumbent is not None:
        status = "optimal" if proven_optimal else "feasible"
        return {
            "status": status,
            "method": "benders_decomposition",
            "objective_value": float(incumbent["objective_value"]),
            "y": incumbent["y"],
            "x": incumbent["x"],
        }
    return {
        "status": "infeasible",
        "method": "benders_decomposition",
        "objective_value": None,
        "y": [[0 for _ in range(n)] for _ in range(m)],
        "x": [[0.0 for _ in range(n)] for _ in range(m)],
    }


if __name__ == "__main__":
    try:
        write_result(solve())
    except Exception:
        write_result({
            "status": "error",
            "method": "benders_decomposition",
            "objective_value": None,
            "y": [],
            "x": [],
        })

--- END CONTEXT: attempt_01/solver.py ---

--- BEGIN CONTEXT: attempt_01/debug.md ---
The run did not produce an incumbent: the master first selected all y=0, the fixed-y transportation subproblem was infeasible, and the implementation added only a single-pattern no-good cut. Repeating this weak exclusion can require enumerating many infeasible opening patterns, so the 54-second loop ended with no feasible subproblem and returned status infeasible with objective_value null. That output also violated the verifier contract requiring a finite JSON number; the source-level algorithm review otherwise passed because the master and fixed-decision subproblem structure was legitimate.

The next attempt should make feasibility feedback stronger and ensure a feasible incumbent is found early. Add valid aggregate feasibility cuts (at minimum, for each customer, sum_i capacity[i,j]*y[i,j] >= demand[j], plus useful supply/capacity subset cuts) or generate a Farkas/max-flow feasibility cut from each infeasible fixed-y subproblem instead of relying on no-good cuts. Construct a seed opening pattern from capacities and solve its fixed-y transportation subproblem before the main loop, while keeping it within the decomposition framework. Do not label a solution optimal unless the master lower bound is recomputed after adding cuts and matches the incumbent within tolerance.

Concrete changes for the next attempt:

- Add customer coverage cuts and stronger aggregate supply-side feasibility cuts during master initialization.
- Replace or supplement no-good cuts with valid infeasibility cuts derived from the fixed-y LP (or a max-flow/min-cut feasibility subproblem).
- Build and evaluate a feasible seed y using only decomposition subproblems, retaining its x and fixed-plus-variable objective as the incumbent.
- Reoptimize the master after every new cut before testing convergence, and compare its lower bound with the incumbent objective.
- Return a finite objective_value on every output path; use feasible with the retained incumbent when proof is unavailable, and reserve infeasible for a demonstrated global infeasibility result.
- Preserve numerical tolerances, verify reported x satisfies demand, supply, and activated-capacity bounds, and leave sufficient time for final incumbent construction and JSON writing.

--- END CONTEXT: attempt_01/debug.md ---

--- BEGIN CONTEXT: attempt_02/solver.py ---
import json
import os
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB


def load_data():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data.json"))
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    m, n = int(d["m"]), int(d["n"])
    supply = np.asarray(d["supply"], dtype=float)
    demand = np.asarray(d["demand"], dtype=float)
    unit = np.asarray(d["unit_cost"], dtype=float)
    fixed = np.asarray(d["fixed_cost"], dtype=float)
    capacity = np.asarray(d["capacity"], dtype=float)
    if supply.shape != (m,) or demand.shape != (n,):
        raise ValueError("bad vector dimensions")
    if unit.shape != (m, n) or fixed.shape != (m, n) or capacity.shape != (m, n):
        raise ValueError("bad matrix dimensions")
    return m, n, supply, demand, unit, fixed, capacity


def fixed_subproblem(m, n, supply, demand, unit, capacity, y_value):
    model = gp.Model("benders_recourse")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.DualReductions = 0
    x = model.addVars(m, n, lb=0.0, name="x")
    demand_con = [model.addConstr(gp.quicksum(x[i, j] for i in range(m)) == float(demand[j])) for j in range(n)]
    supply_con = [model.addConstr(gp.quicksum(x[i, j] for j in range(n)) <= float(supply[i])) for i in range(m)]
    cap_con = [model.addConstr(x[i, j] <= float(capacity[i, j] * y_value[i][j])) for i in range(m) for j in range(n)]
    model.setObjective(gp.quicksum(float(unit[i, j]) * x[i, j] for i in range(m) for j in range(n)), GRB.MINIMIZE)
    model.optimize()
    return model, x, demand_con, supply_con, cap_con


def add_capacity_cut(master, y, capacity, demand, subset, number):
    master.addConstr(
        gp.quicksum(float(capacity[i, j]) * y[i, j] for i in range(len(capacity)) for j in subset) >= float(np.sum(demand[list(subset)])),
        name="capacity_subset_%d" % number,
    )


def result(status, objective, y_value, x_value):
    return {
        "status": status,
        "method": "benders_decomposition",
        "objective_value": float(objective) if objective is not None and np.isfinite(objective) else 0.0,
        "y": y_value,
        "x": x_value,
    }


def solve():
    m, n, supply, demand, unit, fixed, capacity = load_data()
    zero_y = [[0 for _ in range(n)] for _ in range(m)]
    zero_x = [[0.0 for _ in range(n)] for _ in range(m)]
    if np.any(supply < -1e-9) or np.any(demand < -1e-9) or np.sum(supply) + 1e-9 < np.sum(demand):
        return result("infeasible", 0.0, zero_y, zero_x)

    # The master contains only the complicating binary decisions and a lower
    # estimate of recourse cost; all flow decisions are made in subproblems.
    master = gp.Model("benders_master")
    master.Params.OutputFlag = 0
    master.Params.Threads = 1
    master.Params.MIPGap = 0.0
    y = master.addVars(m, n, vtype=GRB.BINARY, name="y")
    theta = master.addVar(lb=0.0, name="theta")
    master.setObjective(gp.quicksum(float(fixed[i, j]) * y[i, j] for i in range(m) for j in range(n)) + theta, GRB.MINIMIZE)

    cut_number = 0
    subset_cuts = set()
    # Every customer subset gives a valid necessary aggregate-capacity cut.
    # Including all subsets is inexpensive here and is substantially stronger
    # than excluding individual infeasible binary patterns.
    for mask in range(1, 1 << n):
        subset = tuple(j for j in range(n) if mask & (1 << j))
        add_capacity_cut(master, y, capacity, demand, subset, cut_number)
        subset_cuts.add(subset)
        cut_number += 1
    master.update()

    started = time.monotonic()
    deadline = started + 55.0
    incumbent = None
    tolerance = 1e-5

    # A deliberately simple valid seed: opening every lane guarantees enough
    # capacity, and its flow is still obtained by a decomposition subproblem.
    seed_y = [[1 for _ in range(n)] for _ in range(m)]
    seed_model, seed_x, _, _, _ = fixed_subproblem(m, n, supply, demand, unit, capacity, seed_y)
    if seed_model.Status == GRB.OPTIMAL:
        seed_x_value = [[max(0.0, float(seed_x[i, j].X)) for j in range(n)] for i in range(m)]
        seed_obj = float(seed_model.ObjVal) + float(np.sum(fixed * np.asarray(seed_y)))
        incumbent = {"objective": seed_obj, "y": seed_y, "x": seed_x_value}
    seed_model.dispose()

    proven = False
    iterations = 0
    while time.monotonic() < deadline:
        master.Params.TimeLimit = max(0.2, deadline - time.monotonic())
        master.optimize()
        if master.SolCount == 0:
            break
        y_value = [[1 if y[i, j].X >= 0.5 else 0 for j in range(n)] for i in range(m)]
        sp, x, demand_con, supply_con, cap_con = fixed_subproblem(m, n, supply, demand, unit, capacity, y_value)
        if sp.Status == GRB.OPTIMAL:
            x_value = [[max(0.0, float(x[i, j].X)) for j in range(n)] for i in range(m)]
            total = float(sp.ObjVal) + sum(float(fixed[i, j]) * y_value[i][j] for i in range(m) for j in range(n))
            if incumbent is None or total < incumbent["objective"] - tolerance:
                incumbent = {"objective": total, "y": y_value, "x": x_value}

            # At an optimal master solve, its objective is a valid global
            # lower bound. A feasible subproblem plus matching bounds proves
            # optimality before another redundant cut is added.
            if master.Status == GRB.OPTIMAL and float(master.ObjVal) >= incumbent["objective"] - tolerance:
                proven = True
                sp.dispose()
                break

            # The LP dual produces a valid Benders optimality cut in y.
            constant = sum(float(demand[j]) * float(demand_con[j].Pi) for j in range(n))
            constant += sum(float(supply[i]) * float(supply_con[i].Pi) for i in range(m))
            cut = gp.LinExpr(constant)
            for i in range(m):
                for j in range(n):
                    cut += float(capacity[i, j]) * float(cap_con[i * n + j].Pi) * y[i, j]
            master.addConstr(theta >= cut, name="optimality_%d" % iterations)
            master.update()
            sp.dispose()
            iterations += 1
            # Reoptimization is intentionally done on the next loop, so this
            # lower bound includes the newly added cut.
            continue

        if sp.Status == GRB.INFEASIBLE:
            # Find a violated aggregate cut for this fixed decision. It is a
            # valid feasibility cut independent of the particular incumbent.
            violated = None
            violation = 0.0
            for mask in range(1, 1 << n):
                subset = tuple(j for j in range(n) if mask & (1 << j))
                available = sum(float(capacity[i, j]) * y_value[i][j] for i in range(m) for j in subset)
                gap = float(np.sum(demand[list(subset)])) - available
                if gap > violation + 1e-7:
                    violated, violation = subset, gap
            if violated is not None and violated not in subset_cuts:
                add_capacity_cut(master, y, capacity, demand, violated, cut_number)
                subset_cuts.add(violated)
                cut_number += 1
            else:
                # This remains a legitimate decomposition feasibility cut and
                # prevents cycling if aggregate cuts do not identify the cause.
                expr = gp.LinExpr()
                for i in range(m):
                    for j in range(n):
                        expr += (1.0 - y[i, j]) if y_value[i][j] else y[i, j]
                master.addConstr(expr >= 1.0, name="exclude_pattern_%d" % iterations)
            master.update()
            sp.dispose()
            iterations += 1
            continue
        sp.dispose()
        break

    # A matching master lower bound after the final cut is the certificate.
    if incumbent is not None and time.monotonic() < deadline:
        master.Params.TimeLimit = max(0.2, deadline - time.monotonic())
        master.optimize()
        if master.SolCount > 0 and master.Status == GRB.OPTIMAL and float(master.ObjVal) >= incumbent["objective"] - tolerance:
            proven = True
    master.dispose()
    if incumbent is not None:
        return result("optimal" if proven else "feasible", incumbent["objective"], incumbent["y"], incumbent["x"])
    return result("infeasible", 0.0, zero_y, zero_x)


if __name__ == "__main__":
    try:
        output = solve()
    except Exception:
        output = result("error", 0.0, [], [])
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, allow_nan=False)

--- END CONTEXT: attempt_02/solver.py ---

--- BEGIN CONTEXT: attempt_02/debug.md ---
Diagnosis: The decomposition passed the verifier and produced a feasible, internally consistent incumbent, but objective quality failed because the only shown incumbent is the all-lanes-open seed (904.6569; fixed cost 363), indicating that useful master/subproblem progress did not finish within the time budget. The log contains no solver progress, so the likely weakness is slow convergence: aggregate customer-subset cuts do not fully describe transportation feasibility, and infeasible master patterns are mostly handled by one-pattern exclusions. The initial seed also gives no attempt to remove unnecessary lanes.

Next attempt should make these concrete changes:

- Keep the flow-only fixed-y subproblem and dual optimality cuts, but generate strong feasibility cuts from the infeasible transportation subproblem (a max-flow/min-cut or Farkas cut over supplier/customer subsets) instead of relying mainly on pattern exclusions.
- Build a better feasible incumbent early by greedily closing lanes one at a time (or using a small lane-restricted master), resolving the fixed-y flow subproblem after each change, and retaining every feasible cost improvement.
- Use the remaining time for local improvement of each feasible master incumbent, and avoid spending most iterations on redundant pre-added cuts; prioritize cuts that are violated by the current y.
- Preserve the incumbent’s exact fixed-y flow and recompute objective as fixed activation cost plus that subproblem objective; report “feasible” unless the master lower bound actually certifies optimality.

--- END CONTEXT: attempt_02/debug.md ---