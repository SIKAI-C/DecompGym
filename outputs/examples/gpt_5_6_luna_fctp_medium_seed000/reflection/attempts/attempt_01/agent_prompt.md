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