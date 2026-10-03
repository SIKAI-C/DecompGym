Implemented `solver.py` with:

- Benders master, feasibility cuts, and optimality cuts
- Validated feasible incumbent retention
- Independent all-open infeasibility check
- Correct `../../data.json` and `../../prompt.txt` paths
- Required `output.json` contract
- Single-threaded Gurobi and time/iteration limits

Syntax validation passed. The generated solver was not executed.