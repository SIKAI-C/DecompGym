"""Shared prospective method contract for generation, reflection, and source review."""
from __future__ import annotations


DECOMPOSITION_POLICY = "strict-decomposition-20261013-v2"
METHOD_HEADER = "Decomposition method requirements:"
METHOD_NAMES = {"bd": "Benders decomposition", "cg": "column generation"}

METHOD_REQUIREMENTS = {
    "bd": "Use a master problem for the complicating decisions and recourse-cost estimates. Solve subproblems with the master decisions fixed, and use subproblem information to generate feasibility or optimality cuts and update subsequent master solves. Classical, integer, and logic-based Benders decomposition are allowed; LP duals are not required for integer or logic-based variants. Solving the master and fixed-decision subproblems as LPs or MIPs is allowed. Construct the reported solution from this decomposition, retaining a feasible incumbent when appropriate.",
    "cg": "Use a restricted master problem and pricing subproblems. Use master dual or reduced-cost information to generate columns, add useful columns to the restricted master, and reoptimize iteratively. A final integer solve of the restricted master over the accumulated columns is allowed. Construct the reported solution from this column-generation procedure.",
}

COMMON_REQUIREMENTS = (
    "Do not build and solve a separate monolithic optimization model of the full original problem, including for an initial incumbent, initialization, a warm start, a fallback, the reported solution, or an optimality certificate. This prohibition includes a full-original-model feasibility solve with a zero objective and applies to every reachable solution-producing path, including conditional fallback paths, even when decomposition also runs.",
    "Heuristic initialization, directly constructed seed columns, phase-I restricted masters with artificial variables and pricing, valid initial bounds or cuts, and warm-starting a legitimate decomposition master are allowed. A permitted initialization must not obtain its solution from a separate full-original-model solve. The restriction concerns that separate full model, not legitimate decomposition masters, fixed-decision subproblems, pricing problems, or integer restricted masters.",
    "Implement the information feedback required by the requested method. There is no minimum iteration count and no requirement that every instance add a cut or column, or improve the initial incumbent: legitimate immediate convergence or termination is allowed. Report mutually consistent decisions and objective values from the same incumbent. Use status feasible when returning an incumbent without a global optimality proof; optimality of an integer restricted master alone does not prove global optimality.",
)


def method_requirements(suite: str) -> str:
    return "\n\n".join((METHOD_HEADER, METHOD_REQUIREMENTS[suite], *COMMON_REQUIREMENTS))
