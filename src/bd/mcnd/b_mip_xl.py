"""Compact MCND MIP and the independent design evaluator (scale-up study).

Strong arc-based formulation:
    min  sum_a f_a y_a + sum_{a,k} c_a x_ak
    s.t. sum_{a in out(i)} x_ak - sum_{a in in(i)} x_ak = b_ik      (every node i, commodity k)
         sum_k x_ak <= u_a y_a                                       (capacity)
         x_ak <= min(d_k, u_a) y_a                                   (strong linking, all of them, as ordinary rows,
                                                                      only where d_k < u_a: otherwise capacity implies it)
         x >= 0, y binary.
The strong linking rows are valid because with nonnegative costs some optimal flow is cycle-free, so no commodity
carries more than its demand on an arc. For the same reason x_ak = 0 is fixed on arcs entering o_k or leaving t_k.
On the largest Canad C instances (30 nodes, ~700 arcs, 400 commodities) this is ~275k flow variables and ~275k
linking rows; `linking="lazy"` keeps them as Gurobi lazy rows (Lazy=3: pulled into the node LPs when violated)
for experiments, but the default is the full strong formulation.
The root relaxation is solved by barrier (Method=2, `root_method`): with one thread Gurobi's default root method
did not finish the c64 root LP in 55 s, while barrier solved it in ~38 s and the MIP reached 131,086 by 45 s
(Gurobi 11.0.3, laptop, Oct 30 2026); node relaxations keep Gurobi's default (dual simplex).

`solve_flows` is the evaluator of a design: the multicommodity min-cost flow LP over the open arcs only
(capacities u_a, no other restriction), returning the flow cost and the flows; `design_cost` adds the fixed costs.
"""
from __future__ import annotations

from typing import Any, Iterable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.mcnd.a_data import MCNDInstance


def useless_arcs(inst: MCNDInstance) -> np.ndarray:
    """(K, m) mask of commodity-arc pairs that carry no flow in some optimal solution: arcs entering the origin
    or leaving the destination of the commodity (only cycles use them; costs are nonnegative)."""
    return (inst.head[None, :] == inst.orig[:, None]) | (inst.tail[None, :] == inst.dest[:, None])


def add_flow_conservation(model: gp.Model, x: gp.MVar, inst: MCNDInstance, arcs: np.ndarray | None = None) -> list:
    """Rows sum_out x - sum_in x = b for every node and commodity; x has one column per arc in `arcs` (default all).
    Returns, per node, the (K,) block of rows (None for a node without arcs)."""
    tail = inst.tail if arcs is None else inst.tail[arcs]
    head = inst.head if arcs is None else inst.head[arcs]
    b = inst.supply()
    rows: list = []
    for i in range(inst.n):
        o, d = np.nonzero(tail == i)[0], np.nonzero(head == i)[0]
        if not len(o) and not len(d):
            if np.any(b[:, i] != 0):
                raise ValueError(f"node {i} has supply but no arcs")
            rows.append(None)
            continue
        expr = 0
        if len(o):
            expr = x[:, o].sum(axis=1)
        if len(d):
            expr = expr - x[:, d].sum(axis=1)
        rows.append(model.addConstr(expr == b[:, i], name=f"flow_{i}"))
    return rows


def build_mcnd_mip(inst: MCNDInstance, env: gp.Env | None = None, linking: str = "all",
                   root_method: int | None = 2) -> tuple[gp.Model, dict[str, Any]]:
    K, m = inst.K, inst.m
    model = gp.Model(env=env) if env else gp.Model()
    if root_method is not None:
        model.Params.Method = root_method
    L = inst.link_bound()
    ub = np.where(useless_arcs(inst), 0.0, L)
    y = model.addMVar(m, vtype=GRB.BINARY, name="y")
    x = model.addMVar((K, m), lb=0.0, ub=ub, name="x")
    add_flow_conservation(model, x, inst)
    model.addConstr(x.sum(axis=0) <= inst.cap * y, name="cap")
    if linking != "none":
        ks, as_ = np.nonzero((L < inst.cap[None, :]) & (ub > 0))
        if len(ks):
            rows = model.addConstr(x[ks, as_] <= L[ks, as_] * y[as_], name="link")
            if linking == "lazy":
                model.update()
                rows.Lazy = 3
    model.setObjective(inst.fixed @ y + (inst.cost[None, :] * x).sum(), GRB.MINIMIZE)
    return model, {"y": y, "x": x}


# ---------------------------------------------------------------- evaluator

def open_vector(inst: MCNDInstance, design: Iterable[int] | np.ndarray) -> np.ndarray:
    """Boolean mask of open arcs from a list of arc indices (a boolean array of length m is taken as the mask)."""
    if isinstance(design, np.ndarray) and design.dtype == bool:
        if len(design) != inst.m:
            raise ValueError("design mask has the wrong length")
        return design.copy()
    idx = np.asarray(list(design), dtype=np.int64)
    if len(idx) and (idx.min() < 0 or idx.max() >= inst.m):
        raise ValueError("arc index out of range")
    y = np.zeros(inst.m, dtype=bool)
    y[idx] = True
    return y


def solve_flows(inst: MCNDInstance, design: Iterable[int] | np.ndarray, env: gp.Env | None = None,
                time_limit: float | None = None) -> tuple[float, np.ndarray] | None:
    """Minimum-cost multicommodity flow over the open arcs. Returns (flow cost, flows (K, m)) or None if infeasible.

    The LP uses capacities u_a on open arcs and nothing else (no strong linking, no fixings), so it is the plain
    recourse of the design. Flows are clipped at 0 and tiny values (< 1e-11) are zeroed.
    """
    y = open_vector(inst, design)
    arcs = np.nonzero(y)[0]
    K = inst.K
    b = inst.supply()
    # Quick infeasibility checks: every origin needs an open outgoing arc and every destination an incoming one.
    if len(arcs) == 0:
        return None
    own = env is None
    if own:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", 1)
        env.start()
    try:
        model = gp.Model(env=env)
        model.Params.OutputFlag = 0
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        if time_limit is not None:
            model.Params.TimeLimit = max(0.01, time_limit)
        x = model.addMVar((K, len(arcs)), lb=0.0, name="x")
        tail, head = inst.tail[arcs], inst.head[arcs]
        for i in range(inst.n):
            o, d = np.nonzero(tail == i)[0], np.nonzero(head == i)[0]
            if not len(o) and not len(d):
                if np.any(b[:, i] != 0):
                    model.dispose()
                    return None
                continue
            expr = 0
            if len(o):
                expr = x[:, o].sum(axis=1)
            if len(d):
                expr = expr - x[:, d].sum(axis=1)
            model.addConstr(expr == b[:, i])
        model.addConstr(x.sum(axis=0) <= inst.cap[arcs])
        model.setObjective((inst.cost[arcs][None, :] * x).sum(), GRB.MINIMIZE)
        model.optimize()
        if model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            model.dispose()
            return None
        if model.Status != GRB.OPTIMAL:
            model.dispose()
            raise RuntimeError(f"flow LP ended with status {model.Status}")
        xs = np.asarray(x.X)
        model.dispose()
    finally:
        if own:
            env.dispose()
    xs = np.where(xs < 1e-11, 0.0, xs)
    flows = np.zeros((K, inst.m))
    flows[:, arcs] = xs
    return float((inst.cost[None, :] * flows).sum()), flows


def plan_cost(inst: MCNDInstance, y_open: Iterable[int] | np.ndarray, flows: np.ndarray) -> float:
    """Objective of a design and its flows: fixed costs of the open arcs + variable costs of the flows."""
    y = open_vector(inst, y_open)
    return float(inst.fixed[y].sum() + (inst.cost[None, :] * np.asarray(flows, dtype=float)).sum())


def design_cost(inst: MCNDInstance, design: Iterable[int] | np.ndarray, env: gp.Env | None = None) -> float | None:
    """True objective of a design (fixed costs + optimal flow cost), or None if no feasible routing exists."""
    y = open_vector(inst, design)
    res = solve_flows(inst, y, env=env)
    if res is None:
        return None
    return plan_cost(inst, y, res[1])


def design_output(inst: MCNDInstance, design: Iterable[int] | np.ndarray, method: str = "benders_decomposition",
                  status: str = "feasible", env: gp.Env | None = None) -> dict[str, Any] | None:
    """A design in the output contract of e_verifier (sparse flows [[k, a, value], ...]), or None if infeasible.
    The objective is recomputed from exactly the flows written, so the verifier's recomputation matches it."""
    y = open_vector(inst, design)
    res = solve_flows(inst, y, env=env)
    if res is None:
        return None
    flows = res[1]
    ks, arcs = np.nonzero(flows)
    return {"status": status, "method": method, "objective_value": plan_cost(inst, y, flows), "y": y.astype(int).tolist(),
            "flows": [[int(k), int(a), float(flows[k, a])] for k, a in zip(ks, arcs)]}
