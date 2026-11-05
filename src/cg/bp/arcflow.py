"""Arc-flow MIP for bin packing / cutting stock (Valerio de Carvalho 1999, Annals of OR 86:629-659).

Nodes are the loads 0..C. An item arc (d, d + w_i) packs one item of type i; a loss arc (d, d + 1) wastes one unit.
A unit of flow from 0 to C is one bin (roll); minimize the flow out of 0 subject to flow conservation and
sum of type-i arcs >= demand_i. Symmetry reduction (same paper): an arc of type i starts only at loads reachable with sizes >= w_i; this keeps
the non-increasing-size path of every pattern (other orders may remain), so no packing is cut off. Its LP relaxation equals the Gilmore-Gomory bound, which
is why it proves optimality where the Kantorovich assignment model stalls on the weak bound sum(w) / C.
"""
from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB


def build_arcflow(sizes: list[int], demands: list[int], capacity: int, name: str) -> tuple[gp.Model, dict[str, Any]]:
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i])
    reach = {0}
    item_arcs: list[tuple[int, int]] = []  # (type, tail)
    for i in order:
        w = sizes[i]
        # Loads reachable with types of size >= w (including type i itself, any multiplicity).
        new = set(reach)
        for d in range(capacity + 1):
            if d in new and d + w <= capacity:
                new.add(d + w)
        tails = sorted(d for d in new if d + w <= capacity)
        item_arcs.extend((i, d) for d in tails)
        reach = new
    model = gp.Model(name)
    x = model.addVars(item_arcs, vtype=GRB.INTEGER, lb=0, name="x")
    loss = model.addVars(range(capacity), vtype=GRB.INTEGER, lb=0, name="loss")  # arc (d, d + 1)
    z = model.addVar(vtype=GRB.INTEGER, lb=0, name="z")
    out_arcs: dict[int, list] = {d: [] for d in range(capacity + 1)}
    in_arcs: dict[int, list] = {d: [] for d in range(capacity + 1)}
    for (i, d), var in x.items():
        out_arcs[d].append(var)
        in_arcs[d + sizes[i]].append(var)
    for d, var in loss.items():
        out_arcs[d].append(var)
        in_arcs[d + 1].append(var)
    for d in range(capacity + 1):
        balance = gp.quicksum(in_arcs[d]) - gp.quicksum(out_arcs[d])
        rhs = -z if d == 0 else (z if d == capacity else 0)
        model.addConstr(balance == rhs, name=f"flow_{d}")
    by_type: dict[int, list] = {i: [] for i in range(len(sizes))}
    for (i, _), var in x.items():
        by_type[i].append(var)
    for i in range(len(sizes)):
        model.addConstr(gp.quicksum(by_type[i]) >= demands[i], name=f"demand_{i}")
    model.setObjective(z, GRB.MINIMIZE)
    return model, {"x": x, "loss": loss, "z": z, "sizes": sizes, "capacity": capacity}


def decompose_flow(state: dict[str, Any]) -> list[tuple[list[int], int]]:
    """Integer flow -> list of (count vector over types, usage)."""
    sizes, capacity = state["sizes"], state["capacity"]
    flow: dict[int, list[list]] = {d: [] for d in range(capacity + 1)}  # tail -> [[head, type or None, remaining]]
    for (i, d), var in state["x"].items():
        if round(var.X) > 0:
            flow[d].append([d + sizes[i], i, round(var.X)])
    for d, var in state["loss"].items():
        if round(var.X) > 0:
            flow[d].append([d + 1, None, round(var.X)])
    patterns: dict[tuple[int, ...], int] = {}
    for _ in range(round(state["z"].X)):
        d, counts = 0, [0] * len(sizes)
        while d < capacity:
            arc = next(a for a in flow[d] if a[2] > 0)
            arc[2] -= 1
            if arc[1] is not None:
                counts[arc[1]] += 1
            d = arc[0]
        patterns[tuple(counts)] = patterns.get(tuple(counts), 0) + 1
    return [(list(k), v) for k, v in patterns.items() if any(k)]
