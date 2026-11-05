from __future__ import annotations

import math
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.cg.cvrp.a_gen_data import CVRPInstance
from src.utils.base_baseline_class import BaseBaseline, BaselineSolution


class CVRPBaseline(BaseBaseline[CVRPInstance, gp.Model]):
    """Directed two-index vehicle flow with a single-commodity load flow (Gavish-Graves), as in the scale-up MIP.

    x_ij = 1 if a vehicle travels i -> j; f_ij is the load on that arc. Every customer has one in- and one
    out-arc and absorbs its demand from the flow; q_j x_ij <= f_ij <= (Q - q_i) x_ij, so loads stay within Q
    and no subtour avoids the depot. At least ceil(q(N) / Q) routes leave the depot. As in the scale-up MIP,
    rounded capacity inequalities x(delta+(S)) >= ceil(q(S) / Q) are separated as user cuts at the root
    (support-graph components, then greedy set growth from every customer).
    """

    def __init__(self, *, params: dict[str, Any] | None = None, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        # Zero gap tolerances: costs are integers, so a bound within 0.5 of the incumbent proves it exactly.
        self.params = {"Threads": 1, "MIPGap": 0.0, "MIPGapAbs": 0.5, **(params or {})}

    def load_instance(self, instance_name: str) -> CVRPInstance:
        return CVRPInstance.from_json_file(instance_name)

    def build_model(self, inst: CVRPInstance) -> gp.Model:
        n, Q, q = inst.n_customers, inst.capacity, inst.demand
        V, C = range(n + 1), range(1, n + 1)
        arcs = [(i, j) for i in V for j in V if i != j and not (i and j and q[i] + q[j] > Q)]
        model = gp.Model("cvrp_monolithic")
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.params.items():
            model.setParam(k, v)
        x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
        f = model.addVars(arcs, lb=0.0, name="f")
        model.addConstrs((x.sum(i, "*") == 1 for i in C), name="out")
        model.addConstrs((x.sum("*", j) == 1 for j in C), name="in")
        model.addConstrs((f.sum("*", i) - f.sum(i, "*") == q[i] for i in C), name="flow")
        for (i, j) in arcs:
            if j == 0:
                model.addConstr(f[i, j] == 0)
            else:
                model.addConstr(f[i, j] >= q[j] * x[i, j])
                model.addConstr(f[i, j] <= (Q - q[i]) * x[i, j])
        model.addConstr(x.sum(0, "*") >= -(-sum(q) // Q), name="min_routes")
        model.setObjective(gp.quicksum(inst.dist(i, j) * x[i, j] for (i, j) in arcs), GRB.MINIMIZE)
        model._x = x  # type: ignore[attr-defined]
        model._arcs = arcs  # type: ignore[attr-defined]
        return model

    @staticmethod
    def violated_sets(n: int, Q: int, q: list[int], flow: dict[tuple[int, int], float], min_viol: float = 0.05) -> list[frozenset[int]]:
        """Customer sets S of 1..n (node 0 = depot, q[0] = 0) whose fractional flow out of S is below ceil(q(S) / Q)."""
        C = range(1, n + 1)
        Y = [[0.0] * (n + 1) for _ in range(n + 1)]
        for (i, j), v in flow.items():
            Y[i][j] += v
            Y[j][i] += v
        found: dict[frozenset[int], float] = {}

        def check(S):
            cross = sum(Y[i][j] for i in S for j in range(n + 1) if j not in S) / 2  # flow leaving S (in = out)
            viol = math.ceil(sum(q[i] for i in S) / Q - 1e-9) - cross
            if viol > min_viol:
                found.setdefault(frozenset(S), viol)

        seen: set[int] = set()
        for i in C:  # components of the customer support graph
            if i in seen:
                continue
            comp, stack = {i}, [i]
            while stack:
                u = stack.pop()
                for v in C:
                    if v not in comp and Y[u][v] > 1e-6:
                        comp.add(v)
                        stack.append(v)
            seen |= comp
            check(comp)
        for i in C:  # greedy growth: add the outside customer most connected to S
            S = {i}
            while len(S) < n:
                j = max((v for v in C if v not in S), key=lambda v: sum(Y[u][v] for u in S))
                if sum(Y[u][j] for u in S) <= 1e-6:
                    break
                S.add(j)
                check(S)
        return [S for S, _ in sorted(found.items(), key=lambda t: -t[1])[:50]]

    def solve_model(self, inst: CVRPInstance, model: gp.Model) -> BaselineSolution[gp.Model]:
        x, arcs = model._x, model._arcs  # type: ignore[attr-defined]
        xs = [x[a] for a in arcs]

        def callback(m, where):
            if where != GRB.Callback.MIPNODE or m.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL or m.cbGet(GRB.Callback.MIPNODE_NODCNT) > 0:
                return
            flow = dict(zip(arcs, m.cbGetNodeRel(xs)))
            for S in self.violated_sets(inst.n_customers, inst.capacity, inst.demand, flow):
                m.cbCut(gp.quicksum(x[i, j] for (i, j) in arcs if i in S and j not in S) >= math.ceil(sum(inst.demand[i] for i in S) / inst.capacity - 1e-9))

        model.Params.PreCrush = 1
        model.optimize(callback)
        if model.Status != GRB.OPTIMAL or math.ceil(model.ObjBound - 1e-6) < model.ObjVal - 1e-6:
            raise RuntimeError(f"Baseline model not proved optimal (status={model.Status}, bound={model.ObjBound}, incumbent={model.ObjVal})")
        used = [(i, j) for (i, j), var in model._x.items() if var.X > 0.5]  # type: ignore[attr-defined]
        succ = {i: j for (i, j) in used if i != 0}
        routes = []
        for start in [j for (i, j) in used if i == 0]:
            route, node = [], start
            while node != 0:
                route.append(node)
                node = succ[node]
            routes.append(route)
        return BaselineSolution(objective=float(model.ObjVal), vars={}, best_bound=float(model.ObjBound), model_state=model, metadata={"routes": routes})
