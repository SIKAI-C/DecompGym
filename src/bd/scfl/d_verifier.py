"""Independent checker for SCFL outputs (standard library only): {"y": [0/1 per facility], ...}.

The solution is the first-stage design y; its value is f.y + sum_s p_s Q_s(y), where Q_s(y) is the optimal cost of
scenario s's transportation problem over the open facilities (demands met exactly, loads <= capacity). Every Q_s is
computed here exactly, without a solver, by the scale-up verifier's min-cost-flow algorithm (copied unchanged):
  start from each customer's cheapest open facility (optimal if capacities are ignored; customer potentials
  pi_j = min_i c_ij, facility potentials 0), then repair overloaded facilities by successive shortest augmenting
  paths (Dijkstra on reduced costs) from facilities with spare capacity; potentials are updated after each path, so
  every residual arc keeps a nonnegative reduced cost and the final flow is optimal (complementary slackness).
With integer demands and capacities every augmentation is integral. A design whose open capacity is below some
scenario's total demand is infeasible. The instance is parsed here, independently of the solver code.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path
from typing import Any


def _load(obj: Any) -> dict[str, Any]:
    if isinstance(obj, (str, Path)):
        return json.loads(Path(obj).read_text())
    if isinstance(obj, dict):
        return obj
    raise TypeError("expected dict or path-like input")


def _result(ok: bool, message: str, objective_value: float | None = None) -> dict[str, Any]:
    return {"ok": ok, "issues": [] if ok else [message], "objective_value": objective_value}


def transport(fac: list[int], u: list[float], c: list[list[float]], dem: list[float]) -> tuple[float, dict] | None:
    """Exact optimal transportation cost over the open facilities `fac` (None if infeasible) and the flows
    {(i, j): q}. See the module docstring for the algorithm."""
    J = len(dem)
    total = math.fsum(dem)
    eps = 1e-9 * max(1.0, total)
    if math.fsum(u[i] for i in fac) < total - eps:
        return None
    if total <= eps:
        return 0.0, {}
    K = len(fac)
    C = [c[i] for i in fac]
    U = [u[i] for i in fac]
    flow: list[dict[int, float]] = [{} for _ in range(K)]
    sup: list[set[int]] = [set() for _ in range(J)]  # open facilities serving customer j
    load = [0.0] * K
    pc = [0.0] * J
    for j in range(J):
        a = min(range(K), key=lambda k: C[k][j])
        pc[j] = C[a][j]
        if dem[j] > 0:
            flow[a][j] = dem[j]
            sup[j].add(a)
            load[a] += dem[j]
    pf = [0.0] * K
    inflow = [min(load[a], U[a]) for a in range(K)]
    deficit = [max(0.0, load[a] - U[a]) for a in range(K)]
    INF = float("inf")
    guard = 0
    while any(x > eps for x in deficit):
        guard += 1
        if guard > 50 * (K + J) ** 2:
            raise RuntimeError("min-cost flow did not terminate")
        distF, distC = [INF] * K, [INF] * J
        prevF, prevC = [-2] * K, [-1] * J
        doneF, doneC = [False] * K, [False] * J
        heap: list[tuple[float, int]] = []
        for a in range(K):
            if U[a] - inflow[a] > eps:
                r = -pf[a]  # reduced cost of the source arc (cost 0, source potential 0)
                if r < distF[a]:
                    distF[a], prevF[a] = r, -1
                    heapq.heappush(heap, (r, a))
        reached = -1
        while heap:
            dist, node = heapq.heappop(heap)
            if node < K:
                a = node
                if doneF[a] or dist > distF[a]:
                    continue
                doneF[a] = True
                if deficit[a] > eps:
                    reached = a
                    break
                base = dist + pf[a]
                Ca = C[a]
                for j in range(J):
                    if not doneC[j]:
                        nd = base + Ca[j] - pc[j]
                        if nd < distC[j]:
                            distC[j], prevC[j] = nd, a
                            heapq.heappush(heap, (nd, K + j))
            else:
                j = node - K
                if doneC[j] or dist > distC[j]:
                    continue
                doneC[j] = True
                base = dist + pc[j]
                for b in sup[j]:
                    if not doneF[b]:
                        nd = base - C[b][j] - pf[b]
                        if nd < distF[b]:
                            distF[b], prevF[b] = nd, j
                            heapq.heappush(heap, (nd, b))
        if reached < 0:
            return None
        dt = distF[reached]
        for a in range(K):
            pf[a] += min(distF[a], dt)
        for j in range(J):
            pc[j] += min(distC[j], dt)
        # Path back from the overloaded facility to the source; delta limited by the deficit, the backward flows and
        # the start facility's spare capacity.
        delta = deficit[reached]
        steps = []
        node = reached
        while True:
            j = prevF[node]
            if j == -1:
                delta = min(delta, U[node] - inflow[node])
                start = node
                break
            delta = min(delta, flow[node][j])
            a = prevC[j]
            steps.append((node, j, a))
            node = a
        if delta <= 0:
            raise RuntimeError("zero augmentation")
        inflow[start] += delta
        for b, j, a in steps:  # customer j moves delta from facility b to facility a
            q = flow[b][j] - delta
            if q <= eps:
                del flow[b][j]
                sup[j].discard(b)
            else:
                flow[b][j] = q
            load[b] -= delta
            flow[a][j] = flow[a].get(j, 0.0) + delta
            sup[j].add(a)
            load[a] += delta
        deficit[reached] = max(0.0, deficit[reached] - delta)
    out = {(fac[a], j): q for a in range(K) for j, q in flow[a].items() if q > 0}
    return math.fsum(c[i][j] * q for (i, j), q in out.items()), out


def verify_solution(data: Any, output: Any, tol: float | None = None) -> dict[str, Any]:
    _ = tol
    try:
        d, output = _load(data), _load(output)
        I, J, S = int(d["n_facilities"]), int(d["n_customers"]), int(d["n_scenarios"])
        u = [float(v) for v in d["capacity"]]
        f = [float(v) for v in d["fixed_cost"]]
        c = [[float(v) for v in row] for row in d["unit_cost"]]
        dem = [[float(v) for v in row] for row in d["demand"]]
        p = [float(v) for v in d["probability"]]
        if len(u) != I or len(f) != I or len(c) != I or any(len(r) != J for r in c) or len(dem) != S or any(len(r) != J for r in dem) or len(p) != S:
            raise ValueError("inconsistent instance dimensions")
    except Exception as exc:
        return _result(False, f"data load error: {exc}")
    y = output.get("y") if isinstance(output, dict) else None
    if not isinstance(y, list) or len(y) != I or not all(type(v) in (int, float) and v in (0, 1) for v in y):
        return _result(False, f"y must be a list of {I} values in {{0, 1}}")
    fac = [i for i in range(I) if y[i] == 1]
    dmax = max(math.fsum(row) for row in dem)
    if math.fsum(u[i] for i in fac) < dmax - 1e-9 * max(1.0, dmax):
        return _result(False, f"open capacity {math.fsum(u[i] for i in fac)} is below the largest scenario demand {dmax}")
    q = []
    for s in range(S):
        res = transport(fac, u, c, dem[s])
        if res is None:
            return _result(False, f"scenario {s} is infeasible for the design")
        q.append(res[0])
    return _result(True, "ok", math.fsum(f[i] for i in fac) + math.fsum(ps * qs for ps, qs in zip(p, q)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an SCFL solution.")
    parser.add_argument("--dir", type=Path, default=Path("."))
    parser.add_argument("--output", default="output.json")
    args = parser.parse_args()
    print(json.dumps(verify_solution(args.dir / "data.json", args.dir / args.output), indent=2))


if __name__ == "__main__":
    main()
