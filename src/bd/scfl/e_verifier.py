"""Independent verifier for SCFL solutions (scfl-v1 JSON instances, scale-up study; standard library only).

Output contract: {"status": "optimal" | "feasible", "method": str, "objective_value": number,
                  "y": [0/1 per facility],
                  optionally "flows": [[scenario, facility, customer, value], ...] (0-based; repeats are summed)}.
The solution is the first-stage design y. Its value is f.y + sum_s p_s Q_s(y), where Q_s(y) is the optimal cost of
the scenario's transportation problem over the open facilities (demands d_sj met exactly, loads <= u_i). This
verifier computes every Q_s itself, exactly, with a min-cost-flow algorithm written here (no solver):
  start from each customer's cheapest open facility (optimal if capacities are ignored; customer potentials
  pi_j = min_i c_ij, facility potentials 0), then repair overloaded facilities by successive shortest augmenting
  paths (Dijkstra on reduced costs) from facilities with spare capacity; potentials are updated after each path, so
  every residual arc keeps a nonnegative reduced cost and the final flow is optimal (complementary slackness).
With integer demands and capacities every augmentation is integral, so the flows are exact.
Checks: y binary of length I; (15d) sum_i u_i y_i >= max_s D_s (otherwise some scenario is infeasible); the claimed
objective must match the recomputed value within tol * max(1, |value|); a missing method or an unknown status is
rejected. If flows are given they must be feasible (nonnegative, only from open facilities, loads within capacity,
demands met exactly, as in the extensive form) and their scenario costs cannot be below the computed optimum (a self-check of this verifier); the
flows' expected cost is reported but the score is the design value. The instance file is parsed here, independently
of the solver code.
"""
from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from typing import Any


def read_instance(path: str | Path) -> dict[str, Any]:
    obj = json.loads(Path(path).read_text())
    if obj.get("format") != "scfl-v1":
        raise ValueError("not an scfl-v1 instance")
    u = [float(v) for v in obj["capacity"]]
    f = [float(v) for v in obj["fixed_cost"]]
    c = [[float(v) for v in row] for row in obj["unit_cost"]]
    d = [[float(v) for v in row] for row in obj["demand"]]
    I, S = len(u), len(d)
    J = len(c[0]) if c else 0
    if len(f) != I or len(c) != I or any(len(r) != J for r in c) or any(len(r) != J for r in d):
        raise ValueError("inconsistent instance dimensions")
    p = [float(v) for v in obj["probability"]] if obj.get("probability") is not None else [1.0 / S] * S
    if len(p) != S or any(v <= 0 for v in p) or abs(math.fsum(p) - 1.0) > 1e-9:
        raise ValueError("invalid scenario probabilities")
    return {"I": I, "J": J, "S": S, "u": u, "f": f, "c": c, "d": d, "p": p}


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


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_index(v: Any, n: int) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v < n


def verify_solution(data_path: str | Path, output_path: str | Path, tol: float = 1e-6) -> dict[str, Any]:
    d = read_instance(data_path)
    I, J, S = d["I"], d["J"], d["S"]
    try:
        out = json.loads(Path(output_path).read_text())
    except (OSError, ValueError) as exc:
        return {"ok": False, "issues": [f"unreadable output: {exc}"]}
    if not isinstance(out, dict):
        return {"ok": False, "issues": ["output must be a JSON object"]}
    issues: list[str] = []
    y = out.get("y")
    if not isinstance(y, list) or len(y) != I or not all(_is_num(v) and v in (0, 1) for v in y):
        return {"ok": False, "issues": [f"y must be a list of {I} values in {{0, 1}}"]}
    fac = [i for i in range(I) if y[i] == 1]
    dmax = max(math.fsum(row) for row in d["d"])
    if math.fsum(d["u"][i] for i in fac) < dmax - 1e-9 * max(1.0, dmax):
        return {"ok": False, "issues": [f"open capacity {math.fsum(d['u'][i] for i in fac)} below the largest scenario "
                                        f"demand {dmax}: some scenario is infeasible"]}
    q = []
    for s in range(S):
        res = transport(fac, d["u"], d["c"], d["d"][s])
        if res is None:
            return {"ok": False, "issues": [f"scenario {s} is infeasible for the design"]}
        q.append(res[0])
    expected = math.fsum(ps * qs for ps, qs in zip(d["p"], q))
    total = math.fsum(d["f"][i] for i in fac) + expected
    result: dict[str, Any] = {"objective_value": total, "expected_recourse": expected, "open": fac}
    if out.get("flows") is not None:
        flows = out["flows"]
        X: list[dict[tuple[int, int], float]] = [{} for _ in range(S)]
        bad = None
        if not isinstance(flows, list):
            bad = "flows must be a list"
        else:
            for e in flows:
                if (not isinstance(e, (list, tuple)) or len(e) != 4 or not _is_index(e[0], S) or not _is_index(e[1], I)
                        or not _is_index(e[2], J) or not _is_num(e[3]) or not math.isfinite(e[3])):
                    bad = f"invalid flow entry {e}"
                    break
                X[e[0]][(e[1], e[2])] = X[e[0]].get((e[1], e[2]), 0.0) + float(e[3])
        if bad:
            issues.append(bad)
        else:
            costs = []
            for s in range(S):
                load = [0.0] * I
                got = [0.0] * J
                for (i, j), v in X[s].items():
                    if v < -tol:
                        issues.append(f"negative flow in scenario {s} on ({i}, {j})")
                    if v > tol and y[i] != 1:
                        issues.append(f"flow from closed facility {i} in scenario {s}")
                    load[i] += v
                    got[j] += v
                if any(load[i] > d["u"][i] + tol * max(1.0, d["u"][i]) for i in range(I)):
                    issues.append(f"capacity exceeded in scenario {s}")
                if any(abs(got[j] - d["d"][s][j]) > tol * max(1.0, d["d"][s][j]) for j in range(J)):
                    issues.append(f"demand not met exactly in scenario {s}")
                cs = math.fsum(d["c"][i][j] * v for (i, j), v in X[s].items())
                if cs < q[s] - tol * max(1.0, abs(q[s])):
                    issues.append(f"flows of scenario {s} cost {cs} < computed optimum {q[s]} (verifier self-check)")
                costs.append(cs)
                if len(issues) > 20:
                    break
            if len(costs) == S:
                result["flows_expected_cost"] = math.fsum(ps * cs for ps, cs in zip(d["p"], costs))
    claimed = out.get("objective_value")
    if not _is_num(claimed) or not math.isfinite(claimed) or abs(claimed - total) > tol * max(1.0, abs(total)):
        issues.append(f"objective mismatch or non-finite: claimed {claimed}, recomputed {total}")
    if out.get("status") not in ("optimal", "feasible"):
        issues.append("status must be 'optimal' or 'feasible' for a returned design")
    if not isinstance(out.get("method"), str) or not out.get("method"):
        issues.append("method missing")
    return {"ok": not issues, "issues": issues, **result}
