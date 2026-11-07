"""RCSP duty-plan evaluator and compact MIP (scale-up study).

Evaluator. `duty_check(inst, depot, tasks)` rebuilds the duty's connections (SOURCE depot -> first task, one
CONNECTION per consecutive pair, SINK last task -> depot; any missing connection makes the duty infeasible) and checks
the labour rules of a_data's docstring: duty length (sum of the connection durations) <= 9.5 h, and some break
connection a with before(a) = (durations before a) + timeBeforeBreak(a) <= 5.5 h and after(a) = timeAfterBreak(a) +
(durations after a) <= 5.5 h. The cost is the sum of the connection costs. `plan_cost(inst, duties)` adds coverage
(every task in at least one duty) and returns the total, or None. On all 200 published best-known solutions it
reproduces the benchmark's best-known costs exactly (tests/large_scale/test_rcsp.py).

Compact MIP: a start-indexed layered duty-flow model ("duty flow"), our strongest compact model (the benchmark paper
compares branch-and-price with a crew-indexed multi-commodity flow model in the style of Hoffmann and Buscher, which
needs an a-priori bound on the number of duties and found feasible solutions on 10 of its 25 smallest instances in
3 h). One commodity per SOURCE connection sigma = (crew base d, first task s); on time-consistent instances (all
benchmark instances, see a_data) a duty of sigma has elapsed time w(k) = end(k) - start(s) at every task k, so the
rules become windows. Commodity sigma has
- a pre-break layer: tasks k reachable from s with w(k) <= 5.5 h and w(k) + (shortest time to a break start) <= 5.5 h,
  joined by every CONNECTION;
- designated-break arcs: break connections a = (i, j) from the pre-break layer with w(i) + timeBeforeBreak(a) <= 5.5 h
  into the post-break layer (exactly one per duty: the layers force it);
- a post-break layer: tasks k with w(k) + (shortest completion to a SINK at d) <= 9.5 h and w(k) <= (latest break end
  of sigma) + 5.5 h, joined by every CONNECTION, and SINK arcs to d from it;
- integer flows x >= 0 with flow conservation per (sigma, layer, task); covering rows sum of the flows entering task t
  (all commodities and layers) >= 1;
- the after-break rule: with e(a) = w(j) - timeAfterBreak(a) (break end) and E(k) = w(k) + SINK duration, for every
  threshold theta in {e(a)}:  sum_{designated a: e(a) <= theta} x_a + sum_{sinks k: E(k) > theta + 5.5 h} x_k <= f_sigma
  (f_sigma = the SOURCE flow), written with cumulative chain variables so that each arc appears once.
Exactness: duty length and the before-break limit hold on every path by construction. A commodity carrying one duty
(f_sigma = 1, a single path) satisfies the after-break rule iff the chain rows hold, so such commodities are exact. When
several duties start with the same SOURCE (f_sigma >= 2; common in single-depot solutions, where crews ride together
from the depot), the chain rows only say that the multiset of break ends and duty ends admits a feasible pairing; the
paths of the integer flow may pair them differently. The model is therefore a relaxation: its bound is valid, and every
integer solution is decoded (a feasible path decomposition is searched per commodity) and re-checked with plan_cost
before it is published; an undecodable solution is not an incumbent.
Size: about (number of SOURCE connections) x (connections inside a 9.5 h window) x 2 integer variables, e.g. 0.3 M for
s6_t482 single, 1.8 M for s6_t482 multi, 2.1 M for s9_t991 single and 4.4 M / 22 M for s14_t1500 single / multi before
reachability pruning (see `mip_size`); in practice it can be built and solved on the tiny tier only.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.cg.rcsp.a_data import RCSPInstance, load_rcsp


def duty_check(inst: RCSPInstance, depot: int, tasks: list[int]) -> tuple[float | None, str]:
    """(cost, "") for a feasible duty, (None, reason) otherwise. Rules: see a_data's module docstring."""
    if not tasks:
        return None, "empty duty"
    if depot not in inst.depots:
        return None, f"station {depot} is not a crew base"
    conns = []
    c = inst.source_of.get((depot, tasks[0]))
    if c is None:
        return None, f"no SOURCE from crew base {depot} to task {tasks[0]}"
    conns.append(c)
    for a, b in zip(tasks, tasks[1:]):
        c = inst.arc_of.get((a, b))
        if c is None:
            return None, f"no connection from task {a} to task {b}"
        conns.append(c)
    c = inst.sink_of.get((tasks[-1], depot))
    if c is None:
        return None, f"no SINK from task {tasks[-1]} to crew base {depot}"
    conns.append(c)
    dur = [int(inst.c_dur[c]) for c in conns]
    length = sum(dur)
    if length > inst.max_duty:
        return None, f"duty length {length} > {inst.max_duty}"
    before = 0
    ok_break = False
    for p, c in enumerate(conns):
        tbb = int(inst.c_tbb[c])
        if tbb >= 0 and before + tbb <= inst.max_work and int(inst.c_tab[c]) + (length - before - dur[p]) <= inst.max_work:
            ok_break = True
            break
        before += dur[p]
    if not ok_break:
        return None, "no break connection with at most 5.5 h of work before and after the break"
    return float(sum(int(inst.c_cost[c]) for c in conns)), ""


def plan_issues(inst: RCSPInstance, duties: list[dict[str, Any]]) -> tuple[float | None, list[str]]:
    issues: list[str] = []
    covered = np.zeros(inst.n, dtype=bool)
    total = 0.0
    for k, d in enumerate(duties):
        try:
            depot, tasks = int(d["depot"]), [int(t) for t in d["tasks"]]
        except (KeyError, TypeError, ValueError):
            issues.append(f"duty {k}: malformed")
            continue
        if any(not 0 <= t < inst.n for t in tasks):
            issues.append(f"duty {k}: task index out of range")
            continue
        cost, why = duty_check(inst, depot, tasks)
        if cost is None:
            issues.append(f"duty {k}: {why}")
            continue
        total += cost
        covered[tasks] = True
    missing = np.nonzero(~covered)[0]
    if len(missing):
        issues.append(f"{len(missing)} tasks uncovered (first: {missing[:5].tolist()})")
    return (None if issues else total), issues


def normalize_plan(duties: list) -> list[dict[str, Any]]:
    return [{"depot": int(d["depot"]), "tasks": [int(t) for t in d["tasks"]]} for d in duties]


def rcsp_output(inst: RCSPInstance, duties: list, status: str = "feasible", method: str = "decomposition") -> dict[str, Any]:
    """The output contract checked by e_verifier.verify_solution."""
    duties = normalize_plan(duties)
    return {"status": status, "method": method, "objective_value": plan_cost(inst, duties), "duties": duties}


def plan_cost(inst: RCSPInstance, duties: list[dict[str, Any]]) -> float | None:
    """Total cost of a duty plan [{"depot": station, "tasks": [task, ...]}], or None if a duty breaks a labour rule
    or a task is uncovered."""
    return plan_issues(inst, duties)[0]


# ---------------------------------------------------------------- compact MIP

PRE, BRK, POST, SNK = 0, 1, 2, 3


def _reach(n: int, tail: np.ndarray, head: np.ndarray, allowed: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Nodes reachable from `start` (bool mask) over the allowed arcs (frontier sweeps)."""
    seen = start.copy()
    t, h = tail[allowed], head[allowed]
    while True:
        new = np.zeros(n, dtype=bool)
        new[h[seen[t]]] = True
        new &= ~seen
        if not new.any():
            return seen
        seen |= new


def commodity_arcs(inst: RCSPInstance, minw: np.ndarray, minb: np.ndarray, d_pos: dict[int, int], snk_dur: np.ndarray,
                   snk_ok: np.ndarray, src_index: int) -> dict[str, Any] | None:
    """The layered network of one SOURCE connection (see the module docstring), as arc index arrays."""
    n, L, LB = inst.n, float(inst.max_duty), float(inst.max_work)
    c = int(inst.src_conn[src_index])
    d, s = int(inst.src_depot[src_index]), int(inst.src_task[src_index])
    di = d_pos[d]
    S = float(inst.start[s])
    w = inst.end.astype(float) - S
    at, ah = inst.arc_tail, inst.arc_head
    tbb, tab, brk = inst.arc_tbb.astype(float), inst.arc_tab.astype(float), inst.arc_break
    pre_ok = (w <= LB) & (w + minb <= LB) & (w >= w[s])
    if not pre_ok[s]:
        return None
    start = np.zeros(n, dtype=bool)
    start[s] = True
    pre = _reach(n, at, ah, pre_ok[at] & pre_ok[ah], start)
    post_cand = (w <= L) & (w + minw[di] <= L)
    b = brk & pre[at] & (w[at] + tbb <= LB) & (tab <= LB) & post_cand[ah]
    if not b.any():
        return None
    e = w[ah] - tab
    maxbe = float(e[b].max())
    post_ok = post_cand & (w <= maxbe + LB)
    b &= post_ok[ah]
    if not b.any():
        return None
    hs = np.zeros(n, dtype=bool)
    hs[ah[b]] = True
    post = _reach(n, at, ah, post_ok[at] & post_ok[ah], hs)
    sinks = post & snk_ok[di] & (w + snk_dur[di] <= L) & (w + snk_dur[di] - maxbe <= LB)
    # Backward pruning: post nodes that reach a sink; pre nodes that reach a designated tail.
    post_b = _reach(n, ah, at, post[at] & post[ah], sinks)
    b &= post_b[ah]
    if not b.any():
        return None
    tails = np.zeros(n, dtype=bool)
    tails[at[b]] = True
    pre_b = _reach(n, ah, at, pre[at] & pre[ah], tails) & pre
    if not pre_b[s]:
        return None
    pre_arcs = np.nonzero(pre_b[at] & pre_b[ah])[0]
    brk_arcs = np.nonzero(b & pre_b[at])[0]
    hs = np.zeros(n, dtype=bool)
    hs[ah[brk_arcs]] = True
    post_f = _reach(n, at, ah, post_b[at] & post_b[ah], hs) & post_b
    post_arcs = np.nonzero(post_f[at] & post_f[ah])[0]
    sink_tasks = np.nonzero(sinks & post_f)[0]
    if not len(sink_tasks):
        return None
    return {"conn": c, "depot": d, "first": s, "w": w, "pre": pre_arcs, "brk": brk_arcs, "post": post_arcs,
            "sinks": sink_tasks, "e": e[brk_arcs], "E": w[sink_tasks] + snk_dur[di, sink_tasks]}


def build_rcsp_mip(inst: RCSPInstance, env: gp.Env | None = None, deadline: float = math.inf) -> tuple[gp.Model, dict[str, Any]]:
    """Build the start-indexed layered duty-flow MIP (module docstring). Returns (model, info); info["vars"] lists
    the integer variables, decode_plan(info, values) turns their values into a duty plan (or None)."""
    from src.cg.rcsp.pricing import DutyPricer

    if not inst.time_consistent:
        raise ValueError("the compact MIP needs time-consistent connection durations (see a_data)")
    pr = DutyPricer(inst)
    n, LB = inst.n, float(inst.max_work)
    snk_ok = np.isfinite(pr.snk_cost)
    comms = []
    for k in range(len(inst.src_conn)):
        if time.perf_counter() > deadline:
            raise TimeoutError("deadline while building the compact MIP")
        cm = commodity_arcs(inst, pr.MINW, pr.MINB, pr.dpos, pr.snk_dur, snk_ok, k)
        if cm is not None:
            comms.append(cm)
    # Variable table: kind, commodity, connection or task, tail node id, head node id (node id = (q, layer, task)).
    kinds, comm_of, ref, tails, heads, costs, cov = [], [], [], [], [], [], []
    for q, cm in enumerate(comms):
        base = q * 2 * n
        at, ah = inst.arc_tail, inst.arc_head
        m = len(cm["pre"]) + len(cm["brk"]) + len(cm["post"]) + len(cm["sinks"]) + 1
        kinds.append(np.concatenate([[4], np.full(len(cm["pre"]), PRE), np.full(len(cm["brk"]), BRK),
                                     np.full(len(cm["post"]), POST), np.full(len(cm["sinks"]), SNK)]))
        comm_of.append(np.full(m, q))
        ref.append(np.concatenate([[cm["conn"]], cm["pre"], cm["brk"], cm["post"], cm["sinks"]]))
        tails.append(np.concatenate([[-1], base + at[cm["pre"]], base + at[cm["brk"]], base + n + at[cm["post"]], base + n + cm["sinks"]]))
        heads.append(np.concatenate([[base + cm["first"]], base + ah[cm["pre"]], base + n + ah[cm["brk"]], base + n + ah[cm["post"]], [-1] * len(cm["sinks"])]))
        di = pr.dpos[cm["depot"]]
        costs.append(np.concatenate([[float(inst.c_cost[cm["conn"]])], inst.arc_cost[cm["pre"]], inst.arc_cost[cm["brk"]],
                                     inst.arc_cost[cm["post"]], pr.snk_cost[di, cm["sinks"]]]))
        cov.append(np.concatenate([[cm["first"]], ah[cm["pre"]], ah[cm["brk"]], ah[cm["post"]], [-1] * len(cm["sinks"])]))
    if not comms:
        raise ValueError("no commodity has a feasible duty")
    kinds = np.concatenate(kinds).astype(np.int64)
    comm_of = np.concatenate(comm_of).astype(np.int64)
    ref = np.concatenate(ref).astype(np.int64)
    tails = np.concatenate(tails).astype(np.int64)
    heads = np.concatenate(heads).astype(np.int64)
    costs = np.concatenate(costs).astype(float)
    cov = np.concatenate(cov).astype(np.int64)
    N = len(kinds)
    model = gp.Model(env=env) if env is not None else gp.Model()
    x = model.addVars(N, lb=0.0, vtype=GRB.INTEGER, obj=costs.tolist())
    xs = [x[i] for i in range(N)]
    # Covering rows.
    o = np.argsort(cov, kind="stable")
    o = o[cov[o] >= 0]
    bounds = np.searchsorted(cov[o], np.arange(n + 1))
    for t in range(n):
        idx = o[bounds[t]:bounds[t + 1]]
        if len(idx) == 0:
            raise ValueError(f"task {t} cannot be covered by the compact model")
        model.addLConstr(gp.LinExpr([1.0] * len(idx), [xs[i] for i in idx.tolist()]), GRB.GREATER_EQUAL, 1.0)
    # Flow conservation per (commodity, layer, task) node: inflow - outflow = 0.
    node = np.concatenate([heads, tails])
    coef = np.concatenate([np.ones(N), -np.ones(N)])
    var = np.concatenate([np.arange(N), np.arange(N)])
    keep = node >= 0
    node, coef, var = node[keep], coef[keep], var[keep]
    o = np.argsort(node, kind="stable")
    node, coef, var = node[o], coef[o], var[o]
    cut = np.nonzero(np.diff(node))[0] + 1
    for a, bnd in zip(np.concatenate([[0], cut]).tolist(), np.concatenate([cut, [len(node)]]).tolist()):
        if time.perf_counter() > deadline:
            raise TimeoutError("deadline while building the compact MIP")
        model.addLConstr(gp.LinExpr(coef[a:bnd].tolist(), [xs[i] for i in var[a:bnd].tolist()]), GRB.EQUAL, 0.0)
    # After-break chain rows per commodity.
    start_of = np.searchsorted(comm_of, np.arange(len(comms) + 1))
    chain_rows = 0
    for q, cm in enumerate(comms):
        lo = int(start_of[q])
        f = xs[lo]
        kq = kinds[lo:start_of[q + 1]]
        bvars = [xs[lo + i] for i in np.nonzero(kq == BRK)[0].tolist()]
        svars = [xs[lo + i] for i in np.nonzero(kq == SNK)[0].tolist()]
        e, E = cm["e"], cm["E"]
        theta = np.unique(e)
        if E.max() <= theta.min() + LB:
            continue  # every pairing is feasible
        U = model.addVars(len(theta), lb=0.0)
        V = model.addVars(len(theta), lb=0.0)
        eb = np.searchsorted(theta, e)  # threshold index of each designated arc
        sb = np.searchsorted(theta + LB, E, side="left") - 1  # sink counted in V_i for i <= sb (E > theta_i + LB)
        for i in range(len(theta)):
            members = [bvars[j] for j in np.nonzero(eb == i)[0].tolist()]
            expr = gp.LinExpr([-1.0] * len(members), members)
            expr.addTerms(1.0, U[i])
            if i:
                expr.addTerms(-1.0, U[i - 1])
            model.addLConstr(expr, GRB.EQUAL, 0.0)
            members = [svars[j] for j in np.nonzero(sb == i)[0].tolist()]
            expr = gp.LinExpr([-1.0] * len(members), members)
            expr.addTerms(1.0, V[i])
            if i + 1 < len(theta):
                expr.addTerms(-1.0, V[i + 1])
            model.addLConstr(expr, GRB.EQUAL, 0.0)
            model.addLConstr(U[i] + V[i] - f, GRB.LESS_EQUAL, 0.0)
            chain_rows += 3
    model.ModelSense = GRB.MINIMIZE
    model.Params.LazyConstraints = 1
    model.update()
    info = {"vars": xs, "kinds": kinds, "comm_of": comm_of, "ref": ref, "comms": comms, "inst": inst, "start_of": start_of,
            "num_vars": N, "chain_rows": chain_rows, "commodities": len(comms)}
    return model, info


def _decompose(inst: RCSPInstance, cm: dict[str, Any], flows: dict[tuple[int, int], int], budget: int = 20000) -> list[list[int]] | None:
    """Split one commodity's integer flow into duties satisfying the after-break rule (depth-first search)."""
    LB = float(inst.max_work)
    f = flows.get((4, -1), 0)
    out: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for (kind, r), v in flows.items():
        if kind == PRE:
            out.setdefault((0, int(inst.arc_tail[r])), []).append((kind, r, int(inst.arc_head[r])))
        elif kind == BRK:
            out.setdefault((0, int(inst.arc_tail[r])), []).append((kind, r, int(inst.arc_head[r])))
        elif kind == POST:
            out.setdefault((1, int(inst.arc_tail[r])), []).append((kind, r, int(inst.arc_head[r])))
        elif kind == SNK:
            out.setdefault((1, r), []).append((kind, r, -1))
    left = dict(flows)
    steps = [0]
    E_of = {int(t): float(E) for t, E in zip(cm["sinks"], cm["E"])}
    e_of = {int(a): float(e) for a, e in zip(cm["brk"], cm["e"])}

    def path(layer: int, v: int, tasks: list[int], be: float | None, remaining: int) -> list[list[int]] | None:
        steps[0] += 1
        if steps[0] > budget:
            return None
        for kind, r, h in out.get((layer, v), []):
            key = (kind, r)
            if left.get(key, 0) <= 0:
                continue
            if kind == SNK:
                if E_of[r] - be > LB + 1e-9:
                    continue
                left[key] -= 1
                rest = start(remaining - 1)
                if rest is not None:
                    return [tasks] + rest
                left[key] += 1
                continue
            left[key] += -1
            nb = e_of[r] if kind == BRK else be
            res = path(1 if kind in (BRK, POST) else 0, h, tasks + [h], nb, remaining)
            left[key] += 1
            if res is not None:
                return res
        return None

    def start(remaining: int) -> list[list[int]] | None:
        if remaining == 0:
            return [] if all(v == 0 for k, v in left.items() if k != (4, -1)) else None
        return path(0, cm["first"], [cm["first"]], None, remaining)

    return start(f)


def decode_plan(info: dict[str, Any], values) -> list[dict[str, Any]] | None:
    """Duty plan of an integer solution (values aligned with info["vars"]), or None if some commodity's flow has no
    decomposition into feasible duties."""
    inst = info["inst"]
    vals = np.rint(np.asarray(values, dtype=float)).astype(np.int64)
    plan = []
    start_of = info["start_of"]
    for q, cm in enumerate(info["comms"]):
        lo, hi = int(start_of[q]), int(start_of[q + 1])
        if vals[lo] <= 0:
            continue
        flows: dict[tuple[int, int], int] = {(4, -1): int(vals[lo])}
        nz = np.nonzero(vals[lo + 1:hi] > 0)[0] + lo + 1
        for i in nz.tolist():
            flows[(int(info["kinds"][i]), int(info["ref"][i]))] = int(vals[i])
        duties = _decompose(inst, cm, flows)
        if duties is None:
            return None
        plan += [{"depot": cm["depot"], "tasks": t} for t in duties]
    return plan


def separate_pairing_cuts(info: dict[str, Any], values, max_cuts: int = 50) -> list[tuple[list[int], list[int]]]:
    """Violated pairing inequalities of an integer solution, as (lhs variable indices, rhs variable indices) with
    sum lhs <= sum rhs. For a commodity sigma, a node set W of its post-break layer and a threshold theta, every duty of
    sigma that uses a designated break a with e(a) <= theta and head(a) in W either ends in W at a SINK with
    E <= theta + 5.5 h or leaves W on a post-break connection, so
        sum_{a: e(a) <= theta, head(a) in W} x_a  <=  sum_{k in W: E(k) <= theta + 5.5 h} x_k + sum_{(u, v): u in W, v not in W} x_uv
    is valid for every feasible plan. W is taken as the set of post-break tasks reachable in the solution's support
    from the heads of the used designated breaks with e <= theta (then the last sum is 0 at the solution)."""
    inst = info["inst"]
    LB = float(inst.max_work)
    vals = np.rint(np.asarray(values, dtype=float)).astype(np.int64)
    kinds, ref, start_of = info["kinds"], info["ref"], info["start_of"]
    at, ah = inst.arc_tail, inst.arc_head
    cuts: list[tuple[list[int], list[int]]] = []
    for q, cm in enumerate(info["comms"]):
        lo, hi = int(start_of[q]), int(start_of[q + 1])
        if vals[lo] < 2:
            continue  # a single duty: the chain rows are exact
        kq = kinds[lo:hi]
        idx = np.arange(lo, hi)
        b_idx, p_idx, k_idx = idx[kq == BRK], idx[kq == POST], idx[kq == SNK]
        b_head, b_e = ah[ref[b_idx]], cm["e"]
        p_tail, p_head = at[ref[p_idx]], ah[ref[p_idx]]
        k_task, k_E = ref[k_idx], cm["E"]
        used_b = vals[b_idx] > 0
        if not used_b.any():
            continue
        supp = vals[p_idx] > 0
        for theta in np.unique(b_e[used_b]):
            seen = np.zeros(inst.n, dtype=bool)
            seen[b_head[used_b & (b_e <= theta)]] = True
            while True:
                new = np.zeros(inst.n, dtype=bool)
                new[p_head[supp & seen[p_tail]]] = True
                new &= ~seen
                if not new.any():
                    break
                seen |= new
            lhs = b_idx[(b_e <= theta) & seen[b_head]]
            rhs_k = k_idx[seen[k_task] & (k_E <= theta + LB + 1e-9)]
            rhs_p = p_idx[seen[p_tail] & ~seen[p_head]]
            if vals[lhs].sum() > vals[rhs_k].sum() + vals[rhs_p].sum():
                cuts.append((lhs.tolist(), rhs_k.tolist() + rhs_p.tolist()))
                break
        if len(cuts) >= max_cuts:
            break
    return cuts


def mip_size(inst: RCSPInstance) -> dict[str, int]:
    """Variable counts of the compact MIP without building it."""
    from src.cg.rcsp.pricing import DutyPricer

    pr = DutyPricer(inst)
    snk_ok = np.isfinite(pr.snk_cost)
    tot = {"commodities": 0, "pre": 0, "brk": 0, "post": 0, "sinks": 0}
    for k in range(len(inst.src_conn)):
        cm = commodity_arcs(inst, pr.MINW, pr.MINB, pr.dpos, pr.snk_dur, snk_ok, k)
        if cm is not None:
            tot["commodities"] += 1
            for key in ("pre", "brk", "post", "sinks"):
                tot[key] += len(cm[key])
    tot["vars"] = tot["commodities"] + tot["pre"] + tot["brk"] + tot["post"] + tot["sinks"]
    return tot


class RCSPCompactMIP:
    """The compact MIP with the solver interface of the decomposition references (incumbents are decoded and
    checked with plan_cost; bounds are Gurobi's best bounds, rounded up: the model is a relaxation)."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0, method: int = 2) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.method = method  # root relaxation: 2 = barrier (default; much faster than dual simplex on this model)
        self.on_incumbent = on_incumbent or (lambda duties: None)
        self.on_bound = on_bound or (lambda bound: None)

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        inst = load_rcsp(data_path)
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", self.threads)
        env.setParam("Seed", self.seed)
        if self.soft_mem_gb:
            env.setParam("SoftMemLimit", self.soft_mem_gb)
        env.start()
        best = {"obj": math.inf, "duties": None, "bound": -math.inf, "undecodable": 0, "cuts": 0}
        try:
            model, info = build_rcsp_mip(inst, env, deadline)
        except TimeoutError:
            env.dispose()
            return {"best_objective": None, "global_lb": None, "termination_reason": "deadline_during_build",
                    "duties": None, "build_seconds": time.perf_counter() - t0, "total_seconds": time.perf_counter() - t0}
        build = time.perf_counter() - t0
        xs = info["vars"]

        def offer(values) -> None:
            plan = decode_plan(info, values)
            if plan is None:
                best["undecodable"] += 1
                return
            offer_plan(plan)

        def offer_plan(plan) -> None:
            obj = plan_cost(inst, plan)
            if obj is not None and obj < best["obj"] - 1e-9:
                best.update(obj=obj, duties=plan)
                self.on_incumbent(plan)

        def bound(value: float) -> None:
            if math.isfinite(value) and value > -1e90:
                b = float(math.ceil(value - 1e-6 * max(1.0, abs(value))))
                if b > best["bound"]:
                    best["bound"] = b
                    self.on_bound(b)

        def callback(m, where):
            if time.perf_counter() > deadline:
                m.terminate()  # Gurobi's TimeLimit can overshoot by many seconds in barrier/crossover on big models
                return
            if where == GRB.Callback.MIPSOL:
                values = m.cbGetSolution(xs)
                plan = decode_plan(info, values)
                if plan is None:
                    best["undecodable"] += 1
                    for lhs, rhs in separate_pairing_cuts(info, values):
                        m.cbLazy(gp.LinExpr([1.0] * len(lhs), [xs[i] for i in lhs]) - gp.LinExpr([1.0] * len(rhs), [xs[i] for i in rhs]) <= 0)
                        best["cuts"] += 1
                else:
                    offer_plan(plan)
            elif where == GRB.Callback.MIP:
                bound(m.cbGet(GRB.Callback.MIP_OBJBND))

        reason = "deadline"
        if deadline - time.perf_counter() > 0.01:
            model.Params.TimeLimit = deadline - time.perf_counter()
            model.Params.Method = self.method
            model.optimize(callback)
            if model.SolCount:
                offer(model.getAttr("X", xs))
            if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.MEM_LIMIT):
                bound(float(model.ObjBound))
            reason = {GRB.OPTIMAL: "finished", GRB.TIME_LIMIT: "time_limit", GRB.MEM_LIMIT: "memory_limit"}.get(model.Status, f"status_{model.Status}")
        out = {"best_objective": best["obj"] if best["duties"] else None, "global_lb": best["bound"] if best["bound"] > -math.inf else None,
               "termination_reason": reason, "duties": best["duties"], "build_seconds": build, "num_vars": info["num_vars"],
               "num_constrs": model.NumConstrs, "commodities": info["commodities"], "undecodable_solutions": best["undecodable"],
               "pairing_cuts": best["cuts"],
               "node_count": float(model.NodeCount) if reason != "deadline" else 0.0, "total_seconds": time.perf_counter() - t0}
        model.dispose()
        env.dispose()
        return out
