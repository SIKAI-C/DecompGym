"""Decomposition-free CLRP control (protocol v4 Section 6, "no grandfathering": CLRP had only a weak control).

A location-allocation + routing iterated local search, Python 3 + NumPy only (no LP, MIP or column generation):

1. Start: depots opened by increasing opening cost per unit of capacity until their capacity covers the demand;
   customers allocated by regret to the open depots and routed by Clarke-Wright savings + 2-opt
   (c_heuristic.plan_for), polished by the engineered reference's local search (d_ref_cg_xl.LocalSearch: relocate,
   swap, 2-opt*, route-to-depot moves; capacities kept).
2. Location phase (until `location_share` of the budget): the neighbourhood of the current depot set S (add one
   depot, drop one, swap one used for one unused; only sets whose capacity covers the demand) is SCREENED by a warm
   start from the best plan (routes of dropped depots dissolved; for an added depot, the customers closer to it than
   to their current depot moved; all of them reinserted by cheapest feasible insertion restricted to the new set) and
   the local search; the `screen_top` best screened sets are then each EVALUATED by a short ruin-and-recreate run
   (src.cg.clrp.f_sisr_xl, `eval_seconds`) restricted to the set. The first evaluated set that beats the best plan is
   accepted and the neighbourhood is rebuilt; when a whole neighbourhood fails, a random kick (one or two depot
   flips of the best set) is evaluated the same way, and the evaluation time doubles (up to 4x).
3. Routing phase (rest of the budget): ruin-and-recreate from the best plan over all depots (a route may still move
   to an unused depot, paying its opening cost; emptying a depot saves it), with the local search polishing new best
   plans at most every `polish_every` seconds.
Every improving plan is re-checked with b_mip_xl.route_cost and published as [{"depot", "customers" (0..n-1)}]
(rate-limited; the search stops up to 0.5 s before the deadline so the pending best is always flushed; never at or
after the deadline). Warm starts are offered before and the result after every evaluation.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np

from src.cg.clrp.a_data import CLRPInstance, load_prodhon
from src.cg.clrp.b_mip_xl import route_cost
from src.cg.clrp.c_heuristic import plan_for
from src.cg.clrp.d_ref_cg_xl import LocalSearch
from src.cg.clrp.f_sisr_xl import RoutingModel, RuinRecreate, copy_routes, insert_all
from src.cg.cvrp.f_ils_xl import Publisher


def routing_model(inst: CLRPInstance, cost: np.ndarray) -> RoutingModel:
    return RoutingModel(cost.astype(float), inst.demand.astype(float), inst.vehicle_capacity, inst.m,
                        W=inst.depot_capacity.astype(float), F=float(inst.vehicle_cost), O=inst.opening_cost.astype(float))


def to_output(m: int, routes: list) -> list[dict[str, Any]]:
    return [{"depot": int(d), "customers": [int(v - m) for v in p]} for d, p in routes if p]


def from_plan(m: int, plan: list[dict[str, Any]]) -> list:
    return [[int(r["depot"]), [m + int(i) for i in r["customers"]]] for r in plan if r["customers"]]


def start_depots(inst: CLRPInstance) -> list[int]:
    order = sorted(range(inst.m), key=lambda d: inst.opening_cost[d] / inst.depot_capacity[d])
    out, cap = [], 0.0
    for d in order:
        out.append(d)
        cap += inst.depot_capacity[d]
        if cap >= inst.demand.sum():
            break
    return sorted(out)


class CLRPLocationRoutingILS:
    """The decomposition-free CLRP control of the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 location_share: float = 0.6, eval_seconds: float | None = None, screen_top: int = 3,
                 T0: float = 100.0, Tf: float = 1.0, polish_every: float = 2.0, publish_every: float = 0.25) -> None:
        self.time_limit, self.threads, self.seed = time_limit, threads, seed
        self.on_incumbent = on_incumbent or (lambda routes: None)
        self.location_share, self.screen_top = location_share, screen_top
        self.eval_seconds = eval_seconds if eval_seconds is not None else min(10.0, max(2.0, 0.01 * time_limit))
        self.T0, self.Tf, self.polish_every, self.publish_every = T0, Tf, polish_every, publish_every

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        publish_deadline = t0 + self.time_limit
        deadline = publish_deadline - min(0.5, 0.05 * self.time_limit)   # search deadline; the reserve is for the flush
        inst = load_prodhon(data_path)
        cost = inst.travel("ceil")
        m, demand = inst.m, float(inst.demand.sum())
        model = routing_model(inst, cost)
        ls = LocalSearch(inst, cost)
        rng = np.random.default_rng(self.seed)
        pub = Publisher(self.on_incumbent, publish_deadline, self.publish_every, stage="lrp_ils")
        best: dict[str, Any] = {"obj": math.inf, "routes": None, "t": None, "first": None}
        stats = {"screened": 0, "evaluated": 0, "location_moves": 0, "kicks": 0, "sisr_iterations": 0}

        def offer(routes: list) -> bool:
            plan = to_output(m, routes)
            obj = route_cost(inst, cost, plan)
            if obj is None or obj >= best["obj"] - 1e-9 or time.perf_counter() >= publish_deadline:
                return False
            best.update(obj=obj, routes=copy_routes(routes), t=time.perf_counter() - t0)
            if best["first"] is None:
                best["first"] = best["t"]
            pub.offer(plan)
            return True

        def polish(routes: list, dl: float) -> list:
            plan = [(d, [v - m for v in p]) for d, p in routes if p]
            out = ls.improve(plan, min(dl, time.perf_counter() + 5.0))
            return [[d, [m + i for i in p]] for d, p in out if p]

        def used(routes: list) -> list[int]:
            return sorted({d for d, p in routes if p})

        def warm(routes: list, S: list[int]) -> list | None:
            Sset = set(S)
            keep = [[d, list(p)] for d, p in routes if d in Sset and p]
            todo = [v for d, p in routes if d not in Sset for v in p]
            new = [a for a in S if a not in set(used(routes))]
            if new:
                moved = set()
                for r in keep:
                    d, p = r
                    stay = []
                    for v in p:
                        if any(cost[a, v] < cost[d, v] for a in new):
                            moved.add(v)
                        else:
                            stay.append(v)
                    r[1] = stay
                keep = [r for r in keep if r[1]]
                todo += sorted(moved)
            todo.sort(key=lambda v: -model.q[v])
            out = insert_all(model, keep, todo, S)
            if out is None:
                return None
            pol = polish(out, deadline)
            return pol if model.feasible(pol, Sset) and model.cost(pol) <= model.cost(out) else out

        def evaluate(start: list, S: list[int], seconds: float) -> tuple[list, float]:
            rr = RuinRecreate(model, seed=int(rng.integers(2**31)), T0=self.T0, Tf=self.Tf)
            dl = min(deadline, time.perf_counter() + seconds)
            offer(start)   # the warm start itself may already beat the best plan (Codex review CG-1 #3)
            routes, c = rr.run(start, dl, allowed=S, on_best=lambda r, _: offer(r), tick=pub.tick)
            offer(routes)
            stats["sisr_iterations"] += rr.iterations
            stats["evaluated"] += 1
            return routes, c

        # 1. start
        S0 = start_depots(inst)
        plan = plan_for(inst, cost, S0)
        routes = from_plan(m, plan) if plan else None
        if routes is None or not model.feasible(routes, set(S0)):
            routes = insert_all(model, [], sorted(range(m, m + inst.n), key=lambda v: -model.q[v]), list(range(m)))
        if routes is None:
            raise RuntimeError("no feasible start plan")
        offer(routes)
        pol = polish(routes, deadline)
        if model.feasible(pol) and model.cost(pol) < model.cost(routes):
            routes = pol
        offer(routes)

        # 2. location phase
        loc_deadline = t0 + self.location_share * self.time_limit
        seconds = self.eval_seconds
        fails = 0
        while time.perf_counter() < loc_deadline:
            S = used(best["routes"])
            cap = lambda T: float(inst.depot_capacity[list(T)].sum()) >= demand if T else False
            closed = [d for d in range(m) if d not in S]
            moves = [sorted(S + [a]) for a in closed]
            moves += [[d for d in S if d != b] for b in S if len(S) > 1]
            moves += [sorted([d for d in S if d != b] + [a]) for b in S for a in closed]
            moves = [T for T in moves if cap(T)]
            screened = []
            for T in moves:
                if time.perf_counter() >= loc_deadline:
                    break
                w = warm(best["routes"], T)
                stats["screened"] += 1
                if w is not None:
                    screened.append((model.cost(w), T, w))
                    offer(w)
            screened.sort(key=lambda x: x[0])
            improved = False
            for _, T, w in screened[:self.screen_top]:
                if time.perf_counter() >= loc_deadline:
                    break
                before = best["obj"]
                evaluate(w, T, seconds)
                if best["obj"] < before - 1e-9:
                    improved = True
                    stats["location_moves"] += 1
                    break
            if improved:
                fails, seconds = 0, self.eval_seconds
                continue
            fails += 1
            seconds = min(4 * self.eval_seconds, 2 * seconds)
            # kick: flip one or two depots of the best set
            for _ in range(20):
                T = set(S)
                for d in rng.choice(m, size=int(rng.integers(1, min(2, m) + 1)), replace=False):
                    T ^= {int(d)}
                T = sorted(T)
                if cap(T) and T != S:
                    break
            else:
                break   # no admissible kick (e.g. only one feasible depot set): the routing phase starts early
            w = warm(best["routes"], T)
            stats["kicks"] += 1
            if w is not None and time.perf_counter() < loc_deadline:
                evaluate(w, T, seconds)

        # 3. routing phase over all depots
        if time.perf_counter() < deadline:
            rr = RuinRecreate(model, seed=self.seed, T0=self.T0, Tf=self.Tf)
            rr.run(best["routes"], deadline, on_best=lambda r, _: offer(r), polish=polish,
                   polish_every=self.polish_every, tick=pub.tick)
            stats["sisr_iterations"] += rr.iterations
        pub.flush()
        return {"best_objective": best["obj"] if best["routes"] else None, "routes": to_output(m, best["routes"] or []),
                "termination_reason": "time_limit", "first_incumbent_seconds": best["first"], "best_seconds": best["t"],
                "open_depots": used(best["routes"] or []), "publications": pub.published, "seed": self.seed, **stats,
                "total_seconds": time.perf_counter() - t0}
