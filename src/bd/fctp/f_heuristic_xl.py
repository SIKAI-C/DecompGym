"""Primal heuristics for the FCTP shared by the engineered Benders reference and the heuristic-only control
(scale-up study).

- Dynamic slope scaling (Kim and Pardalos 1999): transportation LPs over all arcs with unit costs
  rho_ij = c_ij + f_ij / xbar_ij on arcs that carried flow xbar_ij > 0 in the previous LP (unchanged otherwise),
  starting from Balinski's c_ij + f_ij / m_ij; every LP's arcs with flow form a design, evaluated exactly.
- Basis local search: a basic feasible solution of the transportation polytope is a spanning tree of the bipartite
  graph (m + n - 1 arcs, zero-flow tree arcs allowed). A move enters a non-tree arc (i, j): flow delta goes around the
  cycle closed by the tree path from supplier i to customer j (arcs left from a supplier side lose delta, arcs left
  from a customer side gain delta; delta = the smallest flow on the losing arcs); the exact FCTP cost change is
      delta * (c_ij + sum_{gaining} c - sum_{losing} c) + f_ij - sum_{losing arcs emptied} f + sum_{gaining arcs opened} f.
  One depth-first search per supplier evaluates every move (O(m (m + n)) per scan); best improvement until no move
  improves (degenerate moves with delta = 0 are not used for descent). This is the classic exact neighbourhood of
  FCTP local search (Sun et al. 1998; Klose's FCTP local search; the move set of Buson, Roberti and Toth 2014).
- Iterated local search (the heuristic control's main loop): random kicks (k entering arcs regardless of cost),
  descent, record-to-record acceptance (keep the walk within `accept` of the best, otherwise return to the best).

The heuristic control (`FCTPHeuristicControl`) runs initializer -> slope scaling -> descent -> ILS to the deadline,
without any decomposition; it has the solver interface of the references.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from src.bd.fctp.a_data_xl import FCTPInstance, load_fctp
from src.bd.fctp.b_mip_xl import TransportLP, clean_flows, make_env, matrix_to_flows, plan_cost
from src.bd.fctp.c_naive_bd import initial_plan

INF = float("inf")


# ---------------------------------------------------------------- basis local search

class BasisLS:
    """A basic feasible solution (flows + spanning tree) with exact cost and the pivot neighbourhood."""

    def __init__(self, inst: FCTPInstance, X: np.ndarray) -> None:
        self.inst = inst
        self.m, self.n = inst.m, inst.n
        self.c = inst.tcost.tolist()
        self.f = inst.fcost.tolist()
        self.eps = 0.5 if inst.integral else 1e-9 * max(1.0, float(plan_cost(inst, X)))
        self.X = [[float(v) for v in row] for row in np.asarray(X, dtype=float)]
        self.adj_s: list[set[int]] = [set() for _ in range(self.m)]  # tree customers of supplier i
        self.adj_d: list[set[int]] = [set() for _ in range(self.n)]  # tree suppliers of customer j
        self.moves = 0
        self.scans = 0
        self._make_basis()
        self.cost = self.exact_cost()

    # ------------------------------------------------------------ state

    def exact_cost(self) -> float:
        tot = 0.0
        for i in range(self.m):
            xi, ci, fi = self.X[i], self.c[i], self.f[i]
            for j in range(self.n):
                if xi[j] > 0:
                    tot += fi[j] + ci[j] * xi[j]
        return tot

    def matrix(self) -> np.ndarray:
        return np.array(self.X, dtype=float)

    def snapshot(self) -> tuple:
        return ([row[:] for row in self.X], [set(s) for s in self.adj_s], [set(s) for s in self.adj_d], self.cost)

    def restore(self, snap: tuple) -> None:
        X, adj_s, adj_d, cost = snap
        self.X = [row[:] for row in X]
        self.adj_s = [set(s) for s in adj_s]
        self.adj_d = [set(s) for s in adj_d]
        self.cost = cost

    def tree_arcs(self) -> list[tuple[int, int]]:
        return [(i, j) for i in range(self.m) for j in self.adj_s[i]]

    # ------------------------------------------------------------ tree paths

    def path(self, i: int, j: int) -> list[tuple[int, int, int]] | None:
        """Tree path from supplier i to customer j as arcs (a, b, sign): sign -1 when the arc is left from its supplier
        end (it loses flow when (i, j) enters), +1 when left from its customer end. None if j is not reachable."""
        m = self.m
        parent = {i: None}  # node ids: suppliers a -> a, customers b -> m + b
        stack = [i]
        target = m + j
        while stack:
            node = stack.pop()
            if node == target:
                break
            if node < m:
                for b in self.adj_s[node]:
                    if m + b not in parent:
                        parent[m + b] = node
                        stack.append(m + b)
            else:
                b = node - m
                for a in self.adj_d[b]:
                    if a not in parent:
                        parent[a] = node
                        stack.append(a)
        if target not in parent:
            return None
        arcs = []
        node = target
        while parent[node] is not None:
            prev = parent[node]
            if prev < m:  # supplier prev -> customer node: left from the supplier end
                arcs.append((prev, node - m, -1))
            else:  # customer prev -> supplier node
                arcs.append((node, prev - m, +1))
            node = prev
        arcs.reverse()
        return arcs

    def _push(self, i: int, j: int, arcs: list[tuple[int, int, int]], delta: float) -> None:
        self.X[i][j] += delta
        for a, b, sg in arcs:
            self.X[a][b] += sg * delta

    def _make_basis(self) -> None:
        """Make the support acyclic (cancelling cycles in the cost-nonincreasing direction) and complete it to a
        spanning tree with zero-flow arcs."""
        m, n = self.m, self.n
        support = sorted(((self.X[i][j], i, j) for i in range(m) for j in range(n) if self.X[i][j] > 0), reverse=True)
        for _, i, j in support:
            while self.X[i][j] > 0:
                arcs = self.path(i, j)
                if arcs is None:
                    self.adj_s[i].add(j)
                    self.adj_d[j].add(i)
                    break
                # A cycle: push in the direction that does not increase the variable cost until an arc empties.
                g = self.c[i][j] + sum(sg * self.c[a][b] for a, b, sg in arcs)
                if g <= 0:  # increase (i, j)
                    delta = min(self.X[a][b] for a, b, sg in arcs if sg < 0)
                    self._push(i, j, arcs, delta)
                else:  # decrease (i, j)
                    delta = min([self.X[i][j]] + [self.X[a][b] for a, b, sg in arcs if sg > 0])
                    self._push(i, j, arcs, -delta)
                for a, b, _ in arcs:
                    if self.X[a][b] <= 1e-12:
                        self.X[a][b] = 0.0
                        self.adj_s[a].discard(b)
                        self.adj_d[b].discard(a)
                if self.X[i][j] <= 1e-12:
                    self.X[i][j] = 0.0
        # Complete to a spanning tree: join the component of supplier 0 to another one by its cheapest zero-flow arc.
        cost = self.inst.tcost
        while True:
            comp = np.array(self._components())
            if np.all(comp == comp[0]):
                break
            in0 = comp == comp[0]
            cross = in0[:m, None] != in0[None, m:]
            a, b = np.unravel_index(int(np.argmin(np.where(cross, cost, np.inf))), (m, n))
            self.adj_s[int(a)].add(int(b))
            self.adj_d[int(b)].add(int(a))

    def _components(self) -> list[int]:
        m, n = self.m, self.n
        comp = [-1] * (m + n)
        label = 0
        for start in range(m + n):
            if comp[start] >= 0:
                continue
            comp[start] = label
            stack = [start]
            while stack:
                node = stack.pop()
                nbrs = [m + b for b in self.adj_s[node]] if node < m else list(self.adj_d[node - m])
                for nb in nbrs:
                    if comp[nb] < 0:
                        comp[nb] = label
                        stack.append(nb)
            label += 1
        return comp

    # ------------------------------------------------------------ neighbourhood

    def scan_root(self, i: int, collect: list | None = None) -> tuple[float, int]:
        """Best entering arc (i, j) over all customers j (cost change, j); with `collect`, every j with a
        non-degenerate move is appended."""
        X, c, f = self.X, self.c, self.f
        ci, fi = c[i], f[i]
        adj_s, adj_d = self.adj_s, self.adj_d
        best_g, best_j = INF, -1
        stack = [(i, -1, INF, 0.0, 0.0, 0.0)]  # supplier, parent customer, min losing flow, f at min, f opened, path cost
        while stack:
            a, pb, dm, fm, fz, cs = stack.pop()
            xa, ca, fa = X[a], c[a], f[a]
            root = a == i
            for b in adj_s[a]:
                if b == pb:
                    continue
                x = xa[b]
                if x < dm:
                    ndm, nfm = x, fa[b]
                elif x == dm:
                    ndm, nfm = dm, fm + fa[b]
                else:
                    ndm, nfm = dm, fm
                ncs = cs - ca[b]
                if not root and ndm > 0:
                    g = ndm * (ci[b] + ncs) + fi[b] - nfm + fz
                    if g < best_g:
                        best_g, best_j = g, b
                    if collect is not None:
                        collect.append(b)
                for a2 in adj_d[b]:
                    if a2 != a:
                        x2 = X[a2][b]
                        stack.append((a2, b, ndm, nfm, fz + (f[a2][b] if x2 == 0 else 0.0), ncs + c[a2][b]))
        return best_g, best_j

    def best_move(self) -> tuple[float, int, int]:
        self.scans += 1
        best = (INF, -1, -1)
        for i in range(self.m):
            g, j = self.scan_root(i)
            if g < best[0]:
                best = (g, i, j)
        return best

    def pivot(self, i: int, j: int) -> float:
        """Enter (i, j); returns the exact cost change (a non-degenerate move is expected)."""
        arcs = self.path(i, j)
        delta = min(self.X[a][b] for a, b, sg in arcs if sg < 0)
        before = [(a, b, self.X[a][b]) for a, b, _ in arcs] + [(i, j, self.X[i][j])]
        self._push(i, j, arcs, delta)
        change = 0.0
        for a, b, old in before:
            new = self.X[a][b]
            if new <= 1e-12:
                new = self.X[a][b] = 0.0
            change += self.c[a][b] * (new - old)
            if old == 0 and new > 0:
                change += self.f[a][b]
            elif old > 0 and new == 0:
                change -= self.f[a][b]
        # One emptied losing arc leaves the tree (the one with the largest fixed cost), (i, j) enters.
        leave = max(((a, b) for a, b, sg in arcs if sg < 0 and self.X[a][b] == 0), key=lambda e: self.f[e[0]][e[1]])
        self.adj_s[leave[0]].discard(leave[1])
        self.adj_d[leave[1]].discard(leave[0])
        self.adj_s[i].add(j)
        self.adj_d[j].add(i)
        self.cost += change
        self.moves += 1
        return change

    def descend(self, deadline: float, max_moves: int | None = None) -> float:
        """Best-improvement descent until a local optimum, the deadline or max_moves."""
        moves = 0
        while time.perf_counter() < deadline and (max_moves is None or moves < max_moves):
            g, i, j = self.best_move()
            if i < 0 or g > -self.eps:
                break
            self.pivot(i, j)
            moves += 1
        return self.cost

    def kick(self, rng: np.random.Generator, k: int) -> None:
        """k random non-degenerate entering moves regardless of cost."""
        for _ in range(k):
            for _ in range(10):
                i = int(rng.integers(self.m))
                cand: list[int] = []
                self.scan_root(i, cand)
                if cand:
                    self.pivot(i, int(cand[int(rng.integers(len(cand)))]))
                    break


# ---------------------------------------------------------------- slope scaling

def slope_scaling(inst: FCTPInstance, env, X0: np.ndarray, deadline: float, steps: int,
                  on_design: Callable[[np.ndarray], None]) -> int:
    """Dynamic slope scaling from the flows X0 (normally Balinski's plan); calls on_design(design) for every new
    design (arcs with flow); stops when a design repeats, after `steps` LPs or at the deadline. Returns the LP count."""
    rho = inst.tcost + inst.fcost / inst.cap
    xbar = np.asarray(X0, dtype=float)
    lp = TransportLP(inst, env, cost=rho)
    seen = {(xbar > 0).tobytes()}
    count = 0
    try:
        while count < steps and time.perf_counter() < deadline:
            rho = np.where(xbar > 0, inst.tcost + inst.fcost / np.maximum(xbar, 1e-12), rho)
            lp.set_cost(rho)
            res = lp.solve(inst.cap, deadline - time.perf_counter())
            if res["status"] != "optimal":
                break
            count += 1
            xbar = clean_flows(inst, res["flows"])
            design = xbar > 0
            key = design.tobytes()
            if key in seen:
                break
            seen.add(key)
            on_design(design)
    finally:
        lp.dispose()
    return count


# ---------------------------------------------------------------- iterated local search

def iterated_local_search(ls: BasisLS, deadline: float, rng: np.random.Generator, on_improve: Callable[[np.ndarray], None],
                          k_range: tuple[int, int] = (2, 8), accept: float = 0.002) -> dict[str, int]:
    """Random kicks + descent with record-to-record acceptance; on_improve(X) for every new best plan."""
    ls.descend(deadline)
    best = ls.snapshot()
    on_improve(ls.matrix())
    stats = {"ils_iterations": 0, "ils_improvements": 0}
    while time.perf_counter() < deadline:
        ls.kick(rng, int(rng.integers(k_range[0], k_range[1] + 1)))
        ls.descend(deadline)
        stats["ils_iterations"] += 1
        if ls.cost < best[3] - ls.eps:
            best = ls.snapshot()
            stats["ils_improvements"] += 1
            on_improve(ls.matrix())
        elif ls.cost > best[3] * (1.0 + accept):
            ls.restore(best)
    ls.restore(best)
    return stats


# ---------------------------------------------------------------- heuristic-only control

class FCTPHeuristicControl:
    """Initializer (Balinski) -> slope scaling -> basis descent -> iterated local search to the deadline; no
    decomposition. Every plan is published with its exact cost; no bound is claimed."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 ss_steps: int = 30, ss_fraction: float = 0.05, accept: float = 0.002, verbose: bool = False) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.on_incumbent = on_incumbent or (lambda flows, stage=None: None)
        self.on_bound = on_bound or (lambda bound: None)
        self.ss_steps, self.ss_fraction, self.accept = ss_steps, ss_fraction, accept
        self.verbose = verbose

    def run(self, data_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        inst = load_fctp(data_path)
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        state = {"ub": INF, "X": None}
        trace: list[dict] = []

        def offer(X: np.ndarray, stage: str) -> None:
            cost = plan_cost(inst, X)
            if cost < state["ub"] - 1e-9 and time.perf_counter() < deadline:
                state.update(ub=cost, X=X.copy())
                self.on_incumbent(matrix_to_flows(X), stage=stage)
                trace.append({"t": round(time.perf_counter() - t0, 3), "incumbent": cost, "stage": stage})
                if self.verbose:
                    print(trace[-1], flush=True)

        ev = TransportLP(inst, env)
        stats: dict[str, Any] = {"ss_lps": 0}
        reason = "time_limit"
        try:
            init = initial_plan(inst, env, deadline - time.perf_counter())
            if init is not None:
                offer(init["X"], "initializer")

                def on_design(design: np.ndarray) -> None:
                    res = ev.evaluate(design, max(0.01, deadline - time.perf_counter()))
                    if res["status"] == "optimal":
                        offer(res["X"], "slope_scaling")

                on_design(init["X"] > 0)
                stats["ss_lps"] = slope_scaling(inst, env, init["X"], min(deadline, t0 + self.ss_fraction * self.time_limit),
                                                self.ss_steps, on_design)
                if time.perf_counter() < deadline:
                    ls = BasisLS(inst, state["X"])
                    rng = np.random.default_rng(self.seed)
                    stats.update(iterated_local_search(ls, deadline, rng, lambda X: offer(X, "local_search"), accept=self.accept))
                    stats.update(moves=ls.moves, scans=ls.scans)
                if time.perf_counter() < deadline - 1.0:
                    reason = "finished_early"
        finally:
            ev.dispose()
            env.dispose()
        return {"best_objective": state["ub"] if state["X"] is not None else None, "global_lb": None,
                "termination_reason": reason, "budget_exhausted": time.perf_counter() - t0 >= self.time_limit - 1.0,
                "flows": matrix_to_flows(state["X"]) if state["X"] is not None else None, **stats,
                "total_seconds": time.perf_counter() - t0, "trace": trace}
