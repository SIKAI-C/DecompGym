"""UFL at scale, bound side (protocol v4, Sections 3 and 6): the arms of the bound-side screen and the compressed
certificate format, for Koerkel-Ghosh-style instances with n = m in the thousands. Phoenix only (no solver runs on
the laptop). Development code for the 11-03 bound-side probes; the paper's UFL family is untouched.

View: the canonical "ufl-v1" of src/experiments/xl_certificate.ufl_view (read-only use): first stage y_i in [0, 1],
cost f_i, one row 'cover' sum_i y_i >= 1; one subproblem per customer j (id 'c<j>', weight 1) with rows 'd'
(sum_i x_ij = 1) and 'l<i>' (x_ij <= y_i), boxes 0 <= x_ij <= 1. Its deterministic equivalent is the compact LP
    min f.y + sum_ij D_ij x_ij  s.t.  sum_i x_ij = 1,  x_ij <= y_i,  sum_i y_i >= 1,  0 <= x, y <= 1.

Cuts (closed form, Fischetti, Ljubic and Sinnl 2017): for customer j and any v, the dual (pi_d = v, pi_l<i> =
-(v - D_ij)^+) gives  theta_j >= v - sum_i (v - D_ij)^+ y_i, valid for every y in [0, 1]^m with sum y >= 1. At a
point y the oracle fills the facilities in increasing D_ij up to y-mass 1; v_j = the critical cost is an optimal
dual and Q_j(y) = v_j - sum_i (v_j - D_ij)^+ y_i.

Compressed certificate "ufl-cert-c1" (transport compression of the canonical format; the size rule counts the
expanded entries 1 + #{i: D_ij < v} per term):
    {"format": "ufl-cert-c1", "view": "ufl-v1", "version": k, "singles": {"j": [...], "v": [...]},
     "multi": [{"ids": [...], "v": [...]}, ...]}
A single (j, v) is the one-term cut above; a multi cut is the sum of its terms over distinct customers.
`to_generic` writes the canonical certificate (xl_certificate format).

Arms (all with the same interface as the k-median arms: time_limit, threads, on_incumbent(open list, stage),
on_bound(float), on_certificate(bytes)):
  UFLEngLP        engineered Benders, LP phase: disaggregated per-customer cuts, in-out stabilization, dual simplex
                  re-solves; certificate = positive-dual cuts (largest dual first, within the size limit);
  UFLNaiveLP      naive LP-master Benders (unstabilized Kelley, ONE aggregated cut per iteration, same oracle and
                  initializer); certificate = the master-dual combination of its aggregated cuts (one multi cut,
                  which reproduces the master value) then the positive-dual aggregated cuts;
  UFLCompactLP    compact LP (barrier with crossover off, or dual simplex), 1 thread, Gurobi MemLimit; certificate
                  = the assignment-row duals v_j as cuts (j, v_j) (Pi, or BarPi of an incomplete barrier);
  UFLStatic       trivial certificates: 'empty', 'oneshot_greedy' (cuts at the initializer's design);
  UFLPool         no-feedback cut pool (initializer, uniform point, random designs and fractional points);
  UFLLagrangian   subgradient Lagrangian (relaxing the assignment rows; diagnostic): snapshots of u as cuts (j, u_j);
  UFLILS          heuristic-only primal control (add/drop/swap + ILS, src/bd/ufl/b_ref_bd_xl) that also publishes the
                  one-shot certificate of its best design;
  UFLNaiveInt     naive integer-master Benders (primal-side naive baseline);
  UFLCompactMIP   compact MIP (binary y, continuous x), primal baseline;
  UFLEngPrimal    the engineered branch-and-Benders-cut reference src/bd/ufl/b_ref_bd_xl.UFLBendersXL (primal side).
Shared initializer (every Benders arm and the static controls): greedy add with incremental gains from the best
single facility.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from src.bd.scfl.b_mip_xl import is_memory_error, make_env

INF = float("inf")
VIEW_ID = "ufl-v1"
FORMAT = "ufl-cert-c1"
CERT_LIMIT = 2_000_000


# ---------------------------------------------------------------- instances

class UFLInst:
    def __init__(self, f: np.ndarray, D: np.ndarray, name: str = "", meta: dict | None = None) -> None:
        self.f = np.asarray(f, dtype=float)
        self.D = np.asarray(D, dtype=float)
        self.name, self.meta = name, dict(meta or {})
        self.m, self.n = self.D.shape
        if self.f.shape != (self.m,):
            raise ValueError("f must have m entries")
        self.integral = bool(np.all(self.D == np.round(self.D)) and np.all(self.f == np.round(self.f)))


def load_instance(path: str | Path) -> UFLInst:
    obj = json.loads(Path(path).read_text())
    D = np.asarray(obj["D"], dtype=float)
    return UFLInst(np.asarray(obj["f"], dtype=float), D, obj.get("name") or Path(path).stem, obj.get("meta"))


KG_FIXED = {"a": (100, 200), "b": (1000, 2000), "c": (10000, 20000)}


def make_kg(N: int, fclass: str, seed: int, symmetric: bool = True) -> dict:
    """Koerkel-Ghosh-style instance (Koerkel 1989; Ghosh 2003; the UflLib gs/ga classes) at n = m = N: integer
    costs D_ij ~ U{1000..2000} (symmetric: D = D^T, upper triangle drawn), fixed costs ~ U{f_lo..f_hi} by class
    (a: 100-200, b: 1000-2000, c: 10000-20000); numpy default_rng(seed), D first, then f. The keys Xc and Xf are
    placeholders so that src/bd/ufl/a_gen_data.UFLInstance can read the file."""
    rng = np.random.default_rng(seed)
    D = rng.integers(1000, 2001, size=(N, N), dtype=np.int64)
    if symmetric:
        D = np.triu(D) + np.triu(D, 1).T
    lo, hi = KG_FIXED[fclass]
    f = rng.integers(lo, hi + 1, size=N, dtype=np.int64)
    name = f"kg{'s' if symmetric else 'a'}{N}{fclass}_g{seed}"
    return {"name": name, "m": N, "n": N, "f": f.tolist(), "D": D.tolist(), "Xc": [[0, 0]] * N, "Xf": [[0, 0]] * N,
            "meta": {"generator": "Koerkel-Ghosh style (make_kg): D ~ U{1000..2000}, f ~ U by class; numpy default_rng(seed)",
                     "class": fclass, "symmetric": symmetric, "seed": seed, "N": N}}


def write_instance(obj: dict, path: str | Path) -> str:
    text = json.dumps(obj, separators=(",", ":"))
    Path(path).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def design_value(inst: UFLInst, design) -> float | None:
    idx = sorted({int(i) for i in design})
    if not idx or idx[0] < 0 or idx[-1] >= inst.m:
        return None
    return float(inst.f[idx].sum() + inst.D[idx].min(axis=0).sum())


def open_list(y: np.ndarray) -> list[int]:
    return [int(i) for i in np.nonzero(np.asarray(y) > 0.5)[0]]


# ---------------------------------------------------------------- oracle

class Sorted:
    """Per customer, facilities in increasing cost (ties by index): order (n, m) int32, ds (n, m) float64."""

    def __init__(self, D: np.ndarray) -> None:
        DT = np.ascontiguousarray(D.T)
        self.order = np.argsort(DT, axis=1, kind="stable").astype(np.int32)
        self.ds = np.take_along_axis(DT, self.order, axis=1)
        self.n, self.m = DT.shape
        self.d1 = self.ds[:, 0].copy()

    def counts(self, j: np.ndarray, v: np.ndarray) -> np.ndarray:
        """#{i: D_ij < v} for the cuts (j, v) (the expanded size of a term is 1 + this)."""
        j = np.asarray(j, dtype=np.int64)
        v = np.asarray(v, dtype=float)
        out = np.zeros(len(j), dtype=np.int64)
        for a in range(0, len(j), 2048):
            b = min(len(j), a + 2048)
            out[a:b] = (self.ds[j[a:b]] < v[a:b, None]).sum(axis=1)
        return out


class Oracle:
    def __init__(self, inst: UFLInst, srt: Sorted | None = None) -> None:
        self.inst = inst
        self.sc = srt or Sorted(inst.D)
        self.m, self.n = inst.m, inst.n
        self.calls, self.seconds = 0, 0.0

    def separate(self, y: np.ndarray, tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(v, Q, cnt) at y (sum y >= 1): optimal duals, subproblem values, support sizes #{i: D_ij < v_j}."""
        t = time.perf_counter()
        sc = self.sc
        Y = np.asarray(y, dtype=float)[sc.order]
        C = np.cumsum(Y, axis=1)
        hit = C >= 1.0 - tol
        crit = np.argmax(hit, axis=1)
        crit[~hit[:, -1]] = self.m - 1
        rows = np.arange(sc.n)
        v = sc.ds[rows, crit]
        below = sc.ds < v[:, None]
        cnt = below.sum(axis=1)
        Q = v - (np.where(below, v[:, None] - sc.ds, 0.0) * Y).sum(axis=1)
        del Y, C, hit, below
        self.calls += 1
        self.seconds += time.perf_counter() - t
        return v, Q, cnt

    def cut_matrix(self, js: np.ndarray, v: np.ndarray, cnt: np.ndarray) -> sp.csr_matrix:
        sc = self.sc
        js = np.asarray(js, dtype=np.int64)
        c = np.asarray(cnt, dtype=np.int64)
        vv = np.asarray(v, dtype=float)
        r = np.repeat(np.arange(len(js)), c)
        t = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
        jj = js[r]
        return sp.csr_matrix((vv[r] - sc.ds[jj, t], (r, sc.order[jj, t])), shape=(len(js), self.m))

    def aggregate(self, v: np.ndarray, cnt: np.ndarray) -> tuple[float, np.ndarray]:
        A = self.cut_matrix(np.arange(self.n), v, cnt)
        return float(math.fsum(v.tolist())), np.asarray(A.sum(axis=0)).ravel()

    def cut_values(self, v: np.ndarray, cnt: np.ndarray, y: np.ndarray) -> np.ndarray:
        sc = self.sc
        Y = np.asarray(y, dtype=float)[sc.order]
        mask = np.arange(self.m)[None, :] < cnt[:, None]
        return v - (np.where(mask, v[:, None] - sc.ds, 0.0) * Y).sum(axis=1)


def greedy_design(inst: UFLInst) -> np.ndarray:
    """Shared basic initializer: the best single facility, then greedy add (largest saving minus fixed cost) with
    incremental gain updates while the net gain is positive."""
    D, f = inst.D, inst.f
    m, n = D.shape
    y = np.zeros(m, dtype=bool)
    i0 = int(np.argmin(f + D.sum(axis=1)))
    y[i0] = True
    cur = D[i0].copy()
    gain = np.maximum(cur[None, :] - D, 0.0).sum(axis=1)
    while True:
        g = np.where(y, -INF, gain - f)
        i = int(np.argmax(g))
        if g[i] <= 1e-9:
            break
        y[i] = True
        ch = np.nonzero(D[i] < cur)[0]
        if len(ch):
            old, new = cur[ch], D[i, ch]
            for a in range(0, len(ch), 256):
                b = min(len(ch), a + 256)
                sub = D[:, ch[a:b]]
                gain -= (np.maximum(old[None, a:b] - sub, 0.0) - np.maximum(new[None, a:b] - sub, 0.0)).sum(axis=1)
            cur[ch] = new
    return y


# ---------------------------------------------------------------- compressed certificates

def cert_bytes(single_j=(), single_v=(), multi=(), version: Any = None) -> bytes:
    obj = {"format": FORMAT, "view": VIEW_ID, "version": version,
           "singles": {"j": [int(j) for j in single_j], "v": [float(x) for x in single_v]},
           "multi": [{"ids": [int(j) for j in ids], "v": [float(x) for x in v]} for ids, v in multi]}
    return json.dumps(obj, separators=(",", ":")).encode()


def fit(sizes: np.ndarray, limit: int = CERT_LIMIT) -> np.ndarray:
    """Indices of the longest prefix (in the given priority order) whose total size fits the limit."""
    return np.nonzero(np.cumsum(np.asarray(sizes, dtype=np.int64)) <= limit)[0]


def to_generic(data: bytes, inst: UFLInst, srt: Sorted, version: Any = None) -> bytes:
    """Canonical (xl_certificate) certificate equivalent to a ufl-cert-c1 version (implied completions expanded)."""
    obj = json.loads(data)
    if obj.get("format") != FORMAT:
        raise ValueError("format_mismatch")

    def term(j: int, v: float) -> dict:
        c = int(np.searchsorted(srt.ds[j], v, side="left"))
        pi = {"d": v}
        for i, d in zip(srt.order[j, :c].tolist(), srt.ds[j, :c].tolist()):
            pi[f"l{i}"] = -(v - d)
        return {"s": f"c{j}", "pi": pi}

    cuts = [{"kind": "opt", "terms": [term(int(j), float(v))]} for j, v in zip(obj["singles"]["j"], obj["singles"]["v"])]
    for c in obj.get("multi", []):
        cuts.append({"kind": "opt", "terms": [term(int(j), float(v)) for j, v in zip(c["ids"], c["v"])]})
    out = {"view": VIEW_ID, "version": obj.get("version") if version is None else version, "cuts": cuts}
    return json.dumps(out, separators=(",", ":")).encode()


def expanded_size(data: bytes, srt: Sorted) -> int:
    obj = json.loads(data)
    j, v = np.asarray(obj["singles"]["j"], dtype=np.int64), np.asarray(obj["singles"]["v"], dtype=float)
    s = int(len(j) + srt.counts(j, v).sum()) if len(j) else 0
    for c in obj.get("multi", []):
        s += len(c["ids"]) + int(srt.counts(np.asarray(c["ids"]), np.asarray(c["v"])).sum())
    return s


# ---------------------------------------------------------------- exact LP_hi (protocol 3.4)

def exact_master_value(inst: UFLInst, srt: Sorted, y_float, bits: int = 40) -> tuple[Fraction | None, dict]:
    """Exact objective f.y + sum_j Q_j(y) of an exactly feasible compact-LP point: y rounded to the dyadic grid
    2^-bits, clipped to [0, 1], topped up to sum y >= 1, and each customer's exact greedy fill. It is >= LP*
    (the point is feasible), so it is a rigorous LP_hi. Integer arithmetic on the grid (exact for integral data)."""
    if not inst.integral:
        return None, {"reason": "non-integral data"}
    K = 1 << bits
    y = np.clip(np.asarray(y_float, dtype=float), 0.0, 1.0)
    Y = [int(round(v * K)) for v in y.tolist()]
    tot = sum(Y)
    if tot < K:
        for i in sorted(range(inst.m), key=lambda i: -Y[i]):
            add = min(K - Y[i], K - tot)
            Y[i] += add
            tot += add
            if tot >= K:
                break
    if tot < K:
        return None, {"reason": "cover"}
    fi = [int(v) for v in inst.f.tolist()]
    total = sum(a * b for a, b in zip(fi, Y))
    order, ds = srt.order, srt.ds
    for j in range(inst.n):
        rem = K
        oj, dj = order[j], ds[j]
        t = 0
        while rem > 0:
            i = int(oj[t])
            take = Y[i] if Y[i] < rem else rem
            if take:
                total += int(dj[t]) * take
                rem -= take
            t += 1
    return Fraction(total, K), {"bits": bits, "sum_y": float(Fraction(tot, K))}


# ---------------------------------------------------------------- shared bookkeeping

class _Base:
    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, on_certificate: Callable | None = None, soft_mem_gb: float | None = None,
                 seed: int = 0, verbose: bool = False, cert_interval: float = 5.0, cert_limit: int = CERT_LIMIT,
                 mem_limit_gb: float | None = None) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed, self.verbose = time_limit, threads, soft_mem_gb, seed, verbose
        self.mem_limit_gb = mem_limit_gb
        self.on_incumbent = on_incumbent or (lambda design, stage=None: None)
        self.on_bound = on_bound or (lambda bound: None)
        self.on_certificate = on_certificate or (lambda cert: None)
        self.cert_interval, self.cert_limit = cert_interval, cert_limit
        self._last_cert, self._cert_seq, self._cert_key = -INF, 0, None
        self.trace: list[dict[str, Any]] = []

    def start(self, data_path: str) -> None:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        self.inst = load_instance(data_path)
        self.ub, self.lb, self.best = INF, -INF, None

    def left(self) -> float:
        return self.deadline - time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def offer(self, y: np.ndarray, value: float, stage: str) -> bool:
        if value < self.ub - 1e-9 * max(1.0, abs(value)) and self.left() > 0:
            self.ub, self.best = value, np.asarray(y, dtype=bool).copy()
            self.on_incumbent(open_list(self.best), stage=stage)
            return True
        return False

    def publish_bound(self, value: float, stage: str) -> None:
        value = value - 1e-9 * max(1.0, abs(value))
        if value > self.lb and self.left() > 0:
            self.lb = value
            self.on_bound(value)
            self.trace.append({"t": round(self.elapsed(), 3), "bound": value, "stage": stage})

    def emit(self, data_fn: Callable[[int], bytes], force: bool = False, key: Any = None) -> bool:
        now = time.perf_counter()
        if now >= self.deadline or (key is not None and key == self._cert_key):
            return False
        if not force and now - self._last_cert < self.cert_interval:
            return False
        self._cert_seq += 1
        self.on_certificate(data_fn(self._cert_seq))
        self._last_cert, self._cert_key = time.perf_counter(), key
        return True

    def init_cuts(self, oracle: Oracle) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initializer design, and its cuts (v, cnt) plus the lower-bound cuts (j, min_i D_ij) (cnt 0)."""
        y0 = greedy_design(self.inst)
        self.offer(y0, design_value(self.inst, open_list(y0)), "initializer")
        v, Q, cnt = oracle.separate(y0.astype(float))
        return y0, v, cnt, oracle.sc.d1.copy()


# ---------------------------------------------------------------- engineered LP phase

class UFLEngLP(_Base):
    """Engineered Benders, LP phase only (see the module docstring); stops at convergence (master value equals
    f.y + sum_j Q_j(y) at the LP point within `conv_tol`, relative) or at the deadline."""

    def __init__(self, *, inout_lambda: float = 0.3, stabilize: bool = True, conv_tol: float = 1e-8,
                 max_rounds: int = 100000, lagr_warm: float = 0.0, lagr_cap: float = 60.0, purge_age: int = 0,
                 max_add: int | None = None, **kw) -> None:
        super().__init__(**kw)
        self.inout_lambda, self.stabilize, self.conv_tol, self.max_rounds = inout_lambda, stabilize, conv_tol, max_rounds
        # lagr_warm > 0: a subgradient phase of min(lagr_warm * time_limit, lagr_cap) seconds first; its best
        # multipliers u seed the master as the cuts (j, u_j) (the master then starts at L(u)).
        self.lagr_warm, self.lagr_cap, self.purge_age, self.max_add = lagr_warm, lagr_cap, purge_age, max_add

    def add_cuts(self, js, v, cnt) -> None:
        if not len(js):
            return
        A = self.oracle.cut_matrix(js, v, cnt)
        E = sp.csr_matrix((np.ones(len(js)), (np.arange(len(js)), np.asarray(js))), shape=(len(js), self.inst.n))
        rows = self.md.addConstr(A @ self.y + E @ self.th >= np.asarray(v, dtype=float))
        self.md.update()
        self.rows.extend(rows.tolist())
        if hasattr(self, "row_age"):
            self.row_age.extend([0] * len(js))
        self.row_j.extend(int(j) for j in js)
        self.row_v.extend(float(x) for x in v)
        self.row_c.extend(int(c) for c in cnt)
        self.stats["cuts"] += len(js)

    def publish_active(self, force: bool = False) -> None:
        if not self.rows or not (force or time.perf_counter() - self._last_cert >= self.cert_interval):
            return
        try:
            pi = np.asarray(self.md.getAttr("Pi", self.rows))
        except gp.GurobiError:
            return
        act = np.nonzero(pi > 1e-12)[0]
        act = act[np.argsort(-pi[act], kind="stable")]
        rj, rv, rc = np.asarray(self.row_j)[act], np.asarray(self.row_v)[act], np.asarray(self.row_c)[act]
        keep = fit(1 + rc, self.cert_limit)
        self.emit(lambda k: cert_bytes(rj[keep], rv[keep], version=k), force=force)
        self.stats["certificate_versions"] = self._cert_seq

    def run(self, data_path: str) -> dict[str, Any]:
        self.start(data_path)
        inst = self.inst
        self.stats: dict[str, Any] = {"cuts": 0, "rounds": 0, "converged": False, "master_seconds": 0.0,
                                      "oracle_seconds": 0.0, "lp_value": None, "final_lp_point": None}
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        reason = "time_limit"
        self.md = None
        try:
            self.oracle = oracle = Oracle(inst)
            self.stats["setup_seconds"] = self.elapsed()
            y0, v0, c0, d1 = self.init_cuts(oracle)
            md = gp.Model(env=env)
            self.md = md
            md.Params.OutputFlag = 0
            md.Params.Method = 1
            self.y = md.addMVar(inst.m, lb=0.0, ub=1.0, name="y")
            self.th = md.addMVar(inst.n, lb=0.0, name="theta")
            md.addConstr(self.y.sum() >= 1.0, name="cover")
            md.setObjective(inst.f @ self.y + self.th.sum(), GRB.MINIMIZE)
            self.rows, self.row_j, self.row_v, self.row_c = [], [], [], []
            js = np.arange(inst.n)
            self.add_cuts(js, d1, np.zeros(inst.n, dtype=np.int64))
            self.add_cuts(js, v0, c0)
            self.row_age = [0] * len(self.rows)
            self.n_lower = inst.n
            if self.lagr_warm > 0:
                t = time.perf_counter()
                u, L, it = subgradient(inst, inst.D[y0].min(axis=0).astype(float), self.ub,
                                       time.perf_counter() + min(self.lagr_warm * self.time_limit, self.lagr_cap))
                self.stats.update(lagr_seconds=time.perf_counter() - t, lagr_bound=L, lagr_iterations=it)
                self.add_cuts(js, u, oracle.sc.counts(js, u))
            reason = self.phase1(y0.astype(float))
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        finally:
            if self.md is not None:
                self.md.dispose()
            env.dispose()
        self.stats["oracle_seconds"] = getattr(self, "oracle", None) and self.oracle.seconds
        return {"best_objective": self.ub if self.ub < INF else None, "global_lb": self.lb if self.lb > -INF else None,
                "termination_reason": reason, "design": open_list(self.best) if self.best is not None else None,
                "lp_point": self.stats.pop("final_lp_point"), "stats": self.stats, "total_seconds": self.elapsed(),
                "trace": self.trace[-200:]}

    def purge(self) -> None:
        """Remove cuts that have been non-binding (negative slack, zero dual) for purge_age consecutive rounds (the
        lower-bound cuts are kept)."""
        slack = np.asarray(self.md.getAttr("Slack", self.rows))
        pi = np.asarray(self.md.getAttr("Pi", self.rows))
        loose = (slack < -1e-7 * np.maximum(1.0, np.abs(np.asarray(self.row_v)))) & (pi <= 1e-12)
        age = np.where(loose, np.asarray(self.row_age) + 1, 0)
        drop = age >= self.purge_age
        drop[: self.n_lower] = False
        if drop.any():
            self.md.remove([c for c, d in zip(self.rows, drop) if d])
            keep = ~drop
            self.rows = [c for c, k in zip(self.rows, keep) if k]
            self.row_j = np.asarray(self.row_j)[keep].tolist()
            self.row_v = np.asarray(self.row_v)[keep].tolist()
            self.row_c = np.asarray(self.row_c)[keep].tolist()
            self.md.update()
            self.stats["purged"] = self.stats.get("purged", 0) + int(drop.sum())
        self.row_age = age[~drop].tolist()

    def phase1(self, core: np.ndarray) -> str:
        md, oracle, inst = self.md, self.oracle, self.inst
        lam = self.inout_lambda if self.stabilize else 1.0
        best, stall = -INF, 0
        tried: set[bytes] = set()
        reason = "time_limit"
        while self.left() > 0 and self.stats["rounds"] < self.max_rounds:
            t = time.perf_counter()
            md.Params.TimeLimit = max(0.01, self.left())
            md.optimize()
            self.stats["master_seconds"] += time.perf_counter() - t
            if md.Status != GRB.OPTIMAL:
                reason = "time_limit" if md.Status == GRB.TIME_LIMIT else f"master_status_{md.Status}"
                break
            self.stats["rounds"] += 1
            lb_mp = float(md.ObjVal)
            self.publish_bound(lb_mp, "lp_master")
            self.publish_active()
            stall = stall + 1 if lb_mp <= best + 1e-9 * max(1.0, abs(lb_mp)) else 0
            best = max(best, lb_mp)
            y_lp = np.clip(np.asarray(self.y.X), 0.0, 1.0)
            th = np.asarray(self.th.X)
            if self.purge_age > 0:
                self.purge()          # invalidates the master solution: read it first
            self.stats["lp_value"], self.stats["final_lp_point"] = lb_mp, y_lp.tolist()
            # LP rounding (primal side, cheap): open the facilities with y >= 1/2 (or the largest)
            top = y_lp >= 0.5
            if not top.any():
                top[int(np.argmax(y_lp))] = True
            key = top.tobytes()
            if key not in tried:
                tried.add(key)
                self.offer(top, design_value(inst, open_list(top)), "lp_rounding")
            point = y_lp if lam >= 1.0 else lam * y_lp + (1.0 - lam) * core
            v, Q, cnt = oracle.separate(point)
            if lam >= 1.0:
                ub_mp = float(inst.f @ y_lp + Q.sum())
                if ub_mp - lb_mp <= self.conv_tol * max(1.0, abs(lb_mp)):
                    self.stats["converged"] = True
                    reason = "lp_converged"
                    break
                at = Q
            else:
                at = oracle.cut_values(v, cnt, y_lp)
            viol = at - th
            rel = viol / np.maximum(1.0, np.abs(v))
            add = np.nonzero(rel > 1e-9)[0]
            if self.max_add is not None and len(add) > self.max_add:
                add = add[np.argsort(-rel[add], kind="stable")[: self.max_add]]
            if len(add) == 0 or (stall >= 5 and lam < 1.0):
                if lam < 1.0:
                    lam, stall = 1.0, 0
                    continue
                self.stats["converged"] = True
                reason = "lp_converged_no_violated_cut"
                break
            self.add_cuts(add, v[add], cnt[add])
            core = 0.5 * core + 0.5 * point
        if self.left() > 0:
            md.Params.TimeLimit = max(0.01, self.left())
            md.optimize()
            if md.Status == GRB.OPTIMAL:
                self.publish_bound(float(md.ObjVal), "lp_master")
                self.publish_active(force=True)
        self.stats["phase1_seconds"] = self.elapsed()
        return reason


# ---------------------------------------------------------------- naive LP-master Benders

class UFLNaiveLP(_Base):
    """Naive LP-master Benders (bound-side naive baseline): unstabilized Kelley on min f.y + theta, theta >= 0,
    sum y >= 1, y in [0, 1]^m, with ONE aggregated cut per iteration at the master's LP point (same oracle); seeded
    with the aggregated lower-bound and initializer cuts."""

    def run(self, data_path: str) -> dict[str, Any]:
        self.start(data_path)
        inst = self.inst
        stats: dict[str, Any] = {"iterations": 0, "master_seconds": 0.0, "converged": False}
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        reason, point = "time_limit", None
        try:
            oracle = Oracle(inst)
            stats["setup_seconds"] = self.elapsed()
            y0, v0, c0, d1 = self.init_cuts(oracle)
            md = gp.Model(env=env)
            md.Params.OutputFlag = 0
            md.Params.Method = 1
            y = md.addMVar(inst.m, lb=0.0, ub=1.0, name="y")
            th = md.addVar(lb=0.0, name="theta")
            md.addConstr(y.sum() >= 1.0)
            md.setObjective(inst.f @ y + th, GRB.MINIMIZE)
            md.update()
            yl = y.tolist()
            rows: list[gp.Constr] = []
            cuts_v: list[np.ndarray] = []

            def add(v, cnt):
                const, coef = oracle.aggregate(v, cnt)
                nz = np.nonzero(coef)[0]
                rows.append(md.addLConstr(gp.LinExpr(coef[nz].tolist(), [yl[int(i)] for i in nz]) + th, GRB.GREATER_EQUAL, const))
                cuts_v.append(np.asarray(v, dtype=float).copy())

            add(d1, np.zeros(inst.n, dtype=np.int64))
            add(v0, c0)
            th_val = -INF
            point = y0.astype(float)
            while self.left() > 0:
                t = time.perf_counter()
                md.Params.TimeLimit = max(0.01, self.left())
                md.optimize()
                stats["master_seconds"] += time.perf_counter() - t
                if md.Status != GRB.OPTIMAL:
                    reason = "time_limit" if md.Status == GRB.TIME_LIMIT else f"master_status_{md.Status}"
                    break
                stats["iterations"] += 1
                self.publish_bound(float(md.ObjVal), "lp_master")
                lam = np.asarray(md.getAttr("Pi", rows))
                self.publish(lam, cuts_v, oracle)
                point = np.clip(np.asarray(y.X), 0.0, 1.0)
                th_val = float(th.X)
                v, Q, cnt = oracle.separate(point)
                if float(Q.sum()) <= th_val + 1e-9 * max(1.0, abs(th_val)):
                    stats["converged"] = True
                    reason = "lp_converged"
                    break
                add(v, cnt)
            if md.Status == GRB.OPTIMAL and self.left() > 0:
                self.publish(np.asarray(md.getAttr("Pi", rows)), cuts_v, oracle, force=True)
            stats["cuts"] = len(rows)
            md.dispose()
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        finally:
            env.dispose()
        return {"best_objective": self.ub if self.ub < INF else None, "global_lb": self.lb if self.lb > -INF else None,
                "termination_reason": reason, "design": open_list(self.best) if self.best is not None else None,
                "lp_point": point.tolist() if stats["converged"] and point is not None else None, "stats": stats,
                "total_seconds": self.elapsed(), "trace": self.trace[-200:]}

    def publish(self, lam: np.ndarray, cuts_v: list, oracle: Oracle, force: bool = False) -> None:
        """The master-dual combination of the aggregated cuts (sum_k lam_k v^k per customer, implied completion:
        one multi cut that reproduces the master value), then the positive-dual aggregated cuts, within the limit."""
        if not (force or time.perf_counter() - self._last_cert >= self.cert_interval):
            return
        lam = np.maximum(lam, 0.0)
        pos = np.nonzero(lam > 1e-12)[0]
        if not len(pos):
            return
        pos = pos[np.argsort(-lam[pos], kind="stable")]
        w = lam[pos] / lam[pos].sum()
        vbar = sum(wk * cuts_v[k] for wk, k in zip(w, pos))
        sc = oracle.sc
        ids = np.arange(self.inst.n)
        multi, size = [], 0
        for vv in [vbar] + [cuts_v[k] for k in pos]:
            s = int(self.inst.n + sc.counts(ids, vv).sum())
            if size + s > self.cert_limit:
                continue
            multi.append((ids, vv))
            size += s
        self.emit(lambda k: cert_bytes(multi=multi, version=k), force=force)


# ---------------------------------------------------------------- naive integer-master Benders (primal naive)

class UFLNaiveInt(_Base):
    """Textbook outer loop with a binary master re-solved to optimality every iteration and one aggregated cut per
    iteration (primal-side naive baseline; every master design is published)."""

    def run(self, data_path: str) -> dict[str, Any]:
        self.start(data_path)
        inst = self.inst
        stats: dict[str, Any] = {"iterations": 0, "master_seconds": 0.0}
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        reason = "time_limit"
        try:
            oracle = Oracle(inst)
            md = gp.Model(env=env)
            md.Params.OutputFlag = 0
            md.Params.MIPGap = 0.0
            if inst.integral:
                md.Params.MIPGapAbs = 0.999
            y = md.addMVar(inst.m, vtype=GRB.BINARY, name="y")
            th = md.addVar(lb=0.0, name="theta")
            md.addConstr(y.sum() >= 1.0)
            md.setObjective(inst.f @ y + th, GRB.MINIMIZE)
            md.update()
            yl = y.tolist()
            ybar = greedy_design(inst)
            th_val = -INF
            while self.left() > 0:
                v, Q, cnt = oracle.separate(ybar.astype(float))
                self.offer(ybar, design_value(inst, open_list(ybar)), "master_design" if stats["iterations"] else "initializer")
                if float(Q.sum()) <= th_val + 1e-9 * max(1.0, abs(th_val)):
                    reason = "optimal" if self.ub - self.lb <= 1.0 else "stalled"
                    break
                const, coef = oracle.aggregate(v, cnt)
                nz = np.nonzero(coef)[0]
                md.addLConstr(gp.LinExpr(coef[nz].tolist(), [yl[int(i)] for i in nz]) + th, GRB.GREATER_EQUAL, const)
                t = time.perf_counter()
                md.Params.TimeLimit = max(0.01, self.left())
                md.optimize()
                stats["master_seconds"] += time.perf_counter() - t
                stats["iterations"] += 1
                if md.SolCount and math.isfinite(md.ObjBound):
                    self.publish_bound(float(md.ObjBound), "integer_master")
                if md.Status != GRB.OPTIMAL:
                    reason = "time_limit" if md.Status == GRB.TIME_LIMIT else f"master_status_{md.Status}"
                    break
                ybar = np.asarray(y.X) > 0.5
                th_val = float(th.X)
            md.dispose()
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        finally:
            env.dispose()
        return {"best_objective": self.ub if self.ub < INF else None, "global_lb": self.lb if self.lb > -INF else None,
                "termination_reason": reason, "design": open_list(self.best) if self.best is not None else None,
                "stats": stats, "total_seconds": self.elapsed()}


# ---------------------------------------------------------------- compact models

def build_compact(inst: UFLInst, env: gp.Env, relax: bool) -> tuple[gp.Model, dict]:
    m, n = inst.m, inst.n
    N = m * n
    md = gp.Model(env=env)
    y = md.addMVar(m, lb=0.0, ub=1.0, obj=inst.f, vtype=GRB.CONTINUOUS if relax else GRB.BINARY, name="y")
    x = md.addMVar(N, lb=0.0, ub=1.0, obj=np.ascontiguousarray(inst.D.T).ravel(), name="x")  # customer-major
    md.ModelSense = GRB.MINIMIZE
    cj = np.repeat(np.arange(n), m)
    fi = np.tile(np.arange(m), n)
    A = sp.csr_matrix((np.ones(N), (cj, np.arange(N))), shape=(n, N))
    assign = md.addConstr(A @ x == np.ones(n), name="assign")
    del A
    B = sp.csr_matrix((np.ones(N), (np.arange(N), fi)), shape=(N, m))
    md.addConstr(x - B @ y <= 0.0, name="link")
    del B, cj, fi
    md.addConstr(y.sum() >= 1.0, name="cover")
    md.update()
    return md, {"y": y, "x": x, "assign": assign, "num_vars": int(md.NumVars), "num_constrs": int(md.NumConstrs)}


class UFLCompactLP(_Base):
    """Decomposition-free bound control: the compact LP (module docstring), barrier (method=2, Crossover off) or dual
    simplex (method=1), one thread, Gurobi MemLimit. Certificate: (j, v_j) from the assignment-row duals (Pi of a
    complete or SUBOPTIMAL solve; BarPi of an incomplete barrier solve, published publish_margin seconds before the
    deadline). Dual-simplex mid-solve objective values (dual feasible, unperturbed) are solver-reported bounds."""

    def __init__(self, *, method: int = 2, crossover: int = 0, publish_margin: float = 30.0, **kw) -> None:
        super().__init__(**kw)
        self.method, self.crossover, self.publish_margin = method, crossover, publish_margin

    def run(self, data_path: str) -> dict[str, Any]:
        self.start(data_path)
        inst = self.inst
        env = make_env(self.threads, self.soft_mem_gb, self.seed, self.mem_limit_gb)
        out: dict[str, Any] = {"method": self.method, "crossover": self.crossover, "termination_reason": "deadline_during_build"}
        model = None
        try:
            model, info = build_compact(inst, env, relax=True)
            out.update(build_seconds=self.elapsed(), num_vars=info["num_vars"], num_constrs=info["num_constrs"])
            margin = min(self.publish_margin if self.method == 2 else 5.0, 0.1 * self.time_limit)
            if self.left() - margin > 0.01:
                model.Params.TimeLimit = self.left() - margin
                model.Params.Method = self.method
                model.Params.Crossover = self.crossover
                spx: list[tuple[float, float]] = []

                def callback(cb, where):
                    if where == GRB.Callback.SIMPLEX and self.method == 1 and self.left() > 0:
                        if cb.cbGet(GRB.Callback.SPX_ISPERT) == 0 and cb.cbGet(GRB.Callback.SPX_DUALINF) <= 1e-9:
                            val = float(cb.cbGet(GRB.Callback.SPX_OBJVAL))
                            if math.isfinite(val) and (not spx or val > spx[-1][1]):
                                spx.append((round(self.elapsed(), 3), val))
                                self.on_bound(val)

                model.optimize(callback)
                st = int(model.Status)
                out["gurobi_status"] = st
                out["simplex_bounds"] = spx[-100:]
                v = yv = None
                if st in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                    try:
                        v, yv = np.asarray(info["assign"].Pi, dtype=float), np.asarray(info["y"].X, dtype=float)
                        out["certificate_source"] = "Pi"
                    except gp.GurobiError:
                        v = None
                if v is None and self.method == 2:
                    try:
                        v, yv = np.asarray(info["assign"].BarPi, dtype=float), np.asarray(info["y"].BarX, dtype=float)
                        out["certificate_source"] = "BarPi"
                    except (gp.GurobiError, AttributeError):
                        v = None
                if v is not None and self.left() > 0:
                    v = np.where(np.isfinite(v), v, 0.0)
                    srt = Sorted(inst.D)
                    j = np.arange(inst.n)
                    keep = fit(1 + srt.counts(j, v), self.cert_limit)
                    self.emit(lambda k: cert_bytes(j[keep], v[keep], version=k), force=True)
                    out["certificate_seconds"] = self.elapsed()
                    out["lp_point"] = yv.tolist() if yv is not None else None
                if st == GRB.OPTIMAL and self.left() > 0:
                    out["termination_reason"] = "optimal"
                    out["lp_value"] = float(model.ObjVal)
                    self.publish_bound(float(model.ObjVal), "compact_lp")
                else:
                    out["termination_reason"] = {GRB.TIME_LIMIT: "time_limit", GRB.INTERRUPTED: "time_limit",
                                                 GRB.MEM_LIMIT: "memory_limit"}.get(st, f"status_{st}")
                out["barrier_iterations"] = int(model.BarIterCount) if self.method == 2 else None
        except (gp.GurobiError, MemoryError) as err:
            if isinstance(err, gp.GurobiError) and not is_memory_error(err):
                raise
            out["termination_reason"] = "memory_limit" + ("" if model is not None else "_during_build")
            out["error"] = str(err)[:300]
        finally:
            if model is not None:
                model.dispose()
            env.dispose()
        out["total_seconds"] = self.elapsed()
        return out


class UFLCompactMIP(_Base):
    """Compact MIP (binary y, continuous x), primal baseline; incumbents and the MIP bound are published."""

    def run(self, data_path: str) -> dict[str, Any]:
        self.start(data_path)
        inst = self.inst
        env = make_env(self.threads, self.soft_mem_gb, self.seed, self.mem_limit_gb)
        out: dict[str, Any] = {"termination_reason": "deadline_during_build"}
        model = None
        try:
            model, info = build_compact(inst, env, relax=False)
            out["build_seconds"] = self.elapsed()
            yl = info["y"].tolist()
            errors: list[BaseException] = []

            def callback(cb, where):
                try:
                    if self.left() <= 0:
                        cb.terminate()
                        return
                    if where == GRB.Callback.MIPSOL:
                        yy = np.asarray(cb.cbGetSolution(yl)) > 0.5
                        if yy.any():
                            self.offer(yy, design_value(inst, open_list(yy)), "mip")
                    elif where == GRB.Callback.MIP:
                        b = cb.cbGet(GRB.Callback.MIP_OBJBND)
                        if math.isfinite(b) and abs(b) < 1e90:
                            self.publish_bound(float(b), "mip")
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)
                    cb.terminate()

            if self.left() > 0.01:
                model.Params.TimeLimit = self.left()
                model.Params.MIPGap = 1e-6
                model.optimize(callback)
                if errors:
                    raise errors[0]
                st = int(model.Status)
                out["gurobi_status"] = st
                out["termination_reason"] = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit",
                                             GRB.INTERRUPTED: "time_limit", GRB.MEM_LIMIT: "memory_limit"}.get(st, f"status_{st}")
        except (gp.GurobiError, MemoryError) as err:
            if isinstance(err, gp.GurobiError) and not is_memory_error(err):
                raise
            out["termination_reason"] = "memory_limit" + ("" if model is not None else "_during_build")
            out["error"] = str(err)[:300]
        finally:
            if model is not None:
                model.dispose()
            env.dispose()
        out.update(best_objective=self.ub if self.ub < INF else None, global_lb=self.lb if self.lb > -INF else None,
                   design=open_list(self.best) if self.best is not None else None, total_seconds=self.elapsed())
        return out


# ---------------------------------------------------------------- trivial-certificate controls and diagnostics

class UFLStatic(_Base):
    """Trivial certificates published once: 'empty' (no cuts) or 'oneshot_greedy' (one cut per customer at the
    shared initializer's design: v_j = cost of its nearest open facility)."""

    def __init__(self, *, kind: str = "empty", **kw) -> None:
        super().__init__(**kw)
        self.kind = kind

    def run(self, data_path: str) -> dict[str, Any]:
        self.start(data_path)
        inst = self.inst
        if self.kind == "empty":
            self.emit(lambda k: cert_bytes(version=k), force=True)
        elif self.kind == "oneshot_greedy":
            y0 = greedy_design(inst)
            self.offer(y0, design_value(inst, open_list(y0)), "initializer")
            v = inst.D[y0].min(axis=0)
            srt = Sorted(inst.D)
            j = np.arange(inst.n)
            keep = fit(1 + srt.counts(j, v), self.cert_limit)
            self.emit(lambda k: cert_bytes(j[keep], v[keep], version=k), force=True)
        else:
            raise ValueError(self.kind)
        return {"termination_reason": "published", "kind": self.kind, "total_seconds": self.elapsed(),
                "best_objective": self.ub if self.ub < INF else None}


def capped_point(w: np.ndarray, total: float) -> np.ndarray:
    y = total * w / w.sum()
    for _ in range(200):
        over = y > 1.0
        if not over.any():
            break
        excess = float((y[over] - 1.0).sum())
        y[over] = 1.0
        free = y < 1.0
        y[free] += excess * y[free] / y[free].sum()
    return np.minimum(y, 1.0)


class UFLPool(_Base):
    """No-feedback cut pool (frozen recipe, seed 20261103): oracle cuts at points chosen without any master: the
    initializer's design, the uniform point y = k/m (k = its number of open facilities), then alternately a random
    design with k open facilities and a random fractional point (k * Dirichlet(1), capped); new (j, v) cuts in
    customer order until the size limit."""

    def __init__(self, *, pool_seed: int = 20261103, **kw) -> None:
        super().__init__(**kw)
        self.pool_seed = pool_seed

    def run(self, data_path: str) -> dict[str, Any]:
        self.start(data_path)
        inst = self.inst
        m, n = inst.m, inst.n
        oracle = Oracle(inst)
        rng = np.random.default_rng(self.pool_seed)
        y0 = greedy_design(inst)
        self.offer(y0, design_value(inst, open_list(y0)), "initializer")
        k = max(1, int(y0.sum()))
        seen: set[tuple[int, float]] = set()
        js: list[int] = []
        vs: list[float] = []
        size, points, full, t = 0, 0, False, 0
        while not full and self.left() > 0:
            if t == 0:
                point = y0.astype(float)
            elif t == 1:
                point = np.full(m, min(1.0, k / m))
            elif t % 2 == 0:
                point = np.zeros(m)
                point[rng.choice(m, k, replace=False)] = 1.0
            else:
                point = capped_point(rng.dirichlet(np.ones(m)), k)
            t += 1
            v, _, cnt = oracle.separate(point)
            points += 1
            for j in range(n):
                key = (j, float(v[j]))
                if key in seen:
                    continue
                c = 1 + int(cnt[j])
                if size + c > self.cert_limit:
                    full = True
                    break
                seen.add(key)
                js.append(j)
                vs.append(float(v[j]))
                size += c
            self.emit(lambda kk: cert_bytes(js, vs, version=kk), force=full)
        if not full:
            self.emit(lambda kk: cert_bytes(js, vs, version=kk), force=True)
        return {"termination_reason": "pool_full" if full else "time_limit", "points": points, "cuts": len(js),
                "size": size, "total_seconds": self.elapsed(), "best_objective": self.ub if self.ub < INF else None}


def subgradient(inst: UFLInst, u: np.ndarray, ub: float, deadline: float) -> tuple[np.ndarray, float, int]:
    """Subgradient on the assignment-row multipliers (as UFLLagrangian, without publication) until the deadline:
    (best u, best L(u), iterations)."""
    D, f = inst.D, inst.f
    best_L, best_u = -INF, u.copy()
    step, stall, it = 2.0, 0, 0
    while time.perf_counter() < deadline:
        P = u[None, :] - D
        pos = P > 0
        rho = f - np.where(pos, P, 0.0).sum(axis=1)
        S = np.nonzero(rho < 0)[0]
        if not len(S):
            S = np.array([int(np.argmin(rho))])
        L = float(u.sum() + rho[S].sum())
        it += 1
        if L > best_L + 1e-12 * max(1.0, abs(L)):
            best_L, best_u, stall = L, u.copy(), 0
        else:
            stall += 1
            if stall >= 30:
                step, stall = max(step / 2.0, 1e-8), 0
        g = 1.0 - pos[S].sum(axis=0)
        gg = float(g @ g)
        if gg == 0.0:
            break
        u = u + (step * max(ub - L, 1e-6 * max(1.0, abs(L))) / gg) * g
    return best_u, best_L, it


class UFLLagrangian(_Base):
    """Subgradient Lagrangian relaxing the assignment rows (diagnostic; itself a decomposition):
        L(u) = sum_j u_j + min_{y in [0,1]^m, sum y >= 1} sum_i y_i (f_i - sum_j (u_j - D_ij)^+),
    Polyak steps against the best known design value, step halving after 30 non-improving iterations. Snapshots of
    the best u (at geometrically spaced times) are published as the cuts (j, u_j) (the checker master with these
    cuts equals L(u))."""

    def __init__(self, *, snapshot_ratio: float = 2 ** 0.5, **kw) -> None:
        kw.setdefault("cert_interval", 0.0)
        super().__init__(**kw)
        self.snapshot_ratio = snapshot_ratio

    def run(self, data_path: str) -> dict[str, Any]:
        self.start(data_path)
        inst = self.inst
        D, f = inst.D, inst.f
        srt = Sorted(D)
        y0 = greedy_design(inst)
        self.offer(y0, design_value(inst, open_list(y0)), "initializer")
        u = D[y0].min(axis=0).astype(float)
        best_L, best_u = -INF, u.copy()
        step, stall, it = 2.0, 0, 0
        next_snap, snaps = 0.25, 0
        bounds: list[tuple[float, float]] = []
        reason = "time_limit"
        j = np.arange(inst.n)
        while self.left() > 0:
            P = u[None, :] - D
            pos = P > 0
            rho = f - np.where(pos, P, 0.0).sum(axis=1)
            neg = rho < 0
            if neg.any():
                S = np.nonzero(neg)[0]
            else:
                S = np.array([int(np.argmin(rho))])
            L = float(u.sum() + rho[S].sum())
            it += 1
            if L > best_L + 1e-12 * max(1.0, abs(L)):
                best_L, best_u, stall = L, u.copy(), 0
                self.on_bound(L - 1e-9 * max(1.0, abs(L)))
                bounds.append((round(self.elapsed(), 3), L))
            else:
                stall += 1
                if stall >= 30:
                    step, stall = max(step / 2.0, 1e-8), 0
            if self.elapsed() >= next_snap:
                bu = best_u.copy()
                keep = fit(1 + srt.counts(j, bu), self.cert_limit)
                self.emit(lambda k: cert_bytes(j[keep], bu[keep], version=k), force=True)
                snaps += 1
                while next_snap <= self.elapsed():
                    next_snap *= self.snapshot_ratio
            if it % 20 == 1:
                ys = np.zeros(inst.m, dtype=bool)
                ys[S] = True
                self.offer(ys, design_value(inst, open_list(ys)), "lagrangian")
            g = 1.0 - pos[S].sum(axis=0)
            gg = float(g @ g)
            if gg == 0.0:
                reason = "subgradient_zero"
                break
            gap = max(self.ub - L, 1e-6 * max(1.0, abs(L)))
            u = u + (step * gap / gg) * g
        if self.left() > 0:
            bu = best_u.copy()
            keep = fit(1 + srt.counts(j, bu), self.cert_limit)
            self.emit(lambda k: cert_bytes(j[keep], bu[keep], version=k), force=True)
        return {"termination_reason": reason, "iterations": it, "global_lb": best_L if best_L > -INF else None,
                "best_objective": self.ub if self.ub < INF else None, "snapshots": snaps,
                "design": open_list(self.best) if self.best is not None else None, "total_seconds": self.elapsed(),
                "bound_trace": bounds[-100:]}


class UFLILS(_Base):
    """Heuristic-only primal control: src/bd/ufl/b_ref_bd_xl.iterated_local_search (add/drop/swap descent from the
    add/drop start, then random 2-10 flips of the best); it also publishes the one-shot certificate of every new
    best design (the 'one-shot at a heuristic design' trivial-certificate control)."""

    def run(self, data_path: str) -> dict[str, Any]:
        from src.bd.ufl.b_ref_bd_xl import iterated_local_search
        self.start(data_path)
        inst = self.inst
        srt = Sorted(inst.D)
        j = np.arange(inst.n)

        def on_inc(open_):
            y = np.zeros(inst.m, dtype=bool)
            y[list(open_)] = True
            if self.offer(y, design_value(inst, open_), "ils"):
                v = inst.D[y].min(axis=0)
                keep = fit(1 + srt.counts(j, v), self.cert_limit)
                self.emit(lambda k: cert_bytes(j[keep], v[keep], version=k), force=True)

        iterated_local_search(inst.D, inst.f, self.deadline, self.seed, on_inc)
        return {"termination_reason": "time_limit", "best_objective": self.ub if self.ub < INF else None,
                "design": open_list(self.best) if self.best is not None else None, "total_seconds": self.elapsed()}


class UFLEngPrimal(_Base):
    """The engineered branch-and-Benders-cut reference (src/bd/ufl/b_ref_bd_xl.UFLBendersXL, unchanged), primal side;
    its root-LP bound is solver-reported only (no certificate)."""

    def run(self, data_path: str) -> dict[str, Any]:
        from src.bd.ufl.b_ref_bd_xl import UFLBendersXL
        self.start(data_path)
        inst = self.inst

        def on_inc(open_):
            if open_:
                y = np.zeros(inst.m, dtype=bool)
                y[list(open_)] = True
                self.offer(y, design_value(inst, open_), "eng_primal")

        res = UFLBendersXL(time_limit=max(1.0, self.left()), threads=self.threads, on_incumbent=on_inc,
                           on_bound=lambda b: self.publish_bound(float(b), "eng_primal"), soft_mem_gb=self.soft_mem_gb).run(data_path)
        return {k: v for k, v in res.items() if k != "trace"}
