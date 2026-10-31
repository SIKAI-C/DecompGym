"""Correctness checks for the railway crew scheduling family (scale-up study): evaluator and verifier against the
benchmark's published best-known solutions, pricing against enumeration, and the compact MIP, the naive and the
engineered column generation against brute force on synthetic instances with at most 10 tasks."""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

gp = pytest.importorskip("gurobipy")

from src.cg.rcsp.a_data import duties_from_connections, read_instance, read_solution_file, write_spec
from src.cg.rcsp.b_mip_xl import RCSPCompactMIP, duty_check, plan_cost, plan_issues
from src.cg.rcsp.c_naive_cg import RCSPNaiveCG
from src.cg.rcsp.d_ref_cg_xl import RCSPColumnGenerationXL
from src.cg.rcsp.e_verifier import verify_connection_file, verify_solution
from src.cg.rcsp.pricing import DutyPricer, greedy_cover

RAW = Path(__file__).resolve().parents[1] / "outputs" / "scaleup" / "raw" / "rcsp" / "rcsp-benchmark-main"
needs_raw = pytest.mark.skipif(not (RAW / "instances" / "s2_t90" / "tasks.txt").exists(), reason="RCSP benchmark not downloaded")
# Best-known costs (and duty counts) of the benchmark's solutions/README.md, single- and multi-depot.
PUBLISHED = {"s2_t90": ((748080, 14), (748080, 14)), "s3_t262": ((1862520, 32), (1761300, 31)),
             "s6_t482": ((2868240, 50), (2691240, 49)), "s9_t991": ((4831620, 82), (4507920, 79)),
             "s14_t1500": ((8880420, 154), (8245860, 150)), "s20_t3016": ((16287780, 277), (14756220, 265))}


# ---------------------------------------------------------------- synthetic instances

def write_synthetic(rng: np.random.Generator, root: Path, crews: int, p_loop: float = 0.8) -> int:
    """A random instance in the benchmark's format and conventions (time-consistent durations, break connections with
    timeBeforeBreak 300 and a 30 min break, SOURCE cost = duration + 28 800), built around planted duties: each crew
    starts at station 0 between 5:00 and 10:00, makes 2-3 trips of 40-100 min (a trip stays at its station with
    probability p_loop, else goes to the other one; the last trip returns to 0) with short turns and one 35-70 min gap,
    so single-depot coverage is feasible. All station- and time-compatible task pairs are connected, so many other
    duties exist (the 9.5 h and break rules cut some of them). Returns the number of tasks."""
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for _ in range(crews):
        k = int(rng.integers(2, 4))
        t = int(rng.integers(5 * 60, 10 * 60)) * 60
        brk = int(rng.integers(1, k))
        st = 0
        for q in range(k):
            if q:
                t += (int(rng.integers(35, 70)) if q == brk else int(rng.integers(0, 25))) * 60
            d = int(rng.integers(40, 100)) * 60
            nxt = 0 if q == k - 1 else (st if rng.random() < p_loop else 1 - st)
            rows.append((t, t + d, st, nxt))
            t += d
            st = nxt
    rows.sort()
    n = len(rows)
    start, end = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    frm, to = np.array([r[2] for r in rows]), np.array([r[3] for r in rows])
    dur = end - start
    (root / "tasks.txt").write_text("".join(f"{i} {start[i]} {end[i]} {frm[i]} {to[i]} {dur[i]}\n" for i in range(n)))
    (root / "stations.txt").write_text("0 A\n1 B\n")
    (root / "crewbases_single_depot.txt").write_text("0 A\n")
    (root / "crewbases_multi_depot.txt").write_text("0 A\n1 B\n")
    conns = []
    for i in range(n):
        for j in range(n):
            if i != j and to[i] == frm[j] and start[j] >= end[i]:
                gap, d = start[j] - end[i], end[j] - end[i]
                tbb, tab = (300, d - 2100) if gap >= 2100 else (-1, -1)
                conns.append(("CONNECTION", i, j, tbb, tab, d, d))
    for variant, bases in (("single", [0]), ("multi", [0, 1])):
        out = list(conns)
        for b in bases:
            out += [("SOURCE", b, j, -1, -1, int(dur[j]), int(dur[j]) + 28800) for j in range(n) if frm[j] == b]
            out += [("SINK", i, b, -1, -1, 0, 0) for i in range(n) if to[i] == b]
        order = rng.permutation(len(out))
        (root / f"connections_{variant}_depot.txt").write_text("".join(f"{k} {' '.join(str(x) for x in out[r])}\n" for k, r in enumerate(order)))
    return n


def lp_over(inst, duties) -> float:
    m = gp.Model()
    m.Params.OutputFlag = 0
    x = [m.addVar(obj=c) for _, _, c in duties]
    for t in range(inst.n):
        m.addLConstr(gp.quicksum(x[k] for k, (_, p, _) in enumerate(duties) if t in p) >= 1)
    m.optimize()
    return m.ObjVal


def all_duties(inst) -> list[tuple[int, tuple[int, ...], float]]:
    """Every feasible duty, by depth-first search over the task graph and the rule definitions of duty_check."""
    succ = {}
    for i, j in inst.arc_of:
        succ.setdefault(i, []).append(j)
    out = []
    for (d, s) in inst.source_of:
        stack = [(s,)]
        while stack:
            path = stack.pop()
            cost, _ = duty_check(inst, d, list(path))
            if cost is not None:
                out.append((d, path, cost))
            stack += [path + (j,) for j in succ.get(path[-1], [])]
    return out


def brute_force_optimum(inst, duties) -> float:
    """Minimum-cost set cover over all feasible duties by dynamic programming over task subsets."""
    n = inst.n
    masks = {}
    for d, path, cost in duties:
        m = sum(1 << t for t in path)
        masks[m] = min(masks.get(m, math.inf), cost)
    by_low = {}
    for m, c in masks.items():
        for t in range(n):
            if m >> t & 1:
                by_low.setdefault(t, []).append((m, c))
    full = (1 << n) - 1
    f = {0: 0.0}

    def solve(S: int) -> float:
        if S in f:
            return f[S]
        low = (S & -S).bit_length() - 1
        best = math.inf
        for m, c in by_low.get(low, []):
            best = min(best, c + solve(S & ~m))
        f[S] = best
        return best

    return solve(full)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """Synthetic instances with 4-10 tasks, both variants, with brute-force optima and the LP over all duties:
    the first 2 instances with at least 15 feasible duties plus the first 2 whose LP bound is fractional (strictly
    below the optimum), so the integer phase of every method is exercised."""
    rng = np.random.default_rng(2026)
    base = tmp_path_factory.mktemp("rcsp")
    plain, gapped = [], []
    for k in range(400):
        if len(plain) >= 4 and len(gapped) >= 4:
            break
        d = base / f"syn{k}"
        n = write_synthetic(rng, d, int(rng.integers(2, 5)))
        if n > 10:
            continue
        cases = []
        for variant in ("single", "multi"):
            inst = read_instance(d, variant)
            duties = all_duties(inst)
            if len({t for _, p, _ in duties for t in p}) < n:
                break
            cases.append({"inst": inst, "spec": write_spec(d, variant, base / f"syn{k}_{variant}.json"), "duties": duties,
                          "opt": brute_force_optimum(inst, duties), "lp": lp_over(inst, duties)})
        if len(cases) < 2:
            continue
        if any(c["lp"] < c["opt"] - 1e-6 for c in cases) and len(gapped) < 4:
            gapped += cases
        elif len(cases[0]["duties"]) >= 15 and len(plain) < 4:
            plain += cases
    assert len(gapped) >= 4 and len(plain) >= 4
    return plain + gapped


# ---------------------------------------------------------------- published solutions and verifier

@needs_raw
@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_evaluator_reproduces_published_best_known_costs(name, tmp_path):
    for k, variant in enumerate(("single", "multi")):
        inst = read_instance(RAW / "instances" / name, variant)
        sol = RAW / "solutions" / f"{variant}_depot" / f"solution_{variant}_depot_{name}.txt"
        duties = duties_from_connections(inst, read_solution_file(sol))
        cost, issues = plan_issues(inst, duties)
        assert issues == [] and cost == PUBLISHED[name][k][0] and len(duties) == PUBLISHED[name][k][1]
        spec = write_spec(RAW / "instances" / name, variant, tmp_path / f"{variant}.json")
        assert verify_connection_file(spec, sol) == {"ok": True, "issues": [], "objective_value": float(cost), "duties": len(duties)}
        out = tmp_path / "out.json"
        out.write_text(json.dumps({"status": "feasible", "method": "x", "objective_value": cost, "duties": duties}))
        assert verify_solution(spec, out)["ok"]


@needs_raw
def test_benchmark_data_conventions():
    for name in ("s2_t90", "s9_t991"):
        for variant in ("single", "multi"):
            inst = read_instance(RAW / "instances" / name, variant)
            assert inst.time_consistent
            brk = inst.arc_break
            assert np.all(inst.arc_tbb[brk] == 300)
            assert np.all(inst.arc_tbb[brk] + 1800 + inst.arc_tab[brk] == inst.arc_dur[brk])
            assert np.all(inst.c_cost[inst.src_conn] == inst.c_dur[inst.src_conn] + 28800)
            assert np.all(inst.arc_cost == inst.arc_dur)
            assert np.all(inst.start[inst.arc_head] >= inst.end[inst.arc_tail])


@needs_raw
def test_engineered_root_bound_matches_published_root_lb(tmp_path):
    spec = write_spec(RAW / "instances" / "s2_t90", "single", tmp_path / "s.json")
    bounds = []
    res = RCSPColumnGenerationXL(time_limit=10, on_bound=bounds.append).run(str(spec))
    assert res["root_lp"] == pytest.approx(745680, abs=1e-3)  # README root LB
    assert res["best_objective"] == 748080 and res["global_lb"] == 748080 and res["termination_reason"] == "optimal"
    assert max(bounds) <= 748080


def test_verifier_rejects_broken_plans(synthetic, tmp_path):
    case = synthetic[0]
    inst, spec = case["inst"], case["spec"]
    plan = greedy_cover(DutyPricer(inst))
    cost = plan_cost(inst, plan)
    out = tmp_path / "o.json"

    def check(duties, obj=None, **kw):
        out.write_text(json.dumps({"status": "feasible", "method": "m", "objective_value": cost if obj is None else obj, "duties": duties, **kw}))
        return verify_solution(spec, out)

    assert check(plan)["ok"]
    assert not check(plan, obj=cost + 1)["ok"]
    assert not check([d for d in plan if 0 not in d["tasks"]])["ok"]  # task 0 uncovered
    bad = [dict(plan[0], depot=99)] + plan[1:]
    assert not check(bad)["ok"]
    shuffled = [dict(plan[0], tasks=plan[0]["tasks"][::-1])] + plan[1:]
    if len(plan[0]["tasks"]) > 1:
        assert not check(shuffled)["ok"]
    assert not check(plan, status="optimal-ish")["ok"]


def hm(text: str) -> int:
    h, m = text.split(":")
    return 3600 * int(h) + 60 * int(m)


def write_station_instance(root: Path, intervals: list[tuple[str, str]]) -> Path:
    """Tasks at one station (every task starts and ends at the crew base), all time-compatible pairs connected."""
    root.mkdir(parents=True, exist_ok=True)
    start = [hm(a) for a, _ in intervals]
    end = [hm(b) for _, b in intervals]
    n = len(intervals)
    (root / "tasks.txt").write_text("".join(f"{i} {start[i]} {end[i]} 0 0 {end[i] - start[i]}\n" for i in range(n)))
    for f in ("stations.txt", "crewbases_single_depot.txt", "crewbases_multi_depot.txt"):
        (root / f).write_text("0 A\n")
    rows = []
    for i in range(n):
        for j in range(n):
            if i != j and start[j] >= end[i]:
                gap, d = start[j] - end[i], end[j] - end[i]
                rows.append(("CONNECTION", i, j, *((300, d - 2100) if gap >= 2100 else (-1, -1)), d, d))
    rows += [("SOURCE", 0, j, -1, -1, end[j] - start[j], end[j] - start[j] + 28800) for j in range(n)]
    rows += [("SINK", i, 0, -1, -1, 0, 0) for i in range(n)]
    text = "".join(f"{k} {' '.join(str(x) for x in r)}\n" for k, r in enumerate(rows))
    (root / "connections_single_depot.txt").write_text(text)
    (root / "connections_multi_depot.txt").write_text(text)
    return root


@pytest.mark.parametrize("intervals, feasible, why", [
    ([("6:00", "7:00"), ("7:35", "8:35")], True, ""),                       # 35 min gap: break arc
    ([("6:00", "7:00"), ("7:34", "8:35")], False, "no break"),              # 34 min gap: no break arc
    ([("6:00", "11:25"), ("12:00", "13:00")], True, ""),                    # break starts at 5:30 h exactly
    ([("6:00", "11:26"), ("12:01", "13:00")], False, "no break"),           # break starts at 5:31 h
    ([("6:00", "7:00"), ("7:35", "13:05")], True, ""),                      # 5:30 h after the break exactly
    ([("6:00", "7:00"), ("7:35", "13:06")], False, "no break"),             # 5:31 h after the break
    ([("6:00", "10:00"), ("10:35", "15:30")], True, ""),                    # 9:30 h duty exactly
    ([("6:00", "10:00"), ("10:35", "15:31")], False, "duty length"),        # 9:31 h duty
    ([("6:00", "7:00"), ("7:35", "8:00"), ("8:35", "13:30")], True, ""),    # first break fails "after", second works
    ([("6:00", "6:30"), ("7:05", "11:40"), ("12:15", "12:30")], True, ""),  # second break fails "before", first works
    ([("6:00", "7:00")], False, "no break"),                               # a single task has no break connection
])
def test_duty_rules_at_their_limits(tmp_path, intervals, feasible, why):
    d = write_station_instance(tmp_path / "inst", intervals)
    inst = read_instance(d, "single")
    tasks = list(range(len(intervals)))
    cost, reason = duty_check(inst, 0, tasks)
    assert (cost is not None) == feasible and reason.startswith(why)
    if feasible:
        assert cost == 28800 + hm(intervals[-1][1]) - hm(intervals[0][0])
    # The verifier and the pricer (forced-break labelling) agree with duty_check on every path of the instance.
    spec = write_spec(d, "single", tmp_path / "s.json")
    out = tmp_path / "o.json"
    out.write_text(json.dumps({"status": "feasible", "method": "m", "objective_value": cost or 0, "duties": [{"depot": 0, "tasks": tasks}]}))
    assert verify_solution(spec, out)["ok"] == feasible
    enum = DutyPricer(inst).price(np.zeros(inst.n), enumerate_all=True, threshold=1e9, max_cols=10**6)
    paths = [list(c) for r in range(1, inst.n + 1) for c in itertools.combinations(range(inst.n), r)]
    expect = sorted(tuple(p) for p in paths if duty_check(inst, 0, p)[0] is not None)
    assert sorted(tuple(t) for _, _, t in enum["cols"]) == expect


# ---------------------------------------------------------------- pricing

def test_pricing_matches_enumeration(synthetic):
    rng = np.random.default_rng(3)
    for case in synthetic:
        inst, duties = case["inst"], case["duties"]
        pr = DutyPricer(inst)
        for _ in range(6):
            pi = rng.uniform(0, 30000, inst.n) * (rng.random(inst.n) < 0.8)
            best = {}
            for d, path, cost in duties:
                rc = cost - pi[list(path)].sum()
                best[d] = min(best.get(d, math.inf), rc)
            res = pr.price(pi, exact=True, max_cols=10_000)
            assert res["complete"]
            for d in inst.depots:
                expect = min(0.0, best.get(d, math.inf)) if best.get(d, math.inf) < -1e-6 else 0.0
                assert res["min_rc"][pr.dpos[d]] == pytest.approx(expect, abs=1e-6)
            for rc, d, tasks in res["cols"]:
                cost, _ = duty_check(inst, d, tasks)
                assert cost is not None and rc == pytest.approx(cost - pi[tasks].sum(), abs=1e-6)
            fast = pr.price(pi, exact=True, max_cols=10_000, rel=0.5)
            assert fast["min_rc"].min() == pytest.approx(res["min_rc"].min(), abs=1e-6)
            thr = float(rng.uniform(-5000, 20000))
            enum = pr.price(pi, enumerate_all=True, threshold=thr, max_cols=10**6)
            expect = sorted((d, tuple(p)) for d, p, c in duties if c - pi[list(p)].sum() < thr)
            assert enum["complete"] and sorted((d, tuple(t)) for _, d, t in enum["cols"]) == expect
            for beam in (1, 3):
                for rc, d, tasks in pr.price(pi, exact=False, beam=beam)["cols"]:
                    assert duty_check(inst, d, tasks)[0] is not None


def test_dominance_matches_pairwise_definition():
    rng = np.random.default_rng(11)
    for _ in range(60):
        m = int(rng.integers(1, 80))
        c = rng.integers(0, 6, m).astype(float)
        w = rng.integers(0, 5, m).astype(float)
        b = rng.integers(0, 2, m).astype(np.int8)
        beta = np.where(b == 1, np.minimum(w, rng.integers(0, 5, m)), w).astype(float)
        dep = rng.integers(0, 2, m).astype(np.int32)
        keep = set(DutyPricer._dominance(c, w, beta, b, dep).tolist())
        rows = [(c[i], w[i], beta[i], -b[i], dep[i]) for i in range(m)]
        for j in range(m):
            dominated = any(dep[i] == dep[j] and c[i] <= c[j] and w[i] <= w[j] and beta[i] <= beta[j] and b[i] >= b[j]
                            and (rows[i] != rows[j] or i in keep and i != j) for i in range(m))
            # a kept label is never dominated by a different kept one; a dropped label is dominated by some label
            if j in keep:
                assert not any(i in keep and i != j and dep[i] == dep[j] and c[i] <= c[j] and w[i] <= w[j] and beta[i] <= beta[j]
                               and b[i] >= b[j] for i in range(m))
            else:
                assert dominated


# ---------------------------------------------------------------- methods against brute force

def test_methods_reach_brute_force_optimum(synthetic):
    """The compact MIP and the engineered column generation (reduced-cost enumeration) are exact here and prove it;
    the naive control (root column generation + integer restricted master, a heuristic by design) must reach the
    optimum when the LP over all duties is integral and is otherwise only checked for feasibility and valid bounds."""
    naive_missed = 0
    for case in synthetic:
        inst, spec, opt = case["inst"], str(case["spec"]), case["opt"]
        for cls, kw in ((RCSPCompactMIP, {"time_limit": 20}), (RCSPNaiveCG, {"time_limit": 4}), (RCSPColumnGenerationXL, {"time_limit": 4})):
            incs, bounds = [], []
            res = cls(on_incumbent=incs.append, on_bound=bounds.append, **kw).run(spec)
            for plan in incs:
                assert plan_cost(inst, plan) is not None
            assert plan_cost(inst, res["duties"]) == pytest.approx(res["best_objective"])
            if cls is RCSPNaiveCG and case["lp"] < opt - 1e-6:
                assert res["best_objective"] >= opt - 1e-6
                naive_missed += res["best_objective"] > opt + 1e-6
            else:
                assert res["best_objective"] == pytest.approx(opt), (cls.__name__, spec)
            if cls is not RCSPNaiveCG:
                assert res["global_lb"] == pytest.approx(opt)  # optimality proved
            assert all(bnd <= opt + 1e-6 for bnd in bounds), (cls.__name__, bounds, opt)
            if res["global_lb"] is not None:
                assert res["global_lb"] <= opt + 1e-6


def test_converged_bounds_equal_lp_over_all_duties(synthetic):
    for case in synthetic:
        lp = case["lp"]
        res = RCSPNaiveCG(time_limit=4).run(str(case["spec"]))
        assert res["cg_converged"] and res["lp_value"] == pytest.approx(lp, abs=1e-4)
        assert lp - 1.0 <= res["global_lb"] <= math.ceil(lp - 1e-6)
