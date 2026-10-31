"""Correctness checks for the scale-up planning-and-scheduling (LBBD) family: loader and generator, facility oracle,
relaxation and cores, time-indexed MIP exactness, naive and engineered LBBD, decomposition-free controls and the
verifier, against an independent brute force on tiny instances."""
from __future__ import annotations

import functools
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pytest

gp = pytest.importorskip("gurobipy")
pytest.importorskip("ortools")
from gurobipy import GRB  # noqa: E402

from src.bd.lbbd.a_data import (PSInstance, generate, load_hooker_dat, load_instance, parse_hooker_dat,  # noqa: E402
                                random_instance, schedule_cost, write_instance)
from src.bd.lbbd.a_scheduling import (FEASIBLE, INFEASIBLE, UNKNOWN, FacilityOracle, Relaxation, bits,  # noqa: E402
                                      cpsat_check, greedy_solution, sgs_any, small_exact, to_mask)
from src.bd.lbbd.b_mip_xl import PSTimeIndexedMIP, build_time_indexed, fix_solution, make_env, ps_output  # noqa: E402
from src.bd.lbbd.c_naive_bd import PSNaiveLBBD  # noqa: E402
from src.bd.lbbd.d_ref_bd_xl import PSBendersXL  # noqa: E402
from src.bd.lbbd.e_verifier import verify_solution  # noqa: E402
from src.bd.lbbd.f_heuristic import PSCPSatMonolithic, PSLocalSearch  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "outputs" / "scaleup" / "raw" / "lbbd" / "hooker_zip" / "dat"


# ---------------------------------------------------------------- independent brute force

def _earliest(prof: list[int], cap: int, r: int, d: int, p: int, c: int) -> int | None:
    for s in range(r, d - p + 1):
        if all(prof[t] + c <= cap for t in range(s, s + p)):
            return s
    return None


def brute_feasible(inst: PSInstance, i: int, jobs: tuple[int, ...]) -> bool:
    """Feasibility of `jobs` on facility i: depth-first search over serial-SGS job orders (active schedules contain a
    feasible schedule whenever one exists: deadlines = maximum lateness <= 0, a regular objective). A job that cannot be
    placed now cannot be placed later (loads only grow), which prunes the node."""
    cap = int(inst.capacity[i])
    P = [int(v) for v in inst.proc[i]]
    Cr = [int(v) for v in inst.rate[i]]
    R = [int(v) for v in inst.release]
    D = [int(v) for v in inst.deadline]
    seen: set = set()

    def rec(prof: tuple[int, ...], rest: frozenset) -> bool:
        if not rest:
            return True
        key = (prof, rest)
        if key in seen:
            return False
        seen.add(key)
        place = {}
        for j in rest:
            s = _earliest(list(prof), cap, R[j], D[j], P[j], Cr[j])
            if s is None:
                return False
            place[j] = s
        for j in sorted(rest):
            pr = list(prof)
            for t in range(place[j], place[j] + P[j]):
                pr[t] += Cr[j]
            if rec(tuple(pr), rest - {j}):
                return True
        return False

    return rec(tuple([0] * (inst.horizon + 1)), frozenset(jobs))


def brute_feasible_dfs(inst: PSInstance, i: int, jobs: tuple[int, ...]) -> bool:
    """Plain enumeration of all integer start vectors (very small sets only): a check of brute_feasible."""
    ranges = [range(int(inst.release[j]), int(inst.deadline[j]) - int(inst.proc[i, j]) + 1) for j in jobs]
    for starts in itertools.product(*ranges):
        prof = [0] * (inst.horizon + 1)
        ok = True
        for j, s in zip(jobs, starts):
            for t in range(s, s + int(inst.proc[i, j])):
                prof[t] += int(inst.rate[i, j])
                if prof[t] > inst.capacity[i]:
                    ok = False
        if ok:
            return True
    return False


@functools.lru_cache(maxsize=None)
def _feasible_sets(path: str) -> tuple:
    """For every facility, the feasibility of every job subset (by increasing size; a set with an infeasible
    subset is infeasible)."""
    inst = load_instance(path)
    out = []
    for i in range(inst.m):
        feas: dict[int, bool] = {0: True}
        for size in range(1, inst.n + 1):
            for comb in itertools.combinations(range(inst.n), size):
                mask = to_mask(comb)
                if any(not feas[mask & ~(1 << j)] for j in comb):
                    feas[mask] = False
                elif not all(inst.eligible[i, j] for j in comb):
                    feas[mask] = False
                else:
                    feas[mask] = brute_feasible(inst, i, comb)
        out.append(feas)
    return tuple(out)


def brute_optimum(path: str) -> float | None:
    inst = load_instance(path)
    feas = _feasible_sets(path)
    best = None
    for assign in itertools.product(range(inst.m), repeat=inst.n):
        masks = [0] * inst.m
        for j, i in enumerate(assign):
            masks[i] |= 1 << j
        if all(feas[i][masks[i]] for i in range(inst.m)):
            c = sum(int(inst.cost[i, j]) for j, i in enumerate(assign))
            best = c if best is None else min(best, c)
    return best


SHAPES = [(5, 2, 8), (6, 2, 8), (5, 3, 7), (6, 3, 8), (6, 2, 10), (5, 3, 9), (7, 2, 8), (6, 3, 7)]


def tiny(seed: int, tmp_path: Path) -> tuple[PSInstance, str]:
    rng = np.random.default_rng(seed)
    n, m, H = SHAPES[seed % len(SHAPES)]
    inst = random_instance(rng, n, m, horizon=H, pmax=4, capacity=int(rng.integers(3, 5)), windows=seed % 2 == 0)
    path = tmp_path / f"tiny{seed}.json"
    write_instance(inst, path)
    return load_instance(path), str(path)


def write_out(tmp_path: Path, out: dict) -> Path:
    p = tmp_path / "out.json"
    p.write_text(json.dumps(out))
    return p


def test_ortools_cpsat_is_importable():
    import ortools
    from ortools.sat.python import cp_model
    print(f"ortools {ortools.__version__}")
    assert cp_model.CpSolver().parameters.num_workers >= 0


# ---------------------------------------------------------------- data

@pytest.mark.skipif(not RAW.exists(), reason="Hooker instance files not downloaded")
def test_hooker_files_parse_and_roundtrip(tmp_path):
    files = [f for f in sorted(RAW.glob("*.dat")) if "JobCount" in f.read_text()]
    assert len(files) == 335
    for f in files:
        inst = load_hooker_dat(f)
        named = int(f.stem.lstrip("cdef").split("j")[0])
        # the five df20j3m files hold 18 jobs (JobCount = 18), a quirk of the published set
        assert inst.n == (18 if f.stem.startswith("df20j3m") else named), f.stem
        assert inst.m == int(f.stem.split("j")[1].split("m")[0]) and np.all(inst.capacity == 10)
    a = load_hooker_dat(RAW / "c10j2m1.dat")
    assert a.proc[:, 0].tolist() == [4, 6] and a.rate[:, 0].tolist() == [9, 9] and a.cost[:, 0].tolist() == [10, 30]
    assert a.release.tolist() == [0] * 10 and a.deadline.tolist() == [25] * 10
    write_instance(a, tmp_path / "a.json")
    b = load_instance(tmp_path / "a.json")
    for name in ("proc", "rate", "cost", "capacity", "release", "deadline"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    df = load_hooker_dat(RAW / "df14j3m1.dat")
    assert df.release[2] == 44 and df.deadline[2] == 69


def test_generator_follows_the_recipes():
    a, b = generate("c", 40, 4, 3), generate("c", 40, 4, 3)
    assert np.array_equal(a.proc, b.proc) and np.array_equal(a.cost, b.cost) and a.name == "gc40j4m_s3"
    for i in range(4):
        k = 4 - i
        assert a.proc[i].min() >= k and a.proc[i].max() < 10 * k
        assert a.cost[i].min() >= 10 * (i + 1) and a.cost[i].max() < 20 * (i + 1)
    assert np.all(a.rate == a.rate[0]) and a.rate.min() >= 1 and a.rate.max() <= 9
    assert a.deadline[0] == 83 and np.all(a.release == 0)
    e = generate("e", 100, 20, 1)
    assert e.n == 100 and e.m == 20 and np.all(e.deadline == 33)
    assert e.proc[0].max() <= 24 and e.proc[-1].max() <= 14 and e.cost[0].min() >= 16 and e.cost[-1].max() <= 53


@pytest.mark.skipif(not RAW.exists(), reason="Hooker instance files not downloaded")
def test_generator_ranges_match_the_published_e_and_c_files():
    """The largest published p and the published cost ranges per facility agree with the generator's formulas."""
    for m in (4, 6, 8, 10):
        n = 5 * m
        insts = [load_hooker_dat(RAW / f"e{n}j{m}m{k}.dat") for k in range(1, 6)]
        for i in range(m):
            top = int(np.floor(24 - 10 * i / (m - 1) + 0.5))
            assert max(int(x.proc[i].max()) for x in insts) == top, (m, i)
    insts = [load_hooker_dat(RAW / f"e50j10m{k}.dat") for k in range(1, 6)]
    for i in range(10):
        low = int(np.floor(16 + 11 * i / 9 + 0.5))
        assert min(int(x.cost[i].min()) for x in insts) == low and max(int(x.cost[i].max()) for x in insts) == 2 * low - 1
    for m in (2, 3, 4):
        insts = [load_hooker_dat(RAW / f"c30j{m}m{k}.dat") for k in range(1, 6)] if m < 4 else \
            [load_hooker_dat(RAW / f"c32j4m{k}.dat") for k in range(1, 6)]
        for i in range(m):
            k = m - i
            assert min(int(x.proc[i].min()) for x in insts) >= k and max(int(x.proc[i].max()) for x in insts) < 10 * k


# ---------------------------------------------------------------- brute force self-check, oracle, relaxation, cores

def test_brute_force_orders_agree_with_plain_enumeration(tmp_path):
    for seed in range(6):
        inst, _ = tiny(seed, tmp_path)
        for i in range(inst.m):
            for size in (1, 2, 3, 4):
                for comb in itertools.islice(itertools.combinations(range(inst.n), size), 12):
                    if all(inst.eligible[i, j] for j in comb):
                        assert brute_feasible(inst, i, comb) == brute_feasible_dfs(inst, i, comb), (seed, i, comb)


@pytest.mark.parametrize("seed", range(8))
def test_oracle_and_cpsat_agree_with_brute_force(tmp_path, seed):
    inst, path = tiny(seed, tmp_path)
    feas = _feasible_sets(path)
    oracle = FacilityOracle(inst, seed=0)
    for i in range(inst.m):
        for mask, ok in feas[i].items():
            st, starts = oracle.check(i, mask, 10.0)
            assert st == (FEASIBLE if ok else INFEASIBLE), (i, bits(mask))
            if ok:
                assert sorted(starts) == bits(mask)
                from src.bd.lbbd.a_data import machine_load_ok
                assert machine_load_ok(inst, i, list(starts), list(starts.values()))
            jobs = bits(mask)
            if jobs and all(inst.eligible[i, j] for j in jobs):
                st2, _ = cpsat_check(inst, i, jobs, 10.0)
                assert st2 == (FEASIBLE if ok else INFEASIBLE)
                st3, sch = small_exact(inst, i, jobs)
                assert st3 == (FEASIBLE if ok else INFEASIBLE)
                if ok:
                    assert sorted(sch) == jobs
        # cached answers stay consistent
        for mask, ok in itertools.islice(feas[i].items(), 40):
            assert oracle.check(i, mask, 10.0)[0] == (FEASIBLE if ok else INFEASIBLE)


@pytest.mark.parametrize("seed", range(8))
def test_relaxation_rows_hold_for_every_feasible_set(tmp_path, seed):
    inst, path = tiny(seed, tmp_path)
    feas = _feasible_sets(path)
    relax = Relaxation(inst)
    assert relax.size() >= 0
    for i in range(inst.m):
        for mask, ok in feas[i].items():
            if ok:
                x = np.zeros(inst.n)
                x[bits(mask)] = 1.0
                assert relax.violated(i, x) is None, (i, bits(mask))


@pytest.mark.parametrize("seed", range(8))
def test_cores_are_irreducible_infeasible_subsets(tmp_path, seed):
    inst, path = tiny(seed, tmp_path)
    feas = _feasible_sets(path)
    oracle = FacilityOracle(inst, seed=0)
    checked = 0
    for i in range(inst.m):
        for mask, ok in feas[i].items():
            if ok or checked > 40:
                continue
            checked += 1
            core = oracle.core(i, mask, time.perf_counter() + 30)
            assert core and core & ~mask == 0
            assert not feas[i][core]
            for j in bits(core):
                assert feas[i][core & ~(1 << j)], (i, bits(mask), bits(core), j)


def test_small_exact_and_cpsat_agree_on_larger_sets(mid_instance):
    """On a published c instance: random job sets of 6-10 jobs on each facility; the enumeration (when it finishes)
    and CP-SAT give the same answer, and the oracle with the enumeration disabled agrees with the default oracle."""
    inst = load_instance(mid_instance)
    rng = np.random.default_rng(3)
    plain = FacilityOracle(inst, small_size=0)
    agree = 0
    for _ in range(60):
        i = int(rng.integers(inst.m))
        el = [j for j in range(inst.n) if inst.eligible[i, j]]
        jobs = sorted(int(j) for j in rng.choice(el, size=int(rng.integers(6, 11)), replace=False))
        st_cp, _ = cpsat_check(inst, i, jobs, 20.0)
        st_small, sch = small_exact(inst, i, jobs, node_limit=200000)
        if st_small != UNKNOWN and st_cp != UNKNOWN:
            assert st_small == st_cp, (i, jobs)
            agree += 1
        assert plain.check(i, to_mask(jobs), 20.0)[0] in (st_cp, UNKNOWN)
    assert agree >= 30


def test_greedy_and_sgs_schedules_are_exact(tmp_path):
    for seed in range(8):
        inst, path = tiny(seed, tmp_path)
        g = greedy_solution(inst)
        if g is not None:
            assert schedule_cost(inst, g[0]) == g[1]
            assert brute_optimum(path) is not None and g[1] >= brute_optimum(path)
        for i in range(inst.m):
            jobs = [j for j in range(inst.n) if inst.eligible[i, j]][:4]
            st = sgs_any(inst, i, jobs)
            if st is not None:
                from src.bd.lbbd.a_data import machine_load_ok
                assert machine_load_ok(inst, i, list(st), list(st.values()))
                assert _feasible_sets(path)[i][to_mask(jobs)]


# ---------------------------------------------------------------- MIP exactness

def random_schedule(inst: PSInstance, rng: np.random.Generator) -> list[list[int]]:
    sol = []
    for j in range(inst.n):
        i = int(rng.integers(inst.m))
        lo = max(0, int(inst.release[j]) - 1)
        hi = max(lo, int(inst.deadline[j]) - int(inst.proc[i, j]) + 1)
        sol.append([i, int(rng.integers(lo, hi + 1))])
    return sol


def test_time_indexed_mip_admits_a_schedule_iff_the_verifier_accepts(tmp_path):
    env = make_env()
    rng = np.random.default_rng(5)
    admitted = rejected = 0
    for seed in range(6):
        inst, path = tiny(10 + seed, tmp_path)
        model, info = build_time_indexed(inst, env)
        cands = [random_schedule(inst, rng) for _ in range(60)]
        g = greedy_solution(inst)
        if g is not None:
            cands.append(g[0])
        for sol in cands:
            fix_solution(model, info, sol)
            model.optimize()
            out = ps_output(inst, sol, method="test")
            if out["objective_value"] is None:
                out["objective_value"] = 0.0
            ver = verify_solution(path, write_out(tmp_path, out))
            ok = model.Status == GRB.OPTIMAL
            assert ok == ver["ok"] == (schedule_cost(inst, sol) is not None), (seed, sol, ver["issues"])
            if ok:
                admitted += 1
                assert model.ObjVal == pytest.approx(ver["objective_value"], abs=1e-9)
            else:
                rejected += 1
            fix_solution(model, info, None)
        model.dispose()
    env.dispose()
    assert admitted >= 2 and rejected >= 50


# ---------------------------------------------------------------- methods vs brute force

def _check_run(name, res, seen, bounds, opt, path, inst, tmp_path, exact=True):
    vals = [schedule_cost(inst, s) for s in seen]
    assert all(v is not None for v in vals), name
    assert vals == sorted(vals, reverse=True) and len(set(vals)) == len(vals), name
    assert all(b <= opt + 1e-6 for b in bounds) and bounds == sorted(bounds), name
    if exact:
        assert res["termination_reason"] == "optimal", (name, res["termination_reason"])
        assert min(vals) == opt and res["best_objective"] == opt, name
        assert res["global_lb"] <= opt + 1e-6
    else:
        assert min(vals) >= opt
    out = ps_output(inst, seen[-1], status="feasible", method=name)
    assert verify_solution(path, write_out(tmp_path, out))["ok"], name


@pytest.mark.parametrize("seed", range(8))
def test_methods_find_the_optimum_with_valid_bounds(tmp_path, seed):
    inst, path = tiny(seed, tmp_path)
    opt = brute_optimum(path)
    variants = {"mip": PSTimeIndexedMIP, "mip_init": functools.partial(PSTimeIndexedMIP, warm_start=True),
                "naive": PSNaiveLBBD, "naive_relax": functools.partial(PSNaiveLBBD, relaxation=True),
                "eng": PSBendersXL, "eng_nostrong": functools.partial(PSBendersXL, strengthen=False),
                "eng_norelax": functools.partial(PSBendersXL, relaxation=False),
                "eng_norepair_nowarm": functools.partial(PSBendersXL, repair=False, warm_start=False),
                "eng_onecore": functools.partial(PSBendersXL, max_cores=1),
                "cpsat": PSCPSatMonolithic}
    for name, cls in variants.items():
        seen, bounds = [], []
        res = cls(time_limit=60, on_incumbent=lambda s, stage=None: seen.append([list(v) for v in s]),
                  on_bound=bounds.append).run(path)
        if opt is None:
            assert res["termination_reason"] == "infeasible" and not seen, name
            continue
        _check_run(name, res, seen, bounds, opt, path, inst, tmp_path)
    if opt is not None:
        seen = []
        res = PSLocalSearch(time_limit=1.0, on_incumbent=lambda s, stage=None: seen.append([list(v) for v in s])).run(path)
        if seen:
            _check_run("ls", res, seen, [], opt, path, inst, tmp_path, exact=False)
        assert res["global_lb"] is None


def test_runner_callback_conventions(tmp_path):
    """The runner's conventions: the MIP, the naive control and the decomposition-free controls publish with one
    argument; the engineered reference passes its stage (provenance) as a keyword."""
    path = next(p for p in (tiny(s, tmp_path)[1] for s in range(8)) if brute_optimum(p) is not None)
    for cls in (PSTimeIndexedMIP, PSNaiveLBBD, PSCPSatMonolithic, functools.partial(PSLocalSearch, time_limit=0.5)):
        seen = []
        kwargs = {} if isinstance(cls, functools.partial) else {"time_limit": 30}
        res = cls(on_incumbent=lambda s: seen.append(s), on_bound=lambda b: None, **kwargs).run(path)
        assert seen and res["best_stage"] is not None, cls
    stages = []
    PSBendersXL(time_limit=30, on_incumbent=lambda s, stage=None: stages.append(stage)).run(path)
    assert stages and all(isinstance(s, str) for s in stages)


def test_an_unschedulable_instance_is_reported_infeasible(tmp_path):
    inst = PSInstance(proc=np.array([[3, 3, 3], [3, 3, 3]]), rate=np.array([[2, 2, 2], [2, 2, 2]]),
                      cost=np.array([[1, 2, 3], [4, 5, 6]]), capacity=np.array([2, 2]), release=np.array([0, 0, 0]),
                      deadline=np.array([4, 4, 4]), name="unschedulable")
    inst.validate()
    path = tmp_path / "u.json"
    write_instance(inst, path)
    for cls in (PSTimeIndexedMIP, PSNaiveLBBD, PSBendersXL, PSCPSatMonolithic):
        seen = []
        res = cls(time_limit=30, on_incumbent=lambda s, stage=None: seen.append(s)).run(str(path))
        assert res["termination_reason"] == "infeasible" and not seen and res["best_objective"] is None, cls


# ---------------------------------------------------------------- verifier and contract

def test_verifier_rejects_bad_outputs(tmp_path):
    inst, path = tiny(3, tmp_path)
    g = greedy_solution(inst)
    assert g is not None
    good = ps_output(inst, g[0], method="test")

    def check(out):
        return verify_solution(path, write_out(tmp_path, out))

    assert check(good)["ok"] and check(good)["objective_value"] == g[1]
    assert check(good | {"schedule": [dict(e, start=float(e["start"])) for e in good["schedule"]]})["ok"]
    assert not check(good | {"objective_value": good["objective_value"] + 1})["ok"]
    assert not check(good | {"objective_value": float("nan")})["ok"]
    assert not check({k: v for k, v in good.items() if k != "method"})["ok"]
    assert not check(good | {"status": "infeasible"})["ok"]
    assert not check(good | {"schedule": good["schedule"][:-1]})["ok"]
    assert not check(good | {"schedule": good["schedule"] + good["schedule"][:1]})["ok"]
    assert not check(good | {"schedule": [dict(good["schedule"][0], start=0.5)] + good["schedule"][1:]})["ok"]
    assert not check(good | {"schedule": [dict(good["schedule"][0], facility=inst.m)] + good["schedule"][1:]})["ok"]
    e0 = good["schedule"][0]
    late = dict(e0, start=int(inst.deadline[0]) - int(inst.proc[e0["facility"], 0]) + 1)
    assert not check(good | {"schedule": [late] + good["schedule"][1:]})["ok"]
    early = dict(e0, start=int(inst.release[0]) - 1)
    assert not check(good | {"schedule": [early] + good["schedule"][1:]})["ok"]
    # everything on facility 0 at time r_j overloads it (or breaks a window)
    crowded = [{"job": j, "facility": 0, "start": int(inst.release[j])} for j in range(inst.n)]
    out = {"status": "feasible", "method": "t", "objective_value": float(inst.cost[0].sum()), "schedule": crowded}
    assert check(out)["ok"] == (schedule_cost(inst, [[0, int(inst.release[j])] for j in range(inst.n)]) is not None)


def test_schedule_cost_agrees_with_the_verifier(tmp_path):
    rng = np.random.default_rng(11)
    for seed in range(4):
        inst, path = tiny(20 + seed, tmp_path)
        for _ in range(100):
            sol = random_schedule(inst, rng)
            c = schedule_cost(inst, sol)
            out = {"status": "feasible", "method": "t", "objective_value": c if c is not None else 0.0,
                   "schedule": [{"job": j, "facility": i, "start": s} for j, (i, s) in enumerate(sol)]}
            ver = verify_solution(path, write_out(tmp_path, out))
            assert ver["ok"] == (c is not None)


# ---------------------------------------------------------------- deadlines and labels on a mid-size instance

@pytest.fixture(scope="module")
def mid_instance(tmp_path_factory):
    d = tmp_path_factory.mktemp("lbbd")
    src = RAW / "c30j3m1.dat"
    inst = load_hooker_dat(src) if src.exists() else generate("c", 30, 3, 1)
    path = d / f"{inst.name}.json"
    write_instance(inst, path)
    return str(path)


@pytest.mark.parametrize("name", ["mip", "naive", "eng", "cpsat", "ls"])
def test_deadline_is_respected(mid_instance, name):
    cls = {"mip": PSTimeIndexedMIP, "naive": PSNaiveLBBD, "eng": PSBendersXL, "cpsat": PSCPSatMonolithic,
           "ls": PSLocalSearch}[name]
    T = 8.0
    t0 = time.perf_counter()
    seen, bounds = [], []
    res = cls(time_limit=T, on_incumbent=lambda s, stage=None: seen.append((time.perf_counter() - t0, s)),
              on_bound=lambda b: bounds.append((time.perf_counter() - t0, b))).run(mid_instance)
    assert res["total_seconds"] < T + 3.0, name
    assert all(t < T for t, _ in seen) and all(t < T for t, _ in bounds), name
    if res["termination_reason"] != "optimal":
        assert res["termination_reason"] in ("time_limit", "deadline_during_build") and res["budget_exhausted"], res
    inst = load_instance(mid_instance)
    vals = [schedule_cost(inst, s) for _, s in seen]
    assert all(v is not None for v in vals)
    if seen and bounds:
        assert bounds[-1][1] <= min(vals) + 1e-6
    print(name, res["termination_reason"], min(vals) if vals else None, bounds[-1][1] if bounds else None)


def test_mip_labels_a_build_overrun(mid_instance):
    res = PSTimeIndexedMIP(time_limit=1e-4).run(mid_instance)
    assert res["termination_reason"] == "deadline_during_build" and res["best_objective"] is None


def test_callback_errors_are_not_swallowed(tmp_path):
    """gurobipy ignores exceptions raised in callbacks; the reference must re-raise them, never accept an unchecked
    assignment."""
    path = next(p for p in (tiny(s, tmp_path)[1] for s in range(8)) if brute_optimum(p) is not None)

    class Failing(PSBendersXL):
        def search(self, env):
            def boom(*args, **kwargs):
                raise RuntimeError("oracle failure")
            self.oracle.check = boom
            return super().search(env)

    with pytest.raises(RuntimeError, match="oracle failure"):
        Failing(time_limit=30, warm_start=False).run(path)


# ---------------------------------------------------------------- review 37: certification, publication, time allowances

import src.bd.lbbd.a_scheduling as sched_mod  # noqa: E402
import src.bd.lbbd.f_heuristic as heur_mod  # noqa: E402


@pytest.fixture
def clock(monkeypatch):
    """A controlled clock: time.perf_counter() plus an offset the test can jump (Gurobi and CP-SAT keep their own
    clocks, so the solvers run on; only the methods' deadline checks see the jump)."""
    state = {"offset": 0.0}
    real = time.perf_counter
    monkeypatch.setattr(time, "perf_counter", lambda: real() + state["offset"])

    def jump():
        state["offset"] += 1e6

    return jump


def improvable(tmp_path) -> tuple[PSInstance, str, int, int]:
    """A tiny instance whose shared initializer is feasible but not optimal."""
    for s in range(40):
        inst, path = tiny(s, tmp_path)
        g = greedy_solution(inst)
        opt = brute_optimum(path)
        if g is not None and opt is not None and g[1] > opt:
            return inst, path, g[1], opt
    pytest.skip("no tiny instance with a suboptimal initializer")


def test_naive_does_not_certify_a_schedule_published_after_the_deadline(tmp_path, clock):
    """Review 37 (high): the master optimum is schedulable but its publication comes after the deadline; the retained
    incumbent is the initializer, so the run must end as a timeout, not as 'optimal'."""
    inst, path, greedy_cost, opt = improvable(tmp_path)

    class Late(PSNaiveLBBD):
        def publish(self, sol, stage):
            if stage == "master_optimum":
                clock()  # the deadline passes just before the proved optimum is published
            return super().publish(sol, stage)

    seen = []
    res = Late(time_limit=60, on_incumbent=lambda s: seen.append(schedule_cost(inst, s))).run(path)
    assert res["termination_reason"] == "time_limit", res["termination_reason"]
    assert res["best_objective"] == greedy_cost > opt and seen == [greedy_cost]
    assert res["global_lb"] is None or res["global_lb"] <= opt + 1e-6
    # without the clock jump the same run certifies the optimum
    res = PSNaiveLBBD(time_limit=60).run(path)
    assert res["termination_reason"] == "optimal" and res["best_objective"] == opt


def test_engineered_does_not_certify_a_schedule_published_after_the_deadline(tmp_path, clock):
    inst, path, greedy_cost, opt = improvable(tmp_path)
    state = {"jumped": False, "ub": None}

    class Late(PSBendersXL):
        def offer(self, sol, stage):
            if stage == "branch_and_check" and not state["jumped"]:
                state["jumped"], state["ub"] = True, self.ub
                clock()
            return super().offer(sol, stage)

    res = Late(time_limit=60, repair=False, warm_start=False).run(path)
    assert state["jumped"] and state["ub"] == greedy_cost
    assert res["termination_reason"] != "optimal" and res["best_objective"] == greedy_cost, res["termination_reason"]
    assert res["termination_reason"] in ("time_limit", "stuck")


def test_cpsat_does_not_certify_a_solution_found_after_the_deadline(tmp_path, clock, monkeypatch):
    inst, path, greedy_cost, opt = improvable(tmp_path)
    original = heur_mod._Publisher.offer

    def offer(self, sol, stage):
        if stage == "cpsat":
            clock()  # every CP-SAT solution arrives after the deadline
        return original(self, sol, stage)

    monkeypatch.setattr(heur_mod._Publisher, "offer", offer)
    res = PSCPSatMonolithic(time_limit=60).run(path)
    assert res["termination_reason"] == "time_limit" and res["best_objective"] == greedy_cost, res["termination_reason"]


def test_local_search_publishes_an_improvement_before_the_descent_ends(mid_instance, clock):
    """Review 37 (medium): the first improving move happens before the deadline, the deadline passes in the middle of
    the descent; the improvement must already be published (previously only the end of a descent published)."""
    inst = load_instance(mid_instance)

    class Spy(PSLocalSearch):
        def run(self, data_path):
            self.records = []
            return super().run(data_path)

        def after_move(self, stage):
            self.records.append(self.cost())
            super().after_move(stage)
            if len(self.records) == 1:
                clock()

    seen = []
    ls = Spy(time_limit=60, on_incumbent=lambda s: seen.append(schedule_cost(inst, s)))
    res = ls.run(mid_instance)
    assert ls.records and seen[0] > ls.records[0]
    assert seen == [seen[0], ls.records[0]] and res["best_objective"] == ls.records[0]


def test_oracle_uses_one_absolute_deadline(mid_instance, monkeypatch):
    """Review 37 (medium): time spent before CP-SAT (enumeration here) is consumed from the same allowance; CP-SAT is
    not called once the allowance is gone; an expired deadline gives UNKNOWN without solving."""
    inst = load_instance(mid_instance)
    calls = []

    def slow_small(inst_, i, jobs, node_limit=20000, deadline=None):
        time.sleep(0.3)
        return UNKNOWN, None

    def fake_cp(inst_, i, jobs, time_limit=None, seed=0, hint=None, deadline=None):
        calls.append(deadline)
        return UNKNOWN, None

    monkeypatch.setattr(sched_mod, "small_exact", slow_small)
    monkeypatch.setattr(sched_mod, "cpsat_check", fake_cp)
    monkeypatch.setattr(sched_mod, "sgs_any", lambda *a, **k: None)
    oracle = FacilityOracle(inst, small_size=100)
    monkeypatch.setattr(oracle.relax, "violated", lambda i, x: None)
    jobs = [j for j in range(inst.n) if inst.eligible[0, j]][:6]
    t0 = time.perf_counter()
    st, _ = oracle.check(0, to_mask(jobs), 1.0)
    assert st == UNKNOWN and len(calls) == 1
    assert calls[0] <= t0 + 1.0 + 1e-3 and calls[0] - time.perf_counter() < 0.75  # the 0.3 s were consumed
    st, _ = oracle.check(0, to_mask(jobs[:5]), 0.2)  # the enumeration alone exceeds the allowance
    assert st == UNKNOWN and len(calls) == 1
    # a real CP-SAT check with an expired deadline returns at once, unsolved
    monkeypatch.undo()
    t1 = time.perf_counter()
    assert cpsat_check(inst, 0, jobs, deadline=time.perf_counter() - 1.0)[0] == UNKNOWN
    assert time.perf_counter() - t1 < 0.5
    st, _ = small_exact(inst, 0, jobs, deadline=time.perf_counter() - 1.0)
    assert st == UNKNOWN
