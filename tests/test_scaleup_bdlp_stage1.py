"""UFL / stochastic CFL bound-side Stage 1 runner (scripts/phoenix/scaleup_bdlp_stage1.py), its panels and task lists
(scripts/phoenix/scaleup_bdlp_panel.py) and the new XL modules (src/bd/ufl/{a_data_xl,i_controls_xl}.py,
src/bd/scfl/{a_data_xl,i_controls_xl}.py, the lean compact builders in g_bound_xl).

Two groups:
- logic tests (no optimizer: generators, analytical cut sets, panel preregistration and freeze, acceptance, summary
  over synthetic results, the supervisor with solver-free arms) run everywhere;
- solver smoke tests (tiny instances, seconds each) run only on Phoenix (SLURM_JOB_ID) or through the laptop's serial
  wrapper solve1.sh (DECOMPGYM_LOCAL_SOLVE=1), one test per call.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOLVER_OK = bool(os.environ.get("SLURM_JOB_ID") or os.environ.get("DECOMPGYM_LOCAL_SOLVE") == "1")
solver = pytest.mark.skipif(not SOLVER_OK, reason="optimizer runs: Phoenix or the serial wrapper solve1.sh only")


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("_bdlp_runner_t", "scripts/phoenix/scaleup_bdlp_stage1.py")
P = _load("_bdlp_panel_t", "scripts/phoenix/scaleup_bdlp_panel.py")

# a tiny CAP-format base satisfying the SCFL fractional-domain invariant (every capacity below every scenario total)
CAP_TEXT = """ 4 4
 40 500.
 45 700.
 38 400.
 42 600.
 20 40. 60. 80. 50.
 30 90. 30. 60. 45.
 25 50. 75. 25. 40.
 10 30. 20. 10. 15.
"""
# the same costs with capacities above the scenario totals (violates the invariant)
CAP_WIDE = CAP_TEXT.replace(" 40 500.", " 140 500.")


def _ufl(tmp_path: Path, N: int = 10, c: str = "c", seed: int = 5) -> Path:
    from src.bd.ufl.a_data_xl import make_kg, write_instance
    obj = make_kg(N, c, seed)
    p = tmp_path / f"{obj['name']}.json"
    write_instance(obj, p)
    return p


def _scfl(tmp_path: Path, S: int = 6, seed: int = 3) -> Path:
    from src.bd.scfl.a_data_xl import make_instance, read_cap, write_instance
    cap = tmp_path / "captiny.txt"
    cap.write_text(CAP_TEXT)
    obj = make_instance(read_cap(cap), S, seed)
    p = tmp_path / f"{obj['name']}.json"
    write_instance(obj, p)
    return p


# ---------------------------------------------------------------- generators

def test_kg_generator_recipe_and_determinism(tmp_path):
    from src.bd.ufl import g_bound_xl as g
    from src.bd.ufl.a_data_xl import check_recipe, make_kg
    for c in "abc":
        obj = make_kg(30, c, 7)
        assert check_recipe(obj) == []
        D = np.asarray(obj["D"])
        assert np.array_equal(D, D.T) and D.min() >= 1000 and D.max() <= 2000
        assert obj == make_kg(30, c, 7) == g.make_kg(30, c, 7)    # deterministic; the probe entry point delegates
    a = make_kg(30, "b", 7, symmetric=False)
    assert a["name"] == "kga30b_g7" and check_recipe(a) == [] and not np.array_equal(np.asarray(a["D"]), np.asarray(a["D"]).T)
    bad = make_kg(5, "a", 1)
    bad["f"][0] = 5000
    assert check_recipe(bad)
    with pytest.raises(ValueError):
        make_kg(5, "d", 1)


def test_scfl_pinned_generator_matches_shared_module(tmp_path):
    from src.bd.scfl import a_data
    from src.bd.scfl.a_data_xl import make_instance, read_cap, write_instance
    cap = tmp_path / "captiny.txt"
    cap.write_text(CAP_TEXT)
    for S, seed in ((5, 1), (40, 9)):
        obj = make_instance(read_cap(cap), S, seed)
        h1 = write_instance(obj, tmp_path / "a.json")
        inst = a_data.make_stochastic(a_data.read_orlib_cap(cap), S, seed)
        h2 = a_data.write_instance(inst, tmp_path / "b.json")
        assert h1 == h2, "src/bd/scfl/a_data.py and the pinned a_data_xl.py diverged"
        a_data.load_instance(tmp_path / "a.json").validate()


def test_orlib_base_hashes_match_local_raw_files():
    raw = ROOT.parent / "DecompBench" / "outputs" / "scaleup" / "raw" / "scfl" / "orlib"
    if not raw.is_dir():
        pytest.skip("archived raw files not present")
    for base, h in P.CAP_SHA256.items():
        assert R.sha256((raw / f"{base}.txt").read_bytes()) == h


# ---------------------------------------------------------------- analytical controls (closed form, no optimizer)

def test_ufl_analytical_cuts_size_and_exactness(tmp_path):
    from src.bd.ufl.g_bound_xl import Sorted, load_instance
    from src.bd.ufl.i_controls_xl import analytical_cuts
    inst = load_instance(_ufl(tmp_path, N=9, c="b", seed=2))
    srt = Sorted(inst.D)
    j, v, R_ = analytical_cuts(srt, 10**9)
    assert R_ == inst.m
    assert int(len(j) + srt.counts(j, v).sum()) <= 10**9
    rng = np.random.default_rng(0)
    for _ in range(30):     # with every rank the cuts reproduce Q_j(y) = min_{i open} D_ij at every binary design
        y = rng.random(inst.m) < 0.4
        if not y.any():
            y[0] = True
        vals = v - np.array([(np.maximum(vv - inst.D[:, jj], 0.0) * y).sum() for jj, vv in zip(j, v)])
        for jj in range(inst.n):
            assert vals[j == jj].max() == pytest.approx(inst.D[y, jj].min())
            assert (vals[j == jj] <= inst.D[y, jj].min() + 1e-9).all()        # validity
    for limit in (inst.n, 3 * inst.n, 25):
        j2, v2, R2 = analytical_cuts(srt, limit)
        assert int(len(j2) + srt.counts(j2, v2).sum()) <= limit and R2 < inst.m


def test_scfl_analytical_terms_sizes(tmp_path):
    from src.bd.scfl.a_data import load_instance
    from src.bd.scfl.h_view_xl import term_sizes
    from src.bd.scfl.i_controls_xl import analytical_terms
    inst = load_instance(_scfl(tmp_path, S=4))
    lv, R_, size = analytical_terms(inst.cost, inst.S, 10**9)
    assert R_ == inst.I
    z = np.zeros((1, inst.I))
    assert size == inst.S * sum(int(term_sizes(inst.cost, v[None, :], z)[0]) for v in lv)
    lv2, R2, s2 = analytical_terms(inst.cost, inst.S, size - 1)
    assert R2 == inst.I - 1 and s2 <= size - 1
    # validity of the alpha = 0 term (dual feasible: beta = (v - c)^+ >= 0, reduced costs c - v + beta >= 0)
    for v in lv:
        beta = np.maximum(v[None, :] - inst.cost, 0.0)
        assert (inst.cost - v[None, :] + beta >= -1e-12).all()


# ---------------------------------------------------------------- panels and task lists

@pytest.mark.parametrize("fam", ["ufl", "scfl"])
def test_preregistered_panel_matches_builder_and_runner(fam):
    p = json.loads((P.PANELS / f"{fam}_stage1_panel.json").read_text())
    assert p == json.loads(json.dumps(P.build_panel(fam)))          # the committed file is the builder's output
    assert P.prereg_sha(p) == p["preregistration"]["sha256"]
    f = R.family(fam)
    assert p["arms"]["all"] == f.ALL_ARMS and set(f.BOUND_CONTROLS) <= set(f.ALL_ARMS)
    assert p["checkpoints"] == list(R.CHECKPOINTS) and p["T"] == 900.0
    assert p["rules"]["eps_b"] == 1e-5 and p["rules"]["primal_gate"] is False
    assert "analytical" in f.BOUND_CONTROLS and "pool" in f.BOUND_CONTROLS and "empty" in f.BOUND_CONTROLS
    names = set(p["instances"])
    for t in p["tiers"].values():
        assert len(t["stage1"]) == 4 and len(t["holdout"]) == 2 and set(t["stage1"] + t["holdout"]) <= names
    assert len(names) == sum(len(t["stage1"]) + len(t["holdout"]) for t in p["tiers"].values())   # all distinct
    for fname, lines in P.task_lists(p).items():
        assert (P.TASKS / fname).read_text().splitlines() == lines
    runs = (P.TASKS / f"{fam}_stage1_runs.txt").read_text().splitlines()
    assert len(runs) == 4 * len(f.ALL_ARMS)
    for line in runs:
        _, fm, path, arm = line.split()
        assert fm == fam and arm in f.ALL_ARMS and Path(path).stem in p["tiers"][p["tier_order"][0]]["stage1"]


def _ref_row(refdir: Path, fam: str, name: str, inst_sha: str, panel_sha: str, code: dict, lo="999999999/1000", hi="1000000"):
    (refdir / f"{name}__c.json").write_text("{}")
    (refdir / f"{name}__p.json").write_text("[]")
    row = {"family": fam, "instance": name, "instance_sha256": inst_sha,
           "interval": {"lo": lo, "hi": hi, "rigorous": True, "rel_width": 0.0},
           "canonical": {"cert_ok": True, "lb": lo}, "lp_hi": {"value": hi},
           "certificate_file": f"{name}__c.json", "certificate_sha256": R.sha256(b"{}"),
           "lp_point_file": f"{name}__p.json", "lp_point_sha256": R.sha256(b"[]"),
           "identity": {"panel": {"sha256": panel_sha}, "code_sha256": dict(code), "python": "3.11.6", "gurobi": "13.0.1"}}
    if fam == "scfl":
        (refdir / f"{name}__w.npz").write_bytes(b"w")
        row["lp_hi"].update(witness_file=f"{name}__w.npz", witness_sha256=R.sha256(b"w"))
    (refdir / f"{name}__lp_reference.json").write_text(json.dumps(row))
    return row


def test_freeze_is_additive_and_checks_preregistration(tmp_path):
    panel = P.build_panel("ufl")
    pp = tmp_path / "ufl_panel.json"
    pp.write_text(json.dumps(panel))
    raw = pp.read_bytes()
    man = {"family": "ufl", "panel_sha256": R.sha256(raw),
           "instances": {n: {"name": n, "sha256": f"h{n}", "recipe_problems": []} for n in panel["instances"]}}
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(man))
    refs = tmp_path / "refs"
    refs.mkdir()
    code = P.code_hashes(R, "ufl")
    s1 = panel["tiers"]["N4000"]["stage1"]
    for n in s1:
        _ref_row(refs, "ufl", n, f"h{n}", R.sha256(raw), code)
    bad = copy.deepcopy(panel)
    bad["rules"]["eps_b"] = 1e-4
    bp = tmp_path / "bad.json"
    bp.write_text(json.dumps(bad))
    with pytest.raises(SystemExit):
        P.freeze("ufl", mp, refs, None, bp)          # a preregistered field changed
    # a useless interval that declares rel_width 0 is rejected: the width is recomputed exactly
    _ref_row(refs, "ufl", s1[0], f"h{s1[0]}", R.sha256(raw), code, lo="0")
    with pytest.raises(SystemExit):
        P.freeze("ufl", mp, refs, None, pp)
    _ref_row(refs, "ufl", s1[0], f"h{s1[0]}", R.sha256(raw), dict(code, **{"src/bd/ufl/g_bound_xl.py": "x"}))
    with pytest.raises(SystemExit):
        P.freeze("ufl", mp, refs, None, pp)          # produced with other code
    row = _ref_row(refs, "ufl", s1[0], f"h{s1[0]}", R.sha256(raw), code)
    for edit in ({"identity": dict(row["identity"], python="3.12.11")},          # another runtime
                 {"canonical": {"cert_ok": True, "lb": "1"}},                    # LP_lo not the certified bound
                 {"lp_hi": {"value": "2000000"}},                                # LP_hi not the exact value
                 {"canonical": {"cert_ok": False, "lb": row["interval"]["lo"]}}):
        (refs / f"{s1[0]}__lp_reference.json").write_text(json.dumps(dict(row, **edit)))
        with pytest.raises(SystemExit):
            P.freeze("ufl", mp, refs, None, pp)
    _ref_row(refs, "ufl", s1[0], f"h{s1[0]}", R.sha256(raw), code)
    (refs / f"{s1[1]}__p.json").write_text("[1]")   # witness tampered
    with pytest.raises(SystemExit):
        P.freeze("ufl", mp, refs, None, pp)
    (refs / f"{s1[1]}__p.json").write_text("[]")
    out = P.freeze("ufl", mp, refs, "abc", pp)
    assert out["status"].startswith("FROZEN") and len(out["lp_star_frozen"]["intervals"]) == 4
    assert {k: v for k, v in out.items() if k not in P.FROZEN_ONLY} == {k: v for k, v in panel.items() if k not in P.FROZEN_ONLY}
    assert out["code_sha256_at_preparation"] == code and out["freeze"]["task_lists_sha256"]
    with pytest.raises(SystemExit):
        P.freeze("ufl", mp, refs, None, pp)          # already frozen


# ---------------------------------------------------------------- acceptance and summary (synthetic results)

def _per(t_bound, target_T, bgap, infra=None, near=None, best=None):
    return {"t_bound": t_bound, "bound_target_T": target_T, "bgap_LP_T": bgap, "bgap_LP_T_exact": str(Fraction(bgap)),
            "infrastructure": infra or [], "near_best": near, "best": best, "gap_best": None}


def _rules():
    return dict(P.RULES)


def test_accept_rules():
    f = R.UFLFamily()
    base = {a: _per(None, False, 0.01) for a in f.ALL_ARMS}
    base["eng"] = _per(300.0, True, 1e-9)
    base["naive_lp"] = _per(None, False, 0.004)          # quality fails (0.4 points), speed branch: 900 >= 3 * 300
    acc = R.accept(f, base, "final_target", _rules(), 900.0)
    assert acc["pass"] is True and acc["naive_separation_branch"] == "speed (censored at T)"
    slow = dict(base, eng=_per(600.0, True, 1e-9))
    assert R.accept(f, slow, "final_target", _rules(), 900.0)["pass"] is False         # neither branch
    ctrl = dict(base, analytical=_per(900.0, True, 5e-6))
    a2 = R.accept(f, ctrl, "final_target", _rules(), 900.0)
    assert a2["pass"] is False and a2["bound_controls"]["analytical"] == "NOT separated"
    infra = dict(base, pool=_per(None, False, 0.01, infra=["arm_run: x"]))
    assert R.accept(f, infra, "final_target", _rules(), 900.0)["pass"] is None
    diag_infra = dict(base, lagr=_per(None, None, 0.01, infra=["arm_run: x"]))      # diagnostics never block
    assert R.accept(f, diag_infra, "final_target", _rules(), 900.0)["pass"] is True
    miss = {k: v for k, v in base.items() if k != "lp_bar"}
    assert R.accept(f, miss, "final_target", _rules(), 900.0)["pass"] is None
    noeng = dict(base, eng=_per(None, False, 1e-4))
    assert R.accept(f, noeng, "final_target", _rules(), 900.0)["pass"] is False
    gated = dict(_rules(), primal_gate=True)
    g = dict(base, eng=dict(base["eng"], near_best=True, best=100.0), ils=dict(base["ils"], near_best=True, best=100.0),
             naive_int=dict(base["naive_int"], best=110.0))
    assert R.accept(f, g, "final_target", gated, 900.0, z_best=100.0)["pass"] is False      # the ILS control is near best
    sf = R.SCFLFamily()
    sbase = {a: _per(None, False, 0.01) for a in sf.ALL_ARMS}
    sbase["eng"] = _per(300.0, True, 0.0)
    sbase["naive_lp"] = _per(None, False, 0.11)
    assert R.accept(sf, sbase, "final_target", _rules(), 900.0)["naive_separation_branch"] == "quality"
    assert R.accept(sf, dict(sbase, jensen=_per(600.0, True, 1e-6)), "final_target", _rules(), 900.0)["pass"] is False


def _fake_result(fam, name, arm, panel_path, inst_sha, lb, best, T, seal_sha=None, solver_bound=None):
    ident = R.identity(fam, {"instance": name, "instance_sha256": inst_sha, "arm": arm, "T": T,
                             "reference_seal_sha256": seal_sha}, Path(panel_path))
    ident["code_commit"] = None
    cps = [{"checkpoint": T, "seq": 1 if lb is not None else None, "cert_ok": lb is not None, "lb": str(lb if lb is not None else 0),
            "reason": None if lb is not None else "absent"}]
    return {"family": fam.name, "instance": name, "arm": arm, "T": T,
            "run": {"returncode": 0, "killed_at_T": False, "terminal_result": True, "infrastructure": [], "torn_messages": 0},
            "incumbents": {"best": best, "verified": True if best is not None else None, "invalid": 0, "malformed": 0},
            "solver_bounds": {"final": solver_bound}, "certificates": {"canonical": True, "checkpoints": cps}, "identity": ident}


def test_summarize_end_to_end_synthetic(tmp_path):
    fam = R.UFLFamily()
    panel = P.build_panel("ufl")
    panel["T"] = 900.0
    s1 = panel["tiers"]["N4000"]["stage1"]
    for n in panel["instances"]:
        panel["references"][n]["instance_sha256"] = f"sha-{n}"
    panel["lp_star_frozen"]["intervals"] = {n: {"lo": "999999", "hi": "1000000", "rigorous": True, "rel_width": 1e-8,
                                                "instance_sha256": f"sha-{n}"} for n in s1}
    panel["code_sha256_at_preparation"] = P.code_hashes(R, "ufl")
    panel["status"] = "FROZEN (test)"
    panel.pop("runtime")                                # the laptop runtime differs from the Phoenix pins
    pp = tmp_path / "panel.json"
    pp.write_text(json.dumps(panel))
    out = tmp_path / "res"
    good = Fraction(1000000) - Fraction(1, 1000)        # within eps_b of LP_hi
    for i, n in enumerate(s1):
        for arm in fam.ALL_ARMS:
            lb = good if arm == "eng" else (Fraction(990000) if arm != "empty" else None)   # 1% short
            if i == 3 and arm == "analytical":
                lb = good                               # the analytical control reaches the target on one instance
            d = out / f"{n}__{arm}"
            d.mkdir(parents=True)
            r = _fake_result(fam, n, arm, pp, f"sha-{n}", lb, 2000000.0 if arm in ("eng", "ils") else None, 900.0)
            (d / "result.json").write_text(json.dumps(r))
    (out / "stray_dir").mkdir()
    s = R.summarize("ufl", out, pp)
    dec = [s["instances"][n]["decision"] for n in s1]
    assert dec == ["pass", "pass", "pass", "fail"], [s["instances"][n]["acceptance"] for n in s1]
    assert s["tiers"]["N4000"]["stage1"] == "pass" and s["tiers"]["N4000"]["tier"] == "pending"   # holdout not run yet
    assert s["tiers"]["N3000c"]["tier"] == "not_needed" and "stray_dir" in s["unexpected"]
    # holdout: arms need the seal before they start; references are read only once sealed
    hold = panel["tiers"]["N4000"]["holdout"]
    code = panel["code_sha256_at_preparation"]
    for n in hold:
        _ref_row(out, "ufl", n, f"sha-{n}", R.sha256(pp.read_bytes()), code, lo="999999", hi="1000000")
    with pytest.raises(SystemExit):
        R.run_task("ufl", str(tmp_path / f"{hold[0]}.json"), "eng", 900.0, out, pp)       # no seal yet: refused
    s = R.summarize("ufl", out, pp)
    assert [s["instances"][n]["decision"] for n in hold] == ["pending", "pending"]
    P.seal("ufl", out, "N4000", "holdout", pp)
    with pytest.raises(FileExistsError):
        P.seal("ufl", out, "N4000", "holdout", pp)                                          # a seal is written once
    seal_sha = R.seal_sha256(out, "ufl", panel, hold[0])
    assert seal_sha
    for n in hold:
        for arm in fam.ALL_ARMS:
            d = out / f"{n}__{arm}"
            d.mkdir(parents=True)
            lb = good if arm == "eng" else (Fraction(990000) if arm != "empty" else None)
            # pool of the first holdout instance: started without the seal -> identity problem (pending)
            sh = None if (n == hold[0] and arm == "pool") else seal_sha
            (d / "result.json").write_text(json.dumps(_fake_result(fam, n, arm, pp, f"sha-{n}", lb,
                                                                   2000000.0 if arm == "eng" else None, 900.0, sh)))
    s = R.summarize("ufl", out, pp)
    assert [s["instances"][n]["decision"] for n in hold] == ["pending", "pass"]
    # a clean retry of the unsealed arm is scored; a refuted first attempt is replaced by a clean retry as well
    d = out / f"{hold[0]}__pool" / "attempt_2"
    d.mkdir()
    (d / "result.json").write_text(json.dumps(_fake_result(fam, hold[0], "pool", pp, f"sha-{hold[0]}", Fraction(990000),
                                                           None, 900.0, seal_sha)))
    (out / f"{hold[1]}__lp_bar" / "result.json").write_text(json.dumps(_fake_result(
        fam, hold[1], "lp_bar", pp, f"sha-{hold[1]}", Fraction(990000), None, 900.0, seal_sha, solver_bound=3e6)))
    s = R.summarize("ufl", out, pp)
    assert s["instances"][hold[0]]["decision"] == "pass" and s["instances"][hold[0]]["attempts"]["pool"].endswith("attempt_2")
    assert s["instances"][hold[1]]["decision"] == "pending"                                # refuted, no retry yet
    d = out / f"{hold[1]}__lp_bar" / "attempt_2"
    d.mkdir()
    (d / "result.json").write_text(json.dumps(_fake_result(fam, hold[1], "lp_bar", pp, f"sha-{hold[1]}", Fraction(990000),
                                                           None, 900.0, seal_sha)))
    bad = _fake_result(fam, hold[1], "mip", pp, f"sha-{hold[1]}", None, None, 900.0, seal_sha)
    bad["incumbents"] = ["bad"]                       # a malformed attempt never aborts the summary (round 3)
    d = out / f"{hold[1]}__mip" / "attempt_2"
    d.mkdir()
    (d / "result.json").write_text(json.dumps(bad))
    s = R.summarize("ufl", out, pp)
    assert [s["instances"][n]["decision"] for n in hold] == ["pass", "pass"] and s["tiers"]["N4000"]["tier"] == "retained"
    assert s["family_decision"] == "N4000"
    row = json.loads((out / f"{hold[0]}__lp_reference.json").read_text())
    row["interval"]["lo"] = "1000000"                                                         # edited after the seal
    (out / f"{hold[0]}__lp_reference.json").write_text(json.dumps(row))
    assert R.summarize("ufl", out, pp)["instances"][hold[0]]["decision"] == "pending"
    # a solver-reported bound above the verified incumbents refutes the claim: infrastructure, bound fields undefined
    r = json.loads((out / f"{s1[2]}__lagr" / "result.json").read_text())
    r["solver_bounds"]["final"] = 3e6
    (out / f"{s1[2]}__lagr" / "result.json").write_text(json.dumps(r))
    s = R.summarize("ufl", out, pp)
    m = s["instances"][s1[2]]["arms"]["lagr"]
    assert any("refuted" in x for x in m["infrastructure"]) and m["bound_target_T"] is None and m["t_bound"] is None
    # a too-wide reference interval keeps the instance pending (width recomputed from the endpoints, not read)
    panel["lp_star_frozen"]["intervals"][s1[1]]["lo"] = "990000"
    pp.write_text(json.dumps(panel))
    assert R.summarize("ufl", out, pp)["instances"][s1[1]]["acceptance"]["undefined"] == "reference_too_wide"


# ---------------------------------------------------------------- supervisor (solver-free arms)

def test_supervisor_kills_and_records_solver_free(tmp_path):
    path = _ufl(tmp_path, N=8)
    recs, run = R.supervise("ufl", str(path), "_test_sleeper", 4.0, tmp_path / "sl")
    assert run["killed_at_T"] and run["infrastructure"] == [] and run["wall_seconds"] < 30
    kinds = [r["kind"] for r in recs]
    assert kinds == ["incumbent", "certificate"] and (tmp_path / "sl" / "certs" / recs[1]["file"]).exists()
    recs, run = R.supervise("ufl", str(path), "_test_crash", 20.0, tmp_path / "cr")
    assert not run["killed_at_T"] and any("exited unexpectedly" in p for p in run["infrastructure"])
    with pytest.raises(FileExistsError):
        R.supervise("ufl", str(path), "_test_crash", 5.0, tmp_path / "cr")
    recs, run = R.supervise("ufl", str(path), "analytical", 30.0, tmp_path / "an")   # numpy only
    assert run["infrastructure"] == [] and [r["kind"] for r in recs] == ["certificate", "result"]
    inc = R.evaluate_incumbents(R.UFLFamily(), str(path), [{"kind": "incumbent", "t": 1.0, "payload": {"design": [0, 1]}},
                                                          {"kind": "incumbent", "t": 2.0, "payload": {"design": [1]}},
                                                          {"kind": "incumbent", "t": 9.0, "payload": {"design": [2]}}], 5.0)
    from src.bd.ufl.g_bound_xl import design_value, load_instance
    inst = load_instance(path)
    assert inc["late"] == 1 and inc["verified"] is True
    assert inc["best"] == min(design_value(inst, [0, 1]), design_value(inst, [1]))


def test_converter_rejects_malformed(tmp_path):
    import src.experiments.xl_certificate as xc
    path = _ufl(tmp_path, N=6)
    conv = R.UFLFamily().converter(str(path))
    with pytest.raises(xc.CertificateRejected):
        conv(b'{"format": "ufl-cert-c1", "singles": {"j": [0], "v": []}, "multi": "x"}')
    with pytest.raises(xc.CertificateRejected):
        conv(b"not json")
    for bad in ({"singles": {"j": [0], "v": []}, "multi": []},                 # truncation by zip (Codex round 1 #4)
                {"singles": {"j": [6], "v": [1.0]}, "multi": []},              # customer out of range
                {"singles": {"j": [True], "v": [1.0]}, "multi": []},           # bool is not an index
                {"singles": {"j": [], "v": []}, "multi": [{"ids": [1, 1], "v": [1.0, 2.0]}]},   # duplicate ids
                {"singles": {"j": [0], "v": [1e400]}, "multi": []}):           # non-finite (inf)
        with pytest.raises(xc.CertificateRejected):
            conv(json.dumps({"format": "ufl-cert-c1", "view": "ufl-v1", "version": 1, **bad}).encode())
    with pytest.raises(xc.CertificateRejected):
        conv(json.dumps({"format": "ufl-cert-c1", "view": "other", "singles": {"j": [], "v": []}}).encode())
    assert json.loads(conv(R.UFLFamily().empty_certificate()))["cuts"] == []
    sp = _scfl(tmp_path, S=3)
    sconv = R.SCFLFamily().converter(str(sp))
    good = {"format": "scfl-cert-c1", "view": "scfl-v1", "version": 1, "cuts": [{"terms": [[0, [1.0] * 4, [0.0] * 4]]}]}
    assert json.loads(sconv(json.dumps(good).encode()))["cuts"]
    for t in ([[0, [1.0] * 3, [0.0] * 4]], [[0, [1.0] * 4, [0.0]]], [[3, [1.0] * 4, [0.0] * 4]],
              [[0, [1.0] * 4, [0.0] * 4], [0, [1.0] * 4, [0.0] * 4]], []):
        with pytest.raises(xc.CertificateRejected):
            sconv(json.dumps(dict(good, cuts=[{"terms": t}])).encode())


def test_scfl_fractional_domain_invariant(tmp_path):
    from src.bd.scfl.a_data_xl import make_instance, read_cap
    ok = json.loads(_scfl(tmp_path, S=30).read_text())
    assert R.scfl_domain_problems(ok) == []
    wide = tmp_path / "capwide.txt"
    wide.write_text(CAP_WIDE)
    assert R.scfl_domain_problems(make_instance(read_cap(wide), 30, 3))


# ---------------------------------------------------------------- solver smoke tests (tiny; Phoenix or solve1.sh)

@solver
def test_smoke_lean_builders_match_plain(tmp_path):
    import gurobipy as gp
    from src.bd.scfl import g_bound_xl as sg
    from src.bd.scfl.a_data import load_instance as sload
    from src.bd.scfl.b_mip_xl import make_env
    from src.bd.ufl import g_bound_xl as g
    inst = g.load_instance(_ufl(tmp_path, N=12, c="c", seed=4))
    env = make_env(1)
    vals = []
    for b in (g.build_compact, g.build_compact_lean):
        md, info = b(inst, env, relax=True)
        md.optimize()
        vals.append((md.ObjVal, np.asarray(info["assign"].Pi)))
        md.dispose()
    assert vals[0][0] == pytest.approx(vals[1][0], rel=1e-9)
    mip = []
    for b in (g.build_compact, g.build_compact_lean):
        md, info = b(inst, env, relax=False)
        md.Params.MIPGap = 0
        md.optimize()
        mip.append(md.ObjVal)
        md.dispose()
    assert mip[0] == pytest.approx(mip[1], rel=1e-9)
    si = sload(_scfl(tmp_path, S=5))
    ev = []
    for b in (sg.build_ef_lp, sg.build_ef_lp_lean):
        md, info = b(si.I, si.J, si.fixed, si.capacity, si.cost, si.demand, si.prob, env)
        md.optimize()
        ev.append((md.ObjVal, sg.ef_duals(info, "Pi", si.S, si.I, si.J, si.prob)[0].sum()))
        md.dispose()
    assert ev[0][0] == pytest.approx(ev[1][0], rel=1e-9)
    env.dispose()
    del gp


@solver
@pytest.mark.parametrize("fam_name", ["ufl", "scfl"])
def test_smoke_pipeline_tiny(tmp_path, fam_name):
    """gen-free tiny pipeline: reference -> freeze-like panel -> a few arms through run_task -> summarize runs."""
    fam = R.family(fam_name)
    path = _ufl(tmp_path, N=10, c="c", seed=3) if fam_name == "ufl" else _scfl(tmp_path, S=8, seed=2)
    name = path.stem
    ref = R.reference(fam_name, str(path), 20.0, tmp_path / "res")
    iv = ref["interval"]
    assert iv["rigorous"] and iv["rel_width"] <= P.RULES["max_reference_rel_width"], ref
    panel = P.build_panel(fam_name)
    T = 6.0
    panel["T"] = T
    panel["tiers"] = {"tiny": {"headline": "final_target", "stage1": [name], "holdout": []}}
    panel["tier_order"] = ["tiny"]
    panel["references"] = {name: {"instance_sha256": R.sha256(path.read_bytes())}}
    panel["lp_star_frozen"] = {"intervals": {name: iv | {"instance_sha256": ref["instance_sha256"]}}}
    panel["code_sha256_at_preparation"] = P.code_hashes(R, fam_name)
    panel.pop("runtime")
    pp = tmp_path / "panel.json"
    pp.write_text(json.dumps(panel))
    for arm in ("eng", "eng_plain", "naive_lp", "empty", "analytical", "lp_dual"):
        r = R.run_task(fam_name, str(path), arm, T, tmp_path / "res", pp)
        assert r["run"]["infrastructure"] == [], (arm, r["run"])
        assert r["certificates"].get("canonical") in (True, None), (arm, r["certificates"])
        assert "unavailable" not in r["incumbents"], r["incumbents"]
        cps = r["certificates"]["checkpoints"]
        if arm in ("eng", "eng_plain", "naive_lp", "lp_dual"):   # plain/naive too: the domain invariant holds
            assert cps[-1]["cert_ok"] and Fraction(cps[-1]["lb"]) >= R.target(iv, 1e-5), (arm, cps)
        if arm == "empty":
            assert cps[-1]["cert_ok"] and Fraction(cps[-1]["lb"]) <= Fraction(iv["lo"])
    s = R.summarize(fam_name, tmp_path / "res", pp)
    assert s["instances"][name]["decision"] == "pending"          # the other arms were not run


def test_deadline_helper_kills_grandchildren(tmp_path):
    import time
    marker = tmp_path / "pid"

    def fn():
        pid = os.fork()
        if pid == 0:                      # grandchild: would outlive its parent without the group kill
            time.sleep(60)
            os._exit(0)
        marker.write_text(str(pid))
        time.sleep(60)

    t0 = time.monotonic()
    status, _ = R.run_with_deadline(fn, 2.0)
    assert status == "deadline" and time.monotonic() - t0 < 10
    gpid = int(marker.read_text())
    time.sleep(0.5)
    try:
        os.kill(gpid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    if alive:                             # a zombie reparented to init counts as dead; check its state
        import subprocess
        st = subprocess.run(["ps", "-o", "stat=", "-p", str(gpid)], capture_output=True, text=True).stdout.strip()
        alive = bool(st) and not st.startswith("Z")
    assert not alive
    assert R.run_with_deadline(lambda: {"x": 1}, 10.0) == ("done", {"x": 1})
    assert R.run_with_deadline(lambda: 1 / 0, 10.0)[0] == "error"
