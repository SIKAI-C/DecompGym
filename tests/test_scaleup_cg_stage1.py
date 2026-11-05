"""Protocol v4 CG Stage 1 runner, panel logic and decomposition-free controls (CLRP, CVRP, BPPC).

Tiny instances only (<= 15 customers / 20 items). The heuristic controls and the supervised test arms run for a few
seconds; the one Gurobi test (tiny CLRP compact MIP) runs only under DECOMPGYM_LOCAL_SOLVE=1 (scratchpad solve1.sh)
or on Phoenix (SLURM_JOB_ID)."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bd.clrp.f_sisr_xl import RoutingModel, RuinRecreate, insert_all  # noqa: E402

spec = importlib.util.spec_from_file_location("cg_stage1", ROOT / "scripts" / "phoenix" / "scaleup_cg_stage1.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

SOLVE_OK = os.environ.get("DECOMPGYM_LOCAL_SOLVE") == "1" or bool(os.environ.get("SLURM_JOB_ID"))


# ---------------------------------------------------------------- tiny instances

def tiny_cvrp(path: Path, n: int = 12, seed: int = 1) -> Path:
    rng = np.random.default_rng(seed)
    pts = rng.integers(0, 100, size=(n + 1, 2))
    dem = [0] + rng.integers(1, 10, size=n).tolist()
    lines = ["NAME : tiny", "TYPE : CVRP", f"DIMENSION : {n + 1}", "EDGE_WEIGHT_TYPE : EUC_2D", "CAPACITY : 20",
             "NODE_COORD_SECTION"] + [f"{i + 1} {x} {y}" for i, (x, y) in enumerate(pts.tolist())] + \
            ["DEMAND_SECTION"] + [f"{i + 1} {d}" for i, d in enumerate(dem)] + ["DEPOT_SECTION", "1", "-1", "EOF"]
    path.write_text("\n".join(lines) + "\n")
    return path


def tiny_clrp(path: Path, n: int = 10, m: int = 3, seed: int = 2) -> Path:
    rng = np.random.default_rng(seed)
    dep = rng.integers(0, 50, size=(m, 2)).ravel().tolist()
    cus = rng.integers(0, 50, size=(n, 2)).ravel().tolist()
    dem = rng.integers(5, 20, size=n).tolist()
    W = [int(sum(dem) * 0.6)] * m
    O = rng.integers(500, 2000, size=m).tolist()
    tok = [n, m] + dep + cus + [40] + W + dem + O + [100, 0]
    path.write_text(" ".join(str(t) for t in tok) + "\n")
    return path


def tiny_bppc(path: Path, n: int = 16, seed: int = 3) -> Path:
    rng = np.random.default_rng(seed)
    w = rng.integers(20, 60, size=n).tolist()
    lines = [f"{n} 100"]
    for i in range(n):
        conf = [j + 1 for j in range(i + 1, n) if rng.random() < 0.2]
        lines.append(" ".join(str(x) for x in [i + 1, w[i]] + conf))
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------- engine

def model_md(seed=0, n=12, m=2, W=None):
    rng = np.random.default_rng(seed)
    pts = rng.random((m + n, 2)) * 100
    C = np.ceil(np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1)))
    q = rng.integers(1, 6, size=n)
    return RoutingModel(C, q, 10, m, W=W, F=7.0, O=[50.0] * m)


def test_insert_all_matches_bruteforce_cheapest_position():
    model = model_md(W=[1e9, 1e9])
    m = model.m
    routes = [[0, [m + 0, m + 1, m + 2]], [1, [m + 3, m + 4]]]
    c = m + 5
    best = math.inf
    for r, (d, p) in enumerate(routes):
        if sum(model.q[v] for v in p) + model.q[c] > model.Q:
            continue
        for k in range(len(p) + 1):
            cand = [list(x) for x in routes]
            cand[r] = [d, p[:k] + [c] + p[k:]]
            best = min(best, model.cost(cand))
    for d in range(m):
        best = min(best, model.cost(routes + [[d, [c]]]))
    out = insert_all(model, [[d, list(p)] for d, p in routes], [c], [0, 1])
    assert out is not None and abs(model.cost(out) - best) < 1e-9


def test_insert_all_respects_depot_capacity_and_reports_failure():
    model = model_md(W=[3.0, 3.0])
    big = [v for v in range(model.m, model.m + model.n) if model.q[v] > 3]
    assert big
    assert insert_all(model, [], big[:1], [0, 1]) is None


def test_ruin_recreate_keeps_feasibility_and_never_worsens_best():
    model = model_md(seed=4, n=14, m=2, W=[40.0, 40.0])
    start = insert_all(model, [], list(range(model.m, model.m + model.n)), [0, 1])
    assert start is not None and model.feasible(start)
    seen = []
    rr = RuinRecreate(model, seed=0)
    best, cost = rr.run(start, deadline=1e18, max_iterations=300, on_best=lambda r, c: seen.append((r, c)))
    assert model.feasible(best) and abs(model.cost(best) - cost) < 1e-9 and cost <= model.cost(start) + 1e-9
    assert all(model.feasible(r) and abs(model.cost(r) - c) < 1e-9 for r, c in seen)
    assert [c for _, c in seen] == sorted([c for _, c in seen], reverse=True)


def test_ruin_recreate_restricted_depots():
    model = model_md(seed=5, n=10, m=3)
    start = insert_all(model, [], list(range(model.m, model.m + model.n)), [2])
    best, _ = RuinRecreate(model, seed=1).run(start, 1e18, allowed=[2], max_iterations=200)
    assert {d for d, _ in best} == {2}


# ---------------------------------------------------------------- family evaluation

def test_family_value_and_verifier_cvrp_bppc(tmp_path):
    f = R.Family("cvrp", str(tiny_cvrp(tmp_path / "t.vrp")))
    sol = [[i] for i in range(1, 13)]
    z = f.value(sol)
    assert isinstance(z, int) and f.verify(sol, "x")["ok"]
    assert f.value(sol[1:]) is None and f.value("junk") is None and f.value([[0]]) is None
    from src.experiments.xl_scoring import InvalidSolution
    with pytest.raises(InvalidSolution):
        f.objective(b'{"routes": []}')
    assert f.objective(json.dumps({"solution": sol}).encode()) == z
    b = R.Family("bppc", str(tiny_bppc(tmp_path / "t.txt")))
    bins = [[i] for i in range(1, 17)]
    assert b.value(bins) == 16 and b.verify(bins, "x")["ok"] and b.value([[1, 1]] + bins[1:]) is None


def test_family_value_clrp_schema(tmp_path):
    f = R.Family("clrp", str(tiny_clrp(tmp_path / "c.dat")))
    plan = [{"depot": 0, "customers": [i]} for i in range(10)]
    # depot 0 capacity is 60% of the demand: one depot cannot serve all
    assert f.value(plan) is None
    plan = [{"depot": i % 3, "customers": [i]} for i in range(10)]
    z = f.value(plan)
    assert isinstance(z, int) and f.verify(plan, "x")["ok"]
    assert f.value([{"depot": True, "customers": [0]}]) is None and f.value([{"depot": 0, "customers": []}]) is None


# ---------------------------------------------------------------- eps calibration, acceptance, tiers

def F(x):
    return Fraction(x)


def test_separating_eps():
    pairs = [(F("0"), F("0.02")), (F("0.005"), F("0.03")), (F("0.001"), F("0.015")), (F("0.02"), F("0.01"))]
    out = R.separating_eps(pairs, 3)
    eps = Fraction(out["eps_p_exact"])
    assert out["count"] == 3 and F("0.005") <= eps < F("0.015")
    assert R.separating_eps([(F("0.01"), F("0.005"))] * 4, 3)["eps_p"] is None
    big = R.separating_eps([(F(0), None)] * 4, 3)
    assert Fraction(big["eps_p_exact"]) == Fraction(1, 200)


def per(obj, near, t_near=None, refuted=False):
    return {"objective": obj, "success": {"near_best": near}, "time_to_target": {"near_best": t_near}, "t_near_best": t_near,
            "gap_best": 0.0, "matches_or_beats_MIP": None}


RULES = {"naive_quality_gap": 0.005, "naive_speed_ratio": 3.0, "saved_fraction": 0.25}


def test_accept_rules():
    base = {"eng": per(1000, True, 10.0), "naive": per(1005, False), "ls": per(1010, False), "mip": per(1100, False)}
    assert R.accept(base, RULES, 900.0, 1000)["pass"] is True
    a = dict(base, naive=per(1004, False))          # 4 < ceil(5): quality fails, speed: never reached -> censored 900
    assert R.accept(a, RULES, 900.0, 1000)["naive_speed"] is True
    a = dict(base, naive=per(1004, True, 100.0))    # reaches at 100 s: 10x but saved 90 < 225
    assert R.accept(a, RULES, 900.0, 1000)["pass"] is False
    a = dict(base, ls=per(1001, True, 5.0))
    r = R.accept(a, RULES, 900.0, 1000)
    assert r["pass"] is False and any("control" in x for x in r["reasons"])
    a = dict(base, naive=per(None, False))
    assert R.accept(a, RULES, 900.0, 1000)["naive_quality"] is True
    a = dict(base, ls=per(None, None))
    assert R.accept(a, RULES, 900.0, 1000)["pass"] is None


def test_tier_decision():
    assert R.tier_decision(["pass"] * 3 + ["fail"], ["pass", "pass"], 2)["tier"] == "retained"
    assert R.tier_decision(["pass", "fail", "fail", "pending"], None, 2)["tier"] == "rejected"
    assert R.tier_decision(["pass"] * 4, ["pass"], 2)["holdout"] == "pending"
    assert R.tier_decision(["pass"] * 4, [], 0)["holdout"] == "undeclared"
    assert R.tier_decision(["pass"] * 4, ["pass", "fail"], 2)["tier"] == "rejected"


# ---------------------------------------------------------------- supervised runs (test arms: no solver)

def panel_for(tmp_path, family, name, inst):
    from hashlib import sha256
    panel = {"family": family, "T": 4.0, "status": "TEST", "rules": RULES,
             "tiers": {"t": {"stage1": [name], "holdout": [], "eps_p": "0.01"}},
             "references": {name: {"instance_sha256": sha256(Path(inst).read_bytes()).hexdigest(), "z_pre": None}},
             "code_sha256_at_preparation": {f: sha256((ROOT / f).read_bytes()).hexdigest() for f in R.code_files(family)}}
    p = tmp_path / "panel.json"
    p.write_text(json.dumps(panel))
    return p


@pytest.mark.parametrize("mode,expect_infra,expect_best", [
    ("ok", False, True), ("sleep", False, True), ("crash", True, True), ("no_result", True, True),
    ("sigkill_self", True, True), ("invalid", False, True), ("bad_bound", False, True)])
def test_supervised_test_arms(tmp_path, mode, expect_infra, expect_best):
    inst = str(tiny_cvrp(tmp_path / "t.vrp"))
    pp = panel_for(tmp_path, "cvrp", "t", inst)
    row = R.run_task("cvrp", inst, f"_test_{mode}", 4.0, tmp_path / "out", pp)
    assert bool(row["run"]["infrastructure"]) is expect_infra, row["run"]
    inc = row["incumbents"]
    assert (inc["best"] is not None) is expect_best and inc["verified"] is True
    assert row["run"]["killed_at_T"] is (mode == "sleep")
    if mode == "invalid":
        assert inc["infeasible"] == 1
    if mode == "bad_bound":
        sc = R.score(row, inc["best"], Fraction(1, 100), 4.0, None, None, "t")
        assert sc["refuted_claim"] is True and sc["success"]["near_best"] is False
    if mode == "ok":   # the identity matches the panel; a retry takes a fresh attempt directory
        panel = json.loads(pp.read_text())
        import hashlib
        assert R.identity_problems(row, "t", "_test_ok", panel, hashlib.sha256(pp.read_bytes()).hexdigest()) == []
        row2 = R.run_task("cvrp", inst, "_test_ok", 4.0, tmp_path / "out", pp)
        assert row2["attempt"]["index"] == 2


def fake_row(name, arm, best, traj, panel, panel_sha, family="cvrp"):
    return {"family": family, "instance": name, "arm": arm, "T": panel["T"],
            "run": {"infrastructure": []},
            "incumbents": {"best": best, "verified": best is not None, "trajectory": traj, "claims": [], "infrastructure": []},
            "identity": {"task": {"family": family, "arm": arm, "T": panel["T"],
                                  "instance_sha256": panel["references"][name]["instance_sha256"]},
                         "panel": {"sha256": panel_sha}, "code_sha256": panel["code_sha256_at_preparation"]}}


def test_summarize_and_freeze_eps_from_synthetic_results(tmp_path):
    import hashlib
    names = ["a", "b", "c", "d", "h1", "h2"]
    panel = {"family": "cvrp", "T": 900.0, "status": "FROZEN test", "rules": RULES,
             "tiers": {"t": {"dev": ["a", "b", "c", "d"], "stage1": ["a", "b", "c", "d"], "holdout": ["h1", "h2"], "eps_p": "0.004"}},
             "references": {n: {"instance_sha256": n * 8, "z_pre": 1000} for n in names},
             "code_sha256_at_preparation": {"x.py": "0" * 64}}
    pp = tmp_path / "panel.json"
    pp.write_text(json.dumps(panel))
    sha = hashlib.sha256(pp.read_bytes()).hexdigest()
    out = tmp_path / "res"
    vals = {"eng": (1002, [[5.0, 1100], [50.0, 1002]]), "naive": (1030, [[1.0, 1030]]), "ls": (1010, [[2.0, 1010]]),
            "mip": (1050, [[300.0, 1050]])}
    for n in names:
        for a, (b, tr) in vals.items():
            d = out / f"{n}__{a}"
            d.mkdir(parents=True)
            row = fake_row(n, a, b, tr, panel, sha)
            if n == "d" and a == "ls":
                row = fake_row(n, a, 1001, [[2.0, 1001]], panel, sha)   # control reaches near_best on d
            (d / "result.json").write_text(json.dumps(row))
    s = R.summarize(out, pp)
    assert s["instances"]["a"]["decision"] == "pass" and s["instances"]["d"]["decision"] == "fail"
    assert s["tiers"]["t"]["tier"] == "retained"
    assert s["instances"]["a"]["diagnostics"]["mip_baseline"]["z_mip"] == 1050
    # an identity mismatch is pending, never scored
    bad = json.loads((out / "h1__eng" / "result.json").read_text())
    bad["identity"]["code_sha256"] = {"x.py": "1" * 64}
    (out / "h1__eng" / "result.json").write_text(json.dumps(bad))
    s = R.summarize(out, pp)
    assert s["instances"]["h1"]["decision"] == "pending" and s["tiers"]["t"]["tier"] == "pending"
    e = R.freeze_eps(out, pp)
    prop = e["tiers"]["t"]["proposal"]
    assert prop["count"] == 3 and Fraction("0.002") <= Fraction(prop["eps_p_exact"]) < Fraction("0.01")


# ---------------------------------------------------------------- controls (a few seconds each)

def test_cvrp_control_publishes_verified_improving_plans(tmp_path):
    from src.cg.cvrp.f_ils_xl import CVRPRuinRecreate
    inst = str(tiny_cvrp(tmp_path / "t.vrp", n=14))
    f = R.Family("cvrp", inst)
    got = []
    res = CVRPRuinRecreate(time_limit=3.0, on_incumbent=lambda r, stage=None: got.append(f.value(r))).run(inst)
    assert got and None not in got and got == sorted(got, reverse=True) and got[-1] == res["best_objective"]
    assert res["iterations"] > 50


def test_clrp_control_publishes_verified_improving_plans(tmp_path):
    from src.bd.clrp.g_lrp_ils_xl import CLRPLocationRoutingILS
    inst = str(tiny_clrp(tmp_path / "c.dat", n=12))
    f = R.Family("clrp", inst)
    got = []
    res = CLRPLocationRoutingILS(time_limit=4.0, eval_seconds=0.3,
                                 on_incumbent=lambda r, stage=None: got.append(f.value(r))).run(inst)
    assert got and None not in got and got == sorted(got, reverse=True) and got[-1] == res["best_objective"]
    assert res["screened"] > 0


@pytest.mark.skipif(not SOLVE_OK, reason="Gurobi only through solve1.sh or on Phoenix")
def test_clrp_compact_mip_tiny(tmp_path):
    from src.bd.clrp.b_mip_xl import CLRPCompactMIP
    inst = str(tiny_clrp(tmp_path / "c.dat", n=7))
    f = R.Family("clrp", inst)
    got = []
    res = CLRPCompactMIP(time_limit=20.0, on_incumbent=lambda r: got.append(f.value(r))).run(inst)
    assert got and None not in got and res["best_objective"] == got[-1]
    assert res["global_lb"] is not None and res["global_lb"] <= res["best_objective"]


def test_prelaunch_identity(tmp_path):
    inst = str(tiny_cvrp(tmp_path / "t.vrp"))
    pp = panel_for(tmp_path, "cvrp", "t", inst)
    assert R.prelaunch_problems("cvrp", inst, 4.0, pp) == []
    assert R.prelaunch_problems("cvrp", inst, 900.0, pp)
    other = str(tiny_cvrp(tmp_path / "t.vrp", seed=9))   # same name, different bytes
    assert any("instance sha256" in x for x in R.prelaunch_problems("cvrp", other, 4.0, pp))
    tiny_cvrp(tmp_path / "t.vrp")
    bad = json.loads(pp.read_text())
    bad["code_sha256_at_preparation"]["src/cg/cvrp/f_ils_xl.py"] = "0" * 64
    pp.write_text(json.dumps(bad))
    assert any("f_ils_xl" in x for x in R.prelaunch_problems("cvrp", inst, 4.0, pp))


def test_freeze_eps_refuted_and_missing_zbest(tmp_path):
    import hashlib
    panel = {"family": "cvrp", "T": 900.0, "status": "DRAFT", "rules": RULES,
             "tiers": {"t": {"dev": ["a", "b", "c", "d"], "stage1": [], "holdout": [], "eps_p": None}},
             "references": {n: {"instance_sha256": n * 8, "z_pre": (1000 if n != "d" else None)} for n in "abcd"},
             "code_sha256_at_preparation": {"x.py": "0" * 64}}
    pp = tmp_path / "panel.json"
    pp.write_text(json.dumps(panel))
    sha = hashlib.sha256(pp.read_bytes()).hexdigest()
    out = tmp_path / "dev"
    for n in "abcd":
        for a, b in (("eng", 1002), ("ls", 1010)):
            d = out / f"{n}__{a}"
            d.mkdir(parents=True)
            row = fake_row(n, a, None if n == "d" else b, [] if n == "d" else [[1.0, b]], panel, sha)
            if n == "c" and a == "eng":
                row["incumbents"]["claims"] = [{"lower_bound": 5000.0}]   # refuted: eng not separated on c
            (d / "result.json").write_text(json.dumps(row))
    e = R.freeze_eps(out, pp)["tiers"]["t"]
    assert e["dev"]["d"]["z_best"] is None and e["dev"]["c"]["gap_eng"] is None
    assert e["proposal"]["eps_p"] is None and e["proposal"]["count"] == 2 and e["proposal"]["required"] == 3


def test_split_problems():
    spec2 = importlib.util.spec_from_file_location("cg_panel", ROOT / "scripts" / "phoenix" / "scaleup_cg_panel.py")
    P = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(P)
    refs = {n: {"instance_sha256": n} for n in "abcdefgh"}
    ok = {"tiers": {"t": {"dev": ["g", "h"], "stage1": list("abcd"), "holdout": ["e", "f"]}}, "references": refs}
    assert P.split_problems(ok) == []
    dup = {"tiers": {"t": {"dev": ["a"], "stage1": list("abcd"), "holdout": ["e", "e"]}}, "references": refs}
    assert len(P.split_problems(dup)) >= 2
    refs2 = dict(refs, h={"instance_sha256": "a"})
    hashdup = {"tiers": {"t": {"dev": ["h"], "stage1": list("abcd"), "holdout": ["e", "f"]}}, "references": refs2}
    assert P.split_problems(hashdup)


def test_on_best_is_called_before_an_overrunning_polish():
    model = model_md(seed=6, n=12, m=2, W=[1e9, 1e9])
    start = [[0, [v]] for v in range(model.m, model.m + model.n)]   # one route per customer: easy to improve
    calls = []

    def slow_polish(routes, dl):
        return None   # (a polish that returns nothing, e.g. one that ran out of time)
    RuinRecreate(model, seed=0).run(start, 1e18, max_iterations=50, polish=slow_polish, polish_every=0.0,
                                    on_best=lambda r, c: calls.append(c))
    assert calls and calls[0] < model.cost(start)


def test_split_problems_dev_duplicates():
    spec2 = importlib.util.spec_from_file_location("cg_panel2", ROOT / "scripts" / "phoenix" / "scaleup_cg_panel.py")
    P = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(P)
    refs = {n: {"instance_sha256": n} for n in "abcdefgh"}
    bad = {"tiers": {"t": {"dev": ["g", "g", "h"], "stage1": list("abcd"), "holdout": ["e", "f"]}}, "references": refs}
    assert any("distinct" in x for x in P.split_problems(bad))
