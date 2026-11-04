"""v4 scorer (protocol v4 Sections 3.4 and 4; v3.1 Section 20.6 edge cases). Pure Python; run on Phoenix."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from fractions import Fraction

import pytest

from src.experiments.xl_scoring import (Attempt, InvalidSolution, Reference, aggregate_q, attempt_from_supervisor, q_at_k,
                                          score_attempt)


def ref(**kw):
    base = dict(T_f=900.0, z_best=Fraction(100), z_mip=Fraction(110), t_mip=600.0, integral=True, eps_p=Fraction("0.005"))
    base.update(kw)
    return Reference(**base)


def test_primal_metrics_basic():
    att = Attempt(incumbents=[(10.0, 130), (100.0, 108), (500.0, 101), (950.0, 99)], method_ok=True, termination="deadline")
    r = score_attempt(att, ref())
    assert r["feasible"] and r["objective"] == 101 and r["incumbents_eligible"] == 3
    assert r["gap_best"] == pytest.approx(0.01) and r["gap_best_exact"] == "1/100"
    assert r["matches_or_beats_MIP"] is True and r["t_matches_MIP"] == 100.0
    assert r["speedup"] == pytest.approx(6.0) and r["speedup_censored"] is False
    assert r["near_best"] is False and r["t_near_best"] is None
    # p = 1 on [0, 10), 0.3 on [10, 100), 0.08 on [100, 500), 0.01 on [500, 900)
    assert r["primal_integral"] == pytest.approx((10 + 0.3 * 90 + 0.08 * 400 + 0.01 * 400) / 900)
    assert r["success"] == {"matches_or_beats_MIP": True, "near_best": False, "bound_target": None}
    assert r["time_to_target"]["matches_or_beats_MIP"] == 100.0


def test_receipt_exactly_at_T_f_is_not_eligible():
    r = score_attempt(Attempt(incumbents=[(900.0, 100)], method_ok=True), ref())
    assert not r["feasible"] and r["incumbents_eligible"] == 0


def test_tie_rule_integral_and_continuous():
    assert score_attempt(Attempt(incumbents=[(1.0, 110)], method_ok=True), ref())["matches_or_beats_MIP"] is True
    assert score_attempt(Attempt(incumbents=[(1.0, 110.5)], method_ok=True), ref())["matches_or_beats_MIP"] is False
    cont = ref(integral=False, z_mip=Fraction(1000), z_best=Fraction(990))
    # tau = 1e-6 + 1e-9 * 1000 = 2e-6
    assert score_attempt(Attempt(incumbents=[(1.0, 1000.0000015)], method_ok=True), cont)["matches_or_beats_MIP"] is True
    assert score_attempt(Attempt(incumbents=[(1.0, 1000.000003)], method_ok=True), cont)["matches_or_beats_MIP"] is False


def test_no_mip_incumbent_makes_mip_target_undefined():
    r = score_attempt(Attempt(incumbents=[(5.0, 101)], method_ok=True), ref(z_mip=None, t_mip=None))
    assert r["matches_or_beats_MIP"] is None and r["speedup"] is None and r["speedup_censored"] is None
    assert r["success"]["matches_or_beats_MIP"] is None
    assert q_at_k([r, r], 2)["matches_or_beats_MIP"]["defined"] is False


def test_never_reaching_the_target_is_censored_with_the_bound():
    r = score_attempt(Attempt(incumbents=[(5.0, 150), (700.0, 120)], method_ok=True), ref())
    assert r["matches_or_beats_MIP"] is False and r["speedup"] is None
    assert r["speedup_censored"] is True and r["speedup_upper"] == pytest.approx(600 / 900)
    assert r["time_to_target"]["matches_or_beats_MIP"] is None


def test_no_incumbent():
    r = score_attempt(Attempt(incumbents=[], method_ok=True, termination="crash"), ref())
    assert not r["feasible"] and r["gap_best"] == math.inf and r["primal_integral"] == 1.0
    assert r["matches_or_beats_MIP"] is False and r["near_best"] is False
    assert r["success"] == {"matches_or_beats_MIP": False, "near_best": False, "bound_target": None}


def test_crash_carries_the_best_eligible_incumbent_to_T_f():
    r = score_attempt(Attempt(incumbents=[(50.0, 105)], method_ok=True, termination="crash"), ref())
    assert r["feasible"] and r["gap_best"] == pytest.approx(0.05) and r["matches_or_beats_MIP"] is True
    assert r["primal_integral"] == pytest.approx((50 * 1 + 850 * 0.05) / 900)


def test_maximization_uses_flipped_coordinates():
    mx = Reference(T_f=100.0, z_best=Fraction(200), sense=-1, z_mip=Fraction(195), t_mip=50.0, integral=True, eps_p=Fraction("0.06"))
    r = score_attempt(Attempt(incumbents=[(10.0, 150), (20.0, 190)], method_ok=True), mx)
    assert r["objective"] == 190 and r["gap_best"] == pytest.approx(0.05)
    assert r["matches_or_beats_MIP"] is False and r["near_best"] is True
    r2 = score_attempt(Attempt(incumbents=[(10.0, 196)], method_ok=True), mx)
    assert r2["matches_or_beats_MIP"] is True and r2["speedup"] == pytest.approx(5.0)
    # a claimed upper bound below a verified objective is refuted (maximization)
    r3 = score_attempt(Attempt(incumbents=[(10.0, 196)], method_ok=True, claims=[{"bound": 195}]), mx)
    assert r3["refuted_claim"]


def test_refuted_claims_fail_every_target():
    good = Attempt(incumbents=[(1.0, 100)], method_ok=True)
    assert score_attempt(good, ref())["success"]["near_best"] is True
    for claim in ({"lower_bound": 100.5}, {"status": "optimal", "objective": 101}, {"bound": 120, "t": 3.0}):
        att = Attempt(incumbents=[(1.0, 100)], method_ok=True, claims=[claim])
        r = score_attempt(att, ref())
        assert r["refuted_claim"], claim
        assert r["success"] == {"matches_or_beats_MIP": False, "near_best": False, "bound_target": None}
    ok = score_attempt(Attempt(incumbents=[(1.0, 100)], method_ok=True, claims=[{"lower_bound": 100}, {"lower_bound": 90}]), ref())
    assert not ok["refuted_claim"]
    # refuted by a verified value known elsewhere (z_best of the manifest), even without own incumbents
    r = score_attempt(Attempt(incumbents=[], method_ok=True, claims=[{"lower_bound": 100.5}]), ref())
    assert r["refuted_claim"]
    # an 'optimal' claim that differs from a proven optimum
    r = score_attempt(Attempt(incumbents=[(1.0, 100)], method_ok=True, claims=[{"status": "optimal", "objective": 99}]),
                      ref(z_opt=Fraction(100)))
    assert r["refuted_claim"]
    # a late verified incumbent also refutes
    r = score_attempt(Attempt(incumbents=[(1.0, 104)], late_incumbents=[(950.0, 101)], method_ok=True,
                              claims=[{"lower_bound": 102}]), ref(z_best=Fraction(103)))
    assert r["refuted_claim"]


def test_method_audit_gates_success():
    for mo in (False, None):
        r = score_attempt(Attempt(incumbents=[(1.0, 100)], method_ok=mo), ref())
        assert r["matches_or_beats_MIP"] is True and r["success"]["matches_or_beats_MIP"] is False


BD = dict(lp_lo=Fraction("999.9999"), lp_hi=Fraction(1000), eps_b=Fraction("0.0001"), lb_mip=Fraction("999.5"), z_best=Fraction(1010),
          z_mip=Fraction(1012), integral=False, eps_p=Fraction("0.001"))


def cps(*lbs, ok=None):
    ok = ok or [True] * len(lbs)
    return [{"checkpoint": 225.0 * (k + 1), "lb": str(lb), "cert_ok": o, "receipt_t": 100.0 * (k + 1)} for k, (lb, o) in enumerate(zip(lbs, ok))]


def test_bound_metrics_and_t_bound():
    r = score_attempt(Attempt(incumbents=[], method_ok=True, certificate=cps("900", "999.95", "999.95", "999.99")), ref(**BD))
    assert r["cert_ok"] and r["bound_target"] is True and r["success"]["bound_target"] is True
    assert r["bgap_LP"] == pytest.approx(1e-5) and r["bgap_LP_exact"] == "1/100000"
    assert r["t_bound"] == 450.0 and r["t_bound_receipt"] == 200.0
    assert r["beats_MIP_bound"] is True
    assert not r["feasible"]  # a verified incumbent is not required for the bound target


def test_bound_target_comparison_is_exact():
    # bgap = 0.1 / 1000 = 1e-4 exactly; in floats 0.1 / 1000 > 1e-4
    assert 0.1 / 1000 > 1e-4 or (1000 - 999.9) / 1000 > 1e-4
    r = score_attempt(Attempt(incumbents=[], method_ok=True, certificate=cps("999.9", "999.9", "999.9", "999.9")), ref(**BD))
    assert r["bound_target"] is True and r["t_bound"] == 225.0


def test_bound_target_failures():
    # final version broken: trivial bound and cert_ok False at T_f
    r = score_attempt(Attempt(incumbents=[], method_ok=True, certificate=cps("999.99", "999.99", "999.99", "0", ok=[True, True, True, False])),
                      ref(**BD))
    assert r["bound_target"] is False and r["cert_ok"] is False and r["t_bound_raw"] == 225.0
    # no checked certificate checkpoints on a BD-LP instance: the scoring input is incomplete (infrastructure)
    r = score_attempt(Attempt(incumbents=[(1.0, 1011)], method_ok=True), ref(**BD))
    assert r["bound_target"] is None and r["success"]["bound_target"] is None
    assert r["infrastructure_errors"] == ["no_checked_certificate_checkpoints"]
    # gates: method audit and refuted claims
    r = score_attempt(Attempt(incumbents=[], method_ok=False, certificate=cps("1000", "1000", "1000", "1000")), ref(**BD))
    assert r["bound_target"] is False and r["t_bound"] is None and r["t_bound_raw"] == 225.0
    r = score_attempt(Attempt(incumbents=[(1.0, 1011)], method_ok=True, claims=[{"lower_bound": 1100}],
                              certificate=cps("1000", "1000", "1000", "1000")), ref(**BD))
    assert r["refuted_claim"] and r["bound_target"] is False
    # never reaching the bound target: censored
    r = score_attempt(Attempt(incumbents=[], method_ok=True, certificate=cps("990", "995", "998", "999")), ref(**BD))
    assert r["bound_target"] is False and r["t_bound"] is None and r["bgap_LP"] == pytest.approx(1e-3)
    # a primal-only family: the bound side is undefined
    r = score_attempt(Attempt(incumbents=[(1.0, 101)], method_ok=True), ref())
    assert r["bound_target"] is None and r["cert_ok"] is None


def test_inconsistent_references_are_infrastructure_errors_not_successes():
    """Review 45 #6: a certified bound above LP_hi (or z_best) means the view or the reference is wrong."""
    for lbs, why in [(("999", "999", "999", "1001"), "lb_cert_above_lp_hi"), (("1001", "999", "999", "999"), "lb_cert_above_lp_hi"),
                     (("1020", "1020", "1020", "1020"), "lb_cert_above_z_best")]:
        r = score_attempt(Attempt(incumbents=[], method_ok=True, certificate=cps(*lbs)), ref(**BD))
        assert why in r["infrastructure_errors"]
        assert r["bound_target"] is None and r["success"]["bound_target"] is None and r["t_bound"] is None
        assert q_at_k([r], 1)["bound_target"]["q"] == [None]


def test_q_at_k_and_aggregation():
    recs = [{"success": {"matches_or_beats_MIP": False, "near_best": False, "bound_target": None}},
            {"success": {"matches_or_beats_MIP": True, "near_best": False, "bound_target": None}},
            {"success": {"matches_or_beats_MIP": False, "near_best": True, "bound_target": None}}]
    qk = q_at_k(recs, 3)
    assert qk["matches_or_beats_MIP"] == {"q": [0, 1, 1], "first": 2, "defined": True, "infrastructure": False}
    assert qk["near_best"] == {"q": [0, 0, 1], "first": 3, "defined": True, "infrastructure": False}
    assert qk["bound_target"]["defined"] is False
    other = q_at_k([{"success": {"matches_or_beats_MIP": True, "near_best": False, "bound_target": None}}] * 3, 3)
    agg = aggregate_q({"a": qk, "b": other}, 3)
    assert agg["matches_or_beats_MIP"] == {"pairs": 2, "q": [0.5, 1.0, 1.0], "unknown_pairs": [0, 0, 0]}
    assert agg["bound_target"] == {"pairs": 0, "q": [None, None, None], "unknown_pairs": [0, 0, 0]}


def test_headline_and_manifest_parsing():
    entry = {"instance": "gs250a-1", "T_f": 900, "sense": "min", "z_best": "257805", "z_mip": "257805", "t_mip": 212.0,
             "integral": True, "eps_p": "0.001", "eps_b": "0.0001", "lp_interval": {"lp_lo": "257000.5", "lp_hi": "257000.75"},
             "lb_mip": "250000", "headline": "time_to_near_best"}
    rf = Reference.from_manifest(entry)
    assert rf.lp_hi == Fraction(1028003, 4) and rf.eps_b == Fraction(1, 10000) and rf.sense == 1
    r = score_attempt(Attempt(incumbents=[(30.0, 257900), (60.0, 257805)], method_ok=True), rf)
    assert r["headline"] == {"target": "time_to_near_best", "success": True, "time": 30.0}


def test_attempt_from_supervisor_record(tmp_path):
    def put(name, obj):
        data = json.dumps(obj).encode()
        (tmp_path / name).write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    record = {"store_dir": str(tmp_path), "counts": {"late": 2, "torn": 1},
              "incumbents": [{"seq": 1, "t": 1.5, "file": "I1.json", "sha256": put("I1.json", {"open": [0, 1]})},
                             {"seq": 2, "t": 2.5, "file": "I2.json", "sha256": put("I2.json", {"open": []})},
                             {"seq": 3, "t": 3.5, "file": "I3.json", "sha256": put("I3.json", {"open": [1]})},
                             {"seq": 4, "t": 4.5, "file": "I4.json", "sha256": put("I4.json", "not a solution")}],
              "claims": [{"seq": 5, "t": 6.0, "payload": {"lower_bound": "1.0"}, "error": None}], "errors": []}
    costs = {(0, 1): 120.0, (1,): 104.0}

    def objective(payload: bytes):
        obj = json.loads(payload)
        if not isinstance(obj, dict) or "open" not in obj:
            raise InvalidSolution("not a solution")       # explicit rejection: infeasible publication
        return costs.get(tuple(obj["open"]))              # None: invalid solution

    att = attempt_from_supervisor(record, objective, method_ok=True, attempt=1)
    assert att.incumbents == [(1.5, 120.0), (3.5, 104.0)]
    assert att.extra["infeasible_incumbents"] == 2 and att.extra["late"] == 2  # infeasible payload + verifier exception
    assert att.claims == [{"lower_bound": "1.0", "t": 6.0}] and att.infrastructure == []
    r = score_attempt(att, ref())
    assert r["objective"] == 104 and r["matches_or_beats_MIP"] is True and not r["refuted_claim"]
    assert r["infrastructure_errors"] == []
    # a stored payload that fails its receipt hash, or a supervisor error, is an infrastructure error: no score
    tampered = dict(record, incumbents=record["incumbents"] + [{"seq": 9, "t": 5.0, "file": "I3.json", "sha256": "0" * 64}])
    for rec2 in (tampered, dict(record, errors=["stream 3: RuntimeError: boom"])):
        att2 = attempt_from_supervisor(rec2, objective, method_ok=True)
        assert att2.infrastructure
        r2 = score_attempt(att2, ref())
        assert r2["success"] == {"matches_or_beats_MIP": None, "near_best": None, "bound_target": None}
        assert q_at_k([r2], 1)["matches_or_beats_MIP"]["q"] == [None]


# ---------------------------------------------------------------- Codex review 45 regressions

@pytest.mark.parametrize("claim", [
    {"_error": "duplicate_key"}, {"lower_bound": "abc"}, {"lower_bound": float("nan")}, {"lower_bound": True},
    {"lower_bound": "1e999999999"}, {"bound": [1]}, {"status": "optimal"}, {"status": 3}, {"objective": "x"}, "not a dict",
])
def test_review45_2_malformed_claims_fail_the_attempt_without_crashing(claim):
    r = score_attempt(Attempt(incumbents=[(1.0, 100)], method_ok=True, claims=[claim]), ref())
    assert r["refuted_claim"] and r["refutations"][0]["why"].startswith("malformed_claim:")
    assert r["success"] == {"matches_or_beats_MIP": False, "near_best": False, "bound_target": None}


def test_review45_2_claims_are_compared_exactly_from_token_text():
    # 100.0000000000000000001 > 100 only in exact arithmetic (as a float it is 100.0)
    r = score_attempt(Attempt(incumbents=[(1.0, 100)], method_ok=True, claims=[{"lower_bound": "100.0000000000000000001"}]), ref())
    assert r["refuted_claim"]
    r = score_attempt(Attempt(incumbents=[(1.0, 100)], method_ok=True, claims=[{"lower_bound": "100", "status": "feasible"}]), ref())
    assert not r["refuted_claim"]


def test_review45_6_incomplete_checkpoint_records_are_infrastructure():
    full = cps("999.99", "999.99", "999.99", "999.99")
    for record, why in [(full[:3], "incomplete_checkpoints"), (full[1:], "incomplete_checkpoints"),
                        (full[:3] + [dict(full[3], checkpoint=899.0)], "incomplete_checkpoints"),
                        (full[:3] + [dict(full[3], infrastructure="hash_mismatch")], "checkpoint_900.0:hash_mismatch"),
                        (full[:3] + [dict(full[3], cert_ok="yes")], "malformed_checkpoint_record"),
                        (full[:3] + [dict(full[3], lb="abc")], "malformed_checkpoint_record")]:
        r = score_attempt(Attempt(incumbents=[], method_ok=True, certificate=record), ref(**BD))
        assert why in r["infrastructure_errors"], (why, r["infrastructure_errors"])
        assert r["bound_target"] is None and r["success"]["bound_target"] is None
    ok = score_attempt(Attempt(incumbents=[], method_ok=True, certificate=full), ref(**BD))
    assert ok["bound_target"] is True and ok["infrastructure_errors"] == []
    two = score_attempt(Attempt(incumbents=[], method_ok=True, certificate=[dict(full[1], checkpoint=450.0), dict(full[3])]),
                        ref(**BD, checkpoints=2))
    assert two["bound_target"] is True  # a frozen 2-checkpoint schedule


def test_review45_7_speed_headline_uses_the_gated_attainment_time():
    """A qualifying checkpoint at 225 s with an invalid final version: the speed headline succeeds at 225 s, while
    the bound target (final version, bound Q@k) fails."""
    certificate = cps("999.95", "999.95", "999.95", "0", ok=[True, True, True, False])
    speed = score_attempt(Attempt(incumbents=[], method_ok=True, certificate=certificate), ref(**BD, headline="time_to_bound_target"))
    assert speed["headline"] == {"target": "time_to_bound_target", "success": True, "time": 225.0}
    assert speed["bound_target"] is False and q_at_k([speed], 1)["bound_target"]["q"] == [0]
    final = score_attempt(Attempt(incumbents=[], method_ok=True, certificate=certificate), ref(**BD, headline="bound_target"))
    assert final["headline"] == {"target": "bound_target", "success": False, "time": None}
    gated = score_attempt(Attempt(incumbents=[], method_ok=False, certificate=certificate), ref(**BD, headline="time_to_bound_target"))
    assert gated["headline"] == {"target": "time_to_bound_target", "success": False, "time": None}
    never = score_attempt(Attempt(incumbents=[(10.0, 150)], method_ok=True), ref(headline="time_to_near_best"))
    assert never["headline"] == {"target": "time_to_near_best", "success": False, "time": None}


def test_review45_8_censored_speedup_uses_the_same_floor():
    r = score_attempt(Attempt(incumbents=[], method_ok=True), ref(t_mip=0.0001))
    assert r["speedup_censored"] is True and r["speedup_upper"] == pytest.approx(1e-3 / 900)
    r = score_attempt(Attempt(incumbents=[(0.00001, 100)], method_ok=True), ref(t_mip=0.0001))
    assert r["speedup"] == pytest.approx(1.0)


def test_review45_missing_verified_primal_leaves_the_bound_target_undefined():
    no_hi = dict(BD, lp_hi=None)
    r = score_attempt(Attempt(incumbents=[], method_ok=True, certificate=cps("999.99", "999.99", "999.99", "999.99")), ref(**no_hi))
    assert r["bound_target"] is None and r["bgap_LP"] is None and r["infrastructure_errors"] == []
    assert q_at_k([r], 1)["bound_target"]["defined"] is False
    rf = Reference.from_manifest({"T_f": 900, "z_best": "1010", "eps_b": "0.0001", "lp_interval": {"lp_lo": "999", "lp_hi": None}})
    assert rf.lp_hi is None


def test_review45_q_at_k_with_infrastructure_attempts():
    ok = {"success": {"matches_or_beats_MIP": True, "near_best": True, "bound_target": None}, "undefined_by_infrastructure": []}
    no = {"success": {"matches_or_beats_MIP": False, "near_best": False, "bound_target": None}, "undefined_by_infrastructure": []}
    infra = {"success": {"matches_or_beats_MIP": None, "near_best": None, "bound_target": None},
             "undefined_by_infrastructure": ["matches_or_beats_MIP", "near_best", "bound_target"]}
    assert q_at_k([no, infra, ok], 3)["near_best"] == {"q": [0, None, 1], "first": None, "defined": True, "infrastructure": True}
    assert q_at_k([ok, infra], 2)["near_best"] == {"q": [1, 1], "first": 1, "defined": True, "infrastructure": True}
    assert q_at_k([no, no], 3)["near_best"]["q"] == [0, 0, None]   # fewer attempts than k: unknown, not a failure
    agg = aggregate_q({"a": q_at_k([no, infra, ok], 3), "b": q_at_k([no, no, no], 3)}, 3)
    assert agg["near_best"] == {"pairs": 2, "q": [0.0, 0.0, 0.5], "unknown_pairs": [0, 1, 0]}


# ---------------------------------------------------------------- Codex review 47 regressions

def test_review47_4_malformed_checkpoint_fields_are_infrastructure():
    full = cps("999.99", "999.99", "999.99", "999.99")
    nan_times = [dict(c, checkpoint=float("nan")) for c in full]
    for record in (nan_times, full[:3] + [dict(full[3], lb=None)], full[:3] + [dict(full[3], lb=float("nan"))],
                   full[:3] + [dict(full[3], checkpoint=float("inf"))], full[:3] + [dict(full[3], checkpoint=True)], full[:3] + ["x"]):
        r = score_attempt(Attempt(incumbents=[], method_ok=True, certificate=record), ref(**BD, headline="time_to_bound_target"))
        assert r["infrastructure_errors"] == ["malformed_checkpoint_record"]
        assert r["bound_target"] is None and r["headline"] == {"target": "time_to_bound_target", "success": None, "time": None}


@pytest.mark.parametrize("kw,why", [
    (dict(lp_lo=Fraction(101), lp_hi=Fraction(100)), "lp_lo_above_lp_hi"),
    (dict(z_best=Fraction(120)), "z_best_worse_than_z_mip"),
    (dict(z_opt=Fraction(105)), "z_opt_worse_than_z_best"),
    (dict(lp_lo=Fraction(101), lp_hi=Fraction(102)), "lp_lo_above_z_best"),
    (dict(lb_mip=Fraction(101)), "lb_mip_above_z_best"),
    (dict(eps_b=Fraction(-1)), "eps_b_negative"),
    (dict(T_f=float("nan")), "T_f_not_finite_positive"),
    (dict(t_mip=float("nan")), "t_mip_not_finite_nonnegative"),
    (dict(z_mip=110.5), "z_mip_not_exact"),
])
def test_review47_4_inconsistent_references_make_every_target_undefined(kw, why):
    base = dict(BD, lp_lo=Fraction(99), lp_hi=Fraction(100), z_best=Fraction(100), z_mip=Fraction(110), lb_mip=Fraction(99))
    base.update(kw)
    r = score_attempt(Attempt(incumbents=[(1.0, 100)], method_ok=True, certificate=cps("100", "100", "100", "100")),
                      ref(**base, headline="bound_target"))
    assert f"reference: {why}" in r["infrastructure_errors"]
    assert r["success"] == {"matches_or_beats_MIP": None, "near_best": None, "bound_target": None}
    assert r["headline"]["success"] is None


def test_review47_4_malformed_incumbent_records_are_infrastructure():
    for incs in ([(float("nan"), 100)], [(1.0, float("nan"))], [(1.0, None)], [(-1.0, 100)], [(1.0,)]):
        r = score_attempt(Attempt(incumbents=incs, method_ok=True), ref())
        assert r["infrastructure_errors"] == ["malformed_incumbent_record"] and r["success"]["near_best"] is None


# ---------------------------------------------------------------- Codex review 49 regressions

def test_review49_4_each_incumbent_file_is_verified_before_cache_reuse(tmp_path):
    good = b'{"z":100}'
    (tmp_path / "I1.json").write_bytes(good)
    (tmp_path / "I2.json").write_bytes(b'{"z":999}')
    sha = hashlib.sha256(good).hexdigest()
    record = {"store_dir": str(tmp_path), "incumbents": [
        {"seq": 1, "t": 1.0, "file": "I1.json", "bytes": len(good), "sha256": sha},
        {"seq": 2, "t": 20.0, "file": "I2.json", "bytes": len(good), "sha256": sha}]}
    att = attempt_from_supervisor(record, lambda b: json.loads(b)["z"], method_ok=True)
    assert att.incumbents == [(1.0, 100)] and len(att.infrastructure) == 1 and "I2.json" in att.infrastructure[0]
    (tmp_path / "I2.json").unlink()
    att = attempt_from_supervisor(record, lambda b: json.loads(b)["z"], method_ok=True)
    assert att.incumbents == [(1.0, 100)] and len(att.infrastructure) == 1
    assert score_attempt(att, ref())["success"]["near_best"] is None
    # the same file listed twice is read once and reused
    record["incumbents"][1]["file"] = "I1.json"
    att = attempt_from_supervisor(record, lambda b: json.loads(b)["z"], method_ok=True)
    assert att.incumbents == [(1.0, 100), (20.0, 100)] and att.infrastructure == []


def test_review49_5_known_feasible_generators_are_materialized_once():
    for known in ([90], iter([90]), (v for v in [90])):
        r = score_attempt(Attempt(incumbents=[(1.0, 100)], method_ok=True, claims=[{"lower_bound": 95}]),
                          ref(), known_feasible=known)   # z_best = 100: only the known value 90 refutes 95
        assert r["refuted_claim"] and r["success"]["near_best"] is False


# ---------------------------------------------------------------- Codex review 50: k-median Stage 1 runner contract

def _km_runner():
    import importlib.util
    from pathlib import Path as _P
    spec = importlib.util.spec_from_file_location("km_stage1", _P(__file__).resolve().parents[1] / "scripts" / "phoenix" / "scaleup_km_stage1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _km_arm_result(lbs, ok=True, best=100.0, infra=None, canonical=True):
    cps = [{"checkpoint": c, "seq": i + 1, "cert_ok": ok, "lb": lb, "infrastructure": infra if c == 900.0 else None}
           for i, (c, lb) in enumerate(zip((60.0, 300.0, 600.0, 900.0), lbs))]
    return {"certificates": {"checkpoints": cps, "canonical": canonical, "family_agrees": True},
            "incumbents": {"best": best, "verified": True, "t_best": 1.0}, "run": {"killed_at_T": True, "child_peak_rss_mb": 1.0}}


def test_review50_3_stage1_summary_treats_infrastructure_as_undefined(tmp_path):
    pytest.importorskip("gurobipy")
    mod = _km_runner()
    rules = {"eps_b": 1e-5, "eps_p": 1e-3, "naive_quality_gap": 0.005, "naive_speed_ratio": 3.0, "saved_fraction": 0.25,
             "control_speed_ratio": 2.0, "control_saved_seconds": 225.0}
    panel = {"T": 900.0, "rules": rules, "tiers": {"C": {"headline": "final_target", "stage1": ["inst"]}},
             "references": {"inst": {"z_pre": 100.0}},
             "lp_star_frozen": {"intervals": {"inst": {"rigorous": True, "lo": "999.99", "hi": "1000"}}},
             "checker": {"budget_seconds_per_attempt": 600, "max_bytes": 268435456, "memory_bytes": 12 << 30}}
    (tmp_path / "panel.json").write_text(json.dumps(panel))
    arms = {"eng": _km_arm_result(["1000"] * 4), "naive_lp": _km_arm_result(["900"] * 4), "naive_int": _km_arm_result(["900"] * 4, best=150.0),
            "ils": _km_arm_result(["0"] * 4, best=200.0)}
    for c in mod.BOUND_CONTROLS:
        arms.setdefault(c, _km_arm_result(["0"] * 4, best=300.0))

    def write(out, overrides):
        for arm, r in (arms | overrides).items():
            d = out / f"inst__{arm}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "result.json").write_text(json.dumps(r))

    write(tmp_path / "ok", {})
    s = mod.summarize(tmp_path / "ok", tmp_path / "panel.json")
    assert s["instances"]["inst"]["acceptance"]["pass"] is True
    # a control whose checker failed (infrastructure) is undefined, never a beaten control
    write(tmp_path / "infra", {"pool": _km_arm_result(["0"] * 4, ok=False, infra="checker_error")})
    s = mod.summarize(tmp_path / "infra", tmp_path / "panel.json")
    acc = s["instances"]["inst"]["acceptance"]
    assert acc["pass"] is None and acc["infrastructure"] == {"pool": ["checker_error"]}
    assert s["instances"]["inst"]["arms"]["pool"]["bound_target_T"] is None
    # no canonical result (family backend only): undefined as well; the family result is never scored
    write(tmp_path / "fam", {"eng": _km_arm_result(["1000"] * 4, canonical=False)})
    acc = mod.summarize(tmp_path / "fam", tmp_path / "panel.json")["instances"]["inst"]["acceptance"]
    assert acc["pass"] is None and "canonical_unavailable" in acc["infrastructure"]["eng"]
    fz = mod.frozen_checker(tmp_path / "panel.json")
    assert fz["memory_bytes"] == 12 << 30 and fz["budget"] == 600.0 and fz["limits"].max_bytes == 268435456


# ---------------------------------------------------------------- Codex review 52: Stage 1 failure injection

def _km_tiny(tmp_path):
    np = pytest.importorskip("numpy")
    from src.bd.km.a_data_xl import random_instance, write_instance
    inst = random_instance(np.random.default_rng(1), 7, 9, 3, integer=True, same_sites=False, span=8)
    path = tmp_path / "km.json"
    write_instance(inst, path)
    return str(path)


def _clean_arms(mod):
    arms = {"eng": _km_arm_result(["1000"] * 4), "naive_lp": _km_arm_result(["900"] * 4),
            "naive_int": _km_arm_result(["900"] * 4, best=150.0), "ils": _km_arm_result(["0"] * 4, best=200.0)}
    for c in mod.BOUND_CONTROLS:
        arms.setdefault(c, _km_arm_result(["0"] * 4, best=300.0))
    return arms


def _decide(mod, arms, z_best=100.0):
    rules = {"eps_b": 1e-5, "eps_p": 1e-3, "naive_quality_gap": 0.005, "naive_speed_ratio": 3.0, "saved_fraction": 0.25,
             "control_speed_ratio": 2.0, "control_saved_seconds": 225.0}
    iv = {"rigorous": True, "lo": "999.99", "hi": "1000"}
    from src.bd.km.g_certificate_xl import target
    per = {a: mod.arm_metrics(r, iv, target(iv, rules["eps_b"]), z_best, rules["eps_p"]) for a, r in arms.items()}
    return mod.accept(per, "final_target", rules, 900.0)


@pytest.mark.skipif(sys.platform != "linux", reason="the supervised checker forks (Linux)")
def test_review52_1_stage1_conversion_runs_supervised_and_failures_are_not_scores(tmp_path, monkeypatch):
    pytest.importorskip("gurobipy")
    mod = _km_runner()
    import src.bd.km.g_certificate_xl as gcx
    import src.bd.km.h_view_xl as hvx

    path = _km_tiny(tmp_path)
    out = tmp_path / "sleeper"
    records, run = mod.supervise(path, "_test_sleeper", 4.0, out)
    assert run["reader_errors"] == [] and any(r["kind"] == "certificate" for r in records)
    good = mod.certify(path, records, out, 4.0)
    final = good["checkpoints"][-1]
    assert final["cert_ok"] and Fraction(final["lb"]) > 0 and good["family_agrees"]

    def oom(*a, **kw):
        raise MemoryError

    monkeypatch.setattr(hvx, "to_generic", oom)            # Codex: MemoryError in the conversion
    bad = mod.certify(path, records, out, 4.0)
    assert bad["checkpoints"][-1]["infrastructure"] == "memory_error_unattributed" and not bad["checkpoints"][-1]["cert_ok"]
    arm = {"certificates": bad, "incumbents": {"best": 300.0, "verified": True, "t_best": 1.0},
           "run": {"killed_at_T": True, "child_peak_rss_mb": 1.0}}
    assert _decide(mod, _clean_arms(mod))["pass"] is True
    acc = _decide(mod, _clean_arms(mod) | {"pool": arm})
    assert acc["pass"] is None and acc["infrastructure"]["pool"] == ["memory_error_unattributed"]

    def bug(*a, **kw):
        raise RuntimeError("converter bug")

    monkeypatch.setattr(hvx, "to_generic", bug)
    assert mod.certify(path, records, out, 4.0)["checkpoints"][-1]["infrastructure"] == "checker_error"
    monkeypatch.undo()

    def malformed(*a, **kw):
        raise gcx.CertFailure("parse_error")

    monkeypatch.setattr(gcx, "parse_certificate_bytes", malformed)   # malformed input: the agent's own failure
    rej = mod.certify(path, records, out, 4.0)["checkpoints"][-1]
    assert rej["reason"] == "rejected:parse_error" and rej["infrastructure"] is None and not rej["cert_ok"]


def test_review52_2_stage1_publication_reader_failures_make_the_arm_undefined(tmp_path, monkeypatch):
    pytest.importorskip("gurobipy")
    mod = _km_runner()
    path = _km_tiny(tmp_path)
    real_fsync = os.fsync

    def failing_fsync(fd):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    records, run = mod.supervise(path, "_test_sleeper", 3.0, tmp_path / "sleeper")
    monkeypatch.setattr(os, "fsync", real_fsync)
    assert run["reader_errors"] and "injected fsync failure" in run["reader_errors"][0]
    arm = {"certificates": {"checkpoints": [{"checkpoint": 3.0, "seq": None, "cert_ok": False, "lb": "0", "reason": "absent"}],
                            "canonical": True}, "incumbents": {"best": None, "verified": None, "t_best": None}, "run": run}
    m = mod.arm_metrics(arm, {"rigorous": True, "hi": "1000"}, Fraction(999), 100.0, 1e-3)
    assert m["bound_target_T"] is None and m["near_best"] is None and m["infrastructure"]
    acc = _decide(mod, _clean_arms(mod) | {"analytical": arm})
    assert acc["pass"] is None and "analytical" in acc["infrastructure"]
    # a missing arm result is a rerun as well, never a failure or a beaten control
    arms = _clean_arms(mod)
    del arms["pool"]
    assert _decide(mod, arms)["pass"] is None


# ---------------------------------------------------------------- Codex review 53: evaluator failures are never scores

@pytest.mark.parametrize("exc", [MemoryError(), OSError(5, "EIO"), RuntimeError("verifier bug"), KeyError("x")])
def test_review53_1_verifier_exceptions_are_infrastructure(tmp_path, exc):
    good = b'{"z":100}'
    (tmp_path / "I1.json").write_bytes(good)
    record = {"store_dir": str(tmp_path), "incumbents": [{"seq": 1, "t": 1.0, "file": "I1.json", "bytes": len(good),
                                                          "sha256": hashlib.sha256(good).hexdigest()}]}

    def objective(payload):
        raise exc

    att = attempt_from_supervisor(record, objective, method_ok=True)
    assert att.incumbents == [] and len(att.infrastructure) == 1 and "verifier" in att.infrastructure[0]
    r = score_attempt(att, ref())
    assert r["success"]["near_best"] is None and q_at_k([r], 1)["near_best"]["q"] == [None]


def test_review53_1_only_explicit_rejections_are_infeasible(tmp_path):
    good = b'{"z":100}'
    (tmp_path / "I1.json").write_bytes(good)
    record = {"store_dir": str(tmp_path), "incumbents": [{"seq": 1, "t": 1.0, "file": "I1.json", "bytes": len(good),
                                                          "sha256": hashlib.sha256(good).hexdigest()}]}
    for objective in (lambda b: None, lambda b: (_ for _ in ()).throw(InvalidSolution("infeasible"))):
        att = attempt_from_supervisor(record, objective, method_ok=True)
        assert att.incumbents == [] and att.infrastructure == [] and att.extra["infeasible_incumbents"] == 1
        assert score_attempt(att, ref())["success"]["near_best"] is False


def _km_instance_and_design(tmp_path):
    path = _km_tiny(tmp_path)
    from src.bd.km.a_data_xl import load_instance
    from src.bd.km.c_naive_bd_xl import greedy_design
    inst = load_instance(path)
    y = greedy_design(inst.distances(), inst.k)
    return path, inst, [int(i) for i in y.nonzero()[0]]


def test_review53_2_stage1_requires_independent_verification(tmp_path, monkeypatch):
    pytest.importorskip("gurobipy")
    mod = _km_runner()
    import src.bd.km.e_verifier_xl as ver
    path, inst, design = _km_instance_and_design(tmp_path)
    records = [{"kind": "incumbent", "t": 1.0, "payload": {"design": design, "stage": "test"}}]
    ok = mod.evaluate_incumbents(path, records, 10.0, "eng")
    assert ok["verified"] is True and ok["verification"]["ok"] is True

    def unreadable(*a, **kw):
        raise OSError("injected output read error")

    monkeypatch.setattr(ver, "verify_solution", unreadable)
    broken = mod.evaluate_incumbents(path, records, 10.0, "eng")
    assert broken["verified"] is None and "injected output read error" in broken["verification"]["error"]
    monkeypatch.setattr(ver, "verify_solution", lambda *a, **kw: {"ok": False, "issues": ["disagrees"], "objective_value": 1.0})
    disagree = mod.evaluate_incumbents(path, records, 10.0, "eng")
    assert disagree["verified"] is False and disagree["verification"]["issues"] == ["disagrees"]
    for inc in (broken, disagree):
        arm = _km_arm_result(["0"] * 4, best=inc["best"]) | {"incumbents": inc}
        m = mod.arm_metrics(arm, {"rigorous": True, "hi": "1000"}, Fraction(999), inc["best"], 1e-3)
        assert m["near_best"] is None and any("incumbent_not_verified" in x for x in m["infrastructure"])
        assert _decide(mod, _clean_arms(mod) | {"ils": arm})["pass"] is None
    # an invalid design, a malformed message and a content-rejected certificate of an evaluator arm: infrastructure
    bad = mod.evaluate_incumbents(path, [{"kind": "incumbent", "t": 1.0, "payload": {"design": design[:-1]}}], 10.0, "eng")
    assert bad["invalid"] == 1
    arms = {"invalid": _km_arm_result(["0"] * 4) | {"incumbents": bad},
            "malformed": _km_arm_result(["0"] * 4) | {"run": {"killed_at_T": True, "child_peak_rss_mb": 1.0, "malformed_messages": 1}},
            "rejected": _km_arm_result(["0"] * 4)}
    arms["rejected"]["certificates"]["checkpoints"][-1].update(cert_ok=False, reason="rejected:parse_error",
                                                               failure={"stage": "convert", "checkpoint": 3, "category": "certificate"})
    for name, arm in arms.items():
        assert _decide(mod, _clean_arms(mod) | {"pool": arm})["pass"] is None, name
