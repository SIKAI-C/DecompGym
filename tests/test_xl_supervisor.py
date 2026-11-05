"""Supervisor publication channel (protocol v4 Sections 2 and 3.2): framing, hashing, acknowledgement, eligibility
frozen at T_f, torn/corrupt/late/oversize messages, checkpoint versions, and the path publication -> verification ->
certificate check -> scoring. Standard library plus the repository; run on Phoenix."""
from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import subprocess
import sys
import textwrap
import time
from fractions import Fraction
from pathlib import Path

import pytest

from src.experiments import xl_supervisor as xs
from src.experiments.xl_supervisor import ENV, HEADER, MAGIC, PublishLimits, Supervisor, publish, read_payload

REPO = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def channel(monkeypatch):
    """Point this process's client at a supervisor (reset the cached connection between tests)."""
    def use(sup: Supervisor) -> None:
        monkeypatch.setenv(ENV, sup.env()[ENV])
        xs._client.update(sock=None, pid=None, addr=None)
    yield use
    xs._client.update(sock=None, pid=None, addr=None)


def wait_for(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def raw_conn(sup: Supervisor) -> socket.socket:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sup.socket_path)
    return s


SOLVER = textwrap.dedent("""
    import os, sys
    from decompgym_publish import publish_incumbent, publish_certificate, publish_claim
    out = []
    out.append(publish_incumbent({"open": [0, 1]}))
    out.append(publish_certificate({"view": "toy", "version": 1, "cuts": []}))
    out.append(publish_incumbent({"open": [1]}))
    out.append(publish_certificate({"view": "toy", "version": 2, "cuts": [{"kind": "opt", "terms": [{"s": "s", "pi": {"r": "0.25"}}]}]}))
    out.append(publish_claim({"lower_bound": 1.0}))
    pid = os.fork()
    if pid == 0:
        ok = publish_incumbent({"open": [2]})
        os._exit(0 if ok else 3)
    _, status = os.waitpid(pid, 0)
    out.append(os.WEXITSTATUS(status) == 0)
    print(out)
""")


def test_end_to_end_unix_socket_subprocess(tmp_path):
    store = tmp_path / "store"
    sup = Supervisor(store, T_f=60.0).start()
    client_dir = tmp_path / "client"
    client_dir.mkdir()
    xs.write_client(client_dir)
    env = os.environ | sup.env() | {"PYTHONPATH": str(client_dir)}
    sup.start_clock()
    proc = subprocess.run([sys.executable, "-c", SOLVER], env=env, capture_output=True, text=True, timeout=60)
    rec = sup.stop()
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str([True] * 6)
    assert rec["counts"]["eligible"] == 6 and rec["error"] is None
    assert [e["seq"] for e in rec["incumbents"]] == [1, 3, 6]
    for e in rec["incumbents"]:
        path = store / e["file"]
        assert not os.stat(path).st_mode & stat.S_IWUSR  # immutable
        assert hashlib.sha256(path.read_bytes()).hexdigest() == e["sha256"]
        assert 0 < e["t"] < 60.0
    assert json.loads(read_payload(store, rec["incumbents"][1])) == {"open": [1]}
    # both certificate versions arrived before T_f/4: the first is current at no checkpoint and was pruned
    v1, v2 = rec["certificates"]
    assert v1["file"] is None and v1["retained"] is False and v2["retained"] is True
    assert [c["seq"] for c in rec["checkpoints"]] == [v2["seq"]] * 4
    assert [c["checkpoint"] for c in rec["checkpoints"]] == [15.0, 30.0, 45.0, 60.0]
    assert rec["claims"][0]["payload"] == {"lower_bound": "1.0"} and rec["claims"][0]["error"] is None  # exact token text
    assert json.loads((store / "record.json").read_text()) == rec
    lines = [json.loads(x) for x in (store / "index.jsonl").read_text().splitlines()]
    assert len(lines) == 6 and all(x["status"] == "eligible" for x in lines)
    assert not (store / "tmp").exists()


def test_fd_transport(tmp_path):
    sup = Supervisor(tmp_path / "store", T_f=60.0, transport="fd").start()
    code = ("from src.experiments.xl_supervisor import publish_incumbent as p\n"
            "import sys\nprint([p({'k': i}) for i in range(3)])")
    sup.start_clock()
    proc = subprocess.Popen([sys.executable, "-c", code], env=os.environ | sup.env(), cwd=REPO, pass_fds=sup.pass_fds,
                            stdout=subprocess.PIPE, text=True)
    sup.after_spawn()
    out, _ = proc.communicate(timeout=60)
    rec = sup.stop()
    assert out.strip() == "[True, True, True]"
    assert rec["counts"]["eligible"] == 3 and len(rec["incumbents"]) == 3


def test_bare_socket_path_and_executor_start_time(tmp_path, monkeypatch):
    """xl_executor announces the socket as a bare path and hands its monotonic spawn time to on_start(t0)."""
    sup = Supervisor(tmp_path / "store", T_f=60.0).start()
    monkeypatch.setenv(ENV, sup.socket_path)
    xs._client.update(sock=None, pid=None, addr=None)
    sup.start_clock(time.monotonic() - 10.0)  # the solver was spawned 10 s ago
    try:
        assert publish("incumbent", {"a": 1}) is True
    finally:
        xs._client.update(sock=None, pid=None, addr=None)
    rec = sup.stop()
    assert 10.0 <= rec["incumbents"][0]["t"] < 20.0


def test_no_channel_is_a_noop(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert publish("incumbent", {"a": 1}) is False


def test_torn_corrupt_and_bad_magic_messages_are_ignored(tmp_path, channel):
    sup = Supervisor(tmp_path / "store", T_f=60.0).start()
    channel(sup)
    payload = b'{"open": [3]}'
    # torn: the stream ends inside the payload
    s = raw_conn(sup)
    s.sendall(HEADER.pack(MAGIC, b"I", len(payload), hashlib.sha256(payload).digest()) + payload[:5])
    s.close()
    # corrupt: the hash does not match (acknowledged as rejected; the stream stays usable)
    s = raw_conn(sup)
    s.sendall(HEADER.pack(MAGIC, b"I", len(payload), hashlib.sha256(b"other").digest()) + payload)
    ack = s.recv(xs.ACK.size)
    assert xs.ACK.unpack(ack)[3] == b"R"
    s.sendall(xs.frame("incumbent", payload) + payload)
    assert xs.ACK.unpack(s.recv(xs.ACK.size))[3] == b"E"
    s.close()
    # bad magic: the stream is closed
    s = raw_conn(sup)
    s.sendall(b"XXXX" + bytes(44))
    assert s.recv(16) == b""
    s.close()
    # a torn header
    s = raw_conn(sup)
    s.sendall(MAGIC + b"I")
    s.close()
    assert publish("incumbent", {"open": [4]}) is True
    assert wait_for(lambda: sup.counts["torn"] == 2)
    rec = sup.stop()
    assert {k: rec["counts"][k] for k in ("eligible", "late", "torn", "corrupt", "oversize", "over_volume", "error")} == \
        {"eligible": 2, "late": 0, "torn": 2, "corrupt": 2, "oversize": 0, "over_volume": 0, "error": 0}
    assert [json.loads(read_payload(tmp_path / "store", e)) for e in rec["incumbents"]] == [{"open": [3]}, {"open": [4]}]
    assert not list((tmp_path / "store" / "tmp").glob("*")) if (tmp_path / "store" / "tmp").exists() else True


def test_torn_message_at_stop(tmp_path):
    sup = Supervisor(tmp_path / "store", T_f=60.0).start()
    s = raw_conn(sup)
    s.sendall(HEADER.pack(MAGIC, b"C", 1000, bytes(32)) + b"x" * 10)  # the solver is killed mid-write
    assert wait_for(lambda: any(c.got for c in list(sup._conns.values())))
    rec = sup.stop(grace=0.2)
    s.close()
    assert rec["counts"]["torn"] == 1 and rec["certificates"] == []


def test_late_messages_are_drained_and_ignored(tmp_path, channel):
    clock = FakeClock()
    sup = Supervisor(tmp_path / "store", T_f=4.0, clock=clock).start()
    sup.start_clock(0.0)
    channel(sup)
    clock.now = 3.999
    assert publish("incumbent", {"x": 1}) is True
    clock.now = 4.0  # exactly T_f: not strictly before
    assert publish("incumbent", {"x": 2}) is False
    clock.now = 9.0
    assert publish("certificate", {"view": "toy", "cuts": []}) is False
    rec = sup.stop()
    assert rec["counts"]["eligible"] == 1 and rec["counts"]["late"] == 2
    assert len(rec["incumbents"]) == 1 and rec["certificates"] == []
    assert len(list((tmp_path / "store").glob("*.json"))) == 2  # one incumbent + record.json


def test_checkpoint_versions_and_pruning(tmp_path, channel):
    clock = FakeClock()
    store = tmp_path / "store"
    sup = Supervisor(store, T_f=8.0, clock=clock).start()
    sup.start_clock(0.0)
    channel(sup)
    for k, t in enumerate([1.0, 3.0, 6.0, 6.5, 7.0, 8.0]):
        clock.now = t
        ok = publish("certificate", {"view": "toy", "version": k + 1, "cuts": []})
        assert ok is (t < 8.0)
    rec = sup.stop()
    vs = rec["certificates"]
    assert [v["t"] for v in vs] == [1.0, 3.0, 6.0, 6.5, 7.0]
    assert [v["retained"] for v in vs] == [True, True, False, False, True]
    assert [v["file"] is not None for v in vs] == [True, True, False, False, True]
    by_cp = {c["checkpoint"]: c["seq"] for c in rec["checkpoints"]}
    assert by_cp == {2.0: vs[0]["seq"], 4.0: vs[1]["seq"], 6.0: vs[1]["seq"], 8.0: vs[4]["seq"]}  # 6.0 is not < 6.0
    assert json.loads(read_payload(store, rec["checkpoints"][-1]))["version"] == 5


def test_no_certificate_before_first_checkpoint(tmp_path, channel):
    clock = FakeClock()
    sup = Supervisor(tmp_path / "store", T_f=8.0, clock=clock, keep_all_certificates=True).start()
    sup.start_clock(0.0)
    channel(sup)
    for t in (2.0, 2.5):
        clock.now = t
        assert publish("certificate", {"view": "toy", "cuts": []})
    rec = sup.stop()
    assert rec["checkpoints"][0]["seq"] is None and rec["checkpoints"][1]["seq"] == 2
    assert all(v["file"] for v in rec["certificates"])  # keep_all


def test_oversize_and_volume_caps(tmp_path, channel):
    limits = PublishLimits(max_certificate_bytes=100, max_total_incumbent_bytes=50)
    sup = Supervisor(tmp_path / "store", T_f=60.0, limits=limits).start()
    channel(sup)
    assert publish("certificate", {"view": "toy", "cuts": [], "pad": "x" * 200}) is False
    size = len(xs.encode({"pad": "y" * 20}))
    assert 25 < size <= 50
    assert publish("incumbent", {"pad": "y" * 20}) is True
    assert publish("incumbent", {"pad": "z" * 20}) is False  # would exceed 50 bytes in total
    assert publish("certificate", {"view": "toy", "cuts": []}) is True
    rec = sup.stop()
    assert rec["counts"]["oversize"] >= 1 and rec["counts"]["over_volume"] >= 1
    assert rec["counts"]["eligible"] == 2 and rec["volume"]["incumbent"] == size


def test_unacknowledged_publications_do_not_confuse_later_acks(tmp_path, channel):
    sup = Supervisor(tmp_path / "store", T_f=60.0).start()
    channel(sup)
    for k in range(20):
        assert publish("incumbent", {"k": k}, wait=False) is True
    assert publish("incumbent", {"k": "last"}) is True
    rec = sup.stop()
    assert rec["counts"]["eligible"] == 21


def test_numpy_payloads_are_serialized(tmp_path, channel):
    np = pytest.importorskip("numpy")
    sup = Supervisor(tmp_path / "store", T_f=60.0).start()
    channel(sup)
    assert publish("incumbent", {"y": np.array([0, 1]), "k": np.int64(3), "v": np.float64(0.5)}) is True
    rec = sup.stop()
    assert json.loads(read_payload(tmp_path / "store", rec["incumbents"][0])) == {"y": [0, 1], "k": 3, "v": 0.5}


def test_publication_to_score_with_certificate_checkpoints(tmp_path, channel):
    """Publication -> supervisor record -> independent verification -> certificate checkpoints (exact checker with
    fixed master multipliers) -> v4 score."""
    from src.experiments.xl_certificate import BDView, MasterSolve, check_versions
    from src.experiments.xl_scoring import Reference, attempt_from_supervisor, score_attempt

    view = BDView.from_dict({"view": "toy", "first_stage": {"vars": [], "rows": []},
                             "subproblems": [{"id": "s", "weight": 1, "vars": [{"id": "x", "cost": 1, "lb": 0, "ub": 1}],
                                              "rows": [{"id": "r", "sense": ">=", "rhs": 1, "a": {"x": 3}}]}]})
    clock = FakeClock()
    store = tmp_path / "store"
    sup = Supervisor(store, T_f=100.0, clock=clock).start()
    sup.start_clock(0.0)
    channel(sup)
    for t, kind, obj in [(5.0, "incumbent", {"x": 0.5}), (10.0, "certificate", {"view": "toy", "cuts": [
            {"kind": "opt", "terms": [{"s": "s", "pi": {"r": "0.2"}}]}]}),
            (20.0, "incumbent", {"x": 0.25}),   # infeasible: 3 * 0.25 < 1
            (40.0, "incumbent", {"x": 0.34}),
            (60.0, "certificate", {"view": "toy", "cuts": [{"kind": "opt", "terms": [{"s": "s", "pi": {"r": "0.3333"}}]}]}),
            (120.0, "incumbent", {"x": 1 / 3})]:
        clock.now = t
        publish(kind, obj)
    rec = sup.stop()

    def verify(payload: bytes):
        x = Fraction(json.loads(payload)["x"])
        return float(x) if 3 * x >= 1 and 0 <= x <= 1 else None

    ones = lambda lo, hi, obj, rows, tl, lim: MasterSolve("optimal", [1.0] * len(rows))
    checks = check_versions(view, rec["checkpoints"], store_dir=store, solve_master=ones)
    assert [c["lb"] for c in checks] == [Fraction("0.2"), Fraction("0.2"), Fraction("0.3333"), Fraction("0.3333")]
    att = attempt_from_supervisor(rec, verify, method_ok=True, certificate=checks, attempt=1)
    refm = Reference(T_f=100.0, z_best=Fraction(1, 3), z_mip=Fraction("0.34"), t_mip=80.0, eps_p=Fraction("0.01"),
                     eps_b=Fraction("0.001"), lp_lo=Fraction(1, 3), lp_hi=Fraction(1, 3))
    r = score_attempt(att, refm)
    assert r["objective"] == 0.34 and r["incumbents_eligible"] == 2 and att.extra["infeasible_incumbents"] == 1
    assert r["matches_or_beats_MIP"] is True and r["t_matches_MIP"] == 40.0 and r["speedup"] == pytest.approx(2.0)
    assert r["cert_ok"] and r["bound_target"] is True and r["t_bound"] == 75.0 and r["t_bound_receipt"] == 60.0
    assert rec["counts"]["late"] == 1


# ---------------------------------------------------------------- Codex review 45 regressions

def test_review45_2_duplicate_key_and_malformed_claims(tmp_path, channel):
    from src.experiments.xl_scoring import Attempt, Reference, attempt_from_supervisor, score_attempt

    sup = Supervisor(tmp_path / "store", T_f=60.0).start()
    channel(sup)
    assert publish("claim", '{"lower_bound":200,"lower_bound":90}') is False      # stored, recorded as malformed
    assert publish("claim", "[" * 20000 + "]" * 20000) is False                    # 40 KB of nesting: no RecursionError
    assert publish("claim", '{"lower_bound": NaN}') is False
    assert publish("claim", '{"lower_bound": "90"}') is True
    assert publish("incumbent", {"z": 100}) is True                                 # still draining
    rec = sup.stop()
    assert rec["errors"] == [] and rec["counts"]["malformed_claim"] == 3
    assert [c["error"] for c in rec["claims"]] == ["duplicate_key", "too_deep", "non_finite", None]
    assert all(c["file"] for c in rec["claims"])  # raw claims preserved
    att = attempt_from_supervisor(rec, lambda b: json.loads(b)["z"], method_ok=True)
    r = score_attempt(att, Reference(T_f=60.0, z_best=Fraction(100), integral=True, eps_p=Fraction(0)))
    assert r["refuted_claim"] and [x["why"] for x in r["refutations"]][:3] == [
        "malformed_claim:duplicate_key", "malformed_claim:too_deep", "malformed_claim:non_finite"]
    assert r["success"]["near_best"] is False


def test_review45_2_supervisor_errors_keep_draining_and_reach_scoring(tmp_path, channel, monkeypatch):
    from src.experiments.xl_scoring import Reference, attempt_from_supervisor, score_attempt

    sup = Supervisor(tmp_path / "store", T_f=60.0).start()
    channel(sup)

    def boom(data):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(xs, "parse_claim", boom)
    assert publish("claim", {"lower_bound": 1}) is False
    assert publish("incumbent", {"z": 100}) is True
    rec = sup.stop()
    assert rec["errors"] and "injected failure" in rec["errors"][0] and rec["counts"]["eligible"] >= 1
    att = attempt_from_supervisor(rec, lambda b: json.loads(b)["z"], method_ok=True)
    assert att.infrastructure
    r = score_attempt(att, Reference(T_f=60.0, z_best=Fraction(100), integral=True, eps_p=Fraction(0)))
    assert r["success"]["near_best"] is None and r["infrastructure_errors"]


def test_review45_3_capacity_is_reserved_at_header_acceptance(tmp_path):
    sup = Supervisor(tmp_path / "store", T_f=60.0, limits=PublishLimits(max_total_incumbent_bytes=10)).start()
    payload = b'{"a":123}'  # 9 bytes
    hdr = xs.frame("incumbent", payload)
    a, b = raw_conn(sup), raw_conn(sup)
    a.sendall(hdr)
    assert wait_for(lambda: sup.reserved["incumbent"] == 9)
    b.sendall(hdr)
    assert b.recv(16) == b""               # over_volume: closed before its payload is read
    a.sendall(payload)
    assert xs.ACK.unpack(a.recv(xs.ACK.size))[3] == b"E"
    a.close()
    b.close()
    rec = sup.stop()
    assert rec["counts"]["eligible"] == 1 and rec["counts"]["over_volume"] == 1 and rec["volume"]["incumbent"] == 9


def test_review45_3_empty_messages_and_count_event_traffic_caps(tmp_path, channel):
    sup = Supervisor(tmp_path / "store", T_f=60.0, limits=PublishLimits(max_incumbents=2)).start()
    channel(sup)
    s = raw_conn(sup)
    for _ in range(50):
        s.sendall(HEADER.pack(MAGIC, b"I", 0, hashlib.sha256(b"").digest()))
    for _ in range(50):
        assert xs.ACK.unpack(s.recv(xs.ACK.size))[3] == b"R"
    s.sendall(xs.frame("incumbent", b"[1]") + b"[1]")       # the stream is still usable
    assert xs.ACK.unpack(s.recv(xs.ACK.size))[3] == b"E"
    s.close()
    assert publish("incumbent", [2]) is True
    assert publish("incumbent", [3]) is False                # message-count cap (2 incumbents)
    rec = sup.stop()
    assert rec["counts"]["empty"] == 50 and rec["counts"]["over_count"] >= 1 and rec["messages"]["incumbent"] == 2
    assert len(list((tmp_path / "store").glob("I*.json"))) == 2   # empty messages create no files

    sup = Supervisor(tmp_path / "store2", T_f=60.0, limits=PublishLimits(max_events=3)).start()
    s = raw_conn(sup)
    for _ in range(3):
        s.sendall(HEADER.pack(MAGIC, b"I", 0, bytes(32)))
        s.recv(xs.ACK.size)
    s.sendall(HEADER.pack(MAGIC, b"I", 0, bytes(32)))
    assert s.recv(16) == b""                                  # metadata cap reached: closed, not indexed
    s.close()
    rec = sup.stop()
    assert rec["index_lines"] == 3 and rec["counts"]["over_events"] == 1
    assert len((tmp_path / "store2" / "index.jsonl").read_text().splitlines()) == 3

    clock = FakeClock()
    sup = Supervisor(tmp_path / "store3", T_f=1.0, limits=PublishLimits(max_traffic_bytes=20), clock=clock).start()
    sup.start_clock(0.0)
    clock.now = 5.0                                           # late frames are drained but count as traffic
    s = raw_conn(sup)
    s.sendall(xs.frame("incumbent", b"x" * 15) + b"x" * 15)
    assert xs.ACK.unpack(s.recv(xs.ACK.size))[3] == b"L"
    s.sendall(xs.frame("incumbent", b"y" * 15))
    assert s.recv(16) == b""                                  # over_traffic
    s.close()
    rec = sup.stop()
    assert rec["counts"]["late"] == 1 and rec["counts"]["over_traffic"] == 1 and rec["traffic_bytes"] == 15


# ---------------------------------------------------------------- Codex review 47 regressions

def test_review47_1_deep_or_large_claims_are_malformed_and_record_is_always_written(tmp_path, channel, monkeypatch):
    sup = Supervisor(tmp_path / "store", T_f=60.0).start()
    channel(sup)
    deep = '{"lower_bound": "1", "meta": ' + "[" * 1200 + "]" * 1200 + "}"   # Codex: 2,410 bytes, 1,200 nested arrays
    assert publish("claim", deep) is False
    wide = '{"lower_bound": "1", "meta": [' + ",".join(["1"] * 300) + "]}"
    assert publish("claim", wide) is False
    assert publish("claim", '{"lower_bound": "1", "meta": [[1, 2], {"a": [3]}]}') is True   # small nesting is fine
    rec = sup.stop()
    assert [c["error"] for c in rec["claims"]] == ["too_deep", "too_many_nodes", None] and rec["errors"] == []
    assert json.loads((tmp_path / "store" / "record.json").read_text())["claims"][2]["payload"]["meta"] == [["1", "2"], {"a": ["3"]}]

    # an unserializable record (injected) still yields record.json, with the failure recorded
    sup = Supervisor(tmp_path / "store2", T_f=60.0).start()
    channel(sup)
    monkeypatch.setattr(xs, "parse_claim", lambda data: ({"lower_bound": "1", "x": {1, 2}}, None))
    assert publish("claim", {"lower_bound": 1}) is True
    rec = sup.stop()
    saved = json.loads((tmp_path / "store2" / "record.json").read_text())
    assert saved["errors"] and "serialize_full" in saved["errors"][0] and saved["claims"][0]["payload"] is None


def test_review47_5_event_slots_are_reserved_at_header_acceptance(tmp_path):
    sup = Supervisor(tmp_path / "store", T_f=60.0, limits=PublishLimits(max_events=1)).start()
    payload = b'{"a":1}'
    a, b = raw_conn(sup), raw_conn(sup)
    a.sendall(xs.frame("incumbent", payload))
    assert wait_for(lambda: sup.event_slots == 1)
    b.sendall(xs.frame("incumbent", payload))
    assert b.recv(16) == b""                      # no index slot left: rejected before its payload
    a.sendall(payload)
    assert xs.ACK.unpack(a.recv(xs.ACK.size))[3] == b"E"
    a.close()
    b.close()
    rec = sup.stop()
    lines = [json.loads(x) for x in (tmp_path / "store" / "index.jsonl").read_text().splitlines()]
    assert rec["counts"]["eligible"] == 1 and rec["counts"]["over_events"] == 1
    assert [x["seq"] for x in lines if x["status"] == "eligible"] == [e["seq"] for e in rec["incumbents"]]


# ---------------------------------------------------------------- Codex review 49 regressions

def test_review49_2_claims_must_be_strict_utf8(tmp_path, channel):
    from src.experiments.xl_scoring import attempt_from_supervisor, score_attempt
    from src.experiments.xl_scoring import Reference as Ref

    sup = Supervisor(tmp_path / "store", T_f=60.0).start()
    channel(sup)
    codex = '{"lower_bound": "1", "meta": "Ģ", "x": ' + "[" * 997 + "]" * 997 + "}"   # Codex: UTF-16, 997 arrays
    plain = '{"lower_bound": "1"}'
    cases = [codex.encode("utf-16-le"), codex.encode("utf-16"), plain.encode("utf-16-le"), plain.encode("utf-16-be"),
             plain.encode("utf-16"), plain.encode("utf-32"), plain.encode("utf-32-le"), plain.encode("utf-8-sig")]
    for data in cases:
        assert publish("claim", data) is False
    assert publish("claim", plain.encode("utf-8")) is True
    rec = sup.stop()
    errors = [c["error"] for c in rec["claims"]]
    assert errors[-1] is None and all(e in ("not_utf8", "parse_error", "bom", "too_deep", "too_many_nodes") for e in errors[:-1])
    assert errors[1] == "not_utf8" and errors[-2] == "bom" and rec["errors"] == []
    att = attempt_from_supervisor(rec, lambda b: 100, method_ok=True)
    r = score_attempt(att, Ref(T_f=60.0, z_best=Fraction(100), integral=True, eps_p=Fraction(0)))
    assert r["infrastructure_errors"] == [] and r["refuted_claim"]   # a malformed claim fails the attempt


# ---------------------------------------------------------------- Codex review 53 regressions

def test_review53_3_transport_errors_are_infrastructure_peer_disconnect_is_eof(tmp_path, monkeypatch):
    import errno as _errno

    from src.experiments.xl_scoring import Reference, attempt_from_supervisor, score_attempt

    real_read = os.read
    mode = {"fd": None, "raise": None}

    def fake_read(fd, n):
        if fd == mode["fd"] and mode["raise"] is not None:
            exc, mode["raise"] = mode["raise"], None
            raise exc
        return real_read(fd, n)

    monkeypatch.setattr(os, "read", fake_read)
    payload = b'{"z":100}'
    # EIO while a valid incumbent is pending: recorded as infrastructure, never a silent end of stream
    sup = Supervisor(tmp_path / "eio", T_f=60.0, transport="fd").start()
    mode["fd"], mode["raise"] = sup._rfd, OSError(_errno.EIO, "injected EIO")
    os.write(sup._wfd, xs.frame("incumbent", payload) + payload)
    assert wait_for(lambda: sup.errors)
    rec = sup.stop()
    assert rec["errors"] and "EIO" in rec["errors"][0] and rec["incumbents"] == []
    att = attempt_from_supervisor(rec, lambda b: json.loads(b)["z"], method_ok=True)
    r = score_attempt(att, Reference(T_f=60.0, z_best=Fraction(100), integral=True, eps_p=Fraction(0)))
    assert r["success"]["near_best"] is None and r["infrastructure_errors"]
    # a peer disconnect (ECONNRESET) is an end of stream: a partial frame is torn, no infrastructure error
    sup = Supervisor(tmp_path / "reset", T_f=60.0, transport="fd").start()
    mode["fd"] = sup._rfd
    frame = xs.frame("incumbent", payload) + payload
    os.write(sup._wfd, frame[:10])
    assert wait_for(lambda: any(c.partial for c in list(sup._conns.values())))
    mode["raise"] = ConnectionResetError(_errno.ECONNRESET, "peer reset")
    os.write(sup._wfd, frame[10:12])
    assert wait_for(lambda: sup.counts["torn"] == 1)
    rec = sup.stop()
    assert rec["errors"] == [] and rec["counts"]["torn"] == 1


def test_review53_4_finalization_failures_reach_the_returned_and_persisted_record(tmp_path, channel, monkeypatch):
    sup = Supervisor(tmp_path / "store", T_f=60.0).start()
    channel(sup)
    assert publish("incumbent", {"z": 1}) is True

    def bad_fsync(fd):
        raise OSError("injected final fsync failure")

    monkeypatch.setattr(os, "fsync", bad_fsync)
    rec = sup.stop()
    saved = json.loads((tmp_path / "store" / "record.json").read_text())
    assert any("stop/index" in e for e in rec["errors"]) and saved["errors"] == rec["errors"]
    monkeypatch.undo()

    sup = Supervisor(tmp_path / "store2", T_f=60.0).start()
    real_chmod = os.chmod

    def bad_chmod(p, mode, *a, **kw):
        if str(p).endswith("record.json"):
            raise OSError("injected chmod failure")
        return real_chmod(p, mode, *a, **kw)

    monkeypatch.setattr(os, "chmod", bad_chmod)
    rec = sup.stop()
    saved = json.loads((tmp_path / "store2" / "record.json").read_text())
    assert any("chmod_record" in e for e in saved["errors"]) and set(saved["errors"]) <= set(rec["errors"])
