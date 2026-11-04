"""k-median (XL) Stage 1 runner with supervised arms (protocol v4; Codex review 46 #2). Phoenix only.

Modes
  --task INSTANCE ARM     one arm in its own process tree (new session): the supervisor starts its clock just before
                          spawning the child, reads length-framed messages from the child's stdout
                          ("KMMSG <kind> <nbytes>\\n" + payload: incumbent, bound, certificate, result), timestamps each
                          on complete receipt, persists it at once (events.jsonl appended and fsynced; certificate
                          versions as immutable files certs/v<seq>.json with their sha256) and hard-kills the whole
                          process group at T (SIGKILL), whatever the arm is doing. Then, outside T:
                          (1) every incumbent received strictly before T is evaluated exactly, the best one verified
                          with the stdlib verifier; (2) the certificate version current at each frozen checkpoint
                          (60/300/600/900 s, the last complete version received strictly before it) is checked by the
                          CANONICAL evaluator (src/experiments/xl_certificate.check_versions_supervised with
                          h_view_xl.km_view: a forked checker under one hard 600 s deadline, final version first, with
                          the panel's frozen checker limits), and by the optimized family backend as a cross-check
                          (diagnostic only, separately timed, NEVER scored, also when the canonical check is
                          unavailable). Infrastructure failures of the canonical check (view build failure, checker
                          errors, stored-version mismatches) are recorded per checkpoint and make the arm's bound metrics
                          undefined (rerun). --dense adds development diagnostics (family backend on every version).
                          Every run writes into a FRESH directory claimed atomically (mkdir): the first attempt is
                          OUT/<instance>__<arm>, a retry OUT/<instance>__<arm>/attempt_<k> (k >= 2); a non-empty
                          directory is never reused, and result.json is replaced atomically (temp file, fsync, rename).
  --reference INSTANCE    frozen LP* reference interval: engineered Benders Phase 1 only (no branching) to convergence,
                          its final certificate checked canonically and by the family backend; LP_lo = the larger
                          certified exact bound (provenance recorded), LP_hi = the exact master objective of its LP point
                          repaired exactly.
  --summarize             per instance and arm: certified bound at each checkpoint, bgap_LP against the frozen interval,
                          bound_target (cert_ok and bgap_LP <= eps_b), t_bound (first checkpoint meeting it), primal
                          gap to z_best and near_best (<= eps_p), and the frozen acceptance rules (panel file). An arm
                          with an infrastructure marker has undefined bound metrics, and an instance whose decision
                          needs such an arm is undefined ("pass": None, rerun): never a beaten control. Only the 15
                          panel arms are read (other directories are flagged, never scored); per arm exactly one attempt
                          is scored, the earliest complete one with a valid identity and no infrastructure marker; a
                          result whose identity (instance, arm, T, panel sha256, instance sha256, frozen code hashes)
                          does not match the summarized panel is infrastructure (pending, rerun).
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MAGIC = b"KMMSG"
CHECKPOINTS = (60.0, 300.0, 600.0, 900.0)
MESSAGE_KINDS = ("incumbent", "bound", "certificate", "result")
MAX_HEADER = 256                    # bytes of one header line ("KMMSG <kind> <nbytes>\n")
MAX_MESSAGE_BYTES = 256 * 2**20     # the certificate byte limit; other messages are far smaller
CHECKER_BUDGET = 600.0
DEFAULT_PANEL = ROOT / "outputs" / "scaleup" / "references" / "km_stage1_panel.json"
GUROBI_MEM_GB = 15.5          # Gurobi's own cap inside the 16 GB process budget (Slurm cgroup), no soft limit
ALL_ARMS = ["eng", "eng_nostab", "naive_lp", "naive_int", "lp_classical_bar", "lp_classical_dual", "lp_radius_bar",
            "lp_radius_dual", "empty", "analytical", "oneshot_greedy", "pool", "ils", "lagr", "mip"]   # the 15 panel arms
OPTIONAL_CALLBACKS = ("on_certificate", "on_bound")   # dropped for an arm whose constructor has no such parameter
TASK_ALLOCATION = 3600.0      # Slurm time per task (scaleup_km_stage1.sbatch)
DIAGNOSTIC_BUDGET = 300.0     # ONE cumulative hard deadline for all diagnostic family checks of a task
PUBLICATION_VOLUME_CAP = 4 * 2**30   # total certificate bytes stored per arm run (panel: publication_volume_cap_bytes)


def arms() -> dict:
    from src.bd.km.b_mip_xl import KMCompactLP, KMCompactMIP
    from src.bd.km.c_naive_bd_xl import KMNaiveBenders, KMNaiveLPBenders
    from src.bd.km.d_ref_bd_xl import KMBendersXL
    from src.bd.km.f_heuristic_xl import KMLagrangian, KMLocalSearch, KMNoFeedbackPool, KMStaticCertificate
    lp = {"mem_limit_gb": GUROBI_MEM_GB}
    return {
        "eng": (KMBendersXL, {}),
        "eng_nostab": (KMBendersXL, {"stabilize": False}),
        "naive_lp": (KMNaiveLPBenders, {}),
        "naive_int": (KMNaiveBenders, {}),
        "lp_classical_bar": (KMCompactLP, {"formulation": "classical", "method": 2, **lp}),
        "lp_classical_dual": (KMCompactLP, {"formulation": "classical", "method": 1, **lp}),
        "lp_radius_bar": (KMCompactLP, {"formulation": "radius", "method": 2, **lp}),
        "lp_radius_dual": (KMCompactLP, {"formulation": "radius", "method": 1, **lp}),
        "empty": (KMStaticCertificate, {"kind": "empty"}),
        "analytical": (KMStaticCertificate, {"kind": "analytical"}),
        "oneshot_greedy": (KMStaticCertificate, {"kind": "oneshot_greedy"}),
        "pool": (KMNoFeedbackPool, {}),
        "ils": (KMLocalSearch, {}),
        "lagr": (KMLagrangian, {}),
        "mip": (KMCompactMIP, {"formulation": "radius", **lp}),
        "_test_sleeper": (_Sleeper, {}),
        **{f"_test_{m}": (_ProtocolTestArm, {"mode": m}) for m in _ProtocolTestArm.MODES},
    }


class _Sleeper:
    """Test arm: publishes one incumbent and one certificate, then never returns (the supervisor must kill it)."""

    def __init__(self, *, time_limit: float, on_incumbent, on_certificate, **kw) -> None:
        self.on_incumbent, self.on_certificate = on_incumbent, on_certificate

    def run(self, data_path: str) -> dict:
        from src.bd.km.a_data_xl import load_instance
        from src.bd.km.c_naive_bd_xl import greedy_design
        from src.bd.km.g_certificate_xl import one_shot
        inst = load_instance(data_path)
        D = inst.distances()
        y = greedy_design(D, inst.k)
        self.on_incumbent([int(i) for i in y.nonzero()[0]], stage="sleeper")
        cert = one_shot(D, y.nonzero()[0])
        cert.version = 1
        self.on_certificate(cert)
        while True:
            time.sleep(3600)


class _ProtocolTestArm:
    """Test arms for the publication protocol and exit classification (failure injection; never in a panel).
    Every mode first publishes one valid incumbent of the greedy design."""

    MODES = ("ok", "crash", "no_result", "bad_header", "null_incumbent", "empty_incumbent", "unknown_kind",
             "bad_bound", "after_result", "torn_unkilled", "torn_then_sleep", "sigkill_self")

    def __init__(self, *, time_limit: float, on_incumbent, raw=None, mode: str = "ok", **kw) -> None:
        self.on_incumbent, self.raw, self.mode = on_incumbent, raw, mode

    def frame(self, kind: str, payload: bytes) -> None:
        self.raw(MAGIC + b" " + kind.encode() + b" " + str(len(payload)).encode() + b"\n" + payload)

    def run(self, data_path: str) -> dict:
        from src.bd.km.a_data_xl import load_instance
        from src.bd.km.c_naive_bd_xl import greedy_design
        inst = load_instance(data_path)
        self.on_incumbent([int(i) for i in greedy_design(inst.distances(), inst.k).nonzero()[0]], stage="test")
        m = self.mode
        if m == "crash":
            raise RuntimeError("injected arm crash")
        if m == "no_result":
            os._exit(0)                                   # exits normally without its terminal result
        if m == "sigkill_self":                           # dies by SIGKILL before T (an OOM kill looks like this)
            os.kill(os.getpid(), signal.SIGKILL)
        if m == "bad_header":
            self.raw(b"GARBAGE HEADER\n")
        elif m == "null_incumbent":
            self.frame("incumbent", b"null")
        elif m == "empty_incumbent":
            self.frame("incumbent", b"{}")
        elif m == "unknown_kind":
            self.frame("telemetry", b"{}")
        elif m == "bad_bound":
            self.frame("bound", b'{"value": "NaN"}')
        elif m == "after_result":
            self.frame("result", b'{"termination_reason": "finished"}')
            self.frame("incumbent", b'{"design": [0], "stage": "late"}')
            os._exit(0)
        elif m in ("torn_unkilled", "torn_then_sleep"):
            self.raw(MAGIC + b" incumbent 100\n" + b'{"design": ')   # a frame cut inside its payload
            if m == "torn_unkilled":
                os._exit(0)
            while True:
                time.sleep(3600)
        return {"termination_reason": "finished"}


BOUND_CONTROLS = ["lp_classical_bar", "lp_classical_dual", "lp_radius_bar", "lp_radius_dual", "empty", "analytical",
                  "oneshot_greedy", "ils", "pool"]
DIAGNOSTICS = ["lagr", "naive_int", "eng_nostab", "mip"]


def stem(path: str) -> str:
    return Path(path).stem


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_json_atomic(path: Path, obj: Any) -> None:
    """Replace `path` atomically: a unique temporary file in the same directory, written, flushed and fsynced, then
    renamed over `path` (and the directory fsynced). A reader sees the previous complete file or the new complete
    one, never a truncated one (Codex review 56 #3: the diagnostic rewrite cannot damage the canonical result)."""
    data = json.dumps(obj, default=str, indent=1).encode()
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            os.fchmod(fh.fileno(), 0o644)   # mkstemp creates 0600
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:   # (b) the write failed: the previous file is untouched; the error propagates
        try:
            os.unlink(tmp)
        except OSError:   # benign: already renamed or never created
            pass
        raise
    try:
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:   # benign: the directory fsync is a durability hint; the rename itself is atomic
        pass


ATTEMPT_PREFIX = "attempt_"


def attempt_index(name: str) -> int | None:
    """k for a retry directory name attempt_<k> (k >= 2, ASCII digits, no leading zero), else None."""
    k = name[len(ATTEMPT_PREFIX):] if name.startswith(ATTEMPT_PREFIX) else ""
    if not (k.isascii() and k.isdigit()) or k[0] == "0" or int(k) < 2:
        return None
    return int(k)


def claim_attempt(base: Path) -> tuple[int, Path]:
    """A fresh, empty output directory for one run of an arm, claimed atomically with mkdir (never an existing one):
    attempt 1 is `base` itself, a retry is base/attempt_<k> with the smallest free k >= 2."""
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        base.mkdir()
        return 1, base
    except FileExistsError:   # benign: an earlier attempt owns base; this run takes a fresh retry directory
        pass
    k = 2
    while True:
        d = base / f"{ATTEMPT_PREFIX}{k}"
        try:
            d.mkdir()
            return k, d
        except FileExistsError:   # benign: taken by an earlier attempt
            k += 1


def code_commit() -> str | None:
    """The git commit of the code root, recorded with every arm result (provenance; the frozen file hashes are the
    enforced code identity). None outside a git checkout."""
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):   # recorded as unknown
        return None
    commit = r.stdout.strip() if r.returncode == 0 else ""
    return commit or None


# ---------------------------------------------------------------- child

def arm_kwargs(cls, kwargs: dict) -> dict:
    """Constructor keywords for an arm: the optional publication callbacks (on_certificate, on_bound) are dropped when
    the constructor accepts neither them nor **kwargs (e.g. KMCompactMIP publishes no certificates); every other
    keyword is passed as is, so a wrong one still fails loudly."""
    params = inspect.signature(cls).parameters
    if any(q.kind == q.VAR_KEYWORD for q in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params or k not in OPTIONAL_CALLBACKS}


def child(instance: str, arm: str, T: float, start_wall: float) -> None:
    try:   # the arm, not the supervisor, is the kernel's OOM victim inside the job's memory cgroup
        Path("/proc/self/oom_score_adj").write_text("1000")
    except OSError:   # not Linux or not permitted: OOM evidence then decides nothing for this run (infrastructure)
        pass
    out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)                 # stray prints go to stderr, never into the framed channel
    sys.stdout = sys.stderr
    lock = threading.Lock()

    def write_all(data: bytes) -> None:
        view = memoryview(data)
        while view:                     # a raw write may be partial: always complete the frame
            view = view[out.write(view):]

    def send(kind: str, payload: bytes) -> None:
        with lock:
            write_all(MAGIC + b" " + kind.encode() + b" " + str(len(payload)).encode() + b"\n" + payload)

    from src.bd.km.g_certificate_xl import to_bytes

    def on_inc(design, stage=None):
        send("incumbent", json.dumps({"design": [int(i) for i in design], "stage": stage}).encode())

    def on_bound(b):
        send("bound", json.dumps({"value": float(b)}).encode())

    def on_cert(cert):
        send("certificate", to_bytes(cert))

    cls, kw = arms()[arm]
    if arm.startswith("_test_"):
        kw = dict(kw, raw=lambda data: (lock.acquire(), write_all(data), lock.release()))
    left = T - (time.time() - start_wall) - 2.0
    kwargs = arm_kwargs(cls, dict(time_limit=max(1.0, left), threads=1, on_incumbent=on_inc, on_bound=on_bound,
                                  on_certificate=on_cert, **kw))
    res = cls(**kwargs).run(instance)
    slim = {k: v for k, v in res.items() if not isinstance(v, (list, dict)) and k not in ("trace", "cut_log")}
    for key in ("final_lp_point", "lp_point"):
        if res.get(key) is not None:
            slim["lp_point"] = res[key]
    if isinstance(res.get("stats"), dict):
        slim["lp_point"] = slim.get("lp_point") or res["stats"].get("final_lp_point")
        slim["stats"] = {k: v for k, v in res["stats"].items() if not isinstance(v, (list, dict))}
    send("result", json.dumps(slim, default=str).encode())


# ---------------------------------------------------------------- supervisor

def job_memory_cgroup() -> Path | None:
    """This process's cgroup v2 directory (the Slurm job/step cgroup), or None."""
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                return Path("/sys/fs/cgroup" + line[3:])
    except OSError:   # no cgroup v2 information: no OOM evidence (a -9 exit then stays infrastructure)
        return None
    return None


def oom_kill_count() -> int | None:
    """The kernel's oom_kill counter of this job's memory cgroup (memory.events), None if unavailable. The arm runs in
    the same cgroup with oom_score_adj 1000, so an increase during its run while it dies by SIGKILL is a
    kernel-confirmed OOM kill of the arm (as in src/experiments/xl_executor.py)."""
    cg = job_memory_cgroup()
    if cg is None:
        return None
    try:
        for line in (cg / "memory.events").read_text().splitlines():
            key, _, value = line.partition(" ")
            if key == "oom_kill":
                return int(value)
    except (OSError, ValueError):   # unreadable: no evidence
        return None
    return None


def _read_exact(f, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def supervise(instance: str, arm: str, T: float, outdir: Path, volume_cap: int = PUBLICATION_VOLUME_CAP) -> tuple[list[dict], dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    if any(outdir.iterdir()):   # Codex review 56 #3: a retry never mixes with an earlier attempt's files
        raise FileExistsError(f"refusing to reuse the non-empty arm output directory {outdir} "
                              "(a retry takes a fresh attempt directory: claim_attempt)")
    certs = outdir / "certs"
    certs.mkdir()
    events = open(outdir / "events.jsonl", "xb")
    records: list[dict] = []
    torn: list[str] = []            # frames cut at the end of the stream (benign only for the run we killed at T)
    malformed = [0]                 # messages of our own arm that are not valid JSON: an evaluator bug (infrastructure)
    violations: list[str] = []      # publication-protocol violations of the arm (evaluator code): infrastructure
    terminal: list[int] = []        # sequence number of the terminal "result" message
    reader_errors: list[str] = []   # any reader or I/O failure: the arm's result is infrastructure (rerun)
    stored = [0]                    # certificate bytes stored (the total publication-volume cap)
    over_volume = [0]               # certificate versions beyond the cap: received, not stored, never eligible
    oom_before = oom_kill_count()
    start_wall = time.time()
    t0 = time.monotonic()
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child", instance, arm, "--T", str(T),
                             "--start", repr(start_wall)], stdout=subprocess.PIPE, stderr=open(outdir / "stderr.log", "xb"),
                            start_new_session=True, cwd=str(ROOT))

    def reader() -> None:
        try:
            read_all()
        except BaseException as exc:  # noqa: BLE001 - (b) recorded; publications after it are lost: infrastructure
            reader_errors.append(f"{type(exc).__name__}: {exc}"[:300])

    def violation(what: str) -> None:   # (b) the arm (evaluator code) broke the publication protocol
        violations.append(what[:300])

    def read_all() -> None:
        f = proc.stdout
        seq = 0
        while True:
            hdr = f.readline(MAX_HEADER)
            if not hdr:
                break                                     # clean end of stream
            if not hdr.endswith(b"\n"):
                if len(hdr) >= MAX_HEADER:
                    violation(f"header longer than {MAX_HEADER} bytes")
                else:
                    torn.append("header")                 # cut inside a header line at end of stream
                break
            parts = hdr.split()
            if len(parts) != 3 or parts[0] != MAGIC or not parts[2].isdigit():
                violation(f"bad header {hdr[:60]!r}")
                break                                     # framing is lost: nothing after it is trusted
            kind, nbytes = parts[1].decode(errors="replace"), int(parts[2])
            if kind not in MESSAGE_KINDS:
                violation(f"unknown message kind {kind!r}")
                break
            if nbytes > MAX_MESSAGE_BYTES:
                violation(f"{kind} message of {nbytes} bytes")
                break
            payload = _read_exact(f, nbytes)
            if payload is None:
                torn.append(kind)                         # cut inside a payload at end of stream
                break
            t = time.monotonic() - t0
            seq += 1
            if terminal:
                violation(f"{kind} message after the terminal result")
            rec = {"seq": seq, "t": round(t, 6), "kind": kind, "bytes": nbytes, "sha256": sha256(payload)}
            if kind == "certificate" and stored[0] + nbytes > volume_cap:
                rec["file"], rec["over_volume"] = None, True   # the frozen publication-volume cap (same for every arm)
                over_volume[0] += 1
            elif kind == "certificate":
                fname = f"v{seq}.json"
                with open(certs / fname, "xb") as fh:   # immutable: never overwritten
                    fh.write(payload)
                    fh.flush()
                    os.fsync(fh.fileno())
                rec["file"] = fname
                stored[0] += nbytes
            else:
                problem = payload_problem(kind, payload)
                if problem:
                    violation(f"{kind} message {seq}: {problem}")
                    rec["payload"], rec["valid"] = None, False
                    malformed[0] += 1
                else:
                    rec["payload"] = json.loads(payload)
                if kind == "result":
                    terminal.append(seq)
            events.write(json.dumps(rec).encode() + b"\n")
            events.flush()
            os.fsync(events.fileno())
            records.append(rec)

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    killed = False
    try:
        proc.wait(timeout=max(0.0, T - (time.monotonic() - t0)))
    except subprocess.TimeoutExpired:   # the normal end of an arm: hard kill at T
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:   # benign: the process group already exited
            pass
        killed = True
        proc.wait()
    wall = time.monotonic() - t0
    th.join(timeout=60)
    if th.is_alive():
        reader_errors.append("publication reader did not finish")
    try:
        events.close()
    except OSError as exc:   # (b) infrastructure
        reader_errors.append(f"events.jsonl close: {exc}"[:300])
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
    oom_after = oom_kill_count()
    oom = (not killed and proc.returncode == -signal.SIGKILL and oom_before is not None and oom_after is not None
           and oom_after > oom_before)
    run = {"returncode": proc.returncode, "killed_at_T": killed, "wall_seconds": round(wall, 3), "torn_messages": len(torn),
           "torn_kinds": list(torn), "child_peak_rss_mb": round(peak, 1), "reader_errors": list(reader_errors),
           "malformed_messages": malformed[0], "protocol_violations": list(violations), "terminal_result": bool(terminal),
           "certificate_bytes_stored": stored[0], "over_volume_certificates": over_volume[0], "volume_cap_bytes": volume_cap,
           "oom_kill_before": oom_before, "oom_kill_after": oom_after, "oom_confirmed": bool(oom),
           "resource_outcome": "memory_limit (kernel-confirmed OOM kill of the arm)" if oom else None}
    run["infrastructure"] = classify_run(run)
    return records, run


def classify_run(run: dict) -> list[str]:
    """Infrastructure problems of one arm run (the arms are evaluator code), from the exit and the publication log:
      - killed at T by the supervisor (SIGKILL): expected; a frame cut at the end of the stream is benign then;
      - otherwise the arm must exit with code 0 after a valid terminal "result" message; any other exit (nonzero code,
        a signal we did not send, no terminal result) is infrastructure, as is a frame cut without our kill;
      - every protocol violation (framing, unknown kind, payload schema, message after the result) is infrastructure.
    Outcomes the arm itself classified in its result (e.g. termination_reason time_limit or memory_limit) are kept, and
    so is a kernel-confirmed OOM kill (run["oom_confirmed"]: SIGKILL + oom_kill increase in the job's memory.events);
    an exit -9 without that evidence stays infrastructure."""
    problems = [f"protocol: {v}" for v in run.get("protocol_violations") or []]
    if run.get("killed_at_T"):
        if run.get("returncode") not in (-signal.SIGKILL, 0):
            problems.append(f"arm exited with {run.get('returncode')} while being killed at T")
    elif run.get("oom_confirmed"):
        # (a) arm resource outcome: SIGKILL with a kernel-confirmed OOM kill in the job cgroup during the run; its
        # eligible publications received before the kill are kept (scored as at the kill); a cut last frame is benign
        pass
    else:
        if run.get("returncode") != 0:
            rc = run.get("returncode")
            problems.append(f"arm exited unexpectedly: {'signal ' + str(-rc) if isinstance(rc, int) and rc < 0 else 'code ' + str(rc)}")
        elif not run.get("terminal_result"):
            problems.append("arm finished without its terminal result")
        if run.get("torn_messages"):
            problems.append(f"torn publication without a deadline kill: {run.get('torn_kinds')}")
    return problems


def payload_problem(kind: str, payload: bytes) -> str | None:
    """Schema of a non-certificate message: incumbent {"design": [int >= 0, ...], "stage"?: str | null}; bound
    {"value": finite number}; result {...} (an object). Returns the problem or None."""
    try:
        obj = json.loads(payload)
    except ValueError:   # (b) not JSON: protocol violation of an evaluator arm
        return "not JSON"
    if not isinstance(obj, dict):
        return "not a JSON object"
    if kind == "incumbent":
        d = obj.get("design")
        if not isinstance(d, list) or not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in d):
            return "design must be a list of facility indices"
        if obj.get("stage") is not None and not isinstance(obj["stage"], str):
            return "stage must be a string"
    elif kind == "bound":
        v = obj.get("value")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return "value must be a finite number"
    return None


# ---------------------------------------------------------------- evaluation and certification (outside T)

def evaluate_incumbents(instance: str, records: list[dict], T: float, arm: str) -> dict:
    from src.bd.km.a_data_xl import load_instance
    from src.bd.km.b_mip_xl import DesignEvaluator, km_output
    from src.bd.km.e_verifier_xl import verify_solution
    inst = load_instance(instance)
    ev = DesignEvaluator(inst)
    traj, best, best_design, invalid, late, malformed = [], None, None, 0, 0, 0
    for r in records:
        if r["kind"] != "incumbent":
            continue
        if r.get("valid") is False or not isinstance(r.get("payload"), dict) or "design" not in r["payload"]:
            malformed += 1   # (b) recorded as a protocol violation by the reader; counted here too, never skipped silently
            continue
        if r["t"] >= T:
            late += 1
            continue
        v = ev.objective(r["payload"]["design"])
        if v is None:
            invalid += 1
            continue
        if best is None or v < best - 1e-9 * max(1.0, abs(v)):
            best, best_design = v, r["payload"]["design"]
            traj.append((r["t"], v, r["payload"].get("stage")))
    verified, verification = None, None
    if best_design is not None:
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
                json.dump(km_output(inst, best_design, ev.D, method=arm), fh)
            vr = verify_solution(instance, fh.name)
        except Exception as exc:  # noqa: BLE001 - (b) the verifier could not run: infrastructure (arm_metrics)
            verification = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        else:
            verification = {"ok": vr.get("ok"), "issues": (vr.get("issues") or [])[:10], "objective_value": vr.get("objective_value")}
            verified = bool(vr.get("ok") and vr.get("objective_value") is not None
                            and abs(vr["objective_value"] - best) <= 1e-6 * max(1.0, abs(best)))
    return {"best": best, "t_first": traj[0][0] if traj else None, "t_best": traj[-1][0] if traj else None,
            "verified": verified, "verification": verification, "invalid": invalid, "late": late, "malformed": malformed,
            "trajectory": traj[-60:], "design": best_design}


def checkpoint_versions(records: list[dict], T: float) -> list[tuple[float, dict | None]]:
    certs = [r for r in records if r["kind"] == "certificate" and r["t"] < T and r.get("file")]   # stored versions only
    cps = sorted({c for c in CHECKPOINTS if c <= T} | {T})
    out = []
    for c in cps:
        cur = None
        for r in certs:
            if r["t"] < c:
                cur = r
        out.append((c, cur))
    return out


def frozen_checker(panel_path: str | Path | None = None) -> dict:
    """The panel's frozen canonical-checker settings: limits (family Limits overridden by the panel's "checker" values),
    the per-attempt budget, and the optional explicit memory cap `checker.memory_bytes` (RLIMIT_AS of the forked
    checker; without it no memory failure can be charged to a certificate: it is infrastructure)."""
    from src.bd.km.g_certificate_xl import Limits
    panel = {}
    path = Path(panel_path) if panel_path is not None else DEFAULT_PANEL
    if path.exists():
        panel = json.loads(path.read_text())
    ch = panel.get("checker", {})
    base = Limits()
    limits = Limits(max_bytes=int(ch.get("max_bytes", base.max_bytes)), max_nonzeros=int(ch.get("max_nonzeros_expanded", base.max_nonzeros)),
                    max_master_nonzeros=int(ch.get("max_master_nonzeros", base.max_master_nonzeros)),
                    max_seconds=float(ch.get("budget_seconds_per_attempt", base.max_seconds)),
                    max_abs_coef=float(ch.get("max_abs_coef", base.max_abs_coef)), max_token_chars=int(ch.get("token_chars", base.max_token_chars)),
                    max_exponent=int(ch.get("decimal_exponent", base.max_exponent)), gurobi_mem_gb=ch.get("gurobi_mem_gb", base.gurobi_mem_gb))
    return {"limits": limits, "budget": float(ch.get("budget_seconds_per_attempt", CHECKER_BUDGET)),
            "memory_bytes": ch.get("memory_bytes"), "fit_fraction": float(ch.get("memory_fit_fraction", 0.5)),
            "panel": path.name if path.exists() else None}


def certify(instance: str, records: list[dict], outdir: Path, T: float, panel_path: str | Path | None = None) -> dict:
    """CANONICAL certification only (the scored path); the family cross-check runs later in family_diagnostics."""
    from src.bd.km.a_data_xl import load_instance
    from src.bd.km.g_certificate_xl import CertFailure, Checker, parse_certificate_bytes
    from src.bd.km.h_view_xl import generic_limits, km_view, to_generic
    import src.experiments.xl_certificate as xc
    inst = load_instance(instance)
    certs = outdir / "certs"
    cps = checkpoint_versions(records, T)
    out: dict = {"checkpoints": [], "versions": sum(1 for r in records if r["kind"] == "certificate")}
    if not any(cur for _, cur in cps):
        out["checkpoints"] = [{"checkpoint": c, "seq": None, "cert_ok": False, "lb": "0", "lb_float": 0.0,
                               "reason": "absent"} for c, _ in cps]
        return out
    t0 = time.monotonic()
    chk = Checker(inst)
    out["family_setup_seconds"] = round(time.monotonic() - t0, 2)
    fz = frozen_checker(panel_path)
    limits = fz["limits"]
    out["checker"] = {"budget": fz["budget"], "memory_bytes": fz["memory_bytes"], "fit_fraction": fz["fit_fraction"], "panel": fz["panel"]}
    def convert(data: bytes) -> bytes:
        """km-cert-v1 -> the generic certificate, run INSIDE the supervised checker (its deadline and memory gate).
        Malformed input (the family parser's CertFailure) rejects the version with that reason (charged to the agent);
        a MemoryError goes through the memory gate; any other exception is an evaluator failure (infrastructure)."""
        try:
            cert = parse_certificate_bytes(data, chk.m, chk.n, limits)
        except CertFailure as exc:   # (a) malformed input: rejected with its reason (arm_metrics: evaluator arm -> b)
            raise xc.CertificateRejected(exc.reason) from None
        return json.dumps(to_generic(cert, chk), separators=(",", ":")).encode()

    t2 = time.monotonic()
    try:
        view = km_view(inst, chk.D, natural=False)
        out["view_seconds"] = round(time.monotonic() - t2, 2)
        recs = []
        for c, cur in cps:   # the stored km-cert-v1 versions with their receipt hashes (verified by the checker)
            if cur is None:
                recs.append({"checkpoint": c})
            else:
                recs.append({"checkpoint": c, "file": cur.get("file"), "sha256": cur.get("sha256"), "bytes": cur.get("bytes"),
                             "t": cur["t"], "seq": cur["seq"]})
        res = xc.check_versions_supervised(view, recs, store_dir=certs, limits=generic_limits(limits), budget_seconds=fz["budget"],
                                           memory_bytes=fz["memory_bytes"], fit_fraction=fz["fit_fraction"], convert=convert)
        canon = []
        for d in res:
            d = dict(d)
            d["lb"] = str(d["lb"])
            canon.append({k: v for k, v in d.items() if k != "stats"} | {"stats": {k: v for k, v in (d.get("stats") or {}).items()
                                                                                    if not isinstance(v, (list, dict))}})
        out["checkpoints"] = canon
        out["canonical"] = True
        del view
    except Exception as exc:  # noqa: BLE001 - (b) the canonical view could not be built: infrastructure (rerun)
        out["canonical"] = False
        out["canonical_error"] = f"{type(exc).__name__}: {exc}"[:300]
        out["checkpoints"] = [{"checkpoint": c, "seq": cur["seq"] if cur else None, "receipt_t": cur["t"] if cur else None,
                               "cert_ok": None, "lb": None, "lb_float": None, "reason": "infrastructure",
                               "infrastructure": "canonical_view_failed",
                               "failure": {"stage": "view", "checkpoint": i, "category": "infrastructure"}}
                              for i, (c, cur) in enumerate(cps)]
    return out


def run_with_deadline(fn, budget: float, max_bytes: int = 64 * 2**20) -> tuple[str, Any]:
    """fn() in a forked child under ONE hard wall-clock deadline; its JSON result comes back through a pipe. Returns
    ("done", value), ("error", message), ("deadline", None) or ("ipc_error", message). Used for diagnostics only."""
    import select

    if budget <= 0:
        return "deadline", None
    rfd, wfd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(rfd)
        try:
            data = json.dumps({"ok": fn()}, default=str).encode()
        except BaseException as exc:  # noqa: BLE001 - reported to the parent (diagnostics are never scored)
            data = json.dumps({"error": f"{type(exc).__name__}: {exc}"[:300]}).encode()
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(wfd, view):]
        finally:
            os._exit(0)
    os.close(wfd)
    deadline = time.monotonic() + budget
    buf, status = bytearray(), "running"
    try:
        os.set_blocking(rfd, False)
        poller = select.poll()
        poller.register(rfd, select.POLLIN | select.POLLHUP | select.POLLERR)
        while status == "running":
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status = "deadline"
                break
            if not poller.poll(max(1, min(int(remaining * 1000) + 1, 250))):
                continue
            try:
                chunk = os.read(rfd, 1 << 16)
            except BlockingIOError:   # benign: no data yet
                continue
            if not chunk:
                status = "done"
                break
            buf += chunk
            if len(buf) > max_bytes:
                status = "ipc_error"
    except Exception as exc:  # noqa: BLE001 - diagnostics only: recorded
        status = f"ipc_error: {type(exc).__name__}: {exc}"[:300]
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:   # already exited
            pass
        try:
            os.waitpid(pid, 0)
        except OSError:   # already reaped
            pass
        os.close(rfd)
    if status != "done":
        return ("deadline", None) if status == "deadline" else ("ipc_error", status)
    try:
        msg = json.loads(bytes(buf))
    except ValueError:   # the child died before writing a complete result
        return "ipc_error", "incomplete diagnostic result"
    return ("done", msg["ok"]) if "ok" in msg else ("error", msg.get("error"))


def family_diagnostics(instance: str, records: list[dict], outdir: Path, T: float, canonical: dict, budget: float,
                       panel_path: str | Path | None = None, dense: bool = False) -> dict:
    """Family km-cert-v1 backend cross-check (diagnostic only, NEVER scored): the checkpoint versions (final first)
    and, with `dense`, every version, all under ONE cumulative hard deadline `budget` (a forked child, killed at the
    deadline), after the canonical result has been persisted."""
    cps = checkpoint_versions(records, T)
    if not any(cur for _, cur in cps):
        return {"status": "no certificates", "label": "family km-cert-v1 backend: diagnostic cross-check, never scored"}
    certs = outdir / "certs"

    def work() -> dict:
        from src.bd.km.a_data_xl import load_instance
        from src.bd.km.g_certificate_xl import Checker
        chk = Checker(load_instance(instance))
        limits = frozen_checker(panel_path)["limits"]
        fam, errors = {}, []
        for c, cur in reversed(cps):
            if cur is None or str(cur["seq"]) in fam:
                continue
            try:
                r = chk.certify((certs / cur["file"]).read_bytes(), limits=limits)
            except Exception as exc:  # noqa: BLE001 - diagnostic only: recorded, never scored
                errors.append(f"v{cur['seq']}: {type(exc).__name__}: {exc}"[:200])
                continue
            fam[str(cur["seq"])] = {k: r.get(k) for k in ("cert_ok", "lb", "bound", "reason", "lp_status", "nonzeros_expanded",
                                                          "cuts_used", "checker_seconds")}
        dev = []
        if dense:  # development diagnostics only
            for r in records:
                if r["kind"] == "certificate" and r["t"] < T and r.get("file"):
                    try:
                        rr = chk.certify((certs / r["file"]).read_bytes(), limits=limits)
                    except Exception as exc:  # noqa: BLE001 - diagnostic only: recorded, never scored
                        errors.append(f"dense v{r['seq']}: {type(exc).__name__}: {exc}"[:200])
                        continue
                    dev.append({"t": r["t"], "seq": r["seq"], "cert_ok": rr["cert_ok"], "bound": rr["bound"]})
        return {"family": fam, "family_errors": errors, "development_dense_grid": dev}

    t0 = time.monotonic()
    status, value = run_with_deadline(work, budget)
    out = {"status": status, "budget_seconds": budget, "seconds": round(time.monotonic() - t0, 2),
           "label": "family km-cert-v1 backend: diagnostic cross-check, never scored"}
    if status == "done":
        out.update(value)
        fam = value.get("family", {})
        agree = []
        for d in canonical.get("checkpoints", []):
            f = fam.get(str(d.get("seq"))) if d.get("seq") is not None else None
            if f is not None and d.get("cert_ok") is not None and not d.get("infrastructure") and d.get("lb") is not None:
                diff = abs(Fraction(f["lb"]) - Fraction(d["lb"]))
                agree.append(bool(f["cert_ok"] == d["cert_ok"] and diff <= Fraction(1, 10**9) * max(Fraction(1), abs(Fraction(d["lb"])))))
        out["family_agrees"] = all(agree) if agree else None
    else:
        out["detail"] = value
        out["family_agrees"] = None
    return out


def identity(task: dict, panel_path: str | Path | None) -> dict:
    """Runtime, task and panel identity recorded with every result (the frozen code hashes are refreshed at submission)."""
    import platform
    path = Path(panel_path) if panel_path is not None else DEFAULT_PANEL
    panel = json.loads(path.read_text()) if path.exists() else {}
    files = sorted(set(panel.get("code_sha256_at_preparation", {})) | {"scripts/phoenix/scaleup_km_stage1.py",
                                                                         "src/experiments/xl_certificate.py"})
    code = {}
    for f in files:
        try:
            code[f] = sha256((ROOT / f).read_bytes())
        except OSError:   # recorded as missing (reviewed at submission)
            code[f] = None
    try:
        import gurobipy
        grb = ".".join(map(str, gurobipy.gurobi.version()))
    except Exception:  # noqa: BLE001 - recorded as unknown
        grb = None
    return {"python": sys.version.split()[0], "executable": sys.executable, "host": platform.node(), "gurobi": grb,
            "slurm": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_RESTART_COUNT")},
            "argv": sys.argv, "task": task,
            "panel": {"file": path.name if path.exists() else None, "sha256": sha256(path.read_bytes()) if path.exists() else None,
                      "status": panel.get("status")},
            "code_sha256": code}


def run_task(instance: str, arm: str, T: float, out_root: Path, dense: bool, panel_path: str | Path | None = None,
             diag_budget: float | None = None) -> dict:
    """One arm: supervised run, incumbents, CANONICAL certification, persisted (result.json) BEFORE the diagnostic
    family cross-check, which then runs under one cumulative hard deadline that fits the task allocation. The run
    takes a fresh attempt directory (claim_attempt) and result.json is replaced atomically both times."""
    task_t0 = time.monotonic()
    k, outdir = claim_attempt(out_root / f"{stem(instance)}__{arm}")
    path = Path(panel_path) if panel_path is not None else DEFAULT_PANEL
    cap = PUBLICATION_VOLUME_CAP
    if path.exists():
        cap = int(json.loads(path.read_text()).get("publication_volume_cap_bytes", cap))
    records, run = supervise(instance, arm, T, outdir, volume_cap=cap)
    t0 = time.monotonic()
    try:
        inc = evaluate_incumbents(instance, records, T, arm)
    except Exception as exc:  # noqa: BLE001 - (b) evaluator failure: infrastructure (rerun), never a score
        run.setdefault("evaluator_errors", []).append(f"incumbents: {type(exc).__name__}: {exc}"[:300])
        inc = {"best": None, "t_first": None, "t_best": None, "verified": None, "invalid": None, "late": None, "trajectory": [], "design": None}
    sb = [(r["t"], r["payload"]["value"]) for r in records if r["kind"] == "bound" and r.get("payload") and r["t"] < T]
    result = next((r["payload"] for r in records if r["kind"] == "result"), None)
    try:
        cert = certify(instance, records, outdir, T, panel_path)
    except Exception as exc:  # noqa: BLE001 - (b) evaluator failure: infrastructure (rerun), never a score
        run.setdefault("evaluator_errors", []).append(f"certify: {type(exc).__name__}: {exc}"[:300])
        cert = {"canonical": False, "canonical_error": f"{type(exc).__name__}: {exc}"[:300], "checkpoints": []}
    row = {"instance": stem(instance), "arm": arm, "T": T, "run": run, "incumbents": inc,
           "solver_bounds": {"final": max((b for _, b in sb), default=None),
                             "at_checkpoints": {str(c): max((b for t, b in sb if t < c), default=None) for c, _ in
                                                checkpoint_versions(records, T)},
                             "label": "solver-reported internal master bound (never scored)"},
           "certificates": cert, "arm_result": result, "diagnostics": {"status": "pending"},
           "identity": identity({"instance": str(instance), "instance_sha256": sha256(Path(instance).read_bytes()), "arm": arm,
                                 "T": T}, panel_path) | {"code_commit": code_commit()},
           "attempt": {"index": k, "dir": outdir.relative_to(out_root).as_posix()},
           "post_seconds": round(time.monotonic() - t0, 2)}
    write_json_atomic(outdir / "result.json", row)   # the canonical result first
    budget = DIAGNOSTIC_BUDGET if diag_budget is None else diag_budget
    budget = min(budget, TASK_ALLOCATION - (time.monotonic() - task_t0) - 120.0)
    row["diagnostics"] = family_diagnostics(instance, records, outdir, T, cert, budget, panel_path, dense)
    row["certificates"]["family_agrees"] = row["diagnostics"].get("family_agrees")   # diagnostic, never scored
    row["task_seconds"] = round(time.monotonic() - task_t0, 2)
    write_json_atomic(outdir / "result.json", row)
    return row


# ---------------------------------------------------------------- frozen LP* reference

def reference(instance: str, T: float, out_root: Path) -> dict:
    from src.bd.km.a_data_xl import load_instance
    from src.bd.km.d_ref_bd_xl import KMBendersXL
    from src.bd.km.g_certificate_xl import Checker, Limits, lp_interval, to_bytes
    from src.bd.km.h_view_xl import check_canonical, km_view
    inst = load_instance(instance)
    got: list = []
    t0 = time.monotonic()
    res = KMBendersXL(time_limit=T, branch=False, root_fraction=1.0, ls_fraction=0.0, cert_interval=1e9,
                      on_certificate=got.append).run(instance)
    run_s = time.monotonic() - t0
    cert = got[-1] if got else None
    point = (res.get("stats") or {}).get("final_lp_point")
    chk = Checker(inst)
    limits = Limits()
    fam = chk.certify(cert, limits=limits) if cert is not None else None
    t1 = time.monotonic()
    can = None
    try:
        view = km_view(inst, chk.D, natural=False)
        r = check_canonical(view, cert, chk, limits) if cert is not None else None
        can = {"cert_ok": r.cert_ok, "lb": str(r.lb), "lb_float": r.lb_float, "reason": r.reason,
               "seconds": r.stats.get("checker_seconds")} if r is not None else None
    except MemoryError as exc:   # (b) reference preparation: recorded in the row (canonical failed); LP_lo then comes
        # from the family bound with its provenance, LP_hi is unaffected (an exactly repaired primal), reviewed before freezing
        can = {"cert_ok": False, "lb": "0", "reason": f"MemoryError: {exc}", "infrastructure": True}
    can_s = time.monotonic() - t1
    iv = lp_interval(chk, [("canonical xl_certificate, eng Phase-1 final certificate", can),
                           ("family km-cert-v1 backend, eng Phase-1 final certificate", fam)],
                     [("eng Phase-1 final LP point, exact repair", point)])
    out_root.mkdir(parents=True, exist_ok=True)
    cpath = out_root / f"{stem(instance)}__lp_reference_certificate.json"
    if cert is not None:
        cpath.write_bytes(to_bytes(cert))
    # the LP primal witness from which LP_hi is repaired exactly: saved and hashed with the reference
    ppath = out_root / f"{stem(instance)}__lp_reference_point.json"
    if point is not None:
        ppath.write_text(json.dumps({"instance": stem(instance), "source": "eng Phase-1 final LP point (y)",
                                     "point": [repr(float(v)) for v in point]}))
    row = {"instance": stem(instance), "instance_sha256": sha256(Path(instance).read_bytes()), "interval": iv,
           "phase1_converged": (res.get("stats") or {}).get("phase1_converged"), "phase1_seconds": (res.get("stats") or {}).get("phase1_seconds"),
           "certificate_file": cpath.name if cert is not None else None,
           "certificate_sha256": sha256(cpath.read_bytes()) if cert is not None else None,
           "lp_point_file": ppath.name if point is not None else None,
           "lp_point_sha256": sha256(ppath.read_bytes()) if point is not None else None,
           "identity": identity({"instance": str(instance), "instance_sha256": sha256(Path(instance).read_bytes()),
                                 "mode": "reference", "T": T}, None),
           "canonical": can, "family": {k: fam.get(k) for k in ("cert_ok", "lb", "bound", "reason", "nonzeros_expanded",
                                                                 "checker_seconds")} if fam else None,
           "run_seconds": round(run_s, 1), "canonical_seconds": round(can_s, 1),
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)}
    (out_root / f"{stem(instance)}__lp_reference.json").write_text(json.dumps(row, default=str, indent=1))
    return row


# ---------------------------------------------------------------- summary and frozen acceptance

STAGE1_REQUIRED = 3          # jointly >= 3 of the 4 Stage 1 instances of a tier
DECISION_STATES = ("pass", "fail", "pending")


def reference_state(iv: dict | None) -> str:
    """'ready' (a rigorous frozen interval), 'pending_reference' (not computed yet) or 'undefined_reference' (not
    rigorous): only 'ready' instances are decided; the others are never counted as failures."""
    if not iv:
        return "pending_reference"
    return "ready" if iv.get("rigorous") and iv.get("hi") is not None else "undefined_reference"


def identity_problems(r: dict, name: str, arm: str, panel: dict, panel_sha: str) -> list[str]:
    """Codex review 56 #1: an arm result is scored only if it was produced for this instance and arm, at the panel's T,
    under the summarized panel file (sha256), on the frozen instance file (sha256) and with the frozen code (every file
    hash in panel["code_sha256_at_preparation"]; the commit as well if the panel freezes one). Any problem listed here
    makes the result infrastructure (pending, rerun), never a score."""
    p = []
    ident = r.get("identity") if isinstance(r.get("identity"), dict) else {}
    task = ident.get("task") if isinstance(ident.get("task"), dict) else {}
    if r.get("instance") != name:
        p.append(f"instance {r.get('instance')!r} != {name!r}")
    if r.get("arm") != arm or task.get("arm") != arm:
        p.append(f"arm {r.get('arm')!r}/{task.get('arm')!r} != {arm!r}")
    T = panel.get("T")
    if isinstance(T, bool) or not isinstance(T, (int, float)) or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or x != T for x in (r.get("T"), task.get("T"))):
        p.append(f"T {r.get('T')!r}/{task.get('T')!r} != panel T {T!r}")
    got_panel = (ident.get("panel") or {}).get("sha256") if isinstance(ident.get("panel"), dict) else None
    if got_panel != panel_sha:
        p.append(f"panel sha256 {got_panel!r} != summarized panel {panel_sha}")
    want = ((panel.get("references") or {}).get(name) or {}).get("instance_sha256")
    if not want:
        p.append("the panel freezes no instance_sha256 for this instance")
    elif task.get("instance_sha256") != want:
        p.append(f"instance sha256 {task.get('instance_sha256')!r} != frozen {want}")
    frozen = panel.get("code_sha256_at_preparation") or {}
    got = ident.get("code_sha256") if isinstance(ident.get("code_sha256"), dict) else {}
    if not frozen:
        p.append("the panel freezes no code hashes")
    bad = sorted(f for f, h in frozen.items() if got.get(f) != h)
    if bad:
        p.append(f"code sha256 differs from the frozen panel: {bad}")
    if panel.get("code_commit") and ident.get("code_commit") != panel["code_commit"]:
        p.append(f"code commit {ident.get('code_commit')!r} != frozen {panel['code_commit']}")
    return p


def reference_problems(row: dict | None, name: str, panel: dict) -> list[str]:
    """Identity of a reference interval (frozen in the panel, or a holdout reference row): its instance hash, when
    recorded, must be the panel's frozen instance hash."""
    want = ((panel.get("references") or {}).get(name) or {}).get("instance_sha256")
    got = (row or {}).get("instance_sha256")
    if got is not None and got != want:
        return [f"reference instance sha256 {got!r} != frozen {want!r}"]
    return []


def arm_attempts(out_root: Path, name: str, arm: str) -> list[tuple[int, Path]]:
    """Every attempt directory of one arm, in order: base (attempt 1) and base/attempt_<k> (k >= 2)."""
    base = out_root / f"{name}__{arm}"
    if not base.is_dir():
        return []
    found = [(1, base)] + [(attempt_index(d.name), d) for d in base.iterdir() if d.is_dir() and attempt_index(d.name)]
    return sorted(found)


def select_attempt(out_root: Path, name: str, arm: str, panel: dict, panel_sha: str, iv: dict, tgt: Fraction | None) -> dict:
    """Exactly one complete attempt per arm (Codex review 56 #3): the earliest complete attempt with a valid identity and
    no infrastructure marker; if every attempt with a valid identity carries one, the earliest of those (its markers
    keep the arm undefined); if no complete attempt has a valid identity, none (identity problems listed)."""
    seen, valid, clean = [], [], []
    for k, d in arm_attempts(out_root, name, arm):
        rel = d.relative_to(out_root).as_posix()
        f = d / "result.json"
        if not f.is_file():
            seen.append({"attempt": rel, "status": "incomplete"})
            continue
        try:
            r = json.loads(f.read_text())
            if not isinstance(r, dict):
                raise ValueError("result.json is not an object")
            problems = identity_problems(r, name, arm, panel, panel_sha)
            infra = [] if problems else arm_metrics(r, iv, tgt, None, panel["rules"]["eps_p"]).get("infrastructure") or []
        except (OSError, ValueError, KeyError, TypeError, AttributeError, ZeroDivisionError) as exc:   # (b) unreadable or
            # malformed result (an evaluator failure): this attempt is infrastructure, never scored
            seen.append({"attempt": rel, "status": "unreadable", "error": f"{type(exc).__name__}: {exc}"[:300]})
            continue
        seen.append({"attempt": rel, "status": "complete", "identity_problems": problems, "infrastructure": list(infra)})
        if not problems:
            valid.append((rel, r))
            if not infra:
                clean.append((rel, r))
    pick = clean[0] if clean else (valid[0] if valid else None)
    return {"attempt": pick[0] if pick else None, "result": pick[1] if pick else None, "attempts": seen,
            "complete": any(a["status"] != "incomplete" for a in seen),
            "identity_problems": [] if pick else [p for a in seen for p in a.get("identity_problems", [])]
                                                  + [a["error"] for a in seen if a["status"] == "unreadable"]}


def scan_unexpected(out_root: Path, names: list[str]) -> list[str]:
    """Directories that are not an arm of the panel (<instance>__<arm>, the 15 arms) or one of its attempt or certs
    directories: flagged, never read."""
    expected = {f"{n}__{a}" for n in names for a in ALL_ARMS}
    flagged = []
    if not out_root.is_dir():
        return flagged
    for d in sorted(out_root.iterdir()):
        if not d.is_dir():
            continue
        if d.name not in expected:
            flagged.append(d.name)
            continue
        for sub in sorted(d.iterdir()):
            if sub.is_dir() and sub.name != "certs" and attempt_index(sub.name) is None:
                flagged.append(f"{d.name}/{sub.name}")
    return flagged


def instance_decision(name: str, spec: dict, panel: dict, out_root: Path, iv: dict | None, panel_sha: str,
                      ref_problems: list[str] | None = None) -> dict:
    """One instance: reference gate, all 15 arms resolved with a valid identity (z_best is final only then), metrics and
    acceptance. Only the whitelisted arms are read, one attempt each (select_attempt)."""
    from src.bd.km.g_certificate_xl import target
    rules = panel["rules"]
    state = reference_state(iv)
    if ref_problems:
        state = "reference_identity_mismatch"
    base = {"headline": spec["headline"], "lp_interval": iv, "reference_state": state}
    unexpected = [u for u in scan_unexpected(out_root, [name]) if u.startswith(f"{name}__")]
    if unexpected:
        base["unexpected"] = unexpected   # flagged, never read
    if state != "ready":
        return base | {"decision": "pending", "acceptance": {"pass": None, "undefined": state, **({"problems": ref_problems} if ref_problems else {})}}
    tgt = target(iv, rules["eps_b"])
    sel = {a: select_attempt(out_root, name, a, panel, panel_sha, iv, tgt) for a in ALL_ARMS}
    base["attempts"] = {a: s["attempt"] for a, s in sel.items()}
    retried = {a: s["attempts"] for a, s in sel.items() if len(s["attempts"]) > 1}
    if retried:
        base["attempt_history"] = retried
    arms_res = {a: s["result"] for a, s in sel.items() if s["result"] is not None}
    missing = [a for a in ALL_ARMS if not sel[a]["complete"]]
    ident = {a: sel[a]["identity_problems"] for a in ALL_ARMS if sel[a]["complete"] and sel[a]["result"] is None}
    if missing or ident:
        why = "; ".join(x for x in ("arms pending" if missing else "",
                                    "invalid results (identity mismatch or unreadable; infrastructure): rerun" if ident else "") if x)
        return base | {"decision": "pending", "acceptance": {"pass": None, "undefined": why, "missing": missing,
                                                             **({"identity": ident} if ident else {})}}
    # z_best is final only when every arm's primal side is resolved (no run-level infrastructure)
    unresolved = [a for a in ALL_ARMS if arm_metrics(arms_res[a], iv, tgt, None, rules["eps_p"]).get("primal_undefined")]
    if unresolved:
        return base | {"decision": "pending", "acceptance": {"pass": None, "undefined": "z_best not final: rerun", "arms": unresolved}}
    z_pre = panel["references"].get(name, {}).get("z_pre")
    verified = [r["incumbents"]["best"] for r in arms_res.values() if r["incumbents"].get("best") is not None
                and r["incumbents"].get("verified") is True]
    z_best = min([x for x in [z_pre, *verified] if x is not None], default=None)
    per = {a: arm_metrics(r, iv, tgt, z_best, rules["eps_p"]) for a, r in arms_res.items()}
    acc = accept(per, spec["headline"], rules, panel["T"], z_best=z_best)
    decision = {True: "pass", False: "fail", None: "pending"}[acc.get("pass")]
    return base | {"target": str(tgt), "z_best": z_best, "arms": per, "acceptance": acc, "decision": decision}


def tier_decision(stage1: list[str], holdout: list[str] | None) -> dict:
    """Stage 1: pass with >= STAGE1_REQUIRED passes, fail once that is impossible, else pending. The holdout is scored
    only after Stage 1 passed (holdout=None: gated) and needs every holdout instance to pass."""
    n_pass, n_fail = stage1.count("pass"), stage1.count("fail")
    s1 = "pass" if n_pass >= STAGE1_REQUIRED else ("fail" if len(stage1) - n_fail < STAGE1_REQUIRED else "pending")
    if s1 != "pass":
        return {"stage1": s1, "holdout": "not_run", "tier": "rejected" if s1 == "fail" else "pending"}
    if holdout is None:
        return {"stage1": s1, "holdout": "not_run", "tier": "pending"}
    h = "fail" if "fail" in holdout else ("pass" if holdout and all(x == "pass" for x in holdout) else "pending")
    return {"stage1": s1, "holdout": h, "tier": {"pass": "retained", "fail": "rejected", "pending": "pending"}[h]}


def summarize(out_root: Path, panel_path: Path) -> dict:
    """Instance decisions (pass / fail / pending, never failing for a missing or non-rigorous reference or an
    unresolved arm) and the frozen tier decision: Stage 1 jointly >= 3 of 4, then the holdout 2 of 2, the holdout scored
    only after Stage 1 passed and its own reference job (out_root/<name>__lp_reference.json) is done."""
    raw = panel_path.read_bytes()
    panel = json.loads(raw)
    panel_sha = sha256(raw)          # the identity every arm result must carry (identity_problems)
    names = [n for spec in panel["tiers"].values() for n in [*spec["stage1"], *spec.get("holdout", [])]]
    summary: dict = {"panel": panel_path.name, "panel_sha256": panel_sha, "instances": {}, "tiers": {},
                     "unexpected": scan_unexpected(out_root, names)}   # flagged directories, never read
    frozen = panel.get("lp_star_frozen", {}).get("intervals", {})
    for tier, spec in panel["tiers"].items():
        s1 = []
        for name in spec["stage1"]:
            iv = frozen.get(name)
            d = instance_decision(name, spec, panel, out_root, iv, panel_sha, reference_problems(iv, name, panel)) | \
                {"tier": tier, "part": "stage1"}
            summary["instances"][name] = d
            s1.append(d["decision"])
        gate = tier_decision(s1, None)
        hold = None
        if gate["stage1"] == "pass":
            hold = []
            for name in spec.get("holdout", []):
                ref = out_root / f"{name}__lp_reference.json"   # computed by the holdout job before any arm result is read
                row = json.loads(ref.read_text()) if ref.exists() else None
                iv = row["interval"] if row else None
                d = instance_decision(name, spec, panel, out_root, iv, panel_sha, reference_problems(row, name, panel)) | \
                    {"tier": tier, "part": "holdout"}
                summary["instances"][name] = d
                hold.append(d["decision"])
        else:
            for name in spec.get("holdout", []):
                summary["instances"][name] = {"tier": tier, "part": "holdout", "decision": "not_run",
                                              "acceptance": {"pass": None, "undefined": f"Stage 1 {gate['stage1']}"}}
        summary["tiers"][tier] = tier_decision(s1, hold) | {"stage1_decisions": dict(zip(spec["stage1"], s1)),
                                                            "holdout_decisions": dict(zip(spec.get("holdout", []), hold)) if hold is not None else None}
    write_json_atomic(out_root / "stage1_summary.json", summary)
    return summary


def final_label(cp: dict | None, tgt: Fraction | None) -> str:
    """Reporting label of a checkpoint: target / weak_bound / checker_exhausted / certificate_resource:<reason> /
    absent / infrastructure (checker exhaustion is never conflated with a weak bound)."""
    if cp is None:
        return "absent"
    if cp.get("infrastructure"):
        return "infrastructure"
    if cp.get("reason") == "absent":
        return "absent"
    if cp.get("reason") == "checker_budget_exhausted":
        return "checker_exhausted"
    if cp.get("cert_ok"):
        return "target" if tgt is not None and Fraction(cp["lb"]) >= tgt else "weak_bound"
    return f"certificate_resource:{cp.get('reason')}"


def arm_metrics(r: dict, iv: dict, tgt: Fraction | None, z_best: float | None, eps_p: float) -> dict:
    """Bound and primal metrics of one arm. Checkpoints come from the canonical checker only. Any infrastructure marker
    (a checkpoint's "infrastructure", or no canonical result) makes the bound metrics undefined (None) and is listed
    in "infrastructure" (rerun); the family backend never stands in. Gaps are exact (Fractions); the solver's
    internal-master bound is reported next to the certified one (export loss), never scored."""
    cert = r.get("certificates", {})
    cps = cert.get("checkpoints", [])
    infra = sorted({str(cp["infrastructure"]) for cp in cps if cp.get("infrastructure")})
    if cert.get("canonical") is False:
        infra.append("canonical_unavailable")
    if any(cp.get("backend") for cp in cps):
        infra.append("non_canonical_backend")
    run = r.get("run", {})
    inc = r.get("incumbents", {})
    run_infra = [f"publication_reader: {e}" for e in run.get("reader_errors") or []] + \
                [f"evaluator: {e}" for e in run.get("evaluator_errors") or []]
    if run.get("malformed_messages"):
        run_infra.append(f"evaluator_arm_malformed_messages: {run['malformed_messages']}")
    run_infra += [f"arm_run: {p}" for p in (run.get("infrastructure") if "infrastructure" in run else classify_run(run))]
    if inc.get("malformed"):
        run_infra.append(f"evaluator_arm_malformed_incumbents: {inc['malformed']}")
    # Stage 1 arms are evaluator code: an invalid design, a best incumbent without successful independent
    # verification (verifier failure or disagreement), or a content-rejected certificate is an evaluator failure
    if inc.get("invalid"):
        run_infra.append(f"evaluator_arm_invalid_designs: {inc['invalid']}")
    if inc.get("best") is not None and inc.get("verified") is not True:
        run_infra.append(f"incumbent_not_verified: {inc.get('verification')}"[:300])
    for cp in cps:
        if (cp.get("failure") or {}).get("category") == "certificate":
            run_infra.append(f"evaluator_arm_certificate_rejected: {cp.get('reason')}")
    sb = r.get("solver_bounds") or {}
    report = {"solver_bound_T": sb.get("final"), "solver_bound_label": "internal master bound (solver-reported, never scored)",
              "resource_outcome": run.get("resource_outcome"), "killed_at_T": run.get("killed_at_T"),
              "peak_rss_mb": run.get("child_peak_rss_mb"), "family_agrees": cert.get("family_agrees")}
    if run_infra:   # publications may be missing: primal AND bound metrics are undefined
        return report | {"best": None, "gap_best": None, "near_best": None, "t_best": None, "primal_undefined": True,
                         "infrastructure": infra + run_infra, "t_bound": None, "bound_target_T": None, "bgap_LP_T": None,
                         "bgap_LP_T_exact": None, "cert_ok_T": None, "final_label": "infrastructure", "export_loss_T": None}
    best = inc.get("best")
    gap = None
    if best is not None and z_best is not None:
        gap = (Fraction(best) - Fraction(z_best)) / max(Fraction(1), abs(Fraction(z_best)))
    # a resolved run (normal end, deadline kill at T, kernel-confirmed OOM) without any incumbent is not near z_best
    # (Codex review 56 #2): near_best False, never undefined; only infrastructure (above) leaves it undefined
    near = False if best is None else (None if gap is None else gap <= Fraction(str(eps_p)))
    out = report | {"best": best, "gap_best": None if gap is None else float(gap), "gap_best_exact": None if gap is None else str(gap),
                    "near_best": near, "no_incumbent": best is None, "t_best": inc.get("t_best"),
                    "primal_undefined": False, "infrastructure": infra}
    if infra:
        return out | {"t_bound": None, "bound_target_T": None, "bgap_LP_T": None, "bgap_LP_T_exact": None, "cert_ok_T": None,
                      "final_label": "infrastructure", "export_loss_T": None}
    hits = [cp["checkpoint"] for cp in cps if cp.get("cert_ok") and tgt is not None and Fraction(cp["lb"]) >= tgt]
    final = cps[-1] if cps else None
    hi = Fraction(iv["hi"]) if iv.get("hi") else None
    bgap = (hi - Fraction(final["lb"])) / max(abs(hi), Fraction(1)) if (final and final.get("lb") is not None and hi is not None) else None
    lb = Fraction(final["lb"]) if final and final.get("lb") is not None else None
    return out | {"t_bound": min(hits) if hits else None,
                  "bound_target_T": bool(final and final.get("cert_ok") and tgt is not None and Fraction(final["lb"]) >= tgt),
                  "bgap_LP_T": None if bgap is None else float(bgap), "bgap_LP_T_exact": None if bgap is None else str(bgap),
                  "cert_ok_T": final.get("cert_ok") if final else None, "final_label": final_label(final, tgt),
                  "checkpoint_labels": [final_label(cp, tgt) for cp in cps],
                  "export_loss_T": (float(Fraction(sb["final"]) - lb) if sb.get("final") is not None and lb is not None else None)}


DECISION_ARMS = ["eng", "naive_lp", "naive_int", *BOUND_CONTROLS]


def _bgap(a: dict) -> Fraction | None:
    if a.get("bgap_LP_T_exact") is not None:
        return Fraction(a["bgap_LP_T_exact"])
    return None if a.get("bgap_LP_T") is None else Fraction(a["bgap_LP_T"])


def primal_naive_worse(eng: dict, nint: dict, rule: float, z_best: float | None) -> bool:
    """§6 primal implementation separation: the naive objective exceeds the engineered one by at least ceil(rule *
    z_best) units for integer objectives (exact integers), else by rule * max(1, |z_best|) exactly. A resolved naive
    integer run without any incumbent (accept() has excluded infrastructure) is worse than an engineered verified
    incumbent (Codex review 56 #2)."""
    if eng.get("best") is not None and "best" in nint and nint["best"] is None:
        return True
    if eng.get("best") is not None and nint.get("best") is not None and z_best is not None:
        e, n, z = Fraction(eng["best"]), Fraction(nint["best"]), Fraction(z_best)
        if e.denominator == n.denominator == z.denominator == 1:
            return n - e >= math.ceil(Fraction(str(rule)) * z)
        return n - e >= Fraction(str(rule)) * max(Fraction(1), abs(z))
    if eng.get("gap_best") is None or nint.get("gap_best") is None:
        return False
    return Fraction(nint["gap_best"]) - Fraction(eng["gap_best"]) >= Fraction(str(rule))


def accept(per: dict, headline: str, rules: dict, T: float, z_best: float | None = None) -> dict:
    """Frozen acceptance for one instance (protocol v4 §6 with Codex reviews 46 and 55). An arm needed for the
    decision with an infrastructure marker makes the decision undefined ("pass": None, rerun), never a beaten control.
      1. naive separates: naive bgap_LP - engineered bgap_LP >= 0.5 point (exact), or, with a naive arm that never
         reaches the target censored at T, t_naive >= 3 t_eng AND t_naive - t_eng >= 0.25 T;
      2. controls: Tier B (time_to_target) each control misses the target or is >= 2x slower with >= 225 s saved, AND
         the engineered arm also meets the final target (rules["tier_B_requires_final_target"], frozen: yes); Tier C
         (final_target) the engineered arm meets the final target and every control misses it;
      3. primal: engineered near_best, the primal control (ils) misses it, naive integer Benders worse by
         ceil(0.005 z_best) units. A resolved arm without any incumbent is not near_best (an incumbent-less ILS misses
         it; an incumbent-less naive integer Benders is worse than an engineered verified incumbent)."""
    eng, naive = per.get("eng"), per.get("naive_lp")
    why = {}
    missing = [a for a in DECISION_ARMS if a not in per]
    if missing:   # an arm without a result (crash, node failure) is a rerun, never a failure or a beaten control
        return {"pass": None, "undefined": "missing arm results: rerun", "missing": missing}
    infra = {a: per[a]["infrastructure"] for a in DECISION_ARMS if a in per and per[a].get("infrastructure")}
    if infra:
        return {"pass": None, "undefined": "infrastructure: rerun the listed arms", "infrastructure": infra}
    gap_rule = Fraction(str(rules["naive_quality_gap"]))
    ne, nn = _bgap(eng), _bgap(naive)
    q_branch = ne is not None and nn is not None and nn - ne >= gap_rule
    t_eng = eng["t_bound"]
    t_naive = naive["t_bound"] if naive["t_bound"] is not None else T     # censored at T (never reached)
    s_branch = (t_eng is not None and t_naive >= rules["naive_speed_ratio"] * t_eng
                and t_naive - t_eng >= rules["saved_fraction"] * T)
    why["naive_separates"] = bool(q_branch or s_branch)
    why["naive_separation_branch"] = "quality" if q_branch else ("speed (censored at T)" if s_branch and naive["t_bound"] is None
                                                                 else "speed" if s_branch else None)
    ctrl = {}
    for a in BOUND_CONTROLS:
        c = per[a]
        if headline == "time_to_target":
            ok = t_eng is not None and (c["t_bound"] is None or (c["t_bound"] >= rules["control_speed_ratio"] * t_eng
                                                                 and c["t_bound"] - t_eng >= rules["control_saved_seconds"]))
        else:
            ok = eng["bound_target_T"] is True and c["bound_target_T"] is False
        ctrl[a] = "separated" if ok else "NOT separated"
    why["bound_controls"] = ctrl
    why["engineered_bound_target"] = eng["bound_target_T"] is True   # required in both tiers (Tier B: frozen rule)
    ils, nint = per["ils"], per["naive_int"]
    why["primal_engineered_near_best"] = eng.get("near_best") is True
    why["primal_ils_misses"] = ils.get("near_best") is False
    why["primal_naive_worse"] = primal_naive_worse(eng, nint, rules["naive_quality_gap"], z_best)
    ok = (why["naive_separates"] and why["engineered_bound_target"] and all(v == "separated" for v in ctrl.values())
          and why["primal_engineered_near_best"] and why["primal_ils_misses"] and why["primal_naive_worse"])
    why["pass"] = bool(ok)
    why["diagnostics"] = {a: {"t_bound": per[a]["t_bound"], "bgap_LP_T": per[a]["bgap_LP_T"], "final_label": per[a].get("final_label"),
                              "solver_bound_T": per[a].get("solver_bound_T"), "export_loss_T": per[a].get("export_loss_T")}
                          for a in DIAGNOSTICS + ["naive_lp", "eng"] if a in per}
    return why


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", nargs=2, metavar=("INSTANCE", "ARM"))
    ap.add_argument("--child", nargs=2, metavar=("INSTANCE", "ARM"), help=argparse.SUPPRESS)
    ap.add_argument("--start", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--reference", metavar="INSTANCE")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--panel", default=str(DEFAULT_PANEL))
    ap.add_argument("--T", type=float, default=900.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dense", action="store_true")
    a = ap.parse_args()
    if a.child:
        child(a.child[0], a.child[1], a.T, a.start if a.start is not None else time.time())
        return
    out = Path(a.out)
    if a.task:
        r = run_task(a.task[0], a.task[1], a.T, out, a.dense, a.panel)
        print("TASK " + json.dumps({"instance": r["instance"], "arm": r["arm"], "run": r["run"],
                                    "best": r["incumbents"]["best"], "checkpoints": [(c.get("checkpoint"), c.get("cert_ok"), c.get("lb_float"), c.get("reason"))
                                                                                    for c in r["certificates"].get("checkpoints", [])],
                                    "family_agrees": r["certificates"].get("family_agrees")}, default=str), flush=True)
    if a.reference:
        r = reference(a.reference, a.T, out)
        print("REFERENCE " + json.dumps(r, default=str), flush=True)
    if a.summarize:
        s = summarize(out, Path(a.panel))
        if s["unexpected"]:
            print("UNEXPECTED (flagged, never read)", json.dumps(s["unexpected"]), flush=True)
        for name, d in s["instances"].items():
            print("SUMMARY", name, d.get("decision"), json.dumps(d.get("acceptance"), default=str), flush=True)
        for tier, d in s["tiers"].items():
            print("TIER", tier, json.dumps({k: d[k] for k in ("stage1", "holdout", "tier")}), flush=True)


if __name__ == "__main__":
    main()
