"""Supervisor publication channel for incumbents and certificate versions (DecompGym-XL protocol v4, Sections 2 and
3.2; v3.1 Section 4.1). Standard library only, so the same file serves as the solver-side client inside the
bubblewrap sandbox (copied there as `decompgym_publish.py`, see `write_client`).

Solver side (agent code):
    from decompgym_publish import publish_incumbent, publish_certificate
    publish_incumbent({"y": [...], "x": ...})          # the family's solution object
    publish_certificate({"view": "ufl-v1", "version": 3, "cuts": [...]})
Each call sends one immutable message and, on a socket, waits for the supervisor's acknowledgement (True: received
before the deadline and stored; False: late, rejected, or no channel). The channel comes from the environment
variable DECOMPGYM_PUBLISH: "unix:<socket path>" or a bare absolute socket path (one stream per process, re-opened
after fork; acknowledged), or "fd:<n>" (an inherited pipe; single writer, no acknowledgement). Without the variable the
calls are no-ops.

Message framing: a 48-byte header (magic b"DGP1", kind byte I/C/B = incumbent/certificate/claim, 3 pad bytes,
payload length as big-endian uint64, SHA-256 of the payload) followed by the payload (UTF-8 JSON). Acknowledgement:
b"A", the supervisor's sequence number (uint64), the first 8 bytes of the payload hash (so a client matches its own
acknowledgement even after unread ones from wait=False calls) and a status byte (E eligible, L late, R rejected).

Supervisor side (`Supervisor`): a thread drains every connection with a selector. A message is timestamped when its
last byte is read, on the supervisor's own monotonic clock relative to the solver start t0 (`start_clock`, called by
the executor right before spawning the solver). It is eligible iff it is complete, its hash matches and its receipt
time is strictly before T_f; eligibility is frozen at T_f (later messages are drained, counted and discarded). A
connection that ends inside a frame is a torn message (ignored); a bad magic or kind closes the stream (corrupt);
an empty payload is rejected without storage; a declared length above the per-kind cap, or one that does not fit the
remaining per-kind volume, message-count or traffic caps (capacity is reserved when a header is accepted, so
concurrent streams cannot overrun a cap), closes the stream. Eligible payloads are written to `store_dir` (outside
the solver's writable area) as read-only files named by sequence number; the index (`index.jsonl`, one line per
message, at most PublishLimits.max_events lines) records every message. Claims are parsed with `parse_claim`
(duplicate keys, non-finite or non-numeric bounds and deep nesting are recorded as malformed claims, never parsed
loosely). An exception while serving one stream is recorded in record["errors"] (an infrastructure error for
scoring) and the other streams keep being drained.

Certificate checkpoints (protocol 3.3.6): at the frozen schedule (default T_f/4, T_f/2, 3T_f/4, T_f) the version
current at a checkpoint c is the last eligible certificate received strictly before c; the T_f checkpoint is the
final version. Superseded versions that are current at no checkpoint are deleted unless `keep_all_certificates`
(their index lines and hashes stay), so storage is bounded by the number of checkpoints.

Executor interface (src/experiments/xl_executor.py binds `sup.socket_path` read-only at /run/decompgym/publish.sock,
sets DECOMPGYM_PUBLISH to that path and calls on_start(t0) with its monotonic spawn time):
    sup = Supervisor(store_dir, T_f).start()
    executor.run(..., publish_socket=sup.socket_path, on_start=sup.start_clock)
    record = sup.stop()                                   # after the process tree is gone; also store_dir/record.json
Generic use: env = sup.env(path_in_sandbox=...); sup.start_clock(); Popen(cmd, env=..., pass_fds=sup.pass_fds);
sup.after_spawn(). The solver imports the client from a copy made by `write_client(dir)` (dir on its PYTHONPATH).
"""
from __future__ import annotations

import hashlib
import json
import os
import selectors
import socket
import stat
import struct
import sys
import tempfile
import threading
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

ENV = "DECOMPGYM_PUBLISH"
MAGIC = b"DGP1"
HEADER = struct.Struct(">4sc3xQ32s")
ACK = struct.Struct(">cQ8sc")
KINDS = {b"I": "incumbent", b"C": "certificate", b"B": "claim"}
KIND_BYTES = {v: k for k, v in KINDS.items()}
CLIENT_MODULE = "decompgym_publish"


# ---------------------------------------------------------------- solver-side client

_client_lock = threading.Lock()
_client: dict[str, Any] = {"pid": None, "sock": None, "addr": None}


def _default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):  # numpy arrays and scalars
        return obj.tolist()
    if type(obj).__name__ == "Decimal":
        return str(obj)
    if type(obj).__name__ == "Fraction":
        return float(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def encode(obj: Any) -> bytes:
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return bytes(obj)
    if isinstance(obj, str):
        return obj.encode("utf-8")
    return json.dumps(obj, separators=(",", ":"), default=_default).encode("utf-8")


def frame(kind: str, payload: bytes) -> bytes:
    return HEADER.pack(MAGIC, KIND_BYTES[kind], len(payload), hashlib.sha256(payload).digest())


def _connect(addr: str) -> Any:
    if addr.startswith("unix:"):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(addr[5:])
        return s
    if addr.startswith("fd:"):
        return int(addr[3:])
    if addr.startswith("/"):  # a bare socket path (as set by xl_executor)
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(addr)
        return s
    raise ValueError(f"unknown {ENV} address {addr!r}")


def _send(conn: Any, header: bytes, payload: bytes, wait: bool, timeout: float) -> bool:
    if isinstance(conn, int):
        for part in (header, payload):
            view = memoryview(part)
            while view:
                n = os.write(conn, view)
                view = view[n:]
        return True
    conn.settimeout(timeout)
    conn.sendall(header)
    conn.sendall(payload)
    if not wait:
        return True
    want = header[-32:-24]
    while True:
        buf = b""
        while len(buf) < ACK.size:
            chunk = conn.recv(ACK.size - len(buf))
            if not chunk:
                raise ConnectionError("supervisor closed the channel")
            buf += chunk
        tag, _, sha8, status = ACK.unpack(buf)
        if tag != b"A":
            raise ConnectionError("bad acknowledgement")
        if sha8 == want:
            return status == b"E"


def publish(kind: str, obj: Any, *, wait: bool = True, timeout: float = 120.0) -> bool:
    """Send one message; True iff the supervisor acknowledged it as eligible (fd channels: True once written)."""
    addr = os.environ.get(ENV)
    if not addr:
        return False
    try:
        payload = encode(obj)
    except (TypeError, ValueError) as exc:  # (a) agent: its object is not serializable; nothing was published
        print(f"[{CLIENT_MODULE}] {kind} not published: {exc}", file=sys.stderr)
        return False
    header = frame(kind, payload)
    with _client_lock:
        for attempt in (0, 1):
            try:
                if _client["sock"] is None or _client["pid"] != os.getpid() or _client["addr"] != addr:
                    _client.update(sock=_connect(addr), pid=os.getpid(), addr=addr)
                return _send(_client["sock"], header, payload, wait, timeout)
            except (OSError, ConnectionError, ValueError) as exc:
                # client side, inside the agent's process: the call returns False (not acknowledged). A failure on the
                # supervisor side is recorded by the supervisor itself (record["errors"] -> infrastructure).
                if isinstance(_client["sock"], socket.socket) and _client["pid"] == os.getpid():
                    try:
                        _client["sock"].close()
                    except OSError:  # closing a broken client socket: nothing to record
                        pass
                _client.update(sock=None, pid=None, addr=None)
                if attempt or isinstance(exc, ValueError) or addr.startswith("fd:"):
                    print(f"[{CLIENT_MODULE}] {kind} not published: {exc}", file=sys.stderr)
                    return False
    return False


def publish_incumbent(solution: Any, **kw: Any) -> bool:
    """Publish a feasible solution (the family's solution object); the supervisor timestamps and stores it."""
    return publish("incumbent", solution, **kw)


def publish_certificate(certificate: Any, **kw: Any) -> bool:
    """Publish a complete certificate version (each call is one immutable version; the latest one before a
    checkpoint is the one checked)."""
    return publish("certificate", certificate, **kw)


def publish_claim(claim: Any, **kw: Any) -> bool:
    """Publish a bound or status claim, e.g. {"lower_bound": 123.0} (recorded; refuted claims fail the attempt)."""
    return publish("claim", claim, **kw)


def write_client(directory: str | Path) -> Path:
    """Copy this module as `decompgym_publish.py` into a directory the sandbox can import from."""
    out = Path(directory) / f"{CLIENT_MODULE}.py"
    out.write_text(Path(__file__).read_text())
    return out


# ---------------------------------------------------------------- claims (bounded, schema-validated)

CLAIM_NUMBERS = ("lower_bound", "bound", "objective")
CLAIM_MAX_DEPTH = 4      # nesting of arrays/objects, the claim object itself included
CLAIM_MAX_NODES = 256    # structural tokens ([ { , :) outside strings


def json_shape(text: str, max_depth: int, max_nodes: int) -> str | None:
    """Scan decoded JSON text (outside strings) before parsing it: "too_deep" / "too_many_nodes" if the nesting or the
    number of structural tokens exceeds the limits, else None. The same text is then parsed, so the bound holds for
    decoding AND re-encoding."""
    depth = nodes = 0
    in_str = esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "[" or ch == "{":
            depth += 1
            nodes += 1
            if depth > max_depth:
                return "too_deep"
        elif ch == "]" or ch == "}":
            depth -= 1
        elif ch == "," or ch == ":":
            nodes += 1
        if nodes > max_nodes:
            return "too_many_nodes"
    return None


class _ClaimError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _finite_decimal(text: str) -> bool:
    if len(text) > 64:
        return False
    try:
        d = Decimal(text)
    except (InvalidOperation, ValueError):  # (a) agent: a non-numeric claim value -> malformed claim
        return False
    return d.is_finite() and (not d or abs(d.adjusted()) <= 400)


def parse_claim(data: bytes) -> tuple[dict[str, Any] | None, str | None]:
    """A claim is one JSON object in strict UTF-8 without a byte-order mark (other encodings: "not_utf8" / "bom"), at
    most PublishLimits.max_claim_bytes, whose "lower_bound", "bound" and
    "objective" are finite decimal numbers (kept as their token text, so they are compared exactly) and whose
    "status" is a string; other keys are kept. Nesting deeper than CLAIM_MAX_DEPTH or more than CLAIM_MAX_NODES
    structural tokens (checked before decoding), duplicate keys, non-finite values or any other malformation give
    (None, reason): the raw bytes stay stored and the claim counts as a failure of the attempt."""
    def pairs(p: list) -> dict:
        d = dict(p)
        if len(d) != len(p):
            raise _ClaimError("duplicate_key")
        return d

    def constant(tok: str) -> Any:
        raise _ClaimError("non_finite")

    try:
        text = bytes(data).decode("utf-8")   # strict UTF-8 only (json.loads(bytes) would also accept UTF-16/32)
    except UnicodeDecodeError:  # (a) agent: malformed claim
        return None, "not_utf8"
    if text.startswith("\ufeff"):
        return None, "bom"
    shape = json_shape(text, CLAIM_MAX_DEPTH, CLAIM_MAX_NODES)
    if shape is not None:
        return None, shape
    try:
        obj = json.loads(text, parse_float=str, parse_int=str, parse_constant=constant, object_pairs_hook=pairs)
    except _ClaimError as exc:  # (a) agent: malformed claim (duplicate key, non-finite value)
        return None, exc.reason
    except (ValueError, RecursionError):  # (a) agent: malformed claim
        return None, "parse_error"
    if not isinstance(obj, dict):
        return None, "not_an_object"
    for key in CLAIM_NUMBERS:
        v = obj.get(key)
        if v is not None and (not isinstance(v, str) or not _finite_decimal(v)):
            return None, f"bad_{key}"
    if obj.get("status") is not None and not isinstance(obj["status"], str):
        return None, "bad_status"
    return obj, None


# ---------------------------------------------------------------- supervisor

def checkpoint_schedule(T_f: float, k: int = 4) -> list[float]:
    """Frozen checkpoint times T_f/k, 2T_f/k, ..., T_f (the last one is the final version)."""
    return [T_f * i / k for i in range(1, k + 1)]


class PublishLimits:
    """Per-message and per-attempt caps. Capacity is reserved when a frame header is accepted (so concurrent streams
    cannot exceed a cap), and every frame (any status) counts against `max_events`, every payload byte read (any
    status) against `max_traffic_bytes`. The total certificate volume is the protocol's publication cap."""

    def __init__(self, *, max_incumbent_bytes: int = 64 << 20, max_certificate_bytes: int = 256 << 20,
                 max_claim_bytes: int = 64 << 10, max_total_incumbent_bytes: int = 2 << 30,
                 max_total_certificate_bytes: int = 4 << 30, max_total_claim_bytes: int = 16 << 20,
                 max_incumbents: int = 100_000, max_certificates: int = 10_000, max_claims: int = 1_000,
                 max_events: int = 1_000_000, max_traffic_bytes: int = 16 << 30, max_connections: int = 64) -> None:
        self.max_bytes = {"incumbent": max_incumbent_bytes, "certificate": max_certificate_bytes, "claim": max_claim_bytes}
        self.max_total = {"incumbent": max_total_incumbent_bytes, "certificate": max_total_certificate_bytes,
                          "claim": max_total_claim_bytes}
        self.max_messages = {"incumbent": max_incumbents, "certificate": max_certificates, "claim": max_claims}
        self.max_events = max_events
        self.max_traffic = max_traffic_bytes
        self.max_connections = max_connections

    def as_dict(self) -> dict[str, Any]:
        return {"max_bytes": self.max_bytes, "max_total": self.max_total, "max_messages": self.max_messages,
                "max_events": self.max_events, "max_traffic_bytes": self.max_traffic, "max_connections": self.max_connections}


class _Conn:
    __slots__ = ("sock", "fd", "cid", "hdr", "kind", "length", "sha", "got", "hasher", "file", "tmp", "drop", "dead", "reserved",
                 "slot")

    def __init__(self, sock: socket.socket | None, fd: int | None, cid: int) -> None:
        self.sock, self.fd, self.cid = sock, fd, cid
        self.dead = False
        self.reset()

    def reset(self) -> None:
        self.hdr = bytearray()
        self.kind = self.length = self.sha = self.hasher = self.file = self.tmp = None
        self.got = 0
        self.drop = None       # reason a frame in progress is discarded (late)
        self.reserved = False  # capacity reserved for this frame
        self.slot = False      # an index line reserved for this frame

    @property
    def partial(self) -> bool:
        return bool(self.hdr) or self.length is not None


def _short_tmpdir() -> str:
    base = tempfile.gettempdir()
    if len(base) > 60:  # AF_UNIX paths are limited to ~107 bytes
        base = "/tmp"
    return tempfile.mkdtemp(prefix="dgp-", dir=base)


STATUSES = ("eligible", "late", "torn", "corrupt", "empty", "oversize", "over_volume", "over_count", "over_traffic",
            "over_events", "refused", "error", "malformed_claim")


class Supervisor:
    """Drains a publication channel on a background thread; see the module docstring."""

    def __init__(self, store_dir: str | Path, T_f: float, *, limits: PublishLimits | None = None,
                 checkpoints: list[float] | None = None, transport: str = "unix", socket_dir: str | Path | None = None,
                 keep_all_certificates: bool = False, clock: Callable[[], float] = time.monotonic) -> None:
        if transport not in ("unix", "fd"):
            raise ValueError("transport must be 'unix' or 'fd'")
        self.store = Path(store_dir)
        self.store.mkdir(parents=True, exist_ok=True)
        (self.store / "tmp").mkdir(exist_ok=True)
        self.T_f = float(T_f)
        self.limits = limits or PublishLimits()
        self.checkpoints = sorted(checkpoints) if checkpoints is not None else checkpoint_schedule(self.T_f)
        if not self.checkpoints or self.checkpoints[-1] != self.T_f:
            raise ValueError("the last checkpoint must be T_f (the final version)")
        self.transport = transport
        self.keep_all = keep_all_certificates
        self.clock = clock
        self.t0: float | None = None
        self.socket_dir = Path(socket_dir) if socket_dir is not None else None
        self.socket_path: str | None = None
        self._own_socket_dir = False
        self._listener: socket.socket | None = None
        self._rfd = self._wfd = None
        self._sel = selectors.DefaultSelector()
        self._conns: dict[int, _Conn] = {}
        self._next_cid = 0
        self._seq = 0
        self._stop = threading.Event()
        self._stop_deadline = 0.0
        self._thread: threading.Thread | None = None
        self._index = (self.store / "index.jsonl").open("a")
        self.counts = {k: 0 for k in STATUSES}
        self.volume = {k: 0 for k in KIND_BYTES}          # stored (eligible) payload bytes
        self.reserved = {k: 0 for k in KIND_BYTES}        # bytes of accepted frames still in flight
        self.messages = {k: 0 for k in KIND_BYTES}        # eligible messages
        self.inflight = {k: 0 for k in KIND_BYTES}
        self.traffic = 0                                  # payload bytes admitted for reading (any status)
        self.events = 0                                   # index lines written
        self.event_slots = 0                              # index lines reserved by accepted frames in flight
        self.index_dropped = 0                            # rejection lines not written (metadata cap)
        self.incumbents: list[dict[str, Any]] = []
        self.certificates: list[dict[str, Any]] = []
        self.claims: list[dict[str, Any]] = []
        self.record: dict[str, Any] | None = None
        self.errors: list[str] = []      # infrastructure: makes every target of the attempt undefined (rerun)
        self.warnings: list[str] = []    # cleanup of supervisor-owned temporaries: recorded, affects no recorded value

    # ------------------------------------------------------------ lifecycle

    def start(self) -> "Supervisor":
        if self.transport == "unix":
            if self.socket_dir is None:
                self.socket_dir = Path(_short_tmpdir())
                self._own_socket_dir = True
            self.socket_dir.mkdir(parents=True, exist_ok=True)
            self.socket_path = str(self.socket_dir / "pub.sock")
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            lst = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            lst.bind(self.socket_path)
            lst.listen(64)
            lst.setblocking(False)
            self._listener = lst
            self._sel.register(lst, selectors.EVENT_READ, None)
        else:
            self._rfd, self._wfd = os.pipe()
            os.set_blocking(self._rfd, False)
            os.set_inheritable(self._wfd, True)
            c = _Conn(None, self._rfd, self._new_cid())
            self._conns[c.cid] = c
            self._sel.register(self._rfd, selectors.EVENT_READ, c)
        if self.t0 is None:
            self.t0 = self.clock()
        self._thread = threading.Thread(target=self._run, name="decompgym-supervisor", daemon=True)
        self._thread.start()
        return self

    def start_clock(self, t0: float | None = None) -> None:
        """Set the solver start on the supervisor's clock (call right before spawning the solver)."""
        self.t0 = self.clock() if t0 is None else t0

    def elapsed(self) -> float:
        return self.clock() - (self.t0 if self.t0 is not None else self.clock())

    def env(self, path_in_sandbox: str | None = None) -> dict[str, str]:
        if self.transport == "unix":
            return {ENV: "unix:" + (path_in_sandbox or self.socket_path or "")}
        return {ENV: f"fd:{self._wfd}"}

    @property
    def pass_fds(self) -> tuple[int, ...]:
        return (self._wfd,) if self.transport == "fd" and self._wfd is not None else ()

    def after_spawn(self) -> None:
        """Close the parent's copy of the pipe's write end (fd transport), so EOF arrives when the solver exits."""
        if self._wfd is not None:
            os.close(self._wfd)
            self._wfd = None

    def stop(self, grace: float = 1.0) -> dict[str, Any]:
        """Stop accepting, drain open streams for up to `grace` seconds, finalize checkpoints and ALWAYS write
        record.json: a failure in any step is recorded in record["errors"] (an infrastructure error for scoring), and
        if the full record cannot be serialized a reduced one (no claim payloads) is written instead."""
        if self.record is not None:
            return self.record

        def guard(step: str, fn: Callable[[], Any]) -> Any:
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - (b) infrastructure, recorded; finalization continues
                self.errors.append(f"stop/{step}: {type(exc).__name__}: {exc}"[:300])
                return None

        self._stop_deadline = time.monotonic() + grace
        self._stop.set()
        if self._thread is not None:
            self._thread.join(grace + 30.0)
            if self._thread.is_alive():
                self.errors.append("supervisor thread did not stop")
        guard("after_spawn", self.after_spawn)
        for c in list(self._conns.values()):
            if c.partial:
                guard("torn", lambda c=c: self._log(c, "torn", None, c.got))
            guard("close", lambda c=c: self._close(c))
        guard("selector", self._sel.close)
        if self.socket_path:
            guard("socket", lambda: os.path.exists(self.socket_path) and os.unlink(self.socket_path))
        if self._own_socket_dir and self.socket_dir is not None:
            try:
                os.rmdir(self.socket_dir)
            except OSError as exc:  # cleanup of the supervisor's own socket directory: warning only
                self.warnings.append(f"socket dir: {exc}"[:300])
        # the index is completed BEFORE the record is serialized, so its failures are in the record (infrastructure)
        guard("index", lambda: (self._index.flush(), os.fsync(self._index.fileno()), self._index.close()))
        guard("chmod_index", lambda: os.chmod(self.store / "index.jsonl", stat.S_IRUSR | stat.S_IRGRP))
        try:
            os.rmdir(self.store / "tmp")
        except OSError as exc:  # leftover temporary files of discarded frames: warning only
            self.warnings.append(f"tmp dir: {exc}"[:300])
        text = None
        for attempt in ("full", "reduced", "minimal"):
            try:
                record = self._finalize() if attempt != "minimal" else self._minimal()
                if attempt == "reduced":
                    for e in record["claims"]:
                        e["payload"] = None
                text = json.dumps(record, indent=1, allow_nan=False)
                self.record = record
                break
            except Exception as exc:  # noqa: BLE001 - (b) recorded in errors; a reduced record is tried next
                self.errors.append(f"stop/serialize_{attempt}: {type(exc).__name__}: {exc}"[:300])
        if text is None:  # pragma: no cover - the minimal record has only plain numbers and strings
            self.record = {"T_f": self.T_f}
        out = self.store / "record.json"
        tmp = self.store / "record.json.tmp"
        # write, chmod; if writing or chmod fails, the failure is appended and the record rewritten (at most twice),
        # so the returned AND the persisted record carry every finalization error
        for _ in range(3):
            n_errors = len(self.errors)
            self.record["errors"] = list(self.errors)
            self.record["error"] = self.errors[0] if self.errors else None
            self.record["warnings"] = list(self.warnings)
            guard("write_record", lambda: (tmp.write_text(json.dumps(self.record, indent=1, allow_nan=False, default=str)),
                                           os.replace(tmp, out)))
            guard("chmod_record", lambda: os.chmod(out, stat.S_IRUSR | stat.S_IRGRP))
            if len(self.errors) == n_errors:
                break
        self.record["errors"] = list(self.errors)
        self.record["error"] = self.errors[0] if self.errors else None
        return self.record

    def _minimal(self) -> dict[str, Any]:
        strip = lambda es: [{k: e.get(k) for k in ("seq", "t", "bytes", "sha256", "file")} for e in es]
        rec = self._finalize()
        rec.update(incumbents=strip(self.incumbents), certificates=strip(self.certificates),
                   claims=[dict(e, payload=None, error=e.get("error") or "unserializable") for e in strip(self.claims)])
        return rec

    # ------------------------------------------------------------ event loop

    def _new_cid(self) -> int:
        self._next_cid += 1
        return self._next_cid

    def _run(self) -> None:
        """Event loop. An exception while serving one stream is recorded (record["errors"]: an infrastructure error
        for scoring), that stream is closed, and draining continues."""
        while True:
            try:
                stopping = self._stop.is_set()
                if stopping and self._listener is not None:
                    self._sel.unregister(self._listener)
                    self._listener.close()
                    self._listener = None
                events = self._sel.select(timeout=0.01 if stopping else 0.05)
            except Exception as exc:  # noqa: BLE001 - (b) the selector failed: recorded, nothing more can be drained
                self.errors.append(f"selector: {type(exc).__name__}: {exc}")
                return
            for key, _ in events:
                c = key.data
                try:
                    if c is None:
                        self._accept()
                    else:
                        self._read(c)
                except Exception as exc:  # noqa: BLE001 - (b) recorded; keep draining the other streams
                    self.errors.append(f"stream {getattr(c, 'cid', 'listener')}: {type(exc).__name__}: {exc}")
                    if c is not None:
                        try:
                            if c.partial:
                                self._log(c, "error", None, c.got)
                            self._close(c)
                        except Exception as exc2:  # noqa: BLE001 - (b) recorded
                            self.errors.append(f"close: {type(exc2).__name__}: {exc2}")
            if stopping:
                # drain what is buffered; stop when idle (no partial frame, nothing to read) or at the grace deadline
                if (not events and not any(c.partial for c in self._conns.values())) or time.monotonic() > self._stop_deadline:
                    return

    def _accept(self) -> None:
        try:
            sock, _ = self._listener.accept()
        except (BlockingIOError, InterruptedError):  # benign: no pending connection after all
            return
        if len(self._conns) >= self.limits.max_connections:
            self.counts["refused"] += 1
            sock.close()
            return
        sock.setblocking(False)
        c = _Conn(sock, None, self._new_cid())
        self._conns[c.cid] = c
        self._sel.register(sock, selectors.EVENT_READ, c)

    def _release(self, c: _Conn) -> None:
        if c.reserved:
            self.reserved[c.kind] -= c.length
            self.inflight[c.kind] -= 1
            c.reserved = False

    def _close(self, c: _Conn) -> None:
        if c.dead:
            return
        c.dead = True
        self._release(c)
        if c.slot:
            c.slot = False
            self.event_slots -= 1
        if c.file is not None:
            c.file.close()
            c.file = None
            try:
                os.unlink(c.tmp)
            except OSError as exc:  # a discarded frame's temporary file: warning only (never part of the record)
                self.warnings.append(f"stream {c.cid} tmp: {exc}"[:300])
        try:
            self._sel.unregister(c.sock if c.sock is not None else c.fd)
        except (KeyError, ValueError, OSError):  # benign: already unregistered or the descriptor is gone
            pass
        try:
            if c.sock is not None:
                c.sock.close()
            elif c.fd is not None:
                os.close(c.fd)
        except OSError as exc:  # closing a finished stream: warning only
            self.warnings.append(f"stream {c.cid} close: {exc}"[:300])
        self._conns.pop(c.cid, None)

    def _read(self, c: _Conn) -> None:
        try:
            data = c.sock.recv(1 << 20) if c.sock is not None else os.read(c.fd, 1 << 20)
        except (BlockingIOError, InterruptedError):  # benign: nothing to read yet
            return
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            # (a) the solver side disconnected (its process closed or died): end of stream; a partial frame is torn
            data = b""
        except OSError as exc:
            # (b) any other transport error (EIO, ...): infrastructure, recorded; the stream is closed
            self.errors.append(f"stream {c.cid}: transport error: {type(exc).__name__}: {exc}"[:300])
            if c.partial:
                self._log(c, "error", None, c.got)
            self._close(c)
            return
        if not data:
            if c.partial:
                self._log(c, "torn", None, c.got)
            self._close(c)
            return
        view = memoryview(data)
        off, n = 0, len(data)
        while off < n and not c.dead:
            if c.length is None:
                need = HEADER.size - len(c.hdr)
                c.hdr += view[off:off + need]
                off += min(need, n - off)
                if len(c.hdr) == HEADER.size:
                    self._begin(c)
            else:
                chunk = view[off:off + (c.length - c.got)]
                off += len(chunk)
                c.got += len(chunk)
                c.hasher.update(chunk)
                if c.file is not None:
                    c.file.write(chunk)
                if c.got == c.length:
                    self._complete(c)

    def _ack(self, c: _Conn, seq: int, sha: bytes, code: bytes) -> None:
        if c.sock is not None:
            try:
                c.sock.send(ACK.pack(b"A", seq, sha[:8], code))
            except OSError:  # the client went away: its message is already recorded; the client sees no ack
                pass

    def _begin(self, c: _Conn) -> None:
        if self.events + self.event_slots >= self.limits.max_events:
            self.counts["over_events"] += 1  # metadata cap: counted, not indexed; the stream is closed
            self._close(c)
            return
        magic, kb, length, sha = HEADER.unpack(bytes(c.hdr))
        kind = KINDS.get(kb)
        if magic != MAGIC or kind is None:
            self._log(c, "corrupt", None, 0)
            self._close(c)
            return
        if length == 0:  # never a valid payload: no file, one index line, the stream continues
            c.kind = kind
            self._log(c, "empty", kind, 0)
            self._ack(c, 0, sha, b"R")
            c.reset()
            return
        if length > self.limits.max_bytes[kind]:
            self._log(c, "oversize", kind, length)
            self._close(c)
            return
        if self.traffic + length > self.limits.max_traffic:
            self._log(c, "over_traffic", kind, length)
            self._close(c)
            return
        late = self.elapsed() >= self.T_f
        if not late:
            if self.volume[kind] + self.reserved[kind] + length > self.limits.max_total[kind]:
                self._log(c, "over_volume", kind, length)
                self._close(c)
                return
            if self.messages[kind] + self.inflight[kind] >= self.limits.max_messages[kind]:
                self._log(c, "over_count", kind, length)
                self._close(c)
                return
        self.traffic += length
        self.event_slots += 1  # every accepted frame is guaranteed its index line (completion, late, torn)
        c.slot = True
        c.kind, c.length, c.sha, c.got = kind, length, sha, 0
        c.hasher = hashlib.sha256()
        if late:
            c.drop = "late"  # cannot become eligible: drained without storage
        else:
            self.reserved[kind] += length
            self.inflight[kind] += 1
            c.reserved = True
            fd, tmp = tempfile.mkstemp(dir=self.store / "tmp")
            c.file, c.tmp = os.fdopen(fd, "wb"), tmp

    def _complete(self, c: _Conn) -> None:
        t = self.elapsed()  # complete receipt
        self._release(c)
        self._seq += 1
        seq = self._seq
        kind, length = c.kind, c.length
        digest = c.hasher.digest()
        status, fname = "eligible", None
        if digest != c.sha:
            status = "corrupt"
        elif c.drop is not None or t >= self.T_f:
            status = "late"
        if c.file is not None:
            c.file.close()
            c.file = None
            if status == "eligible":
                fname = f"{kind[0].upper()}{seq:07d}.json"
                os.replace(c.tmp, self.store / fname)
                os.chmod(self.store / fname, stat.S_IRUSR | stat.S_IRGRP)
            else:
                os.unlink(c.tmp)
        sha_hex = digest.hex()
        code = {"eligible": b"E", "late": b"L"}.get(status, b"R")
        claim_error = None
        if status == "eligible":
            self.volume[kind] += length
            self.messages[kind] += 1
            entry = {"seq": seq, "t": t, "bytes": length, "sha256": sha_hex, "file": fname}
            if kind == "incumbent":
                self.incumbents.append(entry)
            elif kind == "certificate":
                self._new_certificate(entry)
            else:
                payload, claim_error = parse_claim((self.store / fname).read_bytes())
                entry.update(payload=payload, error=claim_error)
                self.claims.append(entry)
                if claim_error:
                    self.counts["malformed_claim"] += 1
                    code = b"R"  # stored and recorded as a failed claim
        self.counts[status] += 1
        self._index_line({"seq": seq, "conn": c.cid, "kind": kind, "t": t, "bytes": length, "sha256": sha_hex,
                          "status": status, "file": fname, "claim_error": claim_error}, c)
        self._ack(c, seq, c.sha, code)
        c.reset()

    def _index_line(self, rec: dict[str, Any], c: _Conn | None = None) -> None:
        """Write one index line: with the frame's reserved slot if it has one (always written), else only while the
        metadata cap leaves an unreserved slot (otherwise counted in index_dropped)."""
        if c is not None and c.slot:
            c.slot = False
            self.event_slots -= 1
        elif self.events + self.event_slots >= self.limits.max_events:
            self.index_dropped += 1
            return
        self.events += 1
        self._index.write(json.dumps(rec) + "\n")
        self._index.flush()

    def _log(self, c: _Conn, status: str, kind: str | None, length: int) -> None:
        self.counts[status] += 1
        self._index_line({"seq": None, "conn": c.cid, "kind": kind or c.kind, "t": self.elapsed(), "bytes": length,
                          "sha256": None, "status": status, "file": None}, c)

    def _new_certificate(self, entry: dict[str, Any]) -> None:
        if self.certificates:
            prev = self.certificates[-1]
            current_at = [cp for cp in self.checkpoints if prev["t"] < cp <= entry["t"]]
            prev["retained"] = bool(current_at)
            if not current_at and not self.keep_all and prev["file"]:
                try:
                    os.unlink(self.store / prev["file"])
                except OSError as exc:  # a version current at no checkpoint is never checked: warning only
                    self.warnings.append(f"prune {prev['file']}: {exc}"[:300])
                prev["file"] = None
        entry["retained"] = True
        self.certificates.append(entry)

    def _finalize(self) -> dict[str, Any]:
        cps = []
        for cp in self.checkpoints:
            cur = None
            for v in self.certificates:
                if v["t"] < cp:
                    cur = v
            cps.append({"checkpoint": cp} | ({k: cur[k] for k in ("seq", "t", "sha256", "file", "bytes")} if cur else
                                              {"seq": None, "t": None, "sha256": None, "file": None, "bytes": None}))
        return {"T_f": self.T_f, "transport": self.transport, "store_dir": str(self.store), "checkpoint_schedule": self.checkpoints,
                "limits": self.limits.as_dict(), "counts": dict(self.counts), "volume": dict(self.volume),
                "messages": dict(self.messages), "traffic_bytes": self.traffic, "index_lines": self.events,
                "index_dropped": self.index_dropped,
                "incumbents": self.incumbents, "certificates": self.certificates, "claims": self.claims,
                "checkpoints": cps, "errors": list(self.errors), "error": self.errors[0] if self.errors else None}


def load_record(store_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(store_dir) / "record.json").read_text())


def read_payload(store_dir: str | Path, entry: dict[str, Any]) -> bytes:
    """The stored payload of a record entry, checked against its hash."""
    data = (Path(store_dir) / entry["file"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"stored payload {entry['file']} does not match its hash")
    return data
