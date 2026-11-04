"""Integration tests of the XL bubblewrap executor (Linux Slurm job with bwrap and one CPU: Phoenix compute nodes; skipped
elsewhere). Real sandboxed programs, adversarial where Codex review 44 asked: PID-1 signals and prlimit, limit preflight and
affinity widening, track policy and source audit, forwarder filter/budgets/shutdown, scratch on /dev/shm, kernel-confirmed OOM,
feedback frozen at T_f. The full measured battery is `scripts/phoenix/xl_executor_validate.py`.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import tempfile
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(platform.system() != "Linux" or not Path("/usr/bin/bwrap").exists() or "SLURM_JOB_ID" not in os.environ,
                                reason="needs a Linux Slurm job with bubblewrap (Phoenix compute node)")

from src.experiments.xl_build import BuildLimits, manifest_for  # noqa: E402
from src.experiments.xl_executor import GiB, MiB, LicenseForwarder, RunLimits, XLSandboxExecutor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PDPTW = ROOT / "outputs/scaleup/raw/cvrp/Instances-main/PDPTW"


@pytest.fixture(scope="module")
def executor(tmp_path_factory):
    return XLSandboxExecutor(base_dir=tmp_path_factory.mktemp("xl"))


@pytest.fixture
def data(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"n": 3}))
    return path


def run(executor, data, solver, limits=None, cpp=None, track=None, **kwargs):
    files = {"solver.py": solver.encode(), **{"cpp/" + name: text.encode() for name, text in (cpp or {}).items()}}
    track = track or ("cpp_allowed" if cpp else "python_only")
    return executor.execute(manifest_for(files, track=track), data, limits or RunLimits(wall_seconds=60), track=track, **kwargs)


def last_json(record):
    lines = [line for line in ((record["run"] or {}).get("stdout") or "").splitlines() if line.startswith("{")]
    return json.loads(lines[-1]) if lines else None


def test_isolation_minimal_runtime_and_read_only_inputs(executor, data):
    solver = r'''
import json, os, socket, sys
out = {"home": os.path.exists(HOME), "cwd": os.getcwd(), "data": json.load(open("data.json")), "usr": sorted(os.listdir("/usr")),
       "bin_sh": os.path.exists("/bin/sh"), "runtime": sys.version.split()[0]}
for label, path in (("data_write", "data.json"), ("solver", "solver.py")):
    try:
        open(path, "a"); out[label] = "writable"
    except OSError as e:
        out[label] = e.errno
try:
    socket.create_connection(("1.1.1.1", 443), timeout=3); out["net"] = "open"
except OSError as e:
    out["net"] = e.errno
print(json.dumps(out))
'''.replace("HOME", repr(os.path.expanduser("~")))
    record = run(executor, data, solver)
    out = last_json(record)
    assert record["termination"] == "finished" and record["run"]["cleanup_verified"]
    assert out["data"] == {"n": 3} and out["data_write"] == 30 and out["solver"] in (13, 30) and out["net"] == 101
    assert out["home"] is False and out["cwd"] == "/work" and out["usr"] == ["lib", "lib64"] and out["bin_sh"] is False
    assert record["run"]["runtime"]["python"] == out["runtime"] and record["run"]["job_limits"]["memory_max"] <= 16 * GiB


def test_pid1_survives_every_signal_and_prlimit(executor, data):
    solver = r'''
import errno, json, os, resource, signal, time
for number in range(1, 65):
    if number not in (32, 33):
        try:
            os.kill(1, number)
        except OSError:
            pass
time.sleep(0.5)
import ctypes
libc = ctypes.CDLL(None, use_errno=True)
limit = (ctypes.c_ulong * 2)(1, 1)  # RLIMIT_CPU = 1 s on PID 1 (raw prlimit64: this Python's resource module has no prlimit)
rc = libc.syscall(ctypes.c_long(302), ctypes.c_long(1), ctypes.c_long(0), ctypes.c_void_p(ctypes.addressof(limit)), ctypes.c_long(0))
print(json.dumps({"prlimit": "changed" if rc == 0 else errno.errorcode[ctypes.get_errno()]}))
'''
    record = run(executor, data, solver)
    assert record["termination"] == "finished" and last_json(record) == {"prlimit": "EPERM"}
    assert any(event.get("event") == "payload_exit" for event in record["run"]["events"])
    assert record["run"]["sandbox"]["init_handlers_reset"] == [2]  # SIGINT's Python handler is gone from PID 1


def test_seccomp_denies_keyring_ptrace_namespaces(executor, data):
    solver = r'''
import ctypes, errno, json, threading
libc = ctypes.CDLL(None, use_errno=True)
def call(n, *a):
    r = libc.syscall(ctypes.c_long(n), *[ctypes.c_long(x) for x in a]); return r if r != -1 else errno.errorcode[ctypes.get_errno()]
seen = []
t = threading.Thread(target=lambda: seen.append(1)); t.start(); t.join()
print(json.dumps({"keyctl": call(250, 0, -4, 0), "ptrace": call(101, 0, 0, 0, 0), "unshare": call(272, 0x10000000), "getppid": call(110), "thread": seen == [1]}))
'''
    out = last_json(run(executor, data, solver))
    assert out["keyctl"] == out["ptrace"] == out["unshare"] == "EPERM" and isinstance(out["getppid"], int) and out["thread"]


def test_affinity_cannot_widen(executor, data):
    solver = "import json, os\nbefore = sorted(os.sched_getaffinity(0))\ntry:\n    os.sched_setaffinity(0, set(range(os.cpu_count())))\nexcept OSError:\n    pass\nprint(json.dumps({'before': before, 'after': sorted(os.sched_getaffinity(0))}))\n"
    out = last_json(run(executor, data, solver))
    assert out["before"] == out["after"] and len(out["after"]) == 1


def test_deadline_kills_the_whole_tree_and_freezes_feedback(executor, data):
    solver = r'''
import os, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, lambda *a: print("AFTER-TF", flush=True))
subprocess.Popen([sys.executable, "-c", "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nwhile True: time.sleep(1)"])
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        while True: time.sleep(1)
    os._exit(0)
print("BEFORE-TF", flush=True)
while True: time.sleep(1)
'''
    record = run(executor, data, solver, RunLimits(wall_seconds=4, grace_seconds=1))
    assert record["termination"] == "deadline" and record["run"]["kill_sent"] is not None and record["run"]["cleanup_verified"]
    assert "BEFORE-TF" in record["feedback"]["stdout"] and "AFTER-TF" not in record["feedback"]["stdout"] and record["run"]["bytes_after_deadline"] > 0
    assert record["run"]["wall_seconds"] < 8 and record["run"]["startup_seconds"] < 2


def test_fork_bomb_is_contained(executor, data):
    solver = "import os, time\nn = 0\nwhile True:\n    try:\n        if os.fork() == 0:\n            time.sleep(100); os._exit(0)\n        n += 1\n    except OSError:\n        print({'forked': n}, flush=True); break\ntime.sleep(100)\n"
    record = run(executor, data, solver, RunLimits(wall_seconds=5, max_processes=32))
    assert record["termination"] == "deadline" and record["run"]["cleanup_verified"] and "forked" in record["feedback"]["stdout"]


def test_memory_error_is_not_a_kernel_oom(executor, data):
    record = run(executor, data, "chunks = []\nwhile True:\n    chunks.append(bytearray(128 * 1024 * 1024))\n", RunLimits(wall_seconds=60, address_space_bytes=1 * GiB))
    assert record["termination"] == "crash" and "no kernel OOM" in record["detail"]
    forged = run(executor, data, "import sys\nsys.stderr.write('MemoryError\\nOut of memory\\n')\nsys.exit(1)\n")
    assert forged["termination"] == "crash" and "no kernel OOM" in forged["detail"]


def test_file_size_limit(executor, data):
    python = "with open('/work/x', 'wb') as f:\n    f.write(b'0' * (40 * 1024 * 1024))\n"
    record = run(executor, data, python, RunLimits(wall_seconds=60, file_size_bytes=16 * MiB))
    assert record["termination"] == "crash" and "file size limit" in record["detail"]  # CPython ignores SIGXFSZ: EFBIG
    native = "import signal\nsignal.signal(signal.SIGXFSZ, signal.SIG_DFL)\n" + python
    record = run(executor, data, native, RunLimits(wall_seconds=60, file_size_bytes=16 * MiB))
    assert record["termination"] == "crash" and "SIGXFSZ" in record["detail"], record["detail"]


def test_dev_shm_counts_toward_scratch(executor, data):
    solver = "import time\nfor i in range(40):\n    open('/dev/shm/c%d' % i, 'wb').write(b'1' * (32 << 20))\n    time.sleep(0.1)\n"
    record = run(executor, data, solver, RunLimits(wall_seconds=60, scratch_quota_bytes=256 * MiB))
    assert record["termination"] == "crash" and "scratch" in record["detail"]


def test_publication_hook_and_client_modules(executor, data, tmp_path):
    path = tmp_path / "pub.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    got = []

    def serve():
        conn, _ = server.accept()
        with conn:
            got.append(conn.recv(1024))
            conn.sendall(b"ok")

    threading.Thread(target=serve, daemon=True).start()
    module = tmp_path / "dg_client.py"
    module.write_text("VALUE = 7\n")
    solver = "import os, socket, dg_client\ns = socket.socket(socket.AF_UNIX)\ns.connect(os.environ['DECOMPGYM_PUBLISH'])\ns.sendall(b'incumbent')\nprint(s.recv(10).decode(), dg_client.VALUE)\n"
    starts = []
    record = run(executor, data, solver, publish_socket=path, on_start=starts.append, client_modules=[module])
    server.close()
    assert record["termination"] == "finished" and record["run"]["stdout"].strip() == "ok 7" and got == [b"incumbent"]
    assert starts == [record["run"]["t0"]]


def test_gurobi_through_the_license_tunnel(executor):
    result = executor.gurobi_preflight()
    assert result["ok"], result
    tunnel = result["license_tunnel"]
    assert tunnel["admitted"] >= 1 and tunnel["filtered"] == 0 and tunnel["shutdown_confirmed"]


def test_status_queries_are_filtered(executor, data):
    solver = r'''
import ctypes, json, os, socket
import gurobipy as gp
s = socket.socket(socket.AF_UNIX); s.settimeout(5); s.connect("/run/decompgym/license.sock"); s.sendall(b"  16-327 0 0 0 0")
reply = len(s.recv(4096))
lib = ctypes.CDLL(os.path.join(os.path.dirname(gp.__file__), ".libs", "libgurobi130.so")); lib.GRBlisttokens()
m = gp.Model(); x = m.addVar(lb=2); m.setObjective(x); m.Params.OutputFlag = 0; m.optimize()
print(json.dumps({"reply": reply, "obj": m.ObjVal}))
'''
    record = run(executor, data, solver, RunLimits(wall_seconds=120))
    tunnel = record["run"]["license_tunnel"]
    assert last_json(record) == {"reply": 0, "obj": 2.0} and tunnel["filtered"] >= 2 and tunnel["admitted"] == 1


def test_forwarder_filter_rate_budgets_and_shutdown():
    upstream = socket.socket()
    upstream.bind(("127.0.0.1", 0))
    upstream.listen(64)
    upstream.settimeout(0.2)
    port, stop = upstream.getsockname()[1], threading.Event()

    def echo(conn):
        try:
            for block in iter(lambda: conn.recv(65536), b""):
                conn.sendall(block)
        except OSError:
            pass

    def serve():
        while not stop.is_set():
            try:
                conn, _ = upstream.accept()
            except OSError:
                continue
            threading.Thread(target=echo, args=(conn,), daemon=True).start()

    threading.Thread(target=serve, daemon=True).start()
    directory = Path(tempfile.mkdtemp(dir="/tmp"))

    def exchange(path, payload):
        s = socket.socket(socket.AF_UNIX)
        s.settimeout(3)
        try:
            s.connect(str(path))
            s.sendall(payload)
            data = b""
            while len(data) < len(payload):
                block = s.recv(65536)
                if not block:
                    break
                data += block
            return data
        except OSError:
            return b""
        finally:
            s.close()

    fw = LicenseForwarder(directory / "a.sock", ["127.0.0.1"], port, rate_per_second=20, burst=3).start()
    assert exchange(fw.socket_path, b"  433 hello") == b"  433 hello"
    assert exchange(fw.socket_path, b"  16-327 0 0 0 0") == b""
    started = time.monotonic()
    for _ in range(12):
        exchange(fw.socket_path, b"  433 x")
    assert time.monotonic() - started > 0.35
    stats = fw.close()
    assert stats["filtered"] == 1 and stats["rate_delayed"] >= 1 and stats["shutdown_confirmed"]
    fw = LicenseForwarder(directory / "b.sock", ["127.0.0.1"], port, max_bytes_per_stream=1024, max_bytes_total=3000).start()
    assert exchange(fw.socket_path, b"  433 " + b"x" * 4000) == b""
    stats = fw.close()
    assert stats["capped_stream"] == 1
    fw = LicenseForwarder(directory / "c.sock", ["127.0.0.1"], port, max_bytes_per_stream=1 << 30, max_bytes_total=1 << 30).start()
    stalled = socket.socket(socket.AF_UNIX)
    stalled.connect(str(fw.socket_path))
    stalled.settimeout(5)
    threading.Thread(target=lambda: stalled.sendall(b"  433 " + b"z" * (8 << 20)), daemon=True).start()
    time.sleep(1)
    stats = fw.close()
    stalled.close()
    stop.set()
    assert stats["shutdown_confirmed"] and stats["shutdown_seconds"] < 1 and stats["open_at_shutdown"] == 0


def _upstream_server(handler):
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(64)
    server.settimeout(0.2)
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except OSError:
                continue
            threading.Thread(target=handler, args=(conn,), daemon=True).start()

    threading.Thread(target=serve, daemon=True).start()
    return server.getsockname()[1], stop


def _read_all(sock):
    data = b""
    while True:
        block = sock.recv(65536)
        if not block:
            return data
        data += block


def test_forwarder_delivers_reply_then_fin():
    def reply_and_close(conn):
        with conn:
            conn.recv(64)
            conn.sendall(b"R" * 29)  # a token-server-like reply followed immediately by FIN
    port, stop = _upstream_server(reply_and_close)
    fw = LicenseForwarder(Path(tempfile.mkdtemp(dir="/tmp")) / "f.sock", ["127.0.0.1"], port).start()
    replies = []
    for _ in range(30):
        client = socket.socket(socket.AF_UNIX)
        client.settimeout(5)
        client.connect(str(fw.socket_path))
        client.sendall(b"  433 13 request")
        replies.append(_read_all(client))
        client.close()
    stats = fw.close()
    stop.set()
    assert replies == [b"R" * 29] * 30 and stats["shutdown_confirmed"]


def test_forwarder_client_half_close_and_backpressure():
    def echo_until_eof(conn):
        with conn:
            for block in iter(lambda: conn.recv(65536), b""):
                conn.sendall(block)
    port, stop = _upstream_server(echo_until_eof)
    fw = LicenseForwarder(Path(tempfile.mkdtemp(dir="/tmp")) / "g.sock", ["127.0.0.1"], port,
                          max_bytes_per_stream=1 << 30, max_bytes_total=1 << 30).start()
    client = socket.socket(socket.AF_UNIX)
    client.settimeout(5)
    client.connect(str(fw.socket_path))
    client.sendall(b"  433 half-close")
    client.shutdown(socket.SHUT_WR)  # the request is complete; the reply must still arrive, then EOF
    assert _read_all(client) == b"  433 half-close"
    client.close()
    payload = b"  433 " + os.urandom(6 << 20)  # larger than every buffer on the path: exercises backpressure
    client = socket.socket(socket.AF_UNIX)
    client.settimeout(20)
    client.connect(str(fw.socket_path))
    sender = threading.Thread(target=lambda: (client.sendall(payload), client.shutdown(socket.SHUT_WR)), daemon=True)
    sender.start()
    received = _read_all(client)
    sender.join(20)
    client.close()
    stats = fw.close()
    stop.set()
    assert hashlib.sha256(received).digest() == hashlib.sha256(payload).digest() and stats["shutdown_confirmed"]


CORE = r'''
#include <vector>
#include <numeric>
extern "C" long dg_sum(const long* v, int n) { std::vector<long> a(v, v + n); return std::accumulate(a.begin(), a.end(), 0L); }
'''
MAIN = '#include <iostream>\nint main() { std::cout << "core-ok" << std::endl; return 0; }\n'
CALLER = r'''
import ctypes, json, os, subprocess
lib = ctypes.CDLL(os.environ["DECOMPGYM_CORE"]); lib.dg_sum.restype = ctypes.c_long
out = {"sum": lib.dg_sum((ctypes.c_long * 3)(4, 5, 6), 3)}
if "DECOMPGYM_CORE_BIN" in os.environ:
    out["bin"] = subprocess.run([os.environ["DECOMPGYM_CORE_BIN"]], capture_output=True, text=True).stdout.strip()
print(json.dumps(out))
'''


def test_cpp_build_library_and_binary_in_a_memory_step(executor, data):
    lib_only = run(executor, data, CALLER, cpp={"core.cpp": CORE})
    assert lib_only["build"]["status"] == "ok" and set(lib_only["build"]["targets"]) == {"libcore.so"} and last_json(lib_only) == {"sum": 15}
    assert all(step["memory_cgroup"]["memory_max"] == 4 * GiB for step in lib_only["build"]["steps"])
    both = run(executor, data, CALLER, cpp={"core.cpp": CORE, "main.cpp": MAIN})
    assert set(both["build"]["targets"]) == {"libcore.so", "core"} and last_json(both) == {"sum": 15, "bin": "core-ok"}


def test_track_policy_and_informational_audit(executor, data, tmp_path):
    workspace = tmp_path / "delivered"
    (workspace / "cpp").mkdir(parents=True)
    (workspace / "solver.py").write_text("print(1)\n")
    (workspace / "cpp/a.cpp").write_text("int a;\n")
    refused = executor.execute(workspace, data, RunLimits(wall_seconds=30), track="python_only")
    assert refused["termination"] == "build_error" and "python_only" in refused["detail"] and refused["build"] is None
    threads = run(executor, data, "print(1)", cpp={"a.cpp": "%:include <thread>\nint a;\n"})  # audit only records it
    assert threads["termination"] == "finished" and threads["build"]["status"] == "ok"
    assert any("threads" in finding for finding in threads["source_audit"]["cpp"]["findings"])


def test_scipy_optimize_is_masked_and_the_matrix_api_works(executor, data):
    solver = r'''
import json
out = {}
for name in ("scipy.optimize", "scipy.optimize._highspy._core", "scipy.optimize._lsap"):
    try:
        __import__(name); out[name] = "imported"
    except ImportError as e:
        out[name] = "ImportError: " + str(e)[:60]
import numpy as np, scipy.sparse as sp, gurobipy as gp
A = sp.csr_matrix(np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]]))
m = gp.Model(); m.Params.OutputFlag = 0
x = m.addMVar(3, lb=0.0)
m.addConstr(A @ x >= np.array([4.0, 6.0]))
expr = x.sum() + 2 * x[0] - x[1:].sum() * 0.5
m.setObjective(x.sum(), gp.GRB.MINIMIZE)
m.optimize()
out["obj"] = round(m.ObjVal, 6); out["expr_type"] = type(expr).__name__
print(json.dumps(out))
'''
    record = run(executor, data, solver)
    out = last_json(record)
    assert all(value.startswith("ImportError") for key, value in out.items() if key.startswith("scipy")), out
    assert record["termination"] == "finished" and out["obj"] == round(10.0 / 3.0, 6) and out["expr_type"] == "MLinExpr"
    assert "scipy.optimize" in record["run"]["masked_modules"]
    flagged = run(executor, data, "from scipy.optimize import milp\nprint(1)\n")
    assert flagged["termination"] == "crash" and "not available" in flagged["run"]["stderr"]
    assert any("milp" in flag for flag in flagged["source_audit"]["python"]["flags"])


def test_parser_failure_is_a_recorded_candidate_failure(executor, data):
    record = run(executor, data, "x=" + "1+" * 10000 + "1\n")
    assert record["source_audit"]["python"]["error"] and record["termination"] in ("crash", "finished")


def test_late_output_and_late_exit_never_reach_feedback(executor, data, tmp_path):
    path = tmp_path / "t0.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    t0 = {}

    def serve():
        conn, _ = server.accept()
        with conn:
            conn.sendall(repr(t0["t0"]).encode())

    threading.Thread(target=serve, daemon=True).start()
    solver = r'''
import os, signal, socket, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
s = socket.socket(socket.AF_UNIX); s.connect(os.environ["DECOMPGYM_PUBLISH"]); t0 = float(s.recv(64))
print("EARLY", flush=True)
while time.monotonic() < t0 + 2.0 + 0.004:
    pass
print("LATE", flush=True); sys.stderr.write("LATE-ERR\n"); sys.stderr.flush()
os._exit(0)
'''
    record = run(executor, data, solver, RunLimits(wall_seconds=2.0, grace_seconds=1), publish_socket=path, on_start=lambda t: t0.setdefault("t0", t))
    server.close()
    feedback = record["feedback"]["stdout"] + record["feedback"]["stderr"]
    assert record["termination"] == "deadline" and "EARLY" in feedback and "LATE" not in feedback and record["run"]["bytes_after_deadline"] >= 5


CPP_THREADS = r'''
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
static long spin(double seconds) {
    auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    long n = 0;
    while (std::chrono::steady_clock::now() < end) ++n;
    return n;
}
int main() {
    long single = spin(1.5);
    std::vector<long> counts(4);
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) threads.emplace_back([&counts, i] { counts[i] = spin(1.5); });
    for (auto& t : threads) t.join();
    long total = 0;
    for (long c : counts) total += c;
    std::printf("{\"single\": %ld, \"four_total\": %ld, \"ratio\": %.4f, \"hardware\": %u}\n", single, total, (double)total / single,
                std::thread::hardware_concurrency());
    return 0;
}
'''


def test_multithreaded_cpp_gets_one_cpu(executor, data):
    solver = "import os, subprocess\nprint(subprocess.run([os.environ['DECOMPGYM_CORE_BIN']], capture_output=True, text=True).stdout.strip())\n"
    record = run(executor, data, solver, cpp={"spin.cpp": CPP_THREADS})
    out = last_json(record)
    assert record["termination"] == "finished" and out["ratio"] < 1.3, out
    assert any("pthread_create" in name or "_ZNSt6thread" in name for name in record["build"]["native"]["core"]["notable_imports"])


def test_build_step_preflight_fails_closed_before_the_compiler(executor, data, tmp_path):
    from src.experiments.xl_build import BuildLimits as Limits
    attempt = executor.prepare(manifest_for({"solver.py": b"print(1)", "cpp/a.cpp": b"int a;"}, track="cpp_allowed"), track="cpp_allowed")
    try:
        result = executor.compile(attempt, [executor.gcc["gxx"], "-std=c++17", "-O2", "-fPIC", "-shared", "cpp/a.cpp", "-o", "build/libcore.so"],
                                  timeout=60, limits=Limits().rlimits(60), tail_bytes=20000, output_cap=1 << 20, step="probe",
                                  memory_bytes=4 * GiB, preflight_memory_cap=1 * GiB)
        assert result.infrastructure and "preflight" in result.infrastructure and "exceeds the configured cap" in result.infrastructure
        assert result.payload_exit() is None and not (attempt.build / "libcore.so").exists()
        good = executor.compile(attempt, [executor.gcc["gxx"], "-std=c++17", "-O2", "-fPIC", "-shared", "cpp/a.cpp", "-o", "build/libcore.so"],
                                timeout=60, limits=Limits().rlimits(60), tail_bytes=20000, output_cap=1 << 20, step="good")
        facts = good.step_cgroup
        assert facts["ok"] and facts["memory_max"] == 4 * GiB and facts["swap_bound"] == 0 and len(facts["cpuset"]) == 1
    finally:
        executor.cleanup(attempt)


def test_build_sees_only_the_compiler_and_its_inputs(executor, data):
    def incbin(path):
        source = 'asm(".section .rodata\\n.globl blob\\nblob:\\n.incbin \\"%s\\"\\n.previous");\nint a;\n' % path
        return run(executor, data, "print(1)", cpp={"blob.cpp": source})
    for tool in ("/usr/bin/bash", "/bin/sh", "/usr/bin/python3", "/usr/libexec/gcc/x86_64-redhat-linux/11/cc1", "/usr/local/bin"):
        record = incbin(tool)
        assert record["termination"] == "build_error" and "compilation" in record["detail"], (tool, record["detail"])
    permitted = incbin("/usr/bin/as")  # explicitly permitted payload: reachable during the build (protocol §5)
    assert permitted["termination"] == "finished"


def test_build_error_and_time_limit(executor, data):
    broken = run(executor, data, "print(1)", cpp={"bad.cpp": "int f( {\n"})
    assert broken["termination"] == "build_error" and "error" in broken["feedback"]["stderr"] and broken["run"] is None
    slow = "constexpr long w(long n) { long s = 0; for (long i = 0; i < n; ++i) s = (s * 31 + i) % 1000003; return s; }\n"
    slow += "\n".join(f"static_assert(w({200000 + i}) >= 0);" for i in range(400))
    timed = run(executor, data, "print(1)", cpp={"slow.cpp": slow}, build_limits=BuildLimits(budget_seconds=5))
    assert timed["termination"] == "build_error" and "time limit" in timed["detail"] and timed["build"]["seconds"] < 8


def test_pdptw_distance_convention_python_equals_cpp(executor, tmp_path):
    np = pytest.importorskip("numpy")
    from src.cg.pdptw.a_data import load_pdptw
    instances, expected = {}, {}
    for path in sorted((PDPTW / "Li400").glob("lrc1_4_*.vrp"))[:2]:
        inst = load_pdptw(path)
        instances[path.stem] = inst.coords.tolist()
        expected[path.stem] = hashlib.sha256(inst.distances().astype(np.int64).tobytes()).hexdigest()
    ties = [[k / 16.0, 0.0] for k in range(64)]
    instances["ties"] = ties
    arr = np.array(ties)
    expected["ties"] = hashlib.sha256(np.rint(1000.0 * np.abs(arr[:, None, 0] - arr[None, :, 0])).astype(np.int64).tobytes()).hexdigest()
    data = tmp_path / "coords.json"
    data.write_text(json.dumps(instances))
    cpp = r'''
#include <cmath>
extern "C" void dist(const double* p, int n, long long* out) {
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
        double dx = p[2 * i] - p[2 * j], dy = p[2 * i + 1] - p[2 * j + 1];
        out[(long long)i * n + j] = (long long)std::nearbyint(1000.0 * std::sqrt(dx * dx + dy * dy));
    }
}
'''
    solver = r'''
import ctypes, hashlib, json, os
import numpy as np
lib = ctypes.CDLL(os.environ["DECOMPGYM_CORE"])
out = {}
for name, coords in json.load(open("data.json")).items():
    p = np.ascontiguousarray(coords, dtype=np.float64); n = len(p); d = np.zeros(n * n, dtype=np.int64)
    lib.dist(p.ctypes.data_as(ctypes.c_void_p), n, d.ctypes.data_as(ctypes.c_void_p))
    out[name] = hashlib.sha256(d.tobytes()).hexdigest()
    if name == "ties":
        out["tie_row"] = d[:4].tolist()
print(json.dumps(out))
'''
    record = run(executor, data, solver, RunLimits(wall_seconds=120), cpp={"dist.cpp": cpp})
    assert record["termination"] == "finished", record["detail"]
    produced = last_json(record)
    assert produced["tie_row"] == [0, 62, 125, 188]  # 62.5 -> 62, 187.5 -> 188 (round half to even)
    assert {key: value for key, value in produced.items() if key != "tie_row"} == expected and len(expected) == 3
