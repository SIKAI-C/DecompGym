"""Validate the XL sandbox executor, license tunnel and C++ build on a Phoenix compute node (no optimization runs).

Usage (inside a Slurm job, from the code root):
    python scripts/phoenix/xl_executor_validate.py --out <report.json> [--only NAME ...] [--cgroup-oom]

Each check runs real sandboxed programs (adversarial where the review asked for it) and records what it observed. The process
exits with status 1 if any check fails and 2 on an unknown check name. ``--cgroup-oom`` fills the job's 16 GiB memory cgroup
and is opt-in; run it in its own job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments.xl_build import BuildLimits, DeliveryError, collect_delivery, manifest_for  # noqa: E402
from src.experiments.xl_executor import GiB, MiB, LicenseForwarder, RunLimits, XLSandboxExecutor, check_job_limits, persist_record  # noqa: E402

HOME = os.path.expanduser("~")
PDPTW_DIR = ROOT / "outputs/scaleup/raw/cvrp/Instances-main/PDPTW"
# The PDPTW C++ track's instance set (Li & Lim lr1/lrc1, 400 and 600 tasks), frozen with hashes.
EXPECTED_PDPTW_FILE = ROOT / "scripts/phoenix/xl_pdptw_li400_li600_sha256.json"
TRACK_PY, TRACK_CPP = "python_only", "cpp_allowed"


def last_json(text: str):
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


class Validator:
    def __init__(self, out: Path):
        self.out = out
        self.ex = XLSandboxExecutor()
        self.data_dir = Path(tempfile.mkdtemp(prefix="dgxl-val-", dir=self.ex.base_dir))
        self.report = {"node": platform.node(), "job": os.environ.get("SLURM_JOB_ID"), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "environment": self.ex.environment, "checks": []}

    def data(self, value, name="data.json") -> Path:
        path = self.data_dir / name
        path.write_text(json.dumps(value))
        return path

    def run(self, solver: str, *, data=None, limits=None, cpp: dict | None = None, publish=None, build_limits=None, track=None, executor=None,
            **kwargs) -> dict:
        files = {"solver.py": solver.encode(), **{"cpp/" + name: text.encode() for name, text in (cpp or {}).items()}}
        track = track or (TRACK_CPP if cpp else TRACK_PY)
        return (executor or self.ex).execute((manifest_for(files, track=track)), data or self.data({"probe": True}),
                                             limits or RunLimits(wall_seconds=60), track=track, publish_socket=publish, build_limits=build_limits, **kwargs)

    def check(self, name, function):
        started = time.monotonic()
        entry = {"name": name}
        try:
            ok, observed = function()
            entry.update(ok=bool(ok), observed=observed)
        except Exception as error:  # noqa: BLE001 - recorded as a failed check
            entry.update(ok=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()[-4000:])
        entry["seconds"] = round(time.monotonic() - started, 3)
        self.report["checks"].append(entry)
        self.out.write_text(json.dumps(self.report, indent=1, default=str))
        print(f"[{'PASS' if entry.get('ok') else 'FAIL'}] {name} ({entry['seconds']} s)" + (f": {entry.get('error')}" if entry.get("error") else ""), flush=True)
        return entry

    @staticmethod
    def brief(record: dict, keep=("termination", "detail", "wall_seconds", "startup_seconds", "exit_code", "signal", "maxrss_kib", "cpu_seconds",
                                  "oom_kills", "peak_scratch_bytes", "cleanup_verified", "kill_reason", "stdout_bytes", "stderr_bytes",
                                  "bytes_after_deadline")) -> dict:
        run = record.get("run") or {}
        out = {"termination": record["termination"], "detail": record["detail"], **{key: run.get(key) for key in keep if key in run}}
        if run.get("term_sent") is not None:
            out["term_after_t0"] = round(run["term_sent"] - run["t0"], 3)
        if run.get("kill_sent") is not None:
            out["kill_after_t0"] = round(run["kill_sent"] - run["t0"], 3)
        return out

    # --------------------------------------------------------------------------------------------- delivery and limits

    def delivery_rules(self):
        base = Path(tempfile.mkdtemp(prefix="dgxl-del-", dir=self.ex.base_dir))
        outcomes = {}

        def attempt(label, function):
            try:
                function()
                outcomes[label] = "accepted"
            except DeliveryError as error:
                outcomes[label] = f"rejected: {error}"

        root = base / "good"
        (root / "cpp").mkdir(parents=True)
        (root / "solver.py").write_text("print(1)\n")
        (root / "cpp/a.cpp").write_text("int a;\n")
        good, _ = collect_delivery(root, track=TRACK_CPP)
        outcomes["good_files"] = [entry["path"] for entry in good["files"]]
        attempt("python_only_with_cpp", lambda: collect_delivery(root, track=TRACK_PY))
        sparse = base / "sparse"
        sparse.mkdir()
        (sparse / "solver.py").write_text("print(1)\n")
        with open(sparse / "solver.py", "r+b") as stream:
            stream.truncate(3 * GiB)  # sparse: rejected from lstat before any read
        t = time.monotonic()
        attempt("sparse_3GiB_solver", lambda: collect_delivery(sparse, track=TRACK_PY))
        outcomes["sparse_seconds"] = round(time.monotonic() - t, 4)
        many = base / "many"
        (many / "cpp").mkdir(parents=True)
        (many / "solver.py").write_text("print(1)\n")
        for index in range(300):
            (many / f"cpp/f{index}.cpp").write_text("")
        attempt("300_entries", lambda: collect_delivery(many, track=TRACK_CPP))
        total = base / "total"
        (total / "cpp").mkdir(parents=True)
        (total / "solver.py").write_text("print(1)\n")
        for index in range(5):
            (total / f"cpp/f{index}.cpp").write_bytes(b"/" * (2 * MiB - 10))
        attempt("total_over_8MiB", lambda: collect_delivery(total, track=TRACK_CPP))
        attempt("tuple_without_solver", lambda: self.ex.prepare(({"version": good["version"], "track": TRACK_CPP, "files": [], "sha256": ""}, {}), track=TRACK_CPP))
        ok = (outcomes["good_files"] == ["cpp/a.cpp", "solver.py"] and all(outcomes[key].startswith("rejected") for key in
              ("python_only_with_cpp", "sparse_3GiB_solver", "300_entries", "total_over_8MiB", "tuple_without_solver")) and outcomes["sparse_seconds"] < 1)
        return ok, outcomes

    def preflight_limits(self):
        facts = check_job_limits(self.ex.cgroup, cpus=1, memory_cap=16 * GiB, swap_cap=0)
        refused = {}
        for label, kwargs in (("two_cpus_required", {"cpus": 2, "memory_cap": 16 * GiB, "swap_cap": 0}),
                              ("memory_cap_8GiB", {"cpus": 1, "memory_cap": 8 * GiB, "swap_cap": 0})):
            try:
                check_job_limits(self.ex.cgroup, **kwargs)
                refused[label] = "accepted"
            except Exception as error:  # noqa: BLE001
                refused[label] = f"refused: {error}"
        ok = facts["memory_max"] <= 16 * GiB and facts["swap_bound"] == 0 and len(facts["cpuset"]) == 1 and all(v.startswith("refused") for v in refused.values())
        return ok, {"job_limits": facts, "refused": refused}

    # ------------------------------------------------------------------------------------------------ isolation

    def isolation(self):
        abstract = "\0dgxl-probe-" + str(os.getpid())
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(abstract)
        listener.listen(1)
        data = self.data({"probe": True})
        paths = {"repository": str(ROOT), "home": HOME, "ssh": os.path.join(HOME, ".ssh"), "scaleup_base": str(ROOT.parent),
                 "coda_storage": "/storage/coda1", "pace_apps": "/storage/pace-apps", "usr_local": "/usr/local", "usr_bin": "/usr/bin",
                 "bin_sh": "/bin/sh", "usr_libexec": "/usr/libexec", "usr_lib_golang": "/usr/lib/golang", "usr_lib_jvm": "/usr/lib/jvm",
                 "host_license": "/usr/local/pace-apps/manual/packages/gurobi/13.0.1/license/gurobi.lic", "data_host_path": str(data),
                 "data_host_dir": str(data.parent), "munge": "/run/munge", "slurm_conf": "/etc/slurm", "sys": "/sys", "var": "/var",
                 "opt_slurm": "/opt/slurm", "scratch": "/scratch", "conda_ortools": os.path.join(self.ex.python_facts["purelib"], "ortools"),
                 "conda_torch": os.path.join(self.ex.python_facts["purelib"], "torch"), "host_tmp_listing": "/tmp", "project_storage": "/storage/project"}
        solver = r'''
import ctypes, errno, json, os, re, signal, socket, stat, subprocess, sys, time
paths = json.loads(PATHS)
out = {"uid": os.getuid(), "pid": os.getpid(), "hostname": socket.gethostname(), "cwd": os.getcwd(), "root": sorted(os.listdir("/")),
       "usr": sorted(os.listdir("/usr")), "etc": sorted(os.listdir("/etc")), "run": sorted(os.listdir("/run/decompgym")), "work": sorted(os.listdir("/work")),
       "dev": sorted(os.listdir("/dev")), "env": dict(os.environ), "sys_path": sys.path, "proc_pids": sorted(int(p) for p in os.listdir("/proc") if p.isdigit()),
       "site_packages": sorted(os.listdir(next(p for p in sys.path if p.endswith("site-packages")))), "prefix_bin": sorted(os.listdir(os.path.dirname(sys.executable))),
       "affinity": sorted(os.sched_getaffinity(0))}
status = open("/proc/self/status").read()
out["status"] = {line.split(":")[0]: line.split(":", 1)[1].strip() for line in status.splitlines() if line.split(":")[0] in ("CapEff", "NoNewPrivs", "Seccomp", "NSpid")}
mounts = []
for line in open("/proc/self/mountinfo"):
    fields = line.split()
    sep = fields.index("-")
    mounts.append({"mount": fields[4], "options": fields[5], "fstype": fields[sep + 1]})
out["writable_mounts"] = sorted({m["mount"] + ":" + m["fstype"] for m in mounts if m["options"].split(",")[0] == "rw"})
vis = {}
for key, path in paths.items():
    try:
        vis[key] = {"exists": os.path.lexists(path), "listing": sorted(os.listdir(path))[:20] if os.path.isdir(path) else None}
    except OSError as e:
        vis[key] = {"exists": os.path.lexists(path), "error": e.strerror}
out["visibility"] = vis
writes = {}
for label, path in (("data.json", "/work/data.json"), ("solver.py", "/work/solver.py"), ("gurobi.env", "/work/gurobi.env"), ("usr", "/usr/lib64/xx"),
                    ("root", "/xx"), ("etc", "/etc/xx"), ("run", "/run/decompgym/xx"), ("prefix", sys.prefix + "/xx"), ("work_new", "/work/new.txt"),
                    ("tmp_new", "/tmp/new.txt"), ("dev_shm_new", "/dev/shm/new.txt"), ("dev_new", "/dev/new.txt")):
    try:
        with open(path, "a") as f:
            f.write("x")
        writes[label] = "writable"
    except OSError as e:
        writes[label] = errno.errorcode.get(e.errno, str(e.errno))
out["writes"] = writes
net = {}
def attempt(label, fn):
    try:
        net[label] = "OK: " + str(fn())
    except Exception as e:
        net[label] = type(e).__name__ + ": " + str(e)
attempt("tcp_license_direct", lambda: socket.create_connection(("128.61.254.83", 41954), timeout=3).getpeername())
attempt("tcp_internet", lambda: socket.create_connection(("1.1.1.1", 443), timeout=3).getpeername())
attempt("dns_example_com", lambda: socket.getaddrinfo("example.com", 443))
def udp():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.sendto(b"x", ("8.8.8.8", 53)); return "sent"
attempt("udp_8.8.8.8", udp)
attempt("tcp_loopback_22", lambda: socket.create_connection(("127.0.0.1", 22), timeout=3).getpeername())
attempt("tcp_loopback_relay", lambda: socket.create_connection(("127.0.0.1", 41954), timeout=3).getpeername())
def abstract():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(ABSTRACT); return "connected"
attempt("host_abstract_unix_socket", abstract)
attempt("raw_socket", lambda: socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP))
def via_license_socket():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(5); s.connect("/run/decompgym/license.sock")
    s.sendall(b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n")
    try:
        data = s.recv(200)
    except socket.timeout:
        data = b"<timeout>"
    return {"reply_bytes": len(data)}
attempt("http_through_license_socket", via_license_socket)
out["network"] = net
libc = ctypes.CDLL(None, use_errno=True)
def sys_call(number, *args):
    value = libc.syscall(ctypes.c_long(number), *[ctypes.c_long(a) for a in args])
    return value if value != -1 else "errno " + errno.errorcode.get(ctypes.get_errno(), str(ctypes.get_errno()))
limit = (ctypes.c_ulong * 2)(1, 1)
out["syscalls"] = {"keyctl_user_keyring": sys_call(250, 0, -4, 0), "ptrace_attach_init": sys_call(101, 16, 1, 0, 0), "bpf": sys_call(321, 0, 0, 0),
                   "perf_event_open": sys_call(298, 0, 0, -1, -1, 0), "io_uring_setup": sys_call(425, 1, 0), "unshare_newuser": sys_call(272, 0x10000000),
                   "clone_newnet": sys_call(56, 0x40000000 | 17, 0, 0, 0, 0), "prlimit64_init_cpu": sys_call(302, 1, 0, ctypes.addressof(limit), 0),
                   "getpid": sys_call(39)}
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
out["own_setrlimit"] = "ok"
import threading
holder = []
t = threading.Thread(target=lambda: holder.append(1)); t.start(); t.join()
out["threads_work"] = holder == [1]
try:
    os.kill(1, signal.SIGKILL); time.sleep(0.2); out["kill_init"] = "sent; still alive"
except OSError as e:
    out["kill_init"] = errno.errorcode.get(e.errno, str(e.errno))
for label, path in (("init_environ", "/proc/1/environ"), ("init_fd", "/proc/1/fd")):
    try:
        out[label] = (os.listdir(path) if path.endswith("fd") else open(path, "rb").read()[:80].decode(errors="replace"))
    except OSError as e:
        out[label] = errno.errorcode.get(e.errno, str(e.errno))
tool = re.compile(TOOLRE)
usable = []
for top in ("/usr", sys.prefix, "/lib64", "/lib"):
    for directory, subdirs, names in os.walk(top):
        for name in names:
            path = os.path.join(directory, name)
            if tool.fullmatch(name) and os.path.exists(path):
                info = os.stat(path)
                if stat.S_ISREG(info.st_mode) and os.access(path, os.X_OK):
                    usable.append(path)
out["usable_toolchains"] = usable
for label, argv in (("exec_gcc", ["/usr/bin/gcc", "--version"]), ("exec_sh", ["/bin/sh", "-c", "true"]), ("exec_conda_ld", [sys.prefix + "/compiler_compat/ld", "--version"])):
    try:
        out[label] = subprocess.run(argv, capture_output=True, timeout=10).returncode
    except OSError as e:
        out[label] = errno.errorcode.get(e.errno, str(e.errno))
print(json.dumps(out))
'''.replace("PATHS", repr(json.dumps(paths))).replace("ABSTRACT", repr(abstract)).replace("TOOLRE", repr(r"(cc1(plus)?|collect2|lto1|as|ld|ld\.(bfd|gold)|gcc|g\+\+|c\+\+|cpp|tcc|gfortran|clang|go|java|javac|.+-(gcc|g\+\+|ld|as))"))
        try:
            record = self.run(solver, data=data)
        finally:
            listener.close()
        observed = last_json(record["run"]["stdout"]) if record.get("run") else None
        if observed is None:
            return False, {"record": self.brief(record), "stderr": (record.get("run") or {}).get("stderr")}
        vis, writes, net = observed["visibility"], observed["writes"], observed["network"]
        hidden = ("repository", "home", "ssh", "scaleup_base", "coda_storage", "pace_apps", "usr_local", "usr_bin", "bin_sh", "usr_libexec", "usr_lib_golang",
                  "usr_lib_jvm", "host_license", "data_host_path", "data_host_dir", "munge", "slurm_conf", "sys", "var", "opt_slurm", "scratch", "conda_ortools", "conda_torch")
        problems = [key for key in hidden if vis[key]["exists"]]
        problems += [f"write {key}" for key in ("data.json", "solver.py", "gurobi.env", "usr", "root", "etc", "run", "prefix") if writes[key] == "writable"]
        problems += [f"write {key} failed" for key in ("work_new", "tmp_new", "dev_shm_new") if writes[key] != "writable"]
        problems += [f"net {key}" for key in ("tcp_license_direct", "tcp_internet", "dns_example_com", "udp_8.8.8.8", "tcp_loopback_22",
                                              "host_abstract_unix_socket", "raw_socket") if net[key].startswith("OK")]
        if not net["tcp_loopback_relay"].startswith("OK"):
            problems.append("relay not reachable")
        storage = {entry for entry in observed["writable_mounts"] if not entry.endswith((":proc", ":devpts", ":devtmpfs"))}
        if storage != {"/work:tmpfs", "/tmp:tmpfs", "/dev:tmpfs", "/dev/shm:tmpfs"}:
            problems.append(f"writable storage mounts {sorted(storage)}")
        if observed["affinity"] != record["run"]["job_limits"]["cpuset"]:
            problems.append("affinity")
        if observed["status"].get("CapEff") != "0000000000000000":
            problems.append("capabilities")
        if observed["kill_init"] != "sent; still alive":
            problems.append("kill_init")
        problems += [f"syscall {key} allowed" for key, value in observed["syscalls"].items() if key != "getpid" and value != "errno EPERM"]
        if not observed.get("threads_work") or observed.get("own_setrlimit") != "ok":
            problems.append("threads or own setrlimit broken by the seccomp filter")
        if observed["usable_toolchains"]:
            problems.append(f"usable toolchains {observed['usable_toolchains'][:5]}")
        problems += [f"{key} ran" for key in ("exec_gcc", "exec_sh", "exec_conda_ld") if isinstance(observed[key], int)]
        observed["env"] = {key: value for key, value in observed["env"].items()}
        try:
            host_keys = [" ".join(line.split()[7:9]) for line in Path("/proc/keys").read_text().splitlines()]
        except OSError as error:
            host_keys = f"unreadable: {error}"
        return not problems, {"problems": problems, "record": self.brief(record), "sandbox": observed, "license_tunnel": record["run"]["license_tunnel"],
                              "job_limits": record["run"]["job_limits"], "runtime": record["run"]["runtime"], "host_keyring_entries": host_keys,
                              "toolchain_masks": self.ex.inventory}

    def curl_diagnostic(self):
        """Diagnostic mode with all of /usr mounted (not the scored runtime): curl and wget still cannot reach anything."""
        diag = XLSandboxExecutor(base_dir=self.ex.base_dir, expose_usr=True)
        solver = r'''
import json, subprocess
out = {}
for argv in (["curl", "-sS", "-m", "5", "https://example.com"], ["curl", "-sS", "-m", "5", "http://128.61.254.83:41954/"], ["wget", "-q", "-T", "5", "-O-", "https://example.com"], ["gcc", "--version"]):
    try:
        r = subprocess.run(argv, capture_output=True, text=True)
        out[" ".join(argv[:1] + argv[-1:])] = [r.returncode, (r.stderr or r.stdout).strip()[:120]]
    except OSError as e:
        out[" ".join(argv[:1] + argv[-1:])] = [-1, str(e)[:120]]
print(json.dumps(out))
'''
        record = self.run(solver, executor=diag)
        out = last_json(record["run"]["stdout"]) if record.get("run") else None
        ok = out is not None and all(value[0] != 0 for value in out.values())
        return ok, {"results": out, "record": self.brief(record)}

    def init_signals(self):
        """Adversarial: every catchable signal to PID 1, prlimit64 on PID 1; the run must still end with the payload's own report."""
        solver = r'''
import errno, json, os, resource, signal, time
sent = {}
for number in range(1, 65):
    if number in (signal.SIGKILL, signal.SIGSTOP, 32, 33):
        continue
    try:
        os.kill(1, number); sent[number] = "sent"
    except OSError as e:
        sent[number] = errno.errorcode.get(e.errno)
for number in (signal.SIGKILL, signal.SIGSTOP):
    os.kill(1, number)
time.sleep(1.0)
import ctypes
libc = ctypes.CDLL(None, use_errno=True)
def prlimit64(pid, resource_id, new):  # raw syscall: this Python's resource module has no prlimit
    old = (ctypes.c_ulong * 2)()
    rc = libc.syscall(ctypes.c_long(302), ctypes.c_long(pid), ctypes.c_long(resource_id), ctypes.c_void_p(ctypes.addressof(new) if new else 0),
                      ctypes.c_void_p(ctypes.addressof(old)))
    return "ok" if rc == 0 else errno.errorcode[ctypes.get_errno()]
prlimit = prlimit64(1, 0, (ctypes.c_ulong * 2)(1, 1))           # RLIMIT_CPU of PID 1 -> 1 s
read = prlimit64(1, 7, None)                                      # read PID 1's RLIMIT_NOFILE
own = prlimit64(0, 4, (ctypes.c_ulong * 2)(0, 0))                 # own RLIMIT_CORE: allowed
print(json.dumps({"signals_sent": len(sent), "prlimit_init": prlimit, "prlimit_read_init": read, "prlimit_self": own}), flush=True)
'''
        record = self.run(solver)
        out = last_json(record["run"]["stdout"]) if record.get("run") else None
        exit_event = [e for e in (record["run"] or {}).get("events") or [] if e.get("event") == "payload_exit"]
        ok = (record["termination"] == "finished" and bool(exit_event) and out and out["prlimit_init"] == "EPERM" and out["prlimit_read_init"] == "EPERM"
              and out["prlimit_self"] == "ok"
              and record["run"]["sandbox"]["init_handlers_reset"] == [2])
        return ok, {"solver": out, "record": self.brief(record), "init_handlers_reset": record["run"]["sandbox"]["init_handlers_reset"],
                    "stderr": record["run"]["stderr"][-600:]}

    # --------------------------------------------------------------------------------------------------- gurobi

    def gurobi(self):
        preflight = self.ex.gurobi_preflight()
        solver = r'''
import json, gurobipy as gp
env = gp.Env(params={"OutputFlag": 1})
m = gp.Model(env=env)
x = m.addVars(3000, vtype="B", obj=[(i * 37) % 101 + 1 for i in range(3000)])
m.addConstrs(x[i] + x[i + 1] >= 1 for i in range(2999))
m.Params.TimeLimit = 20
m.optimize()
print(json.dumps({"status": m.Status, "obj": m.ObjVal, "threads": m.Params.Threads, "vars": m.NumVars}))
'''
        record = self.run(solver, limits=RunLimits(wall_seconds=90))
        run = record["run"] or {}
        out = last_json(run.get("stdout", ""))
        many = r'''
import json, time, gurobipy as gp
t = time.monotonic(); objs = []
for i in range(40):
    with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as m:
        x = m.addVar(lb=i); m.setObjective(x); m.optimize(); objs.append(m.ObjVal)
seq = time.monotonic() - t
envs = [gp.Env(params={"OutputFlag": 0}) for _ in range(6)]
models = [gp.Model(env=e) for e in envs]
for k, m in enumerate(models):
    x = m.addVar(lb=k); m.setObjective(x); m.optimize()
print(json.dumps({"sequential_ok": objs == [float(i) for i in range(40)], "sequential_seconds": seq, "concurrent": [m.ObjVal for m in models]}))
'''
        many_record = self.run(many, limits=RunLimits(wall_seconds=120))
        many_out = last_json((many_record["run"] or {}).get("stdout", ""))
        ok = (preflight["ok"] and record["termination"] == "finished" and out and out["vars"] == 3000 and out["threads"] == 1
              and many_record["termination"] == "finished" and many_out and many_out["sequential_ok"]
              and many_record["run"]["license_tunnel"]["filtered"] == 0 and many_record["run"]["license_tunnel"]["shutdown_confirmed"])
        return ok, {"preflight": preflight, "solve": out, "record": self.brief(record), "license_tunnel": run.get("license_tunnel"), "many_envs": many_out,
                    "many_envs_tunnel": (many_record["run"] or {}).get("license_tunnel"), "runtime": run.get("runtime")}

    def license_filter(self):
        """Adversarial: GRBlisttokens via ctypes and a raw status request on the socket are dropped before any upstream connection."""
        solver = r'''
import ctypes, json, os, socket, time
import gurobipy as gp
out = {}
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(5); s.connect("/run/decompgym/license.sock")
s.sendall(b"  16-327 0 0 0 0")
try:
    out["raw_status_reply"] = len(s.recv(4096))
except OSError as e:
    out["raw_status_reply"] = str(e)
lib = ctypes.CDLL(os.path.join(os.path.dirname(gp.__file__), ".libs", "libgurobi130.so"))
out["listtokens_rc"] = lib.GRBlisttokens()
m = gp.Model(); x = m.addVar(lb=2); m.setObjective(x); m.Params.OutputFlag = 0; m.optimize(); out["solve_after"] = m.ObjVal
print(json.dumps(out), flush=True)
'''
        record = self.run(solver, limits=RunLimits(wall_seconds=120))
        run = record["run"] or {}
        out = last_json(run.get("stdout", ""))
        tunnel = run.get("license_tunnel") or {}
        status_lines = [line for line in run.get("stdout", "").splitlines() if "token" in line.lower() or "server" in line.lower()][:6]
        ok = (record["termination"] == "finished" and out and out["raw_status_reply"] == 0 and out["solve_after"] == 2.0
              and tunnel.get("filtered", 0) >= 2 and tunnel.get("admitted") == 1 and tunnel.get("shutdown_confirmed"))
        return ok, {"solver": out, "tunnel": tunnel, "listtokens_output": status_lines, "record": self.brief(record)}

    def forwarder_budgets(self):
        """Forwarder against a local echo upstream: prefix filter, rate budget, byte budgets, confirmed shutdown with a stalled peer."""
        upstream = socket.socket()
        upstream.bind(("127.0.0.1", 0))
        upstream.listen(256)
        port = upstream.getsockname()[1]
        stop = threading.Event()

        def echo(conn):
            try:
                for block in iter(lambda: conn.recv(65536), b""):
                    conn.sendall(block)
            except OSError:
                pass

        def serve():
            upstream.settimeout(0.2)
            while not stop.is_set():
                try:
                    conn, _ = upstream.accept()
                except OSError:
                    continue
                threading.Thread(target=echo, args=(conn,), daemon=True).start()

        threading.Thread(target=serve, daemon=True).start()
        directory = Path(tempfile.mkdtemp(prefix="dgxl-fw-", dir="/tmp"))

        def stream(path, payload):
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
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

        results = {}
        fw = LicenseForwarder(directory / "a.sock", ["127.0.0.1"], port, rate_per_second=20, burst=5).start()
        results["token_echo"] = stream(fw.socket_path, b"  433 13 hello").decode(errors="replace")
        results["status_echo"] = stream(fw.socket_path, b"  16-327 0 0 0 0").decode(errors="replace")
        t = time.monotonic()
        for _ in range(25):
            stream(fw.socket_path, b"  433 x")
        results["25_streams_seconds"] = round(time.monotonic() - t, 3)
        rate = fw.close()
        fw = LicenseForwarder(directory / "b.sock", ["127.0.0.1"], port, max_bytes_per_stream=1024).start()
        results["over_stream_cap_reply"] = len(stream(fw.socket_path, b"  433 " + b"x" * 5000))
        per_stream = fw.close()
        fw = LicenseForwarder(directory / "c.sock", ["127.0.0.1"], port, max_bytes_per_stream=1 << 20, max_bytes_total=4096).start()
        results["total_cap_replies"] = [len(stream(fw.socket_path, b"  433 " + b"y" * 1500)) for _ in range(4)]
        total = fw.close()
        fw = LicenseForwarder(directory / "d.sock", ["127.0.0.1"], port, max_bytes_per_stream=1 << 30, max_bytes_total=1 << 30).start()
        stalled = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stalled.connect(str(fw.socket_path))
        stalled.settimeout(8)
        threading.Thread(target=lambda: stalled.sendall(b"  433 " + b"z" * (16 << 20)), daemon=True).start()  # never reads its echo
        time.sleep(2)
        stall = fw.close()
        stalled.close()
        stop.set()
        upstream.close()
        ok = (results["token_echo"] == "  433 13 hello" and results["status_echo"] == "" and rate["filtered"] == 1 and rate["admitted"] == 26
              and rate["rate_delayed"] >= 1 and results["25_streams_seconds"] >= 0.8 and per_stream["capped_stream"] == 1
              and results["over_stream_cap_reply"] == 0 and total["capped_total"] + total["refused"] >= 1 and results["total_cap_replies"][-1] == 0
              and all(item["shutdown_confirmed"] for item in (rate, per_stream, total, stall)) and stall["shutdown_seconds"] < 1.0
              and stall["open_at_shutdown"] == 0)
        return ok, {"results": results, "rate_test": rate, "per_stream_cap": per_stream, "total_cap": total, "stalled_peer_shutdown": stall}

    def license_attribution(self):
        """A candidate's own Gurobi-flavoured crash stays a crash; a broken tunnel becomes infrastructure."""
        own = self.run("import gurobipy\nraise RuntimeError('my gurobi model is wrong')\n")
        broken_license = self.data_dir / "broken.lic"
        broken_license.write_text("TOKENSERVER=no-such-host.invalid\nPORT=41954\n")
        broken = XLSandboxExecutor(license=broken_license, base_dir=self.ex.base_dir)
        record = self.run("import gurobipy as gp\nm = gp.Model()\nm.optimize()\nprint('solved')\n", executor=broken)
        run = record["run"] or {}
        ok = own["termination"] == "crash" and (own["run"].get("gurobi_diagnostic") or {}).get("ok") and record["termination"] == "infrastructure"
        return ok, {"own_error": {"termination": own["termination"], "diagnostic_ok": (own["run"].get("gurobi_diagnostic") or {}).get("ok")},
                    "broken_tunnel": {"termination": record["termination"], "detail": record["detail"], "tunnel": run.get("license_tunnel"),
                                      "stderr_tail": (run.get("stderr") or "")[-200:]}}

    # --------------------------------------------------------------------------------------------------- limits

    def fork_bomb(self):
        solver = r'''
import os, time, json, errno
count, error = 0, None
while count < 100000:
    try:
        pid = os.fork()
    except OSError as e:
        error = errno.errorcode.get(e.errno, str(e.errno)); break
    if pid == 0:
        time.sleep(1000); os._exit(0)
    count += 1
print(json.dumps({"forked": count, "error": error}), flush=True)
time.sleep(1000)
'''
        record = self.run(solver, limits=RunLimits(wall_seconds=8, max_processes=64))
        out = last_json(record["run"]["stdout"])
        ok = record["termination"] == "deadline" and out and out["error"] == "EAGAIN" and out["forked"] < 64 and record["run"]["cleanup_verified"]
        return ok, {"solver": out, "record": self.brief(record)}

    def memory_rlimit(self):
        """With an optional RLIMIT_AS, MemoryError is a candidate-reported failure: a crash with a note, not a kernel OOM."""
        solver = r'''
import json
chunks = []
try:
    while True:
        b = bytearray(256 * 1024 * 1024); b[::4096] = b"\x01" * len(b[::4096]); chunks.append(b)
except MemoryError:
    print(json.dumps({"allocated_mib": 256 * len(chunks)}), flush=True)
    raise
'''
        record = self.run(solver, limits=RunLimits(wall_seconds=60, address_space_bytes=2 * GiB))
        forged = self.run("import sys\nsys.stderr.write('Traceback\\nMemoryError\\nstd::bad_alloc Out of memory\\n')\nsys.exit(1)\n")
        ok = (record["termination"] == "crash" and "no kernel OOM" in record["detail"] and forged["termination"] == "crash" and "no kernel OOM" in forged["detail"])
        return ok, {"address_space": self.brief(record), "forged_stderr": self.brief(forged)}

    def memory_cgroup(self):
        solver = r'''
import json, time
chunks = []; t = time.monotonic()
while True:
    b = bytearray(512 * 1024 * 1024); b[::4096] = b"\x01" * len(b[::4096]); chunks.append(b)
    print(json.dumps({"allocated_mib": 512 * len(chunks), "t": time.monotonic() - t}), flush=True)
'''
        record = self.run(solver, limits=RunLimits(wall_seconds=300))
        out = last_json(record["run"]["stdout"])
        persisted = persist_record(self.out.parent / f"oom_record_{os.environ.get('SLURM_JOB_ID', 'local')}.json", record)
        ok = record["termination"] == "oom" and (record["run"]["oom_kills"] or 0) >= 1 and json.loads(persisted.read_text())["termination"] == "oom"
        return ok, {"solver_last": out, "record": self.brief(record), "executor_alive": True, "persisted_record": str(persisted)}

    def file_size(self):
        python = "with open('/work/big.bin', 'wb') as f:\n    for i in range(200):\n        f.write(b'\\0' * (1024 * 1024))\n"
        record = self.run(python, limits=RunLimits(wall_seconds=60, file_size_bytes=64 * MiB))
        native = "import signal\nsignal.signal(signal.SIGXFSZ, signal.SIG_DFL)\n" + python
        signalled = self.run(native, limits=RunLimits(wall_seconds=60, file_size_bytes=64 * MiB))
        ok = (record["termination"] == "crash" and "file size limit" in record["detail"] and signalled["termination"] == "crash" and "SIGXFSZ" in signalled["detail"])
        return ok, {"efbig": self.brief(record), "sigxfsz": self.brief(signalled)}

    def scratch_quota(self):
        results = {}
        for mount in ("/work", "/dev/shm", "/tmp"):
            solver = ("import json, time\nfor i in range(64):\n"
                      f"    open('{mount}/chunk%d.bin' % i, 'wb').write(b'\\1' * (32 * 1024 * 1024))\n"
                      "    print(json.dumps({'files': i + 1}), flush=True)\n    time.sleep(0.1)\n")
            record = self.run(solver, limits=RunLimits(wall_seconds=60, scratch_quota_bytes=512 * MiB))
            results[mount] = self.brief(record)
        ok = all(entry["termination"] == "crash" and "scratch" in entry["detail"] for entry in results.values())
        return ok, results

    def open_files(self):
        solver = r'''
import json, errno
handles, error = [], None
try:
    for i in range(500):
        handles.append(open("/work/data.json"))
except OSError as e:
    error = errno.errorcode.get(e.errno)
print(json.dumps({"opened": len(handles), "error": error}))
'''
        record = self.run(solver, limits=RunLimits(wall_seconds=30, open_files=64))
        out = last_json(record["run"]["stdout"])
        return record["termination"] == "finished" and out["error"] == "EMFILE" and out["opened"] < 64, {"solver": out, "record": self.brief(record)}

    def deadline_tree(self):
        solver = r'''
import json, os, signal, subprocess, sys, threading, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pids = {"solver": os.getpid()}
pids["sleep_child"] = subprocess.Popen([sys.executable, "-c", "import time\ntime.sleep(1000)"]).pid
ignorer = subprocess.Popen([sys.executable, "-c", "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nwhile True: time.sleep(1)"])
pids["sigterm_ignoring_child"] = ignorer.pid
r, w = os.pipe()
first = os.fork()
if first == 0:
    os.setsid()
    second = os.fork()
    if second == 0:
        os.write(w, str(os.getpid()).encode()); signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True: time.sleep(1)
    os._exit(0)
os.waitpid(first, 0)
pids["daemon_grandchild"] = int(os.read(r, 32))
threading.Thread(target=lambda: [None for _ in iter(int, 1)], daemon=True).start()
print(json.dumps(pids), flush=True)
while True:
    time.sleep(1)
'''
        seen = {}
        record = self.run(solver, data=self.data({"x": 1}), limits=RunLimits(wall_seconds=6, grace_seconds=2), on_start=lambda t0: seen.setdefault("t0", t0))
        run = record["run"]
        ok = (record["termination"] == "deadline" and run["cleanup_verified"] and run["kill_sent"] is not None
              and 5.9 <= run["term_sent"] - run["t0"] <= 6.5 and run["wall_seconds"] < 10 and abs(seen.get("t0", 0) - run["t0"]) < 1e-9)
        return ok, {"inner_pids": last_json(run["stdout"]), "record": self.brief(record)}

    def freeze_feedback(self):
        """Output written after T_f (during the grace period) never reaches the feedback."""
        solver = r'''
import signal, sys, time
signal.signal(signal.SIGTERM, lambda *a: (print("AFTER-TF-SIGTERM", flush=True), print("AFTER-TF-STDERR", file=sys.stderr, flush=True)))
print("BEFORE-TF", flush=True)
time.sleep(1000)
'''
        record = self.run(solver, limits=RunLimits(wall_seconds=3, grace_seconds=2))
        feedback = record["feedback"]["stdout"] + record["feedback"]["stderr"]
        ok = (record["termination"] == "deadline" and "BEFORE-TF" in feedback and "AFTER-TF" not in feedback and record["run"]["bytes_after_deadline"] > 0)
        return ok, {"feedback": feedback[-200:], "bytes_after_deadline": record["run"]["bytes_after_deadline"], "record": self.brief(record)}

    def early_exit_orphan(self):
        solver = r'''
import os, time, json
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        time.sleep(1000)
    os._exit(0)
print(json.dumps({"exiting": True}), flush=True)
'''
        record = self.run(solver, limits=RunLimits(wall_seconds=60))
        run = record["run"]
        return record["termination"] == "finished" and run["wall_seconds"] < 5 and run["cleanup_verified"], {"record": self.brief(record)}

    def cpu_one_core(self):
        """Adversarial: the solver widens its own affinity to every CPU; the cpuset keeps it on one."""
        solver = r'''
import json, os, time, multiprocessing as mp
def spin(seconds, q):
    end = time.monotonic() + seconds; n = 0
    while time.monotonic() < end:
        n += 1
    q.put(n)
if __name__ == "__main__":
    before = sorted(os.sched_getaffinity(0))
    try:
        os.sched_setaffinity(0, set(range(os.cpu_count()))); widened = "accepted"
    except OSError as e:
        widened = str(e)
    after = sorted(os.sched_getaffinity(0))
    ctx = mp.get_context("fork"); q = ctx.Queue()
    p = ctx.Process(target=spin, args=(2.0, q)); p.start(); p.join(); single = q.get()
    procs = [ctx.Process(target=spin, args=(2.0, q)) for _ in range(4)]
    for p in procs: p.start()
    for p in procs: p.join()
    total = sum(q.get() for _ in procs)
    print(json.dumps({"before": before, "widen": widened, "after": after, "single": single, "four_total": total, "ratio": total / single}))
'''
        record = self.run(solver, limits=RunLimits(wall_seconds=60))
        out = last_json(record["run"]["stdout"])
        return out is not None and out["ratio"] < 1.5 and out["after"] == out["before"] and len(out["after"]) == 1, {"solver": out, "record": self.brief(record)}

    # ----------------------------------------------------------------------------------------------- publication

    def publish_hook(self):
        control = Path(tempfile.mkdtemp(prefix="dgxl-pub-", dir=self.ex.base_dir))
        path = control / "publish.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))
        server.listen(4)
        received = []

        def serve():
            while True:
                try:
                    conn, _ = server.accept()
                except OSError:
                    return
                with conn:
                    data = conn.recv(65536)
                    received.append(data)
                    conn.sendall(b"ack:" + data)

        threading.Thread(target=serve, daemon=True).start()
        solver = r'''
import json, os, socket
path = os.environ["DECOMPGYM_PUBLISH"]
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(path)
s.sendall(b'{"objective": 42}'); reply = s.recv(100); s.close()
try:
    os.unlink(path); removed = True
except OSError:
    removed = False
print(json.dumps({"path": path, "reply": reply.decode(), "removed_socket": removed}))
'''
        try:
            record = self.run(solver, publish=path)
        finally:
            server.close()
        out = last_json(record["run"]["stdout"])
        ok = record["termination"] == "finished" and out and out["reply"] == 'ack:{"objective": 42}' and not out["removed_socket"] and received
        return ok, {"solver": out, "record": self.brief(record)}

    def supervisor_end_to_end(self):
        """The supervisor and its client through the sandbox: eligible before T_f, late after T_f, deadline kill."""
        from src.experiments.xl_supervisor import Supervisor, write_client
        work = Path(tempfile.mkdtemp(prefix="dgxl-sup-", dir=self.ex.base_dir))
        client = write_client(work)
        sup = Supervisor(work / "store", 4.0).start()
        solver = r'''
import json, signal, time
from decompgym_publish import publish_incumbent
signal.signal(signal.SIGTERM, signal.SIG_IGN)
first = publish_incumbent({"objective": 10, "x": [1, 0]})
time.sleep(5)
late = publish_incumbent({"objective": 5, "x": [0, 1]})
print(json.dumps({"first": first, "late": late}), flush=True)
time.sleep(100)
'''
        record = self.run(solver, data=self.data({"x": 1}), limits=RunLimits(wall_seconds=4, grace_seconds=3), publish=sup.socket_path,
                          on_start=sup.start_clock, client_modules=[client])
        published = sup.stop()
        ok = (record["termination"] == "deadline" and published["counts"]["eligible"] == 1 and published["counts"]["late"] == 1
              and len(published["incumbents"]) == 1 and published["incumbents"][0]["t"] < 4.0 and record["run"]["cleanup_verified"]
              and "first" not in record["feedback"]["stdout"])  # the solver printed after T_f: not in the feedback
        return ok, {"record": self.brief(record), "supervisor_counts": published["counts"],
                    "incumbents": [{k: v for k, v in entry.items() if k in ("seq", "t", "bytes")} for entry in published["incumbents"]]}

    # ---------------------------------------------------------------------------------------------------- build

    LIB = r'''
#include "core.hpp"
#include <string>
#include <vector>
#include <numeric>
extern "C" long dg_sum(const long* v, int n) { std::vector<long> a(v, v + n); return std::accumulate(a.begin(), a.end(), 0L); }
extern "C" int dg_strlen(const char* s) { return (int)std::string(s).size(); }
'''
    HEADER = "#pragma once\nextern \"C\" long dg_sum(const long* v, int n);\n"
    MAIN = '#include <iostream>\n#include "core.hpp"\nint main() { long v[3] = {1, 2, 3}; std::cout << "core:" << dg_sum(v, 3) << std::endl; return 0; }\n'
    CTYPES = r'''
import ctypes, json, os, subprocess
lib = ctypes.CDLL(os.environ["DECOMPGYM_CORE"])
lib.dg_sum.restype = ctypes.c_long
arr = (ctypes.c_long * 4)(1, 2, 3, 4)
out = {"sum": lib.dg_sum(arr, 4), "env": {k: v for k, v in os.environ.items() if k.startswith("DECOMPGYM")}}
import numpy, gurobipy
m = gurobipy.Model(); x = m.addVar(lb=1); m.setObjective(x); m.Params.OutputFlag = 0; m.optimize(); out["gurobi"] = m.ObjVal
if "DECOMPGYM_CORE_BIN" in os.environ:
    r = subprocess.run([os.environ["DECOMPGYM_CORE_BIN"]], capture_output=True, text=True)
    out["bin"] = [r.returncode, r.stdout.strip(), r.stderr.strip()[-200:]]
maps = open("/proc/self/maps").read()
out["libstdcxx"] = sorted({line.split()[-1] for line in maps.splitlines() if "libstdc++" in line})
try:
    open(os.environ["DECOMPGYM_CORE"], "ab"); out["core_writable"] = True
except OSError:
    out["core_writable"] = False
print(json.dumps(out))
'''

    def build_lib_only(self):
        record = self.run(self.CTYPES, cpp={"core.cpp": self.LIB, "core.hpp": self.HEADER})
        out = last_json(record["run"]["stdout"]) if record["run"] else None
        steps = record["build"]["steps"]
        ok = (record["termination"] == "finished" and out and out["sum"] == 10 and "bin" not in out and out["gurobi"] == 1.0
              and set(record["build"]["targets"]) == {"libcore.so"} and not out["core_writable"]
              and all((step.get("memory_cgroup") or {}).get("memory_max") == 4 * GiB for step in steps))
        return ok, {"solver": out, "build": {k: record["build"][k] for k in ("status", "detail", "seconds", "targets", "sources", "source_audit")},
                    "steps": [{k: s.get(k) for k in ("target", "argv", "exit_code", "seconds", "maxrss_kib", "memory_cgroup", "needed")} for s in steps]}

    def build_with_main(self):
        record = self.run(self.CTYPES, cpp={"core.cpp": self.LIB, "core.hpp": self.HEADER, "main.cpp": self.MAIN})
        out = last_json(record["run"]["stdout"]) if record["run"] else None
        ok = record["termination"] == "finished" and out and out["sum"] == 10 and out.get("bin", [None, None])[1] == "core:6" and set(record["build"]["targets"]) == {"libcore.so", "core"}
        return ok, {"solver": out, "build": {k: record["build"][k] for k in ("status", "detail", "seconds", "targets")}}

    def build_load_order(self):
        solver = r'''
import ctypes, json, os
lib = ctypes.CDLL(os.environ["DECOMPGYM_CORE"])
import numpy, gurobipy
m = gurobipy.Model(); m.Params.OutputFlag = 0; x = m.addMVar(3, lb=numpy.arange(3)); m.setObjective(x.sum()); m.optimize()
print(json.dumps({"obj": m.ObjVal}))
'''
        record = self.run(solver, cpp={"core.cpp": self.LIB, "core.hpp": self.HEADER})
        out = last_json(record["run"]["stdout"]) if record["run"] else None
        return record["termination"] == "finished" and out and out["obj"] == 3.0, {"solver": out, "record": self.brief(record)}

    def build_error(self):
        record = self.run("print('unused')", cpp={"bad.cpp": "int f( { return 0; }\n"})
        feedback = record["feedback"]["stderr"]
        ok = record["termination"] == "build_error" and record["run"] is None and "error" in feedback and len(feedback.encode()) <= 20_000
        return ok, {"detail": record["detail"], "feedback_head": feedback[:300], "build_seconds": record["build"]["seconds"]}

    def build_timeout(self):
        source = "constexpr long work(long n) { long s = 0; for (long i = 0; i < n; ++i) s = (s * 31 + i) % 1000003; return s; }\n"
        source += "\n".join(f"static_assert(work({200000 + i}) >= 0);" for i in range(400))
        record = self.run("print('unused')", cpp={"slow.cpp": source}, build_limits=BuildLimits(budget_seconds=6))
        step = record["build"]["steps"][-1] if record["build"]["steps"] else {}
        ok = record["termination"] == "build_error" and "time limit" in record["detail"] and record["build"]["seconds"] < 9 and step.get("cleanup_verified")
        return ok, {"detail": record["detail"], "build_seconds": record["build"]["seconds"], "step": {k: step.get(k) for k in ("seconds", "kill_reason", "memory_cgroup")}}

    def build_output_cap(self):
        source = "\n".join(f"int v{i} = undefined_symbol_{i};" for i in range(40000))
        record = self.run("print('unused')", cpp={"spam.cpp": source}, build_limits=BuildLimits(output_cap_bytes=1 * MiB))
        large = self.run("print('unused')", cpp={"spam.cpp": "\n".join(f"int v{i} = undefined_symbol_{i};" for i in range(3000))})
        ok = (record["termination"] == "build_error" and "output exceeded" in record["detail"] and len(record["feedback"]["stderr"].encode()) <= 20_000
              and large["termination"] == "build_error" and len(large["feedback"]["stderr"].encode()) <= 20_000 and large["build"]["steps"][0]["output_bytes"] > 20_000)
        return ok, {"capped": record["detail"], "uncapped_output_bytes": large["build"]["steps"][0]["output_bytes"]}

    def build_file_size(self):
        source = "static char big[200 << 20] = {1};\nextern \"C\" char at(long i) { return big[i]; }\n"
        record = self.run("print('unused')", cpp={"big.cpp": source}, build_limits=BuildLimits(file_size_bytes=64 * MiB))
        return record["termination"] == "build_error", {"detail": record["detail"], "feedback_tail": record["feedback"]["stderr"][-200:]}

    def build_memory_step(self):
        """The Slurm step's memory cgroup bounds the compiler tree: a 1 GiB step OOM-kills a compile that a per-process 16 GiB limit allows."""
        source = "".join(f"template<int N, int M> struct G{k} {{ static constexpr long v = G{k}<N-1, M>::v + G{k}<N, M-1>::v + N * M + {k}; }};\n"
                         f"template<int M> struct G{k}<0, M> {{ static constexpr long v = M; }};\n"
                         f"template<int N> struct G{k}<N, 0> {{ static constexpr long v = N; }};\n"
                         f"extern \"C\" long slow{k}() {{ return G{k}<440, 440>::v; }}\n" for k in range(4))
        record = self.run("print('unused')", cpp={"mem.cpp": source},
                          build_limits=BuildLimits(memory_bytes=1 * GiB, address_space_bytes=16 * GiB, budget_seconds=110))
        step = record["build"]["steps"][-1] if record["build"]["steps"] else {}
        ok = record["termination"] == "build_error" and "kernel OOM" in record["detail"] and (step.get("memory_cgroup") or {}).get("memory_max") == 1 * GiB
        return ok, {"detail": record["detail"], "seconds": record["build"]["seconds"], "step": {k: step.get(k) for k in ("memory_cgroup", "oom_kills", "signal", "maxrss_kib")}}

    def build_include_escape(self):
        targets = [os.path.join(HOME, ".bashrc"), os.path.join(HOME, ".ssh/known_hosts"), str(ROOT / "README.md"), str(self.data_dir),
                   "/usr/local/pace-apps/manual/packages/gurobi/13.0.1/license/gurobi.lic", "/etc/slurm/slurm.conf", "/work/data.json", "/run/munge",
                   "/usr/bin/bash", "/bin/sh", "/usr/bin/python3", "/usr/libexec/gcc", "/usr/share/zoneinfo", "/usr/lib/gcc"]
        source = "\n".join(f'#if __has_include("{path}")\nextern "C" int visible_{i} = 1;\n#else\nextern "C" int hidden_{i} = 1;\n#endif' for i, path in enumerate(targets))
        solver = ("import ctypes, json, os\nlib = ctypes.CDLL(os.environ['DECOMPGYM_CORE'])\nseen = {}\n"
                  f"for i in range({len(targets)}):\n"
                  "    for kind in ('visible', 'hidden'):\n"
                  "        try:\n            ctypes.c_int.in_dll(lib, f'{kind}_{i}'); seen[i] = kind\n        except ValueError:\n            pass\n"
                  "print(json.dumps(seen))\n")
        record = self.run(solver, cpp={"inc.cpp": source})
        seen = last_json(record["run"]["stdout"]) if record["run"] else None
        ok = record["termination"] == "finished" and seen is not None and all(kind == "hidden" for kind in seen.values()) and len(seen) == len(targets)
        return ok, {"compiler_view": {targets[int(i)]: kind for i, kind in (seen or {}).items()}}

    def build_source_audit(self):
        """Informational C++ audit (policy 2026-11-04): findings are recorded, no build is rejected for them."""
        cases = {"thread": "#include <thread>\nint a;\n", "pthread": "#include <pthread.h>\nint a;\n", "omp_pragma": "#pragma omp parallel for\nint a;\n",
                 "omp_Pragma": '_Pragma("omp parallel")\nint a;\n', "posix": "#include <unistd.h>\nint a;\n", "computed": "#define H <vector>\n#include H\nint a;\n",
                 "spliced": "#inc\\\nlude <thread>\nint a;\n", "digraph": "%:include <thread>\nint a;\n", "bom": "\ufeff#include <thread>\nint a;\n",
                 "headerless_pthread": 'extern "C" int pthread_create(void*, const void*, void* (*)(void*), void*);\n'
                                       'static void* run(void*) { return nullptr; }\nextern "C" int go() { long t; return pthread_create(&t, nullptr, run, nullptr); }\n'}
        results = {}
        for label, source in cases.items():
            record = self.run("print(1)", cpp={"x.cpp": source})
            audit = record["source_audit"]["cpp"] or {}
            native = (record["build"] or {}).get("native", {}).get("libcore.so", {})
            results[label] = {"termination": record["termination"], "findings": audit.get("findings", [])[:3], "notable_imports": native.get("notable_imports", [])[:5]}
        ok = all(entry["termination"] == "finished" for entry in results.values()) and \
            all(entry["findings"] for label, entry in results.items() if label != "headerless_pthread") and \
            any("pthread_create" in name for name in results["headerless_pthread"]["notable_imports"])
        return ok, results

    def build_incbin(self):
        """Build artifacts can embed only bytes reachable in the narrowed build mount (the compiler tree, /usr/include,
        /usr/lib64, /usr/bin/as and /usr/bin/ld: permitted payload); shells, interpreters and other tools are not reachable."""
        def incbin(path):
            source = 'asm(".section .rodata\\n.globl blob\\nblob:\\n.incbin \\"%s\\"\\n.previous");\nint a;\n' % path
            return self.run("print(1)", cpp={"blob.cpp": source})
        blocked = {}
        for tool in ("/usr/bin/bash", "/bin/sh", "/usr/bin/python3", "/usr/bin/perl", "/usr/libexec/gcc/x86_64-redhat-linux/11/cc1",
                     "/usr/bin/gcc", self.ex.python_facts["purelib"] + "/gurobipy/.libs/libgurobi130.so"):
            record = incbin(tool)
            blocked[tool] = {"termination": record["termination"], "detail": record["detail"][:80],
                             "assembler": next((line for line in record["feedback"]["stderr"].splitlines() if "incbin" in line or "Error" in line), "")[:160]}
        permitted = incbin("/usr/bin/as")
        ok = all(entry["termination"] == "build_error" for entry in blocked.values()) and permitted["termination"] == "finished"
        return ok, {"blocked": blocked, "permitted_as": {"termination": permitted["termination"], "size": (permitted["build"]["targets"].get("libcore.so") or {}).get("size")},
                    "build_mounts": self.ex.environment["build_mounts"]}

    def track_policy(self):
        workspace = Path(tempfile.mkdtemp(prefix="dgxl-track-", dir=self.ex.base_dir))
        (workspace / "cpp").mkdir()
        (workspace / "solver.py").write_text("print(1)\n")
        (workspace / "cpp/a.cpp").write_text("int a;\n")
        rejected = self.ex.execute(workspace, self.data({"x": 1}), RunLimits(wall_seconds=30), track=TRACK_PY)
        python = self.run("import json, os\nprint(json.dumps(sorted(k for k in os.environ if k.startswith('DECOMPGYM'))))\n",
                          track=TRACK_PY, data=self.data({"x": 1}))
        flagged = self.run("import ctypes\nfrom scipy.optimize import milp, linprog\nprint(1)\n", track=TRACK_PY)
        ok = (rejected["termination"] == "build_error" and "not allowed in the python_only track" in rejected["detail"]
              and python["termination"] == "finished" and python["run"]["stdout"].strip() == "[]" and python["run"]["runtime"]["scipy"]
              and any("csgraph" in item for item in python["run"]["runtime_contract"]["allowed"])
              and flagged["termination"] == "crash" and "not available" in flagged["run"]["stderr"]
              and any("milp" in flag for flag in flagged["source_audit"]["python"]["flags"]) and any("ctypes" in flag for flag in flagged["source_audit"]["python"]["flags"]))
        return ok, {"python_only_with_cpp": rejected["detail"], "runtime": python["run"]["runtime"], "python_audit": flagged["source_audit"]["python"]}

    def scipy_masked_matrix_api(self):
        """HiGHS masked: scipy.optimize and its backends do not import; scipy.sparse and gurobipy's matrix API work."""
        solver = r'''
import json
out = {}
for name in ("scipy.optimize", "scipy.optimize._highspy._core", "scipy.optimize._lsap", "scipy.optimize._milp"):
    try:
        __import__(name); out[name] = "imported"
    except ImportError as e:
        out[name] = "ImportError"
import numpy as np, scipy, scipy.sparse as sp, gurobipy as gp
try:
    scipy.optimize; out["attribute"] = "available"
except (AttributeError, ImportError):
    out["attribute"] = "unavailable"
A = sp.csr_matrix(np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]]))
m = gp.Model(); m.Params.OutputFlag = 0
x = m.addMVar(3, lb=0.0)
c = m.addConstr(A @ x >= np.array([4.0, 6.0]))
expr = x.sum() + 2 * x[0] - 0.5 * x[1:].sum()
m.setObjective(x.sum())
m.optimize()
Q = sp.identity(3, format="csr")
m2 = gp.Model(); m2.Params.OutputFlag = 0
y = m2.addMVar(3, lb=-10.0)
m2.setObjective(y @ Q @ y - np.ones(3) @ y)
m2.optimize()
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
rows, cols = min_weight_full_bipartite_matching(sp.csr_matrix(np.array([[4.0, 1.0], [2.0, 3.0]])))
out.update(obj=round(m.ObjVal, 6), duals=[round(v, 6) for v in c.Pi.tolist()], expr=type(expr).__name__, qp=round(m2.ObjVal, 6),
           csgraph_matching=cols.tolist())
print(json.dumps(out))
'''
        record = self.run(solver)
        out = last_json(record["run"]["stdout"]) if record.get("run") else None
        ok = (record["termination"] == "finished" and out is not None and all(out[name] == "ImportError" for name in out if name.startswith("scipy."))
              and out["attribute"] == "unavailable" and out["obj"] == round(10 / 3, 6) and out["qp"] == -0.75 and out["csgraph_matching"] == [1, 0]
              and "scipy.optimize" in record["run"]["masked_modules"])
        return ok, {"solver": out, "masked_modules": record["run"]["masked_modules"], "record": self.brief(record),
                    "stderr": record["run"]["stderr"][-400:]}

    def python_audit_bounded(self):
        """Review 48 repro: a solver whose syntax tree overflows the audit's recursion is a recorded candidate failure."""
        record = self.run("x=" + "1+" * 10000 + "1\n")
        error = record["source_audit"]["python"]["error"]
        ok = bool(error) and "Recursion" in error and record["termination"] in ("crash", "finished") and record["run"] is not None
        return ok, {"audit_error": error, "record": self.brief(record), "stderr": record["run"]["stderr"][-200:]}

    def feedback_cutoff(self):
        """Late output and a late exit between polls: feedback cut by receipt time; the result is a deadline."""
        results = {}
        for label, offset in (("late", 0.004), ("late_15ms", 0.015)):
            control = Path(tempfile.mkdtemp(prefix="dgxl-t0-", dir=self.ex.base_dir))
            path = control / "t0.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(path))
            server.listen(1)
            t0 = {}

            def serve(server=server, t0=t0):
                conn, _ = server.accept()
                with conn:
                    conn.sendall(repr(t0["t0"]).encode())

            threading.Thread(target=serve, daemon=True).start()
            solver = r'''
import os, signal, socket, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)  # the late output is produced whether or not the loop signalled first
s = socket.socket(socket.AF_UNIX); s.connect(os.environ["DECOMPGYM_PUBLISH"]); t0 = float(s.recv(64))
print("EARLY", flush=True)
while time.monotonic() < t0 + 2.0 + OFFSET:
    pass
print("LATE", flush=True); sys.stderr.write("LATE-ERR\n"); sys.stderr.flush()
os._exit(0)
'''.replace("OFFSET", repr(offset))
            record = self.run(solver, limits=RunLimits(wall_seconds=2.0, grace_seconds=1), publish=path, on_start=lambda t, t0=t0: t0.setdefault("t0", t))
            server.close()
            feedback = record["feedback"]["stdout"] + record["feedback"]["stderr"]
            results[label] = {"termination": record["termination"], "detail": record["detail"], "early_in_feedback": "EARLY" in feedback,
                              "late_in_feedback": "LATE" in feedback, "bytes_after_deadline": record["run"]["bytes_after_deadline"],
                              "late_exit": record["run"]["late_exit"]}
        ok = all(entry["termination"] == "deadline" and entry["early_in_feedback"] and not entry["late_in_feedback"] and entry["bytes_after_deadline"] >= 5
                 for entry in results.values())
        return ok, results

    CPP_THREADS = r'''
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

    def cpp_threads_one_cpu(self):
        """Policy (a): the C++ track may use threads; a 4-thread busy loop still gets one CPU of work."""
        solver = "import os, subprocess\nprint(subprocess.run([os.environ['DECOMPGYM_CORE_BIN']], capture_output=True, text=True).stdout.strip())\n"
        record = self.run(solver, cpp={"spin.cpp": self.CPP_THREADS})
        out = last_json(record["run"]["stdout"]) if record.get("run") else None
        imports = (record["build"] or {}).get("native", {}).get("core", {}).get("notable_imports", [])
        ok = record["termination"] == "finished" and out is not None and out["ratio"] < 1.3 and any("pthread_create" in name or "_ZNSt6thread" in name for name in imports)
        return ok, {"solver": out, "notable_imports": imports[:10], "audit_findings": (record["source_audit"]["cpp"] or {}).get("findings"),
                    "record": self.brief(record), "build": (record["build"] or {}).get("detail")}

    def build_preflight_strict(self):
        """The build step's limits are verified inside the step before the compiler; an unmet cap stops the step (no compile)."""
        attempt = self.ex.prepare(manifest_for({"solver.py": b"print(1)", "cpp/a.cpp": b"int a;"}, track=TRACK_CPP), track=TRACK_CPP)
        argv = [self.ex.gcc["gxx"], "-std=c++17", "-O2", "-fPIC", "-shared", "cpp/a.cpp", "-o", "build/libcore.so"]
        try:
            bad = self.ex.compile(attempt, argv, timeout=60, limits=BuildLimits().rlimits(60), tail_bytes=20000, output_cap=1 << 20, step="cap",
                                  memory_bytes=4 * GiB, preflight_memory_cap=1 * GiB)
            produced_after_bad = (attempt.build / "libcore.so").exists()
            good = self.ex.compile(attempt, argv, timeout=60, limits=BuildLimits().rlimits(60), tail_bytes=20000, output_cap=1 << 20, step="ok")
        finally:
            self.ex.cleanup(attempt)
        facts = good.step_cgroup or {}
        ok = (bool(bad.infrastructure) and "exceeds the configured cap" in bad.infrastructure and bad.payload_exit() is None and not produced_after_bad
              and facts.get("ok") and facts.get("memory_max") == 4 * GiB and facts.get("swap_bound") == 0 and len(facts.get("cpuset", [])) == 1
              and all(value == "0" for value in facts.get("oom_group", {}).values()))
        return ok, {"rejected": bad.infrastructure, "accepted_step": facts}

    def forwarder_half_close(self):
        """Reply + FIN delivered on every stream; client half-close; 6 MiB through backpressure intact."""
        def serve_with(handler):
            server = socket.socket()
            server.bind(("127.0.0.1", 0))
            server.listen(64)
            server.settimeout(0.2)
            stop = threading.Event()

            def loop():
                while not stop.is_set():
                    try:
                        conn, _ = server.accept()
                    except OSError:
                        continue
                    threading.Thread(target=handler, args=(conn,), daemon=True).start()
            threading.Thread(target=loop, daemon=True).start()
            return server.getsockname()[1], stop

        def read_all(sock):
            data = b""
            for block in iter(lambda: sock.recv(65536), b""):
                data += block
            return data

        def reply_and_close(conn):
            with conn:
                conn.recv(64)
                conn.sendall(b"R" * 29)

        def echo_until_eof(conn):
            with conn:
                for block in iter(lambda: conn.recv(65536), b""):
                    conn.sendall(block)

        directory = Path(tempfile.mkdtemp(prefix="dgxl-hc-", dir="/tmp"))
        port, stop = serve_with(reply_and_close)
        fw = LicenseForwarder(directory / "a.sock", ["127.0.0.1"], port).start()
        replies = 0
        for _ in range(30):
            client = socket.socket(socket.AF_UNIX)
            client.settimeout(5)
            client.connect(str(fw.socket_path))
            client.sendall(b"  433 13 request")
            replies += read_all(client) == b"R" * 29
            client.close()
        reply_stats = fw.close()
        stop.set()
        port, stop = serve_with(echo_until_eof)
        fw = LicenseForwarder(directory / "b.sock", ["127.0.0.1"], port, max_bytes_per_stream=1 << 30, max_bytes_total=1 << 30).start()
        client = socket.socket(socket.AF_UNIX)
        client.settimeout(5)
        client.connect(str(fw.socket_path))
        client.sendall(b"  433 half-close")
        client.shutdown(socket.SHUT_WR)
        half = read_all(client)
        client.close()
        payload = b"  433 " + os.urandom(6 << 20)
        client = socket.socket(socket.AF_UNIX)
        client.settimeout(20)
        client.connect(str(fw.socket_path))
        sender = threading.Thread(target=lambda: (client.sendall(payload), client.shutdown(socket.SHUT_WR)), daemon=True)
        sender.start()
        received = read_all(client)
        sender.join(20)
        client.close()
        big_stats = fw.close()
        stop.set()
        ok = (replies == 30 and half == b"  433 half-close" and hashlib.sha256(received).digest() == hashlib.sha256(payload).digest()
              and reply_stats["shutdown_confirmed"] and big_stats["shutdown_confirmed"])
        return ok, {"reply_then_fin": f"{replies}/30", "half_close_echo": half.decode(errors="replace"), "backpressure_bytes": len(received),
                    "reply_stats": reply_stats, "big_stats": big_stats}

    # ------------------------------------------------------------------------------------------ pinned conventions

    CONV_CPP = r'''
#include <cmath>
extern "C" void dg_distances(const double* x, const double* y, int n, long long* out) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double dx = x[i] - x[j], dy = y[i] - y[j];
            out[(long long)i * n + j] = (long long)std::nearbyint(1000.0 * std::sqrt(dx * dx + dy * dy));
        }
}
extern "C" double dg_round(double v) { return std::nearbyint(v); }
'''
    CONV_SOLVER = r'''
import ctypes, hashlib, json, math, os
import numpy as np
lib = ctypes.CDLL(os.environ["DECOMPGYM_CORE"])
lib.dg_round.restype = ctypes.c_double; lib.dg_round.argtypes = [ctypes.c_double]
data = json.load(open("data.json"))
ties = [k + 0.5 for k in range(-2000, 2000)]
tie_mismatch = sum(1 for v in ties if lib.dg_round(v) != round(v) or lib.dg_round(v) != float(np.rint(v)))
results = {}
for name, (xs, ys) in data["instances"].items():
    x = np.ascontiguousarray(xs, dtype=np.float64); y = np.ascontiguousarray(ys, dtype=np.float64); n = len(x)
    out = np.zeros(n * n, dtype=np.int64)
    lib.dg_distances(x.ctypes.data_as(ctypes.c_void_p), y.ctypes.data_as(ctypes.c_void_p), n, out.ctypes.data_as(ctypes.c_void_p))
    dx = x[:, None] - x[None, :]; dy = y[:, None] - y[None, :]
    ref = np.rint(1000.0 * np.sqrt(dx * dx + dy * dy)).astype(np.int64).ravel()
    py = np.array([round(1000.0 * math.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)) for i in range(n) for j in range(n)], dtype=np.int64)
    results[name] = {"n": n, "cpp_vs_numpy": int((out != ref).sum()), "cpp_vs_round": int((out != py).sum()),
                     "sha256": hashlib.sha256(out.tobytes()).hexdigest(), "halves": int((np.abs(1000.0 * np.sqrt(dx * dx + dy * dy) % 1 - 0.5) == 0).sum())}
print(json.dumps({"tie_mismatch": tie_mismatch, "results": results}))
'''

    def conventions(self):
        from src.cg.pdptw.a_data import load_pdptw
        import numpy as np
        expected = json.loads(EXPECTED_PDPTW_FILE.read_text())
        found = {str(path.relative_to(PDPTW_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for tier, size in (("Li400", 4), ("Li600", 6)) for family in ("lr1", "lrc1") for path in (PDPTW_DIR / tier).glob(f"{family}_{size}_*.vrp")}
        set_ok = found == expected and len(found) == 40
        instances, reference = {}, {}
        for name in sorted(found):
            inst = load_pdptw(PDPTW_DIR / name)
            instances[name] = [inst.coords[:, 0].tolist(), inst.coords[:, 1].tolist()]
            reference[name] = hashlib.sha256(inst.distances().astype(np.int64).tobytes()).hexdigest()
        grid = [(k / 16.0, 0.0) for k in range(0, 160)] + [(0.0, k / 16.0) for k in range(1, 160)] + [(3 * k / 64.0, 4 * k / 64.0) for k in range(1, 100)]
        random_points = np.random.default_rng(20261103).uniform(0, 100, size=(300, 2)).round(3).tolist()
        for label, points in (("generated_ties", grid), ("generated_random", random_points)):
            arr = np.array(points, dtype=float)
            instances[label] = [arr[:, 0].tolist(), arr[:, 1].tolist()]
            dx = arr[:, None, 0] - arr[None, :, 0]
            dy = arr[:, None, 1] - arr[None, :, 1]
            reference[label] = hashlib.sha256(np.rint(1000.0 * np.sqrt(dx * dx + dy * dy)).astype(np.int64).tobytes()).hexdigest()
        record = self.run(self.CONV_SOLVER, data=self.data({"instances": instances}, "conv.json"), cpp={"conv.cpp": self.CONV_CPP}, limits=RunLimits(wall_seconds=600))
        out = last_json(record["run"]["stdout"]) if record["run"] else None
        if out is None:
            return False, {"record": self.brief(record), "stderr": (record["run"] or {}).get("stderr", "")[-1000:], "build": record["build"]["detail"]}
        mismatches = {name: r for name, r in out["results"].items() if r["cpp_vs_numpy"] or r["cpp_vs_round"] or r["sha256"] != reference[name]}
        ok = set_ok and not mismatches and out["tie_mismatch"] == 0 and out["results"]["generated_ties"]["halves"] > 0 and len(out["results"]) == 42
        return ok, {"pdptw_set_matches_frozen_hashes": set_ok, "pdptw_files": len(found), "pairs": sum(r["n"] ** 2 for r in out["results"].values()),
                    "tie_mismatch": out["tie_mismatch"], "exact_halves_in_generated": out["results"]["generated_ties"]["halves"], "mismatches": mismatches}

    def cleanup_check(self):
        names = {}
        for entry in os.listdir("/proc"):
            if entry.isdigit():
                try:
                    if os.stat(f"/proc/{entry}").st_uid == os.getuid():
                        names[int(entry)] = Path(f"/proc/{entry}/cmdline").read_bytes().replace(b"\0", b" ")[:120].decode(errors="replace")
                except OSError:
                    pass
        leftovers = {pid: cmd for pid, cmd in names.items() if "bwrap" in cmd or "init.py" in cmd or "time.sleep(1000)" in cmd}
        return not leftovers, {"own_processes": len(names), "leftovers": leftovers}


CHECKS = ["delivery_rules", "preflight_limits", "isolation", "curl_diagnostic", "init_signals", "gurobi", "license_filter", "forwarder_budgets",
          "license_attribution", "fork_bomb", "memory_rlimit", "file_size", "scratch_quota", "open_files", "deadline_tree", "freeze_feedback",
          "early_exit_orphan", "cpu_one_core", "publish_hook", "supervisor_end_to_end", "build_lib_only", "build_with_main", "build_load_order",
          "build_error", "build_timeout", "build_output_cap", "build_file_size", "build_memory_step", "build_include_escape", "build_source_audit",
          "build_incbin", "track_policy", "scipy_masked_matrix_api", "python_audit_bounded", "feedback_cutoff", "cpp_threads_one_cpu", "build_preflight_strict",
          "forwarder_half_close", "conventions", "cleanup_check"]
OPT_IN = ["memory_cgroup"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", action="append", default=[], choices=CHECKS + OPT_IN, help="run only these checks (repeatable)")
    parser.add_argument("--cgroup-oom", action="store_true")
    args = parser.parse_args(argv)
    selected = args.only or (CHECKS + (OPT_IN if args.cgroup_oom else []))
    validator = Validator(Path(args.out))
    for name in selected:
        validator.check(name, getattr(validator, name))
    report = validator.report
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["summary"] = {entry["name"]: entry["ok"] for entry in report["checks"]}
    Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    failed = [entry["name"] for entry in report["checks"] if not entry["ok"]]
    print("FAILED: " + ", ".join(failed) if failed else "ALL PASSED", flush=True)
    return 1 if failed or not report["checks"] else 0


if __name__ == "__main__":
    sys.exit(main())
