"""Scored execution of agent-written XL solvers on Phoenix under bubblewrap (protocol v4 §2, §5; Codex reviews 42 #8, 44).

One attempt = (manifest-checked delivery for a track) -> C++ build if the track allows it (`xl_build`) -> one sandboxed
``python solver.py`` run. Tracks: ``python_only`` (paper rules) and ``cpp_allowed`` (PDPTW C++ track); ``execute`` requires one.

Sandbox (``bwrap --unshare-all --as-pid-1``; one per build step and one per run):
- namespaces: new user, PID, network (loopback only), IPC, UTS and cgroup namespaces; ``--die-with-parent``, ``--new-session``;
  the root is a private tmpfs remounted read-only after setup; a seccomp filter (``seccomp_program``) denies the kernel keyring,
  ptrace and other cross-process calls (prlimit64 on any process but the caller), new namespaces and large attack surfaces;
- run filesystem (frozen runtime, read-only): ``/usr/lib64`` and ``/usr/lib/locale`` only (no ``/usr/bin``, no other
  interpreter or toolchain), the solver Python's prefix with ``bin/`` reduced to the interpreter and ``site-packages`` reduced
  to an allowlist (gurobipy, numpy, scipy = ``gurobipy[matrixapi]``; only the dist-info of the imported versions) with
  ``scipy.optimize`` (HiGHS: milp, linprog, linear_sum_assignment) replaced by an ImportError stub (``MASKED_MODULES``), the
  pinned compiler's ``lib64`` in the C++ track, selected ``/etc`` files (synthesized passwd, group, nsswitch.conf, hosts;
  ld.so cache/configuration; localtime), ``/work/{solver.py,data.json,gurobi.env,cpp,build}`` and the control files under
  ``/run/decompgym``. Every compiler, assembler, linker or JIT library found in the mounted trees (`toolchain_inventory`) is
  masked with ``/dev/null``. Build steps mount all of ``/usr`` and the pinned compiler (the compiler is trusted code; the
  candidate controls only its input);
- writable (run): ``/work``, ``/tmp``, ``/dev`` and ``/dev/shm``, all private tmpfs mounts; (build) ``/tmp`` and the
  host-backed ``/work/build``;
- invisible: home and project storage (repository, references, other runs, ``~/.ssh``), host ``/tmp``, ``/run`` (munge),
  ``/sys``, Slurm configuration, host processes, the host Gurobi license file;
- environment: cleared and rebuilt; nothing from the job environment leaks in.

Resource policy (fail closed before every launch, `check_job_limits`): the Slurm job cgroup must give exactly ``cpus``
effective CPUs (affinity cannot widen beyond the cpuset), a memory.max at most ``memory_cap_bytes`` (16 GiB), swap bounded by
``swap_cap_bytes`` (0: the job's memory.swap.max, or no swap on the node) and memory.oom.group = 0 up to the limiting cgroup.
Memory and scratch form one combined budget: process memory plus everything written to the tmpfs mounts (/work, /tmp, /dev,
/dev/shm) is charged to the job's memory cgroup, a hard kernel bound with no swap. ``scratch_quota_bytes`` is a polled
(0.5 s) soft limit on the tmpfs total, not a hard disk quota. No RLIMIT_AS by default, so memory exhaustion surfaces as a
kernel OOM kill. Inside, PID 1 (`xl_sandbox_init`) sets RLIMIT_NPROC (sandbox tasks only), RLIMIT_FSIZE, RLIMIT_NOFILE,
RLIMIT_CPU (backstop) and RLIMIT_CORE = 0, and an OOM score of 1000. C++ builds run in a separate Slurm step
(``srun --overlap --mem=4G``) whose memory cgroup bounds the compiler tree plus its tmpfs scratch (`xl_build`).

Clock: T_f runs on the executor's monotonic clock from the bwrap spawn (identical for every arm; the measured start-up
overhead until the payload starts is recorded). The stdout/stderr readers stop admitting feedback at the first block received
at or after T_f (framework receipt time), independently of the polling loop; only the late byte count is recorded. At T_f
the executor sends SIGTERM to every process of the sandbox's PID namespace except PID 1 and, after the grace period, SIGKILL to
PID 1, which makes the kernel kill the whole namespace. A payload whose exit (the init's report, same clock) is at or after
T_f is a ``deadline`` even if it ended before the executor's signal.

Audits (informational, for the method audit): the Python import audit and the C++ include audit run in a child process with
memory, CPU and wall-clock bounds (`xl_audit.run_bounded_audits`); a parser failure is recorded, never raised. The C++ track
may use anything reachable inside the sandbox (threads, pthread, dlopen); fairness comes from OS enforcement.

Campaign tooling: when a candidate is OOM-killed the executor survives and records a kernel-confirmed ``oom``; Slurm then
labels the job OUT_OF_MEMORY. Persist records with `persist_record`; the record is authoritative and such jobs must not be
discarded, requeued or rescored.

Termination: finished, deadline, oom, crash, build_error, infrastructure. ``oom`` requires a kernel OOM-kill event in the
job cgroup during the run; a candidate's own MemoryError is a crash (noted). ``infrastructure`` requires framework evidence:
a launch or setup failure before the payload started, a failed independent Gurobi probe after a license-flavoured crash, or
sandbox processes surviving cleanup. Once the payload has started, a missing exit report is a crash.

License tunnel: the sandbox has no network except loopback. `LicenseForwarder` (outside) listens on a Unix socket bound into the
sandbox and opens, per accepted stream, one TCP connection to the token server's addresses resolved once from the host license
(TOKENSERVER, PORT). It admits only streams whose first bytes are a token request (``"  433 "``; a status query such as
GRBlisttokens starts with ``"  16-327"`` and is dropped before any upstream connection), and enforces a stream rate, a
concurrency cap and per-stream and per-attempt byte budgets with non-blocking I/O and confirmed shutdown. Inside, the init
starts a relay on 127.0.0.1:PORT -> that socket, and ``GRB_LICENSE_FILE`` names a two-line client license
(``TOKENSERVER=127.0.0.1``, ``PORT``); no license key or other credential enters the sandbox.

Publication hook: ``publish_socket`` (served by `xl_supervisor`) is bound at ``/run/decompgym/publish.sock`` and announced in
``DECOMPGYM_PUBLISH``; ``on_start(t0)`` hands the supervisor the executor's spawn time; ``client_modules`` are bound read-only
under ``/run/decompgym/lib`` on ``PYTHONPATH``:

    sup = Supervisor(store_dir, T_f).start()
    record = executor.execute(delivery, data, RunLimits(wall_seconds=T_f), track="python_only", publish_socket=sup.socket_path,
                              on_start=sup.start_clock, client_modules=[write_client(client_dir)])
    published = sup.stop()
"""
from __future__ import annotations

from dataclasses import dataclass, field
import glob
import grp
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import pwd
import re
import selectors
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable

GiB = 1 << 30
MiB = 1 << 20
KiB = 1 << 10
SANDBOX_POLICY = "decompgym-xl-bwrap-v2"
BWRAP = "/usr/bin/bwrap"
TAIL_BYTES = 20_000
WORK = "/work"
CONTROL = "/run/decompgym"
DEFAULT_PACKAGES = ("gurobipy", "numpy", "scipy")  # gurobipy[matrixapi]: its matrix API imports scipy
TRACKS = ("python_only", "cpp_allowed")
TERMINATIONS = ("finished", "deadline", "oom", "crash", "build_error", "infrastructure")
PACE_LICENSE = "/usr/local/pace-apps/manual/packages/gurobi/13.0.1/license/gurobi.lic"
DEFAULT_LICENSE_PORT = 41954
TOKEN_REQUEST_PREFIXES = (b"  433 ",)  # observed first bytes of a Gurobi 13 token request; "  16-327" is a status query
# Phoenix module gcc/12.3.0 (protocol v4 §5); `module show gcc/12.3.0` sets GCC_ROOT to this prefix.
PHOENIX_GCC_ROOT = "/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/gcc-12.3.0-ukkkutsxfl5kpnnaxflpkq2jtliwthfz"
GCC_VERSION = "12.3.0"
RUN_TREES_USR = ("/usr/lib64", "/usr/lib/locale")
# Toolchains and JIT libraries masked in every tree mounted for a run (`toolchain_inventory`).
TOOL_NAME = re.compile(r"(cc1(plus|obj)?|collect2|lto1|lto-wrapper|as|ld|ld\.(bfd|gold|lld)|gcc|g\+\+|c\+\+|cpp|cc|tcc|gfortran|f951|nvcc|"
                       r"clang(\+\+)?(-\d+)?|go|gofmt|rustc|cargo|javac|java|node|julia|"
                       r".+-(gcc|g\+\+|c\+\+|cc|ld|as|cpp|gfortran)(-\d+(\.\d+)*)?)")
JIT_LIBRARY = re.compile(r"lib(gccjit|LLVM|clang|clang-cpp|tcc|jvm|tinycc)([-.][^/]*)?\.so(\.[0-9.]+)?")
# Diagnostic-only mode that mounts all of /usr (e.g. to show that curl fails): masks for /usr's own toolchains.
USR_FILE_MASKS = ("/usr/bin/gcc", "/usr/bin/g++", "/usr/bin/c++", "/usr/bin/cpp", "/usr/bin/gfortran", "/usr/bin/as", "/usr/bin/ld.bfd",
                  "/usr/bin/ld.gold", "/usr/bin/x86_64-redhat-linux-*", "/usr/bin/clang*", "/usr/bin/tcc", "/usr/bin/gcc-*", "/usr/bin/g++-*",
                  "/usr/bin/go", "/usr/bin/gofmt", "/usr/bin/java", "/usr/bin/javac")
USR_DIR_MASKS = ("/usr/libexec/gcc", "/usr/lib/gcc", "/usr/lib/golang", "/usr/lib/jvm")
GUROBI_CLUES = ("gurobi", "gurobipy", "license", "licence")  # the paper executor's trigger for its independent probe
MEMORY_MARKERS = ("MemoryError", "std::bad_alloc", "Out of memory", "out of memory", "Cannot allocate memory")
SIGNAL_NOTES = {signal.SIGXFSZ: "file size limit", signal.SIGXCPU: "CPU time limit", signal.SIGKILL: "killed", signal.SIGSEGV: "segmentation fault",
                signal.SIGABRT: "abort", signal.SIGBUS: "bus error", signal.SIGFPE: "floating-point exception"}
INIT_SOURCE = Path(__file__).with_name("xl_sandbox_init.py")
# Python modules hidden from the solver (caller decision 2026-11-04): HiGHS (milp, linprog, linear_sum_assignment) lives in
# scipy.optimize; the directory is replaced by an empty tmpfs whose __init__.py raises ImportError. scipy.sparse stays.
MASKED_MODULES = ("scipy.optimize",)
RUNTIME_CONTRACT = {
    "packages": "Python standard library, gurobipy, numpy, scipy (gurobipy[matrixapi])",
    "masked": "scipy.optimize (HiGHS: milp, linprog; linear_sum_assignment) raises ImportError",
    "allowed": ["scipy.sparse and scipy.sparse.linalg",
                "scipy.sparse.csgraph, including min_weight_full_bipartite_matching and maximum_bipartite_matching (not HiGHS; "
                "declared allowed 2026-11-04 for agents, controls and calibration alike)"],
    "cpp_track": "anything reachable inside the sandbox; build artifacts may embed bytes of files reachable during the build "
                 "(the pinned compiler tree, /usr/include, /usr/lib64, /usr/bin/as and /usr/bin/ld); OS limits apply at run time"}
# Run by the build's Slurm step before bwrap: strict check of the step's own cgroup (memory, swap, cpuset, OOM group).
STEP_PREFLIGHT = """import json, sys
sys.path.insert(0, {directory!r})
import xl_executor as x
try:
    facts = x.check_job_limits(x.job_cgroup(), cpus=1, memory_cap={memory_cap}, swap_cap=0)
    print(json.dumps(dict(facts, ok=True)))
except BaseException as error:
    print(json.dumps({{"ok": False, "error": type(error).__name__ + ": " + str(error)}}))
    sys.exit(1)
"""

# Seccomp filter (x86_64, applied by bwrap after setup): EPERM for the kernel keyring, which is not namespaced (keyctl,
# add_key, request_key); cross-process access (ptrace, process_vm_readv/writev, kcmp, pidfd_getfd, process_madvise,
# migrate_pages, move_pages; prlimit64 on any process other than the caller); large attack surfaces (bpf, perf_event_open,
# userfaultfd, io_uring_*); new namespaces (unshare, setns, clone with any CLONE_NEW* flag; clone3 gets ENOSYS so that glibc
# falls back to clone). Every x32-ABI syscall and every foreign architecture is denied too.
SECCOMP_DENY = {"keyctl": 250, "add_key": 248, "request_key": 249, "ptrace": 101, "process_vm_readv": 310, "process_vm_writev": 311,
                "kcmp": 312, "pidfd_getfd": 438, "process_madvise": 440, "migrate_pages": 256, "move_pages": 279, "bpf": 321,
                "perf_event_open": 298, "userfaultfd": 323, "io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427,
                "unshare": 272, "setns": 308}
SECCOMP_CLONE, SECCOMP_CLONE3, SECCOMP_PRLIMIT64 = 56, 435, 302
CLONE_NAMESPACE_FLAGS = 0x7E020000  # NEWNS | NEWCGROUP | NEWUTS | NEWIPC | NEWUSER | NEWPID | NEWNET


def seccomp_program(deny: dict = SECCOMP_DENY) -> bytes:
    """Classic BPF program (struct sock_filter[]) for bwrap --seccomp."""
    LD, JEQ, JGE, JSET, RET = 0x20, 0x15, 0x35, 0x45, 0x06
    ALLOW, EPERM, ENOSYS = 0x7FFF0000, 0x00050000 | 1, 0x00050000 | 38
    code: list = []

    def emit(op, k=0, true=None, false=None):
        code.append([op, true, false, k])

    emit(LD, 4)                                   # seccomp_data.arch
    emit(JEQ, 0xC000003E, false="deny")
    emit(LD, 0)                                   # seccomp_data.nr
    emit(JGE, 0x40000000, true="deny")            # x32 ABI
    emit(JEQ, SECCOMP_CLONE3, true="enosys")
    emit(JEQ, SECCOMP_CLONE, true="clone")
    emit(JEQ, SECCOMP_PRLIMIT64, true="prlimit")
    for number in sorted(set(deny.values())):
        emit(JEQ, number, true="deny")
    labels = {"allow": len(code)}
    emit(RET, ALLOW)
    labels["clone"] = len(code)
    emit(LD, 16)                                  # args[0] (clone flags), low 32 bits
    emit(JSET, CLONE_NAMESPACE_FLAGS, true="deny")
    emit(RET, ALLOW)
    labels["prlimit"] = len(code)
    emit(LD, 16)                                  # args[0] (pid), low 32 bits: only 0 (the caller itself)
    emit(JEQ, 0, false="deny")
    emit(RET, ALLOW)
    labels["deny"] = len(code)
    emit(RET, EPERM)
    labels["enosys"] = len(code)
    emit(RET, ENOSYS)
    words = []
    for index, (op, true, false, k) in enumerate(code):
        jt = labels[true] - index - 1 if true else 0
        jf = labels[false] - index - 1 if false else 0
        if not (0 <= jt < 256 and 0 <= jf < 256):
            raise ValueError("seccomp jump out of range")
        words.append(struct.pack("<HBBI", op, jt, jf, k))
    return b"".join(words)


class InfrastructureError(RuntimeError):
    """The sandbox, tunnel or host environment failed or is misconfigured; nothing was scored."""


def persist_record(path: str | Path, record: dict) -> Path:
    """Durably write an attempt record (temporary file, fsync, rename, directory fsync).

    Campaign tooling: a Slurm job whose state is OUT_OF_MEMORY may contain correctly scored attempts. When a candidate is
    OOM-killed in its sandbox the executor survives (OOM score 1000 for the sandbox, memory.oom.group = 0), records
    ``termination: oom`` (kernel-confirmed) and continues; Slurm still labels the job OUT_OF_MEMORY. The persisted record is
    authoritative: never discard, requeue or rescore attempts because of the job-level state.
    """
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w") as stream:
        json.dump(record, stream, indent=1, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RunLimits:
    """Per-run limits. CPU and memory are the Slurm job's cgroup (checked before launch); the rest is enforced inside it."""
    wall_seconds: float                              # T_f, from the bwrap spawn
    grace_seconds: float = 2.0                       # SIGTERM -> SIGKILL (output after T_f is not fed back)
    address_space_bytes: int | None = None           # optional RLIMIT_AS per process (off: the memory cgroup is the bound)
    max_processes: int = 128                         # RLIMIT_NPROC: tasks (threads included) of the sandbox user namespace
    file_size_bytes: int = 1 * GiB                   # RLIMIT_FSIZE per file
    open_files: int = 1024                           # RLIMIT_NOFILE
    scratch_quota_bytes: int | None = 4 * GiB       # polled soft limit on /work + /tmp + /dev + /dev/shm (tmpfs)
    cpu_seconds_slack: float = 60.0                  # RLIMIT_CPU = wall + slack (backstop only)
    cpus: int = 1                                    # required effective cpuset size
    memory_cap_bytes: int = 16 * GiB                 # required: job memory.max present and at most this
    swap_cap_bytes: int = 0                          # required: swap usable by the job at most this

    def rlimits(self) -> dict:
        return {"address_space": self.address_space_bytes, "processes": self.max_processes, "file_size": self.file_size_bytes,
                "open_files": self.open_files, "core": 0, "cpu_seconds": int(self.wall_seconds + self.cpu_seconds_slack)}


@dataclass
class SandboxResult:
    exit_code: int | None = None          # bwrap's (outside) report of PID 1's exit code
    signal: int | None = None             # payload signal, from the init's report
    started: float = 0.0                  # executor monotonic time at spawn
    elapsed: float = 0.0                  # spawn -> sandbox gone
    payload_started: float | None = None  # init's monotonic time of the payload's start (same clock)
    deadline_reached: bool = False
    late_exit: bool = False               # ended after T_f before the executor had to signal it
    term_sent: float | None = None
    kill_sent: float | None = None
    kill_reason: str | None = None        # deadline, scratch quota, output cap, infrastructure
    stdout_tail: str = ""                 # feedback: frozen at T_f
    stderr_tail: str = ""
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    bytes_after_deadline: int = 0         # stdout + stderr written after T_f (never fed back)
    events: list = field(default_factory=list)
    oom_kills: int | None = None
    step_cgroup: dict | None = None       # preflight facts of a build's Slurm step (checked inside the step before bwrap)
    peak_scratch_bytes: int | None = None
    cleanup_verified: bool = False
    infrastructure: str | None = None

    def payload_exit(self) -> dict | None:
        return next((event for event in self.events if event.get("event") == "payload_exit"), None)


# --------------------------------------------------------------------------------------------------------- host facts

def require_linux() -> None:
    if platform.system() != "Linux":
        raise InfrastructureError("The XL sandbox executor runs only on Linux (Phoenix compute nodes)")


def bwrap_version(bwrap: str = BWRAP) -> str:
    result = subprocess.run([bwrap, "--version"], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}, timeout=10)
    if result.returncode:
        raise InfrastructureError(f"bwrap --version failed: {result.stderr.strip()}")
    return result.stdout.strip()


def job_cgroup(pid: int | str = "self") -> Path | None:
    try:
        for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                return Path("/sys/fs/cgroup" + line[3:])
    except OSError:
        return None
    return None


def cgroup_chain(cgroup: Path, root: Path = Path("/sys/fs/cgroup")) -> list[Path]:
    chain, node = [], Path(cgroup)
    while node != root and root in node.parents:
        chain.append(node)
        node = node.parent
    return chain


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def oom_kills(cgroup: Path | None) -> int | None:
    if cgroup is None:
        return None
    text = _read(cgroup / "memory.events")
    if text is None:
        return None
    for line in text.splitlines():
        key, _, value = line.partition(" ")
        if key == "oom_kill":
            return int(value)
    return None


def parse_cpuset(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in filter(None, (piece.strip() for piece in text.split(","))):
        low, _, high = part.partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    return cpus


def check_job_limits(cgroup: Path | None, *, cpus: int, memory_cap: int, swap_cap: int, affinity: set[int] | None = None,
                     meminfo: str = "/proc/meminfo", root: Path = Path("/sys/fs/cgroup")) -> dict:
    """Fail closed unless the job cgroup gives exactly `cpus` CPUs, memory.max <= memory_cap, swap <= swap_cap and no OOM group."""
    if cgroup is None or not Path(cgroup).is_dir():
        raise InfrastructureError("no cgroup v2 membership: not inside a Slurm job with cgroup limits")
    chain = cgroup_chain(Path(cgroup), root)
    cpuset_text = next((value for node in chain if (value := _read(node / "cpuset.cpus.effective"))), None)
    if cpuset_text is None:
        raise InfrastructureError("cpuset.cpus.effective is unreadable")
    cpuset = parse_cpuset(cpuset_text)
    affinity = set(os.sched_getaffinity(0)) if affinity is None else set(affinity)
    if len(cpuset) != cpus or not affinity <= cpuset:
        raise InfrastructureError(f"effective cpuset {sorted(cpuset)} (affinity {sorted(affinity)}) differs from the required {cpus} CPU(s)")
    limits = [(int(value), node) for node in chain if (value := _read(node / "memory.max")) not in (None, "max")]
    if not limits:
        raise InfrastructureError("no memory.max on the job's cgroup chain: memory would be unbounded")
    memory_max, limiting = min(limits, key=lambda item: item[0])
    if memory_max > memory_cap:
        raise InfrastructureError(f"job memory.max {memory_max} exceeds the configured cap {memory_cap}")
    swap_total = 0
    for line in Path(meminfo).read_text().splitlines():
        if line.startswith("SwapTotal:"):
            swap_total = int(line.split()[1]) * KiB
    swap_limits = [int(value) for node in chain if (value := _read(node / "memory.swap.max")) not in (None, "max")]
    swap_bound = min([swap_total, *swap_limits])
    if swap_bound > swap_cap:
        raise InfrastructureError(f"swap available to the job ({swap_bound} bytes) exceeds the cap {swap_cap}")
    groups = {str(node): _read(node / "memory.oom.group") for node in chain[:chain.index(limiting) + 1]}
    if any(value is None for value in groups.values()):
        raise InfrastructureError(f"memory.oom.group is missing or unreadable on the job's cgroup chain {groups}: cannot verify OOM isolation")
    if any(value != "0" for value in groups.values()):
        raise InfrastructureError(f"memory.oom.group is set on the job's cgroup chain {groups}: an OOM would kill the executor")
    return {"cgroup": str(cgroup), "cpuset": sorted(cpuset), "affinity": sorted(affinity), "memory_max": memory_max, "memory_limited_at": str(limiting),
            "swap_total": swap_total, "swap_bound": swap_bound, "oom_group": groups}


def parse_license(path: str | Path) -> tuple[list[str], int]:
    """Token servers and port of a Gurobi token license (only these two facts leave the host license file)."""
    servers, port = [], DEFAULT_LICENSE_PORT
    for line in Path(path).read_text().splitlines():
        key, _, value = line.strip().partition("=")
        if key.strip().upper() == "TOKENSERVER":
            servers = [item.strip() for item in value.split(",") if item.strip()]
        elif key.strip().upper() == "PORT" and value.strip():
            port = int(value.strip())
    if not servers:
        raise InfrastructureError(f"{path}: not a token-server license (no TOKENSERVER)")
    return servers, port


def namespace_members(ns_inode: int) -> list[int]:
    """Host PIDs of every process whose PID namespace is ns_inode."""
    members = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            if os.stat(f"/proc/{entry}/ns/pid").st_ino == ns_inode:
                members.append(int(entry))
        except OSError:
            continue
    return members


def scratch_usage(pid: int, paths) -> int | None:
    """Bytes used on the sandbox's tmpfs mounts, read through /proc/PID/root of a sandbox process."""
    total = 0
    try:
        for path in paths:
            status = os.statvfs(f"/proc/{pid}/root{path}")
            total += (status.f_blocks - status.f_bfree) * status.f_frsize
    except OSError:
        return None
    return total


def directory_usage(root: Path) -> int:
    """Allocated bytes below a host directory (no links followed)."""
    total, stack = 0, [str(root)]
    while stack:
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    total += info.st_blocks * 512
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
        except OSError:
            continue
    return total


def symlinked_components(path: str) -> list[tuple[str, str]]:
    """(link, target) for every symlinked component of `path` (to replicate them inside a sandbox)."""
    links, current = [], Path("/")
    for part in Path(path).parts[1:]:
        current = current / part
        if current.is_symlink():
            links.append((str(current), os.readlink(current)))
            current = Path(os.path.realpath(current))
    return links


def toolchain_inventory(trees, skip=()) -> dict:
    """Compilers, assemblers, linkers, other language toolchains and JIT libraries below the given trees (regular files)."""
    skip = {os.path.realpath(path) for path in skip}
    found, scanned = [], 0
    for tree in trees:
        if not os.path.isdir(tree):
            continue
        for directory, subdirs, names in os.walk(tree):
            subdirs[:] = [name for name in subdirs if os.path.realpath(os.path.join(directory, name)) not in skip]
            for name in names:
                scanned += 1
                path = os.path.join(directory, name)
                try:
                    info = os.lstat(path)
                except OSError:
                    continue
                if not os.path.isfile(path) or os.path.islink(path):
                    continue
                if (TOOL_NAME.fullmatch(name) and info.st_mode & 0o111) or JIT_LIBRARY.fullmatch(name):
                    found.append(path)
    return {"trees": list(trees), "files_scanned": scanned, "masked": sorted(found)}


# ------------------------------------------------------------------------------------------------- stream capture

class TailReader(threading.Thread):
    """Drain a pipe; keep the last `keep` bytes and the total count (no disk use).

    The feedback is frozen by `freeze()` or, independently of any polling loop, at the first block received at or after the
    absolute `cutoff` (framework receipt time on the monotonic clock): bytes received after T_f never enter the feedback.
    """

    def __init__(self, fd: int, keep: int, cutoff: float | None = None):
        super().__init__(daemon=True)
        self.fd, self.keep, self.cutoff = fd, keep, cutoff
        self.buffer = bytearray()
        self.total = 0
        self.lock = threading.Lock()
        self.frozen: tuple[bytes, int] | None = None

    def run(self) -> None:
        try:
            while True:
                block = os.read(self.fd, 65536)
                if not block:
                    break
                received = time.monotonic()
                with self.lock:
                    if self.frozen is None and self.cutoff is not None and received >= self.cutoff:
                        self.frozen = (bytes(self.buffer[-self.keep:]) if self.keep else b"", self.total)
                    self.total += len(block)
                    if self.frozen is None:
                        self.buffer += block
                        if len(self.buffer) > 4 * self.keep + 65536:
                            del self.buffer[:-self.keep]
        except OSError:
            pass
        finally:
            os.close(self.fd)

    def freeze(self) -> None:
        with self.lock:
            if self.frozen is None:
                self.frozen = (bytes(self.buffer[-self.keep:]) if self.keep else b"", self.total)

    def text(self) -> str:
        with self.lock:
            data = self.frozen[0] if self.frozen else bytes(self.buffer[-self.keep:]) if self.keep else b""
        return data.decode(errors="replace")

    def after_freeze(self) -> int:
        with self.lock:
            return self.total - self.frozen[1] if self.frozen else 0


def _decode_documents(text: str) -> list[dict]:
    decoder, documents, position = json.JSONDecoder(), [], 0
    while True:
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            return documents
        try:
            value, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            return documents
        documents.append(value if isinstance(value, dict) else {"value": value})


class JsonReader(threading.Thread):
    """Collect JSON documents from a pipe."""

    def __init__(self, fd: int):
        super().__init__(daemon=True)
        self.fd = fd
        self.documents: list[dict] = []
        self.changed = threading.Condition()

    def run(self) -> None:
        raw = b""
        try:
            while True:
                block = os.read(self.fd, 65536)
                if not block:
                    break
                raw += block
                documents = _decode_documents(raw.decode(errors="replace"))
                with self.changed:
                    self.documents = documents
                    self.changed.notify_all()
        except OSError:
            pass
        finally:
            os.close(self.fd)
            with self.changed:
                self.changed.notify_all()

    def poll(self) -> None:
        pass

    def wait_for(self, key: str, timeout: float):
        deadline = time.monotonic() + timeout
        with self.changed:
            while True:
                for document in self.documents:
                    if key in document:
                        return document[key]
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self.is_alive():
                    return None
                self.changed.wait(remaining)


class FileJson:
    """JSON documents appended to a file by a process that cannot inherit our pipes (a Slurm step)."""

    def __init__(self, path: Path, alive: Callable[[], bool]):
        self.path, self.alive = path, alive
        self.documents: list[dict] = []

    def poll(self) -> None:
        try:
            self.documents = _decode_documents(self.path.read_text(errors="replace"))
        except OSError:
            pass

    def start(self) -> None:
        pass

    def join(self, timeout=None) -> None:
        self.poll()

    def wait_for(self, key: str, timeout: float):
        deadline = time.monotonic() + timeout
        while True:
            self.poll()
            for document in self.documents:
                if key in document:
                    return document[key]
            if time.monotonic() >= deadline or not self.alive():
                self.poll()
                return next((document[key] for document in self.documents if key in document), None)
            time.sleep(0.01)


# --------------------------------------------------------------------------------------------------- license tunnel

class LicenseForwarder:
    """Unix socket -> the fixed Gurobi token server; one upstream connection per admitted stream; non-blocking I/O.

    Not a general network path: the destination addresses are resolved once from the host license, the client only sends
    payload bytes, and a stream is admitted only if its first bytes are a token request (`TOKEN_REQUEST_PREFIXES`); anything
    else (e.g. a GRBlisttokens status query) is closed before an upstream connection exists. Budgets per attempt: streams
    per second (token bucket), concurrent streams, total streams, bytes per stream and bytes in total. Measured legitimate
    traffic (Gurobi 13): one stream per environment, 47 + 29 bytes up and 29 down, no keepalive while a token is held (300 s
    test); the 4 KiB per-stream and 1 MiB per-attempt budgets bound whatever a stream that only starts like a token request
    could carry.
    """

    BUFFER_CAP = 256 * KiB

    def __init__(self, socket_path: Path, servers: list[str], port: int, *, allowed_prefixes=TOKEN_REQUEST_PREFIXES, prefix_timeout: float = 10.0,
                 rate_per_second: float = 50.0, burst: int = 200, max_concurrent: int = 64, max_streams: int = 4096,
                 max_bytes_per_stream: int = 4 * KiB, max_bytes_total: int = 1 * MiB, connect_timeout: float = 10.0):
        self.socket_path = Path(socket_path)
        self.servers, self.port = list(servers), int(port)
        self.allowed_prefixes = tuple(allowed_prefixes) if allowed_prefixes else ()
        self.prefix_length = max((len(prefix) for prefix in self.allowed_prefixes), default=0)
        self.prefix_timeout, self.connect_timeout = prefix_timeout, connect_timeout
        self.rate, self.burst = rate_per_second, burst
        self.max_concurrent, self.max_streams = max_concurrent, max_streams
        self.max_bytes, self.max_total = max_bytes_per_stream, max_bytes_total
        self.addresses: list = []
        self.stats = {"destination": [f"{host}:{self.port}" for host in self.servers], "resolved": [], "accepted": 0, "admitted": 0,
                      "filtered": 0, "filtered_prefixes": [], "refused": 0, "rate_delayed": 0, "upstream_failures": 0,
                      "bytes_up": 0, "bytes_down": 0, "capped_stream": 0, "capped_total": 0, "max_concurrent_seen": 0,
                      "shutdown_confirmed": None, "shutdown_seconds": None, "open_at_shutdown": None}
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wake_r, self._wake_w = os.pipe()
        os.set_blocking(self._wake_r, False)
        self._streams: list[dict] = []

    def start(self) -> "LicenseForwarder":
        for host in self.servers:
            try:
                for family, _, _, _, address in socket.getaddrinfo(host, self.port, type=socket.SOCK_STREAM):
                    if (family, address) not in self.addresses:
                        self.addresses.append((family, address))
            except OSError:
                continue
        self.stats["resolved"] = [address[0] for _, address in self.addresses]
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        listener.listen(64)
        listener.setblocking(False)
        self._listener = listener
        self._thread = threading.Thread(target=self._loop, name="license-forwarder", daemon=True)
        self._thread.start()
        return self

    # ---- stream helpers. A stream is {"sock": {side: socket}, "out": {side: bytes queued FOR side}, "eof": {side: side sent FIN},
    # "shut": {side: we sent FIN to side}, ...}; an EOF from one side is propagated as SHUT_WR to the other side only after
    # every byte queued for it has been sent, and the stream closes once both directions have ended and both queues are empty.
    SIDES = {"client": "upstream", "upstream": "client"}

    def _close(self, selector, stream) -> None:
        for sock in stream["sock"].values():
            if sock is None:
                continue
            try:
                selector.unregister(sock)
            except (KeyError, ValueError):
                pass
            sock.close()
        stream["sock"] = {"client": None, "upstream": None}
        if stream in self._streams:
            self._streams.remove(stream)

    def _ready(self, stream, side) -> bool:
        """`side` is connected and may be read from or written to."""
        return stream["sock"][side] is not None and (side == "client" or stream["phase"] == "open")

    def _interest(self, selector, stream) -> None:
        for side, other in self.SIDES.items():
            sock = stream["sock"][side]
            if sock is None:
                continue
            if side == "upstream" and stream["phase"] == "connecting":
                events = selectors.EVENT_WRITE
            else:
                events = 0
                if not stream["eof"][side] and self._ready(stream, side) and len(stream["out"][other]) < self.BUFFER_CAP:
                    events |= selectors.EVENT_READ
                if stream["out"][side] and self._ready(stream, side):
                    events |= selectors.EVENT_WRITE
            try:
                registered = sock in selector.get_map()
                if events:
                    selector.modify(sock, events, (stream, side)) if registered else selector.register(sock, events, (stream, side))
                elif registered:
                    selector.unregister(sock)
            except (KeyError, ValueError, OSError):
                pass

    def _settle(self, selector, stream) -> None:
        """Propagate drained EOFs; close the stream when both directions are finished."""
        for side, other in self.SIDES.items():
            target = stream["sock"][other]
            if stream["eof"][side] and not stream["out"][other] and not stream["shut"][other] and self._ready(stream, other):
                try:
                    target.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
                stream["shut"][other] = True
        upstream_gone = stream["sock"]["upstream"] is None and stream["phase"] != "connecting"
        if stream["eof"]["client"] and (stream["eof"]["upstream"] or upstream_gone) and not stream["out"]["client"] and \
                (not stream["out"]["upstream"] or upstream_gone):
            self._close(selector, stream)
        elif stream in self._streams:
            self._interest(selector, stream)

    def _connect(self, selector, stream) -> None:
        while stream["address_index"] < len(self.addresses):
            family, address = self.addresses[stream["address_index"]]
            stream["address_index"] += 1
            upstream = socket.socket(family, socket.SOCK_STREAM)
            upstream.setblocking(False)
            code = upstream.connect_ex(address)
            if code in (0, 115):  # 0 or EINPROGRESS
                stream["sock"]["upstream"], stream["phase"], stream["deadline"] = upstream, "connecting", time.monotonic() + self.connect_timeout
                self._interest(selector, stream)
                return
            upstream.close()
        self.stats["upstream_failures"] += 1
        self._close(selector, stream)

    def _check_prefix(self, selector, stream, final: bool) -> None:
        """Admit (connect upstream), keep waiting, or drop the stream from the first client bytes."""
        head = bytes(stream["out"]["upstream"][:self.prefix_length])
        if self.allowed_prefixes and not final and len(head) < self.prefix_length and any(prefix.startswith(head) for prefix in self.allowed_prefixes):
            self._interest(selector, stream)
            return
        if self.allowed_prefixes and not any(head.startswith(prefix) for prefix in self.allowed_prefixes):
            self.stats["filtered"] += 1
            if len(self.stats["filtered_prefixes"]) < 16:
                self.stats["filtered_prefixes"].append(head[:8].decode("latin-1"))
            self._close(selector, stream)
            return
        self._connect(selector, stream)

    def _loop(self) -> None:
        selector = selectors.DefaultSelector()
        selector.register(self._wake_r, selectors.EVENT_READ, "wake")
        tokens, last = float(self.burst), time.monotonic()
        listening = False
        while not self._stop.is_set():
            now = time.monotonic()
            tokens = min(float(self.burst), tokens + (now - last) * self.rate)
            last = now
            may_accept = len(self._streams) < self.max_concurrent and tokens >= 1.0
            if may_accept != listening:
                if may_accept:
                    selector.register(self._listener, selectors.EVENT_READ, "listen")
                else:
                    selector.unregister(self._listener)
                    if tokens < 1.0:
                        self.stats["rate_delayed"] += 1
                listening = may_accept
            timeout = 0.5 if listening or tokens >= 1.0 else max(0.001, (1.0 - tokens) / self.rate)
            for key, mask in selector.select(timeout=timeout):
                if key.data == "wake":
                    try:
                        os.read(self._wake_r, 64)
                    except OSError:
                        pass
                    continue
                if key.data == "listen":
                    try:
                        client, _ = self._listener.accept()
                    except OSError:
                        continue
                    client.setblocking(False)
                    tokens -= 1.0
                    self.stats["accepted"] += 1
                    if self.max_streams is not None and self.stats["accepted"] > self.max_streams or \
                            self.stats["bytes_up"] + self.stats["bytes_down"] >= self.max_total:
                        self.stats["refused"] += 1
                        client.close()
                        continue
                    stream = {"sock": {"client": client, "upstream": None}, "out": {"client": bytearray(), "upstream": bytearray()},
                              "eof": {"client": False, "upstream": False}, "shut": {"client": False, "upstream": False},
                              "phase": "prefix", "bytes": 0, "deadline": time.monotonic() + self.prefix_timeout, "address_index": 0}
                    self._streams.append(stream)
                    self.stats["max_concurrent_seen"] = max(self.stats["max_concurrent_seen"], len(self._streams))
                    self._interest(selector, stream)
                    continue
                stream, side = key.data
                if stream not in self._streams:
                    continue
                sock = stream["sock"][side]
                if sock is None:
                    continue
                other = self.SIDES[side]
                if stream["phase"] == "connecting" and side == "upstream":
                    if mask & selectors.EVENT_WRITE:
                        if sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR) != 0:
                            selector.unregister(sock)
                            sock.close()
                            stream["sock"]["upstream"] = None
                            self._connect(selector, stream)
                            continue
                        stream["phase"] = "open"
                        self.stats["admitted"] += 1
                        self._settle(selector, stream)
                    continue
                if mask & selectors.EVENT_READ:
                    try:
                        data = sock.recv(65536)
                    except BlockingIOError:
                        data = None
                    except OSError:
                        self._close(selector, stream)
                        continue
                    if data == b"":
                        stream["eof"][side] = True
                        if stream["phase"] == "prefix":
                            self._check_prefix(selector, stream, final=True)
                            continue
                        self._settle(selector, stream)
                        continue
                    if data:
                        stream["bytes"] += len(data)
                        self.stats["bytes_up" if side == "client" else "bytes_down"] += len(data)
                        if stream["bytes"] > self.max_bytes:
                            self.stats["capped_stream"] += 1
                            self._close(selector, stream)
                            continue
                        if self.stats["bytes_up"] + self.stats["bytes_down"] > self.max_total:
                            self.stats["capped_total"] += 1
                            self._close(selector, stream)
                            continue
                        stream["out"][other] += data
                        if stream["phase"] == "prefix":
                            self._check_prefix(selector, stream, final=False)
                            continue
                if stream in self._streams and mask & selectors.EVENT_WRITE and stream["out"][side] and stream["sock"][side] is not None:
                    try:
                        sent = stream["sock"][side].send(stream["out"][side])
                        del stream["out"][side][:sent]
                    except BlockingIOError:
                        pass
                    except OSError:
                        self._close(selector, stream)
                        continue
                if stream in self._streams:
                    self._settle(selector, stream)
            now = time.monotonic()
            for stream in list(self._streams):
                if stream["phase"] in ("prefix", "connecting") and now > stream["deadline"]:
                    if stream["phase"] == "connecting":
                        sock = stream["sock"]["upstream"]
                        try:
                            selector.unregister(sock)
                        except (KeyError, ValueError):
                            pass
                        sock.close()
                        stream["sock"]["upstream"] = None
                        self._connect(selector, stream)
                    else:
                        self.stats["filtered"] += 1
                        self._close(selector, stream)
        for stream in list(self._streams):
            self._close(selector, stream)
        selector.close()

    def close(self) -> dict:
        started = time.monotonic()
        self._stop.set()
        try:
            os.write(self._wake_w, b"x")
        except OSError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5)
        alive = self._thread is not None and self._thread.is_alive()
        if self._listener is not None:
            self._listener.close()
        for fd in (self._wake_r, self._wake_w):
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            self.socket_path.unlink()
        except OSError:
            pass
        self.stats.update(shutdown_confirmed=not alive and not self._streams, shutdown_seconds=round(time.monotonic() - started, 4),
                          open_at_shutdown=len(self._streams))
        return dict(self.stats)


# ----------------------------------------------------------------------------------------------------- the runner

def run_bwrap(arguments: list[str], *, init_argv: list[str], timeout: float, grace: float, tail_bytes: int, cgroup: Path | None,
              scratch_quota: int | None = None, scratch_paths=(WORK, "/tmp", "/dev", "/dev/shm"), host_dirs=(), output_cap: int | None = None,
              on_start: Callable[[float], None] | None = None, bwrap: str = BWRAP, seccomp: bytes | None = None,
              step: dict | None = None) -> SandboxResult:
    """Spawn bwrap ARGUMENTS INIT_ARGV; enforce the deadline; collect tails frozen at the deadline.

    `step` = {"srun": [...], "env": {...}, "dir": Path, "memory_bytes": int} runs bwrap inside a separate Slurm step (the step's
    memory cgroup bounds the whole tree); fds cannot cross srun, so the status channels are files in step["dir"].
    """
    result = SandboxResult()
    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()
    process = None
    closers = [out_w, err_w]
    if step is None:
        status_r, status_w = os.pipe()
        json_r, json_w = os.pipe()
        closers += [status_w, json_w]
        extra = ["--json-status-fd", str(json_w)]
        child_fds = [status_w, json_w]
        if seccomp is not None:
            seccomp_r, seccomp_w = os.pipe()
            os.write(seccomp_w, seccomp)  # a few hundred bytes: fits in the pipe buffer
            os.close(seccomp_w)
            child_fds.append(seccomp_r)
            closers.append(seccomp_r)
            extra += ["--seccomp", str(seccomp_r)]
        command = [bwrap, *arguments, *extra, *[part.replace("{STATUS_FD}", str(status_w)) for part in init_argv]]
        env = {"PATH": "/usr/bin:/bin", "LANG": "C"}
    else:
        directory = Path(step["dir"])
        info_path, status_path, seccomp_path = directory / "bwrap-info.json", directory / "init-status.json", directory / "seccomp.bpf"
        for path in (info_path, status_path):
            path.write_bytes(b"")
        seccomp_path.write_bytes(seccomp or b"")
        preflight_path = directory / "preflight.json"
        extra = ["--json-status-fd", "3"] + (["--seccomp", "4"] if seccomp is not None else [])
        # Inside the step and before bwrap (and so before the compiler): verify the step's memory, swap, cpuset and OOM group.
        wrapper = ('info="$1"; status="$2"; filter="$3"; pre="$4"; py="$5"; check="$6"; shift 6; '
                   '"$py" -I -B -c "$check" >"$pre" 2>&1 || exit 97; exec "$@" 3>"$info" 4<"$filter" 5>"$status"')
        command = [*step["srun"], "/bin/sh", "-c", wrapper, "sh", str(info_path), str(status_path), str(seccomp_path), str(preflight_path),
                   step["preflight_python"], step["preflight_code"], bwrap, *arguments, *extra, *[part.replace("{STATUS_FD}", "5") for part in init_argv]]
        env, child_fds = step["env"], []
    before = oom_kills(cgroup)
    try:
        result.started = time.monotonic()
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out_w, stderr=err_w, pass_fds=tuple(child_fds),
                                   env=env, start_new_session=True, close_fds=True)
    except OSError as error:
        result.infrastructure = f"sandbox launch failed: {error}"
        for fd in (out_r, err_r, *closers, *([status_r, json_r] if step is None else [])):
            os.close(fd)
        return result
    for fd in closers:
        os.close(fd)
    if on_start is not None:
        on_start(result.started)
    deadline = result.started + timeout
    stdout, stderr = TailReader(out_r, tail_bytes, cutoff=deadline), TailReader(err_r, tail_bytes, cutoff=deadline)
    if step is None:
        status, info = JsonReader(status_r), JsonReader(json_r)
    else:
        alive = lambda: process.poll() is None  # noqa: E731
        status, info = FileJson(status_path, alive), FileJson(info_path, alive)
    for reader in (stdout, stderr, status, info):
        reader.start()
    child = info.wait_for("child-pid", timeout=60.0 if step else 30.0)
    ns_inode, step_cgroup = None, None
    if step is not None:  # the in-step preflight has finished before bwrap could report a PID
        try:
            preflight = json.loads(preflight_path.read_text() or "{}")
        except (OSError, json.JSONDecodeError):
            preflight = {}
        if preflight.get("ok") is True and preflight.get("cgroup"):
            result.step_cgroup = preflight
            step_cgroup = Path(preflight["cgroup"])
            before = oom_kills(step_cgroup)
        else:
            text = preflight_path.read_text()[-300:] if preflight_path.exists() else ""
            detail = preflight.get("error") or text or "no preflight output (srun or the step failed before the check)"
            result.infrastructure = f"build step preflight rejected or could not verify the step's limits (the compiler was not started): {detail}"
    if child is not None:
        try:
            ns_inode = os.stat(f"/proc/{child}/ns/pid").st_ino
        except OSError:
            ns_inode = None
    elif result.infrastructure is None:
        result.infrastructure = "bwrap did not report the sandbox PID"
    oom_source = step_cgroup if step is not None else cgroup
    last_oom = before
    next_poll = 0.0
    while process.poll() is None:
        now = time.monotonic()
        status.poll()
        if step is not None and oom_source is not None:
            value = oom_kills(oom_source)
            last_oom = value if value is not None else last_oom
        if result.infrastructure and result.kill_sent is None:
            result.kill_reason, result.kill_sent = "infrastructure", now
            _kill_sandbox(child, process)
        if now >= deadline and result.term_sent is None and result.kill_sent is None:
            stdout.freeze()
            stderr.freeze()
            result.deadline_reached, result.term_sent, result.kill_reason = True, now, "deadline"
            if ns_inode is not None:
                for pid in namespace_members(ns_inode):
                    if pid != child:
                        try:
                            os.kill(pid, signal.SIGTERM)
                        except OSError:
                            pass
        if result.term_sent is not None and result.kill_sent is None and now >= result.term_sent + grace:
            result.kill_sent = now
            _kill_sandbox(child, process)
        if now >= next_poll and result.kill_sent is None:
            next_poll = now + 0.5
            # Only after the init reported `ready`: before pivot_root, /proc/CHILD/root still resolves to the host root.
            ready = any(document.get("event") == "ready" for document in status.documents)
            if scratch_quota is not None and child is not None and ready:
                usage = scratch_usage(child, scratch_paths)
                if usage is not None:
                    usage += sum(directory_usage(Path(path)) for path in host_dirs)
                    result.peak_scratch_bytes = max(result.peak_scratch_bytes or 0, usage)
                    if usage > scratch_quota:
                        result.kill_reason, result.kill_sent = "scratch quota", now
                        stdout.freeze()
                        stderr.freeze()
                        _kill_sandbox(child, process)
            if output_cap is not None and stdout.total + stderr.total > output_cap:
                result.kill_reason, result.kill_sent = "output cap", now
                _kill_sandbox(child, process)
        time.sleep(0.02)
    result.elapsed = time.monotonic() - result.started
    for reader in (stdout, stderr, status, info):
        reader.join(timeout=10)
    exit_doc = next((doc for doc in info.documents if "exit-code" in doc), None)
    result.exit_code = exit_doc["exit-code"] if exit_doc else None
    result.events = list(status.documents)
    payload = result.payload_exit()
    if payload is not None:
        result.signal = payload.get("signal")
    started = next((event for event in result.events if event.get("event") == "payload_started"), None)
    result.payload_started = started["t"] if started else None
    # Completion is judged against T_f independently of the polling loop: the trusted init's exit time (same monotonic clock).
    ended = payload["t"] if payload is not None and "t" in payload else result.started + result.elapsed
    if not result.deadline_reached and result.kill_reason is None and result.payload_started is not None and ended >= deadline:
        result.deadline_reached = result.late_exit = True
    after = oom_kills(oom_source) if oom_source is not None else None
    after = after if after is not None else last_oom  # a finished Slurm step's cgroup may already be gone
    result.oom_kills = (after - before) if (after is not None and before is not None) else None
    result.stdout_tail, result.stderr_tail = stdout.text(), stderr.text()
    result.stdout_bytes, result.stderr_bytes = stdout.total, stderr.total
    result.bytes_after_deadline = stdout.after_freeze() + stderr.after_freeze()
    leftovers = namespace_members(ns_inode) if ns_inode is not None else []
    for pid in leftovers:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    result.cleanup_verified = not leftovers and (ns_inode is None or not namespace_members(ns_inode))
    if result.payload_started is None and result.infrastructure is None and result.kill_sent is None:
        errors = [event for event in result.events if event.get("event") == "error"]
        result.infrastructure = ("sandbox init: " + "; ".join(f"{event.get('stage')}: {event.get('message')}" for event in errors)) if errors \
            else f"the sandbox ended before the payload started (bwrap exit {result.exit_code})"
    return result


def _kill_sandbox(child, process) -> None:
    """SIGKILL to the namespace's PID 1 makes the kernel kill every process of the namespace."""
    try:
        if child is not None:
            os.kill(child, signal.SIGKILL)
        else:
            process.kill()
    except OSError:
        pass


def classify(result: SandboxResult) -> tuple[str, str]:
    """(termination, detail) for a solver run; see the module docstring for the attribution rules."""
    if result.infrastructure:
        return "infrastructure", result.infrastructure
    payload = result.payload_exit()
    if result.deadline_reached:
        if result.late_exit:
            return "deadline", "ended after the wall-clock limit (before the executor's signal)"
        return "deadline", "terminated at the wall-clock limit" + (" (SIGKILL after grace)" if result.kill_sent else "")
    if result.kill_reason == "scratch quota":
        return "crash", "scratch soft limit exceeded (/work + /tmp + /dev + /dev/shm)"
    if result.kill_reason == "output cap":
        return "crash", "output cap exceeded"
    if payload is None:
        if result.payload_started is None:
            return "infrastructure", "the sandbox ended before the payload started"
        if result.oom_kills:
            return "oom", "kernel OOM kill in the job cgroup (no payload exit report)"
        return "crash", "the sandbox ended without a payload exit report after the payload started"
    code, sig = payload.get("exit_code"), payload.get("signal")
    if code == 0:
        return "finished", "exit 0" + (f" ({result.oom_kills} kernel OOM kill(s) of descendant processes)" if result.oom_kills else "")
    how = f"signal {sig}" if sig else f"exit {code}"
    if result.oom_kills:
        return "oom", f"kernel OOM kill in the job cgroup (payload {how})"
    if any(marker in result.stderr_tail for marker in MEMORY_MARKERS):
        return "crash", f"{how} (candidate-reported memory allocation failure; no kernel OOM event)"
    if not sig and "File too large" in result.stderr_tail:  # CPython ignores SIGXFSZ, so the limit surfaces as EFBIG
        return "crash", f"exit {code} (file size limit: EFBIG)"
    if sig:
        name = signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else str(sig)
        note = SIGNAL_NOTES.get(sig)
        return "crash", f"signal {name}" + (f" ({note})" if note else "")
    return "crash", f"exit {code}"


# --------------------------------------------------------------------------------------------------- the executor

@dataclass
class Attempt:
    """Private host directory of one attempt: control files, materialized sources and build outputs."""
    root: Path
    sources: Path
    build: Path
    manifest: dict
    build_record: dict | None = None
    audit: dict | None = None


class XLSandboxExecutor:
    """Phoenix bubblewrap executor. Construct once per job; `execute` runs one attempt (manifest -> build -> run)."""

    def __init__(self, *, python: str | Path | None = None, allowed_packages=DEFAULT_PACKAGES, license: str | Path | None = "auto",
                 gcc_root: str | Path | None = "auto", base_dir: str | Path | None = None, bwrap: str = BWRAP,
                 stream_tail_bytes: int = TAIL_BYTES, keep_files: bool = False, seccomp: bool = True, expose_usr: bool = False,
                 masked_modules=MASKED_MODULES):
        require_linux()
        if not Path(bwrap).is_file():
            raise InfrastructureError(f"{bwrap} is missing")
        self.bwrap = bwrap
        self.seccomp = seccomp_program() if seccomp else None
        self.expose_usr = expose_usr  # diagnostic only: mount all of /usr in runs (with its toolchains masked)
        self.stream_tail_bytes = stream_tail_bytes
        self.keep_files = keep_files
        self.base_dir = Path(base_dir or os.environ.get("TMPDIR") or "/tmp").resolve()
        self.cgroup = job_cgroup()
        self.python = Path(python or sys.executable).resolve()
        self.allowed_packages = tuple(allowed_packages)
        self.python_facts = self._python_facts()
        self.packages = self._package_entries()
        self.masked_modules = self._masked_modules(masked_modules)
        self.license = None
        if license is not None:
            path = os.environ.get("GRB_LICENSE_FILE") or PACE_LICENSE if license == "auto" else str(license)
            servers, port = parse_license(path)
            self.license = {"source": path, "servers": servers, "port": port}
        self.gcc = self._gcc_facts(gcc_root) if gcc_root is not None else None
        # The only /usr/bin entries a build sees: the assembler and linker the pinned driver runs (resolved files).
        self.build_tools = [{"dest": f"/usr/bin/{name}", "real": os.path.realpath(f"/usr/bin/{name}")} for name in ("as", "ld")
                            if os.path.isfile(os.path.realpath(f"/usr/bin/{name}"))]
        self.srun = shutil.which("srun", path="/opt/slurm/current/bin:/usr/bin:/bin")
        prefix, site = self.python_facts["prefix"], self.python_facts["purelib"]
        trees = list(RUN_TREES_USR) + [prefix] + ([self.gcc["lib64"]] if self.gcc else [])
        self.inventory = toolchain_inventory(trees, skip=[site, os.path.join(prefix, "bin")])
        package_dirs = [entry["path"] for entry in self.packages if os.path.isdir(entry["path"])]
        self.inventory["masked"] += toolchain_inventory(package_dirs)["masked"]
        if self.expose_usr:
            self.inventory["masked"] += [path for pattern in USR_FILE_MASKS for path in sorted(glob.glob(pattern))
                                         if os.path.isfile(path) and not os.path.islink(path)]
        self.inventory["masked"] = sorted(set(self.inventory["masked"]))
        user = pwd.getpwuid(os.getuid())
        self.user = {"name": user.pw_name, "uid": os.getuid(), "gid": os.getgid(), "group": grp.getgrgid(os.getgid()).gr_name}
        self.environment = {
            "policy": SANDBOX_POLICY, "bwrap": bwrap_version(bwrap), "bwrap_sha256": sha256_file(Path(bwrap)), "kernel": platform.release(),
            "hostname": platform.node(), "python": self.python_facts, "runtime": self.python_facts["versions"],
            "packages": [entry["name"] for entry in self.packages], "init_sha256": sha256_file(INIT_SOURCE),
            "license": None if self.license is None else {"servers": self.license["servers"], "port": self.license["port"],
                                                          "admitted_prefixes": [prefix.decode() for prefix in TOKEN_REQUEST_PREFIXES]},
            "gcc": None if self.gcc is None else {key: self.gcc[key] for key in ("root", "version", "gxx_sha256", "cc1plus_sha256")},
            "srun": self.srun, "run_trees": trees, "toolchain_masks": self.inventory["masked"], "expose_usr": expose_usr,
            "masked_modules": self.masked_modules, "runtime_contract": RUNTIME_CONTRACT,
            "build_mounts": ["/usr/lib64", "/usr/include", *[tool["dest"] + " -> " + tool["real"] for tool in self.build_tools],
                             "pinned compiler tree and its library directories"],
            "seccomp_denied": sorted([*SECCOMP_DENY, "clone(CLONE_NEW*)", "clone3 (ENOSYS)", "prlimit64(pid != 0)"]) if seccomp else None}

    # ---------------------------------------------------------------------------------------------- environment facts

    def _python_facts(self) -> dict:
        probe = ("import importlib, json, sys, sysconfig\n"
                 f"names = {list(self.allowed_packages)!r}\n"
                 "versions = {'python': sys.version.split()[0]}\n"
                 "for name in names:\n"
                 "    module = importlib.import_module(name)\n"
                 "    versions[name] = getattr(module, '__version__', None)\n"
                 "    if name == 'gurobipy':\n"
                 "        versions['gurobi'] = '.'.join(map(str, module.gurobi.version()))\n"
                 "print(json.dumps({'prefix': sys.prefix, 'version': sys.version.split()[0], 'purelib': sysconfig.get_paths()['purelib'], 'versions': versions}))\n")
        result = subprocess.run([str(self.python), "-I", "-c", probe], capture_output=True, text=True, timeout=120, env={"PATH": "/usr/bin:/bin"})
        if result.returncode:
            raise InfrastructureError(f"cannot inspect {self.python}: {result.stderr.strip()[-500:]}")
        facts = json.loads(result.stdout.strip().splitlines()[-1])
        facts["prefix"] = os.path.realpath(facts["prefix"])
        facts["purelib"] = os.path.realpath(facts["purelib"])
        facts["executable"] = str(self.python)
        if not str(self.python).startswith(facts["prefix"] + "/"):
            raise InfrastructureError("the solver Python must live inside its prefix")
        facts["sha256"] = sha256_file(self.python)
        return facts

    def _package_entries(self) -> list[dict]:
        """Package directories, their `.libs`, and only the dist-info of the version that imports."""
        site = Path(self.python_facts["purelib"])
        versions = self.python_facts["versions"]
        entries = []
        for item in sorted(os.listdir(site)):
            low = item.lower()
            for name in self.allowed_packages:
                base = name.lower()
                if low in (base, base + ".libs") or low == f"{base}-{str(versions.get(name)).lower()}.dist-info":
                    entries.append({"name": item, "path": str(site / item)})
        missing = [name for name in self.allowed_packages if not any(entry["name"].lower() == name.lower() for entry in entries)]
        if missing:
            raise InfrastructureError(f"allowed packages missing from {site}: {missing}")
        return entries

    def _masked_modules(self, names) -> dict:
        """Modules of allowlisted packages replaced by an ImportError stub (path and what is hidden, recorded per run)."""
        site = Path(self.python_facts["purelib"])
        masked = {}
        for name in names:
            if name.split(".")[0] not in self.allowed_packages:
                continue
            directory = site / name.replace(".", "/")
            if directory.is_dir():
                hidden = sorted(entry.name for entry in directory.iterdir())
                backends = sorted(str(path.relative_to(site)) for path in directory.rglob("*") if re.search(r"highs|lsap|linprog|milp", path.name, re.I)
                                  and "__pycache__" not in path.parts)
                masked[name] = {"path": str(directory), "hidden_entries": len(hidden), "highs_and_lp_files": backends}
        return masked

    def _gcc_facts(self, gcc_root) -> dict:
        root = os.environ.get("GCC_ROOT") if gcc_root == "auto" else str(gcc_root)
        root = root or PHOENIX_GCC_ROOT
        gxx = Path(root) / "bin/g++"
        clean = {"PATH": "/usr/bin:/bin"}
        if not gxx.is_file():
            raise InfrastructureError(f"pinned compiler missing: {gxx}")
        version = subprocess.run([str(gxx), "-dumpfullversion"], capture_output=True, text=True, env=clean, timeout=30).stdout.strip()
        if version != GCC_VERSION:
            raise InfrastructureError(f"g++ {GCC_VERSION} required, found {version} at {gxx}")
        tools = [subprocess.run([str(gxx), f"-print-prog-name={name}"], capture_output=True, text=True, env=clean, timeout=30).stdout.strip()
                 for name in ("cc1plus", "collect2", "lto-wrapper")]
        plugin = subprocess.run([str(gxx), "-print-file-name=liblto_plugin.so"], capture_output=True, text=True, env=clean, timeout=30).stdout.strip()
        library_dirs = set()
        for binary in [str(gxx), *tools, plugin]:
            if not os.path.isfile(binary):
                continue
            listing = subprocess.run(["/usr/bin/ldd", binary], capture_output=True, text=True, env=clean, timeout=30).stdout
            for path in re.findall(r"=>\s*(/\S+)", listing):
                real = os.path.realpath(path)
                if not real.startswith(("/usr/", "/lib")):
                    library_dirs.add(os.path.dirname(real))
        real_root = os.path.realpath(root)
        return {"root": root, "real_root": real_root, "version": version, "gxx": str(gxx), "real_gxx": os.path.realpath(gxx), "gxx_sha256": sha256_file(gxx),
                "cc1plus_sha256": sha256_file(Path(tools[0])) if os.path.isfile(tools[0]) else None,
                "library_dirs": sorted(d for d in library_dirs if not d.startswith(real_root + "/")),
                "lib64": os.path.realpath(Path(root) / "lib64"), "symlinks": symlinked_components(str(Path(root) / "lib64")),
                "env": {"PATH": f"{real_root}/bin:/usr/bin", "LD_LIBRARY_PATH": f"{real_root}/lib64:{real_root}/lib",
                        "LIBRARY_PATH": f"{real_root}/lib64:{real_root}/lib", "CPATH": f"{real_root}/include"}}

    # ------------------------------------------------------------------------------------------------ attempt setup

    def prepare(self, delivery: Path | tuple[dict, dict[str, bytes]], *, track: str, strict: bool = True) -> Attempt:
        """Validate the delivery for `track` (canonical validator) and materialize it in a fresh private directory."""
        from src.experiments.xl_build import DeliveryError, collect_delivery, materialize
        if track not in TRACKS:
            raise ValueError(f"unknown track {track!r}; expected one of {TRACKS}")
        if isinstance(delivery, tuple):
            manifest, files = delivery
            if manifest.get("track") != track:
                raise DeliveryError(f"manifest track {manifest.get('track')!r} differs from the attempt's track {track!r}")
        else:
            manifest, files = collect_delivery(Path(delivery), track=track, strict=strict)
        root = Path(tempfile.mkdtemp(prefix="dgxl-", dir=self.base_dir))
        os.chmod(root, 0o700)
        try:
            materialize(manifest, files, root / "src")  # re-validates (canonical validator) and checks the hashes
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise
        build = root / "build"
        build.mkdir(mode=0o700)
        (root / "step").mkdir(mode=0o700)
        etc = root / "etc"
        etc.mkdir(mode=0o700)
        user = self.user
        (etc / "passwd").write_text(f"{user['name']}:x:{user['uid']}:{user['gid']}::{WORK}:/sbin/nologin\n")
        (etc / "group").write_text(f"{user['group']}:x:{user['gid']}:\n")
        (etc / "nsswitch.conf").write_text("passwd: files\ngroup: files\nhosts: files\n")
        (etc / "hosts").write_text("127.0.0.1 localhost\n::1 localhost\n")
        shutil.copyfile(INIT_SOURCE, root / "init.py")
        for name in self.masked_modules:
            (root / f"masked_{name}.py").write_text(f"raise ImportError({name + ' is not available in the DecompGym-XL sandbox (HiGHS and other optimizers are masked)'!r})\n")
        if self.license is not None:
            (root / "gurobi.lic").write_text(f"TOKENSERVER=127.0.0.1\nPORT={self.license['port']}\n")
        for path in [*root.rglob("*")]:
            if path.is_file() and not path.is_symlink() and build not in path.parents:
                os.chmod(path, 0o444)
        return Attempt(root=root, sources=root / "src", build=build, manifest=manifest)

    def cleanup(self, attempt: Attempt) -> None:
        if not self.keep_files:
            for path in attempt.root.rglob("*"):
                if path.is_dir() and not path.is_symlink():
                    os.chmod(path, 0o700)
            shutil.rmtree(attempt.root, ignore_errors=True)

    # --------------------------------------------------------------------------------------------- sandbox arguments

    def _python_mounts(self, attempt: Attempt, packages: bool = True) -> list[str]:
        """The solver Python's prefix; site-packages is the allowlist (runs) or empty (builds: the init needs only the stdlib)."""
        prefix, site = self.python_facts["prefix"], self.python_facts["purelib"]
        args = ["--ro-bind", prefix, prefix, "--tmpfs", site]
        if packages:
            for entry in self.packages:
                args += ["--ro-bind", entry["path"], entry["path"]]
            for name, facts in self.masked_modules.items():  # e.g. scipy.optimize (HiGHS): an empty directory whose import fails
                args += ["--tmpfs", facts["path"], "--ro-bind", str(attempt.root / f"masked_{name}.py"), f"{facts['path']}/__init__.py",
                         "--remount-ro", facts["path"]]
            bundled = Path(site) / "gurobipy/.libs/gurobi.lic"
            if bundled.exists():  # no silent fallback to the pip size-limited license
                args += ["--ro-bind", "/dev/null", str(bundled)]
        args += ["--remount-ro", site, "--tmpfs", f"{prefix}/bin", "--ro-bind", str(self.python), str(self.python)]
        for alias in ("python3", "python"):
            if alias != self.python.name:
                args += ["--symlink", self.python.name, f"{prefix}/bin/{alias}"]
        return args + ["--remount-ro", f"{prefix}/bin"]

    def _common(self, attempt: Attempt, *, phase: str) -> list[str]:
        args = ["--unshare-all", "--die-with-parent", "--new-session", "--as-pid-1", "--hostname", "decompgym", "--clearenv"]
        if phase == "build":  # only what the pinned g++ needs: no /usr/bin (except as, ld), /bin, /usr/libexec or /usr/local
            args += ["--ro-bind", "/usr/lib64", "/usr/lib64", "--ro-bind", "/usr/include", "/usr/include"]
            for tool in self.build_tools:
                args += ["--ro-bind", tool["real"], tool["dest"]]
            # The driver's library search path goes through /usr/lib/../lib64 and /lib/../lib64: an empty /usr/lib suffices.
            args += ["--dir", "/usr/lib", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64"]
        elif self.expose_usr:
            args += ["--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
                     "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64"]
        else:
            for tree in RUN_TREES_USR:
                args += ["--ro-bind-try", tree, tree]
            args += ["--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64"]
        args += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/dev/shm", "--tmpfs", "/tmp"]
        for name in ("passwd", "group", "nsswitch.conf", "hosts"):
            args += ["--ro-bind", str(attempt.root / "etc" / name), f"/etc/{name}"]
        for path in ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d", "/etc/localtime"):
            args += ["--ro-bind-try", path, path]
        args += self._python_mounts(attempt, packages=phase != "build")
        args += ["--tmpfs", WORK, "--ro-bind", str(attempt.root / "init.py"), f"{CONTROL}/init.py"]
        return args

    def _init_argv(self) -> list[str]:
        return [str(self.python), "-I", "-B", "-S", f"{CONTROL}/init.py", f"{CONTROL}/config.json", "{STATUS_FD}"]

    def _write_config(self, attempt: Attempt, name: str, config: dict) -> Path:
        path = attempt.root / name
        path.write_text(json.dumps(config, indent=1))
        os.chmod(path, 0o444)
        return path

    # ---------------------------------------------------------------------------------------------------- build step

    def compile(self, attempt: Attempt, argv: list[str], *, timeout: float, limits: dict, tail_bytes: int, output_cap: int, step: str,
                scratch_quota: int | None = None, memory_step: str = "srun", memory_bytes: int = 4 * GiB,
                preflight_memory_cap: int | None = None) -> SandboxResult:
        """One compiler command inside the sandbox (no network, sources read-only, only /work/build writable), in a memory-limited Slurm step."""
        if self.gcc is None:
            raise InfrastructureError("no pinned compiler configured")
        argv = [self.gcc["real_gxx"] if part == self.gcc["gxx"] else part for part in argv]  # /usr/local is not mounted in builds
        config = {"argv": argv, "cwd": WORK, "env": {**self.gcc["env"], "HOME": WORK, "TMPDIR": "/tmp", "LANG": "C", "LC_ALL": "C"},
                  "limits": limits, "relay": None, "oom_score_adj": 1000}
        config_path = self._write_config(attempt, f"config_{step}.json", config)
        args = self._common(attempt, phase="build")
        args += ["--ro-bind", self.gcc["real_root"], self.gcc["real_root"]]
        for directory in self.gcc["library_dirs"]:
            args += ["--ro-bind", directory, directory]
        site = self.python_facts["purelib"] + "/"
        for path in self.inventory["masked"]:  # JIT libraries etc. in the mounted trees are not build inputs either
            if not path.startswith(site):
                args += ["--ro-bind", "/dev/null", path]
        if (attempt.sources / "cpp").is_dir():
            args += ["--ro-bind", str(attempt.sources / "cpp"), f"{WORK}/cpp"]
        args += ["--bind", str(attempt.build), f"{WORK}/build", "--ro-bind", str(config_path), f"{CONTROL}/config.json",
                 "--chdir", WORK, "--remount-ro", "/"]
        slurm_step = None
        if memory_step == "srun":
            if self.srun is None or "SLURM_JOB_ID" not in os.environ:
                raise InfrastructureError("the build's memory step needs srun inside a Slurm job")
            directory = Path(tempfile.mkdtemp(prefix=f"{step}-", dir=attempt.root / "step"))
            minutes = max(1, math.ceil((timeout + 60) / 60))
            slurm_step = {"srun": [self.srun, "--overlap", f"--mem={memory_bytes // MiB}M", "--ntasks=1", "--nodes=1", "--cpus-per-task=1",
                                   "--quiet", "--export=NONE", f"--time={minutes}", "--job-name=dgxl-build"],
                          "env": {**{key: value for key, value in os.environ.items() if key.startswith("SLURM")},
                                  "PATH": "/opt/slurm/current/bin:/usr/bin:/bin", "LANG": "C"},
                          "dir": directory, "memory_bytes": memory_bytes, "preflight_python": sys.executable,
                          "preflight_code": STEP_PREFLIGHT.format(directory=str(Path(__file__).resolve().parent),
                                                                  memory_cap=preflight_memory_cap or memory_bytes)}
        elif memory_step != "none":
            raise ValueError(f"unknown memory_step {memory_step!r}")
        return run_bwrap(args, init_argv=self._init_argv(), timeout=timeout, grace=0.5, tail_bytes=tail_bytes, cgroup=self.cgroup,
                         output_cap=output_cap, scratch_quota=scratch_quota, scratch_paths=("/tmp", "/dev", "/dev/shm"), host_dirs=(attempt.build,),
                         bwrap=self.bwrap, seccomp=self.seccomp, step=slurm_step)

    def build(self, attempt: Attempt, limits=None) -> dict:
        from src.experiments.xl_build import BuildLimits, build_core
        attempt.build_record = build_core(self, attempt, limits or BuildLimits())
        return attempt.build_record

    # ------------------------------------------------------------------------------------------------------ the run

    def run(self, attempt: Attempt, data: str | Path, limits: RunLimits, *, publish_socket: str | Path | None = None,
            on_start: Callable[[float], None] | None = None, extra_env: dict | None = None, client_modules=()) -> dict:
        """Run `python solver.py` in the sandbox with cwd /work and a read-only /work/data.json (limits checked first)."""
        data = Path(data).resolve(strict=True)
        job_limits = check_job_limits(self.cgroup, cpus=limits.cpus, memory_cap=limits.memory_cap_bytes, swap_cap=limits.swap_cap_bytes)
        track = attempt.manifest["track"]
        built = attempt.build_record or {}
        targets = built.get("targets", {}) if track == "cpp_allowed" else {}
        env = {"PATH": f"{self.python_facts['prefix']}/bin", "HOME": WORK, "TMPDIR": "/tmp", "LANG": "C.UTF-8", "USER": self.user["name"], "LOGNAME": self.user["name"],
               "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
               **{key: str(limits.cpus) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
        if self.expose_usr:
            env["PATH"] += ":/usr/bin:/bin"
        if self.license is not None:
            env["GRB_LICENSE_FILE"] = f"{CONTROL}/gurobi.lic"
        if publish_socket is not None:
            env["DECOMPGYM_PUBLISH"] = f"{CONTROL}/publish.sock"
        modules = [Path(path).resolve(strict=True) for path in client_modules]
        if modules:
            env["PYTHONPATH"] = f"{CONTROL}/lib"
        if "libcore.so" in targets:
            env["DECOMPGYM_CORE"] = f"{WORK}/build/libcore.so"
        if "core" in targets:
            env["DECOMPGYM_CORE_BIN"] = f"{WORK}/build/core"
        env.update(extra_env or {})
        relay = {"socket": f"{CONTROL}/license.sock", "port": self.license["port"]} if self.license is not None else None
        config = {"argv": [str(self.python), "-B", "-s", "-P", "-u", f"{WORK}/solver.py"], "cwd": WORK, "env": env, "limits": limits.rlimits(),
                  "relay": relay, "oom_score_adj": 1000}
        config_path = self._write_config(attempt, "config_run.json", config)
        gurobi_env = attempt.root / "gurobi.env"
        if not gurobi_env.exists():
            gurobi_env.write_text(f"ThreadLimit {limits.cpus}\nThreads {limits.cpus}\n")
            os.chmod(gurobi_env, 0o444)
        args = self._common(attempt, phase="run")
        gcc_lib = self.gcc["lib64"] + "/" if self.gcc else None
        for path in self.inventory["masked"]:  # toolchains in the mounted trees (the compiler's lib64 is handled below)
            if gcc_lib is None or not path.startswith(gcc_lib):
                args += ["--ro-bind", "/dev/null", path]
        if self.expose_usr:
            for path in USR_DIR_MASKS:
                if os.path.isdir(path) and not os.path.islink(path):
                    args += ["--tmpfs", path, "--remount-ro", path]
        if targets and self.gcc is not None:  # runtime libraries of the pinned compiler (the binaries' RPATH)
            args += ["--ro-bind", self.gcc["lib64"], self.gcc["lib64"]]
            for link, target in self.gcc["symlinks"]:  # e.g. /usr/local/pace-apps -> /storage/pace-apps (the RPATH's spelling)
                args += ["--symlink", target, link]
            args += [part for path in self.inventory["masked"] if path.startswith(gcc_lib) for part in ("--ro-bind", "/dev/null", path)]
        args += ["--ro-bind", str(data), f"{WORK}/data.json", "--ro-bind", str(attempt.sources / "solver.py"), f"{WORK}/solver.py",
                 "--ro-bind", str(gurobi_env), f"{WORK}/gurobi.env"]
        if (attempt.sources / "cpp").is_dir():
            args += ["--ro-bind", str(attempt.sources / "cpp"), f"{WORK}/cpp"]
        if targets:
            args += ["--ro-bind", str(attempt.build), f"{WORK}/build"]
        args += ["--ro-bind", str(config_path), f"{CONTROL}/config.json"]
        forwarder = None
        if self.license is not None:
            socket_path = attempt.root / "license.sock"
            forwarder = LicenseForwarder(socket_path, self.license["servers"], self.license["port"]).start()
            args += ["--ro-bind", str(attempt.root / "gurobi.lic"), f"{CONTROL}/gurobi.lic", "--ro-bind", str(socket_path), f"{CONTROL}/license.sock"]
        if publish_socket is not None:
            args += ["--ro-bind", str(Path(publish_socket).resolve(strict=True)), f"{CONTROL}/publish.sock"]
        for module in modules:
            if module.suffix != ".py" or not module.is_file():
                raise InfrastructureError(f"client module must be a .py file: {module}")
            args += ["--ro-bind", str(module), f"{CONTROL}/lib/{module.name}"]
        args += ["--chdir", WORK, "--remount-ro", "/"]
        try:
            result = run_bwrap(args, init_argv=self._init_argv(), timeout=limits.wall_seconds, grace=limits.grace_seconds,
                               tail_bytes=self.stream_tail_bytes, cgroup=self.cgroup, scratch_quota=limits.scratch_quota_bytes,
                               on_start=on_start, bwrap=self.bwrap, seccomp=self.seccomp)
        finally:
            tunnel = forwarder.close() if forwarder is not None else None
        termination, detail = classify(result)
        ready = next((event for event in result.events if event.get("event") == "ready"), {})
        payload = result.payload_exit() or {}
        record = {"policy": SANDBOX_POLICY, "track": track, "termination": termination, "detail": detail, "t0": result.started,
                  "wall_seconds": result.elapsed, "time_limit": limits.wall_seconds,
                  "startup_seconds": (result.payload_started - result.started) if result.payload_started else None,
                  "exit_code": payload.get("exit_code"), "signal": payload.get("signal"), "bwrap_exit_code": result.exit_code,
                  "term_sent": result.term_sent, "kill_sent": result.kill_sent, "kill_reason": result.kill_reason,
                  "maxrss_kib": payload.get("maxrss_kib"), "cpu_seconds": (payload.get("user_seconds") or 0) + (payload.get("system_seconds") or 0) if payload else None,
                  "oom_kills": result.oom_kills, "peak_scratch_bytes": result.peak_scratch_bytes, "cleanup_verified": result.cleanup_verified,
                  "stdout": result.stdout_tail, "stderr": result.stderr_tail, "stdout_bytes": result.stdout_bytes, "stderr_bytes": result.stderr_bytes,
                  "bytes_after_deadline": result.bytes_after_deadline, "late_exit": result.late_exit, "job_limits": job_limits,
                  "runtime": self.python_facts["versions"], "masked_modules": self.masked_modules, "runtime_contract": RUNTIME_CONTRACT,
                  "sandbox": {"affinity": ready.get("affinity"), "rlimits": ready.get("limits"), "uid": ready.get("uid"), "non_dumpable_init": ready.get("non_dumpable"),
                              "oom_score_adj": ready.get("oom_score_adj"), "init_handlers_reset": ready.get("handlers_reset"), "env_keys": sorted(env),
                              "core_targets": sorted(targets)},
                  "license_tunnel": tunnel, "events": result.events, "input_sha256": sha256_file(data)}
        if not result.cleanup_verified and termination != "infrastructure":
            record["termination"], record["detail"] = "infrastructure", "sandbox processes survived cleanup"
        if tunnel is not None and not tunnel["shutdown_confirmed"] and record["termination"] != "infrastructure":
            record["termination"], record["detail"] = "infrastructure", "license forwarder shutdown not confirmed"
        return record

    # ------------------------------------------------------------------------------------------------------ one attempt

    def execute(self, delivery: Path | tuple[dict, dict[str, bytes]], data: str | Path, limits: RunLimits, *, track: str, build_limits=None,
                publish_socket: str | Path | None = None, on_start: Callable[[float], None] | None = None, strict: bool = True,
                diagnose: bool = True, client_modules=()) -> dict:
        """manifest -> build (outside T_f) -> run. Returns {manifest, build, run, termination, detail, feedback, source_audit}."""
        from src.experiments.xl_audit import run_bounded_audits
        from src.experiments.xl_build import DeliveryError
        try:
            attempt = self.prepare(delivery, track=track, strict=strict)
        except DeliveryError as error:
            reasons = "; ".join(f"{item.get('path')}: {item.get('reason')}" for item in error.rejected) if error.rejected else ""
            detail = f"delivery rejected: {error}" + (f" ({reasons})" if reasons else "")
            return {"manifest": None, "build": None, "run": None, "termination": "build_error", "detail": detail, "track": track,
                    "feedback": {"stdout": "", "stderr": f"[framework] {detail}"}, "source_audit": None, "environment": self.environment}
        try:
            files = {entry["path"]: (attempt.sources / entry["path"]).read_bytes() for entry in attempt.manifest["files"]}
            attempt.audit = audit = run_bounded_audits(files)  # bounded child process; never raises for candidate input
            build = self.build(attempt, build_limits)
            base = {"manifest": attempt.manifest, "build": build, "track": track, "source_audit": audit, "environment": self.environment}
            if build["status"] in ("build_error", "infrastructure"):
                return {**base, "run": None, "termination": build["status"], "detail": build["detail"], "feedback": {"stdout": "", "stderr": build["feedback"]}}
            run = self.run(attempt, data, limits, publish_socket=publish_socket, on_start=on_start, client_modules=client_modules)
            if diagnose and self.license is not None and run["termination"] == "crash" and any(clue in run["stderr"].lower() for clue in GUROBI_CLUES):
                # As in the paper's executor: a Gurobi/license failure is scored only if an independent probe still works.
                run["gurobi_diagnostic"] = self.gurobi_preflight()
                if not run["gurobi_diagnostic"]["ok"]:
                    run["termination"], run["detail"] = "infrastructure", "independent Gurobi probe through the tunnel failed"
            return {**base, "run": run, "termination": run["termination"], "detail": run["detail"], "feedback": {"stdout": run["stdout"], "stderr": run["stderr"]}}
        finally:
            self.cleanup(attempt)

    def gurobi_preflight(self, *, variables: int = 2500, timeout: float = 120.0) -> dict:
        """Independent sandboxed Gurobi probe through the tunnel; more variables than the pip size-limited license allows."""
        from src.experiments.xl_build import manifest_for
        solver = ("import gurobipy as gp, json\n"
                  "m = gp.Model()\n"
                  f"x = m.addVars({variables}, lb=0.0, ub=1.0, obj=1.0)\n"
                  f"m.addConstr(x.sum() >= {variables} / 2)\n"
                  "m.optimize()\n"
                  "print(json.dumps({'status': m.Status, 'objective': m.ObjVal, 'threads': m.Params.Threads, 'numvars': m.NumVars}))\n")
        delivery = manifest_for({"solver.py": solver.encode()}, track="python_only")
        with tempfile.NamedTemporaryFile("w", suffix=".json", dir=self.base_dir, delete=False) as stream:
            stream.write("{}")
        try:
            record = self.execute(delivery, stream.name, RunLimits(wall_seconds=timeout), track="python_only", diagnose=False)
        finally:
            os.unlink(stream.name)
        output = None
        for line in ((record["run"] or {}).get("stdout") or "").splitlines():
            if line.startswith("{"):
                try:
                    output = json.loads(line)
                except json.JSONDecodeError:
                    pass
        ok = record["termination"] == "finished" and output is not None and output.get("status") == 2 and output.get("numvars") == variables
        return {"ok": ok, "output": output, "termination": record["termination"], "detail": record["detail"],
                "license_tunnel": (record["run"] or {}).get("license_tunnel"), "stderr": ((record["run"] or {}).get("stderr") or "")[-2000:]}
