"""Run one candidate in a private macOS filesystem, network, and process sandbox."""
from __future__ import annotations

import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from src.experiments.resources import ChildExit, CleanupError, read_commands, read_processes, stop_owned_tree
from src.agent.safe_artifacts import MAX_ARTIFACT_BYTES, UnsafeArtifact, read_regular, read_regular_text
from src.experiments.environment import RuntimeEnvironmentError, require_runtime_environment


NATIVE_SOLVER_POLICY = "private-native-sandbox-v1"
# The candidate's output.json readback limit, on fresh collection and on cached reads. The paper protocol (no policy)
# reads at most MAX_ARTIFACT_BYTES (4 MiB); a campaign config may select OUTPUT_POLICY_32MIB with its top-level
# "solver_output_policy" (src.experiments.plan), which is then recorded in every execution request and record. The
# solver.py and data.json limits are unchanged.
OUTPUT_POLICY_32MIB = "output-json-readback-32MiB-v1"
OUTPUT_LIMITS = {None: MAX_ARTIFACT_BYTES, OUTPUT_POLICY_32MIB: 32 * 1024 * 1024}
# How the committed records store the candidate's output (campaign manifest "solver_evidence_schema", recorded in the
# execution request). None, the paper's and the default, embeds the output text and its parsed object in the evidence
# and the receipt (and execution.json embeds the parsed object). OUTPUT_BY_REFERENCE, which only the 32 MiB policy can
# select, stores the output once: output.json (UTF-8 bytes) is the only copy, and every record keeps its sha256 and
# size; a resumed execution re-reads output.json, checks the sha256 and parses it with the same decoder
# (stored_record, committed_output). A Gurobi diagnostic, or a candidate whose diagnostic failed, keeps embedded evidence.
OUTPUT_BY_REFERENCE = "native-output-by-reference-v1"
EVIDENCE_SCHEMAS = {None: None, OUTPUT_POLICY_32MIB: OUTPUT_BY_REFERENCE}
# The per-solver resident-memory budget (campaign execution setting "solver_memory_policy"; None, the paper's and every
# config's but the lanes variant's, has no budget). Under a policy a watcher samples the summed RSS of the candidate's
# process group (GroupMemoryWatcher) and stops the group above the limit: the run is an execution failure of the
# candidate (exec_ok False, no output, never a Gurobi diagnostic), with MEMORY_MARKER in its stderr. The policy is part
# of the execution request, so a cached result is reused only under the same budget.
MEMORY_POLICY_2GIB = "sampled-process-group-rss-2048MiB-v1"
MEMORY_POLICIES = {None: None, MEMORY_POLICY_2GIB: {"limit_mib": 2048, "interval_seconds": 0.5}}
MEMORY_MARKER = "[framework] memory limit exceeded"
OUTPUT_STATE_UNKNOWN = object()  # the end-of-run output.json snapshot could not be taken (the result is then rejected)


def memory_message(limit_mib):
    return f"{MEMORY_MARKER}: the solver and its child processes used more than {limit_mib} MiB and were stopped"


PROCESS_POLICY = "(deny syscall-unix (syscall-number 82) (syscall-number 147) (syscall-number 244))"
class SolverLaunchError(RuntimeError):
    """A launcher failure, distinct from diagnostics of the generated program."""
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


GUROBI_CLUES = ("gurobi", "gurobipy", "license", "licence")
GUROBI_PROBE = '''import json
from pathlib import Path
import gurobipy as gp
with gp.Env(empty=True) as env:
    env.setParam('OutputFlag', 0)
    env.setParam('ThreadLimit', THREADS)
    env.setParam('Threads', THREADS)
    env.start()
    with gp.Model(env=env) as model:
        x = model.addVar(lb=0)
        model.addConstr(x >= 1)
        model.setObjective(x)
        model.optimize()
        assert model.Status == gp.GRB.OPTIMAL and abs(model.ObjVal - 1) < 1e-9
        Path('output.json').write_text(json.dumps({'status': 'optimal', 'objective_value': model.ObjVal, 'threads': model.Params.Threads}))
'''


def sha(contents):
    return hashlib.sha256(contents).hexdigest()


def stored_record(record, schema):
    """The committed form of an execution record under OUTPUT_BY_REFERENCE: no output text and no parsed output, but
    the sha256 and size of the output.json bytes and whether the parsed object is the run's result (exec_ok)."""
    stored = {key: value for key, value in record.items() if key not in ("output_text", "output")}
    contents = record["output_text"].encode("utf-8") if record["output_text"] is not None else None
    if (record["output"] is not None) != bool(record["exec_ok"]) or (record["output"] is not None and contents is None):
        raise ValueError("An execution record's parsed output must be present exactly for a completed run")
    return stored | {"evidence_schema": schema, "output_sha256": sha(contents) if contents is not None else None,
                     "output_bytes": len(contents) if contents is not None else None, "output_parsed": record["output"] is not None}


def committed_output(contents: bytes | None, result: dict, schema: str):
    """The parsed output of a committed OUTPUT_BY_REFERENCE record from the output.json bytes, whose sha256 the caller
    has checked against result["output_sha256"]: the same strict UTF-8 decoding and json.loads as a fresh run."""
    if (result.get("evidence_schema") != schema or not isinstance(result.get("output_parsed"), bool) or not isinstance(result.get("exec_ok"), bool)
            or result["output_parsed"] != result["exec_ok"] or "output" in result or "output_text" in result
            or result.get("output_bytes") != (len(contents) if contents is not None else None)):
        raise RuntimeEnvironmentError("A committed native execution record has an unknown layout")
    if not result["output_parsed"]:
        return None
    if contents is None or result.get("output_sha256") is None:
        raise RuntimeEnvironmentError("A committed native execution output changed")
    output = json.loads(contents.decode("utf-8"))
    if not isinstance(output, dict):
        raise RuntimeEnvironmentError("A committed native execution output changed")
    return output


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def native_environment():
    observed = require_runtime_environment()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeEnvironmentError("The native sandbox is validated only for macOS arm64")
    executable = Path("/usr/bin/sandbox-exec")
    if not executable.is_file():
        raise RuntimeEnvironmentError("The macOS sandbox launcher is missing")
    # Homebrew's bin/python3.12 is a posix_spawn launcher. Execute its actual runtime directly.
    python = (Path(sys.base_prefix) / "Resources/Python.app/Contents/MacOS/Python").resolve()
    packages = (Path(sys.prefix) / "lib/python3.12/site-packages").resolve()
    if not python.is_file() or not packages.is_dir():
        raise RuntimeEnvironmentError("The validated Python runtime/package layout is unavailable")
    return observed | {"sandbox_binary": str(executable), "sandbox_binary_sha256": sha(executable.read_bytes()), "dyld_profile_sha256": sha(Path("/System/Library/Sandbox/Profiles/dyld-support.sb").read_bytes()), "macos_version": platform.mac_ver()[0], "kernel_release": platform.release(),
                       "python_runtime_binary": str(python), "python_runtime_sha256": sha(python.read_bytes()), "python_runtime_root": str(Path(sys.base_prefix).resolve().parents[3]),
                       "site_packages": str(packages), "dependency_versions": {dist.metadata["Name"]: dist.version for dist in distributions() if dist.metadata["Name"]},
                       "process_policy_sha256": sha(PROCESS_POLICY.encode()), "policy": NATIVE_SOLVER_POLICY}


def same_environment(observed, expected):
    # The recorded invocation spelling may differ while the resolved binary and package tree are identical.
    return {key: value for key, value in observed.items() if key != "python_executable"} == {key: value for key, value in expected.items() if key != "python_executable"}


def seatbelt_profile(case, work, environment, license_file):
    """One OS profile is necessary: macOS rejects applying a second sandbox."""
    literal = lambda path: "(literal " + json.dumps(str(path)) + ")"
    subpath = lambda path: "(subpath " + json.dumps(str(path)) + ")"
    system = ["/System", "/usr/lib", "/usr/share", "/Library/Apple", "/private/var/db/timezone"]
    packages = [environment["site_packages"], environment["python_runtime_root"]]
    return "\n".join([
        "(version 1)", "(deny default)", '(import "dyld-support.sb")', "(allow syscall*)", PROCESS_POLICY,
        "(allow process-exec process-fork sysctl-read)",
        '(allow mach-lookup (global-name "com.apple.system.opendirectoryd.libinfo"))',
        "(allow signal (target same-sandbox))", "(allow file-read-metadata)",
        "(allow file-read* file-map-executable " + " ".join(subpath(path) for path in system + packages) + ")",
        "(allow file-read* " + " ".join(literal(path) for path in (case / "data.json", work / "solver.py", work / "gurobi.env", license_file, "/dev/urandom", "/dev/random")) + ")",
        "(allow file-read* file-write-data " + literal("/dev/null") + " " + literal("/dev/zero") + ")",
        "(allow file-read* file-write* (require-all " + subpath(work) + " (require-not " + literal(work / "solver.py") + ") (require-not " + literal(work / "gurobi.env") + ")))",
    ])


NATIVE_SOLVER_LOCK = threading.Lock()
child_exit = ChildExit  # how a candidate's exit is observed without reaping it (replaced by tests' fake launchers)
# The launch handshake of an executor with an ownership record (lanes variant): the bootstrap prints its token, then
# waits for one line on its stdin pipe, which the controller writes only after the candidate's process group is in the
# durable ownership record. On end of file (the controller died before that record) it exits with HANDSHAKE_EXIT
# without running the candidate, so a launch recorded without its group never ran candidate code.
HANDSHAKE_LINE = "GO\n"
HANDSHAKE_EXIT = 70


def bootstrap_source(token, solver_path, *, handshake=False):
    """The `python -c` program that announces the launch and runs solver.py as __main__ (the paper's without handshake)."""
    wait = f"sys.exit({HANDSHAKE_EXIT}) if sys.stdin.readline()!={HANDSHAKE_LINE!r} else None; " if handshake else ""
    return "import runpy,sys; print(" + repr(token) + ",flush=True); " + wait + "sys.argv=[" + repr(str(solver_path)) + "]; runpy.run_path(sys.argv[0],run_name='__main__')"


OWNERSHIP_RECORD = "native-solver-ownership.json"  # in the private state directory (src.experiments.locks.state_root)


def ownership_problem(path):
    """Read-only: (the unreleased record or None, why another candidate must not start or None).

    A launch whose cleanup was never recorded as verified (its controller died, or its cleanup failed) must be proved
    gone before any candidate starts, whichever campaign root or mode it belonged to. Resolvable records:
    - "running" with its process group (the PID, recorded before the candidate was let run): gone when no process of
      that group and none with its bootstrap token remains;
    - "launching" of a handshake launch (no group recorded yet): its candidate never ran (the bootstrap waits for the
      line the controller writes after the "running" record), gone once no process with its token remains.
    Any other record cannot establish absence and fails closed with the manual check to do. A PID reused since then can
    only refuse."""
    path = Path(path)
    if not path.exists():
        return None, None
    try:
        owner = json.loads(path.read_text())
    except (OSError, ValueError):
        owner = {}
    if owner.get("state") == "released":
        return None, None
    group, token = owner.get("pid"), owner.get("token")
    check = " and ".join(([f"`ps -axww -o pid,pgid,command | awk '$2 == {group}'`"] if isinstance(group, int) else []) + ([f"`pgrep -f {token}`"] if isinstance(token, str) else []))
    resolvable = isinstance(token, str) and ((owner.get("state") == "running" and type(group) is int and group > 1)
                                             or (owner.get("state") == "launching" and group is None and owner.get("handshake") is True))
    if not resolvable:
        return owner, (f"The native solver ownership record {path} (state {owner.get('state')!r}) cannot show that its launch stopped. "
                       f"Check that no process of it runs{(' with ' + check) if check else ' (no group or token is recorded)'}; when nothing remains, delete {path} and resume")
    survivors = sorted({pid for pid, item in read_processes().items() if group is not None and item.group == group and not item.state.startswith("Z")}
                       | {pid for pid, command in read_commands().items() if token in command})
    if survivors:
        return owner, (f"Native computation of an earlier launch ({owner.get('evidence')}) is still running: processes {survivors}. "
                       f"Stop them (kill -9), confirm with {check} that nothing remains, then resume")
    return owner, None


class SolverAdmission:
    """The in-process latch of candidate launches. It closes, while the solver lock is still held, when a launch ends
    without verified cleanup (CleanupError, an interruption during cleanup, any failure after the process started that
    is not a launcher failure with verified cleanup); every later candidate or diagnostic launch of this controller is
    then refused, so no waiting worker can start a solver next to computation that may survive."""
    def __init__(self):
        self.reason = None

    def require_open(self):
        if self.reason is not None:
            raise CleanupError(f"Native solver admission is closed in this controller: {self.reason}; verify that the owned solver processes are gone, then restart")

    def close(self, reason):
        if self.reason is None:
            self.reason = reason


SOLVER_ADMISSION = SolverAdmission()  # read and changed only while NATIVE_SOLVER_LOCK is held


def cleanup_established(error):
    """Whether a failed launch is known to leave no computation behind: a launcher failure before the process started
    or after its verified cleanup."""
    return isinstance(error, SolverLaunchError) and ("pid" not in error.evidence or bool(error.evidence.get("cleanup_verified")))


class GroupMemoryWatcher:
    """Sampled enforcement of a resident-memory budget for one process group: the candidate and every process it starts
    (the sandbox's process policy forbids changing the group or session, so membership is by PGID and survives the
    parent's exit). The summed RSS of the group's live processes is sampled at once and then every `interval` seconds;
    above `limit_mib` the whole group is killed and the violation latched. Shared pages count once per process. A
    process can exceed the budget between two samples, and one sample takes the process query's time (at most 5 s):
    this is sampled enforcement, not an instantaneous ceiling. A failed sample also kills the group, so nothing runs
    unmonitored; the caller reports it as an infrastructure failure, never charged to the candidate. The group is
    signalled only while `anchored()` holds (its leader is unreaped, so the group number is still the candidate's);
    otherwise the watcher stops with an error and signals nothing."""
    def __init__(self, group, *, limit_mib, interval, sampler, anchored=lambda: True):
        self.group, self.limit_mib, self.interval, self.sampler, self.anchored = group, limit_mib, interval, sampler, anchored
        self.peak_mib, self.samples, self.exceeded, self.error = 0.0, 0, False, None
        self.stopping, self.done = threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self.run, name=f"native-memory-watcher-{group}", daemon=True)
        self.thread.start()

    def run(self):
        try:
            while not self.stopping.is_set():
                try:
                    processes = self.sampler()
                    rss_mib = sum(item.rss_kib for item in processes.values() if item.group == self.group and not item.state.startswith("Z")) / 1024
                except Exception as error:  # the process query failed or timed out: no unmonitored execution
                    self.error = f"{type(error).__name__}: {error}"
                    self.stop_group()
                    return
                self.samples += 1
                self.peak_mib = max(self.peak_mib, rss_mib)
                if rss_mib > self.limit_mib:
                    self.exceeded = True
                    self.stop_group()
                    return
                self.stopping.wait(self.interval)
        finally:
            self.done.set()

    def stop_group(self):
        if not self.anchored():
            self.error = self.error or f"process group {self.group} can no longer be identified (its leader was reaped); nothing was signalled"
            return
        try:
            os.killpg(self.group, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass  # already gone, or not ours any more; the caller's verified cleanup decides

    def close(self):
        """Stop sampling and join (before the group's cleanup, any readback and the release of the solver lock)."""
        self.stopping.set()
        self.thread.join()


class NativeSandboxExecutor:
    # Bytes kept from the end of each of stdout and stderr (the paper's 12 000; the scale-up protocol uses 20 000).
    stream_tail_bytes = 12000
    output_policy = None  # OUTPUT_LIMITS key; None is the paper's 4 MiB output.json readback
    output_limit = OUTPUT_LIMITS[None]
    evidence_schema = None  # EVIDENCE_SCHEMAS value; None embeds the output in the records (the paper's layout)
    # The lanes variant only (src.experiments.campaign): the memory budget, the durable host-wide ownership record of
    # the running candidate with the launch handshake (a later controller, of any campaign root, refuses to overlap
    # computation an earlier one left behind) and the solver-lock queue time in each execution record.
    # None/None/False keep the paper's executor and records.
    memory_policy = None  # MEMORY_POLICIES key
    memory_limit_mib = None
    ownership = None  # Path of the ownership record in the private state directory (src.experiments.locks.state_root)
    record_queue_seconds = False
    memory_sampler = staticmethod(read_processes)  # {pid: ProcessSample}; replaced by tests

    def __init__(self, *, license_file: Path, expected_environment: dict | None = None, output_policy: str | None = None, evidence_schema: str | None = None,
                 memory_policy: str | None = None, ownership: Path | None = None, record_queue_seconds: bool = False):
        if output_policy not in OUTPUT_LIMITS:
            raise ValueError(f"Unknown solver output policy: {output_policy!r}")
        if evidence_schema is not None and evidence_schema != EVIDENCE_SCHEMAS[output_policy]:
            raise ValueError(f"Evidence schema {evidence_schema!r} is not available under the solver output policy {output_policy!r}")
        if memory_policy not in MEMORY_POLICIES:
            raise ValueError(f"Unknown solver memory policy: {memory_policy!r}")
        if output_policy is not None:
            self.output_policy, self.output_limit = output_policy, OUTPUT_LIMITS[output_policy]
        if evidence_schema is not None:
            self.evidence_schema = evidence_schema
        if memory_policy is not None:
            self.memory_policy, self.memory_limit_mib = memory_policy, MEMORY_POLICIES[memory_policy]["limit_mib"]
        if ownership is not None:
            self.ownership = Path(ownership)
        self.record_queue_seconds = bool(record_queue_seconds)
        self.license_file = license_file.resolve(strict=True)
        if not self.license_file.is_file():
            raise RuntimeEnvironmentError("The declared local Gurobi license is unavailable")
        self.environment = native_environment()
        if expected_environment is not None:
            if not same_environment(self.environment, expected_environment):
                raise RuntimeEnvironmentError("Native sandbox environment differs from the frozen campaign")
            self.environment = expected_environment

    def __call__(self, attempt_dir, timeout, threads=1):
        attempt_dir = Path(attempt_dir).resolve()
        if not same_environment(native_environment(), self.environment):
            raise RuntimeEnvironmentError("Native sandbox dependencies or launcher changed during execution")
        solver = read_regular(attempt_dir, "solver.py")
        data = read_regular(attempt_dir.parent.parent, "data.json")
        fingerprint = {"solver_sha256": sha(solver), "input_sha256": sha(data), "timeout": timeout, "threads": threads, "environment": self.environment}
        if self.output_policy is not None:
            fingerprint["output_policy"] = self.output_policy
        if self.evidence_schema is not None:
            fingerprint["evidence_schema"] = self.evidence_schema
        if self.memory_policy is not None:
            fingerprint["memory_policy"] = self.memory_policy
        saved = attempt_dir / "native_solver_receipt.json"
        if saved.exists():
            receipt = json.loads(saved.read_text())
            if receipt["request"] != fingerprint:
                raise RuntimeEnvironmentError("A committed native execution request changed")
            expected_output = receipt["result"]["output_sha256"]
            output = read_regular(attempt_dir, "output.json", missing_ok=True, maximum=self.output_limit)
            if (sha(output) if output is not None else None) != expected_output:
                raise RuntimeEnvironmentError("A committed native execution output changed")
            if self.evidence_schema is not None:
                result = receipt["result"]
                return result["exec_ok"], result["stdout"], result["stderr"], committed_output(output, result, self.evidence_schema)
            return tuple(receipt["result"][name] for name in ("exec_ok", "stdout", "stderr", "output"))
        identifier = uuid.uuid4().hex
        evidence = attempt_dir / f"native_execution_{identifier}.json"
        queued = time.monotonic()
        with NATIVE_SOLVER_LOCK:  # one candidate solver on the host at a time, across campaign workers
            waited = time.monotonic() - queued
            record = self.launch(solver, data, timeout, threads, evidence)
        # A known memory-limit stop is the candidate's own failure: it never triggers the Gurobi diagnostic.
        if not record["exec_ok"] and not record["integrity_violations"] and not record.get("memory_limit_exceeded") and any(clue in record["stderr"].lower() for clue in GUROBI_CLUES):
            probe_path = attempt_dir / f"gurobi_diagnostic_{identifier}.json"
            queued = time.monotonic()
            with NATIVE_SOLVER_LOCK:
                waited += time.monotonic() - queued
                probe = self.launch(GUROBI_PROBE.replace("THREADS", str(threads)).encode(), b"{}", timeout, threads, probe_path)
            record["gurobi_diagnostic"] = {"path": probe_path.name, "ok": probe["exec_ok"] and probe["output"] == {"status": "optimal", "objective_value": 1.0, "threads": threads}}
            save(evidence, record)
            if not record["gurobi_diagnostic"]["ok"]:
                raise RuntimeEnvironmentError("The independent Gurobi diagnostic failed; repair the environment before resuming")
        if self.record_queue_seconds:
            # Time spent waiting for the solver lock (candidate and diagnostic), never part of execution_seconds.
            record["solver_queue_seconds"] = waited
        output_path = attempt_dir / "output.json"
        output_path.unlink(missing_ok=True)
        if record["output_text"] is not None:
            temporary = attempt_dir / "output.json.tmp"
            if self.evidence_schema is not None:
                temporary.write_bytes(record["output_text"].encode("utf-8"))  # the bytes whose sha256 the records keep
            else:
                temporary.write_text(record["output_text"])
            temporary.replace(output_path)
        record["output_sha256"] = sha(record["output_text"].encode()) if record["output_text"] is not None else None
        stored = record if self.evidence_schema is None else stored_record(record, self.evidence_schema)
        save(evidence, stored)
        save(attempt_dir / "solver_invocation.json", {"backend": "native-macos-seatbelt", "policy": NATIVE_SOLVER_POLICY, "request": fingerprint, "evidence": evidence.name})
        save(saved, {"request": fingerprint, "result": stored})
        return tuple(record[name] for name in ("exec_ok", "stdout", "stderr", "output"))

    def launch(self, solver, data, timeout, threads, evidence):
        """One execute_bytes while the caller holds NATIVE_SOLVER_LOCK: refused once this controller's solver admission is
        closed or while an earlier launch's computation survives (ownership record); closes admission, before the lock
        is released, when the launch ends without established cleanup."""
        SOLVER_ADMISSION.require_open()
        self.require_released_owner()
        try:
            return self.execute_bytes(solver, data, timeout, threads, evidence=evidence)
        except BaseException as error:
            if not cleanup_established(error):
                SOLVER_ADMISSION.close(f"cleanup of the launch recorded in {evidence.name} was not verified ({type(error).__name__}: {error})")
            raise

    def verify_ownership(self):
        """At controller start: refuse (CleanupError) while computation of an earlier launch of this campaign survives."""
        with NATIVE_SOLVER_LOCK:
            self.require_released_owner()

    def require_released_owner(self):
        """With an ownership record (lanes variant): refuse (ownership_problem) while an earlier launch may survive, of
        whichever campaign root; mark a verified record released."""
        if self.ownership is None:
            return
        owner, problem = ownership_problem(self.ownership)
        if problem:
            raise CleanupError(problem)
        if owner is not None:
            save(self.ownership, owner | {"state": "released", "released_after_verification_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    def own(self, token, evidence, state, pid=None):
        """The durable host-wide ownership record of the launch in progress (lanes variant only; with the handshake)."""
        if self.ownership is not None:
            save(self.ownership, {"state": state, "token": token, "pid": pid, "handshake": True, "evidence": str(evidence), "controller_pid": os.getpid(),
                                  "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    @staticmethod
    def stop_group_now(process):
        """SIGKILL the candidate's whole group at once, while its leader is unreaped (so the group number is still the
        candidate's): the end of the wait is the end of its run, before any blocking step (the watcher's join)."""
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass  # already gone; the verified cleanup decides

    @staticmethod
    def output_state(work):
        """The metadata of output.json (inode, size, modification time in ns) or None, to detect a write after the run
        ended. This is metadata change detection, not a content identity (a same-size rewrite that restores the
        modification time is not detected); a full read here would delay the stop."""
        try:
            status = (work / "output.json").lstat()
        except FileNotFoundError:
            return None
        return status.st_ino, status.st_size, status.st_mtime_ns

    @classmethod
    def output_changed(cls, work, ended) -> bool:
        """Whether output.json may differ from its state when the run ended (an unknown state on either side counts)."""
        if ended is OUTPUT_STATE_UNKNOWN:
            return True
        try:
            return cls.output_state(work) != ended
        except OSError:
            return True

    @staticmethod
    def let_run(process) -> bool:
        """End the launch handshake: the bootstrap may now run the candidate (its group is in the ownership record).
        False when the line could not be delivered (the bootstrap already stopped)."""
        try:
            process.stdin.write(HANDSHAKE_LINE.encode())
            process.stdin.close()
            return True
        except (BrokenPipeError, OSError, ValueError):
            return False

    @staticmethod
    def wait(exits, timeout, watcher):
        """Whether the candidate ran out of time. Its exit is observed without reaping it (resources.ChildExit), so its
        process group keeps its number through the watcher's and the cleanup's signals; stop_owned_tree reaps it after
        the final verification. With a memory watcher the wait also ends once the watcher has stopped the group or
        failed, so the cleanup is immediate."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            if exits.wait(remaining if watcher is None else min(remaining, 0.25)) or (watcher is not None and watcher.done.is_set()):
                return False

    def read_output(self, work: Path) -> str | None:
        """The candidate's output.json in its working directory (None if absent), within this protocol's readback limit."""
        return read_regular_text(work, "output.json", missing_ok=True, maximum=self.output_limit)

    def execute_bytes(self, solver, data, timeout, threads, *, evidence: Path):
        """No archive path is granted to the candidate; cleanup precedes every readback."""
        started = time.monotonic()
        token = "DECOMPBENCH_READY_" + uuid.uuid4().hex
        record = {"policy": NATIVE_SOLVER_POLICY, "solver_sha256": sha(solver), "input_sha256": sha(data), "timeout": timeout, "threads": threads,
                  "exec_ok": False, "output": None, "output_text": None, "stdout": "", "stderr": "", "integrity_violations": [], "cleanup_verified": False, "bootstrap_ready": False}
        if self.output_policy is not None:
            record.update(output_policy=self.output_policy, output_limit_bytes=self.output_limit)
        if self.memory_policy is not None:
            memory = MEMORY_POLICIES[self.memory_policy]
            record.update(memory_policy=self.memory_policy, memory_limit_mib=memory["limit_mib"], memory_sampling="summed RSS of the process group's live processes",
                          memory_sampling_interval_seconds=memory["interval_seconds"])
        evidence.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="decompbench-native-", dir="/private/tmp") as temporary:
            root = Path(temporary)
            case = root / "case"
            work = case / "attempts/attempt_01"
            scratch = work / "tmp"
            scratch.mkdir(parents=True)
            immutable = {case / "data.json": data, work / "solver.py": solver, work / "gurobi.env": f"ThreadLimit {threads}\nThreads {threads}\n".encode()}
            stamps = {}
            for path, contents in immutable.items():
                path.write_bytes(contents)
                path.chmod(0o444)
                status = path.stat()
                stamps[path] = (status.st_ino, stat.S_IMODE(status.st_mode))
            profile = seatbelt_profile(case, work, self.environment, self.license_file)
            handshake = self.ownership is not None
            bootstrap = bootstrap_source(token, work / "solver.py", handshake=handshake)
            command = [self.environment["sandbox_binary"], "-p", profile, self.environment["python_runtime_binary"], "-B", "-s", "-P", "-u", "-c", bootstrap]
            environment = {"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch), "GRB_LICENSE_FILE": str(self.license_file), "PYTHONPATH": self.environment["site_packages"],
                           "PYTHONEXECUTABLE": self.environment["python_runtime_binary"], "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1", "PYTHONDONTWRITEBYTECODE": "1",
                           **{key: str(threads) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}}
            record.update(workspace=str(work), seatbelt_profile=profile, seatbelt_profile_sha256=sha(profile.encode()), network_enabled=False, process_policy=PROCESS_POLICY, environment_keys=sorted(environment), command=command)
            save(evidence, record | {"phase": "launching"})
            timed_out, released, ended = False, True, None
            process = watcher = exits = None
            try:
                self.own(token, evidence, "launching")
                with (root / "stdout").open("wb") as stdout, (root / "stderr").open("wb") as stderr:
                    process = subprocess.Popen(command, cwd=work, env=environment, stdin=subprocess.PIPE if handshake else subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
                    try:
                        record["pid"] = process.pid
                        exits = child_exit(process)  # from here on the candidate is never reaped before the verified cleanup
                        if self.memory_policy is not None:  # the new session's group is the candidate's
                            watcher = GroupMemoryWatcher(process.pid, limit_mib=self.memory_limit_mib, interval=MEMORY_POLICIES[self.memory_policy]["interval_seconds"], sampler=self.memory_sampler,
                                                         anchored=lambda: process.returncode is None)
                        self.own(token, evidence, "running", process.pid)  # durable before the candidate may run
                        save(evidence, record | {"phase": "started"})
                        if handshake:
                            released = self.let_run(process)
                        timed_out = self.wait(exits, timeout, watcher)
                    finally:
                        if handshake and process.stdin is not None and not process.stdin.closed:
                            process.stdin.close()  # end of file: a bootstrap still waiting exits without running the candidate
                        try:
                            if watcher is not None:
                                # The run ends when the wait ends: its output as it stands now, and its whole group stopped
                                # at once (a sample in progress may hold the join for up to its query's 5 s timeout). A
                                # snapshot that cannot be taken (for example a working directory made unsearchable) is
                                # unknown: the result is then rejected, and the stop, join and cleanup still happen.
                                try:
                                    ended = self.output_state(work)
                                except OSError:
                                    ended = OUTPUT_STATE_UNKNOWN
                                try:
                                    self.stop_group_now(process)
                                finally:
                                    watcher.close()  # joined before the cleanup, every readback and the release of the solver lock
                        finally:
                            try:
                                stop_owned_tree(process, grace=0.1)
                            finally:
                                if exits is not None:
                                    exits.close()
                        record["cleanup_verified"] = True
                self.own(token, evidence, "released", process.pid)
            except OSError as error:
                record.update(phase="launch_failed", error_type=type(error).__name__, errno=error.errno)
                save(evidence, record)
                raise SolverLaunchError("Native sandbox launch failed", record) from error
            except BaseException as error:
                record.update(phase="cleanup_failed" if isinstance(error, CleanupError) else "interrupted", error_type=type(error).__name__)
                save(evidence, record)
                raise
            record["returncode"] = process.returncode
            if not released:  # the candidate was never let run: not its failure
                record.update(phase="handshake_failed", execution_seconds=time.monotonic() - started)
                save(evidence, record)
                raise SolverLaunchError("The native sandbox bootstrap stopped before the launch handshake", record)
            exceeded = False
            if watcher is not None:
                # Precedence: a sampling failure, then a missing bootstrap announcement (both infrastructure, never
                # charged: the candidate may never have started), then the memory limit, then the timeout.
                exceeded = watcher.exceeded and watcher.error is None
                record.update(memory_samples=watcher.samples, memory_peak_mib=round(watcher.peak_mib, 1), memory_limit_exceeded=exceeded,
                              termination="memory_limit_exceeded" if exceeded else "timeout" if timed_out else "exited")
                if watcher.error is not None:
                    record.update(phase="memory_sampling_failed", memory_sampling_error=watcher.error, execution_seconds=time.monotonic() - started)
                    save(evidence, record)
                    raise SolverLaunchError("Native memory sampling failed; the candidate was stopped and is not charged", record)
            with (root / "stdout").open("rb") as stream:
                record["bootstrap_ready"] = token.encode() + b"\n" in stream.read(8192)
            for name in ("stdout", "stderr"):
                with (root / name).open("rb") as stream:
                    stream.seek(0, os.SEEK_END)
                    stream.seek(max(0, stream.tell() - self.stream_tail_bytes))
                    record[name] = stream.read().decode(errors="replace").replace(token + "\n", "")
            if not record["bootstrap_ready"]:
                record.update(phase="bootstrap_failed", execution_seconds=time.monotonic() - started, **({"termination": "bootstrap_failed"} if watcher is not None else {}))
                save(evidence, record)
                raise SolverLaunchError("Native sandbox stopped before the candidate started", record)
            # With a watcher, only output as it stood when the wait ended can count: a write after that (a surviving
            # child, or one racing the stop) is never accepted.
            late = watcher is not None and self.output_changed(work, ended)
            if watcher is not None:
                record["output_changed_after_end"] = late
            for path, contents in immutable.items():
                try:
                    observed = read_regular(path.parent, path.name)
                    status = path.lstat()
                    if observed != contents or (status.st_ino, stat.S_IMODE(status.st_mode)) != stamps[path]:
                        raise UnsafeArtifact("immutable input changed")
                except (OSError, UnsafeArtifact):
                    record["integrity_violations"].append(str(path.relative_to(case)))
            try:
                record["output_text"] = self.read_output(work)
                output = json.loads(record["output_text"]) if record["output_text"] is not None else None
                if not isinstance(output, dict):
                    raise ValueError("output.json missing, invalid, or not an object")
                record["output"] = output
                record["exec_ok"] = process.returncode == 0 and not timed_out and not record["integrity_violations"] and not exceeded and not late
            except (UnsafeArtifact, ValueError, RecursionError) as error:
                record["stderr"] += "\n[framework] " + str(error)
            if exceeded:
                record["stderr"] += "\n" + memory_message(self.memory_limit_mib)
            elif timed_out:
                record["stderr"] += f"\n[framework] solver execution timed out after {timeout} seconds"
            if record["integrity_violations"]:
                record["stderr"] += "\n[framework] immutable execution inputs changed"
            if late:
                record["stderr"] += "\n[framework] output.json changed after the run ended; only the output written before it ended counts"
            if not record["exec_ok"]:
                record["output"] = None
                if not record["stderr"]:
                    record["stderr"] = f"returncode={process.returncode}"
            record.update(phase="complete", execution_seconds=time.monotonic() - started, timed_out=timed_out)
            save(evidence, record)
            return record
