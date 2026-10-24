"""Run one candidate in a private macOS filesystem, network, and process sandbox."""
from __future__ import annotations

import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import tempfile
import time
import uuid

from src.experiments.resources import CleanupError, stop_owned_tree
from src.agent.safe_artifacts import UnsafeArtifact, read_regular, read_regular_text
from src.experiments.environment import RuntimeEnvironmentError, require_runtime_environment


NATIVE_SOLVER_POLICY = "private-native-sandbox-v1"
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


class NativeSandboxExecutor:
    def __init__(self, *, license_file: Path, expected_environment: dict | None = None):
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
        saved = attempt_dir / "native_solver_receipt.json"
        if saved.exists():
            receipt = json.loads(saved.read_text())
            if receipt["request"] != fingerprint:
                raise RuntimeEnvironmentError("A committed native execution request changed")
            expected_output = receipt["result"]["output_sha256"]
            output = read_regular(attempt_dir, "output.json", missing_ok=True)
            if (sha(output) if output is not None else None) != expected_output:
                raise RuntimeEnvironmentError("A committed native execution output changed")
            return tuple(receipt["result"][name] for name in ("exec_ok", "stdout", "stderr", "output"))
        identifier = uuid.uuid4().hex
        evidence = attempt_dir / f"native_execution_{identifier}.json"
        record = self.execute_bytes(solver, data, timeout, threads, evidence=evidence)
        if not record["exec_ok"] and not record["integrity_violations"] and any(clue in record["stderr"].lower() for clue in GUROBI_CLUES):
            probe_path = attempt_dir / f"gurobi_diagnostic_{identifier}.json"
            probe = self.execute_bytes(GUROBI_PROBE.replace("THREADS", str(threads)).encode(), b"{}", timeout, threads, evidence=probe_path)
            record["gurobi_diagnostic"] = {"path": probe_path.name, "ok": probe["exec_ok"] and probe["output"] == {"status": "optimal", "objective_value": 1.0, "threads": threads}}
            save(evidence, record)
            if not record["gurobi_diagnostic"]["ok"]:
                raise RuntimeEnvironmentError("The independent Gurobi diagnostic failed; repair the environment before resuming")
        output_path = attempt_dir / "output.json"
        output_path.unlink(missing_ok=True)
        if record["output_text"] is not None:
            temporary = attempt_dir / "output.json.tmp"
            temporary.write_text(record["output_text"])
            temporary.replace(output_path)
        record["output_sha256"] = sha(record["output_text"].encode()) if record["output_text"] is not None else None
        save(evidence, record)
        save(attempt_dir / "solver_invocation.json", {"backend": "native-macos-seatbelt", "policy": NATIVE_SOLVER_POLICY, "request": fingerprint, "evidence": evidence.name})
        save(saved, {"request": fingerprint, "result": record})
        return tuple(record[name] for name in ("exec_ok", "stdout", "stderr", "output"))

    def execute_bytes(self, solver, data, timeout, threads, *, evidence: Path):
        """No archive path is granted to the candidate; cleanup precedes every readback."""
        started = time.monotonic()
        token = "DECOMPBENCH_READY_" + uuid.uuid4().hex
        record = {"policy": NATIVE_SOLVER_POLICY, "solver_sha256": sha(solver), "input_sha256": sha(data), "timeout": timeout, "threads": threads,
                  "exec_ok": False, "output": None, "output_text": None, "stdout": "", "stderr": "", "integrity_violations": [], "cleanup_verified": False, "bootstrap_ready": False}
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
            bootstrap = "import runpy,sys; print(" + repr(token) + ",flush=True); sys.argv=[" + repr(str(work / "solver.py")) + "]; runpy.run_path(sys.argv[0],run_name='__main__')"
            command = [self.environment["sandbox_binary"], "-p", profile, self.environment["python_runtime_binary"], "-B", "-s", "-P", "-u", "-c", bootstrap]
            environment = {"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch), "GRB_LICENSE_FILE": str(self.license_file), "PYTHONPATH": self.environment["site_packages"],
                           "PYTHONEXECUTABLE": self.environment["python_runtime_binary"], "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1", "PYTHONDONTWRITEBYTECODE": "1",
                           **{key: str(threads) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}}
            record.update(workspace=str(work), seatbelt_profile=profile, seatbelt_profile_sha256=sha(profile.encode()), network_enabled=False, process_policy=PROCESS_POLICY, environment_keys=sorted(environment), command=command)
            save(evidence, record | {"phase": "launching"})
            timed_out = False
            process = None
            try:
                with (root / "stdout").open("wb") as stdout, (root / "stderr").open("wb") as stderr:
                    process = subprocess.Popen(command, cwd=work, env=environment, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
                    record["pid"] = process.pid
                    save(evidence, record | {"phase": "started"})
                    try:
                        process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                    finally:
                        stop_owned_tree(process, grace=0.1)
                        record["cleanup_verified"] = True
            except OSError as error:
                record.update(phase="launch_failed", error_type=type(error).__name__, errno=error.errno)
                save(evidence, record)
                raise SolverLaunchError("Native sandbox launch failed", record) from error
            except BaseException as error:
                record.update(phase="cleanup_failed" if isinstance(error, CleanupError) else "interrupted", error_type=type(error).__name__)
                save(evidence, record)
                raise
            record["returncode"] = process.returncode
            with (root / "stdout").open("rb") as stream:
                record["bootstrap_ready"] = token.encode() + b"\n" in stream.read(8192)
            for name in ("stdout", "stderr"):
                with (root / name).open("rb") as stream:
                    stream.seek(0, os.SEEK_END)
                    stream.seek(max(0, stream.tell() - 12000))
                    record[name] = stream.read().decode(errors="replace").replace(token + "\n", "")
            if not record["bootstrap_ready"]:
                record.update(phase="bootstrap_failed", execution_seconds=time.monotonic() - started)
                save(evidence, record)
                raise SolverLaunchError("Native sandbox stopped before the candidate started", record)
            for path, contents in immutable.items():
                try:
                    observed = read_regular(path.parent, path.name)
                    status = path.lstat()
                    if observed != contents or (status.st_ino, stat.S_IMODE(status.st_mode)) != stamps[path]:
                        raise UnsafeArtifact("immutable input changed")
                except (OSError, UnsafeArtifact):
                    record["integrity_violations"].append(str(path.relative_to(case)))
            try:
                record["output_text"] = read_regular_text(work, "output.json", missing_ok=True)
                output = json.loads(record["output_text"]) if record["output_text"] is not None else None
                if not isinstance(output, dict):
                    raise ValueError("output.json missing, invalid, or not an object")
                record["output"] = output
                record["exec_ok"] = process.returncode == 0 and not timed_out and not record["integrity_violations"]
            except (UnsafeArtifact, ValueError, RecursionError) as error:
                record["stderr"] += "\n[framework] " + str(error)
            if timed_out:
                record["stderr"] += f"\n[framework] solver execution timed out after {timeout} seconds"
            if record["integrity_violations"]:
                record["stderr"] += "\n[framework] immutable execution inputs changed"
            if not record["exec_ok"]:
                record["output"] = None
                if not record["stderr"]:
                    record["stderr"] = f"returncode={process.returncode}"
            record.update(phase="complete", execution_seconds=time.monotonic() - started, timed_out=timed_out)
            save(evidence, record)
            return record
