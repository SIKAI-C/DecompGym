"""Multi-file delivery manifest, track policy and the framework-run C++ build (protocol v4 §5; reviews 42 #8, 44, 48).

Tracks: ``python_only`` (paper rules: ``solver.py`` only; any ``cpp/`` file is rejected) and ``cpp_allowed`` (PDPTW C++ track:
``solver.py`` plus flat ``cpp/*.cpp`` / ``cpp/*.hpp``). The hard delivery rules are the track rule and the manifest limits.

Delivery: one canonical validator (`check_files`) serves both delivery paths (a workspace directory via `collect_delivery`,
an in-memory mapping via `manifest_for`) and `verify_manifest`. It requires ``solver.py``, allows only the track's paths
(names ``[A-Za-z0-9_][A-Za-z0-9_.-]*``), at most 64 C++ files, 2 MiB per file and 8 MiB in total; while collecting a
directory these limits are enforced cumulatively from ``lstat`` sizes before any file is read, and every read is capped by the
remaining budget. Symlinks, hard links, special files, nested directories, ``..``/absolute paths and other names are
rejected (strict mode raises; lenient mode records and skips them; limit and track violations always raise). Every delivered
file enters a hashed manifest (path, size, sha256, track), preserved with the attempt and shown to the method audit.

C++ policy (caller decision 2026-11-04): the C++ track may use anything reachable inside the sandbox (threads, libc's
pthread, dlopen, ...); fairness comes from OS enforcement (1-CPU cpuset, memory cgroup, no network, rlimits). The source
audit (`xl_audit`) and the symbols the built files import (`elf_imports`) and need (`elf_needed`) are recorded for the method
audit and never reject a build.

Build (outside T_f, cost reported separately), with the pinned Phoenix compiler gcc/12.3.0, inside the sandbox (no network,
sources read-only, only ``/work/build`` writable, private tmpfs ``/tmp``), in a separate Slurm job step limited to 4 GiB of
memory for the whole compiler tree plus its tmpfs scratch (``srun --overlap --mem=4G``), whose limits (memory, swap, cpuset,
OOM group) are verified inside the step before the compiler starts:

    g++ -std=c++17 -O2 -fPIC -shared cpp/*.cpp -o build/libcore.so
    g++ -std=c++17 -O2 cpp/*.cpp -o build/core          (only if the library defines the symbol ``main``)

``cpp/*.cpp`` is expanded by the framework in byte order. Both commands share one cumulative budget (default 120 s); each also
has RLIMIT_AS, RLIMIT_NPROC, RLIMIT_FSIZE (each object, temporary and output file) and a cap on compiler output (the last
20 KB are kept for reflection). Any failure is ``build_error``. ``build/`` is host-backed: its hard bound is RLIMIT_FSIZE per
file times the two declared targets, and it may contain nothing else (checked after each step). The run then sees
``DECOMPGYM_CORE`` and, if built, ``DECOMPGYM_CORE_BIN``. "Defines main" is read from the built library's symbol tables.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import stat
import struct
import time

from src.agent.safe_artifacts import UnsafeArtifact, read_regular

MANIFEST_VERSION = "decompgym-xl-delivery-v2"
TRACKS = ("python_only", "cpp_allowed")
SOURCE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}\.(cpp|hpp)")
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
MAX_SOURCES = 64
MAX_DIRECTORY_ENTRIES = 256   # entries inspected per directory before the delivery is rejected
GiB = 1 << 30
MiB = 1 << 20

ALLOWED_NEEDED = frozenset({"libstdc++.so.6", "libm.so.6", "libgcc_s.so.1", "libc.so.6", "ld-linux-x86-64.so.2"})
NOTABLE_IMPORTS = re.compile(r"pthread_|thrd_|_ZNSt6thread|dlopen|dlsym|dlmopen|fork|vfork|exec[lv]|system|popen|socket|connect|clone|GRB")


class DeliveryError(ValueError):
    """The delivered workspace violates the multi-file contract."""

    def __init__(self, message: str, rejected: list | None = None):
        super().__init__(message)
        self.rejected = rejected or []


@dataclass(frozen=True)
class BuildLimits:
    budget_seconds: float = 120.0          # cumulative over both targets (Slurm step start-up included)
    memory_bytes: int = 4 * GiB            # whole compiler tree + tmpfs scratch: the Slurm step's memory cgroup
    memory_step: str = "srun"              # "srun" (separate step, hard) or "none" (only the per-process limit below)
    address_space_bytes: int = 4 * GiB     # RLIMIT_AS per compiler process (inner guard)
    max_processes: int = 32                # RLIMIT_NPROC (driver, cc1plus, as, collect2, ld)
    file_size_bytes: int = 256 * MiB       # RLIMIT_FSIZE (objects, temporaries, outputs)
    open_files: int = 256
    output_cap_bytes: int = 16 * MiB       # compiler stdout + stderr per step; beyond it the step is killed
    scratch_quota_bytes: int = 1 * GiB     # build/ (host) + /tmp (tmpfs) usage, polled (diagnostic early stop)
    tail_bytes: int = 20_000               # kept for reflection

    def rlimits(self, remaining: float) -> dict:
        return {"address_space": self.address_space_bytes, "processes": self.max_processes, "file_size": self.file_size_bytes,
                "open_files": self.open_files, "core": 0, "cpu_seconds": int(remaining) + 5}


# ------------------------------------------------------------------------------------------------------- delivery

def path_problem(relative: str, track: str = "cpp_allowed") -> str | None:
    """None if `relative` is an allowed delivery path for `track`, else the reason."""
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        return "invalid path"
    if relative.startswith("/"):
        return "absolute path"
    parts = relative.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return "path traversal or empty component"
    if parts == ["solver.py"]:
        return None
    if len(parts) == 2 and parts[0] == "cpp":
        if track != "cpp_allowed":
            return f"C++ sources are not allowed in the {track} track"
        return None if SOURCE_NAME.fullmatch(parts[1]) else "cpp/ accepts only flat *.cpp and *.hpp names"
    return "not an allowed delivery path (solver.py" + (", cpp/*.cpp, cpp/*.hpp)" if track == "cpp_allowed" else " only)")


def check_files(files: dict[str, bytes], track: str) -> None:
    """The canonical delivery validator (paths, track, required solver, count and size limits)."""
    if track not in TRACKS:
        raise ValueError(f"unknown track {track!r}; expected one of {TRACKS}")
    problems = [{"path": path, "reason": reason} for path in files if (reason := path_problem(path, track))]
    if problems:
        raise DeliveryError("disallowed delivery paths", problems)
    if "solver.py" not in files:
        raise DeliveryError("solver.py is required")
    if sum(1 for path in files if path.startswith("cpp/")) > MAX_SOURCES:
        raise DeliveryError(f"at most {MAX_SOURCES} C++ files")
    for path, contents in files.items():
        if not isinstance(contents, (bytes, bytearray)) or len(contents) > MAX_FILE_BYTES:
            raise DeliveryError(f"{path}: not bytes or larger than {MAX_FILE_BYTES} bytes")
    if sum(len(contents) for contents in files.values()) > MAX_TOTAL_BYTES:
        raise DeliveryError(f"delivery larger than {MAX_TOTAL_BYTES} bytes")


def _entry(path: str, contents: bytes) -> dict:
    return {"path": path, "size": len(contents), "sha256": hashlib.sha256(contents).hexdigest()}


def _digest(track: str, entries: list) -> str:
    return hashlib.sha256(json.dumps({"track": track, "files": entries}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def manifest_for(files: dict[str, bytes], rejected: list | None = None, *, track: str) -> tuple[dict, dict[str, bytes]]:
    """Manifest of an in-memory delivery (path -> bytes); raises DeliveryError on any violation."""
    check_files(files, track)
    entries = [_entry(path, bytes(files[path])) for path in sorted(files)]
    manifest = {"version": MANIFEST_VERSION, "track": track, "files": entries, "rejected": list(rejected or []), "sha256": _digest(track, entries)}
    return manifest, {path: bytes(contents) for path, contents in files.items()}


def collect_delivery(root: Path, *, track: str, strict: bool = True) -> tuple[dict, dict[str, bytes]]:
    """Read a delivered workspace without following links, enforcing every limit before reading; return (manifest, files)."""
    if track not in TRACKS:
        raise ValueError(f"unknown track {track!r}")
    root = Path(root)
    files: dict[str, bytes] = {}
    rejected: list = []
    budget = {"total": 0, "sources": 0}

    def reject(path: str, reason: str) -> None:
        rejected.append({"path": path, "reason": reason})

    def listing(path: str) -> list:
        entries = []
        with os.scandir(path) as iterator:
            for entry in iterator:
                entries.append(entry)
                if len(entries) > MAX_DIRECTORY_ENTRIES:
                    raise DeliveryError(f"more than {MAX_DIRECTORY_ENTRIES} entries in {Path(path).name or path}")
        return sorted(entries, key=lambda item: item.name)

    def read(relative: str, size: int) -> None:
        if size > MAX_FILE_BYTES:
            raise DeliveryError(f"{relative}: larger than {MAX_FILE_BYTES} bytes")
        if budget["total"] + size > MAX_TOTAL_BYTES:
            raise DeliveryError(f"delivery larger than {MAX_TOTAL_BYTES} bytes")
        try:
            contents = read_regular(root, relative, maximum=min(MAX_FILE_BYTES, MAX_TOTAL_BYTES - budget["total"]))
        except UnsafeArtifact as error:
            reject(relative, str(error))
            return
        budget["total"] += len(contents)
        files[relative] = contents

    for entry in listing(str(root)):
        info = entry.stat(follow_symlinks=False)
        if entry.name == "solver.py":
            if stat.S_ISREG(info.st_mode):
                read("solver.py", info.st_size)
            else:
                reject("solver.py", "not a regular file (symlink or special file)")
        elif entry.name == "cpp":
            if not stat.S_ISDIR(info.st_mode):
                reject("cpp", "cpp must be a real directory")
                continue
            for source in listing(entry.path):
                relative = "cpp/" + source.name
                details = source.stat(follow_symlinks=False)
                if track != "cpp_allowed":
                    raise DeliveryError(f"C++ sources are not allowed in the {track} track", [{"path": relative, "reason": "track"}])
                if stat.S_ISLNK(details.st_mode):
                    reject(relative, "symlink")
                elif stat.S_ISDIR(details.st_mode):
                    reject(relative, "nested directory")
                elif not stat.S_ISREG(details.st_mode):
                    reject(relative, "special file")
                elif (reason := path_problem(relative, track)) is not None:
                    reject(relative, reason)
                else:
                    budget["sources"] += 1
                    if budget["sources"] > MAX_SOURCES:
                        raise DeliveryError(f"at most {MAX_SOURCES} C++ files")
                    read(relative, details.st_size)
        else:
            reject(entry.name, "symlink" if stat.S_ISLNK(info.st_mode) else (path_problem(entry.name, track) or "not allowed"))
    if rejected and strict:
        raise DeliveryError("delivery contains disallowed entries: " + ", ".join(item["path"] for item in rejected), rejected)
    return manifest_for(files, rejected, track=track)


def verify_manifest(manifest: dict, files: dict[str, bytes]) -> None:
    """Re-validate `files` with the canonical validator and check that they are exactly the manifest's files."""
    if manifest.get("version") != MANIFEST_VERSION or manifest.get("track") not in TRACKS:
        raise DeliveryError("unknown manifest version or track")
    check_files(files, manifest["track"])
    entries = [_entry(path, files[path]) for path in sorted(files)]
    if entries != manifest.get("files") or _digest(manifest["track"], entries) != manifest.get("sha256"):
        raise DeliveryError("delivered files differ from their manifest")


def materialize(manifest: dict, files: dict[str, bytes], destination: Path) -> None:
    """Write a verified delivery into a fresh directory (files read-only)."""
    verify_manifest(manifest, files)
    destination = Path(destination)
    destination.mkdir(mode=0o700)
    for path in sorted(files):
        target = destination / path
        target.parent.mkdir(mode=0o700, exist_ok=True)
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(files[path])


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


# ------------------------------------------------------------------------------------------------------- ELF reader

def _elf_sections(data, size):
    if size < 64 or data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not an ELF64 little-endian file")
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, _ = struct.unpack_from("<HHH", data, 0x3A)
    if shentsize != 64 or shoff + shnum * 64 > size:
        raise ValueError("malformed section headers")
    return [struct.unpack_from("<IIQQQQIIQQ", data, shoff + 64 * index) for index in range(shnum)]


def _cstring(data, start, limit) -> bytes:
    end = data.find(b"\x00", start, limit)
    if end < 0:
        raise ValueError("unterminated string")
    return data[start:end]


def elf_defines(path: Path, symbol: bytes = b"main") -> bool:
    """True if an ELF64 little-endian file defines a global or weak `symbol` in .symtab or .dynsym."""
    with open(path, "rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if size < 64:
            raise ValueError("not an ELF file")
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            sections = _elf_sections(data, size)
            for _, kind, _, _, offset, length, link, _, _, entsize in sections:
                if kind not in (2, 11) or entsize != 24 or link >= len(sections):  # SHT_SYMTAB, SHT_DYNSYM
                    continue
                strings_offset, strings_size = sections[link][4], sections[link][5]
                if offset + length > size or strings_offset + strings_size > size:
                    raise ValueError("malformed symbol table")
                for start in range(offset, offset + length - 23, 24):
                    name, info, _, shndx, _, _ = struct.unpack_from("<IBBHQQ", data, start)
                    if shndx == 0 or (info >> 4) not in (1, 2) or name >= strings_size:
                        continue
                    begin = strings_offset + name
                    if data[begin:begin + len(symbol) + 1] == symbol + b"\x00":
                        return True
    return False


def elf_needed(path: Path) -> list[str]:
    """DT_NEEDED entries of an ELF64 little-endian file."""
    with open(path, "rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if size < 64:
            raise ValueError("not an ELF file")
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            sections = _elf_sections(data, size)
            needed = []
            for _, kind, _, _, offset, length, link, _, _, _ in sections:
                if kind != 6 or link >= len(sections):  # SHT_DYNAMIC
                    continue
                strings_offset, strings_size = sections[link][4], sections[link][5]
                if offset + length > size or strings_offset + strings_size > size:
                    raise ValueError("malformed dynamic section")
                for start in range(offset, offset + length - 15, 16):
                    tag, value = struct.unpack_from("<qQ", data, start)
                    if tag == 0:
                        break
                    if tag == 1 and value < strings_size:
                        needed.append(_cstring(data, strings_offset + value, strings_offset + strings_size).decode(errors="replace"))
            return needed


def elf_imports(path: Path) -> list[str]:
    """Undefined (imported) symbols of an ELF64 little-endian file's .dynsym."""
    with open(path, "rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if size < 64:
            raise ValueError("not an ELF file")
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            sections = _elf_sections(data, size)
            names = set()
            for _, kind, _, _, offset, length, link, _, _, entsize in sections:
                if kind != 11 or entsize != 24 or link >= len(sections):  # SHT_DYNSYM
                    continue
                strings_offset, strings_size = sections[link][4], sections[link][5]
                if offset + length > size or strings_offset + strings_size > size:
                    raise ValueError("malformed symbol table")
                for start in range(offset, offset + length - 23, 24):
                    name, _, _, shndx, _, _ = struct.unpack_from("<IBBHQQ", data, start)
                    if shndx == 0 and 0 < name < strings_size:
                        names.add(_cstring(data, strings_offset + name, strings_offset + strings_size).decode(errors="replace"))
            return sorted(names)


# --------------------------------------------------------------------------------------------------------- the build

def slurm_noise(text: str) -> str:
    """Drop the Slurm step's own output (the site's prolog banner, srun status lines) from compiler feedback."""
    text = re.sub(r"-{20,}\nBegin Slurm Prolog.*?\n-{20,}\n?", "", text, flags=re.S)
    return "".join(line for line in text.splitlines(keepends=True) if not line.startswith("srun: "))


def build_core(executor, attempt, limits: BuildLimits) -> dict:
    """Run the fixed build commands for `attempt` (an `xl_executor.Attempt`); return the build record (audits informational)."""
    from src.experiments.xl_executor import WORK, sha256_file
    track = attempt.manifest["track"]
    sources = sorted(entry["path"] for entry in attempt.manifest["files"] if entry["path"].startswith("cpp/") and entry["path"].endswith(".cpp"))
    compiler = None if executor.gcc is None else {key: executor.gcc[key] for key in ("gxx", "version", "gxx_sha256", "cc1plus_sha256")}
    audit = (getattr(attempt, "audit", None) or {}).get("cpp")
    record = {"status": "skipped", "detail": f"no cpp/*.cpp ({track})", "track": track, "sources": sources, "targets": {}, "steps": [],
              "seconds": 0.0, "budget_seconds": limits.budget_seconds, "compiler": compiler, "feedback": "", "limits": dict(limits.__dict__),
              "source_audit": audit, "native": {}}
    if track != "cpp_allowed" or not sources:
        if track != "cpp_allowed" and any(entry["path"].startswith("cpp/") for entry in attempt.manifest["files"]):
            record.update(status="build_error", detail=f"C++ sources are not allowed in the {track} track")  # unreachable: validator
        return record
    if executor.gcc is None:
        record.update(status="infrastructure", detail="C++ sources delivered but no pinned compiler is configured")
        return record
    gxx = executor.gcc["gxx"]
    started = time.monotonic()
    plan = [("libcore.so", [gxx, "-std=c++17", "-O2", "-fPIC", "-shared", *sources, "-o", "build/libcore.so"]),
            ("core", [gxx, "-std=c++17", "-O2", *sources, "-o", "build/core"])]
    feedback = []
    for target, argv in plan:
        if target == "core":
            try:
                if not elf_defines(attempt.build / "libcore.so"):
                    break
            except (OSError, ValueError) as error:
                record.update(status="build_error", detail=f"unreadable build/libcore.so: {error}")
                break
        remaining = limits.budget_seconds - (time.monotonic() - started)
        if remaining <= 0:
            record.update(status="build_error", detail=f"build time limit ({limits.budget_seconds:g} s, cumulative) exceeded")
            break
        result = executor.compile(attempt, argv, timeout=remaining, limits=limits.rlimits(remaining), tail_bytes=limits.tail_bytes,
                                  output_cap=limits.output_cap_bytes, step=target.replace(".", "_"), scratch_quota=limits.scratch_quota_bytes,
                                  memory_step=limits.memory_step, memory_bytes=limits.memory_bytes)
        payload = result.payload_exit() or {}
        stdout, stderr = slurm_noise(result.stdout_tail), slurm_noise(result.stderr_tail)
        step = {"target": target, "argv": [part.replace(gxx, "g++") for part in argv], "exit_code": payload.get("exit_code"), "signal": payload.get("signal"),
                "seconds": result.elapsed, "output_bytes": result.stdout_bytes + result.stderr_bytes, "kill_reason": result.kill_reason,
                "oom_kills": result.oom_kills, "maxrss_kib": payload.get("maxrss_kib"), "cleanup_verified": result.cleanup_verified,
                "memory_cgroup": result.step_cgroup, "stdout": stdout, "stderr": stderr}
        record["steps"].append(step)
        feedback.append((stdout + stderr).strip())
        detail = None
        if result.infrastructure:
            record.update(status="infrastructure", detail=result.infrastructure)
            break
        if result.deadline_reached:
            detail = f"build time limit ({limits.budget_seconds:g} s, cumulative) exceeded"
        elif result.kill_reason == "output cap":
            detail = f"compiler output exceeded {limits.output_cap_bytes} bytes"
        elif result.kill_reason == "scratch quota":
            detail = f"build scratch ({limits.scratch_quota_bytes} bytes) exceeded"
        elif result.oom_kills:
            detail = f"compiler memory limit ({limits.memory_bytes} bytes for the compiler tree and scratch; kernel OOM kill)"
        elif payload.get("exit_code") != 0 and any(marker in result.stderr_tail for marker in
                                                   ("virtual memory exhausted", "out of memory", "Cannot allocate memory", "std::bad_alloc")):
            detail = f"compiler memory limit ({limits.address_space_bytes} bytes per process)"
        elif payload.get("signal") == 25:
            detail = f"output file size limit ({limits.file_size_bytes} bytes)"
        elif payload.get("signal"):
            detail = f"compiler killed by signal {payload.get('signal')}"
        elif payload.get("exit_code") != 0:
            detail = f"compilation of build/{target} failed (exit {payload.get('exit_code')})"
        elif not (attempt.build / target).is_file():
            detail = f"build/{target} was not produced"
        if detail is None:
            unexpected = sorted(set(os.listdir(attempt.build)) - {"libcore.so", "core"})
            if unexpected:  # hard: the host-backed build/ is bounded by RLIMIT_FSIZE x the declared targets
                detail = f"unexpected files in build/: {unexpected[:5]}"
        if detail is not None:
            record.update(status="build_error", detail=detail)
            break
        output = attempt.build / target
        record["targets"][target] = {"path": f"{WORK}/build/{target}", "size": output.stat().st_size, "sha256": sha256_file(output)}
        try:  # informational, for the method audit
            needed, imports = elf_needed(output), elf_imports(output)
            record["native"][target] = {"needed": needed, "needed_outside_defaults": sorted(set(needed) - ALLOWED_NEEDED),
                                        "imported_symbols": len(imports), "notable_imports": [name for name in imports if NOTABLE_IMPORTS.search(name)][:200]}
        except (OSError, ValueError) as error:
            record["native"][target] = {"error": str(error)}
    record["seconds"] = time.monotonic() - started
    if record["status"] == "skipped":
        record.update(status="ok", detail="built " + ", ".join(record["targets"]))
    text = "\n".join(part for part in feedback if part)
    if record["status"] == "build_error":
        text += f"\n[framework] build_error: {record['detail']}"
    record["feedback"] = text.encode()[-limits.tail_bytes:].decode(errors="replace")
    return record
