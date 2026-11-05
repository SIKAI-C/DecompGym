"""Unit tests of the XL delivery validator, source audits, ELF readers, build post-checks, job-limit preflight, termination
classifier, seccomp program and validator CLI (pure Python; no sandbox, compiler or solver)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import struct
import sys

import pytest

from src.experiments.xl_audit import audit_cpp_sources, audit_python_source, run_bounded_audits
from src.experiments.xl_build import (MANIFEST_VERSION, BuildLimits, DeliveryError, build_core, collect_delivery, elf_defines, elf_imports,
                                      elf_needed, manifest_for, materialize, path_problem, slurm_noise, verify_manifest)
from src.experiments.xl_executor import (Attempt, InfrastructureError, SandboxResult, TailReader, check_job_limits, classify, parse_license,
                                         persist_record, seccomp_program)

CPP, PY = "cpp_allowed", "python_only"


def delivery(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "delivery"
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


# ------------------------------------------------------------------------------------------------- delivery

@pytest.mark.parametrize("path, ok", [("solver.py", True), ("cpp/core.cpp", True), ("cpp/core.hpp", True), ("cpp/a-b.c_d.cpp", True),
                                      ("cpp/x.h", False), ("cpp/sub/x.cpp", False), ("cpp/../x.cpp", False), ("/solver.py", False),
                                      ("../solver.py", False), ("notes.md", False), ("cpp/.hidden.cpp", False), ("cpp/x.cpp\x00", False),
                                      ("solver.py/", False), ("cpp\\x.cpp", False), ("", False)])
def test_path_rules(path, ok):
    assert (path_problem(path, CPP) is None) is ok


def test_python_only_track_rejects_cpp_everywhere(tmp_path):
    assert path_problem("cpp/a.cpp", PY) is not None and path_problem("solver.py", PY) is None
    with pytest.raises(DeliveryError):
        manifest_for({"solver.py": b"", "cpp/a.cpp": b""}, track=PY)
    root = delivery(tmp_path, {"solver.py": "print(1)\n", "cpp/a.cpp": "int a;\n"})
    with pytest.raises(DeliveryError):
        collect_delivery(root, track=PY, strict=False)  # track violations raise even in lenient mode
    manifest, files = collect_delivery(root, track=CPP)
    with pytest.raises(DeliveryError):
        verify_manifest({**manifest, "track": PY, "sha256": manifest["sha256"]}, files)


def test_manifest_hashes_every_file_and_records_the_track(tmp_path):
    root = delivery(tmp_path, {"solver.py": "print(1)\n", "cpp/b.cpp": "int b;\n", "cpp/a.hpp": "#pragma once\n"})
    manifest, files = collect_delivery(root, track=CPP)
    assert manifest["version"] == MANIFEST_VERSION and manifest["track"] == CPP
    assert [entry["path"] for entry in manifest["files"]] == ["cpp/a.hpp", "cpp/b.cpp", "solver.py"]
    verify_manifest(manifest, files)
    with pytest.raises(DeliveryError):
        verify_manifest(manifest, dict(files, **{"cpp/b.cpp": b"int c;\n"}))
    with pytest.raises(DeliveryError):
        verify_manifest(manifest, {**files, "cpp/extra.cpp": b""})


def forged(files: dict[str, bytes], track: str = CPP) -> dict:
    """A correctly hashed manifest that bypasses manifest_for (review 44 #3)."""
    entries = [{"path": path, "size": len(files[path]), "sha256": hashlib.sha256(files[path]).hexdigest()} for path in sorted(files)]
    digest = hashlib.sha256(json.dumps({"track": track, "files": entries}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"version": MANIFEST_VERSION, "track": track, "files": entries, "rejected": [], "sha256": digest}


@pytest.mark.parametrize("files", [
    {"cpp/a.cpp": b"int a;"},                                                  # no solver.py
    {"solver.py": b"x" * (2 * 1024 * 1024 + 1)},                               # one file over 2 MiB
    {"solver.py": b"", **{f"cpp/f{i}.cpp": b"" for i in range(65)}},           # 65 C++ files
    {"solver.py": b"", **{f"cpp/f{i}.cpp": b"/" * (2 * 1024 * 1024 - 1) for i in range(5)}},  # total over 8 MiB
    {"solver.py": b"", "cpp/../x.cpp": b""},                                   # traversal
])
def test_verify_and_materialize_reject_correctly_hashed_bypasses(tmp_path, files):
    manifest = forged(files)
    with pytest.raises(DeliveryError):
        verify_manifest(manifest, files)
    with pytest.raises(DeliveryError):
        materialize(manifest, files, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_collection_enforces_limits_before_reading(tmp_path):
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    with open(sparse / "solver.py", "wb") as stream:
        stream.truncate(3 * (1 << 30))  # a 3 GiB sparse file: rejected from its size, never read
    with pytest.raises(DeliveryError, match="larger than"):
        collect_delivery(sparse, track=PY)
    many = delivery(tmp_path / "many", {"solver.py": "print(1)\n", **{f"cpp/f{i}.cpp": "" for i in range(300)}})
    with pytest.raises(DeliveryError, match="entries"):
        collect_delivery(many, track=CPP)
    big = delivery(tmp_path / "big", {"solver.py": "print(1)\n", **{f"cpp/f{i}.cpp": "/" * (2 * 1024 * 1024 - 10) for i in range(5)}})
    with pytest.raises(DeliveryError, match="larger than"):
        collect_delivery(big, track=CPP)
    count = delivery(tmp_path / "count", {"solver.py": "print(1)\n", **{f"cpp/f{i}.cpp": "" for i in range(65)}})
    with pytest.raises(DeliveryError, match="at most 64"):
        collect_delivery(count, track=CPP)


def test_strict_and_lenient_rejections(tmp_path):
    root = delivery(tmp_path, {"solver.py": "print(1)\n", "cpp/a.cpp": "int a;\n", "notes.md": "x", "cpp/sub/x.cpp": "", "cpp/readme.txt": ""})
    os.symlink("/etc/passwd", root / "cpp/link.cpp")
    os.symlink(root / "solver.py", root / "other.py")
    with pytest.raises(DeliveryError) as error:
        collect_delivery(root, track=CPP)
    assert {item["path"] for item in error.value.rejected} == {"notes.md", "cpp/sub", "cpp/readme.txt", "cpp/link.cpp", "other.py"}
    manifest, files = collect_delivery(root, track=CPP, strict=False)
    assert sorted(files) == ["cpp/a.cpp", "solver.py"] and len(manifest["rejected"]) == 5


def test_symlinked_solver_hardlinks_and_fifos_are_rejected(tmp_path):
    root = tmp_path / "d"
    root.mkdir()
    (tmp_path / "real.py").write_text("print(1)\n")
    os.symlink(tmp_path / "real.py", root / "solver.py")
    with pytest.raises(DeliveryError):
        collect_delivery(root, track=PY)
    hard = delivery(tmp_path / "x", {"solver.py": "print(1)\n", "cpp/a.cpp": "int a;\n"})
    os.link(hard / "cpp/a.cpp", hard / "cpp/b.cpp")
    with pytest.raises(DeliveryError):
        collect_delivery(hard, track=CPP)
    fifo = delivery(tmp_path / "y", {"solver.py": "print(1)\n"})
    (fifo / "cpp").mkdir()
    os.mkfifo(fifo / "cpp/f.cpp")
    with pytest.raises(DeliveryError):
        collect_delivery(fifo, track=CPP)


def test_materialize_is_read_only_and_fresh(tmp_path):
    manifest, files = manifest_for({"solver.py": b"print(1)\n", "cpp/a.cpp": b"int a;\n"}, track=CPP)
    materialize(manifest, files, tmp_path / "src")
    assert oct((tmp_path / "src/solver.py").stat().st_mode & 0o777) == "0o444"
    with pytest.raises(FileExistsError):
        materialize(manifest, files, tmp_path / "src")


# ------------------------------------------------------------------------------------------------- source audits

@pytest.mark.parametrize("source, needle", [
    ("#include <thread>\n", "threads"), ("#include <mutex>\n", "threads"), ("#include <future>\n", "threads"), ("#include <pthread.h>\n", "threads"),
    ("#include <execution>\n", "threads"), ("#include <omp.h>\n", "OpenMP"), ("#pragma omp parallel for\n", "OpenMP"), ('_Pragma("omp parallel")\n', "OpenMP"),
    ("#include <gurobi_c.h>\n", "Gurobi C API"), ("#include <gurobi_c++.h>\n", "Gurobi C++ API"), ("#include <unistd.h>\n", "standard library"),
    ("#include <dlfcn.h>\n", "dynamic loading"), ("#define H <vector>\n#include H\n", "computed"), ('#include "/etc/passwd"\n', "delivered"),
    ("#include_next <vector>\n", "include_next"), ("#inc\\\nlude <thread>\n", "threads"), ("  #  include   <thread>\n", "threads"),
    ("%:include <thread>\n", "threads"), ("\ufeff#include <thread>\n", "threads"),       # review 48: digraph and BOM
])
def test_cpp_audit_records_findings(source, needle):
    report = audit_cpp_sources({"cpp/x.cpp": source, "cpp/core.hpp": ""})
    assert not report["clean"] and any(needle in finding for finding in report["findings"]), report


def test_cpp_audit_is_clean_for_standard_code_and_ignores_comments_and_strings():
    source = ('// #include <thread>\n/* #pragma omp parallel\n #include <pthread.h> */\n#include <vector>\n#include <cmath>\n#include <math.h>\n'
              '#include "core.hpp"\nconst char* s = "#include <thread>";\nconst char* r = R"x(#pragma omp)x";\nint main() { return 0; }\n')
    report = audit_cpp_sources({"cpp/x.cpp": source, "cpp/core.hpp": "#pragma once\n"})
    assert report["clean"], report
    assert report["files"]["cpp/x.cpp"]["includes"] == ["<vector>", "<cmath>", "<math.h>", '"core.hpp"']


def test_python_audit_flags_scipy_optimization_ctypes_and_dynamic_imports():
    report = audit_python_source("import numpy as np\nfrom scipy.optimize import milp, linprog\nimport ctypes\nimport ortools\n"
                                 "m = __import__('scipy.optimize')\nx = scipy.optimize.linear_sum_assignment\nmethod = 'highs'\n")
    flags = " ".join(report["flags"])
    assert "milp" in flags and "linprog" in flags and "ctypes" in flags and "dynamic __import__" in flags and "highs" in flags
    assert report["non_allowlisted"] == ["ortools"] and "numpy" in report["imports"]
    assert audit_python_source("def f(:\n")["error"].startswith("SyntaxError")


DEEP = "x=" + "1+" * 10000 + "1"  # review 48 repro: RecursionError in ast.parse


def test_python_audit_never_raises_on_parser_failures():
    assert "Recursion" in audit_python_source(DEEP)["error"]
    bounded = run_bounded_audits({"solver.py": DEEP.encode()})
    assert bounded["python"]["error"] and bounded["bounded"]["memory_bytes"] == 1 << 30  # recorded on every platform
    timed_out = run_bounded_audits({"solver.py": b"print(1)\n"}, timeout=0.0001)
    assert "timed out" in timed_out["python"]["error"]
    starved = run_bounded_audits({"solver.py": b"import os\n" * 1000}, memory_bytes=8 << 20)  # too little memory to even start
    assert starved["python"]["error"]


@pytest.mark.skipif(sys.platform != "linux", reason="the audit child's RLIMIT_AS is enforceable on Linux only")
def test_bounded_audit_findings_on_linux():
    assert "Recursion" in run_bounded_audits({"solver.py": DEEP.encode()})["python"]["error"]
    report = run_bounded_audits({"solver.py": b"from scipy.optimize import milp\n", "cpp/a.cpp": b"#include <thread>\n"})
    assert any("milp" in flag for flag in report["python"]["flags"]) and any("threads" in finding for finding in report["cpp"]["findings"])


@pytest.mark.skipif(sys.platform == "linux", reason="covers platforms that reject the audit's limits (e.g. macOS)")
def test_bounded_audit_records_unsupported_limits():
    report = run_bounded_audits({"solver.py": b"import os\n", "cpp/a.cpp": b"#include <thread>\n"})
    assert "limits unavailable" in report["python"]["error"] and report["cpp"]["error"] and report["cpp"]["clean"] is False


# ------------------------------------------------------------------------------------------------- ELF readers

def tiny_elf(symbols: list[tuple[bytes, int, int]], needed: list[bytes] = (), dynamic_symbols: bool = False) -> bytes:
    """ELF64 LE with sections [null, .strtab, .symtab, .dynamic]; symbols are (name, st_info, st_shndx)."""
    strtab = b"\x00"
    offsets = []
    for name in [name for name, _, _ in symbols] + list(needed):
        offsets.append(len(strtab))
        strtab += name + b"\x00"
    symtab = b"\x00" * 24 + b"".join(struct.pack("<IBBHQQ", offsets[i], info, 0, shndx, 0, 0) for i, (_, info, shndx) in enumerate(symbols))
    dynamic = b"".join(struct.pack("<qQ", 1, offsets[len(symbols) + i]) for i in range(len(needed))) + struct.pack("<qQ", 0, 0)
    strtab_offset = 64
    symtab_offset = strtab_offset + len(strtab)
    dynamic_offset = symtab_offset + len(symtab)
    shoff = dynamic_offset + len(dynamic)
    sections = [struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                struct.pack("<IIQQQQIIQQ", 0, 3, 0, 0, strtab_offset, len(strtab), 0, 0, 1, 0),
                struct.pack("<IIQQQQIIQQ", 0, 11 if dynamic_symbols else 2, 0, 0, symtab_offset, len(symtab), 1, 1, 8, 24),
                struct.pack("<IIQQQQIIQQ", 0, 6, 0, 0, dynamic_offset, len(dynamic), 1, 0, 8, 16)]
    header = b"\x7fELF" + bytes([2, 1, 1]) + b"\x00" * 9 + struct.pack("<HHIQQQIHHHHHH", 3, 62, 1, 0, 0, shoff, 0, 64, 0, 0, 64, len(sections), 0)
    return header + strtab + symtab + dynamic + b"".join(sections)


def test_elf_main_detection_and_needed(tmp_path):
    cases = {"global_main": ([(b"main", 0x12, 7)], True), "weak_main": ([(b"main", 0x22, 7)], True), "undefined_main": ([(b"main", 0x12, 0)], False),
             "local_main": ([(b"main", 0x02, 7)], False), "mainly": ([(b"mainly", 0x12, 7)], False), "none": ([], False)}
    for name, (symbols, expected) in cases.items():
        path = tmp_path / name
        path.write_bytes(tiny_elf(symbols))
        assert elf_defines(path) is expected, name
    path = tmp_path / "needs"
    path.write_bytes(tiny_elf([], [b"libstdc++.so.6", b"libpthread.so.0"]))
    assert elf_needed(path) == ["libstdc++.so.6", "libpthread.so.0"]
    (tmp_path / "junk").write_bytes(b"not an elf file at all" * 4)
    with pytest.raises(ValueError):
        elf_defines(tmp_path / "junk")


def test_elf_imports_lists_undefined_dynamic_symbols(tmp_path):
    path = tmp_path / "imports"
    path.write_bytes(tiny_elf([(b"pthread_create", 0x12, 0), (b"dg_sum", 0x12, 7), (b"dlopen", 0x12, 0)], dynamic_symbols=True))
    assert elf_imports(path) == ["dlopen", "pthread_create"]


def test_slurm_step_noise_is_not_compiler_feedback():
    text = ("---------------------------------------\nBegin Slurm Prolog: Nov-03-2026 21:40:41\nJob ID:    1\n---------------------------------------\n"
            "cpp/a.cpp:1: error: x\nsrun: error: node: task 0: Exited with exit code 1\n")
    assert slurm_noise(text) == "cpp/a.cpp:1: error: x\n"


class FakeExecutor:
    """Stands in for XLSandboxExecutor.compile: writes the target (and optionally an extra file) like a compiler would."""

    def __init__(self, extra: str | None = None, needed=()):
        self.gcc = {"gxx": "/x/g++", "version": "12.3.0", "gxx_sha256": "0", "cc1plus_sha256": "0"}
        self.extra, self.needed = extra, list(needed)

    def compile(self, attempt, argv, **kwargs):
        target = attempt.build / argv[-1].split("/")[-1]
        target.write_bytes(tiny_elf([], [name.encode() for name in self.needed]))
        if self.extra:
            (attempt.build / self.extra).write_bytes(b"x")
        result = SandboxResult(events=[{"event": "payload_started", "t": 0.0}, {"event": "payload_exit", "exit_code": 0, "signal": None}])
        result.cleanup_verified = True
        return result


def fake_attempt(tmp_path, files, track=CPP):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest, contents = manifest_for(files, track=track)
    materialize(manifest, contents, tmp_path / "src")
    (tmp_path / "build").mkdir()
    return Attempt(root=tmp_path, sources=tmp_path / "src", build=tmp_path / "build", manifest=manifest)


def test_build_postchecks_reject_extra_outputs_and_record_native_facts(tmp_path):
    files = {"solver.py": b"print(1)", "cpp/a.cpp": b"#include <vector>\nint a;\n"}
    ok = build_core(FakeExecutor(needed=["libstdc++.so.6", "libc.so.6"]), fake_attempt(tmp_path / "ok", files), BuildLimits())
    assert ok["status"] == "ok" and list(ok["targets"]) == ["libcore.so"] and ok["native"]["libcore.so"]["needed_outside_defaults"] == []
    extra = build_core(FakeExecutor(extra="leak.bin"), fake_attempt(tmp_path / "extra", files), BuildLimits())
    assert extra["status"] == "build_error" and "unexpected files" in extra["detail"]  # hard: host-backed build/ bound
    foreign = build_core(FakeExecutor(needed=["libpthread.so.0"]), fake_attempt(tmp_path / "foreign", files), BuildLimits())
    assert foreign["status"] == "ok" and foreign["native"]["libcore.so"]["needed_outside_defaults"] == ["libpthread.so.0"]  # informational
    threads = fake_attempt(tmp_path / "threads", {"solver.py": b"", "cpp/a.cpp": b"#include <thread>\n"})
    threads.audit = {"cpp": audit_cpp_sources({"cpp/a.cpp": "#include <thread>\n"})}  # the bounded run is tested separately
    built = build_core(FakeExecutor(), threads, BuildLimits())
    assert built["status"] == "ok" and built["steps"] and any("threads" in finding for finding in built["source_audit"]["findings"])


# ------------------------------------------------------------------------------------------------- preflight

def fake_cgroup(tmp_path, *, cpus="5", leaf_memory="max", job_memory=str(16 << 30), swap="0", oom_group="0", swap_total_kb=8388604,
                omit_leaf_oom_group=False):
    root = tmp_path / "cg"
    job = root / "system.slice/job"
    user = job / "step_batch/user"
    leaf = user / "task_0"
    leaf.mkdir(parents=True)
    for node, values in ((leaf, {"memory.max": leaf_memory, "memory.swap.max": "max", "memory.oom.group": "0", "cpuset.cpus.effective": cpus}),
                         (user, {"memory.max": job_memory, "memory.swap.max": swap, "memory.oom.group": oom_group, "cpuset.cpus.effective": cpus}),
                         (job / "step_batch", {"memory.max": "max", "memory.swap.max": "max", "memory.oom.group": "0"}),
                         (job, {"memory.max": job_memory, "memory.swap.max": swap, "memory.oom.group": "0"})):
        for name, value in values.items():
            (node / name).write_text(value + "\n")
    if omit_leaf_oom_group:
        (leaf / "memory.oom.group").unlink()
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemTotal: 1 kB\nSwapTotal: {swap_total_kb} kB\n")
    return leaf, root, str(meminfo)


def test_check_job_limits_accepts_the_phoenix_layout(tmp_path):
    leaf, root, meminfo = fake_cgroup(tmp_path)
    facts = check_job_limits(leaf, cpus=1, memory_cap=16 << 30, swap_cap=0, affinity={5}, meminfo=meminfo, root=root)
    assert facts["memory_max"] == 16 << 30 and facts["swap_bound"] == 0 and facts["cpuset"] == [5]


@pytest.mark.parametrize("kwargs, affinity", [({"cpus": "4-5"}, {4}), ({"cpus": "5"}, {4, 5}), ({"job_memory": "max"}, {5}),
                                              ({"job_memory": str(32 << 30)}, {5}), ({"swap": "max"}, {5}), ({"oom_group": "1"}, {5})])
def test_check_job_limits_fails_closed(tmp_path, kwargs, affinity):
    leaf, root, meminfo = fake_cgroup(tmp_path, **kwargs)
    with pytest.raises(InfrastructureError):
        check_job_limits(leaf, cpus=1, memory_cap=16 << 30, swap_cap=0, affinity=affinity, meminfo=meminfo, root=root)


def test_missing_or_unreadable_oom_group_fails_closed(tmp_path):
    leaf, root, meminfo = fake_cgroup(tmp_path / "missing", omit_leaf_oom_group=True)
    with pytest.raises(InfrastructureError, match="missing or unreadable"):
        check_job_limits(leaf, cpus=1, memory_cap=16 << 30, swap_cap=0, affinity={5}, meminfo=meminfo, root=root)
    leaf, root, meminfo = fake_cgroup(tmp_path / "empty", oom_group="")
    with pytest.raises(InfrastructureError):
        check_job_limits(leaf, cpus=1, memory_cap=16 << 30, swap_cap=0, affinity={5}, meminfo=meminfo, root=root)
    if os.geteuid() != 0:
        leaf, root, meminfo = fake_cgroup(tmp_path / "unreadable")
        (leaf.parent / "memory.oom.group").chmod(0)
        with pytest.raises(InfrastructureError, match="missing or unreadable"):
            check_job_limits(leaf, cpus=1, memory_cap=16 << 30, swap_cap=0, affinity={5}, meminfo=meminfo, root=root)


def load_init():
    path = Path(__file__).resolve().parents[1] / "src/experiments/xl_sandbox_init.py"
    spec = importlib.util.spec_from_file_location("xl_sandbox_init_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("behaviour", ["set_fails", "get_reports_dumpable", "raises"])
def test_non_dumpable_fails_closed(monkeypatch, behaviour):
    import ctypes
    init = load_init()

    class Libc:
        def prctl(self, option, *args):
            if behaviour == "raises":
                raise OSError("prctl unavailable")
            if option == 4:
                return -1 if behaviour == "set_fails" else 0
            return 1  # PR_GET_DUMPABLE still 1

    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: Libc())
    assert init.non_dumpable() is False


def test_init_refuses_to_start_candidate_code_when_not_dumpable_off(monkeypatch, tmp_path):
    init = load_init()
    saved = signal.getsignal(signal.SIGINT)
    current = Path("/proc/self/oom_score_adj").read_text().strip() if Path("/proc/self/oom_score_adj").exists() else "0"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"argv": ["/bin/true"], "cwd": "/", "env": {}, "limits": {}, "relay": {"socket": "x", "port": 1},
                                  "oom_score_adj": int(current)}))
    monkeypatch.setattr(init, "non_dumpable", lambda: False)
    monkeypatch.setattr(init.os, "fork", lambda: (_ for _ in ()).throw(AssertionError("candidate code was started")))
    read_end, write_end = os.pipe()
    try:
        assert init.main(str(config), str(write_end)) == 73
    finally:
        signal.signal(signal.SIGINT, saved)
    os.close(write_end)
    events = [json.loads(line) for line in os.read(read_end, 65536).decode().splitlines()]
    os.close(read_end)
    assert [event["event"] for event in events] == ["error"] and events[0]["stage"] == "dumpable"


def test_unbounded_swap_is_fine_only_without_swap_space(tmp_path):
    leaf, root, meminfo = fake_cgroup(tmp_path, swap="max", swap_total_kb=0)
    assert check_job_limits(leaf, cpus=1, memory_cap=16 << 30, swap_cap=0, affinity={5}, meminfo=meminfo, root=root)["swap_bound"] == 0


def test_license_parser_keeps_only_server_and_port(tmp_path):
    path = tmp_path / "gurobi.lic"
    path.write_text("LICENSEID=1\nTYPE=TOKEN\nTOKENSERVER=license-coda.pace.gatech.edu\nKEY=SECRET\n# PORT=1\n")
    assert parse_license(path) == (["license-coda.pace.gatech.edu"], 41954)
    path.write_text("TOKENSERVER=a, b\nPORT=5000\n")
    assert parse_license(path) == (["a", "b"], 5000)


# ------------------------------------------------------------------------------------------------- classification

STARTED = {"event": "payload_started", "t": 1.0}


def exit_event(code=None, sig=None):
    return {"event": "payload_exit", "exit_code": code, "signal": sig}


@pytest.mark.parametrize("result, expected", [
    (SandboxResult(events=[STARTED, exit_event(0)], payload_started=1.0), "finished"),
    (SandboxResult(events=[STARTED, exit_event(1)], payload_started=1.0), "crash"),
    (SandboxResult(events=[STARTED, exit_event(1)], payload_started=1.0, stderr_tail="Traceback\nMemoryError\n"), "crash"),   # forged: no kernel event
    (SandboxResult(events=[STARTED, exit_event(None, signal.SIGKILL)], payload_started=1.0, oom_kills=1), "oom"),
    (SandboxResult(events=[STARTED, exit_event(0)], payload_started=1.0, oom_kills=1), "finished"),                          # a descendant was OOM-killed
    (SandboxResult(events=[STARTED, exit_event(None, signal.SIGXFSZ)], payload_started=1.0), "crash"),
    (SandboxResult(events=[STARTED, exit_event(0)], payload_started=1.0, deadline_reached=True), "deadline"),
    (SandboxResult(events=[STARTED], payload_started=1.0, kill_reason="scratch quota", kill_sent=1.0), "crash"),
    (SandboxResult(events=[STARTED], payload_started=1.0), "crash"),             # init gone after the payload started: no framework evidence
    (SandboxResult(events=[STARTED], payload_started=1.0, oom_kills=1), "oom"),
    (SandboxResult(events=[]), "infrastructure"),                                # the payload never started
    (SandboxResult(events=[STARTED, exit_event(0)], payload_started=1.0, infrastructure="bwrap failed"), "infrastructure"),
])
def test_classify(result, expected):
    assert classify(result)[0] == expected


def test_late_exit_is_a_deadline():
    late = SandboxResult(events=[STARTED, exit_event(0)], payload_started=1.0, deadline_reached=True, late_exit=True)
    assert classify(late) == ("deadline", "ended after the wall-clock limit (before the executor's signal)")


def test_tail_reader_cuts_feedback_at_the_absolute_deadline():
    import time
    read_end, write_end = os.pipe()
    reader = TailReader(read_end, 100, cutoff=time.monotonic() + 0.3)
    reader.start()
    os.write(write_end, b"before ")
    time.sleep(0.6)
    os.write(write_end, b"after")
    os.close(write_end)
    reader.join(5)
    assert reader.text() == "before " and reader.after_freeze() == 5 and reader.total == 12


def test_persist_record_is_atomic_and_durable(tmp_path):
    record = {"termination": "oom", "detail": "kernel OOM kill in the job cgroup (payload signal 9)", "oom_kills": 1}
    path = persist_record(tmp_path / "attempt.json", record)
    assert json.loads(path.read_text()) == record and not (tmp_path / "attempt.json.tmp").exists()


def test_forged_memory_error_is_a_noted_crash():
    termination, detail = classify(SandboxResult(events=[STARTED, exit_event(1)], payload_started=1.0, stderr_tail="MemoryError"))
    assert termination == "crash" and "no kernel OOM" in detail


# ------------------------------------------------------------------------------------------------- seccomp

def run_bpf(program: bytes, nr: int, args=(0,) * 6, arch=0xC000003E) -> int:
    """A tiny classic-BPF interpreter for the instructions the filter uses."""
    data = struct.pack("<iIQ6Q", nr, arch, 0, *args)
    code = [struct.unpack_from("<HBBI", program, i) for i in range(0, len(program), 8)]
    pc, acc = 0, 0
    while True:
        op, jt, jf, k = code[pc]
        if op == 0x20:
            acc = struct.unpack_from("<I", data, k)[0]
            pc += 1
        elif op == 0x15:
            pc += 1 + (jt if acc == k else jf)
        elif op == 0x35:
            pc += 1 + (jt if acc >= k else jf)
        elif op == 0x45:
            pc += 1 + (jt if acc & k else jf)
        elif op == 0x06:
            return k
        else:
            raise AssertionError(op)


def test_seccomp_program_decisions():
    program = seccomp_program()
    allow, eperm, enosys = 0x7FFF0000, 0x50001, 0x50026
    assert run_bpf(program, 39) == allow                              # getpid
    assert run_bpf(program, 302, (0, 7, 1, 0, 0, 0)) == allow         # prlimit64 on itself
    assert run_bpf(program, 302, (1, 0, 1, 0, 0, 0)) == eperm         # prlimit64 on PID 1
    assert run_bpf(program, 56, (17, 0, 0, 0, 0, 0)) == allow         # clone (fork)
    assert run_bpf(program, 56, (0x10000000 | 17,) + (0,) * 5) == eperm  # clone(CLONE_NEWUSER)
    assert run_bpf(program, 435) == enosys                            # clone3 -> glibc falls back to clone
    for number in (250, 248, 249, 101, 272, 308, 321, 298, 425, 440):
        assert run_bpf(program, number) == eperm
    assert run_bpf(program, 0x40000000 | 39) == eperm                 # x32 ABI
    assert run_bpf(program, 39, arch=0x40000003) == eperm             # foreign architecture


# ------------------------------------------------------------------------------------------------- validator CLI

def load_validator():
    path = Path(__file__).resolve().parents[1] / "scripts/phoenix/xl_executor_validate.py"
    spec = importlib.util.spec_from_file_location("xl_executor_validate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validator_rejects_unknown_checks_and_exits_nonzero_on_failure(tmp_path, monkeypatch):
    module = load_validator()
    with pytest.raises(SystemExit) as error:
        module.main(["--out", str(tmp_path / "r.json"), "--only", "no_such_check"])
    assert error.value.code == 2

    class Stub:
        def __init__(self, out):
            self.out, self.report = out, {"checks": []}

        def check(self, name, function):
            self.report["checks"].append({"name": name, "ok": name != "fork_bomb"})

        def __getattr__(self, name):
            return lambda: (True, {})

    monkeypatch.setattr(module, "Validator", Stub)
    assert module.main(["--out", str(tmp_path / "r.json"), "--only", "isolation"]) == 0
    assert module.main(["--out", str(tmp_path / "r.json"), "--only", "isolation", "--only", "fork_bomb"]) == 1
    assert len(json.loads(module.EXPECTED_PDPTW_FILE.read_text())) == 40


def test_round_half_even_convention():
    np = pytest.importorskip("numpy")
    values = [k + 0.5 for k in range(-1000, 1000)]
    assert [round(v) for v in values] == [int(v) for v in np.rint(values)]
    assert round(62.5) == 62 and round(187.5) == 188 and float(np.rint(1000 * 0.0625)) == 62.0
