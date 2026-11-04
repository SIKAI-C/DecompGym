"""Informational source audits of an XL delivery, run in a bounded child process (Codex reviews 44 #6, 48 #1-#2).

Standard library only, so the same file runs as the child: ``python -I -B xl_audit.py MEMORY_BYTES CPU_SECONDS`` reads
``{path: text}`` as JSON on stdin and prints ``{"python": ..., "cpp": ...}``. `run_bounded_audits` starts it with an address
space and CPU limit (set by the child before it reads any input) and a wall-clock timeout; any parser failure (SyntaxError,
RecursionError, MemoryError), timeout or crash becomes a recorded audit error and never an exception in the executor.

Both audits are informational for the method audit (policy of 2026-11-04): nothing here rejects an attempt.
- Python: imports, non-allowlisted packages, and flags for SciPy optimization/HiGHS names, ctypes and dynamic code.
- C++: the direct ``#include`` lines and pragmas (comments removed, line splices joined, a UTF-8 BOM ignored, the ``%:``
  digraph read as ``#``), with findings for headers outside the C++17/C standard library, thread and OpenMP use and the
  Gurobi C/C++ API. A header regex cannot establish semantics (declarations can be written without headers), so the build
  also records the symbols the built files import (`xl_build.elf_imports`).
"""
from __future__ import annotations

import ast
import json
import os
import re
import resource
import subprocess
import sys

CXX17_HEADERS = frozenset("""algorithm any array atomic bitset cassert ccomplex cctype cerrno cfenv cfloat charconv chrono cinttypes
ciso646 climits clocale cmath codecvt complex csetjmp csignal cstdalign cstdarg cstdbool cstddef cstdint cstdio cstdlib cstring
ctgmath ctime cuchar cwchar cwctype deque exception filesystem forward_list fstream functional initializer_list iomanip ios
iosfwd iostream istream iterator limits list locale map memory memory_resource new numeric optional ostream queue random ratio
regex scoped_allocator set sstream stack stdexcept streambuf string string_view strstream system_error tuple type_traits
typeindex typeinfo unordered_map unordered_set utility valarray variant vector thread mutex shared_mutex future condition_variable
execution""".split())
C_HEADERS = frozenset("""assert.h ctype.h errno.h fenv.h float.h inttypes.h iso646.h limits.h locale.h math.h setjmp.h signal.h
stdarg.h stdbool.h stddef.h stdint.h stdio.h stdlib.h string.h time.h uchar.h wchar.h wctype.h""".split())
NOTABLE_HEADERS = {**{name: "threads" for name in ("thread", "mutex", "shared_mutex", "future", "condition_variable", "execution",
                                                    "barrier", "latch", "semaphore", "stop_token", "pthread.h", "threads.h")},
                   "omp.h": "OpenMP", "gurobi_c.h": "Gurobi C API", "gurobi_c++.h": "Gurobi C++ API", "dlfcn.h": "dynamic loading"}
PYTHON_PACKAGES = frozenset({"gurobipy", "numpy", "scipy"})
PYTHON_FLAG_WORDS = re.compile(r"milp|linprog|highs|linear_sum_assignment|scipy\.optimize", re.IGNORECASE)


def strip_comments(text: str) -> str:
    """Drop a BOM, join line splices and replace comments by spaces (newlines kept); literals stay intact."""
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\\\n", "")
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end < 0 else end
            out.append(" ")
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            chunk = text[i:] if end < 0 else text[i:end + 2]
            out.append(" " + "\n" * chunk.count("\n"))
            i = n if end < 0 else end + 2
        elif (match := re.match(r'(?:u8|u|U|L)?R"([^ ()\\\t\n]{0,16})\(', text[i:i + 24])) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            close = ")" + match.group(1) + '"'
            end = text.find(close, i + match.end())
            end = n if end < 0 else end + len(close)
            out.append(text[i:end])
            i = end
        elif ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def audit_cpp_sources(files: dict[str, str]) -> dict:
    """Informational include/pragma audit of the delivered C++ files."""
    delivered = {path.split("/", 1)[1] for path in files if path.startswith("cpp/")}
    report: dict = {"files": {}, "findings": [], "notes": []}
    for path in sorted(files):
        if not path.startswith("cpp/"):
            continue
        text = strip_comments(files[path])
        includes = []
        for number, line in enumerate(text.split("\n"), start=1):
            directive = re.match(r"\s*(?:#|%:)\s*([A-Za-z_]+)(.*)", line)
            if directive:
                name, rest = directive.group(1), directive.group(2).strip()
                if name == "include":
                    angle, quoted = re.match(r"<([^<>]+)>\s*$", rest), re.match(r'"([^"]+)"\s*$', rest)
                    if angle:
                        header = angle.group(1).strip()
                        includes.append(f"<{header}>")
                        if header in NOTABLE_HEADERS:
                            report["findings"].append(f"{path}:{number}: <{header}> ({NOTABLE_HEADERS[header]})")
                        elif header not in CXX17_HEADERS and header not in C_HEADERS:
                            report["findings"].append(f"{path}:{number}: <{header}> is not a C++17 standard library header")
                    elif quoted:
                        header = quoted.group(1)
                        includes.append(f'"{header}"')
                        if header not in delivered:
                            report["findings"].append(f'{path}:{number}: "{header}" is not a delivered cpp/ file')
                    else:
                        report["findings"].append(f"{path}:{number}: computed #include")
                elif name in ("include_next", "import"):
                    report["findings"].append(f"{path}:{number}: #{name}")
                elif name == "pragma" and re.match(r"omp\b", rest):
                    report["findings"].append(f"{path}:{number}: OpenMP pragma")
            if re.search(r'_Pragma\s*\(\s*"\s*omp\b', line):
                report["findings"].append(f"{path}:{number}: OpenMP _Pragma")
            if re.search(r"\b(?:asm|__asm__|__asm)\b", line):
                report["notes"].append(f"{path}:{number}: inline assembly")
        report["files"][path] = {"includes": includes}
    report["clean"] = not report["findings"]
    return report


def audit_python_source(text: str) -> dict:
    """Imports and flagged usages in solver.py (informational)."""
    report: dict = {"imports": [], "flags": [], "non_allowlisted": [], "error": None}
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as error:
        report["error"] = f"{type(error).__name__}: {error}"[:500]
        return report
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    try:
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
            for name in names:
                report["imports"].append(name)
                top = name.split(".")[0]
                if top not in stdlib and top not in PYTHON_PACKAGES and name == names[0]:
                    report["non_allowlisted"].append(name)
                if PYTHON_FLAG_WORDS.search(name):
                    report["flags"].append(f"line {node.lineno}: import {name}")
                if top == "ctypes":
                    report["flags"].append(f"line {node.lineno}: ctypes (native code / shared-library calls)")
            if isinstance(node, ast.Attribute) and PYTHON_FLAG_WORDS.search(node.attr):
                report["flags"].append(f"line {node.lineno}: attribute {node.attr}")
            if isinstance(node, ast.Name) and PYTHON_FLAG_WORDS.search(node.id):
                report["flags"].append(f"line {node.lineno}: name {node.id}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and PYTHON_FLAG_WORDS.search(node.value):
                report["flags"].append(f"line {node.lineno}: string mentions {PYTHON_FLAG_WORDS.search(node.value).group(0)}")
            if isinstance(node, ast.Call):
                func = node.func
                label = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
                if label in ("__import__", "import_module", "exec", "eval", "compile"):
                    report["flags"].append(f"line {node.lineno}: dynamic {label}()")
    except (RecursionError, MemoryError) as error:
        report["error"] = f"{type(error).__name__} while walking the syntax tree"
    report["flags"] = sorted(set(report["flags"]))
    report["imports"] = sorted(set(report["imports"]))
    report["non_allowlisted"] = sorted(set(report["non_allowlisted"]))
    return report


def audit_all(files: dict[str, str]) -> dict:
    out = {"python": audit_python_source(files.get("solver.py", "")), "cpp": None}
    if any(path.startswith("cpp/") for path in files):
        out["cpp"] = audit_cpp_sources(files)
    return out


def run_bounded_audits(files: dict[str, bytes], *, timeout: float = 30.0, memory_bytes: int = 1 << 30, cpu_seconds: int = 20,
                       python: str = sys.executable) -> dict:
    """Both audits in a child process with RLIMIT_AS, RLIMIT_CPU and a wall-clock timeout; never raises for candidate input."""
    payload = json.dumps({path: contents.decode("utf-8", errors="replace") for path, contents in files.items()})
    failure = None
    try:
        result = subprocess.run([python, "-I", "-B", os.path.abspath(__file__), str(memory_bytes), str(cpu_seconds)], input=payload,
                                capture_output=True, text=True, timeout=timeout, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
        if result.returncode == 0:
            try:
                return {**json.loads(result.stdout), "bounded": {"timeout": timeout, "memory_bytes": memory_bytes, "cpu_seconds": cpu_seconds}}
            except json.JSONDecodeError:
                failure = "audit output unreadable"
        else:
            failure = f"audit process failed (exit {result.returncode}): {(result.stderr.strip() or result.stdout.strip())[-300:]}"
    except subprocess.TimeoutExpired:
        failure = f"audit timed out after {timeout:g} s"
    except OSError as error:
        failure = f"audit process could not start: {error}"
    error = {"error": failure}
    return {"python": {"imports": [], "flags": [], "non_allowlisted": [], **error},
            "cpp": {**error, "files": {}, "findings": [], "notes": [], "clean": False} if any(path.startswith("cpp/") for path in files) else None,
            "bounded": {"timeout": timeout, "memory_bytes": memory_bytes, "cpu_seconds": cpu_seconds}}


if __name__ == "__main__":
    memory, cpu = int(sys.argv[1]), int(sys.argv[2])
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    except (ValueError, OSError) as error:  # e.g. macOS rejects RLIMIT_AS: no unbounded audit, record the failure instead
        sys.stdout.write(f"audit limits unavailable on this platform: {error}")
        sys.exit(3)
    try:
        report = audit_all(json.loads(sys.stdin.read()))
    except BaseException as error:  # noqa: BLE001 - every failure becomes a recorded audit error
        report = {"python": {"imports": [], "flags": [], "non_allowlisted": [], "error": f"{type(error).__name__}: {error}"[:500]}, "cpp": None}
    sys.stdout.write(json.dumps(report))
