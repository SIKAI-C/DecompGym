"""Read-only macOS resource sampling and supervision of one owned process tree."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


@dataclass(frozen=True)
class ProcessSample:
    pid: int
    parent: int
    group: int
    cpu_percent: float
    rss_kib: int
    started: str
    name: str
    state: str = "S"


def parse_processes(text: str) -> dict[int, ProcessSample]:
    samples = {}
    for line in text.splitlines():
        values = line.split(None, 11)
        if len(values) != 12:
            continue
        try:
            item = ProcessSample(int(values[0]), int(values[1]), int(values[2]), float(values[3]), int(values[4]), " ".join(values[6:11]), values[11], values[5])
        except ValueError:
            continue
        samples[item.pid] = item
    return samples


def descendants(samples: dict[int, ProcessSample], root_pid: int) -> set[int]:
    found = {root_pid} if root_pid in samples else set()
    while True:
        children = {pid for pid, item in samples.items() if item.parent in found}
        if children <= found:
            return found
        found |= children


def thermal_warnings(text: str) -> list[str]:
    warnings = []
    for label, value in re.findall(r"(CPU_Speed_Limit|CPU_Scheduler_Limit|Thermal_Level|thermal warning level|performance warning level)\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE):
        number = int(value)
        if (label.lower().endswith("limit") and number < 100) or (not label.lower().endswith("limit") and number > 0):
            warnings.append(f"{label}={number}")
    return warnings


def memory_free_percent(text: str) -> int:
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", text)
    if match is None:
        raise RuntimeError("Cannot read macOS memory pressure")
    return int(match.group(1))


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=5)
    if result.returncode:
        raise RuntimeError(f"Resource query failed: {command[0]}: {result.stderr.strip()}")
    return result.stdout


def sample_host() -> tuple[dict[int, ProcessSample], str, int]:
    processes = read_processes()
    if not processes:
        raise RuntimeError("Cannot read processes; refusing unmonitored execution")
    thermal = command_output(["/usr/bin/pmset", "-g", "therm"])
    memory = memory_free_percent(command_output(["/usr/bin/memory_pressure", "-Q"]))
    return processes, thermal, memory


def read_processes() -> dict[int, ProcessSample]:
    processes = parse_processes(command_output(["/bin/ps", "-axo", "pid=,ppid=,pgid=,pcpu=,rss=,stat=,lstart=,comm="]))
    if not processes:
        raise CleanupError("Cannot verify owned process cleanup: process query returned no entries")
    return processes


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


class ResourceGuard:
    def __init__(self, *, max_cpu_percent: float = 150, max_rss_mib: int = 2048, min_free_percent: int = 10, sustained_samples: int = 3) -> None:
        self.max_cpu_percent = max_cpu_percent
        self.max_rss_mib = max_rss_mib
        self.min_free_percent = min_free_percent
        self.sustained_samples = sustained_samples
        self.high_cpu_samples = 0
        self.low_memory_samples = 0

    def check(self, *, cpu_percent: float, rss_mib: float, free_percent: int, thermal: str) -> str | None:
        thermal_issues = thermal_warnings(thermal)
        if thermal_issues:
            return "macOS thermal/performance warning: " + ", ".join(thermal_issues)
        if rss_mib > self.max_rss_mib:
            return f"Experiment resident memory {rss_mib:.1f} MiB exceeds {self.max_rss_mib} MiB"
        self.high_cpu_samples = self.high_cpu_samples + 1 if cpu_percent > self.max_cpu_percent else 0
        self.low_memory_samples = self.low_memory_samples + 1 if free_percent < self.min_free_percent else 0
        if self.high_cpu_samples >= self.sustained_samples:
            return f"Experiment CPU exceeded {self.max_cpu_percent:g}% for {self.sustained_samples} samples (100% is one CPU core)"
        if self.low_memory_samples >= self.sustained_samples:
            return f"System available memory below {self.min_free_percent}% for {self.sustained_samples} samples"
        return None


class CleanupError(RuntimeError):
    """The framework cannot prove that its owned computation stopped."""


def stop_owned_tree(process: subprocess.Popen, known: dict[int, ProcessSample] | None = None, *, grace: float = 0.5) -> None:
    known = dict(known or {})
    errors = []
    def survivors(initial: bool = False) -> dict[int, ProcessSample]:
        try:
            current = read_processes()
        except RuntimeError as error:
            raise CleanupError(str(error)) from error
        owned = {pid for pid, item in known.items() if pid in current and current[pid].started == item.started}
        if initial:
            # Popen created this new session. Its group may outlive an already-exited parent.
            owned |= {pid for pid, item in current.items() if item.group == process.pid}
        while True:
            children = {pid for pid, item in current.items() if item.parent in owned}
            if children <= owned:
                break
            owned |= children
        known.update({pid: current[pid] for pid in owned})
        return {pid: current[pid] for pid in owned if not current[pid].state.startswith("Z")}
    def send(live: dict[int, ProcessSample], signum: int) -> None:
        groups = {item.group for item in live.values()} - {os.getpgrp()}
        for group in groups:
            try:
                os.killpg(group, signum)
            except ProcessLookupError:
                pass
            except PermissionError as error:
                process.poll()
                if any(item.group == group for item in survivors().values()):
                    errors.append(f"Cannot signal live owned group {group}: {error}")
    live = survivors(initial=True)
    send(live, signal.SIGTERM)
    if live:
        time.sleep(grace)
    process.poll()
    send(survivors(), signal.SIGKILL)
    deadline = time.monotonic() + 3
    while True:
        process.poll()
        remaining = survivors()
        if not remaining:
            break
        if time.monotonic() >= deadline:
            raise CleanupError(f"Owned processes remain alive: {sorted(remaining)}; {errors}")
        time.sleep(0.05)
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired as error:
        raise CleanupError(f"Owned root process {process.pid} did not exit") from error


def supervise(command: list[str], monitor_root: Path, *, interval: float = 5, guard: ResourceGuard | None = None, container_sampler=None) -> int:
    monitor_root.mkdir(parents=True, exist_ok=True)
    guard = guard or ResourceGuard()
    _, thermal, free = sample_host()
    initial_issue = guard.check(cpu_percent=0, rss_mib=0, free_percent=free, thermal=thermal)
    if initial_issue:
        atomic_json(monitor_root / "status.json", {"state": "not_started", "reason": initial_issue})
        raise RuntimeError(initial_issue)
    (monitor_root / "protocol.json").write_text(json.dumps({"command": command, "sample_interval_seconds": interval, "max_cpu_percent": guard.max_cpu_percent, "max_rss_mib": guard.max_rss_mib, "min_system_free_percent": guard.min_free_percent, "sustained_samples": guard.sustained_samples, "cpu_units": "100 percent is one logical core", "temperature_celsius_available": False, "container_resources_included": container_sampler is not None, "monitor": "macOS process resources, optional Docker worker resources, and OS thermal/performance warnings"}, indent=2))
    stopped_reason = None
    cleanup_failed = False
    known = {}
    peaks = {"cpu_percent": 0.0, "rss_mib": 0.0}
    process = None
    try:
        with (monitor_root / "run.log").open("a") as log, (monitor_root / "resources.jsonl").open("a") as history:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=os.environ | {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
            print(f"Started supervised experiment PID {process.pid}; monitor: {monitor_root}", flush=True)
            while process.poll() is None:
                processes, thermal, free = sample_host()
                owned = descendants(processes, process.pid)
                owned |= {pid for pid, item in known.items() if pid in processes and processes[pid].started == item.started}
                selected = [processes[pid] for pid in owned]
                known.update({item.pid: item for item in selected})
                cpu = sum(item.cpu_percent for item in selected)
                rss = sum(item.rss_kib for item in selected) / 1024
                containers = container_sampler() if container_sampler else []
                cpu += sum(item["cpu_percent"] for item in containers)
                rss += sum(item["rss_mib"] for item in containers)
                peaks = {"cpu_percent": max(peaks["cpu_percent"], cpu), "rss_mib": max(peaks["rss_mib"], rss)}
                stopped_reason = guard.check(cpu_percent=cpu, rss_mib=rss, free_percent=free, thermal=thermal)
                snapshot = {"sampled_at_utc": datetime.now(timezone.utc).isoformat(), "root_pid": process.pid, "state": "stopping_resource_guard" if stopped_reason else "running", "cpu_percent": cpu, "rss_mib": rss, "system_free_percent": free, "thermal": thermal.strip(), "processes": [asdict(item) for item in selected], "containers": containers, "peaks": peaks, "reason": stopped_reason}
                history.write(json.dumps(snapshot) + "\n")
                history.flush()
                atomic_json(monitor_root / "status.json", snapshot)
                if stopped_reason:
                    stop_owned_tree(process, known)
                    break
                # A live process wait keeps the supervisor responsive without CPU polling loops.
                try:
                    process.wait(timeout=interval)
                except subprocess.TimeoutExpired:
                    pass
    except BaseException as error:
        stopped_reason = f"Supervisor stopped: {type(error).__name__}: {error}"
        cleanup_failed = isinstance(error, CleanupError)
        if process is not None and not isinstance(error, CleanupError):
            try:
                stop_owned_tree(process, known)
            except CleanupError as cleanup_error:
                cleanup_failed = True
                stopped_reason += f"; cleanup failed: {cleanup_error}"
                raise cleanup_error from error
        raise
    finally:
        if process is not None:
            atomic_json(monitor_root / "status.json", {"state": "cleanup_failed" if cleanup_failed else ("stopped_resource_guard" if stopped_reason else ("complete" if process.returncode == 0 else "experiment_failed")), "root_pid": process.pid, "returncode": process.returncode, "reason": stopped_reason, "peaks": peaks, "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    return 75 if stopped_reason else process.returncode
