"""Verify resource guard decisions and cleanup without creating CPU load."""
from __future__ import annotations

import json
import signal
import subprocess
import sys

import pytest

from src.agent import resource_monitor as monitor


def test_process_parsing_and_owned_tree():
    samples = monitor.parse_processes("10 1 10 0.2 2048 S Fri Oct 2 10:00:00 2026 /path with spaces/python\n11 10 11 99.3 100000 S Fri Oct 2 10:00:01 2026 python\n12 11 11 0.0 2000 S Fri Oct 2 10:00:02 2026 child\n13 1 13 200 300000 S Fri Oct 2 10:00:00 2026 unrelated\n")
    assert monitor.descendants(samples, 10) == {10, 11, 12}
    assert samples[10].name == '/path with spaces/python'
    assert samples[11].cpu_percent == 99.3


def test_guard_allows_one_busy_core_and_stops_sustained_excess():
    guard = monitor.ResourceGuard()
    assert guard.check(cpu_percent=100, rss_mib=200, free_percent=38, thermal='Note: No thermal warning level has been recorded') is None
    for _ in range(2):
        assert guard.check(cpu_percent=160, rss_mib=200, free_percent=38, thermal='') is None
    assert 'CPU exceeded' in guard.check(cpu_percent=160, rss_mib=200, free_percent=38, thermal='')


def test_guard_stops_memory_and_thermal_warnings():
    assert 'resident memory' in monitor.ResourceGuard().check(cpu_percent=0, rss_mib=2100, free_percent=40, thermal='')
    assert 'thermal' in monitor.ResourceGuard().check(cpu_percent=0, rss_mib=100, free_percent=40, thermal='CPU_Speed_Limit = 75')
    guard = monitor.ResourceGuard(sustained_samples=2)
    assert guard.check(cpu_percent=0, rss_mib=100, free_percent=9, thermal='') is None
    assert 'available memory' in guard.check(cpu_percent=0, rss_mib=100, free_percent=9, thermal='')


def test_cpu_counter_recovers_and_memory_query_is_read_only():
    guard = monitor.ResourceGuard(sustained_samples=2)
    for value in [160, 0, 160]:
        assert guard.check(cpu_percent=value, rss_mib=100, free_percent=30, thermal='') is None
    assert monitor.memory_free_percent('System-wide memory free percentage: 38%') == 38


def test_cleanup_refreshes_groups_and_kills_only_surviving_child(monkeypatch):
    root = monitor.ProcessSample(10, 1, 10, 0, 1000, 'root start', 'python')
    child = monitor.ProcessSample(11, 10, 11, 0, 1000, 'child start', 'python')
    unrelated = monitor.ProcessSample(13, 1, 13, 0, 1000, 'unrelated', 'python')
    snapshots = iter([{10: root, 11: child, 13: unrelated}, {11: child, 13: unrelated}, {13: unrelated}])
    monkeypatch.setattr(monitor, 'read_processes', lambda: next(snapshots))
    calls = []
    monkeypatch.setattr(monitor.os, 'killpg', lambda group, signum: calls.append((group, signum)))
    class ExitedParent:
        pid = 10
        def poll(self): return -15
        def wait(self, timeout): return -15
    monitor.stop_owned_tree(ExitedParent(), grace=0)
    assert set(calls) == {(10, signal.SIGTERM), (11, signal.SIGTERM), (11, signal.SIGKILL)}
    assert (10, signal.SIGKILL) not in calls


def test_cleanup_does_not_signal_reused_child_pid(monkeypatch):
    original = monitor.ProcessSample(11, 10, 11, 0, 1000, 'original start', 'python')
    reused = monitor.ProcessSample(11, 1, 99, 0, 1000, 'different start', 'python')
    root = monitor.ProcessSample(10, 1, 10, 0, 1000, 'root start', 'python')
    snapshots = iter([{10: root, 11: original}, {11: reused}, {11: reused}])
    monkeypatch.setattr(monitor, 'read_processes', lambda: next(snapshots))
    calls = []
    monkeypatch.setattr(monitor.os, 'killpg', lambda group, signum: calls.append((group, signum)))
    class ExitedParent:
        pid = 10
        def poll(self): return -15
        def wait(self, timeout): return -15
    monitor.stop_owned_tree(ExitedParent(), grace=0)
    assert not any(group == 99 for group, _ in calls)


@pytest.mark.parametrize("container_memory", [False, True])
def test_supervisor_stops_owned_process_on_resource_alarm(tmp_path, monkeypatch, container_memory):
    call = 0
    real_popen = subprocess.Popen
    processes = []
    signals = []
    def popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process
    def sample():
        nonlocal call
        call += 1
        if call == 1:
            return {}, '', 40
        process = processes[0]
        host_memory = 200 if container_memory else 5000
        return {process.pid: monitor.ProcessSample(process.pid, 1, process.pid, 0, host_memory * 1024, 'same start', 'python')}, '', 40
    monkeypatch.setattr(monitor.subprocess, 'Popen', popen)
    monkeypatch.setattr(monitor, 'sample_host', sample)
    def current_processes():
        process = processes[0]
        return {process.pid: monitor.ProcessSample(process.pid, 1, process.pid, 0, 1000, 'same start', 'python')} if process.poll() is None else {}
    monkeypatch.setattr(monitor, 'read_processes', current_processes)
    # Keep group-signal verification portable to sandboxes that deny signaling a reaped group.
    def kill_group(group, signum):
        signals.append((group, signum))
        if signum == signal.SIGTERM:
            processes[0].terminate()
            processes[0].wait(timeout=5)
    monkeypatch.setattr(monitor.os, 'killpg', kill_group)
    sampler = (lambda: [{'name': 'fixture-worker', 'cpu_percent': 75, 'rss_mib': 4800}]) if container_memory else None
    result = monitor.supervise([sys.executable, '-c', 'import time;time.sleep(10)'], tmp_path, interval=0.01, container_sampler=sampler)
    assert result == 75 and processes[0].poll() is not None
    assert signals == [(processes[0].pid, signal.SIGTERM)]
    status = json.loads((tmp_path / 'status.json').read_text())
    assert status['state'] == 'stopped_resource_guard'
    assert 'resident memory' in status['reason']
    assert (tmp_path / 'resources.jsonl').read_text()
    snapshot = json.loads((tmp_path / 'resources.jsonl').read_text().splitlines()[0])
    assert snapshot['rss_mib'] == 5000
    assert snapshot['cpu_percent'] == (75 if container_memory else 0)
    assert bool(snapshot['containers']) == container_memory
