"""Durable container recovery follows the existing handle and preserves uncollected logs."""
import json
import subprocess
import sys
import time

import pytest

from src.agent import container as containers
from src.agent.container import ContainerError, ContainerResult, DockerRuntime, bounded_process, parse_docker_sample
from src.experiments.environment import RuntimeEnvironmentError
from src.experiments.resources import CleanupError


@pytest.mark.parametrize("ownership_matches", [True, False])
def test_campaign_shutdown_uses_owned_receipts_and_preserves_volumes(tmp_path, monkeypatch, ownership_matches):
    runtime = DockerRuntime(seccomp_profile=tmp_path / "profile")
    state_file(tmp_path)
    commands, removed, inspected = [], [], []

    def command(*args):
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, "existing-worker\nother-campaign-worker\n", "")

    def inspect(name):
        inspected.append(name)
        return {"State": {"Running": True}, "Config": {"Labels": {runtime.LABEL: "v2-worker"}}, "Mounts": [{"Type": "volume", "Name": volume} for volume in ("owned-work", "owned-record" if ownership_matches else "foreign-record")]}

    monkeypatch.setattr(runtime, "command", command)
    monkeypatch.setattr(runtime, "inspect", inspect)
    monkeypatch.setattr(runtime, "remove_container", removed.append)
    ticks = iter([0, 9])
    monkeypatch.setattr(containers.time, "monotonic", lambda: next(ticks))
    if ownership_matches:
        runtime.stop_campaign_workers(tmp_path)
        assert removed == ["existing-worker"]
    else:
        with pytest.raises(CleanupError, match="ownership"):
            runtime.stop_campaign_workers(tmp_path)
        assert not removed
    assert inspected == ["existing-worker"]
    assert all(args[0] == "ps" for args in commands)
    assert (tmp_path / "container.json").exists()


def test_heartbeat_channel_stays_open_until_the_worker_finishes():
    program = "import sys;assert sys.stdin.readline().strip()=='request';assert sys.stdin.readline().strip()=='heartbeat';print('complete')"
    result = bounded_process([sys.executable, "-c", program], stdin="request\n", timeout=4, heartbeat=True)
    assert result.returncode == 0 and result.stdout.strip() == "complete" and not result.timed_out


@pytest.mark.parametrize("memory,expected", [("12MiB / 1GiB", 12), ("1GiB / 2GiB", 1024), ("512KiB / 1GiB", 0.5)])
def test_docker_resource_units(memory, expected):
    result = parse_docker_sample({"Name": "owned-worker", "MemUsage": memory, "CPUPerc": "12.5%"})
    assert result == {"name": "owned-worker", "cpu_percent": 12.5, "rss_mib": expected}


class RecoveryRuntime(DockerRuntime):
    def __init__(self, root):
        super().__init__(seccomp_profile=root / "profile")
        self.actions = []
        self.live = True

    def inspect(self, name):
        self.actions.append(("inspect", name))
        return {"State": {"Running": self.live, "Status": "running" if self.live else "exited", "ExitCode": 0}}

    def remove_container(self, name):
        self.actions.append(("remove", name))
        self.live = False

    def _dispose(self, state):
        self.actions.append(("dispose", state["name"]))


def state_file(tmp_path):
    path = tmp_path / "container.json"
    containers.save_state(path, {"phase": "starting", "name": "existing-worker", "work_volume": "owned-work", "record_volume": "owned-record", "image_id": "sha256:fixture", "artifacts": ["solver.py"], "created_at": time.time(), "deadline": time.time() + 10})
    return path


def test_recovery_waits_on_the_same_container_and_collects_after_removal(tmp_path, monkeypatch):
    runtime = RecoveryRuntime(tmp_path)
    commands = []
    def process(command, **kwargs):
        commands.append(command)
        if "wait" in command:
            assert command[-1] == "existing-worker"
            runtime.live = False
            return ContainerResult(0, "", "", 0)
        assert ("remove", "existing-worker") in runtime.actions
        return ContainerResult(0, json.dumps({"ok": True, "artifacts": {"solver.py": "recovered"}}), "", 0)
    monkeypatch.setattr(containers, "bounded_process", process)
    path = state_file(tmp_path)
    result = runtime.recover(path)
    assert result.artifacts["solver.py"] == "recovered" and result.metadata["cleanup_verified"]
    assert not any("create" in command or "start" in command for command in commands)
    count = len(commands)
    assert runtime.recover(path).artifacts == result.artifacts
    assert len(commands) == count


def test_failed_collection_keeps_volumes_for_a_later_recovery(tmp_path, monkeypatch):
    runtime = RecoveryRuntime(tmp_path)
    runtime.live = False
    path = state_file(tmp_path)
    monkeypatch.setattr(containers, "bounded_process", lambda *args, **kwargs: ContainerResult(1, "", "collector unavailable", 0))
    with pytest.raises(ContainerError, match="collector failed"):
        runtime.recover(path)
    assert json.loads(path.read_text())["phase"] == "stopped"
    assert not any(action[0] == "dispose" for action in runtime.actions)
    monkeypatch.setattr(containers, "bounded_process", lambda *args, **kwargs: ContainerResult(0, '{"ok":true,"artifacts":{"solver.py":"preserved"}}', "", 0))
    assert runtime.recover(path).artifacts == {"solver.py": "preserved"}


def test_wrong_engine_is_rejected_before_authenticated_work(tmp_path, monkeypatch):
    runtime = DockerRuntime(seccomp_profile=tmp_path / "profile")
    monkeypatch.setattr(runtime, "command", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "different-engine\n" if args[0] == "info" else "29.4.3 29.4.3\n", ""))
    with pytest.raises(RuntimeEnvironmentError, match="identity or version changed"):
        runtime.bind_environment({"docker_endpoint": "unix:///declared/docker.sock", "docker_engine_id": "expected", "docker_client_version": "29.4.3", "docker_server_version": "29.4.3"})


def test_container_finishing_between_listing_and_sampling_is_not_a_monitor_fault(tmp_path, monkeypatch):
    runtime = DockerRuntime(seccomp_profile=tmp_path / "profile")
    monkeypatch.setattr(runtime, "command", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "finished-worker\n", "") if args[0] == "ps" else subprocess.CompletedProcess(args, 1, "", "No such container"))
    monkeypatch.setattr(runtime, "inspect", lambda _: None)
    assert runtime.resource_samples() == []
