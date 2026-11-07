"""Opt-in lifecycle checks against the built worker, without credentials or inference."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from src.agent import container as containers
from src.agent.container import ContainerError, ContainerResult, DockerRuntime


pytestmark = pytest.mark.skipif(os.environ.get("DECOMPBENCH_DOCKER_TESTS") != "1", reason="requires the reviewed local Docker image")
ROOT = Path(__file__).resolve().parents[1]
FAKE_CLI = """import json,sys,time
from pathlib import Path
with Path('/work/invocations.txt').open('a') as stream: stream.write('started\\n')
Path('/work/solver.py').write_text('print(1)\\n')
if sys.argv[1] == 'wait': time.sleep(120)
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':0,'output_tokens':0}}),flush=True)
"""
FIXTURE = "\n".join([
    "import os,sys",
    "from pathlib import Path",
    "sys.path.insert(0,'/opt/decompbench')",
    "import worker",
    "Path('/tmp/fake_cli.py').write_text(" + repr(FAKE_CLI) + ")",
    "worker.credentials=lambda backend,job:(os.environ.copy(),set(),lambda:None)",
    "worker.cli_command=lambda backend,job:['python3','/tmp/fake_cli.py',job['fixture_mode']]",
    "raise SystemExit(worker.main())",
])


class FixtureRuntime(DockerRuntime):
    """Replace only CLI/auth boundaries; retain the image's actual worker and lifecycle."""

    def command(self, *args, **kwargs):
        args = list(args)
        if args[0] == "create":
            position = args.index("/opt/decompbench/worker.py")
            args[position:position + 1] = ["-c", FIXTURE]
        return super().command(*args, **kwargs)


def runtime():
    return FixtureRuntime(seccomp_profile=ROOT / "docker/agents/seccomp-userns.json")


def run_fixture(engine, path, mode):
    return engine.run(os.environ["DECOMPBENCH_DOCKER_IMAGE"], ["agent", "codex-cli"],
                      payload={"target": "solver.py", "model": "fixture", "role": "diagnostic", "effort": "medium", "prompt": "local fixture only", "timeout": 30, "fixture_mode": mode},
                      state_path=path, timeout=40, artifacts=["solver.py", "worker.json", "events.jsonl", "invocations.txt"])


def assert_disposed(engine, state):
    assert engine.inspect(state["name"]) is None
    for name in (state["work_volume"], state["record_volume"]):
        result = engine.command("volume", "inspect", name)
        assert result.returncode and "no such volume" in result.stderr.lower()


def test_real_worker_finishes_while_heartbeat_reader_is_active(tmp_path):
    engine = runtime()
    path = tmp_path / "container.json"
    result = run_fixture(engine, path, "finish")
    assert result.returncode == 0, result.stderr
    assert "_enter_buffered_busy" not in result.stderr
    assert result.artifacts["invocations.txt"] == "started\n"
    assert json.loads(result.artifacts["worker.json"])["interruption"] is None
    assert_disposed(engine, json.loads(path.read_text()))


def test_real_controller_death_stops_the_worker_and_recovers_once(tmp_path):
    engine = runtime()
    path = tmp_path / "container.json"
    program = "import runpy;from pathlib import Path;m=runpy.run_path(" + repr(str(Path(__file__))) + ");m['run_fixture'](m['runtime'](),Path(" + repr(str(path)) + "),'wait')"
    owner = subprocess.Popen([sys.executable, "-c", program], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    state = None
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if path.exists():
                state = json.loads(path.read_text())
                ready = engine.command("exec", state["name"], "test", "-f", "/work/invocations.txt")
                if ready.returncode == 0:
                    break
            assert owner.poll() is None, "Synthetic controller exited before its worker started"
            time.sleep(0.1)
        else:
            pytest.fail("Synthetic CLI did not start")
        owner.kill()
        owner.wait(timeout=5)
        started = time.monotonic()
        result = engine.recover(path)
        assert time.monotonic() - started < 20
        assert result.metadata["recovered_existing_container"]
        assert result.artifacts["invocations.txt"] == "started\n"
        assert json.loads(result.artifacts["worker.json"])["interruption"] == "controller_connection_lost"
        assert engine.recover(path).artifacts == result.artifacts
        assert_disposed(engine, state)
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)
        if state is not None:
            engine._dispose(state)


def test_real_collector_failure_preserves_artifacts_for_recovery(tmp_path, monkeypatch):
    engine = runtime()
    path = tmp_path / "container.json"
    real_process = containers.bounded_process

    def fail_collection(command, **kwargs):
        if "collect" in command:
            return ContainerResult(1, "", "injected collector failure", 0)
        return real_process(command, **kwargs)

    monkeypatch.setattr(containers, "bounded_process", fail_collection)
    with pytest.raises(ContainerError, match="collector failed"):
        run_fixture(engine, path, "finish")
    state = json.loads(path.read_text())
    assert state["phase"] == "stopped" and engine.inspect(state["name"]) is None
    assert engine.command("volume", "inspect", state["work_volume"]).returncode == 0
    monkeypatch.setattr(containers, "bounded_process", real_process)
    result = engine.recover(path)
    assert result.artifacts["solver.py"] == "print(1)\n"
    assert result.artifacts["invocations.txt"] == "started\n"
    assert_disposed(engine, state)


MOUNT_CLI = """import json,sys
from pathlib import Path
checks={}
checks['data']=Path('/work/instance/data.json').read_text()
checks['history']=Path('/work/history/attempt_01/debug.md').read_text()
def denied(action):
    try:
        action();return False
    except OSError:
        return True
checks['instance_write_denied']=denied(lambda:Path('/work/instance/data.json').write_text('changed'))
checks['instance_create_denied']=denied(lambda:Path('/work/instance/meta.json').write_text('{}'))
checks['history_write_denied']=denied(lambda:Path('/work/history/attempt_01/debug.md').write_text('changed'))
checks['history_create_denied']=denied(lambda:Path('/work/history/attempt_01/solver.py').write_text('print(3)'))
Path('/work/elsewhere').mkdir()
Path('/work/elsewhere/solver.py').write_text('print(2)')
checks['entries']=sorted(p.name for p in Path('/work').iterdir())
Path('/work/solver.py').write_text(json.dumps(checks) if sys.argv[1]=='deliver' else '')
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':0,'output_tokens':0}}),flush=True)
"""
MOUNT_FIXTURE = FIXTURE.replace(repr(FAKE_CLI), repr(MOUNT_CLI))


class MountRuntime(DockerRuntime):
    """The real lifecycle, context loader and worker with a fake CLI; the loader container has no worker path to replace."""

    def command(self, *args, **kwargs):
        args = list(args)
        if args[0] == "create" and "/opt/decompbench/worker.py" in args:
            position = args.index("/opt/decompbench/worker.py")
            args[position:position + 1] = ["-c", MOUNT_FIXTURE]
        return super().command(*args, **kwargs)


@pytest.mark.parametrize("mode", ["deliver", "omit"])
def test_real_context_mounts_are_read_only_and_only_the_target_is_collected(tmp_path, mode):
    engine = MountRuntime(seccomp_profile=ROOT / "docker/agents/seccomp-userns.json")
    context = {"/work/instance": {"data.json": b'{"m": 2}\n'}, "/work/history": {"attempt_01/debug.md": b"notes\n", "attempt_01/solver.py": b"print(1)\n"}}
    path = tmp_path / "container.json"
    result = engine.run(os.environ["DECOMPBENCH_DOCKER_IMAGE"], ["agent", "codex-cli"],
                        payload={"target": "solver.py", "model": "fixture", "role": "diagnostic", "effort": "medium", "prompt": "local fixture only", "timeout": 30, "fixture_mode": mode},
                        state_path=path, timeout=40, network="none", artifacts=["solver.py", "worker.json", "delivery.json"], context=context)
    state = json.loads(path.read_text())
    assert all({"Type": "volume", "Name": state["context_volumes"][d], "Destination": d, "RW": False} in state["mounts"] for d in context)
    assert set(result.artifacts) <= {"solver.py", "worker.json", "delivery.json"}
    delivery = json.loads(result.artifacts["delivery.json"])
    if mode == "deliver":
        checks = json.loads(result.artifacts["solver.py"])
        assert checks["data"] == '{"m": 2}\n' and checks["history"] == "notes\n"
        assert all(checks[name] for name in ("instance_write_denied", "instance_create_denied", "history_write_denied", "history_create_denied"))
        assert checks["entries"] == ["elsewhere", "history", "instance", "solver.py"]
        assert delivery["misplaced"] == {}
    else:
        # An empty target is not rescued: the worker only reports other solver.py files (the mounted copy and the misplaced one).
        assert result.artifacts["solver.py"] == ""
        assert sorted(delivery["misplaced"]) == ["elsewhere/solver.py", "history/attempt_01/solver.py"]
    assert_disposed(engine, state)
    for name in state["context_volumes"].values():
        observed = engine.command("volume", "inspect", name)
        assert observed.returncode and "no such volume" in observed.stderr.lower()
