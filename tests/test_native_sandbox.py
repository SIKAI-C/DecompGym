"""Opt-in real macOS isolation tests; synthetic data and no model invocations."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import time

import pytest

from src.agent.native_solver import GUROBI_PROBE, NativeSandboxExecutor
from src.agent.resource_monitor import read_processes


pytestmark = pytest.mark.skipif(os.environ.get("DECOMPBENCH_NATIVE_TESTS") != "1", reason="requires native macOS process permissions and the existing local Gurobi license")


@pytest.fixture(scope="module")
def executor():
    return NativeSandboxExecutor(license_file=Path.home() / "gurobi.lic")


def execute(executor, tmp_path, program, data=b'{"value":1}', timeout=10):
    return executor.execute_bytes(program.encode(), data, timeout, 1, evidence=tmp_path / "evidence.json")


def test_local_license_known_optimum_inside_full_sandbox(executor, tmp_path):
    result = execute(executor, tmp_path, GUROBI_PROBE.replace("THREADS", "1"))
    assert result["exec_ok"], result["stderr"]
    assert result["output"] == {"status": "optimal", "objective_value": 1.0, "threads": 1}
    assert result["bootstrap_ready"] and result["cleanup_verified"]


def test_input_reference_alias_child_and_network_boundaries(executor, tmp_path):
    private = tmp_path / "reference.json"
    private.write_text("PRIVATE_REFERENCE")
    repository_file = Path(__file__).resolve().parents[1] / "README.md"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
        program = r'''
import errno,json,os,socket,subprocess,sys
from pathlib import Path
checks={}
def denied(action):
    try: action(); return False
    except OSError as error:return error.errno in (errno.EACCES,errno.EPERM)
data=Path('../../data.json')
checks['input_read']=json.loads(data.read_text())['value']==1
for name,action in [('write',lambda:data.write_text('changed')),('delete',lambda:data.unlink()),('rename',lambda:data.rename('stolen')),('chmod',lambda:data.chmod(0o777))]:
    checks['input_'+name+'_denied']=denied(action)
checks['solver_write_denied']=denied(lambda:Path('solver.py').write_text('changed'))
checks['config_write_denied']=denied(lambda:Path('gurobi.env').write_text('Threads 8'))
checks['reference_read_denied']=denied(lambda:Path(PRIVATE).read_text())
checks['data_volume_alias_denied']=denied(lambda:Path('/System/Volumes/Data'+PRIVATE).read_text())
checks['repository_read_denied']=denied(lambda:Path(REPOSITORY).read_text())
Path('reference_alias').symlink_to(PRIVATE)
Path('input_alias').symlink_to(data)
checks['reference_alias_denied']=denied(lambda:Path('reference_alias').read_text())
checks['input_alias_write_denied']=denied(lambda:Path('input_alias').write_text('changed'))
checks['network_denied']=denied(lambda:socket.create_connection(('127.0.0.1',PORT),timeout=1))
Path(os.environ['TMPDIR'],'scratch').write_text('ok')
checks['scratch_write']=True
child="import gurobipy,json,sys;from pathlib import Path;ok=False\ntry:Path(sys.argv[1]).read_text()\nexcept PermissionError:ok=True\nprint(json.dumps({'denied':ok,'gurobi':list(gurobipy.gurobi.version())}))"
r=subprocess.run([sys.executable,'-c',child,PRIVATE],capture_output=True,text=True)
checks['child_python_gurobi_and_read_boundary']=r.returncode==0 and json.loads(r.stdout)=={'denied':True,'gurobi':[11,0,3]}
Path('output.json').write_text(json.dumps(checks))
'''.replace("PRIVATE", repr(str(private))).replace("REPOSITORY", repr(str(repository_file))).replace("PORT", str(port))
        result = execute(executor, tmp_path, program)
    assert result["exec_ok"], result["stderr"]
    assert all(result["output"].values()), result["output"]
    assert not result["integrity_violations"] and private.read_text() == "PRIVATE_REFERENCE"


def test_child_exec_cannot_change_group_or_spawn_a_new_session(executor, tmp_path):
    child = r'''
import errno,json,os,sys
name=sys.argv[1]
try:
    if name=='setsid':os.setsid()
    elif name=='setpgid':os.setpgid(0,0)
    else:
        kwargs={'setpgroup':0} if name=='spawn_group' else ({'setsid':True} if name=='spawn_session' else {})
        pid=os.posix_spawn(sys.executable,[sys.executable,'-c','pass'],os.environ,**kwargs)
        os.waitpid(pid,0)
    print('ALLOWED')
except OSError as error:print('DENIED' if error.errno==errno.EPERM else 'OTHER_ERROR')
'''
    program = "import json,subprocess,sys\nfrom pathlib import Path\nchecks={}\n"
    program += "for name in ('setsid','setpgid','posix_spawn','spawn_group','spawn_session'):\n r=subprocess.run([sys.executable,'-c'," + repr(child) + ",name],capture_output=True,text=True)\n checks[name]=r.returncode==0 and r.stdout.strip()=='DENIED'\n"
    program += "Path('output.json').write_text(json.dumps(checks))\n"
    result = execute(executor, tmp_path, program)
    assert result["exec_ok"] and all(result["output"].values()), result


@pytest.mark.parametrize("wait_for_child", [False, True])
def test_successful_parent_exit_cleans_children_before_collection(executor, tmp_path, wait_for_child):
    child = "import signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path('ready').touch();time.sleep(2);Path('output.json').write_text('{\"tampered\":true}')"
    program = "import subprocess,sys,time,json\nfrom pathlib import Path\n"
    program += "child=subprocess.Popen([sys.executable,'-c'," + repr(child) + "],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
    if wait_for_child:
        program += "while not Path('ready').exists():time.sleep(0.01)\n"
    program += "Path('output.json').write_text(json.dumps({'child_pid':child.pid}))\n"
    result = execute(executor, tmp_path, program)
    assert result["exec_ok"] and result["cleanup_verified"], result
    pid = result["output"]["child_pid"]
    processes = read_processes()
    assert pid not in processes or processes[pid].state.startswith("Z")


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory"])
def test_unsafe_solver_output_is_a_candidate_failure(executor, tmp_path, kind):
    lines = {"symlink": "Path('output.json').symlink_to('../../data.json')", "fifo": "os.mkfifo('output.json')", "directory": "Path('output.json').mkdir()"}
    result = execute(executor, tmp_path, "import os\nfrom pathlib import Path\n" + lines[kind])
    assert not result["exec_ok"] and result["cleanup_verified"]
    assert "[framework]" in result["stderr"] and result["output_text"] is None


def test_two_sandboxed_gurobi_workers_complete_independently(executor, tmp_path):
    program = r'''
import errno,json,os,signal,time
from pathlib import Path
import gurobipy as gp
def denied(action):
    try:action();return False
    except OSError as error:return error.errno in (errno.EPERM,errno.EACCES)
with gp.Env(empty=True) as env:
    env.setParam('OutputFlag',0)
    env.setParam('ThreadLimit',1)
    env.setParam('Threads',1)
    env.start()
    Path('environment.ready').touch()
    while not Path('peer.json').exists():time.sleep(0.01)
    peer=json.loads(Path('peer.json').read_text())
    other=Path(peer['workspace'])
    checks={'peer_input_denied':denied(lambda:(other/'../../data.json').read_text()),'peer_output_read_denied':denied(lambda:(other/'environment.ready').read_text()),'peer_write_denied':denied(lambda:(other/'intrusion').write_text('changed')),'peer_signal_denied':denied(lambda:os.kill(peer['pid'],signal.SIGTERM))}
    with gp.Model(env=env) as model:
        model.Params.Presolve=0
        x=model.addVars(80,vtype=gp.GRB.BINARY)
        model.addConstr(gp.quicksum(x.values())>=40.5)
        model.setObjective(gp.quicksum((1+0.01*i)*x[i] for i in range(80)))
        gated=False
        def barrier(model,where):
            global gated
            if not gated:
                gated=True
                Path('optimize.ready').touch()
                while not Path('release').exists():time.sleep(0.01)
        model.optimize(barrier)
        Path('output.json').write_text(json.dumps({'checks':checks,'objective_value':model.ObjVal,'threads':model.Params.Threads,'optimal':model.Status==gp.GRB.OPTIMAL}))
'''
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(execute, executor, tmp_path / str(i), program) for i in range(2)]
        def wait_for(predicate):
            deadline = time.monotonic() + 7
            while time.monotonic() < deadline:
                if predicate():
                    return
                if any(future.done() for future in futures):
                    pytest.fail("Worker exited before the overlap barrier: " + repr([f.result() for f in futures if f.done()]))
                time.sleep(0.01)
            pytest.fail("Worker did not reach the overlap barrier")
        evidence = [tmp_path / str(i) / "evidence.json" for i in range(2)]
        wait_for(lambda: all(path.exists() and json.loads(path.read_text()).get("pid") for path in evidence))
        packets = [json.loads(path.read_text()) for path in evidence]
        workspaces = [Path(packet["workspace"]) for packet in packets]
        wait_for(lambda: all((path / "environment.ready").exists() for path in workspaces))
        for index, path in enumerate(workspaces):
            other = packets[1 - index]
            (path / "peer.json").write_text(json.dumps({"workspace": other["workspace"], "pid": other["pid"]}))
        wait_for(lambda: all((path / "optimize.ready").exists() for path in workspaces))
        # Both independent optimize calls and license environments are live here.
        time.sleep(0.05)
        for path in workspaces:
            (path / "release").touch()
        results = [future.result() for future in futures]
    assert all(result["exec_ok"] and result["cleanup_verified"] for result in results), results
    assert all(all(result["output"]["checks"].values()) and result["output"]["threads"] == 1 and result["output"]["optimal"] for result in results)
    assert all(abs(result["output"]["objective_value"] - 49.2) < 1e-8 for result in results)
    assert results[0]["pid"] != results[1]["pid"]
    assert results[0]["seatbelt_profile_sha256"] != results[1]["seatbelt_profile_sha256"]


def test_executor_receipt_survives_controller_restart_without_execution(executor, tmp_path, monkeypatch):
    work = tmp_path / "case/attempts/attempt_01"
    work.mkdir(parents=True)
    (work.parent.parent / "data.json").write_text('{"value":1}')
    (work / "solver.py").write_text("from pathlib import Path\nPath('output.json').write_text('{\"objective_value\":1}')\n")
    first = executor(work, 10)
    assert first[0]
    def no_repeat(*args, **kwargs):
        raise AssertionError("Committed execution must not run again")
    monkeypatch.setattr(executor, "execute_bytes", no_repeat)
    assert executor(work, 10) == first
    (work / "output.json").write_text("changed")
    with pytest.raises(Exception, match="committed native execution output changed"):
        executor(work, 10)
