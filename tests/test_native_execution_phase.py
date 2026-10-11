from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from src.agent.codex_client import CodexResult
from src.native_execution_phase import GENERATION_FEEDBACK_CHANGE, NativeCallGate, NULL_SOLUTION_CHANGE, RECOVERY_CHANGE, auth_failure_marker, digest, identity, install_overrides, save, validate_phase


def gate_fixture(tmp_path):
    state, phase = tmp_path / "state", tmp_path / "phase"
    state.mkdir(); phase.mkdir()
    return NativeCallGate(state, phase)


def test_two_calls_overlap_third_waits_and_exception_releases_slot(tmp_path):
    gate = gate_fixture(tmp_path)
    entered = [threading.Event() for _ in range(3)]
    release = threading.Event()
    def call(index):
        with gate.enter():
            entered[index].set()
            if index < 2:
                assert release.wait(3)
            if index == 0:
                raise RuntimeError("controlled client failure")
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(call, 0); second = pool.submit(call, 1)
        try:
            assert entered[0].wait(2) and entered[1].wait(2)
            third = pool.submit(call, 2)
            assert not entered[2].wait(0.1)
        finally:
            release.set()
        with pytest.raises(RuntimeError):first.result()
        second.result(); third.result()
    assert gate.active == 0


def test_auth_fallback_drains_existing_calls_then_remains_serial(tmp_path):
    gate = gate_fixture(tmp_path)
    entered = [threading.Event() for _ in range(3)]
    release = [threading.Event(), threading.Event()]
    def call(index):
        with gate.enter():
            entered[index].set()
            if index < 2:assert release[index].wait(3)
    with ThreadPoolExecutor(max_workers=3) as pool:
        first, second = pool.submit(call, 0), pool.submit(call, 1)
        try:
            assert entered[0].wait(2) and entered[1].wait(2)
            gate.fallback("refresh token", role="algorithm_judge", invocation_id="fixture")
            third = pool.submit(call, 2)
            release[0].set();first.result()
            assert not entered[2].wait(0.1)
        finally:
            release[0].set();release[1].set()
        second.result();third.result()
    assert gate.limit == 1
    assert NativeCallGate(gate.state_directory, gate.phase_directory).limit == 1


@pytest.mark.parametrize("message", ["401 Unauthorized", "refresh_token_reused", "Your access token has expired", "authentication required", "failed to refresh token: rejected", "failed to refresh access token: request timed out"])
def test_auth_failure_detection(message):
    assert auth_failure_marker(CodexResult(False, 1, 0, {}, message), "")
    assert auth_failure_marker(CodexResult(True, 0, 0, {}), message)
    assert auth_failure_marker(CodexResult(True, 0, 0, {}), "") is None


def test_model_catalog_refresh_timeout_is_not_an_authentication_failure():
    diagnostics = "2026-10-08T16:16:36.869718Z ERROR codex_models_manager::manager: failed to refresh available models: request timed out"
    result = CodexResult(True, 0, 111.6, {}, terminal_state="completed")
    assert auth_failure_marker(result, diagnostics) is None
    assert auth_failure_marker(result, diagnostics + "\nERROR native auth: refresh_token_reused") == "refresh_token_reused"


def prepare_phase(tmp_path):
    root=tmp_path/'campaign';phase_dir=root/'execution_phases/native_codex2';phase_dir.mkdir(parents=True)
    base={'protocol_id':'base','workers':2};save(root/'campaign.json',base)
    artifact=root/'runs/completed.json';artifact.parent.mkdir();artifact.write_text('preserve')
    wrapper=phase_dir/'native_execution_phase.py';wrapper.write_text('frozen wrapper')
    inventory=phase_dir/'preserved_artifacts.json';save(inventory,{'runs/completed.json':digest(artifact)})
    phase={'changes':{'active_codex_invocation_limit':{'from':1,'to':2}},'base_manifest_sha256':digest(root/'campaign.json'),'base_protocol_id':'base','wrapper_sha256':digest(wrapper),'preserved_inventory_sha256':digest(inventory)}
    phase['phase_id']=identity(phase);path=phase_dir/'phase.json';save(path,phase)
    return root,path,wrapper,artifact


@pytest.mark.parametrize('changed',['phase','base','wrapper','artifact','inventory'])
def test_execution_phase_rejects_changed_inputs(tmp_path,changed):
    root,path,wrapper,artifact=prepare_phase(tmp_path)
    assert validate_phase(root,path,executing_file=wrapper)['base_protocol_id']=='base'
    target={'phase':path,'base':root/'campaign.json','wrapper':wrapper,'artifact':artifact,'inventory':path.parent/'preserved_artifacts.json'}[changed]
    target.write_text('{}' if changed in {'phase','base','inventory'} else 'changed')
    with pytest.raises((ValueError,KeyError)):
        validate_phase(root,path,executing_file=wrapper)


def test_all_roles_are_annotated_auth_errors_fall_back_and_restart_uses_wrapper(tmp_path):
    root,path,wrapper,_=prepare_phase(tmp_path)
    state=tmp_path/'state';state.mkdir()
    saved=[];supervised=[];legacy_locks=[]
    @contextmanager
    def legacy(name):
        legacy_locks.append(name)
        yield
    paired=SimpleNamespace(runtime_lock=legacy)
    campaign=SimpleNamespace(save=lambda p,v:saved.append((p,v)),supervise=lambda command,monitor,**kwargs:supervised.append(command))
    class Client:
        def __init__(self,role):self.role=role;self.model='fixture'
        def edit_file(self,target,prompt,**kwargs):
            target.write_text('fixture')
            return CodexResult(False,1,0,{},'401 Unauthorized') if self.role=='reflection' else CodexResult(True,0,0,{})
    class Agy(Client):pass
    gate=install_overrides(root,path,json.loads(path.read_text()),campaign=campaign,paired=paired,codex=SimpleNamespace(CodexClient=Client),agy=SimpleNamespace(AgyClient=Agy),state_directory=state,wrapper_path=wrapper)
    for role in ['generation','algorithm_judge','reflection']:
        log=root/role;log.mkdir()
        with paired.runtime_lock('codex-native-auth'):
            Client(role).edit_file(log/'target','prompt',log_dir=log,step='agent')
        annotation=json.loads((log/'execution_phase.json').read_text())
        assert annotation['base_protocol_id']=='base' and annotation['role']==role
    assert gate.limit==1
    with paired.runtime_lock('native-agy-auth'):pass
    assert legacy_locks==['native-agy-auth']
    campaign.supervise(['old-command'],root/'monitor')
    assert str(wrapper) in supervised[0] and 'run' in supervised[0]
    campaign.save(root/'watchdog.json',{'command':['old-command']})
    assert saved[-1][1]['command']==supervised[0] and saved[-1][1]['active_codex_invocation_limit']==1


@pytest.mark.parametrize("recovery_level", [0, 1, 2])
def test_phase_validates_recovery_runner_and_exact_model_reordering(tmp_path, recovery_level):
    root, path, wrapper, _ = prepare_phase(tmp_path)
    original = ['gpt-oss-120b-medium', 'gemini-3.8-flash-medium', 'claude-sonnet-4-6']
    save(root / 'campaign.json', {'protocol_id': 'base', 'workers': 2, 'agy_order': original})
    runner = path.parent / 'paired_runner.py'
    runner.write_text('frozen recovery runner')
    phase = json.loads(path.read_text())
    phase.update(base_manifest_sha256=digest(root / 'campaign.json'), runner_sha256=digest(runner))
    recovery = RECOVERY_CHANGE | (NULL_SOLUTION_CHANGE if recovery_level else {}) | (GENERATION_FEEDBACK_CHANGE if recovery_level == 2 else {})
    phase['changes'].update(runner_recovery=recovery, agy_order={'from': original, 'to': original[1:] + original[:1]})
    def commit_phase():
        phase.pop('phase_id', None)
        phase['phase_id'] = identity(phase)
        save(path, phase)
    commit_phase()
    assert validate_phase(root, path, executing_file=wrapper)['runner_sha256'] == digest(runner)
    phase['changes']['runner_recovery'] = recovery | {'unsupported_recovery': True}
    commit_phase()
    with pytest.raises(ValueError, match='Unrecognized execution-phase change'):
        validate_phase(root, path, executing_file=wrapper)
    phase['changes']['runner_recovery'] = recovery
    commit_phase()
    runner.write_text('changed')
    with pytest.raises(ValueError, match='Recovery runner changed'):
        validate_phase(root, path, executing_file=wrapper)
    runner.write_text('frozen recovery runner')
    phase['changes']['agy_order']['to'] = original[1:] + ['new-model']
    commit_phase()
    with pytest.raises(ValueError, match='exact AGY model set'):
        validate_phase(root, path, executing_file=wrapper)


def test_execution_order_override_preserves_manifest_and_annotates_status(tmp_path):
    root, path, wrapper, _ = prepare_phase(tmp_path)
    state = tmp_path / 'state'
    state.mkdir()
    original = ['oss', 'gemini', 'claude']
    reordered = ['gemini', 'claude', 'oss']
    manifest = {'agy_order': original, 'protocol_id': 'unchanged'}
    saved = []
    campaign = SimpleNamespace(validate=lambda root: manifest, save=lambda path, value: saved.append(value), supervise=lambda *args, **kwargs: None)
    paired = SimpleNamespace(runtime_lock=lambda name: None)
    class Client:
        def edit_file(self, *args, **kwargs):
            raise AssertionError('No model call expected')
    class Agy(Client):
        pass
    phase = json.loads(path.read_text())
    phase['changes']['agy_order'] = {'from': original, 'to': reordered}
    install_overrides(root, path, phase, campaign=campaign, paired=paired, codex=SimpleNamespace(CodexClient=Client), agy=SimpleNamespace(AgyClient=Agy), state_directory=state, wrapper_path=wrapper)
    assert campaign.validate(root)['agy_order'] == reordered
    assert manifest['agy_order'] == original
    campaign.save(root / 'status.json', {'state': 'running'})
    assert saved[-1]['execution_agy_order'] == reordered
