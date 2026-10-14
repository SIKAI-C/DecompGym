"""Prospective wrong-path delivery fails honestly and survives interrupted closure."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys

import pytest

from src.agent import agy_client, policy_completion as completion
from src.agent.agy_client import AgyClient
from src.agent.paired_runner import PairedRunner, ReflectionDeliveryFailure, save
from src.agent.experiment_runner import AttemptResult
from src.native_execution_phase import WORKSPACE_CHANGE, WRONG_TARGET_CHANGE, validate_phase


MODEL = 'claude-sonnet-4-6'


def delivery_case(tmp_path, index=1):
    root = tmp_path / 'campaign'
    source = root / 'dataset/bd/ufl/ufl_small_seed000'
    source.mkdir(parents=True)
    for name, contents in [('data.json', '{"m":2}'), ('prompt.txt', 'Implement Benders decomposition.'), ('meta.json', '{"ground_truth_objective":4}'), ('gold_monolithic.json', '{"objective_value":4}')]:
        (source / name).write_text(contents)
    case = {'suite': 'bd', 'problem': 'ufl', 'instance': source.name, 'size': 'small', 'seed': 0,
            'sha256': {p.name: completion.digest(p) for p in source.iterdir()}}
    manifest = {'protocol_id': 'base', 'workers': 2, 'max_attempts': 10, 'backend': 'native', 'gpt_order': [], 'agy_order': [MODEL], 'instances': [case]}
    save(root / 'campaign.json', manifest)
    phase_dir = root / 'execution_phases/workspace'
    phase_dir.mkdir(parents=True)
    for source_name, frozen_name in [(agy_client.__file__, 'agy_client.py'), (completion.__file__, 'policy_completion.py')]:
        (phase_dir / frozen_name).write_bytes(Path(source_name).read_bytes())
    (phase_dir / 'native_execution_phase.py').write_text('frozen fixture wrapper')
    save(phase_dir / 'preserved_artifacts.json', {})
    phase = {'base_protocol_id': 'base', 'base_manifest_sha256': completion.digest(root / 'campaign.json'),
             'changes': {'active_codex_invocation_limit': {'from': 1, 'to': 2}, 'agy_workspace_clarification': WORKSPACE_CHANGE, 'wrong_target_reflection_failure': WRONG_TARGET_CHANGE},
             'agy_client_sha256': completion.digest(phase_dir / 'agy_client.py'),
             'wrapper_sha256': completion.digest(phase_dir / 'native_execution_phase.py'),
             'preserved_inventory_sha256': completion.digest(phase_dir / 'preserved_artifacts.json'),
             'workspace_change_authorization': 'User: clarify the workspace minimally; a subsequent wrong-path write counts as failure.'}
    phase['phase_id'] = completion.identity(phase)
    save(phase_dir / 'phase.json', phase)
    save(root / 'active_execution_phase.json', {'phase_id': phase['phase_id'], 'phase_file': str(phase_dir / 'phase.json')})
    directory = root / 'runs' / MODEL / 'reflection/bd/ufl' / source.name
    directory.mkdir(parents=True)
    for name in ['data.json', 'prompt.txt']:
        (directory / name).write_bytes((source / name).read_bytes())
    save(directory / 'identity.json', {'model': MODEL, 'protocol_id': 'base'})
    save(directory / 'shared_first.json', {'generation_invocation_id': 'generation-1'})
    scores = []
    for number in range(1, index + 1):
        attempt = directory / 'attempts' / f'attempt_{number:02d}'
        attempt.mkdir(parents=True)
        (attempt / 'solver.py').write_text('')
        score = {'attempt': number, 'status': 'agent_failed', 'exec_ok': False, 'verify_ok': False,
                 'objective': None, 'gold_objective': 4, 'gap_to_gold': None, 'issues': [],
                 'metrics': {'generation': {'ok': False, 'invocation_id': f'generation-{number}'}}}
        save(attempt / 'score_receipt.json', {'result': score, 'artifact_sha256': {'solver.py': completion.digest(attempt / 'solver.py')}})
        save(attempt / 'agent_receipt.json', {'result': score['metrics']['generation']})
        scores.append(score)
    raw = {'model': MODEL, 'suite': 'bd', 'problem': 'ufl', 'instance': source.name, 'protocol_id': 'base', 'max_attempts': 10,
           'reflection': True, 'shared_first_attempt': True, 'backend': 'agy-cli', 'complete': False, 'attempts': deepcopy(scores[:max(1, index - 1)])}
    save(directory / 'instance_summary.json', raw)
    plain = root / 'runs' / MODEL / 'plain/bd/ufl' / source.name
    save(plain / 'instance_summary.json', raw | {'reflection': False, 'complete': True, 'attempts': [scores[0]] * 10})
    executable = tmp_path / 'agy'
    executable.write_text(f'#!{sys.executable}\n' + '''
import json,sys
from pathlib import Path
sys.stdin.read()
Path('attempt_01').mkdir()
Path('attempt_01/debug.md').write_text('A real reflection at the wrong path.')
print(json.dumps({'event':'result','result':{'status':'SUCCESS','response':'written'}}))
''')
    executable.chmod(0o755)

    class Annotated(AgyClient):
        calls = 0
        def edit_file(self, target, prompt, *, context=None, log_dir=None, step='agent'):
            self.calls += 1
            save(log_dir / 'execution_phase.json', {'phase_id': phase['phase_id'], 'role': 'reflection', 'model': MODEL})
            result = super().edit_file(target, prompt, context=context, log_dir=log_dir, step=step)
            # Reconstruct the historical v1 adapter's evidence to keep its validator covered.
            workspace_path = log_dir / f'{step}_workspace.json'
            workspace = json.loads(workspace_path.read_text())
            legacy_sent = (log_dir / f'{step}_prompt_before_workspace.md').read_text()
            note, before = legacy_sent.split('\n\n', 1)
            (log_dir / f'{step}_prompt_before_workspace.md').write_text(before)
            (log_dir / f'{step}_prompt.md').write_text(legacy_sent)
            workspace.update(version=completion.WORKSPACE_AMENDMENT, note=note,
                             before_prompt_sha256=completion.digest(log_dir / f'{step}_prompt_before_workspace.md'),
                             effective_prompt_sha256=completion.digest(log_dir / f'{step}_prompt.md'))
            save(workspace_path, workspace)
            delivery_path = log_dir / f'{step}_delivery_failure.json'
            if delivery_path.exists():
                delivery = json.loads(delivery_path.read_text())
                delivery.update(workspace_note_version=completion.WORKSPACE_AMENDMENT, workspace_record_sha256=completion.digest(workspace_path))
                save(delivery_path, delivery)
            return result

    client = Annotated(MODEL, executable=str(executable), role='reflection')
    class HistoricalDeliveryRunner(PairedRunner):
        # Exercise the preserved historical closure validator explicitly. New campaigns consume an omission and continue.
        def record_reflection_omission(self, call, target, step, metrics):
            request = json.loads((call / 'request.json').read_text())
            self.commit_delivery_failure(call, target, step, request['request_sha256'], metrics)
            return metrics

        def validate_reflection_omission(self, call, target, metrics):
            self.record_reflection_omission(call, target, 'reflection', metrics)

    runner = HistoricalDeliveryRunner(client=client, judge_client=client, reflection_client=client, protocol_id='base', model=MODEL, suite='bd', problem=['ufl'], max_attempts=10)
    target = attempt / 'debug.md'
    return root, source, directory, attempt, target, client, runner, phase_dir


@pytest.mark.parametrize('index', [1, 2, 10])
def test_wrong_path_closes_actual_prefix_without_padding_or_relabeling(tmp_path, index, monkeypatch):
    root, source, directory, attempt, target, client, runner, phase_dir = delivery_case(tmp_path, index)
    with pytest.raises(ReflectionDeliveryFailure):
        runner.edit(client, target, 'Reflect.', [directory / 'data.json'], 'reflection')
    assert client.calls == 1 and target.read_text() == ''
    before = {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    receipt = completion.build_delivery_receipt(directory, attempt, root, recorded_at=datetime.now(timezone.utc).isoformat())
    save(directory / completion.DELIVERY_RECEIPT, receipt)
    summary = completion.load_policy_summary(directory)
    assert completion.policy_is_complete(summary, 10, 'base')
    assert completion.is_terminal_failure(summary) and summary['completion_kind'] == completion.DELIVERY_KIND
    assert summary['outcome'] == 'failed' and summary['actual_attempts'] == index
    assert len(summary['attempts']) == index and summary['unexecuted_attempt_slots'] == 10 - index
    assert all(a['status'] == 'agent_failed' for a in summary['attempts'])
    assert all((directory / name).read_bytes() == contents for name, contents in before.items())
    monkeypatch.setattr(runner, 'prepare', lambda *args: pytest.fail('Closed policy was prepared again'))
    assert runner.run_pair(source, root / 'runs' / MODEL)['reflection']['complete']
    assert client.calls == 1
    analysis = importlib.import_module('src.03_analysis').analyze_instance(directory, 0.01, False)
    assert analysis['actual_attempts'] == index and analysis['unexecuted_attempt_slots'] == 10 - index
    assert len(analysis['solveQ_at_i']) == 10


@pytest.mark.parametrize('lost_stage_receipt', [False, True])
def test_failed_delivery_recovers_after_crash_without_another_call(tmp_path, lost_stage_receipt):
    root, source, directory, attempt, target, client, runner, phase_dir = delivery_case(tmp_path)
    with pytest.raises(ReflectionDeliveryFailure):
        runner.edit(client, target, 'Reflect.', [], 'reflection')
    if lost_stage_receipt:
        (attempt / 'reflection_receipt.json').unlink()
        call = next((attempt / 'calls').iterdir())
        metrics = json.loads((call / 'reflection_metrics.json').read_text())
        metrics.pop('file_delivery_failure')
        save(call / 'reflection_metrics.json', metrics)
    with pytest.raises(ReflectionDeliveryFailure):
        runner.edit(client, target, 'Reflect.', [], 'reflection')
    assert client.calls == 1
    receipt = completion.build_delivery_receipt(directory, attempt, root, recorded_at='now')
    assert receipt['actual_attempts'] == 1


def test_runner_closes_only_reflection_and_continues_without_retrying_wrong_path(tmp_path, monkeypatch):
    root, source, directory, attempt, target, client, runner, phase_dir = delivery_case(tmp_path)
    plain = root / 'runs' / MODEL / 'plain/bd/ufl' / source.name
    plain_first = plain / 'attempts/attempt_01'
    plain_first.mkdir(parents=True)
    for p in attempt.iterdir():
        if p.is_file():
            (plain_first / p.name).write_bytes(p.read_bytes())
    first = json.loads((attempt / 'score_receipt.json').read_text())['result']
    monkeypatch.setattr(runner, 'prepare', lambda source, root, mode: plain if mode == 'plain' else directory)
    monkeypatch.setattr(runner, 'ensure_generation_failure_log', lambda *args: None)
    def execute(index, *args):
        if index == 1:
            return AttemptResult(**deepcopy(first))
        assert runner.mode == 'plain' and index == 2
        return AttemptResult(2, 'ok', True, True, 4, 4, 0, [], feasibility_ok=True)
    monkeypatch.setattr(runner, 'run_attempt', execute)
    score_before = (attempt / 'score_receipt.json').read_bytes()
    summaries = runner.run_pair(source, root / 'runs' / MODEL)
    assert summaries['plain']['complete'] and summaries['plain']['attempts'][1]['status'] == 'ok'
    assert summaries['reflection']['completion_kind'] == completion.DELIVERY_KIND
    assert summaries['reflection']['actual_attempts'] == 1 and client.calls == 1
    assert (attempt / 'score_receipt.json').read_bytes() == score_before
    assert not (attempt / 'debug.md').read_text()
    assert runner.run_pair(source, root / 'runs' / MODEL)['reflection']['complete']
    assert client.calls == 1


@pytest.mark.parametrize('changed', ['prompt', 'alternative', 'phase', 'quota', 'required_target', 'score'])
def test_wrong_target_closure_rejects_changed_or_ineligible_evidence(tmp_path, changed):
    root, source, directory, attempt, target, client, runner, phase_dir = delivery_case(tmp_path)
    with pytest.raises(ReflectionDeliveryFailure):
        runner.edit(client, target, 'Reflect.', [], 'reflection')
    receipt = completion.build_delivery_receipt(directory, attempt, root, recorded_at='now')
    save(directory / completion.DELIVERY_RECEIPT, receipt)
    call = next((attempt / 'calls').iterdir())
    if changed == 'prompt':
        (call / 'reflection_prompt.md').write_text('changed')
    elif changed == 'alternative':
        (call / 'unexpected_outputs/attempt_01/debug.md').write_text('changed')
    elif changed == 'phase':
        phase = json.loads((phase_dir / 'phase.json').read_text())
        phase['workspace_change_authorization'] = ''
        save(phase_dir / 'phase.json', phase)
    elif changed == 'quota':
        metrics = json.loads((call / 'reflection_metrics.json').read_text())
        metrics.update(returncode=3, terminal_state='failed', error='quota reached')
        save(call / 'reflection_metrics.json', metrics)
    elif changed == 'required_target':
        target.write_text('a later correct target')
    elif changed == 'score':
        save(attempt / 'score_receipt.json', {'result': {'status': 'ok'}})
    with pytest.raises(completion.TerminalCompletionError):
        completion.load_policy_summary(directory)


def test_new_phase_pins_adapter_and_keeps_old_phase_validation(tmp_path):
    root, source, directory, attempt, target, client, runner, phase_dir = delivery_case(tmp_path)
    wrapper = phase_dir / 'native_execution_phase.py'
    assert validate_phase(root, phase_dir / 'phase.json', executing_file=wrapper)['changes']['agy_workspace_clarification'] == WORKSPACE_CHANGE
    (phase_dir / 'agy_client.py').write_text('tampered')
    with pytest.raises(ValueError, match='Clarified AGY adapter changed'):
        validate_phase(root, phase_dir / 'phase.json', executing_file=wrapper)
