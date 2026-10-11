from datetime import datetime, timezone
import os

from src.agent.paired_runner import save
from src.dashboard_server import Dashboard


def test_native_first_pass_does_not_imply_retry_pair_is_complete(tmp_path):
    model, instance = 'gpt-6-sol', 'ufl_small_seed000'
    now = datetime.now(timezone.utc).isoformat()
    save(tmp_path / 'campaign.json', {'backend':'native', 'gpt_order':[model], 'agy_order':[], 'max_attempts':10, 'instances':[{'instance':instance,'suite':'bd','problem':'ufl','size':'small','seed':0}]})
    save(tmp_path / 'status.json', {'state':'running','pid':os.getpid(),'active_model':model,'active_instances':{'experiment_0':{'instance':instance,'mode':'reflection','attempt':2,'phase':'agent'}}})
    save(tmp_path / 'monitor/status.json', {'sampled_at_utc':now,'cpu_percent':12,'system_free_percent':40,'thermal':'No warnings'})
    attempt = {'attempt':1,'status':'ok','metrics':{'generation':{'invocation_id':'shared','usage':{'input_tokens':10}}}}
    for mode in ['plain','reflection']:
        save(tmp_path / 'runs' / model / mode / 'bd/ufl' / instance / 'instance_summary.json', {'model':model,'instance':instance,'attempts':[attempt],'complete':False})
    value = Dashboard(tmp_path).snapshot()
    assert value['totals']['pass_rate'] == 1
    assert value['totals']['completed'] == 0 and value['models'][0]['state'] == 'running'
    assert len(value['active_workers']) == 1 and 'attempt 2/10' in value['active_workers'][0]['phase']
    assert value['usage']['generation']['input_tokens'] == 10
    assert value['monitor']['docker_and_controller_cpu_percent'] == 12


def test_dashboard_uses_validated_execution_order(tmp_path):
    original = ['gpt-oss-120b-medium', 'gemini-3.8-flash-medium', 'claude-sonnet-4-6']
    reordered = original[1:] + original[:1]
    save(tmp_path / 'campaign.json', {'backend': 'native', 'gpt_order': [], 'agy_order': original, 'max_attempts': 10, 'instances': []})
    save(tmp_path / 'status.json', {'state': 'running', 'pid': os.getpid(), 'execution_agy_order': reordered})
    assert [model['id'] for model in Dashboard(tmp_path).snapshot()['models']] == reordered
    save(tmp_path / 'status.json', {'state': 'running', 'pid': os.getpid(), 'execution_agy_order': ['different-model']})
    assert [model['id'] for model in Dashboard(tmp_path).snapshot()['models']] == original
