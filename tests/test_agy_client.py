import json
from pathlib import Path
import sys

from src.agent.agy_client import AgyClient


def test_native_agy_uses_exact_catalog_slug_and_fresh_single_turn(tmp_path):
    executable = tmp_path / "agy"
    executable.write_text(f"#!{sys.executable}\n" + '''
import json,sys
from pathlib import Path
assert '--new-project' in sys.argv and '--sandbox' in sys.argv
assert '--continue' not in sys.argv and '--conversation' not in sys.argv
assert '--effort' not in sys.argv and '--disable-slash-commands' in sys.argv
assert sys.argv[sys.argv.index('--model') + 1] == 'claude-opus-4-6-thinking'
messages = [json.loads(line) for line in sys.stdin]
assert len(messages) == 1 and messages[0]['event'] == 'user'
assert 'TASK_DATA' in messages[0]['message']['content']
Path('solver.py').write_text('answer = 42')
Path('unrequested.py').write_text('do not copy')
print(json.dumps({'event':'result','result':{'status':'SUCCESS','response':'done','usage':{'input_tokens':12,'cache_read_tokens':3,'output_tokens':4}}}))
''')
    executable.chmod(0o755)
    data = tmp_path / "data.json"
    data.write_text('"TASK_DATA"')
    result = AgyClient('claude-opus-4-6-thinking', executable=str(executable)).edit_file(tmp_path / 'solver.py', 'Write solver.py', context=[data])
    assert result.ok and result.terminal_state == 'completed'
    assert result.usage['cached_input_tokens'] == 3
    assert (tmp_path / 'solver.py').read_text() == 'answer = 42'
    assert not (tmp_path / 'unrequested.py').exists()
    before = (tmp_path / 'agent_prompt_before_workspace.md').read_text()
    sent = (tmp_path / 'agent_prompt.md').read_text()
    workspace = json.loads((tmp_path / 'agent_workspace.json').read_text())
    assert sent.startswith('Only edit the existing solver.py in the current working directory.')
    assert before.startswith('For this editing call, the workspace is ')
    assert 'For this editing call' not in sent and workspace['workspace'] not in sent
    assert workspace['target'] == str(Path(workspace['workspace']) / 'solver.py')
    assert 'TASK_DATA' in before and 'BEGIN CONTEXT:' in before


def test_agy_error_never_replaces_target(tmp_path):
    executable = tmp_path / 'agy'
    executable.write_text(f"#!{sys.executable}\n" + '''
import json,sys
from pathlib import Path
sys.stdin.read()
Path('solver.py').write_text('partial')
print(json.dumps({'event':'result','result':{'status':'ERROR','error':'authentication required'}}))
sys.exit(1)
''')
    executable.chmod(0o755)
    target = tmp_path / 'solver.py'
    target.write_text('original')
    result = AgyClient('gpt-oss-120b-medium', executable=str(executable)).edit_file(target,'Write')
    assert not result.ok and target.read_text() == 'original'
    assert result.error == 'authentication required'


def test_wrong_target_is_preserved_as_evidence_and_never_promoted(tmp_path):
    executable = tmp_path / 'agy'
    executable.write_text(f'#!{sys.executable}\n' + '''
import json,sys
from pathlib import Path
prompt=json.loads(sys.stdin.readline())['message']['content']
assert prompt.startswith('Only edit the existing debug.md in the current working directory.')
assert str(Path.cwd()) not in prompt
Path('attempt_01').mkdir()
Path('attempt_01/debug.md').write_text('actual model reflection')
print(json.dumps({'event':'result','result':{'status':'SUCCESS','response':'written'}}))
''')
    executable.chmod(0o755)
    target = tmp_path / 'debug.md'
    target.write_text('')
    result = AgyClient('claude-sonnet-4-6', executable=str(executable)).edit_file(target, 'Reflect', step='reflection')
    assert not result.ok and result.returncode == 0 and target.read_text() == ''
    record = json.loads((tmp_path / 'reflection_delivery_failure.json').read_text())
    assert record['required_target_empty'] and record['misplaced_files'][0]['relative_path'] == 'attempt_01/debug.md'
    assert (tmp_path / record['misplaced_files'][0]['archive']).read_text() == 'actual model reflection'


def test_quota_with_an_alternative_file_is_not_a_wrong_target_outcome(tmp_path):
    executable = tmp_path / 'agy'
    executable.write_text(f'#!{sys.executable}\n' + '''
import json,sys
from pathlib import Path
sys.stdin.read()
Path('attempt_01').mkdir()
Path('attempt_01/debug.md').write_text('partial')
print(json.dumps({'event':'result','result':{'status':'ERROR','error':'Individual quota reached'}}))
sys.exit(3)
''')
    executable.chmod(0o755)
    result = AgyClient('claude-sonnet-4-6', executable=str(executable)).edit_file(tmp_path / 'debug.md', 'Reflect', step='reflection')
    assert not result.ok and result.returncode == 3
    assert not (tmp_path / 'reflection_delivery_failure.json').exists()
