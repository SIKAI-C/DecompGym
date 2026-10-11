import json
from pathlib import Path
import sys

import pytest

from src.agent.claude_client import ClaudeClient


def fake_claude(tmp_path, body):
    executable = tmp_path / "claude"
    executable.write_text(f"#!{sys.executable}\nimport sys,json\nfrom pathlib import Path\n" + body)
    executable.chmod(0o755)
    return executable


@pytest.mark.parametrize("filename", ["solver.py", "debug.md"])
def test_claude_uses_pinned_model_existing_target_and_isolated_tools(tmp_path, filename):
    executable = fake_claude(tmp_path, f'''
prompt = sys.stdin.read()
assert prompt.startswith('Only edit the existing {filename} in the current working directory.')
assert Path('{filename}').is_file() and 'TASK_DATA' in prompt and 'ORACLE_SECRET' not in prompt
assert '--restricted' in sys.argv and '--safe-mode' in sys.argv and '--no-session-persistence' in sys.argv
assert '--dangerously-skip-permissions' not in sys.argv and '--bare' not in sys.argv
assert sys.argv[sys.argv.index('--tools')+1] == 'Read,Edit,Write'
assert sys.argv[sys.argv.index('--model')+1] == 'claude-sonnet-5-5'
assert sys.argv[sys.argv.index('--effort')+1] == 'high'
Path('{filename}').write_text('delivered')
Path('unrequested.txt').write_text('not copied')
print(json.dumps({{'type':'result','subtype':'success','is_error':False,'modelUsage':{{'claude-sonnet-5-5':{{}}}},'usage':{{'input_tokens':3,'cache_read_input_tokens':2}}}}))
''')
    (tmp_path / "gold_monolithic.json").write_text("ORACLE_SECRET")
    data = tmp_path / "data.json"
    data.write_text('"TASK_DATA"')
    result = ClaudeClient("claude-sonnet-5-5", executable=str(executable)).edit_file(tmp_path / filename, "Edit.", context=[data])
    assert result.ok and result.usage["cached_input_tokens"] == 2
    assert (tmp_path / filename).read_text() == "delivered"
    assert not (tmp_path / "unrequested.txt").exists()


@pytest.mark.parametrize("failure", ["quota", "wrong_model", "wrong_file"])
def test_claude_does_not_promote_failed_or_mismatched_output(tmp_path, failure):
    body = "sys.stdin.read()\n"
    if failure == "wrong_file":
        body += "Path('nested').mkdir()\nPath('nested/debug.md').write_text('misplaced reflection')\n"
    else:
        body += "Path('debug.md').write_text('partial')\n"
    event = {"type": "result", "subtype": "success", "is_error": False, "modelUsage": {"claude-opus-5-5": {}}}
    if failure == "quota":
        event.update(subtype="error_during_execution", is_error=True, errors=["quota reached"])
    elif failure == "wrong_model":
        event["modelUsage"] = {"claude-sonnet-5-5": {}}
    body += "print(" + repr(json.dumps(event)) + ")\n"
    executable = fake_claude(tmp_path, body)
    target = tmp_path / "debug.md"
    target.write_text("")
    result = ClaudeClient("claude-opus-5-5", executable=str(executable)).edit_file(target, "Reflect", step="reflection")
    assert not result.ok and target.read_text() == ""
    assert (tmp_path / "reflection_delivery_failure.json").exists() == (failure == "wrong_file")
    if failure == "wrong_model":
        assert result.terminal_state == "infrastructure_failed"
