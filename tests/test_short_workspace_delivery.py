"""New wording has its own closure authorization; historical v1 stays independent."""
from pathlib import Path
import sys

import pytest

from src.agent.agy_client import AgyClient
from src.agent.paired_runner import PairedRunner, ProtocolError, ReflectionDeliveryFailure, digest, identity, read, save
from src.agent.policy_completion import DELIVERY_AMENDMENT, validate_delivery_call
from src.agent.prompt_policy import FEEDBACK_POLICY, WORKSPACE_POLICY


def fresh_call(tmp_path):
    root = tmp_path / "campaign"
    source = Path(__file__).resolve().parents[1] / "src/agent"
    hashes = {}
    for name in ("prompt_policy.py", "policy_completion.py", "paired_runner.py", "agy_client.py"):
        target = root / "controller/src/agent" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source / name).read_bytes())
        hashes[str(target.relative_to(root / "controller"))] = digest(target)
    model = "gemini-3.1-pro-high"
    manifest = {"prompt_policies": {"feedback": FEEDBACK_POLICY, "workspace": WORKSPACE_POLICY},
                "wrong_target_reflection_failure": DELIVERY_AMENDMENT, "model_queue": [{"model": model, "backend": "agy-cli"}], "controller_sha256": hashes}
    manifest["protocol_id"] = identity(manifest)
    save(root / "campaign.json", manifest)
    attempt = root / "runs" / model / "reflection/bd/ufl/ufl_small_seed000/attempts/attempt_01"
    attempt.mkdir(parents=True)
    executable = tmp_path / "agy"
    executable.write_text(f"#!{sys.executable}\n" + '''
import json,sys
from pathlib import Path
prompt=json.loads(sys.stdin.readline())['message']['content']
assert prompt.startswith('Only edit the existing debug.md in the current working directory.')
Path('nested').mkdir()
Path('nested/debug.md').write_text('A reflection at the wrong location.')
print(json.dumps({'event':'result','result':{'status':'SUCCESS'}}))
''')
    executable.chmod(0o755)
    client = AgyClient(model, executable=str(executable), role="reflection")
    task = PairedRunner(client=client, judge_client=client, model=model, problem=["ufl"], protocol_id=manifest["protocol_id"])
    return root, attempt, task, client


def test_short_wording_wrong_target_is_committed_once_and_not_retried(tmp_path):
    root, attempt, task, client = fresh_call(tmp_path)
    for _ in range(2):
        with pytest.raises(ReflectionDeliveryFailure):
            task.edit(client, attempt / "debug.md", "Reflect.", [], "reflection")
    calls = list((attempt / "calls").iterdir())
    assert len(calls) == 1 and (attempt / "debug.md").read_text() == ""
    authorization = validate_delivery_call(calls[0], root)
    assert authorization["protocol_id"] == read(root / "campaign.json")["protocol_id"]
    assert read(attempt / "reflection_receipt.json")["result"]["file_delivery_failure"]


@pytest.mark.parametrize("changed", ["authorization", "prompt", "artifact", "controller"])
def test_short_wording_rejects_tampering(tmp_path, changed):
    root, attempt, task, client = fresh_call(tmp_path)
    with pytest.raises(ReflectionDeliveryFailure):
        task.edit(client, attempt / "debug.md", "Reflect.", [], "reflection")
    call = next((attempt / "calls").iterdir())
    if changed == "authorization":
        manifest = read(root / "campaign.json")
        manifest["wrong_target_reflection_failure"] = "not_authorized"
        save(root / "campaign.json", manifest)
    elif changed == "prompt":
        (call / "reflection_prompt.md").write_text("Changed")
    elif changed == "artifact":
        (call / "unexpected_outputs/nested/debug.md").write_text("Changed")
    else:
        (root / "controller/src/agent/agy_client.py").write_text("Changed")
    with pytest.raises(ProtocolError):
        task.edit(client, attempt / "debug.md", "Reflect.", [], "reflection")
