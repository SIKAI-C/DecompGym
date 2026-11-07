"""Real-call check of the mounted-files context delivery (prompt_protocol.context_delivery = "mounted-files-v1").

Run it only after the Docker runtime evidence and the private logins exist:
    python src/02_experiment.py docker prepare
    python src/02_experiment.py docker auth --backend codex|claude|agy      # AGY also needs DECOMPBENCH_AGY_KEYRING_PASSPHRASE
    python scripts/context_delivery_check.py                                 # all three backends
    python scripts/context_delivery_check.py --backends codex --mount-probe-only

1. Mount probe (no model call): the image's real worker runs a fake CLI (the pattern of
   tests/test_docker_worker_integration.py) in a workspace with the mounted layout. It must read the mounted
   /work/instance/data.json and /work/history/attempt_01/debug.md, fail to write to or create anything under them,
   and its target /work/solver.py must be the only collected file.
2. Exactly one real generation call per selected backend through DockerCliClient(context_delivery="mounted-files-v1")
   on one small instance of the combined dataset. The call asks the agent to write into /work/solver.py one Python
   assignment REPORT = {...} with the first key of data.json and its value, a random token from the history file
   debug.md and the objective_value of the history file output.json. The check parses solver.py without executing it
   and compares the answers; it also checks that the random values were never pasted into the prompt and that the
   call evidence records every mounted file's sha256 and read-only mounts.

Evidence goes to outputs/diagnostics/context_delivery_check_<UTC time>/ (summary.json). No solver or optimizer runs.
Exit code 0 only if every selected step passes.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.agent.container import DockerRuntime  # noqa: E402
from src.agent.docker_client import DockerCliClient  # noqa: E402
from src.agent.mounted_context import mounted_files, volume_contents  # noqa: E402
from src.agent.prompts.mounted import CONTEXT_DELIVERY_MOUNTED, HISTORY_DIR, INSTANCE_DIR  # noqa: E402
from src.experiments import docker_setup  # noqa: E402
from src.utils.families import SMALL_SCALE_CAMPAIGN_DATA  # noqa: E402

BACKENDS = docker_setup.BACKENDS  # codex / claude / agy -> CLI backend
EFFORT = {"codex-cli": "medium", "claude-code-cli": "high", "agy-cli": "medium"}  # as `docker auth --verify`

PROBE_CLI = r"""import json
from pathlib import Path
checks = {}
checks['data_read'] = json.loads(Path('/work/instance/data.json').read_text())
checks['history_read'] = Path('/work/history/attempt_01/debug.md').read_text()
def denied(action):
    try:
        action()
        return False
    except OSError:
        return True
checks['instance_write_denied'] = denied(lambda: Path('/work/instance/data.json').write_text('changed'))
checks['instance_create_denied'] = denied(lambda: Path('/work/instance/extra.json').write_text('{}'))
checks['history_write_denied'] = denied(lambda: Path('/work/history/attempt_01/debug.md').write_text('changed'))
checks['history_create_denied'] = denied(lambda: Path('/work/history/attempt_02').mkdir())
Path('/work/elsewhere.py').write_text('print(0)\n')
checks['workspace_entries'] = sorted(path.name for path in Path('/work').iterdir())
Path('/work/solver.py').write_text(json.dumps(checks))
print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 0, 'output_tokens': 0}}), flush=True)
"""
PROBE_WORKER = "\n".join([
    "import os,sys",
    "from pathlib import Path",
    "sys.path.insert(0,'/opt/decompbench')",
    "import worker",
    "Path('/tmp/probe_cli.py').write_text(" + repr(PROBE_CLI) + ")",
    "worker.credentials=lambda backend,job:(os.environ.copy(),set(),lambda:None)",
    "worker.cli_command=lambda backend,job:['python3','/tmp/probe_cli.py']",
    "raise SystemExit(worker.main())",
])


class ProbeRuntime(DockerRuntime):
    """The real lifecycle and worker; only the CLI and the credential plane are replaced (no model, no network)."""

    def command(self, *args, **kwargs):
        args = list(args)
        if args[0] == "create" and self.WORKER in args:
            position = args.index(self.WORKER)
            args[position:position + 1] = ["-c", PROBE_WORKER]
        return super().command(*args, **kwargs)


def make_case(directory: Path, instance: Path) -> tuple[Path, list[Path], dict]:
    """A runner-shaped case: data.json, attempt_01 history files with random values, and the attempt_02 target."""
    case = directory / "case"
    history = case / "attempts/attempt_01"
    history.mkdir(parents=True)
    shutil.copyfile(instance / "data.json", case / "data.json")
    data = json.loads((case / "data.json").read_text())
    first_key, first_value = next(iter(data.items()))
    if isinstance(first_value, (dict, list)):
        raise SystemExit(f"{instance}: the first data.json value must be a scalar for this check")
    token, objective = secrets.token_hex(8), secrets.randbelow(10**9) + 0.5
    (history / "solver.py").write_text("print('attempt 1')\n")
    (history / "debug.md").write_text(f"Reflection after attempt 1.\nCheckpoint token: {token}\n")
    (history / "output.json").write_text(json.dumps({"status": "feasible", "method": "benders_decomposition", "objective_value": objective}))
    target = case / "attempts/attempt_02/solver.py"
    target.parent.mkdir(parents=True)
    target.write_text("")
    context = [case / "data.json", history / "solver.py", history / "debug.md", history / "output.json"]
    return target, context, {"first_key": first_key, "first_value": first_value, "token": token, "objective_value": objective}


TASK = """# Part 2: Task

## Diagnostic
This is a context-delivery diagnostic, not a benchmark task: do not write a solver. Read the mounted files listed in Part 1 and replace the contents of /work/solver.py with exactly one Python assignment and nothing else:
REPORT = {"first_key": <the name of the first top-level key of /work/instance/data.json, a string>, "first_value": <the value of that key, a Python literal>, "token": <the token after "Checkpoint token:" in /work/history/attempt_01/debug.md, a string>, "objective_value": <the objective_value field of /work/history/attempt_01/output.json, a number>}
Copy the values exactly from the files; do not guess."""


def report_of(text: str):
    """REPORT from a solver.py that holds exactly one assignment of a literal (never executed)."""
    tree = ast.parse(text)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign) or [getattr(t, "id", None) for t in tree.body[0].targets] != ["REPORT"]:
        raise ValueError("solver.py is not exactly one REPORT assignment")
    return ast.literal_eval(tree.body[0].value)


def same(left, right) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
        return math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-9)
    return left == right


def mounted_evidence(call: Path, context: list[Path], prompt_values) -> list[str]:
    """Problems with the recorded layout, sha256 values, read-only mounts and the prompt (empty list: all good)."""
    problems = []
    recorded = json.loads((call / "agent_context.json").read_text())
    expected = {item.path: item.sha256 for item in mounted_files(context)}
    if {entry["path"]: entry["sha256"] for entry in recorded["files"]} != expected or recorded["writable_target"] != "/work/solver.py":
        problems.append("agent_context.json differs from the mounted files")
    invocation = json.loads((call / "agent_invocation.json").read_text())
    if invocation.get("context_delivery") != CONTEXT_DELIVERY_MOUNTED:
        problems.append("invocation does not record the mounted-files delivery")
    volumes = invocation.get("context_volumes") or {}
    mounts = invocation.get("mounts") or []
    for destination in (INSTANCE_DIR, HISTORY_DIR):
        if not any(mount.get("Destination") == destination and mount.get("Name") == volumes.get(destination) and mount.get("RW") is False for mount in mounts):
            problems.append(f"{destination} was not mounted read-only")
    manifest = {f"{directory}/{name}": sha for directory, files in (invocation.get("context_manifest") or {}).items() for name, sha in files.items()}
    if manifest != expected:
        problems.append("container context manifest differs from the mounted files")
    prompt = (call / "agent_prompt.md").read_text()
    if any(str(value) in prompt for value in prompt_values) or "/work/instance/data.json" not in prompt or "/work/history/attempt_01/debug.md" not in prompt:
        problems.append("the prompt pastes a checked value or does not name the mounted paths")
    return problems


def mount_probe(engine: ProbeRuntime, image_id: str, directory: Path, instance: Path) -> dict:
    target, context, _ = make_case(directory, instance)
    files = mounted_files(context)
    result = engine.run(image_id, ["agent", "codex-cli"], payload={"target": "solver.py", "model": "probe", "role": "diagnostic", "effort": "medium", "prompt": "mount probe", "timeout": 60},
                        state_path=directory / "probe_container.json", timeout=90, network="none", artifacts=["solver.py", "worker.json", "delivery.json"],
                        context=volume_contents(files))
    checks = json.loads(result.artifacts.get("solver.py") or "{}")
    data = json.loads((context[0]).read_text())
    passed = {"data_read": checks.get("data_read") == data, "history_read": checks.get("history_read") == context[2].read_text(),
              **{name: checks.get(name) is True for name in ("instance_write_denied", "instance_create_denied", "history_write_denied", "history_create_denied")},
              "workspace_layout": checks.get("workspace_entries") == ["elsewhere.py", "history", "instance", "solver.py"],
              "only_target_collected": set(result.artifacts) <= {"solver.py", "worker.json", "delivery.json"},
              "returncode": result.returncode == 0}
    return {"passed": all(passed.values()), "checks": passed, "mounts": result.metadata.get("mounts"), "rejected_artifacts": result.metadata.get("rejected_artifacts")}


def model_call(backend: str, engine: DockerRuntime, image_id: str, evidence_root: Path, directory: Path, instance: Path) -> dict:
    target, context, expected = make_case(directory, instance)
    model = docker_setup.READINESS_MODELS[backend]
    client = DockerCliClient(model, backend=backend, image_id=image_id, runtime=engine, evidence_root=evidence_root, role="context-delivery-check", effort=EFFORT[backend],
                             context_delivery=CONTEXT_DELIVERY_MOUNTED)
    result = client.edit_file(target, TASK, context=context, log_dir=directory / "call", step="agent")
    record = {"backend": backend, "model": model, "edit_ok": result.ok, "terminal_state": result.terminal_state, "error": result.error, "expected": expected}
    try:
        answer = report_of(target.read_text())
        record["answer"] = answer
        record["answers_match"] = isinstance(answer, dict) and set(answer) == set(expected) and all(same(answer[key], expected[key]) for key in expected)
    except (SyntaxError, ValueError) as error:
        record.update(answer=None, answers_match=False, parse_error=str(error))
    record["evidence_problems"] = mounted_evidence(directory / "call", context, [expected["token"], expected["objective_value"]])
    record["passed"] = bool(result.ok and record["answers_match"] and not record["evidence_problems"])
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backends", nargs="+", choices=list(BACKENDS), default=list(BACKENDS))
    parser.add_argument("--instance", default="bd/ufl/ufl_small_seed000", help=f"instance under {SMALL_SCALE_CAMPAIGN_DATA}")
    parser.add_argument("--runtime", type=Path, default=REPO / "outputs/runtime/runtime.json", help="runtime.json written by `docker prepare`")
    parser.add_argument("--mount-probe-only", action="store_true", help="run only the no-model mount probe")
    args = parser.parse_args(argv)
    instance = REPO / SMALL_SCALE_CAMPAIGN_DATA / args.instance
    if not (instance / "data.json").is_file():
        raise SystemExit(f"{instance}/data.json not found")
    environment = docker_setup.load_execution_environment(args.runtime)
    image_id = environment["image_id"]
    output = REPO / "outputs/diagnostics" / ("context_delivery_check_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(parents=True)
    summary = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "image_id": image_id, "instance": args.instance, "context_delivery": CONTEXT_DELIVERY_MOUNTED,
               "data_json_sha256": hashlib.sha256((instance / "data.json").read_bytes()).hexdigest(), "steps": {}}

    def save():
        (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    probe_engine = ProbeRuntime(seccomp_profile=docker_setup.SECCOMP_PROFILE)
    probe_engine.bind_environment(environment)
    if probe_engine.image_id(image_id) != image_id:
        raise SystemExit("The prepared Docker image is unavailable")
    summary["steps"]["mount_probe"] = mount_probe(probe_engine, image_id, output / "mount_probe", instance)
    save()
    if summary["steps"]["mount_probe"]["passed"] and not args.mount_probe_only:
        engine = docker_setup.runtime()
        engine.bind_environment(environment)
        for name in args.backends:
            backend = BACKENDS[name]
            try:
                summary["steps"][backend] = model_call(backend, engine, image_id, output / "preflight", output / backend, instance)
            except Exception as error:  # record and continue with the next backend; each backend gets one call at most
                summary["steps"][backend] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
            save()
    summary["passed"] = all(step.get("passed") for step in summary["steps"].values()) and (args.mount_probe_only or len(summary["steps"]) == 1 + len(args.backends))
    save()
    print(json.dumps({key: summary[key] for key in ("passed", "image_id", "instance")} | {"steps": {name: step.get("passed") for name, step in summary["steps"].items()}, "summary": str(output / "summary.json")}, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    os.umask(0o022)
    main()
