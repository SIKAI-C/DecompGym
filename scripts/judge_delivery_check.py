"""Real-call smoke check of the method judge's mounted delivery (prompt_protocol.judge_context_delivery).

Run it only after the Docker runtime evidence and the Codex login exist:
    python src/02_experiment.py docker prepare
    python src/02_experiment.py docker auth --backend codex --check
    python scripts/judge_delivery_check.py

Four self-contained source fixtures (scripts/judge_fixtures; never executed): a Benders UFL solver and the same solver
with a reachable full-model solve whose result is reported (bd), and a column-generation bin-packing solver and the
same solver with a final compact full model (cg). Each is judged by the campaign's judge (configs/small_scale.json:
judge_model, judge_effort, codex-cli, the frozen image, the v3 review text of algorithm_review_prompt) twice:
- pasted: the paper's transport (the source follows the prompt), with the same v3 review text;
- mounted: the source mounted read-only at /work/review/attempt_01/solver.py (REVIEW_LAYOUT).
That is 8 judge calls; no solver or optimizer runs. Each fixture copy carries a random comment token, so the check can
confirm that the mounted prompt pastes no source. Expected verdicts: PASS for the decompositions, FAIL for the
monolithic variants, under both transports. A smoke check only, not evidence of verdict equivalence across a campaign.
The diagnostic `review_path_in_completed_tool_input` (record only) tells whether a completed Codex command named the
mounted path.

Evidence goes to outputs/diagnostics/judge_delivery_check_<UTC time>/ (summary.json). Exit code 0 only if every call
returns its expected verdict and every mounted call's evidence is consistent.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.agent.docker_client import DockerCliClient  # noqa: E402
from src.agent.mounted_context import mounted_files  # noqa: E402
from src.agent.prompts.judge import algorithm_review_prompt, parse_verdict  # noqa: E402
from src.agent.prompts.mounted import CONTEXT_DELIVERY_MOUNTED  # noqa: E402
from src.agent.specs import REVIEW_LAYOUT  # noqa: E402
from src.experiments import agent_auth, docker_setup  # noqa: E402

CONFIG = REPO / "configs/small_scale.json"
FIXTURES = REPO / "scripts/judge_fixtures"
CASES = [("bd", "bd_ufl_benders.py", "pass"), ("bd", "bd_ufl_monolithic.py", "fail"), ("cg", "cg_bp_column_generation.py", "pass"), ("cg", "cg_bp_monolithic.py", "fail")]
TRANSPORTS = ("pasted", "mounted")


def mentioned(events: str, path: str) -> bool | None:
    """Whether a completed Codex command names `path`; None when the stream holds no usable event object (no evidence)."""
    usable, commands = False, []
    for line in events.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        usable = True
        item = event.get("item")
        if event.get("type") == "item.completed" and isinstance(item, dict) and item.get("type") == "command_execution" and isinstance(item.get("command"), str):
            commands.append(item["command"])
    return any(path in command for command in commands) if usable else None


def marked_source(fixture: str) -> tuple[str, str]:
    """(token, source text): the fixture with one random comment line, used verbatim for both transports."""
    token = secrets.token_hex(8)
    return token, f"# fixture {token}\n" + (FIXTURES / fixture).read_text()


def judge_call(engine, image_id, plan, suite, marked, transport, directory: Path) -> dict:
    token, text = marked
    attempt = directory / "case/attempts/attempt_01"
    attempt.mkdir(parents=True)
    source = attempt / "solver.py"
    source.write_text(text)
    target = attempt / "alg_verifier.md"
    target.write_text("")
    contract = plan["prompt_protocol"]["decomposition_contract"]
    mounted = {"context_delivery": CONTEXT_DELIVERY_MOUNTED, "context_layout": REVIEW_LAYOUT} if transport == "mounted" else {}
    execution = plan["execution"]
    client = DockerCliClient(execution["judge_model"], backend="codex-cli", image_id=image_id, runtime=engine, evidence_root=directory.parent / "preflight", role="algorithm_judge",
                             effort=execution["judge_effort"], timeout=execution["agent_timeout"], **mounted)
    result = client.edit_file(target, algorithm_review_prompt(suite, contract), context=[source], log_dir=directory / "call", step="llm_verifier")
    notes = target.read_text()
    record = {"suite": suite, "transport": transport, "edit_ok": result.ok, "terminal_state": result.terminal_state, "error": result.error,
              "verdict": parse_verdict(notes), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "seconds": result.elapsed_seconds, "usage": result.usage}
    problems = []
    prompt = (directory / "call/llm_verifier_prompt.md").read_text()
    if transport == "mounted":
        evidence = json.loads((directory / "call/llm_verifier_context.json").read_text())
        expected = [(item.path, item.sha256) for item in mounted_files([source], REVIEW_LAYOUT)]
        if [(entry["path"], entry["sha256"]) for entry in evidence["files"]] != expected or evidence.get("layout") != REVIEW_LAYOUT.id:
            problems.append("the mounted-file evidence differs from the reviewed source")
        if token in prompt or "BEGIN CONTEXT" in prompt:
            problems.append("the mounted prompt pastes the source")
        events = directory / "call/llm_verifier_events.jsonl"
        record["review_path_in_completed_tool_input"] = mentioned(events.read_text() if events.exists() else "", "/work/review/attempt_01/solver.py")
    elif token not in prompt:
        problems.append("the pasted prompt lacks the source")
    record["evidence_problems"] = problems
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runtime", type=Path, default=REPO / "outputs/runtime/runtime.json", help="runtime.json written by `docker prepare`")
    args = parser.parse_args(argv)
    plan = json.loads(CONFIG.read_text())
    environment = docker_setup.load_execution_environment(args.runtime)
    image_id = environment["image_id"]
    engine = docker_setup.runtime()
    engine.bind_environment(environment)
    if engine.image_id(image_id) != image_id:
        raise SystemExit("The prepared Docker image is unavailable")
    execution = plan["execution"]
    agent_auth.raise_for(agent_auth.check_backend(engine, image_id, "codex-cli", slots=[0], models=[execution["judge_model"]]))
    output = REPO / "outputs/diagnostics" / ("judge_delivery_check_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(parents=True)
    summary = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "image_id": image_id, "judge_model": execution["judge_model"], "judge_effort": execution["judge_effort"],
               "contract": plan["prompt_protocol"]["decomposition_contract"], "fixtures_sha256": {name: hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() for _, name, _ in CASES}, "calls": []}

    def save():
        (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    for suite, fixture, expected in CASES:
        marked = marked_source(fixture)  # the same bytes for both transports
        for transport in TRANSPORTS:
            try:
                record = judge_call(engine, image_id, plan, suite, marked, transport, output / f"{Path(fixture).stem}_{transport}")
            except Exception as error:  # record and continue; each (fixture, transport) gets one call at most
                record = {"suite": suite, "transport": transport, "error": f"{type(error).__name__}: {error}", "verdict": None, "evidence_problems": ["call failed"]}
            record.update(fixture=fixture, expected=expected, marked_source_sha256=hashlib.sha256(marked[1].encode()).hexdigest())
            if record.get("source_sha256") not in (None, record["marked_source_sha256"]):
                record["evidence_problems"].append("the judged source differs from the marked fixture")
            record["passed"] = bool(record.get("edit_ok") and record["verdict"] == expected and not record["evidence_problems"])
            summary["calls"].append(record)
            save()
    summary["passed"] = len(summary["calls"]) == len(CASES) * len(TRANSPORTS) and all(call["passed"] for call in summary["calls"])
    save()
    print(json.dumps({"passed": summary["passed"], "calls": [(call["fixture"], call["transport"], call["verdict"], call["passed"]) for call in summary["calls"]], "summary": str(output / "summary.json")}, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    os.umask(0o022)
    main()
