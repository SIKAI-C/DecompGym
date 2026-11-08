"""Which mounted files did each agent call open? A read-only report over campaign roots (mounted-files protocol).

    python scripts/file_access_report.py outputs/experiments/<root> [<root> ...] [--output DIR]

For every physical call with a mounted context (`<step>_context.json` next to `<step>_events.jsonl` in a call
directory): the steps are generation (`agent`), reflection and the method judge (`llm_verifier`). A mounted file
counts as "opened" when a completed tool call names it:
- Codex: an `item.completed` `command_execution` command;
- Claude Code: a `tool_use` input (Read, Bash, ...) whose `tool_result` (matched by tool_use_id) is not an error;
- AGY: events of the same two shapes, if present.
It must be named as one token: its full path, or `<directory>/<name>` (agents often `cd /work` and use relative paths
such as `history/attempt_02/debug.md`), after shell brace expansion (`history/attempt_{01..07}/debug.md`); an
`attempt_*` glob over a history file name counts for every attempt's copy. A file name and a directory mentioned
separately never count.

"Opened" is an estimate, not verified read coverage: reads through other globs, shell variables or a bare name after
`cd` into the file's directory are missed, and naming a file is not proof of reading all of it. A call without usable
events is counted under `no_events` and never as "not opened".

Writes calls.csv (one row per call and mounted file), summary.csv (by model, step and file kind) and history.csv
(reflection-policy regenerations by number of earlier attempts) to --output, default
outputs/reports/file_access_<UTC time>/. Nothing in the campaign roots is changed.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]
ATTEMPT = re.compile(r"attempt_(\d+)")


def tool_inputs(events: str) -> list[str] | None:
    """The completed tool-call inputs of an event stream; None when it holds no usable event object. A Claude Code
    tool_use counts only once a tool_result with its id reports no error; an Antigravity tool step once it is DONE
    without an error."""
    usable, inputs, requested, succeeded = False, [], {}, set()
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
            inputs.append(item["command"])
        message = event.get("message")
        for content in (message.get("content") or []) if isinstance(message, dict) and isinstance(message.get("content"), list) else []:
            if isinstance(content, dict) and content.get("type") == "tool_use" and isinstance(content.get("id"), str):
                requested[content["id"]] = json.dumps(content.get("input"))
            elif isinstance(content, dict) and content.get("type") == "tool_result" and not content.get("is_error") and isinstance(content.get("tool_use_id"), str):
                succeeded.add(content["tool_use_id"])
        # Antigravity: a tool step (step_update, step_type "tool") counts once it is DONE without an error.
        step = event.get("step_update")
        if event.get("event") == "step_update" and isinstance(step, dict) and step.get("step_type") == "tool" and step.get("state") == "DONE" and not step.get("error"):
            info = step.get("tool_info")
            inputs.append(json.dumps(info.get("parameters") if isinstance(info, dict) else None))
    inputs += [text for identifier, text in requested.items() if identifier in succeeded]
    return inputs if usable else None


BRACE = re.compile(r"\{([^{}]*)\}")
RANGE = re.compile(r"^(\d+)\.\.(\d+)$")


def expand_braces(text: str, limit: int = 256) -> list[str]:
    """Bash brace expansion of {a,b,...} and {01..07} (zero padding kept), as agents use in shell loops over files. At
    most `limit` results: a range is never materialized beyond the remaining budget, and each nested expansion gets
    only what is left of it."""
    if limit <= 0:
        return []
    match = BRACE.search(text)
    if not match:
        return [text]
    body = match[1]
    numeric = RANGE.match(body)
    if numeric:
        start, end, width = int(numeric[1]), int(numeric[2]), len(numeric[1]) if numeric[1].startswith("0") else 0
        step = 1 if end >= start else -1
        options = (str(start + step * index).zfill(width) for index in range(min(abs(end - start) + 1, limit)))
    elif "," in body:
        options = body.split(",", limit)[:limit]
    else:
        return [text]  # a literal brace, for example a Python dict or set
    expanded = []
    for option in options:
        expanded += expand_braces(text[:match.start()] + option + text[match.end():], limit - len(expanded))
        if len(expanded) >= limit:
            break
    return expanded[:limit]


def token(text: str) -> re.Pattern:
    """`text` as one path token: not inside a longer name (attempt_01/x.md never matches attempt_011/x.md or x.md.bak)."""
    return re.compile(r"(?<![\w.-])" + text + r"(?![\w.-])")


def opened(path: str, inputs: list[str]) -> bool:
    """Named as one token: the full path, `<directory>/<name>`, or an attempt_* glob over the file's name (history)."""
    name, directory = Path(path).name, Path(path).parent.name
    patterns = [token(re.escape(path)), token(re.escape(f"{directory}/{name}"))]
    if "/history/" in path:
        patterns.append(token(r"attempt_(\*|\?\?)/+" + re.escape(name)))
    for text in inputs:
        for variant in expand_braces(text):
            if any(pattern.search(variant) for pattern in patterns):
                return True
    return False


def kind(path: str) -> str:
    if path.endswith("/instance/data.json"):
        return "data.json"
    return ("review " if "/review/" in path else "history ") + Path(path).name


def scan(root: Path):
    for context in sorted(root.glob("runs/*/*/*/*/*/attempts/attempt_*/calls/*/*_context.json")):
        step = context.name[: -len("_context.json")]
        relative = context.relative_to(root / "runs").parts
        model, mode, suite, problem, instance, attempt = relative[0], relative[1], relative[2], relative[3], relative[4], relative[6]
        events = context.with_name(f"{step}_events.jsonl")
        inputs = tool_inputs(events.read_text(errors="replace")) if events.exists() else None
        files = json.loads(context.read_text())["files"]
        history = sorted({int(m[1]) for entry in files for m in [ATTEMPT.search(entry["path"])] if "/history/" in entry["path"] and m})
        for entry in files:
            match = ATTEMPT.search(entry["path"]) if "/history/" in entry["path"] else None
            yield {"root": root.name, "model": model, "mode": mode, "suite": suite, "problem": problem, "instance": instance, "attempt": int(attempt.split("_")[1]),
                   "step": step, "invocation_id": context.parent.name, "file": entry["path"], "file_kind": kind(entry["path"]),
                   "history_attempt": int(match[1]) if match else "", "earlier_attempts": len(history), "latest_history": bool(match and int(match[1]) == max(history)),
                   "events": inputs is not None, "opened": "" if inputs is None else opened(entry["path"], inputs)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or REPO / "outputs/reports" / ("file_access_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(parents=True, exist_ok=True)
    rows = [row for root in args.roots for row in scan(root.resolve())]
    if not rows:
        raise SystemExit("No mounted-context calls found")
    with (output / "calls.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = defaultdict(lambda: {"mounted": 0, "opened": 0, "no_events": 0})
    for row in rows:
        cell = summary[(row["model"], row["step"], row["file_kind"])]
        cell["mounted"] += 1
        cell["no_events"] += not row["events"]
        cell["opened"] += row["opened"] is True
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "step", "file_kind", "mounted", "opened", "no_events", "opened_share_of_calls_with_events"])
        for (model, step, file_kind), cell in sorted(summary.items()):
            with_events = cell["mounted"] - cell["no_events"]
            writer.writerow([model, step, file_kind, cell["mounted"], cell["opened"], cell["no_events"], round(cell["opened"] / with_events, 4) if with_events else ""])

    # Reflection-policy regenerations: per call, was the latest debug.md opened, and was every earlier debug.md opened?
    calls = defaultdict(list)
    for row in rows:
        if row["step"] == "agent" and row["mode"] == "reflection" and row["file_kind"] == "history debug.md" and row["events"]:
            calls[(row["root"], row["model"], row["invocation_id"])].append(row)
    history = defaultdict(lambda: {"calls": 0, "latest_opened": 0, "all_opened": 0})
    for (_, model, _), notes in calls.items():
        cell = history[(model, notes[0]["earlier_attempts"])]
        cell["calls"] += 1
        cell["latest_opened"] += any(row["opened"] for row in notes if row["latest_history"])
        cell["all_opened"] += all(row["opened"] for row in notes)
    with (output / "history.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "earlier_attempts", "calls", "latest_debug_opened", "every_debug_opened"])
        for (model, earlier), cell in sorted(history.items()):
            writer.writerow([model, earlier, cell["calls"], cell["latest_opened"], cell["all_opened"]])
    print(f"{len(rows)} mounted files in {len({(r['root'], r['invocation_id']) for r in rows})} calls -> {output}")


if __name__ == "__main__":
    main()
