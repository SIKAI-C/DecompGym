"""Agent-facing input preview for scale-up instances (the full data.json can exceed the prompt transport).

`preview_json` keeps the structure of a JSON document but shows only the first `keep` items of every list,
followed by a marker with the true length, so the agent learns the format and the sizes and writes a solver
that reads the real file at run time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def preview(value: Any, keep: int = 3) -> Any:
    if isinstance(value, dict):
        return {k: preview(v, keep) for k, v in value.items()}
    if isinstance(value, list):
        shown = [preview(v, keep) for v in value[:keep]]
        return shown + [f"... ({len(value)} items in total)"] if len(value) > keep else shown
    return value


def preview_json(path: str | Path, keep: int = 3) -> str:
    return json.dumps(preview(json.loads(Path(path).read_text()), keep))
