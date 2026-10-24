"""The result record of one agent CLI call, saved next to its call evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class AgentResult:
    ok: bool
    returncode: int | None
    elapsed_seconds: float
    usage: dict[str, int]
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    terminal_state: str | None = None
    artifact_sha256: str | None = None


def save_result(log_dir: Path, step: str, result: AgentResult) -> None:
    path = log_dir / f"{step}_metrics.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(result), indent=2))
    temporary.replace(path)
