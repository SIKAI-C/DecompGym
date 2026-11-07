"""How data.json reaches the agent's generation and reflection context (a recorded prompt-protocol setting).

Without a `data_context` key in the campaign's prompt_protocol (the paper and every older config), the context holds the
full data.json, exactly as before. With DATA_CONTEXT_PREVIEW, a data.json larger than PREVIEW_THRESHOLD_BYTES is
replaced in the context by its structural preview (src.experiments.xl_inputs.preview_json, keep = PREVIEW_KEEP: the
first items of every list and the true list length), written next to the case's data.json as PREVIEW_NAME and clearly
marked as a preview. The solver still reads the real, full data.json at run time; sandbox, verifier, scoring and judge
are unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

from src.agent.prompts.method import CONTRACTS
from src.agent.prompts.snapshot import PROMPT_PROTOCOL
from src.experiments.xl_inputs import preview_json
from src.utils.errors import ProtocolError


DATA_CONTEXT_PREVIEW = "full-or-structural-preview-512KiB-v1"
PREVIEW_THRESHOLD_BYTES = 512 * 1024
PREVIEW_KEEP = 3
PREVIEW_NAME = "data.json.structural-preview.txt"
DATA_CONTEXT_POLICIES = (None, DATA_CONTEXT_PREVIEW)


def accepted_prompt_protocol(protocol) -> bool:
    """The reviewed prompt protocol, optionally with the recorded data-context setting and a registered
    decomposition contract other than the paper's (src.agent.prompts.method.CONTRACTS)."""
    contract = protocol.get("decomposition_contract") if isinstance(protocol, dict) else None
    if not isinstance(contract, str) or contract not in CONTRACTS:
        return False
    reviewed = {**PROMPT_PROTOCOL, "decomposition_contract": protocol["decomposition_contract"]}
    return protocol == reviewed or protocol == {**reviewed, "data_context": DATA_CONTEXT_PREVIEW}


def decomposition_contract_policy(protocol) -> str:
    """The contract id that the instance prompts, reflection and method judge of this protocol use."""
    if not accepted_prompt_protocol(protocol):
        raise ProtocolError("Prompt protocol differs from the reviewed contracts")
    return protocol["decomposition_contract"]


def data_context_policy(protocol) -> str | None:
    if not accepted_prompt_protocol(protocol):
        raise ProtocolError("Prompt protocol differs from the reviewed contracts")
    return protocol.get("data_context")


def preview_text(data: Path) -> str:
    size = data.stat().st_size
    header = (f"STRUCTURAL PREVIEW of data.json, not the data itself: the file has {size} bytes, more than the "
              f"{PREVIEW_THRESHOLD_BYTES}-byte context limit. Every list shows only its first {PREVIEW_KEEP} items, followed by "
              f"\"... (N items in total)\". solver.py must load the complete data.json at run time.")
    return header + "\n" + preview_json(data, keep=PREVIEW_KEEP) + "\n"


def data_context_file(instance: Path, policy: str | None) -> tuple[Path, dict | None]:
    """The context path for the case's data.json and the evidence record (None under the legacy full-file rule)."""
    data = instance / "data.json"
    if policy is None:
        return data, None
    if policy != DATA_CONTEXT_PREVIEW:
        raise ProtocolError(f"Unknown data-context policy: {policy}")
    size = data.stat().st_size
    if size <= PREVIEW_THRESHOLD_BYTES:
        return data, {"policy": policy, "data_json_bytes": size, "delivered": "full"}
    preview = instance / PREVIEW_NAME
    text = preview_text(data)
    if preview.is_symlink() or (preview.exists() and preview.read_text() != text):
        raise ProtocolError("Saved structural preview differs from data.json")
    if not preview.exists():
        # Atomic publication: an interrupted write leaves only a temporary file, never a partial preview.
        handle, temporary = tempfile.mkstemp(dir=instance, prefix=f".{PREVIEW_NAME}.", suffix=".tmp")
        try:
            with os.fdopen(handle, "w") as stream:
                stream.write(text)
            os.replace(temporary, preview)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return preview, {"policy": policy, "data_json_bytes": size, "delivered": "structural_preview", "preview": PREVIEW_NAME, "keep": PREVIEW_KEEP}
