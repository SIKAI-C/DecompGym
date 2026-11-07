"""How data.json reaches the agent's generation and reflection context (a recorded prompt-protocol setting).

Without a `data_context` key in the campaign's prompt_protocol (the paper and every older config), the context holds the
full data.json, exactly as before. With DATA_CONTEXT_PREVIEW, a data.json larger than PREVIEW_THRESHOLD_BYTES is
replaced in the context by its structural preview (src.experiments.xl_inputs.preview_json, keep = PREVIEW_KEEP: the
first items of every list and the true list length), written next to the case's data.json as PREVIEW_NAME and clearly
marked as a preview. The solver still reads the real, full data.json at run time; sandbox, verifier, scoring and judge
are unchanged.

The prompt protocol may instead carry `context_delivery` = CONTEXT_DELIVERY_MOUNTED (src.agent.prompts.mounted): the
context files are mounted read-only in the agent workspace rather than pasted. The reviewed protocols keep the two
settings exclusive (accepted_prompt_protocol). Representation and transport are independent parts of the runner's
ContextSpec (src.experiments.context_spec), which can also mount the preview; DATA_CONTEXT_BOTH (the full data.json and,
above the threshold, its preview as well) is such a representation, used by no config.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

from src.agent.prompts.method import CONTRACTS
from src.agent.prompts.mounted import CONTEXT_DELIVERY_MOUNTED, JUDGE_CONTEXT_DELIVERY
from src.agent.prompts.snapshot import PROMPT_PROTOCOL
from src.agent.specs import PREVIEW_ALONGSIDE_NAME, PREVIEW_NAME
from src.experiments.xl_inputs import preview_json
from src.utils.errors import ProtocolError


DATA_CONTEXT_PREVIEW = "full-or-structural-preview-512KiB-v1"
PREVIEW_THRESHOLD_BYTES = 512 * 1024
PREVIEW_KEEP = 3
# The reviewed data-context settings of a prompt protocol.
DATA_CONTEXT_POLICIES = (None, DATA_CONTEXT_PREVIEW)
DATA_CONTEXT_BOTH = "full-and-structural-preview-512KiB-v1"


def accepted_prompt_protocol(protocol) -> bool:
    """The reviewed prompt protocol, optionally with either the recorded data-context setting or the mounted-files
    context delivery (itself optionally with the mounted method judge, judge_context_delivery), and a registered
    decomposition contract other than the paper's (src.agent.prompts.method.CONTRACTS)."""
    contract = protocol.get("decomposition_contract") if isinstance(protocol, dict) else None
    if not isinstance(contract, str) or contract not in CONTRACTS:
        return False
    reviewed = {**PROMPT_PROTOCOL, "decomposition_contract": protocol["decomposition_contract"]}
    mounted = {**reviewed, "context_delivery": CONTEXT_DELIVERY_MOUNTED}
    return protocol in (reviewed, {**reviewed, "data_context": DATA_CONTEXT_PREVIEW}, mounted, {**mounted, "judge_context_delivery": JUDGE_CONTEXT_DELIVERY})


def decomposition_contract_policy(protocol) -> str:
    """The contract id that the instance prompts, reflection and method judge of this protocol use."""
    if not accepted_prompt_protocol(protocol):
        raise ProtocolError("Prompt protocol differs from the reviewed contracts")
    return protocol["decomposition_contract"]


def data_context_policy(protocol) -> str | None:
    if not accepted_prompt_protocol(protocol):
        raise ProtocolError("Prompt protocol differs from the reviewed contracts")
    return protocol.get("data_context")


def context_delivery_policy(protocol) -> str | None:
    """None: the paper's pasted context; CONTEXT_DELIVERY_MOUNTED: read-only files in the agent workspace."""
    if not accepted_prompt_protocol(protocol):
        raise ProtocolError("Prompt protocol differs from the reviewed contracts")
    return protocol.get("context_delivery")


def judge_context_delivery_policy(protocol) -> str | None:
    """None: the paper's pasted judge context; JUDGE_CONTEXT_DELIVERY: the sources under review mounted read-only."""
    if not accepted_prompt_protocol(protocol):
        raise ProtocolError("Prompt protocol differs from the reviewed contracts")
    return protocol.get("judge_context_delivery")


def preview_text(data: Path, *, with_full: bool = False) -> str:
    size = data.stat().st_size
    if with_full:
        header = (f"STRUCTURAL PREVIEW of data.json, not the data itself: the file has {size} bytes. Every list shows only its first {PREVIEW_KEEP} items, followed by "
                  f"\"... (N items in total)\". The complete data.json is provided as well; solver.py must load the complete data.json at run time.")
    else:
        header = (f"STRUCTURAL PREVIEW of data.json, not the data itself: the file has {size} bytes, more than the "
                  f"{PREVIEW_THRESHOLD_BYTES}-byte context limit. Every list shows only its first {PREVIEW_KEEP} items, followed by "
                  f"\"... (N items in total)\". solver.py must load the complete data.json at run time.")
    return header + "\n" + preview_json(data, keep=PREVIEW_KEEP) + "\n"


def publish_preview(instance: Path, text: str, name: str = PREVIEW_NAME) -> Path:
    """The case's preview file `name` with exactly `text` (written once, atomically; a different saved preview is refused)."""
    preview = instance / name
    if preview.is_symlink() or (preview.exists() and preview.read_text() != text):
        raise ProtocolError("Saved structural preview differs from data.json")
    if not preview.exists():
        # Atomic publication: an interrupted write leaves only a temporary file, never a partial preview.
        handle, temporary = tempfile.mkstemp(dir=instance, prefix=f".{name}.", suffix=".tmp")
        try:
            with os.fdopen(handle, "w") as stream:
                stream.write(text)
            os.replace(temporary, preview)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return preview


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
    preview = publish_preview(instance, preview_text(data))
    return preview, {"policy": policy, "data_json_bytes": size, "delivered": "structural_preview", "preview": PREVIEW_NAME, "keep": PREVIEW_KEEP}


def data_context_files(instance: Path, policy: str | None) -> tuple[list[Path], dict | None]:
    """The instance-data entries of a context under `policy` (data_context_file's one entry, or data.json plus its
    preview under DATA_CONTEXT_BOTH) and the evidence record."""
    if policy != DATA_CONTEXT_BOTH:
        path, record = data_context_file(instance, policy)
        return [path], record
    data = instance / "data.json"
    size = data.stat().st_size
    if size <= PREVIEW_THRESHOLD_BYTES:
        return [data], {"policy": policy, "data_json_bytes": size, "delivered": "full"}
    preview = publish_preview(instance, preview_text(data, with_full=True), PREVIEW_ALONGSIDE_NAME)
    return [data, preview], {"policy": policy, "data_json_bytes": size, "delivered": "full_and_structural_preview", "preview": PREVIEW_ALONGSIDE_NAME, "keep": PREVIEW_KEEP}
