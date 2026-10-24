"""Prompt examples are exported only from files that still match the hashes their campaign recorded."""
import hashlib
import json

import pytest

from src.results.prompt_excerpts import committed


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def test_saved_files_are_checked_against_their_recorded_hashes(tmp_path):
    (tmp_path / "agent_prompt.md").write_text("PROMPT")
    (tmp_path / "agent_workspace.json").write_text(json.dumps({"effective_prompt_sha256": sha("PROMPT")}))
    (tmp_path / "debug.md").write_text("ADVICE")
    (tmp_path / "reflection_receipt.json").write_text(json.dumps({"result": {"artifact_sha256": sha("ADVICE")}}))
    (tmp_path / "alg_verifier.md").write_text("VERDICT: FAIL")
    (tmp_path / "score_receipt.json").write_text(json.dumps({"artifact_sha256": {"alg_verifier.md": sha("VERDICT: FAIL")}}))
    assert [committed(tmp_path, name) for name in ("agent_prompt.md", "debug.md", "alg_verifier.md")] == ["PROMPT", "ADVICE", "VERDICT: FAIL"]
    for name in ("agent_prompt.md", "debug.md", "alg_verifier.md"):
        (tmp_path / name).write_text("edited")
        with pytest.raises(ValueError, match="differs from the hash"):
            committed(tmp_path, name)
