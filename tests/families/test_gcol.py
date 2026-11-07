"""Graph coloring (small scale): the shared verifier path rejects a wrong claimed objective (Codex review 4)."""
from __future__ import annotations

import json

from src.experiments.verification import OutputVerifier


def _write(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


def test_shared_verifier_rejects_a_wrong_claimed_objective(tmp_path):
    # Codex review 4, finding 3: a valid plan with a bogus objective must fail through the shared path.
    data = {"instance_size": "small", "n_vertices": 2, "edges": [[0, 1]]}
    output = {"status": "optimal", "method": "column_generation", "objective_value": -1234567, "solution": {"color_classes": [[0], [1]]}}
    result = OutputVerifier.verify_output("cg", "gcol", _write(tmp_path, "data.json", data), _write(tmp_path, "output.json", output), 1e-6)
    assert not result["ok"]
    assert any("objective mismatch" in issue for issue in result["issues"])
