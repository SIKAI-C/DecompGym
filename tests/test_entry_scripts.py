"""The entry scripts hand campaign commands to the right code."""
import importlib
import json

import pytest

experiment = importlib.import_module("src.02_experiment")


@pytest.mark.parametrize("layout", ["src/experiments/campaign.py", "src/native_campaign.py"])
@pytest.mark.parametrize("spelling", ["separate", "equals"])
def test_run_uses_the_frozen_controller_with_one_absolute_root(tmp_path, monkeypatch, layout, spelling):
    root = tmp_path / "campaign"
    (root / "controller" / layout).parent.mkdir(parents=True)
    (root / "controller" / layout).write_text("")
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(experiment.subprocess, "call", lambda command, cwd: calls.append((command, cwd)) or 0)
    arguments = ["--root", "campaign"] if spelling == "separate" else ["--root=campaign"]
    assert experiment.frozen("run", [*arguments, "--all-models"]) == 0
    command, cwd = calls[0]
    assert cwd == root.resolve() / "controller"
    assert command[3] == ("src.experiments.campaign" if "experiments" in layout else "src.native_campaign")
    assert command[4:] == ["run", "--root", str(root.resolve()), "--all-models"]


def test_a_saved_audit_is_rejected_once_its_campaign_changed(tmp_path):
    from src.results.snapshot import check_cached
    root = tmp_path / "campaign"
    data = root / "dataset/bd/ufl/ufl_small_seed000/data.json"
    data.parent.mkdir(parents=True)
    data.write_text("{}")
    import hashlib
    manifest = {"instances": [{"suite": "bd", "problem": "ufl", "instance": "ufl_small_seed000", "sha256": {"data.json": hashlib.sha256(b"{}").hexdigest()}}]}
    (root / "campaign.json").write_text(json.dumps(manifest))
    check_cached({"cohort_manifest": manifest}, root)
    data.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="changed since the saved audit"):
        check_cached({"cohort_manifest": manifest}, root)
    with pytest.raises(ValueError, match="no longer matches"):
        check_cached({"cohort_manifest": {"instances": []}}, root)
