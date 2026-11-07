"""Instance generation and admission (src.utils.instance_suite): certificates, the strict rule and rejection records."""
from __future__ import annotations

import json

import pytest


def test_certificates_must_be_ordered_meeting_and_match_the_objective():
    from src.utils.instance_suite import certified

    assert certified({"global_lb": 10, "global_ub": 10, "best_objective": 10})
    assert not certified({"global_lb": 20, "global_ub": 10, "best_objective": 10})  # crossed
    assert not certified({"global_lb": 0, "global_ub": 0, "best_objective": 10})  # bound/objective mismatch
    assert not certified({"global_lb": None, "global_ub": 10, "best_objective": 10})


def test_strict_suite_rejects_an_uncertified_reference(monkeypatch, tmp_path):
    # Codex review (routing families) finding 2: strict families need both certificates, and they must agree.
    from src.utils import instance_suite

    data = {"n_customers": 1}
    gen = lambda seed, size, out: (out.mkdir(parents=True, exist_ok=True), (out / "data.json").write_text("{}"), data)[2]
    # api.solve always records a measured total_time and the MIP's closed bounds; the stub does too.
    results = {"monolithic": {"best_objective": 10.0, "global_lb": 10.0, "global_ub": 10.0, "output": {}, "total_time": 1.0}, "decomposition": {"best_objective": 10.0, "global_lb": 9.0, "global_ub": 10.0, "output": {}, "total_time": 1.0}}
    monkeypatch.setattr(instance_suite, "solve", lambda suite, problem, path, method, **kw: dict(results[method]))
    monkeypatch.setattr(instance_suite, "verify", lambda *args: {"ok": True})
    with pytest.raises(RuntimeError, match="matching certified reference"):
        instance_suite.generate_suite("cvrp", gen=gen, render_prompt=lambda d: "p", sizes=["small"], num_instances=1, out_root=tmp_path)
    results["decomposition"]["global_lb"] = 10.0
    metas = instance_suite.generate_suite("cvrp", gen=gen, render_prompt=lambda d: "p", sizes=["small"], num_instances=1, out_root=tmp_path)
    assert metas[0]["ground_truth_source"] == "monolithic_mip" and metas[0]["reference_proved"]
    # Codex full review finding 7: a certified reference over the 60 s budget is not admitted.
    results["decomposition"]["total_time"] = 61.0
    with pytest.raises(RuntimeError, match="reference time"):
        instance_suite.generate_suite("cvrp", gen=gen, render_prompt=lambda d: "p", sizes=["small"], num_instances=1, out_root=tmp_path)


def test_strict_suite_records_generation_failure(tmp_path):
    from src.utils.instance_suite import generate_suite

    folder = tmp_path / "bp_small_seed000"
    folder.mkdir()
    (folder / "meta.json").write_text("{}")

    def boom(seed, size, inst_dir):
        raise ValueError("no draw")

    with pytest.raises(RuntimeError, match="generation or prompt failed"):
        generate_suite("bp", gen=boom, render_prompt=lambda d: "p", sizes=["small"], num_instances=1, out_root=tmp_path)
    assert not (folder / "meta.json").exists() and "no draw" in (folder / "rejection.json").read_text()


def _stub(monkeypatch, results, calls=None):
    from src.utils import instance_suite

    def solve(suite, problem, path, method, **kw):
        if calls is not None:
            calls.append((method, kw))
        if isinstance(results[method], Exception):
            raise results[method]
        return dict(results[method])

    monkeypatch.setattr(instance_suite, "solve", solve)
    monkeypatch.setattr(instance_suite, "verify", lambda *args: {"ok": True})
    return instance_suite


def _gen(data):
    def gen(seed, size, out):
        out.mkdir(parents=True, exist_ok=True)
        (out / "data.json").write_text("{}")
        return data
    return gen


def _mip(value=10.0, lb=None):
    return {"best_objective": value, "global_lb": value if lb is None else lb, "global_ub": value, "output": {}, "total_time": 1.0}


@pytest.mark.parametrize("change,reason", [
    ({"monolithic": _mip(10.0, lb=9.0)}, "MIP bound not closed"),
    ({"monolithic": {**_mip(), "total_time": 1801.0}}, "MIP time"),
    ({"monolithic": RuntimeError("time limit")}, "monolithic MIP not proved optimal"),
    ({"decomposition": {**_mip(9.0), "global_lb": 8.0}}, "decomposition gap"),  # materially below the proved optimum
    ({"decomposition": {**_mip(10.2), "global_lb": 9.0}}, "decomposition gap"),  # above the 1 % rule
    ({"decomposition": _mip(10.05)}, "differs from the MIP optimum"),  # a certified reference must match
])
def test_the_strict_rule_rejects(monkeypatch, tmp_path, change, reason):
    suite = _stub(monkeypatch, {"monolithic": _mip(), "decomposition": {**_mip(10.05), "global_lb": 9.0}, **change})
    with pytest.raises(RuntimeError, match=reason):
        suite.generate_suite("km", gen=_gen({"m": 1, "n": 1}), render_prompt=lambda d: "p", sizes=["small"], num_instances=1, out_root=tmp_path)
    folder = tmp_path / "km_small_seed000"
    assert not (folder / "meta.json").exists() and reason.split()[0] in (folder / "rejection.json").read_text()


def test_the_strict_rule_passes_the_mip_settings_and_reference_options(monkeypatch, tmp_path):
    calls = []
    suite = _stub(monkeypatch, {"monolithic": _mip(102.0), "decomposition": {**_mip(102.0), "global_lb": 101.5}}, calls)
    gen = lambda seed, size, out: (_gen({"items": []})(seed, size, out), {"recipe": "scholl1-C2W2", "n_items": 3})
    [meta] = suite.generate_suite("bp", gen=gen, render_prompt=lambda d: "p", sizes=["small"], num_instances=1, out_root=tmp_path)
    assert calls == [("monolithic", {"params": {"Threads": 1, "MIPGap": 0.0, "TimeLimit": 1800.0}}), ("decomposition", {"time_limit": 60.0, "ffd_warm_start": True})]
    # The recipe-fields layout of bp, csp, gap and bppc: MIP settings, bounds and the recipe fields; golds keep wall_seconds.
    assert meta["mip_time_limit"] == 1800.0 and meta["mip_threads"] == 1 and meta["mip_gap"] == 0.0 and meta["recipe"] == "scholl1-C2W2"
    assert meta["reference_bound"] == 101.5 and meta["reference_gap"] == 0.0
    for name in ("gold_monolithic.json", "gold_cg.json"):
        assert "wall_seconds" in json.loads((tmp_path / "bp_small_seed000" / name).read_text())


def test_the_reused_families_keep_their_historical_rule(monkeypatch, tmp_path):
    calls = []
    certificate = {**_mip(4.0), "global_lb": 4.0}
    suite = _stub(monkeypatch, {"monolithic": RuntimeError("not proved"), "decomposition": certificate}, calls)
    [meta] = suite.generate_suite("gcol", gen=_gen({"n_vertices": 2}), render_prompt=lambda d: "p", sizes=["small"], num_instances=1, out_root=tmp_path)
    assert calls[0] == ("monolithic", {"params": {"TimeLimit": 300.0}}) and calls[1] == ("decomposition", {"time_limit": 60.0})
    assert meta["ground_truth_source"] == "decomposition_certificate" and meta["mip_time_limit"] == 300.0
    assert meta["source"] == {"synthetic": True, "n_vertices": 2}
    # Neither a proved MIP nor a certificate: no meta.json and no exception (the historical behavior).
    _stub(monkeypatch, {"monolithic": RuntimeError("not proved"), "decomposition": {**certificate, "global_lb": 3.0}})
    assert suite.generate_suite("gcol", gen=_gen({"n_vertices": 2}), render_prompt=lambda d: "p", sizes=["small"], num_instances=1, out_root=tmp_path) == []
    assert not (tmp_path / "gcol_small_seed000" / "meta.json").exists()


def test_the_paper_families_keep_their_own_pipeline(tmp_path):
    from src.utils.instance_suite import generate_suite

    with pytest.raises(ValueError, match="own pipeline"):
        generate_suite("tsso", sizes=["small"], num_instances=1, out_root=tmp_path)


def _reused_family_stubs(monkeypatch, case, calls):
    """Stub solve, verify, generation and prompt rendering of instance_suite for one historical-rule outcome of gcol."""
    from src.utils import instance_suite

    mono = {"best_objective": 5.0, "global_lb": 5.0, "global_ub": 5.0, "total_time": 0.1, "output": {"which": "monolithic"}}
    ref = {"best_objective": 5.0, "global_lb": 4.0, "global_ub": 5.0, "total_time": 0.1, "output": {"which": "decomposition"}}
    results = {"mip_rejected_no_certificate": {"monolithic": mono, "decomposition": ref},
               "certificate_over_rejected_mip": {"monolithic": {**mono, "best_objective": 9.0, "global_lb": 9.0, "global_ub": 9.0}, "decomposition": {**ref, "global_lb": 5.0}},
               "certificate_only": {"monolithic": RuntimeError("not proved"), "decomposition": {**ref, "global_lb": 5.0}},
               "mip_only": {"monolithic": mono, "decomposition": RuntimeError("no plan")}}[case]

    def solve(suite, problem, path, method, **kw):
        if isinstance(results[method], Exception):
            raise results[method]
        return dict(results[method])

    def generator(entry):
        def gen(seed, size, out):
            calls.append((seed, size))
            out.mkdir(parents=True, exist_ok=True)
            (out / "data.json").write_text("{}")
            return {"n_vertices": 1}
        return gen

    rejected_mip = case in {"mip_rejected_no_certificate", "certificate_over_rejected_mip"}
    monkeypatch.setattr(instance_suite, "solve", solve)
    monkeypatch.setattr(instance_suite, "verify", lambda suite, problem, path, output: {"ok": not (rejected_mip and output["which"] == "monolithic")})
    monkeypatch.setattr(instance_suite, "_generator", generator)
    monkeypatch.setattr(instance_suite, "_render", lambda entry, contract: lambda data: "p")


@pytest.mark.parametrize("case", ["mip_rejected_no_certificate", "certificate_over_rejected_mip", "certificate_only", "mip_only"])
def test_the_dataset_wrapper_honors_the_reused_families_admission(monkeypatch, tmp_path, case):
    """ensure_suite and write_manifest (01_gen, api.generate_dataset) follow the generator-owned admission: no meta.json
    is synthesized, and completion, reuse and the manifest take the admitted ground truth."""
    from src.utils.dataset import ensure_suite, instance_complete, write_manifest

    calls = []
    _reused_family_stubs(monkeypatch, case, calls)
    records = ensure_suite("cg", "gcol", tmp_path / "cg" / "gcol", ["small"], 1)
    folder = tmp_path / "cg" / "gcol" / "gcol_small_seed000"
    [entry] = write_manifest(tmp_path)["instances"]
    if case == "mip_rejected_no_certificate":
        assert records == [] and not (folder / "meta.json").exists() and not instance_complete(folder, "cg") and not entry["complete"]
        return
    meta = json.loads((folder / "meta.json").read_text())
    expected = {"certificate_over_rejected_mip": ("decomposition_certificate", 5.0, 5.0, 0.0), "certificate_only": ("decomposition_certificate", 5.0, 5.0, 0.0),
                "mip_only": ("monolithic_mip", 5.0, None, None)}[case]
    assert meta["ground_truth_source"] == expected[0] and instance_complete(folder, "cg")
    assert (records[0]["reference_objective"], records[0]["decomposition_objective"], records[0]["decomposition_gap"]) == expected[1:]
    assert entry["complete"] and (entry["reference_objective"], entry["decomposition_objective"], entry["decomposition_gap"]) == expected[1:]
    again = ensure_suite("cg", "gcol", tmp_path / "cg" / "gcol", ["small"], 1)
    assert again[0]["reused"] and len(calls) == 1  # complete, so reused without another generation


def test_the_cli_refreshes_the_manifest_after_a_skipped_regeneration(monkeypatch, tmp_path):
    import importlib
    from types import SimpleNamespace

    generator = importlib.import_module("src.01_gen")
    args = SimpleNamespace(suite="cg", problem="gcol", sizes=["small"], num_instances=1, seed_offset=0, regenerate=False)
    calls = []
    _reused_family_stubs(monkeypatch, "certificate_only", calls)
    generator.generate(tmp_path, args)
    assert json.loads((tmp_path / "manifest.json").read_text())["instances"][0]["complete"]
    _reused_family_stubs(monkeypatch, "mip_rejected_no_certificate", calls)
    generator.generate(tmp_path, SimpleNamespace(**{**vars(args), "regenerate": True}))
    [entry] = json.loads((tmp_path / "manifest.json").read_text())["instances"]
    assert not entry["complete"] and len(calls) == 2
