"""The family registry (src.utils.families) is the single table every tool reads; no optimizer is imported here."""
from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.prompts.method import DECOMPOSITION_POLICY, DECOMPOSITION_POLICY_V3
from src.experiments.data_context import decomposition_contract_policy
from src.experiments.plan import load_plan
from src.utils.families import FAMILIES, PAPER_FAMILIES, SEEDS, SIZES, dataset_families, family, problems
from src.utils.instance_suite import META_LAYOUTS


ROOT = Path(__file__).resolve().parents[1]
FINAL = {"bd": {"ufl", "km", "fctp", "scfl", "lbbd_ps", "tsso", "lrp", "lbbd"},
         "cg": {"bp", "csp", "gap", "bppc", "clrp", "cvrp", "vrptw", "pdptw", "rcsp", "shift", "gcol", "nrp"}}
# The dashboard labels before the registry existed, in their order.
LABELS = {"ufl": "Facility location", "fctp": "Fixed-charge transportation", "km": "K-median", "lrp": "Location routing", "tsso": "Two-stage stochastic optimization", "bp": "Bin packing", "csp": "Cutting stock", "gap": "Generalized assignment", "shift": "Shift scheduling",
          "lbbd": "Assignment and scheduling (logic-based Benders)", "gcol": "Graph coloring", "bppc": "Bin packing with conflicts",
          "vrptw": "Vehicle routing with time windows", "pdptw": "Pickup and delivery with time windows", "nrp": "Nurse rostering",
          "cvrp": "Capacitated vehicle routing", "clrp": "Capacitated location-routing", "scfl": "Stochastic capacitated facility location", "lbbd_ps": "Planning and cumulative scheduling (logic-based Benders)", "rcsp": "Railway crew scheduling"}


def defined(path: Path) -> set[str]:
    return {node.name for node in ast.parse(path.read_text()).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def test_the_registry_holds_exactly_the_final_families():
    assert {suite: set(problems(suite)) for suite in ("bd", "cg")} == FINAL
    assert list(FAMILIES) == list(LABELS) and {name: entry.label for name, entry in FAMILIES.items()} == LABELS
    on_disk = {suite: {p.name for p in (ROOT / "src" / suite).iterdir() if p.is_dir() and p.name != "__pycache__"} for suite in ("bd", "cg")}
    assert on_disk == FINAL  # no family code outside the registry
    assert PAPER_FAMILIES == ("ufl", "fctp", "km", "lrp", "tsso", "bp", "csp", "gap", "shift")
    with pytest.raises(KeyError):
        family("ufl", "cg")
    with pytest.raises(KeyError):
        family("hub")


@pytest.mark.parametrize("name", list(FAMILIES))
def test_every_family_has_its_files_and_entry_points(name):
    entry = FAMILIES[name]
    directory = ROOT / "src" / entry.suite / name
    assert entry.generator in defined(directory / "a_gen_data.py")
    if entry.recipe:
        assert entry.recipe in defined(directory / "a_gen_data.py")
    for spec in (entry.monolithic, entry.reference):
        module, cls = spec.split(".")
        assert cls in defined(directory / f"{module}.py"), spec
    assert "render_prompt" in defined(directory / "b_problem.py")
    assert "generate_suite" in defined(directory / "c_gen.py")
    assert "verify_solution" in defined(ROOT / entry.verifier)
    assert entry.sizes == SIZES and entry.meta_layout in {*META_LAYOUTS, "family"}


@pytest.mark.parametrize("name", list(FAMILIES))
def test_admission_settings_follow_the_dataset(name):
    entry = FAMILIES[name]
    if entry.dataset == "outputs/litgen_data":
        assert (entry.admission, entry.mip_time_limit, entry.contract, entry.config) == ("strict", 1800.0, DECOMPOSITION_POLICY_V3, "configs/small_scale.json")
    elif entry.dataset == "data_harder":
        assert (entry.admission, entry.mip_time_limit, entry.contract, entry.meta_layout) == ("mip_or_certificate", 300.0, DECOMPOSITION_POLICY, "source_summary")
    else:
        assert (entry.dataset, entry.admission, entry.contract, entry.config) == ("data", "family", DECOMPOSITION_POLICY, "configs/paper.json")
    # The generation helpers before the registry: bp warm-starts its reference; SCFL and PDPTW certify to 1e-9.
    assert dict(entry.reference_options) == ({"ffd_warm_start": True} if name == "bp" else {})
    assert entry.tol == (1e-9 if name in {"scfl", "pdptw"} else 1e-6)
    assert entry.reference_certified == (name in {"scfl", "lbbd_ps", "clrp", "cvrp", "vrptw", "pdptw", "rcsp"})
    assert entry.full_verifier == (name in {"lbbd", "scfl", "lbbd_ps"})


@pytest.mark.parametrize("config", ["paper.json", "harder_small_all.json", "small_scale.json"])
def test_each_config_pins_registered_families_and_their_contract(config):
    plan = load_plan(ROOT / "configs" / config)
    names = plan.get("cohort", {}).get("families", list(PAPER_FAMILIES))
    contract = decomposition_contract_policy(plan["prompt_protocol"])
    owned = [name for name in names if FAMILIES[name].config == f"configs/{config}"]
    assert all(FAMILIES[name].contract == contract for name in owned)
    assert sorted(owned) == sorted(name for name, entry in FAMILIES.items() if entry.config == f"configs/{config}")


@pytest.mark.parametrize("dataset", ["data", "data_harder", "outputs/litgen_data"])
def test_each_dataset_holds_its_families_nine_instances(dataset):
    if not (ROOT / dataset).is_dir():
        pytest.skip(f"{dataset} is not present (outputs/ is not part of the repository)")
    for name in dataset_families(dataset):
        entry = FAMILIES[name]
        found = sorted(p.parent.name for p in (ROOT / dataset / entry.suite / name).glob("*/data.json"))
        assert found == sorted(f"{name}_{size}_seed{seed:03d}" for size in SIZES for seed in SEEDS), name


def test_the_paper_analysis_tables_are_unchanged():
    from src.results.combine import PROBLEMS
    from src.results.paper_figures import FAMILIES as PAPER_TABLE

    assert PROBLEMS == [("bd", p) for p in ("ufl", "fctp", "km", "lrp", "tsso")] + [("cg", p) for p in ("bp", "csp", "gap", "shift")]
    assert PAPER_TABLE == ("bd/ufl", "bd/fctp", "bd/km", "bd/lrp", "bd/tsso", "cg/bp", "cg/csp", "cg/gap", "cg/shift")


def test_dispatch_tables_read_the_registry():
    from src.experiments.campaign import FAMILIES as CAMPAIGN_DEFAULT
    from src.experiments.dashboard_server import FAMILIES as DASHBOARD
    from src.experiments.paired_runner import DEFAULT_PROBLEMS
    from src.utils.api import reference_class_name

    assert CAMPAIGN_DEFAULT == list(PAPER_FAMILIES) and DASHBOARD == LABELS
    assert DEFAULT_PROBLEMS == {"bd": ["tsso", "fctp", "km", "lrp", "ufl"], "cg": ["bp", "csp", "gap", "shift"]}
    assert reference_class_name("bd", "ufl", "monolithic") == "b_gold_monolithic.UFLBaseline"
    assert reference_class_name("cg", "vrptw", "decomposition") == "a_gold_cg.BranchAndPrice"
    with pytest.raises(KeyError):
        reference_class_name("bd", "bp", "monolithic")


def test_a_cohort_must_name_registered_families(tmp_path):
    from src.utils.errors import ProtocolError

    plan = json.loads((ROOT / "configs/small_scale.json").read_text())
    for families in (["ufl", "hub"], [], ["ufl", "ufl"]):
        (tmp_path / "config.json").write_text(json.dumps(dict(plan, cohort={"name": "x", "families": families})))
        with pytest.raises(ProtocolError, match="registered problem families"):
            load_plan(tmp_path / "config.json")


def test_generation_never_writes_a_pinned_dataset_root(tmp_path):
    generator = importlib.import_module("src.01_gen")
    args = SimpleNamespace(suite="all", problem=None, sizes=list(SIZES), num_instances=3, seed_offset=0, regenerate=True)
    with pytest.raises(SystemExit, match="pinned family dataset root"):
        generator.generate(ROOT / "data", args)
    assert generator.selected_families(tmp_path, "cg", None) == problems("cg")
    assert generator.selected_families(ROOT / "data", "all", None) == [name for name in FAMILIES if (ROOT / "data" / FAMILIES[name].suite / name).is_dir()]
    with pytest.raises(ValueError, match="Unknown problem"):
        generator.selected_families(ROOT / "data", "all", "cvrp")


@pytest.mark.parametrize("name", list(FAMILIES))
def test_every_family_has_an_output_contract(name):
    from src.utils.output_contract import validate_output_contract

    entry = FAMILIES[name]
    method = "benders_decomposition" if entry.suite == "bd" else "column_generation"
    # The family's own schema is reached (an unknown problem raises there) and reports the missing plan.
    issues = validate_output_contract(entry.suite, name, {}, {"status": "feasible", "method": method, "objective_value": 1.0, "solution": {}})
    assert issues and all("status" not in issue and "method" not in issue for issue in issues)


def test_api_generation_refuses_a_pinned_dataset_root():
    from src.utils.api import generate_dataset

    for dataset in ("data", "data_harder", "outputs/litgen_data"):
        with pytest.raises(ValueError, match="pinned family dataset root"):
            generate_dataset(ROOT / dataset, problems=("ufl",))
