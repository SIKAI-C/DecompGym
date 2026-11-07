"""Build or check the combined small-scale dataset root (outputs/small_scale_data) of configs/small_scale.json.

    python scripts/build_small_scale_data.py build   # once, into a new root; an existing root is never rewritten
    python scripts/build_small_scale_data.py check   # re-verify an existing root against its sources

Every family of the config's cohort is copied from its registered final dataset (src.utils.families: `dataset`), all
instance files byte-identical, sidecars (source.json, provenance.json) included. Then every prompt.txt is re-rendered
through the existing render path (src.agent.prompts.snapshot.render_dataset_prompts) under the config's decomposition
contract. The check proves that the re-render changes nothing but the contract: a prompt either equals its source or,
for a Benders family whose source was rendered under the paper's v2 contract, equals source.replace(v2 sentence,
v3 sentence); column-generation prompts never change; every other file is byte-identical; each stored prompt equals the
render of its data.json. No solver is imported. The source roots are only read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.agent.prompts.method import CONTRACTS, DECOMPOSITION_POLICY, DECOMPOSITION_POLICY_V3  # noqa: E402
from src.agent.prompts.snapshot import render_case_prompt, render_dataset_prompts  # noqa: E402
from src.experiments.data_context import decomposition_contract_policy  # noqa: E402
from src.experiments.plan import load_plan  # noqa: E402
from src.utils.families import FAMILIES, SEEDS, SIZES, SMALL_SCALE_CAMPAIGN_DATA, SMALL_SCALE_CONFIG  # noqa: E402

RECORD = "build_record.json"
# The only sentence that differs between the paper's v2 and the v3 Benders contract (src.agent.prompts.method).
V2_SENTENCE = "Use a master problem for the complicating decisions and recourse-cost estimates."
V3_SENTENCE = "Use a master problem for the complicating decisions and, when applicable, recourse-cost estimates."
assert CONTRACTS[DECOMPOSITION_POLICY_V3]["bd"] == CONTRACTS[DECOMPOSITION_POLICY]["bd"].replace(V2_SENTENCE, V3_SENTENCE)
assert CONTRACTS[DECOMPOSITION_POLICY_V3]["cg"] == CONTRACTS[DECOMPOSITION_POLICY]["cg"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort(config: Path) -> tuple[list[str], str]:
    plan = load_plan(config)
    return plan["cohort"]["families"], decomposition_contract_policy(plan["prompt_protocol"])


def instances(name: str) -> list[str]:
    return [f"{name}_{size}_seed{seed:03d}" for size in SIZES for seed in SEEDS]


def source_dir(name: str) -> Path:
    entry = FAMILIES[name]
    return REPO / entry.dataset / entry.suite / name


def check(root: Path, config: Path) -> dict:
    """Verify `root` against the cohort's source datasets; raise on the first difference. Returns the build summary."""
    families, contract = cohort(config)
    summary = {"contract": contract, "families": {}}
    found = sorted(str(path.parent.relative_to(root)) for path in root.glob("*/*/*/data.json"))
    expected = sorted(f"{FAMILIES[name].suite}/{name}/{instance}" for name in families for instance in instances(name))
    if found != expected:
        raise ValueError(f"{root}: instance set differs from the cohort ({len(found)} found, {len(expected)} expected)")
    extra = {path.name for path in root.iterdir()} - {"bd", "cg", RECORD}
    if extra or any(sorted(path.name for path in (root / suite).iterdir()) != sorted(name for name in families if FAMILIES[name].suite == suite) for suite in ("bd", "cg")):
        raise ValueError(f"{root}: unexpected entries {sorted(extra)} at the root or other family directories")
    for name in families:
        entry = FAMILIES[name]
        source, copy = source_dir(name), root / entry.suite / name
        if sorted(path.name for path in copy.iterdir()) != sorted(instances(name)) or sorted(path.name for path in source.iterdir()) != sorted(instances(name)):
            raise ValueError(f"{name}: family directories hold other entries than its nine instances")
        changed = unchanged = 0
        for instance in instances(name):
            old, new = source / instance, copy / instance
            names = sorted(path.name for path in old.iterdir())
            if sorted(path.name for path in new.iterdir()) != names or any(not (old / item).is_file() or (old / item).is_symlink() or (new / item).is_symlink() for item in names):
                raise ValueError(f"{name}/{instance}: file set or file types differ from the source")
            for item in names:
                if item != "prompt.txt" and (old / item).read_bytes() != (new / item).read_bytes():
                    raise ValueError(f"{name}/{instance}/{item} differs from its source")
            before, after = (old / "prompt.txt").read_text(), (new / "prompt.txt").read_text()
            if after != render_case_prompt(entry.suite, name, json.loads((new / "data.json").read_text()), contract):
                raise ValueError(f"{name}/{instance}: prompt.txt is not the {contract} render of its data.json")
            if after == before:
                unchanged += 1
            elif entry.suite == "bd" and contract == DECOMPOSITION_POLICY_V3 and before.count(V2_SENTENCE) == 1 and after == before.replace(V2_SENTENCE, V3_SENTENCE):
                changed += 1
            else:
                raise ValueError(f"{name}/{instance}: prompt.txt changed beyond the contract sentence")
        summary["families"][name] = {"suite": entry.suite, "source": entry.dataset, "instances": 9, "prompts_changed_v2_to_v3": changed, "prompts_unchanged": unchanged}
    return summary


def build(root: Path, config: Path) -> dict:
    if root.exists():
        raise SystemExit(f"{root} exists; the combined root is built once and never rewritten (use `check`)")
    families, contract = cohort(config)
    staging = root.parent / f".{root.name}.building-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        sources = {}
        for name in families:
            entry = FAMILIES[name]
            destination = staging / entry.suite / name
            destination.mkdir(parents=True)
            for instance in instances(name):
                shutil.copytree(source_dir(name) / instance, destination / instance, copy_function=shutil.copyfile)
            sources[name] = {"dataset": entry.dataset, "files_sha256": {f"{instance}/{path.name}": digest(path) for instance in instances(name) for path in sorted((source_dir(name) / instance).iterdir())}}
        rendered = render_dataset_prompts(staging, contract)
        summary = check(staging, config)
        record = {"config": str(config.resolve().relative_to(REPO)) if config.resolve().is_relative_to(REPO) else str(config), "builder": "scripts/build_small_scale_data.py",
                  "render": rendered, "summary": summary, "sources": sources}
        (staging / RECORD).write_text(json.dumps(record, indent=2) + "\n")
        staging.rename(root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "check"])
    parser.add_argument("--root", type=Path, default=REPO / SMALL_SCALE_CAMPAIGN_DATA)
    parser.add_argument("--config", type=Path, default=REPO / SMALL_SCALE_CONFIG)
    args = parser.parse_args(argv)
    summary = build(args.root, args.config) if args.command == "build" else check(args.root, args.config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
