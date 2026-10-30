"""Import literature instances for the scale-up study into `outputs/scaleup/instances/<family>/<name>/`.

Each instance directory holds `data.json` (the family's DecompGym format) and `meta.json`
(source, raw-file hash, reference value and whether it is a proven optimum or a best known value).
Run as: python -m src.experiments.scaleup_instances ufl-kg --raw <UflLib KoerkelGhosh dir> [--names gs250a-1 ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path("outputs/scaleup/instances")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_ufl_kg(raw: Path, names: list[str] | None, source: str) -> list[Path]:
    from src.bd.ufl.a_gen_data import load_ufllib_instance, read_ufllib_reference

    written = []
    for path in sorted(p for p in raw.rglob("g[sa]*") if p.suffix == ""):
        if names and path.name not in names:
            continue
        ref = next((path.with_suffix(s) for s in (".opt", ".bub") if path.with_suffix(s).exists()), None)
        inst = load_ufllib_instance(path)
        out = ROOT / "ufl" / path.name
        out.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out / "data.json")
        meta = {
            "family": "ufl", "name": path.name, "source": source, "raw_file": str(path), "raw_sha256": sha256(path),
            "m": inst.m, "n": inst.n, "class": path.name[len("gs250")], "symmetric": path.name.startswith("gs"),
            "reference_value": read_ufllib_reference(ref) if ref else None,
            "reference_kind": {".opt": "opt", ".bub": "bks"}.get(ref.suffix) if ref else None,
            "reference_file_sha256": sha256(ref) if ref else None,
            "sense": "min", "data_sha256": sha256(out / "data.json"),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        written.append(out)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    kg = sub.add_parser("ufl-kg")
    kg.add_argument("--raw", type=Path, required=True)
    kg.add_argument("--names", nargs="*")
    kg.add_argument("--source", default="UflLib Koerkel-Ghosh, https://algo.rwth-aachen.de/ufllib/ (mirror of MPI UflLib), downloaded 2026-10-30")
    args = parser.parse_args()
    if args.cmd == "ufl-kg":
        for out in import_ufl_kg(args.raw, args.names, args.source):
            print(out)


if __name__ == "__main__":
    main()
