"""Data for the DecompGym website (site/, published with GitHub Pages).

    python -m src.results.site        # refresh site/data.js and the overview figure from outputs/reports/paper

src/03_analysis.py calls build() after it publishes results/, so one analysis run refreshes the site too.
Reads only the saved paper report (combined.json, resources.json, time/ and tokens/ budget records) and
src/results/styles.json; nothing here calls a model or runs a solver.

The page derives every number in the browser from the per-instance records written here: for each
configuration, policy and instance, the first attempt that reaches E, V and Q, and the cumulative agent,
reflection, judge and solver seconds and tokens through budgets 1..10 (a reflection recorded at attempt k
counts toward budget k + 1, as in time_analysis.py and token_analysis.py). The checks below tie those
records back to every aggregate count in combined.json and to resources.json.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from src.results.combine import STYLE

REPOSITORY = Path(__file__).resolve().parents[2]
SNAPSHOT = "paper v1 (arXiv 2610.38175)"
MODES = ("plain", "reflection")
METRICS = ("execution", "verification", "quality")
SIZES = ("small", "medium", "large")
K = 10
FAMILIES = {
    "bd/ufl": "Uncapacitated facility location",
    "bd/fctp": "Fixed-charge transportation",
    "bd/km": "k-median",
    "bd/lrp": "Location routing",
    "bd/tsso": "Two-stage stochastic production",
    "cg/bp": "Bin packing",
    "cg/csp": "Cutting stock",
    "cg/gap": "Generalized assignment",
    "cg/shift": "Shift scheduling",
}
AGENT = {"OpenAI": "Codex", "Anthropic": "Claude Code", "Google": "Antigravity"}


def provider(model: str) -> str:
    return "OpenAI" if model.startswith("gpt") else "Anthropic" if model.startswith("claude") else "Google"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _key(r):
    return (f'{r["suite"]}/{r["problem"]}', r["instance"])


def site_data(report: Path) -> dict:
    """Per-instance records for the website, checked against the aggregate report."""
    sources = {"combined": report / "combined.json", "resources": report / "resources.json",
               "time": report / "time" / "time_analysis.json", "tokens": report / "tokens" / "token_analysis.json"}
    data = json.loads(sources["combined"].read_text())
    res = json.loads(sources["resources"].read_text())
    time_records = json.loads(sources["time"].read_text())["budget_records"]
    token_records = json.loads(sources["tokens"].read_text())["budget_records"]
    models = data["primary_models"]
    assert len(set(models)) == len(models)

    cases = sorted({(f'{c["suite"]}/{c["problem"]}', c["size"], c["seed"], c["instance"]) for c in data["cases"]},
                   key=lambda c: (list(FAMILIES).index(c[0]), SIZES.index(c[1]), c[2]))
    assert len({c[:3] for c in cases}) == len(cases) == len(data["cases"])
    index = {(c[0], c[3]): i for i, c in enumerate(cases)}

    def keyed(records):
        keys = [(r["model"], r["mode"], _key(r), r["budget"]) for r in records if r["model"] in models]
        assert len(keys) == len(set(keys)) == len(models) * len(MODES) * len(cases) * K, "budget records must cover each key exactly once"
        return {(r["model"], r["mode"], _key(r), r["budget"]): r for r in records if r["model"] in models}
    tmap, kmap = keyed(time_records), keyed(token_records)

    def series(rows, field):
        # Full precision, so time-budget thresholds on the page match the paper; nulls stay per budget.
        vals = [r[field] for r in rows]
        return None if all(x is None for x in vals) else vals

    runs = {}
    for p in data["pairs"]:
        if p["model"] not in models:
            continue
        assert p["state"] == "complete"
        i = index[_key(p)]
        for mode in MODES:
            am = p["policies"][mode]["attempt_metrics"]
            first = []
            for met in METRICS:
                flags = [bool(am[str(k)][met]) for k in range(1, K + 1)]
                assert all(not a or b for a, b in zip(flags, flags[1:])), "cumulative outcomes must be monotone"
                first.append(next((k for k in range(1, K + 1) if flags[k - 1]), 0))
            e, v, q = first
            assert (v == 0 or (e and e <= v)) and (q == 0 or (v and v <= q)), "E >= V >= Q must hold at every budget"
            rows_t = [tmap[(p["model"], mode, _key(p), k)] for k in range(1, K + 1)]
            rows_k = [kmap[(p["model"], mode, _key(p), k)] for k in range(1, K + 1)]
            for k, (rt, rk) in enumerate(zip(rows_t, rows_k), 1):
                assert all(rt[m] == am[str(k)][m] == rk[m] for m in METRICS)
            runs.setdefault(p["model"], {}).setdefault(mode, [None] * len(cases))[i] = [
                e, v, q, series(rows_t, "agent_seconds"), series(rows_t, "reflection_seconds"),
                series(rows_t, "judge_seconds"), series(rows_t, "solver_seconds"), series(rows_k, "agent_accounted_tokens")]

    for m in models:  # the first attempt is shared by both policies
        for P, R in zip(runs[m]["plain"], runs[m]["reflection"]):
            assert all((P[j] == 1) == (R[j] == 1) for j in range(3))
            assert P[3][0] == R[3][0] and (P[7] is None) == (R[7] is None) and (P[7] is None or P[7][0] == R[7][0])

    checked = 0  # the instance records reproduce every aggregate count
    for r in data["rows"]:
        if r["model"] not in models:
            continue
        j = METRICS.index(r["metric"])
        member = {"overall": lambda c: True, "problem": lambda c: c[0] == r["group"], "size": lambda c: c[1] == r["group"],
                  "suite": lambda c: c[0].startswith(r["group"] + "/")}[r["dimension"]]
        sel = [i for i, c in enumerate(cases) if member(c)]
        assert len(sel) == r["paired_cases"]
        for mode in MODES:
            n = sum(1 for i in sel if 0 < runs[r["model"]][mode][i][j] <= r["budget"])
            assert n == r[mode + "_successes"], (r["model"], r["dimension"], r["group"], r["metric"], mode, r["budget"])
        checked += 1

    for r in res["rows"]:  # budget-10 costs are the paper's time and token figures
        if r["model"] not in models:
            continue
        rec = runs[r["model"]][r["mode"]][index[_key(r)]]
        assert rec[3][-1] == r["agent_seconds"]
        assert (rec[7][-1] if rec[7] else None) == r["agent_accounted_tokens"]
        assert not r["quality"] or rec[3][rec[2] - 1] == r["time_to_quality_agent_seconds"]

    styles = STYLE["models"]
    return {
        "meta": {"schema": 2, "snapshot": SNAPSHOT, "style_id": STYLE["style_id"], "sources": {k: _sha(v) for k, v in sources.items()}, "checked_rows": checked},
        "models": [{"id": m, "label": styles[m]["label"].replace(" (M)", ""), "color": styles[m]["color"], "marker": styles[m]["marker"],
                    "dash": styles[m]["linestyle"], "order": styles[m]["order"], "provider": provider(m), "agent": AGENT[provider(m)]} for m in models],
        "families": [{"id": k, "name": v} for k, v in FAMILIES.items()],
        "cases": [{"f": c[0], "size": c[1], "seed": c[2], "id": c[3]} for c in cases],
        "fields": ["first_E", "first_V", "first_Q", "agent_s", "reflection_s", "judge_s", "solver_s", "tokens"],
        "runs": runs,
    }


def render(data: dict) -> str:
    return "// Generated by src/results/site.py from outputs/reports/paper. Do not edit.\nwindow.DECOMPGYM = " + json.dumps(data, separators=(",", ":")) + ";\n"


def build(report: Path, site: Path = REPOSITORY / "site") -> dict:
    """Write site/data.js and site/benchmark_overview.png from a paper report folder."""
    data = site_data(report)
    (site / "data.js").write_text(render(data))
    figure = report / "figures" / "benchmark_overview.png"
    if figure.exists():
        shutil.copyfile(figure, site / "benchmark_overview.png")
    return {"site": str(site), "checked_rows": data["meta"]["checked_rows"], "data_kib": round((site / "data.js").stat().st_size / 1024)}


if __name__ == "__main__":
    print(json.dumps(build(REPOSITORY / "outputs" / "reports" / "paper"), indent=2))
