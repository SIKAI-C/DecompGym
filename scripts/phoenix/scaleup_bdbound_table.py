"""Compact tables of the bound-side probes (reads the summary_<family>.json files written by
scaleup_bdbound_probe.py --summarize). Development only.

    python scripts/phoenix/scaleup_bdbound_table.py RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BOUND = ["eng", "eng_js", "eng_nostab", "eng_lw", "eng_cap", "naive_lp", "lp_bar", "lp_dual", "jensen", "lagr", "pool",
         "oneshot_greedy", "oneshot_saa", "empty", "balinski", "balinski_bar", "lp_bar_x"]
PRIMAL = ["eng_primal", "naive_int", "ils", "heur_saa", "mip", "oneshot_saa"]


def fmt(x, pct=False):
    if x is None:
        return "-"
    if pct:
        return f"{100 * x:.2g}%" if x >= 1e-4 else f"{x:.1e}"
    return f"{x:g}"


def main() -> None:
    for d in sys.argv[1:]:
        for f in sorted(Path(d).glob("summary_*.json")):
            s = json.loads(f.read_text())
            print(f"\n### {f.parent.name} / {f.stem}")
            for name, row in s.items():
                iv = row["interval"]
                print(f"\n{name} {row['info']}  LP* in [{iv.get('lo')}, {iv.get('hi')}] (rel width {fmt(iv.get('rel_width'))}, "
                      f"lo from {iv.get('lo_from')}, hi from {iv.get('hi_from')})")
                print("| arm | termination | t to 1e-5 | t to 1e-4 | bgap_LP at T | peak RSS MB |")
                print("|---|---|---|---|---|---|")
                for arm in BOUND:
                    e = row["arms"].get(arm)
                    if e is None:
                        continue
                    print(f"| {arm} | {e['term']} | {fmt(e.get('t_bound_1e-05'))} | {fmt(e.get('t_bound_0.0001'))} | "
                          f"{fmt(e.get('bgap_T'), True)} | {e['rss_mb']} |")
                prim = [a for a in PRIMAL if a in row["arms"] and row["arms"][a].get("best") is not None]
                if prim:
                    print("| primal arm | best | t_best | verified | gap to z_best |")
                    print("|---|---|---|---|---|")
                    for arm in prim:
                        e = row["arms"][arm]
                        print(f"| {arm} | {e['best']} | {fmt(e.get('t_best'))} | {e.get('verified')} | {fmt(e.get('gap_best'), True)} |")


if __name__ == "__main__":
    main()
