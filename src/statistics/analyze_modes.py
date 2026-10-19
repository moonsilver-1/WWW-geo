# -*- coding: utf-8 -*-
"""Step 2 R3: Attribution mode diagnostics.
For each (engine, page, setting): score morphology = (a) continuous-fraction
(p in (0.05, 0.95) on >=1 unit over half the states), (b) lookup-fraction (p in {0,1}
on >=5 of 6 units), (c) unit-counts with non-trivial variance across states.
Also: seed variance vs state variance on Step 3 L16 design.
"""
import json, os, sys
from collections import defaultdict
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "statistics"))
sys.path.insert(0, os.path.join(ROOT, "src", "evidence_state"))
from p0_data import states  # noqa: E402

ENG = ["qwen3b", "internvl3-2b"]

def load(path):
    if not os.path.exists(path):
        return None
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def per_page_modes(recs):
    by_page = defaultdict(list)
    for r in recs:
        if "p" not in r:
            continue
        by_page[r.get("task") or r.get("page")].append(r["p"])
    out = {}
    for page, Ps in by_page.items():
        P = np.array(Ps)
        continuous = float(np.mean((P > 0.05) & (P < 0.95)))
        per_state_units = ((P > 0.05) & (P < 0.95)).sum(axis=1)
        lookup = float(np.mean(per_state_units <= 1))
        n_active_med = float(np.median(per_state_units))
        var_per_unit = [float(P[:, j].std()) for j in range(6)]
        out[page] = {"continuous_frac": continuous, "lookup_frac": lookup,
                     "n_active_median": n_active_med,
                     "var_per_unit": [round(v, 3) for v in var_per_unit]}
    return out

def seed_vs_state_variance(recs):
    by = defaultdict(list)
    for r in recs:
        if "p" not in r:
            continue
        key = (r["kind"], r["page"], tuple(r["state"]))
        by[key].append(r["p"])  # r["p"] is a list of 6 floats
    by2 = defaultdict(list)
    for (k, page, st), ps_list in by.items():
        # ps_list: list of p-vectors from different seeds; seed variance = std across seeds of p_j
        arr = np.array(ps_list)  # (n_seeds, 6)
        seed_within = float(np.mean(arr.std(axis=0)))  # mean over units
        by2[(k, page)].append((st, seed_within))
    rows = []
    for (k, page), entries in by2.items():
        seed_within_list = [s for (st, s) in entries]
        states_set = list({st for (st, s) in entries})
        if len(seed_within_list) >= 3 and len(states_set) >= 8:
            state_arr = np.array([np.mean(np.array(by[(k, page, st)]), axis=0)
                                  for st in states_set])  # (n_states, 6)
            state_var = float(np.mean(state_arr.std(axis=0)))
            rows.append({"kind": k, "page": page,
                         "median_seed_var": float(np.median(seed_within_list)),
                         "median_state_var": state_var})
    return rows

def main():
    out = {"modes": {}, "seed_vs_state": {}}
    for eng in ENG:
        out["modes"][eng] = {}
        eaq_p02 = load(os.path.join(ROOT, "results", "p0_2", f"eaq_{eng}.jsonl"))
        if eaq_p02:
            out["modes"][eng]["controlled"] = per_page_modes(eaq_p02)
        for label, path in [("real_v1", os.path.join(ROOT, "results", "realweb", f"eaq_{eng}.jsonl")),
                            ("real_v2", os.path.join(ROOT, "results", "realweb", f"eaq_v2_{eng}.jsonl"))]:
            rdata = load(path)
            if rdata:
                out["modes"][eng][label] = per_page_modes(rdata)
        seed_path = os.path.join(ROOT, "results", "tempnull", f"seed_{eng}.jsonl")
        sdata = load(seed_path)
        if sdata:
            out["seed_vs_state"][eng] = seed_vs_state_variance(sdata)
    with open(os.path.join(ROOT, "results", "realweb", "mode_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("=" * 60)
    print(f"{'engine':<12} {'setting':<14} {'page':<14} {'cont':>6} {'lookup':>7} {'#active':>8}")
    for eng in ENG:
        for setting, pages in out["modes"][eng].items():
            for page, m in pages.items():
                print(f"{eng:<12} {setting:<14} {page:<14} "
                      f"{m['continuous_frac']:>6.2f} {m['lookup_frac']:>7.2f} {m['n_active_median']:>8.1f}")
    print("=" * 60)
    print("Seed vs state variance (Step 3, T=0.7):")
    for eng, rows in out["seed_vs_state"].items():
        if rows:
            for r in rows[:8]:
                print(f"  {eng} {r['kind']}/{r['page']}: seed={r['median_seed_var']:.3f} state={r['median_state_var']:.3f}")
    print("written: results/realweb/mode_diagnostics.json")

if __name__ == "__main__":
    main()
