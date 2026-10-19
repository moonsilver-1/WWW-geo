# -*- coding: utf-8 -*-
"""Step 2 analysis: RealWeb-6 pilot — pre-registered criteria A/B/C (see prereg_step2.md).
A: fraction of pages with d_j = I(C_j) - I(A_j) > 0 (pilot target 6/6; full target 9/12)
B: page-level paired bootstrap of median(d_j), 95% CI (no pooled p)
C: sign(median d_real) == sign(d_synthetic)  [d_synthetic ~ +0.27 from 0.2-A]
Plus Gamma per page and the Setting comparison table."""
import itertools, json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "statistics"))
sys.path.insert(0, os.path.join(ROOT, "src", "evidence_state"))
from analyze_p0_1 import wht, GE1_IDX  # noqa: E402
from p0_data import states  # noqa: E402
from realweb_data import SPEC  # noqa: E402

ENG = ["qwen3b", "internvl3-2b"]
RNG = np.random.default_rng(42)

def load(engine):
    path = os.path.join(ROOT, "results", "realweb", f"eaq_{engine}.jsonl")
    recs = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if "error" not in r:
                    recs[(r["page"], tuple(r["state"]))] = r
    return recs

def spec_stats(F):
    F = np.asarray(F, float)
    fh = wht(F)
    E = fh ** 2
    tot = float(E[GE1_IDX].sum())
    if tot <= 1e-12:
        return None
    E1 = sum(E[2 ** i] for i in range(6))
    E2 = E1 + sum(E[s] for s in range(64) if bin(s).count("1") == 2)
    return {"var": tot, "R2_1": E1 / tot, "I_var": 1 - E2 / tot, "fh": fh}

def main():
    sl = [tuple(s) for s in states()]
    per_page = {}
    ok_all = True
    for eng in ENG:
        recs = load(eng)
        per_page[eng] = {}
        for pid in SPEC:
            have = [st for st in sl if (pid, st) in recs]
            if len(have) < 64:
                print(f"[{eng}] {pid}: {len(have)}/64 — skipped")
                ok_all = False
                continue
            A = np.array([recs[(pid, st)]["A_j"] for st in sl])
            P = np.array([recs[(pid, st)]["p"] for st in sl])
            spa = spec_stats(A)
            ius, gam = [], []
            for j in range(6):
                spu = spec_stats(P[:, j])
                if spu and spu["var"] > 0.01:
                    ius.append(spu["I_var"])
                    gam.append(1 - spu["fh"][2 ** j] ** 2 / spu["var"])
            ic = float(np.median(ius)) if ius else None
            d = ic - spa["I_var"] if ic is not None else None
            per_page[eng][pid] = {
                "I_A": spa["I_var"], "R2_1_A": spa["R2_1"], "I_C": ic,
                "d": d, "n_active_units": len(ius),
                "Gamma_median": float(np.median(gam)) if gam else None,
                "parse_fail": sum(0 if recs[(pid, st)]["parse_ok"] else 1 for st in sl)}
            print(f"[{eng}] {pid}: I_A={spa['I_var']:.3f} I_C={ic and round(ic,3)} "
                  f"d={d and round(d,3)} Gamma={per_page[eng][pid]['Gamma_median'] and round(per_page[eng][pid]['Gamma_median'],3)}")
    # pre-registered criteria
    crit = {}
    for eng in ENG:
        ds = [v["d"] for v in per_page.get(eng, {}).values() if v["d"] is not None]
        if len(ds) < 6:
            continue
        ds_arr = np.array(ds)
        boots = [float(np.median(ds_arr[RNG.integers(0, len(ds_arr), len(ds_arr))]))
                 for _ in range(10000)]
        ci = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
        crit[eng] = {
            "A_fraction_positive": float((ds_arr > 0).mean()),
            "A_pages_positive": int((ds_arr > 0).sum()),
            "n_pages": len(ds_arr),
            "B_median_d": float(np.median(ds_arr)), "B_ci95": ci,
            "B_pass": bool(ci[0] > 0),
            "C_same_sign_as_synthetic": bool(np.median(ds_arr) > 0),
        }
    # controlled comparison (from 0.2-A analysis)
    ctrl = json.load(open(os.path.join(ROOT, "results", "p0_2", "analysis_0_2a.json"), encoding="utf-8"))
    ctrl_tbl = {}
    for eng in ENG:
        pt = ctrl.get("per_task", {}).get(eng, {})
        if pt:
            ia = [r["I_A"] for r in pt.values()]
            ic = [r["median_I_C_active"] for r in pt.values() if r.get("median_I_C_active")]
            gm = [g["Gamma_median"] for g in ctrl.get("gamma", {}).values() if g.get("Gamma_median")] \
                if False else None
            ctrl_tbl[eng] = {"I_A_mean": float(np.mean(ia)),
                             "I_C_median_of_medians": float(np.median(ic)),
                             "Delta_I_median": float(np.median([r["median_Delta_I_active"] for r in pt.values()]))}
    # real table
    real_tbl = {}
    for eng in ENG:
        rows = list(per_page.get(eng, {}).values())
        if rows:
            real_tbl[eng] = {"I_A_mean": float(np.mean([r["I_A"] for r in rows])),
                             "I_C_median_of_medians": float(np.median([r["I_C"] for r in rows if r["I_C"]])),
                             "Delta_I_median": float(np.median([r["d"] for r in rows if r["d"]])),
                             "Gamma_mean": float(np.mean([r["Gamma_median"] for r in rows if r["Gamma_median"]]))}
    out = {"per_page": per_page, "criteria": crit,
           "table": {"controlled": ctrl_tbl, "realweb": real_tbl},
           "complete": ok_all}
    with open(os.path.join(ROOT, "results", "realweb", "analysis_step2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("\n== STEP 2 CRITERIA ==")
    print(json.dumps(crit, ensure_ascii=False, indent=1, default=str))
    print("table:", json.dumps(out["table"], indent=1, default=str))
    print("written: results/realweb/analysis_step2.json")

if __name__ == "__main__":
    main()
