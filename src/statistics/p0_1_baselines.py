# -*- coding: utf-8 -*-
"""P0-1 offline baselines (NO engine calls): R^2 of additive / pairwise / full Fourier models
for A(x)=U_ans and C-side utilities (U_R, U_P, per-unit c_i), from existing P0-1 jsonl.
R2_k = sum_{1<=|S|<=k} Fhat(S)^2 / sum_{S!=0} Fhat(S)^2  (variance explained vs mean).
Also: higher-order residual share 1-R2_2, and per-utility interaction complexity I(F).
"""
import itertools, json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "statistics"))
sys.path.insert(0, os.path.join(ROOT, "src", "evidence_state"))
from analyze_p0_1 import load_records, wht, GE1_IDX  # noqa: E402
from p0_data import TASKS, states  # noqa: E402

ENG = ["qwen3b", "internvl3-2b"]

def r2_profile(F):
    F = np.asarray(F, float)
    fh = wht(F)
    tot = float((fh[GE1_IDX] ** 2).sum())   # = Var(F) (sum of squared non-const coefficients)
    if tot <= 1e-12:
        return None
    E1 = sum(fh[2**i] ** 2 for i in range(6))
    E2 = E1 + sum(fh[s] ** 2 for s in range(64) if bin(s).count("1") == 2)
    return {"R2_additive": float(E1 / tot), "R2_pairwise": float(E2 / tot),
            "higher_order_share": float(1 - E2 / tot),
            "I_var": float((tot - E2) / tot), "var": tot}

def main():
    out = {}
    for eng in ENG:
        recs = load_records(eng)
        out[eng] = {}
        for tk in TASKS:
            sl = [tuple(s) for s in states()]
            if not all((tk, st, 0) in recs for st in sl):
                continue
            row = {}
            for key in ("U_ans", "U_R", "U_P"):
                F = np.array([np.mean(r["U_ans"]) if key == "U_ans" else r[key]
                              for st in sl for r in [recs[(tk, st, 0)]]])
                row[key] = r2_profile(F)
            # answered-mask (refusal indicator) and citations conditional on answering
            ins = np.array([1.0 if recs[(tk, st, 0)]["insufficient"] else 0.0 for st in sl])
            row["refusal"] = r2_profile(ins)
            ans = ins == 0
            for j in range(6):
                c = np.array([recs[(tk, st, 0)]["c"][j] for st in sl], dtype=float)
                row[f"c{j+1}"] = r2_profile(c)
            row["n_answered"] = int(ans.sum())
            out[eng][tk] = row
            a, ur = row["U_ans"], row["U_R"]
            msg = (f"[{eng}] {tk}: A: R2_1={a['R2_additive']:.3f} R2_2={a['R2_pairwise']:.3f} I={a['I_var']:.3f} | "
                   f"U_R: R2_1={ur['R2_additive'] if ur else float('nan'):.3f} "
                   f"R2_2={ur['R2_pairwise'] if ur else float('nan'):.3f} "
                   f"ho={ur['higher_order_share'] if ur else float('nan'):.3f}")
            print(msg)
    # aggregate: per-fact citation higher-order share (full cube; note refusal censoring)
    print("\n--- per-unit citation (full cube, T=0), higher-order share 1-R2_2 (var>0.02 only) ---")
    for eng in ENG:
        for tk in TASKS:
            row = out.get(eng, {}).get(tk, {})
            hos = [(k, round(v["higher_order_share"], 3), round(v["var"], 3))
                   for k, v in row.items() if k.startswith("c") and k.endswith("] ") or (k.startswith("c") and v and v["var"] > 0.02)]
            hos = [(k, round(v["higher_order_share"], 3), round(v["var"], 3))
                   for k, v in row.items() if k.startswith("c") and v and v["var"] > 0.02]
            if hos:
                print(f"[{eng}] {tk}: {hos}")
    with open(os.path.join(ROOT, "results", "p0_1", "baseline_r2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("\nwritten: results/p0_1/baseline_r2.json")

if __name__ == "__main__":
    main()
