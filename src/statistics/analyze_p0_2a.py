# -*- coding: utf-8 -*-
"""Phase 0.2-A analysis: Evidence Attribution Query (EAQ) spectra.

Per (task, engine):
  - C_i(x) = p_i(x) in [0,1] from EAQ -> exact Walsh spectrum per unit
  - I(C_i) = 1 - R^2_pairwise (variance-normalized, const excluded)
  - Delta_I = I(C_i) - I(A)  (I(A) loaded from results/p0_1/baseline_r2.json)
  - Interaction Collapse Test: typed concentration kappa (top-3 signature share of
    ge2 energy) under TRUE type labels vs ALL distinct label permutations (exact test;
    note I(C) itself is permutation-invariant, kappa is not)
  - cross-engine spectral similarity per unit
"""
import itertools, json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "statistics"))
sys.path.insert(0, os.path.join(ROOT, "src", "evidence_state"))
from analyze_p0_1 import load_records as _dummy, wht, GE1_IDX, design, sig_of  # noqa: E402
from p0_data import TASKS, states  # noqa: E402

ENG = ["qwen3b", "internvl3-2b"]
_, M = design()

def load_eaq(engine):
    path = os.path.join(ROOT, "results", "p0_2", f"eaq_{engine}.jsonl")
    recs = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if "error" not in r:
                    recs[(r["task"], tuple(r["state"]))] = r
    return recs

def r2_profile(F):
    F = np.asarray(F, float)
    fh = wht(F)
    tot = float((fh[GE1_IDX] ** 2).sum())
    if tot <= 1e-12:
        return None
    E = fh ** 2
    E1 = sum(E[2 ** i] for i in range(6))
    E2 = E1 + sum(E[s] for s in range(64) if bin(s).count("1") == 2)
    return {"R2_1": E1 / tot, "R2_2": E2 / tot, "I_var": 1 - E2 / tot, "E": E, "fh": fh, "var": tot}

def kappa(E, task_key, g=3):
    """typed concentration: share of ge2 energy in top-g signatures (true labels)."""
    sig = {}
    for s in range(1, 64):
        if bin(s).count("1") >= 2:
            k = sig_of(task_key, s)
            sig[k] = sig.get(k, 0.0) + float(E[s])
    tot = sum(sig.values())
    if tot <= 1e-12:
        return None, {}
    top = sorted(sig.values(), reverse=True)[:g]
    return sum(top) / tot, sig

def collapse_test(E, task_key, g=3):
    """Exact permutation test over all distinct type-label assignments of the 6 units."""
    types = [u["type"] if not u.get("caption") else "T" for u in TASKS[task_key]["units"]]
    k_true, _ = kappa(E, task_key, g)
    if k_true is None:
        return None, None, 0
    seen = set()
    nulls = []
    for perm in itertools.permutations(types):
        key = perm
        if key in seen:
            continue
        seen.add(key)
        # temporarily recompute sig_of with permuted labels
        orig = TASKS[task_key]["units"]
        for u, t in zip(orig, perm):
            u["_permtype"] = t if not (u.get("caption") and t == "T") else u["type"]
        # monkey: sig_of uses u["type"] unless caption->T; emulate by swapping types
        saved = [u["type"] for u in orig]
        for u, t in zip(orig, perm):
            u["type"] = t
        k_perm, _ = kappa(E, task_key, g)
        for u, t in zip(orig, saved):
            u["type"] = t
        if k_perm is not None:
            nulls.append(k_perm)
    nulls = np.array(nulls)
    p = float((nulls >= k_true - 1e-12).mean())
    return k_true, p, len(nulls)

def main():
    base = json.load(open(os.path.join(ROOT, "results", "p0_1", "baseline_r2.json"), encoding="utf-8"))
    out = {"per_task": {}, "collapse": {}, "cross_engine": {}, "verdict": {}}
    store = {}
    for eng in ENG:
        recs = load_eaq(eng)
        out["per_task"][eng] = {}
        for tk in TASKS:
            sl = [tuple(s) for s in states()]
            have = [st for st in sl if (tk, st) in recs]
            if len(have) < 64:
                print(f"[{eng}] {tk}: {len(have)}/64 EAQ states — skipped")
                continue
            pf = sum(0 if recs[(tk, st)]["parse_ok"] else 1 for st in sl)
            P = np.array([recs[(tk, st)]["p"] for st in sl])  # 64 x 6
            gold = TASKS[tk].get("gold", [])
            row = {"parse_fail": pf, "units": {}, "gold": gold}
            for j in range(6):
                rep = r2_profile(P[:, j])
                if rep is None:
                    continue
                k_true, p_perm, n_perm = collapse_test(rep["E"], tk)
                row["units"][f"c{j+1}"] = {
                    "var": rep["var"], "I_var": rep["I_var"], "R2_1": rep["R2_1"],
                    "R2_2": rep["R2_2"], "mean_p": float(P[:, j].mean()),
                    "kappa_true": k_true, "collapse_p": p_perm, "n_perm": n_perm,
                    "top_ge2_sigs": {}, }
            # top signatures per unit (with true labels)
            for j in range(6):
                rep = r2_profile(P[:, j])
                if rep is None:
                    continue
                _, sig = kappa(rep["E"], tk)
                row["units"][f"c{j+1}"]["top_ge2_sigs"] = {
                    k: round(v, 5) for k, v in sorted(sig.items(), key=lambda kv: -kv[1])[:4]}
            # gold-directed aggregate
            if gold:
                Cg = P[:, [g - 1 for g in gold]].mean(axis=1)
                rep = r2_profile(Cg)
                if rep:
                    row["C_gold"] = {"I_var": rep["I_var"], "R2_1": rep["R2_1"], "R2_2": rep["R2_2"]}
            I_A = base[eng][tk]["U_ans"]["I_var"]
            row["I_A"] = I_A
            act = [v for v in row["units"].values() if v["var"] > 0.01]
            row["median_I_C_active"] = float(np.median([v["I_var"] for v in act])) if act else None
            row["median_Delta_I_active"] = (float(np.median([v["I_var"] for v in act])) - I_A) if act else None
            row["collapse_p_values"] = {k: v["collapse_p"] for k, v in row["units"].items()
                                        if v["collapse_p"] is not None}
            out["per_task"][eng][tk] = row
            store[(eng, tk)] = P
            print(f"[{eng}] {tk}: I_A={I_A:.3f} med_I_C={row['median_I_C_active']} "
                  f"med_ΔI={row['median_Delta_I_active']} pf={pf} "
                  f"collapse_p={ {k: (round(v,3) if v is not None else None) for k,v in list(row['collapse_p_values'].items())[:3]} }")
    # cross-engine per task: mean over units of spectrum cosine
    ce = {}
    for tk in TASKS:
        if ("qwen3b", tk) in store and ("internvl3-2b", tk) in store:
            cos = []
            for j in range(6):
                fa = wht(store[("qwen3b", tk)][:, j]); fb = wht(store[("internvl3-2b", tk)][:, j])
                fan, fbn = fa[GE1_IDX], fb[GE1_IDX]
                n = np.linalg.norm(fan) * np.linalg.norm(fbn)
                if n > 1e-9:
                    cos.append(float(np.dot(fan, fbn) / n))
            ce[tk] = {"mean_cos_nonconst_over_units": float(np.mean(cos)),
                      "per_unit": [round(c, 3) for c in cos]}
    out["cross_engine"] = ce
    # verdict
    verd = {}
    for eng in ENG:
        gaps, collapse_ok, n_act = [], [], 0
        for tk, row in out["per_task"].get(eng, {}).items():
            if row.get("median_Delta_I_active") is not None:
                gaps.append(row["median_Delta_I_active"]); n_act += 1
            for p in row["collapse_p_values"].values():
                if p is not None and p <= 0.05:
                    collapse_ok.append((tk, p))
        verd[eng] = {"tasks_with_active_C": n_act,
                     "median_Delta_I_over_tasks": float(np.median(gaps)) if gaps else None,
                     "units_collapse_p_le_0.05": len(collapse_ok),
                     "EAQ_confirms_gap": bool(gaps and np.median(gaps) > 0.1)}
    out["verdict"] = verd
    with open(os.path.join(ROOT, "results", "p0_2", "analysis_0_2a.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("\n== 0.2-A VERDICT ==")
    print(json.dumps(verd, ensure_ascii=False, indent=1, default=str))
    print("cross-engine:", json.dumps(ce, indent=1, default=str))
    print("written: results/p0_2/analysis_0_2a.json")

if __name__ == "__main__":
    main()
