# -*- coding: utf-8 -*-
"""Phase 0.2-B analysis: Answer Attribution Separation Test.

H1: support genuinely low-order | H2: teacher-forcing artifact | H3: citation task complexity
Key comparison: SAME fact f_j, TWO readouts —
  A_j   = logprob support of f_j          (expect low I if H1)
  B_j   = binary fact-inclusion in P0-1 generations (same-scale binary support)
  C_A   = statement-scoped attribution scores p_i^{(j)}  (attribution readout)
  C     = whole-answer EAQ attribution (0.2-A)
H2 excluded iff I(C_A) >> I(A_j) for the SAME statement (readout, not estimand, drives complexity).
Also: Gamma(C_i) = 1 - R2_self (configuration share), cross-model similarities, null calibration.
"""
import itertools, json, os, re, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "statistics"))
sys.path.insert(0, os.path.join(ROOT, "src", "evidence_state"))
from analyze_p0_1 import wht, GE1_IDX  # noqa: E402
from p0_data import TASKS, FACTS, states  # noqa: E402

ENG = ["qwen3b", "internvl3-2b"]

def load_jsonl(path, key):
    recs = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if "error" not in r:
                    recs[key(r)] = r
    return recs

def spec(F):
    F = np.asarray(F, float)
    fh = wht(F)
    E = fh ** 2
    tot = float(E[GE1_IDX].sum())
    if tot <= 1e-12:
        return None
    E1 = sum(E[2 ** i] for i in range(6))
    E2 = E1 + sum(E[s] for s in range(64) if bin(s).count("1") == 2)
    return {"var": tot, "R2_1": E1 / tot, "R2_2": E2 / tot, "I_var": 1 - E2 / tot,
            "fh": fh, "E": E}

def self_r2(fh):
    """R2 of model {const, x_i} alone -> Gamma_i = 1 - R2_self."""
    return {f"u{j+1}": float(fh[2 ** j] ** 2) for j in range(6)}

def main():
    sl = [tuple(s) for s in states()]
    out = {"per_task": {}, "gamma": {}, "cross_model": {}, "null": {}, "verdict": {}}
    store = {}
    for eng in ENG:
        sep = load_jsonl(os.path.join(ROOT, "results", "p0_2", f"sep_{eng}.jsonl"),
                         lambda r: (r["task"], tuple(r["state"]), r["fact"], r.get("rerun", 0)))
        p01 = load_jsonl(os.path.join(ROOT, "results", "p0_1", f"{eng}.jsonl"),
                         lambda r: (r["task"], tuple(r["state"]), r.get("rerun", 0)))
        eaq = load_jsonl(os.path.join(ROOT, "results", "p0_2", f"eaq_{eng}.jsonl"),
                         lambda r: (r["task"], tuple(r["state"])))
        out["per_task"][eng] = {}
        for tk in TASKS:
            if not all((tk, st, 0) in sep or True for st in sl):
                pass
            row = {"facts": {}, "n_sep": sum(1 for st in sl for f in FACTS[tk]
                                             if (tk, st, f["id"], 0) in sep)}
            for fact in FACTS[tk]:
                fid = fact["id"]
                fr = {}
                # A_j support spectrum
                if all((tk, st, fid, 0) in sep for st in sl):
                    A = np.array([sep[(tk, st, fid, 0)]["A_j"] for st in sl])
                    sp = spec(A)
                    # rerun noise for A_j
                    diffs = []
                    for st in sl:
                        r1, r2 = sep.get((tk, st, fid, 0)), sep.get((tk, st, fid, 1))
                        if r1 and r2:
                            diffs.append(abs(r1["A_j"] - r2["A_j"]))
                    fr["A_j"] = {"I_var": sp["I_var"], "R2_1": sp["R2_1"], "R2_2": sp["R2_2"],
                                 "var": sp["var"], "mean": float(A.mean()),
                                 "noise_maxdiff": max(diffs) if diffs else None}
                    # additive null for A_j
                    sg = (np.mean(diffs) / np.sqrt(2)) if diffs else 0.0
                    F_add = sp["fh"][0] + sum(sp["fh"][2 ** j] * np.array(
                        [1 if (v >> (5 - j)) & 1 else -1 for v in range(64)]) for j in range(6))
                    rng = np.random.default_rng(7)
                    if sg > 0:
                        null = [(wht(F_add + rng.normal(0, sg, 64))[GE1_IDX] ** 2)[1:].sum() for _ in range(5000)]
                        fr["A_j"]["null_p"] = float((np.array(null) >= sp["I_var"] * sp["var"] * 0 +
                                                     sum(E for s2, E in enumerate(sp["E"]) if bin(s2).count("1") >= 2)).mean())
                    # B_j binary inclusion from P0-1 generations
                    if all((tk, st, 0) in p01 for st in sl):
                        B = np.array([1.0 if re.search(fact["regex"], p01[(tk, st, 0)]["gen"], re.I)
                                      else 0.0 for st in sl])
                        spb = spec(B)
                        if spb:
                            fr["B_j"] = {"I_var": spb["I_var"], "R2_1": spb["R2_1"],
                                         "rate": float(B.mean())}
                    # C_A attribution spectra per unit
                    if all((tk, st, fid, 0) in sep for st in sl):
                        P = np.array([sep[(tk, st, fid, 0)]["p"] for st in sl])
                        units = {}
                        for j in range(6):
                            spu = spec(P[:, j])
                            if spu and spu["var"] > 0.01:
                                units[f"c{j+1}"] = {"I_var": spu["I_var"], "R2_1": spu["R2_1"],
                                                    "var": spu["var"], "mean_p": float(P[:, j].mean())}
                        if units:
                            fr["C_A"] = {"median_I": float(np.median([u["I_var"] for u in units.values()])),
                                         "max_I": max(u["I_var"] for u in units.values()),
                                         "n_active": len(units), "units": units}
                        store[(eng, tk, fid)] = P
                row["facts"][fid] = fr
            # whole-answer C (0.2-A) for comparison + Gamma
            if all((tk, st) in eaq for st in sl):
                Pw = np.array([eaq[(tk, st)]["p"] for st in sl])
                gam, iw = [], []
                for j in range(6):
                    spu = spec(Pw[:, j])
                    if spu and spu["var"] > 0.01:
                        iw.append(spu["I_var"])
                        gam.append(1 - spu["fh"][2 ** j] ** 2 / spu["var"])
                row["C_whole"] = {"median_I": float(np.median(iw)) if iw else None,
                                  "median_Gamma": float(np.median(gam)) if gam else None}
                out["gamma"][f"{eng}|{tk}"] = {"Gamma_median": row["C_whole"]["median_Gamma"],
                                               "Gamma_per_unit": [round(g, 3) for g in gam]}
            out["per_task"][eng][tk] = row
            f1 = row["facts"][FACTS[tk][0]["id"]]
            ca = f1.get("C_A", {}).get("median_I")
            print(f"[{eng}] {tk}: n={row['n_sep']} "
                  f"f1: I(A)={f1.get('A_j',{}).get('I_var')} I(B)={f1.get('B_j',{}).get('I_var')} "
                  f"I(C_A)={ca} | C_whole={row.get('C_whole',{}).get('median_I')} "
                  f"Gamma={row.get('C_whole',{}).get('median_Gamma')}")
    # cross-model similarity
    def cos(a, b):
        n = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / n) if n > 1e-9 else None
    cm = {"A_j": [], "C_A": []}
    for tk in TASKS:
        for fact in FACTS[tk]:
            fid = fact["id"]
            if ("qwen3b", tk, fid) in store and ("internvl3-2b", tk, fid) in store:
                A1 = np.array([sep_q[(tk, st, fid, 0)]["A_j"] for st in sl]) if False else None
        pass
    # recompute A_j spectra per engine for cross-model
    sep_recs = {e: load_jsonl(os.path.join(ROOT, "results", "p0_2", f"sep_{e}.jsonl"),
                              lambda r: (r["task"], tuple(r["state"]), r["fact"], r.get("rerun", 0)))
                for e in ENG}
    for tk in TASKS:
        for fact in FACTS[tk]:
            fid = fact["id"]
            fhs = {}
            for e in ENG:
                if all((tk, st, fid, 0) in sep_recs[e] for st in sl):
                    A = np.array([sep_recs[e][(tk, st, fid, 0)]["A_j"] for st in sl])
                    fhs[e] = wht(A)[GE1_IDX]
            if len(fhs) == 2:
                c = cos(fhs["qwen3b"], fhs["internvl3-2b"])
                if c is not None:
                    cm["A_j"].append(c)
            pas = {}
            for e in ENG:
                if (e, tk, fid) in store:
                    P = store[(e, tk, fid)]
                    for j in range(6):
                        f = wht(P[:, j])[GE1_IDX]
                        if np.linalg.norm(f) > 1e-6 * np.sqrt(63) + 1e-9:
                            pas[(e, j)] = f
            for j in range(6):
                if ("qwen3b", j) in pas and ("internvl3-2b", j) in pas:
                    c = cos(pas[("qwen3b", j)], pas[("internvl3-2b", j)])
                    if c is not None:
                        cm["C_A"].append(c)
    out["cross_model"] = {"A_j_cosine_mean": float(np.mean(cm["A_j"])) if cm["A_j"] else None,
                          "A_j_cosines": [round(c, 3) for c in cm["A_j"]],
                          "C_A_cosine_mean": float(np.mean(cm["C_A"])) if cm["C_A"] else None,
                          "n_C_A_pairs": len(cm["C_A"])}
    # verdict
    gaps, verdicts = [], []
    for eng in ENG:
        for tk in TASKS:
            for fact in FACTS[tk]:
                fr = out["per_task"][eng][tk]["facts"][fact["id"]]
                ia = fr.get("A_j", {}).get("I_var")
                ic = fr.get("C_A", {}).get("median_I")
                if ia is not None and ic is not None:
                    gaps.append({"eng": eng, "task": tk, "fact": fact["id"],
                                 "I_A": round(ia, 3), "I_CA": round(ic, 3),
                                 "d": round(ic - ia, 3), "H2_excluded_for_this_fact": ic - ia > 0.1})
    out["verdict"] = {
        "n_pairs": len(gaps),
        "H2_excluded_fraction": (sum(1 for g in gaps if g["H2_excluded_for_this_fact"]) / len(gaps)) if gaps else None,
        "pairs": gaps,
    }
    with open(os.path.join(ROOT, "results", "p0_2", "analysis_0_2b.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("\n== 0.2-B VERDICT ==")
    print(json.dumps(out["verdict"], ensure_ascii=False, indent=1, default=str)[:2000])
    print("cross-model:", json.dumps(out["cross_model"], indent=1))
    print("written: results/p0_2/analysis_0_2b.json")

if __name__ == "__main__":
    main()
