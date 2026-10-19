# -*- coding: utf-8 -*-
"""Phase 0.2-C analysis: Representation Invariance Test.

Prediction: same fact z, representation r(z) changes -> 
  I(A_j) roughly STABLE across representations (support is representation-invariant),
  I(C) CHANGES across representations (attribution is configuration-dependent).
Key statistics (within engine, per task):
  - I(A_j), median I(C_i), Gamma per representation
  - cross-representation spectral cosine: A_j spectra vs C unit spectra
  - support-invariance ratio = mean_cos_A / mean_cos_C  (>1 supports the claim)
"""
import itertools, json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "statistics"))
sys.path.insert(0, os.path.join(ROOT, "src", "evidence_state"))
from analyze_p0_1 import wht, GE1_IDX  # noqa: E402
from p0_data import TASKS, FACTS, states  # noqa: E402

ENG = ["qwen3b", "internvl3-2b"]
REPS = ["chart", "text", "table", "info"]

def load_rep(engine):
    path = os.path.join(ROOT, "results", "p0_2", f"rep_{engine}.jsonl")
    recs = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if "error" not in r:
                    recs[(r["task"], r["rep"], tuple(r["state"]))] = r
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
    return {"var": tot, "R2_1": E1 / tot, "I_var": 1 - E2 / tot, "fh": fh}

def cos(a, b):
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 1e-9 else None

def main():
    sl = [tuple(s) for s in states()]
    out = {"per_task": {}, "cross_rep": {}, "verdict": {}}
    for eng in ENG:
        recs = load_rep(eng)
        if not recs:
            continue
        out["per_task"][eng] = {}
        for tk in ["t1_solar", "t3_ev"]:
            fact = FACTS[tk][0]
            reps_done = [rp for rp in REPS if all((tk, rp, st) in recs for st in sl)]
            if len(reps_done) < 2:
                print(f"[{eng}] {tk}: only {len(reps_done)} reps complete — skipped")
                continue
            row = {"reps": reps_done, "facts": fact["id"], "per_rep": {}}
            A_specs, C_specs = {}, {}
            for rp in reps_done:
                A = np.array([recs[(tk, rp, st)]["A_j"] for st in sl])
                P = np.array([recs[(tk, rp, st)]["p"] for st in sl])
                spa = spec(A)
                units, gam = [], []
                for j in range(6):
                    spu = spec(P[:, j])
                    if spu and spu["var"] > 0.01:
                        units.append(spu["I_var"])
                        gam.append(1 - spu["fh"][2 ** j] ** 2 / spu["var"])
                row["per_rep"][rp] = {
                    "I_A": spa["I_var"], "R2_1_A": spa["R2_1"],
                    "median_I_C": float(np.median(units)) if units else None,
                    "Gamma_median": float(np.median(gam)) if gam else None,
                    "mean_p_carrier": float(P[:, 3].mean())}
                A_specs[rp] = spa["fh"][GE1_IDX]
                C_specs[rp] = {j: wht(P[:, j])[GE1_IDX] for j in range(6)}
            # cross-rep cosines
            pairs = list(itertools.combinations(reps_done, 2))
            a_cos = [c for c in (cos(A_specs[a], A_specs[b]) for a, b in pairs) if c is not None]
            c_cos = [c for c in (cos(C_specs[a][3], C_specs[b][3]) for a, b in pairs) if c is not None]
            c_cos_all = [c for c in (cos(C_specs[a][j], C_specs[b][j])
                                     for a, b in pairs for j in range(6)) if c is not None]
            a_ivals = [row["per_rep"][rp]["I_A"] for rp in reps_done]
            c_ivals = [row["per_rep"][rp]["median_I_C"] for rp in reps_done if row["per_rep"][rp]["median_I_C"] is not None]
            row["cross_rep"] = {
                "A_cosine_mean": float(np.nanmean(a_cos)), "A_cosines": [round(c, 3) for c in a_cos],
                "C_carrier_cosine_mean": float(np.nanmean(c_cos)), "C_carrier_cosines": [round(c, 3) for c in c_cos],
                "C_all_units_cosine_mean": float(np.nanmean(c_cos_all)),
                "I_A_range": [round(min(a_ivals), 4), round(max(a_ivals), 4)],
                "I_C_range": [round(min(c_ivals), 3), round(max(c_ivals), 3)] if c_ivals else None,
                "invariance_ratio": (float(np.nanmean(a_cos) / np.nanmean(c_cos))
                                     if np.nanmean(c_cos) and abs(np.nanmean(c_cos)) > 1e-6 else None)}
            out["per_task"][eng][tk] = row
            print(f"[{eng}] {tk}: A_cos={row['cross_rep']['A_cosine_mean']:.3f} "
                  f"C_carrier_cos={row['cross_rep']['C_carrier_cosine_mean']:.3f} "
                  f"I_A_range={row['cross_rep']['I_A_range']} "
                  f"I_C_range={row['cross_rep']['I_C_range']} "
                  f"ratio={row['cross_rep']['invariance_ratio'] and round(row['cross_rep']['invariance_ratio'],2)}")
            print(f"   per-rep I(A): " + " ".join(f"{rp}={row['per_rep'][rp]['I_A']:.3f}" for rp in reps_done))
            print(f"   per-rep I(C): " + " ".join(
                f"{rp}={row['per_rep'][rp]['median_I_C'] and round(row['per_rep'][rp]['median_I_C'],3)}"
                for rp in reps_done))
    # verdict
    ratios, a_cos_all, c_cos_all = [], [], []
    for eng in out["per_task"]:
        for tk, row in out["per_task"][eng].items():
            cr = row["cross_rep"]
            if cr["invariance_ratio"]:
                ratios.append(cr["invariance_ratio"])
            a_cos_all.append(cr["A_cosine_mean"])
            c_cos_all.append(cr["C_carrier_cosine_mean"])
    out["verdict"] = {
        "mean_A_crossrep_cosine": float(np.mean(a_cos_all)) if a_cos_all else None,
        "mean_C_carrier_crossrep_cosine": float(np.mean(c_cos_all)) if c_cos_all else None,
        "invariance_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "representation_invariance_holds": bool(ratios and np.mean(ratios) > 1.5),
    }
    with open(os.path.join(ROOT, "results", "p0_2", "analysis_0_2c.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("\n== 0.2-C VERDICT ==")
    print(json.dumps(out["verdict"], indent=1))
    print("written: results/p0_2/analysis_0_2c.json")

if __name__ == "__main__":
    main()
