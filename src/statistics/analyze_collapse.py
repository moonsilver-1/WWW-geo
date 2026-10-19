# -*- coding: utf-8 -*-
"""Interaction Collapse Test suite (0.2-D, analysis-level, zero engine calls).

Sources: EAQ p_i (results/p0_2/eaq_{eng}.jsonl) and explicit-citation c_i (results/p0_1/{eng}.jsonl).

Statistics (each with EXACT permutation test over all distinct type-label assignments):
  kappa_g   : share of ge2 energy in top-g signatures (g=1 and g=3), per unit and pooled over units
  xi_cross  : share of ge2 energy in signatures containing >=2 DISTINCT types (cross-modal share)

Interpretation:
  - I(C) is invariant to label permutation; these statistics are not.
  - p_small => interaction energy is ORGANIZED by the true evidence typing (collapse: shuffle -> gone)
  - p_large => high-order structure is generic complexity, not typed organization
"""
import itertools, json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "statistics"))
sys.path.insert(0, os.path.join(ROOT, "src", "evidence_state"))
from analyze_p0_1 import wht, GE1_IDX, design  # noqa: E402
from p0_data import TASKS, states  # noqa: E402

ENG = ["qwen3b", "internvl3-2b"]
_, M = design()
GE2 = [s for s in range(1, 64) if bin(s).count("1") >= 2]

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

def load_p01(engine):
    path = os.path.join(ROOT, "results", "p0_1", f"{engine}.jsonl")
    recs = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if "error" not in r and r.get("rerun", 0) == 0:
                    recs[(r["task"], tuple(r["state"]))] = r
    return recs

def unit_types(tk):
    return [u["type"] if not u.get("caption") else "T" for u in TASKS[tk]["units"]]

def sig_energies(E, types):
    """signature -> ge2 energy under the given type assignment."""
    sig = {}
    for s in GE2:
        members = [i for i in range(6) if (s >> (5 - i)) & 1]
        key = tuple(sorted(types[i] for i in members))
        sig[key] = sig.get(key, 0.0) + float(E[s])
    return sig

def stats_from_E(E, types):
    sig = sig_energies(E, types)
    tot = sum(sig.values())
    if tot <= 1e-12:
        return None
    vals = sorted(sig.values(), reverse=True)
    k1 = vals[0] / tot
    k3 = sum(vals[:3]) / tot
    xi = sum(v for k, v in sig.items() if len(set(k)) >= 2) / tot
    return {"kappa_1": k1, "kappa_3": k3, "xi_cross": xi, "ge2_total": tot}

def exact_test(E, tk, g_stat="kappa_3"):
    types = unit_types(tk)
    true = stats_from_E(E, types)
    if true is None:
        return None
    nulls = []
    seen = set()
    for perm in itertools.permutations(types):
        if perm in seen:
            continue
        seen.add(perm)
        st = stats_from_E(E, list(perm))
        if st:
            nulls.append(st[g_stat])
    nulls = np.array(nulls)
    p = float((nulls >= true[g_stat] - 1e-12).mean())
    return {"true": true, "p": p, "n_perm": len(nulls),
            "null_mean": float(nulls.mean()), "null_max": float(nulls.max())}

def main():
    out = {}
    for eng in ENG:
        eaq = load_eaq(eng)
        p01 = load_p01(eng)
        out[eng] = {}
        for tk in TASKS:
            sl = [tuple(s) for s in states()]
            row = {"EAQ": {}, "explicit_cit": {}}
            # --- EAQ p_i units (active only) + pooled
            if all((tk, st) in eaq for st in sl):
                P = np.array([eaq[(tk, st)]["p"] for st in sl])
                Es = []
                for j in range(6):
                    fh = wht(P[:, j])
                    E = fh ** 2
                    if (E[GE1_IDX]).sum() > 0.01 * 6:  # active: var > 0.01
                        st = exact_test(E, tk)
                        if st:
                            row["EAQ"][f"c{j+1}"] = {k: (round(v, 3) if isinstance(v, float) else v)
                                                     for k, v in st["true"].items()} | {"p": st["p"]}
                # pooled: sum of unit spectra energies (independent units share the cube)
                E_pool = np.zeros(64)
                for j in range(6):
                    E_pool += wht(P[:, j]) ** 2
                st = exact_test(E_pool, tk)
                row["EAQ"]["POOLED"] = {k: (round(v, 3) if isinstance(v, float) else v)
                                        for k, v in st["true"].items()} | {"p": st["p"]}
            # --- P0-1 explicit citation units
            if all((tk, st) in p01 for st in sl):
                Cs = np.array([p01[(tk, st)]["c"] for st in sl], dtype=float)
                E_pool = np.zeros(64)
                for j in range(6):
                    fh = wht(Cs[:, j])
                    E = fh ** 2
                    if (E[GE1_IDX]).sum() > 0.01 * 6:
                        st = exact_test(E, tk)
                        if st:
                            row["explicit_cit"][f"c{j+1}"] = {k: (round(v, 3) if isinstance(v, float) else v)
                                                              for k, v in st["true"].items()} | {"p": st["p"]}
                    E_pool += E
                if E_pool.sum() > 0:
                    st = exact_test(E_pool, tk)
                    row["explicit_cit"]["POOLED"] = {k: (round(v, 3) if isinstance(v, float) else v)
                                                     for k, v in st["true"].items()} | {"p": st["p"]}
            out[eng][tk] = row
            for mode in ("EAQ", "explicit_cit"):
                if row[mode].get("POOLED"):
                    pl = row[mode]["POOLED"]
                    nps = sum(1 for k, v in row[mode].items() if k != "POOLED" and v["p"] <= 0.05)
                    print(f"[{eng}] {tk} {mode}: POOLED k1={pl['kappa_1']} k3={pl['kappa_3']} "
                          f"xi={pl['xi_cross']} p={pl['p']} | per-unit p<=0.05: {nps}")
    with open(os.path.join(ROOT, "results", "p0_2", "collapse_suite.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("\nwritten: results/p0_2/collapse_suite.json")

if __name__ == "__main__":
    main()
