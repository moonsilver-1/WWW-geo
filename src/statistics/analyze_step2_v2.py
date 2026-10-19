# -*- coding: utf-8 -*-
"""Step 2 R1 retrospective: re-judge with EAQ-v2 data."""
import json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "statistics"))
sys.path.insert(0, os.path.join(ROOT, "src", "evidence_state"))
from analyze_p0_1 import wht, GE1_IDX  # noqa: E402
from p0_data import states  # noqa: E402
from realweb_data import SPEC  # noqa: E402

ENG = ["qwen3b", "internvl3-2b"]
RNG = np.random.default_rng(42)

def load(engine, suffix):
    path = os.path.join(ROOT, "results", "realweb", f"eaq{suffix}_{engine}.jsonl")
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
    crit = {}
    for eng in ENG:
        for suffix in ("", "_v2"):
            recs = load(eng, suffix)
            per_page = {}
            for pid in SPEC:
                have = [st for st in sl if (pid, st) in recs]
                if len(have) < 64:
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
                per_page[pid] = {"I_A": spa["I_var"], "I_C": ic, "d": d}
            if len(per_page) < 6:
                continue
            ds = np.array([v["d"] for v in per_page.values() if v["d"] is not None])
            if len(ds) < 3:
                continue
            boots = [float(np.median(ds[RNG.integers(0, len(ds), len(ds))])) for _ in range(10000)]
            ci = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
            crit[f"{eng}|{suffix or 'v1'}"] = {
                "A_pages_positive": int((ds > 0).sum()),
                "B_median_d": float(np.median(ds)), "B_ci95": ci,
                "B_pass": bool(ci[0] > 0),
                "C_same_sign": bool(np.median(ds) > 0),
                "per_page": {k: round(v["d"], 3) for k, v in per_page.items()}}
    print(json.dumps(crit, indent=1, ensure_ascii=False))
    with open(os.path.join(ROOT, "results", "realweb", "analysis_step2_v2.json"), "w", encoding="utf-8") as f:
        json.dump(crit, f, ensure_ascii=False, indent=1, default=float)
    print("\nwritten: results/realweb/analysis_step2_v2.json")

if __name__ == "__main__":
    main()
